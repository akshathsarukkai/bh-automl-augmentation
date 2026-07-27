"""Tests for reaction-fingerprint cluster and similarity-bounded OOD splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.evaluation.similarity_ood as similarity_module
from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES
from bh_augmentation.evaluation.similarity_ood import (
    FINGERPRINT_EQUALITY_SEMANTICS,
    build_maximum_similarity_bounded_split,
    build_pairwise_reaction_similarity_index,
    build_reaction_fingerprint_cluster_holdouts,
    build_reaction_fingerprint_index,
)


def _signature(*values: str) -> dict[str, str]:
    assert len(values) == len(CANONICAL_ROLE_NAMES)
    return {
        f"canonical_{role}_smiles": value
        for role, value in zip(CANONICAL_ROLE_NAMES, values, strict=True)
    }


_SIGNATURES = (
    _signature("CC", "CN", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "CCNC"),
    _signature(
        "c1ccccc1",
        "c1ccncc1",
        "[Ni]",
        "P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "O=C([O-])[O-].[K+].[K+]",
        "CO",
        "c1ccc(Nc2ccncc2)cc1",
    ),
    _signature(
        "C#N",
        "N#N",
        "[Cu+]",
        "P",
        "[Na+].[OH-]",
        "O",
        "N=C=O",
    ),
    _signature(
        "C1CCCCC1",
        "N1CCCCC1",
        "[Pt]",
        "P(CC)(CC)CC",
        "[Li+].[CH3-]",
        "CCCO",
        "C1CCNCC1",
    ),
    _signature(
        "F[C@@H](Cl)Br",
        "N[C@H](C)C(=O)O",
        "[Ag+]",
        "P(F)(F)F",
        "[Cs+].[F-]",
        "CS(C)=O",
        "N[C@@H](C)C(=O)N[C@H](F)Cl",
    ),
)


def _canonical_frame(*, duplicate_first_group: bool = True) -> pd.DataFrame:
    rows = []
    for index, signature in enumerate(_SIGNATURES):
        rows.append(
            {
                "source_row_id": f"row-{index:02d}",
                "canonical_reaction_key": f"reaction-{index:02d}",
                "all_required_roles_parse_valid": True,
                **signature,
            }
        )
    if duplicate_first_group:
        rows.append(
            {
                **rows[0],
                "source_row_id": "row-duplicate",
            }
        )
    return pd.DataFrame(rows)


def test_reaction_fingerprint_uses_exact_canonical_seven_role_order() -> None:
    first = _signature("CCBr", "CN", "[Pd]", "P", "[Na+].[OH-]", "CCO", "CCNC")
    swapped = _signature(
        "CN",
        "CCBr",
        "[Pd]",
        "P",
        "[Na+].[OH-]",
        "CCO",
        "CCNC",
    )
    frame = pd.DataFrame(
        [
            {
                "source_row_id": "row-a",
                "canonical_reaction_key": "reaction-a",
                **first,
            },
            {
                "source_row_id": "row-b",
                "canonical_reaction_key": "reaction-b",
                **swapped,
            },
        ]
    )

    index = build_reaction_fingerprint_index(frame, n_bits_per_role=64)

    assert index.role_order == CANONICAL_ROLE_NAMES
    assert index.metadata["role_order"] == list(CANONICAL_ROLE_NAMES)
    assert index.tanimoto("reaction-a", "reaction-b") < 1.0
    assert (
        index.fingerprint_hashes["reaction-a"]
        != index.fingerprint_hashes["reaction-b"]
    )


def test_pairwise_similarity_index_is_bulk_built_symmetric_and_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rdkit import DataStructs

    bulk_calls = 0
    production_bulk = DataStructs.BulkTanimotoSimilarity

    def count_bulk(fingerprint: object, targets: object) -> object:
        nonlocal bulk_calls
        bulk_calls += 1
        return production_bulk(fingerprint, targets)

    monkeypatch.setattr(DataStructs, "BulkTanimotoSimilarity", count_bulk)
    pairwise = build_pairwise_reaction_similarity_index(
        _canonical_frame(),
        n_bits_per_role=64,
    )
    matrix = pairwise.matrix

    assert bulk_calls == len(pairwise.canonical_reaction_keys)
    assert matrix.shape == (
        len(pairwise.canonical_reaction_keys),
        len(pairwise.canonical_reaction_keys),
    )
    assert np.isfinite(matrix).all()
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.ones(len(matrix)))
    assert not matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        matrix[0, 0] = 0.0
    assert pairwise.matrix[0, 0] == 1.0
    assert (
        pairwise.fingerprint_metadata["fingerprint_metadata_hash"]
        == pairwise.fingerprint_metadata_hash
    )
    assert pairwise.fingerprint_table_hash
    assert pairwise.similarity_matrix_hash
    assert pairwise.similarity_index_hash


def test_pairwise_similarity_index_is_deterministic_and_matches_fingerprints() -> None:
    frame = _canonical_frame()
    first = build_pairwise_reaction_similarity_index(
        frame,
        n_bits_per_role=128,
    )
    second = build_pairwise_reaction_similarity_index(
        frame.sample(frac=1.0, random_state=44),
        n_bits_per_role=128,
    )
    fingerprints = build_reaction_fingerprint_index(
        frame,
        n_bits_per_role=128,
    )

    assert first.canonical_reaction_keys == tuple(sorted(fingerprints.group_keys))
    assert first.similarity_index_hash == second.similarity_index_hash
    assert np.array_equal(first.matrix, second.matrix)
    for left in first.canonical_reaction_keys:
        for right in first.canonical_reaction_keys:
            assert first.similarity(left, right) == pytest.approx(
                fingerprints.tanimoto(left, right),
                abs=0.0,
            )


def test_pairwise_nearest_query_has_lexical_ties_and_fixed_quantiles() -> None:
    # With one bit per role, all fixture fingerprints collide. The query must
    # retain canonical identities and resolve nearest-key ties lexically.
    pairwise = build_pairwise_reaction_similarity_index(
        _canonical_frame(duplicate_first_group=False),
        n_bits_per_role=1,
    )

    result = pairwise.nearest_training_similarities(
        train_keys={"reaction-01", "reaction-00"},
        test_keys={"reaction-04", "reaction-03", "reaction-02"},
    )

    assert result.train_keys == ("reaction-00", "reaction-01")
    assert result.test_keys == ("reaction-02", "reaction-03", "reaction-04")
    assert result.rows["nearest_train_canonical_reaction_key"].eq(
        "reaction-00"
    ).all()
    assert result.rows["nearest_train_tanimoto"].eq(1.0).all()
    assert result.maximum_similarity == 1.0
    assert result.similarity_quantiles == {
        "q00": 1.0,
        "q25": 1.0,
        "q50": 1.0,
        "q75": 1.0,
        "q90": 1.0,
        "q95": 1.0,
        "q100": 1.0,
    }
    assert result.query_hash
    changed = result.rows
    changed.loc[:, "nearest_train_tanimoto"] = 0.0
    assert result.rows["nearest_train_tanimoto"].eq(1.0).all()


@pytest.mark.parametrize(
    ("train_keys", "test_keys", "match"),
    [
        (set(), {"reaction-02"}, "train_keys must be a non-empty"),
        ({"reaction-00"}, set(), "test_keys must be a non-empty"),
        (
            {"reaction-00", "reaction-01"},
            {"reaction-01", "reaction-02"},
            "requires disjoint",
        ),
        ({"reaction-unknown"}, {"reaction-02"}, "unknown keys"),
        (["reaction-00", "reaction-00"], {"reaction-02"}, "duplicate"),
    ],
)
def test_pairwise_nearest_query_rejects_invalid_key_sets(
    train_keys: object,
    test_keys: object,
    match: str,
) -> None:
    pairwise = build_pairwise_reaction_similarity_index(
        _canonical_frame(),
        n_bits_per_role=64,
    )

    with pytest.raises(ValueError, match=match):
        pairwise.nearest_training_similarities(
            train_keys=train_keys,  # type: ignore[arg-type]
            test_keys=test_keys,  # type: ignore[arg-type]
        )


def test_cluster_holdouts_are_row_order_independent_group_safe_and_audited() -> None:
    frame = _canonical_frame()

    first = build_reaction_fingerprint_cluster_holdouts(
        frame,
        n_bits_per_role=128,
        similarity_cutoff=0.9,
        seed=17,
    )
    second = build_reaction_fingerprint_cluster_holdouts(
        frame.sample(frac=1.0, random_state=91),
        n_bits_per_role=128,
        similarity_cutoff=0.9,
        seed=17,
    )

    pd.testing.assert_frame_equal(
        first.cluster_assignments,
        second.cluster_assignments,
        check_exact=True,
    )
    assert first.cluster_assignment_hash == second.cluster_assignment_hash
    assert [split.audit_record for split in first.splits] == [
        split.audit_record for split in second.splits
    ]
    assert first.included_splits
    for split in first.included_splits:
        assignments = split.assignments
        assert (
            assignments.groupby("canonical_reaction_key")["ood_split"]
            .nunique()
            .max()
            == 1
        )
        assert set(split.similarity_quantiles) == {
            "q00",
            "q25",
            "q50",
            "q75",
            "q90",
            "q95",
            "q100",
        }
        nearest = split.nearest_train_similarities
        assert len(nearest) == split.n_test_rows
        assert nearest["nearest_train_tanimoto"].between(0.0, 1.0).all()


def test_similarity_bounded_split_enforces_bound_for_every_test_row() -> None:
    split = build_maximum_similarity_bounded_split(
        _canonical_frame(),
        maximum_similarity=0.8,
        target_test_fraction=0.3,
        n_bits_per_role=128,
        seed=23,
    )

    assert split.status == "included"
    nearest = split.nearest_train_similarities
    assert not nearest.empty
    assert nearest["nearest_train_tanimoto"].le(0.8 + 1e-12).all()
    assert split.maximum_test_to_train_similarity == pytest.approx(
        nearest["nearest_train_tanimoto"].max()
    )
    test_ids = set(
        split.assignments.loc[
            split.assignments["ood_split"].eq("test"),
            "source_row_id",
        ]
    )
    assert test_ids == set(nearest["source_row_id"])


def test_similarity_bounded_split_is_deterministic_under_row_permutation() -> None:
    frame = _canonical_frame()
    kwargs = {
        "maximum_similarity": 0.8,
        "target_test_fraction": 0.3,
        "n_bits_per_role": 64,
        "seed": 101,
    }

    first = build_maximum_similarity_bounded_split(frame, **kwargs)
    second = build_maximum_similarity_bounded_split(
        frame.iloc[::-1].reset_index(drop=True),
        **kwargs,
    )

    assert first.audit_record == second.audit_record
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(
        first.nearest_train_similarities,
        second.nearest_train_similarities,
    )


def test_impossible_similarity_bound_is_excluded_with_explicit_reason() -> None:
    # One bit per role deliberately creates fingerprint collisions among
    # chemically distinct canonical keys. They remain separate identity groups.
    frame = _canonical_frame(duplicate_first_group=False)

    index = build_reaction_fingerprint_index(frame, n_bits_per_role=1)
    assert len(set(index.fingerprint_hashes.values())) == 1
    assert len(index.group_keys) == len(frame)
    assert (
        index.metadata["fingerprint_equality_semantics"]
        == FINGERPRINT_EQUALITY_SEMANTICS
    )

    split = build_maximum_similarity_bounded_split(
        frame,
        maximum_similarity=0.5,
        n_bits_per_role=1,
        seed=3,
    )

    assert split.status == "excluded"
    assert split.split_hash is None
    assert split.assignments.empty
    assert split.nearest_train_similarities.empty
    assert split.exclusion_reason is not None
    assert split.exclusion_reason.startswith("impossible_similarity_bound:")


def test_cluster_folds_with_insufficient_support_are_explicitly_excluded() -> None:
    plan = build_reaction_fingerprint_cluster_holdouts(
        _canonical_frame(),
        n_bits_per_role=128,
        similarity_cutoff=0.9,
        min_test_rows=100,
    )

    assert plan.excluded_splits
    assert not plan.included_splits
    assert all(
        split.exclusion_reason is not None
        and split.exclusion_reason.startswith("insufficient_support:")
        for split in plan.excluded_splits
    )


def test_same_canonical_key_cannot_map_to_different_role_identities() -> None:
    frame = _canonical_frame()
    frame.loc[
        frame["source_row_id"].eq("row-duplicate"),
        "canonical_product_smiles",
    ] = "CCCCCC"

    with pytest.raises(ValueError, match="multiple seven-role identities"):
        build_reaction_fingerprint_index(frame)


def test_invalid_canonical_molecule_is_rejected_in_scientific_mode() -> None:
    frame = _canonical_frame()
    frame.loc[
        frame["canonical_reaction_key"].eq("reaction-01"),
        "canonical_ligand_smiles",
    ] = "not-a-smiles"

    with pytest.raises(ValueError, match="RDKit cannot parse canonical role"):
        build_reaction_fingerprint_index(frame)

    frame = _canonical_frame()
    frame.loc[0, "all_required_roles_parse_valid"] = False
    with pytest.raises(ValueError, match="invalid molecular roles"):
        build_reaction_fingerprint_index(frame)


def test_scientific_similarity_splits_require_rdkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> tuple[object, object, object, object]:
        raise ImportError(
            "RDKit is required for scientific reaction-similarity OOD splits."
        )

    monkeypatch.setattr(similarity_module, "_load_rdkit", unavailable)

    with pytest.raises(ImportError, match="RDKit is required"):
        build_reaction_fingerprint_index(_canonical_frame())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"maximum_similarity": -0.1}, "maximum_similarity"),
        ({"maximum_similarity": float("nan")}, "maximum_similarity"),
        (
            {"maximum_similarity": 0.8, "target_test_fraction": 1.0},
            "target_test_fraction",
        ),
        (
            {"maximum_similarity": 0.8, "min_train_canonical_groups": 0},
            "min_train_canonical_groups",
        ),
    ],
)
def test_similarity_split_configuration_is_strict(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_maximum_similarity_bounded_split(_canonical_frame(), **kwargs)
