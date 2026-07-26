"""Tests for reaction-aware condition recombination augmentation."""

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation.condition_recombine as recombine_module
from bh_augmentation.augmentation.condition_recombine import (
    CANDIDATE_AUDIT_ATTR,
    build_reaction_smiles,
    condition_recombine_pseudolabel,
    generate_condition_recombined_candidates,
    parse_reaction_smiles,
    pseudo_label_candidates,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    REQUIRED_SYNTHETIC_RANKING_FIELDS,
    REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
    canonicalize_synthetic_roles,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ROLE_TO_COLUMN,
    ensure_reaction_role_columns,
    reaction_roles_from_row,
)


def test_parse_and_build_reaction_smiles() -> None:
    parts = parse_reaction_smiles("A.B.C.D>>P")

    assert parts == {
        "valid": True,
        "left_tokens": ["A", "B", "C", "D"],
        "substrate_tokens": ["A", "B"],
        "condition_tokens": ["C", "D"],
        "product": "P",
    }
    assert build_reaction_smiles(["A", "B"], ["C", "D"], "P") == "A.B.C.D>>P"


@pytest.mark.parametrize("value", ["", "A.B.C", "A>>P", "A.B>>"])
def test_parse_reaction_smiles_rejects_invalid_values(value: str) -> None:
    assert parse_reaction_smiles(value)["valid"] is False


def test_generate_candidates_changes_reactions_without_original_duplicates() -> None:
    train = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2", "r3"],
            "source_row_id": ["row-1", "row-2", "row-3"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )

    candidates = generate_condition_recombined_candidates(
        train,
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=7,
        feature_config={
            "kind": "bh_role_separated",
            "n_bits": 32,
            "fingerprint_backend": "rdkit",
        },
    )

    assert not candidates.empty
    assert candidates["reaction_id"].str.startswith("synthetic_").all()
    assert candidates["is_synthetic"].all()
    assert candidates["pseudo_label"].all()
    assert set(candidates["reaction_smiles"]).isdisjoint(train["reaction_smiles"])
    assert candidates["canonical_reaction_key"].is_unique
    assert candidates["feature_hash"].is_unique
    assert candidates["chemical_parse_valid"].all()
    assert not candidates["source_identical"].any()
    assert not candidates["already_measured"].any()
    assert not candidates["duplicate_synthetic"].any()
    assert not candidates["feature_duplicate_synthetic"].any()
    assert set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(candidates.columns)
    source_reactions = train.set_index("reaction_id")["reaction_smiles"]
    assert all(
        row.reaction_smiles != source_reactions[row.source_reaction_id]
        for row in candidates.itertuples()
    )
    donor_rows = ensure_reaction_role_columns(train).set_index("source_row_id")
    for row in candidates.itertuples():
        synthetic_roles = reaction_roles_from_row(row._asdict())
        donor_identity = canonicalize_synthetic_roles(
            reaction_roles_from_row(donor_rows.loc[row.donor_row_id])
        )
        assert donor_identity.roles is not None
        donor_roles = donor_identity.roles
        for role in ("catalyst", "ligand", "base", "solvent_or_additive"):
            assert getattr(synthetic_roles, role) == getattr(donor_roles, role)
            assert getattr(row, ROLE_TO_COLUMN[role]) == getattr(donor_roles, role)

    audit = candidates.attrs[CANDIDATE_AUDIT_ATTR]
    assert set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(audit.columns)
    assert set(audit["source_row_id"]) <= set(train["source_row_id"])
    assert set(audit["donor_row_id"]) <= set(train["source_row_id"])


def test_canonical_candidate_regeneration_is_deterministic() -> None:
    train = pd.DataFrame(
        {
            "source_row_id": ["a", "b", "c"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )
    config = {
        "kind": "bh_role_separated",
        "n_bits": 32,
        "fingerprint_backend": "rdkit",
    }

    first = generate_condition_recombined_candidates(
        train,
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=11,
        feature_config=config,
    )
    second = generate_condition_recombined_candidates(
        train.copy(),
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=11,
        feature_config=config,
    )

    hash_columns = [
        "canonical_reaction_key",
        "canonical_reaction_hash",
        "feature_hash",
    ]
    assert first[hash_columns].to_dict("records") == second[hash_columns].to_dict(
        "records"
    )
    for role in CANONICAL_ROLE_NAMES:
        assert first[ROLE_TO_COLUMN[role]].tolist() == second[
            ROLE_TO_COLUMN[role]
        ].tolist()


def test_ranked_candidates_are_permutation_invariant_capped_and_auditable() -> None:
    train = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2", "r3", "r4"],
            "source_row_id": ["row-a", "row-b", "row-c", "row-d"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
                "COC.CN.[Pd].P(O)(O)O.N(C)C.CO>>COCN",
            ],
            "yield": [30.0, 50.0, 70.0, 90.0],
        }
    )
    feature_config = {
        "kind": "bh_role_separated",
        "n_bits": 32,
        "fingerprint_backend": "rdkit",
    }
    kwargs = {
        "synthetic_multiplier": 1.0,
        "max_synthetic_rows": 4,
        "random_state": 11,
        "feature_config": feature_config,
        "max_candidates_per_source": 1,
    }

    first = generate_condition_recombined_candidates(train, **kwargs)
    permuted = generate_condition_recombined_candidates(
        train.sample(frac=1.0, random_state=9),
        **kwargs,
    )
    comparison_columns = [
        "canonical_reaction_key",
        "source_row_id",
        "donor_row_id",
        "candidate_rank",
    ]
    assert first[comparison_columns].to_dict("records") == permuted[
        comparison_columns
    ].to_dict("records")
    assert first["candidate_rank"].is_monotonic_increasing

    audit = first.attrs[CANDIDATE_AUDIT_ATTR]
    assert audit.loc[audit["accepted"]].groupby("source_row_id").size().max() == 1
    assert np.isfinite(audit["nearest_training_support_distance"]).all()
    assert audit["calibrated_uncertainty"].isna().all()
    assert audit["uncertainty_rank_value"].notna().all()
    assert audit["uncertainty_rank_basis"].notna().all()
    assert audit["support_distance_metric"].eq(
        "binary_tanimoto_distance"
    ).all()
    assert set(REQUIRED_SYNTHETIC_SUPPORT_FIELDS) <= set(audit)
    assert set(REQUIRED_SYNTHETIC_RANKING_FIELDS) <= set(audit)


def test_recombination_rejects_nonpositive_source_cap() -> None:
    with pytest.raises(ValueError, match="max_candidates_per_source"):
        generate_condition_recombined_candidates(
            pd.DataFrame({"reaction_smiles": ["CC.N.O>>CCN"], "yield": [50.0]}),
            max_candidates_per_source=0,
        )


def test_generation_rejects_broader_measured_keys() -> None:
    train = pd.DataFrame(
        {
            "source_row_id": ["a", "b", "c"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )
    config = {
        "kind": "bh_role_separated",
        "n_bits": 32,
        "fingerprint_backend": "rdkit",
    }
    baseline = generate_condition_recombined_candidates(
        train,
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=19,
        feature_config=config,
    )
    measured_key = baseline.iloc[0]["canonical_reaction_key"]

    filtered = generate_condition_recombined_candidates(
        train,
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=19,
        feature_config=config,
        measured_identity_keys=[measured_key],
    )

    assert measured_key not in set(filtered["canonical_reaction_key"])
    audit = filtered.attrs[CANDIDATE_AUDIT_ATTR]
    measured_rows = audit.loc[audit["canonical_reaction_key"].eq(measured_key)]
    assert not measured_rows.empty
    assert measured_rows["already_measured"].all()
    assert set(measured_rows["rejection_reason"]) == {"already_measured"}


def test_feature_collision_is_not_reported_as_chemical_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame(
        {
            "source_row_id": ["a", "b", "c"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )
    monkeypatch.setattr(
        recombine_module,
        "_features_only",
        lambda df, config: np.ones((len(df), 2), dtype=np.float32),
    )

    accepted = generate_condition_recombined_candidates(
        train,
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=23,
        feature_config={"kind": "bh_role_separated"},
    )

    assert len(accepted) == 1
    audit = accepted.attrs[CANDIDATE_AUDIT_ATTR]
    feature_collisions = audit.loc[
        audit["rejection_reason"].eq("feature_duplicate_synthetic")
    ]
    assert not feature_collisions.empty
    assert (~feature_collisions["duplicate_synthetic"]).any()
    assert feature_collisions["canonical_reaction_key"].notna().all()


def test_pseudolabel_path_enriches_canonical_roles_before_featurization() -> None:
    train = pd.DataFrame(
        {
            "source_row_id": ["a", "b", "c"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )

    augmented = condition_recombine_pseudolabel(
        train,
        {
            "kind": "bh_role_separated",
            "n_bits": 8,
            "fingerprint_backend": "rdkit",
        },
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        teacher_model="ridge",
        min_neighbor_similarity=0.0,
        random_state=5,
    )

    synthetic = augmented.loc[augmented["is_synthetic"]]
    assert not synthetic.empty
    assert synthetic["canonical_reaction_key"].is_unique
    assert synthetic["feature_hash"].is_unique
    assert synthetic["chemical_parse_valid"].all()
    assert not synthetic["already_measured"].any()
    assert not augmented.attrs[CANDIDATE_AUDIT_ATTR].empty


def test_pseudo_label_candidates_clips_teacher_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2"],
            "reaction_smiles": ["CCBr.N.O>>CCN", "CCCl.N.C>>CCN"],
            "yield": [20.0, 80.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "reaction_id": ["synthetic_1", "synthetic_2"],
            "reaction_smiles": ["CCBr.N.C>>CCN", "CCCl.N.O>>CCN"],
            "yield": [np.nan, np.nan],
            "pseudo_label": [True, True],
        }
    )

    def fake_features(
        df: pd.DataFrame, config: dict[str, object]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return (
            np.arange(len(df) * 2, dtype=float).reshape(len(df), 2),
            pd.to_numeric(df["yield"], errors="coerce").fillna(0).to_numpy(),
            ["f0", "f1"],
        )

    monkeypatch.setattr(recombine_module, "build_feature_matrix", fake_features)
    monkeypatch.setattr(recombine_module, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(recombine_module, "train_model", lambda model, X, y: model)
    monkeypatch.setattr(
        recombine_module,
        "predict_model",
        lambda model, X: np.array([-5.0, 120.0]),
    )

    labeled = pseudo_label_candidates(
        train,
        candidates,
        {"kind": "reaction_role_concat"},
        teacher_model_name="random_forest",
    )

    assert labeled["yield"].tolist() == [0.0, 100.0]
    assert labeled["pseudo_label"].all()
    assert set(labeled["pseudo_label_model"]) == {"random_forest"}
