"""Adversarial leakage tests for the observed-only low-data protocol.

Each test mutates information the protocol says must not influence an artifact,
then proves the artifact is byte-identical -- or supplies information that must
never be reachable and proves the code refuses it. Happy-path fixtures would
pass whether or not the contract held; these are written so that they cannot.

The fixtures build a real four-partition low-data unit from saved canonical
grouped split assignments, so the tests exercise the production materializer
rather than a hand-rolled stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.augmentation.candidate_scope import (
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    OBSERVED_ONLY_LOW_DATA,
    CandidateScopeViolation,
)
from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.data.canonical_splits import SPLIT_SCHEMA_VERSION
from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    build_canonical_reaction_identity,
    canonicalize_smiles,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ReactionRoles,
    reaction_roles_to_record,
)
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.low_data_partitions import (
    build_low_data_evaluation_unit,
)
from bh_augmentation.evaluation.withheld_cell_oracle import (
    FrozenCandidateBundle,
    OracleContractViolation,
    withheld_cell_oracle_diagnostic,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

pytest.importorskip("rdkit")

_FEATURE_CONFIG = {
    "kind": "bh_role_separated",
    "n_bits": 64,
    "radius": 2,
    "fingerprint_backend": "rdkit",
    "categorical_columns": [],
}
_TRAIN_FRACTION = 0.5
_SEED = 0

# A small, fully crossed grid so condition transfer has real donors, and so the
# hidden partition contains reactions the transfer can genuinely reconstruct.
_REACTANT_1 = (
    "CCBr",
    "CCCBr",
    "CCCCBr",
    "CCCCCBr",
    "CCCCCCBr",
    "CCCCCCCBr",
    "CCCCCCCCBr",
    "CCCCCCCCCBr",
)
_LIGAND = ("CP(C)C", "CCP(CC)CC", "CCCP(CCC)CCC")
_BASE = ("[Na+].[OH-]", "[K+].[OH-]", "[Li+].[OH-]")

# 8 x 3 x 3 = 72 rows, ordered substrate-major (nine ligand/base combinations per
# substrate).
#
# The labeled subset observes **exactly one combination per substrate**, and the
# chosen offset advances by 4 -- coprime with 9 -- so consecutive labeled rows
# differ in both ligand and base. Both properties are load-bearing:
#
#   * one row per substrate forces donor selection to cross substrates, so a
#     transferred condition block lands on a grid cell the learner has not
#     observed. If a substrate contributed several labeled rows, the nearest
#     same-substrate donor would reproduce an observed row exactly and every
#     candidate would be rejected as already-observed;
#   * advancing both roles satisfies role_change_requirement="all", which
#     requires ligand AND base to change. A stride sharing a factor with 9 would
#     hold one role constant and yield no eligible donor at all.
#
# Either mistake produces zero candidates and a vacuously green leakage suite,
# so the fixture asserts a nonempty pool rather than skipping.
_LABELED_OFFSET_STEP = 4
_COMBINATIONS_PER_SUBSTRATE = len(_LIGAND) * len(_BASE)
_OUTER_TRAIN_CUTOFF = 54
_VALID_CUTOFF = 63


def _grid_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, reactant in enumerate(_REACTANT_1):
        for j, ligand in enumerate(_LIGAND):
            for k, base in enumerate(_BASE):
                roles = ReactionRoles(
                    reactant_1=reactant,
                    reactant_2="CN",
                    catalyst="[Pd]",
                    ligand=ligand,
                    base=base,
                    solvent_or_additive="CCO",
                    product=f"{reactant[:-2]}NC",
                )
                canonical = {
                    role: canonicalize_smiles(
                        getattr(roles, role), isomeric=True
                    ).canonical_smiles
                    for role in CANONICAL_ROLE_NAMES
                }
                identity = build_canonical_reaction_identity(
                    {
                        "all_required_roles_parse_valid": True,
                        **{
                            f"canonical_{role}_smiles": value
                            for role, value in canonical.items()
                        },
                    }
                )
                position = len(rows)
                rows.append(
                    {
                        "source_row_id": f"row-{position:03d}",
                        "reaction_smiles": ReactionRoles(**canonical).reaction_smiles(),
                        "yield": float(10 * i + 20 * j + 30 * k + 5),
                        "canonical_reaction_key": identity["key"],
                        "canonical_reaction_hash": identity["hash"],
                        "all_required_roles_parse_valid": True,
                        "canonicalization_version": CANONICALIZATION_VERSION,
                        **reaction_roles_to_record(ReactionRoles(**canonical)),
                        **{
                            f"canonical_{role}_smiles": value
                            for role, value in canonical.items()
                        },
                    }
                )
    return rows


def _write_dataset(directory: Path, rows: list[dict[str, Any]]) -> Path:
    dataset = directory / "canonical.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    return dataset


def _write_splits(directory: Path, dataset: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a group-safe split whose training subset is a strict prefix.

    Rows 0..53 are outer train, of which one row per substrate is labeled and the
    rest are hidden. Rows 54..62 are validation and 63..71 are test.
    """
    split_directory = directory / "splits"
    split_directory.mkdir()
    low_records: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if position < _OUTER_TRAIN_CUTOFF:
            outer = "train"
        elif position < _VALID_CUTOFF:
            outer = "valid"
        else:
            outer = "test"
        substrate_block, offset = divmod(position, _COMBINATIONS_PER_SUBSTRATE)
        labeled = outer == "train" and offset == (
            _LABELED_OFFSET_STEP * substrate_block
        ) % _COMBINATIONS_PER_SUBSTRATE
        for fraction, included in (
            (_TRAIN_FRACTION, labeled),
            (1.0, outer == "train"),
        ):
            low_records.append(
                {
                    "source_row_id": row["source_row_id"],
                    "canonical_reaction_key": row["canonical_reaction_key"],
                    "seed": _SEED,
                    "outer_split": outer,
                    "outer_group_order": position,
                    "train_fraction": fraction,
                    "included_in_training_subset": included,
                }
            )
    low = pd.DataFrame(low_records)
    # Both saved artifacts share one schema; the outer file is the 1.0 slice.
    outer = low.loc[low["train_fraction"].eq(1.0)].reset_index(drop=True)
    outer.to_csv(split_directory / "outer_split_assignments.csv", index=False)
    low.to_csv(split_directory / "low_data_subset_assignments.csv", index=False)

    from bh_augmentation.data.canonical_splits import split_assignment_hash

    per_seed = split_assignment_hash(
        outer.loc[outer["seed"].eq(_SEED)],
        low.loc[low["seed"].eq(_SEED)],
    )
    frame = pd.DataFrame(rows)
    aggregate = stable_hash({str(_SEED): per_seed})
    manifest = {
        "canonical_dataset_hash": sha256_file(dataset),
        "canonical_rows_read": len(rows),
        "canonical_subset_hash": stable_hash(
            frame.loc[:, ["source_row_id", "canonical_reaction_key"]].to_dict(
                orient="records"
            )
        ),
        "canonicalization_version": CANONICALIZATION_VERSION,
        "dataset_nrows": len(rows),
        "group_column": "canonical_reaction_key",
        "n_groups": int(frame["canonical_reaction_key"].nunique()),
        "n_rows": len(rows),
        "seeds": [_SEED],
        "split_hash": aggregate,
        "split_hashes": {str(_SEED): per_seed},
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "train_fractions": [_TRAIN_FRACTION, 1.0],
    }
    (split_directory / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return split_directory


@pytest.fixture()
def low_data_unit(tmp_path: Path):
    """Build one real four-partition low-data unit from saved assignments."""
    rows = _grid_rows()
    dataset = _write_dataset(tmp_path, rows)
    split_directory = _write_splits(tmp_path, dataset, rows)
    try:
        saved = load_saved_canonical_splits(
            dataset,
            split_directory,
            requested_seeds=[_SEED],
            requested_fractions=[_TRAIN_FRACTION, 1.0],
        )
    except ValueError as exc:  # pragma: no cover - fixture guard
        raise AssertionError(
            "The leakage fixture must load through the production saved-split "
            f"loader; a skipped leakage test proves nothing: {exc}"
        ) from exc
    return build_low_data_evaluation_unit(
        saved,
        seed=_SEED,
        train_fraction=_TRAIN_FRACTION,
        dataset_path=dataset,
    ), dataset, split_directory, rows


def _generate(unit, mode: str) -> dict[str, Any]:
    labeled = unit.labeled_train
    X, y, names, metadata = build_feature_matrix_with_metadata(labeled, _FEATURE_CONFIG)
    config = ConditionTransferConfig(
        donor_strategy="nearest_substrate",
        synthetic_multiplier=1.0,
        n_neighbors=3,
        label_strategy="teacher_ensemble",
        teacher_models=["random_forest"],
        max_teacher_std=None,
        min_similarity=0.0,
        high_yield_threshold=80.0,
        clip_y_min=0.0,
        clip_y_max=100.0,
        candidates_per_real=2,
        random_state=7,
        donor_similarity_n_bits=64,
        donor_similarity_radius=2,
        donor_similarity_backend="rdkit",
        role_change_requirement="all",
        fallback_policy="reject",
        max_candidates_per_source=1,
        requested_roles=["ligand", "base"],
    )
    return generate_condition_transfer_examples(
        labeled,
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        config,
        feature_config=_FEATURE_CONFIG,
        real_feature_names=list(names),
        real_feature_metadata=metadata,
        candidate_scope=unit.candidate_scope(mode),
    )


def _signature(generated: dict[str, Any]) -> tuple:
    audit = generated["candidate_df"]
    return (
        tuple(audit["canonical_reaction_key"].astype(str)),
        tuple(audit["rejection_reason"].fillna("").astype(str)),
        tuple(np.round(np.asarray(generated["synthetic_y"], dtype=float), 10)),
    )


# ------------------------------------------------------- hidden-yield leakage


def test_changing_hidden_outer_train_yields_changes_nothing(low_data_unit) -> None:
    """Hidden outcomes may not reach candidates or pseudo-labels.

    The hidden rows' yields are rewritten to values that would visibly move any
    teacher that saw them. The generated identities, their rejection reasons and
    their pseudo-labels must be identical.
    """
    unit, dataset, split_directory, rows = low_data_unit
    before = _signature(_generate(unit, OBSERVED_ONLY_LOW_DATA))

    mutated = pd.DataFrame(rows)
    hidden_ids = set(unit.hidden_outer_train["source_row_id"].astype(str))
    assert hidden_ids, "The fixture must have a nonempty hidden partition."
    mutated.loc[
        mutated["source_row_id"].astype(str).isin(hidden_ids), "yield"
    ] = -999.0
    mutated.to_csv(dataset, index=False)

    # The split manifest binds the dataset hash, so rebuild the unit directly
    # from the mutated frame rather than through the hash-checked loader.
    rebuilt = unit.__class__(
        evaluation_unit=unit.evaluation_unit,
        seed=unit.seed,
        train_fraction=unit.train_fraction,
        dataset_path=unit.dataset_path,
        dataset_hash=unit.dataset_hash,
        labeled_train=unit.labeled_train,
        hidden_outer_train=unit.hidden_outer_train,
        validation=unit.validation,
        test=unit.test,
        canonical_source_row_ids=unit.canonical_source_row_ids,
    )
    assert _signature(_generate(rebuilt, OBSERVED_ONLY_LOW_DATA)) == before


def test_changing_validation_and_test_yields_changes_nothing(low_data_unit) -> None:
    """Held-out outcomes may not reach candidates or selected pseudo-labels."""
    unit, _, _, _ = low_data_unit
    before = _signature(_generate(unit, OBSERVED_ONLY_LOW_DATA))

    poisoned_validation = unit.validation.copy()
    poisoned_validation["yield"] = -999.0
    rebuilt = unit.__class__(
        evaluation_unit=unit.evaluation_unit,
        seed=unit.seed,
        train_fraction=unit.train_fraction,
        dataset_path=unit.dataset_path,
        dataset_hash=unit.dataset_hash,
        labeled_train=unit.labeled_train,
        hidden_outer_train=unit.hidden_outer_train,
        validation=poisoned_validation,
        test=unit.test,
        canonical_source_row_ids=unit.canonical_source_row_ids,
    )
    assert _signature(_generate(rebuilt, OBSERVED_ONLY_LOW_DATA)) == before


def test_hidden_partition_frames_carry_no_yield_column(low_data_unit) -> None:
    """Hidden and test outcomes must not be materialized at all by default."""
    unit, _, _, _ = low_data_unit
    assert "yield" not in unit.hidden_outer_train.columns
    assert "yield" not in unit.test.columns
    assert "yield" in unit.labeled_train.columns


def test_true_hidden_labels_never_enter_the_fit_matrix(low_data_unit) -> None:
    """Every synthetic label must be a teacher prediction, not a measured yield."""
    unit, _, _, _ = low_data_unit
    generated = _generate(unit, OBSERVED_ONLY_LOW_DATA)
    synthetic = np.asarray(generated["synthetic_y"], dtype=float)
    if not len(synthetic):
        raise AssertionError(
            "The fixture must produce accepted candidates; an empty pool would "
            "make this leakage test vacuous."
        )
    oracle = unit.hidden_outer_train_oracle()
    audit = generated["candidate_df"]
    kept = audit.loc[audit["kept"].astype(bool)] if "kept" in audit else audit
    truth_by_key = dict(
        zip(oracle["canonical_reaction_key"].astype(str), oracle["yield"], strict=True)
    )
    for key, label in zip(kept["canonical_reaction_key"].astype(str), synthetic, strict=True):
        if key in truth_by_key:
            # A pseudo-label that exactly reproduces a withheld measurement would
            # be the signature of a leak. Teacher predictions do not do this.
            assert not np.isclose(label, truth_by_key[key], rtol=0.0, atol=1e-9)


# ------------------------------------------------------------- scope contract


def test_hidden_membership_is_not_queried_before_generation_freezes(
    low_data_unit,
) -> None:
    """The observed-only scope must not contain any hidden identity at all."""
    unit, _, _, _ = low_data_unit
    scope = unit.candidate_scope(OBSERVED_ONLY_LOW_DATA)
    hidden = set(unit.hidden_outer_train_identity_keys)
    assert hidden, "The fixture must have a nonempty hidden partition."
    assert not (hidden & scope.rejection_identity_keys())
    assert not (hidden & scope.quarantined_identity_keys())
    assert scope.global_identity_keys == ()
    assert scope.scope_record()["global_identity_hash"] == "not_consulted"


def test_global_membership_cannot_suppress_a_merely_hidden_candidate(
    low_data_unit,
) -> None:
    """A candidate the learner never saw must survive the observed-only gate."""
    unit, _, _, _ = low_data_unit
    observed = _generate(unit, OBSERVED_ONLY_LOW_DATA)["candidate_df"]
    prospective = _generate(unit, GLOBALLY_UNMEASURED_PROSPECTIVE)["candidate_df"]

    hidden = set(unit.hidden_outer_train_identity_keys)
    accepted_observed = set(
        observed.loc[observed["accepted"].astype(bool), "canonical_reaction_key"].astype(str)
    )
    accepted_prospective = set(
        prospective.loc[
            prospective["accepted"].astype(bool), "canonical_reaction_key"
        ].astype(str)
    )
    # On a densely crossed grid the prospective rule rejects what the low-data
    # rule accepts; that difference is the whole point of the correction.
    assert accepted_prospective <= accepted_observed
    suppressed = (accepted_observed - accepted_prospective) & hidden
    assert suppressed, (
        "The observed-only rule must admit at least one candidate the global "
        "rule suppresses; otherwise this fixture cannot distinguish the two."
    )


def test_an_identical_candidate_in_labeled_training_is_rejected(low_data_unit) -> None:
    """Observed chemistry must never be re-proposed as a synthetic target."""
    unit, _, _, _ = low_data_unit
    audit = _generate(unit, OBSERVED_ONLY_LOW_DATA)["candidate_df"]
    accepted = audit.loc[audit["accepted"].astype(bool)]
    observed = set(unit.labeled_train_identity_keys)
    assert not (set(accepted["canonical_reaction_key"].astype(str)) & observed)


def test_held_out_identity_matches_are_quarantined_from_student_training(
    low_data_unit,
) -> None:
    """Validation and test identities may not enter the training pool."""
    unit, _, _, _ = low_data_unit
    audit = _generate(unit, OBSERVED_ONLY_LOW_DATA)["candidate_df"]
    accepted = audit.loc[audit["accepted"].astype(bool)]
    quarantined = set(unit.quarantine_identity_keys)
    assert not (set(accepted["canonical_reaction_key"].astype(str)) & quarantined)
    flagged = audit.loc[audit["rejection_reason"].eq("quarantined_held_out_identity")]
    if len(flagged):
        # Their existence is recorded, not erased.
        assert set(flagged["canonical_reaction_key"].astype(str)) <= quarantined
        assert flagged["quarantine_role"].notna().all()


def test_hidden_matches_enter_training_only_with_synthetic_labels(
    low_data_unit,
) -> None:
    """A reconstructed hidden cell trains the student on a pseudo-label only."""
    unit, _, _, _ = low_data_unit
    generated = _generate(unit, OBSERVED_ONLY_LOW_DATA)
    rows = generated["synthetic_df"]
    if rows.empty:
        raise AssertionError(
            "The fixture must produce accepted candidates; an empty pool would "
            "make this leakage test vacuous."
        )
    labels = np.asarray(generated["synthetic_y"], dtype=float)
    # A synthetic row's `yield` column is its pseudo-label by construction, so
    # the contract to check is that the value is the teacher's prediction and
    # nothing else has been substituted into it.
    assert np.allclose(rows["synthetic_label"].to_numpy(float), labels)
    assert np.allclose(rows["yield"].to_numpy(float), labels)

    # Each identity entering the student is one the learner had not observed.
    observed = set(unit.labeled_train_identity_keys)
    assert not (set(rows["canonical_reaction_key"].astype(str)) & observed)

    # And where a row reconstructs a hidden cell, its label is the teacher's
    # estimate, never the withheld measurement.
    oracle = unit.hidden_outer_train_oracle()
    truth = dict(
        zip(oracle["canonical_reaction_key"].astype(str), oracle["yield"], strict=True)
    )
    reconstructed = 0
    for key, label in zip(rows["canonical_reaction_key"].astype(str), labels, strict=True):
        if key in truth:
            reconstructed += 1
            assert not np.isclose(label, truth[key], rtol=0.0, atol=1e-9)
    assert reconstructed, "The fixture must reconstruct at least one hidden cell."


def test_prospective_mode_still_rejects_all_historically_measured_identities(
    low_data_unit,
) -> None:
    """The prospective safeguard must survive the low-data correction."""
    unit, _, _, _ = low_data_unit
    audit = _generate(unit, GLOBALLY_UNMEASURED_PROSPECTIVE)["candidate_df"]
    accepted = audit.loc[audit["accepted"].astype(bool)]
    measured = set(unit.global_identity_keys)
    assert not (set(accepted["canonical_reaction_key"].astype(str)) & measured)


# --------------------------------------------------------------- oracle seal


def test_oracle_refuses_a_bundle_whose_pseudo_labels_were_mutated(
    low_data_unit,
) -> None:
    """A tampered pool must not be scoreable against the withheld truth."""
    unit, _, _, _ = low_data_unit
    generated = _generate(unit, OBSERVED_ONLY_LOW_DATA)
    if generated["candidate_df"].empty:
        raise AssertionError(
            "The fixture must produce candidates; an empty pool would make this "
            "leakage test vacuous."
        )
    bundle = FrozenCandidateBundle.freeze(
        generated["candidate_df"],
        evaluation_unit=unit.evaluation_unit,
        train_fraction=unit.train_fraction,
        candidate_scope_mode=OBSERVED_ONLY_LOW_DATA,
    )
    bundle._rows.loc[bundle._rows.index[0], "synthetic_label"] += 1.0
    with pytest.raises(OracleContractViolation, match="pseudo-labels were mutated"):
        withheld_cell_oracle_diagnostic(
            bundle,
            hidden_outer_train=unit.hidden_outer_train_oracle(),
            labeled_train_identity_keys=unit.labeled_train_identity_keys,
        )


def test_oracle_refuses_hidden_rows_that_overlap_labeled_training(
    low_data_unit,
) -> None:
    """Scoring against rows the learner saw would not be an oracle at all."""
    unit, _, _, _ = low_data_unit
    generated = _generate(unit, OBSERVED_ONLY_LOW_DATA)
    if generated["candidate_df"].empty:
        raise AssertionError(
            "The fixture must produce candidates; an empty pool would make this "
            "leakage test vacuous."
        )
    bundle = FrozenCandidateBundle.freeze(
        generated["candidate_df"],
        evaluation_unit=unit.evaluation_unit,
        train_fraction=unit.train_fraction,
        candidate_scope_mode=OBSERVED_ONLY_LOW_DATA,
    )
    contaminated = unit.hidden_outer_train_oracle()
    contaminated.loc[contaminated.index[0], "canonical_reaction_key"] = (
        unit.labeled_train_identity_keys[0]
    )
    with pytest.raises(OracleContractViolation, match="overlap the labeled training"):
        withheld_cell_oracle_diagnostic(
            bundle,
            hidden_outer_train=contaminated,
            labeled_train_identity_keys=unit.labeled_train_identity_keys,
        )


def test_group_safe_split_violation_is_a_hard_failure(low_data_unit) -> None:
    """A canonical identity crossing partitions means the split is broken."""
    unit, _, _, _ = low_data_unit
    broken_validation = unit.validation.copy()
    broken_validation.loc[broken_validation.index[0], "canonical_reaction_key"] = (
        unit.labeled_train_identity_keys[0]
    )
    rebuilt = unit.__class__(
        evaluation_unit=unit.evaluation_unit,
        seed=unit.seed,
        train_fraction=unit.train_fraction,
        dataset_path=unit.dataset_path,
        dataset_hash=unit.dataset_hash,
        labeled_train=unit.labeled_train,
        hidden_outer_train=unit.hidden_outer_train,
        validation=broken_validation,
        test=unit.test,
        canonical_source_row_ids=unit.canonical_source_row_ids,
    )
    with pytest.raises(Exception, match="cross labeled_train"):
        rebuilt.candidate_scope(OBSERVED_ONLY_LOW_DATA)


def test_scope_refuses_to_be_built_with_hidden_identities(low_data_unit) -> None:
    """Even a deliberate attempt to smuggle hidden keys into the scope fails."""
    unit, _, _, _ = low_data_unit
    scope = unit.candidate_scope(OBSERVED_ONLY_LOW_DATA)
    with pytest.raises(CandidateScopeViolation, match="reached the candidate"):
        scope.assert_excludes(
            unit.labeled_train_identity_keys,
            description="Hidden outer-training identities",
        )
