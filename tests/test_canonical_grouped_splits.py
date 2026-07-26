"""Tests for deterministic canonical-reaction grouped outer splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.canonical_test_utils import (
    audit_config,
    grouped_assignment_frame,
    measured_role_frame,
    split_config,
)

from bh_augmentation.data.audit_canonical_dataset import run_canonical_data_audit
from bh_augmentation.data.canonical_splits import (
    _validate_canonical_dataset_provenance,
    build_grouped_outer_assignments,
    build_nested_low_data_assignments,
    run_canonical_grouped_splits,
    split_assignment_hash,
)
from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_to_record


def _outer(seed: int = 0) -> pd.DataFrame:
    return build_grouped_outer_assignments(
        grouped_assignment_frame(),
        seed=seed,
        train_size=0.8,
        valid_size=0.1,
        test_size=0.1,
    )


def test_canonical_groups_and_replicates_never_cross_outer_splits() -> None:
    assignments = _outer()
    assert assignments.groupby("canonical_reaction_key")["outer_split"].nunique().max() == 1
    assert assignments["source_row_id"].nunique() == len(assignments)
    assert set(assignments["outer_split"]) == {"train", "valid", "test"}


def test_outer_splits_are_deterministic_and_seed_dependent() -> None:
    first = _outer(0).sort_values("source_row_id")
    repeated = _outer(0).sort_values("source_row_id")
    different = _outer(1).sort_values("source_row_id")
    pd.testing.assert_frame_equal(first.reset_index(drop=True), repeated.reset_index(drop=True))
    assert not first["outer_split"].reset_index(drop=True).equals(
        different["outer_split"].reset_index(drop=True)
    )


def test_yield_values_do_not_affect_assignments() -> None:
    frame = grouped_assignment_frame()
    first = build_grouped_outer_assignments(
        frame, seed=2, train_size=0.8, valid_size=0.1, test_size=0.1
    )
    frame["yield"] = frame["yield"].iloc[::-1].to_numpy()
    second = build_grouped_outer_assignments(
        frame, seed=2, train_size=0.8, valid_size=0.1, test_size=0.1
    )
    pd.testing.assert_frame_equal(first, second)


def test_split_hash_is_stable_and_detects_tampering() -> None:
    outer = _outer()
    low = build_nested_low_data_assignments(
        outer, train_fractions=[0.01, 0.05, 0.1, 0.2, 1.0]
    )
    first = split_assignment_hash(outer, low)
    assert first == split_assignment_hash(outer.copy(), low.copy())
    tampered = low.copy()
    tampered.loc[0, "included_in_training_subset"] = not bool(
        tampered.loc[0, "included_in_training_subset"]
    )
    assert first != split_assignment_hash(outer, tampered)


def test_tiny_audit_and_split_functions_complete(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    canonical_path = tmp_path / "canonical.csv"
    records = []
    for index in range(12):
        roles = ReactionRoles(
            reactant_1=f"{'C' * (index + 1)}Br",
            reactant_2="CN",
            catalyst="[Pd]",
            ligand="CP(C)C",
            base="[Na+].[OH-]",
            solvent_or_additive="CCO",
            product=f"{'C' * (index + 2)}N",
        )
        record = reaction_roles_to_record(roles)
        record.update({"reaction_id": f"reaction_{index}", "yield": float(index)})
        records.append(record)
    pd.DataFrame(records).to_csv(source, index=False)
    run_canonical_data_audit(
        audit_config(source, canonical_path, tmp_path / "corrected_audit"),
        config_path=tmp_path / "audit.yaml",
    )
    config = split_config(
        source,
        canonical_path,
        tmp_path / "corrected_splits",
        seeds=[0, 1],
    )
    config["dataset"]["nrows"] = 10
    paths = run_canonical_grouped_splits(
        config,
        config_path=tmp_path / "splits.yaml",
    )
    outer = pd.read_csv(paths["outer_assignments"])
    assert outer.groupby(["seed", "canonical_reaction_key"])["outer_split"].nunique().max() == 1
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["canonical_rows_read"] == 10
    assert manifest["dataset_nrows"] == 10
    assert manifest["canonical_subset_hash"]
    with pytest.raises(FileExistsError):
        run_canonical_grouped_splits(
            split_config(source, canonical_path, tmp_path / "corrected_splits"),
            config_path=tmp_path / "splits.yaml",
        )


def test_large_replicate_group_is_allocated_without_extreme_imbalance() -> None:
    records = [
        {"source_row_id": f"large-{index}", "canonical_reaction_key": "large"}
        for index in range(20)
    ]
    records.extend(
        {
            "source_row_id": f"small-{index}",
            "canonical_reaction_key": f"small-{index}",
        }
        for index in range(9)
    )
    frame = pd.DataFrame(records)
    for seed in range(5):
        assignments = build_grouped_outer_assignments(
            frame,
            seed=seed,
            train_size=0.8,
            valid_size=0.1,
            test_size=0.1,
        )
        counts = assignments["outer_split"].value_counts()
        assert counts["train"] >= 20
        assert counts["valid"] < 10
        assert counts["test"] < 10
        assert (
            assignments.groupby("canonical_reaction_key")["outer_split"].nunique().max()
            == 1
        )


def test_split_sizes_must_be_strictly_positive() -> None:
    with pytest.raises(ValueError, match="valid_size"):
        build_grouped_outer_assignments(
            grouped_assignment_frame(),
            seed=0,
            train_size=0.9,
            valid_size=0.0,
            test_size=0.1,
        )


def test_tampered_canonical_key_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    canonical_path = tmp_path / "canonical.csv"
    measured_role_frame().to_csv(source, index=False)
    run_canonical_data_audit(
        audit_config(source, canonical_path, tmp_path / "corrected_audit"),
        config_path=tmp_path / "audit.yaml",
    )
    canonical = pd.read_csv(canonical_path)
    canonical.loc[0, "canonical_reaction_key"] = "tampered"
    canonical.to_csv(canonical_path, index=False)
    with pytest.raises(ValueError, match="do not match canonical roles"):
        _validate_canonical_dataset_provenance(canonical, source)


def test_existing_empty_split_directory_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    canonical_path = tmp_path / "canonical.csv"
    measured_role_frame().to_csv(source, index=False)
    run_canonical_data_audit(
        audit_config(source, canonical_path, tmp_path / "corrected_audit"),
        config_path=tmp_path / "audit.yaml",
    )
    output = tmp_path / "corrected_empty_splits"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_canonical_grouped_splits(
            split_config(source, canonical_path, output),
            config_path=tmp_path / "splits.yaml",
        )
