"""Tests for immutable, common representation benchmark split identities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import bh_augmentation.evaluation.representation_splits as representation_splits
from bh_augmentation.data.canonicalize_roles import CANONICALIZATION_VERSION
from bh_augmentation.data.saved_canonical_splits import SavedCanonicalSplits
from bh_augmentation.evaluation.chemical_ood_artifacts import (
    ASSIGNMENT_COLUMNS,
    CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION,
    FOLD_COLUMNS,
)
from bh_augmentation.evaluation.logo_protocol import build_canonical_logo_contract
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_canonical_logo_split_units,
    build_saved_random_split_unit,
    load_chemical_ood_split_units,
    load_corrected_logo_split_units,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash


def _hash(value: str) -> str:
    return stable_hash(value)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": [f"row-{index}" for index in range(6)],
            "canonical_reaction_key": [f"key-{index}" for index in range(6)],
            "canonical_product_key": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "canonical_substrate_key": ["s1", "s1", "s2", "s2", "s3", "s3"],
        }
    )


def _saved(canonical: pd.DataFrame | None = None) -> SavedCanonicalSplits:
    frame = _canonical() if canonical is None else canonical.copy()
    outer_split = ["train", "train", "train", "train", "valid", "test"]
    included = [True, True, False, False, False, False]
    low = pd.DataFrame(
        {
            "source_row_id": frame["source_row_id"].astype(str),
            "canonical_reaction_key": frame["canonical_reaction_key"].astype(str),
            "seed": 7,
            "outer_split": outer_split,
            "outer_group_order": list(range(len(frame))),
            "train_fraction": 0.5,
            "included_in_training_subset": included,
        }
    )
    outer = low.copy()
    outer["train_fraction"] = 1.0
    outer["included_in_training_subset"] = outer["outer_split"].eq("train")
    dataset_hash = _hash("dataset")
    aggregate_hash = _hash("aggregate")
    return SavedCanonicalSplits(
        canonical=frame,
        outer_assignments=outer,
        low_data_assignments=low,
        manifest={
            "canonical_dataset_hash": dataset_hash,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "canonical_subset_hash": _hash("subset"),
            "split_schema_version": "random-split-schema",
            "train_fractions": [0.5],
        },
        dataset_hash=dataset_hash,
        aggregate_split_hash=aggregate_hash,
        per_seed_split_hashes={7: _hash("seed-7")},
        split_directory=Path("validated-splits"),
        artifact_hashes={},
    )


def test_saved_random_unit_uses_exact_fraction_membership_hash_and_accounts_for_all_rows(
) -> None:
    saved = _saved()

    unit = build_saved_random_split_unit(saved, seed=7, train_fraction=0.5)

    assert unit.train_source_ids == ("row-0", "row-1")
    assert unit.validation_source_ids == ("row-4",)
    assert unit.test_source_ids == ("row-5",)
    assert unit.excluded_source_ids == ("row-2", "row-3")
    assert unit.all_source_ids == tuple(sorted(saved.canonical["source_row_id"]))
    assert unit.exact_split_hash == saved.source_id_split_hash(
        seed=7, train_fraction=0.5
    )
    assert unit.upstream_split_hash == saved.per_seed_split_hashes[7]
    assert unit.audit_record["test_source_id_hash"] == stable_hash(["row-5"])


def test_saved_random_rejects_canonical_key_crossing_partitions() -> None:
    canonical = _canonical()
    canonical.loc[5, "canonical_reaction_key"] = "key-0"

    with pytest.raises(ValueError, match="canonical reaction key crosses"):
        build_saved_random_split_unit(
            _saved(canonical), seed=7, train_fraction=0.5
        )


def test_logo_units_rebuild_every_canonical_group_with_exact_contract_hashes() -> None:
    saved = _saved()
    expected = build_canonical_logo_contract(
        saved.canonical, target="product_key"
    )

    units = build_canonical_logo_split_units(saved, target="product_key")

    assert len(units) == 2
    assert [unit.heldout_group for unit in units] == ["p1", "p2"]
    assert [unit.exact_split_hash for unit in units] == [
        fold.assignment_hash for fold in expected.folds
    ]
    assert all(
        unit.aggregate_assignment_hash == expected.aggregate_assignment_hash
        and unit.canonical_split_dependency_hash == saved.aggregate_split_hash
        and not unit.validation_source_ids
        and not unit.excluded_source_ids
        for unit in units
    )


def _write_corrected_logo_root(root: Path, saved: SavedCanonicalSplits) -> None:
    target = "product_key"
    target_directory = root / target
    target_directory.mkdir(parents=True)
    contract = build_canonical_logo_contract(saved.canonical, target=target)
    assignments_path = target_directory / "fold_assignments.csv"
    contract.assignments.to_csv(assignments_path, index=False)
    manifest = {
        "schema_version": "bh-canonical-logo-v1",
        "status": "corrected_revalidation",
        "split_method": "leave_one_group_out",
        "logo_target": target,
        "expected_group_count": contract.expected_group_count,
        "observed_fold_count": contract.fold_count,
        "heldout_groups": [fold.fold_group for fold in contract.folds],
        "aggregate_assignment_hash": contract.aggregate_assignment_hash,
        "per_fold_assignment_hashes": contract.per_fold_assignment_hashes,
        "dataset": {
            "hash": saved.dataset_hash,
            "n_rows": len(saved.canonical),
        },
        "canonical_split_dependency": {
            "aggregate_split_hash": saved.aggregate_split_hash,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "split_schema_version": saved.manifest["split_schema_version"],
            "artifact_hashes": saved.artifact_hashes,
            "per_seed_split_hashes": {
                str(seed): split_hash
                for seed, split_hash in saved.per_seed_split_hashes.items()
            },
        },
        "output_hashes": {
            "fold_assignments.csv": sha256_file(assignments_path),
        },
    }
    manifest["manifest_payload_hash"] = stable_hash(manifest)
    manifest_path = target_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    completion = {
        "schema_version": "bh-canonical-logo-v1",
        "status": "complete",
        "expected_targets": [target],
        "observed_targets": [target],
        "expected_fold_counts": {target: contract.expected_group_count},
        "observed_fold_counts": {target: contract.fold_count},
        "target_outputs": {
            target: {
                "fold_count": contract.fold_count,
                "aggregate_assignment_hash": contract.aggregate_assignment_hash,
                "manifest_hash": sha256_file(manifest_path),
                "artifact_hashes": {
                    "fold_assignments.csv": sha256_file(assignments_path),
                    "manifest.json": sha256_file(manifest_path),
                },
            }
        },
    }
    completion["completion_payload_hash"] = stable_hash(completion)
    (root / "completion_manifest.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def _rehash_corrected_logo_root(root: Path) -> None:
    target_directory = root / "product_key"
    assignments_path = target_directory / "fold_assignments.csv"
    manifest_path = target_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_hashes"]["fold_assignments.csv"] = sha256_file(
        assignments_path
    )
    manifest.pop("manifest_payload_hash")
    manifest["manifest_payload_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = root / "completion_manifest.json"
    completion = json.loads(completion_path.read_text())
    outputs = completion["target_outputs"]["product_key"]
    outputs["manifest_hash"] = sha256_file(manifest_path)
    outputs["artifact_hashes"]["fold_assignments.csv"] = sha256_file(
        assignments_path
    )
    outputs["artifact_hashes"]["manifest.json"] = sha256_file(manifest_path)
    completion.pop("completion_payload_hash")
    completion["completion_payload_hash"] = stable_hash(completion)
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def test_corrected_logo_loader_verifies_evidence_and_matches_identity_only_rebuild(
    tmp_path: Path,
) -> None:
    saved = _saved()
    assert "yield" not in saved.canonical
    _write_corrected_logo_root(tmp_path, saved)

    loaded = load_corrected_logo_split_units(
        saved,
        logo_root_directory=tmp_path,
        target="product_key",
    )
    rebuilt = build_canonical_logo_split_units(saved, target="product_key")

    assert loaded == rebuilt


def test_corrected_logo_loader_rejects_unhashed_assignment_tamper(
    tmp_path: Path,
) -> None:
    saved = _saved()
    _write_corrected_logo_root(tmp_path, saved)
    path = tmp_path / "product_key" / "fold_assignments.csv"
    assignments = pd.read_csv(path)
    train_index = assignments.loc[
        assignments["outer_split"].eq("train")
    ].index[0]
    assignments.loc[train_index, "outer_split"] = "test"
    assignments.to_csv(path, index=False)

    with pytest.raises(ValueError, match="output hash mismatch"):
        load_corrected_logo_split_units(
            saved,
            logo_root_directory=tmp_path,
            target="product_key",
        )


def test_corrected_logo_loader_rejects_fully_rehashed_membership_tamper(
    tmp_path: Path,
) -> None:
    saved = _saved()
    _write_corrected_logo_root(tmp_path, saved)
    path = tmp_path / "product_key" / "fold_assignments.csv"
    assignments = pd.read_csv(path)
    fold_zero = assignments["fold_index"].eq(0)
    original_test = assignments.loc[
        fold_zero & assignments["outer_split"].eq("test")
    ].index[0]
    original_train = assignments.loc[
        fold_zero & assignments["outer_split"].eq("train")
    ].index[0]
    assignments.loc[original_test, "outer_split"] = "train"
    assignments.loc[original_train, "outer_split"] = "test"
    assignments.to_csv(path, index=False)
    _rehash_corrected_logo_root(tmp_path)

    with pytest.raises(ValueError, match="do not hold out exactly"):
        load_corrected_logo_split_units(
            saved,
            logo_root_directory=tmp_path,
            target="product_key",
        )


def test_evaluation_unit_rejects_unsorted_or_mutable_membership() -> None:
    kwargs = {
        "evaluation_unit": "unit",
        "split_family": "family",
        "split_method": "method",
        "target": "target",
        "fold_index": 0,
        "heldout_group": "group",
        "seed": None,
        "train_fraction": None,
        "validation_source_ids": (),
        "test_source_ids": ("row-2",),
        "excluded_source_ids": (),
        "dataset_hash": _hash("dataset"),
        "exact_split_hash": _hash("split"),
        "aggregate_assignment_hash": _hash("aggregate"),
        "canonical_split_dependency_hash": _hash("dependency"),
        "upstream_split_hash": None,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "split_schema_version": "schema",
    }
    with pytest.raises(ValueError, match="stably sorted"):
        EvaluationSplitUnit(
            **kwargs, train_source_ids=("row-1", "row-0")
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        EvaluationSplitUnit(**kwargs, train_source_ids=["row-0"])  # type: ignore[arg-type]


def _write_chemical_tables(directory: Path, *, permute_key: bool = False) -> None:
    canonical = _canonical()
    records = []
    for family in ("maximum_similarity_bounded", "amine_scaffold"):
        for index, row in canonical.iterrows():
            reaction_key = row["canonical_reaction_key"]
            if permute_key and family == "maximum_similarity_bounded" and index == 0:
                reaction_key = "key-1"
            records.append(
                {
                    "family": family,
                    "source_row_id": row["source_row_id"],
                    "canonical_reaction_key": reaction_key,
                    "group_value": (
                        "train"
                        if family == "maximum_similarity_bounded" and index < 4
                        else (
                            "test"
                            if family == "maximum_similarity_bounded"
                            else "amine-core"
                        )
                    ),
                    "eligible": True,
                    "exclusion_reason": None,
                    "assignment_provenance": "{}",
                }
            )
    pd.DataFrame(records, columns=ASSIGNMENT_COLUMNS).to_csv(
        directory / "group_assignments.csv", index=False
    )
    folds = [
        {
            "family": "maximum_similarity_bounded",
            "fold_index": 0,
            "heldout_group": "similarity-bounded-test",
            "status": "included",
            "exclusion_reason": None,
            "n_train_rows": 4,
            "n_test_rows": 2,
            "n_train_groups": 4,
            "n_test_groups": 2,
            "group_overlap_count": 0,
            "canonical_reaction_key_overlap_count": 0,
            "split_hash": _hash("bounded-split"),
            "upstream_split_hash": _hash("bounded-upstream"),
            "configured_similarity_bound": 0.95,
        },
        {
            "family": "amine_scaffold",
            "fold_index": 0,
            "heldout_group": "__target_excluded__",
            "status": "excluded",
            "exclusion_reason": "insufficient_distinct_scaffold_groups",
            "n_train_rows": 0,
            "n_test_rows": 0,
            "n_train_groups": 0,
            "n_test_groups": 1,
            "group_overlap_count": 0,
            "canonical_reaction_key_overlap_count": 0,
            "split_hash": _hash("amine-excluded"),
            "upstream_split_hash": None,
            "configured_similarity_bound": None,
        },
    ]
    pd.DataFrame(folds, columns=FOLD_COLUMNS).to_csv(
        directory / "fold_definitions.csv", index=False
    )


def _patch_chemical_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    saved: SavedCanonicalSplits,
) -> list[Path]:
    validated: list[Path] = []

    def validate(directory: str | Path, **_: object) -> dict[str, object]:
        validated.append(Path(directory))
        return {
            "schema_version": CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION,
            "manifest_hash": _hash("chemical-manifest"),
            "dataset_hash": saved.dataset_hash,
            "canonical_split_dependency_hash": saved.aggregate_split_hash,
            "canonicalization_version": CANONICALIZATION_VERSION,
        }

    monkeypatch.setattr(
        representation_splits, "validate_chemical_ood_artifacts", validate
    )
    monkeypatch.setattr(
        representation_splits,
        "load_saved_canonical_split_identities",
        lambda *_args, **_kwargs: saved,
    )
    return validated


def test_chemical_ood_factory_strongly_validates_then_replays_only_included_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = _saved()
    _write_chemical_tables(tmp_path)
    validated = _patch_chemical_dependencies(monkeypatch, saved)

    units = load_chemical_ood_split_units(
        tmp_path,
        dataset_path="canonical.csv",
        canonical_split_directory="splits",
        families=["maximum_similarity_bounded", "amine_scaffold"],
    )

    assert validated == [tmp_path]
    assert len(units) == 1
    unit = units[0]
    assert unit.target == "maximum_similarity_bounded"
    assert unit.train_source_ids == ("row-0", "row-1", "row-2", "row-3")
    assert unit.test_source_ids == ("row-4", "row-5")
    assert unit.exact_split_hash == _hash("bounded-split")
    assert unit.aggregate_assignment_hash == _hash("chemical-manifest")
    assert unit.split_schema_version == CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION


def test_chemical_ood_factory_rejects_mapping_changed_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = _saved()
    _write_chemical_tables(tmp_path, permute_key=True)
    _patch_chemical_dependencies(monkeypatch, saved)

    with pytest.raises(ValueError, match="canonical reaction key crosses|mapping"):
        load_chemical_ood_split_units(
            tmp_path,
            dataset_path="canonical.csv",
            canonical_split_directory="splits",
            families=["maximum_similarity_bounded"],
        )


def test_dependency_versions_are_hard_requirements() -> None:
    saved = _saved()
    saved.manifest["canonicalization_version"] = "noncanonical"
    with pytest.raises(ValueError, match="canonicalization dependency"):
        build_saved_random_split_unit(saved, seed=7, train_fraction=0.5)
