"""Tests for strict consumption of saved canonical split assignments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.canonical_test_utils import audit_config, split_config

from bh_augmentation.data.audit_canonical_dataset import run_canonical_data_audit
from bh_augmentation.data.canonical_splits import (
    run_canonical_grouped_splits,
    split_assignment_hash,
)
from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_to_record
from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
    load_saved_canonical_splits,
)
from bh_augmentation.utils.corrected_runs import stable_hash


def _saved_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("rdkit")
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.csv"
    canonical = tmp_path / "canonical.csv"
    records = []
    for index in range(18):
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
        audit_config(source, canonical, tmp_path / "corrected_audit"),
        config_path=tmp_path / "audit.yaml",
    )
    directory = tmp_path / "corrected_splits"
    config = split_config(source, canonical, directory, seeds=[0, 1])
    config["splits"]["train_fractions"] = [0.1, 0.5, 1.0]
    run_canonical_grouped_splits(config, config_path=tmp_path / "splits.yaml")
    return canonical, directory


def _rewrite_manifest_hashes(directory: Path) -> None:
    manifest_path = directory / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outer = pd.read_csv(directory / "outer_split_assignments.csv")
    low = pd.read_csv(directory / "low_data_subset_assignments.csv")
    hashes = {
        str(seed): split_assignment_hash(
            outer.loc[outer["seed"].eq(seed)],
            low.loc[low["seed"].eq(seed)],
        )
        for seed in manifest["seeds"]
    }
    manifest["split_hashes"] = hashes
    manifest["split_hash"] = stable_hash(hashes)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_loader_validates_and_materializes_preserving_indices(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    saved = load_saved_canonical_splits(
        canonical,
        directory,
        requested_seeds=[0, 1],
        requested_fractions=[0.1, 1.0],
    )

    frame = saved.canonical.copy()
    frame.index = pd.Index(range(100, 100 + len(frame)), name="scientific_index")
    variants = saved.materialize_variants(frame, seed=0, train_fractions=[0.1, 1.0])

    assert len(variants) == 2
    assert variants[0][1]["valid"].index.equals(variants[1][1]["valid"].index)
    assert variants[0][1]["test"].index.equals(variants[1][1]["test"].index)
    assert set(variants[0][1]["train"].index).issubset(variants[1][1]["train"].index)
    assert all(split.index.min() >= 100 for _, splits in variants for split in splits.values())
    audit = saved.audit_record(seed=0, train_fraction=0.1)
    assert audit["canonical_dataset_hash"] == saved.dataset_hash
    assert audit["split_aggregate_hash"] == saved.aggregate_split_hash
    assert audit["per_seed_split_hash"] == saved.per_seed_split_hashes[0]
    assert len(audit["source_id_split_hash"]) == 64


def test_tampered_assignment_hash_is_rejected(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    low_path = directory / "low_data_subset_assignments.csv"
    low = pd.read_csv(low_path)
    low.loc[0, "included_in_training_subset"] = not bool(low.loc[0, "included_in_training_subset"])
    low.to_csv(low_path, index=False)

    with pytest.raises(ValueError, match="per-seed split hash mismatch|training subset"):
        load_saved_canonical_splits(canonical, directory)


def test_semantic_nesting_is_checked_even_with_recomputed_hashes(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    low_path = directory / "low_data_subset_assignments.csv"
    low = pd.read_csv(low_path)
    seed_rows = low["seed"].eq(0)
    at_half = seed_rows & low["train_fraction"].eq(0.5)
    at_full = seed_rows & low["train_fraction"].eq(1.0)
    candidate_id = low.loc[at_half & low["included_in_training_subset"], "source_row_id"].iloc[-1]
    low.loc[
        at_full & low["source_row_id"].eq(candidate_id),
        "included_in_training_subset",
    ] = False
    low.to_csv(low_path, index=False)
    _rewrite_manifest_hashes(directory)

    with pytest.raises(ValueError, match="not cumulative|complete outer train"):
        load_saved_canonical_splits(canonical, directory)


def test_low_data_order_must_agree_with_outer_even_when_unhashed(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    low_path = directory / "low_data_subset_assignments.csv"
    low = pd.read_csv(low_path)
    low.loc[0, "outer_group_order"] = int(low.loc[0, "outer_group_order"]) + 1
    low.to_csv(low_path, index=False)

    with pytest.raises(ValueError, match="outer_group_order does not agree"):
        load_saved_canonical_splits(canonical, directory)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("seed", "0.0", "exact integers"),
        ("outer_split", "validation", "unsupported value"),
        ("included_in_training_subset", "true", "exactly 'True' or 'False'"),
    ],
)
def test_assignment_types_and_enums_are_strict(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    outer_path = directory / "outer_split_assignments.csv"
    outer = pd.read_csv(outer_path, dtype=str, keep_default_na=False)
    outer.loc[0, column] = value
    outer.to_csv(outer_path, index=False)

    with pytest.raises(ValueError, match=message):
        load_saved_canonical_splits(canonical, directory)


def test_assignment_schema_and_coverage_are_exact(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    outer_path = directory / "outer_split_assignments.csv"
    outer = pd.read_csv(outer_path)
    outer["unreviewed_field"] = "value"
    outer.to_csv(outer_path, index=False)

    with pytest.raises(ValueError, match="assignment schema mismatch"):
        load_saved_canonical_splits(canonical, directory)

    outer.drop(columns="unreviewed_field").iloc[:-1].to_csv(outer_path, index=False)
    with pytest.raises(ValueError, match="exact canonical row coverage"):
        load_saved_canonical_splits(canonical, directory)


def test_dataset_and_manifest_versions_are_verified(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    canonical_frame = pd.read_csv(canonical)
    canonical_frame.loc[0, "yield"] = 99.0
    canonical_frame.to_csv(canonical, index=False)
    with pytest.raises(ValueError, match="dataset hash"):
        load_saved_canonical_splits(canonical, directory)

    canonical, directory = _saved_split_fixture(tmp_path / "version_case")
    manifest_path = directory / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_schema_version"] = "future-unreviewed-schema"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema version mismatch"):
        load_saved_canonical_splits(canonical, directory)


def test_missing_requested_seed_fraction_and_materialization_rows_fail(
    tmp_path: Path,
) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    with pytest.raises(ValueError, match="seeds are absent"):
        load_saved_canonical_splits(canonical, directory, requested_seeds=[7])
    with pytest.raises(ValueError, match="fraction is absent"):
        load_saved_canonical_splits(canonical, directory, requested_fractions=[0.2])

    saved = load_saved_canonical_splits(canonical, directory)
    incomplete = saved.canonical.iloc[:-1]
    with pytest.raises(ValueError, match="does not exactly cover"):
        saved.materialize_variants(incomplete, seed=0, train_fractions=[0.1])


def test_identity_only_loader_does_not_expose_yields(tmp_path: Path) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    saved = load_saved_canonical_split_identities(canonical, directory)
    assert "yield" not in saved.canonical
    assert saved.dataset_hash


def test_identity_only_loader_never_reads_yield_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, directory = _saved_split_fixture(tmp_path)
    original = pd.read_csv
    canonical_reads: list[dict[str, object]] = []

    def _observed_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if Path(path) == canonical:
            canonical_reads.append(dict(kwargs))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(
        "bh_augmentation.data.saved_canonical_splits.pd.read_csv",
        _observed_read,
    )
    load_saved_canonical_split_identities(canonical, directory)
    value_reads = [
        call for call in canonical_reads if call.get("nrows") != 0
    ]
    assert len(value_reads) == 1
    assert "yield" not in value_reads[0]["usecols"]
