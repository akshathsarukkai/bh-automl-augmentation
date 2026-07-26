"""Tests separating raw duplicates, canonical replicates, and yield conflicts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.canonical_test_utils import audit_config, measured_role_frame

from bh_augmentation.data.audit_canonical_dataset import (
    _validate_audit_config,
    canonical_reaction_group_statistics,
    exact_duplicate_rows,
    run_canonical_data_audit,
)
from bh_augmentation.data.canonicalize_roles import canonicalize_reaction_roles_dataframe


def test_exact_and_canonical_duplicates_are_detected() -> None:
    canonical = canonicalize_reaction_roles_dataframe(
        measured_role_frame(),
        source_file_hash="f" * 64,
    )
    exact = exact_duplicate_rows(canonical)
    stats = canonical_reaction_group_statistics(canonical)
    duplicate_group = stats.loc[stats["replicate_count"] == 4].iloc[0]
    assert len(exact) == 2
    assert duplicate_group["n_unique_yields"] == 2


def test_replicate_yield_statistics_are_correct() -> None:
    canonical = canonicalize_reaction_roles_dataframe(
        measured_role_frame(),
        source_file_hash="1" * 64,
    )
    group = canonical_reaction_group_statistics(canonical).sort_values(
        "replicate_count", ascending=False
    ).iloc[0]
    assert group["replicate_count"] == 4
    assert group["yield_mean"] == pytest.approx(55.0)
    assert group["yield_median"] == pytest.approx(50.0)
    assert group["yield_min"] == 50.0
    assert group["yield_max"] == 70.0
    assert group["yield_range"] == 20.0


def test_audit_preserves_rows_and_does_not_aggregate_yields(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    canonical_path = tmp_path / "canonical.csv"
    output = tmp_path / "corrected_audit"
    measured_role_frame().to_csv(source, index=False)
    paths = run_canonical_data_audit(
        audit_config(source, canonical_path, output),
        config_path=tmp_path / "config.yaml",
    )
    result = pd.read_csv(paths["canonical_dataset"])
    summary = json.loads(paths["summary_json"].read_text())
    assert len(result) == len(measured_role_frame())
    assert result["yield"].tolist() == measured_role_frame()["yield"].tolist()
    assert summary["rows_removed"] == 0
    assert summary["yields_aggregated"] is False
    assert summary["n_replicate_groups"] == 1
    assert summary["n_yield_conflict_groups"] == 1


def test_existing_audit_outputs_are_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    canonical_path = tmp_path / "canonical.csv"
    output = tmp_path / "corrected_audit"
    measured_role_frame().to_csv(source, index=False)
    config = audit_config(source, canonical_path, output)
    run_canonical_data_audit(config, config_path=tmp_path / "config.yaml")
    with pytest.raises(FileExistsError):
        run_canonical_data_audit(config, config_path=tmp_path / "config.yaml")


def test_exact_duplicates_include_source_metadata() -> None:
    frame = measured_role_frame().iloc[[0, 3]].copy()
    frame["temperature"] = ["20", "80"]
    canonical = canonicalize_reaction_roles_dataframe(
        frame,
        source_file_hash="2" * 64,
    )
    assert exact_duplicate_rows(canonical).empty


def test_scientific_config_requires_isomeric_rdkit_role_features(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    config = audit_config(
        source,
        tmp_path / "canonical.csv",
        tmp_path / "corrected_audit",
    )
    config["canonicalization"]["isomeric_smiles"] = False
    with pytest.raises(ValueError, match="isomeric_smiles"):
        _validate_audit_config(config)
    config["canonicalization"]["isomeric_smiles"] = True
    config["features"]["fingerprint_backend"] = "hash"
    with pytest.raises(ValueError, match="fingerprint_backend"):
        _validate_audit_config(config)


def test_invalid_scientific_data_writes_complete_audit_but_no_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    canonical_path = tmp_path / "canonical.csv"
    output = tmp_path / "corrected_audit"
    frame = measured_role_frame().iloc[:1].copy()
    frame["recovered_base_smiles"] = "C1("
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="complete audit"):
        run_canonical_data_audit(
            audit_config(source, canonical_path, output),
            config_path=tmp_path / "config.yaml",
        )
    assert not canonical_path.exists()
    expected = {
        "canonicalization_summary.json",
        "invalid_molecules.csv",
        "invalid_reactions.csv",
        "canonical_role_value_counts.csv",
        "data_audit_report.md",
        "run_manifest.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["construction_failed"] is True


def test_existing_empty_audit_directory_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    measured_role_frame().to_csv(source, index=False)
    output = tmp_path / "corrected_empty_audit"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_canonical_data_audit(
            audit_config(source, tmp_path / "canonical.csv", output),
            config_path=tmp_path / "config.yaml",
        )
