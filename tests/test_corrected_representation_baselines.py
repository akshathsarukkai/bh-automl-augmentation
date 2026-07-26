"""Tests for corrected real-only representation baselines."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from scripts.build_corrected_feature_contract_report import build_report

from bh_augmentation.run_corrected_representation_baselines import (
    REQUIRED_REPRESENTATIONS,
    run_corrected_representation_baselines,
)
from corrected_test_utils import write_corrected_config


def test_production_representation_config_contains_all_canonical_kinds() -> None:
    config = yaml.safe_load(
        Path("configs/corrected_representation_baselines_xgboost.yaml").read_text()
    )
    assert tuple(config["features"]["representations"]) == REQUIRED_REPRESENTATIONS
    assert config["features"]["fingerprint_backend"] == "rdkit"
    assert config["features"]["n_bits"] == 2048


def test_tiny_representation_baseline_completes(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    config_path = write_corrected_config(
        tmp_path, kind="representations", output_name="corrected_representation_tiny"
    )

    paths = run_corrected_representation_baselines(config_path)

    required = {
        "policy_metrics",
        "summary",
        "representation_metadata_csv",
        "representation_metadata_json",
        "split_audit",
        "run_manifest",
    }
    assert all(paths[key].is_file() for key in required)
    metrics = pd.read_csv(paths["policy_metrics"])
    split_audit = pd.read_csv(paths["split_audit"])
    manifest = json.loads(paths["run_manifest"].read_text())
    assert set(metrics["representation"]) == set(REQUIRED_REPRESENTATIONS)
    assert set(metrics["split"]) == {"valid", "test"}
    assert metrics["feature_metadata_hash"].notna().all()
    assert metrics["split_hash"].nunique() == 1
    contract = manifest["canonical_split_contract"]
    assert split_audit["split_hash"].eq(contract["per_seed_split_hashes"]["0"]).all()
    assert split_audit["split_aggregate_hash"].eq(contract["split_aggregate_hash"]).all()
    assert split_audit["source_id_split_hash"].notna().all()


def test_tampered_saved_split_fails_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_corrected_config(
        tmp_path,
        kind="representations",
        output_name="corrected_representation_tamper",
    )
    config = yaml.safe_load(config_path.read_text())
    low_path = (
        Path(config["splits"]["directory"]) / "low_data_subset_assignments.csv"
    )
    low = pd.read_csv(low_path)
    low.loc[0, "included_in_training_subset"] = not bool(
        low.loc[0, "included_in_training_subset"]
    )
    low.to_csv(low_path, index=False)

    def _unexpected_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("model training ran before canonical split validation")

    monkeypatch.setattr(
        "bh_augmentation.run_corrected_representation_baselines.train_model",
        _unexpected_training,
    )
    with pytest.raises(ValueError, match="split hash mismatch|training subset"):
        run_corrected_representation_baselines(config_path)


def test_tiny_representation_widths_follow_contract(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    config_path = write_corrected_config(
        tmp_path, kind="representations", output_name="corrected_representation_widths"
    )
    paths = run_corrected_representation_baselines(config_path)
    metadata = pd.read_csv(paths["representation_metadata_csv"]).set_index(
        "representation_kind"
    )

    assert metadata.loc["reaction_section_concat", "total_width"] == 48
    assert metadata.loc["reaction_section_concat_delta", "total_width"] == 64
    assert metadata.loc["bh_role_separated", "total_width"] == 112
    assert metadata.loc["bh_role_separated_delta", "total_width"] == 160


def test_feature_contract_report_reproduces_identity_and_locality(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    paths = build_report(tmp_path / "corrected_feature_contract")

    identity = pd.read_csv(paths["identity"])
    locality = pd.read_csv(paths["locality"])
    assert identity["bit_identical"].all()
    assert identity["feature_names_identical"].all()
    assert identity["metadata_identical"].all()
    assert locality["only_requested_blocks_changed"].all()
