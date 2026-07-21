"""Tests for corrected run manifests and output isolation."""

from pathlib import Path

import pytest

from bh_augmentation.results.status import InvalidResultError, assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    build_run_manifest,
    prepare_fresh_output_directory,
    stable_hash,
)


def test_manifest_records_required_reproducibility_fields(tmp_path: Path) -> None:
    config = {
        "seed": 0,
        "seeds": [0],
        "low_data": {"train_fractions": [0.01]},
    }
    manifest = build_run_manifest(
        config=config,
        config_path="configs/corrected_tiny.yaml",
        dataset_path="data.csv",
        dataset_hash="dataset",
        output_directory=tmp_path / "corrected_output",
        split_hashes={"split": "hash"},
        feature_metadata_hash="feature",
    )

    required = {
        "run_id",
        "timestamp",
        "git_commit",
        "git_dirty",
        "dataset_path",
        "dataset_hash",
        "resolved_config",
        "config_hash",
        "split_hashes",
        "feature_metadata_hash",
        "dependency_versions",
        "command",
        "seeds",
        "train_fractions",
        "output_directory",
        "historical_results_loaded",
    }
    assert required <= set(manifest)
    assert manifest["historical_results_loaded"] is False


def test_stable_hash_is_order_independent_for_mappings() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_corrected_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "corrected_results"
    output.mkdir()
    (output / "run_manifest.json").write_text("existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_fresh_output_directory(output)


def test_invalid_historical_directories_remain_rejected() -> None:
    with pytest.raises(InvalidResultError, match="RESULT_STATUS.md"):
        assert_result_directory_allowed(
            "results/role_aware_condition_transfer_v2_xgboost"
        )


def test_corrected_role_aware_directory_is_allowed_but_does_not_mask_nested_history() -> None:
    assert_result_directory_allowed(
        "results/corrected_role_aware_condition_transfer_xgboost"
    )
    with pytest.raises(InvalidResultError):
        assert_result_directory_allowed(
            "results/corrected_archive/role_aware_condition_transfer_v2_xgboost"
        )
