"""Tests for corrected paired comparison outputs and interpretation."""

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.compare_corrected_condition_transfer import (
    compare_corrected_condition_transfer,
)


def test_tiny_corrected_comparison_completes(tmp_path: Path) -> None:
    anonymous = tmp_path / "corrected_anonymous"
    role = tmp_path / "corrected_role"
    output = tmp_path / "corrected_comparison"
    _write_run(anonymous, selected_value=9.0, transfer="anonymous")
    _write_run(role, selected_value=8.0, transfer="role_aware")

    paths = compare_corrected_condition_transfer(anonymous, role, output)

    assert all(path.is_file() for key, path in paths.items() if key != "directory")
    decision = pd.read_csv(paths["decision_table"])
    assert decision.loc[0, "role_aware_minus_anonymous_rmse"] == pytest.approx(-1.0)
    assert decision.loc[0, "role_aware_better_seeds"] == 1
    assert "numerically lower" in paths["report"].read_text()


def test_comparison_refuses_to_overwrite_output(tmp_path: Path) -> None:
    anonymous = tmp_path / "corrected_anonymous"
    role = tmp_path / "corrected_role"
    output = tmp_path / "corrected_comparison"
    _write_run(anonymous, selected_value=9.0, transfer="anonymous")
    _write_run(role, selected_value=8.0, transfer="role_aware")
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compare_corrected_condition_transfer(anonymous, role, output)


def _write_run(directory: Path, *, selected_value: float, transfer: str) -> None:
    directory.mkdir()
    baseline_rows = []
    selected_rows = []
    for metric, baseline, selected in [
        ("rmse", 10.0, selected_value),
        ("mae", 8.0, selected_value - 2.0),
        ("r2", 0.1, 0.2),
        ("spearman", 0.2, 0.3),
    ]:
        baseline_rows.append(_row(metric, baseline, "bh_role_separated_real_only", False, 0))
        selected_rows.append(
            _row(metric, selected, f"corrected_{transfer}_condition_transfer", True, 2)
        )
    pd.DataFrame([*baseline_rows, *selected_rows]).to_csv(
        directory / "policy_metrics.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(directory / "selected_policy_metrics.csv", index=False)
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "historical_results_loaded": False,
                "result_status": "corrected_revalidation",
                "dataset_hash": "dataset",
                "feature_metadata_hash": "feature",
                "split_hashes": {"seed=0|train_fraction=0.1": "split"},
            }
        )
    )


def _row(
    metric: str,
    value: float,
    representation: str,
    selected: bool,
    n_synthetic: int,
) -> dict[str, object]:
    return {
        "seed": 0,
        "train_fraction": 0.1,
        "model": "xgboost",
        "split": "test",
        "metric": metric,
        "representation": representation,
        "value": value,
        "feature_metadata_hash": "feature",
        "split_hash": "split",
        "dataset_hash": "dataset",
        "selected_policy": selected,
        "n_synthetic_train": n_synthetic,
        "result_status": "corrected_revalidation",
    }
