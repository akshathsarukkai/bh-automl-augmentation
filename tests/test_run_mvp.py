"""Tests for the complete MVP workflow runner."""

from pathlib import Path

import pandas as pd
import yaml

from bh_augmentation.run_mvp import run_mvp


def test_run_mvp_on_fixture_data_creates_outputs(tmp_path: Path) -> None:
    """The MVP runner should complete quickly on fixture data."""
    results_dir = tmp_path / "results" / "mvp"
    config_path = tmp_path / "mvp.yaml"
    config = {
        "seed": 21,
        "dataset": {
            "name": "fixture_bh",
            "use_fixture": True,
            "fixture_path": "tests/fixtures/sample_bh.csv",
            "fixture_min_rows": 12,
        },
        "splits": {
            "method": "random",
            "train_size": 0.6,
            "valid_size": 0.2,
            "test_size": 0.2,
        },
            "features": {
                "kind": "reaction_role_concat_delta",
                "n_bits": 64,
                "radius": 2,
        },
        "models": ["ridge"],
        "metrics": ["rmse", "mae"],
        "automl": {"enabled": False},
        "output": {"results_dir": str(results_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = run_mvp(config_path)

    assert outputs["prepared_data"].exists()
    assert outputs["baseline_metrics"].exists()
    assert outputs["low_data_metrics"].exists()
    assert outputs["augmentation_metrics"].exists()
    assert outputs["recommendation_metrics"].exists()
    assert outputs["report"].exists()

    assert (results_dir / "baseline" / "baseline_metrics.csv").exists()
    assert (results_dir / "low_data" / "baseline_metrics.csv").exists()
    assert (results_dir / "augmentation" / "safe_aug_metrics.csv").exists()
    assert (results_dir / "recommendation" / "topk_metrics.csv").exists()
    assert (results_dir / "final_report.md").exists()

    baseline = pd.read_csv(results_dir / "baseline" / "baseline_metrics.csv")
    augmentation = pd.read_csv(results_dir / "augmentation" / "safe_aug_metrics.csv")
    recommendation = pd.read_csv(results_dir / "recommendation" / "topk_metrics.csv")
    assert not baseline.empty
    assert not augmentation.empty
    assert set(recommendation["strategy"]) == {"random", "model"}
    assert "Buchwald-Hartwig Augmentation Report" in (results_dir / "final_report.md").read_text(
        encoding="utf-8"
    )
