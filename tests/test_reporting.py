"""Tests for Markdown and plot reporting."""

from pathlib import Path

import pandas as pd

from bh_augmentation.reporting.make_report import generate_markdown_report
from bh_augmentation.run_report import run_report


def test_generate_report_with_partial_results(tmp_path: Path) -> None:
    """Report generation should tolerate missing result files."""
    results_dir = tmp_path / "results"
    baseline_dir = results_dir / "baseline"
    baseline_dir.mkdir(parents=True)
    output_path = results_dir / "final_report.md"
    pd.DataFrame(
        {
            "train_fraction": [0.5, 1.0],
            "split_method": ["random", "random"],
            "group_column": ["", ""],
            "model": ["ridge", "ridge"],
            "split": ["test", "test"],
            "metric": ["rmse", "rmse"],
            "value": [12.0, 8.0],
        }
    ).to_csv(baseline_dir / "baseline_metrics.csv", index=False)

    report_path = generate_markdown_report(results_dir, output_path)

    assert report_path == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "Baseline Metrics Summary" in text
    assert "No recommendation metrics found." in text
    assert (results_dir / "plots" / "train_fraction_vs_rmse.png").exists()


def test_run_report_creates_markdown_and_available_plots(tmp_path: Path) -> None:
    """CLI wrapper should create report and plots from available result CSVs."""
    results_dir = tmp_path / "results"
    recommendation_dir = results_dir / "recommendation"
    recommendation_dir.mkdir(parents=True)
    output_path = results_dir / "final_report.md"
    pd.DataFrame(
        {
            "strategy": ["random", "model", "augmented_order_permutation"],
            "model": ["", "ridge", "ridge"],
            "k": [2, 2, 2],
            "top_k_hit_rate": [0.0, 0.5, 1.0],
            "regret": [40.0, 20.0, 0.0],
            "experiments_to_first_hit": [None, 2, 1],
            "predicted_yields": ["[0.1, 0.2]", "[55.0, 70.0]", "[80.0, 90.0]"],
            "true_yields": ["[10.0, 20.0]", "[60.0, 75.0]", "[85.0, 95.0]"],
        }
    ).to_csv(recommendation_dir / "topk_metrics.csv", index=False)

    report_path = run_report(results_dir, output_path)

    assert report_path == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "Recommendation Simulation Summary" in text
    assert "Augmentation helped" in text
    assert (results_dir / "plots" / "augmentation_type_vs_topk_hit_rate.png").exists()
    assert (results_dir / "plots" / "predicted_vs_true_yield.png").exists()
