"""Summarize the matched condition-transfer plus supervised-AE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        default="results/condition_transfer_supervised_ae_xgboost",
        help="Hybrid result directory.",
    )
    args = parser.parse_args()
    directory = Path(args.directory)
    metrics_path = directory / "policy_metrics.csv"
    selected_path = directory / "selected_hybrid_policies.csv"
    if not metrics_path.exists() or not selected_path.exists():
        raise FileNotFoundError(
            f"Missing hybrid outputs under {directory}. Run the hybrid experiment first."
        )

    metrics = pd.read_csv(metrics_path)
    selected = pd.read_csv(selected_path)
    table = _rmse_table(metrics)
    stats = _paired_stats(directory)
    report_path = directory / "comparison_report.csv"
    table.to_csv(report_path, index=False)

    print("\nMATCHED TEST RMSE")
    print(table.to_string(index=False))
    if not stats.empty:
        print("\nPAIRED HYBRID COMPARISONS")
        print(stats.to_string(index=False))
    if not selected.empty:
        columns = [
            "seed",
            "train_fraction",
            "downstream_model",
            "latent_dim",
            "synthetic_example_weight",
            "condition_transfer_policy_id",
            "valid_rmse",
        ]
        print("\nSELECTED POLICIES")
        print(selected[[column for column in columns if column in selected]].to_string(index=False))
    print(f"\nSaved: {report_path}")
    print("Output files:")
    for path in sorted(directory.glob("*.csv")):
        print(path)


def _rmse_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[(metrics["split"] == "test") & (metrics["metric"] == "rmse")].copy()
    test = test.loc[(test["downstream_model"] == "xgboost")]
    labels = {
        "original_6144_real_only": "real_only",
        "anonymous_condition_transfer": "anonymous_condition_transfer",
        "full_data_original_6144": "full_data_xgboost",
    }
    test["method"] = test["representation"].map(labels)
    test.loc[
        test["representation"].astype(str).str.endswith("_real_only")
        & test["ae_only_selected"].fillna(False).astype(bool),
        "method",
    ] = "supervised_ae_only"
    test.loc[test["selected_policy"].fillna(False).astype(bool), "method"] = "hybrid"
    test = test.loc[test["method"].notna()]
    return (
        test.groupby(["train_fraction", "method"], dropna=False)["value"]
        .agg(rmse_mean="mean", rmse_std="std", n_seeds="count")
        .reset_index()
        .sort_values(["train_fraction", "rmse_mean"])
    )


def _paired_stats(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parent in ["real_only", "anonymous_transfer", "ae_only", "best_parent"]:
        path = directory / f"hybrid_vs_{parent}_by_seed.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for fraction, group in frame.loc[frame["metric"] == "rmse"].groupby("train_fraction"):
            hybrid = group["hybrid_value"].to_numpy(dtype=float)
            baseline = group["parent_value"].to_numpy(dtype=float)
            paired_t = _ttest(hybrid, baseline)
            wilcoxon_p = _wilcoxon(hybrid, baseline)
            rows.append(
                {
                    "train_fraction": fraction,
                    "parent": parent,
                    "hybrid_rmse": float(hybrid.mean()),
                    "parent_rmse": float(baseline.mean()),
                    "delta": float((hybrid - baseline).mean()),
                    "n_hybrid_better": int((hybrid < baseline).sum()),
                    "n_seeds": len(group),
                    "paired_t_pvalue": paired_t,
                    "wilcoxon_pvalue": wilcoxon_p,
                    "synergy_supported": bool(
                        parent == "best_parent"
                        and len(group) >= 2
                        and hybrid.mean() < baseline.mean()
                        and (hybrid < baseline).sum() > len(group) / 2
                        and np.isfinite(paired_t)
                        and paired_t < 0.05
                    ),
                }
            )
    return pd.DataFrame(rows)


def _ttest(left: np.ndarray, right: np.ndarray) -> float:
    return float(ttest_rel(left, right).pvalue) if len(left) >= 2 else float("nan")


def _wilcoxon(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.allclose(left, right):
        return float("nan")
    try:
        return float(wilcoxon(left, right).pvalue)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
