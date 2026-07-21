#!/usr/bin/env python
"""Compare matched corrected anonymous and role-aware condition transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from bh_augmentation.results.status import assert_result_directory_allowed, read_result_csv
from bh_augmentation.utils.corrected_runs import (
    CORRECTED_STATUS,
    prepare_fresh_output_directory,
    stable_hash,
    write_json,
)

BASELINE_REPRESENTATION = "bh_role_separated_real_only"
KEYS = ["seed", "train_fraction", "model", "split", "metric"]


def assert_matched_real_only_baselines(
    anonymous_metrics: pd.DataFrame,
    role_aware_metrics: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Require exact provenance and numerical agreement for matched baselines."""
    anonymous = _baseline_rows(anonymous_metrics).rename(
        columns={
            "value": "anonymous_value",
            "feature_metadata_hash": "anonymous_feature_metadata_hash",
            "split_hash": "anonymous_split_hash",
            "dataset_hash": "anonymous_dataset_hash",
        }
    )
    role_aware = _baseline_rows(role_aware_metrics).rename(
        columns={
            "value": "role_aware_value",
            "feature_metadata_hash": "role_aware_feature_metadata_hash",
            "split_hash": "role_aware_split_hash",
            "dataset_hash": "role_aware_dataset_hash",
        }
    )
    if anonymous.duplicated(KEYS).any() or role_aware.duplicated(KEYS).any():
        raise ValueError("Corrected baseline rows are not unique by matched comparison keys.")
    merged = anonymous.merge(role_aware, on=KEYS, how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Corrected anonymous and role-aware baseline row sets differ.")
    checks = [
        ("split_hash", "anonymous_split_hash", "role_aware_split_hash"),
        (
            "feature metadata",
            "anonymous_feature_metadata_hash",
            "role_aware_feature_metadata_hash",
        ),
        ("dataset_hash", "anonymous_dataset_hash", "role_aware_dataset_hash"),
    ]
    for label, left, right in checks:
        mismatch = ~merged[left].astype(str).eq(merged[right].astype(str))
        if mismatch.any():
            example = merged.loc[mismatch, KEYS + [left, right]].iloc[0].to_dict()
            raise ValueError(f"Corrected baseline {label} mismatch: {example}")
    difference = np.abs(
        merged["anonymous_value"].to_numpy(dtype=float)
        - merged["role_aware_value"].to_numpy(dtype=float)
    )
    if not np.isfinite(difference).all() or (difference > tolerance).any():
        index = int(np.argmax(difference))
        raise ValueError(
            "Corrected real-only baseline metric mismatch exceeds "
            f"{tolerance}: {merged.iloc[index][KEYS].to_dict()}, difference={difference[index]}."
        )
    return merged.drop(columns="_merge")


def compare_corrected_condition_transfer(
    anonymous_dir: str | Path,
    role_aware_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Validate matched inputs and write paired corrected comparison artifacts."""
    anonymous_path = Path(anonymous_dir)
    role_path = Path(role_aware_dir)
    for directory in (anonymous_path, role_path):
        assert_result_directory_allowed(directory)
    anonymous_manifest = _load_corrected_manifest(anonymous_path)
    role_manifest = _load_corrected_manifest(role_path)
    anonymous_metrics = read_result_csv(anonymous_path / "policy_metrics.csv")
    role_metrics = read_result_csv(role_path / "policy_metrics.csv")
    matched_baselines = assert_matched_real_only_baselines(
        anonymous_metrics, role_metrics
    )
    output = prepare_fresh_output_directory(output_dir)
    paths = _output_paths(output)

    anonymous_selected = read_result_csv(anonymous_path / "selected_policy_metrics.csv")
    role_selected = read_result_csv(role_path / "selected_policy_metrics.csv")
    by_seed = _paired_by_seed(
        matched_baselines, anonymous_selected, role_selected
    )
    summary = _paired_summary(by_seed)
    fixed_stats = _fixed_policy_validation_stats(anonymous_metrics, role_metrics)
    decision = _decision_table(summary)

    by_seed.to_csv(paths["by_seed"], index=False)
    summary.to_csv(paths["summary"], index=False)
    fixed_stats.to_csv(paths["fixed_policy_stats"], index=False)
    decision.to_csv(paths["decision_table"], index=False)
    paths["report"].write_text(_report_text(decision, summary))
    write_json(
        paths["manifest"],
        {
            "comparison": "corrected_anonymous_vs_role_aware",
            "anonymous_directory": str(anonymous_path),
            "role_aware_directory": str(role_path),
            "output_directory": str(output),
            "anonymous_manifest_hash": stable_hash(anonymous_manifest),
            "role_aware_manifest_hash": stable_hash(role_manifest),
            "dataset_hash": anonymous_manifest["dataset_hash"],
            "feature_metadata_hash": anonymous_manifest["feature_metadata_hash"],
            "split_hashes": anonymous_manifest["split_hashes"],
            "historical_results_loaded": False,
            "result_status": CORRECTED_STATUS,
        },
    )
    return paths


def _baseline_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        *KEYS,
        "representation",
        "value",
        "feature_metadata_hash",
        "split_hash",
        "dataset_hash",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Corrected metrics are missing baseline columns: {missing}")
    return metrics.loc[
        metrics["representation"].eq(BASELINE_REPRESENTATION),
        KEYS
        + ["value", "feature_metadata_hash", "split_hash", "dataset_hash"],
    ].copy()


def _load_corrected_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing corrected run manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("historical_results_loaded") is not False:
        raise ValueError(f"Manifest does not prove historical_results_loaded=false: {path}")
    if manifest.get("result_status") != CORRECTED_STATUS:
        raise ValueError(f"Manifest is not a corrected revalidation result: {path}")
    return manifest


def _paired_by_seed(
    baselines: pd.DataFrame,
    anonymous_selected: pd.DataFrame,
    role_selected: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baselines.loc[baselines["split"].eq("test"), KEYS + ["anonymous_value"]].rename(
        columns={"anonymous_value": "real_only_value"}
    )
    anonymous = _selected_test_values(anonymous_selected, "anonymous_value")
    role = _selected_test_values(role_selected, "role_aware_value")
    merged = baseline.merge(anonymous, on=KEYS, how="inner").merge(
        role, on=KEYS, how="inner"
    )
    expected = len(baseline)
    if len(merged) != expected:
        raise ValueError(
            "Selected corrected policy test rows do not match every real-only baseline row."
        )
    merged["anonymous_delta_vs_real_only"] = (
        merged["anonymous_value"] - merged["real_only_value"]
    )
    merged["role_aware_delta_vs_real_only"] = (
        merged["role_aware_value"] - merged["real_only_value"]
    )
    merged["role_aware_minus_anonymous"] = (
        merged["role_aware_value"] - merged["anonymous_value"]
    )
    return merged.sort_values(["train_fraction", "metric", "seed"]).reset_index(drop=True)


def _selected_test_values(frame: pd.DataFrame, output_name: str) -> pd.DataFrame:
    selected = frame.loc[
        frame["split"].eq("test")
        & frame["selected_policy"].fillna(False).astype(bool),
        KEYS + ["value", "n_synthetic_train"],
    ].copy()
    if selected.empty or (selected["n_synthetic_train"].astype(int) <= 0).any():
        raise ValueError("Selected corrected policy rows are missing or include zero synthetic rows.")
    if selected["value"].isna().any() or selected.duplicated(KEYS).any():
        raise ValueError("Selected corrected policy test metrics are invalid or duplicated.")
    return selected.drop(columns="n_synthetic_train").rename(columns={"value": output_name})


def _paired_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fraction, metric), group in by_seed.groupby(["train_fraction", "metric"]):
        anonymous = group["anonymous_value"].to_numpy(dtype=float)
        role = group["role_aware_value"].to_numpy(dtype=float)
        rows.append(
            {
                "train_fraction": fraction,
                "metric": metric,
                "real_only_mean": group["real_only_value"].mean(),
                "anonymous_mean": anonymous.mean(),
                "role_aware_mean": role.mean(),
                "anonymous_delta_vs_real_only": group[
                    "anonymous_delta_vs_real_only"
                ].mean(),
                "role_aware_delta_vs_real_only": group[
                    "role_aware_delta_vs_real_only"
                ].mean(),
                "role_aware_minus_anonymous": (role - anonymous).mean(),
                "anonymous_better_seeds": int(_better(metric, anonymous, role).sum()),
                "role_aware_better_seeds": int(_better(metric, role, anonymous).sum()),
                "n_seeds": len(group),
                "paired_t_p": _paired_t(role, anonymous),
                "wilcoxon_p": _wilcoxon(role, anonymous),
            }
        )
    return pd.DataFrame(rows)


def _fixed_policy_validation_stats(
    anonymous_metrics: pd.DataFrame,
    role_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each fixed policy on validation without touching unselected test data."""
    rows: list[dict[str, Any]] = []
    for method, metrics in (
        ("anonymous", anonymous_metrics),
        ("role_aware", role_metrics),
    ):
        baseline = metrics.loc[
            metrics["representation"].eq(BASELINE_REPRESENTATION)
            & metrics["split"].eq("valid"),
            KEYS + ["value"],
        ].rename(columns={"value": "real_only_value"})
        policies = metrics.loc[
            metrics["representation"].eq(f"corrected_{method}_condition_transfer")
            & metrics["split"].eq("valid")
        ].copy()
        if policies.empty:
            continue
        identity_columns = [
            column
            for column in [
                "donor_strategy",
                "label_strategy",
                "synthetic_multiplier",
                "n_neighbors",
                "min_similarity",
                "max_teacher_std",
                "role_transfer_mode",
                "effective_role_transfer_mode",
            ]
            if column in policies.columns
        ]
        paired = policies.merge(baseline, on=KEYS, how="inner")
        paired["delta_vs_real_only"] = paired["value"] - paired["real_only_value"]
        for keys, group in paired.groupby(
            ["train_fraction", "metric", *identity_columns], dropna=False
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_names = ["train_fraction", "metric", *identity_columns]
            record = dict(zip(key_names, keys, strict=True))
            values = group["value"].to_numpy(dtype=float)
            real = group["real_only_value"].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "evaluation_scope": "validation_fixed_policy",
                    **record,
                    "mean_value": values.mean(),
                    "mean_real_only_value": real.mean(),
                    "mean_delta_vs_real_only": (values - real).mean(),
                    "better_seeds": int(
                        _better(str(record["metric"]), values, real).sum()
                    ),
                    "n_seeds": len(group),
                    "paired_t_p": _paired_t(values, real),
                    "wilcoxon_p": _wilcoxon(values, real),
                }
            )
    return pd.DataFrame(rows)


def _decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rmse = summary.loc[summary["metric"].eq("rmse")].copy()
    return rmse.rename(
        columns={
            "real_only_mean": "real_only_rmse",
            "anonymous_mean": "anonymous_rmse",
            "role_aware_mean": "role_aware_rmse",
            "role_aware_minus_anonymous": "role_aware_minus_anonymous_rmse",
        }
    )[
        [
            "train_fraction",
            "real_only_rmse",
            "anonymous_rmse",
            "role_aware_rmse",
            "anonymous_delta_vs_real_only",
            "role_aware_delta_vs_real_only",
            "role_aware_minus_anonymous_rmse",
            "anonymous_better_seeds",
            "role_aware_better_seeds",
            "n_seeds",
            "paired_t_p",
            "wilcoxon_p",
        ]
    ]


def _report_text(decision: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "CORRECTED ANONYMOUS VS ROLE-AWARE CONDITION TRANSFER",
        "",
        "Negative role_aware_minus_anonymous_rmse means role-aware is numerically lower.",
        "Positive values mean anonymous is numerically lower.",
        "Statistical significance alone is not used to select a winner.",
        "",
    ]
    for row in decision.itertuples(index=False):
        lower = "role-aware" if row.role_aware_minus_anonymous_rmse < 0 else "anonymous"
        consistency = (
            row.role_aware_better_seeds
            if lower == "role-aware"
            else row.anonymous_better_seeds
        )
        distinguishable = (
            "paired tests suggest a difference"
            if np.isfinite(row.paired_t_p) and row.paired_t_p < 0.05
            else "not statistically distinguishable by paired t-test"
        )
        lines.append(
            f"fraction={row.train_fraction:g}: {lower} numerically lower by "
            f"{abs(row.role_aware_minus_anonymous_rmse):.6g} RMSE; consistent across "
            f"{consistency} of {row.n_seeds} seeds; {distinguishable}."
        )
    lines.extend(["", "All metric summaries:", summary.to_string(index=False), ""])
    return "\n".join(lines)


def _better(metric: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left < right if metric in {"rmse", "mae"} else left > right


def _paired_t(left: np.ndarray, right: np.ndarray) -> float:
    return float(ttest_rel(left, right).pvalue) if len(left) >= 2 else float("nan")


def _wilcoxon(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.allclose(left, right):
        return float("nan")
    try:
        return float(wilcoxon(left, right).pvalue)
    except ValueError:
        return float("nan")


def _output_paths(directory: Path) -> dict[str, Path]:
    return {
        "directory": directory,
        "by_seed": directory / "anonymous_vs_role_aware_by_seed.csv",
        "summary": directory / "anonymous_vs_role_aware_summary.csv",
        "fixed_policy_stats": directory / "fixed_policy_stats_vs_real_only.csv",
        "decision_table": directory / "final_corrected_decision_table.csv",
        "manifest": directory / "comparison_manifest.json",
        "report": directory / "comparison_report.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anonymous-dir", required=True)
    parser.add_argument("--role-aware-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = compare_corrected_condition_transfer(
        args.anonymous_dir, args.role_aware_dir, args.output_dir
    )
    print(f"Saved corrected comparison to {paths['directory']}")


if __name__ == "__main__":
    main()
