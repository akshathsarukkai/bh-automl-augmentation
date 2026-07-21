#!/usr/bin/env python
"""Build a status-aware data-efficiency table without test-set model selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bh_augmentation.results.status import read_result_csv
from bh_augmentation.utils.corrected_runs import (
    CORRECTED_STATUS,
    prepare_fresh_output_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-development-evidence",
        action="store_true",
        help="Include AE/hybrid development evidence in a separately labeled status.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/corrected_data_efficiency_comparison",
    )
    args = parser.parse_args()
    output = prepare_fresh_output_directory(args.output_dir)

    tables: list[pd.DataFrame] = []
    representation_path = Path(
        "results/corrected_representation_baselines_xgboost/policy_metrics.csv"
    )
    representation = _read_if_present(representation_path)
    if not representation.empty:
        rows = representation.loc[representation["split"].eq("test")].copy()
        rows["method"] = "corrected_real_only__" + rows["representation"].astype(str)
        rows["details"] = rows["representation"].astype(str) + "; no test selection"
        rows["result_status"] = CORRECTED_STATUS
        tables.append(rows)

    for method, path in (
        (
            "corrected_anonymous_condition_transfer",
            Path(
                "results/corrected_anonymous_condition_transfer_xgboost/selected_policy_metrics.csv"
            ),
        ),
        (
            "corrected_role_aware_condition_transfer",
            Path(
                "results/corrected_role_aware_condition_transfer_xgboost/selected_policy_metrics.csv"
            ),
        ),
    ):
        selected = _read_if_present(path)
        if not selected.empty:
            rows = selected.loc[
                selected["split"].eq("test")
                & selected["selected_policy"].fillna(False).astype(bool)
            ].copy()
            rows["method"] = method
            rows["details"] = "policy selected by validation RMSE"
            rows["result_status"] = CORRECTED_STATUS
            tables.append(rows)

    if args.include_development_evidence:
        tables.extend(_development_tables())

    if not tables:
        raise FileNotFoundError(
            "No corrected result files were found. Run corrected revalidation first."
        )
    long = pd.concat(tables, ignore_index=True, sort=False)
    summary = (
        long.groupby(
            ["train_fraction", "method", "details", "result_status", "metric"],
            dropna=False,
        )["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    table = summary.pivot_table(
        index=["train_fraction", "method", "details", "result_status"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    table.to_csv(output / "data_efficiency_comparison.csv", index=False)
    long.to_csv(output / "data_efficiency_long.csv", index=False)
    print("\nCORRECTED DATA EFFICIENCY COMPARISON")
    print(table.to_string(index=False))
    print(f"\nSaved: {output / 'data_efficiency_comparison.csv'}")


def _read_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing corrected evidence: {path}")
        return pd.DataFrame()
    return read_result_csv(path)


def _development_tables() -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for method, path in (
        (
            "development_supervised_ae",
            Path(
                "results/supervised_ae_latent_interpolation_confirm_xgboost/selected_policy_metrics.csv"
            ),
        ),
        (
            "development_condition_transfer_supervised_ae_hybrid",
            Path(
                "results/condition_transfer_supervised_ae_xgboost/selected_hybrid_policy_metrics.csv"
            ),
        ),
    ):
        if not path.exists():
            continue
        rows = read_result_csv(path)
        if "split" in rows:
            rows = rows.loc[rows["split"].eq("test")].copy()
        rows["method"] = method
        rows["details"] = "explicitly included development evidence"
        rows["result_status"] = "development_evidence"
        tables.append(rows)
    return tables


if __name__ == "__main__":
    main()
