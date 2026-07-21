"""Lightweight reporting helpers for experiment outputs."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bh_augmentation.results.status import read_result_csv


def save_metrics_csv(
    metrics: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Save metric records to a CSV file and return the output path."""
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(csv_path, index=False)
    return csv_path


def generate_markdown_report(results_dir: str | Path, output_path: str | Path) -> Path:
    """Generate a Markdown report and simple plots from available result CSVs.

    Missing result files are skipped. Plots are generated only when the required
    columns are available.
    """
    results_path = Path(results_dir)
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = report_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    result_files = _discover_result_files(results_path)
    datasets = {name: _read_csv(path) for name, path in result_files.items()}
    plot_paths = _make_plots(datasets, plots_dir)

    lines = [
        "# Buchwald-Hartwig Augmentation Report",
        "",
        "## Dataset Summary",
        *_dataset_summary_lines(datasets),
        "",
        "## Baseline Metrics Summary",
        *_metrics_summary_lines(datasets.get("baseline"), group_columns=["model", "split", "metric"]),
        "",
        "## Augmentation Metrics Summary",
        *_metrics_summary_lines(
            datasets.get("augmentation"),
            group_columns=["augmentation", "model", "split", "metric"],
        ),
        "",
        "## Low-Data Curve Summary",
        *_low_data_summary_lines(datasets),
        "",
        "## Hard Split Summary",
        *_hard_split_summary_lines(datasets),
        "",
        "## Recommendation Simulation Summary",
        *_recommendation_summary_lines(datasets.get("recommendation")),
        "",
        "## Plots",
        *_plot_lines(plot_paths, report_path.parent),
        "",
        "## Augmentation Helped / Hurt / Neutral",
        *_augmentation_effect_lines(datasets),
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _discover_result_files(results_dir: Path) -> dict[str, Path]:
    candidates = {
        "baseline": [
            results_dir / "baseline" / "baseline_metrics.csv",
            results_dir / "baseline_metrics.csv",
        ],
        "augmentation": [
            results_dir / "augmentation" / "safe_aug_metrics.csv",
            results_dir / "safe_aug_metrics.csv",
        ],
        "recommendation": [
            results_dir / "recommendation" / "topk_metrics.csv",
            results_dir / "topk_metrics.csv",
        ],
        "dataset": [
            results_dir / "dataset_summary.csv",
            results_dir / "data_summary.csv",
        ],
    }
    discovered: dict[str, Path] = {}
    for name, paths in candidates.items():
        for path in paths:
            if path.exists():
                discovered[name] = path
                break
    return discovered


def _read_csv(path: Path) -> pd.DataFrame:
    return read_result_csv(path)


def _dataset_summary_lines(datasets: dict[str, pd.DataFrame]) -> list[str]:
    dataset = datasets.get("dataset")
    if dataset is not None:
        return [_markdown_table(dataset.head(20))]
    return ["No dataset summary file found."]


def _metrics_summary_lines(
    df: pd.DataFrame | None,
    group_columns: list[str],
) -> list[str]:
    if df is None:
        return ["No results file found."]
    required = set(group_columns + ["value"])
    if not required.issubset(df.columns):
        return ["Results file found, but required metric columns are missing."]
    summary = (
        df.groupby(group_columns, dropna=False)["value"]
        .mean()
        .reset_index()
        .sort_values(group_columns)
    )
    return [_markdown_table(summary.head(30))]


def _low_data_summary_lines(datasets: dict[str, pd.DataFrame]) -> list[str]:
    frames = []
    for label in ["baseline", "augmentation"]:
        df = datasets.get(label)
        if df is not None and {"train_fraction", "metric", "value"}.issubset(df.columns):
            subset = df[df["metric"] == "rmse"].copy()
            if not subset.empty:
                subset.insert(0, "source", label)
                frames.append(subset)
    if not frames:
        return ["No low-data metrics found."]

    combined = pd.concat(frames, ignore_index=True)
    group_columns = ["source", "train_fraction"]
    if "augmentation" in combined.columns:
        group_columns.append("augmentation")
    summary = combined.groupby(group_columns, dropna=False)["value"].mean().reset_index()
    return [_markdown_table(summary.head(30))]


def _hard_split_summary_lines(datasets: dict[str, pd.DataFrame]) -> list[str]:
    frames = []
    for label in ["baseline", "augmentation"]:
        df = datasets.get(label)
        if df is not None and "split_method" in df.columns:
            subset = df[df["split_method"] == "heldout_group"].copy()
            if not subset.empty:
                subset.insert(0, "source", label)
                frames.append(subset)
    if not frames:
        return ["No held-out group split metrics found."]

    combined = pd.concat(frames, ignore_index=True)
    columns = [column for column in ["source", "group_column", "model", "split", "metric", "value"] if column in combined.columns]
    return [_markdown_table(combined[columns].head(30))]


def _recommendation_summary_lines(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return ["No recommendation metrics found."]
    columns = [
        column
        for column in [
            "strategy",
            "model",
            "k",
            "top_k_hit_rate",
            "regret",
            "experiments_to_first_hit",
        ]
        if column in df.columns
    ]
    if not columns:
        return ["Recommendation file found, but expected columns are missing."]
    return [_markdown_table(df[columns].head(30))]


def _make_plots(datasets: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
    plot_paths: list[Path] = []
    plot_paths.extend(_plot_train_fraction_vs_rmse(datasets, plots_dir))
    plot_paths.extend(_plot_train_fraction_vs_topk(datasets, plots_dir))
    plot_paths.extend(_plot_augmentation_vs_topk(datasets, plots_dir))
    plot_paths.extend(_plot_predicted_vs_true(datasets, plots_dir))
    return plot_paths


def _plot_train_fraction_vs_rmse(datasets: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
    frames = []
    for label in ["baseline", "augmentation"]:
        df = datasets.get(label)
        if df is not None and {"train_fraction", "metric", "value"}.issubset(df.columns):
            subset = df[df["metric"] == "rmse"].copy()
            if not subset.empty:
                subset["series"] = label if "augmentation" not in subset.columns else subset["augmentation"].fillna(label)
                frames.append(subset)
    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True)
    path = plots_dir / "train_fraction_vs_rmse.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    for series, group in data.groupby("series", dropna=False):
        summary = group.groupby("train_fraction")["value"].mean().reset_index().sort_values("train_fraction")
        ax.plot(summary["train_fraction"], summary["value"], marker="o", label=str(series))
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("RMSE")
    ax.set_title("Train fraction vs RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return [path]


def _plot_train_fraction_vs_topk(datasets: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
    df = datasets.get("recommendation")
    if df is None or not {"train_fraction", "top_k_hit_rate"}.issubset(df.columns):
        return []
    path = plots_dir / "train_fraction_vs_topk_hit_rate.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    series_column = "strategy" if "strategy" in df.columns else None
    if series_column:
        groups = df.groupby(series_column, dropna=False)
    else:
        groups = [("recommendation", df)]
    for series, group in groups:
        summary = group.groupby("train_fraction")["top_k_hit_rate"].mean().reset_index().sort_values("train_fraction")
        ax.plot(summary["train_fraction"], summary["top_k_hit_rate"], marker="o", label=str(series))
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Top-k hit rate")
    ax.set_title("Train fraction vs top-k hit rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return [path]


def _plot_augmentation_vs_topk(datasets: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
    df = datasets.get("recommendation")
    if df is None or not {"strategy", "top_k_hit_rate"}.issubset(df.columns):
        return []
    path = plots_dir / "augmentation_type_vs_topk_hit_rate.png"
    summary = df.groupby("strategy")["top_k_hit_rate"].mean().reset_index().sort_values("strategy")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["strategy"], summary["top_k_hit_rate"])
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Top-k hit rate")
    ax.set_title("Strategy vs top-k hit rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return [path]


def _plot_predicted_vs_true(datasets: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
    df = datasets.get("recommendation")
    if df is None or not {"predicted_yields", "true_yields"}.issubset(df.columns):
        return []
    predicted: list[float] = []
    true: list[float] = []
    for _, row in df.iterrows():
        predicted_values = _parse_list(row["predicted_yields"])
        true_values = _parse_list(row["true_yields"])
        for pred, actual in zip(predicted_values, true_values, strict=False):
            predicted.append(float(pred))
            true.append(float(actual))
    if not predicted:
        return []
    path = plots_dir / "predicted_vs_true_yield.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(predicted, true)
    ax.set_xlabel("Predicted yield")
    ax.set_ylabel("True yield")
    ax.set_title("Predicted vs true yield")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return [path]


def _plot_lines(plot_paths: list[Path], report_dir: Path) -> list[str]:
    if not plot_paths:
        return ["No plots generated from available result files."]
    lines = []
    for path in plot_paths:
        rel_path = path.relative_to(report_dir)
        lines.append(f"- ![{path.stem}]({rel_path.as_posix()})")
    return lines


def _augmentation_effect_lines(datasets: dict[str, pd.DataFrame]) -> list[str]:
    augmentation = datasets.get("augmentation")
    if augmentation is not None and {"augmentation", "metric", "value"}.issubset(augmentation.columns):
        rmse_rows = augmentation[augmentation["metric"] == "rmse"]
        none = rmse_rows[rmse_rows["augmentation"] == "none"]["value"].mean()
        augmented = rmse_rows[rmse_rows["augmentation"] != "none"]["value"].mean()
        if pd.notna(none) and pd.notna(augmented):
            return [_effect_sentence(lower_is_better=True, baseline=none, comparison=augmented)]

    recommendation = datasets.get("recommendation")
    if recommendation is not None and {"strategy", "top_k_hit_rate"}.issubset(recommendation.columns):
        model = recommendation[recommendation["strategy"] == "model"]["top_k_hit_rate"].mean()
        augmented = recommendation[recommendation["strategy"].astype(str).str.startswith("augmented_")]["top_k_hit_rate"].mean()
        if pd.notna(model) and pd.notna(augmented):
            return [_effect_sentence(lower_is_better=False, baseline=model, comparison=augmented)]

    return ["Neutral: insufficient paired augmentation and baseline results were available."]


def _effect_sentence(lower_is_better: bool, baseline: float, comparison: float) -> str:
    tolerance = 1e-9
    delta = comparison - baseline
    if abs(delta) <= tolerance:
        effect = "neutral"
    elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
        effect = "helped"
    else:
        effect = "hurt"
    return f"Augmentation {effect}: baseline={baseline:.4g}, augmented={comparison:.4g}."


def _parse_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    display = df.copy()
    display = display.fillna("")
    headers = [str(column) for column in display.columns]
    rows = [
        [str(value) for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
