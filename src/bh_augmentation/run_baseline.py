"""Command-line baseline experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.split_data import (
    heldout_group_split,
    leave_one_group_out_splits,
    low_data_split,
    random_split,
    subset_train_split,
)
from bh_augmentation.evaluation.metrics import (
    mae,
    pearson_corr,
    r2,
    rmse,
    spearman_corr,
)
from bh_augmentation.features.featurize import (
    REACTION_FEATURE_KINDS,
    build_feature_matrix,
    canonical_feature_kind,
)
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.reporting.make_report import save_metrics_csv
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed

MetricFn = Any

METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "pearson": pearson_corr,
    "spearman": spearman_corr,
}

DEFAULT_CATEGORICAL_COLUMNS = ["solvent"]


def run_baseline(config_path: str | Path) -> Path:
    """Run a baseline experiment and save validation/test metrics."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    data_path = _get_dataset_path(config)
    raw_df = load_reaction_csv(data_path)
    df = clean_buchwald_hartwig(raw_df)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run baseline.")

    baseline_folds = _create_baseline_folds(df, config, seed)
    feature_configs = _resolve_feature_configs(config.get("features", {}), df)

    model_configs = _resolve_model_configs(config.get("models", ["ridge"]))
    metric_names = config.get("metrics", ["rmse", "mae", "r2"])
    split_method = str(config.get("splits", {}).get("method", "random"))
    group_column = _get_group_column(config.get("splits", {}))
    records: list[dict[str, object]] = []

    _save_baseline_split_metadata(baseline_folds, config)

    for feature_config in feature_configs:
        X, y, _ = build_feature_matrix(df, feature_config)
        feature_kind = str(feature_config.get("kind", "custom"))
        n_features = int(X.shape[1])

        for train_fraction, heldout_group_value, splits in baseline_folds:
            for model_config in model_configs:
                model_name, model_kwargs = _parse_model_config(model_config)
                model = get_model(model_name, seed=seed, **model_kwargs)
                train_indices = _split_positions(splits["train"])
                fitted_model = train_model(model, X[train_indices], y[train_indices])

                for split_name in ["valid", "test"]:
                    split_indices = _split_positions(splits[split_name])
                    predictions = predict_model(fitted_model, X[split_indices])
                    y_true = y[split_indices]
                    for metric_name in metric_names:
                        metric_value = _compute_metric(metric_name, y_true, predictions)
                        record = {
                            "train_fraction": train_fraction,
                            "split_method": split_method,
                            "group_column": group_column,
                            "feature_kind": feature_kind,
                            "n_features": n_features,
                            "model": model_name,
                            "split": split_name,
                            "metric": metric_name,
                            "value": metric_value,
                        }
                        if split_method == "leave_one_group_out":
                            record["heldout_group_value"] = heldout_group_value
                        records.append(record)

    output_path = _get_metrics_output_path(config)
    saved_path = save_metrics_csv(records, output_path)
    if split_method == "leave_one_group_out":
        _save_logo_summary(records, config)
    return saved_path


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_baseline`."""
    parser = argparse.ArgumentParser(description="Run baseline yield prediction models.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    output_path = run_baseline(args.config)
    print(f"Saved baseline metrics to {output_path}")


def _get_dataset_path(config: dict[str, Any]) -> str | Path:
    dataset_config = config.get("dataset", {})
    path = dataset_config.get("path")
    if not path:
        raise ValueError("Config must define dataset.path.")
    return path


def _create_splits(df: pd.DataFrame, split_config: dict[str, Any], seed: int) -> dict[str, pd.DataFrame]:
    method = split_config.get("method", "random")
    if method == "random":
        return random_split(
            df,
            train_size=float(split_config.get("train_size", 0.8)),
            valid_size=float(split_config.get("valid_size", 0.1)),
            test_size=float(split_config.get("test_size", 0.1)),
            seed=seed,
        )
    if method == "low_data":
        return low_data_split(
            df,
            train_fraction=float(split_config["train_fraction"]),
            valid_size=float(split_config.get("valid_size", 0.1)),
            test_size=float(split_config.get("test_size", 0.1)),
            seed=seed,
        )
    if method == "heldout_group":
        group_column = str(split_config["group_column"])
        _validate_group_column(df, group_column)
        return heldout_group_split(
            df,
            group_column=group_column,
            heldout_fraction=float(split_config.get("heldout_fraction", 0.2)),
            valid_fraction=float(split_config.get("valid_fraction", 0.1)),
            seed=seed,
        )
    raise ValueError(f"Unknown split method: {method}")


def _create_split_variants(
    df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> list[tuple[float, dict[str, pd.DataFrame]]]:
    low_data_config = config.get("low_data", {})
    split_config = config.get("splits", {})
    if not low_data_config.get("enabled", False):
        return [(1.0, _create_splits(df, split_config, seed))]

    train_fractions = low_data_config.get("train_fractions")
    if not train_fractions:
        raise ValueError("low_data.train_fractions must contain at least one fraction.")

    base_split_config = dict(split_config)
    if base_split_config.get("method") == "low_data":
        base_split_config["method"] = "random"
        base_split_config["train_size"] = 1.0 - float(base_split_config.get("valid_size", 0.1)) - float(
            base_split_config.get("test_size", 0.1)
        )
        base_split_config.pop("train_fraction", None)

    base_splits = _create_splits(df, base_split_config, seed)
    variants = []
    for offset, fraction in enumerate(train_fractions):
        train_fraction = float(fraction)
        variants.append(
            (
                train_fraction,
                subset_train_split(base_splits, train_fraction=train_fraction, seed=seed + offset + 1),
            )
        )
    return variants


def _create_baseline_folds(
    df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> list[tuple[float, object, dict[str, pd.DataFrame]]]:
    split_config = config.get("splits", {})
    if split_config.get("method") != "leave_one_group_out":
        return [
            (train_fraction, "", splits)
            for train_fraction, splits in _create_split_variants(df, config, seed)
        ]
    if config.get("low_data", {}).get("enabled", False):
        raise ValueError("low_data cannot be combined with leave_one_group_out.")

    group_column = str(split_config.get("group_column", ""))
    _validate_group_column(df, group_column)
    folds = leave_one_group_out_splits(
        df,
        group_column=group_column,
        valid_fraction=float(split_config.get("valid_fraction", 0.1)),
        seed=seed,
    )
    return [(1.0, heldout_group, splits) for heldout_group, splits in folds]


def _resolve_feature_config(
    feature_config: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    resolved = dict(feature_config)
    kind = canonical_feature_kind(resolved.get("kind"))
    if kind:
        resolved["kind"] = kind
    if kind in REACTION_FEATURE_KINDS:
        resolved.pop("smiles_columns", None)
    elif "smiles_columns" not in resolved:
        resolved["smiles_columns"] = []
    if "categorical_columns" not in resolved:
        resolved["categorical_columns"] = (
            []
            if kind in REACTION_FEATURE_KINDS
            else [column for column in DEFAULT_CATEGORICAL_COLUMNS if column in df.columns]
        )
    return resolved


def _resolve_feature_configs(
    feature_config: dict[str, Any],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    comparisons = feature_config.get("compare")
    if comparisons is None:
        return [_resolve_feature_config(feature_config, df)]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("features.compare must contain at least one feature configuration.")

    shared = {key: value for key, value in feature_config.items() if key != "compare"}
    resolved: list[dict[str, Any]] = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("Each features.compare entry must be a mapping.")
        merged = {**shared, **comparison}
        if not merged.get("kind"):
            raise ValueError("Each features.compare entry must define kind.")
        resolved.append(_resolve_feature_config(merged, df))
    return resolved


def _parse_model_config(model_config: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(model_config, str):
        return model_config, {}
    if isinstance(model_config, dict):
        name = model_config.get("name")
        if not name:
            raise ValueError("Model config dictionaries must include a name field.")
        params = dict(model_config.get("params", {}))
        return str(name), params
    raise ValueError(f"Unsupported model config: {model_config}")


def _resolve_model_configs(models_config: object) -> list[str | dict[str, Any]]:
    if isinstance(models_config, dict):
        included = models_config.get("include")
        if not isinstance(included, list) or not included:
            raise ValueError("models.include must contain at least one model.")
        return included
    if isinstance(models_config, list) and models_config:
        return models_config
    raise ValueError("models must be a non-empty list or contain models.include.")


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric_name not in METRIC_FUNCTIONS:
        supported = ", ".join(sorted(METRIC_FUNCTIONS))
        raise ValueError(f"Unknown metric: {metric_name}. Supported metrics: {supported}.")
    return float(METRIC_FUNCTIONS[metric_name](y_true, y_pred))


def _split_positions(split: pd.DataFrame) -> np.ndarray:
    return split.index.to_numpy(dtype=int)


def _get_metrics_output_path(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    return Path(output_config.get("metrics_path", "results/baseline/baseline_metrics.csv"))


def _get_split_metadata_output_path(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    default_path = _get_metrics_output_path(config).with_name("split_metadata.csv")
    return Path(output_config.get("split_metadata_path", default_path))


def _get_summary_output_path(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    metrics_path = _get_metrics_output_path(config)
    default_path = metrics_path.with_name(f"{metrics_path.stem}_summary.csv")
    return Path(output_config.get("summary_metrics_path", default_path))


def _save_split_metadata(
    split_variants: list[tuple[float, dict[str, pd.DataFrame]]],
    config: dict[str, Any],
) -> Path:
    split_config = config.get("splits", {})
    split_method = str(split_config.get("method", "random"))
    group_column = _get_group_column(split_config)
    records: list[dict[str, object]] = []
    for train_fraction, splits in split_variants:
        for split_name, split_df in splits.items():
            for _, row in split_df.iterrows():
                records.append(
                    {
                        "train_fraction": train_fraction,
                        "split_method": split_method,
                        "group_column": group_column,
                        "split": split_name,
                        "reaction_id": row.get("reaction_id", ""),
                        "group_value": row.get(group_column, "") if group_column else "",
                    }
                )

    output_path = _get_split_metadata_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def _save_baseline_split_metadata(
    baseline_folds: list[tuple[float, object, dict[str, pd.DataFrame]]],
    config: dict[str, Any],
) -> Path:
    if config.get("splits", {}).get("method") != "leave_one_group_out":
        variants = [(fraction, splits) for fraction, _, splits in baseline_folds]
        return _save_split_metadata(variants, config)

    split_config = config.get("splits", {})
    group_column = _get_group_column(split_config)
    records: list[dict[str, object]] = []
    for train_fraction, heldout_group_value, splits in baseline_folds:
        for split_name, split_df in splits.items():
            for _, row in split_df.iterrows():
                records.append(
                    {
                        "train_fraction": train_fraction,
                        "split_method": "leave_one_group_out",
                        "group_column": group_column,
                        "heldout_group_value": heldout_group_value,
                        "split": split_name,
                        "reaction_id": row.get("reaction_id", ""),
                        "group_value": row.get(group_column, ""),
                    }
                )

    output_path = _get_split_metadata_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def _save_logo_summary(
    records: list[dict[str, object]],
    config: dict[str, Any],
) -> Path:
    metrics = pd.DataFrame(records)
    group_columns = [
        "group_column",
        "feature_kind",
        "n_features",
        "model",
        "split",
        "metric",
    ]
    summary = (
        metrics.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    output_path = _get_summary_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path


def _get_group_column(split_config: dict[str, Any]) -> str:
    if split_config.get("method", "random") not in {
        "heldout_group",
        "leave_one_group_out",
    }:
        return ""
    group_column = str(split_config.get("group_column", ""))
    if not group_column:
        raise ValueError(
            "splits.group_column is required for heldout_group and "
            "leave_one_group_out splits."
        )
    return group_column


def _validate_group_column(df: pd.DataFrame, group_column: str) -> None:
    if group_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Held-out group column is missing after cleaning: {group_column}. "
            f"Available columns: {available}."
        )


if __name__ == "__main__":
    main()
