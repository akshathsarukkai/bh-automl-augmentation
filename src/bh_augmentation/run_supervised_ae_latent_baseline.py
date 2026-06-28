"""Run supervised autoencoder latent baseline experiments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.split_data import random_split, subset_train_split
from bh_augmentation.evaluation.metrics import (
    mae,
    r2,
    rmse,
    spearman_corr,
)
from bh_augmentation.features.featurize import build_feature_matrix, canonical_feature_kind
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.representations.supervised_autoencoder import (
    SupervisedAEConfig,
    encode_with_supervised_autoencoder,
    fit_supervised_autoencoder,
    predict_yield_with_supervised_autoencoder,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed

METRIC_FUNCTIONS = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "spearman": spearman_corr,
}


def run_supervised_ae_latent_baseline(config_path: str | Path) -> dict[str, Path]:
    """Run original, SVD, and supervised-AE latent baselines."""
    config = load_config(config_path)
    seeds = _resolve_seeds(config)
    metric_names = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metric_names)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = clean_buchwald_hartwig(raw_df)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run supervised AE baseline.")

    feature_config = _resolve_feature_config(config.get("features", {}))
    X, y, _ = build_feature_matrix(df, feature_config)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    original_n_features = int(X.shape[1])

    records: list[dict[str, object]] = []
    output_paths = _resolve_output_paths(config)
    output_paths["history_dir"].mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        set_global_seed(seed)
        split_variants = _create_split_variants(df, config, seed)
        for train_fraction, splits in split_variants:
            train_indices = _split_positions(splits["train"])
            valid_indices = _split_positions(splits["valid"])
            test_indices = _split_positions(splits["test"])
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            n_train = int(len(train_indices))

            records.extend(
                _evaluate_downstream_models(
                    X_train,
                    y_train,
                    {"valid": (X_valid, y_valid), "test": (X_test, y_test)},
                    config,
                    seed=seed,
                    train_fraction=float(train_fraction),
                    n_train=n_train,
                    representation="original_6144",
                    requested_latent_dim=original_n_features,
                    actual_latent_dim=original_n_features,
                    original_n_features=original_n_features,
                    student_n_features=original_n_features,
                    metric_names=metric_names,
                    ae_metadata={},
                )
            )

            if config.get("svd_baseline", {}).get("enabled", True):
                for requested_dim in config.get("svd_baseline", {}).get(
                    "components",
                    [16, 30, 64, 128],
                ):
                    svd, actual_dim = fit_svd_on_train_only(
                        X_train,
                        requested_components=int(requested_dim),
                        random_state=seed,
                    )
                    X_train_svd = svd.transform(X_train).astype(np.float32)
                    X_valid_svd = svd.transform(X_valid).astype(np.float32)
                    X_test_svd = svd.transform(X_test).astype(np.float32)
                    records.extend(
                        _evaluate_downstream_models(
                            X_train_svd,
                            y_train,
                            {
                                "valid": (X_valid_svd, y_valid),
                                "test": (X_test_svd, y_test),
                            },
                            config,
                            seed=seed,
                            train_fraction=float(train_fraction),
                            n_train=n_train,
                            representation=f"svd_{int(requested_dim)}",
                            requested_latent_dim=int(requested_dim),
                            actual_latent_dim=actual_dim,
                            original_n_features=original_n_features,
                            student_n_features=actual_dim,
                            metric_names=metric_names,
                            ae_metadata={},
                        )
                    )

            if config.get("supervised_autoencoder", {}).get("enabled", True):
                for latent_dim in config.get("supervised_autoencoder", {}).get(
                    "latent_dims",
                    [16, 32, 64, 128],
                ):
                    ae_artifacts = _fit_ae_for_split(
                        X_train,
                        y_train,
                        original_n_features=original_n_features,
                        latent_dim=int(latent_dim),
                        seed=seed,
                        config=config,
                    )
                    _save_history(
                        ae_artifacts["history"],
                        output_paths["history_dir"],
                        seed=seed,
                        train_fraction=float(train_fraction),
                        latent_dim=int(latent_dim),
                    )
                    X_train_ae = encode_with_supervised_autoencoder(ae_artifacts, X_train)
                    X_valid_ae = encode_with_supervised_autoencoder(ae_artifacts, X_valid)
                    X_test_ae = encode_with_supervised_autoencoder(ae_artifacts, X_test)
                    ae_metadata = _ae_metric_metadata(ae_artifacts)
                    records.extend(
                        _evaluate_downstream_models(
                            X_train_ae,
                            y_train,
                            {"valid": (X_valid_ae, y_valid), "test": (X_test_ae, y_test)},
                            config,
                            seed=seed,
                            train_fraction=float(train_fraction),
                            n_train=n_train,
                            representation=f"supervised_ae_{int(latent_dim)}",
                            requested_latent_dim=int(latent_dim),
                            actual_latent_dim=int(X_train_ae.shape[1]),
                            original_n_features=original_n_features,
                            student_n_features=int(X_train_ae.shape[1]),
                            metric_names=metric_names,
                            ae_metadata=ae_metadata,
                        )
                    )
                    records.extend(
                        _evaluate_ae_yield_head(
                            ae_artifacts,
                            {"valid": (X_valid, y_valid), "test": (X_test, y_test)},
                            seed=seed,
                            train_fraction=float(train_fraction),
                            n_train=n_train,
                            representation=f"supervised_ae_yield_head_{int(latent_dim)}",
                            requested_latent_dim=int(latent_dim),
                            actual_latent_dim=int(X_train_ae.shape[1]),
                            original_n_features=original_n_features,
                            metric_names=metric_names,
                            ae_metadata=ae_metadata,
                        )
                    )

    metrics = pd.DataFrame(records)
    output_paths["metrics_path"].parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_paths["metrics_path"], index=False)

    summary = _summarize_metrics(metrics)
    output_paths["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_paths["summary_path"], index=False)

    by_seed, delta_summary = _save_ae_vs_original_rf(metrics, output_paths)
    _print_test_summary(summary)
    _print_delta_summary(delta_summary)
    return {
        "metrics_path": output_paths["metrics_path"],
        "summary_path": output_paths["summary_path"],
        "history_dir": output_paths["history_dir"],
        "ae_vs_original_rf_by_seed_path": output_paths["ae_vs_original_rf_by_seed_path"],
        "ae_vs_original_rf_summary_path": output_paths["ae_vs_original_rf_summary_path"],
    }


def fit_svd_on_train_only(
    X_train: np.ndarray,
    requested_components: int,
    random_state: int,
) -> tuple[TruncatedSVD, int]:
    """Fit TruncatedSVD on training features only with a safe component cap."""
    X_train_array = np.asarray(X_train, dtype=np.float32)
    if X_train_array.ndim != 2:
        raise ValueError("X_train must be a 2D feature matrix.")
    if X_train_array.shape[1] < 2:
        raise ValueError("TruncatedSVD requires at least two input features.")
    actual_components = _actual_svd_components(
        requested_components,
        n_rows=X_train_array.shape[0],
        n_features=X_train_array.shape[1],
    )
    svd = TruncatedSVD(n_components=actual_components, random_state=random_state)
    svd.fit(X_train_array)
    return svd, actual_components


def make_internal_ae_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    valid_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split low-data train rows into AE train and internal validation rows."""
    X_train_array = np.asarray(X_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(X_train_array) != len(y_train_array):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if len(X_train_array) < 4 or valid_size <= 0:
        return X_train_array, y_train_array, None, None

    n_valid = int(math.floor(len(X_train_array) * valid_size))
    n_valid = min(max(1, n_valid), len(X_train_array) - 1)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X_train_array))
    rng.shuffle(indices)
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]
    return (
        X_train_array[train_indices],
        y_train_array[train_indices],
        X_train_array[valid_indices],
        y_train_array[valid_indices],
    )


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_supervised_ae_latent_baseline`."""
    parser = argparse.ArgumentParser(description="Run supervised AE latent baselines.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    paths = run_supervised_ae_latent_baseline(args.config)
    print(f"Saved supervised AE latent metrics to {paths['metrics_path']}")


def _fit_ae_for_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    original_n_features: int,
    latent_dim: int,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    ae_config = config.get("supervised_autoencoder", {})
    X_ae_train, y_ae_train, X_ae_valid, y_ae_valid = make_internal_ae_split(
        X_train,
        y_train,
        valid_size=float(ae_config.get("internal_valid_size", 0.2)),
        seed=seed + 10_000 + latent_dim,
    )
    training_config = SupervisedAEConfig(
        input_dim=original_n_features,
        latent_dim=latent_dim,
        hidden_dims=[int(dim) for dim in ae_config.get("hidden_dims", [1024, 256])],
        dropout=float(ae_config.get("dropout", 0.1)),
        reconstruction_weight=float(ae_config.get("reconstruction_weight", 1.0)),
        yield_weight=float(ae_config.get("yield_weight", 1.0)),
        latent_l2_weight=float(ae_config.get("latent_l2_weight", 0.0)),
        learning_rate=float(ae_config.get("learning_rate", 0.001)),
        weight_decay=float(ae_config.get("weight_decay", 0.0)),
        batch_size=int(ae_config.get("batch_size", 32)),
        max_epochs=int(ae_config.get("max_epochs", 200)),
        patience=int(ae_config.get("patience", 20)),
        random_state=seed,
        device=str(ae_config.get("device", "auto")),
    )
    return fit_supervised_autoencoder(
        X_ae_train,
        y_ae_train,
        X_ae_valid,
        y_ae_valid,
        training_config,
    )


def _evaluate_downstream_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    evaluation_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seed: int,
    train_fraction: float,
    n_train: int,
    representation: str,
    requested_latent_dim: int,
    actual_latent_dim: int,
    original_n_features: int,
    student_n_features: int,
    metric_names: list[str],
    ae_metadata: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model_config in _resolve_model_configs(config.get("models", ["ridge", "random_forest"])):
        model_name, model_kwargs = _parse_model_config(model_config)
        model = get_model(model_name, seed=seed, **model_kwargs)
        fitted_model = train_model(model, X_train, y_train)
        for split_name, (X_split, y_split) in evaluation_splits.items():
            predictions = predict_model(fitted_model, X_split)
            records.extend(
                _metric_records(
                    y_split,
                    predictions,
                    metric_names=metric_names,
                    seed=seed,
                    train_fraction=train_fraction,
                    n_train=n_train,
                    representation=representation,
                    model=model_name,
                    requested_latent_dim=requested_latent_dim,
                    actual_latent_dim=actual_latent_dim,
                    original_n_features=original_n_features,
                    student_n_features=student_n_features,
                    split=split_name,
                    ae_metadata=ae_metadata,
                )
            )
    return records


def _evaluate_ae_yield_head(
    ae_artifacts: dict[str, Any],
    evaluation_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    train_fraction: float,
    n_train: int,
    representation: str,
    requested_latent_dim: int,
    actual_latent_dim: int,
    original_n_features: int,
    metric_names: list[str],
    ae_metadata: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split_name, (X_split, y_split) in evaluation_splits.items():
        predictions = predict_yield_with_supervised_autoencoder(ae_artifacts, X_split)
        records.extend(
            _metric_records(
                y_split,
                predictions,
                metric_names=metric_names,
                seed=seed,
                train_fraction=train_fraction,
                n_train=n_train,
                representation=representation,
                model="ae_yield_head",
                requested_latent_dim=requested_latent_dim,
                actual_latent_dim=actual_latent_dim,
                original_n_features=original_n_features,
                student_n_features=actual_latent_dim,
                split=split_name,
                ae_metadata=ae_metadata,
            )
        )
    return records


def _metric_records(
    y_true: np.ndarray,
    predictions: np.ndarray,
    metric_names: list[str],
    seed: int,
    train_fraction: float,
    n_train: int,
    representation: str,
    model: str,
    requested_latent_dim: int,
    actual_latent_dim: int,
    original_n_features: int,
    student_n_features: int,
    split: str,
    ae_metadata: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for metric_name in metric_names:
        records.append(
            {
                "seed": seed,
                "train_fraction": train_fraction,
                "n_train": n_train,
                "representation": representation,
                "model": model,
                "requested_latent_dim": requested_latent_dim,
                "actual_latent_dim": actual_latent_dim,
                "original_n_features": original_n_features,
                "student_n_features": student_n_features,
                "split": split,
                "metric": metric_name,
                "value": _compute_metric(metric_name, y_true, predictions),
                "ae_best_epoch": ae_metadata.get("ae_best_epoch", np.nan),
                "ae_best_internal_valid_loss": ae_metadata.get(
                    "ae_best_internal_valid_loss",
                    np.nan,
                ),
                "ae_reconstruction_weight": ae_metadata.get("ae_reconstruction_weight", np.nan),
                "ae_yield_weight": ae_metadata.get("ae_yield_weight", np.nan),
            }
        )
    return records


def _ae_metric_metadata(artifacts: dict[str, Any]) -> dict[str, object]:
    config = artifacts.get("config", {})
    return {
        "ae_best_epoch": int(artifacts.get("best_epoch", 0)),
        "ae_best_internal_valid_loss": float(artifacts.get("best_validation_loss", np.nan)),
        "ae_reconstruction_weight": float(config.get("reconstruction_weight", np.nan)),
        "ae_yield_weight": float(config.get("yield_weight", np.nan)),
    }


def _save_history(
    history: pd.DataFrame,
    history_dir: Path,
    seed: int,
    train_fraction: float,
    latent_dim: int,
) -> Path:
    fraction_label = str(train_fraction).replace(".", "p")
    path = history_dir / f"seed_{seed}_train_fraction_{fraction_label}_latent_{latent_dim}.csv"
    history.to_csv(path, index=False)
    return path


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "train_fraction",
        "representation",
        "model",
        "split",
        "metric",
    ]
    return (
        metrics.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def _save_ae_vs_original_rf(
    metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    baseline = test_metrics.loc[
        (test_metrics["representation"] == "original_6144")
        & (test_metrics["model"] == "random_forest"),
        ["seed", "train_fraction", "metric", "value"],
    ].rename(columns={"value": "original_rf_value"})
    ae_rows = test_metrics.loc[test_metrics["representation"].str.startswith("supervised_ae")].copy()
    by_seed = ae_rows.merge(
        baseline,
        on=["seed", "train_fraction", "metric"],
        how="left",
    )
    by_seed["delta_vs_original_rf"] = by_seed["value"] - by_seed["original_rf_value"]
    by_seed["delta_interpretation"] = np.where(
        by_seed["metric"].isin(["mae", "rmse"]),
        "positive_worse_negative_better",
        "positive_better_negative_worse",
    )
    output_paths["ae_vs_original_rf_by_seed_path"].parent.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(output_paths["ae_vs_original_rf_by_seed_path"], index=False)

    group_columns = [
        "train_fraction",
        "representation",
        "model",
        "metric",
        "delta_interpretation",
    ]
    summary = (
        by_seed.groupby(group_columns, dropna=False)["delta_vs_original_rf"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    summary.to_csv(output_paths["ae_vs_original_rf_summary_path"], index=False)
    return by_seed, summary


def _print_test_summary(summary: pd.DataFrame) -> None:
    test_summary = summary.loc[summary["split"] == "test"]
    if test_summary.empty:
        return
    table = test_summary.pivot_table(
        index=["train_fraction", "representation", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    columns = [
        column
        for column in ["train_fraction", "representation", "model", "mae", "rmse", "r2", "spearman"]
        if column in table.columns
    ]
    print("\nTEST MEAN METRICS ACROSS SEEDS")
    print(table[columns].to_string(index=False))


def _print_delta_summary(delta_summary: pd.DataFrame) -> None:
    if delta_summary.empty:
        return
    table = delta_summary.pivot_table(
        index=["train_fraction", "representation", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    columns = [
        column
        for column in ["train_fraction", "representation", "model", "mae", "rmse", "r2", "spearman"]
        if column in table.columns
    ]
    print("\nTEST DELTAS VS ORIGINAL 6144D RANDOM FOREST")
    print("MAE/RMSE: positive is worse. R2/Spearman: positive is better.")
    print(table[columns].to_string(index=False))


def _resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_config = config.get("output", {})
    metrics_path = Path(
        output_config.get(
            "metrics_path",
            "results/supervised_ae_latent_baseline/metrics.csv",
        )
    )
    summary_path = Path(
        output_config.get(
            "summary_path",
            metrics_path.with_name("summary.csv"),
        )
    )
    history_dir = Path(
        output_config.get(
            "history_dir",
            metrics_path.with_name("history"),
        )
    )
    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "history_dir": history_dir,
        "ae_vs_original_rf_by_seed_path": Path(
            output_config.get(
                "ae_vs_original_rf_by_seed_path",
                metrics_path.with_name("ae_vs_original_rf_by_seed.csv"),
            )
        ),
        "ae_vs_original_rf_summary_path": Path(
            output_config.get(
                "ae_vs_original_rf_summary_path",
                metrics_path.with_name("ae_vs_original_rf_summary.csv"),
            )
        ),
    }


def _resolve_feature_config(feature_config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(feature_config)
    kind = canonical_feature_kind(resolved.get("kind", "reaction_role_concat"))
    resolved["kind"] = kind or "reaction_role_concat"
    resolved.setdefault("n_bits", 2048)
    resolved.setdefault("radius", 2)
    resolved.pop("compare", None)
    if resolved["kind"].startswith("reaction_"):
        resolved["categorical_columns"] = []
    return resolved


def _resolve_seeds(config: dict[str, Any]) -> list[int]:
    seeds = config.get("seeds")
    if seeds is None:
        return [int(config.get("seed", 42))]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list when provided.")
    return [int(seed) for seed in seeds]


def _create_split_variants(
    df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> list[tuple[float, dict[str, pd.DataFrame]]]:
    low_data_config = config.get("low_data", {})
    split_config = dict(config.get("splits", {}))
    if not low_data_config.get("enabled", False):
        return [(1.0, _create_random_splits(df, split_config, seed))]

    train_fractions = low_data_config.get("train_fractions")
    if not train_fractions:
        raise ValueError("low_data.train_fractions must contain at least one fraction.")

    base_splits = _create_random_splits(df, split_config, seed)
    variants: list[tuple[float, dict[str, pd.DataFrame]]] = []
    for offset, fraction in enumerate(train_fractions):
        train_fraction = float(fraction)
        variants.append(
            (
                train_fraction,
                subset_train_split(
                    base_splits,
                    train_fraction=train_fraction,
                    seed=seed + offset + 1,
                ),
            )
        )
    return variants


def _create_random_splits(
    df: pd.DataFrame,
    split_config: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    method = str(split_config.get("method", "random"))
    if method != "random":
        raise ValueError("supervised_ae_latent_baseline currently supports only random splits.")
    return random_split(
        df,
        train_size=float(split_config.get("train_size", 0.8)),
        valid_size=float(split_config.get("valid_size", 0.1)),
        test_size=float(split_config.get("test_size", 0.1)),
        seed=seed,
    )


def _resolve_model_configs(models_config: object) -> list[str | dict[str, Any]]:
    if isinstance(models_config, dict):
        included = models_config.get("include")
        if not isinstance(included, list) or not included:
            raise ValueError("models.include must contain at least one model.")
        return included
    if isinstance(models_config, list) and models_config:
        return models_config
    raise ValueError("models must be a non-empty list or contain models.include.")


def _parse_model_config(model_config: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(model_config, str):
        return model_config, {}
    if isinstance(model_config, dict):
        name = model_config.get("name")
        if not name:
            raise ValueError("Model config dictionaries must include a name field.")
        return str(name), dict(model_config.get("params", {}))
    raise ValueError(f"Unsupported model config: {model_config}")


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric_name not in METRIC_FUNCTIONS:
        supported = ", ".join(sorted(METRIC_FUNCTIONS))
        raise ValueError(f"Unknown metric: {metric_name}. Supported metrics: {supported}.")
    return float(METRIC_FUNCTIONS[metric_name](y_true, y_pred))


def _get_dataset_path(config: dict[str, Any]) -> str | Path:
    path = config.get("dataset", {}).get("path")
    if not path:
        raise ValueError("Config must define dataset.path.")
    return path


def _split_positions(split: pd.DataFrame) -> np.ndarray:
    return split.index.to_numpy(dtype=int)


def _actual_svd_components(requested_components: int, n_rows: int, n_features: int) -> int:
    if requested_components < 1:
        raise ValueError("SVD component count must be at least 1.")
    max_components = max(1, min(int(n_rows), int(n_features) - 1))
    return int(min(requested_components, max_components))


def _validate_metrics(metric_names: list[str]) -> None:
    for metric_name in metric_names:
        if metric_name not in METRIC_FUNCTIONS:
            supported = ", ".join(sorted(METRIC_FUNCTIONS))
            raise ValueError(f"Unknown metric: {metric_name}. Supported metrics: {supported}.")


if __name__ == "__main__":
    main()
