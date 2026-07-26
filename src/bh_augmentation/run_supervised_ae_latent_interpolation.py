"""Run supervised-AE latent interpolation augmentation experiments."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.latent_interpolation import (
    LatentInterpolationConfig,
    generate_latent_interpolations,
)
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.features.compatibility import (
    assert_feature_compatibility,
    coordinate_feature_contract,
)
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.representations.supervised_autoencoder import (
    encode_with_supervised_autoencoder,
)
from bh_augmentation.run_supervised_ae_latent_baseline import (
    _compute_metric,
    _create_split_variants,
    _fit_ae_for_split,
    _parse_model_config,
    _resolve_feature_config,
    _resolve_model_configs,
    _resolve_seeds,
    _validate_metrics,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed
from bh_augmentation.utils.synthetic_audits import (
    build_candidate_audit_frame,
    combine_candidate_audit_frames,
)


def run_supervised_ae_latent_interpolation(config_path: str | Path) -> dict[str, Path]:
    """Run original, AE-only, and AE-latent-interpolation baselines."""
    config = load_config(config_path)
    seeds = _resolve_seeds(config)
    metric_names = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metric_names)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = clean_buchwald_hartwig(raw_df)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run latent interpolation.")

    feature_config = _resolve_feature_config(config.get("features", {}))
    X, y, _ = build_feature_matrix(df, feature_config)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    original_n_features = int(X.shape[1])
    output_paths = _resolve_output_paths(config)
    output_paths["directory"].mkdir(parents=True, exist_ok=True)

    metric_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    candidate_audit_frames: list[pd.DataFrame] = []

    for seed in seeds:
        set_global_seed(seed)
        for train_fraction, splits in _create_split_variants(df, config, seed):
            train_indices = _split_positions(splits["train"])
            valid_indices = _split_positions(splits["valid"])
            test_indices = _split_positions(splits["test"])
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            n_train = int(len(train_indices))

            metric_records.extend(
                _evaluate_models(
                    X_train,
                    y_train,
                    {"valid": (X_valid, y_valid), "test": (X_test, y_test)},
                    config,
                    seed=seed,
                    train_fraction=float(train_fraction),
                    n_train=n_train,
                    representation="original_6144",
                    latent_dim=np.nan,
                    policy_metadata=_empty_policy_metadata(),
                    metric_names=metric_names,
                    policy_id=f"seed={seed}|frac={train_fraction}|original",
                )
            )

            for latent_dim in config.get("supervised_autoencoder", {}).get(
                "latent_dims",
                [16, 32, 64, 128],
            ):
                latent_dim = int(latent_dim)
                ae_artifacts = _fit_ae_for_split(
                    X_train,
                    y_train,
                    original_n_features=original_n_features,
                    latent_dim=latent_dim,
                    seed=seed,
                    config=config,
                )
                z_train = encode_with_supervised_autoencoder(ae_artifacts, X_train)
                z_valid = encode_with_supervised_autoencoder(ae_artifacts, X_valid)
                z_test = encode_with_supervised_autoencoder(ae_artifacts, X_test)

                metric_records.extend(
                    _evaluate_models(
                        z_train,
                        y_train,
                        {"valid": (z_valid, y_valid), "test": (z_test, y_test)},
                        config,
                        seed=seed,
                        train_fraction=float(train_fraction),
                        n_train=n_train,
                        representation=f"supervised_ae_{latent_dim}",
                        latent_dim=latent_dim,
                        policy_metadata=_empty_policy_metadata(),
                        metric_names=metric_names,
                        policy_id=f"seed={seed}|frac={train_fraction}|latent={latent_dim}|ae_only",
                    )
                )

                for policy_index, policy in enumerate(_iter_interpolation_policies(config, seed)):
                    policy_id = _policy_id(seed, float(train_fraction), latent_dim, policy_index, policy)
                    result = generate_latent_interpolations(
                        z_train,
                        y_train,
                        ae_artifacts,
                        policy,
                        source_row_ids=_training_row_ids(splits["train"]),
                    )
                    metadata = result["metadata"]
                    metadata.update(
                        {
                            "seed": seed,
                            "train_fraction": float(train_fraction),
                            "latent_dim": latent_dim,
                            "policy_id": policy_id,
                            "used_validation_or_test_parents": False,
                        }
                    )
                    audit_records.append(dict(metadata))
                    candidate_audit_frames.append(
                        build_candidate_audit_frame(
                            result["candidate_df"],
                            transfer_kind="latent_interpolation",
                            seed=seed,
                            train_fraction=float(train_fraction),
                            policy_id=policy_id,
                        ).assign(
                            latent_dim=latent_dim,
                        )
                    )

                    z_synthetic = result["z_synthetic"]
                    y_synthetic = result["y_synthetic"]
                    if len(z_synthetic) == 0:
                        z_augmented = z_train
                        y_augmented = y_train
                    else:
                        latent_names, latent_metadata = coordinate_feature_contract(
                            f"supervised_ae_latent_{latent_dim}",
                            z_train.shape[1],
                        )
                        assert_feature_compatibility(
                            z_train,
                            latent_names,
                            z_synthetic,
                            latent_names,
                            real_metadata=latent_metadata,
                            synthetic_metadata=latent_metadata,
                        )
                        z_augmented = np.vstack([z_train, z_synthetic]).astype(np.float32)
                        y_augmented = np.concatenate([y_train, y_synthetic]).astype(float)

                    metric_records.extend(
                        _evaluate_models(
                            z_augmented,
                            y_augmented,
                            {"valid": (z_valid, y_valid), "test": (z_test, y_test)},
                            config,
                            seed=seed,
                            train_fraction=float(train_fraction),
                            n_train=n_train,
                            representation=f"supervised_ae_interpolation_{latent_dim}",
                            latent_dim=latent_dim,
                            policy_metadata=metadata,
                            metric_names=metric_names,
                            policy_id=policy_id,
                        )
                    )

    policy_metrics = pd.DataFrame(metric_records)
    policy_metrics["selected_policy"] = False
    selected_policies = _select_policies(policy_metrics, config)
    policy_metrics = _mark_selected_policy_metrics(policy_metrics, selected_policies)
    selected_policy_metrics = policy_metrics.loc[policy_metrics["selected_policy"]].copy()

    audit = pd.DataFrame(audit_records)
    summary = _summarize_for_reporting(policy_metrics, selected_policy_metrics)
    original_delta, original_delta_summary = _save_interpolation_vs_original_rf(
        policy_metrics,
        selected_policy_metrics,
        output_paths,
    )
    ae_delta, ae_delta_summary = _save_interpolation_vs_ae_only(
        policy_metrics,
        selected_policy_metrics,
        output_paths,
    )

    candidate_audit = combine_candidate_audit_frames(candidate_audit_frames)
    _write_outputs(
        policy_metrics,
        selected_policies,
        selected_policy_metrics,
        summary,
        audit,
        candidate_audit,
        output_paths,
    )
    _print_test_summary(summary)
    _print_selected_policy_table(selected_policies, selected_policy_metrics)
    _print_delta_summary("INTERPOLATION DELTAS VS ORIGINAL 6144D RANDOM FOREST", original_delta_summary)
    _print_delta_summary("INTERPOLATION DELTAS VS AE-ONLY LATENT BASELINE", ae_delta_summary)

    return output_paths


def main() -> None:
    """CLI entry point for latent interpolation experiments."""
    parser = argparse.ArgumentParser(description="Run supervised AE latent interpolation.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    paths = run_supervised_ae_latent_interpolation(args.config)
    print(f"Saved supervised AE latent interpolation metrics to {paths['policy_metrics_path']}")


def _evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    evaluation_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seed: int,
    train_fraction: float,
    n_train: int,
    representation: str,
    latent_dim: float,
    policy_metadata: dict[str, Any],
    metric_names: list[str],
    policy_id: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model_config in _resolve_model_configs(config.get("models", ["ridge", "random_forest"])):
        model_name, model_kwargs = _parse_model_config(model_config)
        try:
            model = get_model(model_name, seed=seed, **model_kwargs)
        except ImportError:
            continue
        fitted_model = train_model(model, X_train, y_train)
        for split_name, (X_split, y_split) in evaluation_splits.items():
            predictions = predict_model(fitted_model, X_split)
            for metric_name in metric_names:
                records.append(
                    {
                        "seed": seed,
                        "train_fraction": train_fraction,
                        "n_train": n_train,
                        "representation": representation,
                        "model": model_name,
                        "latent_dim": latent_dim,
                        "policy_id": policy_id,
                        **_policy_metric_fields(policy_metadata),
                        "split": split_name,
                        "metric": metric_name,
                        "value": _compute_metric(metric_name, y_split, predictions),
                    }
                )
    return records


def _iter_interpolation_policies(
    config: dict[str, Any],
    seed: int,
) -> list[LatentInterpolationConfig]:
    interpolation = config.get("latent_interpolation", {})
    if not interpolation.get("enabled", True):
        return []
    multipliers = _as_list(interpolation.get("synthetic_multipliers", [1.0]))
    n_neighbors_values = _as_list(interpolation.get("n_neighbors", [5]))
    alpha_distributions = _as_list(interpolation.get("alpha_distributions", ["uniform"]))
    label_strategies = _as_list(interpolation.get("label_strategies", ["mixup_label"]))
    min_similarities = _as_list(interpolation.get("min_neighbor_similarities", [None]))
    max_teacher_stds = _as_list(interpolation.get("max_teacher_stds", [None]))
    blend_weights = _as_list(interpolation.get("teacher_blend_weights", [0.5]))
    policies: list[LatentInterpolationConfig] = []
    for multiplier, n_neighbors, alpha_distribution, label_strategy, min_similarity in product(
        multipliers,
        n_neighbors_values,
        alpha_distributions,
        label_strategies,
        min_similarities,
    ):
        teacher_std_values = max_teacher_stds if label_strategy in {"teacher_ensemble", "blended"} else [None]
        teacher_blend_values = blend_weights if label_strategy == "blended" else [blend_weights[0]]
        for max_teacher_std, blend_weight in product(teacher_std_values, teacher_blend_values):
            policies.append(
                LatentInterpolationConfig(
                    synthetic_multiplier=float(multiplier),
                    n_neighbors=int(n_neighbors),
                    alpha_min=float(interpolation.get("alpha_min", 0.2)),
                    alpha_max=float(interpolation.get("alpha_max", 0.8)),
                    alpha_distribution=str(alpha_distribution),
                    label_strategy=str(label_strategy),
                    teacher_models=[str(model) for model in interpolation.get("teacher_models", ["ridge", "random_forest"])],
                    teacher_blend_weight=float(blend_weight),
                    min_neighbor_similarity=_none_or_float(min_similarity),
                    max_teacher_std=_none_or_float(max_teacher_std),
                    clip_y_min=float(interpolation.get("clip_y_min", 0.0)),
                    clip_y_max=float(interpolation.get("clip_y_max", 100.0)),
                    random_state=seed,
                    candidates_per_real=int(interpolation.get("candidates_per_real", 20)),
                    max_candidates_per_source=(
                        int(interpolation["max_candidates_per_source"])
                        if interpolation.get("max_candidates_per_source") is not None
                        else None
                    ),
                )
            )
    return policies


def _select_policies(policy_metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config.get("selection", {})
    split = str(selection.get("split", "valid"))
    metric = str(selection.get("metric", "rmse"))
    lower_is_better = bool(selection.get("lower_is_better", True))
    candidates = policy_metrics.loc[
        policy_metrics["representation"].astype(str).str.startswith("supervised_ae_interpolation_")
        & (policy_metrics["split"] == split)
        & (policy_metrics["metric"] == metric)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values(
        "value",
        ascending=lower_is_better,
        kind="mergesort",
    )
    selected = candidates.groupby(
        ["seed", "train_fraction", "latent_dim", "model"],
        dropna=False,
        as_index=False,
    ).head(1)
    return selected.rename(columns={"value": f"{split}_{metric}"}).reset_index(drop=True)


def _mark_selected_policy_metrics(
    policy_metrics: pd.DataFrame,
    selected_policies: pd.DataFrame,
) -> pd.DataFrame:
    if selected_policies.empty:
        return policy_metrics
    key_columns = ["seed", "train_fraction", "latent_dim", "model", "policy_id"]
    selected_keys = selected_policies[key_columns].drop_duplicates().copy()
    selected_keys["selected_policy_marker"] = True
    marked = policy_metrics.drop(columns=["selected_policy"]).merge(
        selected_keys,
        on=key_columns,
        how="left",
    )
    marked["selected_policy"] = marked["selected_policy_marker"].eq(True)
    return marked.drop(columns=["selected_policy_marker"])


def _save_interpolation_vs_original_rf(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_test = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"].copy()
    baseline = policy_metrics.loc[
        (policy_metrics["split"] == "test")
        & (policy_metrics["representation"] == "original_6144")
        & (policy_metrics["model"] == "random_forest"),
        ["seed", "train_fraction", "metric", "value"],
    ].rename(columns={"value": "baseline_value"})
    by_seed = selected_test.merge(
        baseline,
        on=["seed", "train_fraction", "metric"],
        how="left",
    )
    by_seed["delta"] = by_seed["value"] - by_seed["baseline_value"]
    by_seed["delta_interpretation"] = np.where(
        by_seed["metric"].isin(["mae", "rmse"]),
        "negative_better_positive_worse",
        "positive_better_negative_worse",
    )
    by_seed.to_csv(output_paths["interpolation_vs_original_rf_by_seed_path"], index=False)
    summary = _delta_summary(by_seed)
    summary.to_csv(output_paths["interpolation_vs_original_rf_summary_path"], index=False)
    return by_seed, summary


def _save_interpolation_vs_ae_only(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_test = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"].copy()
    baseline = policy_metrics.loc[
        (policy_metrics["split"] == "test")
        & policy_metrics["representation"].astype(str).str.match(r"^supervised_ae_[0-9]+$"),
        ["seed", "train_fraction", "latent_dim", "model", "metric", "value"],
    ].rename(columns={"value": "baseline_value"})
    by_seed = selected_test.merge(
        baseline,
        on=["seed", "train_fraction", "latent_dim", "model", "metric"],
        how="left",
    )
    by_seed["delta"] = by_seed["value"] - by_seed["baseline_value"]
    by_seed["delta_interpretation"] = np.where(
        by_seed["metric"].isin(["mae", "rmse"]),
        "negative_better_positive_worse",
        "positive_better_negative_worse",
    )
    by_seed.to_csv(output_paths["interpolation_vs_ae_only_by_seed_path"], index=False)
    summary = _delta_summary(by_seed)
    summary.to_csv(output_paths["interpolation_vs_ae_only_summary_path"], index=False)
    return by_seed, summary


def _delta_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "train_fraction",
        "latent_dim",
        "model",
        "metric",
        "delta_interpretation",
    ]
    if by_seed.empty:
        return pd.DataFrame(columns=[*group_columns, "mean", "std", "count"])
    return (
        by_seed.groupby(group_columns, dropna=False)["delta"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def _summarize_for_reporting(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    report_metrics = pd.concat(
        [
            policy_metrics.loc[
                policy_metrics["representation"].isin(["original_6144"])
                | policy_metrics["representation"].astype(str).str.match(r"^supervised_ae_[0-9]+$")
            ],
            selected_policy_metrics,
        ],
        ignore_index=True,
    )
    group_columns = ["train_fraction", "representation", "model", "split", "metric"]
    return (
        report_metrics.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def _write_outputs(
    policy_metrics: pd.DataFrame,
    selected_policies: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    for key in [
        "policy_metrics_path",
        "selected_policies_path",
        "selected_policy_metrics_path",
        "summary_path",
        "audit_path",
        "candidate_audit_path",
    ]:
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
    policy_metrics.to_csv(output_paths["policy_metrics_path"], index=False)
    selected_policies.to_csv(output_paths["selected_policies_path"], index=False)
    selected_policy_metrics.to_csv(output_paths["selected_policy_metrics_path"], index=False)
    summary.to_csv(output_paths["summary_path"], index=False)
    audit.to_csv(output_paths["audit_path"], index=False)
    candidate_audit.to_csv(output_paths["candidate_audit_path"], index=False)


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


def _print_selected_policy_table(
    selected_policies: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
) -> None:
    if selected_policies.empty:
        return
    test_metrics = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"]
    pivot = test_metrics.pivot_table(
        index=["seed", "train_fraction", "latent_dim", "model", "policy_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    table = selected_policies.merge(
        pivot,
        on=["seed", "train_fraction", "latent_dim", "model", "policy_id"],
        how="left",
    )
    columns = [
        column
        for column in [
            "train_fraction",
            "latent_dim",
            "model",
            "synthetic_multiplier",
            "n_neighbors",
            "label_strategy",
            "valid_rmse",
            "rmse",
            "r2",
            "spearman",
        ]
        if column in table.columns
    ]
    rename = {"rmse": "test_rmse", "r2": "test_r2", "spearman": "test_spearman"}
    print("\nSELECTED INTERPOLATION POLICIES")
    print(table[columns].rename(columns=rename).to_string(index=False))


def _print_delta_summary(title: str, delta_summary: pd.DataFrame) -> None:
    if delta_summary.empty:
        return
    table = delta_summary.pivot_table(
        index=["train_fraction", "latent_dim", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    columns = [
        column
        for column in ["train_fraction", "latent_dim", "model", "mae", "rmse", "r2", "spearman"]
        if column in table.columns
    ]
    print(f"\n{title}")
    print("MAE/RMSE: negative is better. R2/Spearman: positive is better.")
    print(table[columns].to_string(index=False))


def _policy_metric_fields(metadata: dict[str, Any]) -> dict[str, object]:
    fields = _empty_policy_metadata()
    fields.update({key: metadata.get(key, fields[key]) for key in fields})
    return fields


def _empty_policy_metadata() -> dict[str, object]:
    return {
        "synthetic_multiplier": np.nan,
        "n_neighbors": np.nan,
        "alpha_distribution": "",
        "label_strategy": "",
        "teacher_blend_weight": np.nan,
        "min_neighbor_similarity": np.nan,
        "max_teacher_std": np.nan,
        "n_synthetic_train": 0,
        "n_candidates_generated": 0,
        "n_candidates_accepted": 0,
        "filter_acceptance_rate": np.nan,
        "mean_neighbor_similarity": np.nan,
        "mean_teacher_std": np.nan,
        "mean_synthetic_yield": np.nan,
        "std_synthetic_yield": np.nan,
    }


def _policy_id(
    seed: int,
    train_fraction: float,
    latent_dim: int,
    policy_index: int,
    policy: LatentInterpolationConfig,
) -> str:
    return (
        f"seed={seed}|frac={train_fraction}|latent={latent_dim}|policy={policy_index}|"
        f"mult={policy.synthetic_multiplier}|k={policy.n_neighbors}|"
        f"alpha={policy.alpha_distribution}|label={policy.label_strategy}|"
        f"blend={policy.teacher_blend_weight}|sim={policy.min_neighbor_similarity}|"
        f"std={policy.max_teacher_std}"
    )


def _resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output = config.get("output", {})
    directory = Path(output.get("directory", "results/supervised_ae_latent_interpolation"))
    return {
        "directory": directory,
        "policy_metrics_path": Path(output.get("policy_metrics_path", directory / "policy_metrics.csv")),
        "selected_policies_path": Path(output.get("selected_policies_path", directory / "selected_policies.csv")),
        "selected_policy_metrics_path": Path(
            output.get("selected_policy_metrics_path", directory / "selected_policy_metrics.csv")
        ),
        "summary_path": Path(output.get("summary_path", directory / "summary.csv")),
        "audit_path": Path(output.get("audit_path", directory / "synthetic_audit.csv")),
        "candidate_audit_path": Path(
            output.get(
                "candidate_audit_path",
                directory / "synthetic_candidate_audit.csv",
            )
        ),
        "interpolation_vs_original_rf_by_seed_path": Path(
            output.get("interpolation_vs_original_rf_by_seed_path", directory / "interpolation_vs_original_rf_by_seed.csv")
        ),
        "interpolation_vs_ae_only_by_seed_path": Path(
            output.get("interpolation_vs_ae_only_by_seed_path", directory / "interpolation_vs_ae_only_by_seed.csv")
        ),
        "interpolation_vs_original_rf_summary_path": Path(
            output.get("interpolation_vs_original_rf_summary_path", directory / "interpolation_vs_original_rf_summary.csv")
        ),
        "interpolation_vs_ae_only_summary_path": Path(
            output.get("interpolation_vs_ae_only_summary_path", directory / "interpolation_vs_ae_only_summary.csv")
        ),
    }


def _get_dataset_path(config: dict[str, Any]) -> str | Path:
    path = config.get("dataset", {}).get("path")
    if not path:
        raise ValueError("Config must define dataset.path.")
    return path


def _split_positions(split: pd.DataFrame) -> np.ndarray:
    return split.index.to_numpy(dtype=int)


def _training_row_ids(split: pd.DataFrame) -> list[str]:
    for column in ("source_row_id", "reaction_id", "canonical_reaction_key"):
        if column in split.columns:
            return [str(value) for value in split[column]]
    return [f"dataset_row:{int(index)}" for index in split.index]


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _none_or_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return float(value)


if __name__ == "__main__":
    main()
