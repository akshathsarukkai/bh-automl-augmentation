"""Command-line runner for safe augmentation experiments."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_recombine import (
    assign_synthetic_sample_weights,
    condition_recombine_pseudolabel,
)
from bh_augmentation.augmentation.order_permutation import permute_reaction_components
from bh_augmentation.augmentation.ensemble_filter import (
    condition_recombine_ensemble_filter,
)
from bh_augmentation.augmentation.search import (
    apply_augmentation_policy,
    build_augmentation_policy_grid,
    evaluate_augmentation_policy,
    policy_metadata,
    select_best_policy,
)
from bh_augmentation.augmentation.smiles_randomization import augment_randomized_smiles
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.reporting.make_report import save_metrics_csv
from bh_augmentation.run_baseline import (
    _compute_metric,
    _create_baseline_folds,
    _create_split_variants,
    _get_dataset_path,
    _get_group_column,
    _parse_model_config,
    _resolve_feature_config,
    _resolve_feature_configs,
    _resolve_model_configs,
    _save_baseline_split_metadata,
    _save_split_metadata,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed


def run_augmentation(config_path: str | Path) -> Path:
    """Run no-augmentation and configured safe-augmentation experiments."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = clean_buchwald_hartwig(raw_df).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run augmentation experiment.")

    if config.get("augmentation_search", {}).get("enabled", False):
        return _run_augmentation_search(config, df, seed)

    split_variants = _create_split_variants(df, config, seed)
    feature_config = _resolve_feature_config(config.get("features", {}), df)
    augmentation_config = config.get("augmentation", {})

    model_configs = config.get("models", ["ridge"])
    metric_names = config.get("metrics", ["rmse", "mae", "r2"])
    split_method = str(config.get("splits", {}).get("method", "random"))
    group_column = _get_group_column(config.get("splits", {}))
    records: list[dict[str, object]] = []

    _save_split_metadata(split_variants, config)

    for train_fraction, splits in split_variants:
        train_variants = _build_training_variants(
            splits["train"],
            augmentation_config,
            feature_config,
            seed,
        )

        for variant_name, train_df in train_variants.items():
            train_df = _with_augmentation_metadata(train_df, is_augmented_default=False)
            valid_df = _with_augmentation_metadata(splits["valid"], is_augmented_default=False)
            test_df = _with_augmentation_metadata(splits["test"], is_augmented_default=False)

            combined_df = pd.concat(
                [
                    train_df.assign(__split="train"),
                    valid_df.assign(__split="valid"),
                    test_df.assign(__split="test"),
                ],
                ignore_index=True,
            )
            X, y, _ = build_feature_matrix(combined_df, feature_config)
            feature_kind = str(feature_config.get("kind", "custom"))
            n_features = int(X.shape[1])
            train_mask = combined_df["__split"].to_numpy() == "train"
            variant_metadata = _variant_metrics(
                variant_name,
                train_df,
                len(splits["train"]),
                augmentation_config,
            )

            for model_config in model_configs:
                model_name, model_kwargs = _parse_model_config(model_config)
                model = get_model(model_name, seed=seed, **model_kwargs)
                if bool(variant_metadata["synthetic_weighting_enabled"]):
                    fitted_model = train_model(
                        model,
                        X[train_mask],
                        y[train_mask],
                        sample_weight=train_df["sample_weight"].to_numpy(dtype=float),
                    )
                else:
                    fitted_model = train_model(model, X[train_mask], y[train_mask])

                for split_name in ["valid", "test"]:
                    eval_mask = combined_df["__split"].to_numpy() == split_name
                    predictions = predict_model(fitted_model, X[eval_mask])
                    y_true = y[eval_mask]
                    eval_augmented_rows = int(combined_df.loc[eval_mask, "is_augmented"].sum())
                    for metric_name in metric_names:
                        records.append(
                            {
                                "train_fraction": train_fraction,
                                "split_method": split_method,
                                "group_column": group_column,
                                "feature_kind": feature_kind,
                                "n_features": n_features,
                                "augmentation": variant_name,
                                **variant_metadata,
                                "model": model_name,
                                "split": split_name,
                                "metric": metric_name,
                                "value": _compute_metric(metric_name, y_true, predictions),
                                "train_rows": int(train_mask.sum()),
                                "augmented_train_rows": int(train_df["is_augmented"].sum()),
                                "eval_rows": int(eval_mask.sum()),
                                "eval_augmented_rows": eval_augmented_rows,
                            }
                        )

    return save_metrics_csv(records, _get_metrics_output_path(config))


def _run_augmentation_search(
    config: dict[str, Any],
    df: pd.DataFrame,
    seed: int,
) -> Path:
    search_config = _search_config_with_defaults(config)
    if str(search_config.get("selection_split", "valid")) != "valid":
        raise ValueError("augmentation_search.selection_split must be 'valid'.")

    policies = build_augmentation_policy_grid(search_config)
    folds = _create_baseline_folds(df, config, seed)
    feature_configs = _resolve_feature_configs(config.get("features", {}), df)
    model_configs = _resolve_model_configs(config.get("models", ["ridge"]))
    metrics = list(config.get("metrics", ["rmse", "mae", "r2"]))
    selection_metric = str(search_config.get("selection_metric", "rmse"))
    lower_is_better = bool(search_config.get("lower_is_better", True))
    split_method = str(config.get("splits", {}).get("method", "random"))
    group_column = _get_group_column(config.get("splits", {}))

    search_frames: list[pd.DataFrame] = []
    selected_policy_rows: list[dict[str, object]] = []
    selected_metric_rows: list[dict[str, object]] = []
    _save_baseline_split_metadata(folds, config)

    for feature_config in feature_configs:
        for fold_index, (train_fraction, heldout_group, splits) in enumerate(folds):
            augmentation_cache: dict[object, object] = {}
            context = {
                "fold_index": fold_index,
                "train_fraction": train_fraction,
                "split_method": split_method,
                "group_column": group_column,
                "heldout_group_value": heldout_group,
            }
            for model_config in model_configs:
                model_name, model_kwargs = _parse_model_config(model_config)
                policy_frames: list[pd.DataFrame] = []
                for policy in policies:
                    frame, _ = evaluate_augmentation_policy(
                        splits["train"],
                        splits["valid"],
                        policy,
                        feature_config,
                        model_name,
                        metrics,
                        seed,
                        model_kwargs=model_kwargs,
                        augmentation_cache=augmentation_cache,
                    )
                    for key, value in context.items():
                        frame[key] = value
                    policy_frames.append(frame)

                policy_metrics = pd.concat(policy_frames, ignore_index=True)
                selected = select_best_policy(
                    policy_metrics,
                    selection_metric=selection_metric,
                    lower_is_better=lower_is_better,
                )
                policy_metrics.loc[
                    policy_metrics["policy_id"] == selected["policy_id"],
                    "selected_policy",
                ] = True
                search_frames.append(policy_metrics)
                selected_policy = next(
                    policy for policy in policies if policy["policy_id"] == selected["policy_id"]
                )
                selected_policy_rows.append(
                    {
                        **context,
                        "feature_kind": selected["feature_kind"],
                        "n_features": selected["n_features"],
                        "model": model_name,
                        "selection_metric": selection_metric,
                        "selection_value": selected["value"],
                        "lower_is_better": lower_is_better,
                        **{
                            key: selected[key]
                            for key in [
                                "policy_id",
                                "augmentation_method",
                                "synthetic_multiplier",
                                "min_neighbor_similarity",
                                "teacher_model",
                                "augmentation_random_state",
                                "n_real_train",
                                "n_synthetic_train",
                                "augmentation_factor",
                                "synthetic_weighting_enabled",
                                "mean_synthetic_weight",
                                "min_synthetic_weight",
                                "max_synthetic_weight",
                                "max_teacher_std",
                                "max_prediction_range",
                                "n_candidates_generated",
                                "acceptance_rate",
                                "mean_teacher_std",
                                "mean_prediction_range",
                                "mean_nearest_train_similarity",
                            ]
                        },
                    }
                )
                selected_metric_rows.extend(
                    _evaluate_selected_policy(
                        splits,
                        selected_policy,
                        feature_config,
                        model_name,
                        model_kwargs,
                        metrics,
                        seed,
                        context,
                        augmentation_cache,
                    )
                )

    output = config.get("output", {})
    search_path = Path(
        output.get(
            "search_metrics_path",
            "results/augmentation/policy_search_metrics.csv",
        )
    )
    selected_path = Path(
        output.get(
            "selected_policies_path",
            "results/augmentation/selected_policies.csv",
        )
    )
    metrics_path = _get_metrics_output_path(config)
    search_records = pd.concat(search_frames, ignore_index=True).to_dict("records")
    save_metrics_csv(search_records, search_path)
    save_metrics_csv(selected_policy_rows, selected_path)
    return save_metrics_csv(selected_metric_rows, metrics_path)


def _evaluate_selected_policy(
    splits: dict[str, pd.DataFrame],
    policy: dict[str, Any],
    feature_config: dict[str, Any],
    model_name: str,
    model_kwargs: dict[str, Any],
    metrics: list[str],
    seed: int,
    context: dict[str, object],
    augmentation_cache: dict[object, object],
) -> list[dict[str, object]]:
    augmented_train = apply_augmentation_policy(
        splits["train"], policy, feature_config, cache=augmentation_cache
    )
    combined = pd.concat(
        [
            augmented_train.assign(__split="train"),
            splits["valid"].assign(__split="valid"),
            splits["test"].assign(__split="test"),
        ],
        ignore_index=True,
    )
    X, y, _ = build_feature_matrix(combined, feature_config)
    train_mask = combined["__split"].to_numpy() == "train"
    fitted = train_model(
        get_model(model_name, seed=seed, **model_kwargs),
        X[train_mask],
        y[train_mask],
        sample_weight=(
            augmented_train["sample_weight"].to_numpy(dtype=float)
            if bool(policy.get("synthetic_weighting", {}).get("enabled", False))
            else None
        ),
    )
    metadata = policy_metadata(policy, augmented_train, len(splits["train"]))
    metadata.update(
        {
            "feature_kind": str(feature_config.get("kind", "custom")),
            "n_features": int(X.shape[1]),
            "model": model_name,
        }
    )
    records: list[dict[str, object]] = []
    for split_name in ["valid", "test"]:
        eval_mask = combined["__split"].to_numpy() == split_name
        predictions = predict_model(fitted, X[eval_mask])
        for metric in metrics:
            records.append(
                {
                    **context,
                    **metadata,
                    "selected_policy": True,
                    "split": split_name,
                    "metric": metric,
                    "value": _compute_metric(metric, y[eval_mask], predictions),
                }
            )
    return records


def _search_config_with_defaults(config: dict[str, Any]) -> dict[str, Any]:
    search = dict(config.get("augmentation_search", {}))
    method = str(search.get("method", "condition_recombine_pseudolabel"))
    fixed = config.get("augmentation", {}).get(method, {})
    defaults = {
        "method": method,
        "synthetic_multipliers": [fixed.get("synthetic_multiplier", 1.0)],
        "min_neighbor_similarities": [
            fixed.get("min_neighbor_similarity", 0.3)
        ],
        "teacher_models": [fixed.get("teacher_model", "random_forest")],
        "random_states": [fixed.get("random_state", config.get("seed", 42))],
        "max_synthetic_rows": fixed.get("max_synthetic_rows", 3000),
        "synthetic_weighting": fixed.get("synthetic_weighting", {}),
    }
    if method == "condition_recombine_ensemble_filter":
        uncertainty = fixed.get("uncertainty_filter", {})
        defaults.update(
            {
                "max_teacher_stds": [uncertainty.get("max_teacher_std", 12.0)],
                "max_prediction_ranges": [
                    uncertainty.get("max_prediction_range", 35.0)
                ],
                "ensemble_config": fixed,
            }
        )
    return {**defaults, **search}


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_augmentation`."""
    parser = argparse.ArgumentParser(description="Run safe augmentation comparisons.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    output_path = run_augmentation(args.config)
    print(f"Saved safe augmentation metrics to {output_path}")


def _build_training_variants(
    train_df: pd.DataFrame,
    augmentation_config: dict[str, Any],
    feature_config: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    variants = {"none": _with_augmentation_metadata(train_df, is_augmented_default=False)}

    smiles_config = augmentation_config.get("smiles_randomization", {})
    order_config = augmentation_config.get("order_permutation", {})
    combined_config = augmentation_config.get("combined", {})
    recombine_config = augmentation_config.get(
        "condition_recombine_pseudolabel", {}
    )
    ensemble_config = augmentation_config.get(
        "condition_recombine_ensemble_filter", {}
    )

    if recombine_config.get("enabled", False):
        max_rows = recombine_config.get("max_synthetic_rows", 3000)
        augmented = condition_recombine_pseudolabel(
            train_df,
            feature_config,
            synthetic_multiplier=float(
                recombine_config.get("synthetic_multiplier", 1.0)
            ),
            max_synthetic_rows=None if max_rows is None else int(max_rows),
            teacher_model=str(recombine_config.get("teacher_model", "random_forest")),
            min_neighbor_similarity=float(
                recombine_config.get("min_neighbor_similarity", 0.3)
            ),
            random_state=int(recombine_config.get("random_state", seed)),
        )
        variants["condition_recombine_pseudolabel"] = assign_synthetic_sample_weights(
            augmented,
            recombine_config.get("synthetic_weighting"),
        )

    if ensemble_config.get("enabled", False):
        augmented, _ = condition_recombine_ensemble_filter(
            train_df,
            feature_config,
            ensemble_config,
            random_state=int(ensemble_config.get("random_state", seed)),
        )
        variants["condition_recombine_ensemble_filter"] = (
            assign_synthetic_sample_weights(
                augmented,
                ensemble_config.get("synthetic_weighting"),
            )
        )

    if smiles_config.get("enabled", False):
        _warn_legacy_augmentation("smiles_randomization")
        variants["randomized_smiles"] = _apply_smiles_randomization(
            train_df,
            smiles_config,
            feature_config,
            seed,
        )
    if order_config.get("enabled", False):
        _warn_legacy_augmentation("order_permutation")
        variants["order_permutation"] = _apply_order_permutation(
            train_df,
            order_config,
            seed,
        )
    if combined_config.get("enabled", False):
        _warn_legacy_augmentation("combined_safe")
        combined = train_df
        if smiles_config.get("enabled", False):
            combined = _apply_smiles_randomization(combined, smiles_config, feature_config, seed)
        if order_config.get("enabled", False):
            order_augmented = _apply_order_permutation(combined, order_config, seed)
            combined = pd.concat([combined, order_augmented[order_augmented["is_augmented"]]], ignore_index=True)
        variants["combined_safe"] = _with_augmentation_metadata(combined, is_augmented_default=False)

    return variants


def _apply_smiles_randomization(
    train_df: pd.DataFrame,
    smiles_config: dict[str, Any],
    feature_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows_to_augment = _select_augmentation_rows(
        train_df,
        float(smiles_config.get("ratio", 1.0)),
        int(smiles_config.get("random_state", seed)),
    )
    untouched = train_df.drop(index=rows_to_augment.index)
    augmented = augment_randomized_smiles(
        rows_to_augment,
        smiles_columns=smiles_config.get("smiles_columns", feature_config.get("smiles_columns", [])),
        n_augments=int(smiles_config.get("n_augments", smiles_config.get("n_variants", 1))),
        seed=int(smiles_config.get("random_state", seed)),
        include_original=True,
    )
    return _with_augmentation_metadata(
        pd.concat([untouched, augmented], ignore_index=True),
        is_augmented_default=False,
    )


def _apply_order_permutation(
    train_df: pd.DataFrame,
    order_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows_to_augment = _select_augmentation_rows(
        train_df,
        float(order_config.get("ratio", 1.0)),
        int(order_config.get("random_state", seed)),
    )
    untouched = train_df.drop(index=rows_to_augment.index)
    augmented = permute_reaction_components(
        rows_to_augment,
        component_columns=order_config.get("component_columns", []),
        n_permutations=int(order_config.get("n_permutations", order_config.get("max_permutations", 1))),
        seed=int(order_config.get("random_state", seed)),
        include_original=True,
    )
    return _with_augmentation_metadata(
        pd.concat([untouched, augmented], ignore_index=True),
        is_augmented_default=False,
    )


def _select_augmentation_rows(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    if not 0 <= ratio <= 1:
        raise ValueError("Augmentation ratio must be between 0 and 1.")
    if ratio == 0 or df.empty:
        return df.iloc[0:0].copy()

    n_rows = max(1, int(np.floor(len(df) * ratio)))
    sampled_indices = df.sample(n=n_rows, random_state=seed).index
    return df.loc[sampled_indices].copy()


def _warn_legacy_augmentation(name: str) -> None:
    warnings.warn(
        f"Augmentation '{name}' is legacy and is not used by active configs. "
        "Use condition_recombine_pseudolabel for reaction_smiles data.",
        DeprecationWarning,
        stacklevel=3,
    )


def _with_augmentation_metadata(
    df: pd.DataFrame,
    is_augmented_default: bool,
) -> pd.DataFrame:
    result = df.copy()
    if "is_augmented" not in result.columns:
        result["is_augmented"] = is_augmented_default
    result["is_augmented"] = result["is_augmented"].fillna(is_augmented_default).astype(bool)
    if "augmentation_type" not in result.columns:
        result["augmentation_type"] = "none" if not is_augmented_default else "safe_augmentation"
    if "source_reaction_id" not in result.columns:
        if "reaction_id" in result.columns:
            result["source_reaction_id"] = result["reaction_id"].astype(str)
        else:
            result["source_reaction_id"] = result.index.astype(str)
    return result


def _get_metrics_output_path(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    return Path(output_config.get("metrics_path", "results/augmentation/safe_aug_metrics.csv"))


def _variant_metrics(
    variant_name: str,
    train_df: pd.DataFrame,
    n_real_train: int,
    augmentation_config: dict[str, Any],
) -> dict[str, object]:
    synthetic_mask = (
        train_df.get("is_synthetic", pd.Series(False, index=train_df.index))
        .fillna(False)
        .astype(bool)
    )
    metadata: dict[str, object] = {
        "augmentation_method": variant_name,
        "n_real_train": n_real_train,
        "n_synthetic_train": int(synthetic_mask.sum()),
        "synthetic_multiplier": 0.0,
        "teacher_model": "",
        "min_neighbor_similarity": np.nan,
        "augmentation_random_state": np.nan,
        "synthetic_weighting_enabled": False,
        "mean_synthetic_weight": np.nan,
        "min_synthetic_weight": np.nan,
        "max_synthetic_weight": np.nan,
        "max_teacher_std": np.nan,
        "max_prediction_range": np.nan,
        "n_candidates_generated": int(synthetic_mask.sum()),
        "acceptance_rate": 1.0 if synthetic_mask.any() else 0.0,
        "mean_teacher_std": np.nan,
        "mean_prediction_range": np.nan,
        "mean_nearest_train_similarity": np.nan,
    }
    if variant_name in {
        "condition_recombine_pseudolabel",
        "condition_recombine_ensemble_filter",
    }:
        settings = augmentation_config.get(variant_name, {})
        metadata.update(
            {
                "synthetic_multiplier": float(
                    settings.get("synthetic_multiplier", 1.0)
                ),
                "teacher_model": (
                    "ensemble"
                    if variant_name == "condition_recombine_ensemble_filter"
                    else str(settings.get("teacher_model", "random_forest"))
                ),
                "min_neighbor_similarity": float(
                    settings.get("min_neighbor_similarity", 0.3)
                ),
                "augmentation_random_state": int(settings.get("random_state", 42)),
            }
        )
        weighting_enabled = bool(
            settings.get("synthetic_weighting", {}).get("enabled", False)
        )
        metadata["synthetic_weighting_enabled"] = weighting_enabled
        synthetic_weights = pd.to_numeric(
            train_df.loc[synthetic_mask, "sample_weight"], errors="coerce"
        ).dropna()
        if not synthetic_weights.empty:
            metadata.update(
                {
                    "mean_synthetic_weight": float(synthetic_weights.mean()),
                    "min_synthetic_weight": float(synthetic_weights.min()),
                    "max_synthetic_weight": float(synthetic_weights.max()),
                }
            )
        if variant_name == "condition_recombine_ensemble_filter":
            uncertainty = settings.get("uncertainty_filter", {})
            ensemble_metadata = train_df.attrs.get("augmentation_metadata", {})
            metadata.update(
                {
                    "max_teacher_std": float(
                        uncertainty.get("max_teacher_std", 12.0)
                    ),
                    "max_prediction_range": float(
                        uncertainty.get("max_prediction_range", 35.0)
                    ),
                    **ensemble_metadata,
                }
            )
    return metadata


if __name__ == "__main__":
    main()
