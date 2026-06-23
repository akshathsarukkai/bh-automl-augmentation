"""Validation-only policy search for reaction-aware augmentation."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_recombine import (
    assign_synthetic_sample_weights,
    condition_recombine_pseudolabel,
)
from bh_augmentation.augmentation.ensemble_filter import (
    condition_recombine_ensemble_filter,
)
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_baseline import _compute_metric


def build_augmentation_policy_grid(search_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand augmentation search lists into deterministic policy dictionaries."""
    method = str(search_config.get("method", "condition_recombine_pseudolabel"))
    supported = {
        "condition_recombine_pseudolabel",
        "condition_recombine_ensemble_filter",
        "utility_guided_feature_gan",
    }
    if method not in supported:
        raise ValueError(f"Unsupported augmentation search method: {method}.")

    if method == "condition_recombine_ensemble_filter":
        return _build_ensemble_policy_grid(search_config)
    if method == "utility_guided_feature_gan":
        return _build_utility_gan_policy_grid(search_config)

    dimensions = {
        "synthetic_multipliers": search_config.get("synthetic_multipliers", [1.0]),
        "min_neighbor_similarities": search_config.get(
            "min_neighbor_similarities", [0.3]
        ),
        "teacher_models": search_config.get("teacher_models", ["random_forest"]),
        "random_states": search_config.get("random_states", [42]),
    }
    for name, values in dimensions.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"augmentation_search.{name} must be a non-empty list.")

    max_rows = search_config.get("max_synthetic_rows", 3000)
    weighting_config = dict(search_config.get("synthetic_weighting", {}))
    policies: list[dict[str, Any]] = []
    for multiplier, similarity, teacher, random_state in product(
        dimensions["synthetic_multipliers"],
        dimensions["min_neighbor_similarities"],
        dimensions["teacher_models"],
        dimensions["random_states"],
    ):
        multiplier = float(multiplier)
        similarity = float(similarity)
        if multiplier >= 3.0 and similarity < 0.85:
            continue
        if multiplier >= 2.0 and similarity < 0.80:
            continue
        teacher_name = str(teacher)
        policy = {
            "augmentation_method": method,
            "synthetic_multiplier": multiplier,
            "min_neighbor_similarity": similarity,
            "teacher_model": teacher_name,
            "random_state": int(random_state),
            "max_synthetic_rows": None if max_rows is None else int(max_rows),
            "synthetic_weighting": weighting_config,
        }
        policy["policy_id"] = _policy_id(policy)
        policies.append(policy)
    return policies


def evaluate_augmentation_policy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    policy: dict[str, Any],
    feature_config: dict[str, Any],
    model_name: str,
    metrics: list[str],
    seed: int,
    model_kwargs: dict[str, Any] | None = None,
    augmentation_cache: dict[object, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one policy on real training data and evaluate it only on validation."""
    augmented_train = _apply_policy(
        train_df, policy, feature_config, cache=augmentation_cache
    )
    evaluation_key = (
        "validation_evaluation",
        model_name,
        repr(sorted((model_kwargs or {}).items())),
        _augmented_signature(augmented_train),
    )
    cached_evaluation = (
        augmentation_cache.get(evaluation_key)
        if augmentation_cache is not None
        else None
    )
    if cached_evaluation is None:
        combined = pd.concat(
            [
                augmented_train.assign(__split="train"),
                valid_df.assign(__split="valid"),
            ],
            ignore_index=True,
        )
        X, y, _ = build_feature_matrix(combined, feature_config)
        train_mask = combined["__split"].to_numpy() == "train"
        valid_mask = ~train_mask
        model = train_model(
            get_model(model_name, seed=seed, **(model_kwargs or {})),
            X[train_mask],
            y[train_mask],
            sample_weight=_training_sample_weights(augmented_train, policy),
        )
        predictions = predict_model(model, X[valid_mask])
        metric_values = {
            metric: _compute_metric(metric, y[valid_mask], predictions)
            for metric in metrics
        }
        cached_evaluation = (metric_values, int(X.shape[1]))
        if augmentation_cache is not None:
            augmentation_cache[evaluation_key] = cached_evaluation
    metric_values, n_features = cached_evaluation
    metadata = _policy_metadata(policy, augmented_train, len(train_df))
    metadata.update(
        {
            "feature_kind": str(feature_config.get("kind", "custom")),
            "n_features": n_features,
            "model": model_name,
        }
    )
    rows = [
        {
            **metadata,
            "selected_policy": False,
            "split": "valid",
            "metric": metric,
            "value": metric_values[metric],
        }
        for metric in metrics
    ]
    return pd.DataFrame(rows), metadata


def select_best_policy(
    policy_metrics: pd.DataFrame,
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Select one policy from validation metrics with deterministic tie-breaking."""
    required = {
        "split",
        "metric",
        "value",
        "policy_id",
        "synthetic_multiplier",
        "min_neighbor_similarity",
        "teacher_model",
        "augmentation_random_state",
    }
    missing = sorted(required - set(policy_metrics.columns))
    if missing:
        raise ValueError(f"Policy metrics are missing columns: {', '.join(missing)}.")

    candidates = policy_metrics.loc[
        (policy_metrics["split"] == "valid")
        & (policy_metrics["metric"] == selection_metric)
    ].copy()
    candidates["value"] = pd.to_numeric(candidates["value"], errors="coerce")
    candidates = candidates.dropna(subset=["value"])
    if candidates.empty:
        raise ValueError(
            f"No validation rows found for selection metric {selection_metric!r}."
        )

    candidates = candidates.sort_values(
        by=[
            "value",
            "synthetic_multiplier",
            "min_neighbor_similarity",
            "teacher_model",
            "augmentation_random_state",
            "policy_id",
        ],
        ascending=[
            lower_is_better,
            True,
            False,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )
    return candidates.iloc[0].to_dict()


def apply_augmentation_policy(
    train_df: pd.DataFrame,
    policy: dict[str, Any],
    feature_config: dict[str, Any],
    cache: dict[object, object] | None = None,
) -> pd.DataFrame:
    """Apply a supported policy to real training rows only."""
    return _apply_policy(train_df, policy, feature_config, cache=cache)


def policy_metadata(
    policy: dict[str, Any],
    augmented_train: pd.DataFrame,
    n_real_train: int,
) -> dict[str, Any]:
    """Return serializable policy and augmented-row metadata."""
    return _policy_metadata(policy, augmented_train, n_real_train)


def _apply_policy(
    train_df: pd.DataFrame,
    policy: dict[str, Any],
    feature_config: dict[str, Any],
    cache: dict[object, object] | None = None,
) -> pd.DataFrame:
    method = str(policy.get("augmentation_method", ""))
    if method == "condition_recombine_pseudolabel":
        augmented = condition_recombine_pseudolabel(
            train_df,
            feature_config,
            synthetic_multiplier=float(policy["synthetic_multiplier"]),
            max_synthetic_rows=policy.get("max_synthetic_rows", 3000),
            teacher_model=str(policy["teacher_model"]),
            min_neighbor_similarity=float(policy["min_neighbor_similarity"]),
            random_state=int(policy["random_state"]),
        )
    elif method == "condition_recombine_ensemble_filter":
        ensemble_config = {
            **dict(policy.get("ensemble_config", {})),
            "synthetic_multiplier": float(policy["synthetic_multiplier"]),
            "max_synthetic_rows": policy.get("max_synthetic_rows", 3000),
            "min_neighbor_similarity": float(policy["min_neighbor_similarity"]),
            "uncertainty_filter": {
                **dict(
                    policy.get("ensemble_config", {}).get(
                        "uncertainty_filter", {}
                    )
                ),
                "enabled": True,
                "max_teacher_std": float(policy["max_teacher_std"]),
                "max_prediction_range": float(policy["max_prediction_range"]),
            },
        }
        augmented, _ = condition_recombine_ensemble_filter(
            train_df,
            feature_config,
            ensemble_config,
            random_state=int(policy["random_state"]),
            cache=cache,
        )
    else:
        raise ValueError(f"Unsupported augmentation policy method: {method}.")
    metadata = augmented.attrs.get("augmentation_metadata")
    weighted = assign_synthetic_sample_weights(
        augmented,
        policy.get("synthetic_weighting"),
    )
    if metadata is not None:
        weighted.attrs["augmentation_metadata"] = metadata
    return weighted


def _policy_metadata(
    policy: dict[str, Any],
    augmented_train: pd.DataFrame,
    n_real_train: int,
) -> dict[str, Any]:
    synthetic = (
        augmented_train.get(
            "is_synthetic", pd.Series(False, index=augmented_train.index)
        )
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    weighting = dict(policy.get("synthetic_weighting", {}))
    weighting_enabled = bool(weighting.get("enabled", False))
    synthetic_count = int(synthetic.sum())
    synthetic_weights = pd.to_numeric(
        augmented_train.loc[synthetic, "sample_weight"], errors="coerce"
    ).dropna()
    ensemble_metadata = augmented_train.attrs.get("augmentation_metadata", {})
    return {
        "policy_id": str(policy["policy_id"]),
        "augmentation_method": str(policy["augmentation_method"]),
        "synthetic_multiplier": float(policy["synthetic_multiplier"]),
        "min_neighbor_similarity": float(policy["min_neighbor_similarity"]),
        "teacher_model": str(policy["teacher_model"]),
        "augmentation_random_state": int(policy["random_state"]),
        "n_real_train": int(n_real_train),
        "n_synthetic_train": synthetic_count,
        "augmentation_factor": float(len(augmented_train) / n_real_train)
        if n_real_train
        else 0.0,
        "synthetic_weighting_enabled": weighting_enabled,
        "mean_synthetic_weight": float(synthetic_weights.mean())
        if not synthetic_weights.empty
        else float("nan"),
        "min_synthetic_weight": float(synthetic_weights.min())
        if not synthetic_weights.empty
        else float("nan"),
        "max_synthetic_weight": float(synthetic_weights.max())
        if not synthetic_weights.empty
        else float("nan"),
        "max_teacher_std": policy.get("max_teacher_std", float("nan")),
        "max_prediction_range": policy.get(
            "max_prediction_range", float("nan")
        ),
        "n_candidates_generated": int(
            ensemble_metadata.get("n_candidates_generated", synthetic_count)
        ),
        "acceptance_rate": float(
            ensemble_metadata.get(
                "acceptance_rate", 1.0 if synthetic_count else 0.0
            )
        ),
        "mean_teacher_std": float(
            ensemble_metadata.get("mean_teacher_std", float("nan"))
        ),
        "mean_prediction_range": float(
            ensemble_metadata.get("mean_prediction_range", float("nan"))
        ),
        "mean_nearest_train_similarity": float(
            ensemble_metadata.get(
                "mean_nearest_train_similarity", float("nan")
            )
        ),
    }


def _training_sample_weights(
    augmented_train: pd.DataFrame,
    policy: dict[str, Any],
) -> np.ndarray | None:
    weighting = policy.get("synthetic_weighting", {})
    if not bool(weighting.get("enabled", False)):
        return None
    return augmented_train["sample_weight"].to_numpy(dtype=float)


def _policy_id(policy: dict[str, Any]) -> str:
    if policy["augmentation_method"] == "condition_recombine_ensemble_filter":
        return (
            f"{policy['augmentation_method']}"
            f"__mult_{_number(policy['synthetic_multiplier'])}"
            f"__sim_{_number(policy['min_neighbor_similarity'])}"
            f"__std_{_number(policy['max_teacher_std'])}"
            f"__range_{_number(policy['max_prediction_range'])}"
            f"__seed_{policy['random_state']}"
        )
    if policy["augmentation_method"] == "utility_guided_feature_gan":
        return (
            f"{policy['augmentation_method']}"
            f"__mult_{_number(policy['synthetic_multiplier'])}"
            f"__noise_{policy['noise_dim']}"
            f"__hidden_{policy['hidden_dim']}"
            f"__svd_{policy['svd_components']}"
            f"__epochs_{policy['n_epochs']}"
            f"__rounds_{policy['utility_rounds']}"
            f"__seed_{policy['random_state']}"
        )
    teacher = {"random_forest": "rf"}.get(
        str(policy["teacher_model"]), str(policy["teacher_model"])
    )
    return (
        f"{policy['augmentation_method']}"
        f"__mult_{_number(policy['synthetic_multiplier'])}"
        f"__sim_{_number(policy['min_neighbor_similarity'])}"
        f"__teacher_{teacher}"
        f"__seed_{policy['random_state']}"
    )


def _number(value: object) -> str:
    return format(float(value), "g")


def _augmented_signature(augmented_train: pd.DataFrame) -> tuple[tuple[object, ...], ...]:
    synthetic = augmented_train.get(
        "is_synthetic", pd.Series(False, index=augmented_train.index)
    ).fillna(False)
    rows = augmented_train.loc[synthetic.astype(bool)]
    return tuple(
        (
            str(row.get("reaction_smiles", "")),
            float(row.get("yield", 0.0)),
            float(row.get("sample_weight", 1.0)),
        )
        for _, row in rows.iterrows()
    )


def _build_ensemble_policy_grid(
    search_config: dict[str, Any],
) -> list[dict[str, Any]]:
    dimensions = {
        "synthetic_multipliers": search_config.get("synthetic_multipliers", [1.0]),
        "min_neighbor_similarities": search_config.get(
            "min_neighbor_similarities", [0.8]
        ),
        "max_teacher_stds": search_config.get("max_teacher_stds", [12.0]),
        "max_prediction_ranges": search_config.get(
            "max_prediction_ranges", [35.0]
        ),
        "random_states": search_config.get("random_states", [42]),
    }
    for name, values in dimensions.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"augmentation_search.{name} must be a non-empty list.")

    max_rows = search_config.get("max_synthetic_rows", 3000)
    ensemble_config = dict(search_config.get("ensemble_config", {}))
    weighting = dict(search_config.get("synthetic_weighting", {}))
    policies: list[dict[str, Any]] = []
    for multiplier, similarity, max_std, max_range, random_state in product(
        dimensions["synthetic_multipliers"],
        dimensions["min_neighbor_similarities"],
        dimensions["max_teacher_stds"],
        dimensions["max_prediction_ranges"],
        dimensions["random_states"],
    ):
        policy = {
            "augmentation_method": "condition_recombine_ensemble_filter",
            "synthetic_multiplier": float(multiplier),
            "min_neighbor_similarity": float(similarity),
            "max_teacher_std": float(max_std),
            "max_prediction_range": float(max_range),
            "teacher_model": "ensemble",
            "random_state": int(random_state),
            "max_synthetic_rows": None if max_rows is None else int(max_rows),
            "ensemble_config": ensemble_config,
            "synthetic_weighting": weighting,
        }
        policy["policy_id"] = _policy_id(policy)
        policies.append(policy)
    return policies


def _build_utility_gan_policy_grid(search_config: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = {
        "synthetic_multipliers": search_config.get("synthetic_multipliers", [1.0]),
        "noise_dims": search_config.get("noise_dims", [64]),
        "hidden_dims": search_config.get("hidden_dims", [256]),
        "svd_components": search_config.get("svd_components", [128]),
        "n_epochs": search_config.get("n_epochs", [200]),
        "batch_sizes": search_config.get("batch_sizes", [32]),
        "learning_rates": search_config.get("learning_rates", [0.0002]),
        "gradient_penalties": search_config.get("gradient_penalties", [10.0]),
        "utility_rounds": search_config.get("utility_rounds", [5]),
        "random_states": search_config.get("random_states", [42]),
    }
    for name, values in dimensions.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"augmentation_search.{name} must be a non-empty list.")

    base_config = dict(search_config.get("utility_config", {}))
    max_rows = search_config.get("max_synthetic_rows", base_config.get("max_synthetic_rows", 1000))
    policies: list[dict[str, Any]] = []
    for multiplier, noise_dim, hidden_dim, svd_components, epochs, batch_size, lr, gp, rounds, random_state in product(
        dimensions["synthetic_multipliers"],
        dimensions["noise_dims"],
        dimensions["hidden_dims"],
        dimensions["svd_components"],
        dimensions["n_epochs"],
        dimensions["batch_sizes"],
        dimensions["learning_rates"],
        dimensions["gradient_penalties"],
        dimensions["utility_rounds"],
        dimensions["random_states"],
    ):
        policy = {
            "augmentation_method": "utility_guided_feature_gan",
            "synthetic_multiplier": float(multiplier),
            "min_neighbor_similarity": float(
                base_config.get("filters", {}).get("min_nearest_similarity", 0.6)
            ),
            "teacher_model": "teacher_ensemble",
            "random_state": int(random_state),
            "max_synthetic_rows": int(max_rows),
            "noise_dim": int(noise_dim),
            "hidden_dim": int(hidden_dim),
            "svd_components": int(svd_components),
            "n_epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "gradient_penalty": float(gp),
            "utility_rounds": int(rounds),
            "utility_config": base_config,
        }
        policy["policy_id"] = _policy_id(policy)
        policies.append(policy)
    return policies
