"""Objective functions for small Optuna AutoML searches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.order_permutation import permute_reaction_components
from bh_augmentation.augmentation.smiles_randomization import augment_randomized_smiles
from bh_augmentation.evaluation.metrics import rmse
from bh_augmentation.evaluation.topk import top_k_hit_rate
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_baseline import _resolve_feature_config


MODEL_NAME_MAP = {
    "ridge": "ridge",
    "random_forest": "random_forest",
    "extratrees": "extra_trees",
    "extra_trees": "extra_trees",
}


def suggest_trial_config(trial: Any, automl_config: dict[str, Any]) -> dict[str, Any]:
    """Sample a compact AutoML configuration from an Optuna trial."""
    candidate_models = automl_config.get(
        "candidate_models",
        ["ridge", "random_forest", "extratrees"],
    )
    feature_sets = automl_config.get("feature_sets", ["categorical_conditions"])
    augmentation_types = automl_config.get(
        "augmentation_types",
        ["none", "randomized_smiles", "order_permutation"],
    )
    augmentation_ratios = automl_config.get("augmentation_ratios", [1, 2, 5])

    model_type = trial.suggest_categorical("model_type", candidate_models)
    feature_set = trial.suggest_categorical("feature_set", feature_sets)
    augmentation_type = trial.suggest_categorical("augmentation_type", augmentation_types)
    augmentation_ratio = int(trial.suggest_categorical("augmentation_ratio", augmentation_ratios))

    model_params: dict[str, Any] = {}
    if model_type == "ridge":
        model_params["alpha"] = _suggest_float(trial, "ridge_alpha", 0.01, 10.0, log=True)
    elif model_type == "random_forest":
        model_params["n_estimators"] = _suggest_int(trial, "rf_n_estimators", 20, 80)
        model_params["max_depth"] = trial.suggest_categorical("rf_max_depth", [None, 3, 5])
    elif model_type in {"extratrees", "extra_trees"}:
        model_params["n_estimators"] = _suggest_int(trial, "et_n_estimators", 20, 80)
        model_params["max_depth"] = trial.suggest_categorical("et_max_depth", [None, 3, 5])

    return {
        "model_type": model_type,
        "feature_set": feature_set,
        "augmentation_type": augmentation_type,
        "augmentation_ratio": augmentation_ratio,
        "model_params": model_params,
    }


def evaluate_trial_config(
    trial_config: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    base_feature_config: dict[str, Any],
    automl_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Train/evaluate one sampled configuration on validation data."""
    train_df = apply_trial_augmentation(
        splits["train"],
        trial_config,
        automl_config,
        base_feature_config,
        seed,
    )
    valid_df = splits["valid"].copy()
    combined = pd.concat(
        [train_df.assign(__split="train"), valid_df.assign(__split="valid")],
        ignore_index=True,
    )
    feature_config = resolve_trial_feature_config(
        trial_config["feature_set"],
        base_feature_config,
        combined,
    )
    X, y, _ = build_feature_matrix(combined, feature_config)
    train_mask = combined["__split"].to_numpy() == "train"
    valid_mask = combined["__split"].to_numpy() == "valid"

    model_name = MODEL_NAME_MAP.get(trial_config["model_type"], trial_config["model_type"])
    model = get_model(model_name, seed=seed, **trial_config.get("model_params", {}))
    fitted = train_model(model, X[train_mask], y[train_mask])
    predictions = predict_model(fitted, X[valid_mask])
    y_valid = y[valid_mask]

    hit_rate = top_k_hit_rate(
        y_valid,
        predictions,
        k=int(automl_config.get("top_k", 3)),
        threshold=float(automl_config.get("high_yield_threshold", 70.0)),
    )
    validation_rmse = rmse(y_valid, predictions)
    metric = automl_config.get("primary_metric", "validation_top_k_hit_rate")
    score = hit_rate if metric == "validation_top_k_hit_rate" else -validation_rmse

    return {
        "score": float(score),
        "validation_top_k_hit_rate": float(hit_rate),
        "validation_rmse": float(validation_rmse),
        "train_rows": int(train_mask.sum()),
        "augmented_train_rows": int(train_df.get("is_augmented", pd.Series(False, index=train_df.index)).sum()),
    }


def apply_trial_augmentation(
    train_df: pd.DataFrame,
    trial_config: dict[str, Any],
    automl_config: dict[str, Any],
    base_feature_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Apply only the trial-selected train-set augmentation."""
    augmentation_type = trial_config["augmentation_type"]
    augmentation_ratio = int(trial_config.get("augmentation_ratio", 1))
    if augmentation_type == "none":
        result = train_df.copy()
        result["is_augmented"] = False
        return result

    if augmentation_type == "randomized_smiles":
        return augment_randomized_smiles(
            train_df,
            smiles_columns=automl_config.get(
                "smiles_columns",
                base_feature_config.get("smiles_columns", []),
            ),
            n_augments=augmentation_ratio,
            seed=seed,
            include_original=True,
        )

    if augmentation_type == "order_permutation":
        return permute_reaction_components(
            train_df,
            component_columns=automl_config.get(
                "component_columns",
                ["aryl_halide_smiles", "amine_smiles"],
            ),
            n_permutations=augmentation_ratio,
            seed=seed,
            include_original=True,
        )

    raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def resolve_trial_feature_config(
    feature_set: str,
    base_feature_config: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Resolve a compact feature-set name to an implemented feature config."""
    if feature_set == "categorical_conditions":
        return {
            "smiles_columns": [],
            "categorical_columns": base_feature_config.get("categorical_columns", ["solvent"]),
        }
    if feature_set == "fp_plus_conditions":
        return _resolve_feature_config({**base_feature_config, "kind": "fp_plus_conditions"}, df)
    raise ValueError(f"Unknown feature set: {feature_set}")


def _suggest_float(trial: Any, name: str, low: float, high: float, log: bool = False) -> float:
    if hasattr(trial, "suggest_float"):
        return float(trial.suggest_float(name, low, high, log=log))
    return float(trial.suggest_uniform(name, low, high))


def _suggest_int(trial: Any, name: str, low: int, high: int) -> int:
    if hasattr(trial, "suggest_int"):
        return int(trial.suggest_int(name, low, high))
    return int(trial.suggest_categorical(name, [low, high]))
