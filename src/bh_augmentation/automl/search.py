"""Small Optuna search controller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.automl.objective import (
    apply_trial_augmentation,
    evaluate_trial_config,
    resolve_trial_feature_config,
    suggest_trial_config,
)
from bh_augmentation.evaluation.metrics import mae, r2, rmse
from bh_augmentation.evaluation.topk import top_k_hit_rate
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.reporting.make_report import save_metrics_csv


def run_automl_search(
    splits: dict[str, pd.DataFrame],
    base_feature_config: dict[str, Any],
    automl_config: dict[str, Any],
    output_config: dict[str, Any],
    seed: int = 42,
) -> dict[str, Path]:
    """Run a compact Optuna search and save trial, best-config, and test outputs."""
    optuna = _import_optuna()
    n_trials = int(automl_config.get("n_trials", automl_config.get("search_budget_trials", 10)))
    trial_records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        trial_config = suggest_trial_config(trial, automl_config)
        metrics = evaluate_trial_config(
            trial_config=trial_config,
            splits=splits,
            base_feature_config=base_feature_config,
            automl_config=automl_config,
            seed=seed,
        )
        trial_records.append(
            {
                "trial_number": getattr(trial, "number", len(trial_records)),
                **_flatten_trial_config(trial_config),
                **metrics,
            }
        )
        return float(metrics["score"])

    sampler = None
    if hasattr(optuna, "samplers") and hasattr(optuna.samplers, "TPESampler"):
        sampler = optuna.samplers.TPESampler(seed=seed)
    study_kwargs = {"direction": "maximize"}
    if sampler is not None:
        study_kwargs["sampler"] = sampler
    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, n_trials=n_trials)

    if not trial_records:
        raise ValueError("AutoML search produced no trials.")
    trials_df = pd.DataFrame(trial_records)
    best_row = trials_df.sort_values("score", ascending=False).iloc[0].to_dict()
    best_config = _best_config_from_row(best_row)
    final_metrics = evaluate_best_on_test(
        best_config,
        splits,
        base_feature_config,
        automl_config,
        seed,
    )

    paths = {
        "trials": _trial_table_path(output_config),
        "best_config": _best_config_path(output_config),
        "final_test_metrics": _final_test_metrics_path(output_config),
    }
    paths["trials"].parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(paths["trials"], index=False)
    paths["best_config"].write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    save_metrics_csv(final_metrics, paths["final_test_metrics"])
    return paths


def evaluate_best_on_test(
    best_config: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    base_feature_config: dict[str, Any],
    automl_config: dict[str, Any],
    seed: int,
) -> list[dict[str, object]]:
    """Train the best config on train data and evaluate untouched test data."""
    train_df = apply_trial_augmentation(
        splits["train"],
        best_config,
        automl_config,
        base_feature_config,
        seed,
    )
    test_df = splits["test"].copy()
    test_df["is_augmented"] = False
    combined = pd.concat(
        [train_df.assign(__split="train"), test_df.assign(__split="test")],
        ignore_index=True,
    )
    feature_config = resolve_trial_feature_config(
        best_config["feature_set"],
        base_feature_config,
        combined,
    )
    X, y, _ = build_feature_matrix(combined, feature_config)
    train_mask = combined["__split"].to_numpy() == "train"
    test_mask = combined["__split"].to_numpy() == "test"

    model = get_model(
        _model_name(best_config["model_type"]),
        seed=seed,
        **best_config.get("model_params", {}),
    )
    fitted = train_model(model, X[train_mask], y[train_mask])
    predictions = predict_model(fitted, X[test_mask])
    y_test = y[test_mask]
    return [
        {
            "split": "test",
            "metric": "rmse",
            "value": rmse(y_test, predictions),
            "eval_augmented_rows": int(combined.loc[test_mask, "is_augmented"].sum()),
        },
        {
            "split": "test",
            "metric": "mae",
            "value": mae(y_test, predictions),
            "eval_augmented_rows": int(combined.loc[test_mask, "is_augmented"].sum()),
        },
        {
            "split": "test",
            "metric": "r2",
            "value": r2(y_test, predictions),
            "eval_augmented_rows": int(combined.loc[test_mask, "is_augmented"].sum()),
        },
        {
            "split": "test",
            "metric": "top_k_hit_rate",
            "value": top_k_hit_rate(
                y_test,
                predictions,
                k=int(automl_config.get("top_k", 3)),
                threshold=float(automl_config.get("high_yield_threshold", 70.0)),
            ),
            "eval_augmented_rows": int(combined.loc[test_mask, "is_augmented"].sum()),
        },
    ]


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for AutoML search. Install it explicitly with "
            "`python -m pip install optuna`, then rerun `python -m "
            "bh_augmentation.run_automl --config configs/automl.yaml`."
        ) from exc
    return optuna


def _flatten_trial_config(trial_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": trial_config["model_type"],
        "feature_set": trial_config["feature_set"],
        "augmentation_type": trial_config["augmentation_type"],
        "augmentation_ratio": trial_config["augmentation_ratio"],
        "model_params": json.dumps(trial_config.get("model_params", {}), sort_keys=True),
    }


def _best_config_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": row["model_type"],
        "feature_set": row["feature_set"],
        "augmentation_type": row["augmentation_type"],
        "augmentation_ratio": int(row["augmentation_ratio"]),
        "model_params": json.loads(row.get("model_params", "{}")),
        "validation_score": float(row["score"]),
        "validation_top_k_hit_rate": float(row["validation_top_k_hit_rate"]),
        "validation_rmse": float(row["validation_rmse"]),
    }


def _model_name(model_type: str) -> str:
    return "extra_trees" if model_type == "extratrees" else model_type


def _trial_table_path(output_config: dict[str, Any]) -> Path:
    return Path(output_config.get("trial_table_path", "results/automl/trials.csv"))


def _best_config_path(output_config: dict[str, Any]) -> Path:
    return Path(output_config.get("best_config_path", "results/automl/best_config.json"))


def _final_test_metrics_path(output_config: dict[str, Any]) -> Path:
    return Path(output_config.get("final_test_metrics_path", "results/automl/final_test_metrics.csv"))
