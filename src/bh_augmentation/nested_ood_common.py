"""Shared strict configuration and modeling helpers for nested OOD commands."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.train import train_model
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    stable_hash,
)

METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman_corr}
TARGET_COLUMNS = {
    "product_key": "canonical_product_key",
    "reactant_key": "canonical_substrate_key",
}


def scientific_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return output-independent configuration bound to frozen selections."""
    keys = ("dataset", "splits", "features", "metrics", "nested_ood")
    return json.loads(
        json.dumps({key: config[key] for key in keys if key in config}, sort_keys=True)
    )


def resolve_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the common nested OOD scientific contract."""
    dataset = require_mapping(config.get("dataset"), "dataset")
    splits = require_mapping(config.get("splits"), "splits")
    nested = require_mapping(config.get("nested_ood"), "nested_ood")
    if splits.get("method") != "nested_group_ood":
        raise ValueError("Nested OOD requires splits.method='nested_group_ood'.")
    dataset_path = require_path(dataset.get("path"), "dataset.path")
    split_directory = require_path(
        splits.get("canonical_dependency_directory"),
        "splits.canonical_dependency_directory",
    )
    raw_targets = nested.get("targets")
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or any(value not in TARGET_COLUMNS for value in raw_targets)
        or len(raw_targets) != len(set(raw_targets))
    ):
        raise ValueError(
            f"nested_ood.targets must contain unique values from {sorted(TARGET_COLUMNS)}."
        )
    metrics = config.get("metrics", ["rmse", "mae"])
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(value not in METRICS for value in metrics)
        or len(metrics) != len(set(metrics))
    ):
        raise ValueError(f"Unsupported or duplicate nested OOD metrics: {metrics!r}.")
    selection_metric = nested.get("selection_metric", "rmse")
    if selection_metric not in metrics:
        raise ValueError("nested_ood.selection_metric must be included in metrics.")
    lower_is_better = nested.get("lower_is_better", True)
    if not isinstance(lower_is_better, bool):
        raise ValueError("nested_ood.lower_is_better must be boolean.")
    selection_weighting = nested.get("selection_weighting", "group_weighted")
    if selection_weighting not in {"group_weighted", "sample_weighted"}:
        raise ValueError("Unsupported nested_ood.selection_weighting.")
    raw_policies = nested.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("nested_ood.policies must be a non-empty list.")
    policies: list[dict[str, Any]] = []
    for index, value in enumerate(raw_policies):
        item = require_mapping(value, f"nested_ood.policies[{index}]")
        if set(item) != {"policy_id", "method", "model", "params"}:
            raise ValueError("Nested OOD policy schema mismatch.")
        if item["method"] != "real_only":
            raise ValueError(
                "The Phase 8 production runner currently supports explicit "
                "real-only policies only."
            )
        if not isinstance(item["policy_id"], str) or not item["policy_id"]:
            raise ValueError("Nested OOD policy_id must be non-empty.")
        if not isinstance(item["model"], str) or not item["model"]:
            raise ValueError("Nested OOD model must be non-empty.")
        params = require_mapping(item["params"], f"nested_ood.policies[{index}].params")
        resolved = {
            "policy_id": item["policy_id"],
            "method": "real_only",
            "model": item["model"],
            "params": dict(params),
        }
        resolved["policy_hash"] = stable_hash(resolved)
        policies.append(resolved)
    if len({item["policy_id"] for item in policies}) != len(policies):
        raise ValueError("Nested OOD policy_id values must be unique.")
    feature_config = resolve_corrected_feature_config(
        require_mapping(config.get("features"), "features"),
        required_kind="bh_role_separated",
    )
    seed = config.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    return {
        "dataset_path": dataset_path,
        "split_directory": split_directory,
        "targets": tuple(raw_targets),
        "metrics": tuple(metrics),
        "selection_metric": selection_metric,
        "lower_is_better": lower_is_better,
        "selection_weighting": selection_weighting,
        "policies": policies,
        "feature_config": feature_config,
        "seed": seed,
    }


def build_partition(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one stateless canonical role-fingerprint partition."""
    X, y, names, metadata = build_feature_matrix_with_metadata(frame, feature_config)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(frame) or len(y) != len(frame):
        raise ValueError("Nested OOD feature/label shapes do not match partition rows.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nested OOD partition contains non-finite features or labels.")
    return X, y, feature_contract_record(metadata, names)


def fit_policy(
    policy: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> Any:
    """Fit one explicit real-only candidate policy."""
    estimator = get_model(str(policy["model"]), seed=seed, **dict(policy["params"]))
    return train_model(estimator, X, y)


def metric_value(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    context: str,
) -> float:
    """Evaluate one requested metric and reject undefined results."""
    value = float(METRICS[name](y_true, y_pred))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite nested OOD metric {name!r} for {context}.")
    return value


def assert_fresh_output(path: str | Path) -> Path:
    """Validate an allowed output root without creating or mutating it."""
    output = Path(path)
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite nested OOD output: {output}")
    return output


def create_fresh_output(path: str | Path) -> Path:
    """Create an allowed fresh output root without overwrite."""
    output = assert_fresh_output(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def require_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path.")
    return Path(value)
