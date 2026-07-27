"""Corrected real-only leave-one-group-out evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.logo_protocol import (
    LOGO_SCHEMA_VERSION,
    LOGO_SPLIT_METHOD,
    CanonicalLOGOContract,
    build_canonical_logo_contract,
)
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
)

_REQUIRED_TARGETS = ("product_key", "reactant_key")
_METRICS = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "spearman": spearman_corr,
}


def run_corrected_logo(
    config: str | Path | Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run fixed real-only models over validated canonical LOGO folds.

    The saved canonical random-split artifacts are consumed solely as an
    immutable dataset/canonicalization dependency. LOGO membership itself is
    rebuilt from canonical product and substrate identities.
    """
    resolved, config_path = _load_config(config)
    seed = _integer(resolved.get("seed", 0), name="seed")
    dataset_path, split_directory = _resolve_data_paths(resolved)
    targets = _resolve_targets(resolved)
    feature_config = resolve_corrected_feature_config(
        _mapping(resolved.get("features"), name="features"),
        required_kind="bh_role_separated",
    )
    models = _resolve_models(resolved)
    metrics = _resolve_metrics(resolved)
    raw_output = (
        output_directory
        if output_directory is not None
        else _mapping(resolved.get("output"), name="output").get("directory")
    )
    if not isinstance(raw_output, (str, Path)) or not str(raw_output).strip():
        raise ValueError("output.directory must be a non-empty path.")
    output_path = Path(raw_output)
    assert_result_directory_allowed(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite LOGO output directory: {output_path}")

    saved = load_saved_canonical_splits(dataset_path, split_directory)
    contracts = {
        target: build_canonical_logo_contract(saved.canonical, target=target)
        for target in targets
    }
    for contract in contracts.values():
        _assert_logo_contract(contract)

    if "source_row_id" not in saved.canonical:
        raise ValueError("Canonical dataset is missing source_row_id.")
    source_ids = saved.canonical["source_row_id"].astype(str).tolist()
    X, y, feature_names, metadata = build_feature_matrix_with_metadata(
        saved.canonical,
        feature_config,
    )
    _validate_feature_data(X, y, source_ids)
    feature_contract = feature_contract_record(metadata, feature_names)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Canonical dataset contains duplicate source_row_id values.")
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}

    output_path.mkdir(parents=True, exist_ok=False)
    result_paths: dict[str, Path] = {"output_directory": output_path}
    for target in targets:
        target_directory = output_path / target
        target_directory.mkdir()
        paths = _evaluate_target(
            target_directory=target_directory,
            contract=contracts[target],
            X=X,
            y=y,
            source_index=source_index,
            seed=seed,
            models=models,
            metrics=metrics,
            saved=saved,
            dataset_path=dataset_path,
            split_directory=split_directory,
            feature_config=feature_config,
            feature_contract=feature_contract,
            resolved_config=resolved,
            config_path=config_path,
        )
        result_paths[f"{target}_directory"] = target_directory
        for name, path in paths.items():
            result_paths[f"{target}_{name}"] = path
    target_outputs = {}
    for target in targets:
        target_outputs[target] = {
            "fold_count": contracts[target].fold_count,
            "aggregate_assignment_hash": contracts[target].aggregate_assignment_hash,
            "manifest_hash": sha256_file(result_paths[f"{target}_manifest"]),
            "artifact_hashes": {
                Path(path).name: sha256_file(path)
                for key, path in result_paths.items()
                if key.startswith(f"{target}_") and key != f"{target}_directory"
            },
        }
    completion = {
        "schema_version": LOGO_SCHEMA_VERSION,
        "status": "complete",
        "expected_targets": list(_REQUIRED_TARGETS),
        "observed_targets": list(targets),
        "expected_fold_counts": {
            target: contracts[target].expected_group_count for target in targets
        },
        "observed_fold_counts": {
            target: contracts[target].fold_count for target in targets
        },
        "target_outputs": target_outputs,
    }
    completion["completion_payload_hash"] = stable_hash(completion)
    completion_path = output_path / "completion_manifest.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    result_paths["completion_manifest"] = completion_path
    return result_paths


def _evaluate_target(
    *,
    target_directory: Path,
    contract: CanonicalLOGOContract,
    X: np.ndarray,
    y: np.ndarray,
    source_index: Mapping[str, int],
    seed: int,
    models: list[dict[str, Any]],
    metrics: tuple[str, ...],
    saved: Any,
    dataset_path: Path,
    split_directory: Path,
    feature_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    config_path: Path | None,
) -> dict[str, Path]:
    assignments = contract.assignments
    group_sizes = pd.DataFrame(contract.group_size_records)
    group_sizes.insert(0, "group_column", contract.group_column)
    group_sizes.insert(0, "logo_target", contract.target)
    group_sizes.insert(0, "split_method", LOGO_SPLIT_METHOD)
    overlap = contract.overlap_audit
    overlap.insert(0, "group_column", contract.group_column)
    overlap.insert(0, "logo_target", contract.target)
    overlap.insert(0, "split_method", LOGO_SPLIT_METHOD)

    metric_rows: list[dict[str, Any]] = []
    for fold in contract.folds:
        train_indices = np.asarray(
            [source_index[source_id] for source_id in fold.train_source_ids],
            dtype=int,
        )
        test_indices = np.asarray(
            [source_index[source_id] for source_id in fold.test_source_ids],
            dtype=int,
        )
        for model_index, model_config in enumerate(models):
            model_name = model_config["name"]
            model_params = model_config["params"]
            model_seed = seed + fold.fold_index * len(models) + model_index
            estimator = get_model(model_name, seed=model_seed, **model_params)
            estimator = train_model(estimator, X[train_indices], y[train_indices])
            test_prediction = predict_model(estimator, X[test_indices])
            if not np.isfinite(test_prediction).all():
                raise ValueError(
                    "Model produced non-finite held-out predictions for "
                    f"target={contract.target!r}, fold={fold.fold_group!r}, "
                    f"model={model_name!r}."
                )
            model_hash = stable_hash(model_config)
            for metric_name in metrics:
                value = float(_METRICS[metric_name](y[test_indices], test_prediction))
                if not np.isfinite(value):
                    raise ValueError(
                        "Non-finite requested LOGO metric for "
                        f"target={contract.target!r}, fold={fold.fold_group!r}, "
                        f"model={model_name!r}, metric={metric_name!r}."
                    )
                metric_rows.append(
                    {
                        "logo_target": contract.target,
                        "split_method": LOGO_SPLIT_METHOD,
                        "group_column": contract.group_column,
                        "fold_index": fold.fold_index,
                        "fold_group": fold.fold_group,
                        "fold_assignment_hash": fold.assignment_hash,
                        "n_train": fold.train_size,
                        "n_test": fold.test_size,
                        "model": model_name,
                        "model_config_hash": model_hash,
                        "model_seed": model_seed,
                        "metric": metric_name,
                        "value": value,
                        "test_evaluation_count": 1,
                        "test_used_for_policy_selection": False,
                    }
                )
    per_fold = pd.DataFrame(metric_rows)
    expected_metric_rows = contract.fold_count * len(models) * len(metrics)
    if len(per_fold) != expected_metric_rows:
        raise AssertionError(
            "Every fold/model/metric must be evaluated exactly once: "
            f"expected={expected_metric_rows}, observed={len(per_fold)}."
        )
    summary = (
        per_fold.groupby(
            [
                "logo_target",
                "split_method",
                "group_column",
                "model",
                "model_config_hash",
                "metric",
            ],
            sort=True,
            dropna=False,
        )["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    expected_summary_rows = len(models) * len(metrics)
    if len(summary) != expected_summary_rows:
        raise AssertionError(
            "LOGO summary must contain exactly one row per method/metric: "
            f"expected={expected_summary_rows}, observed={len(summary)}."
        )
    invalid_counts = summary.loc[summary["count"].ne(contract.fold_count)]
    if not invalid_counts.empty:
        details = invalid_counts[
            ["model", "model_config_hash", "metric", "count"]
        ].to_dict(orient="records")
        raise AssertionError(
            "Every LOGO method/metric summary must contain every held-out fold: "
            f"expected_count={contract.fold_count}, invalid={details}."
        )

    paths = {
        "fold_assignments": target_directory / "fold_assignments.csv",
        "group_sizes": target_directory / "group_sizes.csv",
        "overlap_audit": target_directory / "overlap_audit.csv",
        "per_fold_metrics": target_directory / "per_fold_metrics.csv",
        "summary_metrics": target_directory / "summary_metrics.csv",
    }
    assignments.to_csv(paths["fold_assignments"], index=False)
    group_sizes.to_csv(paths["group_sizes"], index=False)
    overlap.to_csv(paths["overlap_audit"], index=False)
    per_fold.to_csv(paths["per_fold_metrics"], index=False)
    summary.to_csv(paths["summary_metrics"], index=False)
    output_hashes = {path.name: sha256_file(path) for path in paths.values()}

    manifest = {
        "schema_version": LOGO_SCHEMA_VERSION,
        "runner": "corrected_real_only_logo",
        "status": "corrected_revalidation",
        "output_directory": str(target_directory.parent),
        "target_output_directory": str(target_directory),
        "split_method": LOGO_SPLIT_METHOD,
        "logo_target": contract.target,
        "group_column": contract.group_column,
        "fold_group_present": all(bool(fold.fold_group) for fold in contract.folds),
        "expected_group_count": contract.expected_group_count,
        "observed_fold_count": contract.fold_count,
        "heldout_groups": [fold.fold_group for fold in contract.folds],
        "group_sizes": contract.group_size_records,
        "all_train_test_overlaps_zero": bool(overlap["all_overlaps_zero"].all()),
        "aggregate_assignment_hash": contract.aggregate_assignment_hash,
        "per_fold_assignment_hashes": contract.per_fold_assignment_hashes,
        "dataset": {
            "path": str(dataset_path),
            "hash": saved.dataset_hash,
            "n_rows": len(saved.canonical),
        },
        "canonical_split_dependency": {
            "directory": str(split_directory),
            "aggregate_split_hash": saved.aggregate_split_hash,
            "per_seed_split_hashes": {
                str(key): value for key, value in saved.per_seed_split_hashes.items()
            },
            "artifact_hashes": saved.artifact_hashes,
            "canonicalization_version": saved.manifest["canonicalization_version"],
            "split_schema_version": saved.manifest["split_schema_version"],
        },
        "feature_config": dict(feature_config),
        "feature_contract": dict(feature_contract),
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "seed": seed,
        "models": models,
        "metrics": list(metrics),
        "model_selection": "none_fixed_configured_models",
        "test_used_for_policy_selection": False,
        "test_evaluation_count_per_fold_method": 1,
        "resolved_config_hash": stable_hash(resolved_config),
        "config_path": str(config_path) if config_path is not None else None,
        "output_hashes": output_hashes,
    }
    manifest["manifest_payload_hash"] = stable_hash(manifest)
    manifest_path = target_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["manifest"] = manifest_path
    return paths


def _assert_logo_contract(contract: CanonicalLOGOContract) -> None:
    if contract.split_method != LOGO_SPLIT_METHOD:
        raise ValueError(
            f"Corrected LOGO requires split_method={LOGO_SPLIT_METHOD!r}; "
            f"got {contract.split_method!r}."
        )
    if contract.expected_group_count != contract.fold_count:
        raise ValueError("Expected LOGO group count does not equal observed fold count.")
    if not contract.folds or any(not fold.fold_group for fold in contract.folds):
        raise ValueError("Every LOGO fold must identify a non-empty held-out group.")
    if len({fold.fold_group for fold in contract.folds}) != contract.fold_count:
        raise ValueError("Every LOGO held-out group must appear exactly once.")
    if not bool(contract.overlap_audit["all_overlaps_zero"].all()):
        raise ValueError("LOGO train/test overlap audit failed.")


def _validate_feature_data(
    X: np.ndarray,
    y: np.ndarray,
    source_ids: list[str],
) -> None:
    if X.ndim != 2:
        raise ValueError(f"LOGO feature matrix must be two-dimensional; got shape={X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"LOGO label vector must be one-dimensional; got shape={y.shape}.")
    n_sources = len(source_ids)
    if len(X) != n_sources or len(y) != n_sources:
        raise ValueError(
            "Canonical sources, features, and labels must have identical lengths: "
            f"sources={n_sources}, features={len(X)}, labels={len(y)}."
        )
    if not np.isfinite(X).all():
        raise ValueError("LOGO feature matrix contains non-finite values.")
    if not np.isfinite(y).all():
        raise ValueError("LOGO label vector contains non-finite values.")


def _load_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, Mapping):
        return dict(config), None
    path = Path(config)
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Corrected LOGO config must contain a YAML mapping.")
    return value, path


def _resolve_data_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    dataset = _mapping(config.get("dataset"), name="dataset")
    raw_dataset_path = dataset.get("path")
    if not isinstance(raw_dataset_path, (str, Path)) or not str(raw_dataset_path).strip():
        raise ValueError("dataset.path must be a non-empty path.")
    dataset_path = Path(raw_dataset_path)
    splits = _mapping(config.get("splits"), name="splits")
    if splits.get("method") != LOGO_SPLIT_METHOD:
        raise ValueError(
            f"Corrected LOGO requires splits.method={LOGO_SPLIT_METHOD!r}; "
            f"got {splits.get('method')!r}."
        )
    dependency = splits.get("canonical_dependency_directory")
    if not isinstance(dependency, str) or not dependency.strip():
        raise ValueError("splits.canonical_dependency_directory must be a non-empty path.")
    return dataset_path, Path(dependency)


def _resolve_targets(config: Mapping[str, Any]) -> tuple[str, ...]:
    splits = _mapping(config.get("splits"), name="splits")
    raw = splits.get("targets")
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError("splits.targets must be a list of LOGO target names.")
    if len(raw) != len(set(raw)):
        raise ValueError("splits.targets must not contain duplicates.")
    if set(raw) != set(_REQUIRED_TARGETS):
        raise ValueError(
            "Corrected LOGO must run both logical targets: "
            f"{list(_REQUIRED_TARGETS)}."
        )
    return _REQUIRED_TARGETS


def _resolve_models(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("models")
    if not isinstance(raw, list) or not raw:
        raise ValueError("models must be a non-empty list.")
    models: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        model = _mapping(value, name=f"models[{index}]")
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"models[{index}].name must be non-empty.")
        params = _mapping(model.get("params", {}), name=f"models[{index}].params")
        models.append({"name": name, "params": dict(params)})
    if len({stable_hash(model) for model in models}) != len(models):
        raise ValueError("models contains duplicate fixed configurations.")
    return models


def _resolve_metrics(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("metrics", ["rmse", "mae"])
    if not isinstance(raw, list) or not raw or any(not isinstance(value, str) for value in raw):
        raise ValueError("metrics must be a non-empty list of metric names.")
    unknown = sorted(set(raw) - set(_METRICS))
    if unknown:
        raise ValueError(f"Unsupported corrected LOGO metrics: {unknown}.")
    if len(raw) != len(set(raw)):
        raise ValueError("metrics must not contain duplicates.")
    return tuple(raw)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return value
