"""Run matched corrected real-only representation baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_supervised_ae_latent_baseline import (
    _compute_metric,
    _parse_model_config,
    _resolve_model_configs,
    _resolve_seeds,
    _split_positions,
    _validate_metrics,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    CORRECTED_STATUS,
    build_run_manifest,
    canonical_split_audit_record,
    feature_contract_record,
    load_corrected_bh_dataframe,
    prepare_fresh_output_directory,
    resolve_corrected_feature_config,
    sha256_file,
    write_json,
)
from bh_augmentation.utils.seed import set_global_seed

REQUIRED_REPRESENTATIONS = (
    "reaction_section_concat",
    "reaction_section_concat_delta",
    "bh_role_separated",
    "bh_role_separated_delta",
)


def run_corrected_representation_baselines(config_path: str | Path) -> dict[str, Path]:
    """Evaluate all canonical representations on exactly matched random splits."""
    config = load_config(config_path)
    metric_names = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metric_names)
    seeds = _resolve_seeds(config)
    dataset_path = Path(_dataset_path(config))
    train_fractions = _corrected_train_fractions(config)
    split_directory = _corrected_split_directory(config)
    saved_splits = load_saved_canonical_splits(
        dataset_path,
        split_directory,
        requested_seeds=seeds,
        requested_fractions=train_fractions,
    )
    dataset_hash = sha256_file(dataset_path)
    frame = load_corrected_bh_dataframe(dataset_path)
    output_dir = prepare_fresh_output_directory(
        config.get("output", {}).get(
            "directory", "results/corrected_representation_baselines_xgboost"
        )
    )

    feature_configs = _representation_configs(config)
    metric_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    metadata_json: dict[str, Any] = {}
    split_rows_by_key: dict[str, dict[str, Any]] = {}
    split_hashes: dict[str, str] = {}

    for feature_config in feature_configs:
        X, y, feature_names, feature_metadata = build_feature_matrix_with_metadata(
            frame, feature_config
        )
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        contract = feature_contract_record(feature_metadata, feature_names)
        metadata_json[feature_metadata.representation_kind] = contract
        metadata_rows.append(_flatten_metadata_record(contract))

        for seed in seeds:
            set_global_seed(seed)
            for train_fraction, splits in saved_splits.materialize_variants(
                frame,
                seed=seed,
                train_fractions=train_fractions,
            ):
                split_record = canonical_split_audit_record(
                    seed,
                    float(train_fraction),
                    splits,
                    dataset_hash=dataset_hash,
                    saved_audit=saved_splits.audit_record(
                        seed=seed,
                        train_fraction=float(train_fraction),
                    ),
                )
                split_key = _split_key(seed, float(train_fraction))
                previous = split_rows_by_key.setdefault(split_key, split_record)
                if previous["split_hash"] != split_record["split_hash"]:
                    raise ValueError(f"Split changed across representations for {split_key}.")
                split_hashes[split_key] = str(split_record["split_hash"])

                train_idx = _split_positions(splits["train"])
                valid_idx = _split_positions(splits["valid"])
                test_idx = _split_positions(splits["test"])
                for model_config in _resolve_model_configs(config.get("models", ["xgboost"])):
                    model_name, model_kwargs = _parse_model_config(model_config)
                    model = train_model(
                        get_model(model_name, seed=seed, **model_kwargs),
                        X[train_idx],
                        y[train_idx],
                    )
                    for split_name, indices in (("valid", valid_idx), ("test", test_idx)):
                        predictions = predict_model(model, X[indices])
                        if not np.isfinite(predictions).all():
                            raise ValueError(
                                f"Non-finite {split_name} predictions for "
                                f"{feature_metadata.representation_kind}, seed={seed}."
                            )
                        for metric_name in metric_names:
                            metric_rows.append(
                                {
                                    "seed": seed,
                                    "train_fraction": float(train_fraction),
                                    "representation": feature_metadata.representation_kind,
                                    "model": model_name,
                                    "split": split_name,
                                    "metric": metric_name,
                                    "value": _compute_metric(
                                        metric_name, y[indices], predictions
                                    ),
                                    "n_train": len(train_idx),
                                    "n_valid": len(valid_idx),
                                    "n_test": len(test_idx),
                                    "feature_width": X.shape[1],
                                    "feature_metadata_hash": contract[
                                        "feature_metadata_hash"
                                    ],
                                    "split_hash": split_record["split_hash"],
                                    "dataset_hash": dataset_hash,
                                    "result_status": CORRECTED_STATUS,
                                }
                            )

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(
            ["train_fraction", "representation", "model", "split", "metric"],
            dropna=False,
        )["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    split_audit = pd.DataFrame(split_rows_by_key.values()).sort_values(
        ["seed", "train_fraction"]
    )
    paths = _output_paths(output_dir)
    metrics.to_csv(paths["policy_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(metadata_rows).to_csv(paths["representation_metadata_csv"], index=False)
    write_json(paths["representation_metadata_json"], metadata_json)
    split_audit.to_csv(paths["split_audit"], index=False)
    manifest = build_run_manifest(
        config=config,
        config_path=config_path,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        output_directory=output_dir,
        split_hashes=split_hashes,
        feature_metadata_hash={
            kind: value["feature_metadata_hash"] for kind, value in metadata_json.items()
        },
        canonical_split_contract={
            **saved_splits.audit_metadata,
            "split_directory": str(split_directory),
        },
    )
    write_json(paths["run_manifest"], manifest)
    return paths


def _representation_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    features = dict(config.get("features", {}))
    requested = list(features.pop("representations", REQUIRED_REPRESENTATIONS))
    missing = [kind for kind in REQUIRED_REPRESENTATIONS if kind not in requested]
    if missing:
        raise ValueError(
            "Corrected representation baseline config is missing required kinds: "
            + ", ".join(missing)
        )
    return [
        resolve_corrected_feature_config({**features, "kind": str(kind)})
        for kind in requested
    ]


def _flatten_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "role_ordering": "|".join(record["role_ordering"]),
        "block_slices": "|".join(
            f"{block['name']}:{block['start']}:{block['stop']}"
            for block in record["block_slices"]
        ),
    }


def _output_paths(directory: Path) -> dict[str, Path]:
    return {
        "directory": directory,
        "policy_metrics": directory / "policy_metrics.csv",
        "summary": directory / "summary.csv",
        "representation_metadata_csv": directory / "representation_metadata.csv",
        "representation_metadata_json": directory / "representation_metadata.json",
        "split_audit": directory / "split_audit.csv",
        "run_manifest": directory / "run_manifest.json",
    }


def _dataset_path(config: dict[str, Any]) -> str:
    path = config.get("dataset", {}).get("path")
    if not path:
        raise ValueError("Config must define dataset.path.")
    return str(path)


def _corrected_split_directory(config: dict[str, Any]) -> Path:
    split_config = config.get("splits", {})
    if split_config.get("method") != "canonical_saved":
        raise ValueError("Corrected scientific runs require splits.method='canonical_saved'.")
    directory = split_config.get("directory")
    if not directory:
        raise ValueError("Corrected scientific runs require splits.directory.")
    return Path(directory)


def _corrected_train_fractions(config: dict[str, Any]) -> list[float]:
    low_data = config.get("low_data", {})
    if not low_data.get("enabled", False):
        return [1.0]
    fractions = low_data.get("train_fractions")
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("Corrected low_data.train_fractions must be a non-empty list.")
    return [float(value) for value in fractions]


def _split_key(seed: int, fraction: float) -> str:
    return f"seed={seed}|train_fraction={fraction:.12g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    paths = run_corrected_representation_baselines(args.config)
    print(f"Saved corrected representation metrics to {paths['policy_metrics']}")


if __name__ == "__main__":
    main()
