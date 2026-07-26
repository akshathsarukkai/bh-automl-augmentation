"""Shared provenance and integrity helpers for corrected revalidation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.data.bh_condition_reader import RECOVERED_COLUMNS, augment_bh_dataframe
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.reaction_roles import ensure_reaction_role_columns
from bh_augmentation.features.compatibility import FeatureMetadata
from bh_augmentation.features.featurize import DEPRECATED_FEATURE_ALIASES, normalize_feature_config
from bh_augmentation.results.status import assert_result_directory_allowed

CORRECTED_ROLE_FEATURE_KIND = "bh_role_separated"
CORRECTED_STATUS = "corrected_revalidation"
REQUIRED_METRICS = ("rmse", "mae", "r2", "spearman")


def load_corrected_bh_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    """Load and identically validate the BH rows used by corrected experiments."""
    frame = clean_buchwald_hartwig(load_reaction_csv(dataset_path))
    if not set(RECOVERED_COLUMNS).issubset(frame.columns) or "role_validation_status" not in frame:
        frame = augment_bh_dataframe(frame, strict=False)
    valid_statuses = {"valid", "repaired", "valid_position_fallback"}
    valid = frame["condition_parse_status"].eq("ok") & frame["role_validation_status"].isin(
        valid_statuses
    )
    result = frame.loc[valid].copy()
    if result.empty:
        raise ValueError("No rows remain after corrected condition-role validation.")
    return ensure_reaction_role_columns(result, parse_if_missing=False).reset_index(drop=True)


def resolve_corrected_feature_config(
    feature_config: Mapping[str, Any],
    *,
    required_kind: str | None = None,
) -> dict[str, Any]:
    """Resolve an explicit canonical feature config and reject legacy aliases."""
    raw_kind = str(feature_config.get("kind", "")).strip()
    if not raw_kind:
        raise ValueError("Corrected configs must explicitly define features.kind.")
    if raw_kind in DEPRECATED_FEATURE_ALIASES:
        raise ValueError(
            f"Corrected configs may not use legacy feature alias {raw_kind!r}; "
            f"use {DEPRECATED_FEATURE_ALIASES[raw_kind]!r}."
        )
    resolved = normalize_feature_config(dict(feature_config))
    if required_kind is not None and resolved["kind"] != required_kind:
        raise ValueError(
            f"Corrected condition transfer requires features.kind={required_kind!r}; "
            f"got {resolved['kind']!r}."
        )
    resolved["categorical_columns"] = []
    return resolved


def prepare_fresh_output_directory(path: str | Path) -> Path:
    """Create a corrected output directory, refusing to overwrite any content."""
    directory = Path(path)
    if "corrected" not in directory.name.lower() and "corrected" not in str(directory).lower():
        raise ValueError(
            f"Corrected revalidation outputs require a fresh path containing 'corrected': {directory}"
        )
    assert_result_directory_allowed(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty corrected result directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256_file(path: str | Path) -> str:
    """Hash a file exactly as stored on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable scientific metadata deterministically."""
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_contract_record(
    metadata: FeatureMetadata,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Return the complete stable representation contract and its hashes."""
    record = {
        "representation_kind": metadata.representation_kind,
        "n_bits": metadata.n_bits,
        "radius": metadata.radius,
        "fingerprint_backend": metadata.fingerprint_backend,
        "role_ordering": list(metadata.role_ordering),
        "block_slices": [
            {"name": block.name, "start": block.start, "stop": block.stop}
            for block in metadata.block_slices
        ],
        "total_width": metadata.total_width,
        "feature_name_hash": stable_hash(list(feature_names)),
    }
    record["feature_metadata_hash"] = stable_hash(record)
    return record


def split_audit_record(
    seed: int,
    train_fraction: float,
    splits: Mapping[str, pd.DataFrame],
    *,
    dataset_hash: str,
) -> dict[str, Any]:
    """Describe and hash one train/validation/test split without row-order ambiguity."""
    indices = {
        split: [int(index) for index in splits[split].index.tolist()]
        for split in ("train", "valid", "test")
    }
    split_hash = stable_hash(indices)
    return {
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "n_train": len(indices["train"]),
        "n_valid": len(indices["valid"]),
        "n_test": len(indices["test"]),
        "train_index_hash": stable_hash(indices["train"]),
        "valid_index_hash": stable_hash(indices["valid"]),
        "test_index_hash": stable_hash(indices["test"]),
        "split_hash": split_hash,
        "dataset_hash": dataset_hash,
    }


def assert_training_only_parents(
    candidate_df: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
) -> None:
    """Fail if any synthetic source or donor is outside the current training subset."""
    if candidate_df.empty:
        return
    train = set(splits["train"].index.tolist())
    forbidden = set(splits["valid"].index.tolist()) | set(splits["test"].index.tolist())
    for column in ("source_index", "donor_index"):
        if column not in candidate_df.columns:
            raise ValueError(f"Synthetic candidate audit is missing parent column {column!r}.")
        parents = set(candidate_df[column].dropna().tolist())
        if not parents.issubset(train) or parents & forbidden:
            raise ValueError(
                f"Synthetic parent leakage detected in {column}: "
                f"outside_train={sorted(parents - train)[:5]}."
            )


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    dataset_path: str | Path,
    dataset_hash: str,
    output_directory: str | Path,
    split_hashes: Mapping[str, str],
    feature_metadata_hash: str | Mapping[str, str],
) -> dict[str, Any]:
    """Build the mandatory provenance manifest for one corrected run."""
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": f"corrected-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp": timestamp,
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["git", "status", "--porcelain"])),
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "resolved_config": _jsonable(config),
        "config_path": str(config_path),
        "config_hash": stable_hash(config),
        "split_hashes": dict(sorted(split_hashes.items())),
        "feature_metadata_hash": _jsonable(feature_metadata_hash),
        "dependency_versions": _dependency_versions(),
        "command": " ".join(sys.argv),
        "seeds": [int(seed) for seed in config.get("seeds", [config.get("seed", 42)])],
        "train_fractions": [
            float(value)
            for value in config.get("low_data", {}).get("train_fractions", [1.0])
        ],
        "output_directory": str(output_directory),
        "historical_results_loaded": False,
        "result_status": CORRECTED_STATUS,
    }


def write_json(path: str | Path, value: Any) -> Path:
    """Write stable, human-readable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")
    return output


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "rdkit", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _git_value(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
