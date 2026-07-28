"""One shared, tamper-evident manifest contract for scientific runs.

The manifest binds a result directory to the exact code, data, split, features,
configuration, plan, and outputs that produced it, plus the dependency and
hardware environment that determines floating-point reproducibility.

``verify_manifest`` recomputes the manifest hash and every recorded output hash.
It works both for manifests written by :func:`build_scientific_manifest` and for
the pre-existing runner manifests that already carry ``output_hashes`` and
``manifest_hash`` (for example the Phase 13 low-complexity benchmark), so a
single verification entry point covers every hash-addressed bundle in the
repository.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

SCIENTIFIC_MANIFEST_SCHEMA_VERSION = "bh-scientific-run-manifest-v1"
SCIENTIFIC_MANIFEST_FILENAME = "scientific_manifest.json"

#: Manifest file names understood by :func:`verify_manifest`, most specific first.
KNOWN_MANIFEST_FILENAMES = (
    SCIENTIFIC_MANIFEST_FILENAME,
    "manifest.json",
    "run_manifest.json",
)

#: Dependencies whose versions materially change scientific outputs.
TRACKED_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "rdkit",
    "xgboost",
    "PyYAML",
)

#: Environment variables that change BLAS/OpenMP threading and therefore the
#: exact floating-point reduction order of a run.
THREADING_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "KMP_AFFINITY",
    "KMP_DUPLICATE_LIB_OK",
    "MKL_THREADING_LAYER",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "created_at",
        "git_commit",
        "git_dirty_at_execution",
        "dataset_path",
        "dataset_hash",
        "split_hash",
        "feature_metadata_hash",
        "config_hash",
        "plan_hash",
        "row_counts",
        "output_directory",
        "output_hashes",
        "dependency_versions",
        "platform_record",
        "command",
        "resolved_scientific_config",
    }
)


class ManifestError(ValueError):
    """Raised when a manifest is malformed, incomplete, or fails verification."""


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    """Result of recomputing every hash recorded by one manifest."""

    directory: Path
    manifest_path: Path
    schema_version: str
    manifest_hash: str
    verified_output_count: int
    verified_outputs: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verification outcome."""
        return {
            "directory": str(self.directory),
            "manifest_path": str(self.manifest_path),
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "verified_output_count": self.verified_output_count,
            "verified_outputs": list(self.verified_outputs),
        }


def dependency_versions() -> dict[str, str]:
    """Report installed versions of every dependency that can change results."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in TRACKED_DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def platform_record() -> dict[str, Any]:
    """Report the hardware and threading environment affecting float results."""
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "cpu_count_logical": os.cpu_count(),
        "byte_order": sys.byteorder,
        "max_size": sys.maxsize,
        "float_repr_style": sys.float_repr_style,
        "environment": {
            name: os.environ[name]
            for name in THREADING_ENVIRONMENT_VARIABLES
            if name in os.environ
        },
        "torch": _torch_record(),
    }
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:  # pragma: no cover - platform dependent
        try:
            record["cpu_count_affinity"] = len(affinity(0))
        except OSError:
            record["cpu_count_affinity"] = None
    else:
        record["cpu_count_affinity"] = None
    return record


def compute_output_hashes(
    directory: str | Path,
    outputs: Iterable[str],
) -> dict[str, str]:
    """Hash every declared output relative to the result directory."""
    root = Path(directory)
    hashes: dict[str, str] = {}
    for name in sorted(set(outputs)):
        path = root / name
        if not path.is_file():
            raise ManifestError(f"Declared manifest output is missing: {path}")
        hashes[name] = sha256_file(path)
    if not hashes:
        raise ManifestError("A scientific manifest must declare at least one output.")
    return hashes


def build_scientific_manifest(
    *,
    status: str,
    output_directory: str | Path,
    outputs: Sequence[str],
    dataset_path: str | Path,
    dataset_hash: str,
    split_hash: str,
    feature_metadata_hash: str,
    config_hash: str,
    plan_hash: str,
    resolved_scientific_config: Mapping[str, Any],
    row_counts: Mapping[str, int],
    command: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared manifest, including its self-describing hash."""
    directory = Path(output_directory)
    manifest: dict[str, Any] = {
        "schema_version": SCIENTIFIC_MANIFEST_SCHEMA_VERSION,
        "status": str(status),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": git_commit(),
        "git_dirty_at_execution": git_dirty(),
        "dataset_path": str(dataset_path),
        "dataset_hash": str(dataset_hash),
        "split_hash": str(split_hash),
        "feature_metadata_hash": str(feature_metadata_hash),
        "config_hash": str(config_hash),
        "plan_hash": str(plan_hash),
        "row_counts": {str(key): int(value) for key, value in dict(row_counts).items()},
        "output_directory": str(directory),
        "output_hashes": compute_output_hashes(directory, outputs),
        "dependency_versions": dependency_versions(),
        "platform_record": platform_record(),
        "command": str(command),
        "resolved_scientific_config": _jsonable(resolved_scientific_config),
    }
    if extra:
        overlap = sorted(set(extra) & set(manifest))
        if overlap:
            raise ManifestError(f"Manifest extras may not redefine core fields: {overlap}.")
        manifest.update({str(key): _jsonable(value) for key, value in extra.items()})
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:  # pragma: no cover - defensive
        raise ManifestError(f"Scientific manifest is missing fields: {missing}.")
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def write_scientific_manifest(
    directory: str | Path,
    manifest: Mapping[str, Any],
    *,
    filename: str = SCIENTIFIC_MANIFEST_FILENAME,
) -> Path:
    """Write a manifest without overwriting an existing one."""
    path = Path(directory) / filename
    if path.exists():
        raise ManifestError(f"Refusing to overwrite an existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(directory: str | Path, *, filename: str | None = None) -> dict[str, Any]:
    """Read the manifest document from a result directory."""
    return _load_manifest(Path(directory), filename)[1]


def verify_manifest(
    directory: str | Path,
    *,
    filename: str | None = None,
) -> ManifestVerification:
    """Recompute the manifest hash and every recorded output hash.

    Raises :class:`ManifestError` when the manifest hash does not match its own
    contents or when any declared output no longer hashes to its recorded value.
    """
    root = Path(directory)
    manifest_path, manifest = _load_manifest(root, filename)
    claimed = manifest.pop("manifest_hash", None)
    if not isinstance(claimed, str) or not claimed:
        raise ManifestError(f"Manifest does not record a manifest_hash: {manifest_path}")
    recomputed = stable_hash(manifest)
    if recomputed != claimed:
        raise ManifestError(
            f"Manifest hash mismatch for {manifest_path}: "
            f"recorded={claimed}, recomputed={recomputed}."
        )
    output_hashes = manifest.get("output_hashes")
    if not isinstance(output_hashes, Mapping) or not output_hashes:
        raise ManifestError(f"Manifest does not declare output_hashes: {manifest_path}")
    verified: list[str] = []
    for name in sorted(output_hashes):
        path = root / str(name)
        if not path.is_file():
            raise ManifestError(f"Manifest output is missing: {path}")
        actual = sha256_file(path)
        if actual != output_hashes[name]:
            raise ManifestError(
                f"Manifest output hash mismatch for {path}: "
                f"recorded={output_hashes[name]}, recomputed={actual}."
            )
        verified.append(str(name))
    return ManifestVerification(
        directory=root,
        manifest_path=manifest_path,
        schema_version=str(manifest.get("schema_version", "unknown")),
        manifest_hash=claimed,
        verified_output_count=len(verified),
        verified_outputs=tuple(verified),
    )


def git_commit() -> str:
    """Return the current commit; scientific runs require a resolvable commit."""
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError("Scientific runs require an available Git commit.") from exc
    if not value:
        raise ManifestError("Scientific runs require a nonempty Git commit.")
    return value


def git_dirty() -> bool:
    """Return whether the worktree had uncommitted changes at execution time."""
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError("Unable to determine Git worktree state.") from exc


def _load_manifest(root: Path, filename: str | None) -> tuple[Path, dict[str, Any]]:
    if not root.is_dir():
        raise ManifestError(f"Result directory does not exist: {root}")
    candidates = (filename,) if filename else KNOWN_MANIFEST_FILENAMES
    for name in candidates:
        path = root / str(name)
        if path.is_file():
            try:
                document = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ManifestError(f"Cannot read manifest: {path}") from exc
            if not isinstance(document, dict):
                raise ManifestError(f"Manifest must be a JSON object: {path}")
            return path, document
    raise ManifestError(
        f"No manifest found in {root}; expected one of {list(candidates)}."
    )


def _torch_record() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"version": "not_installed"}
    record: dict[str, Any] = {"version": str(torch.__version__)}
    for name, getter in (
        ("num_threads", getattr(torch, "get_num_threads", None)),
        ("num_interop_threads", getattr(torch, "get_num_interop_threads", None)),
    ):
        try:
            record[name] = int(getter()) if getter is not None else None
        except (RuntimeError, ValueError):  # pragma: no cover - defensive
            record[name] = None
    try:
        record["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    except (AttributeError, RuntimeError):  # pragma: no cover - version dependent
        record["deterministic_algorithms"] = None
    try:
        record["cuda_available"] = bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):  # pragma: no cover - environment dependent
        record["cuda_available"] = None
    return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError, AttributeError):
            pass
    return value
