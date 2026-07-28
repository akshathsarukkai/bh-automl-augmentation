"""Bounded end-to-end smoke over a real scientific production path.

This exercises the genuine runners -- canonical data audit, canonical
group-safe splitting, and the Phase 13 low-complexity benchmark -- on a tiny
slice of the committed ``data/processed/bh_clean_stress.csv`` fixture, then
verifies the resulting bundle with the production validator and the shared
manifest verifier.

It uses only data that is committed to git, so it runs from a clean checkout
and in CI without the gitignored canonical dataset or canonical split
directory. Everything is written beneath a single fresh output directory named
by the caller; nothing is written into ``results/``.

Example
-------
    python -B scripts/run_production_path_smoke.py --output-directory /tmp/bh-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bh_augmentation.data.audit_canonical_dataset import (  # noqa: E402
    run_canonical_data_audit,
)
from bh_augmentation.data.canonical_splits import (  # noqa: E402
    run_canonical_grouped_splits,
)
from bh_augmentation.low_complexity_benchmark import (  # noqa: E402
    run_low_complexity_benchmark,
    validate_low_complexity_benchmark,
)
from bh_augmentation.representations.low_complexity import (  # noqa: E402
    LOW_COMPLEXITY_METHODS,
)
from bh_augmentation.utils.scientific_manifest import verify_manifest  # noqa: E402
from bh_augmentation.utils.strict_config import (  # noqa: E402
    require_fresh_output_directory,
)

SOURCE_DATASET = REPOSITORY_ROOT / "data" / "processed" / "bh_clean_stress.csv"
SMOKE_SCHEMA_VERSION = "bh-production-path-smoke-v1"


def checkout_relative_source() -> Path:
    """Return the source dataset as a path relative to the current directory.

    Absolute paths are recorded verbatim inside split manifests and therefore
    inside downstream config and plan hashes. Using a checkout-relative path
    keeps every artifact byte-identical across two different clean checkouts,
    which is what makes the reproduction claim checkable.
    """
    try:
        return Path(os.path.relpath(SOURCE_DATASET, Path.cwd()))
    except ValueError:  # pragma: no cover - different drives on Windows
        return SOURCE_DATASET


def _audit_config(source: Path, canonical: Path, output: Path, rows: int) -> dict[str, Any]:
    return {
        "scientific_run": True,
        "allow_hash_fingerprint_fallback": False,
        "invalid_smiles_policy": "audit_then_error_for_scientific_dataset",
        "dataset": {"path": str(source), "nrows": rows},
        "canonicalization": {
            "backend": "rdkit",
            "isomeric_smiles": True,
            "preserve_role_order": True,
        },
        "duplicate_audit": {
            "yield_conflict_reporting_threshold": 10.0,
            "exclusion_policy": "none",
        },
        "features": {
            "kind": "bh_role_separated",
            "n_bits": 32,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "output": {
            "canonical_dataset_path": str(canonical),
            "directory": str(output),
        },
    }


def _split_config(source: Path, canonical: Path, output: Path) -> dict[str, Any]:
    return {
        "scientific_run": True,
        "allow_hash_fingerprint_fallback": False,
        "invalid_smiles_policy": "audit_then_error_for_scientific_dataset",
        "canonicalization": {
            "backend": "rdkit",
            "isomeric_smiles": True,
            "preserve_role_order": True,
        },
        "dataset": {"source_path": str(source), "path": str(canonical)},
        "splits": {
            "group_column": "canonical_reaction_key",
            "seeds": [0],
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "train_fractions": [0.5, 1.0],
        },
        "output": {"directory": str(output)},
    }


def _benchmark_config(canonical: Path, splits: Path, output: Path) -> dict[str, Any]:
    return {
        "dataset": {"path": str(canonical)},
        "splits": {
            "canonical_directory": str(splits),
            "random_seeds": [0],
            "random_fractions": [1.0],
        },
        "features": {
            "kind": "bh_role_separated",
            "n_bits": 32,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "methods": {
            "families": list(LOW_COMPLEXITY_METHODS),
            "latent_widths": [8, 16],
            "ridge_alpha": 1.0,
            "linear_autoencoder": {"epochs": 2, "learning_rate": 0.01},
            "neural": {
                "max_epochs": 3,
                "patience": 2,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "batch_size": 128,
                "internal_validation_fraction": 0.2,
            },
        },
        "metrics": ["rmse", "mae", "r2", "spearman"],
        "selection_metric": "rmse",
        "base_seed": 0,
        "output": {"directory": str(output)},
    }


def run_smoke(output_directory: str | Path, *, rows: int = 400) -> dict[str, Any]:
    """Run the bounded production path and return its summary record."""
    if not SOURCE_DATASET.is_file():
        raise FileNotFoundError(
            f"The committed smoke source dataset is missing: {SOURCE_DATASET}"
        )
    root = require_fresh_output_directory(output_directory, name="smoke output directory")
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = checkout_relative_source()

    canonical_dataset = root / "corrected_canonical_roles_smoke.csv"
    audit_directory = root / "corrected_canonical_audit"
    split_directory = root / "corrected_canonical_splits"
    benchmark_directory = root / "corrected_low_complexity_smoke"

    run_canonical_data_audit(
        _audit_config(source, canonical_dataset, audit_directory, rows),
        config_path=root / "smoke_audit_config.yaml",
    )
    run_canonical_grouped_splits(
        _split_config(source, canonical_dataset, split_directory),
        config_path=root / "smoke_split_config.yaml",
    )
    run_low_complexity_benchmark(
        _benchmark_config(canonical_dataset, split_directory, benchmark_directory)
    )

    runner_manifest = validate_low_complexity_benchmark(benchmark_directory)
    shared = verify_manifest(benchmark_directory, filename="scientific_manifest.json")
    legacy = verify_manifest(benchmark_directory, filename="manifest.json")
    summary = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": "passed",
        "rows": rows,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "source_dataset": str(source),
        "output_directory": str(root),
        "benchmark_directory": str(benchmark_directory),
        "runner_manifest_hash": runner_manifest["manifest_hash"],
        "shared_manifest_verification": shared.to_dict(),
        "runner_manifest_verification": legacy.to_dict(),
        "method_family_count": runner_manifest["method_family_count"],
        "test_used_for_selection": runner_manifest["test_used_for_selection"],
    }
    (root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded production-path smoke."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args(argv)
    summary = run_smoke(args.output_directory, rows=args.rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
