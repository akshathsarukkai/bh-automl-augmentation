"""Run four tiny corrected paths against one validated canonical split slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bh_augmentation.run_condition_transfer import run_condition_transfer
from bh_augmentation.run_condition_transfer_supervised_ae import (
    run_condition_transfer_supervised_ae,
)
from bh_augmentation.run_corrected_representation_baselines import (
    run_corrected_representation_baselines,
)
from bh_augmentation.run_role_aware_condition_transfer import (
    run_role_aware_condition_transfer,
)

DATASET_PATH = "data/processed/bh_canonical_roles_v1.csv"
SPLIT_DIRECTORY = "results/corrected_canonical_splits"
HASH_COLUMNS = (
    "canonical_dataset_hash",
    "split_aggregate_hash",
    "per_seed_split_hash",
    "source_id_split_hash",
    "valid_source_id_hash",
    "test_source_id_hash",
    "canonicalization_version",
    "split_schema_version",
)


def run_smoke(output_directory: str | Path) -> dict[str, Any]:
    """Execute and validate the smallest common corrected split integration."""
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 4 smoke output: {root}")
    root.mkdir(parents=True)

    common = _common_config()
    configs = {
        "real_only": {
            **common,
            "features": {
                **common["features"],
                "representations": [
                    "reaction_section_concat",
                    "reaction_section_concat_delta",
                    "bh_role_separated",
                    "bh_role_separated_delta",
                ],
            },
            "output": {"directory": str(root / "corrected_real_only")},
        },
        "anonymous": {
            **common,
            "condition_transfer": {
                **_anonymous_transfer_config(),
            },
            "output": {"directory": str(root / "corrected_anonymous")},
        },
        "role_aware": {
            **common,
            "role_aware_condition_transfer": {
                **_role_aware_transfer_config(),
            },
            "output": {"directory": str(root / "corrected_role_aware")},
        },
        "hybrid": {
            **common,
            "condition_transfer": {
                **_anonymous_transfer_config(),
                "selection_model": "ridge",
                "policy_pairs": [
                    {
                        "donor_strategy": "random",
                        "label_strategy": "teacher_ensemble",
                    }
                ],
            },
            "supervised_autoencoder": {
                "latent_dims": [4],
                "hidden_dims": [8],
                "dropout": 0.0,
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 8,
                "internal_valid_size": 0.2,
                "device": "cpu",
                "synthetic_example_weights": [0.5],
            },
            "evaluate_full_data_reference": False,
            "output": {"directory": str(root / "corrected_hybrid")},
        },
    }
    config_paths: dict[str, Path] = {}
    for name, config in configs.items():
        path = root / f"{name}_config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        config_paths[name] = path

    outputs = {
        "real_only": run_corrected_representation_baselines(config_paths["real_only"]),
        "anonymous": run_condition_transfer(config_paths["anonymous"]),
        "role_aware": run_role_aware_condition_transfer(config_paths["role_aware"]),
        "hybrid": run_condition_transfer_supervised_ae(config_paths["hybrid"]),
    }
    audit_paths = {
        "real_only": outputs["real_only"]["split_audit"],
        "anonymous": outputs["anonymous"]["split_audit"],
        "role_aware": outputs["role_aware"]["split_audit"],
        "hybrid": outputs["hybrid"]["split_audit_path"],
    }
    audit_records = {
        name: _single_audit_record(path) for name, path in audit_paths.items()
    }
    reference = audit_records["real_only"]
    for name, record in audit_records.items():
        for column in HASH_COLUMNS:
            if record[column] != reference[column]:
                raise AssertionError(
                    f"Corrected runner {name} disagrees on {column}: "
                    f"{record[column]!r} != {reference[column]!r}."
                )

    required_paths = {
        name: sorted(
            {
                str(path)
                for path in result.values()
                if isinstance(path, Path) and path.is_file()
            }
        )
        for name, result in outputs.items()
    }
    if any(not paths for paths in required_paths.values()):
        raise AssertionError("At least one corrected runner produced no files.")
    output_hashes = {
        path: _sha256_file(Path(path))
        for paths in required_paths.values()
        for path in paths
    }
    manifest = {
        "dataset_path": DATASET_PATH,
        "split_directory": SPLIT_DIRECTORY,
        "seed": 0,
        "train_fraction": 0.01,
        "runner_hash_contract": {
            column: reference[column] for column in HASH_COLUMNS
        },
        "runner_outputs": required_paths,
        "output_hashes": output_hashes,
        "anonymous_selection_exclusions": _csv_rows(
            outputs["anonymous"]["selection_exclusions"]
        ),
        "role_aware_selection_exclusions": _csv_rows(
            outputs["role_aware"]["selection_exclusions"]
        ),
        "hybrid_selected_policies": _csv_rows(
            outputs["hybrid"]["selected_hybrid_policies_path"]
        ),
        "status": "passed",
    }
    manifest_path = root / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _common_config() -> dict[str, Any]:
    return {
        "seed": 0,
        "seeds": [0],
        "corrected_revalidation": {"enabled": True},
        "dataset": {"path": DATASET_PATH},
        "splits": {
            "method": "canonical_saved",
            "directory": SPLIT_DIRECTORY,
        },
        "low_data": {"enabled": True, "train_fractions": [0.01]},
        "features": {
            "kind": "bh_role_separated",
            "n_bits": 8,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "models": ["ridge"],
        "metrics": ["rmse", "mae", "r2", "spearman"],
        "selection": {
            "split": "valid",
            "metric": "rmse",
            "lower_is_better": True,
        },
    }


def _anonymous_transfer_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "role_change_requirement": "all",
        "fallback_policy": "reject",
        "donor_strategies": ["random"],
        "label_strategies": ["teacher_ensemble"],
        "synthetic_multipliers": [0.5],
        "n_neighbors": [3],
        "min_similarities": [None],
        "max_teacher_stds": [None],
        "candidates_per_real": 2,
        "teacher_models": ["ridge"],
        "donor_similarity_n_bits": 32,
        "donor_similarity_radius": 2,
        "donor_similarity_backend": "rdkit",
    }


def _role_aware_transfer_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "role_change_requirement": "all",
        "fallback_policy": "reject",
        "exclude_invariant_roles": True,
        "allow_zero_synthetic_policy_selection": False,
        "role_transfer_modes": ["ligand_base"],
        "donor_strategies": ["random"],
        "label_strategies": ["teacher_ensemble"],
        "synthetic_multipliers": [0.5],
        "max_candidates_per_source": 2,
        "max_teacher_stds": [None],
        "min_similarities": [0.0],
        "max_resample_attempts": 4,
        "teacher_models": ["ridge"],
        "donor_similarity_n_bits": 32,
        "donor_similarity_radius": 2,
        "donor_similarity_backend": "rdkit",
    }


def _single_audit_record(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if frame.shape[0] != 1:
        raise AssertionError(f"Expected one split audit row in {path}, got {len(frame)}.")
    return frame.iloc[0].to_dict()


def _csv_rows(path: Path) -> int:
    return len(pd.read_csv(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory")
    args = parser.parse_args()
    manifest = run_smoke(args.output_directory)
    print(json.dumps(manifest["runner_hash_contract"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
