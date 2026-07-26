"""Fixtures shared by corrected-revalidation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bh_augmentation.data.audit_canonical_dataset import run_canonical_data_audit
from bh_augmentation.data.canonical_splits import run_canonical_grouped_splits
from canonical_test_utils import audit_config, split_config


def corrected_dataset(n_rows: int = 72) -> pd.DataFrame:
    reactant_1 = ["Brc1ccccc1", "Clc1ccccc1", "Brc1ccncc1", "Ic1ccccc1"]
    reactant_2 = ["CN", "CCN", "Nc1ccccc1"]
    ligands = ["P(C)(C)C", "P(CC)(CC)CC", "P(c1ccccc1)(c1ccccc1)c1ccccc1"]
    bases = ["O=P(O)(O)O", "O=C(O)O", "CN(C)C(=NC(C)(C)C)N(C)C"]
    solvents = ["CCO", "CCCO", "O1CCOCC1"]
    catalysts = ["[Pd]", "[Pt]", "[Ni]"]
    products = ["c1ccccc1NC", "c1ccccc1NCC", "c1ccncc1NC"]
    rows: list[dict[str, Any]] = []
    for index in range(n_rows):
        r1 = reactant_1[index % len(reactant_1)]
        r2 = reactant_2[(index // 2) % len(reactant_2)]
        ligand = ligands[(index // 3) % len(ligands)]
        base = bases[(index // 5) % len(bases)]
        solvent = solvents[(index // 7) % len(solvents)]
        product = products[index % len(products)]
        catalyst = catalysts[(index // 11) % len(catalysts)]
        rows.append(
            {
                "reaction_id": f"corrected_{index}",
                "reaction_smiles": (
                    f"{r1}.{r2}.{catalyst}.{ligand}.{base}.{solvent}>>{product}"
                ),
                "yield": float((index * 13) % 101),
                "reactant_key": f"{r1}.{r2}",
                "product_key": product,
                "recovered_reactant_1_smiles": r1,
                "recovered_reactant_2_smiles": r2,
                "recovered_catalyst_smiles": catalyst,
                "recovered_ligand_smiles": ligand,
                "recovered_base_smiles": base,
                "recovered_solvent_or_additive_smiles": solvent,
                "recovered_product_smiles": product,
                "recovered_temperature": 25.0,
                "condition_parse_status": "ok",
                "role_validation_status": "valid",
            }
        )
    return pd.DataFrame(rows)


def write_corrected_config(
    tmp_path: Path,
    *,
    kind: str,
    output_name: str,
) -> Path:
    source_path = tmp_path / "corrected_source_dataset.csv"
    dataset_path = tmp_path / "corrected_canonical_dataset.csv"
    audit_directory = tmp_path / "corrected_fixture_audit"
    split_directory = tmp_path / "corrected_fixture_splits"
    if not source_path.exists():
        corrected_dataset().to_csv(source_path, index=False)
    if not dataset_path.exists():
        run_canonical_data_audit(
            audit_config(source_path, dataset_path, audit_directory),
            config_path=tmp_path / "fixture_audit.yaml",
        )
    if not split_directory.exists():
        grouped_config = split_config(
            source_path,
            dataset_path,
            split_directory,
            seeds=[0],
        )
        grouped_config["splits"].update(
            {
                "train_size": 0.6,
                "valid_size": 0.2,
                "test_size": 0.2,
                "train_fractions": [0.5, 1.0],
            }
        )
        run_canonical_grouped_splits(
            grouped_config,
            config_path=tmp_path / "fixture_splits.yaml",
        )
    output_dir = tmp_path / output_name
    base: dict[str, Any] = {
        "seed": 0,
        "seeds": [0],
        "corrected_revalidation": {"enabled": True},
        "dataset": {"path": str(dataset_path)},
        "splits": {
            "method": "canonical_saved",
            "directory": str(split_directory),
        },
        "low_data": {"enabled": True, "train_fractions": [0.5]},
        "features": {
            "kind": "bh_role_separated",
            "n_bits": 16,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "models": [
            {
                "name": "xgboost",
                "params": {
                    "n_estimators": 2,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "n_jobs": 1,
                    "tree_method": "hist",
                },
            }
        ],
        "metrics": ["rmse", "mae", "r2", "spearman"],
        "selection": {"split": "valid", "metric": "rmse", "lower_is_better": True},
        "output": {"directory": str(output_dir)},
    }
    if kind == "representations":
        base["features"] = {
            "representations": [
                "reaction_section_concat",
                "reaction_section_concat_delta",
                "bh_role_separated",
                "bh_role_separated_delta",
            ],
            "n_bits": 16,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        }
    elif kind == "anonymous":
        base["condition_transfer"] = {
            "enabled": True,
            "role_change_requirement": "all",
            "fallback_policy": "reject",
            "donor_strategies": ["random"],
            "label_strategies": ["teacher_ensemble"],
            "synthetic_multipliers": [0.5],
            "n_neighbors": [3],
            "min_similarities": [None],
            "max_teacher_stds": [None],
            "high_yield_threshold": 70.0,
            "clip_y_min": 0.0,
            "clip_y_max": 100.0,
            "candidates_per_real": 3,
            "teacher_models": ["ridge"],
            "donor_similarity_n_bits": 64,
            "donor_similarity_radius": 2,
            "donor_similarity_backend": "rdkit",
        }
    elif kind == "role_aware":
        base["role_aware_condition_transfer"] = {
            "enabled": True,
            "role_change_requirement": "all",
            "fallback_policy": "reject",
            "exclude_invariant_roles": True,
            "allow_zero_synthetic_policy_selection": False,
            "role_transfer_modes": ["ligand_base_solvent_or_additive"],
            "donor_strategies": ["matched_product_or_reactant_key"],
            "label_strategies": ["teacher_ensemble"],
            "synthetic_multipliers": [0.5],
            "max_candidates_per_source": 3,
            "max_teacher_stds": [None],
            "min_similarities": [0.0],
            "clip_y_min": 0.0,
            "clip_y_max": 100.0,
            "max_resample_attempts": 8,
            "teacher_models": ["ridge"],
            "donor_similarity_n_bits": 64,
            "donor_similarity_radius": 2,
            "donor_similarity_backend": "rdkit",
        }
    else:
        raise ValueError(kind)
    path = tmp_path / f"{kind}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path
