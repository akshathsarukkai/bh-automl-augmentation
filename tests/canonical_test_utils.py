"""Shared deterministic fixtures for canonical-data tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_to_record


def measured_role_frame() -> pd.DataFrame:
    """Return valid rows containing canonical duplicates and yield conflicts."""
    raw_roles = [
        ReactionRoles("C(C)Br", "CN", "[Pd]", "CP(C)C", "[Na+].[OH-]", "CCO", "CCN"),
        ReactionRoles("CCBr", "CN", "[Pd]", "CP(C)C", "[OH-].[Na+]", "OCC", "CCN"),
        ReactionRoles("C(C)Br", "CN", "[Pd]", "CP(C)C", "[Na+].[OH-]", "CCO", "CCN"),
        ReactionRoles("C(C)Br", "CN", "[Pd]", "CP(C)C", "[Na+].[OH-]", "CCO", "CCN"),
        ReactionRoles("CCCl", "CN", "[Pd]", "CP(C)C", "[K+].[OH-]", "CO", "CCCN"),
    ]
    yields = [50.0, 50.0, 70.0, 50.0, 20.0]
    records = []
    for position, (roles, measured_yield) in enumerate(zip(raw_roles, yields, strict=True)):
        record = reaction_roles_to_record(roles)
        record.update(
            {
                "reaction_id": f"reaction_{position}",
                "yield": measured_yield,
                "temperature": "UNKNOWN",
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def grouped_assignment_frame(n_groups: int = 30) -> pd.DataFrame:
    """Return source IDs and canonical keys with deterministic replicates."""
    records = []
    for group in range(n_groups):
        replicate_count = 2 if group % 7 == 0 else 1
        for replicate in range(replicate_count):
            records.append(
                {
                    "source_row_id": f"row-{group:03d}-{replicate}",
                    "canonical_reaction_key": f"group-{group:03d}",
                    "yield": float((group + replicate) % 100),
                }
            )
    return pd.DataFrame(records)


def audit_config(source: Path, canonical: Path, output: Path) -> dict[str, Any]:
    return {
        "scientific_run": True,
        "allow_hash_fingerprint_fallback": False,
        "invalid_smiles_policy": "audit_then_error_for_scientific_dataset",
        "dataset": {"path": str(source)},
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
            "n_bits": 16,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "output": {
            "canonical_dataset_path": str(canonical),
            "directory": str(output),
        },
    }


def split_config(
    source: Path,
    canonical: Path,
    output: Path,
    *,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "scientific_run": True,
        "allow_hash_fingerprint_fallback": False,
        "invalid_smiles_policy": "audit_then_error_for_scientific_dataset",
        "canonicalization": {
            "backend": "rdkit",
            "isomeric_smiles": True,
            "preserve_role_order": True,
        },
        "dataset": {
            "source_path": str(source),
            "path": str(canonical),
        },
        "splits": {
            "group_column": "canonical_reaction_key",
            "seeds": seeds or [0],
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "train_fractions": [0.01, 0.05, 0.1, 0.2, 1.0],
        },
        "output": {"directory": str(output)},
    }
