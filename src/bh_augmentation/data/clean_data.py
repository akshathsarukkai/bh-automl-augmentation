"""Cleaning helpers for Buchwald-Hartwig reaction-yield data."""

from __future__ import annotations

import re

import pandas as pd

NORMALIZED_SCHEMA = [
    "reaction_id",
    "aryl_halide_smiles",
    "amine_smiles",
    "ligand_smiles",
    "base_smiles",
    "additive_smiles",
    "solvent",
    "temperature",
    "reaction_smiles",
    "yield",
]

OPTIONAL_CATEGORICAL_COLUMNS = [
    "ligand_smiles",
    "base_smiles",
    "additive_smiles",
    "solvent",
    "reaction_smiles",
]

COLUMN_ALIASES = {
    "id": "reaction_id",
    "rxn_id": "reaction_id",
    "reactionid": "reaction_id",
    "yield_percent": "yield",
    "yield_percentage": "yield",
    "yield_pct": "yield",
    "yield_": "yield",
    "aryl_halide": "aryl_halide_smiles",
    "amine": "amine_smiles",
    "ligand": "ligand_smiles",
    "base": "base_smiles",
    "additive": "additive_smiles",
}


def clean_buchwald_hartwig(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean Buchwald-Hartwig reaction-yield records.

    Column names are standardized to lowercase snake case. Rows with missing,
    non-numeric, or out-of-range yields are dropped. The accepted yield range is
    `0 <= yield <= 100`; values outside that range are treated as invalid
    measurements rather than clipped.

    Missing optional categorical fields are filled with `"UNKNOWN"`. The
    returned DataFrame includes `attrs["original_row_count"]` and
    `attrs["dropped_invalid_yield_count"]` metadata.
    """
    cleaned = df.copy()
    original_row_count = len(cleaned)
    cleaned.columns = [_normalize_column_name(column) for column in cleaned.columns]
    cleaned = cleaned.rename(columns=COLUMN_ALIASES)

    if "reaction_id" not in cleaned.columns:
        cleaned.insert(0, "reaction_id", [_make_reaction_id(i) for i in range(original_row_count)])

    if "yield" not in cleaned.columns:
        cleaned["yield"] = pd.NA

    cleaned["yield"] = pd.to_numeric(cleaned["yield"], errors="coerce")
    valid_yield = cleaned["yield"].between(0, 100, inclusive="both")
    cleaned = cleaned.loc[valid_yield].copy()

    for column in NORMALIZED_SCHEMA:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA

    for column in OPTIONAL_CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna("UNKNOWN")
        cleaned[column] = cleaned[column].replace("", "UNKNOWN")

    cleaned = cleaned[NORMALIZED_SCHEMA].reset_index(drop=True)
    cleaned.attrs["original_row_count"] = original_row_count
    cleaned.attrs["dropped_invalid_yield_count"] = original_row_count - len(cleaned)
    return cleaned


def _normalize_column_name(column: object) -> str:
    """Convert a raw column name to lowercase snake case."""
    normalized = str(column).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _make_reaction_id(index: int) -> str:
    """Create a stable synthetic reaction identifier for a row index."""
    return f"rxn_{index + 1:06d}"
