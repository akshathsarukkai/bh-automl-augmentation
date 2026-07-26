"""RDKit canonicalization and stable identities for seven-role BH reactions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    CANONICAL_ROLE_NAMES,
    ROLE_TO_COLUMN,
)

CANONICALIZATION_VERSION = "bh-rdkit-isomeric-v1"
REACTION_KEY_SCHEMA_VERSION = "bh-seven-role-v1"
PARTIAL_KEY_SCHEMA_VERSION = "bh-partial-role-v1"


@dataclass(frozen=True)
class CanonicalMolecule:
    """One raw molecular string and its deterministic RDKit canonicalization."""

    raw_smiles: str
    canonical_smiles: str | None
    parse_valid: bool
    error_type: str | None
    error_message: str | None


def canonicalize_smiles(
    smiles: str,
    *,
    isomeric: bool = True,
) -> CanonicalMolecule:
    """Canonicalize one molecule without standardizing its chemical state.

    RDKit canonical SMILES preserves stereochemistry (when ``isomeric=True``),
    isotopes, formal charges, aromaticity, and disconnected fragments. This
    function intentionally performs no neutralization, tautomer handling, salt
    stripping, fragment removal, or protonation-state normalization.
    """
    raw = smiles if isinstance(smiles, str) else str(smiles)
    text = raw.strip()
    if not isinstance(smiles, str) or not text:
        return CanonicalMolecule(
            raw_smiles=raw,
            canonical_smiles=None,
            parse_valid=False,
            error_type="missing_or_non_string_smiles",
            error_message="Required molecular role must be a non-empty SMILES string.",
        )

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for scientific molecular canonicalization. "
            "Install it with `pip install rdkit`."
        ) from exc

    try:
        molecule = Chem.MolFromSmiles(text)
    except Exception as exc:  # pragma: no cover - defensive around native RDKit errors
        return CanonicalMolecule(
            raw_smiles=raw,
            canonical_smiles=None,
            parse_valid=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    if molecule is None:
        return CanonicalMolecule(
            raw_smiles=raw,
            canonical_smiles=None,
            parse_valid=False,
            error_type="rdkit_parse_failed",
            error_message=f"RDKit MolFromSmiles returned None for {text!r}.",
        )

    try:
        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=isomeric,
        )
    except Exception as exc:  # pragma: no cover - defensive around native RDKit errors
        return CanonicalMolecule(
            raw_smiles=raw,
            canonical_smiles=None,
            parse_valid=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return CanonicalMolecule(
        raw_smiles=raw,
        canonical_smiles=canonical,
        parse_valid=True,
        error_type=None,
        error_message=None,
    )


def canonicalize_reaction_roles_dataframe(
    df: pd.DataFrame,
    *,
    source_file_hash: str,
    isomeric: bool = True,
    source_row_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Add raw/canonical columns and versioned identities for all seven roles.

    The source row ID is derived from the immutable source-file hash and the
    original row position, not from the mutable DataFrame index. No rows are
    removed and existing recovered-role/source columns are retained.
    """
    if not source_file_hash:
        raise ValueError("source_file_hash is required to construct stable source_row_id values.")
    enriched = _ensure_role_columns_for_audit(df)
    if "source_row_id" in enriched.columns:
        raise ValueError("Input already contains source_row_id; refusing to replace row identity.")

    original_indices = enriched.index.tolist()
    positions = _resolve_source_row_positions(enriched, source_row_positions)
    enriched.insert(0, "source_row_id", [
        source_row_id(source_file_hash, position)
        for position in positions
    ])
    if "source_row_position" not in enriched.columns:
        enriched.insert(1, "source_row_position", positions)
    enriched.insert(2, "source_dataframe_index", [str(value) for value in original_indices])

    invalid_names_per_row: list[list[str]] = [[] for _ in range(len(enriched))]
    for role in CANONICAL_ROLE_NAMES:
        source_column = ROLE_TO_COLUMN[role]
        results = [
            canonicalize_smiles(value, isomeric=isomeric)
            for value in enriched[source_column].tolist()
        ]
        enriched[f"raw_{role}_smiles"] = [result.raw_smiles for result in results]
        enriched[f"canonical_{role}_smiles"] = [
            result.canonical_smiles for result in results
        ]
        enriched[f"{role}_parse_valid"] = [result.parse_valid for result in results]
        enriched[f"{role}_parse_error"] = [
            _format_parse_error(result) for result in results
        ]
        for position, result in enumerate(results):
            if not result.parse_valid:
                invalid_names_per_row[position].append(role)

    enriched["all_required_roles_parse_valid"] = [
        not invalid_names for invalid_names in invalid_names_per_row
    ]
    enriched["n_invalid_roles"] = [len(invalid_names) for invalid_names in invalid_names_per_row]
    enriched["invalid_role_names"] = [
        json.dumps(invalid_names, separators=(",", ":"))
        for invalid_names in invalid_names_per_row
    ]
    enriched["canonicalization_version"] = CANONICALIZATION_VERSION

    identity_records = [
        build_canonical_reaction_identity(row)
        for row in enriched.to_dict(orient="records")
    ]
    enriched["canonical_reaction_key"] = [record["key"] for record in identity_records]
    enriched["canonical_reaction_hash"] = [record["hash"] for record in identity_records]
    enriched["canonical_substrate_key"] = [
        _partial_key(record, ("reactant_1", "reactant_2"), "substrate")
        for record in enriched.to_dict(orient="records")
    ]
    enriched["canonical_condition_key"] = [
        _partial_key(
            record,
            ("catalyst", "ligand", "base", "solvent_or_additive"),
            "condition",
        )
        for record in enriched.to_dict(orient="records")
    ]
    enriched["canonical_product_key"] = [
        _partial_key(record, ("product",), "product")
        for record in enriched.to_dict(orient="records")
    ]
    return enriched


def build_canonical_reaction_identity(row: Mapping[str, Any]) -> dict[str, str | None]:
    """Return stable JSON and SHA-256 identity for one valid seven-role reaction."""
    if _explicitly_false(row.get("all_required_roles_parse_valid")):
        return {"key": None, "hash": None}
    values: dict[str, str] = {}
    for role in CANONICAL_ROLE_NAMES:
        if _explicitly_false(row.get(f"{role}_parse_valid")):
            return {"key": None, "hash": None}
        column = f"canonical_{role}_smiles"
        value = row.get(column)
        if value is None or bool(pd.isna(value)) or not str(value):
            return {"key": None, "hash": None}
        values[role] = str(value)

    payload = {"schema": REACTION_KEY_SCHEMA_VERSION, **values}
    key = stable_json(payload)
    return {"key": key, "hash": sha256_text(key)}


def source_row_id(source_file_hash: str, original_row_position: int) -> str:
    """Build immutable row identity from source-file content and row position."""
    payload = {
        "schema": "source-row-v1",
        "source_file_sha256": source_file_hash,
        "original_row_position": int(original_row_position),
    }
    return sha256_text(stable_json(payload))


def stable_json(value: Mapping[str, Any]) -> str:
    """Serialize identity data without dependence on mapping insertion order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _partial_key(
    row: Mapping[str, Any],
    roles: tuple[str, ...],
    key_type: str,
) -> str | None:
    values: dict[str, str] = {}
    for role in roles:
        value = row.get(f"canonical_{role}_smiles")
        if value is None or bool(pd.isna(value)) or not str(value):
            return None
        values[role] = str(value)
    return stable_json(
        {
            "schema": PARTIAL_KEY_SCHEMA_VERSION,
            "key_type": key_type,
            **values,
        }
    )


def _format_parse_error(result: CanonicalMolecule) -> str | None:
    if result.parse_valid:
        return None
    return f"{result.error_type}: {result.error_message}"


def _ensure_role_columns_for_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Recover missing role columns without rejecting invalid role values."""
    enriched = df.copy()
    missing = [column for column in CANONICAL_ROLE_COLUMNS if column not in enriched.columns]
    if missing:
        from bh_augmentation.data.bh_condition_reader import augment_bh_dataframe

        enriched = augment_bh_dataframe(enriched, strict=False)
        missing = [
            column for column in CANONICAL_ROLE_COLUMNS if column not in enriched.columns
        ]
    if missing:
        raise ValueError(
            "Cannot audit canonical roles because required recovered columns are missing: "
            + ", ".join(missing)
        )
    return enriched


def _resolve_source_row_positions(
    df: pd.DataFrame,
    supplied: Sequence[int] | None,
) -> list[int]:
    if supplied is not None:
        positions = [int(value) for value in supplied]
    elif "source_row_position" in df.columns:
        positions = pd.to_numeric(
            df["source_row_position"], errors="raise"
        ).astype(int).tolist()
    elif df.index.is_unique and pd.api.types.is_integer_dtype(df.index.dtype):
        positions = [int(value) for value in df.index.tolist()]
    else:
        raise ValueError(
            "Original source-row positions are required for stable row identity. "
            "Pass source_row_positions or retain a unique integer source index."
        )
    if len(positions) != len(df):
        raise ValueError("source_row_positions must contain one position per input row.")
    if len(set(positions)) != len(positions) or any(position < 0 for position in positions):
        raise ValueError("Original source-row positions must be unique non-negative integers.")
    return positions


def _explicitly_false(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return value is False or value == 0
