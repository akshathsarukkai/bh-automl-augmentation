"""Canonical seven-role schema for processed Buchwald-Hartwig reactions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_ROLE_NAMES = (
    "reactant_1",
    "reactant_2",
    "catalyst",
    "ligand",
    "base",
    "solvent_or_additive",
    "product",
)

CANONICAL_ROLE_COLUMNS = (
    "recovered_reactant_1_smiles",
    "recovered_reactant_2_smiles",
    "recovered_catalyst_smiles",
    "recovered_ligand_smiles",
    "recovered_base_smiles",
    "recovered_solvent_or_additive_smiles",
    "recovered_product_smiles",
)

ROLE_TO_COLUMN = dict(zip(CANONICAL_ROLE_NAMES, CANONICAL_ROLE_COLUMNS, strict=True))


@dataclass(frozen=True)
class ReactionRoles:
    """One reaction expressed in the dataset-specific canonical BH role order."""

    reactant_1: str
    reactant_2: str
    catalyst: str
    ligand: str
    base: str
    solvent_or_additive: str
    product: str

    def reaction_smiles(self) -> str:
        """Return the stable six-left-token reaction string for these roles."""
        left = ".".join(
            [
                self.reactant_1,
                self.reactant_2,
                self.catalyst,
                self.ligand,
                self.base,
                self.solvent_or_additive,
            ]
        )
        return f"{left}>>{self.product}"


def reaction_roles_from_row(row: Mapping[str, object]) -> ReactionRoles:
    """Build validated roles from canonical recovered columns in one row."""
    missing_columns = [column for column in CANONICAL_ROLE_COLUMNS if column not in row]
    if missing_columns:
        raise ValueError(
            "Missing canonical Buchwald-Hartwig role columns: " + ", ".join(missing_columns)
        )

    values: dict[str, str] = {}
    invalid: list[str] = []
    for role, column in ROLE_TO_COLUMN.items():
        value = row[column]
        if _missing_required_molecule(value):
            invalid.append(column)
        else:
            values[role] = str(value).strip()
    if invalid:
        raise ValueError(
            "Canonical Buchwald-Hartwig roles require non-empty molecule identities; "
            "invalid fields: " + ", ".join(invalid)
        )
    return ReactionRoles(**values)


def reaction_roles_to_record(roles: ReactionRoles) -> dict[str, str]:
    """Convert roles to canonical recovered columns plus reaction SMILES."""
    record = {
        ROLE_TO_COLUMN[role]: getattr(roles, role)
        for role in CANONICAL_ROLE_NAMES
    }
    record["reaction_smiles"] = roles.reaction_smiles()
    return record


def reaction_roles_dataframe(
    roles: Sequence[ReactionRoles],
    *,
    yields: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build a canonical role DataFrame without relying on column insertion order."""
    records = [reaction_roles_to_record(item) for item in roles]
    frame = pd.DataFrame(records, columns=[*CANONICAL_ROLE_COLUMNS, "reaction_smiles"])
    if yields is None:
        frame["yield"] = np.nan
    else:
        if len(yields) != len(roles):
            raise ValueError("yields must contain one value per ReactionRoles item.")
        frame["yield"] = np.asarray(yields, dtype=np.float32)
    return frame


def ensure_reaction_role_columns(
    df: pd.DataFrame,
    *,
    parse_if_missing: bool = True,
) -> pd.DataFrame:
    """Return a copy with complete, validated canonical BH role columns.

    Positional recovery is delegated to the dataset-specific condition reader.
    Missing or UNKNOWN required molecules are rejected rather than represented
    as empty strings or zero fingerprints.
    """
    result = df.copy()
    missing = [column for column in CANONICAL_ROLE_COLUMNS if column not in result.columns]
    if missing and parse_if_missing:
        from bh_augmentation.data.bh_condition_reader import augment_bh_dataframe

        result = augment_bh_dataframe(result, strict=False)
        missing = [column for column in CANONICAL_ROLE_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(
            "Canonical Buchwald-Hartwig role columns are missing: "
            + ", ".join(missing)
            + ". Generate the condition-enriched CSV with bh_condition_reader.py first."
        )

    invalid_rows: list[str] = []
    for index, row in result.iterrows():
        try:
            reaction_roles_from_row(row)
        except ValueError as exc:
            invalid_rows.append(f"row {index}: {exc}")
            if len(invalid_rows) >= 3:
                break
    if invalid_rows:
        raise ValueError(
            "Cannot build canonical Buchwald-Hartwig roles because required molecule "
            "identities are missing or invalid. " + " | ".join(invalid_rows)
        )
    return result


def _missing_required_molecule(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip().upper()
    return normalized in {"", "UNKNOWN", "NOT_RECOVERABLE", "NAN", "NONE"}
