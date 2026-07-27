"""Conservative canonical substrate-scaffold definitions for BH OOD splits.

Role assignment is chemistry-based rather than positional:

* the electrophile is the unique canonical reactant containing a carbon atom
  directly bonded to Cl, Br, or I (SMARTS ``[#6]-[Cl,Br,I]``);
* the amine is the complementary canonical reactant and must contain a neutral
  primary/secondary or protonated amine-like nitrogen (SMARTS
  ``[N;H1,H2,H3;+0,+1]``).

This deliberately excludes pseudohalides, tertiary amines, ambiguous
multi-electrophile pairs, and unsupported partners instead of guessing roles.
Canonical isomeric SMILES is retained and hashed before scaffold derivation.
No neutralization, salt stripping, scaffold genericization, or fingerprint
equivalence is applied; different molecular identities may intentionally share
the same directly derived scaffold group.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bh_augmentation.data.canonicalize_roles import canonicalize_smiles
from bh_augmentation.utils.corrected_runs import stable_hash

SCAFFOLD_SCHEMA_VERSION = "bh-canonical-scaffold-ood-v1"
ELECTROPHILE_SMARTS = "[#6]-[Cl,Br,I]"
AMINE_SMARTS = "[N;H1,H2,H3;+0,+1]"
SCAFFOLD_TARGETS = ("electrophile_scaffold", "amine_scaffold")
_ROLE_COLUMNS = {
    "reactant_1": "canonical_reactant_1_smiles",
    "reactant_2": "canonical_reactant_2_smiles",
}
SCAFFOLD_ASSIGNMENT_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "scaffold_target",
    "assigned_role",
    "assigned_canonical_smiles",
    "molecule_identity_hash",
    "electrophile_match_roles",
    "complementary_amine_match",
    "chemical_parse_valid",
    "role_assignment_valid",
    "bemis_murcko_scaffold_smiles",
    "scaffold_key",
    "scaffold_hash",
    "included",
    "exclusion_reason",
)


@dataclass(frozen=True, slots=True)
class ScaffoldHoldout:
    """One deterministic scaffold-group holdout over included reactions."""

    target: str
    heldout_scaffold_key: str
    train_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]
    excluded_source_ids: tuple[str, ...]
    split_hash: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ScaffoldOODAssignments:
    """Immutable audited scaffold assignments for one chemical target."""

    target: str
    included_scaffold_groups: tuple[str, ...]
    assignment_hash: str
    excluded_reason_counts: tuple[tuple[str, int], ...]
    _assignments: pd.DataFrame = field(repr=False, compare=False)

    @property
    def assignments(self) -> pd.DataFrame:
        """Return a defensive copy of per-reaction scaffold assignments."""
        return self._assignments.copy(deep=True)

    @property
    def included(self) -> pd.DataFrame:
        """Return included rows in deterministic source-ID order."""
        return self._assignments.loc[self._assignments["included"]].copy(
            deep=True
        )

    @property
    def excluded(self) -> pd.DataFrame:
        """Return excluded rows with explicit reasons."""
        return self._assignments.loc[~self._assignments["included"]].copy(
            deep=True
        )

    def holdout(self, scaffold_key: str) -> ScaffoldHoldout:
        """Hold out exactly one included scaffold group."""
        if len(self.included_scaffold_groups) < 2:
            raise ValueError(
                f"Insufficient included {self.target} scaffold groups for holdout."
            )
        if scaffold_key not in self.included_scaffold_groups:
            raise ValueError(
                f"Scaffold key is absent from included {self.target} groups."
            )
        included = self.included
        test_ids = tuple(
            included.loc[
                included["scaffold_key"].eq(scaffold_key),
                "source_row_id",
            ]
        )
        train_ids = tuple(
            included.loc[
                ~included["scaffold_key"].eq(scaffold_key),
                "source_row_id",
            ]
        )
        excluded_ids = tuple(self.excluded["source_row_id"])
        split_hash = stable_hash(
            {
                "schema_version": SCAFFOLD_SCHEMA_VERSION,
                "target": self.target,
                "heldout_scaffold_key": scaffold_key,
                "train_source_ids": list(train_ids),
                "test_source_ids": list(test_ids),
                "excluded_source_ids": list(excluded_ids),
                "assignment_hash": self.assignment_hash,
            }
        )
        return ScaffoldHoldout(
            target=self.target,
            heldout_scaffold_key=scaffold_key,
            train_source_ids=train_ids,
            test_source_ids=test_ids,
            excluded_source_ids=excluded_ids,
            split_hash=split_hash,
        )


def build_scaffold_ood_assignments(
    canonical: pd.DataFrame,
    *,
    target: str,
) -> ScaffoldOODAssignments:
    """Assign conservative BH substrate roles and Bemis–Murcko scaffolds."""
    if target not in SCAFFOLD_TARGETS:
        raise ValueError(
            f"Unsupported scaffold OOD target {target!r}; "
            f"supported={list(SCAFFOLD_TARGETS)}."
        )
    frame = _validate_input(canonical)
    records = [
        _assignment_record(row, target=target)
        for row in frame.to_dict(orient="records")
    ]
    assignments = pd.DataFrame(
        records,
        columns=SCAFFOLD_ASSIGNMENT_COLUMNS,
    ).sort_values("source_row_id", kind="mergesort").reset_index(drop=True)
    included_groups = tuple(
        sorted(
            assignments.loc[assignments["included"], "scaffold_key"].unique()
        )
    )
    excluded_counts = tuple(
        sorted(
            (
                str(reason),
                int(count),
            )
            for reason, count in assignments.loc[
                ~assignments["included"],
                "exclusion_reason",
            ]
            .value_counts()
            .items()
        )
    )
    assignment_hash = stable_hash(
        {
            "schema_version": SCAFFOLD_SCHEMA_VERSION,
            "target": target,
            "assignments": assignments.to_dict(orient="records"),
        }
    )
    return ScaffoldOODAssignments(
        target=target,
        included_scaffold_groups=included_groups,
        assignment_hash=assignment_hash,
        excluded_reason_counts=excluded_counts,
        _assignments=assignments.copy(deep=True),
    )


def _assignment_record(
    row: Mapping[str, Any],
    *,
    target: str,
) -> dict[str, Any]:
    base = {
        "source_row_id": str(row["source_row_id"]),
        "canonical_reaction_key": str(row["canonical_reaction_key"]),
        "scaffold_target": target,
        "assigned_role": None,
        "assigned_canonical_smiles": None,
        "molecule_identity_hash": None,
        "electrophile_match_roles": "[]",
        "complementary_amine_match": False,
        "chemical_parse_valid": False,
        "role_assignment_valid": False,
        "bemis_murcko_scaffold_smiles": None,
        "scaffold_key": None,
        "scaffold_hash": None,
        "included": False,
        "exclusion_reason": None,
    }
    molecules: dict[str, tuple[str, Any]] = {}
    for role, column in _ROLE_COLUMNS.items():
        result = canonicalize_smiles(row[column], isomeric=True)
        if not result.parse_valid or result.canonical_smiles is None:
            return {
                **base,
                "exclusion_reason": f"invalid_{role}_canonical_identity",
            }
        if result.canonical_smiles != str(row[column]).strip():
            return {
                **base,
                "exclusion_reason": f"noncanonical_{role}_identity",
            }
        molecules[role] = (
            result.canonical_smiles,
            _rdkit_molecule(result.canonical_smiles),
        )
    base["chemical_parse_valid"] = True
    electrophile_roles = tuple(
        role
        for role, (_, molecule) in molecules.items()
        if _matches_smarts(molecule, ELECTROPHILE_SMARTS)
    )
    base["electrophile_match_roles"] = json.dumps(
        list(electrophile_roles),
        separators=(",", ":"),
    )
    if not electrophile_roles:
        return {
            **base,
            "exclusion_reason": "no_unique_carbon_halogen_electrophile",
        }
    if len(electrophile_roles) != 1:
        return {
            **base,
            "exclusion_reason": "ambiguous_carbon_halogen_electrophile",
        }
    electrophile_role = electrophile_roles[0]
    amine_role = (
        "reactant_2" if electrophile_role == "reactant_1" else "reactant_1"
    )
    amine_match = _matches_smarts(molecules[amine_role][1], AMINE_SMARTS)
    base["complementary_amine_match"] = amine_match
    if not amine_match:
        return {
            **base,
            "exclusion_reason": "complementary_reactant_not_amine_like",
        }
    assigned_role = (
        electrophile_role if target == "electrophile_scaffold" else amine_role
    )
    assigned_smiles, assigned_molecule = molecules[assigned_role]
    base.update(
        {
            "assigned_role": assigned_role,
            "assigned_canonical_smiles": assigned_smiles,
            "molecule_identity_hash": stable_hash(
                {
                    "schema_version": SCAFFOLD_SCHEMA_VERSION,
                    "canonical_isomeric_smiles": assigned_smiles,
                }
            ),
            "role_assignment_valid": True,
        }
    )
    scaffold_smiles = _bemis_murcko_scaffold(assigned_molecule)
    if not scaffold_smiles:
        return {
            **base,
            "exclusion_reason": "empty_bemis_murcko_scaffold",
        }
    scaffold_payload = {
        "schema_version": SCAFFOLD_SCHEMA_VERSION,
        "target": target,
        "bemis_murcko_scaffold_smiles": scaffold_smiles,
    }
    scaffold_key = json.dumps(
        scaffold_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **base,
        "bemis_murcko_scaffold_smiles": scaffold_smiles,
        "scaffold_key": scaffold_key,
        "scaffold_hash": stable_hash(scaffold_payload),
        "included": True,
        "exclusion_reason": None,
    }


def _validate_input(canonical: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(canonical, pd.DataFrame) or canonical.empty:
        raise ValueError("Scaffold OOD input must be a non-empty DataFrame.")
    required = {
        "source_row_id",
        "canonical_reaction_key",
        *_ROLE_COLUMNS.values(),
    }
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Scaffold OOD input is missing columns: {missing}.")
    frame = canonical.copy(deep=True)
    for column in required:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Scaffold OOD input column {column} has missing values.")
    frame["source_row_id"] = frame["source_row_id"].astype(str)
    frame["canonical_reaction_key"] = frame[
        "canonical_reaction_key"
    ].astype(str)
    if frame["source_row_id"].duplicated().any():
        raise ValueError("Scaffold OOD source_row_id values must be unique.")
    return frame.sort_values("source_row_id", kind="mergesort").reset_index(
        drop=True
    )


def _rdkit_molecule(smiles: str) -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for scientific scaffold OOD definitions."
        ) from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Validated canonical SMILES unexpectedly failed RDKit parsing.")
    return molecule


def _matches_smarts(molecule: Any, smarts: str) -> bool:
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:  # pragma: no cover - constant patterns are tested
        raise RuntimeError(f"Invalid internal scaffold role SMARTS: {smarts}.")
    return bool(molecule.HasSubstructMatch(pattern))


def _bemis_murcko_scaffold(molecule: Any) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return str(
        Chem.MolToSmiles(
            scaffold,
            canonical=True,
            isomericSmiles=True,
        )
    )
