"""Reaction-family adapter seam for applying the canonical framework to new chemistry.

Phase 16 status
---------------
``implementation passed``
``external empirical validation blocked``

The blocking reason is that no licensed Suzuki-Miyaura dataset is present in this
repository. The only reaction data available locally is Buchwald-Hartwig, derived
from the TDC ``Yields(name="Buchwald-Hartwig")`` export. Acquiring a real
Suzuki-Miyaura high-throughput-experimentation (HTE) dataset requires a human with
network access and the ability to accept the source license; see
``docs/EXTERNAL_DATASETS.md``. Every artefact in this module is therefore exercised
against a clearly labelled synthetic fixture and never against measured Suzuki
yields.

What this module is for
-----------------------
The canonical framework built in earlier phases -- RDKit canonical molecular
identity, versioned reaction keys, complete-group train/validation/test
separation, training-only augmentation, validation-only policy selection, frozen
policies before test access, one test evaluation per unit, and hash-verifiable
outputs -- contains nothing that is specific to Buchwald-Hartwig chemistry except
the *role vocabulary*. This module isolates that vocabulary behind
:class:`ReactionFamilyAdapter` so a chemically distinct reaction family can be
plugged in without redesigning the system.

Reused, family-agnostic (imported, not reimplemented):

* :func:`bh_augmentation.data.canonicalize_roles.canonicalize_smiles` -- RDKit
  canonicalization of one molecule, with no neutralization or tautomer handling.
* :func:`bh_augmentation.data.canonicalize_roles.stable_json`,
  :func:`~bh_augmentation.data.canonicalize_roles.sha256_text`, and
  :func:`~bh_augmentation.data.canonicalize_roles.source_row_id` -- stable
  identity encodings.
* :func:`bh_augmentation.data.canonical_splits.build_grouped_outer_assignments`
  and :func:`~bh_augmentation.data.canonical_splits.build_nested_low_data_assignments`
  -- grouped splits that already operate on any ``canonical_reaction_key`` column.
* :mod:`bh_augmentation.evaluation.policy_protocol` -- search/freeze/final
  evaluation, which never mentions a role name.

Family-specific (declared once, in an adapter instance): role names, source
columns, the substrate/condition/product partition, the transferable-role subset,
the canonicalization and reaction-key schema versions, the reaction-SMILES
assembly order, dataset provenance/license metadata, and the eligibility rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION as BH_CANONICALIZATION_VERSION,
)
from bh_augmentation.data.canonicalize_roles import (
    PARTIAL_KEY_SCHEMA_VERSION as BH_PARTIAL_KEY_SCHEMA_VERSION,
)
from bh_augmentation.data.canonicalize_roles import (
    REACTION_KEY_SCHEMA_VERSION as BH_REACTION_KEY_SCHEMA_VERSION,
)
from bh_augmentation.data.canonicalize_roles import (
    canonicalize_smiles,
    sha256_text,
    source_row_id,
    stable_json,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    CANONICAL_ROLE_NAMES,
)

REACTION_FAMILY_ADAPTER_VERSION = "reaction-family-adapter-v1"
PROVENANCE_SCHEMA_VERSION = "external-dataset-provenance-v1"

PROVENANCE_FIELDS = (
    "source_name",
    "source_url",
    "license",
    "citation",
    "retrieved_date",
    "raw_file_sha256",
)

MISSING_MOLECULE_TOKENS = frozenset(
    {"", "UNKNOWN", "NOT_RECOVERABLE", "NAN", "NONE"}
)

#: Returned rejection reason (or ``None``) for one canonical role mapping.
FamilyEligibilityRule = Callable[[Mapping[str, str]], "str | None"]


def no_family_eligibility_rule(canonical_roles: Mapping[str, str]) -> str | None:
    """Accept every chemically parseable reaction (the default, no-op hook)."""
    del canonical_roles
    return None


@dataclass(frozen=True)
class DatasetProvenance:
    """Mandatory provenance and licensing record for one external dataset.

    Every field is required. A reaction-family adapter cannot be constructed
    without one, so no external dataset can enter the pipeline anonymously. When
    a value is genuinely unknown the honest action is to record a string that
    says so (for example ``"verify before use"``) rather than to invent a URL,
    licence, or date.
    """

    source_name: str
    source_url: str
    license: str
    citation: str
    retrieved_date: str
    raw_file_sha256: str

    def __post_init__(self) -> None:
        for name in PROVENANCE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Dataset provenance field {name!r} is required and must be a "
                    "non-empty string."
                )
        digest = self.raw_file_sha256.strip()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(
                "Dataset provenance raw_file_sha256 must be a 64-character lowercase "
                f"hex SHA-256 digest of the raw source file; got {self.raw_file_sha256!r}."
            )

    def to_dict(self) -> dict[str, str]:
        """Serialize the exact provenance schema."""
        return {
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            **{name: getattr(self, name) for name in PROVENANCE_FIELDS},
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetProvenance:
        """Parse a strict provenance mapping and reject missing or unknown fields."""
        if not isinstance(value, Mapping):
            raise ValueError("Dataset provenance must be a mapping.")
        supplied = {
            str(key): item
            for key, item in value.items()
            if str(key) != "provenance_schema_version"
        }
        missing = sorted(set(PROVENANCE_FIELDS) - set(supplied))
        unknown = sorted(set(supplied) - set(PROVENANCE_FIELDS))
        if missing or unknown:
            raise ValueError(
                "Dataset provenance schema mismatch: "
                f"missing={missing}, unknown={unknown}."
            )
        return cls(**{name: supplied[name] for name in PROVENANCE_FIELDS})


@dataclass(frozen=True)
class CanonicalFamilyReaction:
    """Canonical form and versioned identity of one typed reaction."""

    family_id: str
    canonical_roles: Mapping[str, str] | None
    reaction_key: str | None
    reaction_hash: str | None
    parse_valid: bool
    invalid_roles: tuple[str, ...]
    canonicalization_version: str
    eligibility_reason: str | None = None

    @property
    def eligible(self) -> bool:
        """Return whether the reaction parsed and passed the family rule."""
        return self.parse_valid and self.eligibility_reason is None


@dataclass(frozen=True)
class ReactionFamilyAdapter:
    """Everything that is specific to one reaction family, declared in one place.

    Parameters
    ----------
    family_id:
        Machine identifier, embedded in every family-scoped artefact.
    family_name:
        Human-readable reaction-family name.
    role_names:
        Ordered tuple of typed role names. Order is the canonical role order.
    role_columns:
        Source-dataset column supplying each role, positionally aligned with
        ``role_names``.
    substrate_roles:
        Roles that contribute atoms to the product skeleton. These are never
        transferable: changing one changes what reaction is being described.
    condition_roles:
        Roles describing how the substrates are coupled rather than what is
        coupled.
    transferable_roles:
        The subset of ``condition_roles`` that typed condition transfer is
        allowed to vary. Defaults to all condition roles.
    product_role:
        The single measured product role.
    assembly_order:
        Left-hand token order of the reaction SMILES. The product is always the
        single right-hand token, so ``assembly_order`` is a permutation of
        ``role_names`` minus ``product_role``.
    provenance:
        Required dataset provenance/licensing record.
    eligibility_rule:
        Family-specific validation hook applied to canonical role values. It
        returns ``None`` for an eligible reaction or a short machine-readable
        rejection reason.
    """

    family_id: str
    family_name: str
    role_names: tuple[str, ...]
    role_columns: tuple[str, ...]
    substrate_roles: tuple[str, ...]
    condition_roles: tuple[str, ...]
    product_role: str
    canonicalization_version: str
    reaction_key_schema_version: str
    partial_key_schema_version: str
    assembly_order: tuple[str, ...]
    provenance: DatasetProvenance
    transferable_roles: tuple[str, ...] = ()
    eligibility_rule: FamilyEligibilityRule = field(
        default=no_family_eligibility_rule, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("family_id", "family_name", "product_role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Reaction family adapter {name!r} must be a non-empty string.")
        for name in (
            "canonicalization_version",
            "reaction_key_schema_version",
            "partial_key_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Reaction family adapter {name!r} must be a non-empty version string."
                )
        object.__setattr__(self, "role_names", tuple(str(role) for role in self.role_names))
        object.__setattr__(self, "role_columns", tuple(str(col) for col in self.role_columns))
        object.__setattr__(
            self, "substrate_roles", tuple(str(role) for role in self.substrate_roles)
        )
        object.__setattr__(
            self, "condition_roles", tuple(str(role) for role in self.condition_roles)
        )
        object.__setattr__(
            self, "assembly_order", tuple(str(role) for role in self.assembly_order)
        )
        if not self.transferable_roles:
            object.__setattr__(self, "transferable_roles", tuple(self.condition_roles))
        else:
            object.__setattr__(
                self, "transferable_roles", tuple(str(r) for r in self.transferable_roles)
            )

        if not self.role_names:
            raise ValueError("A reaction family requires at least one typed role.")
        if len(set(self.role_names)) != len(self.role_names):
            raise ValueError("Reaction family role_names must be unique.")
        if len(self.role_columns) != len(self.role_names):
            raise ValueError("role_columns must contain exactly one column per role.")
        if len(set(self.role_columns)) != len(self.role_columns):
            raise ValueError("Reaction family role_columns must be unique.")

        roles = set(self.role_names)
        for name in ("substrate_roles", "condition_roles"):
            unknown = sorted(set(getattr(self, name)) - roles)
            if unknown:
                raise ValueError(f"{name} contains unknown roles: {unknown}.")
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty.")
        if set(self.substrate_roles) & set(self.condition_roles):
            raise ValueError("A role cannot be both a substrate and a condition.")
        if self.product_role not in roles:
            raise ValueError(f"product_role {self.product_role!r} is not a declared role.")
        if self.product_role in set(self.substrate_roles) | set(self.condition_roles):
            raise ValueError("product_role must not also be a substrate or condition role.")
        covered = set(self.substrate_roles) | set(self.condition_roles) | {self.product_role}
        if covered != roles:
            raise ValueError(
                "Every typed role must be classified as substrate, condition, or product; "
                f"unclassified={sorted(roles - covered)}."
            )
        unknown_transferable = sorted(set(self.transferable_roles) - set(self.condition_roles))
        if unknown_transferable:
            raise ValueError(
                "transferable_roles must be a subset of condition_roles; "
                f"unknown={unknown_transferable}."
            )
        if set(self.assembly_order) != roles - {self.product_role} or len(
            self.assembly_order
        ) != len(roles) - 1:
            raise ValueError(
                "assembly_order must be a permutation of every non-product role."
            )
        if not isinstance(self.provenance, DatasetProvenance):
            raise ValueError(
                "Reaction family adapters require a DatasetProvenance record; "
                "external datasets may not be registered without provenance."
            )
        if not callable(self.eligibility_rule):
            raise ValueError("eligibility_rule must be callable.")

    # ------------------------------------------------------------------
    # Schema accessors
    # ------------------------------------------------------------------
    @property
    def role_to_column(self) -> dict[str, str]:
        """Return the role -> source column mapping."""
        return dict(zip(self.role_names, self.role_columns, strict=True))

    @property
    def column_to_role(self) -> dict[str, str]:
        """Return the source column -> role mapping."""
        return dict(zip(self.role_columns, self.role_names, strict=True))

    @property
    def reaction_smiles_role_order(self) -> tuple[str, ...]:
        """Return the full left-to-right role order of the reaction SMILES."""
        return (*self.assembly_order, self.product_role)

    @property
    def canonical_role_columns(self) -> tuple[str, ...]:
        """Return the canonicalized-SMILES column for each role, in role order."""
        return tuple(f"canonical_{role}_smiles" for role in self.role_names)

    def schema_record(self) -> dict[str, Any]:
        """Return the complete, hashable declaration of this family schema."""
        return {
            "adapter_version": REACTION_FAMILY_ADAPTER_VERSION,
            "family_id": self.family_id,
            "family_name": self.family_name,
            "role_names": list(self.role_names),
            "role_columns": list(self.role_columns),
            "substrate_roles": list(self.substrate_roles),
            "condition_roles": list(self.condition_roles),
            "transferable_roles": list(self.transferable_roles),
            "product_role": self.product_role,
            "assembly_order": list(self.assembly_order),
            "canonicalization_version": self.canonicalization_version,
            "reaction_key_schema_version": self.reaction_key_schema_version,
            "partial_key_schema_version": self.partial_key_schema_version,
        }

    def provenance_record(self) -> dict[str, Any]:
        """Return provenance joined to the family schema for run manifests."""
        return {**self.schema_record(), "provenance": self.provenance.to_dict()}

    def with_provenance(self, provenance: DatasetProvenance) -> ReactionFamilyAdapter:
        """Return the same family schema bound to a different dataset provenance."""
        return ReactionFamilyAdapter(
            family_id=self.family_id,
            family_name=self.family_name,
            role_names=self.role_names,
            role_columns=self.role_columns,
            substrate_roles=self.substrate_roles,
            condition_roles=self.condition_roles,
            product_role=self.product_role,
            canonicalization_version=self.canonicalization_version,
            reaction_key_schema_version=self.reaction_key_schema_version,
            partial_key_schema_version=self.partial_key_schema_version,
            assembly_order=self.assembly_order,
            provenance=provenance,
            transferable_roles=self.transferable_roles,
            eligibility_rule=self.eligibility_rule,
        )

    # ------------------------------------------------------------------
    # Row -> roles
    # ------------------------------------------------------------------
    def reaction_smiles(self, values: Mapping[str, str]) -> str:
        """Assemble the family reaction SMILES in the declared assembly order."""
        missing = [role for role in self.role_names if role not in values]
        if missing:
            raise ValueError(f"Reaction SMILES assembly is missing roles: {missing}.")
        left = ".".join(str(values[role]) for role in self.assembly_order)
        return f"{left}>>{values[self.product_role]}"

    def roles_from_row(self, row: Mapping[str, Any]) -> dict[str, str]:
        """Extract and validate every required role value from one source row."""
        missing_columns = [column for column in self.role_columns if column not in row]
        if missing_columns:
            raise ValueError(
                f"Missing {self.family_name} role columns: " + ", ".join(missing_columns)
            )
        values: dict[str, str] = {}
        invalid: list[str] = []
        for role, column in self.role_to_column.items():
            value = row[column]
            if is_missing_molecule(value):
                invalid.append(column)
            else:
                values[role] = str(value).strip()
        if invalid:
            raise ValueError(
                f"Canonical {self.family_name} roles require non-empty molecule "
                "identities; invalid fields: " + ", ".join(invalid)
            )
        return values

    # ------------------------------------------------------------------
    # Canonical identity
    # ------------------------------------------------------------------
    def canonicalize_roles(
        self,
        values: Mapping[str, str],
        *,
        isomeric: bool = True,
    ) -> CanonicalFamilyReaction:
        """Canonicalize all roles and build the family-scoped reaction identity."""
        canonical: dict[str, str] = {}
        invalid: list[str] = []
        for role in self.role_names:
            raw = values.get(role)
            result = canonicalize_smiles(raw if isinstance(raw, str) else str(raw), isomeric=isomeric)
            if not result.parse_valid or result.canonical_smiles is None:
                invalid.append(role)
            else:
                canonical[role] = result.canonical_smiles
        if invalid:
            return CanonicalFamilyReaction(
                family_id=self.family_id,
                canonical_roles=None,
                reaction_key=None,
                reaction_hash=None,
                parse_valid=False,
                invalid_roles=tuple(invalid),
                canonicalization_version=self.canonicalization_version,
                eligibility_reason="chemical_parse_invalid",
            )
        key = self.reaction_key(canonical)
        return CanonicalFamilyReaction(
            family_id=self.family_id,
            canonical_roles=canonical,
            reaction_key=key,
            reaction_hash=sha256_text(key),
            parse_valid=True,
            invalid_roles=(),
            canonicalization_version=self.canonicalization_version,
            eligibility_reason=self.validate_eligibility(canonical),
        )

    def reaction_key(self, canonical_roles: Mapping[str, str]) -> str:
        """Return the stable family-scoped canonical reaction key."""
        missing = [role for role in self.role_names if not canonical_roles.get(role)]
        if missing:
            raise ValueError(f"Canonical reaction key is missing roles: {missing}.")
        payload = {
            "schema": self.reaction_key_schema_version,
            **{role: str(canonical_roles[role]) for role in self.role_names},
        }
        return stable_json(payload)

    def partial_key(
        self,
        canonical_roles: Mapping[str, str],
        roles: Sequence[str],
        key_type: str,
    ) -> str | None:
        """Return a stable partial identity over a subset of canonical roles."""
        values: dict[str, str] = {}
        for role in roles:
            value = canonical_roles.get(role)
            if value is None or not str(value):
                return None
            values[role] = str(value)
        return stable_json(
            {
                "schema": self.partial_key_schema_version,
                "key_type": str(key_type),
                **values,
            }
        )

    def substrate_key(self, canonical_roles: Mapping[str, str]) -> str | None:
        """Return the canonical identity of the substrate block only."""
        return self.partial_key(canonical_roles, self.substrate_roles, "substrate")

    def condition_key(self, canonical_roles: Mapping[str, str]) -> str | None:
        """Return the canonical identity of the condition block only."""
        return self.partial_key(canonical_roles, self.condition_roles, "condition")

    def product_key(self, canonical_roles: Mapping[str, str]) -> str | None:
        """Return the canonical identity of the product role only."""
        return self.partial_key(canonical_roles, (self.product_role,), "product")

    def validate_eligibility(self, canonical_roles: Mapping[str, str]) -> str | None:
        """Apply the family-specific eligibility hook to canonical role values."""
        reason = self.eligibility_rule(dict(canonical_roles))
        if reason is None:
            return None
        text = str(reason).strip()
        if not text:
            raise ValueError("An eligibility rule must return None or a non-empty reason.")
        return text

    # ------------------------------------------------------------------
    # DataFrame level
    # ------------------------------------------------------------------
    def canonicalize_dataframe(
        self,
        df: pd.DataFrame,
        *,
        source_file_hash: str,
        isomeric: bool = True,
        source_row_positions: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Add raw/canonical role columns and family-scoped identities to a frame.

        Mirrors the Buchwald-Hartwig audit contract exactly: immutable row IDs
        derived from the source-file hash and original row position, no rows
        dropped, invalid rows retained and flagged rather than silently removed.
        """
        if not source_file_hash:
            raise ValueError(
                "source_file_hash is required to construct stable source_row_id values."
            )
        missing = [column for column in self.role_columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"{self.family_name} source frame is missing role columns: {missing}."
            )
        enriched = df.copy()
        if "source_row_id" in enriched.columns:
            raise ValueError(
                "Input already contains source_row_id; refusing to replace row identity."
            )
        positions = (
            [int(value) for value in source_row_positions]
            if source_row_positions is not None
            else list(range(len(enriched)))
        )
        if len(positions) != len(enriched):
            raise ValueError("source_row_positions must contain one position per input row.")
        if len(set(positions)) != len(positions) or any(value < 0 for value in positions):
            raise ValueError("source-row positions must be unique non-negative integers.")

        enriched.insert(0, "source_row_id", [
            source_row_id(source_file_hash, position) for position in positions
        ])
        enriched.insert(1, "source_row_position", positions)
        enriched.insert(2, "reaction_family_id", self.family_id)

        identities: list[CanonicalFamilyReaction] = []
        for record in enriched.to_dict(orient="records"):
            try:
                values = self.roles_from_row(record)
            except ValueError:
                identities.append(
                    CanonicalFamilyReaction(
                        family_id=self.family_id,
                        canonical_roles=None,
                        reaction_key=None,
                        reaction_hash=None,
                        parse_valid=False,
                        invalid_roles=tuple(self.role_names),
                        canonicalization_version=self.canonicalization_version,
                        eligibility_reason="missing_required_role",
                    )
                )
                continue
            identities.append(self.canonicalize_roles(values, isomeric=isomeric))

        for role in self.role_names:
            column = self.role_to_column[role]
            enriched[f"raw_{role}_smiles"] = [
                "" if is_missing_molecule(value) else str(value).strip()
                for value in enriched[column].tolist()
            ]
            enriched[f"canonical_{role}_smiles"] = [
                (identity.canonical_roles or {}).get(role) for identity in identities
            ]
            enriched[f"{role}_parse_valid"] = [
                identity.parse_valid or role not in identity.invalid_roles
                for identity in identities
            ]
        enriched["all_required_roles_parse_valid"] = [
            identity.parse_valid for identity in identities
        ]
        enriched["n_invalid_roles"] = [len(identity.invalid_roles) for identity in identities]
        enriched["invalid_role_names"] = [
            stable_json({"roles": list(identity.invalid_roles)}) for identity in identities
        ]
        enriched["canonicalization_version"] = self.canonicalization_version
        enriched["reaction_key_schema_version"] = self.reaction_key_schema_version
        enriched["canonical_reaction_key"] = [identity.reaction_key for identity in identities]
        enriched["canonical_reaction_hash"] = [identity.reaction_hash for identity in identities]
        enriched["canonical_substrate_key"] = [
            self.substrate_key(identity.canonical_roles) if identity.canonical_roles else None
            for identity in identities
        ]
        enriched["canonical_condition_key"] = [
            self.condition_key(identity.canonical_roles) if identity.canonical_roles else None
            for identity in identities
        ]
        enriched["canonical_product_key"] = [
            self.product_key(identity.canonical_roles) if identity.canonical_roles else None
            for identity in identities
        ]
        enriched["family_eligibility_reason"] = [
            identity.eligibility_reason for identity in identities
        ]
        enriched["family_eligible"] = [identity.eligible for identity in identities]
        enriched["canonical_reaction_smiles"] = [
            self.reaction_smiles(identity.canonical_roles) if identity.canonical_roles else None
            for identity in identities
        ]
        return enriched

    def measured_canonical_keys(self, canonical: pd.DataFrame) -> set[str]:
        """Collect measured chemical identities from an already-canonicalized frame.

        Chemical identity comes only from canonical reaction keys. Fingerprint or
        feature-vector equality is never treated as chemical identity.
        """
        if "canonical_reaction_key" not in canonical.columns:
            raise ValueError(
                "measured_canonical_keys requires a canonicalized frame with "
                "canonical_reaction_key."
            )
        return {
            str(value)
            for value in canonical["canonical_reaction_key"].dropna().tolist()
            if str(value)
        }


def is_missing_molecule(value: Any) -> bool:
    """Return whether a role value is missing, using the shared BH sentinel set."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().upper() in MISSING_MOLECULE_TOKENS


# ----------------------------------------------------------------------
# Buchwald-Hartwig: the existing schema, expressed as an adapter
# ----------------------------------------------------------------------
BUCHWALD_HARTWIG_FAMILY_ID = "buchwald_hartwig"

#: Provenance of the Buchwald-Hartwig CSV committed at data/raw/.
TDC_BUCHWALD_HARTWIG_PROVENANCE = DatasetProvenance(
    source_name="Therapeutics Data Commons (TDC), Yields(name='Buchwald-Hartwig')",
    source_url="https://tdcommons.ai/single_pred_tasks/yields",
    license="TDC redistribution terms; verify before use",
    citation=(
        "Ahneman, D. T.; Estrada, J. G.; Lin, S.; Dreher, S. D.; Doyle, A. G. "
        "Predicting reaction performance in C-N cross-coupling using machine "
        "learning. Science 2018, 360, 186-190. Redistributed via Therapeutics "
        "Data Commons (Huang et al., NeurIPS Datasets and Benchmarks 2021)."
    ),
    retrieved_date="2026-06-22",
    raw_file_sha256="f97e4fdc19e1d098946c3e38dc0c920318d94e630f43a1da43f0acf09c58aec5",
)


def buchwald_hartwig_adapter(
    provenance: DatasetProvenance = TDC_BUCHWALD_HARTWIG_PROVENANCE,
) -> ReactionFamilyAdapter:
    """Return the Buchwald-Hartwig adapter, built from the existing constants.

    Nothing here is a re-declaration: the role names, source columns, and schema
    versions are imported from :mod:`bh_augmentation.data.reaction_roles` and
    :mod:`bh_augmentation.data.canonicalize_roles`, and the assembly order is
    derived from the canonical role order exactly as
    :meth:`bh_augmentation.data.reaction_roles.ReactionRoles.reaction_smiles`
    builds it. The adapter is therefore definitionally identical to today's
    behaviour, and the existing Buchwald-Hartwig code path is untouched.
    """
    return ReactionFamilyAdapter(
        family_id=BUCHWALD_HARTWIG_FAMILY_ID,
        family_name="Buchwald-Hartwig amination",
        role_names=CANONICAL_ROLE_NAMES,
        role_columns=CANONICAL_ROLE_COLUMNS,
        substrate_roles=("reactant_1", "reactant_2"),
        condition_roles=("catalyst", "ligand", "base", "solvent_or_additive"),
        product_role="product",
        canonicalization_version=BH_CANONICALIZATION_VERSION,
        reaction_key_schema_version=BH_REACTION_KEY_SCHEMA_VERSION,
        partial_key_schema_version=BH_PARTIAL_KEY_SCHEMA_VERSION,
        assembly_order=tuple(role for role in CANONICAL_ROLE_NAMES if role != "product"),
        provenance=provenance,
        transferable_roles=("catalyst", "ligand", "base", "solvent_or_additive"),
        eligibility_rule=no_family_eligibility_rule,
    )


BUCHWALD_HARTWIG_ADAPTER = buchwald_hartwig_adapter()


# ----------------------------------------------------------------------
# Suzuki-Miyaura
# ----------------------------------------------------------------------
SUZUKI_MIYAURA_FAMILY_ID = "suzuki_miyaura"

SUZUKI_MIYAURA_ROLE_NAMES = (
    "organohalide",
    "organoboron",
    "catalyst",
    "ligand",
    "base",
    "solvent_or_additive",
    "product",
)
SUZUKI_MIYAURA_ROLE_COLUMNS = (
    "smiles_organohalide",
    "smiles_organoboron",
    "smiles_catalyst",
    "smiles_ligand",
    "smiles_base",
    "smiles_solvent_or_additive",
    "smiles_product",
)
SUZUKI_MIYAURA_SUBSTRATE_ROLES = ("organohalide", "organoboron")
SUZUKI_MIYAURA_CONDITION_ROLES = ("catalyst", "ligand", "base", "solvent_or_additive")
SUZUKI_MIYAURA_TRANSFERABLE_ROLES = ("ligand", "base", "solvent_or_additive")
SUZUKI_MIYAURA_CANONICALIZATION_VERSION = "suzuki-rdkit-isomeric-v1"
SUZUKI_MIYAURA_REACTION_KEY_SCHEMA_VERSION = "suzuki-seven-role-v1"
SUZUKI_MIYAURA_PARTIAL_KEY_SCHEMA_VERSION = "suzuki-partial-role-v1"

_HALIDE_LEAVING_GROUP_SMARTS = "[#6][Cl,Br,I]"
_PSEUDOHALIDE_SMARTS = "[#6][OX2][SX4](=O)(=O)"


def suzuki_miyaura_eligibility(canonical_roles: Mapping[str, str]) -> str | None:
    """Reject rows that are not chemically a Suzuki-Miyaura coupling.

    The rules are deliberately narrow and structural, checked with RDKit
    substructure queries rather than string matching:

    * the ``organoboron`` role must actually contain boron (boronic acid, boronic
      ester, trifluoroborate, or MIDA boronate all satisfy this);
    * the ``organohalide`` role must carry a C-Cl/C-Br/C-I bond or a
      carbon-bound sulfonate-ester pseudohalide (OTf/OTs/OMs). C-F is excluded
      because it is not a competent leaving group for Pd(0) oxidative addition
      under normal Suzuki conditions;
    * the ``catalyst`` role must contain palladium or nickel;
    * the ``product`` must be distinct from both substrates, so a row that
      merely restates a starting material is never treated as a coupling.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - RDKit is a scientific requirement
        raise ImportError(
            "RDKit is required for Suzuki-Miyaura eligibility validation. "
            "Install it with `pip install rdkit`."
        ) from exc

    boron = Chem.MolFromSmiles(canonical_roles.get("organoboron", ""))
    if boron is None:
        return "organoboron_unparseable"
    if not any(atom.GetSymbol() == "B" for atom in boron.GetAtoms()):
        return "organoboron_contains_no_boron"

    halide = Chem.MolFromSmiles(canonical_roles.get("organohalide", ""))
    if halide is None:
        return "organohalide_unparseable"
    has_halide = halide.HasSubstructMatch(Chem.MolFromSmarts(_HALIDE_LEAVING_GROUP_SMARTS))
    has_pseudohalide = halide.HasSubstructMatch(Chem.MolFromSmarts(_PSEUDOHALIDE_SMARTS))
    if not (has_halide or has_pseudohalide):
        return "organohalide_has_no_leaving_group"

    catalyst = Chem.MolFromSmiles(canonical_roles.get("catalyst", ""))
    if catalyst is None:
        return "catalyst_unparseable"
    if not any(atom.GetSymbol() in {"Pd", "Ni"} for atom in catalyst.GetAtoms()):
        return "catalyst_is_not_palladium_or_nickel"

    product = canonical_roles.get("product", "")
    if product in {
        canonical_roles.get("organohalide"),
        canonical_roles.get("organoboron"),
    }:
        return "product_identical_to_substrate"
    return None


def suzuki_miyaura_adapter(
    provenance: DatasetProvenance,
    *,
    role_columns: Sequence[str] | None = None,
    transferable_roles: Sequence[str] = SUZUKI_MIYAURA_TRANSFERABLE_ROLES,
) -> ReactionFamilyAdapter:
    """Return the typed Suzuki-Miyaura adapter for one provenanced dataset.

    Role list and why it is not Buchwald-Hartwig terminology
    -------------------------------------------------------
    A Suzuki-Miyaura coupling joins an electrophilic organohalide/pseudohalide to
    a nucleophilic organoboron reagent under Pd(0)/Pd(II) catalysis. Reusing the
    Buchwald-Hartwig vocabulary would be chemically wrong, so the roles are:

    ``organohalide``
        The electrophilic partner consumed by oxidative addition (Ar-I, Ar-Br,
        Ar-Cl, vinyl/heteroaryl halides, and OTf/OTs/OMs pseudohalides). It is
        *not* called ``aryl_halide``: heteroaryl, alkenyl and pseudohalide
        electrophiles are routine, and "halide" alone would exclude triflates.
    ``organoboron``
        The nucleophilic transmetalating partner (boronic acid, pinacol/neopentyl
        boronate, potassium trifluoroborate, or MIDA boronate). There is no
        Buchwald-Hartwig analogue: the corresponding BH partner is an amine
        nitrogen nucleophile, so calling this ``reactant_2`` or ``amine`` would
        assert chemistry that is not happening.
    ``catalyst``
        The Pd (or Ni) source or precatalyst.
    ``ligand``
        The supporting phosphine/NHC ligand controlling oxidative addition and
        reductive elimination.
    ``base``
        The base required to activate the boron reagent toward transmetalation.
    ``solvent_or_additive``
        The reaction medium and any additive reported as part of the medium.
        Public HTE exports usually report a solvent *system* (often aqueous
        co-solvent plus salt) rather than cleanly separable solvent and additive
        fields, so a single role is the honest representation. This mirrors the
        existing Buchwald-Hartwig ``solvent_or_additive`` role for exactly the
        same reason.
    ``product``
        The measured coupled product.

    Why the boron reagent is a substrate, not a condition
    ----------------------------------------------------
    A condition role describes *how* two partners are coupled; a substrate role
    describes *what* is coupled. The organoboron reagent contributes the carbon
    fragment that ends up in the product skeleton, so replacing it changes the
    product constitution and therefore changes which reaction is being measured.
    Typed condition transfer must preserve substrate and product identity, so
    treating boron as transferable would silently fabricate a different reaction
    and attach a donor's yield to it. The boron reagent is mechanistically
    coupled to the base (base activates boron for transmetalation), but that
    coupling is a reason to keep base transfer honest, not a reason to
    reclassify boron.

    ``catalyst`` is a condition role but is excluded from the default
    ``transferable_roles`` because in most Suzuki HTE designs the Pd source is
    held fixed (or is bound to the ligand as a precatalyst) while ligand, base
    and solvent are varied. Callers can widen the set explicitly.
    """
    return ReactionFamilyAdapter(
        family_id=SUZUKI_MIYAURA_FAMILY_ID,
        family_name="Suzuki-Miyaura cross-coupling",
        role_names=SUZUKI_MIYAURA_ROLE_NAMES,
        role_columns=tuple(role_columns or SUZUKI_MIYAURA_ROLE_COLUMNS),
        substrate_roles=SUZUKI_MIYAURA_SUBSTRATE_ROLES,
        condition_roles=SUZUKI_MIYAURA_CONDITION_ROLES,
        product_role="product",
        canonicalization_version=SUZUKI_MIYAURA_CANONICALIZATION_VERSION,
        reaction_key_schema_version=SUZUKI_MIYAURA_REACTION_KEY_SCHEMA_VERSION,
        partial_key_schema_version=SUZUKI_MIYAURA_PARTIAL_KEY_SCHEMA_VERSION,
        assembly_order=tuple(
            role for role in SUZUKI_MIYAURA_ROLE_NAMES if role != "product"
        ),
        provenance=provenance,
        transferable_roles=tuple(transferable_roles),
        eligibility_rule=suzuki_miyaura_eligibility,
    )


REACTION_FAMILY_FACTORIES: dict[str, Callable[..., ReactionFamilyAdapter]] = {
    BUCHWALD_HARTWIG_FAMILY_ID: buchwald_hartwig_adapter,
    SUZUKI_MIYAURA_FAMILY_ID: suzuki_miyaura_adapter,
}


def available_reaction_families() -> tuple[str, ...]:
    """Return every registered reaction-family identifier."""
    return tuple(sorted(REACTION_FAMILY_FACTORIES))


def adapter_for_family(
    family_id: str,
    provenance: DatasetProvenance,
    **kwargs: Any,
) -> ReactionFamilyAdapter:
    """Build the registered adapter for ``family_id`` bound to ``provenance``."""
    factory = REACTION_FAMILY_FACTORIES.get(str(family_id))
    if factory is None:
        raise ValueError(
            f"Unknown reaction family {family_id!r}; registered families: "
            + ", ".join(available_reaction_families())
        )
    return factory(provenance, **kwargs)
