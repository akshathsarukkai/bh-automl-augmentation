"""Canonical chemical identities and collision audits for synthetic candidates."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    OBSERVED_ONLY_LOW_DATA,
    CandidateScopePolicy,
    CandidateScopeViolation,
    legacy_scope_from_measured_identity_keys,
)
from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    build_canonical_reaction_identity,
    canonicalize_smiles,
    stable_json,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ReactionRoles,
    reaction_roles_from_row,
    reaction_roles_to_record,
)

FEATURE_HASH_SCHEMA_VERSION = "configured-feature-vector-v1"
#: Bumped to v2 when the candidate audit gained its explicit candidate-scope
#: columns. The eligibility *decisions* are unchanged for any pre-existing
#: configuration -- a legacy `measured_identity_keys` call still reports
#: rejection_reason="already_measured" against the same key set -- but the audit
#: frame now carries seven additional columns naming which rule produced it.
#: Recording that as a schema version is deliberate: a candidate audit that
#: cannot say which eligibility rule it was produced under is exactly what let
#: two different scientific questions share one gate.
SYNTHETIC_IDENTITY_AUDIT_VERSION = "canonical-synthetic-identity-v2"
ROLE_CHANGE_REQUIREMENTS = {"all", "any"}

REQUIRED_SYNTHETIC_AUDIT_FIELDS = (
    "source_row_id",
    "donor_row_id",
    "canonical_reaction_key",
    "canonical_reaction_hash",
    "feature_hash",
    "source_identical",
    "already_measured",
    "duplicate_synthetic",
    "feature_duplicate_synthetic",
    "chemical_parse_valid",
    "rejection_reason",
)
REQUIRED_CANDIDATE_SCOPE_AUDIT_FIELDS = (
    "candidate_scope_mode",
    "candidate_scope_provenance",
    "candidate_scope_hash",
    "observed_in_labeled_train",
    "globally_measured_outside_labeled_train",
    "quarantined_held_out_identity",
    "quarantine_role",
)
REQUIRED_SYNTHETIC_SUPPORT_FIELDS = (
    "substrate_similarity",
    "product_similarity",
    "nontransferred_role_similarity",
    "condition_similarity",
    "overall_similarity",
    "nearest_training_support_distance",
    "support_distance_metric",
    "support_distance_backend",
)
REQUIRED_SYNTHETIC_RANKING_FIELDS = (
    "calibrated_uncertainty",
    "uncertainty_rank_value",
    "uncertainty_rank_basis",
    "relevant_context_similarity",
    "diversity_contribution",
    "out_of_support_distance",
    "candidate_rank",
)


@dataclass(frozen=True)
class CanonicalSyntheticReaction:
    """Canonical form and stable identity of one seven-role candidate."""

    roles: ReactionRoles | None
    canonical_reaction_key: str | None
    canonical_reaction_hash: str | None
    chemical_parse_valid: bool
    invalid_roles: tuple[str, ...]
    canonicalization_version: str = CANONICALIZATION_VERSION


@dataclass
class CandidateIdentitySets:
    """Distinct identity sets; feature collisions never imply chemical equality."""

    measured_canonical_keys: set[str] = field(default_factory=set)
    generated_canonical_keys: set[str] = field(default_factory=set)
    generated_feature_hashes: set[str] = field(default_factory=set)


def canonicalize_synthetic_roles(roles: ReactionRoles) -> CanonicalSyntheticReaction:
    """Canonicalize all roles without changing charge, isotope, or stereochemistry."""
    canonical_values: dict[str, str] = {}
    invalid_roles: list[str] = []
    for role in CANONICAL_ROLE_NAMES:
        result = canonicalize_smiles(getattr(roles, role), isomeric=True)
        if not result.parse_valid or result.canonical_smiles is None:
            invalid_roles.append(role)
        else:
            canonical_values[role] = result.canonical_smiles
    if invalid_roles:
        return CanonicalSyntheticReaction(
            roles=None,
            canonical_reaction_key=None,
            canonical_reaction_hash=None,
            chemical_parse_valid=False,
            invalid_roles=tuple(invalid_roles),
        )

    canonical_roles = ReactionRoles(**canonical_values)
    identity_row = {
        f"canonical_{role}_smiles": getattr(canonical_roles, role)
        for role in CANONICAL_ROLE_NAMES
    }
    identity_row["all_required_roles_parse_valid"] = True
    identity = build_canonical_reaction_identity(identity_row)
    return CanonicalSyntheticReaction(
        roles=canonical_roles,
        canonical_reaction_key=identity["key"],
        canonical_reaction_hash=identity["hash"],
        chemical_parse_valid=True,
        invalid_roles=(),
    )


def canonical_candidate_record(roles: ReactionRoles) -> dict[str, Any]:
    """Return canonical candidate fields while retaining invalid candidates for audit."""
    identity = canonicalize_synthetic_roles(roles)
    canonical_roles = identity.roles or roles
    record: dict[str, Any] = {
        "reaction_roles": canonical_roles,
        "canonical_reaction_roles": identity.roles,
        "canonical_reaction_key": identity.canonical_reaction_key,
        "canonical_reaction_hash": identity.canonical_reaction_hash,
        "chemical_parse_valid": identity.chemical_parse_valid,
        "invalid_chemical_roles": stable_json({"roles": list(identity.invalid_roles)}),
        "canonicalization_version": identity.canonicalization_version,
        **reaction_roles_to_record(canonical_roles),
    }
    for role in CANONICAL_ROLE_NAMES:
        record[f"canonical_{role}_smiles"] = (
            getattr(identity.roles, role) if identity.roles is not None else None
        )
    return record


def configured_feature_hash(feature_vector: np.ndarray | Sequence[float]) -> str | None:
    """Hash the exact configured finite feature vector in a platform-stable encoding."""
    vector = np.asarray(feature_vector)
    if vector.ndim != 1:
        raise ValueError("configured feature vectors must be one-dimensional.")
    if not np.isfinite(vector).all():
        return None
    normalized = np.ascontiguousarray(vector, dtype="<f4")
    header = stable_json(
        {
            "schema": FEATURE_HASH_SCHEMA_VERSION,
            "dtype": "float32-little-endian",
            "shape": [int(value) for value in normalized.shape],
        }
    )
    digest = hashlib.sha256()
    digest.update(header.encode("utf-8"))
    digest.update(b"\0")
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def measured_canonical_keys(
    measured: pd.DataFrame,
    *,
    additional_keys: Iterable[str] = (),
) -> set[str]:
    """Build measured chemical identities without using feature-vector equality."""
    keys = {str(value) for value in additional_keys if isinstance(value, str) and value}
    if "canonical_reaction_key" in measured:
        keys.update(
            str(value)
            for value in measured["canonical_reaction_key"].dropna()
            if str(value)
        )
    for row in measured.to_dict(orient="records"):
        if isinstance(row.get("canonical_reaction_key"), str) and row["canonical_reaction_key"]:
            continue
        roles = reaction_roles_from_row(row)
        identity = canonicalize_synthetic_roles(roles)
        if not identity.chemical_parse_valid or identity.canonical_reaction_key is None:
            raise ValueError(
                "Measured data contain a chemically invalid canonical role record; "
                f"invalid_roles={identity.invalid_roles}."
            )
        keys.add(identity.canonical_reaction_key)
    return keys


def assert_stored_identities_match_roles(
    frame: pd.DataFrame,
    *,
    description: str,
    sample: int | None = None,
) -> None:
    """Fail if a frame's stored canonical keys disagree with live canonicalization.

    Candidate identities are computed live from roles, while a partition's
    identities are usually read from the stored ``canonical_reaction_key``
    column.  If the installed RDKit canonicalizes differently from the version
    that built the dataset, the two vocabularies silently stop intersecting --
    and an eligibility gate that compares them would then accept *every*
    candidate, including ones identical to reactions the learner has observed.
    That failure is invisible in the metrics, so it must be loud here.

    ``sample`` bounds the work for large frames; the default checks every row,
    which is what a low-data labeled subset can afford.
    """
    if frame.empty or "canonical_reaction_key" not in frame:
        return
    rows = frame if sample is None else frame.head(int(sample))
    for row in rows.to_dict(orient="records"):
        stored = row.get("canonical_reaction_key")
        if not isinstance(stored, str) or not stored:
            continue
        identity = canonicalize_synthetic_roles(reaction_roles_from_row(row))
        if identity.canonical_reaction_key == stored:
            continue
        raise CandidateScopeViolation(
            f"{description}: stored canonical identities disagree with live "
            "canonicalization, so candidate eligibility would compare two "
            "different identity vocabularies and accept everything. This "
            "usually means the installed RDKit differs from the version that "
            "built the dataset (see the dependency_versions block of the "
            "canonical data audit manifest). Offending source_row_id="
            f"{row.get('source_row_id')!r}."
        )


def audit_candidate_identities(
    candidate_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    source_rows: Sequence[Mapping[str, Any]],
    measured_keys: Iterable[str] | None = None,
    candidate_scope: CandidateScopePolicy | None = None,
) -> pd.DataFrame:
    """Audit candidates sequentially using separate chemical and feature sets.

    Exactly one eligibility rule applies per call.  ``candidate_scope`` is the
    explicit typed policy; ``measured_keys`` is the historical keyword, which is
    reconstructed into an equivalent policy carrying legacy provenance so that
    pre-audit configurations reproduce their candidate audits unchanged.
    """
    scope = _resolve_audit_scope(measured_keys, candidate_scope)
    result = candidate_df.copy()
    features = np.asarray(feature_matrix)
    if len(result) != len(features):
        raise ValueError("candidate_df and feature_matrix must contain the same row count.")
    rejection_keys = scope.rejection_identity_keys()
    quarantine_keys = scope.quarantined_identity_keys()
    observed_keys = frozenset(scope.observed_identity_keys)
    global_keys = frozenset(scope.global_identity_keys)
    identity_sets = CandidateIdentitySets(measured_canonical_keys=set(rejection_keys))
    source_identities = [
        canonicalize_synthetic_roles(reaction_roles_from_row(row))
        for row in source_rows
    ]

    audit_records: list[dict[str, Any]] = []
    for position, (_, candidate) in enumerate(result.iterrows()):
        source_position = int(candidate["source_position"])
        donor_position = int(candidate["donor_position"])
        if source_position < 0 or source_position >= len(source_rows):
            raise ValueError(f"Invalid source_position {source_position} in candidate audit.")
        if donor_position < 0 or donor_position >= len(source_rows):
            raise ValueError(f"Invalid donor_position {donor_position} in candidate audit.")

        candidate_identity = _candidate_identity(candidate)
        source_identity = source_identities[source_position]
        key = candidate_identity.canonical_reaction_key
        feature_hash = configured_feature_hash(features[position])
        source_identical = bool(
            key is not None and key == source_identity.canonical_reaction_key
        )
        already_measured = bool(key is not None and key in rejection_keys)
        observed_in_labeled_train = bool(key is not None and key in observed_keys)
        globally_measured_outside = (
            bool(key is not None and key in global_keys and key not in observed_keys)
            if scope.consults_complete_dataset
            else pd.NA
        )
        quarantined = bool(
            key is not None and not already_measured and key in quarantine_keys
        )
        duplicate_synthetic = bool(
            key is not None and key in identity_sets.generated_canonical_keys
        )
        feature_duplicate = bool(
            feature_hash is not None
            and feature_hash in identity_sets.generated_feature_hashes
        )
        rejection_reason = _identity_rejection_reason(
            chemical_parse_valid=candidate_identity.chemical_parse_valid,
            feature_hash=feature_hash,
            source_identical=source_identical,
            scope_rejection_reason=scope.rejection_reason_for(key),
            duplicate_synthetic=duplicate_synthetic,
            feature_duplicate_synthetic=feature_duplicate,
        )
        if rejection_reason is None:
            if key is None or feature_hash is None:  # pragma: no cover - defensive
                raise AssertionError("Accepted candidate is missing a required identity.")
            identity_sets.generated_canonical_keys.add(key)
            identity_sets.generated_feature_hashes.add(feature_hash)

        source_row = source_rows[source_position]
        donor_row = source_rows[donor_position]
        audit_records.append(
            {
                "source_row_id": _parent_row_id(source_row, candidate.get("source_index")),
                "donor_row_id": _parent_row_id(donor_row, candidate.get("donor_index")),
                "canonical_reaction_key": key,
                "canonical_reaction_hash": candidate_identity.canonical_reaction_hash,
                "feature_hash": feature_hash,
                "source_identical": source_identical,
                "already_measured": already_measured,
                "candidate_scope_mode": scope.mode,
                "candidate_scope_provenance": scope.provenance,
                "candidate_scope_hash": scope.scope_hash,
                "observed_in_labeled_train": observed_in_labeled_train,
                "globally_measured_outside_labeled_train": globally_measured_outside,
                "quarantined_held_out_identity": quarantined,
                "quarantine_role": scope.quarantine_role if quarantined else None,
                "duplicate_synthetic": duplicate_synthetic,
                "feature_duplicate_synthetic": feature_duplicate,
                "chemical_parse_valid": candidate_identity.chemical_parse_valid,
                "rejection_reason": rejection_reason,
                "synthetic_identity_audit_version": SYNTHETIC_IDENTITY_AUDIT_VERSION,
            }
        )

    audit = pd.DataFrame(audit_records, index=result.index)
    for column in audit:
        result[column] = audit[column]
    return result


def apply_filter_rejection(
    candidate_df: pd.DataFrame,
    rejected: np.ndarray | pd.Series,
    reason: str,
) -> pd.DataFrame:
    """Record a later filter reason only when no earlier rejection already applies."""
    result = candidate_df.copy()
    mask = np.asarray(rejected, dtype=bool)
    if len(mask) != len(result):
        raise ValueError("rejection mask must contain one value per candidate.")
    unclassified = result["rejection_reason"].isna().to_numpy(dtype=bool)
    result.loc[mask & unclassified, "rejection_reason"] = str(reason)
    return result


def audit_role_changes(
    candidate_df: pd.DataFrame,
    *,
    source_rows: Sequence[Mapping[str, Any]],
    requested_roles: Sequence[str],
    role_change_requirement: str,
) -> pd.DataFrame:
    """Compare canonical roles and enforce an exact requested change contract."""
    requested = tuple(str(role) for role in requested_roles)
    unknown = sorted(set(requested) - set(CANONICAL_ROLE_NAMES))
    if not requested or unknown:
        raise ValueError(
            "requested_roles must be a non-empty subset of canonical roles"
            + (f"; unknown={unknown}" if unknown else ".")
        )
    if len(set(requested)) != len(requested):
        raise ValueError("requested_roles must not contain duplicates.")
    if role_change_requirement not in ROLE_CHANGE_REQUIREMENTS:
        raise ValueError(
            "role_change_requirement must be one of: "
            + ", ".join(sorted(ROLE_CHANGE_REQUIREMENTS))
        )

    result = candidate_df.copy()
    source_identities = [
        canonicalize_synthetic_roles(reaction_roles_from_row(row))
        for row in source_rows
    ]
    records: list[dict[str, Any]] = []
    invalid_contract = np.zeros(len(result), dtype=bool)
    unexpected_contract = np.zeros(len(result), dtype=bool)
    requested_set = set(requested)
    for output_position, (_, candidate) in enumerate(result.iterrows()):
        source_position = int(candidate["source_position"])
        source_identity = source_identities[source_position]
        candidate_identity = _candidate_identity(candidate)
        if source_identity.roles is None:
            raise ValueError(
                f"Source row {source_position} has no valid canonical role identity."
            )

        changed = []
        if candidate_identity.roles is not None:
            changed = [
                role
                for role in CANONICAL_ROLE_NAMES
                if getattr(candidate_identity.roles, role)
                != getattr(source_identity.roles, role)
            ]
        actual_set = set(changed)
        unchanged_requested = [
            role for role in requested if role not in actual_set
        ]
        unexpected = [
            role for role in CANONICAL_ROLE_NAMES
            if role in actual_set and role not in requested_set
        ]
        requested_change_valid = (
            not unchanged_requested
            if role_change_requirement == "all"
            else bool(actual_set & requested_set)
        )
        valid = bool(
            candidate_identity.chemical_parse_valid
            and requested_change_valid
            and not unexpected
        )
        invalid_contract[output_position] = not valid
        unexpected_contract[output_position] = bool(unexpected)
        records.append(
            {
                "requested_roles": "|".join(requested),
                "actual_changed_roles": "|".join(changed),
                "unchanged_requested_roles": "|".join(unchanged_requested),
                "unexpected_changed_roles": "|".join(unexpected),
                "change_mask": "".join(
                    "1" if role in actual_set else "0"
                    for role in CANONICAL_ROLE_NAMES
                ),
                "role_change_requirement": role_change_requirement,
                "role_change_valid": valid,
            }
        )

    audit = pd.DataFrame(records, index=result.index)
    for column in audit:
        result[column] = audit[column]
    result = apply_filter_rejection(
        result,
        unexpected_contract,
        "unexpected_role_change",
    )
    result = apply_filter_rejection(
        result,
        invalid_contract,
        "role_change_requirement_not_met",
    )
    return result


def assert_accepted_role_change_invariants(candidate_df: pd.DataFrame) -> None:
    """Hard-fail if a final accepted candidate violates its declared role change."""
    required = {
        "requested_roles",
        "actual_changed_roles",
        "unchanged_requested_roles",
        "unexpected_changed_roles",
        "change_mask",
        "role_change_requirement",
        "role_change_valid",
    }
    missing = sorted(required - set(candidate_df))
    if missing:
        raise ValueError("Role-change audit is missing fields: " + ", ".join(missing))
    accepted = candidate_df.loc[candidate_df["accepted"].astype(bool)]
    if accepted.empty:
        return
    if not accepted["role_change_valid"].astype(bool).all():
        raise AssertionError("An accepted candidate violates its role-change contract.")
    if accepted["unexpected_changed_roles"].fillna("").astype(str).str.len().gt(0).any():
        raise AssertionError("An accepted candidate changes an unrequested role.")
    strict = accepted["role_change_requirement"].eq("all")
    if (
        accepted.loc[strict, "unchanged_requested_roles"]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .any()
    ):
        raise AssertionError("An accepted strict candidate leaves a requested role unchanged.")


def assert_accepted_identity_invariants(candidate_df: pd.DataFrame) -> None:
    """Hard-fail if final accepted candidates violate the canonical identity gate."""
    accepted = candidate_df.loc[candidate_df["accepted"].astype(bool)]
    missing = [field for field in REQUIRED_SYNTHETIC_AUDIT_FIELDS if field not in candidate_df]
    if missing:
        raise ValueError("Synthetic candidate audit is missing fields: " + ", ".join(missing))
    if accepted.empty:
        return
    if not accepted["chemical_parse_valid"].astype(bool).all():
        raise AssertionError("An accepted synthetic candidate has invalid chemistry.")
    if accepted["canonical_reaction_key"].isna().any():
        raise AssertionError("An accepted synthetic candidate lacks a canonical key.")
    if accepted["canonical_reaction_hash"].isna().any():
        raise AssertionError("An accepted synthetic candidate lacks a canonical hash.")
    if accepted["feature_hash"].isna().any():
        raise AssertionError("An accepted synthetic candidate lacks a feature hash.")
    if accepted["already_measured"].astype(bool).any():
        raise AssertionError("An accepted synthetic candidate duplicates measured chemistry.")
    _assert_accepted_scope_invariants(accepted)
    if accepted["canonical_reaction_key"].duplicated().any():
        raise AssertionError("An accepted synthetic canonical key appears more than once.")
    if accepted["feature_hash"].duplicated().any():
        raise AssertionError("An accepted synthetic feature hash appears more than once.")


def _assert_accepted_scope_invariants(accepted: pd.DataFrame) -> None:
    """Hard-fail if an accepted candidate breaks its declared eligibility scope."""
    if "candidate_scope_mode" not in accepted:
        return
    modes = set(accepted["candidate_scope_mode"].dropna().astype(str))
    if len(modes) > 1:
        raise AssertionError(
            "One candidate pool mixes candidate-scope modes: " + ", ".join(sorted(modes))
        )
    if accepted["observed_in_labeled_train"].astype(bool).any():
        raise AssertionError(
            "An accepted synthetic candidate repeats chemistry the simulated "
            "low-data learner already observed."
        )
    if accepted["quarantined_held_out_identity"].astype(bool).any():
        raise AssertionError(
            "A quarantined held-out identity was accepted into a training pool."
        )
    global_flags = accepted["globally_measured_outside_labeled_train"]
    if modes == {OBSERVED_ONLY_LOW_DATA}:
        if not global_flags.isna().all():
            raise AssertionError(
                "An observed-only low-data candidate audit recorded complete-dataset "
                "membership, which the protocol forbids consulting at generation time."
            )
    elif global_flags.fillna(False).astype(bool).any():
        raise AssertionError(
            "An accepted prospective candidate duplicates globally measured chemistry."
        )


def _candidate_identity(candidate: pd.Series) -> CanonicalSyntheticReaction:
    roles = candidate.get("reaction_roles")
    if not isinstance(roles, ReactionRoles):
        roles = reaction_roles_from_row(candidate)
    return canonicalize_synthetic_roles(roles)


def _resolve_audit_scope(
    measured_keys: Iterable[str] | None,
    candidate_scope: CandidateScopePolicy | None,
) -> CandidateScopePolicy:
    """Return exactly one eligibility policy for a candidate audit."""
    if candidate_scope is not None and measured_keys is not None:
        raise CandidateScopeViolation(
            "Pass either candidate_scope or measured_keys to audit_candidate_identities, "
            "not both: two eligibility rules cannot apply to one candidate audit."
        )
    if candidate_scope is not None:
        if not isinstance(candidate_scope, CandidateScopePolicy):
            raise CandidateScopeViolation(
                "candidate_scope must be a CandidateScopePolicy instance."
            )
        return candidate_scope
    if measured_keys is None:
        raise CandidateScopeViolation(
            "audit_candidate_identities requires candidate_scope or measured_keys."
        )
    return legacy_scope_from_measured_identity_keys(
        labeled_train_identity_keys=[str(value) for value in measured_keys],
        measured_identity_keys=(),
    )


def _identity_rejection_reason(
    *,
    chemical_parse_valid: bool,
    feature_hash: str | None,
    source_identical: bool,
    scope_rejection_reason: str | None,
    duplicate_synthetic: bool,
    feature_duplicate_synthetic: bool,
) -> str | None:
    if not chemical_parse_valid:
        return "chemical_parse_invalid"
    if feature_hash is None:
        return "invalid_feature_vector"
    if source_identical:
        return "source_identical"
    if scope_rejection_reason is not None:
        return scope_rejection_reason
    if duplicate_synthetic:
        return "duplicate_synthetic"
    if feature_duplicate_synthetic:
        return "feature_duplicate_synthetic"
    return None


def _parent_row_id(row: Mapping[str, Any], fallback: Any) -> str:
    value = row.get("source_row_id")
    if isinstance(value, str) and value:
        return value
    return str(fallback)
