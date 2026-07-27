"""Leakage-safe construction of evaluation-only hidden-condition probes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.role_aware_condition_transfer import (
    build_role_transferred_reaction_smiles,
)
from bh_augmentation.augmentation.synthetic_identity import canonicalize_synthetic_roles
from bh_augmentation.data.canonicalize_roles import (
    PARTIAL_KEY_SCHEMA_VERSION,
    stable_json,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    ReactionRoles,
    reaction_roles_from_row,
    reaction_roles_to_record,
)
from bh_augmentation.features.featurize import morgan_fingerprint

VARIABLE_CONDITION_ROLES = ("ligand", "base", "solvent_or_additive")
SUPPORTED_PROBE_DONOR_STRATEGIES = ("random", "nearest_substrate")
_ROLE_MODE_BY_SET = {
    frozenset(("ligand",)): "ligand_only",
    frozenset(("base",)): "base_only",
    frozenset(("solvent_or_additive",)): "solvent_or_additive_only",
    frozenset(("ligand", "base")): "ligand_base",
    frozenset(("ligand", "solvent_or_additive")): "ligand_solvent_or_additive",
    frozenset(("base", "solvent_or_additive")): "base_solvent_or_additive",
}
_PROBE_COLUMNS = (
    "hidden_row_id",
    "source_row_id",
    "donor_row_id",
    "held_condition_key",
    "requested_roles",
    "role_transfer_mode",
    "donor_strategy",
    "substrate_similarity",
    "canonical_reaction_key",
    "canonical_reaction_hash",
    "hidden_canonical_reaction_key",
    "hidden_canonical_reaction_hash",
    "canonical_identity_exact_hidden_match",
    "source_train_only",
    "donor_train_only",
    "eligible_for_training",
    "probe_purpose",
    *CANONICAL_ROLE_COLUMNS,
    "reaction_smiles",
)
_EXCLUSION_COLUMNS = (
    "hidden_row_id",
    "held_condition_key",
    "requested_roles",
    "donor_strategy",
    "exclusion_reason",
)


@dataclass(frozen=True, slots=True)
class HiddenConditionProbeResult:
    """Frozen construction result with probes and explicit target exclusions."""

    held_condition_key: str
    requested_roles: tuple[str, ...]
    donor_strategy: str
    seed: int
    probes: pd.DataFrame
    exclusions: pd.DataFrame
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _PreparedProbeRecord:
    """One sealed canonical row and its cached substrate fingerprint."""

    source_row_id: str
    canonical_condition_key: str
    canonical_reaction_key: str
    canonical_reaction_hash: str
    roles: ReactionRoles
    substrate_fingerprint_bytes: bytes

    def role_record(self) -> dict[str, str]:
        """Return the canonical row mapping expected by transfer utilities."""
        return {
            **reaction_roles_to_record(self.roles),
            "source_row_id": self.source_row_id,
            "canonical_condition_key": self.canonical_condition_key,
            "canonical_reaction_key": self.canonical_reaction_key,
            "canonical_reaction_hash": self.canonical_reaction_hash,
        }

    def substrate_fingerprint(self) -> np.ndarray:
        """Return a read-only float32 view backed by immutable bytes."""
        return np.frombuffer(self.substrate_fingerprint_bytes, dtype="<f4")


@dataclass(frozen=True, slots=True)
class PreparedHiddenConditionProbeContext:
    """Validated, immutable inputs reusable across role and donor policies."""

    held_condition_key: str
    similarity_n_bits: int
    similarity_radius: int
    train_records: tuple[_PreparedProbeRecord, ...]
    hidden_records: tuple[_PreparedProbeRecord, ...]
    train_source_ids: frozenset[str]


def build_hidden_condition_probes(
    train_frame: pd.DataFrame,
    hidden_identity_frame: pd.DataFrame,
    *,
    requested_roles: Sequence[str],
    donor_strategy: str,
    seed: int,
    similarity_n_bits: int = 256,
    similarity_radius: int = 2,
) -> HiddenConditionProbeResult:
    """Recreate globally held condition rows without exposing hidden outcomes."""
    context = prepare_hidden_condition_probe_context(
        train_frame,
        hidden_identity_frame,
        similarity_n_bits=similarity_n_bits,
        similarity_radius=similarity_radius,
    )
    return build_hidden_condition_probes_from_context(
        context,
        requested_roles=requested_roles,
        donor_strategy=donor_strategy,
        seed=seed,
    )


def prepare_hidden_condition_probe_context(
    train_frame: pd.DataFrame,
    hidden_identity_frame: pd.DataFrame,
    *,
    similarity_n_bits: int = 256,
    similarity_radius: int = 2,
) -> PreparedHiddenConditionProbeContext:
    """Validate and canonicalize frames once for repeated probe construction."""
    train, hidden, held_key = _validate_frames(train_frame, hidden_identity_frame)
    n_bits = _positive_integer(similarity_n_bits, "similarity_n_bits")
    radius = _nonnegative_integer(similarity_radius, "similarity_radius")
    train_records = _canonical_records(train, n_bits=n_bits, radius=radius)
    hidden_records = _canonical_records(hidden, n_bits=n_bits, radius=radius)
    return PreparedHiddenConditionProbeContext(
        held_condition_key=held_key,
        similarity_n_bits=n_bits,
        similarity_radius=radius,
        train_records=train_records,
        hidden_records=hidden_records,
        train_source_ids=frozenset(record.source_row_id for record in train_records),
    )


def build_hidden_condition_probes_from_context(
    context: PreparedHiddenConditionProbeContext,
    *,
    requested_roles: Sequence[str],
    donor_strategy: str,
    seed: int,
) -> HiddenConditionProbeResult:
    """Build one role/strategy probe set from sealed, cached canonical inputs."""
    if not isinstance(context, PreparedHiddenConditionProbeContext):
        raise TypeError(
            "context must be a PreparedHiddenConditionProbeContext returned by "
            "prepare_hidden_condition_probe_context."
        )
    roles = _normalize_requested_roles(requested_roles)
    strategy = str(donor_strategy)
    if strategy not in SUPPORTED_PROBE_DONOR_STRATEGIES:
        raise ValueError(
            f"Unsupported probe donor_strategy {strategy!r}; expected one of "
            f"{SUPPORTED_PROBE_DONOR_STRATEGIES}."
        )
    random_state = _integer(seed, "seed")
    unsupported_reason = _unsupported_role_reason(roles)
    if unsupported_reason is not None:
        exclusions = _unsupported_exclusions(
            context.hidden_records,
            held_key=context.held_condition_key,
            requested_roles=roles,
            donor_strategy=strategy,
            reason=unsupported_reason,
        )
        return _package_result(
            context.held_condition_key,
            roles,
            strategy,
            random_state,
            _empty_probes(),
            exclusions,
    )

    role_mode = _ROLE_MODE_BY_SET[frozenset(roles)]
    probe_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for target in context.hidden_records:
        source_candidates = _eligible_sources(context.train_records, target, roles)
        if not source_candidates:
            exclusion_rows.append(
                _exclusion_record(
                    target,
                    context.held_condition_key,
                    roles,
                    strategy,
                    "no_exact_train_source",
                )
            )
            continue
        source = min(source_candidates, key=_stable_record_key)
        donor_candidates = _eligible_donors(context.train_records, target, roles)
        if not donor_candidates:
            exclusion_rows.append(
                _exclusion_record(
                    target,
                    context.held_condition_key,
                    roles,
                    strategy,
                    "no_train_donor_with_target_requested_roles",
                )
            )
            continue
        donor, substrate_similarity = _choose_donor(
            donor_candidates,
            target,
            strategy=strategy,
            seed=random_state,
            requested_roles=roles,
        )
        generated = build_role_transferred_reaction_smiles(
            source.role_record(),
            donor.role_record(),
            role_mode,
        )
        generated_identity = canonicalize_synthetic_roles(generated["reaction_roles"])
        if (
            not generated_identity.chemical_parse_valid
            or generated_identity.canonical_reaction_key
            != target.canonical_reaction_key
            or generated_identity.canonical_reaction_hash
            != target.canonical_reaction_hash
        ):
            raise ValueError(
                "Hidden-condition probe failed exact canonical target reconstruction "
                f"for hidden_row_id={target.source_row_id!r}."
            )
        generated_roles = generated_identity.roles
        if generated_roles is None:  # pragma: no cover - valid identity invariant
            raise AssertionError("Valid generated identity is missing canonical roles.")
        source_id = source.source_row_id
        donor_id = donor.source_row_id
        if (
            source_id not in context.train_source_ids
            or donor_id not in context.train_source_ids
        ):
            raise ValueError("Hidden-condition source or donor is not train-only.")
        probe_rows.append(
            {
                "hidden_row_id": target.source_row_id,
                "source_row_id": source_id,
                "donor_row_id": donor_id,
                "held_condition_key": context.held_condition_key,
                "requested_roles": "|".join(roles),
                "role_transfer_mode": role_mode,
                "donor_strategy": strategy,
                "substrate_similarity": substrate_similarity,
                "canonical_reaction_key": generated_identity.canonical_reaction_key,
                "canonical_reaction_hash": generated_identity.canonical_reaction_hash,
                "hidden_canonical_reaction_key": target.canonical_reaction_key,
                "hidden_canonical_reaction_hash": target.canonical_reaction_hash,
                "canonical_identity_exact_hidden_match": True,
                "source_train_only": True,
                "donor_train_only": True,
                "eligible_for_training": False,
                "probe_purpose": "hidden_measured_evaluation_only",
                **reaction_roles_to_record(generated_roles),
            }
        )

    probes = pd.DataFrame(probe_rows, columns=_PROBE_COLUMNS)
    exclusions = pd.DataFrame(exclusion_rows, columns=_EXCLUSION_COLUMNS)
    return _package_result(
        context.held_condition_key,
        roles,
        strategy,
        random_state,
        probes,
        exclusions,
    )


def _validate_frames(
    train_frame: pd.DataFrame,
    hidden_identity_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if "yield" in hidden_identity_frame:
        raise ValueError(
            "hidden_identity_frame must not expose hidden yield outcomes to probe construction."
        )
    required = {
        "source_row_id",
        "canonical_condition_key",
        "canonical_reaction_key",
        "canonical_reaction_hash",
        *CANONICAL_ROLE_COLUMNS,
    }
    for name, frame in (
        ("train_frame", train_frame),
        ("hidden_identity_frame", hidden_identity_frame),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing canonical probe fields: {missing}.")
        if frame.empty:
            raise ValueError(f"{name} must be nonempty.")
        ids = frame["source_row_id"]
        if (
            ids.isna().any()
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or ids.duplicated().any()
        ):
            raise ValueError(f"{name} source_row_id values must be nonempty unique strings.")
    held_keys = tuple(hidden_identity_frame["canonical_condition_key"].dropna().unique())
    if len(held_keys) != 1 or not isinstance(held_keys[0], str) or not held_keys[0]:
        raise ValueError(
            "hidden_identity_frame must contain exactly one canonical_condition_key."
        )
    held_key = held_keys[0]
    if train_frame["canonical_condition_key"].eq(held_key).any():
        raise ValueError(
            "Global condition holdout violated: held canonical_condition_key remains "
            "in train_frame."
        )
    train_ids = set(train_frame["source_row_id"])
    hidden_ids = set(hidden_identity_frame["source_row_id"])
    if train_ids & hidden_ids:
        raise ValueError("Hidden-condition rows overlap train_frame source identities.")
    return train_frame.copy(), hidden_identity_frame.copy(), held_key


def _canonical_records(
    frame: pd.DataFrame,
    *,
    n_bits: int,
    radius: int,
) -> tuple[_PreparedProbeRecord, ...]:
    records: list[_PreparedProbeRecord] = []
    for row in frame.to_dict("records"):
        identity = canonicalize_synthetic_roles(reaction_roles_from_row(row))
        if not identity.chemical_parse_valid or identity.roles is None:
            raise ValueError(
                f"Probe row {row['source_row_id']!r} has invalid canonical molecular roles."
            )
        if (
            identity.canonical_reaction_key != row["canonical_reaction_key"]
            or identity.canonical_reaction_hash != row["canonical_reaction_hash"]
        ):
            raise ValueError(
                f"Probe row {row['source_row_id']!r} has inconsistent canonical identity."
            )
        if _canonical_condition_key(identity.roles) != row["canonical_condition_key"]:
            raise ValueError(
                f"Probe row {row['source_row_id']!r} has inconsistent canonical "
                "condition identity."
            )
        records.append(
            _PreparedProbeRecord(
                source_row_id=str(row["source_row_id"]),
                canonical_condition_key=str(row["canonical_condition_key"]),
                canonical_reaction_key=identity.canonical_reaction_key,
                canonical_reaction_hash=identity.canonical_reaction_hash,
                roles=identity.roles,
                substrate_fingerprint_bytes=_substrate_fingerprint(
                    identity.roles,
                    n_bits=n_bits,
                    radius=radius,
                )
                .astype("<f4", copy=False)
                .tobytes(),
            )
        )
    return tuple(sorted(records, key=_stable_record_key))


def _canonical_condition_key(roles: ReactionRoles) -> str:
    return stable_json(
        {
            "schema": PARTIAL_KEY_SCHEMA_VERSION,
            "key_type": "condition",
            "catalyst": roles.catalyst,
            "ligand": roles.ligand,
            "base": roles.base,
            "solvent_or_additive": roles.solvent_or_additive,
        }
    )


def _eligible_sources(
    train_records: Sequence[_PreparedProbeRecord],
    target: _PreparedProbeRecord,
    requested_roles: tuple[str, ...],
) -> list[_PreparedProbeRecord]:
    target_roles = target.roles
    fixed_roles = (
        "reactant_1",
        "reactant_2",
        "catalyst",
        *(
            role
            for role in VARIABLE_CONDITION_ROLES
            if role not in requested_roles
        ),
        "product",
    )
    return [
        record
        for record in train_records
        if all(
            getattr(record.roles, role) == getattr(target_roles, role)
            for role in fixed_roles
        )
        and all(
            getattr(record.roles, role) != getattr(target_roles, role)
            for role in requested_roles
        )
    ]


def _eligible_donors(
    train_records: Sequence[_PreparedProbeRecord],
    target: _PreparedProbeRecord,
    requested_roles: tuple[str, ...],
) -> list[_PreparedProbeRecord]:
    target_roles = target.roles
    return [
        record
        for record in train_records
        if all(
            getattr(record.roles, role) == getattr(target_roles, role)
            for role in requested_roles
        )
    ]


def _choose_donor(
    candidates: Sequence[_PreparedProbeRecord],
    target: _PreparedProbeRecord,
    *,
    strategy: str,
    seed: int,
    requested_roles: tuple[str, ...],
) -> tuple[_PreparedProbeRecord, float]:
    ordered = sorted(candidates, key=_stable_record_key)
    target_fp = target.substrate_fingerprint()
    similarities = np.asarray(
        [
            _cosine_similarity(
                target_fp,
                candidate.substrate_fingerprint(),
            )
            for candidate in ordered
        ],
        dtype=float,
    )
    if strategy == "nearest_substrate":
        position = int(np.argmax(similarities))
    else:
        rng = np.random.default_rng(
            _stable_seed(
                seed,
                target.source_row_id,
                "|".join(requested_roles),
            )
        )
        position = int(rng.integers(0, len(ordered)))
    return ordered[position], float(similarities[position])


def _substrate_fingerprint(
    roles: ReactionRoles,
    *,
    n_bits: int,
    radius: int,
) -> np.ndarray:
    return (
        morgan_fingerprint(
            roles.reactant_1,
            n_bits=n_bits,
            radius=radius,
            warn_invalid=False,
            backend="rdkit",
        )
        + morgan_fingerprint(
            roles.reactant_2,
            n_bits=n_bits,
            radius=radius,
            warn_invalid=False,
            backend="rdkit",
        )
    ).astype(np.float32)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _normalize_requested_roles(requested_roles: Sequence[str]) -> tuple[str, ...]:
    if isinstance(requested_roles, (str, bytes)) or not isinstance(
        requested_roles, Sequence
    ):
        raise ValueError("requested_roles must be a sequence of canonical role names.")
    roles = tuple(requested_roles)
    if not roles or any(not isinstance(role, str) for role in roles):
        raise ValueError("requested_roles must contain canonical role names.")
    if len(roles) != len(set(roles)):
        raise ValueError("requested_roles must not contain duplicates.")
    canonical_order = ("catalyst", *VARIABLE_CONDITION_ROLES)
    unknown = sorted(set(roles) - set(canonical_order))
    if unknown:
        raise ValueError("requested_roles contains unsupported roles: " + ", ".join(unknown))
    selected = set(roles)
    return tuple(role for role in canonical_order if role in selected)


def _unsupported_role_reason(requested_roles: tuple[str, ...]) -> str | None:
    selected = set(requested_roles)
    if selected == {"catalyst", *VARIABLE_CONDITION_ROLES}:
        return "unsupported_full_condition_transfer_global_holdout"
    if "catalyst" in selected:
        return "unsupported_invariant_catalyst"
    if len(selected) == 3:
        return "unsupported_triple_variable_role_transfer_global_holdout"
    return None


def _unsupported_exclusions(
    hidden: Sequence[_PreparedProbeRecord],
    *,
    held_key: str,
    requested_roles: tuple[str, ...],
    donor_strategy: str,
    reason: str,
) -> pd.DataFrame:
    rows = [
        {
            "hidden_row_id": record.source_row_id,
            "held_condition_key": held_key,
            "requested_roles": "|".join(requested_roles),
            "donor_strategy": donor_strategy,
            "exclusion_reason": reason,
        }
        for record in hidden
    ]
    return pd.DataFrame(rows, columns=_EXCLUSION_COLUMNS)


def _exclusion_record(
    target: _PreparedProbeRecord,
    held_key: str,
    requested_roles: tuple[str, ...],
    donor_strategy: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "hidden_row_id": target.source_row_id,
        "held_condition_key": held_key,
        "requested_roles": "|".join(requested_roles),
        "donor_strategy": donor_strategy,
        "exclusion_reason": reason,
    }


def _package_result(
    held_key: str,
    requested_roles: tuple[str, ...],
    donor_strategy: str,
    seed: int,
    probes: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> HiddenConditionProbeResult:
    metadata = MappingProxyType(
        {
            "protocol": "canonical_global_condition_holdout_v1",
            "held_condition_key": held_key,
            "requested_roles": list(requested_roles),
            "donor_strategy": donor_strategy,
            "seed": seed,
            "n_probes": int(len(probes)),
            "n_exclusions": int(len(exclusions)),
            "hidden_outcomes_accessible": False,
            "eligible_for_training": False,
        }
    )
    return HiddenConditionProbeResult(
        held_condition_key=held_key,
        requested_roles=requested_roles,
        donor_strategy=donor_strategy,
        seed=seed,
        probes=probes,
        exclusions=exclusions,
        metadata=metadata,
    )


def _empty_probes() -> pd.DataFrame:
    return pd.DataFrame(columns=_PROBE_COLUMNS)


def _stable_record_key(record: _PreparedProbeRecord) -> tuple[str, str]:
    return record.canonical_reaction_key, record.source_row_id


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result
