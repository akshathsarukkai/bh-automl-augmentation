"""Frozen train-only chemical augmentation controls for matched comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    LEGACY_PROVENANCE,
    CandidateScopePolicy,
)
from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    clear_role_aware_teacher_cache,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.features.compatibility import FeatureMetadata

_VARIABLE_ANONYMOUS_ROLES = ("ligand", "base", "solvent_or_additive")
_TYPED_ROLES = ("ligand", "base")
_TEACHER_MODELS = ("ridge", "random_forest")
_POOL_HASH_SCHEMA = "phase11-chemical-prefilter-pool-v1"
_UNCALIBRATED_UNCERTAINTY = "raw_teacher_std_proxy_phase12_pending"


@dataclass(frozen=True, slots=True)
class ChemicalControlSpec:
    """One immutable, scientifically named chemical control."""

    control_number: int
    control_id: str
    control_name: str
    generator_family: str
    requested_roles: tuple[str, ...]
    role_transfer_mode: str | None
    donor_strategy: str
    label_strategy: str
    context_definition: str
    uncertainty_filter_enabled: bool
    role_change_requirement: str = "all"
    fallback_policy: str = "reject"
    donor_similarity_backend: str = "rdkit"


CHEMICAL_CONTROL_SPECS = (
    ChemicalControlSpec(
        9,
        "anonymous_condition_transfer",
        "Anonymous condition transfer",
        "anonymous",
        _VARIABLE_ANONYMOUS_ROLES,
        None,
        "random",
        "teacher_ensemble",
        "no_typed_context_requirement",
        False,
    ),
    ChemicalControlSpec(
        10,
        "random_typed_transfer",
        "Random typed transfer",
        "role_aware",
        _TYPED_ROLES,
        "ligand_base",
        "random",
        "average_source_donor_label",
        "random_role_valid_donor",
        False,
    ),
    ChemicalControlSpec(
        11,
        "strict_context_matched_typed_transfer",
        "Strict exact-substrate context-matched typed transfer",
        "role_aware",
        _TYPED_ROLES,
        "ligand_base",
        "same_substrate_different_role",
        "average_source_donor_label",
        "exact_reactant_key",
        False,
    ),
    ChemicalControlSpec(
        12,
        "typed_transfer_without_uncertainty_filtering",
        "Typed transfer without uncertainty filtering",
        "role_aware",
        _TYPED_ROLES,
        "ligand_base",
        "same_substrate_different_role",
        "teacher_ensemble",
        "exact_reactant_key",
        False,
    ),
    ChemicalControlSpec(
        13,
        "typed_transfer_with_uncertainty_filtering",
        "Typed transfer with raw teacher-std filtering",
        "role_aware",
        _TYPED_ROLES,
        "ligand_base",
        "same_substrate_different_role",
        "uncertainty_filtered_teacher",
        "exact_reactant_key",
        True,
    ),
)
CHEMICAL_CONTROL_IDS = tuple(spec.control_id for spec in CHEMICAL_CONTROL_SPECS)
_SPEC_BY_ID = MappingProxyType({spec.control_id: spec for spec in CHEMICAL_CONTROL_SPECS})


@dataclass(frozen=True, slots=True)
class ChemicalAugmentationControlResult:
    """One frozen, fully audited train-only chemical augmentation result."""

    control_id: str
    control_name: str
    spec: ChemicalControlSpec
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray | None
    row_audit: pd.DataFrame
    candidate_audit: pd.DataFrame
    requested_added_count: int
    effective_added_count: int
    total_sample_weight: float
    chemical_identity_applicable: bool
    nonchemical_semantics: str | None
    budget_underfill_count: int
    budget_underfill_reason: str | None
    generator_metadata: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    prefilter_pool_hash: str | None
    accepted_candidate_hash: str
    uncertainty_semantics: str | None
    uncertainty_accepted_keys_subset: bool | None

    @property
    def nominal_added_budget(self) -> int:
        """Return the requested matched augmentation budget."""
        return self.requested_added_count


def build_chemical_augmentation_control(
    control_id: str,
    *,
    train_frame: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_config: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_metadata: FeatureMetadata,
    measured_identity_keys: Iterable[str] = (),
    candidate_scope: CandidateScopePolicy | None = None,
    nominal_added_budget: int,
    seed: int,
    max_teacher_std: float | None = None,
    teacher_models: Sequence[str] = _TEACHER_MODELS,
) -> ChemicalAugmentationControlResult:
    """Build one frozen chemical control from explicit measured training rows."""
    spec = _resolve_spec(control_id)
    frame, X, y, source_ids = _normalize_training_inputs(
        train_frame,
        X_train,
        y_train,
        feature_names,
        feature_metadata,
    )
    budget = _nonnegative_integer(nominal_added_budget, "nominal_added_budget")
    random_state = _integer(seed, "seed")
    resolved_teachers = _teacher_models(teacher_models)
    # Forward exactly one eligibility rule to the generator, which folds in the
    # training frame's own identities and resolves the policy. Resolving it here
    # would duplicate that work against a frame that has not yet been role-
    # normalized by the generator.
    scope_kwargs: dict[str, Any] = (
        {"candidate_scope": candidate_scope}
        if candidate_scope is not None
        else {"measured_identity_keys": tuple(measured_identity_keys)}
    )
    threshold = _uncertainty_threshold(spec, max_teacher_std)
    multiplier = _budget_multiplier(budget, len(frame))
    source_cap = max(3, math.ceil(budget / len(frame)) * 3)

    if spec.generator_family == "anonymous":
        generator_config: ConditionTransferConfig | RoleAwareConditionTransferConfig = (
            ConditionTransferConfig(
                donor_strategy=spec.donor_strategy,
                synthetic_multiplier=multiplier,
                n_neighbors=max(1, len(frame) - 1),
                label_strategy=spec.label_strategy,
                teacher_models=list(resolved_teachers),
                max_teacher_std=None,
                min_similarity=None,
                high_yield_threshold=70.0,
                clip_y_min=0.0,
                clip_y_max=100.0,
                candidates_per_real=source_cap,
                random_state=random_state,
                donor_similarity_n_bits=256,
                donor_similarity_radius=2,
                donor_similarity_backend=spec.donor_similarity_backend,
                role_change_requirement=spec.role_change_requirement,
                fallback_policy=spec.fallback_policy,
                max_candidates_per_source=source_cap,
                requested_roles=spec.requested_roles,
            )
        )
        generated = generate_condition_transfer_examples(
            frame,
            X,
            y,
            generator_config,
            feature_config=dict(feature_config),
            real_feature_names=list(feature_names),
            real_feature_metadata=feature_metadata,
            **scope_kwargs,
        )
        reference_unfiltered = None
    else:
        generator_config = _role_aware_config(
            spec,
            multiplier=multiplier,
            source_cap=source_cap,
            seed=random_state,
            max_teacher_std=threshold,
            teacher_models=resolved_teachers,
        )
        clear_role_aware_teacher_cache()
        if spec.control_number == 13:
            reference_unfiltered = generate_role_aware_condition_transfer_examples(
                frame,
                X,
                y,
                _role_aware_config(
                    _SPEC_BY_ID["typed_transfer_without_uncertainty_filtering"],
                    multiplier=multiplier,
                    source_cap=source_cap,
                    seed=random_state,
                    max_teacher_std=None,
                    teacher_models=resolved_teachers,
                ),
                feature_config=dict(feature_config),
                real_feature_names=list(feature_names),
                real_feature_metadata=feature_metadata,
                **scope_kwargs,
            )
        else:
            reference_unfiltered = None
        generated = generate_role_aware_condition_transfer_examples(
            frame,
            X,
            y,
            generator_config,
            feature_config=dict(feature_config),
            real_feature_names=list(feature_names),
            real_feature_metadata=feature_metadata,
            **scope_kwargs,
        )

    candidate_audit = generated["candidate_df"].copy()
    _assert_train_only_parents(candidate_audit, source_ids)
    effective_count = int(len(generated["synthetic_y"]))
    if effective_count > budget:
        raise ValueError(
            "Chemical generator exceeded the frozen nominal added budget: "
            f"requested={budget}, generated={effective_count}."
        )
    if len(generated["X_synthetic"]) != effective_count:
        raise ValueError("Chemical generator returned inconsistent synthetic row counts.")

    prefilter_pool_hash = None
    subset_valid: bool | None = None
    if spec.control_number in {12, 13}:
        prefilter_pool_hash = _candidate_pool_hash(candidate_audit)
    if spec.control_number == 13:
        if reference_unfiltered is None:  # pragma: no cover - construction invariant
            raise AssertionError("Filtered control is missing its unfiltered reference pool.")
        reference_audit = reference_unfiltered["candidate_df"].copy()
        _assert_train_only_parents(reference_audit, source_ids)
        reference_hash = _candidate_pool_hash(reference_audit)
        if prefilter_pool_hash != reference_hash:
            raise ValueError(
                "Filtered and unfiltered typed controls produced different prefilter pools."
            )
        filtered_accepted = _accepted_candidate_identities(candidate_audit)
        unfiltered_accepted = _accepted_candidate_identities(reference_audit)
        subset_valid = filtered_accepted.issubset(unfiltered_accepted)
        if not subset_valid:
            raise ValueError(
                "Uncertainty-filtered accepted candidates are not a subset of the "
                "common unfiltered candidate pool."
            )

    X_synthetic = np.asarray(generated["X_synthetic"], dtype=np.float32)
    y_synthetic = np.asarray(generated["synthetic_y"], dtype=np.float32).reshape(-1)
    kept_indices = _ordered_kept_indices(candidate_audit)
    if len(kept_indices) != effective_count:
        raise ValueError(
            "Chemical candidate kept order disagrees with synthetic output count."
        )
    if effective_count:
        candidate_audit.loc[kept_indices, "synthetic_label"] = (
            y_synthetic.astype(float)
        )
    X_out = np.vstack([X, X_synthetic]).astype(np.float32)
    y_out = np.concatenate([y, y_synthetic]).astype(np.float32)
    row_audit = _build_row_audit(
        source_ids,
        candidate_audit,
        effective_count,
        kept_indices=kept_indices,
    )
    underfill = budget - effective_count
    rejection_counts = _rejection_counts(candidate_audit)
    underfill_reason = _underfill_reason(underfill, rejection_counts)
    uncertainty_semantics = (
        _UNCALIBRATED_UNCERTAINTY if spec.control_number in {12, 13} else None
    )
    metadata = {
        **dict(generated["metadata"]),
        "control_number": spec.control_number,
        "control_id": spec.control_id,
        "requested_added_count": budget,
        "effective_added_count": effective_count,
        "budget_underfill_count": underfill,
        "budget_underfill_reason": underfill_reason,
        "budget_backfill_performed": False,
        **{f"rejected_{reason}_count": count for reason, count in rejection_counts.items()},
        **_scope_metadata(candidate_audit, candidate_scope),
        "context_definition": spec.context_definition,
        "prefilter_pool_hash": prefilter_pool_hash,
        "uncertainty_semantics": uncertainty_semantics,
        "uncertainty_accepted_keys_subset": subset_valid,
        "used_validation_or_test_parents": False,
    }
    resolved = {
        "spec": asdict(spec),
        "generator_config": asdict(generator_config),
        "candidate_scope_mode": (
            candidate_scope.mode if candidate_scope is not None else LEGACY_PROVENANCE
        ),
        "candidate_scope_hash": (
            candidate_scope.scope_hash if candidate_scope is not None else None
        ),
        "nominal_added_budget": budget,
        "seed": random_state,
        "max_teacher_std": threshold,
        "teacher_models": list(resolved_teachers),
    }
    return ChemicalAugmentationControlResult(
        control_id=spec.control_id,
        control_name=spec.control_name,
        spec=spec,
        X=_readonly(X_out),
        y=_readonly(y_out),
        sample_weight=None,
        row_audit=row_audit,
        candidate_audit=candidate_audit,
        requested_added_count=budget,
        effective_added_count=effective_count,
        total_sample_weight=float(len(y_out)),
        chemical_identity_applicable=True,
        nonchemical_semantics=None,
        budget_underfill_count=underfill,
        budget_underfill_reason=underfill_reason,
        generator_metadata=MappingProxyType(metadata),
        resolved_config=MappingProxyType(resolved),
        prefilter_pool_hash=prefilter_pool_hash,
        accepted_candidate_hash=_accepted_candidate_hash(candidate_audit),
        uncertainty_semantics=uncertainty_semantics,
        uncertainty_accepted_keys_subset=subset_valid,
    )


#: Rejection reasons a chemical control counts separately.  Splitting these
#: apart is what makes a suppressed treatment visible: "collides with a reaction
#: the learner observed" and "collides with a reaction the learner was never
#: shown" have opposite scientific meanings under the low-data protocol.
_COUNTED_REJECTION_REASONS = (
    "observed_in_labeled_train",
    "already_measured",
    "quarantined_held_out_identity",
    "source_identical",
    "duplicate_synthetic",
    "feature_duplicate_synthetic",
    "chemical_parse_invalid",
    "role_change_requirement_not_met",
    "unexpected_role_change",
)


def _scope_metadata(
    candidate_audit: pd.DataFrame,
    candidate_scope: CandidateScopePolicy | None,
) -> dict[str, Any]:
    """Record which eligibility rule produced this control's candidate pool."""
    if candidate_scope is not None:
        return candidate_scope.scope_record()
    return {
        "candidate_scope_mode": LEGACY_PROVENANCE,
        "candidate_scope_provenance": LEGACY_PROVENANCE,
        "consults_complete_dataset": None,
        "candidate_scope_hash": (
            str(candidate_audit["candidate_scope_hash"].iloc[0])
            if "candidate_scope_hash" in candidate_audit and not candidate_audit.empty
            else None
        ),
    }


def _rejection_counts(candidate_audit: pd.DataFrame) -> dict[str, int]:
    if candidate_audit.empty or "rejection_reason" not in candidate_audit:
        return dict.fromkeys(_COUNTED_REJECTION_REASONS, 0)
    reasons = candidate_audit["rejection_reason"].fillna("").astype(str)
    return {reason: int(reasons.eq(reason).sum()) for reason in _COUNTED_REJECTION_REASONS}


def _underfill_reason(underfill: int, counts: Mapping[str, int]) -> str | None:
    """Name the dominant reason a control could not spend its budget."""
    if not underfill:
        return None
    dominant = max(counts.items(), key=lambda item: (item[1], item[0]))
    if dominant[1] == 0:
        return "no_role_and_context_eligible_donor"
    return f"candidate_rejection:{dominant[0]}"


def _resolve_spec(control_id: str) -> ChemicalControlSpec:
    try:
        return _SPEC_BY_ID[control_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported chemical augmentation control {control_id!r}; "
            f"expected one of {CHEMICAL_CONTROL_IDS}."
        ) from exc


def _normalize_training_inputs(
    train_frame: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    feature_metadata: FeatureMetadata,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, tuple[str, ...]]:
    if "source_row_id" not in train_frame:
        raise ValueError("train_frame requires strict source_row_id parent identities.")
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if X.ndim != 2 or not len(X):
        raise ValueError("X_train must be a nonempty dense two-dimensional matrix.")
    if len(train_frame) != len(X) or len(y) != len(X):
        raise ValueError("train_frame, X_train, and y_train row counts must match.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Measured training features and labels must be finite.")
    if len(feature_names) != X.shape[1] or feature_metadata.total_width != X.shape[1]:
        raise ValueError("Feature contract width does not match X_train.")
    raw_source_ids = train_frame["source_row_id"]
    if raw_source_ids.isna().any():
        raise ValueError("train_frame source_row_id values must be nonempty and unique.")
    source_ids = tuple(raw_source_ids)
    if (
        any(not isinstance(value, str) or not value.strip() for value in source_ids)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("train_frame source_row_id values must be nonempty and unique.")
    order = np.asarray(sorted(range(len(source_ids)), key=source_ids.__getitem__), dtype=int)
    frame = train_frame.iloc[order].copy()
    ordered_ids = tuple(source_ids[position] for position in order)
    return frame, X[order].copy(), y[order].copy(), ordered_ids


def _role_aware_config(
    spec: ChemicalControlSpec,
    *,
    multiplier: float,
    source_cap: int,
    seed: int,
    max_teacher_std: float | None,
    teacher_models: tuple[str, ...],
) -> RoleAwareConditionTransferConfig:
    if spec.role_transfer_mode is None:  # pragma: no cover - spec invariant
        raise AssertionError("Role-aware control is missing role_transfer_mode.")
    return RoleAwareConditionTransferConfig(
        role_transfer_mode=spec.role_transfer_mode,
        donor_strategy=spec.donor_strategy,
        label_strategy=spec.label_strategy,
        synthetic_multiplier=multiplier,
        max_candidates_per_source=source_cap,
        teacher_models=list(teacher_models),
        max_teacher_std=max_teacher_std,
        min_similarity=None,
        clip_y_min=0.0,
        clip_y_max=100.0,
        random_state=seed,
        donor_similarity_n_bits=256,
        donor_similarity_radius=2,
        donor_similarity_backend=spec.donor_similarity_backend,
        role_change_requirement=spec.role_change_requirement,
        fallback_policy=spec.fallback_policy,
    )


def _uncertainty_threshold(
    spec: ChemicalControlSpec,
    value: float | None,
) -> float | None:
    if not spec.uncertainty_filter_enabled:
        if value is not None:
            raise ValueError(
                "max_teacher_std applies only to "
                "'typed_transfer_with_uncertainty_filtering'."
            )
        return None
    if value is None:
        raise ValueError("Filtered typed transfer requires max_teacher_std.")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("max_teacher_std must be finite and nonnegative.")
    return threshold


def _teacher_models(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("teacher_models must be a nonempty sequence of model names.")
    models = tuple(values)
    if any(not isinstance(model, str) or not model.strip() for model in models):
        raise ValueError("teacher_models must contain nonempty strings.")
    if len(models) != len(set(models)):
        raise ValueError("teacher_models must not contain duplicates.")
    return models


def _assert_train_only_parents(
    candidate_audit: pd.DataFrame,
    allowed_source_ids: tuple[str, ...],
) -> None:
    if candidate_audit.empty:
        return
    allowed = set(allowed_source_ids)
    for column in ("source_row_id", "donor_row_id"):
        if column not in candidate_audit:
            raise ValueError(f"Chemical candidate audit is missing strict parent ID {column!r}.")
        values = candidate_audit[column]
        if values.isna().any():
            raise ValueError(f"Chemical candidate audit contains missing {column}.")
        parent_ids = set(values.astype(str))
        if not parent_ids.issubset(allowed):
            raise ValueError(
                f"Chemical candidate audit contains non-training {column}: "
                f"{sorted(parent_ids - allowed)[:5]}."
            )


def _candidate_pool_hash(candidate_audit: pd.DataFrame) -> str:
    required = (
        "canonical_reaction_key",
        "canonical_reaction_hash",
        "source_row_id",
        "donor_row_id",
    )
    missing = [column for column in required if column not in candidate_audit]
    if missing and not candidate_audit.empty:
        raise ValueError(
            "Chemical candidate pool cannot be hashed; missing fields: "
            + ", ".join(missing)
        )
    records = (
        candidate_audit.loc[:, required].fillna("").astype(str).to_dict("records")
        if not candidate_audit.empty
        else []
    )
    records.sort(
        key=lambda row: tuple(row[column] for column in required)
    )
    payload = json.dumps(
        {"schema": _POOL_HASH_SCHEMA, "candidates": records},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _accepted_candidate_identities(candidate_audit: pd.DataFrame) -> set[tuple[str, str, str]]:
    if candidate_audit.empty:
        return set()
    accepted = candidate_audit.loc[candidate_audit["accepted"].astype(bool)]
    return {
        (
            str(row["canonical_reaction_key"]),
            str(row["source_row_id"]),
            str(row["donor_row_id"]),
        )
        for _, row in accepted.iterrows()
    }


def _accepted_candidate_hash(candidate_audit: pd.DataFrame) -> str:
    identities = sorted(_accepted_candidate_identities(candidate_audit))
    payload = json.dumps(identities, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_row_audit(
    source_ids: tuple[str, ...],
    candidate_audit: pd.DataFrame,
    effective_count: int,
    *,
    kept_indices: list[int] | None = None,
) -> pd.DataFrame:
    real = pd.DataFrame(
        {
            "audit_record_type": "measured",
            "output_row_index": np.arange(len(source_ids), dtype=int),
            "row_id": source_ids,
            "is_real": True,
            "is_added": False,
            "accepted": True,
            "source_row_id": source_ids,
            "donor_row_id": None,
            "rejection_reason": None,
            "chemical_identity_applicable": True,
        }
    )
    if candidate_audit.empty:
        return real
    candidates = candidate_audit.copy()
    candidates.insert(0, "audit_record_type", "candidate")
    candidates["is_real"] = False
    candidates["is_added"] = candidates["kept"].astype(bool)
    candidates["chemical_identity_applicable"] = True
    candidates["row_id"] = candidates["canonical_reaction_hash"]
    output_indices = pd.Series(pd.NA, index=candidates.index, dtype="Int64")
    kept = candidates["kept"].astype(bool)
    kept_count = int(kept.sum())
    if kept_count != effective_count:
        raise ValueError("Chemical candidate kept audit disagrees with synthetic row count.")
    ordered_indices = (
        _ordered_kept_indices(candidates)
        if kept_indices is None
        else kept_indices
    )
    if set(ordered_indices) != set(candidates.index[kept]):
        raise ValueError("Chemical candidate kept indices are inconsistent.")
    output_indices.loc[ordered_indices] = np.arange(
        len(source_ids),
        len(source_ids) + kept_count,
        dtype=int,
    )
    candidates["output_row_index"] = output_indices
    return pd.concat([real, candidates], ignore_index=True, sort=False)


def _ordered_kept_indices(candidate_audit: pd.DataFrame) -> list[int]:
    """Return generator output order, which is the frozen candidate-rank order."""
    kept = candidate_audit.loc[candidate_audit["kept"].astype(bool)]
    if "candidate_rank" in kept and kept["candidate_rank"].notna().all():
        kept = kept.sort_values("candidate_rank", kind="mergesort")
    return [int(index) for index in kept.index]


def _budget_multiplier(budget: int, n_real: int) -> float:
    if not budget:
        return 0.0
    ratio = float(budget) / float(n_real)
    return float(np.nextafter(ratio, 0.0))


def _nonnegative_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result
