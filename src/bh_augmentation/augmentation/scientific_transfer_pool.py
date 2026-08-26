"""Immutable, leakage-bounded synthetic transfer pools for scientific runners."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    EXPLICIT_PROVENANCE,
    GENERATION_SCOPE_MODES,
    CandidateScopePolicy,
    CandidateScopeViolation,
    legacy_scope_from_measured_identity_keys,
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
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    assert_accepted_identity_invariants,
    assert_accepted_role_change_invariants,
    canonicalize_synthetic_roles,
    configured_feature_hash,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ReactionRoles,
    ensure_reaction_role_columns,
    reaction_roles_from_row,
)
from bh_augmentation.features.compatibility import (
    FeatureMetadata,
    assert_feature_compatibility,
)
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    canonical_feature_kind,
)
from bh_augmentation.utils.corrected_runs import feature_contract_record, stable_hash

SCIENTIFIC_TRANSFER_POOL_SCHEMA_VERSION = "bh-scientific-transfer-pool-v1"
_STRICT_TYPED_DONOR_STRATEGY = "same_substrate_different_role"
_POOL_KINDS = {"anonymous", "strict_context_matched_typed"}


@dataclass(frozen=True, slots=True)
class ScientificTransferPool:
    """One defensive-copying synthetic pool and its immutable provenance."""

    pool_kind: str
    schema_version: str
    training_source_id_hash: str
    global_measured_identity_hash: str
    candidate_scope_mode: str
    feature_metadata_hash: str
    config_hash: str
    candidate_audit_hash: str
    accepted_identity_hash: str
    pool_hash: str
    feature_names: tuple[str, ...]
    feature_metadata: FeatureMetadata
    _rows: pd.DataFrame = field(repr=False, compare=False)
    _features: np.ndarray = field(repr=False, compare=False)
    _labels: np.ndarray = field(repr=False, compare=False)
    _audit: pd.DataFrame = field(repr=False, compare=False)
    _generator_metadata: dict[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.pool_kind not in _POOL_KINDS:
            raise ValueError(f"Unknown scientific transfer pool kind {self.pool_kind!r}.")
        if self.candidate_scope_mode not in GENERATION_SCOPE_MODES:
            raise ValueError(
                f"Unknown pool candidate scope mode {self.candidate_scope_mode!r}."
            )
        if self.schema_version != SCIENTIFIC_TRANSFER_POOL_SCHEMA_VERSION:
            raise ValueError("Scientific transfer pool schema version mismatch.")
        for name in (
            "training_source_id_hash",
            "global_measured_identity_hash",
            "feature_metadata_hash",
            "config_hash",
            "candidate_audit_hash",
            "accepted_identity_hash",
            "pool_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest.")
        rows = self._rows.copy(deep=True).reset_index(drop=True)
        features = np.ascontiguousarray(self._features, dtype=np.float32).copy()
        labels = np.ascontiguousarray(self._labels, dtype=float).reshape(-1).copy()
        audit = self._audit.copy(deep=True).reset_index(drop=True)
        if (
            features.ndim != 2
            or len(rows) != len(features)
            or len(rows) != len(labels)
            or features.shape[1] != self.feature_metadata.total_width
            or not np.isfinite(features).all()
            or not np.isfinite(labels).all()
        ):
            raise ValueError("Scientific transfer pool rows, features, and labels diverge.")
        features.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_features", features)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_audit", audit)
        object.__setattr__(
            self, "_generator_metadata", dict(self._generator_metadata)
        )

    @property
    def rows(self) -> pd.DataFrame:
        """Return a defensive copy of accepted, selected synthetic rows."""
        return self._rows.copy(deep=True)

    @property
    def features(self) -> np.ndarray:
        """Return a defensive copy of aligned synthetic features."""
        return self._features.copy()

    @property
    def labels(self) -> np.ndarray:
        """Return a defensive copy of aligned pseudo-labels."""
        return self._labels.copy()

    @property
    def audit(self) -> pd.DataFrame:
        """Return a defensive copy of the complete candidate identity audit."""
        return self._audit.copy(deep=True)

    @property
    def accepted_audit(self) -> pd.DataFrame:
        """Return accepted candidate audits, including candidates truncated by budget."""
        if self._audit.empty:
            return self._audit.copy(deep=True)
        return self._audit.loc[self._audit["accepted"].astype(bool)].copy(deep=True)

    @property
    def generator_metadata(self) -> dict[str, Any]:
        """Return a copy of generator diagnostics."""
        return dict(self._generator_metadata)

    @property
    def candidate_count(self) -> int:
        return len(self._audit)

    @property
    def accepted_count(self) -> int:
        if self._audit.empty:
            return 0
        return int(self._audit["accepted"].astype(bool).sum())

    @property
    def selected_count(self) -> int:
        return len(self._rows)


def build_anonymous_transfer_pool(
    *,
    training_frame: pd.DataFrame,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    feature_config: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_metadata: FeatureMetadata,
    global_measured_identity_keys: Iterable[str] = (),
    candidate_scope: CandidateScopePolicy | None = None,
    config: ConditionTransferConfig,
) -> ScientificTransferPool:
    """Generate one strict anonymous pool without an undeclared fallback."""
    resolved = ConditionTransferConfig(**asdict(config))
    _validate_common_config(
        resolved,
        donor_similarity_backend=resolved.donor_similarity_backend,
    )
    prepared = _prepare_inputs(
        training_frame=training_frame,
        training_features=training_features,
        training_labels=training_labels,
        feature_config=feature_config,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        global_measured_identity_keys=global_measured_identity_keys,
        candidate_scope=candidate_scope,
    )
    result = generate_condition_transfer_examples(
        prepared.frame,
        prepared.features,
        prepared.labels,
        resolved,
        feature_config=prepared.feature_config,
        real_feature_names=list(prepared.feature_names),
        real_feature_metadata=prepared.feature_metadata,
        candidate_scope=prepared.scope,
    )
    return _validate_and_package(
        pool_kind="anonymous",
        config=resolved,
        prepared=prepared,
        result=result,
    )


def build_strict_context_matched_typed_transfer_pool(
    *,
    training_frame: pd.DataFrame,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    feature_config: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_metadata: FeatureMetadata,
    global_measured_identity_keys: Iterable[str] = (),
    candidate_scope: CandidateScopePolicy | None = None,
    config: RoleAwareConditionTransferConfig,
) -> ScientificTransferPool:
    """Generate a strict typed pool using exact-substrate context matching."""
    resolved = RoleAwareConditionTransferConfig(**asdict(config))
    _validate_common_config(
        resolved,
        donor_similarity_backend=resolved.donor_similarity_backend,
    )
    if resolved.donor_strategy != _STRICT_TYPED_DONOR_STRATEGY:
        raise ValueError(
            "Strict context-matched typed pools require "
            f"donor_strategy={_STRICT_TYPED_DONOR_STRATEGY!r}."
        )
    prepared = _prepare_inputs(
        training_frame=training_frame,
        training_features=training_features,
        training_labels=training_labels,
        feature_config=feature_config,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        global_measured_identity_keys=global_measured_identity_keys,
        candidate_scope=candidate_scope,
    )
    clear_role_aware_teacher_cache()
    try:
        result = generate_role_aware_condition_transfer_examples(
            prepared.frame,
            prepared.features,
            prepared.labels,
            resolved,
            feature_config=prepared.feature_config,
            real_feature_names=list(prepared.feature_names),
            real_feature_metadata=prepared.feature_metadata,
            candidate_scope=prepared.scope,
        )
    finally:
        clear_role_aware_teacher_cache()
    return _validate_and_package(
        pool_kind="strict_context_matched_typed",
        config=resolved,
        prepared=prepared,
        result=result,
    )


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    frame: pd.DataFrame = field(repr=False, compare=False)
    features: np.ndarray = field(repr=False, compare=False)
    labels: np.ndarray = field(repr=False, compare=False)
    feature_config: dict[str, Any]
    feature_names: tuple[str, ...]
    feature_metadata: FeatureMetadata
    training_source_ids: tuple[str, ...]
    scope: CandidateScopePolicy
    feature_contract: dict[str, Any]


def _prepare_inputs(
    *,
    training_frame: pd.DataFrame,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    feature_config: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_metadata: FeatureMetadata,
    global_measured_identity_keys: Iterable[str],
    candidate_scope: CandidateScopePolicy | None,
) -> _PreparedInputs:
    if not isinstance(feature_metadata, FeatureMetadata):
        raise TypeError("feature_metadata must be a FeatureMetadata instance.")
    resolved_feature_config = dict(feature_config)
    if (
        resolved_feature_config.get("fingerprint_backend") != "rdkit"
        or feature_metadata.fingerprint_backend != "rdkit"
    ):
        raise ValueError("Scientific transfer pools require RDKit feature semantics.")
    if canonical_feature_kind(resolved_feature_config.get("kind")) != "bh_role_separated":
        raise ValueError(
            "Scientific transfer pools require canonical seven-role separated features."
        )
    if (
        feature_metadata.representation_kind != "bh_role_separated"
        or feature_metadata.role_ordering != CANONICAL_ROLE_NAMES
        or tuple(block.name for block in feature_metadata.block_slices)
        != CANONICAL_ROLE_NAMES
    ):
        raise ValueError("Feature metadata do not describe canonical seven-role order.")

    frame = ensure_reaction_role_columns(
        training_frame, parse_if_missing=False
    ).copy()
    if "source_row_id" not in frame:
        raise ValueError("Scientific transfer pools require source_row_id.")
    source_ids = tuple(frame["source_row_id"].astype(str))
    if (
        any(not source_id for source_id in source_ids)
        or len(source_ids) != len(set(source_ids))
    ):
        raise ValueError("Training source_row_id values must be nonempty and unique.")
    features = np.asarray(training_features, dtype=np.float32)
    labels = np.asarray(training_labels, dtype=float).reshape(-1)
    names = tuple(str(name) for name in feature_names)
    if (
        features.ndim != 2
        or len(features) != len(frame)
        or len(labels) != len(frame)
        or not np.isfinite(features).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("Training frame, features, and labels are invalid or unaligned.")
    if "yield" in frame and not np.array_equal(
        pd.to_numeric(frame["yield"], errors="raise").to_numpy(float), labels
    ):
        raise ValueError("training_labels differ from training_frame yield values.")

    global_keys = tuple(
        sorted(
            {
                str(key)
                for key in global_measured_identity_keys
                if isinstance(key, str) and key
            }
        )
    )
    if candidate_scope is not None and global_keys:
        raise CandidateScopeViolation(
            "Pass either candidate_scope or global_measured_identity_keys to a "
            "scientific transfer pool, not both."
        )
    if candidate_scope is None and not global_keys:
        raise ValueError(
            "A scientific transfer pool requires either an explicit candidate_scope "
            "or a nonempty global_measured_identity_keys contract."
        )
    computed_training_keys: set[str] = set()
    for row in frame.to_dict(orient="records"):
        identity = canonicalize_synthetic_roles(reaction_roles_from_row(row))
        if (
            not identity.chemical_parse_valid
            or identity.canonical_reaction_key is None
        ):
            raise ValueError("Training frame contains an invalid canonical identity.")
        supplied_key = row.get("canonical_reaction_key")
        if (
            isinstance(supplied_key, str)
            and supplied_key
            and supplied_key != identity.canonical_reaction_key
        ):
            raise ValueError("Training canonical_reaction_key does not match its roles.")
        computed_training_keys.add(identity.canonical_reaction_key)
    if candidate_scope is None:
        scope = legacy_scope_from_measured_identity_keys(
            labeled_train_identity_keys=computed_training_keys,
            measured_identity_keys=global_keys,
        )
        outside_contract = computed_training_keys - set(global_keys)
        if outside_contract:
            raise ValueError(
                "Global measured identity contract omits training reactions: "
                f"{sorted(outside_contract)[:3]}."
            )
    else:
        scope = candidate_scope
        outside_scope = computed_training_keys - set(scope.observed_identity_keys)
        if outside_scope:
            raise CandidateScopeViolation(
                "The candidate scope omits training reactions handed to the pool: "
                f"{sorted(outside_scope)[:3]}."
            )

    replay_frame = frame.copy()
    replay_frame["yield"] = 0.0
    replay_X, _, replay_names, replay_metadata = build_feature_matrix_with_metadata(
        replay_frame, resolved_feature_config
    )
    replay_X = np.asarray(replay_X, dtype=np.float32)
    assert_feature_compatibility(
        replay_X,
        replay_names,
        features,
        names,
        real_metadata=replay_metadata,
        synthetic_metadata=feature_metadata,
    )
    if not np.array_equal(replay_X, features):
        raise ValueError("Provided training features do not match canonical identities.")
    feature_contract = feature_contract_record(feature_metadata, names)
    return _PreparedInputs(
        frame=frame,
        features=np.ascontiguousarray(features),
        labels=np.ascontiguousarray(labels),
        feature_config=resolved_feature_config,
        feature_names=names,
        feature_metadata=feature_metadata,
        training_source_ids=source_ids,
        scope=scope,
        feature_contract=feature_contract,
    )


def _validate_common_config(
    config: ConditionTransferConfig | RoleAwareConditionTransferConfig,
    *,
    donor_similarity_backend: str,
) -> None:
    if config.role_change_requirement != "all":
        raise ValueError("Scientific transfer pools require all requested roles to change.")
    if config.fallback_policy != "reject":
        raise ValueError("Scientific transfer pools require fallback_policy='reject'.")
    if donor_similarity_backend != "rdkit":
        raise ValueError("Scientific transfer pools require RDKit donor similarity.")


def _validate_and_package(
    *,
    pool_kind: str,
    config: ConditionTransferConfig | RoleAwareConditionTransferConfig,
    prepared: _PreparedInputs,
    result: Mapping[str, Any],
) -> ScientificTransferPool:
    required_result = {
        "synthetic_df",
        "synthetic_y",
        "X_synthetic",
        "metadata",
        "candidate_df",
        "feature_names",
        "feature_metadata",
    }
    if set(result) != required_result:
        raise ValueError("Synthetic generator result schema mismatch.")
    rows = result["synthetic_df"]
    audit = result["candidate_df"]
    metadata = result["metadata"]
    if (
        not isinstance(rows, pd.DataFrame)
        or not isinstance(audit, pd.DataFrame)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("Synthetic generator returned invalid row or audit containers.")
    features = np.asarray(result["X_synthetic"], dtype=np.float32)
    labels = np.asarray(result["synthetic_y"], dtype=float).reshape(-1)
    assert_feature_compatibility(
        prepared.features,
        prepared.feature_names,
        features,
        result["feature_names"],
        real_metadata=prepared.feature_metadata,
        synthetic_metadata=result["feature_metadata"],
    )
    if len(rows) != len(features) or len(rows) != len(labels):
        raise ValueError("Synthetic generator output rows, features, and labels diverge.")
    if not audit.empty:
        missing = sorted(
            set(
                [
                    *REQUIRED_SYNTHETIC_AUDIT_FIELDS,
                    "accepted",
                    "kept",
                    "role_change_valid",
                ]
            )
            - set(audit)
        )
        if missing:
            raise ValueError(f"Synthetic candidate audit is missing fields: {missing}.")
        assert_accepted_identity_invariants(audit)
        assert_accepted_role_change_invariants(audit)
        accepted_candidates = audit.loc[audit["accepted"].astype(bool)]
        ineligible = set(
            accepted_candidates["canonical_reaction_key"].astype(str)
        ) & prepared.scope.rejection_identity_keys()
        if ineligible:
            raise ValueError(
                "An accepted synthetic candidate duplicates chemistry the "
                f"{prepared.scope.mode} scope declares ineligible."
            )
        quarantined = set(
            accepted_candidates["canonical_reaction_key"].astype(str)
        ) & prepared.scope.quarantined_identity_keys()
        if quarantined:
            raise ValueError(
                "An accepted synthetic candidate matches a quarantined held-out "
                "evaluation identity."
            )
        for key, digest in zip(
            accepted_candidates["canonical_reaction_key"],
            accepted_candidates["canonical_reaction_hash"],
            strict=True,
        ):
            if hashlib.sha256(str(key).encode()).hexdigest() != str(digest):
                raise ValueError("Accepted synthetic canonical reaction hash mismatch.")
        parent_ids = set(audit["source_row_id"].dropna().astype(str)) | set(
            audit["donor_row_id"].dropna().astype(str)
        )
        outside = parent_ids - set(prepared.training_source_ids)
        if outside:
            raise ValueError(
                f"Synthetic candidate parents are outside provided training IDs: {sorted(outside)}."
            )
    kept = (
        audit.loc[audit["kept"].astype(bool)]
        .sort_values(
            ["candidate_rank", "canonical_reaction_key"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        if not audit.empty
        else audit.copy()
    )
    if len(kept) != len(rows):
        raise ValueError("Selected synthetic rows do not match kept candidate audits.")
    if len(rows):
        if (
            "canonical_reaction_key" not in rows
            or "yield" not in rows
            or rows["canonical_reaction_key"].astype(str).tolist()
            != kept["canonical_reaction_key"].astype(str).tolist()
            or not np.array_equal(
                pd.to_numeric(rows["yield"], errors="raise").to_numpy(float),
                labels,
            )
        ):
            raise ValueError("Synthetic row identity or label alignment mismatch.")
        expected_feature_hashes = [
            configured_feature_hash(vector) for vector in features
        ]
        if expected_feature_hashes != kept["feature_hash"].tolist():
            raise ValueError("Synthetic feature hashes do not align with pool features.")
        if (
            set(kept["canonical_reaction_key"].astype(str))
            & (
                prepared.scope.rejection_identity_keys()
                | prepared.scope.quarantined_identity_keys()
            )
            or kept["canonical_reaction_key"].duplicated().any()
            or kept["feature_hash"].duplicated().any()
            or not kept["chemical_parse_valid"].astype(bool).all()
        ):
            raise ValueError("Selected pool violates canonical identity invariants.")
        for key, digest in zip(
            kept["canonical_reaction_key"],
            kept["canonical_reaction_hash"],
            strict=True,
        ):
            if hashlib.sha256(str(key).encode()).hexdigest() != str(digest):
                raise ValueError("Synthetic canonical reaction hash mismatch.")

    config_hash = stable_hash(asdict(config))
    candidate_audit_hash = stable_hash(
        sorted(
            (_clean_json(record) for record in audit.to_dict(orient="records")),
            key=stable_hash,
        )
    )
    accepted_records = sorted(
        (
            {
                "canonical_reaction_key": str(row["canonical_reaction_key"]),
                "canonical_reaction_hash": str(row["canonical_reaction_hash"]),
                "feature_hash": str(row["feature_hash"]),
                "source_row_id": str(row["source_row_id"]),
                "donor_row_id": str(row["donor_row_id"]),
                "synthetic_label": float(row["synthetic_label"]),
                "kept": bool(row["kept"]),
            }
            for row in audit.loc[
                audit["accepted"].astype(bool)
            ].to_dict(orient="records")
        ),
        key=lambda record: (
            record["canonical_reaction_key"],
            record["source_row_id"],
            record["donor_row_id"],
        ),
    ) if not audit.empty else []
    accepted_identity_hash = stable_hash(accepted_records)
    training_source_id_hash = stable_hash(sorted(prepared.training_source_ids))
    global_measured_identity_hash = stable_hash(prepared.scope.global_identity_keys)
    payload = {
        "schema_version": SCIENTIFIC_TRANSFER_POOL_SCHEMA_VERSION,
        "pool_kind": pool_kind,
        "training_source_id_hash": training_source_id_hash,
        "global_measured_identity_hash": global_measured_identity_hash,
        "feature_metadata_hash": prepared.feature_contract["feature_metadata_hash"],
        "config_hash": config_hash,
        "candidate_audit_hash": candidate_audit_hash,
        "accepted_identity_hash": accepted_identity_hash,
        "selected": [
            {
                "canonical_reaction_key": str(row["canonical_reaction_key"]),
                "canonical_reaction_hash": str(row["canonical_reaction_hash"]),
                "feature_hash": str(row["feature_hash"]),
                "source_row_id": str(row["source_row_id"]),
                "donor_row_id": str(row["donor_row_id"]),
                "synthetic_label": float(row["synthetic_label"]),
            }
            for row in kept.to_dict(orient="records")
        ],
    }
    if prepared.scope.provenance == EXPLICIT_PROVENANCE:
        payload["candidate_scope"] = prepared.scope.scope_record()
    return ScientificTransferPool(
        pool_kind=pool_kind,
        schema_version=SCIENTIFIC_TRANSFER_POOL_SCHEMA_VERSION,
        training_source_id_hash=training_source_id_hash,
        global_measured_identity_hash=global_measured_identity_hash,
        candidate_scope_mode=prepared.scope.mode,
        feature_metadata_hash=prepared.feature_contract["feature_metadata_hash"],
        config_hash=config_hash,
        candidate_audit_hash=candidate_audit_hash,
        accepted_identity_hash=accepted_identity_hash,
        pool_hash=stable_hash(payload),
        feature_names=prepared.feature_names,
        feature_metadata=prepared.feature_metadata,
        _rows=rows,
        _features=features,
        _labels=labels,
        _audit=audit,
        _generator_metadata=dict(metadata),
    )


def _clean_json(value: Any) -> Any:
    if is_dataclass(value):
        return _clean_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, ReactionRoles):
        return asdict(value)
    return value
