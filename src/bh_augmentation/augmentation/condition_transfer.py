"""Chemically constrained condition-transfer augmentation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    apply_filter_rejection,
    assert_accepted_identity_invariants,
    assert_accepted_role_change_invariants,
    audit_candidate_identities,
    audit_role_changes,
    canonical_candidate_record,
    canonicalize_synthetic_roles,
    measured_canonical_keys,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    ReactionRoles,
    ensure_reaction_role_columns,
    reaction_roles_from_row,
)
from bh_augmentation.features.compatibility import FeatureMetadata, assert_feature_compatibility
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    morgan_fingerprint,
)
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model

DONOR_STRATEGIES = {"random", "nearest_reaction", "nearest_substrate", "high_yield_nearest"}
FALLBACK_POLICIES = {"reject", "same_product", "nearest_substrate", "random"}
ANONYMOUS_TRANSFER_ROLES = (
    "catalyst",
    "ligand",
    "base",
    "solvent_or_additive",
)
LABEL_STRATEGIES = {
    "teacher_ensemble",
    "uncertainty_filtered_teacher",
    "source_label",
    "donor_label",
    "average_label",
}


@dataclass(frozen=True)
class ParsedReaction:
    """Parsed condition-transfer view of a reaction string."""

    reaction_smiles: str
    left_tokens: list[str]
    right: str
    substrates: tuple[str, str]
    condition_tokens: tuple[str, ...]
    substrate_block: str
    condition_block: str
    product: str


@dataclass
class ConditionTransferConfig:
    """Configuration for condition-transfer augmentation."""

    donor_strategy: str
    synthetic_multiplier: float
    n_neighbors: int
    label_strategy: str
    teacher_models: list[str]
    max_teacher_std: float | None
    min_similarity: float | None
    high_yield_threshold: float
    clip_y_min: float
    clip_y_max: float
    candidates_per_real: int
    random_state: int
    donor_similarity_n_bits: int = 2048
    donor_similarity_radius: int = 2
    donor_similarity_backend: str = "auto"
    role_change_requirement: str = "all"
    fallback_policy: str = "reject"
    max_candidates_per_source: int | None = None
    requested_roles: tuple[str, ...] | list[str] | None = None


def parse_condition_transfer_reaction(value: object) -> ParsedReaction | None:
    """Parse `substrate.substrate.conditions>>product` for condition transfer."""
    if not isinstance(value, str) or value.count(">>") != 1:
        return None
    left, right = (part.strip() for part in value.split(">>", 1))
    left_tokens = [token.strip() for token in left.split(".") if token.strip()]
    product = right.strip()
    if len(left_tokens) < 3 or not product:
        return None
    substrates = (left_tokens[0], left_tokens[1])
    condition_tokens = tuple(left_tokens[2:])
    return ParsedReaction(
        reaction_smiles=value,
        left_tokens=left_tokens,
        right=product,
        substrates=substrates,
        condition_tokens=condition_tokens,
        substrate_block=".".join(substrates),
        condition_block=".".join(condition_tokens),
        product=product,
    )


def build_condition_transfer_reaction(
    source: ParsedReaction,
    donor: ParsedReaction,
) -> str:
    """Build a synthetic reaction with source substrates/product and donor conditions."""
    return f"{source.substrate_block}.{donor.condition_block}>>{source.product}"


def build_anonymous_condition_transfer_roles(
    source_row: pd.Series | dict[str, Any],
    donor_row: pd.Series | dict[str, Any],
    *,
    requested_roles: tuple[str, ...] | list[str] | None = None,
) -> ReactionRoles:
    """Transfer exactly the requested donor conditions and preserve other roles."""
    source = reaction_roles_from_row(source_row)
    donor = reaction_roles_from_row(donor_row)
    requested = set(_canonical_requested_roles(requested_roles))
    return ReactionRoles(
        reactant_1=source.reactant_1,
        reactant_2=source.reactant_2,
        catalyst=donor.catalyst if "catalyst" in requested else source.catalyst,
        ligand=donor.ligand if "ligand" in requested else source.ligand,
        base=donor.base if "base" in requested else source.base,
        solvent_or_additive=(
            donor.solvent_or_additive
            if "solvent_or_additive" in requested
            else source.solvent_or_additive
        ),
        product=source.product,
    )


def generate_condition_transfer_examples(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ConditionTransferConfig,
    *,
    feature_config: dict[str, Any],
    real_feature_names: list[str],
    real_feature_metadata: FeatureMetadata,
    measured_identity_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Generate condition-transfer examples using only low-data training rows."""
    _validate_config(config)
    requested_roles = _canonical_requested_roles(config.requested_roles)
    if "reaction_smiles" not in df_train.columns:
        raise ValueError("condition transfer requires a reaction_smiles column.")

    role_train = ensure_reaction_role_columns(df_train, parse_if_missing=True)
    all_measured_keys = measured_canonical_keys(
        role_train,
        additional_keys=measured_identity_keys,
    )
    train = role_train.reset_index(drop=False).rename(columns={"index": "_source_dataframe_index"})
    train_records = train.to_dict(orient="records")
    canonical_train_identities = [
        canonicalize_synthetic_roles(reaction_roles_from_row(row))
        for row in train_records
    ]
    canonical_train_roles = [identity.roles for identity in canonical_train_identities]
    if any(roles is None for roles in canonical_train_roles):
        raise ValueError("Condition-transfer source rows require valid canonical roles.")
    X_train_array = np.asarray(X_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(train) != len(X_train_array) or len(train) != len(y_train_array):
        raise ValueError("df_train, X_train, and y_train must contain the same number of rows.")

    parsed = [parse_condition_transfer_reaction(value) for value in train["reaction_smiles"]]
    valid_positions = [position for position, item in enumerate(parsed) if item is not None]
    invalid_parse_count = len(train) - len(valid_positions)
    n_real_train = len(train)
    target_count = int(math.ceil(max(0.0, config.synthetic_multiplier) * n_real_train))
    max_candidates_per_source = _configured_max_candidates_per_source(config)
    if target_count == 0 or len(valid_positions) < 2:
        metadata = _metadata_from_candidates(
            _empty_candidate_df(),
            config,
            n_real_train=n_real_train,
            invalid_parse_count=invalid_parse_count,
            invalid_featurization_count=0,
            duplicate_synthetic_count=0,
            existing_real_duplicate_count=0,
            high_yield_fallback_count=0,
        )
        metadata["fallback_attempt_count"] = 0
        metadata["fallback_success_count"] = 0
        return _package_result(
            _empty_synthetic_df(), np.empty(0), metadata, _empty_candidate_df(),
            np.empty((0, X_train_array.shape[1])), real_feature_names, real_feature_metadata,
        )

    similarity_config = {
        **feature_config,
        "kind": "reaction_section_concat",
        "categorical_columns": [],
    }
    reaction_features, _, _, _ = build_feature_matrix_with_metadata(train, similarity_config)
    reaction_similarity = _cosine_similarity_matrix(reaction_features)
    np.fill_diagonal(reaction_similarity, -np.inf)
    substrate_similarity = _substrate_similarity_matrix(
        parsed,
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    np.fill_diagonal(substrate_similarity, -np.inf)
    product_similarity = _token_group_similarity_matrix(
        [(item.product,) if item is not None else () for item in parsed],
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    condition_similarity = _token_group_similarity_matrix(
        [item.condition_tokens if item is not None else () for item in parsed],
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    np.fill_diagonal(product_similarity, -np.inf)
    np.fill_diagonal(condition_similarity, -np.inf)

    rows: list[dict[str, Any]] = []
    high_yield_fallback_count = 0
    fallback_attempt_count = 0
    fallback_success_count = 0
    stable_row_keys = [
        _stable_training_row_key(
            train_records[position],
            canonical_train_identities[position].canonical_reaction_key,
            float(y_train_array[position]),
        )
        for position in range(len(train))
    ]
    source_order = sorted(valid_positions, key=lambda position: stable_row_keys[position])
    generated_per_source = {position: 0 for position in source_order}
    used_donors = {position: set() for position in source_order}
    source_rngs = {
        position: np.random.default_rng(
            _stable_seed(config.random_state, stable_row_keys[position])
        )
        for position in source_order
    }
    candidate_id = 0
    for source_position in source_order:
        while generated_per_source[source_position] < max_candidates_per_source:
            available_positions = [
                position
                for position in sorted(valid_positions, key=lambda item: stable_row_keys[item])
                if position == source_position or position not in used_donors[source_position]
            ]
            if len(available_positions) < 2:
                break
            donor_position, donor_similarity, used_fallback = _select_donor_position(
                source_position=source_position,
                valid_positions=available_positions,
                y_train=y_train_array,
                reaction_similarity=reaction_similarity,
                substrate_similarity=substrate_similarity,
                canonical_roles=canonical_train_roles,
                config=config,
                rng=source_rngs[source_position],
            )
            if used_fallback:
                fallback_attempt_count += 1
            if donor_position is None:
                break
            used_donors[source_position].add(donor_position)
            if used_fallback:
                high_yield_fallback_count += 1
                fallback_success_count += 1
            source = parsed[source_position]
            donor = parsed[donor_position]
            if source is None or donor is None:  # pragma: no cover - valid_positions invariant
                continue
            synthetic_roles = build_anonymous_condition_transfer_roles(
                train.loc[source_position],
                train.loc[donor_position],
                requested_roles=requested_roles,
            )
            synthetic_record = canonical_candidate_record(synthetic_roles)
            generated_per_source[source_position] += 1
            source_candidate_ordinal = generated_per_source[source_position]
            substrate_score = float(substrate_similarity[source_position, donor_position])
            product_score = float(product_similarity[source_position, donor_position])
            condition_score = float(condition_similarity[source_position, donor_position])
            relevant_context_score = float(np.mean([substrate_score, product_score]))
            overall_score = float(
                np.mean([substrate_score, product_score, condition_score])
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "reaction_smiles": synthetic_record["reaction_smiles"],
                    "source_position": source_position,
                    "donor_position": donor_position,
                    "source_index": train.loc[source_position, "_source_dataframe_index"],
                    "donor_index": train.loc[donor_position, "_source_dataframe_index"],
                    "source_reaction_smiles": source.reaction_smiles,
                    "donor_reaction_smiles": donor.reaction_smiles,
                    "source_yield": float(y_train_array[source_position]),
                    "donor_yield": float(y_train_array[donor_position]),
                    "donor_strategy": config.donor_strategy,
                    "effective_donor_strategy": (
                        config.fallback_policy if used_fallback else config.donor_strategy
                    ),
                    "fallback_policy": config.fallback_policy,
                    "fallback_used": bool(used_fallback),
                    "label_strategy": config.label_strategy,
                    "donor_similarity": float(donor_similarity),
                    "substrate_similarity": substrate_score,
                    "product_similarity": product_score,
                    "nontransferred_role_similarity": relevant_context_score,
                    "condition_similarity": condition_score,
                    "overall_similarity": overall_score,
                    "relevant_context_similarity": relevant_context_score,
                    "source_candidate_ordinal": source_candidate_ordinal,
                    "generated_per_source": 0,
                    "max_candidates_per_source": max_candidates_per_source,
                    "source_product": source.product,
                    "source_substrate_block": source.substrate_block,
                    "donor_condition_block": donor.condition_block,
                    "teacher_mean": np.nan,
                    "teacher_std": np.nan,
                    "synthetic_label": np.nan,
                    "synthetic_was_duplicate": False,
                    "existing_real_duplicate": False,
                    "accepted": False,
                    "kept": False,
                    **synthetic_record,
                }
            )
            candidate_id += 1

    for row in rows:
        row["generated_per_source"] = generated_per_source[int(row["source_position"])]

    candidate_df = pd.DataFrame(rows) if rows else _empty_candidate_df()
    if candidate_df.empty:
        metadata = _metadata_from_candidates(
            candidate_df,
            config,
            n_real_train=n_real_train,
            invalid_parse_count=invalid_parse_count,
            invalid_featurization_count=0,
            duplicate_synthetic_count=0,
            existing_real_duplicate_count=0,
            high_yield_fallback_count=high_yield_fallback_count,
        )
        metadata["fallback_attempt_count"] = fallback_attempt_count
        metadata["fallback_success_count"] = fallback_success_count
        return _package_result(
            _empty_synthetic_df(), np.empty(0), metadata, candidate_df,
            np.empty((0, X_train_array.shape[1])), real_feature_names, real_feature_metadata,
        )

    feature_frame = candidate_df.copy()
    feature_frame["yield"] = 0.0
    X_synthetic, _, synthetic_feature_names, synthetic_feature_metadata = (
        build_feature_matrix_with_metadata(feature_frame, feature_config)
    )
    assert_feature_compatibility(
        X_train_array,
        real_feature_names,
        X_synthetic,
        synthetic_feature_names,
        real_metadata=real_feature_metadata,
        synthetic_metadata=synthetic_feature_metadata,
    )
    candidate_df = audit_candidate_identities(
        candidate_df,
        X_synthetic,
        measured_keys=all_measured_keys,
        source_rows=train_records,
    )
    candidate_df = audit_role_changes(
        candidate_df,
        source_rows=train_records,
        requested_roles=requested_roles,
        role_change_requirement=config.role_change_requirement,
    )
    candidate_df["synthetic_was_duplicate"] = (
        candidate_df["duplicate_synthetic"].astype(bool)
        | candidate_df["feature_duplicate_synthetic"].astype(bool)
    )
    candidate_df["existing_real_duplicate"] = candidate_df["already_measured"].astype(bool)
    invalid_featurization_count = 0
    teacher_mean, teacher_std, teachers_used = _teacher_predictions(
        X_train_array,
        y_train_array,
        X_synthetic,
        config,
    )
    candidate_df["teacher_mean"] = teacher_mean
    candidate_df["teacher_std"] = teacher_std
    candidate_df["teacher_models_used"] = ",".join(teachers_used)
    support_distance = _nearest_support_distance(X_synthetic, X_train_array)
    candidate_df["nearest_training_support_distance"] = support_distance
    candidate_df["support_distance_metric"] = "cosine_distance"
    candidate_df["support_distance_backend"] = (
        f"{real_feature_metadata.representation_kind}:"
        f"{real_feature_metadata.fingerprint_backend}"
    )
    candidate_df["out_of_support_distance"] = support_distance
    candidate_df["diversity_contribution"] = _diversity_contribution(X_synthetic)
    uncertainty_rank_value, uncertainty_rank_basis = _uncertainty_ranking_proxy(
        candidate_df,
        y_train_array,
    )
    candidate_df["calibrated_uncertainty"] = np.nan
    candidate_df["uncertainty_rank_value"] = uncertainty_rank_value
    candidate_df["uncertainty_rank_basis"] = uncertainty_rank_basis
    labels = _assign_labels(candidate_df, config)
    candidate_df["synthetic_label"] = np.clip(
        labels,
        float(config.clip_y_min),
        float(config.clip_y_max),
    )
    candidate_df = apply_filter_rejection(
        candidate_df,
        ~np.isfinite(candidate_df["synthetic_label"].to_numpy(dtype=float)),
        "invalid_synthetic_label",
    )
    if config.min_similarity is not None:
        candidate_df = apply_filter_rejection(
            candidate_df,
            ~np.isfinite(candidate_df["donor_similarity"].to_numpy(dtype=float))
            | (
                candidate_df["donor_similarity"].to_numpy(dtype=float)
                < float(config.min_similarity)
            ),
            "similarity_below_threshold",
        )
    if (
        config.label_strategy == "uncertainty_filtered_teacher"
        and config.max_teacher_std is not None
    ):
        candidate_df = apply_filter_rejection(
            candidate_df,
            candidate_df["teacher_std"].to_numpy(dtype=float)
            > float(config.max_teacher_std),
            "teacher_uncertainty_above_threshold",
        )
    accepted = _acceptance_mask(candidate_df, X_synthetic, config)
    candidate_df["accepted"] = accepted
    assert_accepted_identity_invariants(candidate_df)
    assert_accepted_role_change_invariants(candidate_df)
    accepted_indices = np.flatnonzero(accepted)
    if len(accepted_indices):
        candidate_df.loc[accepted_indices, "diversity_contribution"] = (
            _diversity_contribution(X_synthetic[accepted_indices])
        )
    candidate_df["candidate_rank"] = pd.Series(
        pd.array([pd.NA] * len(candidate_df), dtype="Int64"),
        index=candidate_df.index,
    )
    ranked_indices = _rank_accepted_candidates(candidate_df)
    if ranked_indices:
        candidate_df.loc[ranked_indices, "candidate_rank"] = np.arange(
            1, len(ranked_indices) + 1
        )
    kept_indices = ranked_indices[:target_count]
    candidate_df.loc[kept_indices, "kept"] = True

    synthetic_df = candidate_df.loc[kept_indices, _synthetic_columns()].copy()
    synthetic_df["yield"] = candidate_df.loc[
        kept_indices, "synthetic_label"
    ].to_numpy(dtype=float)
    synthetic_y = synthetic_df["yield"].to_numpy(dtype=float)
    duplicate_synthetic_count = int(candidate_df["synthetic_was_duplicate"].sum())
    existing_real_duplicate_count = int(candidate_df["existing_real_duplicate"].sum())
    metadata = _metadata_from_candidates(
        candidate_df,
        config,
        n_real_train=n_real_train,
        invalid_parse_count=invalid_parse_count,
        invalid_featurization_count=invalid_featurization_count,
        duplicate_synthetic_count=duplicate_synthetic_count,
        existing_real_duplicate_count=existing_real_duplicate_count,
        high_yield_fallback_count=high_yield_fallback_count,
    )
    metadata["teacher_models_used"] = ",".join(teachers_used)
    metadata["fallback_attempt_count"] = fallback_attempt_count
    metadata["fallback_success_count"] = fallback_success_count
    return _package_result(
        synthetic_df.reset_index(drop=True),
        synthetic_y,
        metadata,
        candidate_df,
        X_synthetic[np.asarray(kept_indices, dtype=int)].astype(np.float32),
        synthetic_feature_names,
        synthetic_feature_metadata,
    )


def _select_donor_position(
    source_position: int,
    valid_positions: list[int],
    y_train: np.ndarray,
    reaction_similarity: np.ndarray,
    substrate_similarity: np.ndarray,
    canonical_roles: list[ReactionRoles | None],
    config: ConditionTransferConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, bool]:
    candidates = [position for position in valid_positions if position != source_position]
    candidates = _role_change_eligible_candidates(
        source_position,
        candidates,
        canonical_roles=canonical_roles,
        requested_roles=_canonical_requested_roles(config.requested_roles),
        requirement=config.role_change_requirement,
    )
    if not candidates:
        return None, float("nan"), False
    if config.donor_strategy == "random":
        candidates = _similarity_eligible_candidates(
            source_position,
            candidates,
            reaction_similarity,
            config.min_similarity,
        )
        if not candidates:
            return None, float("nan"), False
        donor = int(rng.choice(candidates))
        return donor, float(reaction_similarity[source_position, donor]), False
    if config.donor_strategy == "nearest_substrate":
        donor, similarity = _nearest_from_candidates(
            source_position,
            candidates,
            substrate_similarity,
            config,
        )
        if donor is not None:
            return donor, similarity, False
        return _select_fallback_donor(
            source_position,
            candidates,
            reaction_similarity=reaction_similarity,
            substrate_similarity=substrate_similarity,
            canonical_roles=canonical_roles,
            config=config,
            rng=rng,
        )
    if config.donor_strategy == "high_yield_nearest":
        high_yield_candidates = [
            position
            for position in candidates
            if float(y_train[position]) >= float(config.high_yield_threshold)
        ]
        if high_yield_candidates:
            donor, similarity = _nearest_from_candidates(
                source_position,
                high_yield_candidates,
                reaction_similarity,
                config,
            )
            if donor is not None:
                return donor, similarity, False
        return _select_fallback_donor(
            source_position,
            candidates,
            reaction_similarity=reaction_similarity,
            substrate_similarity=substrate_similarity,
            canonical_roles=canonical_roles,
            config=config,
            rng=rng,
        )
    donor, similarity = _nearest_from_candidates(
        source_position,
        candidates,
        reaction_similarity,
        config,
    )
    if donor is not None:
        return donor, similarity, False
    return _select_fallback_donor(
        source_position,
        candidates,
        reaction_similarity=reaction_similarity,
        substrate_similarity=substrate_similarity,
        canonical_roles=canonical_roles,
        config=config,
        rng=rng,
    )


def _role_change_eligible_candidates(
    source_position: int,
    candidates: list[int],
    *,
    canonical_roles: list[ReactionRoles | None],
    requested_roles: tuple[str, ...],
    requirement: str,
) -> list[int]:
    source = canonical_roles[source_position]
    if source is None:
        return []
    eligible: list[int] = []
    for position in candidates:
        donor = canonical_roles[position]
        if donor is None:
            continue
        changed = [
            role
            for role in requested_roles
            if getattr(source, role) != getattr(donor, role)
        ]
        if (requirement == "all" and len(changed) == len(requested_roles)) or (
            requirement == "any" and changed
        ):
            eligible.append(position)
    return eligible


def _select_fallback_donor(
    source_position: int,
    candidates: list[int],
    *,
    reaction_similarity: np.ndarray,
    substrate_similarity: np.ndarray,
    canonical_roles: list[ReactionRoles | None],
    config: ConditionTransferConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, bool]:
    if config.fallback_policy == "reject":
        return None, float("nan"), True
    if config.fallback_policy == "same_product":
        source = canonical_roles[source_position]
        same_product = [
            position
            for position in candidates
            if source is not None
            and canonical_roles[position] is not None
            and canonical_roles[position].product == source.product
        ]
        donor, similarity = _nearest_from_candidates(
            source_position,
            same_product,
            reaction_similarity,
            config,
        )
        return donor, similarity, True
    if config.fallback_policy == "nearest_substrate":
        donor, similarity = _nearest_from_candidates(
            source_position,
            candidates,
            substrate_similarity,
            config,
        )
        return donor, similarity, True
    if config.fallback_policy == "random":
        candidates = _similarity_eligible_candidates(
            source_position,
            candidates,
            reaction_similarity,
            config.min_similarity,
        )
        if not candidates:
            return None, float("nan"), True
        donor = int(rng.choice(candidates))
        return donor, float(reaction_similarity[source_position, donor]), True
    raise ValueError(f"Unsupported fallback policy: {config.fallback_policy}")


def _nearest_from_candidates(
    source_position: int,
    candidates: list[int],
    similarity: np.ndarray,
    config: ConditionTransferConfig,
) -> tuple[int | None, float]:
    if not candidates:
        return None, float("nan")
    scores = np.asarray([similarity[source_position, position] for position in candidates], dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    n_neighbors = max(1, min(int(config.n_neighbors), len(order)))
    for order_index in order[:n_neighbors]:
        donor = int(candidates[int(order_index)])
        score = float(scores[int(order_index)])
        if config.min_similarity is None or score >= float(config.min_similarity):
            return donor, score
    return None, float("nan")


def _similarity_eligible_candidates(
    source_position: int,
    candidates: list[int],
    similarity: np.ndarray,
    min_similarity: float | None,
) -> list[int]:
    if min_similarity is None:
        return candidates
    threshold = float(min_similarity)
    return [
        position
        for position in candidates
        if np.isfinite(similarity[source_position, position])
        and float(similarity[source_position, position]) >= threshold
    ]


def _teacher_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synthetic: np.ndarray,
    config: ConditionTransferConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    needs_teacher = config.label_strategy in {"teacher_ensemble", "uncertainty_filtered_teacher"}
    if not needs_teacher:
        return (
            np.full(len(X_synthetic), np.nan, dtype=float),
            np.full(len(X_synthetic), np.nan, dtype=float),
            [],
        )
    predictions: list[np.ndarray] = []
    teachers_used: list[str] = []
    for model_name in config.teacher_models or ["ridge", "random_forest"]:
        try:
            model = get_model(str(model_name), seed=int(config.random_state))
        except ImportError:
            continue
        fitted = train_model(model, X_train, y_train)
        predictions.append(predict_model(fitted, X_synthetic).reshape(-1))
        teachers_used.append(str(model_name))
    if not predictions:
        raise ValueError("No teacher models could be fit for condition-transfer labels.")
    matrix = np.vstack(predictions)
    return matrix.mean(axis=0), matrix.std(axis=0), teachers_used


def _assign_labels(candidate_df: pd.DataFrame, config: ConditionTransferConfig) -> np.ndarray:
    if config.label_strategy in {"teacher_ensemble", "uncertainty_filtered_teacher"}:
        return candidate_df["teacher_mean"].to_numpy(dtype=float)
    if config.label_strategy == "source_label":
        return candidate_df["source_yield"].to_numpy(dtype=float)
    if config.label_strategy == "donor_label":
        return candidate_df["donor_yield"].to_numpy(dtype=float)
    if config.label_strategy == "average_label":
        return 0.5 * (
            candidate_df["source_yield"].to_numpy(dtype=float)
            + candidate_df["donor_yield"].to_numpy(dtype=float)
        )
    raise ValueError(f"Unsupported label strategy: {config.label_strategy}")


def _acceptance_mask(
    candidate_df: pd.DataFrame,
    X_synthetic: np.ndarray,
    config: ConditionTransferConfig,
) -> np.ndarray:
    del X_synthetic, config
    return candidate_df["rejection_reason"].isna().to_numpy(dtype=bool)


def _mark_duplicates(candidate_df: pd.DataFrame, real_reactions: set[str]) -> pd.DataFrame:
    result = candidate_df.copy()
    seen: set[str] = set()
    duplicate_flags: list[bool] = []
    existing_flags: list[bool] = []
    for reaction in result["reaction_smiles"].astype(str):
        duplicate_flags.append(reaction in seen)
        existing_flags.append(reaction in real_reactions)
        seen.add(reaction)
    result["synthetic_was_duplicate"] = duplicate_flags
    result["existing_real_duplicate"] = existing_flags
    return result


def _substrate_similarity_matrix(
    parsed: list[ParsedReaction | None],
    *,
    n_bits: int,
    radius: int,
    backend: str,
) -> np.ndarray:
    """Preserve the legacy summed-substrate cosine geometry explicitly."""
    fingerprints: list[np.ndarray] = []
    for item in parsed:
        if item is None:
            fingerprints.append(np.zeros(n_bits, dtype=np.float32))
            continue
        summed = np.zeros(n_bits, dtype=np.float32)
        for token in item.substrates:
            summed += morgan_fingerprint(
                token,
                radius=radius,
                n_bits=n_bits,
                warn_invalid=False,
                backend=backend,
            )
        fingerprints.append(summed)
    return _cosine_similarity_matrix(np.vstack(fingerprints).astype(np.float32))


def _token_group_similarity_matrix(
    token_groups: list[tuple[str, ...]],
    *,
    n_bits: int,
    radius: int,
    backend: str,
) -> np.ndarray:
    fingerprints: list[np.ndarray] = []
    for tokens in token_groups:
        summed = np.zeros(n_bits, dtype=np.float32)
        for token in tokens:
            summed += morgan_fingerprint(
                token,
                radius=radius,
                n_bits=n_bits,
                warn_invalid=False,
                backend=backend,
            )
        fingerprints.append(summed)
    return _cosine_similarity_matrix(np.vstack(fingerprints).astype(np.float32))


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = X / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _nearest_support_distance(
    X_candidates: np.ndarray,
    X_train: np.ndarray,
) -> np.ndarray:
    candidates = np.asarray(X_candidates, dtype=np.float64)
    train = np.asarray(X_train, dtype=np.float64)
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    train_norms = np.linalg.norm(train, axis=1, keepdims=True)
    normalized_candidates = candidates / np.maximum(candidate_norms, 1e-12)
    normalized_train = train / np.maximum(train_norms, 1e-12)
    similarities = np.clip(normalized_candidates @ normalized_train.T, -1.0, 1.0)
    return 1.0 - np.max(similarities, axis=1)


def _diversity_contribution(X_candidates: np.ndarray) -> np.ndarray:
    candidates = np.asarray(X_candidates, dtype=np.float64)
    if len(candidates) <= 1:
        return np.ones(len(candidates), dtype=float)
    similarities = _cosine_similarity_matrix(candidates)
    np.fill_diagonal(similarities, -np.inf)
    return 1.0 - np.max(similarities, axis=1)


def _uncertainty_ranking_proxy(
    candidate_df: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    teacher_std = candidate_df["teacher_std"].to_numpy(dtype=float)
    disagreement = 0.5 * np.abs(
        candidate_df["source_yield"].to_numpy(dtype=float)
        - candidate_df["donor_yield"].to_numpy(dtype=float)
    )
    raw = np.where(np.isfinite(teacher_std), teacher_std, disagreement)
    calibration_scale = max(float(np.std(np.asarray(y_train, dtype=float))), 1.0)
    basis = np.where(
        np.isfinite(teacher_std),
        "teacher_std_normalized_proxy_phase12_pending",
        "source_donor_disagreement_normalized_proxy_phase12_pending",
    )
    return raw / calibration_scale, basis


def _rank_accepted_candidates(candidate_df: pd.DataFrame) -> list[int]:
    accepted = candidate_df.loc[candidate_df["accepted"].astype(bool)].copy()
    if accepted.empty:
        return []
    accepted["_stable_canonical_key"] = accepted["canonical_reaction_key"].fillna("").astype(str)
    ranked = accepted.sort_values(
        by=[
            "uncertainty_rank_value",
            "relevant_context_similarity",
            "diversity_contribution",
            "out_of_support_distance",
            "_stable_canonical_key",
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    return [int(index) for index in ranked.index]


def _stable_training_row_key(
    row: dict[str, Any],
    canonical_reaction_key: str | None,
    yield_value: float,
) -> tuple[str, str, str]:
    row_id = row.get("reaction_id")
    stable_row_id = str(row_id) if row_id is not None and str(row_id) else ""
    return (
        str(canonical_reaction_key or row.get("reaction_smiles", "")),
        stable_row_id,
        f"{yield_value:.17g}",
    )


def _stable_seed(base_seed: int, key: tuple[str, str, str]) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("utf-8"))
    for value in key:
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _configured_max_candidates_per_source(config: ConditionTransferConfig) -> int:
    legacy_limit = max(1, int(config.candidates_per_real))
    if config.max_candidates_per_source is None:
        return legacy_limit
    return min(legacy_limit, max(1, int(config.max_candidates_per_source)))


def _metadata_from_candidates(
    candidate_df: pd.DataFrame,
    config: ConditionTransferConfig,
    n_real_train: int,
    invalid_parse_count: int,
    invalid_featurization_count: int,
    duplicate_synthetic_count: int,
    existing_real_duplicate_count: int,
    high_yield_fallback_count: int,
) -> dict[str, Any]:
    kept = candidate_df.loc[candidate_df.get("kept", False)] if not candidate_df.empty else candidate_df
    accepted = candidate_df.loc[candidate_df.get("accepted", False)] if not candidate_df.empty else candidate_df
    n_generated = int(len(candidate_df))
    n_accepted = int(len(accepted))
    return {
        "n_real_train": int(n_real_train),
        "n_candidates_generated": n_generated,
        "n_candidates_accepted": n_accepted,
        "n_synthetic_train": int(len(kept)),
        "filter_acceptance_rate": float(n_accepted / n_generated) if n_generated else 0.0,
        "donor_strategy": config.donor_strategy,
        "role_change_requirement": config.role_change_requirement,
        "fallback_policy": config.fallback_policy,
        "requested_roles": "|".join(
            _canonical_requested_roles(config.requested_roles)
        ),
        "label_strategy": config.label_strategy,
        "synthetic_multiplier": float(config.synthetic_multiplier),
        "n_neighbors": int(config.n_neighbors),
        "candidates_per_real": int(config.candidates_per_real),
        "max_candidates_per_source": _configured_max_candidates_per_source(config),
        "max_generated_per_source": (
            int(candidate_df["generated_per_source"].max())
            if not candidate_df.empty and "generated_per_source" in candidate_df
            else 0
        ),
        "source_cap_violation_count": (
            int(
                (
                    candidate_df.groupby("source_position").size()
                    > _configured_max_candidates_per_source(config)
                ).sum()
            )
            if not candidate_df.empty and "source_position" in candidate_df
            else 0
        ),
        "min_similarity": config.min_similarity,
        "max_teacher_std": config.max_teacher_std,
        "high_yield_threshold": float(config.high_yield_threshold),
        "invalid_parse_count": int(invalid_parse_count),
        "invalid_featurization_count": int(invalid_featurization_count),
        "duplicate_synthetic_count": int(duplicate_synthetic_count),
        "existing_real_duplicate_count": int(existing_real_duplicate_count),
        "high_yield_fallback_count": int(high_yield_fallback_count),
        "used_validation_or_test_parents": False,
        "mean_donor_similarity": _safe_mean(kept, "donor_similarity"),
        "mean_teacher_std": _safe_mean(kept, "teacher_std"),
        "mean_synthetic_yield": _safe_mean(kept, "synthetic_label"),
        "std_synthetic_yield": _safe_std(kept, "synthetic_label"),
        "min_synthetic_yield": _safe_min(kept, "synthetic_label"),
        "max_synthetic_yield": _safe_max(kept, "synthetic_label"),
    }


def _validate_config(config: ConditionTransferConfig) -> None:
    config.requested_roles = _canonical_requested_roles(config.requested_roles)
    if config.donor_strategy not in DONOR_STRATEGIES:
        raise ValueError(f"Unsupported donor strategy: {config.donor_strategy}")
    if config.label_strategy not in LABEL_STRATEGIES:
        raise ValueError(f"Unsupported label strategy: {config.label_strategy}")
    if config.role_change_requirement not in {"all", "any"}:
        raise ValueError("role_change_requirement must be one of: all, any")
    if config.fallback_policy not in FALLBACK_POLICIES:
        raise ValueError(
            "fallback_policy must be one of: "
            + ", ".join(sorted(FALLBACK_POLICIES))
        )
    if config.synthetic_multiplier < 0:
        raise ValueError("synthetic_multiplier must be non-negative.")
    if config.n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if config.candidates_per_real < 1:
        raise ValueError("candidates_per_real must be at least 1.")
    if (
        config.max_candidates_per_source is not None
        and config.max_candidates_per_source < 1
    ):
        raise ValueError("max_candidates_per_source must be at least 1.")
    if config.clip_y_min > config.clip_y_max:
        raise ValueError("clip_y_min must be less than or equal to clip_y_max.")


def _canonical_requested_roles(
    requested_roles: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Validate and normalize anonymous roles to canonical condition order."""
    if requested_roles is None:
        return ANONYMOUS_TRANSFER_ROLES
    if not isinstance(requested_roles, (tuple, list)) or not requested_roles:
        raise ValueError("requested_roles must be a nonempty tuple or list.")
    if any(not isinstance(role, str) for role in requested_roles):
        raise ValueError("requested_roles must contain only role names.")
    roles = tuple(requested_roles)
    if len(roles) != len(set(roles)):
        raise ValueError("requested_roles must not contain duplicates.")
    unsupported = sorted(set(roles) - set(ANONYMOUS_TRANSFER_ROLES))
    if unsupported:
        raise ValueError(
            "requested_roles contains unsupported anonymous condition roles: "
            + ", ".join(unsupported)
        )
    selected = set(roles)
    return tuple(
        role for role in ANONYMOUS_TRANSFER_ROLES if role in selected
    )


def _synthetic_columns() -> list[str]:
    return [
        "reaction_smiles",
        "source_index",
        "donor_index",
        "source_reaction_smiles",
        "donor_reaction_smiles",
        "source_yield",
        "donor_yield",
        "donor_strategy",
        "effective_donor_strategy",
        "fallback_policy",
        "fallback_used",
        "label_strategy",
        "donor_similarity",
        "substrate_similarity",
        "product_similarity",
        "nontransferred_role_similarity",
        "condition_similarity",
        "overall_similarity",
        "nearest_training_support_distance",
        "support_distance_metric",
        "support_distance_backend",
        "calibrated_uncertainty",
        "uncertainty_rank_value",
        "uncertainty_rank_basis",
        "relevant_context_similarity",
        "diversity_contribution",
        "out_of_support_distance",
        "candidate_rank",
        "source_candidate_ordinal",
        "generated_per_source",
        "max_candidates_per_source",
        "teacher_mean",
        "teacher_std",
        "synthetic_label",
        "source_product",
        "source_substrate_block",
        "donor_condition_block",
        "synthetic_was_duplicate",
        *REQUIRED_SYNTHETIC_AUDIT_FIELDS,
        "requested_roles",
        "actual_changed_roles",
        "unchanged_requested_roles",
        "unexpected_changed_roles",
        "change_mask",
        "role_change_requirement",
        "role_change_valid",
        "synthetic_identity_audit_version",
        "canonicalization_version",
        *CANONICAL_ROLE_COLUMNS,
    ]


def _empty_synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_synthetic_columns(), "yield"])


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_id",
            *_synthetic_columns(),
            "source_position",
            "donor_position",
            "existing_real_duplicate",
            "accepted",
            "kept",
        ]
    )


def _package_result(
    synthetic_df: pd.DataFrame,
    synthetic_y: np.ndarray,
    metadata: dict[str, Any],
    candidate_df: pd.DataFrame,
    X_synthetic: np.ndarray,
    feature_names: list[str],
    feature_metadata: FeatureMetadata,
) -> dict[str, Any]:
    return {
        "synthetic_df": synthetic_df,
        "synthetic_y": np.asarray(synthetic_y, dtype=float).reshape(-1),
        "X_synthetic": np.asarray(X_synthetic, dtype=np.float32),
        "metadata": dict(metadata),
        "candidate_df": candidate_df,
        "feature_names": list(feature_names),
        "feature_metadata": feature_metadata,
    }


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _safe_std(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").std(ddof=0))


def _safe_min(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").min())


def _safe_max(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").max())
