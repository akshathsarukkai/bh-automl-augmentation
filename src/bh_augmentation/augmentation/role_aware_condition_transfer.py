"""Role-aware Buchwald-Hartwig condition-transfer augmentation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    ROLE_CHANGE_REQUIREMENTS,
    apply_filter_rejection,
    assert_accepted_identity_invariants,
    assert_accepted_role_change_invariants,
    audit_candidate_identities,
    audit_role_changes,
    canonical_candidate_record,
    measured_canonical_keys,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    ROLE_TO_COLUMN,
    ReactionRoles,
    reaction_roles_from_row,
    reaction_roles_to_record,
)
from bh_augmentation.features.compatibility import (
    FeatureMetadata,
    assert_feature_compatibility,
)
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    morgan_fingerprint,
)
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model

ROLE_COLUMNS = ROLE_TO_COLUMN
CONDITION_ROLE_COLUMNS = [
    ROLE_COLUMNS["catalyst"],
    ROLE_COLUMNS["ligand"],
    ROLE_COLUMNS["base"],
    ROLE_COLUMNS["solvent_or_additive"],
]
ROLE_TRANSFER_MODES = {
    "catalyst_only": ["catalyst"],
    "ligand_only": ["ligand"],
    "base_only": ["base"],
    "solvent_or_additive_only": ["solvent_or_additive"],
    "catalyst_ligand": ["catalyst", "ligand"],
    "ligand_base": ["ligand", "base"],
    "ligand_solvent_or_additive": ["ligand", "solvent_or_additive"],
    "base_solvent_or_additive": ["base", "solvent_or_additive"],
    "catalyst_ligand_base": ["catalyst", "ligand", "base"],
    "ligand_base_solvent_or_additive": ["ligand", "base", "solvent_or_additive"],
    "all_transferable_roles": ["catalyst", "ligand", "base", "solvent_or_additive"],
    "full_condition_block": ["catalyst", "ligand", "base", "solvent_or_additive"],
}
DONOR_STRATEGIES = {
    "random",
    "nearest_substrate",
    "nearest_condition",
    "high_yield_nearest",
    "diverse_role_value",
    "same_substrate_different_role",
    "same_nontransferred_roles",
    "matched_product_or_reactant_key",
    "high_yield_same_context",
}
LABEL_STRATEGIES = {
    "teacher_ensemble",
    "uncertainty_filtered_teacher",
    "source_label",
    "average_source_donor_label",
}
FALLBACK_POLICIES = {"reject", "same_product", "nearest_substrate", "random"}
_TEACHER_MODEL_CACHE: dict[tuple[object, ...], list[Any]] = {}


def clear_role_aware_teacher_cache() -> None:
    """Discard fitted teachers before changing the real training subset."""
    _TEACHER_MODEL_CACHE.clear()


@dataclass
class RoleAwareConditionTransferConfig:
    """Configuration for role-aware condition-transfer augmentation."""

    role_transfer_mode: str
    donor_strategy: str
    label_strategy: str
    synthetic_multiplier: float
    max_candidates_per_source: int
    teacher_models: list[str]
    max_teacher_std: float | None
    min_similarity: float | None
    clip_y_min: float = 0.0
    clip_y_max: float = 100.0
    random_state: int = 0
    max_resample_attempts: int = 10
    effective_role_transfer_mode: str | None = None
    invariant_roles_excluded: str = ""
    n_unique_catalysts: int = 0
    n_unique_ligands: int = 0
    n_unique_bases: int = 0
    n_unique_solvents: int = 0
    donor_similarity_n_bits: int = 256
    donor_similarity_radius: int = 2
    donor_similarity_backend: str = "auto"
    role_change_requirement: str = "any"
    fallback_policy: str = "random"


def build_role_transferred_reaction_smiles(
    source_row: pd.Series | dict[str, Any],
    donor_row: pd.Series | dict[str, Any],
    role_transfer_mode: str,
) -> dict[str, Any]:
    """Build a synthetic reaction by replacing selected condition roles."""
    if role_transfer_mode not in ROLE_TRANSFER_MODES:
        raise ValueError(f"Unknown role_transfer_mode: {role_transfer_mode}")

    source = reaction_roles_from_row(_row_mapping(source_row))
    donor = reaction_roles_from_row(_row_mapping(donor_row))
    transferred_roles = set(ROLE_TRANSFER_MODES[role_transfer_mode])
    roles = ReactionRoles(
        reactant_1=source.reactant_1,
        reactant_2=source.reactant_2,
        catalyst=donor.catalyst if "catalyst" in transferred_roles else source.catalyst,
        ligand=donor.ligand if "ligand" in transferred_roles else source.ligand,
        base=donor.base if "base" in transferred_roles else source.base,
        solvent_or_additive=(
            donor.solvent_or_additive
            if "solvent_or_additive" in transferred_roles
            else source.solvent_or_additive
        ),
        product=source.product,
    )
    synthetic = {
        "synthetic_reactant_1_smiles": roles.reactant_1,
        "synthetic_reactant_2_smiles": roles.reactant_2,
        "synthetic_catalyst_smiles": roles.catalyst,
        "synthetic_ligand_smiles": roles.ligand,
        "synthetic_base_smiles": roles.base,
        "synthetic_solvent_or_additive_smiles": roles.solvent_or_additive,
        "synthetic_product_smiles": roles.product,
        "synthetic_reaction_smiles": roles.reaction_smiles(),
        "reaction_roles": roles,
        **reaction_roles_to_record(roles),
    }
    synthetic["changed_catalyst"] = roles.catalyst != source.catalyst
    synthetic["changed_ligand"] = roles.ligand != source.ligand
    synthetic["changed_base"] = roles.base != source.base
    synthetic["changed_solvent_or_additive"] = roles.solvent_or_additive != source.solvent_or_additive
    return synthetic


def generate_role_aware_condition_transfer_examples(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RoleAwareConditionTransferConfig,
    *,
    feature_config: dict[str, Any],
    real_feature_names: list[str],
    real_feature_metadata: FeatureMetadata,
    measured_identity_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Generate role-aware synthetic reactions using only training rows."""
    _validate_config(config)
    _validate_training_frame(df_train)
    all_measured_keys = measured_canonical_keys(
        df_train,
        additional_keys=measured_identity_keys,
    )
    train = df_train.reset_index(drop=False).rename(columns={"index": "_source_dataframe_index"})
    X_train_array = np.asarray(X_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(train) != len(X_train_array) or len(train) != len(y_train_array):
        raise ValueError("df_train, X_train, and y_train must contain the same number of rows.")

    n_real_train = len(train)
    target_count = int(math.ceil(max(0.0, config.synthetic_multiplier) * n_real_train))
    if target_count == 0 or n_real_train < 2:
        candidate_df = _empty_candidate_df()
        metadata = _metadata_from_candidates(candidate_df, train, config)
        return _package_result(
            _empty_synthetic_df(), np.empty(0), metadata, candidate_df,
            np.empty((0, X_train_array.shape[1])), real_feature_names, real_feature_metadata,
        )

    rng = np.random.default_rng(config.random_state)
    canonical_source_order = _canonical_position_order(train, y_train_array)
    effective_mode = _effective_role_transfer_mode(config)
    legacy_donor_context_similarity = _role_similarity_matrix(
        train,
        ["reactant_1", "reactant_2", "product"],
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    substrate_similarity = _role_similarity_matrix(
        train,
        ["reactant_1", "reactant_2"],
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    condition_similarity = _role_similarity_matrix(
        train,
        ["catalyst", "ligand", "base", "solvent_or_additive"],
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    product_similarity = _role_similarity_matrix(
        train,
        ["product"],
        n_bits=config.donor_similarity_n_bits,
        radius=config.donor_similarity_radius,
        backend=config.donor_similarity_backend,
    )
    nontransferred_roles = [
        role
        for role in ["catalyst", "ligand", "base", "solvent_or_additive"]
        if role not in ROLE_TRANSFER_MODES[effective_mode]
    ]
    nontransferred_role_similarity = (
        _role_similarity_matrix(
            train,
            nontransferred_roles,
            n_bits=config.donor_similarity_n_bits,
            radius=config.donor_similarity_radius,
            backend=config.donor_similarity_backend,
        )
        if nontransferred_roles
        else np.ones((n_real_train, n_real_train), dtype=np.float32)
    )
    role_values = {
        role: train[ROLE_COLUMNS[role]].astype(str).to_numpy()
        for role in ["catalyst", "ligand", "base", "solvent_or_additive"]
    }
    context_values = _context_values(train)
    train_records = train.to_dict("records")
    source_dataframe_indices = train["_source_dataframe_index"].to_numpy()
    np.fill_diagonal(legacy_donor_context_similarity, -np.inf)
    np.fill_diagonal(condition_similarity, -np.inf)

    rows: list[dict[str, Any]] = []
    n_identical_skipped = 0
    n_invalid_role_parse_skipped = 0
    n_same_context_donors_found = 0
    n_role_changed_candidates_found = 0
    per_source_cap = max(0, int(config.max_candidates_per_source))
    generated_per_source = {position: 0 for position in range(n_real_train)}
    candidate_id = 0
    for source_position in canonical_source_order:
        used_donors: set[int] = set()
        while generated_per_source[source_position] < per_source_cap:
            donor_position, donor_similarity, donor_fallback_level, donor_stats = _select_donor_position(
                train=train,
                source_position=source_position,
                y_train=y_train_array,
                substrate_similarity=legacy_donor_context_similarity,
                condition_similarity=condition_similarity,
                role_values=role_values,
                context_values=context_values,
                config=config,
                rng=rng,
                excluded_donor_positions=used_donors,
            )
            n_identical_skipped += donor_stats["n_source_identical_donors_rejected"]
            n_same_context_donors_found += donor_stats["n_same_context_donors_found"]
            n_role_changed_candidates_found += donor_stats["n_role_changed_candidates_found"]
            if donor_position is None:
                break
            used_donors.add(donor_position)

            synthetic = build_role_transferred_reaction_smiles(
                train_records[source_position],
                train_records[donor_position],
                effective_mode,
            )
            synthetic = {
                **synthetic,
                **canonical_candidate_record(synthetic["reaction_roles"]),
            }
            if synthetic["reaction_smiles"] == str(train_records[source_position]["reaction_smiles"]):
                n_identical_skipped += 1

            substrate_score = float(substrate_similarity[source_position, donor_position])
            product_score = float(product_similarity[source_position, donor_position])
            nontransferred_score = float(
                nontransferred_role_similarity[source_position, donor_position]
            )
            condition_score = float(condition_similarity[source_position, donor_position])
            relevant_context_score = float(
                np.mean([substrate_score, product_score, nontransferred_score])
            )
            overall_score = float(
                np.mean(
                    [
                        substrate_score,
                        product_score,
                        nontransferred_score,
                        condition_score,
                    ]
                )
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_position": source_position,
                    "donor_position": donor_position,
                    "source_index": source_dataframe_indices[source_position],
                    "donor_index": source_dataframe_indices[donor_position],
                    "source_yield": float(y_train_array[source_position]),
                    "donor_yield": float(y_train_array[donor_position]),
                    "donor_similarity": float(donor_similarity),
                    "legacy_donor_context_similarity": float(
                        legacy_donor_context_similarity[
                            source_position, donor_position
                        ]
                    ),
                    "substrate_similarity": substrate_score,
                    "product_similarity": product_score,
                    "nontransferred_role_similarity": nontransferred_score,
                    "condition_similarity": condition_score,
                    "overall_similarity": overall_score,
                    "relevant_context_similarity": relevant_context_score,
                    "diversity_contribution": float(
                        np.clip(1.0 - condition_score, 0.0, 2.0)
                    ),
                    "nearest_training_support_distance": np.nan,
                    "out_of_support_distance": np.nan,
                    "calibrated_uncertainty": np.nan,
                    "uncertainty_rank_value": np.nan,
                    "uncertainty_rank_basis": "teacher_std_proxy_phase12_pending",
                    "source_candidate_ordinal": (
                        generated_per_source[source_position] + 1
                    ),
                    "generated_per_source": 0,
                    "max_candidates_per_source": per_source_cap,
                    "candidate_rank": pd.NA,
                    "donor_fallback_level": donor_fallback_level,
                    "fallback_policy": config.fallback_policy,
                    "fallback_used": donor_fallback_level.startswith("fallback_"),
                    "teacher_mean": np.nan,
                    "teacher_std": np.nan,
                    "synthetic_label": np.nan,
                    "accepted": False,
                    "kept": False,
                    **synthetic,
                }
            )
            generated_per_source[source_position] += 1
            candidate_id += 1

    for row in rows:
        row["generated_per_source"] = generated_per_source[
            int(row["source_position"])
        ]

    candidate_df = pd.DataFrame(rows) if rows else _empty_candidate_df()
    if candidate_df.empty:
        metadata = _metadata_from_candidates(
            candidate_df,
            train,
            config,
            n_identical_skipped=n_identical_skipped,
            n_invalid_role_parse_skipped=n_invalid_role_parse_skipped,
            n_same_context_donors_found=n_same_context_donors_found,
            n_role_changed_candidates_found=n_role_changed_candidates_found,
        )
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
        requested_roles=ROLE_TRANSFER_MODES[effective_mode],
        role_change_requirement=config.role_change_requirement,
    )
    n_identical_skipped += int(candidate_df["source_identical"].astype(bool).sum())
    n_invalid_role_parse_skipped += int(
        (~candidate_df["chemical_parse_valid"].astype(bool)).sum()
    )
    teacher_order = np.asarray(canonical_source_order, dtype=int)
    teacher_mean, teacher_std, teachers_used = _teacher_predictions(
        X_train_array[teacher_order],
        y_train_array[teacher_order],
        X_synthetic,
        config,
    )
    candidate_df["teacher_mean"] = teacher_mean
    candidate_df["teacher_std"] = teacher_std
    # Ensemble spread is useful for ordering but is not calibrated uncertainty.
    candidate_df["calibrated_uncertainty"] = np.nan
    candidate_df["uncertainty_rank_value"] = teacher_std
    candidate_df["uncertainty_rank_basis"] = (
        "teacher_std_proxy_phase12_pending"
    )
    support_distance = _nearest_training_support_distance(X_synthetic, X_train_array)
    candidate_df["nearest_training_support_distance"] = support_distance
    candidate_df["support_distance_metric"] = "euclidean_distance"
    candidate_df["support_distance_backend"] = (
        f"{real_feature_metadata.representation_kind}:"
        f"{real_feature_metadata.fingerprint_backend}"
    )
    candidate_df["out_of_support_distance"] = support_distance
    candidate_df["teacher_models_used"] = ",".join(teachers_used)
    candidate_df["synthetic_label"] = np.clip(
        _assign_labels(candidate_df, config),
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
            candidate_df["donor_similarity"].to_numpy(dtype=float)
            < float(config.min_similarity),
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
    candidate_df = _rank_candidates(candidate_df)
    kept_indices = (
        candidate_df.loc[candidate_df["accepted"]]
        .sort_values("candidate_rank", kind="mergesort")
        .head(target_count)
        .index.to_numpy(dtype=int)
    )
    candidate_df.loc[kept_indices, "kept"] = True

    kept_candidates = candidate_df.loc[kept_indices].sort_values(
        "candidate_rank", kind="mergesort"
    )
    synthetic_df = kept_candidates[_synthetic_columns()].copy()
    synthetic_df["yield"] = kept_candidates["synthetic_label"].to_numpy(dtype=float)
    synthetic_y = synthetic_df["yield"].to_numpy(dtype=float)
    metadata = _metadata_from_candidates(
        candidate_df,
        train,
        config,
        n_identical_skipped=n_identical_skipped,
        n_invalid_role_parse_skipped=n_invalid_role_parse_skipped,
        n_same_context_donors_found=n_same_context_donors_found,
        n_role_changed_candidates_found=n_role_changed_candidates_found,
    )
    metadata["teacher_models_used"] = ",".join(teachers_used)
    return _package_result(
        synthetic_df.reset_index(drop=True),
        synthetic_y,
        metadata,
        candidate_df,
        X_synthetic[kept_candidates.index.to_numpy(dtype=int)].astype(np.float32),
        synthetic_feature_names,
        synthetic_feature_metadata,
    )


def _select_donor_position(
    train: pd.DataFrame,
    source_position: int,
    y_train: np.ndarray,
    substrate_similarity: np.ndarray,
    condition_similarity: np.ndarray,
    role_values: dict[str, np.ndarray],
    context_values: dict[str, np.ndarray],
    config: RoleAwareConditionTransferConfig,
    rng: np.random.Generator,
    excluded_donor_positions: set[int] | None = None,
) -> tuple[int | None, float, str, dict[str, int]]:
    excluded = excluded_donor_positions or set()
    candidates = [
        position
        for position in _canonical_position_order(train, y_train)
        if position != source_position and position not in excluded
    ]
    transferred_roles = ROLE_TRANSFER_MODES[_effective_role_transfer_mode(config)]
    role_changed_candidates = _filter_role_changed(
        candidates,
        source_position,
        role_values,
        transferred_roles,
        requirement=config.role_change_requirement,
    )
    stats = {
        "n_same_context_donors_found": 0,
        "n_role_changed_candidates_found": len(role_changed_candidates),
        "n_source_identical_donors_rejected": len(
            candidates
        )
        - len(
            _filter_role_changed(
                candidates,
                source_position,
                role_values,
                transferred_roles,
                requirement="any",
            )
        ),
    }

    if config.donor_strategy in {
        "same_substrate_different_role",
        "same_nontransferred_roles",
        "matched_product_or_reactant_key",
        "high_yield_same_context",
    }:
        return _select_context_aware_donor(
            candidates=role_changed_candidates,
            source_position=source_position,
            y_train=y_train,
            substrate_similarity=substrate_similarity,
            role_values=role_values,
            context_values=context_values,
            transferred_roles=transferred_roles,
            strategy=config.donor_strategy,
            fallback_policy=config.fallback_policy,
            min_similarity=config.min_similarity,
            rng=rng,
            stats=stats,
        )

    candidates = role_changed_candidates
    if not candidates:
        return None, float("nan"), "unavailable", stats

    if config.donor_strategy == "random":
        eligible = _similarity_eligible(
            candidates,
            source_position,
            substrate_similarity,
            config.min_similarity,
        )
        if not eligible:
            return _select_declared_fallback(
                candidates=candidates,
                source_position=source_position,
                y_train=y_train,
                substrate_similarity=substrate_similarity,
                context_values=context_values,
                fallback_policy=config.fallback_policy,
                min_similarity=config.min_similarity,
                rng=rng,
                stats=stats,
            )
        donor_position = int(rng.choice(eligible))
        return donor_position, float(substrate_similarity[source_position, donor_position]), "random", stats

    similarity = condition_similarity if config.donor_strategy == "nearest_condition" else substrate_similarity
    candidate_array = np.asarray(candidates, dtype=int)
    scores = similarity[source_position, candidate_array]
    if config.min_similarity is not None:
        keep = scores >= float(config.min_similarity)
        candidate_array = candidate_array[keep]
        scores = scores[keep]
    if len(candidate_array) == 0:
        return _select_declared_fallback(
            candidates=candidates,
            source_position=source_position,
            y_train=y_train,
            substrate_similarity=substrate_similarity,
            context_values=context_values,
            fallback_policy=config.fallback_policy,
            min_similarity=config.min_similarity,
            rng=rng,
            stats=stats,
        )

    if config.donor_strategy == "high_yield_nearest":
        order = np.lexsort((-scores, -y_train[candidate_array]))
        donor_position = int(candidate_array[order[0]])
        return donor_position, float(similarity[source_position, donor_position]), "high_yield_nearest", stats

    best_index = int(np.argmax(scores))
    donor_position = int(candidate_array[best_index])
    return donor_position, float(scores[best_index]), config.donor_strategy, stats


def _select_context_aware_donor(
    candidates: list[int],
    source_position: int,
    y_train: np.ndarray,
    substrate_similarity: np.ndarray,
    role_values: dict[str, np.ndarray],
    context_values: dict[str, np.ndarray],
    transferred_roles: list[str],
    strategy: str,
    fallback_policy: str,
    min_similarity: float | None,
    rng: np.random.Generator,
    stats: dict[str, int],
) -> tuple[int | None, float, str, dict[str, int]]:
    if not candidates:
        return None, float("nan"), "unavailable", stats

    nontransferred_roles = [
        role for role in ["catalyst", "ligand", "base", "solvent_or_additive"] if role not in transferred_roles
    ]
    same_nontransferred = [
        position
        for position in candidates
        if all(role_values[role][position] == role_values[role][source_position] for role in nontransferred_roles)
    ]
    same_reactant = [
        position
        for position in candidates
        if context_values["reactant_key"][position] == context_values["reactant_key"][source_position]
    ]
    same_product = [
        position
        for position in candidates
        if context_values["product_key"][position] == context_values["product_key"][source_position]
    ]
    same_product_or_reactant_set = set(same_product) | set(same_reactant)
    same_product_or_reactant = [
        position for position in candidates if position in same_product_or_reactant_set
    ]

    if strategy == "same_substrate_different_role":
        levels = [("strict_same_reactant_key", same_reactant)]
    elif strategy == "same_nontransferred_roles":
        levels = [("strict_same_nontransferred_roles", same_nontransferred)]
    elif strategy == "matched_product_or_reactant_key":
        levels = [("strict_matched_product_or_reactant_key", same_product_or_reactant)]
    else:
        levels = [("strict_same_nontransferred_roles", same_nontransferred)]

    for fallback_level, level_candidates in levels:
        unique_candidates = list(dict.fromkeys(level_candidates))
        unique_candidates = _similarity_eligible(
            unique_candidates,
            source_position,
            substrate_similarity,
            min_similarity,
        )
        if not unique_candidates:
            continue
        stats["n_same_context_donors_found"] += len(unique_candidates)
        donor_position = _choose_from_candidates(
            unique_candidates,
            source_position=source_position,
            y_train=y_train,
            substrate_similarity=substrate_similarity,
            prefer_high_yield=strategy == "high_yield_same_context",
            random_choice=False,
            rng=rng,
        )
        return donor_position, float(substrate_similarity[source_position, donor_position]), fallback_level, stats
    return _select_declared_fallback(
        candidates=candidates,
        source_position=source_position,
        y_train=y_train,
        substrate_similarity=substrate_similarity,
        context_values=context_values,
        fallback_policy=fallback_policy,
        min_similarity=min_similarity,
        rng=rng,
        stats=stats,
    )


def _select_declared_fallback(
    candidates: list[int],
    source_position: int,
    y_train: np.ndarray,
    substrate_similarity: np.ndarray,
    context_values: dict[str, np.ndarray],
    fallback_policy: str,
    min_similarity: float | None,
    rng: np.random.Generator,
    stats: dict[str, int],
) -> tuple[int | None, float, str, dict[str, int]]:
    """Apply exactly one named fallback to role-valid donor candidates."""
    if fallback_policy == "reject" or not candidates:
        return None, float("nan"), "fallback_reject", stats
    fallback_candidates = list(dict.fromkeys(candidates))
    if fallback_policy == "same_product":
        fallback_candidates = [
            position
            for position in fallback_candidates
            if context_values["product_key"][position]
            == context_values["product_key"][source_position]
        ]
        if not fallback_candidates:
            return None, float("nan"), "fallback_same_product", stats
        stats["n_same_context_donors_found"] += len(fallback_candidates)
    fallback_candidates = _similarity_eligible(
        fallback_candidates,
        source_position,
        substrate_similarity,
        min_similarity,
    )
    if not fallback_candidates:
        return None, float("nan"), f"fallback_{fallback_policy}", stats
    donor_position = _choose_from_candidates(
        fallback_candidates,
        source_position=source_position,
        y_train=y_train,
        substrate_similarity=substrate_similarity,
        prefer_high_yield=False,
        random_choice=fallback_policy == "random",
        rng=rng,
    )
    return (
        donor_position,
        float(substrate_similarity[source_position, donor_position]),
        f"fallback_{fallback_policy}",
        stats,
    )


def _choose_from_candidates(
    candidates: list[int],
    source_position: int,
    y_train: np.ndarray,
    substrate_similarity: np.ndarray,
    prefer_high_yield: bool,
    random_choice: bool,
    rng: np.random.Generator,
) -> int:
    if random_choice:
        return int(rng.choice(candidates))
    candidate_array = np.asarray(candidates, dtype=int)
    scores = substrate_similarity[source_position, candidate_array]
    if prefer_high_yield:
        order = np.lexsort((-scores, -y_train[candidate_array]))
    else:
        order = np.argsort(-scores, kind="mergesort")
    return int(candidate_array[order[0]])


def _filter_role_changed(
    candidates: list[int],
    source_position: int,
    role_values: dict[str, np.ndarray],
    transferred_roles: list[str],
    requirement: str,
) -> list[int]:
    predicate = all if requirement == "all" else any
    return [
        position
        for position in candidates
        if predicate(
            role_values[role][position] != role_values[role][source_position]
            for role in transferred_roles
        )
    ]


def _canonical_position_order(train: pd.DataFrame, y_train: np.ndarray) -> list[int]:
    """Return a row-order-independent traversal without changing public positions."""
    role_columns = [ROLE_COLUMNS[role] for role in ROLE_COLUMNS]
    reaction_ids = (
        train["reaction_id"].astype(str).to_numpy()
        if "reaction_id" in train
        else np.full(len(train), "", dtype=object)
    )
    role_values = [
        train[column].astype(str).to_numpy()
        for column in role_columns
    ]
    reactions = train["reaction_smiles"].astype(str).to_numpy()
    yields = np.asarray(y_train, dtype=float)

    def canonical_key(position: int) -> tuple[str, ...]:
        return (
            *(values[position] for values in role_values),
            reactions[position],
            reaction_ids[position],
            float(yields[position]).hex(),
        )

    return sorted(range(len(train)), key=canonical_key)


def _similarity_eligible(
    candidates: list[int],
    source_position: int,
    similarity: np.ndarray,
    min_similarity: float | None,
) -> list[int]:
    if min_similarity is None:
        return list(candidates)
    threshold = float(min_similarity)
    return [
        position
        for position in candidates
        if float(similarity[source_position, position]) >= threshold
    ]


def _role_similarity_matrix(
    train: pd.DataFrame,
    roles: list[str],
    *,
    n_bits: int,
    radius: int,
    backend: str,
) -> np.ndarray:
    """Preserve the legacy donor geometry: sum role fingerprints, then cosine.

    This helper is intentionally separate from scientific model featurization.
    Defaults in ``RoleAwareConditionTransferConfig`` retain the historical
    radius-2, 256-bit summed-role donor ranking behavior.
    """
    vectors = []
    for _, row in train.iterrows():
        fingerprint = np.zeros(n_bits, dtype=np.float32)
        for role in roles:
            fingerprint += _cached_similarity_fingerprint(
                str(row[ROLE_COLUMNS[role]]),
                radius,
                n_bits,
                backend,
            )
        vectors.append(fingerprint)
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return normalized @ normalized.T


@lru_cache(maxsize=8192)
def _cached_similarity_fingerprint(
    smiles: str,
    radius: int,
    n_bits: int,
    backend: str,
) -> np.ndarray:
    return morgan_fingerprint(
        smiles,
        radius=radius,
        n_bits=n_bits,
        warn_invalid=False,
        backend=backend,
    )


def _context_values(train: pd.DataFrame) -> dict[str, np.ndarray]:
    reactant_pair = (
        train[ROLE_COLUMNS["reactant_1"]].astype(str)
        + "."
        + train[ROLE_COLUMNS["reactant_2"]].astype(str)
    )
    reactant_key = train["reactant_key"].astype(str) if "reactant_key" in train else reactant_pair
    product_key = train["product_key"].astype(str) if "product_key" in train else train[ROLE_COLUMNS["product"]].astype(str)
    return {
        "reactant_key": reactant_key.to_numpy(),
        "product_key": product_key.to_numpy(),
    }


def _effective_role_transfer_mode(config: RoleAwareConditionTransferConfig) -> str:
    return config.effective_role_transfer_mode or config.role_transfer_mode


def _teacher_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synthetic: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(X_synthetic) == 0:
        return np.empty(0), np.empty(0), []
    predictions: list[np.ndarray] = []
    teachers = _fit_teacher_models(X_train, y_train, config)
    for fitted in teachers:
        predictions.append(np.asarray(predict_model(fitted, X_synthetic), dtype=float))
    if not predictions:
        raise ValueError("At least one teacher model is required for teacher label strategies.")
    stacked = np.vstack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0), list(config.teacher_models)


def _fit_teacher_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> list[Any]:
    key = (
        id(X_train),
        id(y_train),
        tuple(X_train.shape),
        tuple(y_train.shape),
        float(np.sum(y_train)),
        tuple(config.teacher_models),
        int(config.random_state),
    )
    if key not in _TEACHER_MODEL_CACHE:
        fitted_models = []
        for model_name in config.teacher_models:
            model = get_model(model_name, seed=config.random_state)
            fitted_models.append(train_model(model, X_train, y_train))
        _TEACHER_MODEL_CACHE[key] = fitted_models
    return _TEACHER_MODEL_CACHE[key]


def _assign_labels(candidate_df: pd.DataFrame, config: RoleAwareConditionTransferConfig) -> np.ndarray:
    if config.label_strategy in {"teacher_ensemble", "uncertainty_filtered_teacher"}:
        return candidate_df["teacher_mean"].to_numpy(dtype=float)
    if config.label_strategy == "source_label":
        return candidate_df["source_yield"].to_numpy(dtype=float)
    if config.label_strategy == "average_source_donor_label":
        return 0.5 * (
            candidate_df["source_yield"].to_numpy(dtype=float)
            + candidate_df["donor_yield"].to_numpy(dtype=float)
        )
    raise ValueError(f"Unknown label_strategy: {config.label_strategy}")


def _acceptance_mask(
    candidate_df: pd.DataFrame,
    X_synthetic: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> np.ndarray:
    del X_synthetic, config
    return candidate_df["rejection_reason"].isna().to_numpy(dtype=bool)


def _nearest_training_support_distance(
    X_synthetic: np.ndarray,
    X_train: np.ndarray,
) -> np.ndarray:
    synthetic = np.asarray(X_synthetic, dtype=np.float32)
    training = np.asarray(X_train, dtype=np.float32)
    if len(synthetic) == 0:
        return np.empty(0, dtype=float)
    if len(training) == 0:
        return np.full(len(synthetic), np.inf, dtype=float)
    result = np.empty(len(synthetic), dtype=float)
    for start in range(0, len(synthetic), 256):
        stop = min(start + 256, len(synthetic))
        deltas = synthetic[start:stop, None, :] - training[None, :, :]
        squared = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
        result[start:stop] = np.sqrt(np.min(squared, axis=1))
    return result


def _rank_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Rank independently of generation order using the declared Phase-3 keys."""
    result = candidate_df.copy()
    ranking = pd.DataFrame(index=result.index)
    calibrated = pd.to_numeric(
        result["calibrated_uncertainty"], errors="coerce"
    )
    proxy = pd.to_numeric(
        result["uncertainty_rank_value"], errors="coerce"
    )
    ranking["_uncertainty"] = calibrated.where(calibrated.notna(), proxy).fillna(np.inf)
    ranking["_context"] = pd.to_numeric(
        result["relevant_context_similarity"], errors="coerce"
    ).fillna(-np.inf)
    ranking["_diversity"] = pd.to_numeric(
        result["diversity_contribution"], errors="coerce"
    ).fillna(-np.inf)
    ranking["_support"] = pd.to_numeric(
        result["out_of_support_distance"], errors="coerce"
    ).fillna(np.inf)
    ranking["_canonical_key"] = result["canonical_reaction_key"].fillna("~").astype(str)
    ranking["_source_key"] = result.get(
        "source_row_id", pd.Series("", index=result.index)
    ).fillna("").astype(str)
    ranking["_donor_key"] = result.get(
        "donor_row_id", pd.Series("", index=result.index)
    ).fillna("").astype(str)
    ordered_indices = ranking.sort_values(
        [
            "_uncertainty",
            "_context",
            "_diversity",
            "_support",
            "_canonical_key",
            "_source_key",
            "_donor_key",
        ],
        ascending=[True, False, False, True, True, True, True],
        kind="mergesort",
    ).index
    ranks = pd.Series(
        np.arange(1, len(result) + 1, dtype=int),
        index=ordered_indices,
        dtype="Int64",
    )
    result["candidate_rank"] = ranks.reindex(result.index)
    return result


def _metadata_from_candidates(
    candidate_df: pd.DataFrame,
    train: pd.DataFrame,
    config: RoleAwareConditionTransferConfig,
    n_identical_skipped: int = 0,
    n_invalid_role_parse_skipped: int = 0,
    n_same_context_donors_found: int = 0,
    n_role_changed_candidates_found: int = 0,
) -> dict[str, Any]:
    kept = candidate_df.loc[candidate_df.get("kept", False).eq(True)] if not candidate_df.empty else candidate_df
    accepted_count = int(candidate_df.get("accepted", pd.Series(dtype=bool)).sum()) if not candidate_df.empty else 0
    teacher_std = kept["teacher_std"].to_numpy(dtype=float) if not kept.empty else np.array([])
    synthetic_y = kept["synthetic_label"].to_numpy(dtype=float) if not kept.empty else np.array([])
    uncertainty_rejected = 0
    if (
        not candidate_df.empty
        and config.label_strategy == "uncertainty_filtered_teacher"
        and config.max_teacher_std is not None
    ):
        uncertainty_rejected = int((candidate_df["teacher_std"].to_numpy(dtype=float) > float(config.max_teacher_std)).sum())
    return {
        "role_transfer_mode": config.role_transfer_mode,
        "effective_role_transfer_mode": _effective_role_transfer_mode(config),
        "donor_strategy": config.donor_strategy,
        "role_change_requirement": config.role_change_requirement,
        "fallback_policy": config.fallback_policy,
        "label_strategy": config.label_strategy,
        "n_unique_catalysts": int(config.n_unique_catalysts),
        "n_unique_ligands": int(config.n_unique_ligands),
        "n_unique_bases": int(config.n_unique_bases),
        "n_unique_solvents": int(config.n_unique_solvents),
        "invariant_roles_excluded": config.invariant_roles_excluded,
        "synthetic_multiplier": float(config.synthetic_multiplier),
        "min_similarity": np.nan if config.min_similarity is None else float(config.min_similarity),
        "max_teacher_std": np.nan if config.max_teacher_std is None else float(config.max_teacher_std),
        "n_candidates_generated": int(len(candidate_df)),
        "n_candidates_accepted": accepted_count,
        "n_synthetic_train": int(len(kept)),
        "max_candidates_per_source": int(config.max_candidates_per_source),
        "max_generated_per_source": (
            int(candidate_df["generated_per_source"].max())
            if "generated_per_source" in candidate_df and not candidate_df.empty
            else 0
        ),
        "source_cap_violation_count": (
            int(
                (
                    candidate_df.groupby("source_position").size()
                    > int(config.max_candidates_per_source)
                ).sum()
            )
            if "source_position" in candidate_df and not candidate_df.empty
            else 0
        ),
        "filter_acceptance_rate": float(accepted_count / len(candidate_df)) if len(candidate_df) else 0.0,
        "mean_donor_similarity": _safe_mean(candidate_df.get("donor_similarity", pd.Series(dtype=float))),
        "mean_teacher_std": _safe_mean(teacher_std),
        "mean_synthetic_yield": _safe_mean(synthetic_y),
        "std_synthetic_yield": _safe_std(synthetic_y),
        "n_identical_skipped": int(n_identical_skipped),
        "n_invalid_role_parse_skipped": int(n_invalid_role_parse_skipped),
        "n_teacher_uncertainty_rejected": int(uncertainty_rejected),
        "n_zero_synthetic_policy_skipped": int(len(kept) == 0),
        "donor_fallback_level": _most_common(kept, "donor_fallback_level"),
        "n_fallback_used": (
            int(candidate_df["fallback_used"].astype(bool).sum())
            if "fallback_used" in candidate_df
            else 0
        ),
        "n_same_context_donors_found": int(n_same_context_donors_found),
        "n_role_changed_candidates_found": int(n_role_changed_candidates_found),
        "unique_source_catalysts": int(train[ROLE_COLUMNS["catalyst"]].nunique(dropna=False)),
        "unique_source_ligands": int(train[ROLE_COLUMNS["ligand"]].nunique(dropna=False)),
        "unique_source_bases": int(train[ROLE_COLUMNS["base"]].nunique(dropna=False)),
        "unique_source_solvents": int(train[ROLE_COLUMNS["solvent_or_additive"]].nunique(dropna=False)),
        "unique_synthetic_catalysts": _safe_nunique(kept, "synthetic_catalyst_smiles"),
        "unique_synthetic_ligands": _safe_nunique(kept, "synthetic_ligand_smiles"),
        "unique_synthetic_bases": _safe_nunique(kept, "synthetic_base_smiles"),
        "unique_synthetic_solvents": _safe_nunique(kept, "synthetic_solvent_or_additive_smiles"),
        "changed_catalyst_fraction": _safe_bool_mean(kept, "changed_catalyst"),
        "changed_ligand_fraction": _safe_bool_mean(kept, "changed_ligand"),
        "changed_base_fraction": _safe_bool_mean(kept, "changed_base"),
        "changed_solvent_fraction": _safe_bool_mean(kept, "changed_solvent_or_additive"),
        "changed_any_transferred_role_fraction": _changed_any_transferred_role_fraction(kept, _effective_role_transfer_mode(config)),
    }


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
        "synthetic_y": synthetic_y,
        "metadata": metadata,
        "candidate_df": candidate_df,
        "X_synthetic": X_synthetic,
        "feature_names": list(feature_names),
        "feature_metadata": feature_metadata,
    }


def _synthetic_columns() -> list[str]:
    return [
        "reaction_smiles",
        "source_index",
        "donor_index",
        "source_yield",
        "donor_yield",
        "donor_similarity",
        "legacy_donor_context_similarity",
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
        "donor_fallback_level",
        "fallback_policy",
        "fallback_used",
        "teacher_mean",
        "teacher_std",
        "synthetic_label",
        "synthetic_reactant_1_smiles",
        "synthetic_reactant_2_smiles",
        "synthetic_catalyst_smiles",
        "synthetic_ligand_smiles",
        "synthetic_base_smiles",
        "synthetic_solvent_or_additive_smiles",
        "synthetic_product_smiles",
        "changed_catalyst",
        "changed_ligand",
        "changed_base",
        "changed_solvent_or_additive",
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


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_synthetic_columns(), "candidate_id", "source_position", "donor_position", "accepted", "kept"])


def _empty_synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_synthetic_columns(), "yield"])


def _validate_config(config: RoleAwareConditionTransferConfig) -> None:
    if config.role_transfer_mode not in ROLE_TRANSFER_MODES:
        raise ValueError(f"Unknown role_transfer_mode: {config.role_transfer_mode}")
    if config.donor_strategy not in DONOR_STRATEGIES:
        raise ValueError(f"Unknown donor_strategy: {config.donor_strategy}")
    if config.label_strategy not in LABEL_STRATEGIES:
        raise ValueError(f"Unknown label_strategy: {config.label_strategy}")
    if config.role_change_requirement not in ROLE_CHANGE_REQUIREMENTS:
        raise ValueError(
            "role_change_requirement must be one of: "
            + ", ".join(sorted(ROLE_CHANGE_REQUIREMENTS))
        )
    if config.fallback_policy not in FALLBACK_POLICIES:
        raise ValueError(
            "fallback_policy must be one of: "
            + ", ".join(sorted(FALLBACK_POLICIES))
        )
    if config.max_candidates_per_source <= 0:
        raise ValueError("max_candidates_per_source must be greater than zero.")
    if not config.teacher_models and config.label_strategy in {"teacher_ensemble", "uncertainty_filtered_teacher"}:
        raise ValueError("teacher_models must be non-empty for teacher label strategies.")


def _validate_training_frame(df: pd.DataFrame) -> None:
    missing = [column for column in [*ROLE_COLUMNS.values(), "reaction_smiles"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing recovered role columns for role-aware condition transfer: {missing}")


def _row_mapping(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return row.to_dict() if isinstance(row, pd.Series) else dict(row)


def _safe_mean(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def _safe_std(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=0)) if len(array) else float("nan")


def _safe_nunique(df: pd.DataFrame, column: str) -> int:
    return int(df[column].nunique(dropna=False)) if column in df and not df.empty else 0


def _safe_bool_mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].astype(bool).mean()) if column in df and not df.empty else float("nan")


def _most_common(df: pd.DataFrame, column: str) -> str:
    if column not in df or df.empty:
        return ""
    counts = df[column].astype(str).value_counts()
    return str(counts.index[0]) if not counts.empty else ""


def _changed_any_transferred_role_fraction(df: pd.DataFrame, mode: str) -> float:
    if df.empty:
        return float("nan")
    columns = [
        f"changed_{role}" if role != "solvent_or_additive" else "changed_solvent_or_additive"
        for role in ROLE_TRANSFER_MODES[mode]
    ]
    existing = [column for column in columns if column in df]
    if not existing:
        return float("nan")
    return float(df[existing].astype(bool).any(axis=1).mean())
