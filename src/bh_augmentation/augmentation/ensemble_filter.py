"""Teacher-ensemble uncertainty filtering for recombined reactions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_recombine import (
    CANDIDATE_AUDIT_ATTR,
    _updated_candidate_audit,
    _with_candidate_audit,
    filter_candidates_by_nearest_neighbor_similarity,
    generate_condition_recombined_candidates,
)
from bh_augmentation.augmentation.synthetic_identity import (
    apply_filter_rejection,
    assert_accepted_identity_invariants,
)
from bh_augmentation.data.reaction_roles import ensure_reaction_role_columns
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model


def condition_recombine_ensemble_filter(
    train_df: pd.DataFrame,
    feature_config: dict[str, Any],
    config: dict[str, Any],
    random_state: int,
    cache: dict[object, object] | None = None,
    *,
    measured_identity_keys: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return real rows plus ensemble-labeled candidates passing all filters."""
    cache = cache if cache is not None else {}
    role_train = ensure_reaction_role_columns(train_df, parse_if_missing=True)
    multiplier = float(config.get("synthetic_multiplier", 1.0))
    max_rows = config.get("max_synthetic_rows", 3000)
    scored_key = ("scored_candidates", multiplier, max_rows, int(random_state))
    if scored_key in cache:
        scored = cache[scored_key].copy()
    else:
        candidates = generate_condition_recombined_candidates(
            role_train,
            synthetic_multiplier=multiplier,
            max_synthetic_rows=None if max_rows is None else int(max_rows),
            random_state=random_state,
            feature_config=feature_config,
            measured_identity_keys=measured_identity_keys,
        )
        candidates = filter_candidates_by_nearest_neighbor_similarity(
            role_train,
            candidates,
            feature_config,
            min_similarity=0.0,
        )
        scored = add_teacher_ensemble_predictions(
            role_train,
            candidates,
            feature_config,
            list(config.get("teachers", [])),
            cache=cache,
        )
        cache[scored_key] = scored.copy()

    accepted = filter_ensemble_candidates(scored, config)
    accepted = assign_ensemble_pseudo_labels(
        accepted, config.get("pseudo_label", {})
    )
    accepted["is_synthetic"] = True
    accepted["is_augmented"] = True
    accepted["augmentation_method"] = "condition_recombine_ensemble_filter"
    accepted["augmentation_type"] = "condition_recombine_ensemble_filter"
    accepted["pseudo_label"] = True
    accepted["pseudo_label_model"] = "teacher_ensemble_mean"

    real = role_train.copy()
    real["is_synthetic"] = False
    real["is_augmented"] = False
    real["augmentation_method"] = "real"
    real["augmentation_type"] = "real"
    real["pseudo_label"] = False
    if "source_reaction_id" not in real.columns:
        real["source_reaction_id"] = real.get(
            "reaction_id", pd.Series(real.index, index=real.index)
        ).astype(str)

    metadata = _ensemble_metadata(scored, accepted)
    audit = accepted.attrs.get(CANDIDATE_AUDIT_ATTR, pd.DataFrame()).copy()
    metadata["rejection_reason_counts"] = (
        {
            str(reason): int(count)
            for reason, count in audit["rejection_reason"]
            .fillna("accepted")
            .value_counts()
            .sort_index()
            .items()
        }
        if "rejection_reason" in audit
        else {}
    )
    combined = pd.concat([real, accepted], ignore_index=True, sort=False)
    combined.attrs["augmentation_metadata"] = metadata
    combined.attrs[CANDIDATE_AUDIT_ATTR] = audit
    return combined, metadata


def add_teacher_ensemble_predictions(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict[str, Any],
    teachers: list[dict[str, Any]],
    cache: dict[object, object] | None = None,
) -> pd.DataFrame:
    """Add teacher prediction summary columns to synthetic candidates."""
    if not teachers:
        raise ValueError("Ensemble filtering requires at least one teacher.")
    if candidate_df.empty:
        return _empty_prediction_columns(candidate_df)

    cache = cache if cache is not None else {}
    teacher_key = (
        "teachers",
        tuple(
            (str(item.get("model", "")), int(item.get("random_state", 0)))
            for item in teachers
        ),
    )
    fitted_teachers = cache.get(teacher_key)
    if fitted_teachers is None:
        train_x, train_y, _ = build_feature_matrix(train_df, feature_config)
        fitted_teachers = []
        for teacher_config in teachers:
            model_name = str(teacher_config.get("model", ""))
            if not model_name:
                raise ValueError("Each ensemble teacher must define model.")
            teacher_seed = int(teacher_config.get("random_state", 0))
            model = get_model(model_name, seed=teacher_seed)
            fitted_teachers.append(train_model(model, train_x, train_y))
        cache[teacher_key] = fitted_teachers

    candidate_data = candidate_df.copy()
    candidate_data["yield"] = 0.0
    candidate_x, _, _ = build_feature_matrix(candidate_data, feature_config)
    predictions = np.vstack(
        [predict_model(model, candidate_x) for model in fitted_teachers]
    )
    result = candidate_df.copy()
    result["teacher_mean_prediction"] = predictions.mean(axis=0)
    result["teacher_std_prediction"] = predictions.std(axis=0, ddof=0)
    result["teacher_min_prediction"] = predictions.min(axis=0)
    result["teacher_max_prediction"] = predictions.max(axis=0)
    result["teacher_prediction_range"] = (
        result["teacher_max_prediction"] - result["teacher_min_prediction"]
    )
    if CANDIDATE_AUDIT_ATTR in candidate_df.attrs:
        result.attrs[CANDIDATE_AUDIT_ATTR] = candidate_df.attrs[
            CANDIDATE_AUDIT_ATTR
        ].copy()
    return result


def filter_ensemble_candidates(
    candidate_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Keep candidates passing nearest-neighbor and teacher uncertainty limits."""
    result = candidate_df.copy()
    if result.empty:
        result["accepted_by_similarity_filter"] = pd.Series(dtype=bool)
        result["accepted_by_uncertainty_filter"] = pd.Series(dtype=bool)
        return result

    minimum_similarity = float(config.get("min_neighbor_similarity", 0.8))
    uncertainty = config.get("uncertainty_filter", {})
    uncertainty_enabled = bool(uncertainty.get("enabled", True))
    max_std = float(uncertainty.get("max_teacher_std", float("inf")))
    max_range = float(uncertainty.get("max_prediction_range", float("inf")))
    result["accepted_by_similarity_filter"] = result[
        "nearest_train_similarity"
    ].ge(minimum_similarity)
    result["accepted_by_uncertainty_filter"] = True
    if uncertainty_enabled:
        result["accepted_by_uncertainty_filter"] = (
            result["teacher_std_prediction"].le(max_std)
            & result["teacher_prediction_range"].le(max_range)
        )
    accepted_mask = (
        result["accepted_by_similarity_filter"]
        & result["accepted_by_uncertainty_filter"]
    )
    if "rejection_reason" not in result.columns:
        return result.loc[accepted_mask].reset_index(drop=True)

    result = apply_filter_rejection(
        result,
        ~result["accepted_by_similarity_filter"].to_numpy(dtype=bool),
        "nearest_train_similarity_below_threshold",
    )
    result = apply_filter_rejection(
        result,
        ~result["accepted_by_uncertainty_filter"].to_numpy(dtype=bool),
        "teacher_uncertainty_above_threshold",
    )
    result["accepted"] = result["rejection_reason"].isna()
    assert_accepted_identity_invariants(result)
    audit = _updated_candidate_audit(result)
    accepted = result.loc[result["accepted"]].reset_index(drop=True)
    return _with_candidate_audit(accepted, audit)


def assign_ensemble_pseudo_labels(
    candidate_df: pd.DataFrame,
    pseudo_label_config: dict[str, Any],
) -> pd.DataFrame:
    """Assign clipped ensemble-mean pseudo-labels to accepted candidates."""
    result = candidate_df.copy()
    mode = str(pseudo_label_config.get("mode", "ensemble_mean"))
    if mode != "ensemble_mean":
        raise ValueError("Ensemble pseudo_label.mode must be 'ensemble_mean'.")
    minimum = float(pseudo_label_config.get("clip_min", 0.0))
    maximum = float(pseudo_label_config.get("clip_max", 100.0))
    if maximum < minimum:
        raise ValueError("pseudo_label clip_max must be at least clip_min.")
    result["yield"] = result["teacher_mean_prediction"].clip(minimum, maximum)
    if CANDIDATE_AUDIT_ATTR in candidate_df.attrs:
        result.attrs[CANDIDATE_AUDIT_ATTR] = candidate_df.attrs[
            CANDIDATE_AUDIT_ATTR
        ].copy()
    return result


def _ensemble_metadata(
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
) -> dict[str, Any]:
    generated = len(candidates)
    accepted_count = len(accepted)
    return {
        "n_candidates_generated": generated,
        "n_candidates_accepted": accepted_count,
        "acceptance_rate": float(accepted_count / generated) if generated else 0.0,
        "mean_teacher_std": _mean_or_nan(accepted.get("teacher_std_prediction")),
        "mean_prediction_range": _mean_or_nan(
            accepted.get("teacher_prediction_range")
        ),
        "mean_nearest_train_similarity": _mean_or_nan(
            accepted.get("nearest_train_similarity")
        ),
    }


def _mean_or_nan(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").mean())


def _empty_prediction_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    result = candidate_df.copy()
    for column in [
        "teacher_mean_prediction",
        "teacher_std_prediction",
        "teacher_min_prediction",
        "teacher_max_prediction",
        "teacher_prediction_range",
    ]:
        result[column] = pd.Series(dtype=float)
    if CANDIDATE_AUDIT_ATTR in candidate_df.attrs:
        result.attrs[CANDIDATE_AUDIT_ATTR] = candidate_df.attrs[
            CANDIDATE_AUDIT_ATTR
        ].copy()
    return result
