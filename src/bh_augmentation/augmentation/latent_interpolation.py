"""Synthetic latent-point generation by supervised-AE interpolation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    REPRESENTATION_AUGMENTATION,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_RANKING_FIELDS,
    REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
    configured_feature_hash,
)
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.representations.supervised_autoencoder import predict_yield_from_latent

SUPPORTED_ALPHA_DISTRIBUTIONS = {"uniform", "beta_2_2"}
SUPPORTED_LABEL_STRATEGIES = {
    "mixup_label",
    "ae_yield_head",
    "teacher_ensemble",
    "blended",
}


@dataclass
class LatentInterpolationConfig:
    """Configuration for latent interpolation augmentation."""

    synthetic_multiplier: float
    n_neighbors: int
    alpha_min: float
    alpha_max: float
    alpha_distribution: str
    label_strategy: str
    teacher_models: list[str]
    teacher_blend_weight: float
    min_neighbor_similarity: float | None
    max_teacher_std: float | None
    clip_y_min: float
    clip_y_max: float
    random_state: int
    candidates_per_real: int = 20
    max_candidates_per_source: int | None = None


def generate_latent_interpolations(
    z_train: np.ndarray,
    y_train: np.ndarray,
    ae_artifacts: dict[str, Any],
    config: LatentInterpolationConfig,
    source_row_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate train-only latent interpolations as an explicit non-chemical control.

    A latent coordinate does not define a seven-role reaction. Consequently these
    candidates receive deterministic feature identities, but never fabricated
    canonical reaction identities or claims of valid molecular parsing.
    """
    _validate_config(config)
    z_train_array = np.asarray(z_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if z_train_array.ndim != 2:
        raise ValueError("z_train must be a 2D latent matrix.")
    if len(z_train_array) != len(y_train_array):
        raise ValueError("z_train and y_train must contain the same number of rows.")

    n_real_train, latent_dim = z_train_array.shape
    parent_row_ids = _normalize_parent_row_ids(
        source_row_ids,
        z_train_array,
        y_train_array,
    )
    original_positions = np.arange(n_real_train, dtype=int)
    stable_order = np.asarray(
        sorted(
            range(n_real_train),
            key=lambda index: (
                parent_row_ids[index],
                configured_feature_hash(z_train_array[index]) or "",
                f"{float(y_train_array[index]):.17g}",
            ),
        ),
        dtype=int,
    )
    z_train_array = z_train_array[stable_order]
    y_train_array = y_train_array[stable_order]
    parent_row_ids = [parent_row_ids[index] for index in stable_order]
    original_positions = original_positions[stable_order]
    target_count = int(math.ceil(max(0.0, float(config.synthetic_multiplier)) * n_real_train))
    if n_real_train < 2 or target_count == 0:
        candidate_df = _empty_candidate_df()
        metadata = _metadata_from_candidates(candidate_df, config, n_real_train, 0)
        return _package_result(np.empty((0, latent_dim), dtype=np.float32), np.empty(0), metadata, candidate_df)

    rng = np.random.default_rng(int(config.random_state))
    similarity_matrix = _cosine_similarity_matrix(z_train_array)
    np.fill_diagonal(similarity_matrix, -np.inf)
    neighbor_count = min(max(1, int(config.n_neighbors)), n_real_train - 1)
    neighbor_indices = np.argsort(
        -similarity_matrix,
        axis=1,
        kind="stable",
    )[:, :neighbor_count]

    n_candidates = max(target_count, int(math.ceil(max(1, config.candidates_per_real) * n_real_train)))
    parent_i = rng.integers(0, n_real_train, size=n_candidates)
    neighbor_choice = rng.integers(0, neighbor_count, size=n_candidates)
    parent_j = neighbor_indices[parent_i, neighbor_choice]
    alpha = _sample_alpha(rng, n_candidates, config)
    z_candidates = (
        alpha[:, None] * z_train_array[parent_i]
        + (1.0 - alpha[:, None]) * z_train_array[parent_j]
    ).astype(np.float32)

    parent_similarity = similarity_matrix[parent_i, parent_j].astype(float)
    nearest_train_similarity = _max_cosine_similarity(z_candidates, z_train_array)
    mixup_labels = alpha * y_train_array[parent_i] + (1.0 - alpha) * y_train_array[parent_j]

    teacher_mean, teacher_std, teachers_used = _teacher_predictions(
        z_train_array,
        y_train_array,
        z_candidates,
        ae_artifacts,
        config,
    )
    y_synthetic = _assign_labels(z_candidates, mixup_labels, teacher_mean, ae_artifacts, config)
    y_synthetic = np.clip(y_synthetic, float(config.clip_y_min), float(config.clip_y_max))

    candidate_df = pd.DataFrame(
        {
            "candidate_id": np.arange(n_candidates, dtype=int),
            "parent_i": original_positions[parent_i].astype(int),
            "parent_j": original_positions[parent_j].astype(int),
            "source_row_id": [parent_row_ids[index] for index in parent_i],
            "donor_row_id": [parent_row_ids[index] for index in parent_j],
            "alpha": alpha.astype(float),
            "parent_similarity": parent_similarity,
            "nearest_train_similarity": nearest_train_similarity.astype(float),
            "mixup_label": mixup_labels.astype(float),
            "teacher_mean": teacher_mean.astype(float),
            "teacher_std": teacher_std.astype(float),
            "y_synthetic": y_synthetic.astype(float),
        }
    )
    accepted_mask = _acceptance_mask(candidate_df, z_candidates, config)
    candidate_df = _audit_nonchemical_candidates(
        candidate_df,
        z_candidates,
        accepted_mask,
    )
    accepted_mask = candidate_df["accepted"].to_numpy(dtype=bool)
    candidate_df["label_strategy"] = config.label_strategy
    candidate_df["teacher_models_used"] = ",".join(teachers_used)
    candidate_df["neighbor_metric"] = "cosine"
    candidate_df["substrate_similarity"] = np.nan
    candidate_df["product_similarity"] = np.nan
    candidate_df["nontransferred_role_similarity"] = np.nan
    candidate_df["condition_similarity"] = np.nan
    candidate_df["overall_similarity"] = candidate_df["nearest_train_similarity"]
    candidate_df["relevant_context_similarity"] = candidate_df["parent_similarity"]
    candidate_df["nearest_training_support_distance"] = (
        1.0 - candidate_df["nearest_train_similarity"].to_numpy(dtype=float)
    )
    candidate_df["out_of_support_distance"] = candidate_df[
        "nearest_training_support_distance"
    ]
    candidate_df["support_distance_metric"] = "cosine_distance"
    candidate_df["support_distance_backend"] = "numpy_latent_coordinates"
    candidate_df["diversity_contribution"] = _static_cosine_diversity(z_candidates)
    teacher_proxy = candidate_df["teacher_std"].to_numpy(dtype=float)
    interpolation_proxy = (
        np.abs(y_train_array[parent_i] - y_train_array[parent_j])
        * np.minimum(alpha, 1.0 - alpha)
    )
    scale = max(float(np.std(y_train_array)), 1.0)
    candidate_df["calibrated_uncertainty"] = np.nan
    candidate_df["uncertainty_rank_value"] = np.where(
        np.isfinite(teacher_proxy),
        teacher_proxy,
        interpolation_proxy,
    ) / scale
    candidate_df["uncertainty_rank_basis"] = np.where(
        np.isfinite(teacher_proxy),
        "teacher_ensemble_std_normalized_uncalibrated_proxy_phase12_pending",
        "parent_label_interpolation_gap_normalized_proxy_phase12_pending",
    )

    ranked_indices = _rank_candidate_indices(candidate_df)
    candidate_df["candidate_rank"] = pd.Series(pd.NA, index=candidate_df.index, dtype="Int64")
    for rank, index in enumerate(ranked_indices, start=1):
        candidate_df.at[index, "candidate_rank"] = rank

    source_cap = _configured_source_cap(config)
    source_counts: dict[str, int] = {}
    for index in ranked_indices:
        source_id = str(candidate_df.at[index, "source_row_id"])
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        candidate_df.at[index, "source_candidate_ordinal"] = source_counts[source_id]
        if source_counts[source_id] > source_cap:
            candidate_df.at[index, "accepted"] = False
            candidate_df.at[index, "rejection_reason"] = "per_source_cap"

    accepted_indices = _rank_candidate_indices(candidate_df)[:target_count]
    candidate_df["kept"] = False
    candidate_df.loc[accepted_indices, "kept"] = True
    z_synthetic = z_candidates[accepted_indices].astype(np.float32)
    y_selected = y_synthetic[accepted_indices].astype(float)
    metadata = _metadata_from_candidates(candidate_df, config, n_real_train, len(accepted_indices))
    metadata["latent_dim"] = int(latent_dim)
    metadata["neighbor_metric"] = "cosine"
    metadata["teacher_models_used"] = ",".join(teachers_used)
    return _package_result(z_synthetic, y_selected, metadata, candidate_df)


def _assign_labels(
    z_candidates: np.ndarray,
    mixup_labels: np.ndarray,
    teacher_mean: np.ndarray,
    ae_artifacts: dict[str, Any],
    config: LatentInterpolationConfig,
) -> np.ndarray:
    if config.label_strategy == "mixup_label":
        return mixup_labels.astype(float)
    if config.label_strategy == "ae_yield_head":
        return predict_yield_from_latent(ae_artifacts, z_candidates)
    if config.label_strategy == "teacher_ensemble":
        return teacher_mean.astype(float)
    if config.label_strategy == "blended":
        weight = float(config.teacher_blend_weight)
        return (1.0 - weight) * mixup_labels.astype(float) + weight * teacher_mean.astype(float)
    raise ValueError(f"Unsupported label strategy: {config.label_strategy}")


def _teacher_predictions(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_candidates: np.ndarray,
    ae_artifacts: dict[str, Any],
    config: LatentInterpolationConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    needs_teacher = config.label_strategy in {"teacher_ensemble", "blended"} or (
        config.max_teacher_std is not None
        and config.label_strategy in {"teacher_ensemble", "blended"}
    )
    if not needs_teacher:
        return (
            np.full(len(z_candidates), np.nan, dtype=float),
            np.full(len(z_candidates), np.nan, dtype=float),
            [],
        )

    cache_key = (tuple(config.teacher_models or ["ridge", "random_forest"]), int(config.random_state))
    cache = ae_artifacts.setdefault("_latent_interpolation_teacher_cache", {})
    fitted_teachers = cache.get(cache_key)
    if fitted_teachers is None:
        fitted_teachers = []
        for model_name in config.teacher_models or ["ridge", "random_forest"]:
            try:
                model = get_model(str(model_name), seed=int(config.random_state))
            except ImportError:
                continue
            fitted_teachers.append((str(model_name), train_model(model, z_train, y_train)))
        cache[cache_key] = fitted_teachers

    predictions: list[np.ndarray] = []
    teachers_used: list[str] = []
    for model_name, fitted in fitted_teachers:
        predictions.append(predict_model(fitted, z_candidates).reshape(-1))
        teachers_used.append(str(model_name))

    if not predictions:
        raise ValueError("No teacher models could be fit for teacher-based latent interpolation.")
    matrix = np.vstack(predictions)
    return matrix.mean(axis=0), matrix.std(axis=0), teachers_used


def _acceptance_mask(
    candidate_df: pd.DataFrame,
    z_candidates: np.ndarray,
    config: LatentInterpolationConfig,
) -> np.ndarray:
    finite_vectors = np.isfinite(z_candidates).all(axis=1)
    finite_labels = np.isfinite(candidate_df["y_synthetic"].to_numpy(dtype=float))
    accepted = finite_vectors & finite_labels
    if config.min_neighbor_similarity is not None:
        accepted &= (
            candidate_df["nearest_train_similarity"].to_numpy(dtype=float)
            >= float(config.min_neighbor_similarity)
        )
    if config.max_teacher_std is not None and config.label_strategy in {"teacher_ensemble", "blended"}:
        accepted &= candidate_df["teacher_std"].to_numpy(dtype=float) <= float(config.max_teacher_std)
    return accepted


def _audit_nonchemical_candidates(
    candidate_df: pd.DataFrame,
    z_candidates: np.ndarray,
    initial_acceptance: np.ndarray,
) -> pd.DataFrame:
    """Attach exact coordinate identities without implying chemical identity."""
    result = candidate_df.copy()
    initially_accepted = np.asarray(initial_acceptance, dtype=bool)
    if len(result) != len(initially_accepted):
        raise ValueError("initial_acceptance must contain one value per candidate.")

    feature_hashes = [configured_feature_hash(row) for row in z_candidates]
    duplicate_flags = np.zeros(len(result), dtype=bool)
    accepted = initially_accepted.copy()
    rejection_reasons: list[str | None] = [
        None if value else "candidate_filter_rejected" for value in initially_accepted
    ]
    seen: set[str] = set()
    for position, feature_hash in enumerate(feature_hashes):
        if feature_hash is None:
            accepted[position] = False
            rejection_reasons[position] = "invalid_feature_vector"
            continue
        if not initially_accepted[position]:
            continue
        if feature_hash in seen:
            duplicate_flags[position] = True
            accepted[position] = False
            rejection_reasons[position] = "feature_duplicate_synthetic"
            continue
        seen.add(feature_hash)

    # Chemical duplicate questions are not false; they are not evaluable from a
    # latent vector. Nullable fields preserve that distinction in serialized audits.
    result["canonical_reaction_key"] = None
    result["canonical_reaction_hash"] = None
    result["feature_hash"] = feature_hashes
    result["source_identical"] = pd.array([pd.NA] * len(result), dtype="boolean")
    result["already_measured"] = pd.array([pd.NA] * len(result), dtype="boolean")
    # Semantic rule: representation_augmentation. A latent coordinate carries no
    # seven-role reaction identity, so the chemical-identity gate is
    # inapplicable rather than passed. Chemical duplicate questions stay NULL
    # (not False): they are unanswerable here, which is a different statement
    # from being answered "no". See docs/CANDIDATE_SCOPE.md.
    result["candidate_scope_mode"] = REPRESENTATION_AUGMENTATION
    result["duplicate_synthetic"] = pd.array([pd.NA] * len(result), dtype="boolean")
    result["feature_duplicate_synthetic"] = duplicate_flags
    result["chemical_parse_valid"] = False
    result["identity_classification"] = "feature_space_nonchemical"
    result["chemical_identity_available"] = False
    result["scientific_candidate_eligible"] = False
    result["development_control_only"] = True
    result["source_id_semantics"] = "interpolation_parent"
    result["donor_id_semantics"] = "interpolation_neighbor_parent"
    result["chemical_identity_limitation"] = (
        "latent_coordinate_has_no_seven_role_reaction_identity"
    )
    result["rejection_reason"] = rejection_reasons
    result["accepted"] = accepted
    return result


def _normalize_parent_row_ids(
    source_row_ids: Sequence[str] | None,
    z_train: np.ndarray,
    y_train: np.ndarray,
) -> list[str]:
    n_rows = len(z_train)
    if source_row_ids is None:
        return [
            "feature:"
            f"{configured_feature_hash(z_train[index])}:"
            f"yield:{float(y_train[index]):.17g}"
            for index in range(n_rows)
        ]
    if len(source_row_ids) != n_rows:
        raise ValueError("source_row_ids must contain one identifier per training row.")
    normalized = [str(value) for value in source_row_ids]
    if any(not value for value in normalized):
        raise ValueError("source_row_ids must not contain empty identifiers.")
    return normalized


def _metadata_from_candidates(
    candidate_df: pd.DataFrame,
    config: LatentInterpolationConfig,
    n_real_train: int,
    n_synthetic_train: int,
) -> dict[str, Any]:
    kept = candidate_df.loc[candidate_df.get("kept", False)] if not candidate_df.empty else candidate_df
    accepted = candidate_df.loc[candidate_df.get("accepted", False)] if not candidate_df.empty else candidate_df
    n_generated = int(len(candidate_df))
    n_accepted = int(len(accepted))
    return {
        "n_real_train": int(n_real_train),
        "n_candidates_generated": n_generated,
        "n_candidates_accepted": n_accepted,
        "n_synthetic_train": int(n_synthetic_train),
        "filter_acceptance_rate": float(n_accepted / n_generated) if n_generated else 0.0,
        "synthetic_multiplier": float(config.synthetic_multiplier),
        "n_neighbors": int(config.n_neighbors),
        "alpha_min": float(config.alpha_min),
        "alpha_max": float(config.alpha_max),
        "alpha_distribution": config.alpha_distribution,
        "label_strategy": config.label_strategy,
        "teacher_blend_weight": float(config.teacher_blend_weight),
        "min_neighbor_similarity": config.min_neighbor_similarity,
        "max_teacher_std": config.max_teacher_std,
        "identity_classification": "feature_space_nonchemical",
        "chemical_identity_available": False,
        "scientific_candidate_eligible": False,
        "development_control_only": True,
        "mean_alpha": _safe_mean(kept, "alpha"),
        "std_alpha": _safe_std(kept, "alpha"),
        "mean_neighbor_similarity": _safe_mean(kept, "nearest_train_similarity"),
        "mean_parent_similarity": _safe_mean(kept, "parent_similarity"),
        "mean_teacher_std": _safe_mean(kept, "teacher_std"),
        "mean_synthetic_yield": _safe_mean(kept, "y_synthetic"),
        "std_synthetic_yield": _safe_std(kept, "y_synthetic"),
        "min_synthetic_yield": _safe_min(kept, "y_synthetic"),
        "max_synthetic_yield": _safe_max(kept, "y_synthetic"),
    }


def _cosine_similarity_matrix(z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    normalized = z / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _max_cosine_similarity(z_candidates: np.ndarray, z_train: np.ndarray) -> np.ndarray:
    train_norms = np.linalg.norm(z_train, axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(z_candidates, axis=1, keepdims=True)
    normalized_train = z_train / np.maximum(train_norms, 1e-12)
    normalized_candidates = z_candidates / np.maximum(candidate_norms, 1e-12)
    return np.max(np.clip(normalized_candidates @ normalized_train.T, -1.0, 1.0), axis=1)


def _static_cosine_diversity(z_candidates: np.ndarray) -> np.ndarray:
    """Return each point's fixed distance to its nearest other candidate."""
    values = np.asarray(z_candidates, dtype=np.float64)
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    nearest = np.full(len(values), -np.inf, dtype=float)
    for start in range(0, len(values), 128):
        stop = min(start + 128, len(values))
        scores = np.clip(normalized[start:stop] @ normalized.T, -1.0, 1.0)
        rows = np.arange(stop - start)
        scores[rows, np.arange(start, stop)] = -np.inf
        nearest[start:stop] = np.max(scores, axis=1)
    return 1.0 - nearest


def _rank_candidate_indices(candidate_df: pd.DataFrame) -> list[int]:
    accepted = candidate_df.loc[candidate_df["accepted"].astype(bool)].copy()
    if accepted.empty:
        return []
    accepted["_stable_feature_key"] = accepted["feature_hash"].fillna("~").astype(str)
    accepted["_stable_source_key"] = accepted["source_row_id"].fillna("").astype(str)
    accepted["_stable_donor_key"] = accepted["donor_row_id"].fillna("").astype(str)
    ranked = accepted.sort_values(
        [
            "uncertainty_rank_value",
            "relevant_context_similarity",
            "diversity_contribution",
            "out_of_support_distance",
            "_stable_feature_key",
            "_stable_source_key",
            "_stable_donor_key",
        ],
        ascending=[True, False, False, True, True, True, True],
        kind="mergesort",
    )
    return [int(index) for index in ranked.index]


def _configured_source_cap(config: LatentInterpolationConfig) -> int:
    return int(
        config.max_candidates_per_source
        if config.max_candidates_per_source is not None
        else config.candidates_per_real
    )


def _sample_alpha(
    rng: np.random.Generator,
    n_candidates: int,
    config: LatentInterpolationConfig,
) -> np.ndarray:
    if config.alpha_distribution == "uniform":
        raw = rng.uniform(0.0, 1.0, size=n_candidates)
    elif config.alpha_distribution == "beta_2_2":
        raw = rng.beta(2.0, 2.0, size=n_candidates)
    else:
        raise ValueError(f"Unsupported alpha distribution: {config.alpha_distribution}")
    return float(config.alpha_min) + raw * (float(config.alpha_max) - float(config.alpha_min))


def _validate_config(config: LatentInterpolationConfig) -> None:
    if config.alpha_distribution not in SUPPORTED_ALPHA_DISTRIBUTIONS:
        raise ValueError(f"Unsupported alpha distribution: {config.alpha_distribution}")
    if config.label_strategy not in SUPPORTED_LABEL_STRATEGIES:
        raise ValueError(f"Unsupported label strategy: {config.label_strategy}")
    if not 0 <= config.alpha_min <= config.alpha_max <= 1:
        raise ValueError("alpha_min and alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")
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
    if not 0 <= config.teacher_blend_weight <= 1:
        raise ValueError("teacher_blend_weight must be between 0 and 1.")
    if config.clip_y_min > config.clip_y_max:
        raise ValueError("clip_y_min must be less than or equal to clip_y_max.")


def _empty_candidate_df() -> pd.DataFrame:
    result = pd.DataFrame(
        columns=[
            "candidate_id",
            "parent_i",
            "parent_j",
            "source_row_id",
            "donor_row_id",
            "alpha",
            "parent_similarity",
            "nearest_train_similarity",
            "mixup_label",
            "teacher_mean",
            "teacher_std",
            "y_synthetic",
            "accepted",
            "kept",
            "label_strategy",
            "teacher_models_used",
            "neighbor_metric",
            "substrate_similarity",
            "product_similarity",
            "nontransferred_role_similarity",
            "condition_similarity",
            "overall_similarity",
            "relevant_context_similarity",
            "nearest_training_support_distance",
            "out_of_support_distance",
            "support_distance_metric",
            "support_distance_backend",
            "diversity_contribution",
            "calibrated_uncertainty",
            "uncertainty_rank_value",
            "uncertainty_rank_basis",
            "candidate_rank",
            "source_candidate_ordinal",
            "canonical_reaction_key",
            "canonical_reaction_hash",
            "feature_hash",
            "source_identical",
            "already_measured",
            "duplicate_synthetic",
            "feature_duplicate_synthetic",
            "chemical_parse_valid",
            "identity_classification",
            "chemical_identity_available",
            "scientific_candidate_eligible",
            "development_control_only",
            "source_id_semantics",
            "donor_id_semantics",
            "chemical_identity_limitation",
            "rejection_reason",
        ]
    )
    for column in (
        *REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
        *REQUIRED_SYNTHETIC_RANKING_FIELDS,
    ):
        if column not in result:
            result[column] = pd.Series(dtype=object)
    return result


def _package_result(
    z_synthetic: np.ndarray,
    y_synthetic: np.ndarray,
    metadata: dict[str, Any],
    candidate_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "z_synthetic": z_synthetic.astype(np.float32),
        "y_synthetic": np.asarray(y_synthetic, dtype=float).reshape(-1),
        "metadata": dict(metadata),
        "candidate_df": candidate_df,
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
