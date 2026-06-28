"""Synthetic latent-point generation by supervised-AE interpolation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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


def generate_latent_interpolations(
    z_train: np.ndarray,
    y_train: np.ndarray,
    ae_artifacts: dict[str, Any],
    config: LatentInterpolationConfig,
) -> dict[str, Any]:
    """Generate synthetic latent vectors from train-only nearest-neighbor interpolation."""
    _validate_config(config)
    z_train_array = np.asarray(z_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if z_train_array.ndim != 2:
        raise ValueError("z_train must be a 2D latent matrix.")
    if len(z_train_array) != len(y_train_array):
        raise ValueError("z_train and y_train must contain the same number of rows.")

    n_real_train, latent_dim = z_train_array.shape
    target_count = int(math.ceil(max(0.0, float(config.synthetic_multiplier)) * n_real_train))
    if n_real_train < 2 or target_count == 0:
        candidate_df = _empty_candidate_df()
        metadata = _metadata_from_candidates(candidate_df, config, n_real_train, 0)
        return _package_result(np.empty((0, latent_dim), dtype=np.float32), np.empty(0), metadata, candidate_df)

    rng = np.random.default_rng(int(config.random_state))
    similarity_matrix = _cosine_similarity_matrix(z_train_array)
    np.fill_diagonal(similarity_matrix, -np.inf)
    neighbor_count = min(max(1, int(config.n_neighbors)), n_real_train - 1)
    neighbor_indices = np.argsort(-similarity_matrix, axis=1)[:, :neighbor_count]

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
            "parent_i": parent_i.astype(int),
            "parent_j": parent_j.astype(int),
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
    accepted_mask = _deduplicate_mask(z_candidates, accepted_mask)
    candidate_df["accepted"] = accepted_mask
    candidate_df["label_strategy"] = config.label_strategy
    candidate_df["teacher_models_used"] = ",".join(teachers_used)
    candidate_df["neighbor_metric"] = "cosine"

    accepted_indices = np.flatnonzero(accepted_mask)[:target_count]
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


def _deduplicate_mask(z_candidates: np.ndarray, accepted_mask: np.ndarray) -> np.ndarray:
    deduped = np.array(accepted_mask, copy=True)
    seen: set[tuple[float, ...]] = set()
    for index, row in enumerate(np.round(z_candidates, decimals=6)):
        if not deduped[index]:
            continue
        key = tuple(float(value) for value in row)
        if key in seen:
            deduped[index] = False
        else:
            seen.add(key)
    return deduped


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
    if not 0 <= config.teacher_blend_weight <= 1:
        raise ValueError("teacher_blend_weight must be between 0 and 1.")
    if config.clip_y_min > config.clip_y_max:
        raise ValueError("clip_y_min must be less than or equal to clip_y_max.")


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_id",
            "parent_i",
            "parent_j",
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
        ]
    )


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
