"""Chemically constrained condition-transfer augmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    ReactionRoles,
    ensure_reaction_role_columns,
    reaction_roles_from_row,
    reaction_roles_to_record,
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
) -> ReactionRoles:
    """Transfer all four typed donor conditions while preserving source chemistry."""
    source = reaction_roles_from_row(source_row)
    donor = reaction_roles_from_row(donor_row)
    return ReactionRoles(
        reactant_1=source.reactant_1,
        reactant_2=source.reactant_2,
        catalyst=donor.catalyst,
        ligand=donor.ligand,
        base=donor.base,
        solvent_or_additive=donor.solvent_or_additive,
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
) -> dict[str, Any]:
    """Generate condition-transfer examples using only low-data training rows."""
    _validate_config(config)
    if "reaction_smiles" not in df_train.columns:
        raise ValueError("condition transfer requires a reaction_smiles column.")

    role_train = ensure_reaction_role_columns(df_train, parse_if_missing=True)
    train = role_train.reset_index(drop=False).rename(columns={"index": "_source_dataframe_index"})
    X_train_array = np.asarray(X_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(train) != len(X_train_array) or len(train) != len(y_train_array):
        raise ValueError("df_train, X_train, and y_train must contain the same number of rows.")

    parsed = [parse_condition_transfer_reaction(value) for value in train["reaction_smiles"]]
    valid_positions = [position for position, item in enumerate(parsed) if item is not None]
    invalid_parse_count = len(train) - len(valid_positions)
    n_real_train = len(train)
    target_count = int(math.ceil(max(0.0, config.synthetic_multiplier) * n_real_train))
    n_candidates = int(math.ceil(max(1, config.candidates_per_real) * n_real_train))
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
        return _package_result(
            _empty_synthetic_df(), np.empty(0), metadata, _empty_candidate_df(),
            np.empty((0, X_train_array.shape[1])), real_feature_names, real_feature_metadata,
        )

    rng = np.random.default_rng(int(config.random_state))
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

    rows: list[dict[str, Any]] = []
    attempts = max(n_candidates, target_count * 5)
    high_yield_fallback_count = 0
    for candidate_id in range(attempts):
        source_position = int(rng.choice(valid_positions))
        donor_position, donor_similarity, used_fallback = _select_donor_position(
            source_position=source_position,
            valid_positions=valid_positions,
            y_train=y_train_array,
            reaction_similarity=reaction_similarity,
            substrate_similarity=substrate_similarity,
            config=config,
            rng=rng,
        )
        if donor_position is None:
            continue
        if used_fallback:
            high_yield_fallback_count += 1
        source = parsed[source_position]
        donor = parsed[donor_position]
        if source is None or donor is None:
            continue
        synthetic_roles = build_anonymous_condition_transfer_roles(
            train.loc[source_position],
            train.loc[donor_position],
        )
        synthetic_record = reaction_roles_to_record(synthetic_roles)
        synthetic_reaction = synthetic_roles.reaction_smiles()
        rows.append(
            {
                "candidate_id": candidate_id,
                "reaction_smiles": synthetic_reaction,
                "source_position": source_position,
                "donor_position": donor_position,
                "source_index": train.loc[source_position, "_source_dataframe_index"],
                "donor_index": train.loc[donor_position, "_source_dataframe_index"],
                "source_reaction_smiles": source.reaction_smiles,
                "donor_reaction_smiles": donor.reaction_smiles,
                "source_yield": float(y_train_array[source_position]),
                "donor_yield": float(y_train_array[donor_position]),
                "donor_strategy": config.donor_strategy,
                "label_strategy": config.label_strategy,
                "donor_similarity": float(donor_similarity),
                "teacher_mean": np.nan,
                "teacher_std": np.nan,
                "synthetic_label": np.nan,
                "source_product": source.product,
                "source_substrate_block": source.substrate_block,
                "donor_condition_block": donor.condition_block,
                "synthetic_was_duplicate": False,
                "existing_real_duplicate": False,
                "accepted": False,
                "kept": False,
                "reaction_roles": synthetic_roles,
                **synthetic_record,
            }
        )
        if len(rows) >= n_candidates:
            break

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
        return _package_result(
            _empty_synthetic_df(), np.empty(0), metadata, candidate_df,
            np.empty((0, X_train_array.shape[1])), real_feature_names, real_feature_metadata,
        )

    candidate_df = _mark_duplicates(candidate_df, set(train["reaction_smiles"].astype(str)))
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
    labels = _assign_labels(candidate_df, config)
    candidate_df["synthetic_label"] = np.clip(
        labels,
        float(config.clip_y_min),
        float(config.clip_y_max),
    )
    accepted = _acceptance_mask(candidate_df, X_synthetic, config)
    candidate_df["accepted"] = accepted
    kept_indices = np.flatnonzero(accepted)[:target_count]
    candidate_df.loc[kept_indices, "kept"] = True

    synthetic_df = candidate_df.loc[candidate_df["kept"], _synthetic_columns()].copy()
    synthetic_df["yield"] = candidate_df.loc[candidate_df["kept"], "synthetic_label"].to_numpy(dtype=float)
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
    return _package_result(
        synthetic_df.reset_index(drop=True),
        synthetic_y,
        metadata,
        candidate_df,
        X_synthetic[candidate_df["kept"].to_numpy(dtype=bool)].astype(np.float32),
        synthetic_feature_names,
        synthetic_feature_metadata,
    )


def _select_donor_position(
    source_position: int,
    valid_positions: list[int],
    y_train: np.ndarray,
    reaction_similarity: np.ndarray,
    substrate_similarity: np.ndarray,
    config: ConditionTransferConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float, bool]:
    candidates = [position for position in valid_positions if position != source_position]
    if not candidates:
        return None, float("nan"), False
    if config.donor_strategy == "random":
        donor = int(rng.choice(candidates))
        return donor, float(reaction_similarity[source_position, donor]), False
    if config.donor_strategy == "nearest_substrate":
        return _nearest_from_candidates(
            source_position,
            candidates,
            substrate_similarity,
            config,
        ) + (False,)
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
            return donor, similarity, False
        donor, similarity = _nearest_from_candidates(
            source_position,
            candidates,
            reaction_similarity,
            config,
        )
        return donor, similarity, True
    return _nearest_from_candidates(
        source_position,
        candidates,
        reaction_similarity,
        config,
    ) + (False,)


def _nearest_from_candidates(
    source_position: int,
    candidates: list[int],
    similarity: np.ndarray,
    config: ConditionTransferConfig,
) -> tuple[int | None, float]:
    scores = np.asarray([similarity[source_position, position] for position in candidates], dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    n_neighbors = max(1, min(int(config.n_neighbors), len(order)))
    for order_index in order[:n_neighbors]:
        donor = int(candidates[int(order_index)])
        score = float(scores[int(order_index)])
        if config.min_similarity is None or score >= float(config.min_similarity):
            return donor, score
    return None, float("nan")


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
    accepted = np.isfinite(X_synthetic).all(axis=1)
    accepted &= np.isfinite(candidate_df["synthetic_label"].to_numpy(dtype=float))
    accepted &= ~candidate_df["synthetic_was_duplicate"].to_numpy(dtype=bool)
    accepted &= ~candidate_df["existing_real_duplicate"].to_numpy(dtype=bool)
    if config.min_similarity is not None and config.donor_strategy in {
        "nearest_reaction",
        "nearest_substrate",
        "high_yield_nearest",
    }:
        accepted &= candidate_df["donor_similarity"].to_numpy(dtype=float) >= float(config.min_similarity)
    if config.label_strategy == "uncertainty_filtered_teacher" and config.max_teacher_std is not None:
        accepted &= candidate_df["teacher_std"].to_numpy(dtype=float) <= float(config.max_teacher_std)
    return accepted


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


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = X / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


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
        "label_strategy": config.label_strategy,
        "synthetic_multiplier": float(config.synthetic_multiplier),
        "n_neighbors": int(config.n_neighbors),
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
    if config.donor_strategy not in DONOR_STRATEGIES:
        raise ValueError(f"Unsupported donor strategy: {config.donor_strategy}")
    if config.label_strategy not in LABEL_STRATEGIES:
        raise ValueError(f"Unsupported label strategy: {config.label_strategy}")
    if config.synthetic_multiplier < 0:
        raise ValueError("synthetic_multiplier must be non-negative.")
    if config.n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if config.candidates_per_real < 1:
        raise ValueError("candidates_per_real must be at least 1.")
    if config.clip_y_min > config.clip_y_max:
        raise ValueError("clip_y_min must be less than or equal to clip_y_max.")


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
        "label_strategy",
        "teacher_mean",
        "teacher_std",
        "synthetic_label",
        "source_product",
        "source_substrate_block",
        "donor_condition_block",
        "synthetic_was_duplicate",
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
            "donor_similarity",
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
