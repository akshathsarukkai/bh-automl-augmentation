"""Reaction-aware condition recombination with teacher pseudo-labeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model


def parse_reaction_smiles(rxn: str) -> dict[str, Any]:
    """Parse an MVP reaction string into substrate, condition, and product tokens."""
    invalid = {
        "valid": False,
        "left_tokens": [],
        "substrate_tokens": [],
        "condition_tokens": [],
        "product": "",
    }
    if not isinstance(rxn, str) or not rxn.strip() or rxn.count(">>") != 1:
        return invalid
    left, product = (part.strip() for part in rxn.split(">>", 1))
    left_tokens = [token.strip() for token in left.split(".") if token.strip()]
    if len(left_tokens) < 2 or not product:
        return invalid
    return {
        "valid": True,
        "left_tokens": left_tokens,
        "substrate_tokens": left_tokens[:2],
        "condition_tokens": left_tokens[2:],
        "product": product,
    }


def build_reaction_smiles(
    substrate_tokens: list[str],
    condition_tokens: list[str],
    product: str,
) -> str:
    """Build `substrates.conditions>>product` from parsed reaction sections."""
    left = ".".join([*substrate_tokens, *condition_tokens])
    return f"{left}>>{product}"


def generate_condition_recombined_candidates(
    train_df: pd.DataFrame,
    synthetic_multiplier: float = 1.0,
    max_synthetic_rows: int | None = 3000,
    random_state: int = 42,
    min_condition_tokens: int = 1,
) -> pd.DataFrame:
    """Generate unique reaction candidates by replacing source conditions with donor conditions."""
    if synthetic_multiplier < 0:
        raise ValueError("synthetic_multiplier must be non-negative.")
    if max_synthetic_rows is not None and max_synthetic_rows < 0:
        raise ValueError("max_synthetic_rows must be non-negative or None.")
    if min_condition_tokens < 0:
        raise ValueError("min_condition_tokens must be non-negative.")
    if "reaction_smiles" not in train_df.columns:
        raise ValueError("condition recombination requires reaction_smiles.")

    target = int(synthetic_multiplier * len(train_df))
    if max_synthetic_rows is not None:
        target = min(target, max_synthetic_rows)
    if target <= 0 or train_df.empty:
        return _empty_candidates(train_df)

    parsed = [parse_reaction_smiles(value) for value in train_df["reaction_smiles"]]
    source_positions = [index for index, item in enumerate(parsed) if item["valid"]]
    donor_positions = [
        index
        for index, item in enumerate(parsed)
        if item["valid"] and len(item["condition_tokens"]) >= min_condition_tokens
    ]
    if not source_positions or not donor_positions:
        return _empty_candidates(train_df)

    rng = np.random.default_rng(random_state)
    original_reactions = set(train_df["reaction_smiles"].astype(str))
    generated: set[str] = set()
    rows: list[pd.Series] = []
    max_attempts = max(100, target * 30)

    for _ in range(max_attempts):
        if len(rows) >= target:
            break
        source_position = int(rng.choice(source_positions))
        donor_position = int(rng.choice(donor_positions))
        if source_position == donor_position and len(donor_positions) > 1:
            continue
        source_parts = parsed[source_position]
        donor_parts = parsed[donor_position]
        reaction = build_reaction_smiles(
            source_parts["substrate_tokens"],
            donor_parts["condition_tokens"],
            source_parts["product"],
        )
        source_reaction = str(train_df.iloc[source_position]["reaction_smiles"])
        if reaction == source_reaction or reaction in original_reactions or reaction in generated:
            continue

        source_row = train_df.iloc[source_position].copy()
        source_id = _reaction_id(train_df.iloc[source_position], source_position)
        donor_id = _reaction_id(train_df.iloc[donor_position], donor_position)
        source_row["reaction_id"] = f"synthetic_{len(rows) + 1:06d}"
        source_row["reaction_smiles"] = reaction
        source_row["yield"] = np.nan
        source_row["is_synthetic"] = True
        source_row["is_augmented"] = True
        source_row["augmentation_method"] = "condition_recombine_pseudolabel"
        source_row["augmentation_type"] = "condition_recombine_pseudolabel"
        source_row["source_reaction_id"] = source_id
        source_row["donor_reaction_id"] = donor_id
        source_row["pseudo_label"] = True
        rows.append(source_row)
        generated.add(reaction)

    if not rows:
        return _empty_candidates(train_df)
    return pd.DataFrame(rows).reset_index(drop=True)


def filter_candidates_by_nearest_neighbor_similarity(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict[str, Any],
    min_similarity: float = 0.3,
) -> pd.DataFrame:
    """Keep candidates whose binary fingerprint is sufficiently similar to real training data."""
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1.")
    if candidate_df.empty:
        result = candidate_df.copy()
        result["nearest_train_similarity"] = pd.Series(dtype=float)
        return result

    train_x = _features_only(train_df, feature_config)
    candidate_x = _features_only(candidate_df, feature_config)
    train_binary = sparse.csr_matrix(train_x != 0, dtype=np.float32)
    candidate_binary = sparse.csr_matrix(candidate_x != 0, dtype=np.float32)
    train_counts = np.asarray(train_binary.sum(axis=1)).reshape(-1)

    similarities: list[float] = []
    for row_index in range(candidate_binary.shape[0]):
        candidate = candidate_binary.getrow(row_index)
        intersections = (train_binary @ candidate.T).toarray().reshape(-1)
        candidate_count = float(candidate.sum())
        unions = train_counts + candidate_count - intersections
        scores = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        similarities.append(float(scores.max()) if len(scores) else 0.0)

    result = candidate_df.copy()
    result["nearest_train_similarity"] = similarities
    return result.loc[result["nearest_train_similarity"] >= min_similarity].reset_index(drop=True)


def pseudo_label_candidates(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict[str, Any],
    teacher_model_name: str = "random_forest",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a teacher on real rows only and predict clipped synthetic yields."""
    if candidate_df.empty:
        result = candidate_df.copy()
        result["pseudo_label_model"] = pd.Series(dtype=object)
        return result

    train_x, train_y, _ = build_feature_matrix(train_df, feature_config)
    teacher = train_model(
        get_model(teacher_model_name, seed=random_state),
        train_x,
        train_y,
    )
    candidate_x = _features_only(candidate_df, feature_config)
    predictions = np.clip(predict_model(teacher, candidate_x), 0.0, 100.0)

    result = candidate_df.copy()
    result["yield"] = predictions
    result["pseudo_label_model"] = teacher_model_name
    result["pseudo_label"] = True
    return result


def condition_recombine_pseudolabel(
    train_df: pd.DataFrame,
    feature_config: dict[str, Any],
    synthetic_multiplier: float = 1.0,
    max_synthetic_rows: int | None = 3000,
    teacher_model: str = "random_forest",
    min_neighbor_similarity: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return real training rows plus filtered, teacher-labeled recombined reactions."""
    real = train_df.copy()
    real["is_synthetic"] = False
    real["is_augmented"] = False
    real["augmentation_method"] = "none"
    real["augmentation_type"] = "none"
    real["pseudo_label"] = False
    if "source_reaction_id" not in real.columns:
        real["source_reaction_id"] = real.get(
            "reaction_id", pd.Series(real.index, index=real.index)
        ).astype(str)

    candidates = generate_condition_recombined_candidates(
        train_df,
        synthetic_multiplier=synthetic_multiplier,
        max_synthetic_rows=max_synthetic_rows,
        random_state=random_state,
    )
    candidates = filter_candidates_by_nearest_neighbor_similarity(
        train_df,
        candidates,
        feature_config,
        min_similarity=min_neighbor_similarity,
    )
    candidates = pseudo_label_candidates(
        train_df,
        candidates,
        feature_config,
        teacher_model_name=teacher_model,
        random_state=random_state,
    )
    return pd.concat([real, candidates], ignore_index=True, sort=False)


def assign_synthetic_sample_weights(
    train_df: pd.DataFrame,
    weighting_config: dict[str, Any] | None,
) -> pd.DataFrame:
    """Assign real and similarity-based synthetic sample weights."""
    result = train_df.copy()
    config = weighting_config or {}
    real_weight = float(config.get("real_weight", 1.0))
    minimum = float(config.get("min_synthetic_weight", 0.25))
    maximum = float(config.get("max_synthetic_weight", 0.85))
    if minimum < 0 or maximum < minimum:
        raise ValueError("Synthetic weight bounds must satisfy 0 <= min <= max.")

    synthetic = (
        result.get("is_synthetic", pd.Series(False, index=result.index))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    result["sample_weight"] = real_weight
    if not bool(config.get("enabled", False)) or not synthetic.any():
        return result
    if str(config.get("synthetic_weight_mode", "nearest_similarity")) != "nearest_similarity":
        raise ValueError("synthetic_weight_mode must be 'nearest_similarity'.")

    similarities = pd.to_numeric(
        result.loc[synthetic, "nearest_train_similarity"], errors="coerce"
    ).fillna(minimum)
    result.loc[synthetic, "sample_weight"] = similarities.clip(minimum, maximum)
    return result


def _features_only(df: pd.DataFrame, feature_config: dict[str, Any]) -> np.ndarray:
    data = df.copy()
    if "yield" not in data.columns:
        data["yield"] = 0.0
    else:
        data["yield"] = pd.to_numeric(data["yield"], errors="coerce").fillna(0.0)
    X, _, _ = build_feature_matrix(data, feature_config)
    return X


def _reaction_id(row: pd.Series, position: int) -> str:
    value = row.get("reaction_id")
    return str(value) if pd.notna(value) else str(position)


def _empty_candidates(train_df: pd.DataFrame) -> pd.DataFrame:
    result = train_df.iloc[0:0].copy()
    for column, dtype in {
        "is_synthetic": bool,
        "is_augmented": bool,
        "augmentation_method": object,
        "source_reaction_id": object,
        "donor_reaction_id": object,
        "pseudo_label": bool,
    }.items():
        result[column] = pd.Series(dtype=dtype)
    return result
