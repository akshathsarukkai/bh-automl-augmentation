"""Simulated single-round reaction recommendation evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.evaluation.regret import simple_regret
from bh_augmentation.evaluation.topk import experiments_to_first_hit, top_k_hit_rate
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_augmentation import _build_training_variants
from bh_augmentation.run_baseline import _parse_model_config, _resolve_feature_config


def simulate_single_round_recommendation(
    dataframe: pd.DataFrame,
    model_config: str | dict[str, Any],
    feature_config: dict[str, Any],
    seed_size: int | None = None,
    seed_fraction: float | None = None,
    candidate_pool: pd.DataFrame | list[str] | None = None,
    k: int = 5,
    high_yield_threshold: float = 70.0,
    augmentation_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate one round of model-guided reaction recommendation.

    A measured seed set is sampled from `dataframe`. Models train only on that
    seed set, rank the remaining candidate reactions, and reveal the true yields
    of the top-k selected candidates. The returned DataFrame compares random
    selection, a non-augmented model, and configured safe augmentation variants.
    """
    if "yield" not in dataframe.columns:
        raise ValueError("Recommendation simulation requires a yield column.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    df = _ensure_reaction_ids(dataframe).reset_index(drop=True)
    seed_df = _select_seed_set(df, seed_size=seed_size, seed_fraction=seed_fraction, seed=seed)
    candidates = _select_candidate_pool(df, seed_df, candidate_pool)
    if candidates.empty:
        raise ValueError("Candidate pool is empty after removing seed reactions.")

    resolved_feature_config = _resolve_feature_config(feature_config, df)
    records = [
        _evaluate_random_baseline(candidates, k, high_yield_threshold, seed),
        _evaluate_model_strategy(
            strategy_name="model",
            train_df=seed_df,
            candidate_df=candidates,
            model_config=model_config,
            feature_config=resolved_feature_config,
            k=k,
            high_yield_threshold=high_yield_threshold,
            seed=seed,
        ),
    ]

    if augmentation_config:
        train_variants = _build_training_variants(
            seed_df,
            augmentation_config,
            resolved_feature_config,
            seed,
        )
        for variant_name, train_variant in train_variants.items():
            if variant_name == "none":
                continue
            records.append(
                _evaluate_model_strategy(
                    strategy_name=f"augmented_{variant_name}",
                    train_df=train_variant,
                    candidate_df=candidates,
                    model_config=model_config,
                    feature_config=resolved_feature_config,
                    k=k,
                    high_yield_threshold=high_yield_threshold,
                    seed=seed,
                )
            )

    result = pd.DataFrame(records)
    result.insert(1, "seed_size", len(seed_df))
    result.insert(2, "candidate_pool_size", len(candidates))
    return result


def _evaluate_random_baseline(
    candidate_df: pd.DataFrame,
    k: int,
    high_yield_threshold: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    scores = rng.random(len(candidate_df))
    return _recommendation_record(
        strategy="random",
        candidate_df=candidate_df,
        scores=scores,
        k=k,
        high_yield_threshold=high_yield_threshold,
    )


def _evaluate_model_strategy(
    strategy_name: str,
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    model_config: str | dict[str, Any],
    feature_config: dict[str, Any],
    k: int,
    high_yield_threshold: float,
    seed: int,
) -> dict[str, object]:
    combined = pd.concat(
        [train_df.assign(__role="train"), candidate_df.assign(__role="candidate")],
        ignore_index=True,
    )
    X, y, _ = build_feature_matrix(combined, feature_config)
    train_mask = combined["__role"].to_numpy() == "train"
    candidate_mask = combined["__role"].to_numpy() == "candidate"

    model_name, model_kwargs = _parse_model_config(model_config)
    model = get_model(model_name, seed=seed, **model_kwargs)
    fitted_model = train_model(model, X[train_mask], y[train_mask])
    scores = predict_model(fitted_model, X[candidate_mask])
    return _recommendation_record(
        strategy=strategy_name,
        candidate_df=candidate_df,
        scores=scores,
        k=k,
        high_yield_threshold=high_yield_threshold,
        model=model_name,
    )


def _recommendation_record(
    strategy: str,
    candidate_df: pd.DataFrame,
    scores: np.ndarray,
    k: int,
    high_yield_threshold: float,
    model: str = "",
) -> dict[str, object]:
    effective_k = min(k, len(candidate_df))
    ranked_indices = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    selected_positions = ranked_indices[:effective_k]
    y_true = candidate_df["yield"].to_numpy(dtype=float)

    selected = candidate_df.iloc[selected_positions]
    selected_scores = np.asarray(scores, dtype=float)[selected_positions]
    return {
        "strategy": strategy,
        "model": model,
        "k": effective_k,
        "high_yield_threshold": high_yield_threshold,
        "selected_reaction_ids": selected["reaction_id"].astype(str).tolist(),
        "predicted_yields": selected_scores.tolist(),
        "true_yields": selected["yield"].astype(float).tolist(),
        "top_k_hit_rate": top_k_hit_rate(y_true, scores, effective_k, high_yield_threshold),
        "regret": simple_regret(y_true, scores, k=effective_k),
        "experiments_to_first_hit": experiments_to_first_hit(
            y_true,
            scores,
            high_yield_threshold,
        ),
    }


def _select_seed_set(
    df: pd.DataFrame,
    seed_size: int | None,
    seed_fraction: float | None,
    seed: int,
) -> pd.DataFrame:
    if seed_size is None:
        if seed_fraction is None:
            raise ValueError("Either seed_size or seed_fraction must be provided.")
        if not 0 < seed_fraction < 1:
            raise ValueError("seed_fraction must be between 0 and 1.")
        seed_size = max(1, int(np.floor(len(df) * seed_fraction)))
    if not 0 < seed_size < len(df):
        raise ValueError("seed_size must be greater than 0 and smaller than the dataset.")
    return df.sample(n=seed_size, random_state=seed).copy()


def _select_candidate_pool(
    df: pd.DataFrame,
    seed_df: pd.DataFrame,
    candidate_pool: pd.DataFrame | list[str] | None,
) -> pd.DataFrame:
    seed_ids = set(seed_df["reaction_id"].astype(str))
    if candidate_pool is None:
        candidates = df.loc[~df["reaction_id"].astype(str).isin(seed_ids)].copy()
    elif isinstance(candidate_pool, pd.DataFrame):
        candidates = _ensure_reaction_ids(candidate_pool).copy()
        candidates = candidates.loc[~candidates["reaction_id"].astype(str).isin(seed_ids)]
    else:
        candidate_ids = {str(reaction_id) for reaction_id in candidate_pool}
        candidates = df.loc[df["reaction_id"].astype(str).isin(candidate_ids - seed_ids)].copy()
    return candidates.reset_index(drop=True)


def _ensure_reaction_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "reaction_id" not in result.columns:
        result.insert(0, "reaction_id", [f"rxn_{index:06d}" for index in range(len(result))])
    return result
