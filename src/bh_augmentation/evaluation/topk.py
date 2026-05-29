"""Top-k decision metrics for predicted reaction yields."""

from __future__ import annotations

import numpy as np


def top_k_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    threshold: float,
) -> float:
    """Return the fraction of top-k predicted reactions above a true-yield threshold."""
    true, pred = _validate_decision_inputs(y_true, y_pred)
    selected = _top_k_indices(pred, k)
    return float(np.mean(true[selected] >= threshold))


def top_k_average_true_yield(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Return the average true yield among the top-k predicted reactions."""
    true, pred = _validate_decision_inputs(y_true, y_pred)
    selected = _top_k_indices(pred, k)
    return float(np.mean(true[selected]))


def experiments_to_first_hit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> int | None:
    """Return rank position of the first hit, or None if no hit is found."""
    true, pred = _validate_decision_inputs(y_true, y_pred)
    ranked_indices = _top_k_indices(pred, len(pred))
    for rank, index in enumerate(ranked_indices, start=1):
        if true[index] >= threshold:
            return rank
    return None


def _validate_decision_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {true.shape} and {pred.shape}."
        )
    if len(true) == 0:
        raise ValueError("Decision metric inputs must contain at least one value.")
    return true, pred


def _top_k_indices(y_pred: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    effective_k = min(k, len(y_pred))
    ranked_indices = np.argsort(-y_pred, kind="mergesort")
    return ranked_indices[:effective_k]
