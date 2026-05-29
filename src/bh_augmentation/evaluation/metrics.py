"""Regression metrics for yield prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    true, pred = _validate_arrays(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(true, pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute error."""
    true, pred = _validate_arrays(y_true, y_pred)
    return float(mean_absolute_error(true, pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return coefficient of determination."""
    true, pred = _validate_arrays(y_true, y_pred)
    return float(r2_score(true, pred))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when either input is constant."""
    true, pred = _validate_arrays(y_true, y_pred)
    if _is_constant(true) or _is_constant(pred):
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spearman rank correlation, or NaN when either input is constant."""
    true, pred = _validate_arrays(y_true, y_pred)
    true_ranks = _rankdata(true)
    pred_ranks = _rankdata(pred)
    if _is_constant(true_ranks) or _is_constant(pred_ranks):
        return float("nan")
    return float(np.corrcoef(true_ranks, pred_ranks)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the standard regression metrics used in the MVP."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }


def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {true.shape} and {pred.shape}."
        )
    if len(true) == 0:
        raise ValueError("Metric inputs must contain at least one value.")
    return true, pred


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.all(values == values[0]))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for values, matching Spearman tie handling."""
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=float)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[sorter[start:end]] = average_rank
        start = end

    return ranks
