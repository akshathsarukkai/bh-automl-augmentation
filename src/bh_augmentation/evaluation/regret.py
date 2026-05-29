"""Regret-style decision metrics."""

from __future__ import annotations

import numpy as np

from bh_augmentation.evaluation.topk import _top_k_indices, _validate_decision_inputs


def simple_regret(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1) -> float:
    """Return true best yield minus best true yield among top-k predictions."""
    true, pred = _validate_decision_inputs(y_true, y_pred)
    selected = _top_k_indices(pred, k)
    return float(np.max(true) - np.max(true[selected]))
