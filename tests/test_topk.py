"""Tests for top-k and regret-style decision metrics."""

import numpy as np
import pytest

from bh_augmentation.evaluation.regret import simple_regret
from bh_augmentation.evaluation.topk import (
    experiments_to_first_hit,
    top_k_average_true_yield,
    top_k_hit_rate,
)


def test_top_k_hit_rate_returns_hand_computed_value() -> None:
    """Hit rate should use true yields among highest predicted reactions."""
    y_true = np.array([50.0, 90.0, 80.0, 10.0])
    y_pred = np.array([0.2, 0.4, 0.3, 0.9])

    assert top_k_hit_rate(y_true, y_pred, k=2, threshold=70.0) == pytest.approx(0.5)


def test_top_k_average_true_yield_returns_hand_computed_value() -> None:
    """Average selected yield should use top-k predicted reactions."""
    y_true = np.array([50.0, 90.0, 80.0, 10.0])
    y_pred = np.array([0.2, 0.4, 0.3, 0.9])

    assert top_k_average_true_yield(y_true, y_pred, k=3) == pytest.approx(60.0)


def test_simple_regret_returns_hand_computed_value() -> None:
    """Regret should compare the true best against best selected true yield."""
    y_true = np.array([50.0, 90.0, 80.0, 10.0])
    y_pred = np.array([0.2, 0.4, 0.3, 0.9])

    assert simple_regret(y_true, y_pred, k=1) == pytest.approx(80.0)
    assert simple_regret(y_true, y_pred, k=2) == pytest.approx(0.0)


def test_experiments_to_first_hit_returns_rank_or_none() -> None:
    """First-hit metric should return a one-based rank or None."""
    y_true = np.array([50.0, 90.0, 80.0, 10.0])
    y_pred = np.array([0.2, 0.4, 0.3, 0.9])

    assert experiments_to_first_hit(y_true, y_pred, threshold=70.0) == 2
    assert experiments_to_first_hit(y_true, y_pred, threshold=95.0) is None


def test_k_larger_than_dataset_uses_all_rows() -> None:
    """Oversized k should select the whole dataset."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([0.1, 0.3, 0.2])

    assert top_k_hit_rate(y_true, y_pred, k=10, threshold=20.0) == pytest.approx(2 / 3)
    assert top_k_average_true_yield(y_true, y_pred, k=10) == pytest.approx(20.0)
    assert simple_regret(y_true, y_pred, k=10) == pytest.approx(0.0)


def test_top_k_metrics_validate_k() -> None:
    """k must be positive."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="k must be greater than 0"):
        top_k_hit_rate(y_true, y_pred, k=0, threshold=15.0)


def test_top_k_metrics_validate_shapes() -> None:
    """Decision metric inputs should have matching shapes."""
    with pytest.raises(ValueError, match="same shape"):
        top_k_average_true_yield(np.array([10.0, 20.0]), np.array([0.1]), k=1)
