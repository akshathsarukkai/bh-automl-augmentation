"""Tests for regression prediction metrics."""

import math

import numpy as np
import pytest

from bh_augmentation.evaluation.metrics import (
    mae,
    pearson_corr,
    r2,
    regression_metrics,
    rmse,
    spearman_corr,
)


def test_regression_metrics_return_hand_computed_values() -> None:
    """Standard metrics should match hand-computed arrays."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])

    assert rmse(y_true, y_pred) == pytest.approx(0.5)
    assert mae(y_true, y_pred) == pytest.approx(0.25)
    assert r2(y_true, y_pred) == pytest.approx(0.8)
    assert pearson_corr(y_true, y_pred) == pytest.approx(0.9827076298)
    assert spearman_corr(y_true, y_pred) == pytest.approx(1.0)


def test_regression_metrics_dict_contains_expected_keys() -> None:
    """Combined metric helper should return the MVP metric names."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics) == {"rmse", "mae", "r2", "pearson", "spearman"}


def test_constant_predictions_return_nan_correlations() -> None:
    """Correlation is undefined for constant predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    assert math.isnan(pearson_corr(y_true, y_pred))
    assert math.isnan(spearman_corr(y_true, y_pred))


def test_metrics_validate_matching_shapes() -> None:
    """Metric inputs should have the same number of values."""
    with pytest.raises(ValueError, match="same shape"):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))
