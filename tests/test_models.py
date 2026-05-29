"""Tests for baseline model creation, training, and prediction."""

import sys

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model


def _synthetic_regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    return X, y


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("ridge", Ridge),
        ("random_forest", RandomForestRegressor),
        ("extra_trees", ExtraTreesRegressor),
    ],
)
def test_get_model_returns_supported_sklearn_models(name: str, expected_type: type) -> None:
    """Supported default models should instantiate without optional packages."""
    model = get_model(name, seed=123, n_estimators=5) if name != "ridge" else get_model(name)

    assert isinstance(model, expected_type)


@pytest.mark.parametrize("name", ["ridge", "random_forest", "extra_trees"])
def test_train_and_predict_model_output_shape(name: str) -> None:
    """Baseline models should train and produce one prediction per row."""
    X, y = _synthetic_regression_data()
    kwargs = {"n_estimators": 5} if name in {"random_forest", "extra_trees"} else {}
    model = get_model(name, seed=42, **kwargs)

    fitted = train_model(model, X, y)
    predictions = predict_model(fitted, X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_get_model_unknown_name_raises_helpful_error() -> None:
    """Unknown model names should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown model name"):
        get_model("not_a_model")


def test_optional_xgboost_missing_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting missing XGBoost should not break package import."""
    monkeypatch.setitem(sys.modules, "xgboost", None)

    with pytest.raises(ImportError, match="python -m pip install xgboost"):
        get_model("xgboost")


def test_optional_catboost_missing_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting missing CatBoost should not break package import."""
    monkeypatch.setitem(sys.modules, "catboost", None)

    with pytest.raises(ImportError, match="python -m pip install catboost"):
        get_model("catboost")


def test_train_model_validates_fit_method() -> None:
    """Training helper should require a fit-compatible object."""
    X, y = _synthetic_regression_data()

    with pytest.raises(ValueError, match="fit method"):
        train_model(object(), X, y)


def test_predict_model_validates_predict_method() -> None:
    """Prediction helper should require a predict-compatible object."""
    X, _ = _synthetic_regression_data()

    with pytest.raises(ValueError, match="predict method"):
        predict_model(object(), X)
