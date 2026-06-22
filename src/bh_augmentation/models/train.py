"""Model training helpers."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit a regression model and return the fitted estimator."""
    if not hasattr(model, "fit"):
        raise ValueError("Model must implement a fit method.")
    if sample_weight is None:
        model.fit(X_train, y_train)
        return model

    try:
        parameters = inspect.signature(model.fit).parameters.values()
        supports_weights = any(
            parameter.name == "sample_weight"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_weights = True

    if supports_weights:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return model
        except TypeError:
            supports_weights = False

    if not supports_weights:
        warnings.warn(
            f"Model {type(model).__name__} does not support sample_weight; "
            "fitting without synthetic weighting.",
            UserWarning,
            stacklevel=2,
        )
        model.fit(X_train, y_train)
    return model
