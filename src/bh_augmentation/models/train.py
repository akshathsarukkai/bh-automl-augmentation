"""Model training helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit a regression model and return the fitted estimator."""
    if not hasattr(model, "fit"):
        raise ValueError("Model must implement a fit method.")
    model.fit(X_train, y_train)
    return model
