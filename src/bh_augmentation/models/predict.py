"""Model prediction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_model(model: Any, X: np.ndarray) -> np.ndarray:
    """Generate numeric predictions from a fitted regression model."""
    if not hasattr(model, "predict"):
        raise ValueError("Model must implement a predict method.")
    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)
