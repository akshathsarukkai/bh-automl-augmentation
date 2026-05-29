"""Baseline model factory for yield prediction."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

SUPPORTED_MODELS = {
    "ridge",
    "random_forest",
    "extra_trees",
    "xgboost",
    "xgb",
    "catboost",
}


def get_model(name: str, seed: int = 42, **kwargs: Any) -> Any:
    """Create a baseline regression model by name.

    Parameters
    ----------
    name:
        Model identifier. Supported names are `ridge`, `random_forest`,
        `extra_trees`, `xgboost`/`xgb`, and `catboost`.
    seed:
        Deterministic random seed for models that support one.
    **kwargs:
        Additional estimator keyword arguments.

    Returns
    -------
    Any
        A scikit-learn compatible regressor.

    Raises
    ------
    ValueError
        If the model name is unknown.
    ImportError
        If an optional model package is requested but not installed.
    """
    normalized_name = name.strip().lower().replace("-", "_")

    if normalized_name == "ridge":
        return Ridge(**kwargs)
    if normalized_name == "random_forest":
        defaults = {"n_estimators": 100, "random_state": seed, "n_jobs": -1}
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)
    if normalized_name == "extra_trees":
        defaults = {"n_estimators": 100, "random_state": seed, "n_jobs": -1}
        defaults.update(kwargs)
        return ExtraTreesRegressor(**defaults)
    if normalized_name in {"xgboost", "xgb"}:
        return _get_xgboost_model(seed=seed, **kwargs)
    if normalized_name == "catboost":
        return _get_catboost_model(seed=seed, **kwargs)

    supported = ", ".join(sorted(SUPPORTED_MODELS))
    raise ValueError(f"Unknown model name: {name}. Supported models: {supported}.")


def _get_xgboost_model(seed: int, **kwargs: Any) -> Any:
    """Create an XGBoost regressor if XGBoost is installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is optional and is not installed. Install it explicitly "
            "with `python -m pip install xgboost`, or use a scikit-learn model "
            "such as `ridge`, `random_forest`, or `extra_trees`."
        ) from exc

    defaults = {
        "n_estimators": 100,
        "random_state": seed,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }
    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def _get_catboost_model(seed: int, **kwargs: Any) -> Any:
    """Create a CatBoost regressor if CatBoost is installed."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost is optional and is not installed. Install it explicitly "
            "with `python -m pip install catboost`, or use a scikit-learn model "
            "such as `ridge`, `random_forest`, or `extra_trees`."
        ) from exc

    defaults = {
        "iterations": 100,
        "random_seed": seed,
        "verbose": False,
    }
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)
