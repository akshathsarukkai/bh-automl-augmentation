"""Shared uncertainty estimators with explicit split-conformal calibration.

Models are fitted only on the declared training partition. Calibration outcomes
are used only after model fitting, to convert raw uncertainty signals or raw
quantile bounds into intervals at the configured marginal coverage.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from bh_augmentation.models.baselines import get_model
from bh_augmentation.utils.corrected_runs import stable_hash

UncertaintyMethod = Literal[
    "bootstrap_extra_trees",
    "seeded_xgboost",
    "quantile_gradient_boosting",
    "split_conformal_ridge",
    "distance_aware_ridge",
    "heterogeneous_disagreement",
]

SUPPORTED_UNCERTAINTY_METHODS = (
    "bootstrap_extra_trees",
    "seeded_xgboost",
    "quantile_gradient_boosting",
    "split_conformal_ridge",
    "distance_aware_ridge",
    "heterogeneous_disagreement",
)


@dataclass(frozen=True, slots=True)
class UncertaintyEstimatorConfig:
    """Complete deterministic configuration for one uncertainty estimator."""

    method: UncertaintyMethod
    coverage: float = 0.9
    random_state: int = 0
    ridge_alpha: float = 1.0
    minimum_raw_scale: float = 1.0e-6
    bootstrap_members: int = 8
    bootstrap_trees_per_member: int = 16
    xgboost_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    xgboost_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "tree_method": "hist",
        }
    )
    quantile_n_estimators: int = 60
    quantile_max_depth: int = 2
    quantile_learning_rate: float = 0.05
    heterogeneous_tree_estimators: int = 32

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_UNCERTAINTY_METHODS:
            raise ValueError(
                f"Unsupported uncertainty method {self.method!r}; "
                f"supported={list(SUPPORTED_UNCERTAINTY_METHODS)}."
            )
        if not math.isfinite(float(self.coverage)) or not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must be finite and in (0, 1).")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError("random_state must be an integer.")
        for name in ("ridge_alpha", "minimum_raw_scale", "quantile_learning_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "bootstrap_members",
            "bootstrap_trees_per_member",
            "quantile_n_estimators",
            "quantile_max_depth",
            "heterogeneous_tree_estimators",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.bootstrap_members < 2:
            raise ValueError("bootstrap_members must be at least two.")
        if (
            not isinstance(self.xgboost_seeds, tuple)
            or len(self.xgboost_seeds) < 2
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.xgboost_seeds)
            or len(set(self.xgboost_seeds)) != len(self.xgboost_seeds)
        ):
            raise ValueError("xgboost_seeds must contain at least two unique integer seeds.")
        params = dict(self.xgboost_params)
        if not params:
            raise ValueError("xgboost_params must be a non-empty mapping.")
        if "random_state" in params or "seed" in params:
            raise ValueError(
                "xgboost_params must not override the declared xgboost_seeds."
            )
        if int(params.get("n_jobs", 1)) != 1:
            raise ValueError("Deterministic seeded XGBoost requires n_jobs=1.")
        try:
            stable_hash(params)
        except (TypeError, ValueError) as exc:
            raise ValueError("xgboost_params must be deterministically JSON serializable.") from exc
        _reject_nonfinite_config(params, path="xgboost_params")
        object.__setattr__(self, "xgboost_params", params)

    @property
    def config_hash(self) -> str:
        """Return a stable hash of every resolved setting."""
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class EstimatorAudit:
    """Immutable provenance for model fitting and interval calibration."""

    method: str
    config_hash: str
    train_source_id_hash: str
    calibration_source_id_hash: str
    train_data_hash: str
    calibration_data_hash: str
    model_names: tuple[str, ...]
    model_seeds: tuple[int, ...]
    seed_hash: str
    sample_hash: str
    bootstrap_sample_hashes: tuple[str, ...]
    calibration_score_hash: str
    calibration_quantile: float
    calibration_quantile_level: float
    calibration_kind: str
    train_size: int
    calibration_size: int
    feature_width: int

    @property
    def audit_hash(self) -> str:
        """Hash all fitting and calibration provenance."""
        return stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class UncertaintyPrediction:
    """Point predictions, raw uncertainty, and calibrated intervals."""

    point: np.ndarray
    raw_score: np.ndarray
    raw_lower: np.ndarray
    raw_upper: np.ndarray
    interval_lower: np.ndarray
    interval_upper: np.ndarray
    calibrated_uncertainty: np.ndarray
    source_id_hash: str
    estimator_audit_hash: str
    prediction_hash: str

    def __post_init__(self) -> None:
        arrays = (
            self.point,
            self.raw_score,
            self.raw_lower,
            self.raw_upper,
            self.interval_lower,
            self.interval_upper,
            self.calibrated_uncertainty,
        )
        lengths = {len(np.asarray(array).reshape(-1)) for array in arrays}
        if len(lengths) != 1:
            raise ValueError("Uncertainty prediction arrays must have identical lengths.")
        for array in arrays:
            values = _readonly(array)
            if not np.isfinite(values).all():
                raise ValueError("Uncertainty predictions must be finite.")
        for name, array in zip(
            (
                "point",
                "raw_score",
                "raw_lower",
                "raw_upper",
                "interval_lower",
                "interval_upper",
                "calibrated_uncertainty",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, _readonly(array))
        if np.any(self.raw_lower > self.raw_upper):
            raise ValueError("Raw uncertainty bounds cross.")
        if np.any(self.interval_lower > self.interval_upper):
            raise ValueError("Calibrated uncertainty bounds cross.")


@dataclass(slots=True)
class FittedUncertaintyEstimator:
    """Run-local fitted uncertainty estimator; no global cache is used."""

    config: UncertaintyEstimatorConfig
    audit: EstimatorAudit
    _models: tuple[Any, ...]
    _training_features: np.ndarray
    _calibration_quantile: float
    _calibration_kind: str

    def predict(
        self,
        features: np.ndarray,
        *,
        source_ids: Sequence[str],
    ) -> UncertaintyPrediction:
        """Predict without accepting outcomes or scientific partition objects."""
        X = _validate_prediction_features(
            features,
            source_ids,
            expected_width=self.audit.feature_width,
        )
        ids = _validate_source_ids(source_ids, len(X), role="prediction")
        point, raw_score, raw_lower, raw_upper = _raw_prediction(
            self.config,
            self._models,
            self._training_features,
            X,
        )
        if self._calibration_kind == "additive_quantile_bounds":
            interval_lower = raw_lower - self._calibration_quantile
            interval_upper = raw_upper + self._calibration_quantile
        elif self._calibration_kind == "additive_residual":
            interval_lower = point - self._calibration_quantile
            interval_upper = point + self._calibration_quantile
        elif self._calibration_kind == "scaled_residual":
            scale = np.maximum(raw_score, self.config.minimum_raw_scale)
            interval_lower = point - self._calibration_quantile * scale
            interval_upper = point + self._calibration_quantile * scale
        else:  # pragma: no cover - construction is private and exhaustive
            raise AssertionError(f"Unknown calibration kind {self._calibration_kind!r}.")
        calibrated_uncertainty = 0.5 * (interval_upper - interval_lower)
        source_id_hash = stable_hash(list(ids))
        payload_hash = _prediction_payload_hash(
            point,
            raw_score,
            raw_lower,
            raw_upper,
            interval_lower,
            interval_upper,
            ids,
            self.audit.audit_hash,
        )
        return UncertaintyPrediction(
            point=_readonly(point),
            raw_score=_readonly(raw_score),
            raw_lower=_readonly(raw_lower),
            raw_upper=_readonly(raw_upper),
            interval_lower=_readonly(interval_lower),
            interval_upper=_readonly(interval_upper),
            calibrated_uncertainty=_readonly(calibrated_uncertainty),
            source_id_hash=source_id_hash,
            estimator_audit_hash=self.audit.audit_hash,
            prediction_hash=payload_hash,
        )


def fit_uncertainty_estimator(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    train_source_ids: Sequence[str],
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_source_ids: Sequence[str],
    config: UncertaintyEstimatorConfig,
) -> FittedUncertaintyEstimator:
    """Fit models on training rows, then calibrate intervals on disjoint rows."""
    X_train, y_train, train_ids = _validate_labeled_partition(
        train_features,
        train_labels,
        train_source_ids,
        role="train",
    )
    X_calibration, y_calibration, calibration_ids = _validate_labeled_partition(
        calibration_features,
        calibration_labels,
        calibration_source_ids,
        role="calibration",
    )
    if X_train.shape[1] != X_calibration.shape[1]:
        raise ValueError("Training and calibration feature widths differ.")
    overlap = set(train_ids) & set(calibration_ids)
    if overlap:
        raise ValueError(
            "Training and calibration source IDs overlap: "
            f"examples={sorted(overlap)[:5]}."
        )

    models, training_features, seeds, sample_hashes = _fit_models(config, X_train, y_train)
    point, raw_score, raw_lower, raw_upper = _raw_prediction(
        config,
        models,
        training_features,
        X_calibration,
    )
    calibration_kind, calibration_scores = _calibration_scores(
        config,
        y_calibration,
        point,
        raw_score,
        raw_lower,
        raw_upper,
    )
    calibration_quantile, quantile_level = _finite_sample_quantile(
        calibration_scores,
        coverage=config.coverage,
    )
    seed_hash = stable_hash(list(seeds))
    sample_hash = stable_hash(list(sample_hashes))
    audit = EstimatorAudit(
        method=config.method,
        config_hash=config.config_hash,
        train_source_id_hash=stable_hash(list(train_ids)),
        calibration_source_id_hash=stable_hash(list(calibration_ids)),
        train_data_hash=_labeled_data_hash(X_train, y_train, train_ids),
        calibration_data_hash=_labeled_data_hash(
            X_calibration, y_calibration, calibration_ids
        ),
        model_names=tuple(type(model).__name__ for model in models),
        model_seeds=seeds,
        seed_hash=seed_hash,
        sample_hash=sample_hash,
        bootstrap_sample_hashes=(
            sample_hashes if config.method == "bootstrap_extra_trees" else ()
        ),
        calibration_score_hash=_array_hash(calibration_scores),
        calibration_quantile=float(calibration_quantile),
        calibration_quantile_level=float(quantile_level),
        calibration_kind=calibration_kind,
        train_size=len(X_train),
        calibration_size=len(X_calibration),
        feature_width=X_train.shape[1],
    )
    frozen_training = np.asarray(training_features, dtype=np.float64).copy()
    frozen_training.setflags(write=False)
    return FittedUncertaintyEstimator(
        config=config,
        audit=audit,
        _models=models,
        _training_features=frozen_training,
        _calibration_quantile=float(calibration_quantile),
        _calibration_kind=calibration_kind,
    )


def _fit_models(
    config: UncertaintyEstimatorConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[tuple[Any, ...], np.ndarray, tuple[int, ...], tuple[str, ...]]:
    if config.method == "bootstrap_extra_trees":
        rng = np.random.default_rng(config.random_state)
        models = []
        seeds = []
        sample_hashes = []
        for _ in range(config.bootstrap_members):
            sample = rng.integers(0, len(X_train), size=len(X_train), endpoint=False)
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            model = ExtraTreesRegressor(
                n_estimators=config.bootstrap_trees_per_member,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(X_train[sample], y_train[sample])
            models.append(model)
            seeds.append(seed)
            sample_hashes.append(stable_hash([int(value) for value in sample]))
        return tuple(models), X_train, tuple(seeds), tuple(sample_hashes)

    if config.method == "seeded_xgboost":
        models = []
        for seed in config.xgboost_seeds:
            model = get_model("xgboost", seed=seed, **dict(config.xgboost_params))
            model.fit(X_train, y_train)
            models.append(model)
        return (
            tuple(models),
            X_train,
            config.xgboost_seeds,
            (stable_hash(list(range(len(X_train)))),),
        )

    if config.method == "quantile_gradient_boosting":
        alpha = 0.5 * (1.0 - config.coverage)
        models = []
        for offset, quantile in enumerate((0.5, alpha, 1.0 - alpha)):
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=config.quantile_n_estimators,
                max_depth=config.quantile_max_depth,
                learning_rate=config.quantile_learning_rate,
                random_state=config.random_state + offset,
            )
            model.fit(X_train, y_train)
            models.append(model)
        seeds = tuple(config.random_state + offset for offset in range(3))
        return tuple(models), X_train, seeds, (stable_hash(list(range(len(X_train)))),)

    if config.method in {"split_conformal_ridge", "distance_aware_ridge"}:
        model = Ridge(alpha=config.ridge_alpha)
        model.fit(X_train, y_train)
        return (model,), X_train, (), (stable_hash(list(range(len(X_train)))),)

    if config.method == "heterogeneous_disagreement":
        seeds = (config.random_state + 1001, config.random_state + 2001)
        models = (
            Ridge(alpha=config.ridge_alpha).fit(X_train, y_train),
            RandomForestRegressor(
                n_estimators=config.heterogeneous_tree_estimators,
                random_state=seeds[0],
                n_jobs=1,
            ).fit(X_train, y_train),
            ExtraTreesRegressor(
                n_estimators=config.heterogeneous_tree_estimators,
                random_state=seeds[1],
                n_jobs=1,
            ).fit(X_train, y_train),
        )
        return models, X_train, seeds, (stable_hash(list(range(len(X_train)))),)

    raise AssertionError(f"Unhandled uncertainty method {config.method!r}.")


def _raw_prediction(
    config: UncertaintyEstimatorConfig,
    models: tuple[Any, ...],
    training_features: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.method == "quantile_gradient_boosting":
        point = np.asarray(models[0].predict(features), dtype=float)
        first = np.asarray(models[1].predict(features), dtype=float)
        second = np.asarray(models[2].predict(features), dtype=float)
        raw_lower = np.minimum.reduce([first, second, point])
        raw_upper = np.maximum.reduce([first, second, point])
        raw_score = 0.5 * (raw_upper - raw_lower)
        return point, raw_score, raw_lower, raw_upper

    if config.method == "split_conformal_ridge":
        point = np.asarray(models[0].predict(features), dtype=float)
        raw_score = np.zeros(len(features), dtype=float)
        return point, raw_score, point.copy(), point.copy()

    if config.method == "distance_aware_ridge":
        point = np.asarray(models[0].predict(features), dtype=float)
        raw_score = np.maximum(
            _nearest_cosine_distance(features, training_features),
            config.minimum_raw_scale,
        )
        return point, raw_score, point - raw_score, point + raw_score

    matrix = np.vstack(
        [np.asarray(model.predict(features), dtype=float).reshape(-1) for model in models]
    )
    point = matrix.mean(axis=0)
    raw_score = matrix.std(axis=0, ddof=0)
    return point, raw_score, point - raw_score, point + raw_score


def _calibration_scores(
    config: UncertaintyEstimatorConfig,
    labels: np.ndarray,
    point: np.ndarray,
    raw_score: np.ndarray,
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
) -> tuple[str, np.ndarray]:
    if config.method == "quantile_gradient_boosting":
        scores = np.maximum.reduce(
            [raw_lower - labels, labels - raw_upper, np.zeros(len(labels))]
        )
        return "additive_quantile_bounds", scores
    residuals = np.abs(labels - point)
    if config.method == "split_conformal_ridge":
        return "additive_residual", residuals
    scale = np.maximum(raw_score, config.minimum_raw_scale)
    return "scaled_residual", residuals / scale


def _finite_sample_quantile(
    scores: np.ndarray,
    *,
    coverage: float,
) -> tuple[float, float]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Calibration scores must be nonempty, finite, and nonnegative.")
    rank = int(math.ceil((len(values) + 1) * coverage))
    rank = min(max(rank, 1), len(values))
    level = rank / len(values)
    return float(np.partition(values, rank - 1)[rank - 1]), float(level)


def _nearest_cosine_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    query_values = np.asarray(query, dtype=np.float64)
    support_values = np.asarray(support, dtype=np.float64)
    query_norm = np.linalg.norm(query_values, axis=1, keepdims=True)
    support_norm = np.linalg.norm(support_values, axis=1, keepdims=True)
    similarities = (
        query_values / np.maximum(query_norm, 1.0e-12)
    ) @ (support_values / np.maximum(support_norm, 1.0e-12)).T
    return 1.0 - np.max(np.clip(similarities, -1.0, 1.0), axis=1)


def _validate_labeled_partition(
    features: np.ndarray,
    labels: np.ndarray,
    source_ids: Sequence[str],
    *,
    role: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{role} features must be a nonempty two-dimensional array.")
    if len(X) != len(y):
        raise ValueError(f"{role} features, labels, and source IDs must have equal lengths.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{role} features and labels must be finite.")
    ids = _validate_source_ids(source_ids, len(X), role=role)
    return X, y, ids


def _validate_prediction_features(
    features: np.ndarray,
    source_ids: Sequence[str],
    *,
    expected_width: int,
) -> np.ndarray:
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Prediction features must be a nonempty two-dimensional array.")
    if X.shape[1] != expected_width:
        raise ValueError(
            f"Prediction feature width differs: expected={expected_width}, observed={X.shape[1]}."
        )
    if not np.isfinite(X).all():
        raise ValueError("Prediction features must be finite.")
    _validate_source_ids(source_ids, len(X), role="prediction")
    return X


def _validate_source_ids(
    source_ids: Sequence[str],
    expected_length: int,
    *,
    role: str,
) -> tuple[str, ...]:
    if isinstance(source_ids, (str, bytes)):
        raise ValueError(f"{role} source IDs must be a sequence of strings.")
    ids = tuple(source_ids)
    if len(ids) != expected_length:
        raise ValueError(f"{role} features, labels, and source IDs must have equal lengths.")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{role} source IDs must be nonempty strings.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{role} source IDs must be unique.")
    return ids


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1).copy()
    result.setflags(write=False)
    return result


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(
        stable_hash({"dtype": "float64-little-endian", "shape": list(array.shape)}).encode(
            "utf-8"
        )
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _labeled_data_hash(
    features: np.ndarray,
    labels: np.ndarray,
    source_ids: Sequence[str],
) -> str:
    return stable_hash(
        {
            "feature_hash": _array_hash(features),
            "label_hash": _array_hash(labels),
            "source_ids": list(source_ids),
        }
    )


def _prediction_payload_hash(
    point: np.ndarray,
    raw_score: np.ndarray,
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    source_ids: Sequence[str],
    audit_hash: str,
) -> str:
    return stable_hash(
        {
            "point": _array_hash(point),
            "raw_score": _array_hash(raw_score),
            "raw_lower": _array_hash(raw_lower),
            "raw_upper": _array_hash(raw_upper),
            "interval_lower": _array_hash(interval_lower),
            "interval_upper": _array_hash(interval_upper),
            "source_ids": list(source_ids),
            "estimator_audit_hash": audit_hash,
        }
    )


def _reject_nonfinite_config(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_config(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_config(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{path} contains a nonfinite value.")
