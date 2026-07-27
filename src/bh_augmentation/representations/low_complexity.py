"""Deterministic low-complexity representation-learning baselines.

The public fit function deliberately receives evaluation features but no
evaluation labels.  Every learned transformation and predictor is therefore
fit from the training partition alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Number
from typing import Any

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from torch import nn

from bh_augmentation.utils.corrected_runs import stable_hash

LOW_COMPLEXITY_SCHEMA_VERSION = "bh-low-complexity-representation-v1"
LOW_COMPLEXITY_METHODS = (
    "truncated_svd",
    "linear_autoencoder",
    "partial_least_squares",
    "sparse_feature_selection",
    "small_bottleneck_mlp",
    "direct_mlp_regressor",
)


@dataclass(frozen=True, slots=True)
class LowComplexityConfig:
    """Configuration shared by the six low-complexity baselines."""

    method: str
    latent_width: int
    ridge_alpha: float = 1.0
    random_state: int = 42
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 10
    internal_valid_fraction: float = 0.2
    batch_size: int = 32
    pls_max_iter: int = 500
    pls_tolerance: float = 1e-6


@dataclass(frozen=True, slots=True)
class LowComplexityFitResult:
    """Predictions and complete scientific audit for one fitted baseline."""

    method: str
    requested_latent_width: int
    actual_latent_width: int
    predictions: np.ndarray
    prediction_hash: str
    state_hash: str
    fitted_state_scalar_count: int
    deployed_predictive_parameter_count: int
    selected_epoch: int | None
    internal_split_hash: str | None
    internal_fit_source_id_hash: str | None
    internal_validation_source_id_hash: str | None
    metadata: Mapping[str, Any]


class _TiedLinearAutoencoder(nn.Module):
    """Bias-free linear autoencoder with one shared encoder/decoder matrix."""

    def __init__(self, input_width: int, latent_width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(latent_width, input_width))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(features, self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.encode(features)
        return nn.functional.linear(latent, self.weight.T)


class _MatchedBottleneckRegressor(nn.Module):
    """The common D -> K ReLU -> 1 model used by both neural controls."""

    def __init__(self, input_width: int, latent_width: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_width, latent_width)
        self.head = nn.Linear(latent_width, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features)).squeeze(-1)


def fit_low_complexity_representation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    config: LowComplexityConfig,
    *,
    train_source_ids: Sequence[str] | None = None,
    train_group_ids: Sequence[str] | None = None,
) -> LowComplexityFitResult:
    """Fit one baseline and predict a label-free evaluation feature matrix.

    Neural supervised models choose an epoch using a deterministic,
    train-derived group split, then restart from the same initialization and
    refit all training rows for exactly that frozen epoch count.
    """
    X_fit, y_fit, X_eval = _validate_inputs(X_train, y_train, X_evaluation, config)
    source_ids = _validate_source_ids(train_source_ids, len(X_fit))
    group_ids = _validate_group_ids(train_group_ids, len(X_fit))
    method = config.method
    input_width = int(X_fit.shape[1])
    latent_width = int(config.latent_width)
    config_record = asdict(config)
    fit_warnings: list[str] = []
    frozen_epoch: int | None = None
    internal_split_hash: str | None = None
    internal_fit_source_id_hash: str | None = None
    internal_validation_source_id_hash: str | None = None
    internal_fit_index_hash: str | None = None
    internal_validation_index_hash: str | None = None
    internal_fit_count: int | None = None
    internal_validation_count: int | None = None
    state: dict[str, np.ndarray] = {}

    if method == "truncated_svd":
        representation = TruncatedSVD(
            n_components=latent_width,
            algorithm="randomized",
            n_iter=7,
            random_state=int(config.random_state),
        )
        Z_train = representation.fit_transform(X_fit).astype(np.float32)
        Z_eval = representation.transform(X_eval).astype(np.float32)
        predictor = _fit_ridge(Z_train, y_fit, config.ridge_alpha)
        predictions = np.asarray(predictor.predict(Z_eval), dtype=float)
        state.update(_numeric_fitted_state(representation, "representation"))
        state.update(_numeric_fitted_state(predictor, "predictor"))
        deployed_count = input_width * latent_width + latent_width + 1
        parameter_formula = "D*K + K + 1"

    elif method == "linear_autoencoder":
        representation = _fit_tied_linear_autoencoder(X_fit, config)
        Z_train = _encode_tied(representation, X_fit)
        Z_eval = _encode_tied(representation, X_eval)
        predictor = _fit_ridge(Z_train, y_fit, config.ridge_alpha)
        predictions = np.asarray(predictor.predict(Z_eval), dtype=float)
        state.update(_torch_state(representation, "representation"))
        state.update(_numeric_fitted_state(predictor, "predictor"))
        frozen_epoch = int(config.max_epochs)
        deployed_count = input_width * latent_width + latent_width + 1
        parameter_formula = "D*K_tied + K + 1"

    elif method == "partial_least_squares":
        representation = PLSRegression(
            n_components=latent_width,
            scale=False,
            max_iter=int(config.pls_max_iter),
            tol=float(config.pls_tolerance),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            representation.fit(X_fit, y_fit)
        fit_warnings.extend(
            f"{item.category.__name__}: {item.message}" for item in captured
        )
        Z_train = np.asarray(representation.transform(X_fit), dtype=np.float32)
        Z_eval = np.asarray(representation.transform(X_eval), dtype=np.float32)
        predictor = _fit_ridge(Z_train, y_fit, config.ridge_alpha)
        predictions = np.asarray(predictor.predict(Z_eval), dtype=float)
        state.update(_numeric_fitted_state(representation, "representation"))
        state.update(_numeric_fitted_state(predictor, "predictor"))
        deployed_count = (
            input_width * latent_width + input_width + latent_width + 1
        )
        parameter_formula = "D*K_rotation + D_center + K + 1"

    elif method == "sparse_feature_selection":
        selected_indices, scores, selector_warnings = _stable_f_regression_top_k(
            X_fit,
            y_fit,
            latent_width,
        )
        fit_warnings.extend(selector_warnings)
        Z_train = X_fit[:, selected_indices]
        Z_eval = X_eval[:, selected_indices]
        predictor = _fit_ridge(Z_train, y_fit, config.ridge_alpha)
        predictions = np.asarray(predictor.predict(Z_eval), dtype=float)
        state["representation.scores"] = scores
        state["representation.selected_indices"] = selected_indices
        state.update(_numeric_fitted_state(predictor, "predictor"))
        deployed_count = latent_width + 1
        parameter_formula = "K + 1 (selected feature indices are nonparametric)"

    else:
        fit_indices, valid_indices, internal_split_hash = _make_internal_group_split(
            group_ids,
            valid_fraction=float(config.internal_valid_fraction),
            random_state=int(config.random_state),
        )
        internal_fit_source_id_hash = stable_hash(
            [source_ids[index] for index in fit_indices]
        )
        internal_validation_source_id_hash = stable_hash(
            [source_ids[index] for index in valid_indices]
        )
        internal_fit_index_hash = stable_hash(fit_indices.tolist())
        internal_validation_index_hash = stable_hash(valid_indices.tolist())
        internal_fit_count = len(fit_indices)
        internal_validation_count = len(valid_indices)
        frozen_epoch = _select_neural_epoch(
            X_fit[fit_indices],
            y_fit[fit_indices],
            X_fit[valid_indices],
            y_fit[valid_indices],
            config,
        )
        representation = _refit_matched_neural_model(
            X_fit,
            y_fit,
            config,
            epochs=frozen_epoch,
        )
        neural_state = _torch_state(representation, "neural_model")
        state.update(neural_state)
        y_mean, y_scale = _target_scaling(y_fit)
        state["target.y_mean"] = np.asarray([y_mean], dtype=np.float64)
        state["target.y_scale"] = np.asarray([y_scale], dtype=np.float64)
        if method == "small_bottleneck_mlp":
            Z_train = _encode_matched(representation, X_fit)
            Z_eval = _encode_matched(representation, X_eval)
            predictor = _fit_ridge(Z_train, y_fit, config.ridge_alpha)
            predictions = np.asarray(predictor.predict(Z_eval), dtype=float)
            state.update(_numeric_fitted_state(predictor, "predictor"))
        else:
            predictions = _predict_matched(
                representation,
                X_eval,
                y_mean=y_mean,
                y_scale=y_scale,
            )
        deployed_count = input_width * latent_width + 2 * latent_width + 1
        parameter_formula = "D*K + K_encoder_bias + K_head + 1"

    if not np.isfinite(predictions).all() or predictions.shape != (len(X_eval),):
        raise ValueError("Low-complexity baseline produced invalid predictions.")
    state_hash = _state_hash(
        state,
        {
            "schema_version": LOW_COMPLEXITY_SCHEMA_VERSION,
            "config": config_record,
            "input_width": input_width,
            "frozen_epoch": frozen_epoch,
            "internal_split_hash": internal_split_hash,
        },
    )
    prediction_hash = _array_hash(np.asarray(predictions, dtype=np.float64))
    fitted_count = sum(int(value.size) for value in state.values())
    metadata = {
        "schema_version": LOW_COMPLEXITY_SCHEMA_VERSION,
        "config": config_record,
        "config_hash": stable_hash(config_record),
        "input_width": input_width,
        "train_row_count": len(X_fit),
        "evaluation_row_count": len(X_eval),
        "training_feature_dtype": str(X_fit.dtype),
        "requested_latent_width": latent_width,
        "actual_latent_width": latent_width,
        "strict_width_applied": True,
        "ridge_alpha": float(config.ridge_alpha),
        "frozen_epoch": frozen_epoch,
        "internal_split_hash": internal_split_hash,
        "internal_fit_source_id_hash": internal_fit_source_id_hash,
        "internal_validation_source_id_hash": internal_validation_source_id_hash,
        "internal_fit_index_hash": internal_fit_index_hash,
        "internal_validation_index_hash": internal_validation_index_hash,
        "internal_fit_count": internal_fit_count,
        "internal_validation_count": internal_validation_count,
        "fit_warnings": tuple(fit_warnings),
        "parameter_count_formula": parameter_formula,
        "deployed_predictive_parameter_count": deployed_count,
        "fitted_state_scalar_count": fitted_count,
        "state_hash": state_hash,
        "prediction_hash": prediction_hash,
        "evaluation_labels_received": False,
    }
    if method in {"small_bottleneck_mlp", "direct_mlp_regressor"}:
        metadata["matched_neural_architecture"] = "D->K(ReLU)->1"
        metadata["neural_model_state_hash"] = _state_hash(
            neural_state,
            {
                "input_width": input_width,
                "latent_width": latent_width,
                "frozen_epoch": frozen_epoch,
            },
        )
    return LowComplexityFitResult(
        method=method,
        requested_latent_width=latent_width,
        actual_latent_width=latent_width,
        predictions=np.asarray(predictions, dtype=np.float64),
        prediction_hash=prediction_hash,
        state_hash=state_hash,
        fitted_state_scalar_count=fitted_count,
        deployed_predictive_parameter_count=deployed_count,
        selected_epoch=frozen_epoch,
        internal_split_hash=internal_split_hash,
        internal_fit_source_id_hash=internal_fit_source_id_hash,
        internal_validation_source_id_hash=internal_validation_source_id_hash,
        metadata=metadata,
    )


def fit_predict_low_complexity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    train_source_ids: Sequence[str],
    config: LowComplexityConfig,
    train_group_ids: Sequence[str] | None = None,
) -> LowComplexityFitResult:
    """Runner-oriented alias requiring explicit audited training source IDs."""
    return fit_low_complexity_representation(
        X_train,
        y_train,
        X_eval,
        config,
        train_source_ids=train_source_ids,
        train_group_ids=train_group_ids,
    )


def _validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    config: LowComplexityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(config, LowComplexityConfig):
        raise TypeError("config must be a LowComplexityConfig.")
    if config.method not in LOW_COMPLEXITY_METHODS:
        raise ValueError(
            f"Unsupported low-complexity method {config.method!r}; "
            f"expected one of {LOW_COMPLEXITY_METHODS}."
        )
    X_fit = np.asarray(X_train, dtype=np.float32)
    X_eval = np.asarray(X_evaluation, dtype=np.float32)
    y_fit = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if X_fit.ndim != 2 or X_eval.ndim != 2:
        raise ValueError("Training and evaluation features must be 2D matrices.")
    if len(X_fit) != len(y_fit) or len(X_fit) < 3:
        raise ValueError("Training features/labels require at least three paired rows.")
    if len(X_eval) == 0 or X_eval.shape[1] != X_fit.shape[1]:
        raise ValueError("Evaluation features must be nonempty with matching width.")
    if X_fit.shape[1] < 2:
        raise ValueError("Low-complexity baselines require at least two features.")
    if not np.isfinite(X_fit).all() or not np.isfinite(X_eval).all():
        raise ValueError("Feature matrices must contain only finite values.")
    if not np.isfinite(y_fit).all():
        raise ValueError("Training labels must contain only finite values.")
    width = config.latent_width
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("latent_width must be a positive integer.")
    max_width = min(int(X_fit.shape[1]), int(len(X_fit) - 1))
    if width > max_width:
        raise ValueError(
            f"latent_width={width} exceeds strict common maximum {max_width}."
        )
    _finite_nonnegative(config.ridge_alpha, "ridge_alpha")
    _finite_positive(config.learning_rate, "learning_rate")
    _finite_nonnegative(config.weight_decay, "weight_decay")
    _positive_int(config.max_epochs, "max_epochs")
    _positive_int(config.patience, "patience")
    _positive_int(config.batch_size, "batch_size")
    _positive_int(config.pls_max_iter, "pls_max_iter")
    _finite_positive(config.pls_tolerance, "pls_tolerance")
    if (
        isinstance(config.random_state, bool)
        or not isinstance(config.random_state, int)
    ):
        raise ValueError("random_state must be an integer.")
    if not 0 < float(config.internal_valid_fraction) < 1:
        raise ValueError("internal_valid_fraction must be in (0, 1).")
    if float(np.std(y_fit)) < 1e-12 and config.method in {
        "partial_least_squares",
        "sparse_feature_selection",
    }:
        raise ValueError(f"{config.method} requires nonconstant training labels.")
    return X_fit, y_fit, X_eval


def _validate_source_ids(
    train_source_ids: Sequence[str] | None,
    n_rows: int,
) -> tuple[str, ...]:
    if train_source_ids is None:
        return tuple(f"row-{index:012d}" for index in range(n_rows))
    if isinstance(train_source_ids, (str, bytes)) or len(train_source_ids) != n_rows:
        raise ValueError("train_source_ids must contain one ID per training row.")
    source_ids = tuple(str(value) for value in train_source_ids)
    if any(not value for value in source_ids) or len(set(source_ids)) != n_rows:
        raise ValueError("train_source_ids must be nonempty and unique.")
    return source_ids


def _validate_group_ids(
    train_group_ids: Sequence[str] | None,
    n_rows: int,
) -> tuple[str, ...]:
    if train_group_ids is None:
        return tuple(f"row-{index:012d}" for index in range(n_rows))
    if isinstance(train_group_ids, (str, bytes)) or len(train_group_ids) != n_rows:
        raise ValueError("train_group_ids must contain one group ID per training row.")
    groups = tuple(str(value) for value in train_group_ids)
    if any(not value for value in groups) or len(set(groups)) < 2:
        raise ValueError("train_group_ids must contain at least two nonempty groups.")
    return groups


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    estimator = Ridge(alpha=float(alpha), solver="lsqr")
    estimator.fit(np.asarray(X, dtype=np.float32), y)
    return estimator


def _fit_tied_linear_autoencoder(
    X_train: np.ndarray,
    config: LowComplexityConfig,
) -> _TiedLinearAutoencoder:
    with _deterministic_torch(int(config.random_state)):
        model = _TiedLinearAutoencoder(
            input_width=X_train.shape[1],
            latent_width=int(config.latent_width),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        features = torch.from_numpy(X_train)
        for epoch in range(int(config.max_epochs)):
            for indices in _epoch_batches(
                len(features),
                batch_size=int(config.batch_size),
                random_state=int(config.random_state),
                epoch=epoch,
            ):
                batch = features[indices]
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                loss = nn.functional.mse_loss(reconstruction, batch)
                loss.backward()
                optimizer.step()
    model.eval()
    return model


def _encode_tied(
    model: _TiedLinearAutoencoder,
    features: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        return model.encode(torch.from_numpy(features)).numpy().astype(np.float32)


def _stable_f_regression_top_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        scores, _ = f_regression(X_train, y_train, center=True, force_finite=True)
    audit_warnings = tuple(
        f"{item.category.__name__}: {item.message}" for item in captured
    )
    scores = np.asarray(scores, dtype=np.float64)
    # scikit-learn 1.2 can round r**2 slightly above one and consequently
    # return a large negative F value for exact correlation.  F statistics
    # are nonnegative; this edge is the limiting positive-infinity case.
    scores = np.where(scores < 0.0, np.finfo(np.float64).max, scores)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    feature_indices = np.arange(X_train.shape[1], dtype=np.int64)
    ranked = np.lexsort((feature_indices, -scores))
    selected = np.sort(ranked[:width].astype(np.int64))
    if len(selected) != width:
        raise ValueError("Sparse feature selector could not select the requested width.")
    return selected, scores, audit_warnings


def _make_internal_group_split(
    group_ids: tuple[str, ...],
    *,
    valid_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    unique_groups = np.asarray(sorted(set(group_ids)), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("Neural epoch selection requires at least two groups.")
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)
    target_valid_rows = max(1, int(math.floor(len(group_ids) * valid_fraction)))
    valid_groups: set[str] = set()
    valid_rows = 0
    counts = {group: group_ids.count(group) for group in unique_groups}
    for group in unique_groups[:-1]:
        if valid_rows >= target_valid_rows:
            break
        valid_groups.add(str(group))
        valid_rows += counts[group]
    if not valid_groups:
        valid_groups.add(str(unique_groups[0]))
    valid_indices = np.asarray(
        [index for index, group in enumerate(group_ids) if group in valid_groups],
        dtype=np.int64,
    )
    fit_indices = np.asarray(
        [index for index, group in enumerate(group_ids) if group not in valid_groups],
        dtype=np.int64,
    )
    if len(fit_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("Neural internal group split produced an empty partition.")
    split_record = {
        "schema_version": "bh-low-complexity-internal-group-split-v1",
        "random_state": random_state,
        "valid_fraction": valid_fraction,
        "fit_indices": fit_indices.tolist(),
        "valid_indices": valid_indices.tolist(),
        "fit_groups": sorted(set(group_ids) - valid_groups),
        "valid_groups": sorted(valid_groups),
    }
    return fit_indices, valid_indices, stable_hash(split_record)


def _select_neural_epoch(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: LowComplexityConfig,
) -> int:
    y_mean, y_scale = _target_scaling(y_fit)
    fit_targets = torch.from_numpy(((y_fit - y_mean) / y_scale).astype(np.float32))
    valid_targets = torch.from_numpy(
        ((y_valid - y_mean) / y_scale).astype(np.float32)
    )
    with _deterministic_torch(int(config.random_state)):
        model = _MatchedBottleneckRegressor(
            input_width=X_fit.shape[1],
            latent_width=int(config.latent_width),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        fit_features = torch.from_numpy(X_fit)
        valid_features = torch.from_numpy(X_valid)
        best_epoch = 1
        best_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, int(config.max_epochs) + 1):
            model.train()
            for indices in _epoch_batches(
                len(fit_features),
                batch_size=int(config.batch_size),
                random_state=int(config.random_state),
                epoch=epoch - 1,
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(
                    model(fit_features[indices]),
                    fit_targets[indices],
                )
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(
                    nn.functional.mse_loss(
                        model(valid_features),
                        valid_targets,
                    ).item()
                )
            if valid_loss < best_loss - 1e-10:
                best_loss = valid_loss
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= int(config.patience):
                break
    return best_epoch


def _refit_matched_neural_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: LowComplexityConfig,
    *,
    epochs: int,
) -> _MatchedBottleneckRegressor:
    y_mean, y_scale = _target_scaling(y_train)
    targets = torch.from_numpy(((y_train - y_mean) / y_scale).astype(np.float32))
    with _deterministic_torch(int(config.random_state)):
        model = _MatchedBottleneckRegressor(
            input_width=X_train.shape[1],
            latent_width=int(config.latent_width),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        features = torch.from_numpy(X_train)
        for epoch in range(epochs):
            model.train()
            for indices in _epoch_batches(
                len(features),
                batch_size=int(config.batch_size),
                random_state=int(config.random_state),
                epoch=epoch,
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(
                    model(features[indices]),
                    targets[indices],
                )
                loss.backward()
                optimizer.step()
    model.eval()
    return model


def _encode_matched(
    model: _MatchedBottleneckRegressor,
    features: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        return model.encode(torch.from_numpy(features)).numpy().astype(np.float32)


def _predict_matched(
    model: _MatchedBottleneckRegressor,
    features: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    with torch.no_grad():
        scaled = model(torch.from_numpy(features)).numpy().astype(np.float64)
    return scaled * y_scale + y_mean


def _target_scaling(targets: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(targets, dtype=np.float64))
    scale = float(np.std(targets, dtype=np.float64))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return mean, scale


def _epoch_batches(
    n_rows: int,
    *,
    batch_size: int,
    random_state: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(random_state + epoch * 1_000_003)
    order = rng.permutation(n_rows).astype(np.int64)
    width = min(batch_size, n_rows)
    return tuple(order[start : start + width] for start in range(0, n_rows, width))


class _deterministic_torch:
    """Temporarily enable deterministic single-thread CPU torch execution."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __enter__(self) -> None:
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)

    def __exit__(self, *_: object) -> None:
        torch.set_num_threads(self.previous_threads)
        torch.use_deterministic_algorithms(self.previous_deterministic)


def _torch_state(module: nn.Module, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.{name}": value.detach().cpu().numpy().copy()
        for name, value in sorted(module.state_dict().items())
    }


def _numeric_fitted_state(estimator: Any, prefix: str) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name, value in sorted(vars(estimator).items()):
        if not name.endswith("_"):
            continue
        key = f"{prefix}.{name}"
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            state[key] = np.asarray(value).copy()
        elif isinstance(value, Number) and not isinstance(value, complex):
            state[key] = np.asarray([value])
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, Number) for item in value)
        ):
            state[key] = np.asarray(value)
    return state


def _state_hash(
    state: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(state):
        value = np.ascontiguousarray(state[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative.")


def _finite_positive(value: float, name: str) -> None:
    _finite_nonnegative(value, name)
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
