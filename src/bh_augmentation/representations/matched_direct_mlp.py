"""Deterministic parameter-matched direct-MLP control.

This control uses the same deployed ``D -> H -> L -> 1`` parameterization as
a supervised autoencoder encoder followed by its linear yield head. Epoch
selection uses only a group-disjoint split of *measured* training rows. The
selected epoch count is then refit from scratch on every training row before
predicting a label-free evaluation matrix.

The control accepts the same teacher-labelled synthetic transfer rows as the
supervised autoencoder so that "AE beats direct MLP" cannot be confounded with
access to a synthetic pool. Synthetic rows are weighted by
``synthetic_supervised_weight`` exactly as the autoencoder weights its
synthetic supervised loss, and they are excluded from both the target scaler
and the early-stopping monitor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from bh_augmentation.representations.redesigned_supervised_autoencoder import (
    fixed_denominator_weighted_mean,
)
from bh_augmentation.utils.corrected_runs import stable_hash

MATCHED_DIRECT_MLP_SCHEMA_VERSION = "bh-matched-direct-mlp-v1"


@dataclass(frozen=True, slots=True)
class MatchedDirectMLPConfig:
    """Strict configuration for the measured-only direct-MLP control."""

    input_dim: int
    hidden_dim: int
    latent_dim: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    internal_valid_fraction: float = 0.2
    random_state: int = 42
    synthetic_supervised_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class MatchedDirectMLPFitResult:
    """Predictions and immutable scientific audit for one fitted control."""

    predictions: np.ndarray
    prediction_hash: str
    state_hash: str
    deployed_predictor_parameter_count: int
    selected_epoch: int
    measured_train_source_id_hash: str
    synthetic_train_source_id_hash: str
    measured_train_group_assignment_hash: str
    internal_fit_source_id_hash: str
    internal_validation_source_id_hash: str
    internal_fit_group_id_hash: str
    internal_validation_group_id_hash: str
    internal_split_hash: str
    target_scaler_source_id_hash: str
    target_scaler_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _InternalGroupSplit:
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    fit_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    split_hash: str


class _MatchedDirectRegressor(nn.Module):
    """Exact ``D -> H -> L -> 1`` deployed predictor."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.yield_head = nn.Linear(latent_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.yield_head(self.encoder(features)).squeeze(-1)


def matched_direct_mlp_parameter_count(
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> int:
    """Return the exact deployed scalar count for ``D -> H -> L -> 1``."""
    dimensions = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
    }
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    return (
        input_dim * hidden_dim
        + hidden_dim
        + hidden_dim * latent_dim
        + latent_dim
        + latent_dim
        + 1
    )


def fit_predict_matched_direct_mlp(
    X_measured_train: np.ndarray,
    y_measured_train: np.ndarray,
    X_evaluation: np.ndarray,
    *,
    measured_source_ids: Sequence[str],
    measured_group_ids: Sequence[str],
    config: MatchedDirectMLPConfig,
    X_synthetic: np.ndarray | None = None,
    y_synthetic: np.ndarray | None = None,
    synthetic_source_ids: Sequence[str] = (),
    synthetic_group_ids: Sequence[str] = (),
) -> MatchedDirectMLPFitResult:
    """Fit measured (and optional synthetic) rows without evaluation labels."""
    X_train, y_train, X_eval = _validate_arrays(
        X_measured_train,
        y_measured_train,
        X_evaluation,
        config,
    )
    source_ids, group_ids = _validate_identities(
        measured_source_ids,
        measured_group_ids,
        len(X_train),
    )

    # Stable source ordering makes fitting independent of input traversal.
    order = np.asarray(
        sorted(range(len(source_ids)), key=lambda index: source_ids[index]),
        dtype=np.int64,
    )
    X_train = X_train[order]
    y_train = y_train[order]
    source_ids = tuple(source_ids[index] for index in order)
    group_ids = tuple(group_ids[index] for index in order)
    group_assignment = [
        {"source_row_id": source_id, "canonical_group": group_id}
        for source_id, group_id in zip(source_ids, group_ids, strict=True)
    ]
    measured_source_hash = stable_hash(list(source_ids))
    measured_group_hash = stable_hash(group_assignment)
    (
        X_added,
        y_added,
        added_source_ids,
        added_group_ids,
    ) = _validate_synthetic(
        X_synthetic,
        y_synthetic,
        synthetic_source_ids,
        synthetic_group_ids,
        config,
        measured_source_ids=source_ids,
    )
    synthetic_source_hash = stable_hash(list(added_source_ids))
    synthetic_weight = float(config.synthetic_supervised_weight)

    split = _make_group_disjoint_split(
        source_ids,
        group_ids,
        valid_fraction=float(config.internal_valid_fraction),
        random_state=int(config.random_state),
    )
    fit_source_ids = tuple(source_ids[index] for index in split.fit_indices)
    validation_source_ids = tuple(
        source_ids[index] for index in split.validation_indices
    )
    # Target scaling and the early-stopping monitor both stay measured-only.
    development_mean, development_scale = _target_scaling(
        y_train[split.fit_indices]
    )
    selected_epoch = _select_epoch(
        np.concatenate([X_train[split.fit_indices], X_added]),
        np.concatenate([y_train[split.fit_indices], y_added]),
        _supervised_weights(len(split.fit_indices), len(y_added), synthetic_weight),
        X_train[split.validation_indices],
        y_train[split.validation_indices],
        y_mean=development_mean,
        y_scale=development_scale,
        config=config,
    )

    final_mean, final_scale = _target_scaling(y_train)
    model = _fit_exact_epochs(
        np.concatenate([X_train, X_added]),
        np.concatenate([y_train, y_added]),
        _supervised_weights(len(y_train), len(y_added), synthetic_weight),
        y_mean=final_mean,
        y_scale=final_scale,
        epochs=selected_epoch,
        config=config,
    )
    predictions = _predict(
        model,
        X_eval,
        y_mean=final_mean,
        y_scale=final_scale,
    )
    state = {
        name: value.detach().cpu().numpy().copy()
        for name, value in sorted(model.state_dict().items())
    }
    scaler_source_hash = measured_source_hash
    scaler_record = {
        "source_id_hash": scaler_source_hash,
        "measured_row_count": len(source_ids),
        "y_mean": final_mean,
        "y_scale": final_scale,
    }
    scaler_hash = stable_hash(scaler_record)
    state_hash = _state_hash(
        state,
        {
            "schema_version": MATCHED_DIRECT_MLP_SCHEMA_VERSION,
            "config": asdict(config),
            "selected_epoch": selected_epoch,
            "measured_train_source_id_hash": measured_source_hash,
            "synthetic_train_source_id_hash": synthetic_source_hash,
            "synthetic_supervised_weight": synthetic_weight,
            "measured_train_group_assignment_hash": measured_group_hash,
            "internal_split_hash": split.split_hash,
            "target_scaler_hash": scaler_hash,
        },
    )
    prediction_hash = _array_hash(predictions)
    parameter_count = matched_direct_mlp_parameter_count(
        config.input_dim,
        config.hidden_dim,
        config.latent_dim,
    )
    actual_parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    if actual_parameter_count != parameter_count:
        raise RuntimeError(
            "Matched direct-MLP parameter accounting disagrees with the model."
        )
    metadata = {
        "schema_version": MATCHED_DIRECT_MLP_SCHEMA_VERSION,
        "architecture": "D->H(ReLU)->L->1",
        "resolved_config": asdict(config),
        "config_hash": stable_hash(asdict(config)),
        "parameter_count_formula": "D*H + H + H*L + L + L + 1",
        "deployed_predictor_parameter_count": parameter_count,
        "measured_train_row_count": len(source_ids),
        "synthetic_train_row_count": len(added_source_ids),
        "synthetic_train_source_id_hash": synthetic_source_hash,
        "synthetic_supervised_weight": synthetic_weight,
        "supervised_weight_semantics": (
            "measured=1,synthetic=synthetic_supervised_weight"
        ),
        "effective_supervised_weight_sum": float(
            len(source_ids) + len(added_source_ids) * synthetic_weight
        ),
        "evaluation_row_count": len(X_eval),
        "selected_epoch": selected_epoch,
        "epoch_selection_metric": "real_internal_validation_predictive_mse",
        "epoch_selection_rows": "group_disjoint_measured_training_rows_only",
        "epoch_fit_rows": "measured_epoch_fit_rows_plus_all_synthetic_rows",
        "refit_rows": "all_measured_training_rows_plus_all_synthetic_rows",
        "refit_epoch_count": selected_epoch,
        "development_target_scaler_source_id_hash": stable_hash(
            list(fit_source_ids)
        ),
        "development_target_scaler_hash": stable_hash(
            {
                "source_id_hash": stable_hash(list(fit_source_ids)),
                "y_mean": development_mean,
                "y_scale": development_scale,
            }
        ),
        "target_scaler_fit_rows": "all_measured_training_rows_only",
        "target_y_mean": final_mean,
        "target_y_scale": final_scale,
        "evaluation_labels_received": False,
        "internal_group_overlap_count": 0,
        "state_hash": state_hash,
        "prediction_hash": prediction_hash,
    }
    return MatchedDirectMLPFitResult(
        predictions=predictions,
        prediction_hash=prediction_hash,
        state_hash=state_hash,
        deployed_predictor_parameter_count=parameter_count,
        selected_epoch=selected_epoch,
        measured_train_source_id_hash=measured_source_hash,
        synthetic_train_source_id_hash=synthetic_source_hash,
        measured_train_group_assignment_hash=measured_group_hash,
        internal_fit_source_id_hash=stable_hash(list(fit_source_ids)),
        internal_validation_source_id_hash=stable_hash(
            list(validation_source_ids)
        ),
        internal_fit_group_id_hash=stable_hash(list(split.fit_groups)),
        internal_validation_group_id_hash=stable_hash(
            list(split.validation_groups)
        ),
        internal_split_hash=split.split_hash,
        target_scaler_source_id_hash=scaler_source_hash,
        target_scaler_hash=scaler_hash,
        metadata=metadata,
    )


def _validate_arrays(
    X_measured_train: np.ndarray,
    y_measured_train: np.ndarray,
    X_evaluation: np.ndarray,
    config: MatchedDirectMLPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(config, MatchedDirectMLPConfig):
        raise TypeError("config must be a MatchedDirectMLPConfig.")
    _validate_config(config)
    X_train = np.asarray(X_measured_train, dtype=np.float32)
    y_train = np.asarray(y_measured_train, dtype=np.float32).reshape(-1)
    X_eval = np.asarray(X_evaluation, dtype=np.float32)
    if X_train.ndim != 2 or X_eval.ndim != 2:
        raise ValueError("Training and evaluation features must be 2D.")
    if len(X_train) != len(y_train) or len(X_train) < 4:
        raise ValueError(
            "Measured training features and outcomes require at least four "
            "paired rows."
        )
    if X_train.shape[1] != config.input_dim:
        raise ValueError(
            "Measured training width does not match config.input_dim."
        )
    if len(X_eval) == 0 or X_eval.shape[1] != config.input_dim:
        raise ValueError(
            "Evaluation features must be nonempty with matching input width."
        )
    if (
        not np.isfinite(X_train).all()
        or not np.isfinite(y_train).all()
        or not np.isfinite(X_eval).all()
    ):
        raise ValueError("Features and measured outcomes must be finite.")
    return (
        np.ascontiguousarray(X_train),
        np.ascontiguousarray(y_train),
        np.ascontiguousarray(X_eval),
    )


def _validate_config(config: MatchedDirectMLPConfig) -> None:
    matched_direct_mlp_parameter_count(
        config.input_dim,
        config.hidden_dim,
        config.latent_dim,
    )
    for name in ("batch_size", "max_epochs", "patience"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        not isinstance(config.random_state, int)
        or isinstance(config.random_state, bool)
    ):
        raise ValueError("random_state must be an integer.")
    for name in (
        "learning_rate",
        "weight_decay",
        "internal_valid_fraction",
        "synthetic_supervised_weight",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be finite.")
    if float(config.learning_rate) <= 0:
        raise ValueError("learning_rate must be positive.")
    if float(config.weight_decay) < 0:
        raise ValueError("weight_decay must be non-negative.")
    if float(config.synthetic_supervised_weight) < 0:
        raise ValueError("synthetic_supervised_weight must be non-negative.")
    if not 0 < float(config.internal_valid_fraction) < 1:
        raise ValueError("internal_valid_fraction must be in (0, 1).")


def _supervised_weights(
    measured_count: int,
    synthetic_count: int,
    synthetic_weight: float,
) -> np.ndarray:
    """Mirror the AE weighting: measured rows 1.0, synthetic rows the weight."""
    return np.concatenate(
        [
            np.ones(measured_count, dtype=np.float32),
            np.full(synthetic_count, float(synthetic_weight), dtype=np.float32),
        ]
    )


def _validate_synthetic(
    X_synthetic: np.ndarray | None,
    y_synthetic: np.ndarray | None,
    synthetic_source_ids: Sequence[str],
    synthetic_group_ids: Sequence[str],
    config: MatchedDirectMLPConfig,
    *,
    measured_source_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    X_added = (
        np.empty((0, config.input_dim), dtype=np.float32)
        if X_synthetic is None
        else np.asarray(X_synthetic, dtype=np.float32)
    )
    y_added = (
        np.empty(0, dtype=np.float32)
        if y_synthetic is None
        else np.asarray(y_synthetic, dtype=np.float32).reshape(-1)
    )
    if (
        X_added.ndim != 2
        or X_added.shape[1] != config.input_dim
        or len(X_added) != len(y_added)
        or not np.isfinite(X_added).all()
        or not np.isfinite(y_added).all()
    ):
        raise ValueError("Synthetic features and outcomes must be finite and aligned.")
    if isinstance(synthetic_source_ids, (str, bytes)) or len(
        synthetic_source_ids
    ) != len(X_added):
        raise ValueError("synthetic_source_ids must contain one ID per row.")
    if isinstance(synthetic_group_ids, (str, bytes)) or len(
        synthetic_group_ids
    ) != len(X_added):
        raise ValueError("synthetic_group_ids must contain one group per row.")
    added_source_ids = tuple(str(value).strip() for value in synthetic_source_ids)
    added_group_ids = tuple(str(value).strip() for value in synthetic_group_ids)
    if (
        any(not value for value in added_source_ids)
        or len(set(added_source_ids)) != len(X_added)
        or set(added_source_ids) & set(measured_source_ids)
    ):
        raise ValueError("Synthetic source IDs must be unique and disjoint from measured.")
    if any(not value for value in added_group_ids):
        raise ValueError("Synthetic group IDs must be nonempty.")
    order = np.asarray(
        sorted(range(len(added_source_ids)), key=lambda index: added_source_ids[index]),
        dtype=np.int64,
    )
    return (
        np.ascontiguousarray(X_added[order]),
        np.ascontiguousarray(y_added[order]),
        tuple(added_source_ids[index] for index in order),
        tuple(added_group_ids[index] for index in order),
    )


def _validate_identities(
    measured_source_ids: Sequence[str],
    measured_group_ids: Sequence[str],
    n_rows: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(measured_source_ids, (str, bytes)) or len(
        measured_source_ids
    ) != n_rows:
        raise ValueError("measured_source_ids must contain one ID per row.")
    if isinstance(measured_group_ids, (str, bytes)) or len(
        measured_group_ids
    ) != n_rows:
        raise ValueError("measured_group_ids must contain one ID per row.")
    source_ids = tuple(str(value).strip() for value in measured_source_ids)
    group_ids = tuple(str(value).strip() for value in measured_group_ids)
    if (
        any(not value for value in source_ids)
        or len(set(source_ids)) != n_rows
    ):
        raise ValueError("Measured source IDs must be nonempty and unique.")
    if any(not value for value in group_ids) or len(set(group_ids)) < 2:
        raise ValueError(
            "Measured group IDs must contain at least two nonempty groups."
        )
    return source_ids, group_ids


def _make_group_disjoint_split(
    source_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    *,
    valid_fraction: float,
    random_state: int,
) -> _InternalGroupSplit:
    unique_groups = tuple(sorted(set(group_ids)))
    if len(unique_groups) < 2:
        raise ValueError(
            "Internal validation requires at least two measured groups."
        )
    group_sizes = {
        group: sum(candidate == group for candidate in group_ids)
        for group in unique_groups
    }
    traversal = sorted(
        unique_groups,
        key=lambda group: (
            stable_hash(
                {
                    "schema_version": MATCHED_DIRECT_MLP_SCHEMA_VERSION,
                    "random_state": random_state,
                    "group_id": group,
                }
            ),
            group,
        ),
    )
    target_validation_rows = max(
        1, int(math.floor(len(source_ids) * valid_fraction))
    )
    validation_groups: list[str] = []
    validation_rows = 0
    for group in traversal[:-1]:
        validation_groups.append(group)
        validation_rows += group_sizes[group]
        if validation_rows >= target_validation_rows:
            break
    validation_group_set = set(validation_groups)
    fit_groups = tuple(
        sorted(set(unique_groups) - validation_group_set)
    )
    validation_groups_tuple = tuple(sorted(validation_group_set))
    fit_indices = np.asarray(
        [
            index
            for index, group in enumerate(group_ids)
            if group not in validation_group_set
        ],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [
            index
            for index, group in enumerate(group_ids)
            if group in validation_group_set
        ],
        dtype=np.int64,
    )
    if (
        len(fit_indices) == 0
        or len(validation_indices) == 0
        or set(fit_groups) & set(validation_groups_tuple)
    ):
        raise RuntimeError("Internal measured group split is not disjoint.")
    split_record = {
        "schema_version": MATCHED_DIRECT_MLP_SCHEMA_VERSION,
        "random_state": random_state,
        "valid_fraction": valid_fraction,
        "fit_source_ids": [source_ids[index] for index in fit_indices],
        "validation_source_ids": [
            source_ids[index] for index in validation_indices
        ],
        "fit_groups": list(fit_groups),
        "validation_groups": list(validation_groups_tuple),
    }
    return _InternalGroupSplit(
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        fit_groups=fit_groups,
        validation_groups=validation_groups_tuple,
        split_hash=stable_hash(split_record),
    )


def _select_epoch(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    sample_weights: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
    config: MatchedDirectMLPConfig,
) -> int:
    with _deterministic_torch(config.random_state):
        model = _MatchedDirectRegressor(
            config.input_dim,
            config.hidden_dim,
            config.latent_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        fit_features = torch.from_numpy(X_fit)
        fit_targets = torch.from_numpy(
            ((y_fit - y_mean) / y_scale).astype(np.float32)
        )
        fit_weights = torch.from_numpy(sample_weights)
        validation_features = torch.from_numpy(X_validation)
        best_epoch = 1
        best_mse = float("inf")
        stale_epochs = 0
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            for indices in _epoch_batches(
                len(X_fit),
                batch_size=config.batch_size,
                random_state=config.random_state,
                epoch=epoch - 1,
            ):
                batch_weights = fit_weights[indices]
                if float(batch_weights.sum()) == 0.0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                prediction = model(fit_features[indices])
                loss = fixed_denominator_weighted_mean(
                    nn.functional.mse_loss(
                        prediction,
                        fit_targets[indices],
                        reduction="none",
                    ),
                    batch_weights,
                )
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                scaled_prediction = model(validation_features).numpy()
            prediction = scaled_prediction.astype(float) * y_scale + y_mean
            validation_mse = float(
                np.mean(
                    np.square(
                        prediction - np.asarray(y_validation, dtype=float)
                    )
                )
            )
            if validation_mse < best_mse - 1e-12:
                best_mse = validation_mse
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    return best_epoch


def _fit_exact_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
    epochs: int,
    config: MatchedDirectMLPConfig,
) -> _MatchedDirectRegressor:
    with _deterministic_torch(config.random_state):
        model = _MatchedDirectRegressor(
            config.input_dim,
            config.hidden_dim,
            config.latent_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        features = torch.from_numpy(X_train)
        targets = torch.from_numpy(
            ((y_train - y_mean) / y_scale).astype(np.float32)
        )
        weights = torch.from_numpy(sample_weights)
        for epoch in range(epochs):
            model.train()
            for indices in _epoch_batches(
                len(X_train),
                batch_size=config.batch_size,
                random_state=config.random_state,
                epoch=epoch,
            ):
                batch_weights = weights[indices]
                if float(batch_weights.sum()) == 0.0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = fixed_denominator_weighted_mean(
                    nn.functional.mse_loss(
                        model(features[indices]),
                        targets[indices],
                        reduction="none",
                    ),
                    batch_weights,
                )
                loss.backward()
                optimizer.step()
    model.eval()
    return model


def _predict(
    model: _MatchedDirectRegressor,
    X_evaluation: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    with torch.no_grad():
        scaled = model(torch.from_numpy(X_evaluation)).numpy().astype(float)
    prediction = np.asarray(scaled * y_scale + y_mean, dtype=np.float64)
    if prediction.shape != (len(X_evaluation),) or not np.isfinite(
        prediction
    ).all():
        raise ValueError("Matched direct MLP produced invalid predictions.")
    return prediction


def _target_scaling(targets: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(targets, dtype=np.float64))
    scale = float(np.std(targets, dtype=np.float64))
    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError("Measured target scaling must be finite.")
    if scale < 1e-8:
        scale = 1.0
    return mean, scale


def _epoch_batches(
    n_rows: int,
    *,
    batch_size: int,
    random_state: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(random_state + epoch * 1_000_003)
    order = generator.permutation(n_rows).astype(np.int64)
    width = min(batch_size, n_rows)
    return tuple(
        order[start : start + width]
        for start in range(0, n_rows, width)
    )


class _deterministic_torch(AbstractContextManager[None]):
    """Temporarily enforce deterministic single-thread CPU execution."""

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state

    def __enter__(self) -> None:
        self._previous_deterministic = (
            torch.are_deterministic_algorithms_enabled()
        )
        self._previous_threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        return None

    def __exit__(self, *args: object) -> None:
        torch.set_num_threads(self._previous_threads)
        torch.use_deterministic_algorithms(self._previous_deterministic)


def _state_hash(
    state: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    for name in sorted(state):
        value = np.ascontiguousarray(state[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()
