"""Leakage-resistant, provenance-aware supervised autoencoder core."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from bh_augmentation.utils.corrected_runs import stable_hash

REDESIGNED_AE_SCHEMA_VERSION = "bh-redesigned-supervised-ae-v1"
RECONSTRUCTION_OBJECTIVES = (
    "mse",
    "positive_bit_weighted_mse",
    "binary_cross_entropy",
    "count_aware_mse",
)
SYNTHETIC_WEIGHT_CHOICES = (0.0, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class RoleBlock:
    """One contiguous molecular-role feature block."""

    name: str
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class RedesignedSupervisedAEConfig:
    """Configuration for a compact D -> H -> L supervised autoencoder."""

    input_dim: int
    hidden_dim: int
    latent_dim: int
    reconstruction_objective: str = "positive_bit_weighted_mse"
    role_blocks: tuple[RoleBlock, ...] = ()
    role_balanced_reconstruction: bool = True
    positive_bit_weight: float = 5.0
    reconstruction_weight: float = 1.0
    supervised_weight: float = 1.0
    synthetic_reconstruction_weight: float = 0.25
    synthetic_supervised_weight: float = 0.25
    masking_probability: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    internal_valid_fraction: float = 0.2
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class MeasuredInternalSplit:
    """Exact measured fit/validation boundary used for AE epoch selection."""

    measured_fit_indices: tuple[int, ...]
    measured_validation_indices: tuple[int, ...]
    measured_fit_source_ids: tuple[str, ...]
    measured_validation_source_ids: tuple[str, ...]
    synthetic_source_ids: tuple[str, ...]
    measured_fit_source_id_hash: str
    measured_validation_source_id_hash: str
    synthetic_source_id_hash: str
    all_source_id_hash: str
    split_hash: str


@dataclass(frozen=True, slots=True)
class RedesignedSupervisedAEFit:
    """Encoded features, predictions, and complete training audit."""

    train_latent: np.ndarray
    evaluation_latent: np.ndarray
    yield_predictions: np.ndarray
    selected_epoch: int
    parameter_count: int
    state_hash: str
    prediction_hash: str
    train_latent_hash: str
    evaluation_latent_hash: str
    measured_scaling_source_id_hash: str
    synthetic_source_id_hash: str
    internal_fit_source_id_hash: str
    internal_validation_source_id_hash: str
    internal_split_hash: str
    target_mean: float
    target_scale: float
    training_reconstruction_metrics: Mapping[
        str, Mapping[str, float | int | None]
    ]
    evaluation_reconstruction_metrics: Mapping[str, float | int | None]
    metadata: Mapping[str, Any]


class RedesignedSupervisedAutoencoder(nn.Module):
    """Compact symmetric AE with a supervised latent yield head."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder_input = nn.Linear(input_dim, hidden_dim)
        self.encoder_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        self.yield_head = nn.Linear(latent_dim, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.encoder_input(features))
        return self.encoder_latent(hidden)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(features)
        decoder_hidden = torch.relu(self.decoder_hidden(latent))
        reconstruction_raw = self.decoder_output(decoder_hidden)
        yield_scaled = self.yield_head(latent).squeeze(-1)
        return latent, reconstruction_raw, yield_scaled


def supervised_ae_parameter_count(
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> int:
    """Return the exact number of learned scalars in the symmetric AE."""
    return (
        2 * input_dim * hidden_dim
        + 2 * hidden_dim * latent_dim
        + 2 * hidden_dim
        + input_dim
        + 2 * latent_dim
        + 1
    )


def build_measured_internal_split(
    train_source_ids: Sequence[str],
    sample_origins: Sequence[str],
    *,
    train_group_ids: Sequence[str] | None = None,
    valid_fraction: float = 0.2,
    random_state: int = 42,
) -> MeasuredInternalSplit:
    """Build the public measured-only epoch-fit boundary.

    Runners should call this before generating synthetic examples and permit
    transfers to use only ``measured_fit_source_ids`` as sources and donors.
    Every supplied source ID is represented exactly once in the hashed split
    membership.
    """
    source_ids = _validate_source_ids(train_source_ids, len(train_source_ids))
    origins = _validate_origins(sample_origins, len(source_ids))
    groups = _validate_groups(train_group_ids, source_ids)
    if not 0.0 < float(valid_fraction) < 1.0:
        raise ValueError("valid_fraction must be in (0, 1).")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer.")
    measured_indices = np.asarray(
        [index for index, origin in enumerate(origins) if origin == "measured"],
        dtype=np.int64,
    )
    if len(measured_indices) < 3:
        raise ValueError("At least three measured rows are required.")
    measured_groups = tuple(groups[index] for index in measured_indices)
    fit_local, validation_local, _ = _make_measured_internal_split(
        measured_groups,
        valid_fraction=valid_fraction,
        random_state=random_state,
    )
    fit_indices = measured_indices[fit_local]
    validation_indices = measured_indices[validation_local]
    fit_ids = tuple(source_ids[index] for index in fit_indices)
    validation_ids = tuple(source_ids[index] for index in validation_indices)
    synthetic_ids = tuple(
        source_id
        for source_id, origin in zip(source_ids, origins, strict=True)
        if origin == "synthetic"
    )
    fit_groups = {groups[index] for index in fit_indices}
    validation_groups = {groups[index] for index in validation_indices}
    if fit_groups & validation_groups:
        raise RuntimeError("Measured internal split is not group-disjoint.")
    accounted_ids = set(fit_ids) | set(validation_ids) | set(synthetic_ids)
    if (
        len(accounted_ids) != len(source_ids)
        or accounted_ids != set(source_ids)
        or set(fit_ids) & set(validation_ids)
        or set(fit_ids) & set(synthetic_ids)
        or set(validation_ids) & set(synthetic_ids)
    ):
        raise RuntimeError("Measured internal split does not account for every source.")
    membership = []
    fit_id_set = set(fit_ids)
    validation_id_set = set(validation_ids)
    for source_id, origin, group in zip(source_ids, origins, groups, strict=True):
        if source_id in fit_id_set:
            role = "measured_epoch_fit"
        elif source_id in validation_id_set:
            role = "measured_internal_validation"
        else:
            role = "synthetic_not_in_measured_split"
        membership.append(
            {
                "source_row_id": source_id,
                "sample_origin": origin,
                "group_id": group,
                "internal_role": role,
            }
        )
    split_record = {
        "schema_version": "bh-redesigned-ae-public-measured-split-v1",
        "valid_fraction": float(valid_fraction),
        "random_state": random_state,
        "membership": membership,
    }
    return MeasuredInternalSplit(
        measured_fit_indices=tuple(int(value) for value in fit_indices),
        measured_validation_indices=tuple(int(value) for value in validation_indices),
        measured_fit_source_ids=fit_ids,
        measured_validation_source_ids=validation_ids,
        synthetic_source_ids=synthetic_ids,
        measured_fit_source_id_hash=stable_hash(list(fit_ids)),
        measured_validation_source_id_hash=stable_hash(list(validation_ids)),
        synthetic_source_id_hash=stable_hash(list(synthetic_ids)),
        all_source_id_hash=stable_hash(list(source_ids)),
        split_hash=stable_hash(split_record),
    )


def measured_validation_yield_mse(
    predicted_yield_scaled: np.ndarray,
    measured_yield: np.ndarray,
    *,
    target_mean: float,
    target_scale: float,
) -> float:
    """Compute the sole early-stopping monitor in original yield units."""
    predicted = np.asarray(predicted_yield_scaled, dtype=np.float64).reshape(-1)
    observed = np.asarray(measured_yield, dtype=np.float64).reshape(-1)
    if (
        len(predicted) == 0
        or predicted.shape != observed.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(target_mean)
        or not np.isfinite(target_scale)
        or target_scale <= 0.0
    ):
        raise ValueError("Invalid inputs for measured validation yield MSE.")
    predicted_original = predicted * target_scale + target_mean
    return float(np.mean((predicted_original - observed) ** 2))


def fit_redesigned_supervised_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    *,
    train_source_ids: Sequence[str],
    sample_origins: Sequence[str],
    config: RedesignedSupervisedAEConfig,
    train_group_ids: Sequence[str] | None = None,
) -> RedesignedSupervisedAEFit:
    """Fit using measured/synthetic training rows and label-free evaluation X.

    Epoch selection uses only a train-derived, measured-only validation split.
    The selected epoch is then frozen and a fresh model is fit on all training
    rows.  Target scaling always uses measured outcomes only.
    """
    X_fit, y_fit, X_eval = _validate_arrays(X_train, y_train, X_evaluation, config)
    source_ids = _validate_source_ids(train_source_ids, len(X_fit))
    origins = _validate_origins(sample_origins, len(X_fit))
    groups = _validate_groups(train_group_ids, source_ids)
    measured_mask = np.asarray([origin == "measured" for origin in origins])
    synthetic_mask = ~measured_mask
    measured_indices = np.flatnonzero(measured_mask).astype(np.int64)
    measured_split = build_measured_internal_split(
        source_ids,
        origins,
        train_group_ids=groups,
        valid_fraction=float(config.internal_valid_fraction),
        random_state=int(config.random_state),
    )
    measured_fit_indices = np.asarray(
        measured_split.measured_fit_indices,
        dtype=np.int64,
    )
    measured_valid_indices = np.asarray(
        measured_split.measured_validation_indices,
        dtype=np.int64,
    )
    internal_split_hash = measured_split.split_hash
    epoch_fit_indices = np.concatenate(
        [measured_fit_indices, np.flatnonzero(synthetic_mask)]
    ).astype(np.int64)
    internal_fit_source_id_hash = measured_split.measured_fit_source_id_hash
    internal_validation_source_id_hash = (
        measured_split.measured_validation_source_id_hash
    )
    epoch_fit_source_id_hash = stable_hash(
        [source_ids[index] for index in epoch_fit_indices]
    )

    early_mean, early_scale = _target_scaling(y_fit[measured_fit_indices])
    selected_epoch = _select_epoch(
        X_fit[epoch_fit_indices],
        y_fit[epoch_fit_indices],
        tuple(origins[index] for index in epoch_fit_indices),
        X_fit[measured_valid_indices],
        y_fit[measured_valid_indices],
        target_mean=early_mean,
        target_scale=early_scale,
        config=config,
    )
    target_mean, target_scale = _target_scaling(y_fit[measured_indices])
    model = _refit_all_training_rows(
        X_fit,
        y_fit,
        origins,
        target_mean=target_mean,
        target_scale=target_scale,
        epochs=selected_epoch,
        config=config,
    )
    model.eval()
    with torch.no_grad():
        train_tensor = torch.from_numpy(X_fit)
        eval_tensor = torch.from_numpy(X_eval)
        train_latent_tensor, train_reconstruction_raw, _ = model(train_tensor)
        eval_latent_tensor, eval_reconstruction_raw, eval_yield_scaled = model(
            eval_tensor
        )
    train_latent = train_latent_tensor.numpy().astype(np.float32)
    evaluation_latent = eval_latent_tensor.numpy().astype(np.float32)
    yield_predictions = (
        eval_yield_scaled.numpy().astype(np.float64) * target_scale + target_mean
    )
    train_reconstructed = _decoded_features(train_reconstruction_raw, config)
    eval_reconstructed = _decoded_features(eval_reconstruction_raw, config)
    training_metrics = {
        "measured": _reconstruction_metrics(
            torch.from_numpy(X_fit[measured_mask]),
            train_reconstructed[measured_mask],
        ),
        "synthetic": _reconstruction_metrics(
            torch.from_numpy(X_fit[synthetic_mask]),
            train_reconstructed[synthetic_mask],
        ),
    }
    evaluation_metrics = _reconstruction_metrics(
        torch.from_numpy(X_eval),
        eval_reconstructed,
    )

    parameter_count = supervised_ae_parameter_count(
        config.input_dim,
        config.hidden_dim,
        config.latent_dim,
    )
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != actual_parameter_count:
        raise RuntimeError("Supervised AE parameter-count formula mismatch.")
    state = {
        name: value.detach().cpu().numpy().copy()
        for name, value in sorted(model.state_dict().items())
    }
    state["target_mean"] = np.asarray([target_mean], dtype=np.float64)
    state["target_scale"] = np.asarray([target_scale], dtype=np.float64)
    config_record = _config_record(config)
    state_hash = _state_hash(
        state,
        {
            "schema_version": REDESIGNED_AE_SCHEMA_VERSION,
            "config": config_record,
            "selected_epoch": selected_epoch,
            "internal_split_hash": internal_split_hash,
        },
    )
    prediction_hash = _array_hash(yield_predictions)
    train_latent_hash = _array_hash(train_latent)
    evaluation_latent_hash = _array_hash(evaluation_latent)
    measured_source_ids = tuple(source_ids[index] for index in measured_indices)
    synthetic_source_ids = tuple(source_ids[index] for index in np.flatnonzero(synthetic_mask))
    measured_scaling_source_id_hash = stable_hash(list(measured_source_ids))
    synthetic_source_id_hash = stable_hash(list(synthetic_source_ids))
    provenance_records = [
        {"source_row_id": source_id, "sample_origin": origin}
        for source_id, origin in zip(source_ids, origins, strict=True)
    ]
    metadata = {
        "schema_version": REDESIGNED_AE_SCHEMA_VERSION,
        "config": config_record,
        "config_hash": stable_hash(config_record),
        "architecture": (
            f"{config.input_dim}->{config.hidden_dim}->{config.latent_dim}"
        ),
        "parameter_count": parameter_count,
        "parameter_count_formula": "2*D*H + 2*H*L + 2*H + D + 2*L + 1",
        "train_row_count": len(X_fit),
        "evaluation_row_count": len(X_eval),
        "measured_row_count": int(measured_mask.sum()),
        "synthetic_row_count": int(synthetic_mask.sum()),
        "sample_provenance_hash": stable_hash(provenance_records),
        "measured_scaling_source_id_hash": measured_scaling_source_id_hash,
        "synthetic_source_id_hash": synthetic_source_id_hash,
        "target_scaling": "measured_training_rows_only",
        "target_mean": target_mean,
        "target_scale": target_scale,
        "selected_epoch": selected_epoch,
        "early_stopping_rows": "train_derived_measured_only",
        "early_stopping_monitor": "real_internal_validation_yield_mse_only",
        "early_stopping_tie_break": "earliest_epoch",
        "internal_fit_source_id_hash": internal_fit_source_id_hash,
        "internal_validation_source_id_hash": (
            internal_validation_source_id_hash
        ),
        "internal_fit_count": len(measured_fit_indices),
        "internal_validation_count": len(measured_valid_indices),
        "epoch_fit_source_id_hash": epoch_fit_source_id_hash,
        "epoch_fit_count": len(epoch_fit_indices),
        "epoch_fit_measured_count": len(measured_fit_indices),
        "epoch_fit_synthetic_count": int(synthetic_mask.sum()),
        "internal_split_hash": internal_split_hash,
        "synthetic_supervised_weight": config.synthetic_supervised_weight,
        "synthetic_reconstruction_weight": (
            config.synthetic_reconstruction_weight
        ),
        "effective_supervised_weight_sum": float(
            measured_mask.sum()
            + synthetic_mask.sum() * config.synthetic_supervised_weight
        ),
        "effective_reconstruction_weight_sum": float(
            measured_mask.sum()
            + synthetic_mask.sum() * config.synthetic_reconstruction_weight
        ),
        "masking_probability": config.masking_probability,
        "evaluation_labels_received": False,
        "state_hash": state_hash,
        "prediction_hash": prediction_hash,
        "train_latent_hash": train_latent_hash,
        "evaluation_latent_hash": evaluation_latent_hash,
    }
    return RedesignedSupervisedAEFit(
        train_latent=train_latent,
        evaluation_latent=evaluation_latent,
        yield_predictions=yield_predictions,
        selected_epoch=selected_epoch,
        parameter_count=parameter_count,
        state_hash=state_hash,
        prediction_hash=prediction_hash,
        train_latent_hash=train_latent_hash,
        evaluation_latent_hash=evaluation_latent_hash,
        measured_scaling_source_id_hash=measured_scaling_source_id_hash,
        synthetic_source_id_hash=synthetic_source_id_hash,
        internal_fit_source_id_hash=internal_fit_source_id_hash,
        internal_validation_source_id_hash=internal_validation_source_id_hash,
        internal_split_hash=internal_split_hash,
        target_mean=target_mean,
        target_scale=target_scale,
        training_reconstruction_metrics=training_metrics,
        evaluation_reconstruction_metrics=evaluation_metrics,
        metadata=metadata,
    )


def _validate_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    config: RedesignedSupervisedAEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(config, RedesignedSupervisedAEConfig):
        raise TypeError("config must be a RedesignedSupervisedAEConfig.")
    _validate_config(config)
    X_fit = np.asarray(X_train, dtype=np.float32)
    X_eval = np.asarray(X_evaluation, dtype=np.float32)
    y_fit = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if X_fit.ndim != 2 or X_eval.ndim != 2:
        raise ValueError("Training and evaluation features must be 2D.")
    if len(X_fit) != len(y_fit) or len(X_fit) < 3:
        raise ValueError("Training features and labels must contain paired rows.")
    if (
        len(X_eval) == 0
        or X_fit.shape[1] != config.input_dim
        or X_eval.shape[1] != config.input_dim
    ):
        raise ValueError("Feature widths must match config.input_dim.")
    if (
        not np.isfinite(X_fit).all()
        or not np.isfinite(X_eval).all()
        or not np.isfinite(y_fit).all()
    ):
        raise ValueError("Features and training labels must be finite.")
    if config.reconstruction_objective == "binary_cross_entropy":
        if not (
            np.logical_or(X_fit == 0.0, X_fit == 1.0).all()
            and np.logical_or(X_eval == 0.0, X_eval == 1.0).all()
        ):
            raise ValueError("Binary cross-entropy requires binary features.")
    if config.reconstruction_objective == "count_aware_mse":
        if (X_fit < 0.0).any() or (X_eval < 0.0).any():
            raise ValueError("Count-aware reconstruction requires nonnegative features.")
    return X_fit, y_fit, X_eval


def _validate_config(config: RedesignedSupervisedAEConfig) -> None:
    for name in ("input_dim", "hidden_dim", "latent_dim", "batch_size", "max_epochs", "patience"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.reconstruction_objective not in RECONSTRUCTION_OBJECTIVES:
        raise ValueError("Unsupported reconstruction_objective.")
    if not isinstance(config.role_balanced_reconstruction, bool):
        raise ValueError("role_balanced_reconstruction must be boolean.")
    for name in (
        "positive_bit_weight",
        "reconstruction_weight",
        "supervised_weight",
        "learning_rate",
    ):
        _finite_positive(getattr(config, name), name)
    _finite_nonnegative(config.weight_decay, "weight_decay")
    if config.synthetic_reconstruction_weight not in SYNTHETIC_WEIGHT_CHOICES:
        raise ValueError("Unsupported synthetic_reconstruction_weight.")
    if config.synthetic_supervised_weight not in SYNTHETIC_WEIGHT_CHOICES:
        raise ValueError("Unsupported synthetic_supervised_weight.")
    if not 0.0 <= float(config.masking_probability) < 1.0:
        raise ValueError("masking_probability must be in [0, 1).")
    if not 0.0 < float(config.internal_valid_fraction) < 1.0:
        raise ValueError("internal_valid_fraction must be in (0, 1).")
    if (
        not isinstance(config.random_state, int)
        or isinstance(config.random_state, bool)
    ):
        raise ValueError("random_state must be an integer.")
    blocks = _effective_role_blocks(config)
    expected_start = 0
    names: set[str] = set()
    for block in blocks:
        if (
            not block.name
            or block.name in names
            or block.start != expected_start
            or block.stop <= block.start
        ):
            raise ValueError("role_blocks must uniquely and contiguously cover input_dim.")
        names.add(block.name)
        expected_start = block.stop
    if expected_start != config.input_dim:
        raise ValueError("role_blocks must exactly cover input_dim.")


def _validate_source_ids(
    source_ids: Sequence[str],
    n_rows: int,
) -> tuple[str, ...]:
    if isinstance(source_ids, (str, bytes)) or len(source_ids) != n_rows:
        raise ValueError("train_source_ids must contain one ID per training row.")
    normalized = tuple(str(value) for value in source_ids)
    if any(not value for value in normalized) or len(set(normalized)) != n_rows:
        raise ValueError("train_source_ids must be nonempty and unique.")
    return normalized


def _validate_origins(origins: Sequence[str], n_rows: int) -> tuple[str, ...]:
    if isinstance(origins, (str, bytes)) or len(origins) != n_rows:
        raise ValueError("sample_origins must contain one value per training row.")
    normalized = tuple(str(value) for value in origins)
    if any(value not in {"measured", "synthetic"} for value in normalized):
        raise ValueError("sample_origins values must be 'measured' or 'synthetic'.")
    return normalized


def _validate_groups(
    group_ids: Sequence[str] | None,
    source_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if group_ids is None:
        return source_ids
    if isinstance(group_ids, (str, bytes)) or len(group_ids) != len(source_ids):
        raise ValueError("train_group_ids must contain one value per training row.")
    normalized = tuple(str(value) for value in group_ids)
    if any(not value for value in normalized):
        raise ValueError("train_group_ids must be nonempty.")
    return normalized


def _make_measured_internal_split(
    measured_groups: tuple[str, ...],
    *,
    valid_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    unique_groups = np.asarray(sorted(set(measured_groups)), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("Measured early stopping requires at least two groups.")
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)
    target_rows = max(1, int(math.floor(len(measured_groups) * valid_fraction)))
    group_counts = {group: measured_groups.count(group) for group in unique_groups}
    valid_groups: set[str] = set()
    valid_rows = 0
    for group in unique_groups[:-1]:
        if valid_rows >= target_rows:
            break
        valid_groups.add(str(group))
        valid_rows += group_counts[group]
    if not valid_groups:
        valid_groups.add(str(unique_groups[0]))
    valid_indices = np.asarray(
        [
            index
            for index, group in enumerate(measured_groups)
            if group in valid_groups
        ],
        dtype=np.int64,
    )
    fit_indices = np.asarray(
        [
            index
            for index, group in enumerate(measured_groups)
            if group not in valid_groups
        ],
        dtype=np.int64,
    )
    if len(fit_indices) == 0 or len(valid_indices) == 0:
        raise ValueError("Measured internal split produced an empty partition.")
    record = {
        "schema_version": "bh-redesigned-ae-measured-internal-split-v1",
        "valid_fraction": valid_fraction,
        "random_state": random_state,
        "fit_indices": fit_indices.tolist(),
        "validation_indices": valid_indices.tolist(),
        "fit_groups": sorted(set(measured_groups) - valid_groups),
        "validation_groups": sorted(valid_groups),
    }
    return fit_indices, valid_indices, stable_hash(record)


def _select_epoch(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    origins: tuple[str, ...],
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    target_mean: float,
    target_scale: float,
    config: RedesignedSupervisedAEConfig,
) -> int:
    with _deterministic_torch(config.random_state):
        model = RedesignedSupervisedAutoencoder(
            config.input_dim,
            config.hidden_dim,
            config.latent_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_epoch = 1
        best_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, config.max_epochs + 1):
            _train_epoch(
                model,
                optimizer,
                X_fit,
                y_fit,
                origins,
                target_mean=target_mean,
                target_scale=target_scale,
                config=config,
                epoch=epoch - 1,
            )
            valid_loss = _real_validation_predictive_yield_mse(
                model,
                X_valid,
                y_valid,
                target_mean=target_mean,
                target_scale=target_scale,
            )
            if valid_loss < best_loss - 1e-10:
                best_loss = valid_loss
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    return best_epoch


def _refit_all_training_rows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    origins: tuple[str, ...],
    *,
    target_mean: float,
    target_scale: float,
    epochs: int,
    config: RedesignedSupervisedAEConfig,
) -> RedesignedSupervisedAutoencoder:
    with _deterministic_torch(config.random_state):
        model = RedesignedSupervisedAutoencoder(
            config.input_dim,
            config.hidden_dim,
            config.latent_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for epoch in range(epochs):
            _train_epoch(
                model,
                optimizer,
                X_train,
                y_train,
                origins,
                target_mean=target_mean,
                target_scale=target_scale,
                config=config,
                epoch=epoch,
            )
    model.eval()
    return model


def _train_epoch(
    model: RedesignedSupervisedAutoencoder,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    origins: tuple[str, ...],
    *,
    target_mean: float,
    target_scale: float,
    config: RedesignedSupervisedAEConfig,
    epoch: int,
) -> None:
    features = torch.from_numpy(X_train)
    targets = torch.from_numpy(
        ((y_train - target_mean) / target_scale).astype(np.float32)
    )
    reconstruction_weights = torch.as_tensor(
        [
            1.0 if origin == "measured" else config.synthetic_reconstruction_weight
            for origin in origins
        ],
        dtype=torch.float32,
    )
    supervised_weights = torch.as_tensor(
        [
            1.0 if origin == "measured" else config.synthetic_supervised_weight
            for origin in origins
        ],
        dtype=torch.float32,
    )
    model.train()
    for batch_indices in _epoch_batches(
        len(X_train),
        batch_size=config.batch_size,
        random_state=config.random_state,
        epoch=epoch,
    ):
        batch_reconstruction_weight = reconstruction_weights[batch_indices]
        batch_supervised_weight = supervised_weights[batch_indices]
        if (
            float(batch_reconstruction_weight.sum()) == 0.0
            and float(batch_supervised_weight.sum()) == 0.0
        ):
            continue
        clean_features = features[batch_indices]
        noisy_features = _masked_features(
            clean_features,
            probability=config.masking_probability,
            random_state=config.random_state,
            epoch=epoch,
            batch_number=int(batch_indices[0]),
        )
        optimizer.zero_grad(set_to_none=True)
        _, reconstruction_raw, yield_scaled = model(noisy_features)
        reconstruction_per_sample = _reconstruction_loss_per_sample(
            reconstruction_raw,
            clean_features,
            config,
        )
        supervised_per_sample = nn.functional.mse_loss(
            yield_scaled,
            targets[batch_indices],
            reduction="none",
        )
        reconstruction_loss = fixed_denominator_weighted_mean(
            reconstruction_per_sample,
            batch_reconstruction_weight,
        )
        supervised_loss = fixed_denominator_weighted_mean(
            supervised_per_sample,
            batch_supervised_weight,
        )
        total_loss = (
            config.reconstruction_weight * reconstruction_loss
            + config.supervised_weight * supervised_loss
        )
        total_loss.backward()
        optimizer.step()


def _real_validation_predictive_yield_mse(
    model: RedesignedSupervisedAutoencoder,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    target_mean: float,
    target_scale: float,
) -> float:
    model.eval()
    features = torch.from_numpy(X_valid)
    with torch.no_grad():
        _, _, yield_scaled = model(features)
        return measured_validation_yield_mse(
            yield_scaled.numpy(),
            y_valid,
            target_mean=target_mean,
            target_scale=target_scale,
        )


def _reconstruction_loss_per_sample(
    reconstruction_raw: torch.Tensor,
    targets: torch.Tensor,
    config: RedesignedSupervisedAEConfig,
) -> torch.Tensor:
    if config.reconstruction_objective == "binary_cross_entropy":
        element_losses = nn.functional.binary_cross_entropy_with_logits(
            reconstruction_raw,
            targets,
            reduction="none",
        )
        element_weights = torch.ones_like(targets)
    elif config.reconstruction_objective == "count_aware_mse":
        predicted_counts = torch.nn.functional.softplus(reconstruction_raw)
        element_losses = (
            torch.log1p(predicted_counts) - torch.log1p(targets)
        ).pow(2)
        element_weights = torch.ones_like(targets)
    else:
        element_losses = (reconstruction_raw - targets).pow(2)
        if config.reconstruction_objective == "positive_bit_weighted_mse":
            element_weights = torch.where(
                targets > 0.0,
                torch.full_like(targets, config.positive_bit_weight),
                torch.ones_like(targets),
            )
        else:
            element_weights = torch.ones_like(targets)
    weighted_losses = element_losses * element_weights
    if not config.role_balanced_reconstruction:
        return weighted_losses.sum(dim=1) / element_weights.sum(dim=1).clamp_min(
            torch.finfo(element_weights.dtype).eps
        )
    role_losses = []
    for block in _effective_role_blocks(config):
        numerator = weighted_losses[:, block.start : block.stop].sum(dim=1)
        denominator = element_weights[:, block.start : block.stop].sum(dim=1)
        role_losses.append(
            numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
        )
    return torch.stack(role_losses, dim=1).mean(dim=1)


def _decoded_features(
    reconstruction_raw: torch.Tensor,
    config: RedesignedSupervisedAEConfig,
) -> torch.Tensor:
    if config.reconstruction_objective == "binary_cross_entropy":
        return torch.sigmoid(reconstruction_raw)
    if config.reconstruction_objective == "count_aware_mse":
        return torch.nn.functional.softplus(reconstruction_raw)
    return reconstruction_raw


def _reconstruction_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, float | int | None]:
    if targets.numel() == 0:
        return {
            "mse": None,
            "positive_bit_mse": None,
            "zero_bit_mse": None,
            "positive_bit_count": 0,
            "zero_bit_count": 0,
        }
    errors = (predictions - targets).pow(2)
    positive = targets > 0.0
    zero = ~positive
    positive_count = int(positive.sum().item())
    zero_count = int(zero.sum().item())
    return {
        "mse": float(errors.mean().item()),
        "positive_bit_mse": (
            float(errors[positive].mean().item()) if positive_count else None
        ),
        "zero_bit_mse": (
            float(errors[zero].mean().item()) if zero_count else None
        ),
        "positive_bit_count": positive_count,
        "zero_bit_count": zero_count,
    }


def _masked_features(
    features: torch.Tensor,
    *,
    probability: float,
    random_state: int,
    epoch: int,
    batch_number: int,
) -> torch.Tensor:
    if probability == 0.0:
        return features
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        random_state + epoch * 1_000_003 + batch_number * 97
    )
    keep = torch.rand(
        features.shape,
        generator=generator,
        dtype=features.dtype,
    ) >= probability
    return features * keep


def fixed_denominator_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Scale sample contributions without normalizing away their weights."""
    if values.ndim != 1 or weights.shape != values.shape or values.numel() == 0:
        raise ValueError("Weighted objectives require aligned nonempty vectors.")
    return (values * weights).sum() / values.numel()


def _effective_role_blocks(
    config: RedesignedSupervisedAEConfig,
) -> tuple[RoleBlock, ...]:
    return config.role_blocks or (RoleBlock("all_features", 0, config.input_dim),)


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


def _config_record(config: RedesignedSupervisedAEConfig) -> dict[str, Any]:
    record = asdict(config)
    record["role_blocks"] = [asdict(block) for block in config.role_blocks]
    return record


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
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative.")


def _finite_positive(value: float, name: str) -> None:
    _finite_nonnegative(value, name)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive.")
