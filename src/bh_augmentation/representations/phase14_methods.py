"""Shared label-isolated fitting adapters for the Phase 14 benchmark."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bh_augmentation.models.baselines import get_model
from bh_augmentation.representations.low_complexity import (
    LowComplexityConfig,
    fit_low_complexity_representation,
)
from bh_augmentation.representations.matched_direct_mlp import (
    MatchedDirectMLPConfig,
    fit_predict_matched_direct_mlp,
)
from bh_augmentation.representations.redesigned_supervised_autoencoder import (
    RedesignedSupervisedAEConfig,
    fit_redesigned_supervised_autoencoder,
)
from bh_augmentation.utils.corrected_runs import stable_hash

PHASE14_METHOD_SCHEMA_VERSION = "bh-phase14-method-fit-v1"
PHASE14_METHOD_FAMILIES = (
    "redesigned_supervised_ae",
    "truncated_svd",
    "linear_autoencoder",
    "direct_xgboost",
    "matched_direct_mlp",
    "anonymous_transfer_without_ae",
    "typed_transfer_without_ae",
)
_TRANSFER_FAMILIES = {
    "anonymous_transfer_without_ae",
    "typed_transfer_without_ae",
}
# Families that may be selected over {real_only, anonymous, typed} exactly like
# the autoencoder family, so that architecture is never confounded with access
# to a teacher-labelled synthetic pool.
PHASE14_PROTOCOL_SELECTED_FAMILIES = (
    "matched_direct_mlp",
    "redesigned_supervised_ae",
)
_REAL_ONLY_FAMILIES = {
    "truncated_svd",
    "linear_autoencoder",
    "direct_xgboost",
}


@dataclass(frozen=True, slots=True)
class Phase14MethodConfig:
    """One fully resolved fitting policy."""

    method_family: str
    input_dim: int
    random_state: int
    data_protocol: str = "real_only"
    latent_dim: int | None = None
    hidden_dim: int | None = None
    ridge_alpha: float = 1.0
    max_epochs: int = 20
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 128
    internal_valid_fraction: float = 0.2
    synthetic_supervised_weight: float = 1.0
    xgboost_params: tuple[tuple[str, Any], ...] = ()
    ae_config: RedesignedSupervisedAEConfig | None = None


@dataclass(frozen=True, slots=True)
class Phase14MethodFitResult:
    """Predictions and scientific provenance for one label-isolated fit."""

    method_family: str
    predictions: np.ndarray
    prediction_hash: str
    state_hash: str
    selected_epoch: int | None
    training_parameter_count: int | None
    deployed_predictor_parameter_count: int | None
    measured_fit_source_id_hash: str
    synthetic_fit_source_id_hash: str
    fit_group_assignment_hash: str
    effective_supervised_weight_sum: float
    reconstruction_metrics: Mapping[str, Any]
    reconstruction_metrics_hash: str
    metadata: Mapping[str, Any]


def fit_phase14_method(
    X_measured_train: np.ndarray,
    y_measured_train: np.ndarray,
    X_evaluation: np.ndarray,
    *,
    measured_source_ids: Sequence[str],
    measured_group_ids: Sequence[str],
    config: Phase14MethodConfig,
    X_synthetic: np.ndarray | None = None,
    y_synthetic: np.ndarray | None = None,
    synthetic_source_ids: Sequence[str] = (),
    synthetic_group_ids: Sequence[str] = (),
    transfer_pool_hash: str | None = None,
    evaluation_unit: str | None = None,
    phase: str | None = None,
) -> Phase14MethodFitResult:
    """Fit one method without receiving evaluation labels."""
    (
        X_measured,
        y_measured,
        X_eval,
        measured_ids,
        measured_groups,
        X_added,
        y_added,
        added_ids,
        added_groups,
    ) = _validate_inputs(
        X_measured_train,
        y_measured_train,
        X_evaluation,
        measured_source_ids,
        measured_group_ids,
        config,
        X_synthetic,
        y_synthetic,
        synthetic_source_ids,
        synthetic_group_ids,
        transfer_pool_hash,
        evaluation_unit,
        phase,
    )
    family = config.method_family
    measured_hash = stable_hash(sorted(measured_ids))
    synthetic_hash = stable_hash(sorted(added_ids))
    assignments = [
        {
            "source_row_id": source_id,
            "canonical_group": group_id,
            "sample_origin": "measured",
        }
        for source_id, group_id in zip(
            measured_ids, measured_groups, strict=True
        )
    ] + [
        {
            "source_row_id": source_id,
            "canonical_group": group_id,
            "sample_origin": "synthetic",
        }
        for source_id, group_id in zip(added_ids, added_groups, strict=True)
    ]
    assignment_hash = stable_hash(
        sorted(assignments, key=lambda row: row["source_row_id"])
    )
    selected_epoch: int | None = None
    training_count: int | None = None
    deployed_count: int | None = None
    reconstruction: Mapping[str, Any] = {}
    method_metadata: dict[str, Any]

    if family in {"truncated_svd", "linear_autoencoder"}:
        low_method = (
            "truncated_svd"
            if family == "truncated_svd"
            else "linear_autoencoder"
        )
        low_config = LowComplexityConfig(
            method=low_method,
            latent_width=int(config.latent_dim),
            ridge_alpha=float(config.ridge_alpha),
            random_state=int(config.random_state),
            learning_rate=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            max_epochs=int(config.max_epochs),
            patience=int(config.patience),
            internal_valid_fraction=float(config.internal_valid_fraction),
            batch_size=int(config.batch_size),
        )
        low = fit_low_complexity_representation(
            X_measured,
            y_measured,
            X_eval,
            low_config,
            train_source_ids=measured_ids,
            train_group_ids=measured_groups,
        )
        predictions = low.predictions
        scientific_state_hash = low.state_hash
        selected_epoch = low.selected_epoch
        training_count = low.fitted_state_scalar_count
        deployed_count = low.deployed_predictive_parameter_count
        effective_weight_sum = float(len(y_measured))
        method_metadata = dict(low.metadata)

    elif family == "matched_direct_mlp":
        direct_config = MatchedDirectMLPConfig(
            input_dim=config.input_dim,
            hidden_dim=int(config.hidden_dim),
            latent_dim=int(config.latent_dim),
            learning_rate=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            batch_size=int(config.batch_size),
            max_epochs=int(config.max_epochs),
            patience=int(config.patience),
            internal_valid_fraction=float(config.internal_valid_fraction),
            random_state=int(config.random_state),
            synthetic_supervised_weight=float(config.synthetic_supervised_weight),
        )
        direct = fit_predict_matched_direct_mlp(
            X_measured,
            y_measured,
            X_eval,
            measured_source_ids=measured_ids,
            measured_group_ids=measured_groups,
            config=direct_config,
            X_synthetic=X_added,
            y_synthetic=y_added,
            synthetic_source_ids=added_ids,
            synthetic_group_ids=added_groups,
        )
        predictions = direct.predictions
        scientific_state_hash = direct.state_hash
        selected_epoch = direct.selected_epoch
        training_count = direct.deployed_predictor_parameter_count
        deployed_count = direct.deployed_predictor_parameter_count
        effective_weight_sum = float(
            len(y_measured)
            + len(y_added) * config.synthetic_supervised_weight
        )
        method_metadata = dict(direct.metadata)
        if direct.measured_train_source_id_hash != stable_hash(
            sorted(measured_ids)
        ):
            raise RuntimeError(
                "Matched direct-MLP measured-only scaler provenance mismatch."
            )

    elif family == "redesigned_supervised_ae":
        X_fit = np.vstack([X_measured, X_added]).astype(np.float32)
        y_fit = np.concatenate([y_measured, y_added]).astype(np.float32)
        fit_ids = (*measured_ids, *added_ids)
        fit_groups = (*measured_groups, *added_groups)
        origins = ("measured",) * len(measured_ids) + (
            "synthetic",
        ) * len(added_ids)
        ae = fit_redesigned_supervised_autoencoder(
            X_fit,
            y_fit,
            X_eval,
            train_source_ids=fit_ids,
            sample_origins=origins,
            train_group_ids=fit_groups,
            config=config.ae_config,
        )
        predictions = ae.yield_predictions
        scientific_state_hash = ae.state_hash
        selected_epoch = ae.selected_epoch
        training_count = ae.parameter_count
        deployed_count = _ae_deployed_predictor_count(config.ae_config)
        effective_weight_sum = float(
            len(y_measured)
            + len(y_added) * config.synthetic_supervised_weight
        )
        reconstruction = {
            "training": ae.training_reconstruction_metrics,
            "evaluation": ae.evaluation_reconstruction_metrics,
        }
        method_metadata = dict(ae.metadata)
        if (
            ae.measured_scaling_source_id_hash
            != stable_hash(list(measured_ids))
        ):
            raise RuntimeError("AE measured-only scaler provenance mismatch.")

    else:
        X_fit = np.vstack([X_measured, X_added]).astype(np.float32)
        y_fit = np.concatenate([y_measured, y_added]).astype(np.float32)
        weights = np.concatenate(
            [
                np.ones(len(y_measured), dtype=np.float32),
                np.full(
                    len(y_added),
                    float(config.synthetic_supervised_weight),
                    dtype=np.float32,
                ),
            ]
        )
        estimator = get_model(
            "xgboost",
            seed=int(config.random_state),
            **dict(config.xgboost_params),
        )
        estimator.fit(X_fit, y_fit, sample_weight=weights)
        predictions = np.asarray(estimator.predict(X_eval), dtype=np.float64)
        scientific_state_hash = _xgboost_state_hash(estimator, config)
        effective_weight_sum = float(weights.sum(dtype=np.float64))
        method_metadata = {
            "estimator": type(estimator).__name__,
            "xgboost_params": dict(config.xgboost_params),
            "fit_row_count": len(X_fit),
            "measured_fit_row_count": len(X_measured),
            "synthetic_fit_row_count": len(X_added),
            "sample_weight_semantics": (
                "measured=1,synthetic=synthetic_supervised_weight"
            ),
        }

    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if predictions.shape != (len(X_eval),) or not np.isfinite(
        predictions
    ).all():
        raise ValueError("Phase 14 method produced invalid predictions.")
    prediction_hash = _array_hash(predictions)
    config_record = _config_record(config)
    state_hash = stable_hash(
        {
            "schema_version": PHASE14_METHOD_SCHEMA_VERSION,
            "method_family": family,
            "resolved_config": config_record,
            "scientific_state_hash": scientific_state_hash,
            "measured_fit_source_id_hash": measured_hash,
            "synthetic_fit_source_id_hash": synthetic_hash,
            "fit_group_assignment_hash": assignment_hash,
            "transfer_pool_hash": transfer_pool_hash,
        }
    )
    metadata = {
        "schema_version": PHASE14_METHOD_SCHEMA_VERSION,
        "method_family": family,
        "resolved_config": config_record,
        "config_hash": stable_hash(config_record),
        "data_protocol": config.data_protocol,
        "measured_fit_row_count": len(X_measured),
        "synthetic_fit_row_count": len(X_added),
        "measured_fit_source_id_hash": measured_hash,
        "synthetic_fit_source_id_hash": synthetic_hash,
        "fit_group_assignment_hash": assignment_hash,
        "transfer_pool_hash": transfer_pool_hash,
        "effective_supervised_weight_sum": effective_weight_sum,
        "reconstruction_metrics_hash": stable_hash(reconstruction),
        "evaluation_labels_received": False,
        "method_metadata": method_metadata,
    }
    return Phase14MethodFitResult(
        method_family=family,
        predictions=predictions,
        prediction_hash=prediction_hash,
        state_hash=state_hash,
        selected_epoch=selected_epoch,
        training_parameter_count=training_count,
        deployed_predictor_parameter_count=deployed_count,
        measured_fit_source_id_hash=measured_hash,
        synthetic_fit_source_id_hash=synthetic_hash,
        fit_group_assignment_hash=assignment_hash,
        effective_supervised_weight_sum=effective_weight_sum,
        reconstruction_metrics=reconstruction,
        reconstruction_metrics_hash=stable_hash(reconstruction),
        metadata=metadata,
    )


def _validate_inputs(
    X_measured_train: np.ndarray,
    y_measured_train: np.ndarray,
    X_evaluation: np.ndarray,
    measured_source_ids: Sequence[str],
    measured_group_ids: Sequence[str],
    config: Phase14MethodConfig,
    X_synthetic: np.ndarray | None,
    y_synthetic: np.ndarray | None,
    synthetic_source_ids: Sequence[str],
    synthetic_group_ids: Sequence[str],
    transfer_pool_hash: str | None,
    evaluation_unit: str | None,
    phase: str | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(config, Phase14MethodConfig):
        raise TypeError("config must be Phase14MethodConfig.")
    if config.method_family not in PHASE14_METHOD_FAMILIES:
        raise ValueError("Unsupported Phase 14 method family.")
    if config.data_protocol not in {"real_only", "anonymous", "typed"}:
        raise ValueError("Unsupported Phase 14 data protocol.")
    if (
        not isinstance(config.input_dim, int)
        or isinstance(config.input_dim, bool)
        or config.input_dim <= 0
        or not isinstance(config.random_state, int)
        or isinstance(config.random_state, bool)
    ):
        raise ValueError("Invalid Phase 14 dimensions or random state.")
    X_measured = np.asarray(X_measured_train, dtype=np.float32)
    y_measured = np.asarray(y_measured_train, dtype=np.float32).reshape(-1)
    X_eval = np.asarray(X_evaluation, dtype=np.float32)
    measured_ids = _validated_ids(
        measured_source_ids, len(X_measured), "measured_source_ids"
    )
    measured_groups = _validated_groups(
        measured_group_ids, len(X_measured), "measured_group_ids"
    )
    if (
        X_measured.ndim != 2
        or X_eval.ndim != 2
        or len(X_measured) != len(y_measured)
        or len(X_measured) < 4
        or len(X_eval) == 0
        or X_measured.shape[1] != config.input_dim
        or X_eval.shape[1] != config.input_dim
        or not np.isfinite(X_measured).all()
        or not np.isfinite(y_measured).all()
        or not np.isfinite(X_eval).all()
    ):
        raise ValueError("Invalid measured/evaluation arrays.")
    if X_synthetic is None:
        X_added = np.empty((0, config.input_dim), dtype=np.float32)
    else:
        X_added = np.asarray(X_synthetic, dtype=np.float32)
    if y_synthetic is None:
        y_added = np.empty(0, dtype=np.float32)
    else:
        y_added = np.asarray(y_synthetic, dtype=np.float32).reshape(-1)
    added_ids = _validated_ids(
        synthetic_source_ids, len(X_added), "synthetic_source_ids"
    )
    added_groups = _validated_groups(
        synthetic_group_ids, len(X_added), "synthetic_group_ids"
    )
    if (
        X_added.ndim != 2
        or X_added.shape[1] != config.input_dim
        or len(X_added) != len(y_added)
        or not np.isfinite(X_added).all()
        or not np.isfinite(y_added).all()
        or set(measured_ids) & set(added_ids)
    ):
        raise ValueError("Invalid or overlapping synthetic arrays.")
    expects_pool = (
        config.data_protocol != "real_only"
        or config.method_family in _TRANSFER_FAMILIES
    )
    if expects_pool != (transfer_pool_hash is not None):
        raise ValueError("Transfer-pool declaration does not match method.")
    if transfer_pool_hash is not None and (
        len(transfer_pool_hash) != 64
        or any(character not in "0123456789abcdef" for character in transfer_pool_hash)
    ):
        raise ValueError("transfer_pool_hash must be a SHA-256 digest.")
    expected_protocol = {
        "anonymous_transfer_without_ae": "anonymous",
        "typed_transfer_without_ae": "typed",
    }.get(config.method_family)
    if expected_protocol is not None and config.data_protocol != expected_protocol:
        raise ValueError("Transfer family and data protocol disagree.")
    if (
        config.method_family in _REAL_ONLY_FAMILIES
        and config.data_protocol != "real_only"
    ):
        raise ValueError("Unaugmented control must use real_only data.")
    if expects_pool and len(X_added) == 0:
        raise ValueError(
            "Degenerate transfer pool: method family "
            f"{config.method_family!r} declared data_protocol "
            f"{config.data_protocol!r} and transfer pool "
            f"{transfer_pool_hash} but received zero synthetic rows for "
            f"evaluation_unit={evaluation_unit!r} phase={phase!r}. An "
            "augmented family that silently degrades to measured-only rows "
            "is not a valid transfer control."
        )
    if config.method_family == "redesigned_supervised_ae":
        if (
            not isinstance(
                config.ae_config, RedesignedSupervisedAEConfig
            )
            or config.ae_config.input_dim != config.input_dim
            or config.ae_config.hidden_dim != config.hidden_dim
            or config.ae_config.latent_dim != config.latent_dim
            or config.ae_config.random_state != config.random_state
            or config.ae_config.synthetic_supervised_weight
            != config.synthetic_supervised_weight
        ):
            raise ValueError("Resolved AE configuration mismatch.")
    elif config.ae_config is not None:
        raise ValueError("Non-AE method cannot carry ae_config.")
    xgboost_family = config.method_family in {
        "direct_xgboost",
        *_TRANSFER_FAMILIES,
    }
    if bool(config.xgboost_params) != xgboost_family:
        raise ValueError("XGBoost parameter binding does not match method family.")
    if len({key for key, _ in config.xgboost_params}) != len(
        config.xgboost_params
    ):
        raise ValueError("XGBoost parameter keys must be unique.")
    if config.method_family in {
        "truncated_svd",
        "linear_autoencoder",
        "matched_direct_mlp",
    } and (
        not isinstance(config.latent_dim, int)
        or isinstance(config.latent_dim, bool)
        or config.latent_dim <= 0
    ):
        raise ValueError("Latent methods require a positive latent_dim.")
    if config.method_family in {
        "redesigned_supervised_ae",
        "matched_direct_mlp",
    } and (
        not isinstance(config.hidden_dim, int)
        or isinstance(config.hidden_dim, bool)
        or config.hidden_dim <= 0
    ):
        raise ValueError("Neural methods require a positive hidden_dim.")
    weight = float(config.synthetic_supervised_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("Synthetic supervised weight must be nonnegative.")
    if config.data_protocol == "real_only" and len(X_added):
        raise ValueError("real_only method received synthetic rows.")
    return (
        np.ascontiguousarray(X_measured),
        np.ascontiguousarray(y_measured),
        np.ascontiguousarray(X_eval),
        measured_ids,
        measured_groups,
        np.ascontiguousarray(X_added),
        np.ascontiguousarray(y_added),
        added_ids,
        added_groups,
    )


def _validated_ids(
    values: Sequence[str], n_rows: int, name: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or len(values) != n_rows:
        raise ValueError(f"{name} must contain one ID per row.")
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized) or len(set(normalized)) != n_rows:
        raise ValueError(f"{name} values must be nonempty and unique.")
    return normalized


def _validated_groups(
    values: Sequence[str], n_rows: int, name: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or len(values) != n_rows:
        raise ValueError(f"{name} must contain one group per row.")
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} values must be nonempty.")
    return normalized


def _ae_deployed_predictor_count(
    config: RedesignedSupervisedAEConfig,
) -> int:
    return (
        config.input_dim * config.hidden_dim
        + config.hidden_dim
        + config.hidden_dim * config.latent_dim
        + config.latent_dim
        + config.latent_dim
        + 1
    )


def _xgboost_state_hash(
    estimator: Any, config: Phase14MethodConfig
) -> str:
    raw = bytes(estimator.get_booster().save_raw())
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _config_record(config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(raw)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _config_record(config: Phase14MethodConfig) -> dict[str, Any]:
    record = asdict(config)
    record["xgboost_params"] = [
        [str(key), value] for key, value in config.xgboost_params
    ]
    return record
