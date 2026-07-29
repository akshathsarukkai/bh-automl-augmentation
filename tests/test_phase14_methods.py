"""Focused tests for shared Phase 14 method fitting."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from bh_augmentation.representations.phase14_methods import (
    PHASE14_METHOD_FAMILIES,
    Phase14MethodConfig,
    fit_phase14_method,
)
from bh_augmentation.representations.redesigned_supervised_autoencoder import (
    RedesignedSupervisedAEConfig,
    RoleBlock,
)


def _bundle() -> dict[str, object]:
    rng = np.random.default_rng(1414)
    X = rng.binomial(1, 0.2, size=(30, 28)).astype(np.float32)
    y = (30 + 8 * X[:, 2] - 4 * X[:, 9]).astype(np.float32)
    return {
        "X_measured_train": X,
        "y_measured_train": y,
        "X_evaluation": rng.binomial(
            1, 0.2, size=(6, 28)
        ).astype(np.float32),
        "measured_source_ids": tuple(f"row-{i:03d}" for i in range(30)),
        "measured_group_ids": tuple(f"group-{i:03d}" for i in range(30)),
    }


def _ae_method(
    *, data_protocol: str = "real_only"
) -> Phase14MethodConfig:
    ae = RedesignedSupervisedAEConfig(
        input_dim=28,
        hidden_dim=12,
        latent_dim=4,
        reconstruction_objective="binary_cross_entropy",
        role_blocks=(
            RoleBlock("a", 0, 8),
            RoleBlock("b", 8, 16),
            RoleBlock("c", 16, 28),
        ),
        synthetic_reconstruction_weight=0.25,
        synthetic_supervised_weight=0.5,
        batch_size=10,
        max_epochs=2,
        patience=1,
        random_state=14,
    )
    return Phase14MethodConfig(
        method_family="redesigned_supervised_ae",
        input_dim=28,
        random_state=14,
        data_protocol=data_protocol,
        hidden_dim=12,
        latent_dim=4,
        synthetic_supervised_weight=0.5,
        ae_config=ae,
    )


def test_public_fit_api_cannot_receive_evaluation_labels() -> None:
    parameters = inspect.signature(fit_phase14_method).parameters
    assert "y_evaluation" not in parameters
    assert "y_eval" not in parameters
    assert tuple(PHASE14_METHOD_FAMILIES) == (
        "redesigned_supervised_ae",
        "truncated_svd",
        "linear_autoencoder",
        "direct_xgboost",
        "matched_direct_mlp",
        "anonymous_transfer_without_ae",
        "typed_transfer_without_ae",
    )


def test_ae_fit_is_deterministic_and_parameter_matched() -> None:
    bundle = _bundle()
    first = fit_phase14_method(**bundle, config=_ae_method())
    second = fit_phase14_method(**bundle, config=_ae_method())

    assert first.state_hash == second.state_hash
    assert first.prediction_hash == second.prediction_hash
    assert first.training_parameter_count == 829
    assert first.deployed_predictor_parameter_count == 405
    assert first.metadata["evaluation_labels_received"] is False


@pytest.mark.parametrize(
    ("family", "latent_dim"),
    [
        ("truncated_svd", 4),
        ("linear_autoencoder", 4),
        ("matched_direct_mlp", 4),
    ],
)
def test_real_only_latent_methods_are_label_isolated(
    family: str, latent_dim: int
) -> None:
    config = Phase14MethodConfig(
        method_family=family,
        input_dim=28,
        random_state=14,
        hidden_dim=12 if family == "matched_direct_mlp" else None,
        latent_dim=latent_dim,
        max_epochs=2,
        patience=1,
        batch_size=10,
    )
    result = fit_phase14_method(**_bundle(), config=config)

    assert result.predictions.shape == (6,)
    assert np.isfinite(result.predictions).all()
    assert result.synthetic_fit_source_id_hash


def test_xgboost_transfer_weights_and_pool_provenance_are_exact() -> None:
    pytest.importorskip("xgboost")
    bundle = _bundle()
    rng = np.random.default_rng(8)
    X_added = rng.binomial(1, 0.2, size=(3, 28)).astype(np.float32)
    y_added = np.asarray([40, 50, 60], dtype=np.float32)
    config = Phase14MethodConfig(
        method_family="anonymous_transfer_without_ae",
        input_dim=28,
        random_state=14,
        data_protocol="anonymous",
        synthetic_supervised_weight=0.25,
        xgboost_params=(
            ("n_estimators", 4),
            ("max_depth", 2),
            ("n_jobs", 1),
            ("tree_method", "hist"),
        ),
    )
    result = fit_phase14_method(
        **bundle,
        config=config,
        X_synthetic=X_added,
        y_synthetic=y_added,
        synthetic_source_ids=("synthetic-a", "synthetic-b", "synthetic-c"),
        synthetic_group_ids=("sg-a", "sg-b", "sg-c"),
        transfer_pool_hash="a" * 64,
    )

    assert result.effective_supervised_weight_sum == 30.75
    assert result.metadata["transfer_pool_hash"] == "a" * 64
    assert result.metadata["method_metadata"]["synthetic_fit_row_count"] == 3


def test_transfer_protocol_or_ae_binding_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="Transfer family"):
        fit_phase14_method(
            **_bundle(),
            config=Phase14MethodConfig(
                method_family="typed_transfer_without_ae",
                input_dim=28,
                random_state=14,
                data_protocol="anonymous",
            ),
            transfer_pool_hash="a" * 64,
        )
    wrong = _ae_method()
    with pytest.raises(ValueError, match="AE configuration mismatch"):
        fit_phase14_method(
            **_bundle(),
            config=replace(wrong, synthetic_supervised_weight=1.0),
        )


def _pool_rows() -> dict[str, object]:
    rng = np.random.default_rng(909)
    return {
        "X_synthetic": rng.binomial(1, 0.2, size=(4, 28)).astype(np.float32),
        "y_synthetic": np.asarray([41, 52, 63, 47], dtype=np.float32),
        "synthetic_source_ids": tuple(f"synthetic-{i}" for i in range(4)),
        "synthetic_group_ids": tuple(f"sg-{i}" for i in range(4)),
        "transfer_pool_hash": "b" * 64,
    }


def _direct_mlp(*, data_protocol: str, weight: float) -> Phase14MethodConfig:
    return Phase14MethodConfig(
        method_family="matched_direct_mlp",
        input_dim=28,
        random_state=14,
        data_protocol=data_protocol,
        hidden_dim=12,
        latent_dim=4,
        max_epochs=2,
        patience=1,
        batch_size=10,
        synthetic_supervised_weight=weight,
    )


@pytest.mark.parametrize("data_protocol", ["anonymous", "typed"])
def test_matched_direct_mlp_is_protocol_selected_like_the_ae(
    data_protocol: str,
) -> None:
    # The direct-MLP control must be allowed exactly the same synthetic-pool
    # access as the AE family, otherwise architecture is confounded with data.
    augmented = fit_phase14_method(
        **_bundle(),
        config=_direct_mlp(data_protocol=data_protocol, weight=0.25),
        **_pool_rows(),
    )
    measured_only = fit_phase14_method(
        **_bundle(),
        config=_direct_mlp(data_protocol="real_only", weight=0.25),
    )

    assert augmented.metadata["data_protocol"] == data_protocol
    assert augmented.metadata["synthetic_fit_row_count"] == 4
    assert augmented.metadata["transfer_pool_hash"] == "b" * 64
    assert augmented.effective_supervised_weight_sum == pytest.approx(31.0)
    assert augmented.state_hash != measured_only.state_hash
    assert augmented.deployed_predictor_parameter_count == (
        measured_only.deployed_predictor_parameter_count
    )
    assert augmented.metadata["method_metadata"][
        "synthetic_supervised_weight"
    ] == 0.25


@pytest.mark.parametrize(
    "family",
    ["truncated_svd", "linear_autoencoder", "direct_xgboost"],
)
def test_unaugmented_controls_remain_real_only(family: str) -> None:
    config = Phase14MethodConfig(
        method_family=family,
        input_dim=28,
        random_state=14,
        data_protocol="anonymous",
        latent_dim=4,
        xgboost_params=(("n_estimators", 4),) if family == "direct_xgboost" else (),
    )
    with pytest.raises(ValueError, match="real_only"):
        fit_phase14_method(**_bundle(), config=config, **_pool_rows())


def test_linear_autoencoder_receives_the_configured_weight_decay() -> None:
    config = Phase14MethodConfig(
        method_family="linear_autoencoder",
        input_dim=28,
        random_state=14,
        latent_dim=4,
        max_epochs=2,
        patience=1,
        batch_size=10,
        weight_decay=0.01,
    )
    result = fit_phase14_method(**_bundle(), config=config)

    assert result.metadata["method_metadata"]["config"]["weight_decay"] == 0.01


@pytest.mark.parametrize(
    ("family", "data_protocol"),
    [
        ("anonymous_transfer_without_ae", "anonymous"),
        ("typed_transfer_without_ae", "typed"),
        ("matched_direct_mlp", "typed"),
        ("redesigned_supervised_ae", "anonymous"),
    ],
)
def test_zero_row_transfer_pool_is_a_loud_failure(
    family: str,
    data_protocol: str,
) -> None:
    if family == "redesigned_supervised_ae":
        config = _ae_method(data_protocol=data_protocol)
    elif family == "matched_direct_mlp":
        config = _direct_mlp(data_protocol=data_protocol, weight=0.25)
    else:
        config = Phase14MethodConfig(
            method_family=family,
            input_dim=28,
            random_state=14,
            data_protocol=data_protocol,
            synthetic_supervised_weight=0.25,
            xgboost_params=(("n_estimators", 4),),
        )
    with pytest.raises(ValueError, match="Degenerate transfer pool") as excinfo:
        fit_phase14_method(
            **_bundle(),
            config=config,
            X_synthetic=np.empty((0, 28), dtype=np.float32),
            y_synthetic=np.empty(0, dtype=np.float32),
            synthetic_source_ids=(),
            synthetic_group_ids=(),
            transfer_pool_hash="c" * 64,
            evaluation_unit="seed=0|fraction=0.2",
            phase="placement",
        )

    message = str(excinfo.value)
    assert "seed=0|fraction=0.2" in message
    assert "placement" in message


def test_nonfinite_and_duplicate_source_inputs_fail() -> None:
    bundle = _bundle()
    bad_X = np.asarray(bundle["X_measured_train"]).copy()
    bad_X[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid measured"):
        fit_phase14_method(
            **{**bundle, "X_measured_train": bad_X},
            config=Phase14MethodConfig(
                method_family="direct_xgboost",
                input_dim=28,
                random_state=14,
                xgboost_params=(("n_estimators", 4),),
            ),
        )
    duplicate_ids = (*bundle["measured_source_ids"][:-1], "row-000")
    with pytest.raises(ValueError, match="unique"):
        fit_phase14_method(
            **{**bundle, "measured_source_ids": duplicate_ids},
            config=Phase14MethodConfig(
                method_family="direct_xgboost",
                input_dim=28,
                random_state=14,
                xgboost_params=(("n_estimators", 4),),
            ),
        )
