"""Focused tests for the redesigned supervised-autoencoder core."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bh_augmentation.representations.redesigned_supervised_autoencoder import (
    SYNTHETIC_WEIGHT_CHOICES,
    RedesignedSupervisedAEConfig,
    RedesignedSupervisedAutoencoder,
    RoleBlock,
    _decoded_features,
    _reconstruction_loss_per_sample,
    _reconstruction_metrics,
    build_measured_internal_split,
    fit_redesigned_supervised_autoencoder,
    fixed_denominator_weighted_mean,
    measured_validation_yield_mse,
    supervised_ae_parameter_count,
)


def _fixture(
    *,
    synthetic_rows: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    rng = np.random.default_rng(803)
    measured_rows = 16
    n_rows = measured_rows + synthetic_rows
    X = rng.binomial(1, 0.2, size=(n_rows, 28)).astype(np.float32)
    y = (
        10.0 * X[:, 2]
        - 4.0 * X[:, 11]
        + rng.normal(40.0, 1.0, size=n_rows)
    ).astype(np.float32)
    X_eval = rng.binomial(1, 0.2, size=(5, 28)).astype(np.float32)
    source_ids = tuple(f"source-{index:03d}" for index in range(n_rows))
    origins = ("measured",) * measured_rows + ("synthetic",) * synthetic_rows
    return X, y, X_eval, source_ids, origins


def _config(**changes: object) -> RedesignedSupervisedAEConfig:
    values: dict[str, object] = {
        "input_dim": 28,
        "hidden_dim": 12,
        "latent_dim": 4,
        "role_blocks": (
            RoleBlock("role-a", 0, 4),
            RoleBlock("role-b", 4, 12),
            RoleBlock("role-c", 12, 28),
        ),
        "batch_size": 6,
        "max_epochs": 6,
        "patience": 2,
        "random_state": 19,
    }
    values.update(changes)
    return RedesignedSupervisedAEConfig(**values)


def test_exact_parameter_count_matches_actual_compact_model() -> None:
    model = RedesignedSupervisedAutoencoder(28, 12, 4)
    expected = 2 * 28 * 12 + 2 * 12 * 4 + 2 * 12 + 28 + 2 * 4 + 1

    assert supervised_ae_parameter_count(28, 12, 4) == expected
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


@pytest.mark.parametrize(
    ("input_dim", "hidden_dim", "latent_dim"),
    [(14336, 128, 16), (14336, 256, 32), (31, 9, 3)],
)
def test_parameter_formula_supports_primary_and_other_input_widths(
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> None:
    model = RedesignedSupervisedAutoencoder(input_dim, hidden_dim, latent_dim)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        supervised_ae_parameter_count(input_dim, hidden_dim, latent_dim)
    )


def test_fit_is_deterministic_and_returns_label_safe_outputs() -> None:
    X, y, X_eval, source_ids, origins = _fixture()
    config = _config(masking_probability=0.2)

    first = fit_redesigned_supervised_autoencoder(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        sample_origins=origins,
        config=config,
    )
    second = fit_redesigned_supervised_autoencoder(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        sample_origins=origins,
        config=config,
    )

    assert first.state_hash == second.state_hash
    assert first.prediction_hash == second.prediction_hash
    np.testing.assert_array_equal(first.train_latent, second.train_latent)
    np.testing.assert_array_equal(first.evaluation_latent, second.evaluation_latent)
    assert first.train_latent.shape == (len(X), config.latent_dim)
    assert first.evaluation_latent.shape == (len(X_eval), config.latent_dim)
    assert first.yield_predictions.shape == (len(X_eval),)
    assert first.metadata["evaluation_labels_received"] is False


def test_target_scaling_uses_measured_rows_only_and_records_provenance() -> None:
    X, y, X_eval, source_ids, origins = _fixture()
    y[-4:] = 1_000_000.0

    result = fit_redesigned_supervised_autoencoder(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        sample_origins=origins,
        config=_config(max_epochs=2),
    )

    np.testing.assert_allclose(result.target_mean, np.mean(y[:16], dtype=np.float64))
    np.testing.assert_allclose(result.target_scale, np.std(y[:16], dtype=np.float64))
    assert result.metadata["measured_row_count"] == 16
    assert result.metadata["synthetic_row_count"] == 4
    assert result.measured_scaling_source_id_hash
    assert result.synthetic_source_id_hash


def test_early_stopping_validation_is_real_only_and_group_disjoint() -> None:
    X, y, X_eval, source_ids, origins = _fixture()
    groups = tuple(
        f"group-{index // 2:03d}" if origin == "measured" else f"synthetic-{index}"
        for index, origin in enumerate(origins)
    )

    result = fit_redesigned_supervised_autoencoder(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        sample_origins=origins,
        train_group_ids=groups,
        config=_config(),
    )

    assert result.internal_fit_source_id_hash
    assert result.internal_validation_source_id_hash
    assert result.metadata["internal_fit_count"] + result.metadata[
        "internal_validation_count"
    ] == 16
    assert result.metadata["early_stopping_rows"] == "train_derived_measured_only"
    assert (
        result.metadata["early_stopping_monitor"]
        == "real_internal_validation_yield_mse_only"
    )
    assert result.metadata["early_stopping_tie_break"] == "earliest_epoch"


def test_public_measured_split_accounts_for_ids_and_is_group_disjoint() -> None:
    _, _, _, source_ids, origins = _fixture()
    groups = tuple(
        f"group-{index // 2:03d}" if origin == "measured" else f"synthetic-{index}"
        for index, origin in enumerate(origins)
    )

    first = build_measured_internal_split(
        source_ids,
        origins,
        train_group_ids=groups,
        valid_fraction=0.25,
        random_state=31,
    )
    second = build_measured_internal_split(
        source_ids,
        origins,
        train_group_ids=groups,
        valid_fraction=0.25,
        random_state=31,
    )

    assert first == second
    accounted = (
        set(first.measured_fit_source_ids)
        | set(first.measured_validation_source_ids)
        | set(first.synthetic_source_ids)
    )
    assert accounted == set(source_ids)
    assert not (
        set(first.measured_fit_source_ids)
        & set(first.measured_validation_source_ids)
    )
    group_by_source = dict(zip(source_ids, groups, strict=True))
    assert {
        group_by_source[source_id] for source_id in first.measured_fit_source_ids
    }.isdisjoint(
        {
            group_by_source[source_id]
            for source_id in first.measured_validation_source_ids
        }
    )
    assert set(first.synthetic_source_ids) == set(source_ids[-4:])
    assert len(first.split_hash) == 64


def test_validation_monitor_is_analytic_yield_mse_not_reconstruction_loss() -> None:
    predicted_scaled = np.asarray([0.0, 1.0, -1.0])
    measured_yield = np.asarray([10.0, 12.0, 8.0])

    monitor = measured_validation_yield_mse(
        predicted_scaled,
        measured_yield,
        target_mean=10.0,
        target_scale=2.0,
    )

    assert monitor == 0.0
    # There is deliberately no reconstruction prediction, objective, or
    # reconstruction weight argument in the monitor contract.


@pytest.mark.parametrize("weight", SYNTHETIC_WEIGHT_CHOICES)
def test_all_declared_synthetic_weights_execute(weight: float) -> None:
    X, y, X_eval, source_ids, origins = _fixture(synthetic_rows=2)
    result = fit_redesigned_supervised_autoencoder(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        sample_origins=origins,
        config=_config(
            synthetic_reconstruction_weight=weight,
            synthetic_supervised_weight=weight,
            max_epochs=1,
        ),
    )

    assert result.metadata["synthetic_reconstruction_weight"] == weight
    assert result.metadata["synthetic_supervised_weight"] == weight


def test_all_synthetic_batch_weights_do_not_normalize_away() -> None:
    losses = torch.tensor([2.0, 4.0, 6.0])
    low = fixed_denominator_weighted_mean(
        losses, torch.full_like(losses, 0.1)
    )
    full = fixed_denominator_weighted_mean(
        losses, torch.ones_like(losses)
    )

    assert float(low) == pytest.approx(0.1 * float(full))


@pytest.mark.parametrize(
    "objective",
    (
        "mse",
        "positive_bit_weighted_mse",
        "binary_cross_entropy",
        "count_aware_mse",
    ),
)
def test_every_reconstruction_objective_executes(objective: str) -> None:
    X, y, X_eval, source_ids, origins = _fixture(synthetic_rows=0)
    result = fit_redesigned_supervised_autoencoder(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        sample_origins=origins,
        config=_config(reconstruction_objective=objective, max_epochs=1),
    )

    assert np.isfinite(result.yield_predictions).all()
    assert np.isfinite(result.evaluation_reconstruction_metrics["mse"])


def test_positive_bit_weight_changes_only_positive_error_contribution() -> None:
    raw = torch.zeros((1, 4))
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    unweighted = _reconstruction_loss_per_sample(
        raw,
        target,
        RedesignedSupervisedAEConfig(
            input_dim=4,
            hidden_dim=2,
            latent_dim=1,
            reconstruction_objective="mse",
            role_balanced_reconstruction=False,
        ),
    )
    weighted = _reconstruction_loss_per_sample(
        raw,
        target,
        RedesignedSupervisedAEConfig(
            input_dim=4,
            hidden_dim=2,
            latent_dim=1,
            reconstruction_objective="positive_bit_weighted_mse",
            role_balanced_reconstruction=False,
            positive_bit_weight=5.0,
        ),
    )

    assert float(unweighted) == 0.5
    assert float(weighted) > float(unweighted)


def test_role_balancing_gives_equal_weight_to_unequal_width_blocks() -> None:
    raw = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = torch.zeros_like(raw)
    balanced = _reconstruction_loss_per_sample(
        raw,
        target,
        RedesignedSupervisedAEConfig(
            input_dim=4,
            hidden_dim=2,
            latent_dim=1,
            reconstruction_objective="mse",
            role_blocks=(
                RoleBlock("narrow", 0, 1),
                RoleBlock("wide", 1, 4),
            ),
            role_balanced_reconstruction=True,
        ),
    )
    flat = _reconstruction_loss_per_sample(
        raw,
        target,
        RedesignedSupervisedAEConfig(
            input_dim=4,
            hidden_dim=2,
            latent_dim=1,
            reconstruction_objective="mse",
            role_balanced_reconstruction=False,
        ),
    )

    assert float(balanced) == 0.5
    assert float(flat) == 0.25


def test_positive_and_zero_reconstruction_metrics_are_reported_separately() -> None:
    targets = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    predictions = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    metrics = _reconstruction_metrics(targets, predictions)

    assert metrics["positive_bit_count"] == 2
    assert metrics["zero_bit_count"] == 2
    assert metrics["positive_bit_mse"] == 1.0
    assert metrics["zero_bit_mse"] == 0.5


def test_bce_decoding_is_probability_bounded() -> None:
    raw = torch.tensor([[-100.0, 0.0, 100.0]])
    config = RedesignedSupervisedAEConfig(
        input_dim=3,
        hidden_dim=2,
        latent_dim=1,
        reconstruction_objective="binary_cross_entropy",
    )

    decoded = _decoded_features(raw, config)

    assert torch.all(decoded >= 0.0)
    assert torch.all(decoded <= 1.0)


def test_binary_and_count_contracts_reject_incompatible_features() -> None:
    X, y, X_eval, source_ids, origins = _fixture(synthetic_rows=0)
    X[0, 0] = 2.0
    with pytest.raises(ValueError, match="binary features"):
        fit_redesigned_supervised_autoencoder(
            X,
            y,
            X_eval,
            train_source_ids=source_ids,
            sample_origins=origins,
            config=_config(
                reconstruction_objective="binary_cross_entropy",
                max_epochs=1,
            ),
        )
    X[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        fit_redesigned_supervised_autoencoder(
            X,
            y,
            X_eval,
            train_source_ids=source_ids,
            sample_origins=origins,
            config=_config(
                reconstruction_objective="count_aware_mse",
                max_epochs=1,
            ),
        )


def test_invalid_role_partition_or_synthetic_weight_fails_hard() -> None:
    X, y, X_eval, source_ids, origins = _fixture()
    with pytest.raises(ValueError, match="exactly cover"):
        fit_redesigned_supervised_autoencoder(
            X,
            y,
            X_eval,
            train_source_ids=source_ids,
            sample_origins=origins,
            config=_config(role_blocks=(RoleBlock("partial", 0, 20),)),
        )
    with pytest.raises(ValueError, match="synthetic_supervised_weight"):
        fit_redesigned_supervised_autoencoder(
            X,
            y,
            X_eval,
            train_source_ids=source_ids,
            sample_origins=origins,
            config=_config(synthetic_supervised_weight=0.3),
        )
