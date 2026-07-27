"""Tests for deterministic low-complexity representation baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from bh_augmentation.representations.low_complexity import (
    LOW_COMPLEXITY_METHODS,
    LowComplexityConfig,
    _make_internal_group_split,
    _stable_f_regression_top_k,
    fit_low_complexity_representation,
    fit_predict_low_complexity,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(731)
    X = rng.binomial(1, 0.25, size=(24, 40)).astype(np.float32)
    y = (
        8.0 * X[:, 1]
        - 5.0 * X[:, 7]
        + 3.0 * X[:, 13]
        + rng.normal(0.0, 0.2, size=len(X))
    ).astype(np.float32)
    X_eval = rng.binomial(1, 0.25, size=(7, 40)).astype(np.float32)
    return X, y, X_eval


@pytest.mark.parametrize("method", LOW_COMPLEXITY_METHODS)
def test_all_methods_return_finite_predictions_and_complete_hashes(method: str) -> None:
    X, y, X_eval = _fixture()
    result = fit_low_complexity_representation(
        X,
        y,
        X_eval,
        LowComplexityConfig(
            method=method,
            latent_width=4,
            random_state=17,
            max_epochs=8,
            patience=3,
        ),
    )

    assert result.predictions.shape == (len(X_eval),)
    assert np.isfinite(result.predictions).all()
    assert len(result.state_hash) == 64
    assert len(result.prediction_hash) == 64
    assert result.actual_latent_width == 4
    assert result.fitted_state_scalar_count > 0
    assert result.deployed_predictive_parameter_count > 0
    assert result.metadata["evaluation_labels_received"] is False


@pytest.mark.parametrize("method", LOW_COMPLEXITY_METHODS)
def test_every_method_is_deterministic_for_same_seed(method: str) -> None:
    X, y, X_eval = _fixture()
    config = LowComplexityConfig(
        method=method,
        latent_width=4,
        random_state=29,
        max_epochs=6,
        patience=2,
    )

    first = fit_low_complexity_representation(X, y, X_eval, config)
    second = fit_low_complexity_representation(X, y, X_eval, config)

    assert first.state_hash == second.state_hash
    assert first.prediction_hash == second.prediction_hash
    np.testing.assert_array_equal(first.predictions, second.predictions)


def test_strict_width_rejects_capacity_change_instead_of_capping() -> None:
    X, y, X_eval = _fixture()
    with pytest.raises(ValueError, match="strict common maximum"):
        fit_low_complexity_representation(
            X[:8],
            y[:8],
            X_eval,
            LowComplexityConfig(method="truncated_svd", latent_width=8),
        )


def test_sparse_selector_breaks_equal_score_ties_by_feature_index() -> None:
    X = np.zeros((12, 6), dtype=np.float32)
    X[:, 0] = np.arange(12) % 2
    X[:, 1] = X[:, 0]
    X[:, 2] = X[:, 0]
    y = X[:, 0].copy()

    selected, scores, captured = _stable_f_regression_top_k(X, y, width=2)

    np.testing.assert_array_equal(selected, np.asarray([0, 1]))
    assert scores[0] == scores[1] == scores[2]
    assert isinstance(captured, tuple)


def test_sparse_selector_records_numerical_warnings_instead_of_suppressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y, X_eval = _fixture()

    def warning_f_regression(
        features: np.ndarray,
        targets: np.ndarray,
        *,
        center: bool,
        force_finite: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        del targets, center, force_finite
        warnings.warn("synthetic selector warning", RuntimeWarning, stacklevel=2)
        return np.arange(features.shape[1], dtype=float), np.ones(features.shape[1])

    monkeypatch.setattr(
        "bh_augmentation.representations.low_complexity.f_regression",
        warning_f_regression,
    )

    result = fit_low_complexity_representation(
        X,
        y,
        X_eval,
        LowComplexityConfig(method="sparse_feature_selection", latent_width=4),
    )

    assert result.metadata["fit_warnings"] == (
        "RuntimeWarning: synthetic selector warning",
    )


def test_group_internal_split_is_deterministic_and_group_disjoint() -> None:
    groups = ("a", "a", "b", "b", "c", "c", "d", "d")
    first_fit, first_valid, first_hash = _make_internal_group_split(
        groups,
        valid_fraction=0.25,
        random_state=3,
    )
    second_fit, second_valid, second_hash = _make_internal_group_split(
        groups,
        valid_fraction=0.25,
        random_state=3,
    )

    np.testing.assert_array_equal(first_fit, second_fit)
    np.testing.assert_array_equal(first_valid, second_valid)
    assert first_hash == second_hash
    assert {groups[index] for index in first_fit}.isdisjoint(
        {groups[index] for index in first_valid}
    )


def test_bottleneck_and_direct_mlp_have_exactly_matched_architecture_capacity() -> None:
    X, y, X_eval = _fixture()
    common = {
        "latent_width": 4,
        "random_state": 43,
        "max_epochs": 8,
        "patience": 3,
    }
    bottleneck = fit_low_complexity_representation(
        X,
        y,
        X_eval,
        LowComplexityConfig(method="small_bottleneck_mlp", **common),
    )
    direct = fit_low_complexity_representation(
        X,
        y,
        X_eval,
        LowComplexityConfig(method="direct_mlp_regressor", **common),
    )

    expected_count = X.shape[1] * 4 + 2 * 4 + 1
    assert bottleneck.deployed_predictive_parameter_count == expected_count
    assert direct.deployed_predictive_parameter_count == expected_count
    assert (
        bottleneck.metadata["neural_model_state_hash"]
        == direct.metadata["neural_model_state_hash"]
    )
    assert bottleneck.selected_epoch == direct.selected_epoch
    assert bottleneck.internal_split_hash == direct.internal_split_hash


def test_tied_linear_autoencoder_count_does_not_double_decoder_weights() -> None:
    X, y, X_eval = _fixture()
    result = fit_low_complexity_representation(
        X,
        y,
        X_eval,
        LowComplexityConfig(
            method="linear_autoencoder",
            latent_width=4,
            max_epochs=3,
        ),
    )

    assert result.deployed_predictive_parameter_count == X.shape[1] * 4 + 4 + 1
    assert "K_tied" in result.metadata["parameter_count_formula"]


def test_eval_features_affect_predictions_but_not_fitted_state() -> None:
    X, y, X_eval = _fixture()
    config = LowComplexityConfig(method="partial_least_squares", latent_width=4)

    ordinary = fit_low_complexity_representation(X, y, X_eval, config)
    poisoned = fit_low_complexity_representation(X, y, X_eval + 50.0, config)

    assert ordinary.state_hash == poisoned.state_hash
    assert ordinary.prediction_hash != poisoned.prediction_hash


@pytest.mark.parametrize(
    ("method", "match"),
    [
        ("partial_least_squares", "nonconstant"),
        ("sparse_feature_selection", "nonconstant"),
    ],
)
def test_supervised_screeners_reject_constant_training_labels(
    method: str,
    match: str,
) -> None:
    X, _, X_eval = _fixture()
    with pytest.raises(ValueError, match=match):
        fit_low_complexity_representation(
            X,
            np.ones(len(X), dtype=np.float32),
            X_eval,
            LowComplexityConfig(method=method, latent_width=4),
        )


def test_nonfinite_or_mismatched_eval_features_are_rejected() -> None:
    X, y, X_eval = _fixture()
    config = LowComplexityConfig(method="truncated_svd", latent_width=4)
    bad = X_eval.copy()
    bad[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        fit_low_complexity_representation(X, y, bad, config)
    with pytest.raises(ValueError, match="matching width"):
        fit_low_complexity_representation(X, y, X_eval[:, :-1], config)


def test_runner_api_binds_internal_partitions_to_supplied_source_ids() -> None:
    X, y, X_eval = _fixture()
    source_ids = tuple(f"source-{index:03d}" for index in range(len(X)))
    groups = tuple(f"group-{index // 2:03d}" for index in range(len(X)))

    result = fit_predict_low_complexity(
        X,
        y,
        X_eval,
        train_source_ids=source_ids,
        train_group_ids=groups,
        config=LowComplexityConfig(
            method="direct_mlp_regressor",
            latent_width=4,
            random_state=13,
            max_epochs=5,
            patience=2,
            batch_size=7,
        ),
    )

    assert result.internal_fit_source_id_hash
    assert result.internal_validation_source_id_hash
    assert result.metadata["internal_fit_index_hash"]
    assert result.metadata["internal_validation_index_hash"]
    assert (
        result.metadata["internal_fit_count"]
        + result.metadata["internal_validation_count"]
        == len(X)
    )
