"""Focused gates for the parameter-matched direct-MLP control."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from bh_augmentation.representations.matched_direct_mlp import (
    MatchedDirectMLPConfig,
    _make_group_disjoint_split,
    fit_predict_matched_direct_mlp,
    matched_direct_mlp_parameter_count,
)


def _fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    generator = np.random.default_rng(1407)
    X_train = generator.normal(size=(24, 12)).astype(np.float32)
    y_train = (
        4.0 * X_train[:, 1]
        - 2.0 * X_train[:, 5]
        + generator.normal(0, 0.1, size=len(X_train))
    ).astype(np.float32)
    X_eval = generator.normal(size=(5, 12)).astype(np.float32)
    source_ids = tuple(f"measured-{index:03d}" for index in range(24))
    group_ids = tuple(f"group-{index // 3:03d}" for index in range(24))
    return X_train, y_train, X_eval, source_ids, group_ids


def _config() -> MatchedDirectMLPConfig:
    return MatchedDirectMLPConfig(
        input_dim=12,
        hidden_dim=8,
        latent_dim=4,
        learning_rate=0.005,
        batch_size=6,
        max_epochs=8,
        patience=3,
        internal_valid_fraction=0.25,
        random_state=31,
    )


@pytest.mark.parametrize(
    ("input_dim", "hidden_dim", "latent_dim", "expected"),
    [
        (14336, 128, 16, 1_837_217),
        (14336, 256, 32, 3_678_529),
    ],
)
def test_canonical_parameter_formulas_without_large_fit(
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    expected: int,
) -> None:
    assert (
        matched_direct_mlp_parameter_count(
            input_dim,
            hidden_dim,
            latent_dim,
        )
        == expected
    )


def test_api_cannot_receive_evaluation_labels() -> None:
    parameters = inspect.signature(
        fit_predict_matched_direct_mlp
    ).parameters
    assert "y_evaluation" not in parameters
    assert "y_eval" not in parameters
    assert set(parameters) == {
        "X_measured_train",
        "y_measured_train",
        "X_evaluation",
        "measured_source_ids",
        "measured_group_ids",
        "config",
        "X_synthetic",
        "y_synthetic",
        "synthetic_source_ids",
        "synthetic_group_ids",
    }


def test_fit_is_deterministic_and_input_order_invariant() -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    first = fit_predict_matched_direct_mlp(
        X_train,
        y_train,
        X_eval,
        measured_source_ids=source_ids,
        measured_group_ids=group_ids,
        config=_config(),
    )
    second = fit_predict_matched_direct_mlp(
        X_train,
        y_train,
        X_eval,
        measured_source_ids=source_ids,
        measured_group_ids=group_ids,
        config=_config(),
    )
    order = np.random.default_rng(19).permutation(len(X_train))
    reordered = fit_predict_matched_direct_mlp(
        X_train[order],
        y_train[order],
        X_eval,
        measured_source_ids=tuple(source_ids[index] for index in order),
        measured_group_ids=tuple(group_ids[index] for index in order),
        config=_config(),
    )

    assert first.state_hash == second.state_hash == reordered.state_hash
    assert (
        first.prediction_hash
        == second.prediction_hash
        == reordered.prediction_hash
    )
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_array_equal(first.predictions, reordered.predictions)
    assert first.metadata["evaluation_labels_received"] is False


def test_internal_validation_is_group_disjoint_and_train_derived() -> None:
    _, _, _, source_ids, group_ids = _fixture()
    split = _make_group_disjoint_split(
        source_ids,
        group_ids,
        valid_fraction=0.25,
        random_state=31,
    )
    fit_groups = {group_ids[index] for index in split.fit_indices}
    validation_groups = {
        group_ids[index] for index in split.validation_indices
    }

    assert fit_groups
    assert validation_groups
    assert fit_groups.isdisjoint(validation_groups)
    assert fit_groups == set(split.fit_groups)
    assert validation_groups == set(split.validation_groups)


def test_final_scaler_and_refit_are_bound_to_all_measured_rows() -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    ordinary = fit_predict_matched_direct_mlp(
        X_train,
        y_train,
        X_eval,
        measured_source_ids=source_ids,
        measured_group_ids=group_ids,
        config=_config(),
    )
    poisoned_eval = fit_predict_matched_direct_mlp(
        X_train,
        y_train,
        X_eval + 100.0,
        measured_source_ids=source_ids,
        measured_group_ids=group_ids,
        config=_config(),
    )

    assert ordinary.state_hash == poisoned_eval.state_hash
    assert ordinary.target_scaler_hash == poisoned_eval.target_scaler_hash
    assert ordinary.target_scaler_source_id_hash == stable_source_hash(
        source_ids
    )
    assert ordinary.metadata["target_y_mean"] == pytest.approx(
        float(np.mean(y_train, dtype=np.float64))
    )
    assert ordinary.metadata["refit_epoch_count"] == ordinary.selected_epoch
    assert (
        ordinary.metadata["refit_rows"]
        == "all_measured_training_rows_plus_all_synthetic_rows"
    )
    assert ordinary.prediction_hash != poisoned_eval.prediction_hash


def _synthetic() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    generator = np.random.default_rng(5150)
    X_added = generator.normal(size=(6, 12)).astype(np.float32)
    # Far outside the measured target range: any leak into the target scaler or
    # into the early-stopping monitor would be visible immediately.
    y_added = np.full(6, 900.0, dtype=np.float32)
    return (
        X_added,
        y_added,
        tuple(f"synthetic-{index:03d}" for index in range(6)),
        tuple(f"synthetic-group-{index:03d}" for index in range(6)),
    )


def test_synthetic_rows_train_the_control_but_never_the_scaler_or_monitor() -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    X_added, y_added, added_ids, added_groups = _synthetic()
    measured_only = fit_predict_matched_direct_mlp(
        X_train,
        y_train,
        X_eval,
        measured_source_ids=source_ids,
        measured_group_ids=group_ids,
        config=_config(),
    )
    augmented = fit_predict_matched_direct_mlp(
        X_train,
        y_train,
        X_eval,
        measured_source_ids=source_ids,
        measured_group_ids=group_ids,
        config=replace(_config(), synthetic_supervised_weight=0.25),
        X_synthetic=X_added,
        y_synthetic=y_added,
        synthetic_source_ids=added_ids,
        synthetic_group_ids=added_groups,
    )

    # The synthetic pool genuinely participates in supervised training.
    assert augmented.state_hash != measured_only.state_hash
    assert augmented.prediction_hash != measured_only.prediction_hash
    assert augmented.metadata["synthetic_train_row_count"] == 6
    assert augmented.metadata[
        "effective_supervised_weight_sum"
    ] == pytest.approx(24 + 6 * 0.25)
    assert augmented.metadata["supervised_weight_semantics"] == (
        "measured=1,synthetic=synthetic_supervised_weight"
    )
    assert augmented.synthetic_train_source_id_hash == stable_source_hash(
        added_ids
    )
    # Target scaling and early stopping stay measured-only and unchanged.
    assert augmented.target_scaler_hash == measured_only.target_scaler_hash
    assert (
        augmented.target_scaler_source_id_hash
        == measured_only.target_scaler_source_id_hash
    )
    assert augmented.metadata["target_y_mean"] == pytest.approx(
        float(np.mean(y_train, dtype=np.float64))
    )
    assert augmented.internal_split_hash == measured_only.internal_split_hash
    assert (
        augmented.internal_validation_source_id_hash
        == measured_only.internal_validation_source_id_hash
    )
    assert (
        augmented.internal_fit_source_id_hash
        == measured_only.internal_fit_source_id_hash
    )
    assert augmented.metadata["epoch_selection_rows"] == (
        "group_disjoint_measured_training_rows_only"
    )


def test_synthetic_supervised_weight_changes_the_fitted_control() -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    X_added, y_added, added_ids, added_groups = _synthetic()

    def fit(weight: float) -> object:
        return fit_predict_matched_direct_mlp(
            X_train,
            y_train,
            X_eval,
            measured_source_ids=source_ids,
            measured_group_ids=group_ids,
            config=replace(_config(), synthetic_supervised_weight=weight),
            X_synthetic=X_added,
            y_synthetic=y_added,
            synthetic_source_ids=added_ids,
            synthetic_group_ids=added_groups,
        )

    assert fit(0.25).state_hash != fit(1.0).state_hash


def test_synthetic_identities_must_be_disjoint_from_measured_rows() -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    X_added, y_added, _added_ids, added_groups = _synthetic()

    with pytest.raises(ValueError, match="disjoint"):
        fit_predict_matched_direct_mlp(
            X_train,
            y_train,
            X_eval,
            measured_source_ids=source_ids,
            measured_group_ids=group_ids,
            config=_config(),
            X_synthetic=X_added,
            y_synthetic=y_added,
            synthetic_source_ids=source_ids[:6],
            synthetic_group_ids=added_groups,
        )


def stable_source_hash(source_ids: tuple[str, ...]) -> str:
    from bh_augmentation.utils.corrected_runs import stable_hash

    return stable_hash(sorted(source_ids))


@pytest.mark.parametrize(("array_name", "bad_value"), [
    ("X_train", np.nan),
    ("y_train", np.inf),
    ("X_eval", -np.inf),
])
def test_nonfinite_values_are_rejected(
    array_name: str,
    bad_value: float,
) -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    arrays = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_eval": X_eval.copy(),
    }
    arrays[array_name].reshape(-1)[0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        fit_predict_matched_direct_mlp(
            arrays["X_train"],
            arrays["y_train"],
            arrays["X_eval"],
            measured_source_ids=source_ids,
            measured_group_ids=group_ids,
            config=_config(),
        )


def test_duplicate_sources_or_single_group_are_rejected() -> None:
    X_train, y_train, X_eval, source_ids, group_ids = _fixture()
    duplicate_sources = (*source_ids[:-1], source_ids[0])

    with pytest.raises(ValueError, match="unique"):
        fit_predict_matched_direct_mlp(
            X_train,
            y_train,
            X_eval,
            measured_source_ids=duplicate_sources,
            measured_group_ids=group_ids,
            config=_config(),
        )
    with pytest.raises(ValueError, match="at least two"):
        fit_predict_matched_direct_mlp(
            X_train,
            y_train,
            X_eval,
            measured_source_ids=source_ids,
            measured_group_ids=("one-group",) * len(source_ids),
            config=_config(),
        )
