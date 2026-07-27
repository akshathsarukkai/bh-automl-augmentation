"""Focused tests for deterministic uncertainty estimators."""

from __future__ import annotations

import numpy as np
import pytest

from bh_augmentation.uncertainty.estimators import (
    SUPPORTED_UNCERTAINTY_METHODS,
    UncertaintyEstimatorConfig,
    fit_uncertainty_estimator,
)


def _partitions() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(41)
    X = rng.normal(size=(30, 4))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.4 * X[:, 2] ** 2
    return (
        X[:20],
        y[:20],
        [f"train-{index}" for index in range(20)],
        X[20:],
        y[20:] + np.linspace(-0.5, 0.5, 10),
        [f"calibration-{index}" for index in range(10)],
    )


def _small_config(method: str) -> UncertaintyEstimatorConfig:
    return UncertaintyEstimatorConfig(
        method=method,  # type: ignore[arg-type]
        coverage=0.8,
        random_state=17,
        bootstrap_members=3,
        bootstrap_trees_per_member=4,
        xgboost_seeds=(5, 7),
        xgboost_params={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "tree_method": "hist",
        },
        quantile_n_estimators=10,
        heterogeneous_tree_estimators=6,
    )


@pytest.mark.parametrize("method", SUPPORTED_UNCERTAINTY_METHODS)
def test_all_six_methods_are_deterministic_and_produce_ordered_intervals(
    method: str,
) -> None:
    X_train, y_train, train_ids, X_cal, y_cal, cal_ids = _partitions()
    config = _small_config(method)
    fitted = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=config,
    )
    repeated = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=config,
    )
    prediction = fitted.predict(X_cal[:3], source_ids=["query-0", "query-1", "query-2"])
    repeated_prediction = repeated.predict(
        X_cal[:3], source_ids=["query-0", "query-1", "query-2"]
    )

    assert fitted.audit == repeated.audit
    assert fitted.audit.audit_hash == repeated.audit.audit_hash
    assert prediction.prediction_hash == repeated_prediction.prediction_hash
    for name in (
        "point",
        "raw_score",
        "raw_lower",
        "raw_upper",
        "interval_lower",
        "interval_upper",
        "calibrated_uncertainty",
    ):
        first = getattr(prediction, name)
        second = getattr(repeated_prediction, name)
        assert np.array_equal(first, second)
        assert np.isfinite(first).all()
        assert not first.flags.writeable
    assert np.all(prediction.raw_lower <= prediction.raw_upper)
    assert np.all(prediction.interval_lower <= prediction.interval_upper)
    assert np.all(prediction.calibrated_uncertainty >= 0.0)


def test_fit_rejects_partition_overlap_and_malformed_data() -> None:
    X_train, y_train, train_ids, X_cal, y_cal, cal_ids = _partitions()
    config = _small_config("split_conformal_ridge")
    cal_ids[0] = train_ids[0]
    with pytest.raises(ValueError, match="overlap"):
        fit_uncertainty_estimator(
            X_train,
            y_train,
            train_source_ids=train_ids,
            calibration_features=X_cal,
            calibration_labels=y_cal,
            calibration_source_ids=cal_ids,
            config=config,
        )

    with pytest.raises(ValueError, match="finite"):
        fit_uncertainty_estimator(
            X_train,
            y_train,
            train_source_ids=train_ids,
            calibration_features=X_cal,
            calibration_labels=np.full(len(X_cal), np.nan),
            calibration_source_ids=[f"cal-{index}" for index in range(len(X_cal))],
            config=config,
        )
    with pytest.raises(ValueError, match="widths differ"):
        fit_uncertainty_estimator(
            X_train,
            y_train,
            train_source_ids=train_ids,
            calibration_features=X_cal[:, :2],
            calibration_labels=y_cal,
            calibration_source_ids=[f"cal-{index}" for index in range(len(X_cal))],
            config=config,
        )


def test_split_conformal_has_exact_finite_sample_residual_quantile() -> None:
    fitted = fit_uncertainty_estimator(
        np.zeros((3, 1)),
        np.zeros(3),
        train_source_ids=["train-0", "train-1", "train-2"],
        calibration_features=np.zeros((4, 1)),
        calibration_labels=np.array([1.0, 2.0, 3.0, 4.0]),
        calibration_source_ids=["cal-0", "cal-1", "cal-2", "cal-3"],
        config=UncertaintyEstimatorConfig(
            method="split_conformal_ridge",
            coverage=0.8,
        ),
    )
    prediction = fitted.predict(np.zeros((1, 1)), source_ids=["query"])

    assert fitted.audit.calibration_quantile_level == 1.0
    assert fitted.audit.calibration_quantile == 4.0
    assert prediction.point.tolist() == [0.0]
    assert prediction.raw_lower.tolist() == [0.0]
    assert prediction.raw_upper.tolist() == [0.0]
    assert prediction.interval_lower.tolist() == [-4.0]
    assert prediction.interval_upper.tolist() == [4.0]


def test_calibration_labels_change_only_calibrated_interval_not_raw_prediction() -> None:
    X_train, y_train, train_ids, X_cal, y_cal, cal_ids = _partitions()
    config = _small_config("heterogeneous_disagreement")
    first = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=config,
    )
    second = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal + 100.0,
        calibration_source_ids=cal_ids,
        config=config,
    )
    query_ids = ["query-0", "query-1"]
    first_prediction = first.predict(X_cal[:2], source_ids=query_ids)
    second_prediction = second.predict(X_cal[:2], source_ids=query_ids)

    assert np.array_equal(first_prediction.point, second_prediction.point)
    assert np.array_equal(first_prediction.raw_score, second_prediction.raw_score)
    assert np.array_equal(first_prediction.raw_lower, second_prediction.raw_lower)
    assert np.array_equal(first_prediction.raw_upper, second_prediction.raw_upper)
    assert not np.array_equal(
        first_prediction.calibrated_uncertainty,
        second_prediction.calibrated_uncertainty,
    )


def test_bootstrap_audit_records_each_sample_and_seed_hash() -> None:
    X_train, y_train, train_ids, X_cal, y_cal, cal_ids = _partitions()
    fitted = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=_small_config("bootstrap_extra_trees"),
    )

    hashes = fitted.audit.bootstrap_sample_hashes
    assert len(hashes) == 3
    assert len(set(hashes)) == 3
    assert len(fitted.audit.seed_hash) == 64
    assert len(fitted.audit.sample_hash) == 64
    assert len(fitted.audit.config_hash) == 64
    assert fitted.audit.model_names == (
        "ExtraTreesRegressor",
        "ExtraTreesRegressor",
        "ExtraTreesRegressor",
    )


def test_distance_aware_raw_score_is_nearest_cosine_distance_with_floor() -> None:
    config = UncertaintyEstimatorConfig(
        method="distance_aware_ridge",
        coverage=0.5,
        minimum_raw_scale=0.01,
    )
    fitted = fit_uncertainty_estimator(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 2.0]),
        train_source_ids=["train-0", "train-1"],
        calibration_features=np.array([[1.0, 1.0], [-1.0, 1.0]]),
        calibration_labels=np.array([1.5, 2.5]),
        calibration_source_ids=["cal-0", "cal-1"],
        config=config,
    )
    prediction = fitted.predict(
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        source_ids=["query-0", "query-1"],
    )

    assert prediction.raw_score[0] == pytest.approx(0.01)
    assert prediction.raw_score[1] == pytest.approx(1.0 - 1.0 / np.sqrt(2.0))


def test_method_specific_audit_identifies_seeded_xgboost_and_fixed_heterogeneous_family() -> None:
    X_train, y_train, train_ids, X_cal, y_cal, cal_ids = _partitions()
    xgboost = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=_small_config("seeded_xgboost"),
    )
    heterogeneous = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=_small_config("heterogeneous_disagreement"),
    )

    assert xgboost.audit.seed_hash != heterogeneous.audit.seed_hash
    assert heterogeneous.audit.model_names == (
        "Ridge",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
    )


def test_prediction_rejects_duplicate_ids_and_wrong_width() -> None:
    X_train, y_train, train_ids, X_cal, y_cal, cal_ids = _partitions()
    fitted = fit_uncertainty_estimator(
        X_train,
        y_train,
        train_source_ids=train_ids,
        calibration_features=X_cal,
        calibration_labels=y_cal,
        calibration_source_ids=cal_ids,
        config=_small_config("split_conformal_ridge"),
    )
    with pytest.raises(ValueError, match="unique"):
        fitted.predict(X_cal[:2], source_ids=["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="width differs"):
        fitted.predict(X_cal[:2, :2], source_ids=["query-0", "query-1"])
