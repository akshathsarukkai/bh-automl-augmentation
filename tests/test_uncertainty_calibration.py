"""Tests for leakage-resistant Phase 12 calibration primitives."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bh_augmentation.uncertainty.calibration import (
    CalibrationDataRoles,
    IntervalCalibrationMetrics,
    SelectiveCurvePoint,
    ValidationCalibrationCandidate,
    build_selective_prediction_curve,
    finite_sample_conformal_quantile,
    interval_calibration_metrics,
    select_validation_policy,
    uncertainty_error_spearman,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(
    *,
    empirical_coverage: float = 0.8,
    mean_width: float = 4.0,
    correlation: float | None = 0.7,
) -> IntervalCalibrationMetrics:
    return IntervalCalibrationMetrics(
        sample_count=5,
        nominal_coverage=0.8,
        empirical_coverage=empirical_coverage,
        mean_interval_width=mean_width,
        median_interval_width=mean_width,
        calibration_error=abs(empirical_coverage - 0.8),
        uncertainty_error_spearman=correlation,
    )


def _point(
    cutoff: float,
    retained: int,
    rmse: float,
    *,
    mae: float | None = None,
) -> SelectiveCurvePoint:
    return SelectiveCurvePoint(
        uncertainty_cutoff=cutoff,
        retained_count=retained,
        total_count=5,
        retained_fraction=retained / 5,
        rmse=rmse,
        mae=rmse if mae is None else mae,
        mean_uncertainty=cutoff / 2,
    )


def _candidate(
    method_id: str,
    *,
    metrics: IntervalCalibrationMetrics | None = None,
    curve: tuple[SelectiveCurvePoint, ...] | None = None,
    config_hash: str | None = None,
    roles: CalibrationDataRoles | None = None,
) -> ValidationCalibrationCandidate:
    return ValidationCalibrationCandidate(
        method_id=method_id,
        method_config_hash=_digest(method_id) if config_hash is None else config_hash,
        metrics=_metrics() if metrics is None else metrics,
        selective_curve=(
            (_point(0.2, 3, 3.0), _point(0.8, 5, 4.0))
            if curve is None
            else curve
        ),
        data_roles=CalibrationDataRoles() if roles is None else roles,
    )


def test_finite_sample_conformal_quantile_uses_conservative_order_statistic() -> None:
    result = finite_sample_conformal_quantile(
        [4.0, 1.0, 3.0, 2.0],
        nominal_coverage=0.5,
    )

    assert result.rank == 3
    assert result.value == 3.0
    assert result.n_calibration == 4
    assert result.finite_sample_level == 0.75


def test_conformal_quantile_caps_unattainable_rank_at_maximum_residual() -> None:
    result = finite_sample_conformal_quantile(
        [1.0, 2.0, 9.0],
        nominal_coverage=0.95,
    )

    assert result.rank == 3
    assert result.value == 9.0
    assert result.finite_sample_level == 1.0


@pytest.mark.parametrize(
    ("residuals", "coverage"),
    (
        ([], 0.8),
        ([1.0, -0.1], 0.8),
        ([1.0, np.nan], 0.8),
        ([1.0, np.inf], 0.8),
        ([1.0], 0.0),
        ([1.0], 1.0),
    ),
)
def test_conformal_quantile_rejects_invalid_inputs(
    residuals: list[float],
    coverage: float,
) -> None:
    with pytest.raises(ValueError):
        finite_sample_conformal_quantile(
            residuals,
            nominal_coverage=coverage,
        )


def test_interval_metrics_replay_coverage_width_error_and_correlation() -> None:
    result = interval_calibration_metrics(
        y_true=[0.0, 1.0, 2.0, 3.0],
        y_pred=[0.0, 0.0, 0.0, 0.0],
        lower=[-1.0, -1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0, 1.0],
        uncertainty=[0.0, 1.0, 2.0, 3.0],
        nominal_coverage=0.75,
    )

    assert result.sample_count == 4
    assert result.empirical_coverage == 0.5
    assert result.mean_interval_width == 2.0
    assert result.median_interval_width == 2.0
    assert result.calibration_error == 0.25
    assert result.uncertainty_error_spearman == pytest.approx(1.0)


def test_constant_uncertainty_correlation_is_explicitly_undefined() -> None:
    assert (
        uncertainty_error_spearman(
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("lower", [0.0, 3.0], "cross"),
        ("upper", [1.0, np.nan], "finite"),
        ("uncertainty", [0.1, -0.2], "nonnegative"),
        ("y_pred", [0.0], "length"),
    ),
)
def test_interval_metrics_reject_crossing_nonfinite_or_misaligned_arrays(
    field: str,
    replacement: list[float],
    error: str,
) -> None:
    values = {
        "y_true": [0.0, 1.0],
        "y_pred": [0.0, 1.0],
        "lower": [-1.0, 0.0],
        "upper": [1.0, 2.0],
        "uncertainty": [0.1, 0.2],
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=error):
        interval_calibration_metrics(
            **values,
            nominal_coverage=0.8,
        )


def test_selective_curve_is_deterministic_and_permutation_invariant() -> None:
    kwargs = {
        "y_true": np.asarray([0.0, 5.0, 2.0, 8.0]),
        "y_pred": np.asarray([1.0, 1.0, 1.0, 1.0]),
        "uncertainty": np.asarray([0.2, 0.1, 0.2, 0.9]),
        "sample_ids": ("b", "a", "c", "d"),
    }
    expected = build_selective_prediction_curve(**kwargs)
    permutation = np.asarray([3, 1, 0, 2])
    observed = build_selective_prediction_curve(
        y_true=kwargs["y_true"][permutation],
        y_pred=kwargs["y_pred"][permutation],
        uncertainty=kwargs["uncertainty"][permutation],
        sample_ids=tuple(kwargs["sample_ids"][index] for index in permutation),
    )

    assert observed == expected
    assert tuple(point.uncertainty_cutoff for point in expected) == (0.1, 0.2, 0.9)
    assert tuple(point.retained_count for point in expected) == (1, 3, 4)
    assert expected[-1].rmse == pytest.approx(
        np.sqrt(np.mean(np.asarray([1.0, 4.0, 1.0, 7.0]) ** 2))
    )


@pytest.mark.parametrize(
    "sample_ids",
    (("a",), ("a", "a"), ("a", "")),
)
def test_selective_curve_requires_complete_unique_stable_ids(
    sample_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="sample_ids"):
        build_selective_prediction_curve(
            [0.0, 1.0],
            [0.0, 1.0],
            [0.1, 0.2],
            sample_ids,
        )


def test_validation_selector_enforces_constraints_and_minimum_retention() -> None:
    poorly_calibrated = _candidate(
        "poor-calibration",
        metrics=_metrics(empirical_coverage=0.4),
        curve=(_point(0.1, 5, 0.1),),
    )
    eligible = _candidate(
        "eligible",
        curve=(_point(0.1, 2, 0.1), _point(0.4, 4, 2.0)),
    )

    result = select_validation_policy(
        [poorly_calibrated, eligible],
        nominal_coverage=0.8,
        max_calibration_error=0.1,
        min_retained_fraction=0.6,
    )

    assert result.method_id == "eligible"
    assert result.uncertainty_cutoff == 0.4
    assert result.retained_fraction == 0.8
    assert result.selection_data_role == "policy_validation"
    assert "validation selective RMSE" in result.tie_break_protocol


def test_validation_selector_uses_stable_ties_not_input_order() -> None:
    lower_hash = _candidate(
        "method-z",
        config_hash="0" * 64,
        curve=(_point(0.5, 4, 2.0),),
    )
    higher_hash = _candidate(
        "method-a",
        config_hash="f" * 64,
        curve=(_point(0.5, 4, 2.0),),
    )

    first = select_validation_policy(
        [higher_hash, lower_hash],
        nominal_coverage=0.8,
        max_calibration_error=0.1,
        min_retained_fraction=0.6,
    )
    second = select_validation_policy(
        [lower_hash, higher_hash],
        nominal_coverage=0.8,
        max_calibration_error=0.1,
        min_retained_fraction=0.6,
    )

    assert first == second
    assert first.method_id == "method-z"


def test_validation_selector_rejects_no_eligible_or_duplicate_candidates() -> None:
    candidate = _candidate("method")
    with pytest.raises(ValueError, match="No validation"):
        select_validation_policy(
            [candidate],
            nominal_coverage=0.9,
            max_calibration_error=0.01,
            min_retained_fraction=0.8,
        )
    with pytest.raises(ValueError, match="identities"):
        select_validation_policy(
            [candidate, candidate],
            nominal_coverage=0.8,
            max_calibration_error=0.1,
            min_retained_fraction=0.6,
        )


@pytest.mark.parametrize(
    ("field", "contaminated"),
    (
        ("teacher_fit_roles", ("teacher_fit", "hidden_measured_evaluation")),
        ("uncertainty_calibration_roles", ("outer_test",)),
        ("policy_selection_roles", ("hidden_measured_evaluation",)),
        ("candidate_source_roles", ("policy_validation",)),
        ("candidate_donor_roles", ("outer_test",)),
    ),
)
def test_role_metadata_rejects_every_contaminated_path(
    field: str,
    contaminated: tuple[str, ...],
) -> None:
    metadata = {
        "teacher_fit_roles": ["teacher_fit"],
        "uncertainty_calibration_roles": ["interval_calibration"],
        "policy_selection_roles": ["policy_validation"],
        "candidate_source_roles": ["teacher_fit"],
        "candidate_donor_roles": ["teacher_fit"],
    }
    metadata[field] = list(contaminated)

    with pytest.raises(ValueError, match="Contaminated"):
        CalibrationDataRoles.from_mapping(metadata)


def test_role_metadata_schema_is_strict() -> None:
    with pytest.raises(ValueError, match="schema"):
        CalibrationDataRoles.from_mapping(
            {
                "teacher_fit_roles": ["teacher_fit"],
                "uncertainty_calibration_roles": ["interval_calibration"],
                "policy_selection_roles": ["policy_validation"],
                "candidate_source_roles": ["teacher_fit"],
                "candidate_donor_roles": ["teacher_fit"],
                "hidden_labels": ["allowed"],
            }
        )
