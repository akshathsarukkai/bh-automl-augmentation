"""Leakage-resistant uncertainty calibration metrics and policy selection.

This module is deliberately unaware of hidden-measured and outer-test outcomes.
Its selection records require explicit scientific data roles and permit policy
selection from validation evidence only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bh_augmentation.evaluation.metrics import spearman_corr

_ROLE_FIELDS = (
    "teacher_fit_roles",
    "uncertainty_calibration_roles",
    "policy_selection_roles",
    "candidate_source_roles",
    "candidate_donor_roles",
)
_EXPECTED_ROLES = {
    "teacher_fit_roles": ("teacher_fit",),
    "uncertainty_calibration_roles": ("interval_calibration",),
    "policy_selection_roles": ("policy_validation",),
    "candidate_source_roles": ("teacher_fit",),
    "candidate_donor_roles": ("teacher_fit",),
}
_SELECTION_TIE_BREAK = (
    "lowest validation selective RMSE; lowest calibration error; lowest mean "
    "interval width; highest uncertainty-error Spearman; highest retained "
    "fraction; lowest cutoff; stable method-config hash; stable method ID"
)


@dataclass(frozen=True, slots=True)
class CalibrationDataRoles:
    """Exact allowed data roles for fitting, calibration, and selection."""

    teacher_fit_roles: tuple[str, ...] = ("teacher_fit",)
    uncertainty_calibration_roles: tuple[str, ...] = ("interval_calibration",)
    policy_selection_roles: tuple[str, ...] = ("policy_validation",)
    candidate_source_roles: tuple[str, ...] = ("teacher_fit",)
    candidate_donor_roles: tuple[str, ...] = ("teacher_fit",)

    def __post_init__(self) -> None:
        for field in _ROLE_FIELDS:
            roles = getattr(self, field)
            if not isinstance(roles, tuple):
                raise TypeError(f"{field} must be an immutable tuple.")
            if roles != _EXPECTED_ROLES[field]:
                raise ValueError(
                    f"Contaminated calibration role metadata for {field}: "
                    f"expected {_EXPECTED_ROLES[field]}, observed {roles}."
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CalibrationDataRoles:
        """Strictly parse role metadata, rejecting unknown or missing fields."""
        if not isinstance(value, Mapping) or set(value) != set(_ROLE_FIELDS):
            raise ValueError("Calibration role metadata schema mismatch.")
        return cls(
            **{
                field: tuple(value[field])
                if isinstance(value[field], (list, tuple))
                else value[field]
                for field in _ROLE_FIELDS
            }
        )


@dataclass(frozen=True, slots=True)
class ConformalQuantile:
    """One finite-sample split-conformal absolute-residual quantile."""

    value: float
    rank: int
    n_calibration: int
    nominal_coverage: float
    finite_sample_level: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.value)
            or self.value < 0
            or isinstance(self.rank, bool)
            or not 1 <= self.rank <= self.n_calibration
        ):
            raise ValueError("Conformal quantile record is invalid.")
        _open_unit_interval(self.nominal_coverage, name="nominal_coverage")
        if not math.isclose(
            self.finite_sample_level,
            self.rank / self.n_calibration,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Conformal finite-sample level is inconsistent.")


@dataclass(frozen=True, slots=True)
class IntervalCalibrationMetrics:
    """Replayable interval and uncertainty calibration measurements."""

    sample_count: int
    nominal_coverage: float
    empirical_coverage: float
    mean_interval_width: float
    median_interval_width: float
    calibration_error: float
    uncertainty_error_spearman: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer.")
        nominal = _open_unit_interval(
            self.nominal_coverage, name="nominal_coverage"
        )
        empirical = _closed_unit_interval(
            self.empirical_coverage, name="empirical_coverage"
        )
        for name in ("mean_interval_width", "median_interval_width"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isclose(
            self.calibration_error,
            abs(empirical - nominal),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("calibration_error is inconsistent with coverage.")
        correlation = self.uncertainty_error_spearman
        if correlation is not None and (
            not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0
        ):
            raise ValueError("uncertainty_error_spearman must be finite and bounded.")


@dataclass(frozen=True, slots=True)
class SelectiveCurvePoint:
    """Metrics after retaining predictions no more uncertain than a cutoff."""

    uncertainty_cutoff: float
    retained_count: int
    total_count: int
    retained_fraction: float
    rmse: float
    mae: float
    mean_uncertainty: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.uncertainty_cutoff)
            or self.uncertainty_cutoff < 0
            or isinstance(self.retained_count, bool)
            or isinstance(self.total_count, bool)
            or not 1 <= self.retained_count <= self.total_count
        ):
            raise ValueError("Selective curve counts or cutoff are invalid.")
        if not math.isclose(
            self.retained_fraction,
            self.retained_count / self.total_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Selective retained fraction is inconsistent.")
        for name in ("rmse", "mae", "mean_uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Selective {name} must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class ValidationCalibrationCandidate:
    """One method's validation-only calibration evidence."""

    method_id: str
    method_config_hash: str
    metrics: IntervalCalibrationMetrics
    selective_curve: tuple[SelectiveCurvePoint, ...]
    data_roles: CalibrationDataRoles

    def __post_init__(self) -> None:
        _nonempty_string(self.method_id, name="method_id")
        _require_sha256(self.method_config_hash, name="method_config_hash")
        if not isinstance(self.selective_curve, tuple) or not self.selective_curve:
            raise ValueError("selective_curve must be a nonempty immutable tuple.")
        if not isinstance(self.data_roles, CalibrationDataRoles):
            raise TypeError("data_roles must be CalibrationDataRoles.")
        if any(
            point.total_count != self.metrics.sample_count
            for point in self.selective_curve
        ):
            raise ValueError("Selective curve and calibration sample counts differ.")
        cutoffs = tuple(point.uncertainty_cutoff for point in self.selective_curve)
        retained = tuple(point.retained_count for point in self.selective_curve)
        if cutoffs != tuple(sorted(set(cutoffs))) or retained != tuple(
            sorted(retained)
        ):
            raise ValueError("Selective curve must have increasing unique cutoffs.")


@dataclass(frozen=True, slots=True)
class ValidationCalibrationSelection:
    """Deterministic method and uncertainty cutoff frozen from validation."""

    method_id: str
    method_config_hash: str
    uncertainty_cutoff: float
    retained_count: int
    retained_fraction: float
    validation_rmse: float
    validation_mae: float
    nominal_coverage: float
    empirical_coverage: float
    calibration_error: float
    mean_interval_width: float
    uncertainty_error_spearman: float | None
    selection_data_role: str
    tie_break_protocol: str


def finite_sample_conformal_quantile(
    absolute_residuals: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float,
) -> ConformalQuantile:
    """Return the conservative finite-sample split-conformal quantile.

    The selected one-indexed order statistic has rank
    ``ceil((n + 1) * nominal_coverage)``, capped at ``n`` when the requested
    finite-sample level is unattainable.
    """
    residuals = _finite_vector(absolute_residuals, name="absolute_residuals")
    if (residuals < 0).any():
        raise ValueError("absolute_residuals must be nonnegative.")
    coverage = _open_unit_interval(nominal_coverage, name="nominal_coverage")
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * coverage))
    ordered = np.sort(residuals, kind="mergesort")
    return ConformalQuantile(
        value=float(ordered[rank - 1]),
        rank=rank,
        n_calibration=len(residuals),
        nominal_coverage=coverage,
        finite_sample_level=rank / len(residuals),
    )


def uncertainty_error_spearman(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    uncertainty: Sequence[float] | np.ndarray,
) -> float | None:
    """Return Spearman correlation of uncertainty with absolute error.

    ``None`` represents the scientifically undefined constant-vector case.
    """
    true, pred, score = _prediction_vectors(y_true, y_pred, uncertainty)
    correlation = float(spearman_corr(np.abs(true - pred), score))
    return correlation if math.isfinite(correlation) else None


def interval_calibration_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    uncertainty: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float,
) -> IntervalCalibrationMetrics:
    """Measure interval calibration and uncertainty-error correspondence."""
    true, pred, score = _prediction_vectors(y_true, y_pred, uncertainty)
    low = _finite_vector(lower, name="lower")
    high = _finite_vector(upper, name="upper")
    _same_length(true, low, name="lower")
    _same_length(true, high, name="upper")
    if (low > high).any():
        raise ValueError("Prediction intervals must not cross.")
    coverage = _open_unit_interval(nominal_coverage, name="nominal_coverage")
    empirical = float(np.mean((true >= low) & (true <= high)))
    widths = high - low
    correlation = float(spearman_corr(np.abs(true - pred), score))
    return IntervalCalibrationMetrics(
        sample_count=len(true),
        nominal_coverage=coverage,
        empirical_coverage=empirical,
        mean_interval_width=float(np.mean(widths)),
        median_interval_width=float(np.median(widths)),
        calibration_error=abs(empirical - coverage),
        uncertainty_error_spearman=(
            correlation if math.isfinite(correlation) else None
        ),
    )


def build_selective_prediction_curve(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    uncertainty: Sequence[float] | np.ndarray,
    sample_ids: Sequence[str],
) -> tuple[SelectiveCurvePoint, ...]:
    """Build a deterministic low-to-high uncertainty retention curve.

    All samples tied at a cutoff are retained together. Stable source IDs are
    validated and used to make prediction-table ordering irrelevant.
    """
    true, pred, score = _prediction_vectors(y_true, y_pred, uncertainty)
    ids = tuple(sample_ids)
    if len(ids) != len(true):
        raise ValueError("sample_ids and prediction arrays must have equal length.")
    if (
        any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("sample_ids must contain unique nonempty strings.")
    order = sorted(range(len(ids)), key=lambda index: (score[index], ids[index]))
    true = true[order]
    pred = pred[order]
    score = score[order]

    points = []
    for cutoff in np.unique(score):
        retained = score <= cutoff
        errors = true[retained] - pred[retained]
        points.append(
            SelectiveCurvePoint(
                uncertainty_cutoff=float(cutoff),
                retained_count=int(retained.sum()),
                total_count=len(true),
                retained_fraction=float(retained.mean()),
                rmse=float(np.sqrt(np.mean(errors**2))),
                mae=float(np.mean(np.abs(errors))),
                mean_uncertainty=float(np.mean(score[retained])),
            )
        )
    return tuple(points)


def select_validation_policy(
    candidates: Sequence[ValidationCalibrationCandidate],
    *,
    nominal_coverage: float,
    max_calibration_error: float,
    min_retained_fraction: float,
) -> ValidationCalibrationSelection:
    """Select a method and cutoff using constrained validation evidence only."""
    expected_coverage = _open_unit_interval(
        nominal_coverage, name="nominal_coverage"
    )
    maximum_error = _closed_unit_interval(
        max_calibration_error, name="max_calibration_error"
    )
    minimum_retained = _open_unit_interval(
        min_retained_fraction, name="min_retained_fraction"
    )
    records = tuple(candidates)
    if not records:
        raise ValueError("Validation calibration selection requires candidates.")
    identities = [
        (candidate.method_id, candidate.method_config_hash)
        for candidate in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Validation calibration candidate identities must be unique.")

    eligible: list[
        tuple[ValidationCalibrationCandidate, SelectiveCurvePoint]
    ] = []
    for candidate in records:
        metrics = candidate.metrics
        if not math.isclose(
            metrics.nominal_coverage,
            expected_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            continue
        if metrics.calibration_error > maximum_error:
            continue
        eligible.extend(
            (candidate, point)
            for point in candidate.selective_curve
            if point.retained_fraction >= minimum_retained
        )
    if not eligible:
        raise ValueError(
            "No validation calibration candidate satisfies the predefined constraints."
        )

    def rank(
        record: tuple[ValidationCalibrationCandidate, SelectiveCurvePoint],
    ) -> tuple[Any, ...]:
        candidate, point = record
        correlation = candidate.metrics.uncertainty_error_spearman
        return (
            point.rmse,
            candidate.metrics.calibration_error,
            candidate.metrics.mean_interval_width,
            -(correlation if correlation is not None else -math.inf),
            -point.retained_fraction,
            point.uncertainty_cutoff,
            candidate.method_config_hash,
            candidate.method_id,
        )

    selected_candidate, selected_point = min(eligible, key=rank)
    metrics = selected_candidate.metrics
    return ValidationCalibrationSelection(
        method_id=selected_candidate.method_id,
        method_config_hash=selected_candidate.method_config_hash,
        uncertainty_cutoff=selected_point.uncertainty_cutoff,
        retained_count=selected_point.retained_count,
        retained_fraction=selected_point.retained_fraction,
        validation_rmse=selected_point.rmse,
        validation_mae=selected_point.mae,
        nominal_coverage=metrics.nominal_coverage,
        empirical_coverage=metrics.empirical_coverage,
        calibration_error=metrics.calibration_error,
        mean_interval_width=metrics.mean_interval_width,
        uncertainty_error_spearman=metrics.uncertainty_error_spearman,
        selection_data_role="policy_validation",
        tie_break_protocol=_SELECTION_TIE_BREAK,
    )


def _prediction_vectors(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    uncertainty: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = _finite_vector(y_true, name="y_true")
    pred = _finite_vector(y_pred, name="y_pred")
    score = _finite_vector(uncertainty, name="uncertainty")
    _same_length(true, pred, name="y_pred")
    _same_length(true, score, name="uncertainty")
    if (score < 0).any():
        raise ValueError("uncertainty must be nonnegative.")
    return true, pred, score


def _finite_vector(
    values: Sequence[float] | np.ndarray, *, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _same_length(reference: np.ndarray, value: np.ndarray, *, name: str) -> None:
    if len(reference) != len(value):
        raise ValueError(
            f"{name} must have length {len(reference)}; observed {len(value)}."
        )


def _open_unit_interval(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one.")
    return result


def _closed_unit_interval(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one.")
    return result


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    text = _nonempty_string(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return text
