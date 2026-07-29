"""Validation-only retention decision for the redesigned supervised AE."""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bh_augmentation.utils.corrected_runs import stable_hash

AE_RETENTION_SCHEMA_VERSION = "bh-ae-retention-decision-v1"
MINIMUM_RETENTION_SEEDS = 3
AE_RETENTION_CONTROL_FAMILIES = (
    "truncated_svd",
    "linear_autoencoder",
    "direct_xgboost",
    "matched_direct_mlp",
    "anonymous_transfer_without_ae",
    "typed_transfer_without_ae",
)

_RECORD_FIELDS = {
    "evaluation_unit",
    "seed",
    "train_fraction",
    "method_family",
    "metric",
    "value",
    "selected_policy_hash",
    "frozen_policy_hash",
    "data_role",
}
_DISALLOWED_SELECTION_OR_TEST_FIELDS = {
    "selected",
    "selected_within_family",
    "selected_within_method",
    "winner",
    "overall_winner",
    "split",
    "test_evaluated",
    "test_used_for_selection",
    "outer_test_labels_accessed",
    "outer_test_predictions_generated",
}


@dataclass(frozen=True, slots=True)
class AERetentionConfig:
    """Frozen validation-placement gate and deterministic bootstrap settings."""

    fractions: tuple[float, ...]
    seeds: tuple[int, ...]
    ae_family: str
    control_families: tuple[str, ...]
    practical_margin: float
    required_margin_wins_per_fraction: int
    no_practical_loss_threshold: float
    bootstrap_reps: int
    bootstrap_alpha: float
    bootstrap_seed: int


class ImmutableJSONPayload(dict[str, Any]):
    """A recursively immutable dict subclass supported by ``json.dumps``."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        dict.__init__(self, value)

    def _immutable(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("Retention decision payload is immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def decide_ae_retention(
    placement_validation_records: Sequence[Mapping[str, Any]],
    config: AERetentionConfig,
) -> ImmutableJSONPayload:
    """Classify AE placement without accepting outer-test data or fields."""
    resolved = _validate_config(config)
    records = _validate_records(placement_validation_records, resolved)
    by_unit = {
        (seed, fraction): {
            str(record["method_family"]): record
            for record in records
            if int(record["seed"]) == seed
            and float(record["train_fraction"]) == fraction
        }
        for seed in resolved.seeds
        for fraction in resolved.fractions
    }

    unit_rows: list[dict[str, Any]] = []
    for fraction in resolved.fractions:
        for seed in resolved.seeds:
            family_rows = by_unit[(seed, fraction)]
            ae = family_rows[resolved.ae_family]
            ranked_controls = sorted(
                (
                    (
                        float(family_rows[family]["value"]),
                        family,
                        family_rows[family],
                    )
                    for family in resolved.control_families
                ),
                key=lambda item: (item[0], item[1]),
            )
            best_value, best_family, best_record = ranked_controls[0]
            delta = best_value - float(ae["value"])
            unit_rows.append(
                {
                    "evaluation_unit": str(ae["evaluation_unit"]),
                    "seed": seed,
                    "train_fraction": fraction,
                    "ae_family": resolved.ae_family,
                    "ae_rmse": float(ae["value"]),
                    "ae_selected_policy_hash": str(
                        ae["selected_policy_hash"]
                    ),
                    "ae_frozen_policy_hash": str(ae["frozen_policy_hash"]),
                    "strongest_control_family": best_family,
                    "strongest_control_rmse": best_value,
                    "strongest_control_selected_policy_hash": str(
                        best_record["selected_policy_hash"]
                    ),
                    "strongest_control_frozen_policy_hash": str(
                        best_record["frozen_policy_hash"]
                    ),
                    "delta_best_control_rmse_minus_ae_rmse": delta,
                    "practical_margin_win": (
                        delta >= resolved.practical_margin
                    ),
                    "practical_loss": (
                        delta < -resolved.no_practical_loss_threshold
                    ),
                    "control_rmse_rows": tuple(
                        {
                            "method_family": family,
                            "rmse": value,
                            "selected_policy_hash": str(
                                record["selected_policy_hash"]
                            ),
                            "frozen_policy_hash": str(
                                record["frozen_policy_hash"]
                            ),
                        }
                        for value, family, record in ranked_controls
                    ),
                }
            )

    fraction_rows: list[dict[str, Any]] = []
    for fraction in resolved.fractions:
        deltas = np.asarray(
            [
                row["delta_best_control_rmse_minus_ae_rmse"]
                for row in unit_rows
                if row["train_fraction"] == fraction
            ],
            dtype=float,
        )
        margin_wins = int(np.count_nonzero(deltas >= resolved.practical_margin))
        practical_losses = int(
            np.count_nonzero(
                deltas < -resolved.no_practical_loss_threshold
            )
        )
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        mean_met = mean_delta >= resolved.practical_margin
        median_met = median_delta >= resolved.practical_margin
        wins_met = (
            margin_wins >= resolved.required_margin_wins_per_fraction
        )
        no_loss_met = practical_losses == 0
        fraction_rows.append(
            {
                "train_fraction": fraction,
                "seed_count": len(resolved.seeds),
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "practical_margin_win_count": margin_wins,
                "required_margin_win_count": (
                    resolved.required_margin_wins_per_fraction
                ),
                "practical_loss_count": practical_losses,
                "mean_practical_margin_met": mean_met,
                "median_practical_margin_met": median_met,
                "required_margin_wins_met": wins_met,
                "no_practical_loss_met": no_loss_met,
                "all_fraction_criteria_met": (
                    mean_met and median_met and wins_met and no_loss_met
                ),
            }
        )

    bootstrap = _seed_cluster_bootstrap(unit_rows, resolved)
    criteria = {
        "all_records_complete_exactly_once": True,
        "all_fraction_mean_practical_margin_met": all(
            row["mean_practical_margin_met"] for row in fraction_rows
        ),
        "all_fraction_median_practical_margin_met": all(
            row["median_practical_margin_met"] for row in fraction_rows
        ),
        "all_fraction_required_margin_wins_met": all(
            row["required_margin_wins_met"] for row in fraction_rows
        ),
        "all_fraction_no_practical_loss_met": all(
            row["no_practical_loss_met"] for row in fraction_rows
        ),
        "pooled_bootstrap_lower_bound_at_least_practical_margin": bootstrap[
            "pooled_lower_bound_at_least_practical_margin"
        ],
    }
    criteria["primary_criteria_met"] = all(criteria.values())
    payload: dict[str, Any] = {
        "schema_version": AE_RETENTION_SCHEMA_VERSION,
        "placement": (
            "primary_benchmark"
            if criteria["primary_criteria_met"]
            else "secondary_ablation"
        ),
        "metric": "rmse",
        "data_role": "placement_validation",
        "ae_family": resolved.ae_family,
        "control_families": resolved.control_families,
        "configured_fractions": resolved.fractions,
        "configured_seeds": resolved.seeds,
        "practical_margin": resolved.practical_margin,
        "required_margin_wins_per_fraction": (
            resolved.required_margin_wins_per_fraction
        ),
        "no_practical_loss_threshold": (
            resolved.no_practical_loss_threshold
        ),
        "expected_record_count": len(resolved.seeds)
        * len(resolved.fractions)
        * (len(resolved.control_families) + 1),
        "unit_count": len(unit_rows),
        "per_unit_rows": tuple(unit_rows),
        "per_fraction_rows": tuple(fraction_rows),
        "bootstrap": bootstrap,
        "complete_criteria": criteria,
        "resolved_config": asdict(resolved),
        "input_records_hash": stable_hash(records),
        "outer_test_inputs_accepted": False,
        "outer_test_used_for_decision": False,
    }
    payload["decision_hash"] = stable_hash(payload)
    return _freeze_json(payload)


def _validate_config(config: AERetentionConfig) -> AERetentionConfig:
    if not isinstance(config, AERetentionConfig):
        raise TypeError("config must be an AERetentionConfig.")
    if (
        not isinstance(config.fractions, tuple)
        or not config.fractions
        or len(config.fractions) != len(set(config.fractions))
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 < float(value) <= 1
            for value in config.fractions
        )
    ):
        raise ValueError("fractions must be unique finite values in (0, 1].")
    if (
        not isinstance(config.seeds, tuple)
        or not config.seeds
        or len(config.seeds) != len(set(config.seeds))
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in config.seeds
        )
    ):
        raise ValueError("seeds must be a nonempty tuple of unique integers.")
    if len(config.seeds) < MINIMUM_RETENTION_SEEDS:
        raise ValueError(
            "The seed-cluster bootstrap is degenerate below "
            f"{MINIMUM_RETENTION_SEEDS} configured seeds; got {len(config.seeds)}."
        )
    if (
        not isinstance(config.ae_family, str)
        or not config.ae_family.strip()
        or config.ae_family in config.control_families
    ):
        raise ValueError("ae_family must be nonempty and distinct.")
    if config.control_families != AE_RETENTION_CONTROL_FAMILIES:
        raise ValueError(
            "control_families must equal the canonical six families in order."
        )
    for name in ("practical_margin", "no_practical_loss_threshold"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative.")
    if (
        not isinstance(config.required_margin_wins_per_fraction, int)
        or isinstance(config.required_margin_wins_per_fraction, bool)
        or not 0
        <= config.required_margin_wins_per_fraction
        <= len(config.seeds)
    ):
        raise ValueError(
            "required_margin_wins_per_fraction must be between zero and "
            "the configured seed count."
        )
    if (
        not isinstance(config.bootstrap_reps, int)
        or isinstance(config.bootstrap_reps, bool)
        or config.bootstrap_reps <= 0
    ):
        raise ValueError("bootstrap_reps must be a positive integer.")
    if (
        isinstance(config.bootstrap_alpha, bool)
        or not isinstance(config.bootstrap_alpha, (int, float))
        or not math.isfinite(float(config.bootstrap_alpha))
        or not 0 < float(config.bootstrap_alpha) < 1
    ):
        raise ValueError("bootstrap_alpha must be in (0, 1).")
    if not isinstance(config.bootstrap_seed, int) or isinstance(
        config.bootstrap_seed, bool
    ):
        raise ValueError("bootstrap_seed must be an integer.")
    return AERetentionConfig(
        fractions=tuple(float(value) for value in config.fractions),
        seeds=config.seeds,
        ae_family=config.ae_family.strip(),
        control_families=config.control_families,
        practical_margin=float(config.practical_margin),
        required_margin_wins_per_fraction=(
            config.required_margin_wins_per_fraction
        ),
        no_practical_loss_threshold=float(
            config.no_practical_loss_threshold
        ),
        bootstrap_reps=config.bootstrap_reps,
        bootstrap_alpha=float(config.bootstrap_alpha),
        bootstrap_seed=config.bootstrap_seed,
    )


def _validate_records(
    placement_validation_records: Sequence[Mapping[str, Any]],
    config: AERetentionConfig,
) -> list[dict[str, Any]]:
    if isinstance(placement_validation_records, (str, bytes)):
        raise TypeError("placement_validation_records must be a sequence.")
    expected_families = {config.ae_family, *config.control_families}
    expected_keys = {
        (seed, fraction, family)
        for seed in config.seeds
        for fraction in config.fractions
        for family in expected_families
    }
    normalized: list[dict[str, Any]] = []
    observed_keys: set[tuple[int, float, str]] = set()
    evaluation_units: dict[tuple[int, float], str] = {}
    for raw in placement_validation_records:
        if not isinstance(raw, Mapping):
            raise TypeError("Every placement metric record must be a mapping.")
        disallowed = set(raw) & _DISALLOWED_SELECTION_OR_TEST_FIELDS
        disallowed.update(
            key
            for key in raw
            if key.startswith("test_") or key.startswith("outer_test_")
        )
        if disallowed:
            raise ValueError(
                "Selection/test fields are forbidden in retention inputs: "
                f"{sorted(disallowed)}."
            )
        if set(raw) != _RECORD_FIELDS:
            raise ValueError(
                "Placement validation metric record fields mismatch."
            )
        if raw["metric"] != "rmse" or raw["data_role"] != (
            "placement_validation"
        ):
            raise ValueError(
                "Retention inputs must be placement-validation RMSE rows."
            )
        seed = raw["seed"]
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed not in config.seeds
        ):
            raise ValueError("Placement record has an unconfigured seed.")
        fraction = _configured_fraction(raw["train_fraction"], config)
        family = raw["method_family"]
        if not isinstance(family, str) or family not in expected_families:
            raise ValueError(
                "Placement record has an unconfigured method family."
            )
        value = raw["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("Placement RMSE values must be finite and non-negative.")
        evaluation_unit = raw["evaluation_unit"]
        if not isinstance(evaluation_unit, str) or not evaluation_unit.strip():
            raise ValueError("evaluation_unit must be a nonempty string.")
        for field in ("selected_policy_hash", "frozen_policy_hash"):
            if not _is_sha256(raw[field]):
                raise ValueError(f"{field} must be a SHA-256 digest.")
        key = (seed, fraction, family)
        if key in observed_keys:
            raise ValueError(
                "Placement RMSE rows must occur exactly once per unit/family."
            )
        unit_key = (seed, fraction)
        claimed_unit = evaluation_units.setdefault(
            unit_key, evaluation_unit.strip()
        )
        if claimed_unit != evaluation_unit.strip():
            raise ValueError(
                "Method families disagree on the placement evaluation unit."
            )
        observed_keys.add(key)
        normalized.append(
            {
                "evaluation_unit": evaluation_unit.strip(),
                "seed": seed,
                "train_fraction": fraction,
                "method_family": family,
                "metric": "rmse",
                "value": float(value),
                "selected_policy_hash": str(raw["selected_policy_hash"]),
                "frozen_policy_hash": str(raw["frozen_policy_hash"]),
                "data_role": "placement_validation",
            }
        )
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            "Placement RMSE matrix is incomplete: "
            f"missing={missing[:3]}, extra={extra[:3]}."
        )
    if len(set(evaluation_units.values())) != len(evaluation_units):
        raise ValueError("evaluation_unit must be unique per seed/fraction.")
    return sorted(
        normalized,
        key=lambda row: (
            row["train_fraction"],
            row["seed"],
            row["method_family"],
        ),
    )


def _configured_fraction(value: Any, config: AERetentionConfig) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("train_fraction must be finite.")
    matches = [
        fraction
        for fraction in config.fractions
        if math.isclose(
            float(value), fraction, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if len(matches) != 1:
        raise ValueError("Placement record has an unconfigured fraction.")
    return matches[0]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _seed_cluster_bootstrap(
    unit_rows: Sequence[Mapping[str, Any]],
    config: AERetentionConfig,
) -> dict[str, Any]:
    delta_by_seed_fraction = {
        (int(row["seed"]), float(row["train_fraction"])): float(
            row["delta_best_control_rmse_minus_ae_rmse"]
        )
        for row in unit_rows
    }
    generator = np.random.default_rng(config.bootstrap_seed)
    pooled_samples = np.empty(config.bootstrap_reps, dtype=float)
    fraction_samples = {
        fraction: np.empty(config.bootstrap_reps, dtype=float)
        for fraction in config.fractions
    }
    seeds = np.asarray(config.seeds, dtype=int)
    for replicate in range(config.bootstrap_reps):
        sampled_seeds = generator.choice(
            seeds, size=len(seeds), replace=True
        )
        pooled_deltas: list[float] = []
        for fraction in config.fractions:
            deltas = [
                delta_by_seed_fraction[(int(seed), fraction)]
                for seed in sampled_seeds
            ]
            fraction_samples[fraction][replicate] = float(np.mean(deltas))
            pooled_deltas.extend(deltas)
        pooled_samples[replicate] = float(np.mean(pooled_deltas))
    pooled_lower, pooled_upper = _interval(
        pooled_samples, config.bootstrap_alpha
    )
    per_fraction = []
    for fraction in config.fractions:
        lower, upper = _interval(
            fraction_samples[fraction], config.bootstrap_alpha
        )
        per_fraction.append(
            {
                "train_fraction": fraction,
                "mean_delta": float(
                    np.mean(
                        [
                            delta_by_seed_fraction[(seed, fraction)]
                            for seed in config.seeds
                        ]
                    )
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "lower_bound_above_zero": lower > 0,
                "lower_bound_at_least_practical_margin": (
                    lower >= config.practical_margin
                ),
            }
        )
    return {
        "method": "deterministic_seed_cluster_percentile_bootstrap",
        "cluster": "seed",
        "fractions_resampled_together": True,
        "repetitions": config.bootstrap_reps,
        "alpha": config.bootstrap_alpha,
        "random_seed": config.bootstrap_seed,
        "pooled_mean_delta": float(
            np.mean(
                [
                    delta_by_seed_fraction[(seed, fraction)]
                    for seed in config.seeds
                    for fraction in config.fractions
                ]
            )
        ),
        "pooled_ci_lower": pooled_lower,
        "pooled_ci_upper": pooled_upper,
        "practical_margin": config.practical_margin,
        "pooled_lower_bound_at_least_practical_margin": (
            pooled_lower >= config.practical_margin
        ),
        "per_fraction": tuple(per_fraction),
    }


def _interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lower, upper = np.quantile(
        samples,
        [alpha / 2, 1 - alpha / 2],
        method="linear",
    )
    return float(lower), float(upper)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ImmutableJSONPayload(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def decision_json(payload: ImmutableJSONPayload) -> str:
    """Serialize a decision without converting through mutable containers."""
    if not isinstance(payload, ImmutableJSONPayload):
        raise TypeError("payload must be an ImmutableJSONPayload.")
    return json.dumps(payload, sort_keys=True, allow_nan=False)
