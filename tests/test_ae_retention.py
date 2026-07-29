"""Focused tests for the validation-only AE retention decision."""

from __future__ import annotations

import inspect
import json

import pytest

from bh_augmentation.evaluation.ae_retention import (
    AE_RETENTION_CONTROL_FAMILIES,
    AERetentionConfig,
    ImmutableJSONPayload,
    decide_ae_retention,
    decision_json,
)
from bh_augmentation.utils.corrected_runs import stable_hash

AE_FAMILY = "redesigned_supervised_autoencoder"


def _config(**overrides: object) -> AERetentionConfig:
    values = {
        "fractions": (0.2, 1.0),
        "seeds": (0, 1, 2),
        "ae_family": AE_FAMILY,
        "control_families": AE_RETENTION_CONTROL_FAMILIES,
        "practical_margin": 0.5,
        "required_margin_wins_per_fraction": 2,
        "no_practical_loss_threshold": 0.5,
        "bootstrap_reps": 250,
        "bootstrap_alpha": 0.05,
        "bootstrap_seed": 1414,
    }
    values.update(overrides)
    return AERetentionConfig(**values)


def _records(
    *,
    ae_values: dict[tuple[int, float], float] | None = None,
) -> list[dict[str, object]]:
    ae_values = ae_values or {
        (seed, fraction): 8.8
        for seed in (0, 1, 2)
        for fraction in (0.2, 1.0)
    }
    rows = []
    families = (AE_FAMILY, *AE_RETENTION_CONTROL_FAMILIES)
    for fraction in (0.2, 1.0):
        for seed in (0, 1, 2):
            for index, family in enumerate(families):
                value = (
                    ae_values[(seed, fraction)]
                    if family == AE_FAMILY
                    else 10.0 + index / 10
                )
                identity = {
                    "unit": f"seed={seed}|fraction={fraction}",
                    "family": family,
                }
                rows.append(
                    {
                        "evaluation_unit": identity["unit"],
                        "seed": seed,
                        "train_fraction": fraction,
                        "method_family": family,
                        "metric": "rmse",
                        "value": value,
                        "selected_policy_hash": stable_hash(
                            {**identity, "kind": "selected"}
                        ),
                        "frozen_policy_hash": stable_hash(
                            {**identity, "kind": "frozen"}
                        ),
                        "data_role": "placement_validation",
                    }
                )
    return rows


def test_primary_decision_replays_strongest_control_and_delta() -> None:
    decision = decide_ae_retention(_records(), _config())

    assert decision["placement"] == "primary_benchmark"
    assert decision["unit_count"] == 6
    assert decision["expected_record_count"] == 42
    assert decision["complete_criteria"]["primary_criteria_met"] is True
    assert decision["outer_test_inputs_accepted"] is False
    assert decision["outer_test_used_for_decision"] is False
    for row in decision["per_unit_rows"]:
        assert row["strongest_control_family"] == "truncated_svd"
        assert row["strongest_control_rmse"] == pytest.approx(10.1)
        assert row[
            "delta_best_control_rmse_minus_ae_rmse"
        ] == pytest.approx(1.3)
        assert len(row["control_rmse_rows"]) == 6


def test_payload_is_immutable_and_directly_json_serializable() -> None:
    decision = decide_ae_retention(_records(), _config())

    assert isinstance(decision, ImmutableJSONPayload)
    assert json.loads(decision_json(decision))["placement"] == "primary_benchmark"
    assert json.loads(json.dumps(decision))["placement"] == "primary_benchmark"
    with pytest.raises(TypeError, match="immutable"):
        decision["placement"] = "secondary_ablation"
    with pytest.raises(TypeError, match="immutable"):
        decision["complete_criteria"]["primary_criteria_met"] = False


def test_bootstrap_and_decision_hash_are_deterministic() -> None:
    first = decide_ae_retention(_records(), _config())
    second = decide_ae_retention(list(reversed(_records())), _config())

    assert first == second
    assert first["decision_hash"] == second["decision_hash"]
    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["fractions_resampled_together"] is True
    assert first["bootstrap"]["pooled_ci_lower"] > 0


def test_failed_fraction_or_practical_loss_makes_ae_secondary() -> None:
    values = {
        (seed, fraction): 8.8
        for seed in (0, 1, 2)
        for fraction in (0.2, 1.0)
    }
    values[(0, 1.0)] = 11.0
    decision = decide_ae_retention(
        _records(ae_values=values),
        _config(required_margin_wins_per_fraction=3),
    )
    full_fraction = next(
        row
        for row in decision["per_fraction_rows"]
        if row["train_fraction"] == 1.0
    )

    assert decision["placement"] == "secondary_ablation"
    assert full_fraction["practical_loss_count"] == 1
    assert full_fraction["no_practical_loss_met"] is False
    assert full_fraction["required_margin_wins_met"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_rows_must_be_complete_and_exactly_once(mutation: str) -> None:
    rows = _records()
    if mutation == "missing":
        rows.pop()
        match = "incomplete"
    else:
        rows.append(dict(rows[0]))
        match = "exactly once"

    with pytest.raises(ValueError, match=match):
        decide_ae_retention(rows, _config())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("value", float("nan"), "finite"),
        ("value", -1.0, "finite"),
        ("metric", "mae", "RMSE"),
        ("data_role", "outer_test", "placement-validation"),
    ],
)
def test_only_finite_validation_rmse_is_accepted(
    field: str,
    value: object,
    match: str,
) -> None:
    rows = _records()
    rows[0][field] = value

    with pytest.raises(ValueError, match=match):
        decide_ae_retention(rows, _config())


@pytest.mark.parametrize(
    "field",
    ["test_evaluated", "test_used_for_selection", "selected_within_family"],
)
def test_selection_and_test_fields_are_rejected(field: str) -> None:
    rows = _records()
    rows[0][field] = False

    with pytest.raises(ValueError, match="Selection/test fields"):
        decide_ae_retention(rows, _config())


def test_pooled_bootstrap_conjunct_requires_the_practical_margin() -> None:
    # Every delta is +0.3: strictly above zero, but below the 0.5 margin the
    # mean/median conjuncts use. The bootstrap conjunct must not be vacuous.
    values = {
        (seed, fraction): 10.1 - 0.3
        for seed in (0, 1, 2)
        for fraction in (0.2, 1.0)
    }
    decision = decide_ae_retention(_records(ae_values=values), _config())
    bootstrap = decision["bootstrap"]

    assert bootstrap["pooled_ci_lower"] == pytest.approx(0.3)
    assert "pooled_lower_bound_above_zero" not in bootstrap
    assert bootstrap["practical_margin"] == 0.5
    assert bootstrap["pooled_lower_bound_at_least_practical_margin"] is False
    assert (
        decision["complete_criteria"][
            "pooled_bootstrap_lower_bound_at_least_practical_margin"
        ]
        is False
    )
    assert decision["placement"] == "secondary_ablation"


def test_pooled_bootstrap_conjunct_passes_at_the_practical_margin() -> None:
    values = {
        (seed, fraction): 10.1 - 1.3
        for seed in (0, 1, 2)
        for fraction in (0.2, 1.0)
    }
    decision = decide_ae_retention(_records(ae_values=values), _config())

    assert decision["bootstrap"][
        "pooled_lower_bound_at_least_practical_margin"
    ] is True
    assert decision["placement"] == "primary_benchmark"


@pytest.mark.parametrize("seeds", [(0,), (0, 1)])
def test_cluster_bootstrap_requires_at_least_three_seeds(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        decide_ae_retention(
            _records(),
            _config(seeds=seeds, required_margin_wins_per_fraction=1),
        )


def test_control_family_contract_is_exact() -> None:
    wrong_controls = (
        *AE_RETENTION_CONTROL_FAMILIES[:-1],
        "post_hoc_oracle",
    )
    with pytest.raises(ValueError, match="canonical six"):
        decide_ae_retention(
            _records(),
            _config(control_families=wrong_controls),
        )


def test_api_has_no_outer_test_inputs() -> None:
    parameters = inspect.signature(decide_ae_retention).parameters
    assert set(parameters) == {"placement_validation_records", "config"}
    assert not any("test" in parameter for parameter in parameters)
