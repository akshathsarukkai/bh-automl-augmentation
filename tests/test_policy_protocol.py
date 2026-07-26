"""Tests for validation-only search and frozen final-evaluation protocol."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pytest

from bh_augmentation.evaluation.policy_protocol import (
    FinalEvaluationInputs,
    PolicySearchInputs,
    ResolvedPolicy,
    ScientificBinding,
    evaluate_frozen_policy_once,
    freeze_policy,
    load_frozen_policy,
    run_policy_search,
)
from bh_augmentation.utils.corrected_runs import stable_hash


def _binding(**changes: str) -> ScientificBinding:
    values = {
        "dataset_hash": stable_hash("dataset"),
        "split_hash": stable_hash("split"),
        "split_aggregate_hash": stable_hash("aggregate"),
        "per_seed_split_hash": stable_hash("seed"),
        "source_id_split_hash": stable_hash("source IDs"),
        "canonicalization_version": "canonical-v1",
        "split_schema_version": "split-v1",
        "feature_metadata_hash": stable_hash("features"),
        "config_hash": stable_hash("config"),
        "commit_hash": "a" * 40,
    }
    values.update(changes)
    return ScientificBinding(**values)


def _search_result():
    inputs = PolicySearchInputs(
        train={"rows": ["train"]},
        validation={"rows": ["validation"]},
        binding=_binding(),
    )
    policies = [
        ResolvedPolicy("later", {"weight": 1.0}),
        ResolvedPolicy("winner", {"weight": 0.5}),
    ]

    def evaluator(search_inputs, policy):
        assert search_inputs.train == {"rows": ["train"]}
        assert search_inputs.validation == {"rows": ["validation"]}
        value = {"later": 4.0, "winner": 3.0}[policy.policy_id]
        return [{"split": "valid", "metric": "rmse", "value": value}]

    return run_policy_search(inputs, policies, evaluator)


def _frozen():
    return freeze_policy(
        _search_result(),
        training_protocol={"refit_roles": ["outer_train"]},
        search_manifest_hash=stable_hash("search manifest"),
    )


def test_search_inputs_have_no_outer_test_field_and_search_never_calls_outer_spy() -> None:
    assert {field.name for field in fields(PolicySearchInputs)} == {
        "train",
        "validation",
        "binding",
    }
    outer_calls = []
    result = _search_result()

    assert outer_calls == []
    assert result.selected_policy.policy_id == "winner"
    assert set(result.search_metrics["split"]) == {"valid"}
    assert result.search_metrics.groupby("policy_id")["selected_policy"].first().sum() == 1


def test_selection_ties_use_stable_policy_hash_then_id() -> None:
    inputs = PolicySearchInputs("train", "validation", _binding())
    policies = [
        ResolvedPolicy("z", {"weight": 1}),
        ResolvedPolicy("a", {"weight": 2}),
    ]

    def evaluator(_, __):
        return {"split": "valid", "metric": "rmse", "value": 2.0}

    result = run_policy_search(inputs, policies, evaluator)
    expected = min(policies, key=lambda policy: (policy.policy_hash, policy.policy_id))
    assert result.selected_policy == expected


def test_search_rejects_test_rows_and_nonfinite_metrics() -> None:
    inputs = PolicySearchInputs("train", "validation", _binding())
    policy = ResolvedPolicy("p", {"weight": 1})

    with pytest.raises(ValueError, match="non-validation"):
        run_policy_search(
            inputs,
            [policy],
            lambda *_: {"split": "test", "metric": "rmse", "value": 1.0},
        )
    with pytest.raises(ValueError, match="must be finite"):
        run_policy_search(
            inputs,
            [policy],
            lambda *_: {"split": "valid", "metric": "rmse", "value": float("nan")},
        )


def test_resolved_policy_rejects_nonfinite_configuration() -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        ResolvedPolicy("nonfinite", {"weight": float("nan")})


def test_frozen_envelope_round_trip_and_strict_schema(tmp_path) -> None:
    frozen = _frozen()
    path = frozen.write_json(tmp_path / "frozen_policy.json")
    loaded = load_frozen_policy(path, expected_binding=frozen.binding)

    assert loaded.to_dict() == frozen.to_dict()
    assert loaded.frozen_policy_hash == stable_hash(
        {key: value for key, value in loaded.to_dict().items() if key != "frozen_policy_hash"}
    )

    payload = frozen.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="schema mismatch"):
        load_frozen_policy(payload)


def test_unfrozen_and_policy_hash_tampering_are_rejected() -> None:
    payload = _frozen().to_dict()
    payload["status"] = "selected_not_frozen"
    with pytest.raises(ValueError, match="unfrozen"):
        load_frozen_policy(payload)

    payload = _frozen().to_dict()
    payload["resolved_policy"]["weight"] = 999
    with pytest.raises(ValueError, match="resolved policy hash mismatch"):
        load_frozen_policy(payload)


def test_envelope_hash_and_binding_mismatches_are_rejected() -> None:
    frozen = _frozen()
    payload = frozen.to_dict()
    payload["training_protocol"]["refit_roles"] = ["outer_train", "validation"]
    with pytest.raises(ValueError, match="envelope hash mismatch"):
        load_frozen_policy(payload)

    with pytest.raises(ValueError, match="scientific binding mismatch"):
        load_frozen_policy(
            frozen.to_dict(),
            expected_binding=_binding(dataset_hash=stable_hash("other dataset")),
        )


def test_final_outer_callback_runs_once_and_duplicate_unit_is_rejected() -> None:
    frozen = _frozen()
    inputs = FinalEvaluationInputs(
        refit="refit rows",
        outer_test="outer test rows",
        binding=frozen.binding,
        evaluation_unit="seed=0|fraction=0.1|method=typed",
    )
    calls = []
    evaluated_units: set[str] = set()

    def outer_evaluator(final_inputs, selected):
        calls.append((final_inputs.outer_test, selected.resolved_policy.policy_id))
        return {"split": "test", "metric": "rmse", "value": 2.5}

    result = evaluate_frozen_policy_once(
        inputs,
        frozen,
        outer_evaluator,
        evaluated_units=evaluated_units,
    )
    assert calls == [("outer test rows", "winner")]
    assert result.metrics.loc[0, "evaluation_unit"] == inputs.evaluation_unit
    assert result.metrics.loc[0, "frozen_policy_hash"] == frozen.frozen_policy_hash

    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_frozen_policy_once(
            inputs,
            frozen,
            outer_evaluator,
            evaluated_units=evaluated_units,
        )
    assert len(calls) == 1


def test_final_binding_is_checked_before_outer_callback() -> None:
    frozen = _frozen()
    calls = []
    inputs = FinalEvaluationInputs(
        "refit",
        "test",
        _binding(config_hash=stable_hash("changed config")),
        "unit",
    )

    with pytest.raises(ValueError, match="scientific binding mismatch"):
        evaluate_frozen_policy_once(
            inputs,
            frozen,
            lambda *_: calls.append("called"),
            evaluated_units=set(),
        )
    assert calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_final_rejects_nonfinite_metrics(value: float) -> None:
    frozen = _frozen()
    inputs = FinalEvaluationInputs("refit", "test", frozen.binding, "unit")

    with pytest.raises(ValueError, match="must be finite"):
        evaluate_frozen_policy_once(
            inputs,
            frozen,
            lambda *_: pd.DataFrame([{"split": "test", "metric": "rmse", "value": value}]),
            evaluated_units=set(),
        )
