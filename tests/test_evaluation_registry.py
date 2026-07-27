"""Tests for the repository-global outer-evaluation registry."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bh_augmentation.evaluation.evaluation_registry import (
    EvaluationAlreadyClaimedError,
    EvaluationIdentity,
    EvaluationRegistry,
    EvaluationRegistryCorruptionError,
    EvaluationTransitionError,
    hash_metrics_payload,
)
from bh_augmentation.utils.corrected_runs import stable_hash

_DATASET_HASH = "a" * 64
_SEARCH_HASH = "b" * 64
_FROZEN_HASH = "c" * 64
_PREDICTION_HASH = "d" * 64
_METRICS_PAYLOAD = [
    {
        "evaluation_unit": "target=product|outer=group-a",
        "metric": "rmse",
        "value": 1.25,
        "n_test": 4,
    },
    {
        "evaluation_unit": "target=product|outer=group-a",
        "metric": "mae",
        "value": 1.0,
        "n_test": 4,
    },
]
_METRICS_HASH = hash_metrics_payload(_METRICS_PAYLOAD)
_START = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _identity(unit: str = "target=product|outer=group-a") -> EvaluationIdentity:
    return EvaluationIdentity(
        dataset_hash=_DATASET_HASH,
        split_or_search_manifest_hash=_SEARCH_HASH,
        frozen_policy_hash=_FROZEN_HASH,
        evaluation_unit=unit,
    )


def test_identity_and_claim_are_independent_of_search_bundle_location(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "repository-global-registry"
    original_bundle = tmp_path / "original-search-bundle"
    copied_bundle = tmp_path / "copied-search-bundle"
    original_bundle.mkdir()
    copied_bundle.mkdir()
    original_identity = _identity()
    copied_identity = _identity()
    registry = EvaluationRegistry(registry_root)

    first = registry.reserve(original_identity, now=_START)

    assert original_bundle != copied_bundle
    assert original_identity.registry_key == copied_identity.registry_key
    assert first.identity == copied_identity
    with pytest.raises(EvaluationAlreadyClaimedError, match="durable"):
        registry.reserve(copied_identity, now=_START + timedelta(seconds=1))


def test_exclusive_reservation_allows_exactly_one_concurrent_winner(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()

    def reserve() -> str:
        try:
            return registry.reserve(identity, now=_START).status
        except EvaluationAlreadyClaimedError:
            return "already_claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: reserve(), range(2)))

    assert sorted(outcomes) == ["already_claimed", "reserved"]
    assert registry.inspect(identity).status == "reserved"  # type: ignore[union-attr]


def test_prediction_and_metrics_completion_are_atomic_and_discoverable(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)

    prediction = registry.mark_prediction_complete(
        identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(minutes=1),
    )
    complete = registry.mark_metrics_complete(
        identity,
        metrics_payload=_METRICS_PAYLOAD,
        metrics_hash=_METRICS_HASH,
        now=_START + timedelta(minutes=2),
    )

    assert prediction.status == "prediction_complete"
    assert prediction.prediction_hash == _PREDICTION_HASH
    assert complete.status == "metrics_complete"
    assert complete.metrics_hash == _METRICS_HASH
    assert complete.metrics_payload == _METRICS_PAYLOAD
    assert registry.completed_records() == (complete,)
    with pytest.raises(EvaluationAlreadyClaimedError):
        registry.reserve(identity, now=_START + timedelta(minutes=3))


def test_stale_reservation_becomes_permanently_failed_and_cannot_run(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)

    failed = registry.inspect(
        identity,
        stale_after=timedelta(minutes=10),
        now=_START + timedelta(minutes=11),
    )

    assert failed is not None
    assert failed.status == "failed_or_uncertain"
    assert "must never be reevaluated" in failed.failure_reason  # type: ignore[operator]
    assert registry.completed_records() == ()
    with pytest.raises(EvaluationAlreadyClaimedError):
        registry.reserve(identity, now=_START + timedelta(minutes=12))
    with pytest.raises(EvaluationTransitionError):
        registry.mark_prediction_complete(
            identity,
            prediction_hash=_PREDICTION_HASH,
        )


def test_stale_prediction_completion_becomes_permanently_uncertain(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)
    registry.mark_prediction_complete(
        identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(minutes=1),
    )

    failed = registry.inspect(
        identity,
        stale_after=timedelta(minutes=10),
        now=_START + timedelta(minutes=12),
    )

    assert failed is not None
    assert failed.status == "failed_or_uncertain"
    assert failed.prediction_hash == _PREDICTION_HASH
    with pytest.raises(EvaluationTransitionError):
        registry.mark_metrics_complete(
            identity,
            metrics_payload=_METRICS_PAYLOAD,
            metrics_hash=_METRICS_HASH,
        )


def test_resume_discovery_separates_complete_active_and_uncertain_units(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    complete_identity = _identity("complete-unit")
    active_identity = _identity("active-unit")
    stale_identity = _identity("stale-unit")
    registry.reserve(complete_identity, now=_START)
    registry.mark_prediction_complete(
        complete_identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(seconds=1),
    )
    registry.mark_metrics_complete(
        complete_identity,
        metrics_payload=_METRICS_PAYLOAD,
        metrics_hash=_METRICS_HASH,
        now=_START + timedelta(seconds=2),
    )
    registry.reserve(active_identity, now=_START + timedelta(minutes=9))
    registry.reserve(stale_identity, now=_START)

    records = registry.records(
        stale_after=timedelta(minutes=10),
        now=_START + timedelta(minutes=11),
    )
    by_unit = {record.identity.evaluation_unit: record.status for record in records}

    assert by_unit == {
        "complete-unit": "metrics_complete",
        "active-unit": "reserved",
        "stale-unit": "failed_or_uncertain",
    }
    assert {
        record.identity.evaluation_unit for record in registry.completed_records()
    } == {"complete-unit"}


def test_invalid_state_transitions_cannot_claim_completion(tmp_path: Path) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)

    with pytest.raises(EvaluationTransitionError, match="prediction_complete"):
        registry.mark_metrics_complete(
            identity,
            metrics_payload=_METRICS_PAYLOAD,
            metrics_hash=_METRICS_HASH,
        )

    failed = registry.mark_failed_or_uncertain(
        identity,
        reason="Prediction call raised after test materialization.",
        now=_START + timedelta(seconds=1),
    )
    assert failed.status == "failed_or_uncertain"
    with pytest.raises(EvaluationTransitionError):
        registry.mark_metrics_complete(
            identity,
            metrics_payload=_METRICS_PAYLOAD,
            metrics_hash=_METRICS_HASH,
        )


def test_fresh_caller_reconstructs_exact_rows_from_completed_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "registry"
    identity = _identity()
    writer = EvaluationRegistry(root)
    writer.reserve(identity, now=_START)
    writer.mark_prediction_complete(
        identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(seconds=1),
    )
    writer.mark_metrics_complete(
        identity,
        metrics_payload=_METRICS_PAYLOAD,
        now=_START + timedelta(seconds=2),
    )

    fresh_reader = EvaluationRegistry(root)
    completed = fresh_reader.completed_records()

    assert len(completed) == 1
    assert completed[0].metrics_payload == _METRICS_PAYLOAD
    assert completed[0].metrics_hash == hash_metrics_payload(
        completed[0].metrics_payload  # type: ignore[arg-type]
    )


def test_consistently_rehashed_tampered_metrics_payload_is_corrupt(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)
    registry.mark_prediction_complete(
        identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(seconds=1),
    )
    registry.mark_metrics_complete(
        identity,
        metrics_payload=_METRICS_PAYLOAD,
        now=_START + timedelta(seconds=2),
    )
    path = registry.path_for(identity)
    document = json.loads(path.read_text())
    document["metrics_payload"][0]["value"] = 999.0
    unhashed = dict(document)
    unhashed.pop("record_payload_hash")
    document["record_payload_hash"] = stable_hash(unhashed)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        EvaluationRegistryCorruptionError,
        match="record state is invalid",
    ):
        registry.inspect(identity)


def test_supplied_metrics_hash_must_match_canonical_payload(tmp_path: Path) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)
    registry.mark_prediction_complete(
        identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="does not match"):
        registry.mark_metrics_complete(
            identity,
            metrics_payload=_METRICS_PAYLOAD,
            metrics_hash="f" * 64,
        )

    assert registry.inspect(identity).status == "prediction_complete"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "payload",
    [
        [{"metric": "rmse", "value": float("nan")}],
        [{"metric": "rmse", "value": float("inf")}],
        [{"metric": "rmse", "value": {"not_json": {1, 2}}}],
    ],
)
def test_nonfinite_or_nonserializable_metrics_payload_is_rejected(
    tmp_path: Path,
    payload: list[dict[str, object]],
) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_START)
    registry.mark_prediction_complete(
        identity,
        prediction_hash=_PREDICTION_HASH,
        now=_START + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="finite and JSON serializable"):
        registry.mark_metrics_complete(identity, metrics_payload=payload)

    assert registry.inspect(identity).status == "prediction_complete"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "field",
    [
        "dataset_hash",
        "split_or_search_manifest_hash",
        "frozen_policy_hash",
    ],
)
def test_identity_rejects_non_sha256_hashes(field: str) -> None:
    values = {
        "dataset_hash": _DATASET_HASH,
        "split_or_search_manifest_hash": _SEARCH_HASH,
        "frozen_policy_hash": _FROZEN_HASH,
        "evaluation_unit": "unit",
    }
    values[field] = "not-a-sha256"

    with pytest.raises(ValueError, match="SHA-256"):
        EvaluationIdentity(**values)
