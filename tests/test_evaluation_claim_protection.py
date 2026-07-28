"""Cross-run protection guarantees for outer-test evaluation claims."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bh_augmentation import nested_ood_final
from bh_augmentation.evaluation.evaluation_registry import (
    REPOSITORY_RELATIVE_REGISTRY_DIRECTORY,
    EvaluationAlreadyClaimedError,
    EvaluationIdentity,
    EvaluationRegistry,
    default_evaluation_registry_root,
    hash_metrics_payload,
    repository_root,
)

_DATASET_HASH = "a" * 64
_SPLIT_HASH = "b" * 64
_FROZEN_HASH = "c" * 64
_PREDICTION_HASH = "d" * 64
_METRICS = [{"evaluation_unit": "outer=group-a", "metric": "rmse", "value": 1.5}]
_NOW = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)


def _identity(unit: str = "outer=group-a") -> EvaluationIdentity:
    return EvaluationIdentity(
        dataset_hash=_DATASET_HASH,
        split_or_search_manifest_hash=_SPLIT_HASH,
        frozen_policy_hash=_FROZEN_HASH,
        evaluation_unit=unit,
    )


def test_changing_the_output_directory_does_not_release_a_claim(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    first_output = tmp_path / "corrected-run-a"
    second_output = tmp_path / "corrected-run-b"
    first_output.mkdir()
    identity = _identity()

    registry = EvaluationRegistry(registry_root)
    registry.reserve(identity, now=_NOW)
    (first_output / "final_test_metrics.csv").write_text("metric,value\nrmse,1.5\n")

    # A second run writes to a completely different result directory, and even
    # relocates the first bundle. The identity is artifact-location-independent,
    # so the claim survives untouched.
    shutil.move(str(first_output), str(second_output))
    assert not first_output.exists()

    with pytest.raises(EvaluationAlreadyClaimedError, match="already has a durable"):
        EvaluationRegistry(registry_root).reserve(identity, now=_NOW)
    assert registry.path_for(identity).is_file()
    assert len(registry.records()) == 1


def test_completed_identity_cannot_be_reserved_again(tmp_path: Path) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_NOW)
    registry.mark_prediction_complete(
        identity, prediction_hash=_PREDICTION_HASH, now=_NOW
    )
    record = registry.mark_metrics_complete(
        identity,
        metrics_payload=_METRICS,
        metrics_hash=hash_metrics_payload(_METRICS),
        now=_NOW,
    )
    assert record.is_complete

    with pytest.raises(EvaluationAlreadyClaimedError, match="'metrics_complete'"):
        registry.reserve(identity, now=_NOW)
    with pytest.raises(EvaluationAlreadyClaimedError):
        EvaluationRegistry(tmp_path / "registry").reserve(identity, now=_NOW)
    assert len(registry.completed_records()) == 1


def test_failed_or_uncertain_identity_cannot_be_reserved_again(tmp_path: Path) -> None:
    registry = EvaluationRegistry(tmp_path / "registry")
    identity = _identity()
    registry.reserve(identity, now=_NOW)
    registry.mark_failed_or_uncertain(identity, reason="worker crashed", now=_NOW)

    with pytest.raises(EvaluationAlreadyClaimedError, match="'failed_or_uncertain'"):
        registry.reserve(identity, now=_NOW)


def test_identical_science_from_a_different_bundle_maps_to_the_same_record(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"
    identity = _identity()
    same_identity = EvaluationIdentity(
        dataset_hash=_DATASET_HASH,
        split_or_search_manifest_hash=_SPLIT_HASH,
        frozen_policy_hash=_FROZEN_HASH,
        evaluation_unit="outer=group-a",
    )
    assert identity.registry_key == same_identity.registry_key

    registry = EvaluationRegistry(registry_root)
    registry.reserve(identity, now=_NOW)
    assert registry.path_for(same_identity) == registry.path_for(identity)
    with pytest.raises(EvaluationAlreadyClaimedError):
        registry.reserve(same_identity, now=_NOW)


def test_registry_root_is_working_directory_independent(tmp_path: Path) -> None:
    original_cwd = Path.cwd()
    expected = default_evaluation_registry_root()
    assert expected.is_absolute()
    assert expected == repository_root() / REPOSITORY_RELATIVE_REGISTRY_DIRECTORY
    try:
        os.chdir(tmp_path)
        assert default_evaluation_registry_root() == expected
    finally:
        os.chdir(original_cwd)


def test_nested_ood_final_registry_directory_is_anchored_to_the_checkout() -> None:
    directory = nested_ood_final.EVALUATION_REGISTRY_DIRECTORY
    assert directory.is_absolute()
    assert directory == default_evaluation_registry_root()


def test_registry_exposes_no_delete_or_release_operation() -> None:
    forbidden = ("delete", "remove", "release", "unclaim", "reset", "clear", "purge")
    public = [name for name in dir(EvaluationRegistry) if not name.startswith("_")]
    assert not [name for name in public if any(word in name.lower() for word in forbidden)]
    assert set(public) == {
        "completed_records",
        "inspect",
        "mark_failed_or_uncertain",
        "mark_metrics_complete",
        "mark_prediction_complete",
        "path_for",
        "records",
        "reserve",
    }
