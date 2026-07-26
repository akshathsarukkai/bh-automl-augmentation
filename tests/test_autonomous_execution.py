"""Tests for autonomous phase-state validation and resume behavior."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from bh_augmentation.autonomous_execution import (
    first_resumable_phase,
    load_and_validate_state,
    verify_passed_phases,
)


def test_repository_state_has_all_required_phases() -> None:
    state = load_and_validate_state("results/autonomous_execution/state.json")

    assert len(state["phases"]) == 18
    expected = next(
        phase["phase_number"]
        for phase in state["phases"]
        if phase["status"] != "passed"
    )
    assert first_resumable_phase(state) == expected


def test_missing_passed_artifact_reopens_phase(tmp_path: Path) -> None:
    state = load_and_validate_state("results/autonomous_execution/state.json")
    fixture = copy.deepcopy(state)
    phase = fixture["phases"][0]
    phase.update(
        {
            "status": "passed",
            "completed_at": "2026-07-26T00:00:00Z",
            "git_commit_at_end": "abc123",
            "targeted_tests": {"status": "passed"},
            "full_suite_result": {"status": "passed"},
            "smoke_test_result": {"status": "passed"},
            "output_paths": [{"path": "missing.csv", "sha256": "deadbeef"}],
        }
    )
    repository = tmp_path
    canonical = repository / "data/processed"
    splits = repository / "results/corrected_canonical_splits"
    canonical.mkdir(parents=True)
    splits.mkdir(parents=True)
    (canonical / "bh_canonical_roles_v1.csv").write_text("row\n1\n")
    (splits / "outer_split_assignments.csv").write_text("row\n1\n")
    (splits / "low_data_subset_assignments.csv").write_text("row\n1\n")
    (splits / "split_manifest.json").write_text(json.dumps({"split_hash": "fixture"}))

    issues = verify_passed_phases(fixture, repository=repository)

    assert issues
    assert fixture["phases"][0]["status"] == "not_started"
    assert fixture["phases"][0]["git_commit_at_end"] is None
