"""Validation and resume helpers for the autonomous 18-phase roadmap."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_PHASE_STATUSES = {
    "not_started",
    "in_progress",
    "testing",
    "repairing",
    "passed",
    "blocked",
    "skipped_due_to_dependency",
}
REQUIRED_PHASE_FIELDS = {
    "phase_number",
    "phase_name",
    "status",
    "started_at",
    "completed_at",
    "git_commit_at_start",
    "git_commit_at_end",
    "targeted_tests",
    "full_suite_result",
    "smoke_test_result",
    "output_paths",
    "warnings",
    "blockers",
    "dependencies",
}


def load_and_validate_state(path: str | Path) -> dict[str, Any]:
    """Load the state ledger and reject malformed phase records."""
    state = json.loads(Path(path).read_text())
    phases = state.get("phases")
    if not isinstance(phases, list) or len(phases) != 18:
        raise ValueError("Autonomous state must contain exactly 18 phases.")
    if [phase.get("phase_number") for phase in phases] != list(range(1, 19)):
        raise ValueError("Autonomous phases must be numbered sequentially from 1 to 18.")
    for phase in phases:
        missing = sorted(REQUIRED_PHASE_FIELDS - set(phase))
        if missing:
            raise ValueError(
                f"Phase {phase.get('phase_number')} is missing fields: {', '.join(missing)}."
            )
        if phase["status"] not in VALID_PHASE_STATUSES:
            raise ValueError(
                f"Phase {phase['phase_number']} has invalid status {phase['status']!r}."
            )
    return state


def current_dependency_hashes(repository: str | Path) -> dict[str, str]:
    """Hash the immutable Batch 3 inputs used by downstream scientific phases."""
    root = Path(repository)
    manifest_path = root / "results/corrected_canonical_splits/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "canonical_dataset_sha256": sha256_file(
            root / "data/processed/bh_canonical_roles_v1.csv"
        ),
        "split_manifest_sha256": sha256_file(manifest_path),
        "outer_assignments_sha256": sha256_file(
            root / "results/corrected_canonical_splits/outer_split_assignments.csv"
        ),
        "low_data_assignments_sha256": sha256_file(
            root / "results/corrected_canonical_splits/low_data_subset_assignments.csv"
        ),
        "split_aggregate_hash": str(manifest["split_hash"]),
    }


def verify_passed_phases(
    state: dict[str, Any],
    *,
    repository: str | Path,
) -> list[dict[str, Any]]:
    """Reopen the first invalid passed phase and every dependent later pass."""
    root = Path(repository)
    issues: list[dict[str, Any]] = []
    expected_dependencies = state.get("dependency_hashes", {})
    actual_dependencies = current_dependency_hashes(root)
    dependency_changed = expected_dependencies != actual_dependencies
    first_invalid: int | None = None

    for phase in state["phases"]:
        if phase["status"] != "passed":
            continue
        phase_issues = _passed_phase_issues(phase, root)
        if dependency_changed:
            phase_issues.append("canonical dependency hashes changed")
        if phase_issues and first_invalid is None:
            first_invalid = int(phase["phase_number"])
        if phase_issues:
            issues.append(
                {
                    "phase_number": int(phase["phase_number"]),
                    "issues": phase_issues,
                }
            )

    if first_invalid is not None:
        for phase in state["phases"]:
            if phase["status"] == "passed" and phase["phase_number"] >= first_invalid:
                phase["status"] = "not_started"
                phase["completed_at"] = None
                phase["git_commit_at_end"] = None
                phase["warnings"].append(
                    f"Automatically reopened after verification failure at phase {first_invalid}."
                )
    return issues


def first_resumable_phase(state: Mapping[str, Any]) -> int | None:
    """Return the first phase requiring work, excluding externally blocked phases."""
    for phase in state["phases"]:
        if phase["status"] in {"not_started", "in_progress", "testing", "repairing"}:
            return int(phase["phase_number"])
    return None


def verify_and_resume(
    state_path: str | Path,
    *,
    repository: str | Path,
    events_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int | None]:
    """Validate passed artifacts, persist reopenings, and identify resume position."""
    path = Path(state_path)
    state = load_and_validate_state(path)
    issues = verify_passed_phases(state, repository=repository)
    resume_phase = first_resumable_phase(state)
    if issues:
        timestamp = datetime.now(timezone.utc).isoformat()
        state["updated_at"] = timestamp
        _atomic_write_json(path, state)
        if events_path is not None:
            event = {
                "event": "passed_phase_reopened",
                "issues": issues,
                "resume_phase": resume_phase,
                "timestamp": timestamp,
            }
            with Path(events_path).open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
    return state, issues, resume_phase


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passed_phase_issues(phase: Mapping[str, Any], root: Path) -> list[str]:
    issues: list[str] = []
    if not phase["completed_at"] or not phase["git_commit_at_end"]:
        issues.append("completion timestamp or commit is missing")
    for field in ("targeted_tests", "full_suite_result", "smoke_test_result"):
        result = phase[field]
        if not isinstance(result, Mapping) or result.get("status") != "passed":
            issues.append(f"{field} is not a verified passing result")
    if not phase["output_paths"]:
        issues.append("no required output artifacts are recorded")
    for artifact in phase["output_paths"]:
        if isinstance(artifact, str):
            relative_path = artifact
            expected_hash = None
        elif isinstance(artifact, Mapping):
            relative_path = artifact.get("path")
            expected_hash = artifact.get("sha256")
        else:
            issues.append(f"invalid output artifact record: {artifact!r}")
            continue
        if not relative_path:
            issues.append("output artifact record has no path")
            continue
        output = root / str(relative_path)
        if not output.exists():
            issues.append(f"required output is missing: {relative_path}")
        elif expected_hash and output.is_file() and sha256_file(output) != expected_hash:
            issues.append(f"required output hash changed: {relative_path}")
    return issues


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify autonomous state and show resume phase.")
    parser.add_argument(
        "--state",
        default="results/autonomous_execution/state.json",
    )
    parser.add_argument(
        "--events",
        default="results/autonomous_execution/events.jsonl",
    )
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()
    _, issues, resume_phase = verify_and_resume(
        args.state,
        repository=args.repository,
        events_path=args.events,
    )
    print(f"Passed-phase verification issues: {len(issues)}")
    print(f"Resume phase: {resume_phase if resume_phase is not None else 'none'}")


if __name__ == "__main__":
    main()
