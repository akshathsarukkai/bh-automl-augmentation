"""Bounded search + final-evaluation smoke on the SYNTHETIC Suzuki fixture.

Everything here writes to pytest's ``tmp_path``; nothing is written under
``results/``. The data is the synthetic fixture from ``tests/suzuki_fixture.py``
and is not an experimental measurement, so no empirical Suzuki-Miyaura claim is
made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.evaluation.policy_protocol import load_frozen_policy
from bh_augmentation.run_external_family_validation import run_external_family_validation
from suzuki_fixture import suzuki_fixture_config, write_suzuki_fixture_csv


class _FitRecorder:
    """Record which source rows each fitting phase actually saw."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, tuple[str, ...]]] = []

    def __call__(self, phase: str, unit: str, source_row_ids: list[str]) -> None:
        self.events.append((phase, unit, tuple(source_row_ids)))

    def rows(self, phase: str) -> set[str]:
        return {
            row_id
            for event_phase, _, ids in self.events
            if event_phase == phase
            for row_id in ids
        }


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("suzuki_smoke")
    dataset_path = write_suzuki_fixture_csv(root)
    output_directory = root / "external_validation_output"
    config = suzuki_fixture_config(dataset_path, output_directory, n_bits=64)
    recorder = _FitRecorder()
    paths = run_external_family_validation(
        config,
        config_path=root / "suzuki_fixture.yaml",
        fit_row_observer=recorder,
    )
    manifest = json.loads(Path(paths["manifest"]).read_text())
    return {
        "root": root,
        "paths": paths,
        "manifest": manifest,
        "recorder": recorder,
        "config": config,
        "dataset_path": dataset_path,
    }


def test_smoke_writes_only_into_the_temporary_directory(smoke_run) -> None:
    root = smoke_run["root"]
    for path in smoke_run["paths"].values():
        assert Path(path).is_file()
        assert root in Path(path).resolve().parents
    repository = Path(__file__).resolve().parents[1]
    assert repository not in root.resolve().parents


def test_search_and_final_phases_both_ran_and_produced_metrics(smoke_run) -> None:
    search = pd.read_csv(smoke_run["paths"]["search_metrics"])
    final = pd.read_csv(smoke_run["paths"]["final_metrics"])
    assert set(search["split"]) == {"valid"}
    assert set(final["split"]) == {"test"}
    assert search["policy_id"].nunique() >= 2
    assert search.groupby("evaluation_unit")["selected_policy"].any().all()
    assert final["value"].notna().all()


def test_no_validation_or_test_row_ever_entered_fitting(smoke_run) -> None:
    manifest = smoke_run["manifest"]
    recorder = smoke_run["recorder"]
    canonical_ids = _partition_source_ids(smoke_run)

    fitted = recorder.rows("search") | recorder.rows("final")
    assert fitted
    assert fitted <= canonical_ids["train"]
    assert not fitted & canonical_ids["valid"]
    assert not fitted & canonical_ids["test"]
    assert manifest["n_test_evaluations"] == len(manifest["evaluation_units"])


def test_policies_are_frozen_and_verifiable_before_any_test_evaluation(smoke_run) -> None:
    recorder = smoke_run["recorder"]
    manifest = smoke_run["manifest"]
    for record in manifest["frozen_policies"]:
        unit = record["evaluation_unit"]
        phases = [phase for phase, event_unit, _ in recorder.events if event_unit == unit]
        assert "search" in phases
        assert "final" in phases
        assert phases.index("final") > max(
            index for index, phase in enumerate(phases) if phase == "search"
        )

        frozen_path = Path(smoke_run["paths"]["manifest"]).parent / record["frozen_policy_path"]
        frozen = load_frozen_policy(frozen_path)
        assert frozen.status == "frozen"
        assert frozen.frozen_policy_hash == record["frozen_policy_hash"]
        assert frozen.resolved_policy.policy_id == record["policy_id"]
        assert frozen.training_protocol["test_access"] == "single_shot_after_freeze"


def test_exactly_one_test_evaluation_per_unit(smoke_run) -> None:
    final = pd.read_csv(smoke_run["paths"]["final_metrics"])
    per_unit = final.groupby(["evaluation_unit", "metric"]).size()
    assert (per_unit == 1).all()
    assert final["evaluation_unit"].nunique() == smoke_run["manifest"]["n_test_evaluations"]
    assert final["frozen_policy_hash"].nunique() == final["evaluation_unit"].nunique()


def test_a_second_run_into_the_same_directory_is_refused(smoke_run) -> None:
    with pytest.raises(FileExistsError, match="non-empty"):
        run_external_family_validation(
            smoke_run["config"],
            config_path=smoke_run["root"] / "suzuki_fixture.yaml",
        )


def test_outputs_are_hash_verifiable_and_provenance_bound(smoke_run) -> None:
    manifest = smoke_run["manifest"]
    directory = Path(smoke_run["paths"]["manifest"]).parent
    for name, digest in manifest["output_file_hashes"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert manifest["dataset_hash"] == hashlib.sha256(
        Path(smoke_run["dataset_path"]).read_bytes()
    ).hexdigest()
    provenance = manifest["reaction_family"]["provenance"]
    assert provenance["raw_file_sha256"] == manifest["dataset_hash"]
    assert manifest["reaction_family"]["family_id"] == "suzuki_miyaura"
    assert manifest["reaction_family"]["transferable_roles"] == [
        "ligand",
        "base",
        "solvent_or_additive",
    ]
    assert manifest["historical_results_loaded"] is False


def test_dataset_hash_must_match_the_declared_provenance(tmp_path) -> None:
    dataset_path = write_suzuki_fixture_csv(tmp_path)
    config = suzuki_fixture_config(dataset_path, tmp_path / "out", n_bits=32)
    config["reaction_family"]["provenance"]["raw_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match the declared provenance"):
        run_external_family_validation(config, config_path=tmp_path / "config.yaml")
    assert not (tmp_path / "out").exists()


def test_synthetic_candidates_are_audited_and_training_only(smoke_run) -> None:
    audit = pd.read_csv(smoke_run["paths"]["candidate_audit"])
    assert not audit.empty
    canonical_ids = _partition_source_ids(smoke_run)
    assert set(audit["source_row_id"]) <= canonical_ids["train"]
    assert set(audit["donor_row_id"]) <= canonical_ids["train"]
    accepted = audit.loc[audit["accepted"].astype(bool)]
    assert not accepted.empty
    for value in accepted["changed_roles"].fillna(""):
        changed = {role for role in str(value).split("|") if role}
        assert changed <= {"ligand", "base", "solvent_or_additive"}


def _partition_source_ids(smoke_run) -> dict[str, set[str]]:
    from bh_augmentation.data.canonical_splits import build_grouped_outer_assignments
    from suzuki_fixture import suzuki_fixture_adapter, suzuki_fixture_frame

    adapter = suzuki_fixture_adapter(smoke_run["dataset_path"])
    frame = suzuki_fixture_frame()
    canonical = adapter.canonicalize_dataframe(
        frame,
        source_file_hash=smoke_run["manifest"]["dataset_hash"],
        source_row_positions=range(len(frame)),
    )
    usable = canonical.loc[canonical["family_eligible"].astype(bool)].reset_index(drop=True)
    outer = build_grouped_outer_assignments(
        usable,
        seed=0,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )
    return {
        split: set(outer.loc[outer["outer_split"].eq(split), "source_row_id"].astype(str))
        for split in ("train", "valid", "test")
    }
