"""Tests for the read-only registry CLI, bundle builder, and smoke runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from scripts import build_reproducibility_bundle, inspect_evaluation_registry
from scripts import run_production_path_smoke as smoke

from bh_augmentation.evaluation.evaluation_registry import (
    EvaluationIdentity,
    EvaluationRegistry,
    hash_metrics_payload,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash
from bh_augmentation.utils.scientific_manifest import (
    ManifestError,
    build_scientific_manifest,
    write_scientific_manifest,
)
from bh_augmentation.utils.strict_config import OutputDirectoryReuseError

_NOW = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)
_METRICS = [{"evaluation_unit": "outer=group-a", "metric": "rmse", "value": 1.5}]


def _identity(unit: str) -> EvaluationIdentity:
    return EvaluationIdentity(
        dataset_hash="a" * 64,
        split_or_search_manifest_hash="b" * 64,
        frozen_policy_hash="c" * 64,
        evaluation_unit=unit,
    )


def _populated_registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    registry = EvaluationRegistry(root)
    reserved = _identity("outer=group-reserved")
    completed = _identity("outer=group-complete")
    registry.reserve(reserved, now=_NOW)
    registry.reserve(completed, now=_NOW)
    registry.mark_prediction_complete(completed, prediction_hash="d" * 64, now=_NOW)
    registry.mark_metrics_complete(
        completed,
        metrics_payload=_METRICS,
        metrics_hash=hash_metrics_payload(_METRICS),
        now=_NOW,
    )
    return root


def _result_directory(tmp_path: Path, name: str = "corrected-result") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "final_test_metrics.csv").write_text("metric,value\nrmse,1.5\n")
    dataset = tmp_path / "canonical_dataset.csv"
    dataset.write_text("source_row_id,yield\nrow-0,50.0\n")
    manifest = build_scientific_manifest(
        status="corrected_revalidation_complete",
        output_directory=directory,
        outputs=("final_test_metrics.csv",),
        dataset_path=dataset,
        dataset_hash=sha256_file(dataset),
        split_hash="b" * 64,
        feature_metadata_hash="c" * 64,
        config_hash="d" * 64,
        plan_hash="e" * 64,
        resolved_scientific_config={"dataset_path": str(dataset), "seeds": [0]},
        row_counts={"final_test_metrics": 1},
        command="python -B scripts/run_production_path_smoke.py --output-directory OUT",
    )
    write_scientific_manifest(directory, manifest)
    return directory


# --- read-only registry CLI -------------------------------------------------


def test_registry_cli_lists_records_without_mutating_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _populated_registry(tmp_path)
    before = {path: sha256_file(path) for path in sorted(root.rglob("*.json"))}

    exit_code = inspect_evaluation_registry.main(
        ["--registry-directory", str(root), "list", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["record_count"] == 2
    assert {record["status"] for record in payload["records"]} == {
        "reserved",
        "metrics_complete",
    }
    assert {path: sha256_file(path) for path in sorted(root.rglob("*.json"))} == before


def test_registry_cli_filters_and_shows_one_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _populated_registry(tmp_path)
    inspect_evaluation_registry.main(
        ["--registry-directory", str(root), "list", "--status", "metrics_complete", "--json"]
    )
    listed = json.loads(capsys.readouterr().out)
    assert listed["record_count"] == 1
    key = listed["records"][0]["registry_key"]

    assert (
        inspect_evaluation_registry.main(
            ["--registry-directory", str(root), "show", "--registry-key", key]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "metrics_complete"
    assert document["record_payload_hash"] == stable_hash(
        {key_: value for key_, value in document.items() if key_ != "record_payload_hash"}
    )


def test_registry_cli_offers_no_mutating_subcommand(tmp_path: Path) -> None:
    for argv in (
        ["delete", "--registry-key", "x"],
        ["release", "--registry-key", "x"],
        ["reset"],
    ):
        with pytest.raises(SystemExit):
            inspect_evaluation_registry.main(
                ["--registry-directory", str(tmp_path), *argv]
            )
    source = Path(inspect_evaluation_registry.__file__).read_text()
    for forbidden in ("unlink(", "rmtree(", "os.remove", "shutil.move", "write_text("):
        assert forbidden not in source


def test_registry_cli_handles_an_empty_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = inspect_evaluation_registry.main(
        ["--registry-directory", str(tmp_path / "absent"), "list"]
    )
    assert exit_code == 0
    assert "No evaluation claims found." in capsys.readouterr().out


# --- local reproducibility bundle -------------------------------------------


def test_bundle_collects_manifest_config_environment_inputs_and_commands(
    tmp_path: Path,
) -> None:
    result = _result_directory(tmp_path)
    config = tmp_path / "example_config.yaml"
    config.write_text("dataset:\n  path: canonical_dataset.csv\n")
    bundle = build_reproducibility_bundle.build_bundle(
        result_directory=result,
        bundle_directory=tmp_path / "bundle",
        config_path=config,
    )

    assert sorted(path.name for path in bundle.iterdir()) == [
        "REPRODUCE.md",
        "bundle_manifest.json",
        "config",
        "environment.json",
        "git.json",
        "input_hashes.json",
        "run_manifest.json",
    ]
    bundle_manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert bundle_manifest["published"] is False
    assert bundle_manifest["network_access"] is False
    assert bundle_manifest["result_verification"]["verified_output_count"] == 1
    assert bundle_manifest["bundle_hash"] == stable_hash(
        {k: v for k, v in bundle_manifest.items() if k != "bundle_hash"}
    )
    environment = json.loads((bundle / "environment.json").read_text())
    assert environment["recorded_at_execution"]["platform_record"]["machine"]
    assert environment["observed_at_bundle_time"]["dependency_versions"]["python"]
    inputs = json.loads((bundle / "input_hashes.json").read_text())
    assert inputs["missing_input_count"] == 0
    assert all(item["sha256"] for item in inputs["inputs"])
    reproduce = (bundle / "REPRODUCE.md").read_text()
    assert "git worktree add" in reproduce
    assert "run_production_path_smoke.py" in reproduce
    assert "config/example_config.yaml" in reproduce
    assert (bundle / "config" / "example_config.yaml").read_text() == config.read_text()


def test_bundle_reports_inputs_a_human_must_supply(tmp_path: Path) -> None:
    result = _result_directory(tmp_path, name="corrected-missing-input")
    bundle = build_reproducibility_bundle.build_bundle(
        result_directory=result,
        bundle_directory=tmp_path / "bundle-missing",
        extra_inputs=[str(tmp_path / "not_committed.csv")],
    )
    inputs = json.loads((bundle / "input_hashes.json").read_text())
    assert inputs["missing_input_count"] == 1
    assert "must be supplied" in (bundle / "REPRODUCE.md").read_text()


def test_bundle_refuses_to_write_into_an_existing_directory(tmp_path: Path) -> None:
    result = _result_directory(tmp_path, name="corrected-existing-bundle")
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(OutputDirectoryReuseError):
        build_reproducibility_bundle.build_bundle(
            result_directory=result, bundle_directory=existing
        )
    assert not any(existing.iterdir())


def test_bundle_refuses_a_result_whose_outputs_were_tampered_with(tmp_path: Path) -> None:
    result = _result_directory(tmp_path, name="corrected-tampered")
    (result / "final_test_metrics.csv").write_text("metric,value\nrmse,0.0\n")
    with pytest.raises(ManifestError, match="output hash mismatch"):
        build_reproducibility_bundle.build_bundle(
            result_directory=result, bundle_directory=tmp_path / "bundle-tampered"
        )
    assert not (tmp_path / "bundle-tampered").exists()


def test_bundle_performs_no_network_access() -> None:
    source = Path(build_reproducibility_bundle.__file__).read_text()
    for forbidden in ("requests", "urllib", "http", "socket", "boto3", "upload"):
        assert forbidden not in source


# --- production-path smoke runner -------------------------------------------


def test_smoke_uses_only_committed_input_data() -> None:
    assert smoke.SOURCE_DATASET.is_file()
    assert smoke.SOURCE_DATASET.name == "bh_clean_stress.csv"


def test_smoke_configs_are_strict_and_declare_all_six_families(tmp_path: Path) -> None:
    from bh_augmentation.low_complexity_benchmark import _resolve_contract
    from bh_augmentation.representations.low_complexity import LOW_COMPLEXITY_METHODS

    config = smoke._benchmark_config(
        tmp_path / "canonical.csv", tmp_path / "splits", tmp_path / "corrected-out"
    )
    contract = _resolve_contract(config)
    assert contract["families"] == LOW_COMPLEXITY_METHODS
    assert contract["latent_widths"] == (8, 16)
    assert contract["output_directory"] == tmp_path / "corrected-out"


def test_smoke_refuses_to_reuse_an_output_directory(tmp_path: Path) -> None:
    existing = tmp_path / "smoke-out"
    existing.mkdir()
    with pytest.raises(OutputDirectoryReuseError):
        smoke.run_smoke(existing)


# --- CI tamper-evidence assertion -------------------------------------------


def test_tamper_assertion_passes_and_leaves_the_original_bundle_untouched(
    tmp_path: Path,
) -> None:
    from scripts import assert_manifest_tamper_evident as tamper

    result = _result_directory(tmp_path, name="corrected-tamper-check")
    before = {
        path.name: sha256_file(path) for path in sorted(result.iterdir()) if path.is_file()
    }
    tamper.assert_tamper_evident(result, "final_test_metrics.csv")
    after = {
        path.name: sha256_file(path) for path in sorted(result.iterdir()) if path.is_file()
    }
    assert before == after


def test_tamper_assertion_fails_when_verification_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import assert_manifest_tamper_evident as tamper

    result = _result_directory(tmp_path, name="corrected-blind-verifier")
    monkeypatch.setattr(tamper, "verify_manifest", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit, match="failed to detect a mutated output byte"):
        tamper.assert_tamper_evident(result, "final_test_metrics.csv")


# --- result-family classification document ----------------------------------


def test_result_status_document_classifies_and_asserts_no_phase_14_outcome() -> None:
    text = Path("RESULT_STATUS.md").read_text()
    for status in (
        "Supported",
        "Development evidence",
        "Confirmatory evidence",
        "Experimental",
        "Invalidated",
        "Deprecated",
        "Externally blocked",
    ):
        assert status in text
    # The pre-existing invalidations must stay intact and clearly excluded.
    assert "Role-aware condition transfer v2 | **Invalidated**" in text
    assert "results/stress/logo_product/` outputs | **Invalidated**" in text
    assert "results/stress/logo_reactant/` outputs | **Invalidated**" in text
    # Phase 16 is externally blocked; no external result is claimed.
    assert "Phase 16 external typed-reaction dataset adapter | Externally blocked" in text
    # Phase 14 is still executing: its outcome must not be asserted anywhere.
    assert "No Phase 14 result, positive or negative, is asserted" in text
    assert "The confirmatory-evidence class is currently empty." in text
