"""Contract and tamper-evidence tests for the shared scientific manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bh_augmentation.utils.corrected_runs import stable_hash
from bh_augmentation.utils.scientific_manifest import (
    REQUIRED_MANIFEST_FIELDS,
    SCIENTIFIC_MANIFEST_FILENAME,
    SCIENTIFIC_MANIFEST_SCHEMA_VERSION,
    ManifestError,
    build_scientific_manifest,
    compute_output_hashes,
    dependency_versions,
    platform_record,
    read_manifest,
    verify_manifest,
    write_scientific_manifest,
)

_HASH = "a" * 64


def _bundle(tmp_path: Path, name: str = "corrected-bundle") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "final_test_metrics.csv").write_text("metric,value\nrmse,1.5\n")
    (directory / "summary.csv").write_text("method,rmse\nridge,1.5\n")
    return directory


def _write_bundle_manifest(directory: Path) -> Path:
    manifest = build_scientific_manifest(
        status="corrected_revalidation_complete",
        output_directory=directory,
        outputs=("final_test_metrics.csv", "summary.csv"),
        dataset_path="data/processed/example.csv",
        dataset_hash=_HASH,
        split_hash="b" * 64,
        feature_metadata_hash="c" * 64,
        config_hash="d" * 64,
        plan_hash="e" * 64,
        resolved_scientific_config={"seeds": [0], "metrics": ["rmse"]},
        row_counts={"final_test_metrics": 1, "summary": 1},
        command="python -B scripts/example.py",
    )
    return write_scientific_manifest(directory, manifest)


def test_manifest_declares_every_required_provenance_field(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    path = _write_bundle_manifest(directory)
    manifest = json.loads(path.read_text())

    assert path.name == SCIENTIFIC_MANIFEST_FILENAME
    assert REQUIRED_MANIFEST_FIELDS.issubset(set(manifest))
    assert manifest["schema_version"] == SCIENTIFIC_MANIFEST_SCHEMA_VERSION
    assert manifest["git_commit"] and manifest["git_commit"] != "unknown"
    assert isinstance(manifest["git_dirty_at_execution"], bool)
    assert set(manifest["output_hashes"]) == {"final_test_metrics.csv", "summary.csv"}
    assert manifest["row_counts"] == {"final_test_metrics": 1, "summary": 1}
    assert manifest["manifest_hash"] == stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )


def test_manifest_records_hardware_and_threading_environment(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    manifest = read_manifest(_write_bundle_manifest(directory).parent)
    record = manifest["platform_record"]

    for field in (
        "platform",
        "system",
        "machine",
        "processor",
        "python_implementation",
        "python_version",
        "cpu_count_logical",
        "cpu_count_affinity",
        "environment",
        "torch",
    ):
        assert field in record
    assert record["python_implementation"] == platform_record()["python_implementation"]
    assert "version" in record["torch"]
    if record["torch"]["version"] != "not_installed":
        assert isinstance(record["torch"]["num_threads"], int)
    assert set(manifest["dependency_versions"]) >= {"python", "numpy", "pandas"}
    assert manifest["dependency_versions"] == dependency_versions()


def test_manifest_captures_set_blas_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    record = platform_record()
    assert record["environment"]["OMP_NUM_THREADS"] == "3"
    assert "MKL_NUM_THREADS" not in record["environment"]
    assert os.environ.get("OMP_NUM_THREADS") == "3"


def test_verification_recomputes_manifest_and_every_output_hash(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    _write_bundle_manifest(directory)
    verification = verify_manifest(directory)

    assert verification.verified_output_count == 2
    assert verification.verified_outputs == ("final_test_metrics.csv", "summary.csv")
    assert verification.schema_version == SCIENTIFIC_MANIFEST_SCHEMA_VERSION
    assert verification.to_dict()["manifest_hash"] == verification.manifest_hash


def test_single_mutated_output_byte_fails_verification(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    _write_bundle_manifest(directory)
    verify_manifest(directory)

    target = directory / "summary.csv"
    original = target.read_bytes()
    tampered = bytearray(original)
    tampered[-2] = tampered[-2] ^ 0x01
    target.write_bytes(bytes(tampered))
    assert len(bytes(tampered)) == len(original)

    with pytest.raises(ManifestError, match="output hash mismatch"):
        verify_manifest(directory)


def test_mutated_manifest_field_fails_manifest_hash_verification(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    path = _write_bundle_manifest(directory)
    document = json.loads(path.read_text())
    document["dataset_hash"] = "f" * 64
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ManifestError, match="Manifest hash mismatch"):
        verify_manifest(directory)


def test_deleted_output_fails_verification(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    _write_bundle_manifest(directory)
    (directory / "summary.csv").unlink()
    with pytest.raises(ManifestError, match="output is missing"):
        verify_manifest(directory)


def test_verification_covers_preexisting_runner_manifests(tmp_path: Path) -> None:
    directory = _bundle(tmp_path, name="corrected-runner-style")
    manifest = {
        "schema_version": "bh-low-complexity-benchmark-v1",
        "status": "corrected_revalidation_complete",
        "output_hashes": compute_output_hashes(
            directory, ("final_test_metrics.csv", "summary.csv")
        ),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    verification = verify_manifest(directory)
    assert verification.manifest_path.name == "manifest.json"
    assert verification.schema_version == "bh-low-complexity-benchmark-v1"

    (directory / "final_test_metrics.csv").write_text("metric,value\nrmse,9.9\n")
    with pytest.raises(ManifestError, match="output hash mismatch"):
        verify_manifest(directory)


def test_manifest_write_refuses_to_overwrite(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    _write_bundle_manifest(directory)
    with pytest.raises(ManifestError, match="Refusing to overwrite"):
        _write_bundle_manifest(directory)


def test_manifest_extras_may_not_redefine_core_fields(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    with pytest.raises(ManifestError, match="may not redefine core fields"):
        build_scientific_manifest(
            status="corrected_revalidation_complete",
            output_directory=directory,
            outputs=("summary.csv",),
            dataset_path="data/processed/example.csv",
            dataset_hash=_HASH,
            split_hash="b" * 64,
            feature_metadata_hash="c" * 64,
            config_hash="d" * 64,
            plan_hash="e" * 64,
            resolved_scientific_config={},
            row_counts={"summary": 1},
            command="python -B scripts/example.py",
            extra={"dataset_hash": "0" * 64},
        )


def test_missing_declared_output_is_rejected_at_build_time(tmp_path: Path) -> None:
    directory = _bundle(tmp_path)
    with pytest.raises(ManifestError, match="output is missing"):
        compute_output_hashes(directory, ("summary.csv", "absent.csv"))


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    directory = _bundle(tmp_path, name="corrected-no-manifest")
    with pytest.raises(ManifestError, match="No manifest found"):
        verify_manifest(directory)
