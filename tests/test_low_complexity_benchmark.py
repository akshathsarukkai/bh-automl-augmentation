"""Scientific and artifact gates for the Phase 13 benchmark."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bh_augmentation.low_complexity_benchmark import (
    LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION,
    run_low_complexity_benchmark,
    validate_low_complexity_benchmark,
)
from bh_augmentation.representations.low_complexity import (
    LOW_COMPLEXITY_METHODS,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    ROOT / "configs/corrected_low_complexity_benchmark_phase13_smoke.yaml"
)


@pytest.fixture(scope="module")
def completed_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("phase13") / "corrected-phase13"
    run_low_complexity_benchmark(SMOKE_CONFIG, output_directory=output)
    return output


def _copy_bundle(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "phase13-tampered"
    shutil.copytree(source, target)
    return target


def _rehash(root: Path, changed_names: list[str]) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in changed_names:
        manifest["output_hashes"][name] = sha256_file(root / name)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_configs_predefine_equal_six_family_selection_budget() -> None:
    for name in (
        "corrected_low_complexity_benchmark_phase13_smoke.yaml",
        "corrected_low_complexity_benchmark_phase13.yaml",
    ):
        config = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert tuple(config["methods"]["families"]) == LOW_COMPLEXITY_METHODS
        assert config["methods"]["latent_widths"] == [8, 16]
        assert config["selection_metric"] == "rmse"
        if name.endswith("phase13.yaml"):
            assert config["features"]["n_bits"] == 2048


def test_bundle_reports_all_methods_without_cross_method_selection(
    completed_bundle: Path,
) -> None:
    manifest = validate_low_complexity_benchmark(completed_bundle)
    plan = json.loads((completed_bundle / "benchmark_plan.json").read_text())
    frozen = json.loads(
        (completed_bundle / "frozen_method_policies.json").read_text()
    )
    final = pd.read_csv(completed_bundle / "final_test_metrics.csv")
    claims = pd.read_csv(completed_bundle / "evaluation_claims.csv")

    assert manifest["schema_version"] == LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION
    assert manifest["method_family_count"] == 6
    assert manifest["cross_method_selection"] is False
    assert manifest["test_used_for_selection"] is False
    assert plan["status"] == "frozen_before_any_outcome_loading"
    assert plan["test_labels_accessed"] is False
    assert frozen["status"] == "frozen_before_outer_test_outcome_loading"
    assert frozen["cross_method_selection"] is False
    assert {record["method_family"] for record in frozen["policies"]} == set(
        LOW_COMPLEXITY_METHODS
    )
    assert set(final["method_family"]) == set(LOW_COMPLEXITY_METHODS)
    assert final["test_evaluation_count"].eq(1).all()
    assert len(claims) == 6
    assert claims["status"].eq("complete").all()
    assert claims["outer_test_prediction_batches"].eq(1).all()


def test_all_methods_use_identical_outer_memberships_and_train_only_fits(
    completed_bundle: Path,
) -> None:
    audit = pd.read_csv(completed_bundle / "refit_audit.csv")

    assert audit["representation_fit_excludes_saved_validation_and_test"].all()
    assert audit["fit_validation_overlap_count"].eq(0).all()
    assert audit["fit_test_overlap_count"].eq(0).all()
    assert audit["evaluation_labels_received_by_fit"].eq(False).all()
    for _, rows in audit.groupby("evaluation_unit"):
        for field in (
            "representation_fit_source_id_hash",
            "saved_validation_source_id_hash",
            "outer_test_source_id_hash",
            "exact_split_hash",
            "aggregate_split_hash",
            "feature_metadata_hash",
        ):
            assert rows[field].nunique() == 1


def test_matched_neural_controls_have_exact_parameter_counts(
    completed_bundle: Path,
) -> None:
    audit = pd.read_csv(completed_bundle / "search_fit_audit.csv")
    paired = audit.loc[
        audit["method_family"].isin(
            {"small_bottleneck_mlp", "direct_mlp_regressor"}
        )
    ]
    pivot = paired.pivot(
        index=["evaluation_unit", "requested_latent_width"],
        columns="method_family",
        values="deployed_predictive_parameter_count",
    )

    assert (
        pivot["small_bottleneck_mlp"]
        == pivot["direct_mlp_regressor"]
    ).all()
    state_pivot = paired.pivot(
        index=["evaluation_unit", "requested_latent_width"],
        columns="method_family",
        values="matched_neural_model_state_hash",
    )
    assert (
        state_pivot["small_bottleneck_mlp"]
        == state_pivot["direct_mlp_regressor"]
    ).all()


def test_validator_rejects_rehashed_validation_oracle_attack(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "search_predictions.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    rows["prediction"] = rows["measured_yield"]
    rows["signed_error"] = 0.0
    rows["absolute_error"] = 0.0
    rows.to_csv(path, index=False)
    _rehash(root, [path.name])

    with pytest.raises(ValueError, match="prediction replay"):
        validate_low_complexity_benchmark(root)


def test_validator_rejects_rehashed_fit_contamination_claim(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "refit_audit.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    rows.loc[0, "fit_test_overlap_count"] = 1
    rows.to_csv(path, index=False)
    _rehash(root, [path.name])

    with pytest.raises(ValueError, match="fit audit"):
        validate_low_complexity_benchmark(root)


def test_validator_rejects_cross_method_winner_field_even_when_rehashed(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "frozen_method_policies.json"
    document = json.loads(path.read_text())
    document["overall_winner"] = "partial_least_squares"
    document.pop("frozen_document_hash")
    document["frozen_document_hash"] = stable_hash(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    _rehash(root, [path.name])

    with pytest.raises(ValueError, match="frozen"):
        validate_low_complexity_benchmark(root)


def test_validator_rejects_extra_completion_field_after_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["overall_winner"] = "partial_least_squares"
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="completion manifest"):
        validate_low_complexity_benchmark(root)


def test_runner_refuses_to_overwrite_completed_output(
    completed_bundle: Path,
) -> None:
    with pytest.raises(FileExistsError, match="overwrite"):
        run_low_complexity_benchmark(
            SMOKE_CONFIG, output_directory=completed_bundle
        )
