"""Scientific and artifact gates for the Phase 18 recommendation simulation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from bh_augmentation.recommendation_simulation import (
    ACQUISITION_STRATEGIES,
    ALL_STRATEGIES,
    INFORMED_STRATEGIES,
    PHASE12_SURVIVING_UNCERTAINTY_METHODS,
    RANDOM_STRATEGY,
    RECOMMENDATION_SIMULATION_SCHEMA_VERSION,
    AcquisitionContext,
    AcquisitionSelection,
    EvaluationOracleGate,
    EvaluationOracleGateError,
    FutureLabelAccessError,
    LabelOracle,
    run_recommendation_simulation,
    validate_recommendation_simulation,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash
from canonical_artifacts import require_canonical_artifacts

pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs/corrected_recommendation_simulation_phase18_smoke.yaml"
PRODUCTION_CONFIG = ROOT / "configs/corrected_recommendation_simulation_phase18.yaml"

_SCIENTIFIC_ARTIFACTS = (
    "campaign_plan.json",
    "split_units.csv",
    "campaign_pools.csv",
    "acquisition_audit.csv",
    "acquisitions.csv",
    "uncertainty_diagnostics.csv",
    "round_trajectory.csv",
    "random_distribution.csv",
    "strategy_percentiles.csv",
    "summary.csv",
)


def _tiny_config(tmp_path: Path) -> dict[str, Any]:
    """Return the smallest config the contract allows, for fast unit gates."""
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    config["campaign"]["rounds"] = 2
    config["campaign"]["batch_size"] = 4
    config["campaign"]["random_replicates"] = 200
    config["output"]["directory"] = str(tmp_path / "corrected-unused")
    return config


@pytest.fixture(scope="module")
def completed_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    require_canonical_artifacts()
    output = tmp_path_factory.mktemp("phase18") / "corrected-phase18"
    run_recommendation_simulation(SMOKE_CONFIG, output_directory=output)
    return output


def _copy_bundle(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "phase18-tampered"
    shutil.copytree(source, target)
    return target


def _rehash(root: Path, changed_names: list[str]) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in changed_names:
        manifest["output_hashes"][name] = sha256_file(root / name)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_configs_declare_only_phase12_surviving_uncertainty() -> None:
    for path in (SMOKE_CONFIG, PRODUCTION_CONFIG):
        config = yaml.safe_load(path.read_text())
        assert (
            config["model"]["uncertainty_method"] in PHASE12_SURVIVING_UNCERTAINTY_METHODS
        )
        assert config["campaign"]["random_replicates"] >= 200
        assert config["features"]["kind"] == "bh_role_separated"
        assert config["features"]["fingerprint_backend"] == "rdkit"
    production = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    assert production["features"]["n_bits"] == 256
    assert production["splits"]["random_seeds"] == [0, 1, 2, 3, 4]


def test_unsupported_uncertainty_method_is_rejected(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    config["model"]["uncertainty_method"] = "split_conformal_ridge"
    with pytest.raises(ValueError, match="survived Phase 12"):
        run_recommendation_simulation(config, output_directory=tmp_path / "corrected-x")


def test_small_random_replicate_count_is_rejected(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    config["campaign"]["random_replicates"] = 25
    with pytest.raises(ValueError, match="at least 200"):
        run_recommendation_simulation(config, output_directory=tmp_path / "corrected-y")


def test_bundle_reports_every_strategy_without_autoencoder_or_leakage(
    completed_bundle: Path,
) -> None:
    manifest = validate_recommendation_simulation(completed_bundle)
    plan = json.loads((completed_bundle / "campaign_plan.json").read_text())
    trajectory = pd.read_csv(completed_bundle / "round_trajectory.csv")

    assert manifest["schema_version"] == RECOMMENDATION_SIMULATION_SCHEMA_VERSION
    assert manifest["future_label_access_events"] == 0
    assert manifest["supervised_autoencoder_used"] is False
    assert manifest["prospective_validation_claimed"] is False
    assert manifest["pool_outcomes_loaded_after_campaigns"] is True
    assert manifest["uncertainty_method"] in PHASE12_SURVIVING_UNCERTAINTY_METHODS
    assert plan["status"] == "frozen_before_any_outcome_loading"
    assert plan["pool_outcomes_loaded"] is False
    assert set(trajectory["strategy"]) == set(ALL_STRATEGIES)


def test_acquisition_never_sees_a_future_label(completed_bundle: Path) -> None:
    audit = pd.read_csv(completed_bundle / "acquisition_audit.csv")

    assert audit["future_label_access_events"].eq(0).all()
    assert audit["candidate_labeled_overlap"].eq(0).all()
    assert (
        audit["scorer_visible_label_id_hash"].eq(audit["labeled_before_id_hash"]).all()
    )
    assert (
        audit["oracle_cumulative_revealed_id_hash"]
        .eq(audit["labeled_before_id_hash"])
        .all()
    )
    expected = audit["seed_pool_size"] + (audit["round_index"] - 1) * audit["batch_size"]
    assert audit["labeled_before_count"].eq(expected).all()
    assert audit["oracle_cumulative_revealed_count"].eq(expected).all()
    informed = audit.loc[audit["strategy"].ne(RANDOM_STRATEGY)]
    assert informed["labels_consumed_by_acquisition"].all()
    assert not audit.loc[audit["strategy"].eq(RANDOM_STRATEGY)][
        "labels_consumed_by_acquisition"
    ].any()


def test_context_refuses_unacquired_labels() -> None:
    context = AcquisitionContext(
        evaluation_unit="unit",
        strategy="greedy_predicted_yield",
        replicate=0,
        round_index=1,
        batch_size=1,
        candidate_source_ids=("candidate-a", "candidate-b"),
        candidate_pool_positions=np.asarray([2, 3], dtype=np.int64),
        labeled_source_ids=("seed-a",),
        labeled_values={"seed-a": 10.0},
        similarity=np.eye(4),
        nearest_similarity=np.zeros(2),
        point=np.asarray([1.0, 2.0]),
        calibrated_uncertainty=np.asarray([1.0, 1.0]),
        rng=np.random.default_rng(0),
        params={},
    )

    assert context.label_of("seed-a") == 10.0
    with pytest.raises(FutureLabelAccessError, match="unacquired label"):
        context.label_of("candidate-a")


def test_context_rejects_labeled_candidate_overlap() -> None:
    with pytest.raises(FutureLabelAccessError, match="already labeled"):
        AcquisitionContext(
            evaluation_unit="unit",
            strategy="greedy_predicted_yield",
            replicate=0,
            round_index=1,
            batch_size=1,
            candidate_source_ids=("seed-a",),
            candidate_pool_positions=np.asarray([0], dtype=np.int64),
            labeled_source_ids=("seed-a",),
            labeled_values={"seed-a": 10.0},
            similarity=np.eye(1),
            nearest_similarity=np.zeros(1),
            point=np.asarray([1.0]),
            calibrated_uncertainty=np.asarray([1.0]),
            rng=np.random.default_rng(0),
            params={},
        )


def test_random_strategy_may_not_read_any_outcome() -> None:
    context = AcquisitionContext(
        evaluation_unit="unit",
        strategy=RANDOM_STRATEGY,
        replicate=0,
        round_index=1,
        batch_size=1,
        candidate_source_ids=("candidate-a", "candidate-b"),
        candidate_pool_positions=np.asarray([2, 3], dtype=np.int64),
        labeled_source_ids=("seed-a",),
        labeled_values=None,
        similarity=np.eye(4),
        nearest_similarity=np.zeros(2),
        point=None,
        calibrated_uncertainty=None,
        rng=np.random.default_rng(0),
        params={},
    )

    assert context.consumes_labels is False
    with pytest.raises(FutureLabelAccessError, match="label-free"):
        context.label_of("seed-a")


def test_saboteur_strategy_reading_a_future_label_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strategy that peeks at an unacquired outcome must abort the simulation."""
    require_canonical_artifacts()

    def _saboteur(context: AcquisitionContext) -> AcquisitionSelection:
        stolen = [context.label_of(source_id) for source_id in context.candidate_source_ids]
        order = np.argsort(-np.asarray(stolen, dtype=float), kind="stable")
        positions = tuple(int(value) for value in np.sort(order[: context.batch_size]))
        return AcquisitionSelection(
            positions=positions,
            scores=tuple(float(stolen[index]) for index in positions),
            score_kind="oracle_cheat",
        )

    monkeypatch.setitem(ACQUISITION_STRATEGIES, "greedy_predicted_yield", _saboteur)
    with pytest.raises(FutureLabelAccessError, match="unacquired label"):
        run_recommendation_simulation(
            _tiny_config(tmp_path), output_directory=tmp_path / "corrected-saboteur"
        )


def test_label_oracle_refuses_rows_outside_the_pool(tmp_path: Path) -> None:
    oracle = LabelOracle(
        dataset_path=ROOT / "data/processed/bh_canonical_roles_v1.csv",
        canonical_source_ids=("a", "b"),
        allowed_pool_ids=frozenset({"a"}),
        campaign_id="unit|test|replicate=0",
    )
    with pytest.raises(ValueError, match="outside the campaign pool"):
        oracle.reveal(["b"], round_index=1)


def test_evaluation_gate_blocks_pool_outcomes_before_campaigns() -> None:
    gate = EvaluationOracleGate()
    gate.require_locked()
    with pytest.raises(EvaluationOracleGateError, match="after every campaign"):
        gate.require_unlocked()
    gate.unlock()
    gate.require_unlocked()
    with pytest.raises(EvaluationOracleGateError, match="during acquisition"):
        gate.require_locked()


def test_every_strategy_shares_matched_pools_budget_and_rounds(
    completed_bundle: Path,
) -> None:
    audit = pd.read_csv(completed_bundle / "acquisition_audit.csv")
    manifest = json.loads((completed_bundle / "manifest.json").read_text())

    for _, rows in audit.groupby("evaluation_unit"):
        for column in (
            "initial_seed_source_id_hash",
            "initial_candidate_source_id_hash",
            "seed_pool_size",
            "candidate_pool_size",
            "rounds",
            "batch_size",
            "total_budget",
        ):
            assert rows[column].nunique() == 1
        rounds_per_campaign = rows.groupby(["strategy", "replicate"]).size()
        assert rounds_per_campaign.nunique() == 1
        assert int(rounds_per_campaign.iloc[0]) == manifest["rounds"]
    assert manifest["total_budget"] == manifest["rounds"] * manifest["batch_size"]


def test_group_separation_between_seed_pool_and_discoveries(
    completed_bundle: Path,
) -> None:
    pools = pd.read_csv(completed_bundle / "campaign_pools.csv")
    acquisitions = pd.read_csv(completed_bundle / "acquisitions.csv")

    assert pools["seed_candidate_group_overlap"].eq(0).all()
    assert pools["pool_group_count"].eq(pools["pool_row_count"]).all()
    assert pools["pool_rows_per_canonical_reaction"].eq(1).all()
    assert not acquisitions["acquired_reaction_in_seed_pool"].any()
    duplicated = acquisitions.duplicated(
        subset=["evaluation_unit", "strategy", "canonical_reaction_key"]
    )
    assert not duplicated.any()


def test_random_distribution_is_complete_and_independent(
    completed_bundle: Path,
) -> None:
    manifest = json.loads((completed_bundle / "manifest.json").read_text())
    trajectory = pd.read_csv(completed_bundle / "round_trajectory.csv")
    distribution = pd.read_csv(completed_bundle / "random_distribution.csv")
    percentiles = pd.read_csv(completed_bundle / "strategy_percentiles.csv")

    random_rows = trajectory.loc[trajectory["strategy"].eq(RANDOM_STRATEGY)]
    replicate_counts = random_rows.groupby(["evaluation_unit", "round_index"])[
        "replicate"
    ].nunique()
    assert manifest["random_replicate_count"] >= 200
    assert replicate_counts.eq(manifest["random_replicate_count"]).all()
    acquiring = random_rows.loc[random_rows["round_index"].gt(0)]
    distinct = acquiring.groupby(["evaluation_unit", "round_index"])[
        "selected_source_id_hash"
    ].nunique()
    assert distinct.gt(1).all()
    reported = distribution.loc[distribution["metric"].eq("best_yield_discovered")]
    assert reported["replicate_count"].eq(manifest["random_replicate_count"]).all()
    assert set(percentiles["strategy"]) == set(INFORMED_STRATEGIES)
    scored = percentiles.loc[percentiles["random_percentile"].notna()]
    assert scored["random_percentile"].between(0.0, 100.0).all()


def test_manifest_and_artifact_hashes_validate(completed_bundle: Path) -> None:
    manifest = json.loads((completed_bundle / "manifest.json").read_text())
    for name, digest in manifest["output_hashes"].items():
        assert sha256_file(completed_bundle / name) == digest
    claimed = manifest.pop("manifest_hash")
    assert claimed == stable_hash(manifest)


def test_validator_rejects_tampered_trajectory_after_rehash(
    completed_bundle: Path, tmp_path: Path
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "round_trajectory.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    informed = rows["strategy"].ne(RANDOM_STRATEGY)
    rows.loc[informed, "best_yield_discovered"] = 100.0
    rows.to_csv(path, index=False)
    _rehash(root, [path.name])

    with pytest.raises(ValueError, match="replay mismatch"):
        validate_recommendation_simulation(root)


def test_validator_rejects_manifest_hash_tampering(
    completed_bundle: Path, tmp_path: Path
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["future_label_access_events"] = 3
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="completion manifest"):
        validate_recommendation_simulation(root)


def test_validator_rejects_unhashed_artifact_edit(
    completed_bundle: Path, tmp_path: Path
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "summary.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    rows.loc[0, "mean_value"] = -1.0
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_recommendation_simulation(root)


def test_runner_refuses_to_overwrite_completed_output(completed_bundle: Path) -> None:
    with pytest.raises(FileExistsError, match="overwrite"):
        run_recommendation_simulation(SMOKE_CONFIG, output_directory=completed_bundle)


def test_two_runs_produce_byte_identical_scientific_artifacts(tmp_path: Path) -> None:
    require_canonical_artifacts()
    first = tmp_path / "corrected-run-a"
    second = tmp_path / "corrected-run-b"
    config = _tiny_config(tmp_path)
    run_recommendation_simulation(config, output_directory=first)
    run_recommendation_simulation(config, output_directory=second)

    for name in _SCIENTIFIC_ARTIFACTS:
        assert sha256_file(first / name) == sha256_file(second / name), name
    left = json.loads((first / "manifest.json").read_text())
    right = json.loads((second / "manifest.json").read_text())
    assert left["output_hashes"] == right["output_hashes"]
    assert left["plan_hash"] == right["plan_hash"]
    assert left["config_hash"] == right["config_hash"]
