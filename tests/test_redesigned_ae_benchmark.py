from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import yaml

import bh_augmentation.redesigned_ae_benchmark as benchmark_module
from bh_augmentation.evaluation.ae_retention import (
    AERetentionConfig,
    decide_ae_retention,
    decision_json,
)
from bh_augmentation.evaluation.evaluation_registry import EvaluationIdentity
from bh_augmentation.features.reconstruction_contract import (
    CANONICAL_COUNT_AWARE_EXCLUSION_REASON,
)
from bh_augmentation.redesigned_ae_benchmark import (
    MINIMUM_CONTROL_SEARCH_BUDGET,
    PHASE14_METHOD_FAMILIES,
    frozen_policy_identity_payload,
    run_redesigned_ae_benchmark,
    search_budget_by_family,
    validate_redesigned_ae_benchmark,
)
from bh_augmentation.representations.redesigned_supervised_autoencoder import RoleBlock
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs/redesigned_ae_benchmark_phase14_smoke.yaml"
PRODUCTION_CONFIG = ROOT / "configs/redesigned_ae_benchmark_phase14.yaml"
DATASET = ROOT / "data/processed/bh_canonical_roles_v1.csv"


@dataclass(frozen=True)
class _Unit:
    evaluation_unit: str = "seed=0|fraction=0.2"


def _candidate_configs(raw: dict[str, object]) -> list[tuple[str, object]]:
    contract = benchmark_module._resolve_contract(raw)
    return benchmark_module._candidate_configs(
        unit=_Unit(),
        contract=contract,
        input_dim=224,
        role_blocks=(RoleBlock("all_features", 0, 224),),
        common_seed=7,
    )


@pytest.fixture(scope="module")
def completed_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("phase14") / "bundle"
    registry = tmp_path_factory.mktemp("phase14-registry")
    run_redesigned_ae_benchmark(
        SMOKE_CONFIG,
        output_directory=output,
        evaluation_registry_directory=registry,
    )
    return output


@pytest.fixture(scope="module")
def structurally_invalid_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, BaseException]:
    """Run the benchmark with one structural replay forced to fail."""
    output = tmp_path_factory.mktemp("phase14-structural") / "bundle"
    registry = tmp_path_factory.mktemp("phase14-structural-registry")
    original = benchmark_module._assert_selection

    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("Phase 14 inner-selection replay mismatch.")

    benchmark_module._assert_selection = broken
    try:
        with pytest.raises(ValueError, match="inner-selection") as excinfo:
            run_redesigned_ae_benchmark(
                SMOKE_CONFIG,
                output_directory=output,
                evaluation_registry_directory=registry,
            )
    finally:
        benchmark_module._assert_selection = original
    return registry, excinfo.value


def _rehash_output(bundle: Path, name: str) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_hashes"][name] = sha256_file(bundle / name)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _set_manifest_row_count(bundle: Path, name: str, count: int) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_counts"][name] = count
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_production_and_smoke_configs_freeze_canonical_scope() -> None:
    production = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    smoke = yaml.safe_load(SMOKE_CONFIG.read_text())
    assert production["splits"]["seeds"] == [0, 1, 2, 3, 4]
    assert production["splits"]["fractions"] == [0.2, 1.0]
    assert production["features"]["n_bits"] == 2048
    assert smoke["splits"]["seeds"] == [0, 1, 2]
    assert smoke["splits"]["fractions"] == [0.2]
    assert smoke["features"]["n_bits"] == 32
    for config in (production, smoke):
        candidates = config["methods"]["ae_candidates"]
        by_id = {row["candidate_id"]: row for row in candidates}
        baseline_id = "ae-baseline-128-16-rec025-anonymous"
        assert len(candidates) == 11
        assert baseline_id in by_id
        assert {
            (row["hidden_dim"], row["latent_dim"]) for row in candidates
        } >= {(128, 16), (256, 32)}
        assert {
            row["synthetic_reconstruction_weight"] for row in candidates
        } == {0.0, 0.1, 0.25, 0.5, 1.0}
        assert {
            row["reconstruction_objective"] for row in candidates
        } == {"binary_cross_entropy", "positive_bit_weighted_mse"}
        assert {row["data_protocol"] for row in candidates} == {
            "anonymous",
            "typed",
            "real_only",
        }
        baseline = {
            key: value
            for key, value in by_id[baseline_id].items()
            if key != "candidate_id"
        }
        expected_differences = {
            "ae-rec0-anonymous": {"synthetic_reconstruction_weight"},
            "ae-rec01-anonymous": {"synthetic_reconstruction_weight"},
            "ae-rec05-anonymous": {"synthetic_reconstruction_weight"},
            "ae-rec1-anonymous": {"synthetic_reconstruction_weight"},
            "ae-bce-anonymous": {"reconstruction_objective"},
            "ae-256-32-anonymous": {"hidden_dim", "latent_dim"},
            "ae-mask01-anonymous": {"masking_probability"},
            "ae-supervised1-anonymous": {"synthetic_supervised_weight"},
            "ae-typed": {"data_protocol"},
            "ae-real-only": {"data_protocol"},
        }
        for candidate_id, expected in expected_differences.items():
            row = {
                key: value
                for key, value in by_id[candidate_id].items()
                if key != "candidate_id"
            }
            assert {
                key for key in baseline if row[key] != baseline[key]
            } == expected
        assert {
            row["synthetic_supervised_weight"]
            for row in candidates
            if row["data_protocol"] != "real_only"
        } <= set(config["methods"]["transfer_supervised_weights"])


def test_configs_declare_a_meaningful_teacher_ensemble() -> None:
    # A one-model "ensemble" has identically zero standard deviation, which
    # makes the uncertainty ranking and the max_teacher_std filter inert.
    for path in (PRODUCTION_CONFIG, SMOKE_CONFIG):
        config = yaml.safe_load(path.read_text())
        for kind in ("anonymous", "typed"):
            teachers = config["transfer"][kind]["teacher_models"]
            assert len(teachers) >= 2
            assert len(set(teachers)) == len(teachers)
            assert set(teachers) <= {
                "ridge",
                "random_forest",
                "extra_trees",
                "xgboost",
                "catboost",
            }


def test_control_families_get_a_real_search_grid() -> None:
    production = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    assert production["methods"]["ridge_alphas"] == [0.1, 1.0, 10.0]
    budget = search_budget_by_family(
        config.method_family for _, config in _candidate_configs(production)
    )

    assert budget == {
        "redesigned_supervised_ae": 11,
        "truncated_svd": 6,
        "linear_autoencoder": 6,
        "direct_xgboost": 3,
        "matched_direct_mlp": 10,
        "anonymous_transfer_without_ae": 6,
        "typed_transfer_without_ae": 6,
    }
    assert min(
        count
        for family, count in budget.items()
        if family != "redesigned_supervised_ae"
    ) >= MINIMUM_CONTROL_SEARCH_BUDGET
    policy_ids = [policy_id for policy_id, _ in _candidate_configs(production)]
    assert len(policy_ids) == len(set(policy_ids))
    assert "truncated_svd:latent=16:alpha=0.1" in policy_ids
    assert "anonymous_transfer_without_ae:xgb=2:weight=1" in policy_ids
    assert "matched_direct_mlp:256-32:typed:weight=0.25" in policy_ids


def test_starved_control_search_budget_is_rejected_at_plan_time() -> None:
    raw = copy.deepcopy(yaml.safe_load(PRODUCTION_CONFIG.read_text()))
    raw["methods"]["ridge_alphas"] = [1.0]
    raw["methods"]["control_latent_dims"] = [16, 32]

    with pytest.raises(ValueError, match="fairness floor"):
        _candidate_configs(raw)


def test_missing_canonical_ae_architecture_is_rejected() -> None:
    raw = copy.deepcopy(yaml.safe_load(PRODUCTION_CONFIG.read_text()))
    for candidate in raw["methods"]["ae_candidates"]:
        if candidate["candidate_id"] == "ae-256-32-anonymous":
            candidate["hidden_dim"] = 64
            candidate["latent_dim"] = 8

    # A proper-subset test would have accepted {(128, 16), (64, 8)}.
    with pytest.raises(ValueError, match="architectures are incomplete"):
        _candidate_configs(raw)


def test_matched_direct_mlp_is_protocol_selected_like_the_ae() -> None:
    production = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    protocols: dict[str, set[str]] = {}
    for _policy_id, config in _candidate_configs(production):
        protocols.setdefault(config.method_family, set()).add(
            config.data_protocol
        )

    assert protocols["matched_direct_mlp"] == {"real_only", "anonymous", "typed"}
    assert protocols["redesigned_supervised_ae"] == protocols["matched_direct_mlp"]
    for family in ("truncated_svd", "linear_autoencoder", "direct_xgboost"):
        assert protocols[family] == {"real_only"}


def test_plan_and_manifest_record_the_search_budget(
    completed_bundle: Path,
) -> None:
    plan = json.loads((completed_bundle / "benchmark_plan.json").read_text())
    manifest = json.loads((completed_bundle / "manifest.json").read_text())
    candidates = pd.read_csv(completed_bundle / "candidate_policies.csv")

    assert plan["search_budget_by_family"] == manifest["search_budget_by_family"]
    assert plan["minimum_control_search_budget"] == MINIMUM_CONTROL_SEARCH_BUDGET
    assert plan["protocol_selected_families"] == [
        "matched_direct_mlp",
        "redesigned_supervised_ae",
    ]
    for _unit, rows in candidates.groupby("evaluation_unit"):
        observed = search_budget_by_family(rows["method_family"].astype(str))
        assert observed == plan["search_budget_by_family"]


def _identity_fields(**overrides: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": benchmark_module.REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
        "status": "frozen_before_outer_test",
        "evaluation_unit": "seed=0|fraction=0.2",
        "method_family": "redesigned_supervised_ae",
        "policy_id": "ae-real-only",
        "selected_policy_hash": "1" * 64,
        "resolved_config": {"method_family": "redesigned_supervised_ae"},
        "plan_hash": "2" * 64,
        "config_hash": "3" * 64,
        "dataset_hash": "4" * 64,
        "canonical_split_hash": "5" * 64,
        "split_hash": "6" * 64,
        "feature_metadata_hash": "7" * 64,
    }
    payload.update(overrides)
    return payload


_RETENTION_CONFIG = {"practical_margin": 1.0, "seeds": [0, 1, 2]}


def _registry_key(
    policy: dict[str, object],
    *,
    implementation_commit: str = "a" * 40,
    search_evidence_hash: str = "8" * 64,
    placement_evidence_hash: str = "9" * 64,
) -> str:
    document = {
        **policy,
        "implementation_commit": implementation_commit,
        "search_prediction_hash": "b" * 64,
        "search_evidence_hash": search_evidence_hash,
    }
    document["frozen_policy_hash"] = stable_hash(
        frozen_policy_identity_payload(document)
    )
    del placement_evidence_hash  # never an identity component
    anchor = benchmark_module._final_evaluation_anchor_hash(
        plan_hash=str(policy["plan_hash"]),
        config_hash=str(policy["config_hash"]),
        dataset_hash=str(policy["dataset_hash"]),
        canonical_split_hash=str(policy["canonical_split_hash"]),
        retention_config=_RETENTION_CONFIG,
    )
    return EvaluationIdentity(
        dataset_hash=str(policy["dataset_hash"]),
        split_or_search_manifest_hash=anchor,
        frozen_policy_hash=str(document["frozen_policy_hash"]),
        evaluation_unit=(
            f"{policy['evaluation_unit']}|family={policy['method_family']}"
        ),
    ).registry_key


def test_evaluation_identity_ignores_commit_and_float_evidence() -> None:
    baseline = _registry_key(_identity_fields())

    # An unrelated commit must not mint a fresh outer-test evaluation identity.
    assert _registry_key(
        _identity_fields(), implementation_commit="f" * 40
    ) == baseline
    # Nor may a 1-ULP difference in any float-derived evidence hash.
    assert _registry_key(
        _identity_fields(), search_evidence_hash="c" * 64
    ) == baseline
    assert _registry_key(
        _identity_fields(), placement_evidence_hash="d" * 64
    ) == baseline


def test_evaluation_identity_ignores_which_policy_won_the_search() -> None:
    """One outer-test evaluation per declared unit and family, full stop.

    The winning policy is itself a function of float inner-validation metrics,
    so binding it into the identity would let any change that merely reordered
    the search mint a fresh claim on an already-consumed outer-test unit. The
    declared experiment is pinned by ``plan_hash`` and ``config_hash``, both of
    which already cover the entire candidate grid.
    """
    baseline = _registry_key(_identity_fields())

    assert _registry_key(_identity_fields(policy_id="ae-typed")) == baseline
    assert _registry_key(_identity_fields(selected_policy_hash="e" * 64)) == baseline
    assert (
        _registry_key(
            _identity_fields(resolved_config={"method_family": "truncated_svd"})
        )
        == baseline
    )


@pytest.mark.parametrize(
    "field",
    [
        "dataset_hash",
        "canonical_split_hash",
        "split_hash",
        "config_hash",
        "plan_hash",
        "method_family",
        "evaluation_unit",
    ],
)
def test_evaluation_identity_tracks_declared_scientific_scope(field: str) -> None:
    baseline = _registry_key(_identity_fields())
    changed = _identity_fields(
        **{
            field: (
                "e" * 64
                if field.endswith("_hash")
                else f"{_identity_fields()[field]}-changed"
            )
        }
    )

    assert _registry_key(changed) != baseline


def test_frozen_policy_identity_payload_excludes_provenance(
    completed_bundle: Path,
) -> None:
    frozen = json.loads(
        (completed_bundle / "frozen_method_policies.json").read_text()
    )
    for policy in frozen["policies"]:
        assert policy["implementation_commit"]
        payload = frozen_policy_identity_payload(policy)
        assert "implementation_commit" not in payload
        assert "search_evidence_hash" not in payload
        assert "search_prediction_hash" not in payload
        assert policy["frozen_policy_hash"] == stable_hash(payload)


def test_transfer_pool_keys_never_consume_validation_or_test_identities(
    completed_bundle: Path,
) -> None:
    partitions = pd.read_csv(completed_bundle / "partition_assignments.csv")
    summary = pd.read_csv(completed_bundle / "pool_summary.csv")
    dataset = pd.read_csv(
        DATASET, usecols=["source_row_id", "canonical_reaction_key"]
    )
    keys = {
        str(row["source_row_id"]): str(row["canonical_reaction_key"])
        for row in dataset.to_dict(orient="records")
    }

    def role_keys(unit: str, roles: list[str]) -> set[str]:
        rows = partitions.loc[
            partitions["evaluation_unit"].astype(str).eq(unit)
            & partitions["data_role"].isin(roles)
        ]
        return {keys[str(value)] for value in rows["source_row_id"]}

    assert summary["measured_identity_key_count"].gt(0).all()
    for row in summary.to_dict(orient="records"):
        unit = str(row["evaluation_unit"])
        visible = (
            ["policy_fit"]
            if row["phase"] == "search"
            else ["policy_fit", "policy_holdout"]
        )
        expected = role_keys(unit, visible)
        assert row["measured_identity_key_count"] == len(expected)
        assert row["measured_identity_key_hash"] == stable_hash(
            sorted(expected)
        )
        withheld = role_keys(unit, ["saved_validation", "outer_test"]) - expected
        assert not (expected & withheld)


def test_zero_row_transfer_pools_are_flagged(completed_bundle: Path) -> None:
    summary = pd.read_csv(completed_bundle / "pool_summary.csv")

    assert "degenerate_pool" in summary
    assert summary["degenerate_pool"].astype(bool).eq(
        summary["selected_count"].eq(0)
    ).all()
    assert not summary["degenerate_pool"].astype(bool).any()


def test_structural_failure_precedes_any_outer_test_reservation(
    structurally_invalid_run: tuple[Path, BaseException],
) -> None:
    registry, error = structurally_invalid_run
    records = sorted(Path(registry).rglob("*.json"))

    assert "inner-selection" in str(error)
    assert records == []


def test_bundle_validates_deterministically_and_is_immutable(
    completed_bundle: Path,
) -> None:
    first = validate_redesigned_ae_benchmark(completed_bundle)
    second = validate_redesigned_ae_benchmark(completed_bundle)
    assert first == second
    with pytest.raises(FileExistsError, match="overwrite"):
        run_redesigned_ae_benchmark(
            SMOKE_CONFIG, output_directory=completed_bundle
        )


def test_inner_selection_and_refits_are_leakage_safe(
    completed_bundle: Path,
) -> None:
    partitions = pd.read_csv(completed_bundle / "partition_assignments.csv")
    candidates = pd.read_csv(completed_bundle / "candidate_policies.csv")
    metrics = pd.read_csv(completed_bundle / "search_metrics.csv")
    refit = pd.read_csv(completed_bundle / "refit_audit.csv")
    assert set(candidates["method_family"]) == set(PHASE14_METHOD_FAMILIES)
    assert set(metrics["data_role"]) == {"inner_policy_validation"}
    assert not metrics["test_used_for_selection_or_retention"].astype(bool).any()
    winners = metrics.loc[
        metrics["metric"].eq("rmse")
        & metrics["selected_within_family"].astype(bool)
    ]
    assert winners.groupby(["evaluation_unit", "method_family"]).size().eq(1).all()
    assert not refit["evaluation_labels_received"].astype(bool).any()
    assert refit["fit_forbidden_overlap_count"].eq(0).all()
    assert refit["expected_measured_source_id_hash"].equals(
        refit["measured_fit_source_id_hash"]
    )
    # Boundaries are per evaluation unit: one unit's training rows are legally
    # another unit's test rows, so pooling identities across units is meaningless.
    pool_audit = pd.read_csv(completed_bundle / "pool_candidate_audit.csv")
    assert set(pool_audit["evaluation_unit"].astype(str)) == set(
        partitions["evaluation_unit"].astype(str)
    )
    for unit, rows in pool_audit.groupby("evaluation_unit"):
        parents: set[str] = set()
        for column in ("source_row_id", "donor_row_id"):
            parents.update(rows[column].dropna().astype(str))
        test_ids = set(
            partitions.loc[
                partitions["evaluation_unit"].astype(str).eq(str(unit))
                & partitions["data_role"].eq("outer_test"),
                "source_row_id",
            ].astype(str)
        )
        validation_ids = set(
            partitions.loc[
                partitions["evaluation_unit"].astype(str).eq(str(unit))
                & partitions["data_role"].eq("saved_validation"),
                "source_row_id",
            ].astype(str)
        )
        assert parents
        assert not (parents & test_ids)
        assert not (parents & validation_ids)


def test_shared_transfer_pool_parity_and_ae_validation_parent_exclusion(
    completed_bundle: Path,
) -> None:
    refit = pd.read_csv(completed_bundle / "refit_audit.csv")
    summary = pd.read_csv(completed_bundle / "pool_summary.csv")
    assert summary["parents_exclude_internal_validation"].astype(bool).all()
    for protocol, families in {
        "anonymous": {
            "redesigned_supervised_ae",
            "anonymous_transfer_without_ae",
        },
        "typed": {"redesigned_supervised_ae", "typed_transfer_without_ae"},
    }.items():
        rows = refit.loc[refit["data_protocol"].eq(protocol)]
        assert families <= set(rows["method_family"])
        assert (
            rows.groupby(["evaluation_unit", "phase"])["pool_hash"]
            .nunique()
            .eq(1)
            .all()
        )


def test_retention_is_frozen_before_exactly_once_outer_test(
    completed_bundle: Path,
) -> None:
    retention = json.loads((completed_bundle / "retention_decision.json").read_text())
    claims = pd.read_csv(completed_bundle / "evaluation_claims.csv")
    predictions = pd.read_csv(completed_bundle / "final_predictions.csv")
    metrics = pd.read_csv(completed_bundle / "final_test_metrics.csv")
    assert retention["placement"] in {"primary_benchmark", "secondary_ablation"}
    assert claims.groupby(["evaluation_unit", "method_family"]).size().eq(1).all()
    assert claims["outer_test_prediction_batches"].eq(1).all()
    assert predictions["test_evaluation_count"].eq(1).all()
    assert metrics["test_evaluation_count"].eq(1).all()
    assert not predictions["test_used_for_selection_or_retention"].astype(bool).any()
    assert not metrics["test_used_for_selection_or_retention"].astype(bool).any()
    assert metrics.groupby(["evaluation_unit", "method_family", "metric"]).size().eq(
        1
    ).all()


def test_reconstruction_plan_is_binary_and_excludes_count_aware(
    completed_bundle: Path,
) -> None:
    plan = json.loads((completed_bundle / "benchmark_plan.json").read_text())
    assert plan["count_aware_reconstruction"] == "excluded_not_run"
    assert (
        plan["count_aware_exclusion_reason"]
        == CANONICAL_COUNT_AWARE_EXCLUSION_REASON
    )
    assert set(plan["reconstruction_contracts"]) == {
        "binary_cross_entropy",
        "positive_bit_weighted_mse",
    }
    assert all(
        not contract["count_aware_applicable"]
        for contract in plan["reconstruction_contracts"].values()
    )
    candidates = pd.read_csv(completed_bundle / "candidate_policies.csv")
    ae_candidates = candidates.loc[
        candidates["method_family"].eq("redesigned_supervised_ae")
    ]
    assert ae_candidates["reconstruction_contract_hash"].notna().all()
    assert all(
        set(audit["block_domains_verified"])
        == {
            "reactant_1",
            "reactant_2",
            "catalyst",
            "ligand",
            "base",
            "solvent_or_additive",
            "product",
        }
        for audit in plan["reconstruction_domain_audits"].values()
    )


def test_semantically_rehashed_leakage_tamper_fails(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(completed_bundle, tampered)
    path = tampered / "refit_audit.csv"
    refit = pd.read_csv(path)
    refit.loc[0, "fit_forbidden_overlap_count"] = 1
    refit.to_csv(path, index=False)
    _rehash_output(tampered, "refit_audit.csv")
    with pytest.raises(ValueError, match="fit boundary"):
        validate_redesigned_ae_benchmark(tampered)


def test_semantically_rehashed_reconstruction_row_tamper_fails(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "reconstruction-tampered"
    shutil.copytree(completed_bundle, tampered)
    path = tampered / "reconstruction_metrics.csv"
    reconstruction = pd.read_csv(path).iloc[1:].reset_index(drop=True)
    reconstruction.to_csv(path, index=False)
    _rehash_output(tampered, "reconstruction_metrics.csv")
    _set_manifest_row_count(
        tampered, "reconstruction_metrics", len(reconstruction)
    )
    with pytest.raises(ValueError, match="reconstruction row coverage"):
        validate_redesigned_ae_benchmark(tampered)


def test_semantically_rehashed_reconstruction_value_tamper_fails(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "reconstruction-value-tampered"
    shutil.copytree(completed_bundle, tampered)
    path = tampered / "reconstruction_metrics.csv"
    reconstruction = pd.read_csv(path)
    reconstruction.loc[0, "value"] = float(reconstruction.loc[0, "value"]) + 0.1
    reconstruction.to_csv(path, index=False)
    _rehash_output(tampered, "reconstruction_metrics.csv")
    with pytest.raises(ValueError, match="reconstruction fit hash"):
        validate_redesigned_ae_benchmark(tampered)


def test_semantically_rehashed_observed_outcome_tamper_fails(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "observed-outcome-tampered"
    shutil.copytree(completed_bundle, tampered)
    path = tampered / "placement_predictions.csv"
    predictions = pd.read_csv(path)
    predictions.loc[0, "observed_yield"] += 1.0
    predictions.to_csv(path, index=False)
    _rehash_output(tampered, "placement_predictions.csv")
    with pytest.raises(ValueError, match="canonical outcome replay"):
        validate_redesigned_ae_benchmark(tampered)


def test_semantically_rehashed_claim_status_tamper_fails(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "claim-tampered"
    shutil.copytree(completed_bundle, tampered)
    path = tampered / "evaluation_claims.csv"
    claims = pd.read_csv(path)
    claims.loc[0, "status"] = "complete"
    payload = {
        key: value
        for key, value in claims.iloc[0].to_dict().items()
        if key != "claim_hash"
    }
    claims.loc[0, "claim_hash"] = stable_hash(payload)
    claims.to_csv(path, index=False)
    _rehash_output(tampered, "evaluation_claims.csv")
    with pytest.raises(ValueError, match="claims"):
        validate_redesigned_ae_benchmark(tampered)


def test_fully_rehashed_retention_promotion_fails_global_anchor(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "retention-promoted"
    shutil.copytree(completed_bundle, tampered)
    predictions_path = tampered / "placement_predictions.csv"
    metrics_path = tampered / "placement_metrics.csv"
    claims_path = tampered / "evaluation_claims.csv"
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    claims = pd.read_csv(claims_path)
    ae_family = "redesigned_supervised_ae"
    prediction_mask = predictions["method_family"].eq(ae_family)
    predictions.loc[prediction_mask, "predicted_yield"] = predictions.loc[
        prediction_mask, "observed_yield"
    ]
    oracle = predictions.loc[prediction_mask, "predicted_yield"].to_numpy(float)
    observed = predictions.loc[prediction_mask, "observed_yield"].to_numpy(float)
    prediction_hash = benchmark_module._array_hash(oracle)
    predictions.loc[prediction_mask, "prediction_hash"] = prediction_hash
    metric_mask = metrics["method_family"].eq(ae_family)
    for index in metrics.index[metric_mask]:
        metric = str(metrics.loc[index, "metric"])
        metrics.loc[index, "value"] = benchmark_module._METRICS[metric](
            observed, oracle
        )
        metrics.loc[index, "prediction_hash"] = prediction_hash
    plan = json.loads((tampered / "benchmark_plan.json").read_text())
    retention_config = AERetentionConfig(
        **{
            **plan["retention_config"],
            "fractions": tuple(plan["retention_config"]["fractions"]),
            "seeds": tuple(plan["retention_config"]["seeds"]),
            "control_families": tuple(
                plan["retention_config"]["control_families"]
            ),
        }
    )
    promoted = decide_ae_retention(
        benchmark_module._retention_inputs(metrics),
        retention_config,
    )
    assert promoted["placement"] == "primary_benchmark"
    (tampered / "retention_decision.json").write_text(decision_json(promoted))
    refit = pd.read_csv(tampered / "refit_audit.csv")
    placement_evidence_hash = benchmark_module._method_evidence_hash(
        phase="placement",
        evaluation_unit=None,
        method_family=None,
        predictions=predictions,
        metrics=metrics,
        refits=refit,
    )
    final_anchor = benchmark_module._final_evaluation_anchor_hash(
        plan_hash=plan["plan_hash"],
        config_hash=plan["config_hash"],
        dataset_hash=plan["dataset_hash"],
        canonical_split_hash=plan["canonical_split_hash"],
        retention_config=plan["retention_config"],
    )
    claims["retention_decision_hash"] = promoted["decision_hash"]
    for index in claims.index:
        body = {
            key: value
            for key, value in claims.loc[index].to_dict().items()
            if key != "claim_hash"
        }
        claims.loc[index, "claim_hash"] = stable_hash(body)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    claims.to_csv(claims_path, index=False)
    for name in (
        "placement_predictions.csv",
        "placement_metrics.csv",
        "retention_decision.json",
        "evaluation_claims.csv",
    ):
        _rehash_output(tampered, name)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retention_decision_hash"] = promoted["decision_hash"]
    manifest["retention_placement"] = promoted["placement"]
    manifest["placement_evidence_hash"] = placement_evidence_hash
    manifest["final_evaluation_anchor_hash"] = final_anchor
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # The stabilized evaluation identity no longer moves with placement floats,
    # so the durable registry payload is what refuses the promotion.
    with pytest.raises(ValueError, match="registry selection-evidence linkage"):
        validate_redesigned_ae_benchmark(tampered)


def test_fully_rehashed_oracle_prediction_attack_fails_live_registry_link(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "oracle-tampered"
    shutil.copytree(completed_bundle, tampered)
    predictions_path = tampered / "final_predictions.csv"
    metrics_path = tampered / "final_test_metrics.csv"
    claims_path = tampered / "evaluation_claims.csv"
    registry_path = tampered / "test_evaluation_registry.json"
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    claims = pd.read_csv(claims_path)
    registry = json.loads(registry_path.read_text())
    for (unit, family), _rows in predictions.groupby(
        ["evaluation_unit", "method_family"], sort=False
    ):
        mask = (
            predictions["evaluation_unit"].eq(unit)
            & predictions["method_family"].eq(family)
        )
        predictions.loc[mask, "predicted_yield"] = predictions.loc[
            mask, "observed_yield"
        ]
        oracle = predictions.loc[mask, "predicted_yield"].to_numpy(float)
        prediction_hash = benchmark_module._array_hash(oracle)
        predictions.loc[mask, "prediction_hash"] = prediction_hash
        metric_mask = (
            metrics["evaluation_unit"].eq(unit)
            & metrics["method_family"].eq(family)
        )
        observed = predictions.loc[mask, "observed_yield"].to_numpy(float)
        for index in metrics.index[metric_mask]:
            metric = str(metrics.loc[index, "metric"])
            metrics.loc[index, "value"] = benchmark_module._METRICS[metric](
                observed, oracle
            )
            metrics.loc[index, "prediction_hash"] = prediction_hash
        metric_payload = json.loads(
            metrics.loc[metric_mask].to_json(orient="records")
        )
        registry_unit = f"{unit}|family={family}"
        document = next(
            record
            for record in registry["records"]
            if record["identity"]["evaluation_unit"] == registry_unit
        )
        document["prediction_hash"] = prediction_hash
        document["metrics_payload"] = metric_payload
        document["metrics_hash"] = stable_hash(metric_payload)
        document_body = {
            key: value
            for key, value in document.items()
            if key != "record_payload_hash"
        }
        document["record_payload_hash"] = stable_hash(document_body)
        claim_mask = (
            claims["evaluation_unit"].eq(unit)
            & claims["method_family"].eq(family)
        )
        claims.loc[claim_mask, "prediction_hash"] = prediction_hash
        claims.loc[claim_mask, "registry_metrics_hash"] = document[
            "metrics_hash"
        ]
        claim_index = claims.index[claim_mask][0]
        claim_body = {
            key: value
            for key, value in claims.loc[claim_index].to_dict().items()
            if key != "claim_hash"
        }
        claims.loc[claim_index, "claim_hash"] = stable_hash(claim_body)
    registry.pop("snapshot_hash")
    registry["snapshot_hash"] = stable_hash(registry)
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    claims.to_csv(claims_path, index=False)
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    summary = benchmark_module._summarize(metrics)
    summary.to_csv(tampered / "summary.csv", index=False)
    for name in (
        "final_predictions.csv",
        "final_test_metrics.csv",
        "evaluation_claims.csv",
        "test_evaluation_registry.json",
        "summary.csv",
    ):
        _rehash_output(tampered, name)
    with pytest.raises(ValueError, match="live evaluation-registry linkage"):
        validate_redesigned_ae_benchmark(tampered)
