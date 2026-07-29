"""Leakage-safe Phase 14 benchmark and validation-only AE retention placement."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.augmentation.condition_transfer import ConditionTransferConfig
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
)
from bh_augmentation.augmentation.scientific_transfer_pool import (
    ScientificTransferPool,
    build_anonymous_transfer_pool,
    build_strict_context_matched_typed_transfer_pool,
)
from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.ae_retention import (
    AE_RETENTION_CONTROL_FAMILIES,
    AERetentionConfig,
    decide_ae_retention,
    decision_json,
)
from bh_augmentation.evaluation.evaluation_registry import (
    EvaluationIdentity,
    EvaluationRegistry,
)
from bh_augmentation.evaluation.isolated_fit_worker import run_isolated_fit
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_saved_random_split_unit,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.features.reconstruction_contract import (
    CANONICAL_COUNT_AWARE_EXCLUSION_REASON,
    build_canonical_role_separated_reconstruction_contract,
)
from bh_augmentation.representations.phase14_methods import (
    PHASE14_METHOD_FAMILIES,
    PHASE14_PROTOCOL_SELECTED_FAMILIES,
    Phase14MethodConfig,
    Phase14MethodFitResult,
    fit_phase14_method,
)
from bh_augmentation.representations.redesigned_supervised_autoencoder import (
    RedesignedSupervisedAEConfig,
    RoleBlock,
    build_measured_internal_split,
)
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    sha256_file,
    stable_hash,
)

REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION = "bh-redesigned-ae-benchmark-v1"
PHASE14_AE_FAMILY = "redesigned_supervised_ae"
# Every control family must search at least this many candidate policies so
# that within-family min-over-inner-validation selection does not give the AE
# family an unmatched winner's-curse advantage.
MINIMUM_CONTROL_SEARCH_BUDGET = 3
# Recorded provenance that must never enter the outer-test evaluation identity.
_NON_IDENTITY_POLICY_FIELDS = frozenset(
    {
        "frozen_policy_hash",
        "implementation_commit",
        "search_prediction_hash",
        "search_evidence_hash",
        # The winning policy is itself a function of float search metrics, so a
        # code change that merely reordered the search could otherwise mint a
        # fresh claim on an already-consumed outer-test unit. The declared
        # experiment is pinned by plan_hash and config_hash instead.
        "policy_id",
        "selected_policy_hash",
        "resolved_config",
    }
)
_FROZEN_POLICY_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "evaluation_unit",
        "method_family",
        "plan_hash",
        "config_hash",
        "dataset_hash",
        "canonical_split_hash",
        "split_hash",
        "feature_metadata_hash",
    }
)
PHASE14_EVALUATION_REGISTRY_DIRECTORY = Path(
    "results/autonomous_execution/evaluation_registry"
)
_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman_corr}
_OUTPUTS = (
    "benchmark_plan.json",
    "split_units.csv",
    "partition_assignments.csv",
    "candidate_policies.csv",
    "pool_candidate_audit.csv",
    "pool_summary.csv",
    "search_predictions.csv",
    "search_metrics.csv",
    "placement_predictions.csv",
    "placement_metrics.csv",
    "reconstruction_metrics.csv",
    "frozen_method_policies.json",
    "retention_decision.json",
    "refit_audit.csv",
    "resource_metrics.csv",
    "test_evaluation_registry.json",
    "evaluation_claims.csv",
    "final_predictions.csv",
    "final_test_metrics.csv",
    "summary.csv",
)
_TRAINING_PROTOCOL = {
    "inner_policy_selection": "group_disjoint_subset_of_saved_train",
    "pool_sources_and_donors": "public_ae_measured_fit_rows_only",
    "placement_refit": "full_saved_train",
    "placement_evaluation": "untouched_saved_validation",
    "retention": "placement_validation_only_frozen_before_outer_test",
    "outer_test_prediction_batches_per_unit_family": 1,
}


def run_redesigned_ae_benchmark(
    config: str | Path | Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
    evaluation_registry_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run search, validation placement, freeze retention, then test once."""
    raw = _load_config(config)
    contract = _resolve_contract(raw)
    saved = load_saved_canonical_split_identities(
        contract["dataset_path"],
        contract["split_directory"],
        requested_seeds=contract["seeds"],
        requested_fractions=contract["fractions"],
    )
    units = tuple(
        build_saved_random_split_unit(saved, seed=seed, train_fraction=fraction)
        for seed in contract["seeds"]
        for fraction in contract["fractions"]
    )
    identity = saved.canonical.copy()
    identity["source_row_id"] = identity["source_row_id"].astype(str)
    identity_no_y = identity.copy()
    identity_no_y["yield"] = 0.0
    X_all, _, feature_names, feature_metadata = build_feature_matrix_with_metadata(
        identity_no_y, contract["feature_config"]
    )
    X_all = np.asarray(X_all, dtype=np.float32)
    feature_contract = feature_contract_record(feature_metadata, feature_names)
    role_blocks = tuple(
        RoleBlock(block.name, block.start, block.stop)
        for block in feature_metadata.block_slices
    )
    positions = {
        source_id: index
        for index, source_id in enumerate(identity["source_row_id"])
    }
    candidate_configs: dict[tuple[str, str], Phase14MethodConfig] = {}
    candidate_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    partitions: dict[str, dict[str, tuple[str, ...]]] = {}
    for unit in units:
        common_seed = _derived_seed(
            contract["base_seed"], unit.evaluation_unit, "common-ae-split"
        )
        split = build_measured_internal_split(
            unit.train_source_ids,
            ("measured",) * len(unit.train_source_ids),
            train_group_ids=_groups(identity, positions, unit.train_source_ids),
            valid_fraction=contract["policy_valid_fraction"],
            random_state=common_seed,
        )
        partition = {
            "policy_fit": tuple(sorted(split.measured_fit_source_ids)),
            "policy_holdout": tuple(sorted(split.measured_validation_source_ids)),
            "saved_validation": unit.validation_source_ids,
            "outer_test": unit.test_source_ids,
        }
        partitions[unit.evaluation_unit] = partition
        for role, source_ids in partition.items():
            partition_rows.extend(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "source_row_id": source_id,
                    "data_role": role,
                    "source_id_hash": stable_hash(sorted(source_ids)),
                }
                for source_id in source_ids
            )
        configs = _candidate_configs(
            unit=unit,
            contract=contract,
            input_dim=X_all.shape[1],
            role_blocks=role_blocks,
            common_seed=common_seed,
        )
        for policy_id, method_config in configs:
            key = (unit.evaluation_unit, policy_id)
            candidate_configs[key] = method_config
            record = _config_record(method_config)
            candidate_rows.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "seed": unit.seed,
                    "train_fraction": unit.train_fraction,
                    "method_family": method_config.method_family,
                    "policy_id": policy_id,
                    "data_protocol": method_config.data_protocol,
                    "resolved_config": json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    ),
                    "config_hash": stable_hash(record),
                    "policy_hash": stable_hash(
                        {
                            "evaluation_unit": unit.evaluation_unit,
                            "policy_id": policy_id,
                            "config": record,
                        }
                    ),
                    "exact_split_hash": unit.exact_split_hash,
                }
            )
    search_budget = _assert_uniform_search_budget(candidate_rows, units)
    scientific_config = _scientific_config(contract)
    reconstruction_contracts: dict[str, Any] = {}
    reconstruction_domain_audits: dict[str, Any] = {}
    ae_objectives = sorted(
        {
            config.ae_config.reconstruction_objective
            for config in candidate_configs.values()
            if config.ae_config is not None
        }
    )
    for objective in ae_objectives:
        reconstruction_contract, domain_audit = (
            build_canonical_role_separated_reconstruction_contract(
                feature_metadata, X_all, objective=objective
            )
        )
        reconstruction_contracts[objective] = reconstruction_contract.to_dict()
        reconstruction_domain_audits[objective] = asdict(domain_audit)
    for row in candidate_rows:
        method_config = candidate_configs[
            (str(row["evaluation_unit"]), str(row["policy_id"]))
        ]
        if method_config.ae_config is None:
            row["reconstruction_contract_hash"] = None
            row["observed_domain_audit_hash"] = None
            continue
        objective = method_config.ae_config.reconstruction_objective
        row["reconstruction_contract_hash"] = reconstruction_contracts[objective][
            "contract_hash"
        ]
        row["observed_domain_audit_hash"] = stable_hash(
            reconstruction_domain_audits[objective]
        )
    plan = {
        "schema_version": REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
        "status": "frozen_before_any_outcome_loading",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "feature_contract": feature_contract,
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "training_protocol": _TRAINING_PROTOCOL,
        "ae_predictor_scope": (
            "fixed_end_to_end_supervised_yield_head_ablation; "
            "does_not_supersede_phase_06_joint_ae_selection"
        ),
        "parameter_matching_scope": (
            "architecture-paired AE and direct-MLP candidates; "
            "within-family policies are selected independently"
        ),
        "method_families": PHASE14_METHOD_FAMILIES,
        "protocol_selected_families": list(PHASE14_PROTOCOL_SELECTED_FAMILIES),
        "protocol_selection_scope": (
            "matched_direct_mlp enumerates real_only/anonymous/typed candidates "
            "exactly like redesigned_supervised_ae; truncated_svd, "
            "linear_autoencoder and direct_xgboost stay real_only and their "
            "augmented counterparts are the two *_transfer_without_ae families"
        ),
        "search_budget_by_family": search_budget,
        "minimum_control_search_budget": MINIMUM_CONTROL_SEARCH_BUDGET,
        "candidate_policies": candidate_rows,
        "split_units": [unit.audit_record for unit in units],
        "resolved_scientific_config": scientific_config,
        "config_hash": stable_hash(scientific_config),
        "retention_config": asdict(contract["retention_config"]),
        "reconstruction_contracts": reconstruction_contracts,
        "reconstruction_domain_audits": reconstruction_domain_audits,
        "count_aware_reconstruction": "excluded_not_run",
        "count_aware_exclusion_reason": CANONICAL_COUNT_AWARE_EXCLUSION_REASON,
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metrics_generated": False,
    }
    plan["plan_hash"] = stable_hash(plan)

    output = Path(
        output_directory if output_directory is not None else contract["output"]
    )
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 14 output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {name.removesuffix(".json").removesuffix(".csv"): output / name for name in _OUTPUTS}
    paths["directory"] = output
    paths["manifest"] = output / "manifest.json"
    _write_json(paths["benchmark_plan"], plan)
    pd.DataFrame([unit.audit_record for unit in units]).to_csv(
        paths["split_units"], index=False
    )
    pd.DataFrame(partition_rows).to_csv(paths["partition_assignments"], index=False)
    pd.DataFrame(candidate_rows).to_csv(paths["candidate_policies"], index=False)

    pool_audits: list[dict[str, Any]] = []
    pool_summaries: list[dict[str, Any]] = []
    search_prediction_rows: list[dict[str, Any]] = []
    search_metric_rows: list[dict[str, Any]] = []
    search_fit_rows: list[dict[str, Any]] = []
    placement_prediction_rows: list[dict[str, Any]] = []
    placement_metric_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    placement_results: dict[tuple[str, str], Phase14MethodFitResult] = {}
    placement_pools: dict[tuple[str, str], ScientificTransferPool] = {}
    # Search and placement may obtain train/validation labels, never outer-test labels.
    for unit in units:
        partition = partitions[unit.evaluation_unit]
        allowed_ids = set(unit.train_source_ids) | set(unit.validation_source_ids)
        outcomes = _read_allowed_outcomes(contract["dataset_path"], allowed_ids)
        by_outcome = outcomes.set_index("source_row_id")["yield"]
        policy_fit = partition["policy_fit"]
        policy_holdout = partition["policy_holdout"]
        search_pools = _build_pools(
            phase="search",
            unit=unit,
            measured_ids=policy_fit,
            outcomes=by_outcome,
            identity=identity,
            X_all=X_all,
            positions=positions,
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            contract=contract,
            pool_audits=pool_audits,
            pool_summaries=pool_summaries,
        )
        for candidate in [
            row for row in candidate_rows
            if row["evaluation_unit"] == unit.evaluation_unit
        ]:
            policy_id = str(candidate["policy_id"])
            method_config = candidate_configs[(unit.evaluation_unit, policy_id)]
            pool = _pool_for_config(method_config, search_pools)
            execution = _isolated_method_fit(
                method_config,
                measured_ids=policy_fit,
                evaluation_ids=policy_holdout,
                outcomes=by_outcome,
                identity=identity,
                X_all=X_all,
                positions=positions,
                pool=pool,
                timeout=contract["fit_timeout_seconds"],
                evaluation_unit=unit.evaluation_unit,
                phase="search",
            )
            result = execution.scientific_result
            resource_rows.append(
                _resource_row(
                    unit, method_config, policy_id, "search", execution.resource_usage
                )
            )
            search_fit_rows.append(
                _fit_audit_row(
                    unit, method_config, policy_id, result, "search",
                    measured_ids=policy_fit,
                    forbidden_ids=set(policy_holdout)
                    | set(unit.validation_source_ids)
                    | set(unit.test_source_ids),
                    pool=pool,
                    plan_hash=plan["plan_hash"],
                )
            )
            y_holdout = by_outcome.loc[list(policy_holdout)].to_numpy(float)
            prediction_hash = result.prediction_hash
            search_prediction_rows.extend(
                _prediction_rows(
                    unit, method_config.method_family, policy_id,
                    candidate["policy_hash"], policy_holdout, y_holdout,
                    result.predictions, "inner_policy_validation",
                    prediction_hash, result.state_hash, plan["plan_hash"],
                )
            )
            search_metric_rows.extend(
                _metric_rows(
                    unit, method_config.method_family, policy_id,
                    candidate["policy_hash"], y_holdout, result.predictions,
                    "inner_policy_validation", contract["metrics"],
                    prediction_hash, result.state_hash, plan["plan_hash"],
                )
            )
            reconstruction_rows.extend(
                _reconstruction_rows(
                    unit, method_config.method_family, policy_id, "search",
                    result.reconstruction_metrics, plan["plan_hash"]
                )
            )
        unit_search = [
            row for row in search_metric_rows
            if row["evaluation_unit"] == unit.evaluation_unit
            and row["metric"] == contract["selection_metric"]
        ]
        for family in PHASE14_METHOD_FAMILIES:
            candidates = [row for row in unit_search if row["method_family"] == family]
            winner = sorted(
                candidates,
                key=lambda row: (
                    float(row["value"]), str(row["policy_hash"]), str(row["policy_id"])
                ),
            )[0]
            selected[(unit.evaluation_unit, family)] = dict(winner)
    for row in search_metric_rows:
        winner = selected[(row["evaluation_unit"], row["method_family"])]
        row["selected_within_family"] = row["policy_id"] == winner["policy_id"]
    for row in search_prediction_rows:
        winner = selected[(row["evaluation_unit"], row["method_family"])]
        row["selected_within_family"] = row["policy_id"] == winner["policy_id"]

    frozen_policies = []
    for unit in units:
        outcomes = _read_allowed_outcomes(
            contract["dataset_path"],
            set(unit.train_source_ids) | set(unit.validation_source_ids),
        ).set_index("source_row_id")["yield"]
        full_pools = _build_pools(
            phase="placement",
            unit=unit,
            measured_ids=unit.train_source_ids,
            outcomes=outcomes,
            identity=identity,
            X_all=X_all,
            positions=positions,
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            contract=contract,
            pool_audits=pool_audits,
            pool_summaries=pool_summaries,
        )
        placement_pools[(unit.evaluation_unit, "anonymous")] = full_pools["anonymous"]
        placement_pools[(unit.evaluation_unit, "typed")] = full_pools["typed"]
        for family in PHASE14_METHOD_FAMILIES:
            search_winner = selected[(unit.evaluation_unit, family)]
            policy_id = str(search_winner["policy_id"])
            method_config = candidate_configs[(unit.evaluation_unit, policy_id)]
            selected_policy_hash = str(search_winner["policy_hash"])
            # Only scientifically declared, stable quantities may enter the
            # hashed identity payload. Provenance that varies with unrelated
            # commits or with float outcome values is recorded beside it but
            # never hashed into the outer-test evaluation identity.
            identity_payload = {
                "schema_version": REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
                "status": "frozen_before_outer_test",
                "evaluation_unit": unit.evaluation_unit,
                "method_family": family,
                "plan_hash": plan["plan_hash"],
                "config_hash": plan["config_hash"],
                "dataset_hash": saved.dataset_hash,
                "canonical_split_hash": saved.aggregate_split_hash,
                "split_hash": unit.exact_split_hash,
                "feature_metadata_hash": feature_contract["feature_metadata_hash"],
            }
            frozen_policy_hash = stable_hash(identity_payload)
            frozen_payload = {
                **identity_payload,
                "policy_id": policy_id,
                "selected_policy_hash": selected_policy_hash,
                "resolved_config": _config_record(method_config),
                "search_prediction_hash": search_winner["prediction_hash"],
                "search_evidence_hash": _method_evidence_hash(
                    phase="search",
                    evaluation_unit=unit.evaluation_unit,
                    method_family=family,
                    predictions=search_prediction_rows,
                    metrics=search_metric_rows,
                    refits=search_fit_rows,
                ),
                "implementation_commit": _git_commit(),
                "frozen_policy_hash": frozen_policy_hash,
            }
            frozen_policies.append(frozen_payload)
            pool = _pool_for_config(method_config, full_pools)
            execution = _isolated_method_fit(
                method_config,
                measured_ids=unit.train_source_ids,
                evaluation_ids=unit.validation_source_ids,
                outcomes=outcomes,
                identity=identity,
                X_all=X_all,
                positions=positions,
                pool=pool,
                timeout=contract["fit_timeout_seconds"],
                evaluation_unit=unit.evaluation_unit,
                phase="placement",
            )
            result = execution.scientific_result
            placement_results[(unit.evaluation_unit, family)] = result
            resource_rows.append(
                _resource_row(
                    unit, method_config, policy_id, "placement",
                    execution.resource_usage
                )
            )
            search_fit_rows.append(
                _fit_audit_row(
                    unit, method_config, policy_id, result, "placement",
                    measured_ids=unit.train_source_ids,
                    forbidden_ids=set(unit.validation_source_ids)
                    | set(unit.test_source_ids),
                    pool=pool,
                    plan_hash=plan["plan_hash"],
                    frozen_policy_hash=frozen_policy_hash,
                )
            )
            y_valid = outcomes.loc[list(unit.validation_source_ids)].to_numpy(float)
            placement_prediction_rows.extend(
                _prediction_rows(
                    unit, family, policy_id, selected_policy_hash,
                    unit.validation_source_ids, y_valid, result.predictions,
                    "placement_validation", result.prediction_hash,
                    result.state_hash, plan["plan_hash"],
                    frozen_policy_hash=frozen_policy_hash,
                )
            )
            placement_metric_rows.extend(
                _metric_rows(
                    unit, family, policy_id, selected_policy_hash,
                    y_valid, result.predictions, "placement_validation",
                    contract["metrics"], result.prediction_hash,
                    result.state_hash, plan["plan_hash"],
                    frozen_policy_hash=frozen_policy_hash,
                )
            )
            reconstruction_rows.extend(
                _reconstruction_rows(
                    unit, family, policy_id, "placement",
                    result.reconstruction_metrics, plan["plan_hash"]
                )
            )
    frozen_document = {
        "schema_version": REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
        "status": "frozen_before_outer_test",
        "selection_scope": "within_family_inner_policy_validation",
        "cross_family_test_selection": False,
        "plan_hash": plan["plan_hash"],
        "policy_count": len(frozen_policies),
        "policies": frozen_policies,
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metrics_generated": False,
    }
    frozen_document["frozen_document_hash"] = stable_hash(frozen_document)
    _write_json(paths["frozen_method_policies"], frozen_document)

    retention_records = []
    retention_family = {family: family for family in PHASE14_METHOD_FAMILIES}
    for row in placement_metric_rows:
        if row["metric"] != "rmse":
            continue
        retention_records.append(
            {
                "evaluation_unit": row["evaluation_unit"],
                "seed": int(row["seed"]),
                "train_fraction": float(row["train_fraction"]),
                "method_family": retention_family[row["method_family"]],
                "metric": "rmse",
                "value": float(row["value"]),
                "selected_policy_hash": row["policy_hash"],
                "frozen_policy_hash": row["frozen_policy_hash"],
                "data_role": "placement_validation",
            }
        )
    retention = decide_ae_retention(retention_records, contract["retention_config"])
    paths["retention_decision"].write_text(decision_json(retention))

    # Reload the immutable selection artifacts and persist every test reservation
    # before any outer-test outcome is requested.
    frozen_for_final = _load_frozen_for_final(
        paths["frozen_method_policies"],
        expected_document_hash=frozen_document["frozen_document_hash"],
    )
    retention_for_final = _load_retention_for_final(
        paths["retention_decision"],
        expected_decision_hash=retention["decision_hash"],
    )
    placement_evidence_hash = _method_evidence_hash(
        phase="placement",
        evaluation_unit=None,
        method_family=None,
        predictions=placement_prediction_rows,
        metrics=placement_metric_rows,
        refits=search_fit_rows,
    )
    final_evaluation_anchor_hash = _final_evaluation_anchor_hash(
        plan_hash=plan["plan_hash"],
        config_hash=plan["config_hash"],
        dataset_hash=saved.dataset_hash,
        canonical_split_hash=saved.aggregate_split_hash,
        retention_config=plan["retention_config"],
    )
    claim_rows: list[dict[str, Any]] = []
    final_prediction_rows: list[dict[str, Any]] = []
    final_metric_rows: list[dict[str, Any]] = []
    frozen_by_key = {
        (row["evaluation_unit"], row["method_family"]): row
        for row in frozen_for_final["policies"]
    }
    # Structural validation must precede the reservation block: a structural
    # misconfiguration discovered after the outer test has been consumed is
    # irrecoverable, because the registry forbids reevaluation.
    validate_redesigned_ae_benchmark_structure(
        plan=plan,
        canonical_hashes={
            "dataset_hash": saved.dataset_hash,
            "canonical_split_hash": saved.aggregate_split_hash,
            "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        },
        frozen=frozen_for_final,
        split_units=pd.DataFrame([unit.audit_record for unit in units]),
        partitions=pd.DataFrame(partition_rows),
        candidates=pd.DataFrame(candidate_rows),
        pool_audit=pd.DataFrame(pool_audits),
        pool_summary=pd.DataFrame(pool_summaries),
        search_predictions=pd.DataFrame(search_prediction_rows),
        search_metrics=pd.DataFrame(search_metric_rows),
        placement_predictions=pd.DataFrame(placement_prediction_rows),
        refit=pd.DataFrame(search_fit_rows),
        reconstruction_metrics=pd.DataFrame(reconstruction_rows),
    )
    registry = EvaluationRegistry(
        evaluation_registry_directory
        if evaluation_registry_directory is not None
        else PHASE14_EVALUATION_REGISTRY_DIRECTORY
    )
    evaluation_identities = {
        (unit.evaluation_unit, family): EvaluationIdentity(
            dataset_hash=saved.dataset_hash,
            split_or_search_manifest_hash=final_evaluation_anchor_hash,
            frozen_policy_hash=frozen_by_key[
                (unit.evaluation_unit, family)
            ]["frozen_policy_hash"],
            evaluation_unit=f"{unit.evaluation_unit}|family={family}",
        )
        for unit in units
        for family in PHASE14_METHOD_FAMILIES
    }
    reserved: list[EvaluationIdentity] = []
    try:
        for identity_record in evaluation_identities.values():
            if registry.inspect(identity_record) is not None:
                raise RuntimeError(
                    "Outer-test evaluation identity was already reserved; "
                    "reevaluation is prohibited."
                )
        for identity_record in evaluation_identities.values():
            registry.reserve(identity_record)
            reserved.append(identity_record)
    except Exception as exc:
        for identity_record in reserved:
            current = registry.inspect(identity_record)
            if current is not None and current.status in {
                "reserved",
                "prediction_complete",
            }:
                registry.mark_failed_or_uncertain(
                    identity_record,
                    reason=(
                        "Batch reservation failed before outer-test access: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
        raise
    _write_registry_snapshot(
        paths["test_evaluation_registry"],
        registry,
        evaluation_identities,
        status="reserved_before_outer_test_outcome_access",
    )
    for unit in units:
        try:
            test_outcomes = _read_allowed_outcomes(
                contract["dataset_path"], set(unit.test_source_ids)
            ).set_index("source_row_id")["yield"]
            train_outcomes = _read_allowed_outcomes(
                contract["dataset_path"], set(unit.train_source_ids)
            ).set_index("source_row_id")["yield"]
            pools = {
                "anonymous": placement_pools[(unit.evaluation_unit, "anonymous")],
                "typed": placement_pools[(unit.evaluation_unit, "typed")],
            }
            for family in PHASE14_METHOD_FAMILIES:
                frozen = frozen_by_key[(unit.evaluation_unit, family)]
                policy_id = frozen["policy_id"]
                method_config = candidate_configs[(unit.evaluation_unit, policy_id)]
                pool = _pool_for_config(method_config, pools)
                execution = _isolated_method_fit(
                    method_config,
                    measured_ids=unit.train_source_ids,
                    evaluation_ids=unit.test_source_ids,
                    outcomes=train_outcomes,
                    identity=identity,
                    X_all=X_all,
                    positions=positions,
                    pool=pool,
                    timeout=contract["fit_timeout_seconds"],
                    evaluation_unit=unit.evaluation_unit,
                    phase="final",
                )
                result = execution.scientific_result
                placement_state = placement_results[
                    (unit.evaluation_unit, family)
                ].state_hash
                if result.state_hash != placement_state:
                    raise RuntimeError(
                        "Final scientific state differs from frozen placement refit."
                    )
                resource_rows.append(
                    _resource_row(
                        unit, method_config, policy_id, "final",
                        execution.resource_usage
                    )
                )
                search_fit_rows.append(
                    _fit_audit_row(
                        unit, method_config, policy_id, result, "final",
                        measured_ids=unit.train_source_ids,
                        forbidden_ids=set(unit.validation_source_ids)
                        | set(unit.test_source_ids),
                        pool=pool,
                        plan_hash=plan["plan_hash"],
                        frozen_policy_hash=frozen["frozen_policy_hash"],
                    )
                )
                y_test = test_outcomes.loc[list(unit.test_source_ids)].to_numpy(float)
                unit_prediction_rows = _prediction_rows(
                    unit, family, policy_id, frozen["selected_policy_hash"],
                    unit.test_source_ids, y_test, result.predictions, "test",
                    result.prediction_hash, result.state_hash, plan["plan_hash"],
                    frozen_policy_hash=frozen["frozen_policy_hash"],
                    test_evaluation_count=1,
                )
                unit_metric_rows = _metric_rows(
                    unit, family, policy_id, frozen["selected_policy_hash"],
                    y_test, result.predictions, "test", contract["metrics"],
                    result.prediction_hash, result.state_hash, plan["plan_hash"],
                    frozen_policy_hash=frozen["frozen_policy_hash"],
                    test_evaluation_count=1,
                )
                evaluation_identity = evaluation_identities[
                    (unit.evaluation_unit, family)
                ]
                registry.mark_prediction_complete(
                    evaluation_identity,
                    prediction_hash=result.prediction_hash,
                )
                registry_record = registry.mark_metrics_complete(
                    evaluation_identity,
                    metrics_payload=[
                        {
                            "payload_role": "outer_metric",
                            "metric_row": row,
                        }
                        for row in unit_metric_rows
                    ]
                    + [
                        {
                            "payload_role": "final_fit_evidence",
                            "state_hash": result.state_hash,
                            "reconstruction_metrics_hash": (
                                result.reconstruction_metrics_hash
                            ),
                            "training_parameter_count": (
                                result.training_parameter_count
                            ),
                            "deployed_predictor_parameter_count": (
                                result.deployed_predictor_parameter_count
                            ),
                        }
                    ]
                    + [
                        {
                            "payload_role": "frozen_selection_evidence",
                            "frozen_document_hash": frozen_for_final[
                                "frozen_document_hash"
                            ],
                            "retention_decision_hash": retention_for_final[
                                "decision_hash"
                            ],
                            "placement_evidence_hash": placement_evidence_hash,
                        }
                    ],
                )
                _write_registry_snapshot(
                    paths["test_evaluation_registry"],
                    registry,
                    evaluation_identities,
                    status="outer_test_evaluation_in_progress",
                )
                final_prediction_rows.extend(unit_prediction_rows)
                final_metric_rows.extend(unit_metric_rows)
                claim_payload = {
                    "evaluation_unit": unit.evaluation_unit,
                    "method_family": family,
                    "frozen_policy_hash": frozen["frozen_policy_hash"],
                    "retention_decision_hash": retention_for_final["decision_hash"],
                    "registry_key": registry_record.registry_key,
                    "registry_metrics_hash": registry_record.metrics_hash,
                    "status": "metrics_complete",
                    "outer_test_prediction_batches": 1,
                    "prediction_hash": result.prediction_hash,
                    "state_hash": result.state_hash,
                }
                claim_rows.append(
                    {
                        **claim_payload,
                        "claim_hash": stable_hash(claim_payload),
                    }
                )
                reconstruction_rows.extend(
                    _reconstruction_rows(
                        unit, family, policy_id, "final",
                        result.reconstruction_metrics, plan["plan_hash"]
                    )
                )
        except Exception as exc:
            for family in PHASE14_METHOD_FAMILIES:
                evaluation_identity = evaluation_identities[
                    (unit.evaluation_unit, family)
                ]
                current = registry.inspect(evaluation_identity)
                if current is not None and current.status in {
                    "reserved",
                    "prediction_complete",
                }:
                    registry.mark_failed_or_uncertain(
                        evaluation_identity,
                        reason=(
                            "Outer-test access or evaluation failed and must not "
                            f"be retried: {type(exc).__name__}: {exc}"
                        ),
                    )
            _write_registry_snapshot(
                paths["test_evaluation_registry"],
                registry,
                evaluation_identities,
                status="failed_or_uncertain",
            )
            raise
    _write_registry_snapshot(
        paths["test_evaluation_registry"],
        registry,
        evaluation_identities,
        status="complete",
    )

    tables = {
        "pool_candidate_audit": pd.DataFrame(pool_audits),
        "pool_summary": pd.DataFrame(pool_summaries),
        "search_predictions": pd.DataFrame(search_prediction_rows),
        "search_metrics": pd.DataFrame(search_metric_rows),
        "placement_predictions": pd.DataFrame(placement_prediction_rows),
        "placement_metrics": pd.DataFrame(placement_metric_rows),
        "reconstruction_metrics": pd.DataFrame(reconstruction_rows),
        "refit_audit": pd.DataFrame(search_fit_rows),
        "resource_metrics": pd.DataFrame(resource_rows),
        "evaluation_claims": pd.DataFrame(claim_rows),
        "final_predictions": pd.DataFrame(final_prediction_rows),
        "final_test_metrics": pd.DataFrame(final_metric_rows),
    }
    tables["summary"] = _summarize(tables["final_test_metrics"])
    for name, frame in tables.items():
        frame.to_csv(paths[name], index=False)
    output_hashes = {name: sha256_file(output / name) for name in _OUTPUTS}
    manifest = {
        "schema_version": REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
        "status": "corrected_revalidation_complete",
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "config_hash": plan["config_hash"],
        "plan_hash": plan["plan_hash"],
        "frozen_document_hash": frozen_document["frozen_document_hash"],
        "retention_decision_hash": retention_for_final["decision_hash"],
        "retention_placement": retention_for_final["placement"],
        "placement_evidence_hash": placement_evidence_hash,
        "final_evaluation_anchor_hash": final_evaluation_anchor_hash,
        "method_family_count": len(PHASE14_METHOD_FAMILIES),
        "evaluation_unit_count": len(units),
        "candidate_count": len(candidate_rows),
        "search_budget_by_family": search_budget,
        "minimum_control_search_budget": MINIMUM_CONTROL_SEARCH_BUDGET,
        "selection_scope": "within_family_inner_policy_validation",
        "retention_data_role": "placement_validation",
        "outer_test_labels_accessed_after_retention_freeze": True,
        "test_used_for_selection_or_retention": False,
        "test_evaluation_count_per_unit_family": 1,
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "dependency_versions": _dependency_versions(),
        "resolved_scientific_config": scientific_config,
        "command": _command(config, output),
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    _write_json(paths["manifest"], manifest)
    validate_redesigned_ae_benchmark(output)
    return paths


def validate_redesigned_ae_benchmark(directory: str | Path) -> dict[str, Any]:
    """Validate hashes and replay leakage, selection, retention, and metrics."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed = manifest.pop("manifest_hash", None)
    if claimed != stable_hash(manifest):
        raise ValueError("Phase 14 manifest hash mismatch.")
    manifest["manifest_hash"] = claimed
    if (
        set(manifest)
        != {
            "schema_version",
            "status",
            "git_commit",
            "git_dirty_at_execution",
            "dataset_hash",
            "canonical_split_hash",
            "feature_metadata_hash",
            "config_hash",
            "plan_hash",
            "frozen_document_hash",
            "retention_decision_hash",
            "retention_placement",
            "placement_evidence_hash",
            "final_evaluation_anchor_hash",
            "method_family_count",
            "evaluation_unit_count",
            "candidate_count",
            "search_budget_by_family",
            "minimum_control_search_budget",
            "selection_scope",
            "retention_data_role",
            "outer_test_labels_accessed_after_retention_freeze",
            "test_used_for_selection_or_retention",
            "test_evaluation_count_per_unit_family",
            "row_counts",
            "dependency_versions",
            "resolved_scientific_config",
            "command",
            "output_hashes",
            "manifest_hash",
        }
        or manifest.get("schema_version")
        != REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION
        or manifest.get("status") != "corrected_revalidation_complete"
        or manifest.get("selection_scope")
        != "within_family_inner_policy_validation"
        or manifest.get("retention_data_role") != "placement_validation"
        or manifest.get("test_used_for_selection_or_retention") is not False
        or manifest.get("test_evaluation_count_per_unit_family") != 1
        or set(manifest.get("output_hashes", {})) != set(_OUTPUTS)
    ):
        raise ValueError("Phase 14 manifest contract mismatch.")
    for name, digest in manifest["output_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Phase 14 output hash mismatch: {name}.")
    plan = json.loads((root / "benchmark_plan.json").read_text())
    plan_hash = plan.pop("plan_hash", None)
    if (
        set(plan)
        != {
            "schema_version",
            "status",
            "dataset_hash",
            "canonical_split_hash",
            "feature_contract",
            "feature_metadata_hash",
            "training_protocol",
            "ae_predictor_scope",
            "parameter_matching_scope",
            "method_families",
            "protocol_selected_families",
            "protocol_selection_scope",
            "search_budget_by_family",
            "minimum_control_search_budget",
            "candidate_policies",
            "split_units",
            "resolved_scientific_config",
            "config_hash",
            "retention_config",
            "reconstruction_contracts",
            "reconstruction_domain_audits",
            "count_aware_reconstruction",
            "count_aware_exclusion_reason",
            "outer_test_labels_accessed",
            "outer_test_predictions_generated",
            "outer_test_metrics_generated",
        }
        or plan_hash != stable_hash(plan)
        or plan.get("status") != "frozen_before_any_outcome_loading"
        or plan.get("outer_test_labels_accessed") is not False
        or plan.get("outer_test_predictions_generated") is not False
        or plan.get("outer_test_metrics_generated") is not False
        or tuple(plan.get("method_families", ())) != PHASE14_METHOD_FAMILIES
        or tuple(plan.get("protocol_selected_families", ()))
        != PHASE14_PROTOCOL_SELECTED_FAMILIES
        or plan.get("minimum_control_search_budget")
        != MINIMUM_CONTROL_SEARCH_BUDGET
        or plan.get("count_aware_reconstruction") != "excluded_not_run"
        or plan.get("count_aware_exclusion_reason")
        != CANONICAL_COUNT_AWARE_EXCLUSION_REASON
        or set(plan.get("reconstruction_contracts", {}))
        != {"binary_cross_entropy", "positive_bit_weighted_mse"}
        or any(
            item.get("count_aware_applicable") is not False
            for item in plan.get("reconstruction_contracts", {}).values()
        )
    ):
        raise ValueError("Phase 14 frozen plan mismatch.")
    plan["plan_hash"] = plan_hash
    if (
        manifest["plan_hash"] != plan_hash
        or manifest["config_hash"] != plan["config_hash"]
        or manifest["resolved_scientific_config"]
        != plan["resolved_scientific_config"]
    ):
        raise ValueError("Phase 14 plan-manifest linkage mismatch.")
    read = {"float_precision": "round_trip"}
    split_units = pd.read_csv(root / "split_units.csv", **read)
    partitions = pd.read_csv(root / "partition_assignments.csv", **read)
    candidates = pd.read_csv(root / "candidate_policies.csv", **read)
    pool_audit = pd.read_csv(root / "pool_candidate_audit.csv", **read)
    pool_summary = pd.read_csv(root / "pool_summary.csv", **read)
    search_predictions = pd.read_csv(root / "search_predictions.csv", **read)
    search_metrics = pd.read_csv(root / "search_metrics.csv", **read)
    placement_predictions = pd.read_csv(root / "placement_predictions.csv", **read)
    placement_metrics = pd.read_csv(root / "placement_metrics.csv", **read)
    reconstruction_metrics = pd.read_csv(
        root / "reconstruction_metrics.csv", **read
    )
    refit = pd.read_csv(root / "refit_audit.csv", **read)
    resources = pd.read_csv(root / "resource_metrics.csv", **read)
    claims = pd.read_csv(root / "evaluation_claims.csv", **read)
    final_predictions = pd.read_csv(root / "final_predictions.csv", **read)
    final_metrics = pd.read_csv(root / "final_test_metrics.csv", **read)
    summary = pd.read_csv(root / "summary.csv", **read)
    for name, frame in {
        "pool_candidate_audit": pool_audit,
        "pool_summary": pool_summary,
        "search_predictions": search_predictions,
        "search_metrics": search_metrics,
        "placement_predictions": placement_predictions,
        "placement_metrics": placement_metrics,
        "reconstruction_metrics": reconstruction_metrics,
        "refit_audit": refit,
        "resource_metrics": resources,
        "evaluation_claims": claims,
        "final_predictions": final_predictions,
        "final_test_metrics": final_metrics,
        "summary": summary,
    }.items():
        if int(manifest["row_counts"][name]) != len(frame):
            raise ValueError(f"Phase 14 row count mismatch: {name}.")
    if (
        len(split_units) != manifest["evaluation_unit_count"]
        or len(candidates) != manifest["candidate_count"]
        or candidates.groupby(["evaluation_unit", "method_family"]).size().min() < 1
    ):
        raise ValueError("Phase 14 split/candidate coverage mismatch.")
    frozen = json.loads((root / "frozen_method_policies.json").read_text())
    frozen_hash = frozen.pop("frozen_document_hash", None)
    if (
        set(frozen)
        != {
            "schema_version",
            "status",
            "selection_scope",
            "cross_family_test_selection",
            "plan_hash",
            "policy_count",
            "policies",
            "outer_test_labels_accessed",
            "outer_test_predictions_generated",
            "outer_test_metrics_generated",
        }
        or frozen_hash != stable_hash(frozen)
        or frozen.get("status") != "frozen_before_outer_test"
        or frozen.get("outer_test_labels_accessed") is not False
        or frozen.get("outer_test_predictions_generated") is not False
        or frozen.get("outer_test_metrics_generated") is not False
        or len(frozen.get("policies", ()))
        != manifest["evaluation_unit_count"] * len(PHASE14_METHOD_FAMILIES)
    ):
        raise ValueError("Phase 14 frozen policy contract mismatch.")
    frozen["frozen_document_hash"] = frozen_hash
    validate_redesigned_ae_benchmark_structure(
        plan=plan,
        canonical_hashes={
            "dataset_hash": manifest["dataset_hash"],
            "canonical_split_hash": manifest["canonical_split_hash"],
            "feature_metadata_hash": manifest["feature_metadata_hash"],
        },
        frozen=frozen,
        split_units=split_units,
        partitions=partitions,
        candidates=candidates,
        pool_audit=pool_audit,
        pool_summary=pool_summary,
        search_predictions=search_predictions,
        search_metrics=search_metrics,
        placement_predictions=placement_predictions,
        refit=refit,
        reconstruction_metrics=reconstruction_metrics,
        final_predictions=final_predictions,
    )
    retention = json.loads((root / "retention_decision.json").read_text())
    retention_hash = retention.pop("decision_hash", None)
    if retention_hash != stable_hash(retention):
        raise ValueError("Phase 14 retention hash mismatch.")
    retention["decision_hash"] = retention_hash
    retention_inputs = _retention_inputs(placement_metrics)
    retention_config = AERetentionConfig(**{
        **plan["retention_config"],
        "fractions": tuple(plan["retention_config"]["fractions"]),
        "seeds": tuple(plan["retention_config"]["seeds"]),
        "control_families": tuple(plan["retention_config"]["control_families"]),
    })
    replayed_retention = decide_ae_retention(retention_inputs, retention_config)
    if stable_hash(retention) != stable_hash(replayed_retention):
        raise ValueError("Phase 14 retention replay mismatch.")
    if manifest["retention_decision_hash"] != retention_hash:
        raise ValueError("Phase 14 retention manifest linkage mismatch.")
    replayed_placement_evidence_hash = _method_evidence_hash(
        phase="placement",
        evaluation_unit=None,
        method_family=None,
        predictions=placement_predictions,
        metrics=placement_metrics,
        refits=refit,
    )
    replayed_final_anchor = _final_evaluation_anchor_hash(
        plan_hash=plan["plan_hash"],
        config_hash=plan["config_hash"],
        dataset_hash=manifest["dataset_hash"],
        canonical_split_hash=manifest["canonical_split_hash"],
        retention_config=plan["retention_config"],
    )
    if (
        manifest["placement_evidence_hash"]
        != replayed_placement_evidence_hash
        or manifest["final_evaluation_anchor_hash"] != replayed_final_anchor
        or manifest["frozen_document_hash"] != frozen_hash
    ):
        raise ValueError("Phase 14 placement/final evidence anchor mismatch.")
    registry_snapshot = _assert_registry_snapshot(
        root / "test_evaluation_registry.json",
        frozen,
        final_predictions,
        final_metrics,
        refit,
        final_evaluation_anchor_hash=replayed_final_anchor,
        selection_evidence={
            "frozen_document_hash": frozen_hash,
            "retention_decision_hash": retention_hash,
            "placement_evidence_hash": replayed_placement_evidence_hash,
        },
    )
    _assert_claims(
        claims,
        frozen,
        retention_hash,
        final_predictions,
        final_metrics,
        registry_snapshot,
    )
    _assert_prediction_metric_replay(search_predictions, search_metrics)
    _assert_prediction_metric_replay(placement_predictions, placement_metrics)
    _assert_prediction_metric_replay(final_predictions, final_metrics)
    if (
        not final_predictions["data_role"].eq("test").all()
        or not final_metrics["data_role"].eq("test").all()
        or not final_predictions["test_evaluation_count"].eq(1).all()
        or not final_metrics["test_evaluation_count"].eq(1).all()
        or final_predictions["test_used_for_selection_or_retention"].astype(bool).any()
        or final_metrics["test_used_for_selection_or_retention"].astype(bool).any()
    ):
        raise ValueError("Phase 14 exactly-once final test gate failed.")
    expected_summary = _summarize(final_metrics)
    if not _frames_close(summary, expected_summary):
        raise ValueError("Phase 14 summary replay mismatch.")
    if (
        not np.isfinite(resources["elapsed_seconds"].to_numpy(float)).all()
        or (resources["elapsed_seconds"] < 0).any()
        or (resources["rss_peak_bytes"] < resources["rss_baseline_bytes"]).any()
        or not (
            resources["rss_increment_bytes"]
            == resources["rss_peak_bytes"] - resources["rss_baseline_bytes"]
        ).all()
    ):
        raise ValueError("Phase 14 resource metric contract mismatch.")
    return manifest


def validate_redesigned_ae_benchmark_structure(
    *,
    plan: Mapping[str, Any],
    canonical_hashes: Mapping[str, str],
    frozen: Mapping[str, Any],
    split_units: pd.DataFrame,
    partitions: pd.DataFrame,
    candidates: pd.DataFrame,
    pool_audit: pd.DataFrame,
    pool_summary: pd.DataFrame,
    search_predictions: pd.DataFrame,
    search_metrics: pd.DataFrame,
    placement_predictions: pd.DataFrame,
    refit: pd.DataFrame,
    reconstruction_metrics: pd.DataFrame,
    final_predictions: pd.DataFrame | None = None,
) -> None:
    """Replay every structural guarantee that needs no outer-test output.

    This runs *before* the outer-test registry reservation so a structural
    misconfiguration cannot irrecoverably consume the held-out test set, and
    again as part of the complete post-run validation.
    """
    prediction_frames: list[tuple[pd.DataFrame, str]] = [
        (search_predictions, "inner_policy_validation"),
        (placement_predictions, "placement_validation"),
    ]
    if final_predictions is not None:
        prediction_frames.append((final_predictions, "test"))
    _assert_canonical_dependency_replay(
        plan,
        canonical_hashes,
        split_units,
        partitions,
        [frame for frame, _ in prediction_frames],
    )
    _assert_candidate_plan_replay(candidates, plan)
    _assert_search_budget_replay(candidates, plan)
    _assert_prediction_partitions(partitions, prediction_frames)
    _assert_fit_boundaries(refit, partitions, split_units)
    _assert_pool_boundaries(pool_audit, pool_summary, partitions, split_units)
    _assert_selection(search_predictions, search_metrics, candidates, plan)
    _assert_frozen_policy_replay(
        frozen,
        candidates,
        plan,
        canonical_hashes,
        search_predictions,
        search_metrics,
        refit,
    )
    _assert_reconstruction_metrics(reconstruction_metrics, candidates, refit, plan)
    _assert_parameter_count_pairs(refit, candidates)


def _assert_search_budget_replay(
    candidates: pd.DataFrame,
    plan: Mapping[str, Any],
) -> None:
    planned = {
        str(family): int(count)
        for family, count in dict(plan["search_budget_by_family"]).items()
    }
    for unit, rows in candidates.groupby("evaluation_unit", sort=True):
        observed = search_budget_by_family(rows["method_family"].astype(str))
        assert_search_budget_by_family(observed, evaluation_unit=str(unit))
        if observed != planned:
            raise ValueError(
                "Phase 14 recorded search budget does not replay for "
                f"{unit}: observed={observed}, planned={planned}."
            )


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return json.loads(json.dumps(dict(config)))
    path = Path(config)
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("Phase 14 configuration must be a mapping.")
    return loaded


def _resolve_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "dataset",
        "splits",
        "features",
        "policy_split",
        "transfer",
        "methods",
        "retention",
        "metrics",
        "selection_metric",
        "base_seed",
        "fit_timeout_seconds",
        "output",
    }
    if set(raw) != required:
        raise ValueError(
            "Phase 14 configuration keys mismatch: "
            f"missing={sorted(required-set(raw))}, unknown={sorted(set(raw)-required)}."
        )
    dataset = raw["dataset"]
    splits = raw["splits"]
    features = raw["features"]
    policy = raw["policy_split"]
    transfer = raw["transfer"]
    methods = raw["methods"]
    retention = raw["retention"]
    for name, value in {
        "dataset": dataset,
        "splits": splits,
        "features": features,
        "policy_split": policy,
        "transfer": transfer,
        "methods": methods,
        "retention": retention,
    }.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping.")
    if set(dataset) != {"path"}:
        raise ValueError("dataset must contain exactly path.")
    if set(splits) != {"canonical_directory", "seeds", "fractions"}:
        raise ValueError("splits keys mismatch.")
    if set(features) != {"kind", "n_bits", "radius", "fingerprint_backend"}:
        raise ValueError("features keys mismatch.")
    feature_config = dict(features)
    if (
        feature_config["kind"] != "bh_role_separated"
        or feature_config["fingerprint_backend"] != "rdkit"
        or int(feature_config["n_bits"]) <= 0
        or int(feature_config["radius"]) <= 0
    ):
        raise ValueError("Phase 14 scientific features require role-separated RDKit.")
    if set(policy) != {"valid_fraction"}:
        raise ValueError("policy_split must contain exactly valid_fraction.")
    policy_fraction = float(policy["valid_fraction"])
    if not 0.0 < policy_fraction < 1.0:
        raise ValueError("policy valid_fraction must be in (0, 1).")
    if set(transfer) != {"anonymous", "typed"}:
        raise ValueError("transfer must declare anonymous and typed.")
    anonymous = dict(transfer["anonymous"])
    typed = dict(transfer["typed"])
    if (
        anonymous.get("role_change_requirement") != "all"
        or anonymous.get("fallback_policy") != "reject"
        or anonymous.get("donor_similarity_backend") != "rdkit"
        or typed.get("role_change_requirement") != "all"
        or typed.get("fallback_policy") != "reject"
        or typed.get("donor_similarity_backend") != "rdkit"
        or typed.get("donor_strategy") != "same_substrate_different_role"
    ):
        raise ValueError("Phase 14 transfer pools must be strict RDKit scientific pools.")
    expected_methods = {
        "ae_candidates",
        "control_latent_dims",
        "ridge_alphas",
        "xgboost_candidates",
        "transfer_supervised_weights",
        "training",
    }
    if set(methods) != expected_methods:
        raise ValueError("methods keys mismatch.")
    ridge_alphas = [float(value) for value in methods["ridge_alphas"]]
    if (
        not ridge_alphas
        or len(ridge_alphas) != len(set(ridge_alphas))
        or any(not math.isfinite(value) or value <= 0.0 for value in ridge_alphas)
    ):
        raise ValueError("methods.ridge_alphas must be unique positive finite values.")
    training = dict(methods["training"])
    expected_training = {
        "max_epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "internal_valid_fraction",
    }
    if set(training) != expected_training:
        raise ValueError("methods.training keys mismatch.")
    if float(training["internal_valid_fraction"]) != policy_fraction:
        raise ValueError("Policy and model internal validation fractions must match.")
    if int(training["max_epochs"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("Training epochs and batch size must be positive.")
    seeds = tuple(int(value) for value in splits["seeds"])
    fractions = tuple(float(value) for value in splits["fractions"])
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or not fractions
        or any(not 0.0 < value <= 1.0 for value in fractions)
        or tuple(sorted(fractions)) != fractions
    ):
        raise ValueError("Invalid seeds or nested fractions.")
    metric_names = tuple(str(value) for value in raw["metrics"])
    if not metric_names or len(metric_names) != len(set(metric_names)):
        raise ValueError("metrics must be a nonempty unique sequence.")
    if any(value not in _METRICS for value in metric_names):
        raise ValueError("Unsupported metric.")
    selection_metric = str(raw["selection_metric"])
    if selection_metric != "rmse" or selection_metric not in metric_names:
        raise ValueError("Phase 14 selection metric must be rmse.")
    expected_retention = {
        "practical_margin",
        "required_margin_wins_per_fraction",
        "no_practical_loss_threshold",
        "bootstrap_reps",
        "bootstrap_alpha",
        "bootstrap_seed",
    }
    if set(retention) != expected_retention:
        raise ValueError("retention keys mismatch.")
    retention_config = AERetentionConfig(
        fractions=fractions,
        seeds=seeds,
        ae_family="redesigned_supervised_ae",
        control_families=AE_RETENTION_CONTROL_FAMILIES,
        practical_margin=float(retention["practical_margin"]),
        required_margin_wins_per_fraction=int(
            retention["required_margin_wins_per_fraction"]
        ),
        no_practical_loss_threshold=float(
            retention["no_practical_loss_threshold"]
        ),
        bootstrap_reps=int(retention["bootstrap_reps"]),
        bootstrap_alpha=float(retention["bootstrap_alpha"]),
        bootstrap_seed=int(retention["bootstrap_seed"]),
    )
    timeout = float(raw["fit_timeout_seconds"])
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("fit_timeout_seconds must be positive.")
    return {
        "dataset_path": Path(str(dataset["path"])),
        "split_directory": Path(str(splits["canonical_directory"])),
        "seeds": seeds,
        "fractions": fractions,
        "feature_config": feature_config,
        "policy_valid_fraction": policy_fraction,
        "anonymous_config": anonymous,
        "typed_config": typed,
        "methods": json.loads(json.dumps(methods)),
        "retention_config": retention_config,
        "metrics": metric_names,
        "selection_metric": selection_metric,
        "base_seed": int(raw["base_seed"]),
        "fit_timeout_seconds": timeout,
        "output": Path(str(raw["output"])),
    }


def _groups(
    identity: pd.DataFrame,
    positions: Mapping[str, int],
    source_ids: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        str(identity.iloc[positions[str(source_id)]]["canonical_reaction_key"])
        for source_id in source_ids
    )


def _candidate_configs(
    *,
    unit: EvaluationSplitUnit,
    contract: Mapping[str, Any],
    input_dim: int,
    role_blocks: tuple[RoleBlock, ...],
    common_seed: int,
) -> list[tuple[str, Phase14MethodConfig]]:
    methods = contract["methods"]
    training = methods["training"]
    configs: list[tuple[str, Phase14MethodConfig]] = []
    ae_seen: set[str] = set()
    ae_grid_records: list[dict[str, Any]] = []
    rec_weights: set[float] = set()
    architectures: set[tuple[int, int]] = set()
    objectives: set[str] = set()
    role_balanced = False
    masking = False
    for raw_candidate in methods["ae_candidates"]:
        candidate = dict(raw_candidate)
        ae_grid_records.append(dict(candidate))
        candidate_id = str(candidate.pop("candidate_id"))
        data_protocol = str(candidate.pop("data_protocol"))
        required = {
            "hidden_dim",
            "latent_dim",
            "reconstruction_objective",
            "role_balanced_reconstruction",
            "positive_bit_weight",
            "synthetic_reconstruction_weight",
            "synthetic_supervised_weight",
            "masking_probability",
        }
        if set(candidate) != required:
            raise ValueError(f"AE candidate {candidate_id!r} keys mismatch.")
        architecture = (int(candidate["hidden_dim"]), int(candidate["latent_dim"]))
        architectures.add(architecture)
        rec_weights.add(float(candidate["synthetic_reconstruction_weight"]))
        objectives.add(str(candidate["reconstruction_objective"]))
        role_balanced |= bool(candidate["role_balanced_reconstruction"])
        masking |= float(candidate["masking_probability"]) > 0.0
        if data_protocol not in {"real_only", "anonymous", "typed"}:
            raise ValueError("Unsupported AE data protocol.")
        ae_config = RedesignedSupervisedAEConfig(
            input_dim=input_dim,
            hidden_dim=architecture[0],
            latent_dim=architecture[1],
            reconstruction_objective=str(candidate["reconstruction_objective"]),
            role_blocks=role_blocks,
            role_balanced_reconstruction=bool(
                candidate["role_balanced_reconstruction"]
            ),
            positive_bit_weight=float(candidate["positive_bit_weight"]),
            reconstruction_weight=1.0,
            supervised_weight=1.0,
            synthetic_reconstruction_weight=float(
                candidate["synthetic_reconstruction_weight"]
            ),
            synthetic_supervised_weight=float(
                candidate["synthetic_supervised_weight"]
            ),
            masking_probability=float(candidate["masking_probability"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            internal_valid_fraction=float(training["internal_valid_fraction"]),
            random_state=common_seed,
        )
        config = Phase14MethodConfig(
            method_family="redesigned_supervised_ae",
            input_dim=input_dim,
            random_state=common_seed,
            data_protocol=data_protocol,
            latent_dim=architecture[1],
            hidden_dim=architecture[0],
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            internal_valid_fraction=float(training["internal_valid_fraction"]),
            synthetic_supervised_weight=float(
                candidate["synthetic_supervised_weight"]
            ),
            ae_config=ae_config,
        )
        signature = stable_hash(_config_record(config))
        if signature in ae_seen:
            raise ValueError("Duplicate AE candidate configuration.")
        ae_seen.add(signature)
        configs.append((candidate_id, config))
    if not architectures >= {(128, 16), (256, 32)}:
        raise ValueError("Canonical compact AE architectures are incomplete.")
    if rec_weights != {0.0, 0.1, 0.25, 0.5, 1.0}:
        raise ValueError("Synthetic AE reconstruction-weight grid is incomplete.")
    if objectives != {"binary_cross_entropy", "positive_bit_weighted_mse"}:
        raise ValueError("Canonical binary AE objectives are incomplete.")
    if not role_balanced or not masking:
        raise ValueError("Role-balanced and masking AE candidates are required.")
    _assert_identifiable_ae_grid(ae_grid_records)
    augmented_supervised_weights = {
        float(row["synthetic_supervised_weight"])
        for row in ae_grid_records
        if row["data_protocol"] != "real_only"
    }
    if not augmented_supervised_weights <= {
        float(value) for value in methods["transfer_supervised_weights"]
    }:
        raise ValueError(
            "Every augmented AE supervised weight requires a matched no-AE control."
        )
    for latent in methods["control_latent_dims"]:
        for alpha in methods["ridge_alphas"]:
            for family in ("truncated_svd", "linear_autoencoder"):
                policy_id = f"{family}:latent={int(latent)}:alpha={float(alpha):g}"
                configs.append(
                    (
                        policy_id,
                        Phase14MethodConfig(
                            method_family=family,
                            input_dim=input_dim,
                            random_state=common_seed,
                            latent_dim=int(latent),
                            ridge_alpha=float(alpha),
                            max_epochs=int(training["max_epochs"]),
                            patience=int(training["patience"]),
                            learning_rate=float(training["learning_rate"]),
                            weight_decay=float(training["weight_decay"]),
                            batch_size=int(training["batch_size"]),
                            internal_valid_fraction=float(
                                training["internal_valid_fraction"]
                            ),
                        ),
                    )
                )
    for index, xgb in enumerate(methods["xgboost_candidates"]):
        if not isinstance(xgb, Mapping):
            raise ValueError("XGBoost candidates must be mappings.")
        configs.append(
            (
                f"direct_xgboost:{index}",
                Phase14MethodConfig(
                    method_family="direct_xgboost",
                    input_dim=input_dim,
                    random_state=common_seed,
                    xgboost_params=tuple(sorted(dict(xgb).items())),
                ),
            )
        )
    transfer_weights = [float(value) for value in methods["transfer_supervised_weights"]]
    if any(value < 0.0 for value in transfer_weights):
        raise ValueError("Transfer supervised weights cannot be negative.")

    def direct_mlp(
        hidden: int,
        latent: int,
        *,
        protocol: str,
        weight: float,
    ) -> Phase14MethodConfig:
        return Phase14MethodConfig(
            method_family="matched_direct_mlp",
            input_dim=input_dim,
            random_state=common_seed,
            data_protocol=protocol,
            hidden_dim=hidden,
            latent_dim=latent,
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            internal_valid_fraction=float(training["internal_valid_fraction"]),
            synthetic_supervised_weight=weight,
        )

    # The direct-MLP control is protocol-selected exactly like the AE family so
    # that "AE beats matched_direct_mlp" cannot be explained by pool access.
    for hidden, latent in ((128, 16), (256, 32)):
        configs.append(
            (
                f"matched_direct_mlp:{hidden}-{latent}:real_only",
                direct_mlp(hidden, latent, protocol="real_only", weight=1.0),
            )
        )
        for protocol in ("anonymous", "typed"):
            for value in transfer_weights:
                configs.append(
                    (
                        f"matched_direct_mlp:{hidden}-{latent}:"
                        f"{protocol}:weight={value:g}",
                        direct_mlp(
                            hidden, latent, protocol=protocol, weight=value
                        ),
                    )
                )
    for index, xgb in enumerate(methods["xgboost_candidates"]):
        for value in transfer_weights:
            for family, protocol in (
                ("anonymous_transfer_without_ae", "anonymous"),
                ("typed_transfer_without_ae", "typed"),
            ):
                configs.append(
                    (
                        f"{family}:xgb={index}:weight={value:g}",
                        Phase14MethodConfig(
                            method_family=family,
                            input_dim=input_dim,
                            random_state=common_seed,
                            data_protocol=protocol,
                            synthetic_supervised_weight=value,
                            xgboost_params=tuple(sorted(dict(xgb).items())),
                        ),
                    )
                )
    policy_ids = [policy_id for policy_id, _ in configs]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError(f"Duplicate policy IDs for {unit.evaluation_unit}.")
    if {config.method_family for _, config in configs} != set(
        PHASE14_METHOD_FAMILIES
    ):
        raise ValueError("Phase 14 candidate families are incomplete.")
    assert_search_budget_by_family(
        search_budget_by_family(config.method_family for _, config in configs),
        evaluation_unit=unit.evaluation_unit,
    )
    return configs


def _assert_uniform_search_budget(
    candidate_rows: Sequence[Mapping[str, Any]],
    units: Sequence[EvaluationSplitUnit],
) -> dict[str, int]:
    budgets: dict[str, dict[str, int]] = {}
    for unit in units:
        budget = search_budget_by_family(
            row["method_family"]
            for row in candidate_rows
            if str(row["evaluation_unit"]) == unit.evaluation_unit
        )
        assert_search_budget_by_family(
            budget, evaluation_unit=unit.evaluation_unit
        )
        budgets[unit.evaluation_unit] = budget
    if len({stable_hash(budget) for budget in budgets.values()}) != 1:
        raise ValueError(
            "Phase 14 search budget must be identical across evaluation units: "
            f"{budgets}."
        )
    return next(iter(budgets.values()))


def search_budget_by_family(families: Any) -> dict[str, int]:
    """Count candidate policies per method family in canonical family order."""
    counted = list(str(value) for value in families)
    return {
        family: counted.count(family) for family in PHASE14_METHOD_FAMILIES
    }


def assert_search_budget_by_family(
    budget: Mapping[str, int],
    *,
    evaluation_unit: str,
) -> None:
    """Fail when any control family searches a smaller grid than the floor.

    Candidate counts are recorded, never silently equalized: a control family
    with fewer than ``MINIMUM_CONTROL_SEARCH_BUDGET`` candidates would hand the
    autoencoder family an unearned winner's-curse advantage under
    min-over-inner-validation selection.
    """
    if set(budget) != set(PHASE14_METHOD_FAMILIES):
        raise ValueError("Phase 14 search-budget families mismatch.")
    starved = {
        family: int(budget[family])
        for family in PHASE14_METHOD_FAMILIES
        if family != PHASE14_AE_FAMILY
        and int(budget[family]) < MINIMUM_CONTROL_SEARCH_BUDGET
    }
    if starved:
        raise ValueError(
            "Phase 14 control search budget is below the fairness floor of "
            f"{MINIMUM_CONTROL_SEARCH_BUDGET} candidate policies for "
            f"{evaluation_unit}: {sorted(starved.items())} against an AE budget "
            f"of {int(budget[PHASE14_AE_FAMILY])}."
        )


def _assert_identifiable_ae_grid(records: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(row["candidate_id"]): dict(row) for row in records}
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
    baseline_id = "ae-baseline-128-16-rec025-anonymous"
    if set(by_id) != {baseline_id, *expected_differences}:
        raise ValueError("Phase 14 AE grid must match the frozen identifiable design.")
    baseline = {
        key: value
        for key, value in by_id[baseline_id].items()
        if key != "candidate_id"
    }
    for candidate_id, expected in expected_differences.items():
        candidate = {
            key: value
            for key, value in by_id[candidate_id].items()
            if key != "candidate_id"
        }
        differences = {
            key
            for key in baseline
            if candidate.get(key) != baseline[key]
        }
        if differences != expected or set(candidate) != set(baseline):
            raise ValueError(
                f"AE candidate {candidate_id!r} is not a one-factor-at-a-time "
                f"ablation: observed={sorted(differences)}, "
                f"expected={sorted(expected)}."
            )


def _scientific_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(contract["dataset_path"]),
        "split_directory": str(contract["split_directory"]),
        "seeds": list(contract["seeds"]),
        "fractions": list(contract["fractions"]),
        "feature_config": dict(contract["feature_config"]),
        "policy_valid_fraction": contract["policy_valid_fraction"],
        "anonymous_config": dict(contract["anonymous_config"]),
        "typed_config": dict(contract["typed_config"]),
        "methods": contract["methods"],
        "retention_config": asdict(contract["retention_config"]),
        "metrics": list(contract["metrics"]),
        "selection_metric": contract["selection_metric"],
        "base_seed": contract["base_seed"],
    }


def _derived_seed(base_seed: int, evaluation_unit: str, purpose: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}|{evaluation_unit}|{purpose}".encode()
    ).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _config_record(config: Phase14MethodConfig) -> dict[str, Any]:
    return asdict(config)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_frozen_for_final(
    path: Path,
    *,
    expected_document_hash: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    claimed = payload.pop("frozen_document_hash", None)
    if (
        claimed != expected_document_hash
        or claimed != stable_hash(payload)
        or payload.get("status") != "frozen_before_outer_test"
        or payload.get("outer_test_labels_accessed") is not False
        or payload.get("outer_test_predictions_generated") is not False
        or payload.get("outer_test_metrics_generated") is not False
    ):
        raise RuntimeError("Persisted frozen policy document is invalid.")
    for policy in payload.get("policies", ()):
        policy_claimed = policy.get("frozen_policy_hash")
        if (
            policy_claimed != stable_hash(frozen_policy_identity_payload(policy))
            or policy.get("status") != "frozen_before_outer_test"
        ):
            raise RuntimeError("Persisted frozen method policy is invalid.")
    payload["frozen_document_hash"] = claimed
    return payload


def _load_retention_for_final(
    path: Path,
    *,
    expected_decision_hash: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    claimed = payload.pop("decision_hash", None)
    if (
        claimed != expected_decision_hash
        or claimed != stable_hash(payload)
        or payload.get("data_role") != "placement_validation"
        or payload.get("outer_test_used_for_decision") is not False
        or payload.get("placement")
        not in {"primary_benchmark", "secondary_ablation"}
    ):
        raise RuntimeError("Persisted AE retention decision is invalid.")
    payload["decision_hash"] = claimed
    return payload


def _write_registry_snapshot(
    path: Path,
    registry: EvaluationRegistry,
    identities: Mapping[tuple[str, str], EvaluationIdentity],
    *,
    status: str,
) -> None:
    records = []
    for key, identity in sorted(identities.items()):
        record = registry.inspect(identity)
        if record is None:
            raise RuntimeError(f"Missing outer-test registry record for {key}.")
        records.append(record.to_document())
    payload = {
        "schema_version": REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
        "status": status,
        "registry_directory": str(registry.root),
        "record_count": len(records),
        "records": records,
    }
    payload["snapshot_hash"] = stable_hash(payload)
    _write_json(path, payload)


def _read_allowed_outcomes(path: Path, allowed_ids: set[str]) -> pd.DataFrame:
    if not allowed_ids:
        raise ValueError("Outcome access requires a nonempty explicit allowlist.")
    frame = pd.read_csv(path, usecols=["source_row_id", "yield"])
    frame["source_row_id"] = frame["source_row_id"].astype(str)
    selected = frame.loc[frame["source_row_id"].isin(allowed_ids)].copy()
    if set(selected["source_row_id"]) != set(allowed_ids):
        raise ValueError("Outcome allowlist contains unknown source IDs.")
    if selected["source_row_id"].duplicated().any():
        raise ValueError("Outcome allowlist contains duplicate source IDs.")
    selected["yield"] = pd.to_numeric(selected["yield"], errors="raise")
    if not np.isfinite(selected["yield"].to_numpy(float)).all():
        raise ValueError("Allowed outcomes must be finite.")
    return selected.sort_values("source_row_id").reset_index(drop=True)


def _build_pools(
    *,
    phase: str,
    unit: EvaluationSplitUnit,
    measured_ids: Sequence[str],
    outcomes: pd.Series,
    identity: pd.DataFrame,
    X_all: np.ndarray,
    positions: Mapping[str, int],
    feature_names: Sequence[str],
    feature_metadata: Any,
    contract: Mapping[str, Any],
    pool_audits: list[dict[str, Any]],
    pool_summaries: list[dict[str, Any]],
) -> dict[str, ScientificTransferPool]:
    measured_ids = tuple(sorted(str(value) for value in measured_ids))
    # Deduplication keys may only describe rows this phase is allowed to see:
    # the inner policy-fit subset during search and the saved training rows
    # during placement/final. Validation and outer-test identities are never
    # consumed, matching the declared pool_sources_and_donors protocol.
    measured_identity_keys = tuple(
        sorted(set(_groups(identity, positions, measured_ids)))
    )
    measured_identity_key_hash = stable_hash(list(measured_identity_keys))
    common_seed = _derived_seed(
        contract["base_seed"], unit.evaluation_unit, "common-ae-split"
    )
    split = build_measured_internal_split(
        measured_ids,
        ("measured",) * len(measured_ids),
        train_group_ids=_groups(identity, positions, measured_ids),
        valid_fraction=float(contract["methods"]["training"]["internal_valid_fraction"]),
        random_state=common_seed,
    )
    eligible_ids = split.measured_fit_source_ids
    eligible_positions = [positions[source_id] for source_id in eligible_ids]
    training_frame = identity.iloc[eligible_positions].copy().reset_index(drop=True)
    training_frame["yield"] = outcomes.loc[list(eligible_ids)].to_numpy(float)
    training_features = X_all[eligible_positions]
    training_labels = training_frame["yield"].to_numpy(float)
    pool_seed = _derived_seed(
        contract["base_seed"], unit.evaluation_unit, f"{phase}-pool"
    )
    anonymous_config = ConditionTransferConfig(
        **{**contract["anonymous_config"], "random_state": pool_seed}
    )
    typed_config = RoleAwareConditionTransferConfig(
        **{**contract["typed_config"], "random_state": pool_seed}
    )
    pools = {
        "anonymous": build_anonymous_transfer_pool(
            training_frame=training_frame,
            training_features=training_features,
            training_labels=training_labels,
            feature_config=contract["feature_config"],
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            global_measured_identity_keys=measured_identity_keys,
            config=anonymous_config,
        ),
        "typed": build_strict_context_matched_typed_transfer_pool(
            training_frame=training_frame,
            training_features=training_features,
            training_labels=training_labels,
            feature_config=contract["feature_config"],
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            global_measured_identity_keys=measured_identity_keys,
            config=typed_config,
        ),
    }
    for pool in pools.values():
        if pool.global_measured_identity_hash != measured_identity_key_hash:
            raise RuntimeError(
                "Phase 14 transfer pool consumed an unexpected measured "
                "identity key set."
            )
    eligible_hash = stable_hash(sorted(eligible_ids))
    internal_validation_hash = stable_hash(
        sorted(split.measured_validation_source_ids)
    )
    for short_kind, pool in pools.items():
        audit = pool.audit
        if audit.empty:
            pool_audits.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "phase": phase,
                    "pool_kind": short_kind,
                    "pool_hash": pool.pool_hash,
                    "audit_record_type": "zero_candidate_pool",
                    "source_row_id": "",
                    "donor_row_id": "",
                    "accepted": False,
                    "kept_after_budget": False,
                    "parent_eligible_source_id_hash": eligible_hash,
                }
            )
        else:
            for record in audit.to_dict(orient="records"):
                pool_audits.append(
                    {
                        "evaluation_unit": unit.evaluation_unit,
                        "phase": phase,
                        "pool_kind": short_kind,
                        "pool_hash": pool.pool_hash,
                        "audit_record_type": "candidate",
                        "parent_eligible_source_id_hash": eligible_hash,
                        **record,
                    }
                )
        parent_ids: set[str] = set()
        if not audit.empty:
            for column in ("source_row_id", "donor_row_id"):
                parent_ids.update(
                    str(value)
                    for value in audit[column].dropna().astype(str)
                    if str(value)
                )
        internal_validation = set(split.measured_validation_source_ids)
        pool_summaries.append(
            {
                "evaluation_unit": unit.evaluation_unit,
                "phase": phase,
                "pool_kind": short_kind,
                "pool_hash": pool.pool_hash,
                "parent_eligible_source_id_hash": eligible_hash,
                "internal_validation_source_id_hash": internal_validation_hash,
                "observed_parent_source_id_hash": stable_hash(sorted(parent_ids)),
                "parent_count": len(parent_ids),
                "candidate_count": pool.candidate_count,
                "accepted_count": pool.accepted_count,
                "selected_count": pool.selected_count,
                "degenerate_pool": pool.selected_count == 0,
                "measured_identity_key_count": len(measured_identity_keys),
                "measured_identity_key_hash": measured_identity_key_hash,
                "measured_identity_key_scope": (
                    "inner_policy_fit_rows"
                    if phase == "search"
                    else "saved_training_rows"
                ),
                "parents_exclude_internal_validation": not bool(
                    parent_ids & internal_validation
                ),
                "pool_training_source_id_hash": pool.training_source_id_hash,
            }
        )
    return pools


def _pool_for_config(
    config: Phase14MethodConfig,
    pools: Mapping[str, ScientificTransferPool],
) -> ScientificTransferPool | None:
    if config.data_protocol == "real_only":
        return None
    if config.data_protocol == "anonymous":
        return pools["anonymous"]
    if config.data_protocol == "typed":
        return pools["typed"]
    raise ValueError(f"Unknown data protocol {config.data_protocol!r}.")


def _isolated_method_fit(
    config: Phase14MethodConfig,
    *,
    measured_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    outcomes: pd.Series,
    identity: pd.DataFrame,
    X_all: np.ndarray,
    positions: Mapping[str, int],
    pool: ScientificTransferPool | None,
    timeout: float,
    evaluation_unit: str,
    phase: str,
) -> Any:
    measured_ids = tuple(str(value) for value in measured_ids)
    evaluation_ids = tuple(str(value) for value in evaluation_ids)
    measured_positions = [positions[value] for value in measured_ids]
    evaluation_positions = [positions[value] for value in evaluation_ids]
    kwargs: dict[str, Any] = {
        "measured_source_ids": measured_ids,
        "measured_group_ids": _groups(identity, positions, measured_ids),
        "config": config,
        "evaluation_unit": evaluation_unit,
        "phase": phase,
    }
    if pool is not None:
        rows = pool.rows
        kwargs.update(
            {
                "X_synthetic": pool.features,
                "y_synthetic": pool.labels,
                "synthetic_source_ids": tuple(
                    f"synthetic:{value}"
                    for value in rows["canonical_reaction_hash"].astype(str)
                ),
                "synthetic_group_ids": tuple(
                    rows["canonical_reaction_key"].astype(str)
                ),
                "transfer_pool_hash": pool.pool_hash,
            }
        )
    return run_isolated_fit(
        fit_phase14_method,
        args=(
            X_all[measured_positions],
            outcomes.loc[list(measured_ids)].to_numpy(float),
            X_all[evaluation_positions],
        ),
        kwargs=kwargs,
        timeout_seconds=timeout,
    )


def _resource_row(
    unit: EvaluationSplitUnit,
    config: Phase14MethodConfig,
    policy_id: str,
    phase: str,
    resource_usage: Any,
) -> dict[str, Any]:
    return {
        "evaluation_unit": unit.evaluation_unit,
        "seed": unit.seed,
        "train_fraction": unit.train_fraction,
        "method_family": config.method_family,
        "policy_id": policy_id,
        "phase": phase,
        "hidden_width": config.hidden_dim,
        "latent_width": config.latent_dim,
        **resource_usage.to_dict(),
    }


def _fit_audit_row(
    unit: EvaluationSplitUnit,
    config: Phase14MethodConfig,
    policy_id: str,
    result: Phase14MethodFitResult,
    phase: str,
    *,
    measured_ids: Sequence[str],
    forbidden_ids: set[str],
    pool: ScientificTransferPool | None,
    plan_hash: str,
    frozen_policy_hash: str | None = None,
) -> dict[str, Any]:
    measured = set(str(value) for value in measured_ids)
    method_metadata = dict(result.metadata.get("method_metadata", {}))
    return {
        "evaluation_unit": unit.evaluation_unit,
        "seed": unit.seed,
        "train_fraction": unit.train_fraction,
        "method_family": config.method_family,
        "policy_id": policy_id,
        "phase": phase,
        "data_protocol": config.data_protocol,
        "expected_measured_source_id_hash": stable_hash(sorted(measured)),
        "measured_fit_row_count": len(measured_ids),
        "synthetic_fit_row_count": 0 if pool is None else pool.selected_count,
        "evaluation_row_count": len(result.predictions),
        "measured_fit_source_id_hash": result.measured_fit_source_id_hash,
        "synthetic_fit_source_id_hash": result.synthetic_fit_source_id_hash,
        "fit_group_assignment_hash": result.fit_group_assignment_hash,
        "fit_forbidden_overlap_count": len(measured & forbidden_ids),
        "evaluation_labels_received": False,
        "pool_hash": None if pool is None else pool.pool_hash,
        "pool_training_source_id_hash": (
            None if pool is None else pool.training_source_id_hash
        ),
        "internal_validation_source_id_hash": method_metadata.get(
            "internal_validation_source_id_hash"
        ),
        "state_hash": result.state_hash,
        "selected_epoch": result.selected_epoch,
        "training_parameter_count": result.training_parameter_count,
        "deployed_predictor_parameter_count": (
            result.deployed_predictor_parameter_count
        ),
        "effective_supervised_weight_sum": result.effective_supervised_weight_sum,
        "reconstruction_metrics_hash": result.reconstruction_metrics_hash,
        "plan_hash": plan_hash,
        "frozen_policy_hash": frozen_policy_hash,
    }


def _prediction_rows(
    unit: EvaluationSplitUnit,
    family: str,
    policy_id: str,
    policy_hash: str,
    source_ids: Sequence[str],
    observed: np.ndarray,
    predicted: np.ndarray,
    data_role: str,
    prediction_hash: str,
    state_hash: str,
    plan_hash: str,
    *,
    frozen_policy_hash: str | None = None,
    test_evaluation_count: int = 0,
) -> list[dict[str, Any]]:
    return [
        {
            "evaluation_unit": unit.evaluation_unit,
            "seed": unit.seed,
            "train_fraction": unit.train_fraction,
            "method_family": family,
            "policy_id": policy_id,
            "policy_hash": policy_hash,
            "source_row_id": str(source_id),
            "observed_yield": float(y_true),
            "predicted_yield": float(y_pred),
            "data_role": data_role,
            "prediction_hash": prediction_hash,
            "state_hash": state_hash,
            "plan_hash": plan_hash,
            "frozen_policy_hash": frozen_policy_hash,
            "test_evaluation_count": test_evaluation_count,
            "test_used_for_selection_or_retention": False,
        }
        for source_id, y_true, y_pred in zip(
            source_ids, observed, predicted, strict=True
        )
    ]


def _metric_rows(
    unit: EvaluationSplitUnit,
    family: str,
    policy_id: str,
    policy_hash: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    data_role: str,
    metrics: Sequence[str],
    prediction_hash: str,
    state_hash: str,
    plan_hash: str,
    *,
    frozen_policy_hash: str | None = None,
    test_evaluation_count: int = 0,
) -> list[dict[str, Any]]:
    return [
        {
            "evaluation_unit": unit.evaluation_unit,
            "seed": unit.seed,
            "train_fraction": unit.train_fraction,
            "method_family": family,
            "policy_id": policy_id,
            "policy_hash": policy_hash,
            "metric": metric,
            "value": float(_METRICS[metric](observed, predicted)),
            "n_samples": len(observed),
            "data_role": data_role,
            "prediction_hash": prediction_hash,
            "state_hash": state_hash,
            "plan_hash": plan_hash,
            "frozen_policy_hash": frozen_policy_hash,
            "test_evaluation_count": test_evaluation_count,
            "test_used_for_selection_or_retention": False,
        }
        for metric in metrics
    ]


def _reconstruction_rows(
    unit: EvaluationSplitUnit,
    family: str,
    policy_id: str,
    phase: str,
    metrics: Mapping[str, Any],
    plan_hash: str,
) -> list[dict[str, Any]]:
    if not metrics:
        return []
    rows: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif value is None or isinstance(value, (int, float, np.number)):
            rows.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "seed": unit.seed,
                    "train_fraction": unit.train_fraction,
                    "method_family": family,
                    "policy_id": policy_id,
                    "phase": phase,
                    "reconstruction_measure": prefix,
                    "value": np.nan if value is None else float(value),
                    "plan_hash": plan_hash,
                }
            )

    visit("", metrics)
    return rows


def _retention_inputs(placement_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in placement_metrics.loc[placement_metrics["metric"].eq("rmse")].to_dict(
        orient="records"
    ):
        records.append(
            {
                "evaluation_unit": str(row["evaluation_unit"]),
                "seed": int(row["seed"]),
                "train_fraction": float(row["train_fraction"]),
                "method_family": str(row["method_family"]),
                "metric": "rmse",
                "value": float(row["value"]),
                "selected_policy_hash": str(row["policy_hash"]),
                "frozen_policy_hash": str(row["frozen_policy_hash"]),
                "data_role": "placement_validation",
            }
        )
    return records


def _method_evidence_hash(
    *,
    phase: str,
    evaluation_unit: str | None,
    method_family: str | None,
    predictions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    metrics: Sequence[Mapping[str, Any]] | pd.DataFrame,
    refits: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> str:
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metrics)
    refit_frame = pd.DataFrame(refits)

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        selected = frame.loc[frame["phase"].eq(phase)].copy() if "phase" in frame else frame.copy()
        if "data_role" in selected:
            expected_role = {
                "search": "inner_policy_validation",
                "placement": "placement_validation",
            }[phase]
            selected = selected.loc[selected["data_role"].eq(expected_role)]
        if evaluation_unit is not None:
            selected = selected.loc[
                selected["evaluation_unit"].astype(str).eq(evaluation_unit)
            ]
        if method_family is not None:
            selected = selected.loc[
                selected["method_family"].astype(str).eq(method_family)
            ]
        return selected

    prediction_frame = select(prediction_frame)
    metric_frame = select(metric_frame)
    refit_frame = select(refit_frame)
    prediction_records = (
        prediction_frame[
            [
                "evaluation_unit",
                "method_family",
                "policy_id",
                "policy_hash",
                "prediction_hash",
                "state_hash",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["evaluation_unit", "method_family", "policy_id"],
            kind="mergesort",
        )
        .to_dict(orient="records")
    )
    metric_records = [
        {
            "evaluation_unit": str(row["evaluation_unit"]),
            "method_family": str(row["method_family"]),
            "policy_id": str(row["policy_id"]),
            "metric": str(row["metric"]),
            "value": float(row["value"]),
            "prediction_hash": str(row["prediction_hash"]),
            "state_hash": str(row["state_hash"]),
        }
        for row in metric_frame.sort_values(
            ["evaluation_unit", "method_family", "policy_id", "metric"],
            kind="mergesort",
        ).to_dict(orient="records")
    ]
    refit_records = [
        {
            "evaluation_unit": str(row["evaluation_unit"]),
            "method_family": str(row["method_family"]),
            "policy_id": str(row["policy_id"]),
            "state_hash": str(row["state_hash"]),
            "training_parameter_count": (
                None
                if pd.isna(row["training_parameter_count"])
                else int(row["training_parameter_count"])
            ),
            "deployed_predictor_parameter_count": (
                None
                if pd.isna(row["deployed_predictor_parameter_count"])
                else int(row["deployed_predictor_parameter_count"])
            ),
            "reconstruction_metrics_hash": str(
                row["reconstruction_metrics_hash"]
            ),
        }
        for row in refit_frame.sort_values(
            ["evaluation_unit", "method_family", "policy_id"],
            kind="mergesort",
        ).to_dict(orient="records")
    ]
    return stable_hash(
        {
            "phase": phase,
            "prediction_records": prediction_records,
            "metric_records": metric_records,
            "refit_records": refit_records,
        }
    )


def frozen_policy_identity_payload(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the stable, declared subset of a frozen policy document.

    ``implementation_commit`` and the float-derived search evidence fields are
    recorded provenance only: including them would let an unrelated commit or a
    1-ULP metric difference mint a fresh outer-test evaluation identity.
    """
    payload = {
        key: value
        for key, value in policy.items()
        if key not in _NON_IDENTITY_POLICY_FIELDS
    }
    if set(payload) != _FROZEN_POLICY_IDENTITY_FIELDS:
        raise ValueError("Phase 14 frozen policy identity schema mismatch.")
    return payload


def _final_evaluation_anchor_hash(
    *,
    plan_hash: str,
    config_hash: str,
    dataset_hash: str,
    canonical_split_hash: str,
    retention_config: Mapping[str, Any],
) -> str:
    """Anchor the outer-test identity to declarations, never to outcomes.

    The retention *configuration* is bound in; the retention decision, the
    placement evidence and the frozen-document hash are not, because all three
    are functions of float metric values produced by this very run.
    """
    return stable_hash(
        {
            "schema_version": REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION,
            "anchor_scope": "declared_scientific_configuration_only",
            "plan_hash": plan_hash,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "canonical_split_hash": canonical_split_hash,
            "retention_config": json.loads(json.dumps(dict(retention_config))),
        }
    )


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["method_family", "metric"], sort=True)["value"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={"std": "std_value", "count": "n_evaluation_units"}
    )
    return grouped


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def _dependency_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for package in ("rdkit", "scikit-learn", "torch", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _command(config: str | Path | Mapping[str, Any], output: Path) -> str:
    config_text = "<mapping>" if isinstance(config, Mapping) else str(config)
    return (
        "python scripts/run_redesigned_ae_benchmark.py "
        f"--config {config_text} --output-directory {output}"
    )


def _parse_phase14_method_config(value: Mapping[str, Any]) -> Phase14MethodConfig:
    expected = {field.name for field in fields(Phase14MethodConfig)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Resolved Phase 14 method config schema mismatch.")
    record = json.loads(json.dumps(dict(value)))
    ae_record = record.get("ae_config")
    if ae_record is not None:
        ae_expected = {field.name for field in fields(RedesignedSupervisedAEConfig)}
        if not isinstance(ae_record, dict) or set(ae_record) != ae_expected:
            raise ValueError("Resolved redesigned-AE config schema mismatch.")
        role_blocks = ae_record.get("role_blocks")
        if not isinstance(role_blocks, list):
            raise ValueError("Resolved redesigned-AE role blocks are invalid.")
        ae_record["role_blocks"] = tuple(
            RoleBlock(**block) for block in role_blocks
        )
        record["ae_config"] = RedesignedSupervisedAEConfig(**ae_record)
    params = record.get("xgboost_params")
    if not isinstance(params, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in params
    ):
        raise ValueError("Resolved XGBoost parameter schema mismatch.")
    record["xgboost_params"] = tuple((item[0], item[1]) for item in params)
    config = Phase14MethodConfig(**record)
    if stable_hash(_config_record(config)) != stable_hash(value):
        raise ValueError("Resolved Phase 14 method config does not round-trip.")
    return config


def _assert_canonical_dependency_replay(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    split_units: pd.DataFrame,
    partitions: pd.DataFrame,
    prediction_frames: Sequence[pd.DataFrame],
) -> None:
    resolved = plan["resolved_scientific_config"]
    saved = load_saved_canonical_split_identities(
        Path(resolved["dataset_path"]),
        Path(resolved["split_directory"]),
        requested_seeds=tuple(int(value) for value in resolved["seeds"]),
        requested_fractions=tuple(float(value) for value in resolved["fractions"]),
    )
    if (
        saved.dataset_hash != plan["dataset_hash"]
        or saved.dataset_hash != manifest["dataset_hash"]
        or saved.aggregate_split_hash != plan["canonical_split_hash"]
        or saved.aggregate_split_hash != manifest["canonical_split_hash"]
    ):
        raise ValueError("Phase 14 canonical dependency hash replay failed.")
    expected_units = {
        unit.evaluation_unit: unit.audit_record
        for seed in resolved["seeds"]
        for fraction in resolved["fractions"]
        for unit in (
            build_saved_random_split_unit(
                saved, seed=int(seed), train_fraction=float(fraction)
            ),
        )
    }
    if set(split_units["evaluation_unit"].astype(str)) != set(expected_units):
        raise ValueError("Phase 14 split-unit replay coverage failed.")
    replay_fields = {
        "exact_split_hash",
        "train_source_id_hash",
        "validation_source_id_hash",
        "test_source_id_hash",
    }
    for row in split_units.to_dict(orient="records"):
        expected = expected_units[str(row["evaluation_unit"])]
        if any(str(row[field]) != str(expected[field]) for field in replay_fields):
            raise ValueError("Phase 14 saved split assignment replay failed.")
    expected_partition_rows: list[dict[str, Any]] = []
    for unit_name, audit in expected_units.items():
        unit = build_saved_random_split_unit(
            saved,
            seed=int(audit["seed"]),
            train_fraction=float(audit["train_fraction"]),
        )
        common_seed = _derived_seed(
            int(resolved["base_seed"]), unit_name, "common-ae-split"
        )
        split = build_measured_internal_split(
            unit.train_source_ids,
            ("measured",) * len(unit.train_source_ids),
            train_group_ids=_groups(
                saved.canonical,
                {
                    str(source_id): index
                    for index, source_id in enumerate(
                        saved.canonical["source_row_id"].astype(str)
                    )
                },
                unit.train_source_ids,
            ),
            valid_fraction=float(resolved["policy_valid_fraction"]),
            random_state=common_seed,
        )
        replay_partition = {
            "policy_fit": tuple(sorted(split.measured_fit_source_ids)),
            "policy_holdout": tuple(
                sorted(split.measured_validation_source_ids)
            ),
            "saved_validation": unit.validation_source_ids,
            "outer_test": unit.test_source_ids,
        }
        for role, source_ids in replay_partition.items():
            expected_partition_rows.extend(
                {
                    "evaluation_unit": unit_name,
                    "source_row_id": str(source_id),
                    "data_role": role,
                    "source_id_hash": stable_hash(sorted(source_ids)),
                }
                for source_id in source_ids
            )
    expected_partitions = pd.DataFrame(expected_partition_rows)
    partition_columns = {
        "evaluation_unit",
        "source_row_id",
        "data_role",
        "source_id_hash",
    }
    sort_columns = ["evaluation_unit", "data_role", "source_row_id"]
    if (
        set(partitions) != partition_columns
        or not _frames_close(
            partitions.sort_values(sort_columns).reset_index(drop=True),
            expected_partitions.sort_values(sort_columns).reset_index(drop=True),
        )
    ):
        raise ValueError("Phase 14 partition-assignment replay failed.")
    all_source_ids = set(saved.canonical["source_row_id"].astype(str))
    yields = _read_allowed_outcomes(
        Path(resolved["dataset_path"]), all_source_ids
    ).set_index("source_row_id")["yield"]
    for predictions in prediction_frames:
        expected_outcomes = yields.loc[
            predictions["source_row_id"].astype(str)
        ].to_numpy(float)
        if not np.array_equal(
            predictions["observed_yield"].to_numpy(float),
            expected_outcomes,
        ):
            raise ValueError("Phase 14 canonical outcome replay failed.")
    identity = saved.canonical.copy()
    identity["yield"] = 0.0
    X_all, _, feature_names, metadata = build_feature_matrix_with_metadata(
        identity, resolved["feature_config"]
    )
    feature_contract = feature_contract_record(metadata, feature_names)
    if (
        feature_contract != plan["feature_contract"]
        or feature_contract["feature_metadata_hash"]
        != manifest["feature_metadata_hash"]
    ):
        raise ValueError("Phase 14 feature dependency replay failed.")
    for objective, expected_contract in plan["reconstruction_contracts"].items():
        contract, audit = build_canonical_role_separated_reconstruction_contract(
            metadata, np.asarray(X_all, dtype=np.float32), objective=objective
        )
        if (
            stable_hash(contract.to_dict()) != stable_hash(expected_contract)
            or stable_hash(asdict(audit))
            != stable_hash(plan["reconstruction_domain_audits"][objective])
        ):
            raise ValueError("Phase 14 reconstruction-contract replay failed.")


def _assert_candidate_plan_replay(
    candidates: pd.DataFrame,
    plan: Mapping[str, Any],
) -> None:
    expected_columns = {
        "evaluation_unit",
        "seed",
        "train_fraction",
        "method_family",
        "policy_id",
        "data_protocol",
        "resolved_config",
        "config_hash",
        "policy_hash",
        "exact_split_hash",
        "reconstruction_contract_hash",
        "observed_domain_audit_hash",
    }
    if set(candidates) != expected_columns:
        raise ValueError("Phase 14 candidate artifact schema mismatch.")
    planned = {
        (str(row["evaluation_unit"]), str(row["policy_id"])): row
        for row in plan["candidate_policies"]
    }
    if len(planned) != len(candidates):
        raise ValueError("Phase 14 candidate-plan cardinality mismatch.")
    for row in candidates.to_dict(orient="records"):
        key = (str(row["evaluation_unit"]), str(row["policy_id"]))
        expected = planned.get(key)
        if expected is None:
            raise ValueError("Phase 14 candidate is absent from the frozen plan.")
        config_record = json.loads(str(row["resolved_config"]))
        config = _parse_phase14_method_config(config_record)
        if (
            str(row["resolved_config"]) != str(expected["resolved_config"])
            or str(row["config_hash"]) != stable_hash(_config_record(config))
            or str(row["config_hash"]) != str(expected["config_hash"])
            or str(row["policy_hash"]) != str(expected["policy_hash"])
            or str(row["exact_split_hash"]) != str(expected["exact_split_hash"])
            or config.method_family != str(row["method_family"])
            or config.data_protocol != str(row["data_protocol"])
        ):
            raise ValueError("Phase 14 candidate-plan replay failed.")
        for nullable in (
            "reconstruction_contract_hash",
            "observed_domain_audit_hash",
        ):
            observed_value = None if pd.isna(row[nullable]) else str(row[nullable])
            expected_value = expected[nullable]
            if observed_value != expected_value:
                raise ValueError("Phase 14 candidate contract linkage failed.")


def _assert_frozen_policy_replay(
    frozen: Mapping[str, Any],
    candidates: pd.DataFrame,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    search_predictions: pd.DataFrame,
    search_metrics: pd.DataFrame,
    refit: pd.DataFrame,
) -> None:
    candidate_by_key = {
        (str(row["evaluation_unit"]), str(row["policy_id"])): row
        for row in candidates.to_dict(orient="records")
    }
    expected_policy_fields = _FROZEN_POLICY_IDENTITY_FIELDS | _NON_IDENTITY_POLICY_FIELDS
    for policy in frozen["policies"]:
        if set(policy) != expected_policy_fields:
            raise ValueError("Phase 14 frozen policy schema mismatch.")
        claimed = policy["frozen_policy_hash"]
        body = frozen_policy_identity_payload(policy)
        candidate = candidate_by_key.get(
            (str(policy["evaluation_unit"]), str(policy["policy_id"]))
        )
        config = _parse_phase14_method_config(policy["resolved_config"])
        family_metrics = search_metrics.loc[
            search_metrics["evaluation_unit"].astype(str).eq(
                str(policy["evaluation_unit"])
            )
            & search_metrics["method_family"].astype(str).eq(
                str(policy["method_family"])
            )
            & search_metrics["metric"].eq("rmse")
        ]
        winner = family_metrics.sort_values(
            ["value", "policy_hash", "policy_id"], kind="mergesort"
        ).iloc[0]
        evidence_hash = _method_evidence_hash(
            phase="search",
            evaluation_unit=str(policy["evaluation_unit"]),
            method_family=str(policy["method_family"]),
            predictions=search_predictions,
            metrics=search_metrics,
            refits=refit,
        )
        if (
            claimed != stable_hash(body)
            or candidate is None
            or claimed is None
            or str(policy["selected_policy_hash"])
            != str(candidate["policy_hash"])
            or stable_hash(_config_record(config))
            != str(candidate["config_hash"])
            or policy["method_family"] != config.method_family
            or policy["plan_hash"] != plan["plan_hash"]
            or policy["config_hash"] != plan["config_hash"]
            or policy["dataset_hash"] != manifest["dataset_hash"]
            or policy["canonical_split_hash"]
            != manifest["canonical_split_hash"]
            or policy["feature_metadata_hash"]
            != manifest["feature_metadata_hash"]
            or str(policy["policy_id"]) != str(winner["policy_id"])
            or str(policy["selected_policy_hash"])
            != str(winner["policy_hash"])
            or str(policy["search_prediction_hash"])
            != str(winner["prediction_hash"])
            or str(policy["search_evidence_hash"]) != evidence_hash
        ):
            raise ValueError("Phase 14 frozen policy replay failed.")


def _assert_prediction_partitions(
    partitions: pd.DataFrame,
    prediction_frames: Sequence[tuple[pd.DataFrame, str]],
) -> None:
    expected_roles = {
        "inner_policy_validation": "policy_holdout",
        "placement_validation": "saved_validation",
        "test": "outer_test",
    }
    for predictions, role in prediction_frames:
        for (unit, _family, _policy), rows in predictions.groupby(
            ["evaluation_unit", "method_family", "policy_id"], sort=False
        ):
            expected_ids = set(
                partitions.loc[
                    partitions["evaluation_unit"].astype(str).eq(str(unit))
                    & partitions["data_role"].eq(expected_roles[role]),
                    "source_row_id",
                ].astype(str)
            )
            if (
                set(rows["source_row_id"].astype(str)) != expected_ids
                or rows["source_row_id"].astype(str).duplicated().any()
            ):
                raise ValueError("Phase 14 prediction partition replay failed.")


def _assert_registry_snapshot(
    path: Path,
    frozen: Mapping[str, Any],
    final_predictions: pd.DataFrame,
    final_metrics: pd.DataFrame,
    refit: pd.DataFrame,
    *,
    final_evaluation_anchor_hash: str,
    selection_evidence: Mapping[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    snapshot = json.loads(path.read_text())
    claimed = snapshot.pop("snapshot_hash", None)
    if (
        set(snapshot)
        != {
            "schema_version",
            "status",
            "registry_directory",
            "record_count",
            "records",
        }
        or claimed != stable_hash(snapshot)
        or snapshot.get("schema_version")
        != REDESIGNED_AE_BENCHMARK_SCHEMA_VERSION
        or snapshot.get("status") != "complete"
        or snapshot.get("record_count") != len(snapshot.get("records", ()))
    ):
        raise ValueError("Phase 14 evaluation-registry snapshot mismatch.")
    snapshot["snapshot_hash"] = claimed
    records_by_key = {
        str(document["registry_key"]): document
        for document in snapshot["records"]
    }
    if len(records_by_key) != len(frozen["policies"]):
        raise ValueError("Phase 14 evaluation-registry coverage mismatch.")
    live_registry = EvaluationRegistry(snapshot["registry_directory"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for policy in frozen["policies"]:
        key = (str(policy["evaluation_unit"]), str(policy["method_family"]))
        identity = EvaluationIdentity(
            dataset_hash=policy["dataset_hash"],
            split_or_search_manifest_hash=final_evaluation_anchor_hash,
            frozen_policy_hash=policy["frozen_policy_hash"],
            evaluation_unit=(
                f"{policy['evaluation_unit']}|family={policy['method_family']}"
            ),
        )
        document = records_by_key.get(identity.registry_key)
        live = live_registry.inspect(identity)
        if (
            document is None
            or live is None
            or document != live.to_document()
            or document["status"] != "metrics_complete"
        ):
            raise ValueError("Phase 14 live evaluation-registry linkage failed.")
        prediction_rows = final_predictions.loc[
            final_predictions["evaluation_unit"].astype(str).eq(key[0])
            & final_predictions["method_family"].astype(str).eq(key[1])
        ]
        metric_rows = final_metrics.loc[
            final_metrics["evaluation_unit"].astype(str).eq(key[0])
            & final_metrics["method_family"].astype(str).eq(key[1])
        ]
        payload = document["metrics_payload"]
        outer_payload = [
            row["metric_row"]
            for row in payload
            if row.get("payload_role") == "outer_metric"
            and set(row) == {"payload_role", "metric_row"}
        ]
        fit_payload = [
            row
            for row in payload
            if row.get("payload_role") == "final_fit_evidence"
        ]
        # The durable registry record, not the rewritable artifact directory,
        # is the authority for the frozen selection evidence. The evaluation
        # identity itself must stay free of these float-derived hashes.
        evidence_payload = [
            row
            for row in payload
            if row.get("payload_role") == "frozen_selection_evidence"
        ]
        if len(evidence_payload) != 1 or evidence_payload[0] != {
            "payload_role": "frozen_selection_evidence",
            **dict(selection_evidence),
        }:
            raise ValueError(
                "Phase 14 registry selection-evidence linkage failed."
            )
        if len(payload) != len(outer_payload) + len(fit_payload) + 1:
            raise ValueError("Phase 14 registry payload role coverage failed.")
        final_refit = refit.loc[
            refit["evaluation_unit"].astype(str).eq(key[0])
            & refit["method_family"].astype(str).eq(key[1])
            & refit["phase"].eq("final")
        ]
        if (
            document["prediction_hash"]
            != str(prediction_rows["prediction_hash"].iloc[0])
            or len(outer_payload) != len(metric_rows)
            or not _frames_close(
                pd.DataFrame(outer_payload)[metric_rows.columns],
                metric_rows.reset_index(drop=True),
            )
            or len(fit_payload) != 1
            or len(final_refit) != 1
            or fit_payload[0]
            != {
                "payload_role": "final_fit_evidence",
                "state_hash": str(final_refit.iloc[0]["state_hash"]),
                "reconstruction_metrics_hash": str(
                    final_refit.iloc[0]["reconstruction_metrics_hash"]
                ),
                "training_parameter_count": (
                    None
                    if pd.isna(final_refit.iloc[0]["training_parameter_count"])
                    else int(final_refit.iloc[0]["training_parameter_count"])
                ),
                "deployed_predictor_parameter_count": (
                    None
                    if pd.isna(
                        final_refit.iloc[0][
                            "deployed_predictor_parameter_count"
                        ]
                    )
                    else int(
                        final_refit.iloc[0][
                            "deployed_predictor_parameter_count"
                        ]
                    )
                ),
            }
        ):
            raise ValueError("Phase 14 registry scientific payload linkage failed.")
        result[key] = document
    return result


def _assert_reconstruction_metrics(
    reconstruction: pd.DataFrame,
    candidates: pd.DataFrame,
    refit: pd.DataFrame,
    plan: Mapping[str, Any],
) -> None:
    expected_columns = {
        "evaluation_unit",
        "seed",
        "train_fraction",
        "method_family",
        "policy_id",
        "phase",
        "reconstruction_measure",
        "value",
        "plan_hash",
    }
    expected_measures = {
        f"{scope}.{measure}"
        for scope in ("evaluation", "training.measured", "training.synthetic")
        for measure in (
            "mse",
            "positive_bit_count",
            "positive_bit_mse",
            "zero_bit_count",
            "zero_bit_mse",
        )
    }
    if (
        set(reconstruction) != expected_columns
        or not reconstruction["method_family"].eq(
            "redesigned_supervised_ae"
        ).all()
        or not reconstruction["plan_hash"].eq(plan["plan_hash"]).all()
        or not set(reconstruction["phase"]) <= {"search", "placement", "final"}
    ):
        raise ValueError("Phase 14 reconstruction artifact contract mismatch.")
    ae_refits = refit.loc[
        refit["method_family"].eq("redesigned_supervised_ae")
    ]
    observed_groups = {
        (str(unit), str(policy), str(phase)): set(rows["reconstruction_measure"])
        for (unit, policy, phase), rows in reconstruction.groupby(
            ["evaluation_unit", "policy_id", "phase"], sort=False
        )
    }
    expected_groups = {
        (str(row["evaluation_unit"]), str(row["policy_id"]), str(row["phase"]))
        for row in ae_refits.to_dict(orient="records")
    }
    if set(observed_groups) != expected_groups or any(
        measures != expected_measures for measures in observed_groups.values()
    ):
        raise ValueError("Phase 14 reconstruction row coverage mismatch.")
    candidate_configs = {
        (str(row["evaluation_unit"]), str(row["policy_id"])):
        _parse_phase14_method_config(json.loads(str(row["resolved_config"])))
        for row in candidates.to_dict(orient="records")
    }
    refit_by_group = {
        (str(row["evaluation_unit"]), str(row["policy_id"]), str(row["phase"])):
        row
        for row in ae_refits.to_dict(orient="records")
    }
    for group, rows in reconstruction.groupby(
        ["evaluation_unit", "policy_id", "phase"], sort=False
    ):
        key = tuple(str(value) for value in group)
        refit_row = refit_by_group[key]
        config = candidate_configs[(key[0], key[1])]
        values: dict[str, Any] = {}
        for record in rows.to_dict(orient="records"):
            measure = str(record["reconstruction_measure"])
            value = float(record["value"])
            if measure.endswith("_count"):
                if (
                    not math.isfinite(value)
                    or value < 0
                    or not value.is_integer()
                ):
                    raise ValueError(
                        "Phase 14 reconstruction counts must be integers."
                    )
                values[measure] = int(value)
            else:
                values[measure] = None if math.isnan(value) else value
        nested: dict[str, Any] = {}
        for measure, value in values.items():
            cursor = nested
            parts = measure.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        if stable_hash(nested) != str(
            refit_row["reconstruction_metrics_hash"]
        ):
            raise ValueError("Phase 14 reconstruction fit hash mismatch.")
        expected_totals = {
            "evaluation": int(refit_row["evaluation_row_count"])
            * config.input_dim,
            "training.measured": int(refit_row["measured_fit_row_count"])
            * config.input_dim,
            "training.synthetic": int(refit_row["synthetic_fit_row_count"])
            * config.input_dim,
        }
        for scope, expected_total in expected_totals.items():
            observed_total = (
                values[f"{scope}.positive_bit_count"]
                + values[f"{scope}.zero_bit_count"]
            )
            if observed_total != expected_total:
                raise ValueError(
                    "Phase 14 reconstruction count total mismatch."
                )
        for measure, value in values.items():
            if measure.endswith("_count"):
                continue
            if value is None:
                if not measure.startswith("training.synthetic") or int(
                    refit_row["synthetic_fit_row_count"]
                ):
                    raise ValueError(
                        "Phase 14 reconstruction value is unexpectedly missing."
                    )
            elif not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("Phase 14 reconstruction values are invalid.")
    non_ae_refits = refit.loc[
        ~refit["method_family"].eq("redesigned_supervised_ae")
    ]
    if not non_ae_refits["reconstruction_metrics_hash"].eq(
        stable_hash({})
    ).all():
        raise ValueError("Phase 14 control reconstruction hash mismatch.")


def _assert_parameter_count_pairs(
    refit: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    configs = {
        (str(row["evaluation_unit"]), str(row["policy_id"])):
        _parse_phase14_method_config(json.loads(str(row["resolved_config"])))
        for row in candidates.to_dict(orient="records")
    }
    search = refit.loc[refit["phase"].eq("search")]
    for row in refit.loc[
        refit["method_family"].eq("redesigned_supervised_ae")
    ].to_dict(orient="records"):
        config = configs[(str(row["evaluation_unit"]), str(row["policy_id"]))]
        D, H, L = config.input_dim, int(config.hidden_dim), int(config.latent_dim)
        training_expected = 2 * D * H + 2 * H * L + 2 * H + D + 2 * L + 1
        deployed_expected = D * H + H * L + H + 2 * L + 1
        if (
            int(row["training_parameter_count"]) != training_expected
            or int(row["deployed_predictor_parameter_count"])
            != deployed_expected
        ):
            raise ValueError("Phase 14 AE parameter count replay failed.")
        paired = []
        for mlp_row in search.loc[
            search["evaluation_unit"].astype(str).eq(str(row["evaluation_unit"]))
            & search["method_family"].eq("matched_direct_mlp")
        ].to_dict(orient="records"):
            mlp_config = configs[
                (str(mlp_row["evaluation_unit"]), str(mlp_row["policy_id"]))
            ]
            if (
                mlp_config.hidden_dim == H
                and mlp_config.latent_dim == L
            ):
                paired.append(mlp_row)
        # Every protocol variant of the paired direct-MLP architecture must
        # deploy exactly the AE's predictor parameter count.
        if not paired or any(
            int(row["deployed_predictor_parameter_count"]) != deployed_expected
            for row in paired
        ):
            raise ValueError("Phase 14 AE/direct-MLP parameter pairing failed.")


def _assert_fit_boundaries(
    refit: pd.DataFrame,
    partitions: pd.DataFrame,
    split_units: pd.DataFrame,
) -> None:
    if (
        refit["evaluation_labels_received"].astype(bool).any()
        or not refit["fit_forbidden_overlap_count"].eq(0).all()
        or not refit["expected_measured_source_id_hash"].eq(
            refit["measured_fit_source_id_hash"]
        ).all()
        or not refit["plan_hash"].nunique() == 1
    ):
        raise ValueError("Phase 14 fit boundary audit failed.")
    expected: dict[tuple[str, str], str] = {}
    for unit, rows in partitions.groupby("evaluation_unit", sort=False):
        policy_fit = sorted(
            rows.loc[rows["data_role"].eq("policy_fit"), "source_row_id"].astype(str)
        )
        expected[(str(unit), "search")] = stable_hash(policy_fit)
    for row in split_units.to_dict(orient="records"):
        for phase in ("placement", "final"):
            expected[(str(row["evaluation_unit"]), phase)] = str(
                row["train_source_id_hash"]
            )
    for row in refit.to_dict(orient="records"):
        if str(row["measured_fit_source_id_hash"]) != expected[
            (str(row["evaluation_unit"]), str(row["phase"]))
        ]:
            raise ValueError("Phase 14 refit membership mismatch.")
    for (_, phase, protocol), rows in refit.groupby(
        ["evaluation_unit", "phase", "data_protocol"], dropna=False
    ):
        if protocol == "real_only":
            continue
        if rows["pool_hash"].nunique(dropna=True) != 1:
            raise ValueError(
                f"Phase 14 shared immutable pool parity failed in {phase}/{protocol}."
            )


def _assert_pool_boundaries(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    partitions: pd.DataFrame,
    split_units: pd.DataFrame,
) -> None:
    if (
        not summary["parents_exclude_internal_validation"].astype(bool).all()
        or not summary["parent_eligible_source_id_hash"].eq(
            summary["pool_training_source_id_hash"]
        ).all()
    ):
        raise ValueError("Phase 14 transfer-pool boundary summary failed.")
    allowed: dict[tuple[str, str], set[str]] = {}
    for unit, rows in partitions.groupby("evaluation_unit", sort=False):
        allowed[(str(unit), "search")] = set(
            rows.loc[rows["data_role"].eq("policy_fit"), "source_row_id"].astype(str)
        )
    for row in split_units.to_dict(orient="records"):
        unit = str(row["evaluation_unit"])
        train_rows = partitions.loc[
            partitions["evaluation_unit"].astype(str).eq(unit)
            & partitions["data_role"].eq("policy_fit")
        ]
        # The precise full-train identities are cryptographically checked through
        # the pool training hash; candidate parents are additionally checked
        # against every non-test/non-validation identity in the audit itself.
        allowed[(unit, "placement")] = set(
            partitions.loc[
                partitions["evaluation_unit"].astype(str).eq(unit)
                & partitions["data_role"].isin(["policy_fit", "policy_holdout"]),
                "source_row_id",
            ].astype(str)
        )
        del train_rows
    candidates = audit.loc[audit["audit_record_type"].eq("candidate")]
    for row in candidates.to_dict(orient="records"):
        key = (str(row["evaluation_unit"]), str(row["phase"]))
        for column in ("source_row_id", "donor_row_id"):
            value = str(row[column])
            if value not in allowed[key]:
                raise ValueError("Phase 14 transfer-pool parent boundary failed.")


def _assert_selection(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    plan: Mapping[str, Any],
) -> None:
    if (
        not predictions["data_role"].eq("inner_policy_validation").all()
        or not metrics["data_role"].eq("inner_policy_validation").all()
        or predictions["test_evaluation_count"].ne(0).any()
        or metrics["test_evaluation_count"].ne(0).any()
        or not predictions["plan_hash"].eq(plan["plan_hash"]).all()
        or not metrics["plan_hash"].eq(plan["plan_hash"]).all()
    ):
        raise ValueError("Phase 14 inner-selection data-role gate failed.")
    for row in candidates.to_dict(orient="records"):
        resolved = json.loads(str(row["resolved_config"]))
        ae_config = resolved.get("ae_config")
        if ae_config is None:
            if not pd.isna(row["reconstruction_contract_hash"]):
                raise ValueError("Control unexpectedly declares reconstruction contract.")
            continue
        objective = str(ae_config["reconstruction_objective"])
        if str(row["reconstruction_contract_hash"]) != str(
            plan["reconstruction_contracts"][objective]["contract_hash"]
        ):
            raise ValueError("AE candidate reconstruction contract linkage mismatch.")
    expected = candidates.groupby(["evaluation_unit", "method_family"]).size()
    observed = metrics.loc[metrics["metric"].eq("rmse")].groupby(
        ["evaluation_unit", "method_family"]
    ).size()
    if not expected.equals(observed):
        raise ValueError("Phase 14 inner-selection candidate coverage mismatch.")
    winners = metrics.loc[
        metrics["metric"].eq("rmse") & metrics["selected_within_family"].astype(bool)
    ]
    if not winners.groupby(["evaluation_unit", "method_family"]).size().eq(1).all():
        raise ValueError("Phase 14 inner-selection winner cardinality mismatch.")
    for _, rows in metrics.loc[metrics["metric"].eq("rmse")].groupby(
        ["evaluation_unit", "method_family"], sort=False
    ):
        expected_winner = rows.sort_values(
            ["value", "policy_hash", "policy_id"], kind="mergesort"
        ).iloc[0]["policy_id"]
        actual = rows.loc[rows["selected_within_family"].astype(bool), "policy_id"].iloc[
            0
        ]
        if str(expected_winner) != str(actual):
            raise ValueError("Phase 14 inner-selection replay mismatch.")


def _assert_claims(
    claims: pd.DataFrame,
    frozen: Mapping[str, Any],
    retention_hash: str,
    final_predictions: pd.DataFrame,
    final_metrics: pd.DataFrame,
    registry_records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    if set(claims) != {
        "evaluation_unit",
        "method_family",
        "frozen_policy_hash",
        "retention_decision_hash",
        "registry_key",
        "registry_metrics_hash",
        "status",
        "outer_test_prediction_batches",
        "prediction_hash",
        "state_hash",
        "claim_hash",
    }:
        raise ValueError("Phase 14 evaluation-claim schema mismatch.")
    expected = {
        (str(row["evaluation_unit"]), str(row["method_family"])): row
        for row in frozen["policies"]
    }
    if (
        len(claims) != len(expected)
        or claims.duplicated(["evaluation_unit", "method_family"]).any()
        or not claims["status"].eq("metrics_complete").all()
        or not claims["outer_test_prediction_batches"].eq(1).all()
        or not claims["retention_decision_hash"].eq(retention_hash).all()
    ):
        raise ValueError("Phase 14 test-evaluation claims mismatch.")
    for row in claims.to_dict(orient="records"):
        payload = {key: value for key, value in row.items() if key != "claim_hash"}
        key = (str(row["evaluation_unit"]), str(row["method_family"]))
        prediction_rows = final_predictions.loc[
            final_predictions["evaluation_unit"].astype(str).eq(key[0])
            & final_predictions["method_family"].astype(str).eq(key[1])
        ]
        metric_rows = final_metrics.loc[
            final_metrics["evaluation_unit"].astype(str).eq(key[0])
            & final_metrics["method_family"].astype(str).eq(key[1])
        ]
        registry = registry_records.get(key)
        if (
            str(row["claim_hash"]) != stable_hash(payload)
            or str(row["frozen_policy_hash"])
            != expected[key]["frozen_policy_hash"]
            or registry is None
            or str(row["registry_key"]) != str(registry["registry_key"])
            or str(row["registry_metrics_hash"])
            != str(registry["metrics_hash"])
            or prediction_rows["prediction_hash"].nunique() != 1
            or str(row["prediction_hash"])
            != str(prediction_rows["prediction_hash"].iloc[0])
            or prediction_rows["state_hash"].nunique() != 1
            or str(row["state_hash"]) != str(prediction_rows["state_hash"].iloc[0])
            or metric_rows["prediction_hash"].nunique() != 1
            or str(row["prediction_hash"])
            != str(metric_rows["prediction_hash"].iloc[0])
        ):
            raise ValueError("Phase 14 test-evaluation claim linkage mismatch.")


def _assert_prediction_metric_replay(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    if set(predictions) != {
        "evaluation_unit",
        "seed",
        "train_fraction",
        "method_family",
        "policy_id",
        "policy_hash",
        "source_row_id",
        "observed_yield",
        "predicted_yield",
        "data_role",
        "prediction_hash",
        "state_hash",
        "plan_hash",
        "frozen_policy_hash",
        "test_evaluation_count",
        "test_used_for_selection_or_retention",
        "selected_within_family",
    } - (set() if "selected_within_family" in predictions else {"selected_within_family"}):
        raise ValueError("Phase 14 prediction artifact schema mismatch.")
    if set(metrics) != {
        "evaluation_unit",
        "seed",
        "train_fraction",
        "method_family",
        "policy_id",
        "policy_hash",
        "metric",
        "value",
        "n_samples",
        "data_role",
        "prediction_hash",
        "state_hash",
        "plan_hash",
        "frozen_policy_hash",
        "test_evaluation_count",
        "test_used_for_selection_or_retention",
        "selected_within_family",
    } - (set() if "selected_within_family" in metrics else {"selected_within_family"}):
        raise ValueError("Phase 14 metric artifact schema mismatch.")
    keys = [
        "evaluation_unit",
        "method_family",
        "policy_id",
        "policy_hash",
        "data_role",
        "prediction_hash",
        "state_hash",
    ]
    for key, rows in predictions.groupby(keys, dropna=False, sort=False):
        observed = rows["observed_yield"].to_numpy(float)
        predicted = rows["predicted_yield"].to_numpy(float)
        if (
            rows["prediction_hash"].nunique(dropna=False) != 1
            or str(rows["prediction_hash"].iloc[0]) != _array_hash(predicted)
        ):
            raise ValueError("Phase 14 ordered prediction hash mismatch.")
        metric_rows = metrics
        for column, value in zip(keys, key, strict=True):
            if pd.isna(value):
                metric_rows = metric_rows.loc[metric_rows[column].isna()]
            else:
                metric_rows = metric_rows.loc[metric_rows[column].eq(value)]
        expected_metrics = set(_METRICS) & set(metrics["metric"])
        if (
            set(metric_rows["metric"]) != expected_metrics
            or len(metric_rows) != len(expected_metrics)
            or metric_rows["metric"].duplicated().any()
        ):
            raise ValueError("Phase 14 metric coverage mismatch.")
        for metric_row in metric_rows.to_dict(orient="records"):
            expected = float(_METRICS[str(metric_row["metric"])](observed, predicted))
            if not math.isclose(
                expected, float(metric_row["value"]), rel_tol=1e-10, abs_tol=1e-10
            ):
                raise ValueError("Phase 14 metric replay mismatch.")


def _frames_close(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if tuple(left.columns) != tuple(right.columns) or len(left) != len(right):
        return False
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
            check_dtype=False,
        )
    except AssertionError:
        return False
    return True
