"""Leakage-safe benchmark for low-complexity learned representations."""

from __future__ import annotations

import importlib.metadata
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.data.outcome_access import read_allowed_outcomes
from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.isolated_fit_worker import run_isolated_fit
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_saved_random_split_unit,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.representations.low_complexity import (
    LOW_COMPLEXITY_METHODS,
    LowComplexityConfig,
    fit_low_complexity_representation,
)
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
)

LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION = "bh-low-complexity-benchmark-v1"
TRAINING_PROTOCOL = {
    "search_representation_fit_rows": "saved_nested_training_subset_only",
    "search_width_selection_rows": "saved_outer_validation_only",
    "selection_scope": "within_method_family_hyperparameters",
    "cross_method_selection": False,
    "final_representation_refit_rows": "saved_nested_training_subset_only",
    "outer_test_prediction_batches_per_unit_method": 1,
}
_METRICS = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "spearman": spearman_corr,
}
_OUTPUTS = (
    "benchmark_plan.json",
    "split_units.csv",
    "candidate_policies.csv",
    "search_predictions.csv",
    "search_metrics.csv",
    "search_fit_audit.csv",
    "frozen_method_policies.json",
    "evaluation_claims.csv",
    "refit_audit.csv",
    "resource_metrics.csv",
    "final_predictions.csv",
    "final_test_metrics.csv",
    "summary.csv",
)
_PLAN_FIELDS = {
    "schema_version",
    "status",
    "dataset_hash",
    "canonical_split_hash",
    "canonical_split_artifact_hashes",
    "feature_contract",
    "feature_metadata_hash",
    "split_units",
    "candidate_policies",
    "training_protocol",
    "resolved_scientific_config",
    "config_hash",
    "selection_metric",
    "selection_scope",
    "cross_method_selection",
    "test_labels_accessed",
    "test_predictions_generated",
    "test_metrics_generated",
}
_MANIFEST_FIELDS = {
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
    "resolved_scientific_config",
    "dependency_versions",
    "command",
    "evaluation_unit_count",
    "method_family_count",
    "candidate_count",
    "frozen_policy_count",
    "search_prediction_row_count",
    "search_metric_row_count",
    "search_fit_audit_row_count",
    "refit_audit_row_count",
    "evaluation_claim_row_count",
    "resource_metric_row_count",
    "final_prediction_row_count",
    "final_test_metric_row_count",
    "summary_row_count",
    "selection_data_roles",
    "selection_scope",
    "cross_method_selection",
    "outer_test_labels_accessed_after_policy_freeze",
    "test_evaluated",
    "test_used_for_selection",
    "test_evaluation_count_per_unit_method",
    "output_hashes",
}


def run_low_complexity_benchmark(
    config: str | Path | Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Freeze validation-selected settings, then score all six test methods."""
    contract = _resolve_contract(_load_config(config))
    saved = load_saved_canonical_split_identities(
        contract["dataset_path"],
        contract["canonical_split_directory"],
        requested_seeds=contract["random_seeds"],
        requested_fractions=contract["random_fractions"],
    )
    units = tuple(
        build_saved_random_split_unit(
            saved, seed=seed, train_fraction=fraction
        )
        for seed in contract["random_seeds"]
        for fraction in contract["random_fractions"]
    )
    if not units:
        raise ValueError("Low-complexity benchmark has no split units.")
    canonical = saved.canonical.copy()
    source_ids = canonical["source_row_id"].astype(str).tolist()
    positions = {
        source_id: position
        for position, source_id in enumerate(source_ids)
    }
    identity_frame = canonical.copy()
    identity_frame["yield"] = 0.0
    X_all, _, feature_names, feature_metadata = (
        build_feature_matrix_with_metadata(
            identity_frame, contract["feature_config"]
        )
    )
    X_all = np.asarray(X_all, dtype=np.float32)
    feature_contract = feature_contract_record(
        feature_metadata, feature_names
    )
    candidate_rows = _candidate_policy_rows(contract, units)
    scientific_config = {
        "dataset_path": str(contract["dataset_path"]),
        "canonical_split_directory": str(
            contract["canonical_split_directory"]
        ),
        "random_seeds": list(contract["random_seeds"]),
        "random_fractions": list(contract["random_fractions"]),
        "feature_config": contract["feature_config"],
        "families": list(contract["families"]),
        "latent_widths": list(contract["latent_widths"]),
        "ridge_alpha": contract["ridge_alpha"],
        "linear_autoencoder": contract["linear_autoencoder"],
        "neural": contract["neural"],
        "metrics": list(contract["metrics"]),
        "selection_metric": contract["selection_metric"],
        "base_seed": contract["base_seed"],
    }
    plan = {
        "schema_version": LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION,
        "status": "frozen_before_any_outcome_loading",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "canonical_split_artifact_hashes": saved.artifact_hashes,
        "feature_contract": feature_contract,
        "feature_metadata_hash": feature_contract[
            "feature_metadata_hash"
        ],
        "split_units": [unit.audit_record for unit in units],
        "candidate_policies": candidate_rows,
        "training_protocol": TRAINING_PROTOCOL,
        "resolved_scientific_config": scientific_config,
        "config_hash": stable_hash(scientific_config),
        "selection_metric": contract["selection_metric"],
        "selection_scope": "within_method_family_hyperparameters",
        "cross_method_selection": False,
        "test_labels_accessed": False,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
    }
    plan["plan_hash"] = stable_hash(plan)

    output = Path(
        output_directory
        if output_directory is not None
        else contract["output_directory"]
    )
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite low-complexity output: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        name.removesuffix(".json").removesuffix(".csv"): output / name
        for name in _OUTPUTS
    }
    paths["output_directory"] = output
    paths["manifest"] = output / "manifest.json"
    _write_json(paths["benchmark_plan"], plan)
    pd.DataFrame([unit.audit_record for unit in units]).to_csv(
        paths["split_units"], index=False
    )
    pd.DataFrame(candidate_rows).to_csv(
        paths["candidate_policies"], index=False
    )

    by_source = canonical.set_index("source_row_id", drop=False)
    search_prediction_rows: list[dict[str, Any]] = []
    search_metric_rows: list[dict[str, Any]] = []
    search_audit_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    candidate_by_unit = _candidate_rows_by_unit(candidate_rows)
    candidate_by_hash = {
        row["policy_hash"]: row for row in candidate_rows
    }
    for unit in units:
        visible_ids = tuple(
            sorted(unit.train_source_ids + unit.validation_source_ids)
        )
        outcomes = read_allowed_outcomes(
            contract["dataset_path"], source_ids, visible_ids
        )
        X_train, y_train = _partition_arrays(
            unit.train_source_ids, X_all, positions, outcomes
        )
        X_validation, y_validation = _partition_arrays(
            unit.validation_source_ids, X_all, positions, outcomes
        )
        train_groups = tuple(
            by_source.loc[
                list(unit.train_source_ids), "canonical_reaction_key"
            ].astype(str)
        )
        unit_search_rows: list[dict[str, Any]] = []
        for policy in candidate_by_unit[unit.evaluation_unit]:
            model_config = _model_config(policy)
            execution = run_isolated_fit(
                fit_low_complexity_representation,
                args=(X_train, y_train, X_validation, model_config),
                kwargs={
                    "train_source_ids": unit.train_source_ids,
                    "train_group_ids": train_groups,
                },
                timeout_seconds=contract["worker_timeout_seconds"],
            )
            result = execution.scientific_result
            _assert_result_identity(result, policy)
            search_prediction_rows.extend(
                _prediction_rows(
                    unit,
                    policy,
                    unit.validation_source_ids,
                    y_validation,
                    result,
                    split="valid",
                    plan_hash=plan["plan_hash"],
                )
            )
            metrics = _metric_values(
                y_validation, result.predictions, contract["metrics"]
            )
            for metric_name, value in metrics.items():
                row = {
                    **_evaluation_fields(unit, policy),
                    "split": "valid",
                    "metric": metric_name,
                    "value": value,
                    "n_fit": len(unit.train_source_ids),
                    "n_evaluation": len(unit.validation_source_ids),
                    "prediction_hash": result.prediction_hash,
                    "state_hash": result.state_hash,
                    "selected_within_method": False,
                    "test_used_for_selection": False,
                    "plan_hash": plan["plan_hash"],
                }
                search_metric_rows.append(row)
                unit_search_rows.append(row)
            search_audit_rows.append(
                _fit_audit_row(
                    unit,
                    policy,
                    result,
                    phase="search_refit",
                    feature_metadata_hash=plan["feature_metadata_hash"],
                    plan_hash=plan["plan_hash"],
                )
            )
            resource_rows.append(
                {
                    **_evaluation_fields(unit, policy),
                    "phase": "search_refit",
                    **execution.resource_usage.to_dict(),
                    "training_time_seconds": (
                        execution.resource_usage.elapsed_seconds
                    ),
                    "training_time_scope": (
                        "fit_transform_and_prediction_upper_bound"
                    ),
                    "peak_memory_bytes": (
                        execution.resource_usage.rss_peak_bytes
                    ),
                    "plan_hash": plan["plan_hash"],
                }
            )
        for family in contract["families"]:
            family_rows = [
                row
                for row in unit_search_rows
                if row["method_family"] == family
                and row["metric"] == contract["selection_metric"]
            ]
            winner = _select_candidate(family_rows)
            winner_policy = candidate_by_hash[winner["policy_hash"]]
            for row in search_metric_rows:
                if (
                    row["evaluation_unit"] == unit.evaluation_unit
                    and row["method_family"] == family
                ):
                    row["selected_within_method"] = (
                        row["policy_hash"] == winner["policy_hash"]
                    )
            payload = {
                "schema_version": LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION,
                "status": "frozen_before_outer_test_outcome_loading",
                "evaluation_unit": unit.evaluation_unit,
                "method_family": family,
                "selected_policy_id": winner["policy_id"],
                "selected_policy_hash": winner["policy_hash"],
                "selected_model_config": json.loads(
                    winner_policy["resolved_model_config"]
                ),
                "selection_metric": contract["selection_metric"],
                "selection_value": winner["value"],
                "selection_data_roles": ["saved_train", "saved_validation"],
                "training_protocol": TRAINING_PROTOCOL,
                "dataset_hash": saved.dataset_hash,
                "exact_split_hash": unit.exact_split_hash,
                "aggregate_split_hash": unit.aggregate_assignment_hash,
                "feature_metadata_hash": plan["feature_metadata_hash"],
                "config_hash": plan["config_hash"],
                "commit_hash": _git_commit(),
                "validation_prediction_hash": winner["prediction_hash"],
                "plan_hash": plan["plan_hash"],
                "test_labels_accessed": False,
                "test_predictions_generated": False,
                "test_metrics_generated": False,
            }
            payload["frozen_policy_hash"] = stable_hash(payload)
            selected_records.append(payload)

    pd.DataFrame(search_prediction_rows).to_csv(
        paths["search_predictions"], index=False
    )
    pd.DataFrame(search_metric_rows).to_csv(
        paths["search_metrics"], index=False
    )
    pd.DataFrame(search_audit_rows).to_csv(
        paths["search_fit_audit"], index=False
    )
    frozen_document = {
        "schema_version": LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION,
        "status": "frozen_before_outer_test_outcome_loading",
        "plan_hash": plan["plan_hash"],
        "selection_scope": "within_method_family_hyperparameters",
        "cross_method_selection": False,
        "policy_count": len(selected_records),
        "policies": selected_records,
        "test_labels_accessed": False,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
    }
    frozen_document["frozen_document_hash"] = stable_hash(frozen_document)
    _write_json(paths["frozen_method_policies"], frozen_document)
    loaded_frozen = _load_frozen_for_final(
        paths["frozen_method_policies"], plan
    )
    selected_records = list(loaded_frozen["policies"])
    frozen_by_key = {
        (record["evaluation_unit"], record["method_family"]): record
        for record in selected_records
    }
    claim_rows = _initial_evaluation_claims(selected_records)
    pd.DataFrame(claim_rows).to_csv(paths["evaluation_claims"], index=False)
    final_prediction_rows: list[dict[str, Any]] = []
    final_metric_rows: list[dict[str, Any]] = []
    refit_audit_rows: list[dict[str, Any]] = []
    for unit in units:
        allowed_ids = tuple(
            sorted(unit.train_source_ids + unit.test_source_ids)
        )
        outcomes = read_allowed_outcomes(
            contract["dataset_path"], source_ids, allowed_ids
        )
        X_train, y_train = _partition_arrays(
            unit.train_source_ids, X_all, positions, outcomes
        )
        X_test, y_test = _partition_arrays(
            unit.test_source_ids, X_all, positions, outcomes
        )
        train_groups = tuple(
            by_source.loc[
                list(unit.train_source_ids), "canonical_reaction_key"
            ].astype(str)
        )
        for family in contract["families"]:
            frozen = frozen_by_key[(unit.evaluation_unit, family)]
            policy = candidate_by_hash[frozen["selected_policy_hash"]]
            execution = run_isolated_fit(
                fit_low_complexity_representation,
                args=(
                    X_train,
                    y_train,
                    X_test,
                    _model_config(policy),
                ),
                kwargs={
                    "train_source_ids": unit.train_source_ids,
                    "train_group_ids": train_groups,
                },
                timeout_seconds=contract["worker_timeout_seconds"],
            )
            result = execution.scientific_result
            _assert_result_identity(result, policy)
            final_prediction_rows.extend(
                _prediction_rows(
                    unit,
                    policy,
                    unit.test_source_ids,
                    y_test,
                    result,
                    split="test",
                    plan_hash=plan["plan_hash"],
                    frozen_policy_hash=frozen["frozen_policy_hash"],
                )
            )
            metrics = _metric_values(
                y_test, result.predictions, contract["metrics"]
            )
            final_metric_rows.extend(
                {
                    **_evaluation_fields(unit, policy),
                    "split": "test",
                    "metric": metric_name,
                    "value": value,
                    "n_refit": len(unit.train_source_ids),
                    "n_test": len(unit.test_source_ids),
                    "prediction_hash": result.prediction_hash,
                    "state_hash": result.state_hash,
                    "actual_latent_width": result.actual_latent_width,
                    "deployed_predictive_parameter_count": (
                        result.deployed_predictive_parameter_count
                    ),
                    "fitted_state_scalar_count": (
                        result.fitted_state_scalar_count
                    ),
                    "frozen_policy_hash": frozen["frozen_policy_hash"],
                    "test_evaluation_count": 1,
                    "test_used_for_selection": False,
                    "plan_hash": plan["plan_hash"],
                }
                for metric_name, value in metrics.items()
            )
            refit_audit_rows.append(
                _fit_audit_row(
                    unit,
                    policy,
                    result,
                    phase="final_refit",
                    feature_metadata_hash=plan["feature_metadata_hash"],
                    plan_hash=plan["plan_hash"],
                    frozen_policy_hash=frozen["frozen_policy_hash"],
                )
            )
            resource_rows.append(
                {
                    **_evaluation_fields(unit, policy),
                    "phase": "final_refit",
                    **execution.resource_usage.to_dict(),
                    "training_time_seconds": (
                        execution.resource_usage.elapsed_seconds
                    ),
                    "training_time_scope": (
                        "fit_transform_and_prediction_upper_bound"
                    ),
                    "peak_memory_bytes": (
                        execution.resource_usage.rss_peak_bytes
                    ),
                    "plan_hash": plan["plan_hash"],
                }
            )
            claim = next(
                row
                for row in claim_rows
                if row["evaluation_unit"] == unit.evaluation_unit
                and row["method_family"] == family
            )
            claim["status"] = "complete"
            claim["outer_test_prediction_batches"] = 1
            claim["prediction_hash"] = result.prediction_hash
            claim["state_hash"] = result.state_hash
            claim["claim_hash"] = _claim_hash(claim)
            pd.DataFrame(claim_rows).to_csv(
                paths["evaluation_claims"], index=False
            )

    final_predictions = pd.DataFrame(final_prediction_rows)
    final_metrics = pd.DataFrame(final_metric_rows)
    refit_audit = pd.DataFrame(refit_audit_rows)
    resources = pd.DataFrame(resource_rows)
    summary = _summarize(final_metrics)
    refit_audit.to_csv(paths["refit_audit"], index=False)
    resources.to_csv(paths["resource_metrics"], index=False)
    final_predictions.to_csv(paths["final_predictions"], index=False)
    final_metrics.to_csv(paths["final_test_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)

    output_hashes = {
        name: sha256_file(output / name) for name in _OUTPUTS
    }
    manifest = {
        "schema_version": LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION,
        "status": "corrected_revalidation_complete",
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "feature_metadata_hash": plan["feature_metadata_hash"],
        "config_hash": plan["config_hash"],
        "plan_hash": plan["plan_hash"],
        "frozen_document_hash": frozen_document[
            "frozen_document_hash"
        ],
        "resolved_scientific_config": scientific_config,
        "dependency_versions": _dependency_versions(),
        "command": _command_record(config, output),
        "evaluation_unit_count": len(units),
        "method_family_count": len(contract["families"]),
        "candidate_count": len(candidate_rows),
        "frozen_policy_count": len(selected_records),
        "search_prediction_row_count": len(search_prediction_rows),
        "search_metric_row_count": len(search_metric_rows),
        "search_fit_audit_row_count": len(search_audit_rows),
        "refit_audit_row_count": len(refit_audit),
        "evaluation_claim_row_count": len(claim_rows),
        "resource_metric_row_count": len(resources),
        "final_prediction_row_count": len(final_predictions),
        "final_test_metric_row_count": len(final_metrics),
        "summary_row_count": len(summary),
        "selection_data_roles": ["saved_train", "saved_validation"],
        "selection_scope": "within_method_family_hyperparameters",
        "cross_method_selection": False,
        "outer_test_labels_accessed_after_policy_freeze": True,
        "test_evaluated": True,
        "test_used_for_selection": False,
        "test_evaluation_count_per_unit_method": 1,
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    _write_json(paths["manifest"], manifest)
    validate_low_complexity_benchmark(output)
    return paths


def validate_low_complexity_benchmark(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Replay plans, model fits, within-family selection, and final metrics."""
    root = Path(output_directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed_manifest_hash = manifest.pop("manifest_hash", None)
    if (
        claimed_manifest_hash != stable_hash(manifest)
        or set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema_version")
        != LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION
        or manifest.get("status") != "corrected_revalidation_complete"
        or manifest.get("selection_scope")
        != "within_method_family_hyperparameters"
        or manifest.get("selection_data_roles")
        != ["saved_train", "saved_validation"]
        or manifest.get("cross_method_selection") is not False
        or manifest.get("outer_test_labels_accessed_after_policy_freeze")
        is not True
        or manifest.get("test_evaluated") is not True
        or manifest.get("test_used_for_selection") is not False
        or manifest.get("test_evaluation_count_per_unit_method") != 1
        or int(manifest.get("method_family_count", -1))
        != len(LOW_COMPLEXITY_METHODS)
    ):
        raise ValueError("Invalid low-complexity completion manifest.")
    manifest["manifest_hash"] = claimed_manifest_hash
    if set(manifest.get("output_hashes", {})) != set(_OUTPUTS):
        raise ValueError("Low-complexity output hash coverage mismatch.")
    for name, digest in manifest["output_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(
                f"Low-complexity output hash mismatch: {name}."
            )
    plan = json.loads((root / "benchmark_plan.json").read_text())
    claimed_plan_hash = plan.pop("plan_hash", None)
    if (
        claimed_plan_hash != stable_hash(plan)
        or set(plan) != _PLAN_FIELDS
        or plan.get("schema_version")
        != LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION
        or plan.get("status") != "frozen_before_any_outcome_loading"
        or plan.get("training_protocol") != TRAINING_PROTOCOL
        or plan.get("selection_scope")
        != "within_method_family_hyperparameters"
        or plan.get("selection_metric")
        != plan.get("resolved_scientific_config", {}).get(
            "selection_metric"
        )
        or plan.get("config_hash")
        != stable_hash(plan.get("resolved_scientific_config"))
        or plan.get("cross_method_selection") is not False
        or plan.get("test_labels_accessed") is not False
        or plan.get("test_predictions_generated") is not False
        or plan.get("test_metrics_generated") is not False
    ):
        raise ValueError("Invalid frozen low-complexity plan.")
    plan["plan_hash"] = claimed_plan_hash
    if (
        manifest["plan_hash"] != claimed_plan_hash
        or manifest["config_hash"] != plan["config_hash"]
        or manifest["resolved_scientific_config"]
        != plan["resolved_scientific_config"]
        or manifest["config_hash"]
        != stable_hash(manifest["resolved_scientific_config"])
    ):
        raise ValueError("Low-complexity manifest-plan mismatch.")
    scientific = plan["resolved_scientific_config"]
    saved = load_saved_canonical_split_identities(
        scientific["dataset_path"],
        scientific["canonical_split_directory"],
        requested_seeds=scientific["random_seeds"],
        requested_fractions=scientific["random_fractions"],
    )
    if (
        manifest["dataset_hash"] != saved.dataset_hash
        or manifest["canonical_split_hash"] != saved.aggregate_split_hash
        or plan["dataset_hash"] != saved.dataset_hash
        or plan["canonical_split_hash"] != saved.aggregate_split_hash
    ):
        raise ValueError("Low-complexity canonical dependency mismatch.")
    units = tuple(
        build_saved_random_split_unit(
            saved, seed=seed, train_fraction=fraction
        )
        for seed in scientific["random_seeds"]
        for fraction in scientific["random_fractions"]
    )
    if stable_hash([unit.audit_record for unit in units]) != stable_hash(
        plan["split_units"]
    ):
        raise ValueError("Low-complexity split replay mismatch.")
    contract = _contract_from_scientific(scientific, root)
    if (
        contract["families"] != LOW_COMPLEXITY_METHODS
        or contract["latent_widths"] != (8, 16)
        or len(units) != int(manifest.get("evaluation_unit_count", -1))
    ):
        raise ValueError("Low-complexity resolved contract mismatch.")
    expected_candidates = _candidate_policy_rows(contract, units)
    if (
        int(manifest.get("candidate_count", -1))
        != len(expected_candidates)
        or int(manifest.get("frozen_policy_count", -1))
        != len(units) * len(LOW_COMPLEXITY_METHODS)
    ):
        raise ValueError("Low-complexity planned count mismatch.")
    if stable_hash(expected_candidates) != stable_hash(
        plan["candidate_policies"]
    ):
        raise ValueError("Low-complexity candidate plan replay mismatch.")
    read_options = {"float_precision": "round_trip"}
    split_table = pd.read_csv(root / "split_units.csv", **read_options)
    candidate_table = pd.read_csv(
        root / "candidate_policies.csv", **read_options
    )
    search_predictions = pd.read_csv(
        root / "search_predictions.csv", **read_options
    )
    search_metrics = pd.read_csv(
        root / "search_metrics.csv", **read_options
    )
    search_audit = pd.read_csv(
        root / "search_fit_audit.csv", **read_options
    )
    frozen_document = json.loads(
        (root / "frozen_method_policies.json").read_text()
    )
    refit_audit = pd.read_csv(root / "refit_audit.csv", **read_options)
    claims = pd.read_csv(
        root / "evaluation_claims.csv", **read_options
    )
    resources = pd.read_csv(
        root / "resource_metrics.csv", **read_options
    )
    final_predictions = pd.read_csv(
        root / "final_predictions.csv", **read_options
    )
    final_metrics = pd.read_csv(
        root / "final_test_metrics.csv", **read_options
    )
    summary = pd.read_csv(root / "summary.csv", **read_options)
    _assert_manifest_counts(
        manifest,
        search_predictions=search_predictions,
        search_metrics=search_metrics,
        search_fit_audit=search_audit,
        refit_audit=refit_audit,
        evaluation_claims=claims,
        resource_metrics=resources,
        final_predictions=final_predictions,
        final_test_metrics=final_metrics,
        summary=summary,
    )
    if (
        stable_hash(_frame_records(split_table))
        != stable_hash(
            _frame_records(pd.DataFrame([u.audit_record for u in units]))
        )
        or stable_hash(_frame_records(candidate_table))
        != stable_hash(_frame_records(pd.DataFrame(expected_candidates)))
    ):
        raise ValueError("Low-complexity split or candidate artifact mismatch.")
    _assert_resource_metrics(
        resources, expected_candidates, units, claimed_plan_hash
    )

    canonical = saved.canonical.copy()
    source_ids = canonical["source_row_id"].astype(str).tolist()
    positions = {
        source_id: position
        for position, source_id in enumerate(source_ids)
    }
    identity_frame = canonical.copy()
    identity_frame["yield"] = 0.0
    X_all, _, feature_names, feature_metadata = (
        build_feature_matrix_with_metadata(
            identity_frame, scientific["feature_config"]
        )
    )
    X_all = np.asarray(X_all, dtype=np.float32)
    replay_feature_contract = feature_contract_record(
        feature_metadata, feature_names
    )
    if (
        replay_feature_contract != plan["feature_contract"]
        or manifest["feature_metadata_hash"]
        != replay_feature_contract["feature_metadata_hash"]
    ):
        raise ValueError("Low-complexity feature contract mismatch.")
    by_source = canonical.set_index("source_row_id", drop=False)
    replayed_policies = _replay_search(
        units,
        expected_candidates,
        search_predictions,
        search_metrics,
        search_audit,
        X_all,
        positions,
        by_source,
        source_ids,
        scientific,
        claimed_plan_hash,
        manifest["git_commit"],
        plan["feature_metadata_hash"],
    )
    claimed_document_hash = frozen_document.pop(
        "frozen_document_hash", None
    )
    if (
        claimed_document_hash != stable_hash(frozen_document)
        or set(frozen_document)
        != {
            "schema_version",
            "status",
            "plan_hash",
            "selection_scope",
            "cross_method_selection",
            "policy_count",
            "policies",
            "test_labels_accessed",
            "test_predictions_generated",
            "test_metrics_generated",
        }
        or frozen_document.get("schema_version")
        != LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION
        or frozen_document.get("status")
        != "frozen_before_outer_test_outcome_loading"
        or frozen_document.get("plan_hash") != claimed_plan_hash
        or frozen_document.get("selection_scope")
        != "within_method_family_hyperparameters"
        or frozen_document.get("cross_method_selection") is not False
        or frozen_document.get("test_labels_accessed") is not False
        or frozen_document.get("test_predictions_generated") is not False
        or frozen_document.get("test_metrics_generated") is not False
        or frozen_document.get("policy_count")
        != len(units) * len(LOW_COMPLEXITY_METHODS)
        or stable_hash(frozen_document.get("policies"))
        != stable_hash(replayed_policies)
    ):
        raise ValueError("Invalid frozen low-complexity policies.")
    frozen_document["frozen_document_hash"] = claimed_document_hash
    if manifest["frozen_document_hash"] != claimed_document_hash:
        raise ValueError("Low-complexity frozen-document linkage mismatch.")
    _assert_final_resource_policies(resources, replayed_policies)
    _assert_evaluation_claims(
        claims, replayed_policies, final_predictions
    )
    _replay_final(
        units,
        expected_candidates,
        replayed_policies,
        refit_audit,
        final_predictions,
        final_metrics,
        X_all,
        positions,
        by_source,
        source_ids,
        scientific,
        claimed_plan_hash,
        plan["feature_metadata_hash"],
    )
    expected_summary = _summarize(final_metrics)
    if not _records_close(
        _frame_records(expected_summary), _frame_records(summary)
    ):
        raise ValueError("Low-complexity summary replay mismatch.")
    return manifest


def _resolve_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "dataset",
            "splits",
            "features",
            "methods",
            "metrics",
            "selection_metric",
            "base_seed",
            "output",
        },
        "config",
    )
    dataset = _mapping(raw["dataset"], "dataset")
    _exact_keys(dataset, {"path"}, "dataset")
    splits = _mapping(raw["splits"], "splits")
    _exact_keys(
        splits,
        {"canonical_directory", "random_seeds", "random_fractions"},
        "splits",
    )
    methods = _mapping(raw["methods"], "methods")
    _exact_keys(
        methods,
        {
            "families",
            "latent_widths",
            "ridge_alpha",
            "linear_autoencoder",
            "neural",
        },
        "methods",
    )
    families = tuple(methods["families"])
    if families != LOW_COMPLEXITY_METHODS:
        raise ValueError(
            "methods.families must contain the canonical six families in order."
        )
    widths = tuple(methods["latent_widths"])
    if widths != (8, 16):
        raise ValueError(
            "methods.latent_widths must be [8, 16] for equal selection budgets."
        )
    linear = _mapping(
        methods["linear_autoencoder"], "methods.linear_autoencoder"
    )
    _exact_keys(
        linear, {"epochs", "learning_rate"}, "methods.linear_autoencoder"
    )
    neural = _mapping(methods["neural"], "methods.neural")
    _exact_keys(
        neural,
        {
            "max_epochs",
            "patience",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "internal_validation_fraction",
        },
        "methods.neural",
    )
    metrics = tuple(raw["metrics"])
    if (
        not metrics
        or len(metrics) != len(set(metrics))
        or any(metric not in _METRICS for metric in metrics)
    ):
        raise ValueError("metrics contains duplicates or unsupported values.")
    selection_metric = str(raw["selection_metric"])
    if selection_metric not in metrics:
        raise ValueError("selection_metric must be included in metrics.")
    output = _mapping(raw["output"], "output")
    _exact_keys(output, {"directory"}, "output")
    return {
        "dataset_path": Path(str(dataset["path"])),
        "canonical_split_directory": Path(
            str(splits["canonical_directory"])
        ),
        "random_seeds": _unique_ints(
            splits["random_seeds"], "splits.random_seeds"
        ),
        "random_fractions": _unique_fractions(
            splits["random_fractions"], "splits.random_fractions"
        ),
        "feature_config": resolve_corrected_feature_config(
            _mapping(raw["features"], "features"),
            required_kind="bh_role_separated",
        ),
        "families": families,
        "latent_widths": widths,
        "ridge_alpha": _positive_float(
            methods["ridge_alpha"], "methods.ridge_alpha"
        ),
        "linear_autoencoder": {
            "epochs": _positive_int(
                linear["epochs"], "methods.linear_autoencoder.epochs"
            ),
            "learning_rate": _positive_float(
                linear["learning_rate"],
                "methods.linear_autoencoder.learning_rate",
            ),
        },
        "neural": {
            "max_epochs": _positive_int(
                neural["max_epochs"], "methods.neural.max_epochs"
            ),
            "patience": _positive_int(
                neural["patience"], "methods.neural.patience"
            ),
            "learning_rate": _positive_float(
                neural["learning_rate"], "methods.neural.learning_rate"
            ),
            "weight_decay": _nonnegative_float(
                neural["weight_decay"], "methods.neural.weight_decay"
            ),
            "batch_size": _positive_int(
                neural["batch_size"], "methods.neural.batch_size"
            ),
            "internal_validation_fraction": _open_unit(
                neural["internal_validation_fraction"],
                "methods.neural.internal_validation_fraction",
            ),
        },
        "metrics": metrics,
        "selection_metric": selection_metric,
        "base_seed": _int(raw["base_seed"], "base_seed"),
        "output_directory": Path(str(output["directory"])),
        "worker_timeout_seconds": 600.0,
    }


def _contract_from_scientific(
    scientific: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    return {
        "dataset_path": Path(scientific["dataset_path"]),
        "canonical_split_directory": Path(
            scientific["canonical_split_directory"]
        ),
        "random_seeds": tuple(scientific["random_seeds"]),
        "random_fractions": tuple(scientific["random_fractions"]),
        "feature_config": dict(scientific["feature_config"]),
        "families": tuple(scientific["families"]),
        "latent_widths": tuple(scientific["latent_widths"]),
        "ridge_alpha": float(scientific["ridge_alpha"]),
        "linear_autoencoder": dict(scientific["linear_autoencoder"]),
        "neural": dict(scientific["neural"]),
        "metrics": tuple(scientific["metrics"]),
        "selection_metric": str(scientific["selection_metric"]),
        "base_seed": int(scientific["base_seed"]),
        "output_directory": output,
        "worker_timeout_seconds": 600.0,
    }


def _candidate_policy_rows(
    contract: Mapping[str, Any],
    units: Sequence[EvaluationSplitUnit],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        for family in contract["families"]:
            for width in contract["latent_widths"]:
                seed_parts = [unit.evaluation_unit]
                if family not in {
                    "small_bottleneck_mlp",
                    "direct_mlp_regressor",
                }:
                    seed_parts.append(family)
                seed = _derived_seed(contract["base_seed"], *seed_parts)
                if family == "linear_autoencoder":
                    learning_rate = contract["linear_autoencoder"][
                        "learning_rate"
                    ]
                    max_epochs = contract["linear_autoencoder"]["epochs"]
                    patience = max_epochs
                    weight_decay = 0.0
                    batch_size = contract["neural"]["batch_size"]
                    internal_fraction = contract["neural"][
                        "internal_validation_fraction"
                    ]
                elif family in {
                    "small_bottleneck_mlp",
                    "direct_mlp_regressor",
                }:
                    learning_rate = contract["neural"]["learning_rate"]
                    max_epochs = contract["neural"]["max_epochs"]
                    patience = contract["neural"]["patience"]
                    weight_decay = contract["neural"]["weight_decay"]
                    batch_size = contract["neural"]["batch_size"]
                    internal_fraction = contract["neural"][
                        "internal_validation_fraction"
                    ]
                else:
                    learning_rate = 0.01
                    max_epochs = 1
                    patience = 1
                    weight_decay = 0.0
                    batch_size = contract["neural"]["batch_size"]
                    internal_fraction = contract["neural"][
                        "internal_validation_fraction"
                    ]
                model_config = LowComplexityConfig(
                    method=family,
                    latent_width=int(width),
                    ridge_alpha=float(contract["ridge_alpha"]),
                    random_state=seed,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    max_epochs=int(max_epochs),
                    patience=int(patience),
                    internal_valid_fraction=float(internal_fraction),
                    batch_size=int(batch_size),
                )
                config_json = json.dumps(
                    asdict(model_config),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                payload = {
                    "evaluation_unit": unit.evaluation_unit,
                    "method_family": family,
                    "policy_id": f"{family}:width={int(width)}",
                    "requested_latent_width": int(width),
                    "model_seed": seed,
                    "resolved_model_config": config_json,
                    "model_config_hash": stable_hash(asdict(model_config)),
                    "exact_split_hash": unit.exact_split_hash,
                    "aggregate_split_hash": unit.aggregate_assignment_hash,
                }
                payload["policy_hash"] = stable_hash(payload)
                rows.append(payload)
    return sorted(
        rows,
        key=lambda row: (
            row["evaluation_unit"],
            row["method_family"],
            row["requested_latent_width"],
        ),
    )


def _candidate_rows_by_unit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["evaluation_unit"]), []).append(row)
    return result


def _model_config(policy: Mapping[str, Any]) -> LowComplexityConfig:
    record = json.loads(str(policy["resolved_model_config"]))
    config = LowComplexityConfig(**record)
    if (
        stable_hash(asdict(config)) != policy["model_config_hash"]
        or config.method != policy["method_family"]
        or config.latent_width != int(policy["requested_latent_width"])
    ):
        raise ValueError("Low-complexity policy config mismatch.")
    return config


def _load_frozen_for_final(
    path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    document = json.loads(path.read_text())
    claimed = document.pop("frozen_document_hash", None)
    if (
        claimed != stable_hash(document)
        or document.get("schema_version")
        != LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION
        or document.get("status")
        != "frozen_before_outer_test_outcome_loading"
        or document.get("plan_hash") != plan["plan_hash"]
        or document.get("selection_scope")
        != "within_method_family_hyperparameters"
        or document.get("cross_method_selection") is not False
        or document.get("test_labels_accessed") is not False
        or document.get("test_predictions_generated") is not False
        or document.get("test_metrics_generated") is not False
        or document.get("policy_count") != len(document.get("policies", []))
    ):
        raise ValueError("Final evaluation requires valid frozen method policies.")
    document["frozen_document_hash"] = claimed
    return document


def _initial_evaluation_claims(
    policies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        row = {
            "evaluation_unit": policy["evaluation_unit"],
            "method_family": policy["method_family"],
            "frozen_policy_hash": policy["frozen_policy_hash"],
            "selected_policy_hash": policy["selected_policy_hash"],
            "exact_split_hash": policy["exact_split_hash"],
            "status": "claimed_before_test_outcome_loading",
            "outer_test_prediction_batches": 0,
            "prediction_hash": None,
            "state_hash": None,
        }
        row["claim_id"] = stable_hash(
            {
                "evaluation_unit": row["evaluation_unit"],
                "method_family": row["method_family"],
                "frozen_policy_hash": row["frozen_policy_hash"],
                "exact_split_hash": row["exact_split_hash"],
            }
        )
        row["claim_hash"] = _claim_hash(row)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (row["evaluation_unit"], row["method_family"]),
    )


def _claim_hash(claim: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            key: _jsonable(value)
            for key, value in claim.items()
            if key != "claim_hash"
        }
    )


def _partition_arrays(
    source_ids: Sequence[str],
    X_all: np.ndarray,
    positions: Mapping[str, int],
    outcomes: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(
        [positions[source_id] for source_id in source_ids], dtype=int
    )
    return (
        X_all[indices],
        np.asarray(
            [outcomes[source_id] for source_id in source_ids], dtype=float
        ),
    )


def _assert_result_identity(
    result: Any, policy: Mapping[str, Any]
) -> None:
    if (
        result.method != policy["method_family"]
        or int(result.requested_latent_width)
        != int(policy["requested_latent_width"])
        or int(result.actual_latent_width)
        != int(policy["requested_latent_width"])
        or result.metadata.get("config_hash")
        != policy["model_config_hash"]
        or result.metadata.get("evaluation_labels_received") is not False
    ):
        raise ValueError("Low-complexity fitted result identity mismatch.")


def _evaluation_fields(
    unit: EvaluationSplitUnit, policy: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "evaluation_unit": unit.evaluation_unit,
        "seed": unit.seed,
        "train_fraction": unit.train_fraction,
        "method_family": policy["method_family"],
        "policy_id": policy["policy_id"],
        "policy_hash": policy["policy_hash"],
        "model_config_hash": policy["model_config_hash"],
        "requested_latent_width": policy["requested_latent_width"],
        "model_seed": policy["model_seed"],
        "exact_split_hash": unit.exact_split_hash,
        "aggregate_split_hash": unit.aggregate_assignment_hash,
    }


def _assert_row_identity(
    rows: pd.DataFrame,
    unit: EvaluationSplitUnit,
    policy: Mapping[str, Any],
) -> None:
    expected = _evaluation_fields(unit, policy)
    if rows.empty:
        raise ValueError("Low-complexity identity audit received no rows.")
    for field, value in expected.items():
        if value is None:
            valid = rows[field].isna().all()
        elif isinstance(value, float):
            valid = np.allclose(
                pd.to_numeric(rows[field], errors="coerce").to_numpy(float),
                value,
                rtol=0.0,
                atol=1e-12,
            )
        else:
            valid = rows[field].astype(str).eq(str(value)).all()
        if not valid:
            raise ValueError(
                f"Low-complexity row identity mismatch: {field}."
            )


def _prediction_rows(
    unit: EvaluationSplitUnit,
    policy: Mapping[str, Any],
    source_ids: Sequence[str],
    measured_yields: np.ndarray,
    result: Any,
    *,
    split: str,
    plan_hash: str,
    frozen_policy_hash: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            **_evaluation_fields(unit, policy),
            "split": split,
            "source_row_id": source_id,
            "measured_yield": float(measured_yields[index]),
            "prediction": float(result.predictions[index]),
            "signed_error": float(
                result.predictions[index] - measured_yields[index]
            ),
            "absolute_error": float(
                abs(result.predictions[index] - measured_yields[index])
            ),
            "actual_latent_width": result.actual_latent_width,
            "deployed_predictive_parameter_count": (
                result.deployed_predictive_parameter_count
            ),
            "fitted_state_scalar_count": result.fitted_state_scalar_count,
            "selected_epoch": result.selected_epoch,
            "state_hash": result.state_hash,
            "prediction_hash": result.prediction_hash,
            "frozen_policy_hash": frozen_policy_hash,
            "test_evaluation_count": 1 if split == "test" else 0,
            "test_used_for_selection": False,
            "plan_hash": plan_hash,
        }
        for index, source_id in enumerate(source_ids)
    ]


def _metric_values(
    y_true: np.ndarray,
    prediction: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float]:
    result = {
        metric: float(
            _METRICS[metric](
                np.asarray(y_true, dtype=float),
                np.asarray(prediction, dtype=float),
            )
        )
        for metric in metrics
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Low-complexity metrics must be finite.")
    return result


def _fit_audit_row(
    unit: EvaluationSplitUnit,
    policy: Mapping[str, Any],
    result: Any,
    *,
    phase: str,
    feature_metadata_hash: str,
    plan_hash: str,
    frozen_policy_hash: str | None = None,
) -> dict[str, Any]:
    payload = {
        **_evaluation_fields(unit, policy),
        "phase": phase,
        "feature_metadata_hash": feature_metadata_hash,
        "representation_fit_source_id_hash": stable_hash(
            list(unit.train_source_ids)
        ),
        "saved_validation_source_id_hash": stable_hash(
            list(unit.validation_source_ids)
        ),
        "outer_test_source_id_hash": stable_hash(
            list(unit.test_source_ids)
        ),
        "fit_validation_overlap_count": 0,
        "fit_test_overlap_count": 0,
        "internal_fit_source_id_hash": result.internal_fit_source_id_hash,
        "internal_validation_source_id_hash": (
            result.internal_validation_source_id_hash
        ),
        "internal_split_hash": result.internal_split_hash,
        "n_fit": len(unit.train_source_ids),
        "n_saved_validation": len(unit.validation_source_ids),
        "n_test": len(unit.test_source_ids),
        "actual_latent_width": result.actual_latent_width,
        "selected_epoch": result.selected_epoch,
        "deployed_predictive_parameter_count": (
            result.deployed_predictive_parameter_count
        ),
        "fitted_state_scalar_count": result.fitted_state_scalar_count,
        "parameter_count_formula": result.metadata[
            "parameter_count_formula"
        ],
        "matched_neural_model_state_hash": result.metadata.get(
            "neural_model_state_hash"
        ),
        "fit_warnings": json.dumps(
            list(result.metadata.get("fit_warnings", ())),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "state_hash": result.state_hash,
        "prediction_hash": result.prediction_hash,
        "frozen_policy_hash": frozen_policy_hash,
        "representation_fit_excludes_saved_validation_and_test": True,
        "evaluation_labels_received_by_fit": False,
        "plan_hash": plan_hash,
    }
    payload["fit_audit_hash"] = stable_hash(_jsonable(payload))
    return payload


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("Within-method selection has no candidates.")
    if len(rows) != len({row["policy_hash"] for row in rows}):
        raise ValueError("Within-method selection candidates are duplicated.")
    return min(
        rows,
        key=lambda row: (
            float(row["value"]),
            str(row["policy_hash"]),
            str(row["policy_id"]),
        ),
    )


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["train_fraction", "method_family", "metric"], dropna=False
        )["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values(
            ["train_fraction", "method_family", "metric"],
            kind="mergesort",
        )
    )


def _replay_search(
    units: Sequence[EvaluationSplitUnit],
    candidates: Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    audit: pd.DataFrame,
    X_all: np.ndarray,
    positions: Mapping[str, int],
    by_source: pd.DataFrame,
    canonical_source_ids: Sequence[str],
    scientific: Mapping[str, Any],
    plan_hash: str,
    run_commit: str,
    feature_metadata_hash: str,
) -> list[dict[str, Any]]:
    candidates_by_unit = _candidate_rows_by_unit(candidates)
    expected_prediction_count = sum(
        len(unit.validation_source_ids)
        * len(candidates_by_unit[unit.evaluation_unit])
        for unit in units
    )
    expected_metric_count = len(candidates) * len(scientific["metrics"])
    if (
        len(predictions) != expected_prediction_count
        or len(metrics) != expected_metric_count
        or len(audit) != len(candidates)
        or not predictions["split"].eq("valid").all()
        or not metrics["split"].eq("valid").all()
        or predictions["test_used_for_selection"].map(_strict_bool).any()
        or metrics["test_used_for_selection"].map(_strict_bool).any()
        or set(predictions["plan_hash"]) != {plan_hash}
        or set(metrics["plan_hash"]) != {plan_hash}
        or set(audit["plan_hash"]) != {plan_hash}
    ):
        raise ValueError("Low-complexity search artifact coverage mismatch.")
    records: list[dict[str, Any]] = []
    for unit in units:
        visible_ids = tuple(
            sorted(unit.train_source_ids + unit.validation_source_ids)
        )
        outcomes = read_allowed_outcomes(
            scientific["dataset_path"],
            canonical_source_ids,
            visible_ids,
        )
        X_train, y_train = _partition_arrays(
            unit.train_source_ids, X_all, positions, outcomes
        )
        X_validation, y_validation = _partition_arrays(
            unit.validation_source_ids, X_all, positions, outcomes
        )
        train_groups = tuple(
            by_source.loc[
                list(unit.train_source_ids), "canonical_reaction_key"
            ].astype(str)
        )
        selection_rows: list[dict[str, Any]] = []
        for policy in candidates_by_unit[unit.evaluation_unit]:
            result = fit_low_complexity_representation(
                X_train,
                y_train,
                X_validation,
                _model_config(policy),
                train_source_ids=unit.train_source_ids,
                train_group_ids=train_groups,
            )
            _assert_result_identity(result, policy)
            observed_predictions = predictions.loc[
                predictions["evaluation_unit"].eq(unit.evaluation_unit)
                & predictions["policy_hash"].eq(policy["policy_hash"])
            ].sort_values("source_row_id", kind="mergesort")
            _assert_row_identity(observed_predictions, unit, policy)
            _assert_prediction_replay(
                observed_predictions,
                unit.validation_source_ids,
                y_validation,
                result,
                expected_split="valid",
            )
            observed_metrics = metrics.loc[
                metrics["evaluation_unit"].eq(unit.evaluation_unit)
                & metrics["policy_hash"].eq(policy["policy_hash"])
            ]
            _assert_row_identity(observed_metrics, unit, policy)
            replayed_metrics = _metric_values(
                y_validation, result.predictions, scientific["metrics"]
            )
            if (
                set(observed_metrics["metric"]) != set(scientific["metrics"])
                or observed_metrics["metric"].duplicated().any()
                or set(observed_metrics["prediction_hash"])
                != {result.prediction_hash}
                or set(observed_metrics["state_hash"]) != {result.state_hash}
            ):
                raise ValueError(
                    "Low-complexity search metric provenance mismatch."
                )
            for metric_name, value in replayed_metrics.items():
                observed_value = observed_metrics.loc[
                    observed_metrics["metric"].eq(metric_name), "value"
                ]
                if len(observed_value) != 1 or not math.isclose(
                    float(observed_value.iloc[0]),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        "Low-complexity validation metric replay mismatch."
                    )
            expected_audit = _fit_audit_row(
                unit,
                policy,
                result,
                phase="search_refit",
                feature_metadata_hash=feature_metadata_hash,
                plan_hash=plan_hash,
            )
            observed_audit = audit.loc[
                audit["evaluation_unit"].eq(unit.evaluation_unit)
                & audit["policy_hash"].eq(policy["policy_hash"])
            ]
            if len(observed_audit) != 1 or not _records_close(
                [_jsonable(expected_audit)],
                _frame_records(observed_audit),
            ):
                raise ValueError("Low-complexity search fit audit mismatch.")
            selected_metric = replayed_metrics[
                scientific["selection_metric"]
            ]
            selection_rows.append(
                {
                    **policy,
                    "value": selected_metric,
                    "prediction_hash": result.prediction_hash,
                }
            )
        for family in scientific["families"]:
            winner = _select_candidate(
                [
                    row
                    for row in selection_rows
                    if row["method_family"] == family
                ]
            )
            observed_selected = metrics.loc[
                metrics["evaluation_unit"].eq(unit.evaluation_unit)
                & metrics["method_family"].eq(family),
                ["policy_hash", "selected_within_method"],
            ]
            expected_selected = observed_selected["policy_hash"].eq(
                winner["policy_hash"]
            )
            observed_selected_flags = observed_selected[
                "selected_within_method"
            ].map(_strict_bool)
            selected_hashes = set(
                observed_selected.loc[
                    observed_selected_flags,
                    "policy_hash",
                ]
            )
            if (
                selected_hashes != {winner["policy_hash"]}
                or not observed_selected_flags.eq(expected_selected).all()
            ):
                raise ValueError(
                    "Low-complexity within-method winner replay mismatch."
                )
            payload = {
                "schema_version": LOW_COMPLEXITY_BENCHMARK_SCHEMA_VERSION,
                "status": "frozen_before_outer_test_outcome_loading",
                "evaluation_unit": unit.evaluation_unit,
                "method_family": family,
                "selected_policy_id": winner["policy_id"],
                "selected_policy_hash": winner["policy_hash"],
                "selected_model_config": json.loads(
                    winner["resolved_model_config"]
                ),
                "selection_metric": scientific["selection_metric"],
                "selection_value": winner["value"],
                "selection_data_roles": ["saved_train", "saved_validation"],
                "training_protocol": TRAINING_PROTOCOL,
                "dataset_hash": unit.dataset_hash,
                "exact_split_hash": unit.exact_split_hash,
                "aggregate_split_hash": unit.aggregate_assignment_hash,
                "feature_metadata_hash": feature_metadata_hash,
                "config_hash": stable_hash(scientific),
                "commit_hash": run_commit,
                "validation_prediction_hash": winner["prediction_hash"],
                "plan_hash": plan_hash,
                "test_labels_accessed": False,
                "test_predictions_generated": False,
                "test_metrics_generated": False,
            }
            payload["frozen_policy_hash"] = stable_hash(payload)
            records.append(payload)
    return records


def _replay_final(
    units: Sequence[EvaluationSplitUnit],
    candidates: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    audit: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    X_all: np.ndarray,
    positions: Mapping[str, int],
    by_source: pd.DataFrame,
    canonical_source_ids: Sequence[str],
    scientific: Mapping[str, Any],
    plan_hash: str,
    feature_metadata_hash: str,
) -> None:
    candidate_by_hash = {
        row["policy_hash"]: row for row in candidates
    }
    policy_by_key = {
        (row["evaluation_unit"], row["method_family"]): row
        for row in policies
    }
    expected_fit_count = len(units) * len(scientific["families"])
    expected_prediction_count = sum(
        len(unit.test_source_ids) * len(scientific["families"])
        for unit in units
    )
    if (
        len(audit) != expected_fit_count
        or len(predictions) != expected_prediction_count
        or len(metrics)
        != expected_fit_count * len(scientific["metrics"])
        or not predictions["split"].eq("test").all()
        or not metrics["split"].eq("test").all()
        or not predictions["test_evaluation_count"].eq(1).all()
        or not metrics["test_evaluation_count"].eq(1).all()
        or predictions["test_used_for_selection"].map(_strict_bool).any()
        or metrics["test_used_for_selection"].map(_strict_bool).any()
        or set(predictions["plan_hash"]) != {plan_hash}
        or set(metrics["plan_hash"]) != {plan_hash}
        or set(audit["plan_hash"]) != {plan_hash}
    ):
        raise ValueError("Low-complexity final artifact coverage mismatch.")
    for unit in units:
        allowed_ids = tuple(
            sorted(unit.train_source_ids + unit.test_source_ids)
        )
        outcomes = read_allowed_outcomes(
            scientific["dataset_path"],
            canonical_source_ids,
            allowed_ids,
        )
        X_train, y_train = _partition_arrays(
            unit.train_source_ids, X_all, positions, outcomes
        )
        X_test, y_test = _partition_arrays(
            unit.test_source_ids, X_all, positions, outcomes
        )
        train_groups = tuple(
            by_source.loc[
                list(unit.train_source_ids), "canonical_reaction_key"
            ].astype(str)
        )
        for family in scientific["families"]:
            frozen = policy_by_key[(unit.evaluation_unit, family)]
            policy = candidate_by_hash[frozen["selected_policy_hash"]]
            result = fit_low_complexity_representation(
                X_train,
                y_train,
                X_test,
                _model_config(policy),
                train_source_ids=unit.train_source_ids,
                train_group_ids=train_groups,
            )
            observed_predictions = predictions.loc[
                predictions["evaluation_unit"].eq(unit.evaluation_unit)
                & predictions["method_family"].eq(family)
            ].sort_values("source_row_id", kind="mergesort")
            _assert_row_identity(observed_predictions, unit, policy)
            _assert_prediction_replay(
                observed_predictions,
                unit.test_source_ids,
                y_test,
                result,
                expected_split="test",
                frozen_policy_hash=frozen["frozen_policy_hash"],
            )
            observed_metrics = metrics.loc[
                metrics["evaluation_unit"].eq(unit.evaluation_unit)
                & metrics["method_family"].eq(family)
            ]
            _assert_row_identity(observed_metrics, unit, policy)
            replayed_metrics = _metric_values(
                y_test, result.predictions, scientific["metrics"]
            )
            if (
                set(observed_metrics["metric"]) != set(scientific["metrics"])
                or observed_metrics["metric"].duplicated().any()
                or set(observed_metrics["prediction_hash"])
                != {result.prediction_hash}
                or set(observed_metrics["state_hash"]) != {result.state_hash}
                or set(observed_metrics["frozen_policy_hash"])
                != {frozen["frozen_policy_hash"]}
                or not observed_metrics["actual_latent_width"]
                .eq(result.actual_latent_width)
                .all()
                or not observed_metrics[
                    "deployed_predictive_parameter_count"
                ]
                .eq(result.deployed_predictive_parameter_count)
                .all()
                or not observed_metrics["fitted_state_scalar_count"]
                .eq(result.fitted_state_scalar_count)
                .all()
            ):
                raise ValueError(
                    "Low-complexity final metric provenance mismatch."
                )
            for metric_name, value in replayed_metrics.items():
                observed = observed_metrics.loc[
                    observed_metrics["metric"].eq(metric_name), "value"
                ]
                if len(observed) != 1 or not math.isclose(
                    float(observed.iloc[0]),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        "Low-complexity final metric replay mismatch."
                    )
            observed_audit = audit.loc[
                audit["evaluation_unit"].eq(unit.evaluation_unit)
                & audit["method_family"].eq(family)
            ]
            if len(observed_audit) != 1:
                raise ValueError(
                    "Low-complexity final fit audit coverage mismatch."
                )
            expected_audit = _fit_audit_row(
                unit,
                policy,
                result,
                phase="final_refit",
                feature_metadata_hash=feature_metadata_hash,
                plan_hash=plan_hash,
                frozen_policy_hash=frozen["frozen_policy_hash"],
            )
            if not _records_close(
                [_jsonable(expected_audit)],
                _frame_records(observed_audit),
            ):
                raise ValueError("Low-complexity final fit audit mismatch.")


def _assert_prediction_replay(
    rows: pd.DataFrame,
    source_ids: Sequence[str],
    measured_yields: np.ndarray,
    result: Any,
    *,
    expected_split: str,
    frozen_policy_hash: str | None = None,
) -> None:
    if (
        tuple(rows["source_row_id"].astype(str)) != tuple(source_ids)
        or not rows["split"].eq(expected_split).all()
        or set(rows["prediction_hash"]) != {result.prediction_hash}
        or set(rows["state_hash"]) != {result.state_hash}
        or not rows["actual_latent_width"]
        .eq(result.actual_latent_width)
        .all()
        or not rows["deployed_predictive_parameter_count"]
        .eq(result.deployed_predictive_parameter_count)
        .all()
        or not rows["fitted_state_scalar_count"]
        .eq(result.fitted_state_scalar_count)
        .all()
        or not _nullable_scalar_equal(
            rows["selected_epoch"], result.selected_epoch
        )
    ):
        raise ValueError("Low-complexity prediction provenance mismatch.")
    prediction = np.asarray(result.predictions, dtype=float)
    measured = np.asarray(measured_yields, dtype=float)
    if (
        not np.allclose(
            rows["measured_yield"].to_numpy(float),
            measured,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            rows["prediction"].to_numpy(float),
            prediction,
            rtol=0.0,
            atol=1e-10,
        )
        or not np.allclose(
            rows["signed_error"].to_numpy(float),
            prediction - measured,
            rtol=0.0,
            atol=1e-10,
        )
        or not np.allclose(
            rows["absolute_error"].to_numpy(float),
            np.abs(prediction - measured),
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise ValueError("Low-complexity prediction replay mismatch.")
    if frozen_policy_hash is not None and set(
        rows["frozen_policy_hash"]
    ) != {frozen_policy_hash}:
        raise ValueError("Low-complexity prediction-policy mismatch.")


def _assert_resource_metrics(
    resources: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    units: Sequence[EvaluationSplitUnit],
    plan_hash: str,
) -> None:
    expected_search = {
        (
            row["evaluation_unit"],
            row["method_family"],
            row["policy_hash"],
            "search_refit",
        )
        for row in candidates
    }
    expected_final_prefixes = {
        (unit.evaluation_unit, family, "final_refit")
        for unit in units
        for family in LOW_COMPLEXITY_METHODS
    }
    observed_search = set(
        resources.loc[
            resources["phase"].eq("search_refit"),
            [
                "evaluation_unit",
                "method_family",
                "policy_hash",
                "phase",
            ],
        ].itertuples(index=False, name=None)
    )
    observed_final = set(
        resources.loc[
            resources["phase"].eq("final_refit"),
            ["evaluation_unit", "method_family", "phase"],
        ].itertuples(index=False, name=None)
    )
    numeric_columns = (
        "elapsed_seconds",
        "training_time_seconds",
        "rss_baseline_bytes",
        "rss_peak_bytes",
        "rss_increment_bytes",
        "peak_memory_bytes",
    )
    numeric = resources.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        observed_search != expected_search
        or observed_final != expected_final_prefixes
        or resources[
            [
                "evaluation_unit",
                "method_family",
                "policy_hash",
                "phase",
            ]
        ].duplicated().any()
        or numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(float)).all()
        or (numeric.to_numpy(float) < 0.0).any()
        or (
            numeric["rss_peak_bytes"].to_numpy(float)
            < numeric["rss_baseline_bytes"].to_numpy(float)
        ).any()
        or not np.array_equal(
            numeric["rss_increment_bytes"].to_numpy(float),
            (
                numeric["rss_peak_bytes"] - numeric["rss_baseline_bytes"]
            ).to_numpy(float),
        )
        or set(resources["timing_scope"]) != {"task_callable_only"}
        or set(resources["training_time_scope"])
        != {"fit_transform_and_prediction_upper_bound"}
        or set(resources["rss_scope"])
        != {"isolated_child_process_ru_maxrss"}
        or set(resources["plan_hash"]) != {plan_hash}
        or not np.array_equal(
            numeric["training_time_seconds"].to_numpy(float),
            numeric["elapsed_seconds"].to_numpy(float),
        )
        or not np.array_equal(
            numeric["peak_memory_bytes"].to_numpy(float),
            numeric["rss_peak_bytes"].to_numpy(float),
        )
    ):
        raise ValueError("Low-complexity resource audit mismatch.")


def _assert_final_resource_policies(
    resources: pd.DataFrame,
    policies: Sequence[Mapping[str, Any]],
) -> None:
    expected = {
        (
            record["evaluation_unit"],
            record["method_family"],
            record["selected_policy_hash"],
        )
        for record in policies
    }
    observed = set(
        resources.loc[
            resources["phase"].eq("final_refit"),
            ["evaluation_unit", "method_family", "policy_hash"],
        ].itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            "Low-complexity final resource-policy linkage mismatch."
        )


def _assert_evaluation_claims(
    claims: pd.DataFrame,
    policies: Sequence[Mapping[str, Any]],
    predictions: pd.DataFrame,
) -> None:
    policies_by_key = {
        (row["evaluation_unit"], row["method_family"]): row
        for row in policies
    }
    if (
        len(claims) != len(policies_by_key)
        or claims[["evaluation_unit", "method_family"]].duplicated().any()
        or not claims["status"].eq("complete").all()
        or not claims["outer_test_prediction_batches"].eq(1).all()
    ):
        raise ValueError("Low-complexity evaluation claim coverage mismatch.")
    for record in claims.to_dict("records"):
        normalized = _jsonable(record)
        policy = policies_by_key[
            (record["evaluation_unit"], record["method_family"])
        ]
        prediction_rows = predictions.loc[
            predictions["evaluation_unit"].eq(record["evaluation_unit"])
            & predictions["method_family"].eq(record["method_family"])
        ]
        if (
            record["frozen_policy_hash"] != policy["frozen_policy_hash"]
            or record["selected_policy_hash"]
            != policy["selected_policy_hash"]
            or record["exact_split_hash"] != policy["exact_split_hash"]
            or set(prediction_rows["prediction_hash"])
            != {record["prediction_hash"]}
            or set(prediction_rows["state_hash"]) != {record["state_hash"]}
            or record["claim_hash"] != _claim_hash(normalized)
        ):
            raise ValueError("Low-complexity evaluation claim mismatch.")


def _assert_manifest_counts(
    manifest: Mapping[str, Any], **tables: pd.DataFrame
) -> None:
    fields = {
        "search_predictions": "search_prediction_row_count",
        "search_metrics": "search_metric_row_count",
        "search_fit_audit": "search_fit_audit_row_count",
        "refit_audit": "refit_audit_row_count",
        "evaluation_claims": "evaluation_claim_row_count",
        "resource_metrics": "resource_metric_row_count",
        "final_predictions": "final_prediction_row_count",
        "final_test_metrics": "final_test_metric_row_count",
        "summary": "summary_row_count",
    }
    for name, table in tables.items():
        if int(manifest.get(fields[name], -1)) != len(table):
            raise ValueError(
                f"Low-complexity manifest row count mismatch: {name}."
            )


def _nullable_scalar_equal(series: pd.Series, expected: Any) -> bool:
    if expected is None:
        return series.isna().all()
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().all() and numeric.eq(float(expected)).all()


def _load_config(
    config: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Low-complexity config must be a mapping.")
    return loaded


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys mismatch: expected={sorted(expected)}, "
            f"observed={sorted(value)}."
        )


def _unique_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    result = tuple(_int(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values.")
    return result


def _unique_fractions(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    result = tuple(float(item) for item in value)
    if (
        len(result) != len(set(result))
        or any(
            not math.isfinite(item) or not 0.0 < item <= 1.0
            for item in result
        )
    ):
        raise ValueError(f"{name} must contain unique values in (0, 1].")
    return result


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _open_unit(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return result


def _derived_seed(base_seed: int, *parts: str) -> int:
    return int(base_seed) + int(stable_hash(list(parts))[:12], 16) % 1_000_000


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ValueError(f"Expected strict boolean, received {value!r}.")


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonable(record) for record in frame.to_dict("records")]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records_close(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> bool:
    if len(expected) != len(observed):
        return False
    return all(
        _values_close(_jsonable(left), _jsonable(right))
        for left, right in zip(expected, observed, strict=True)
    )


def _values_close(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if left is None or right is None:
        return left is None and right is None
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-10
        )
    return left == right


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dependency in ("numpy", "pandas", "scikit-learn", "torch", "rdkit"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = "not-installed"
    return versions


def _git_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Scientific runs require an available Git commit.") from exc
    if not value:
        raise ValueError("Scientific runs require a nonempty Git commit.")
    return value


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Unable to determine Git worktree state.") from exc


def _command_record(config: Any, output: Path) -> str:
    return (
        "python -B scripts/run_low_complexity_benchmark.py "
        f"--config {config} --output-directory {output}"
    )
