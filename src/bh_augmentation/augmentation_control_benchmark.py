"""Matched canonical benchmark for simple and chemical augmentation controls."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.augmentation.chemical_controls import (
    CHEMICAL_CONTROL_IDS,
    CHEMICAL_CONTROL_SPECS,
    build_chemical_augmentation_control,
)
from bh_augmentation.augmentation.condition_transfer import ConditionTransferConfig
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
)
from bh_augmentation.augmentation.simple_controls import (
    SIMPLE_CONTROL_IDS,
    SIMPLE_CONTROL_SPECS,
    build_simple_augmentation_control,
)
from bh_augmentation.augmentation.synthetic_identity import (
    configured_feature_hash,
    measured_canonical_keys,
)
from bh_augmentation.data.canonicalize_roles import (
    build_canonical_reaction_identity,
    canonicalize_smiles,
)
from bh_augmentation.data.reaction_roles import ROLE_TO_COLUMN
from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_saved_random_split_unit,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
)

AUGMENTATION_CONTROL_SCHEMA_VERSION = "bh-augmentation-controls-v1"
CONTROL_IDS = SIMPLE_CONTROL_IDS + CHEMICAL_CONTROL_IDS
COMPARATOR_IDS = (
    "real_only",
    "exact_duplication",
    "random_oversampling",
    "yield_stratified_oversampling",
)
_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman_corr}
_LOWER_IS_BETTER = {"rmse": True, "mae": True, "r2": False, "spearman": False}
_RAW_UNCERTAINTY_SEMANTICS = "raw_teacher_std_proxy_phase12_pending"
_UNCERTAINTY_DESCRIPTION = (
    "raw teacher standard-deviation development ablation; uncalibrated "
    "proxy pending Phase 12 and not calibrated evidence"
)
_NONCHEMICAL_DESCRIPTION = (
    "Convex feature coordinates are nonchemical controls; no seven-role "
    "reaction identity, canonical reaction key, or chemical equivalence is claimed."
)
_OUTPUT_NAMES = (
    "benchmark_plan.json",
    "split_units.csv",
    "control_metadata.csv",
    "budget_audit.csv",
    "training_audit.csv",
    "chemical_candidate_audit.csv",
    "predictions.csv",
    "metrics.csv",
    "paired_comparisons.csv",
    "summary.csv",
)


def run_augmentation_control_benchmark(
    config: str | Path | Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run all 13 frozen controls once per saved split evaluation unit."""
    raw = _load_config(config)
    contract = _resolve_contract(raw)
    saved = load_saved_canonical_split_identities(
        contract["dataset_path"],
        contract["canonical_split_directory"],
        requested_seeds=contract["random_seeds"],
        requested_fractions=contract["random_fractions"],
    )
    units = tuple(
        build_saved_random_split_unit(saved, seed=seed, train_fraction=fraction)
        for seed in contract["random_seeds"]
        for fraction in contract["random_fractions"]
    )
    if not units:
        raise ValueError("Augmentation-control benchmark has no evaluation units.")

    feature_config = resolve_corrected_feature_config(
        contract["features"], required_kind="bh_role_separated"
    )
    identity_feature_frame = saved.canonical.copy()
    identity_feature_frame["yield"] = 0.0
    X_all, _, feature_names, feature_metadata = (
        build_feature_matrix_with_metadata(identity_feature_frame, feature_config)
    )
    X_all = np.asarray(X_all, dtype=np.float32)
    feature_contract = feature_contract_record(feature_metadata, feature_names)
    model_protocol = {
        "name": contract["model"]["name"],
        "params": contract["model"]["params"],
        "fit_protocol": (
            "one frozen configuration per control; current saved training subset "
            "only; validation and test excluded from fitting"
        ),
        "seed_derivation": "base_seed_plus_stable_evaluation_unit_hash_mod_1e6",
        "sample_weight_requirement": "hard_fail_if_requested_and_unsupported",
    }
    model_protocol_hash = stable_hash(model_protocol)
    budget_protocol = {
        "policy_selection_budget_per_control": 1,
        "policy_selection_method": "predefined_before_label_loading_no_grid_search",
        "nominal_added_multiplier": contract["augmentation"][
            "nominal_added_multiplier"
        ],
        "underfill_policy": "record_rejections_never_backfill",
        "sample_reweighting_match": "total_weight_equals_n_plus_nominal_budget",
    }
    budget_protocol_hash = stable_hash(budget_protocol)
    scientific_config = {
        "dataset_path": str(contract["dataset_path"]),
        "canonical_split_directory": str(
            contract["canonical_split_directory"]
        ),
        "random_seeds": list(contract["random_seeds"]),
        "random_fractions": list(contract["random_fractions"]),
        "controls": list(CONTROL_IDS),
        "augmentation": contract["augmentation"],
        "features": feature_config,
        "model": contract["model"],
        "metrics": list(contract["metrics"]),
        "base_seed": contract["base_seed"],
    }
    plan = {
        "schema_version": AUGMENTATION_CONTROL_SCHEMA_VERSION,
        "status": "frozen_before_label_loading",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "split_units": [unit.audit_record for unit in units],
        "controls": _control_plan_records(),
        "control_selection": "none_all_predefined_controls_reported",
        "policy_selection_budget_per_control": 1,
        "model_protocol": model_protocol,
        "model_protocol_hash": model_protocol_hash,
        "budget_protocol": budget_protocol,
        "budget_protocol_hash": budget_protocol_hash,
        "feature_contract": feature_contract,
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "resolved_scientific_config": scientific_config,
        "config_hash": stable_hash(scientific_config),
        "metrics": list(contract["metrics"]),
        "test_used_for_selection": False,
        "test_evaluations_planned_per_unit_control": 1,
        "uncertainty_filter_semantics": _UNCERTAINTY_DESCRIPTION,
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
            f"Refusing to overwrite augmentation-control output: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        name.removesuffix(".json").removesuffix(".csv"): output / name
        for name in _OUTPUT_NAMES
    }
    paths["output_directory"] = output
    paths["manifest"] = output / "manifest.json"
    paths["benchmark_plan"].write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame([unit.audit_record for unit in units]).to_csv(
        paths["split_units"], index=False
    )

    canonical = pd.read_csv(contract["dataset_path"])
    if sha256_file(contract["dataset_path"]) != saved.dataset_hash:
        raise ValueError("Canonical dataset changed after Phase 11 plan freeze.")
    identity_columns = list(saved.canonical.columns)
    if not canonical.loc[:, identity_columns].equals(
        saved.canonical.reset_index(drop=True)
    ):
        raise ValueError("Canonical identities changed after Phase 11 plan freeze.")
    if "yield" not in canonical:
        raise ValueError("Canonical labels unavailable after Phase 11 plan freeze.")
    canonical["source_row_id"] = canonical["source_row_id"].astype(str)
    by_source = canonical.set_index("source_row_id", drop=False)
    positions = {
        source_id: position
        for position, source_id in enumerate(canonical["source_row_id"])
    }
    global_measured_keys = frozenset(measured_canonical_keys(canonical))

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    training_audits: list[pd.DataFrame] = []
    chemical_audits: list[pd.DataFrame] = []
    metadata_seen: dict[str, dict[str, Any]] = {}
    for unit in units:
        train_positions = np.asarray(
            [positions[source_id] for source_id in unit.train_source_ids],
            dtype=int,
        )
        test_positions = np.asarray(
            [positions[source_id] for source_id in unit.test_source_ids],
            dtype=int,
        )
        train_frame = by_source.loc[
            list(unit.train_source_ids)
        ].reset_index(drop=True)
        y_train = pd.to_numeric(
            train_frame["yield"], errors="raise"
        ).to_numpy(dtype=np.float32)
        X_train = X_all[train_positions]
        y_test = pd.to_numeric(
            by_source.loc[list(unit.test_source_ids), "yield"], errors="raise"
        ).to_numpy(dtype=np.float32)
        nominal_budget = int(
            np.ceil(
                contract["augmentation"]["nominal_added_multiplier"]
                * len(train_frame)
            )
        )
        unit_seed = contract["base_seed"] + (
            int(stable_hash(unit.evaluation_unit)[:12], 16) % 1_000_000
        )
        for control_id in CONTROL_IDS:
            if control_id in SIMPLE_CONTROL_IDS:
                result = build_simple_augmentation_control(
                    control_id,
                    X_train,
                    y_train,
                    unit.train_source_ids,
                    added_count=nominal_budget,
                    random_state=unit_seed,
                    n_yield_strata=contract["augmentation"]["n_yield_strata"],
                    mixup_alpha=contract["augmentation"]["mixup_alpha"],
                    teacher_model_config=contract["augmentation"][
                        "self_training_teacher"
                    ],
                )
                candidate_audit = pd.DataFrame()
                resolved_control = {
                    "control_id": control_id,
                    "nominal_added_budget": nominal_budget,
                    "random_state": unit_seed,
                    "n_yield_strata": contract["augmentation"][
                        "n_yield_strata"
                    ],
                    "mixup_alpha": contract["augmentation"]["mixup_alpha"],
                    "self_training_teacher": contract["augmentation"][
                        "self_training_teacher"
                    ],
                }
                if control_id in {"real_only", "sample_reweighting"}:
                    underfill_count = 0
                    underfill_reason = None
                else:
                    underfill_count = (
                        result.requested_added_count
                        - result.effective_added_count
                    )
                    underfill_reason = (
                        "unique_nonchemical_candidate_pool_exhausted"
                        if underfill_count
                        and control_id
                        in {
                            "nearest_neighbor_pseudo_labeling",
                            "self_training",
                            "feature_mixup",
                        }
                        else None
                    )
                prefilter_pool_hash = None
                accepted_candidate_hash = None
                uncertainty_semantics = None
            else:
                result = build_chemical_augmentation_control(
                    control_id,
                    train_frame=train_frame,
                    X_train=X_train,
                    y_train=y_train,
                    feature_config=feature_config,
                    feature_names=feature_names,
                    feature_metadata=feature_metadata,
                    measured_identity_keys=global_measured_keys,
                    nominal_added_budget=nominal_budget,
                    seed=unit_seed,
                    max_teacher_std=(
                        contract["augmentation"]["raw_teacher_std_threshold"]
                        if control_id
                        == "typed_transfer_with_uncertainty_filtering"
                        else None
                    ),
                    teacher_models=contract["augmentation"][
                        "chemical_teacher_models"
                    ],
                )
                candidate_audit = result.candidate_audit.copy()
                resolved_control = dict(result.resolved_config)
                underfill_count = result.budget_underfill_count
                underfill_reason = result.budget_underfill_reason
                prefilter_pool_hash = result.prefilter_pool_hash
                accepted_candidate_hash = result.accepted_candidate_hash
                uncertainty_semantics = result.uncertainty_semantics
                if not candidate_audit.empty:
                    chemical_audits.append(
                        _contextualize_audit(
                            candidate_audit,
                            unit=unit,
                            control_id=control_id,
                        )
                    )
            _assert_control_result(
                result,
                unit=unit,
                nominal_budget=nominal_budget,
            )
            control_config_hash = stable_hash(resolved_control)
            metadata_seen.setdefault(
                control_id,
                {
                    "control_id": control_id,
                    "control_name": result.control_name,
                    "chemical_identity_applicable": (
                        result.chemical_identity_applicable
                    ),
                    "nonchemical_semantics": result.nonchemical_semantics,
                },
            )
            contextual_training_audit = _contextualize_training_audit(
                result,
                unit=unit,
                control_id=control_id,
            )
            training_audits.append(contextual_training_audit)
            augmented_training_hash = _audited_training_hash(
                contextual_training_audit
            )
            estimator = get_model(
                contract["model"]["name"],
                seed=unit_seed,
                **contract["model"]["params"],
            )
            estimator = _fit_control_estimator(
                estimator, result.X, result.y, result.sample_weight
            )
            predictions = np.asarray(
                predict_model(estimator, X_all[test_positions]), dtype=float
            )
            if not np.isfinite(predictions).all():
                raise ValueError(
                    f"Non-finite predictions for {unit.evaluation_unit}/{control_id}."
                )
            prediction_rows.extend(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "control_id": control_id,
                    "source_row_id": source_id,
                    "prediction": float(value),
                }
                for source_id, value in zip(
                    unit.test_source_ids, predictions, strict=True
                )
            )
            fit_instance_hash = stable_hash(
                {
                    "model_protocol_hash": model_protocol_hash,
                    "model_seed": unit_seed,
                    "exact_split_hash": unit.exact_split_hash,
                    "control_config_hash": control_config_hash,
                    "augmented_training_hash": augmented_training_hash,
                }
            )
            budget_measure = (
                "not_applicable"
                if control_id == "real_only"
                else (
                    "added_weight"
                    if control_id == "sample_reweighting"
                    else "added_rows"
                )
            )
            budget_effective_units = (
                0.0
                if control_id == "real_only"
                else (
                    result.total_sample_weight - len(train_frame)
                    if control_id == "sample_reweighting"
                    else float(result.effective_added_count)
                )
            )
            budget_utilization = (
                0.0
                if control_id == "real_only"
                else (
                    budget_effective_units / nominal_budget
                    if nominal_budget
                    else 1.0
                )
            )
            budget_rows.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "seed": unit.seed,
                    "train_fraction": unit.train_fraction,
                    "control_id": control_id,
                    "control_name": result.control_name,
                    "control_config_hash": control_config_hash,
                    "resolved_control_config": json.dumps(
                        resolved_control,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "policy_selection_budget": 1,
                    "nominal_added_sample_count": nominal_budget,
                    "augmentation_budget_applicable": (
                        control_id != "real_only"
                    ),
                    "budget_measure": budget_measure,
                    "budget_effective_units": budget_effective_units,
                    "effective_added_sample_count": (
                        result.effective_added_count
                    ),
                    "effective_added_weight": (
                        result.total_sample_weight - len(train_frame)
                    ),
                    "final_train_row_count": len(result.y),
                    "real_sample_weight": float(len(train_frame)),
                    "synthetic_or_added_sample_weight": (
                        result.total_sample_weight - len(train_frame)
                    ),
                    "total_sample_weight": result.total_sample_weight,
                    "budget_underfill_count": underfill_count,
                    "budget_underfill_reason": underfill_reason,
                    "budget_utilization": budget_utilization,
                    "budget_backfill_performed": False,
                    "chemical_identity_applicable": (
                        result.chemical_identity_applicable
                    ),
                    "prefilter_pool_hash": prefilter_pool_hash,
                    "accepted_candidate_hash": accepted_candidate_hash,
                    "uncertainty_semantics": uncertainty_semantics,
                    "exact_split_hash": unit.exact_split_hash,
                    "aggregate_assignment_hash": (
                        unit.aggregate_assignment_hash
                    ),
                    "feature_metadata_hash": feature_contract[
                        "feature_metadata_hash"
                    ],
                    "model_protocol_hash": model_protocol_hash,
                    "budget_protocol_hash": budget_protocol_hash,
                    "model_seed": unit_seed,
                    "augmented_training_hash": augmented_training_hash,
                    "fit_instance_hash": fit_instance_hash,
                    "plan_hash": plan["plan_hash"],
                    "used_validation_or_test_sources": False,
                }
            )
            for metric_name in contract["metrics"]:
                value = float(_METRICS[metric_name](y_test, predictions))
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite {metric_name} for "
                        f"{unit.evaluation_unit}/{control_id}."
                    )
                metric_rows.append(
                    {
                        "evaluation_unit": unit.evaluation_unit,
                        "seed": unit.seed,
                        "train_fraction": unit.train_fraction,
                        "control_id": control_id,
                        "metric": metric_name,
                        "value": value,
                        "n_train_real": len(train_frame),
                        "n_train_final": len(result.y),
                        "n_test": len(unit.test_source_ids),
                        "nominal_added_sample_count": nominal_budget,
                        "effective_added_sample_count": (
                            result.effective_added_count
                        ),
                        "total_sample_weight": result.total_sample_weight,
                        "exact_split_hash": unit.exact_split_hash,
                        "aggregate_assignment_hash": (
                            unit.aggregate_assignment_hash
                        ),
                        "feature_metadata_hash": feature_contract[
                            "feature_metadata_hash"
                        ],
                        "model_protocol_hash": model_protocol_hash,
                        "budget_protocol_hash": budget_protocol_hash,
                        "model_seed": unit_seed,
                        "fit_instance_hash": fit_instance_hash,
                        "test_evaluation_count": 1,
                        "test_used_for_selection": False,
                        "plan_hash": plan["plan_hash"],
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    budgets = pd.DataFrame(budget_rows)
    training_audit = pd.concat(training_audits, ignore_index=True, sort=False)
    chemical_audit = (
        pd.concat(chemical_audits, ignore_index=True, sort=False)
        if chemical_audits
        else pd.DataFrame(
            columns=[
                "evaluation_unit",
                "control_id",
                "source_row_id",
                "donor_row_id",
                "rejection_reason",
            ]
        )
    )
    control_metadata = pd.DataFrame(
        [metadata_seen[control_id] for control_id in CONTROL_IDS]
    )
    paired = _paired_comparisons(metrics, budgets)
    summary = _summarize(metrics, paired)
    _assert_paired_gate(metrics, budgets, units, plan)
    _assert_uncertainty_pair(budgets, chemical_audit)
    control_metadata.to_csv(paths["control_metadata"], index=False)
    budgets.to_csv(paths["budget_audit"], index=False)
    training_audit.to_csv(paths["training_audit"], index=False)
    chemical_audit.to_csv(paths["chemical_candidate_audit"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    paired.to_csv(paths["paired_comparisons"], index=False)
    summary.to_csv(paths["summary"], index=False)
    output_hashes = {
        name: sha256_file(output / name) for name in _OUTPUT_NAMES
    }
    manifest = {
        "schema_version": AUGMENTATION_CONTROL_SCHEMA_VERSION,
        "status": "complete",
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "plan_hash": plan["plan_hash"],
        "config_hash": plan["config_hash"],
        "model_protocol_hash": model_protocol_hash,
        "budget_protocol_hash": budget_protocol_hash,
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "resolved_scientific_config": scientific_config,
        "command": " ".join(sys.argv),
        "dependency_versions": _dependency_versions(),
        "evaluation_unit_count": len(units),
        "control_count": len(CONTROL_IDS),
        "metric_row_count": len(metrics),
        "prediction_row_count": len(predictions),
        "budget_row_count": len(budgets),
        "training_audit_row_count": len(training_audit),
        "chemical_candidate_audit_row_count": len(chemical_audit),
        "selection_performed_across_controls": False,
        "policy_selection_budget_per_control": 1,
        "test_used_for_selection": False,
        "test_evaluations_per_unit_control": 1,
        "uncertainty_calibrated": False,
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_augmentation_control_benchmark(output)
    return paths


def validate_augmentation_control_benchmark(
    directory: str | Path,
) -> dict[str, Any]:
    """Rehash, replay splits, and semantically validate Phase 11 artifacts."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed_manifest_hash = manifest.pop("manifest_hash", None)
    if claimed_manifest_hash != stable_hash(manifest):
        raise ValueError("Augmentation-control manifest hash mismatch.")
    manifest["manifest_hash"] = claimed_manifest_hash
    if (
        manifest.get("schema_version") != AUGMENTATION_CONTROL_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or int(manifest.get("control_count", -1)) != len(CONTROL_IDS)
        or manifest.get("selection_performed_across_controls") is not False
        or manifest.get("test_used_for_selection") is not False
        or manifest.get("uncertainty_calibrated") is not False
        or set(manifest.get("output_hashes", {})) != set(_OUTPUT_NAMES)
    ):
        raise ValueError("Augmentation-control manifest contract mismatch.")
    for name, digest in manifest["output_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(
                f"Augmentation-control output hash mismatch: {name}."
            )
    plan = json.loads((root / "benchmark_plan.json").read_text())
    claimed_plan_hash = plan.pop("plan_hash", None)
    if claimed_plan_hash != stable_hash(plan):
        raise ValueError("Augmentation-control plan hash mismatch.")
    plan["plan_hash"] = claimed_plan_hash
    if (
        plan.get("status") != "frozen_before_label_loading"
        or tuple(item["control_id"] for item in plan.get("controls", []))
        != CONTROL_IDS
        or plan.get("controls")
        != json.loads(json.dumps(_control_plan_records()))
        or plan.get("model_protocol_hash")
        != stable_hash(plan.get("model_protocol"))
        or plan.get("budget_protocol_hash")
        != stable_hash(plan.get("budget_protocol"))
        or plan.get("feature_metadata_hash")
        != plan.get("feature_contract", {}).get("feature_metadata_hash")
        or plan.get("config_hash")
        != stable_hash(plan.get("resolved_scientific_config"))
        or plan.get("test_used_for_selection") is not False
        or plan.get("uncertainty_filter_semantics")
        != _UNCERTAINTY_DESCRIPTION
    ):
        raise ValueError("Augmentation-control frozen plan mismatch.")
    scientific = plan["resolved_scientific_config"]
    replay_saved = load_saved_canonical_split_identities(
        scientific["dataset_path"],
        scientific["canonical_split_directory"],
        requested_seeds=scientific["random_seeds"],
        requested_fractions=scientific["random_fractions"],
    )
    replay_units = tuple(
        build_saved_random_split_unit(
            replay_saved, seed=seed, train_fraction=fraction
        )
        for seed in scientific["random_seeds"]
        for fraction in scientific["random_fractions"]
    )
    if stable_hash([unit.audit_record for unit in replay_units]) != stable_hash(
        plan["split_units"]
    ):
        raise ValueError("Augmentation-control split replay mismatch.")
    if (
        manifest["dataset_hash"] != replay_saved.dataset_hash
        or manifest["canonical_split_dependency_hash"]
        != replay_saved.aggregate_split_hash
        or manifest["plan_hash"] != claimed_plan_hash
        or manifest["model_protocol_hash"] != plan["model_protocol_hash"]
        or manifest["budget_protocol_hash"] != plan["budget_protocol_hash"]
        or manifest["feature_metadata_hash"] != plan["feature_metadata_hash"]
        or manifest["config_hash"] != plan["config_hash"]
        or manifest["resolved_scientific_config"]
        != plan["resolved_scientific_config"]
        or int(manifest["evaluation_unit_count"]) != len(replay_units)
    ):
        raise ValueError("Augmentation-control provenance linkage mismatch.")
    _assert_split_unit_table(
        root / "split_units.csv", [unit.audit_record for unit in replay_units]
    )
    metrics = pd.read_csv(root / "metrics.csv")
    budgets = pd.read_csv(root / "budget_audit.csv")
    if (
        len(metrics) != int(manifest["metric_row_count"])
        or len(budgets) != int(manifest["budget_row_count"])
        or len(metrics)
        != len(replay_units) * len(CONTROL_IDS) * len(plan["metrics"])
        or len(budgets) != len(replay_units) * len(CONTROL_IDS)
        or set(metrics["plan_hash"]) != {claimed_plan_hash}
        or set(budgets["plan_hash"]) != {claimed_plan_hash}
    ):
        raise ValueError("Augmentation-control table coverage mismatch.")
    _assert_paired_gate(metrics, budgets, replay_units, plan)
    chemical_audit = pd.read_csv(root / "chemical_candidate_audit.csv")
    training_audit = pd.read_csv(root / "training_audit.csv")
    if (
        len(training_audit)
        != int(manifest.get("training_audit_row_count", -1))
        or len(chemical_audit)
        != int(manifest.get("chemical_candidate_audit_row_count", -1))
    ):
        raise ValueError("Phase 11 audit row-count manifest mismatch.")
    audited_training_hashes = _assert_training_audit(
        training_audit,
        budgets,
        replay_units,
        scientific,
        replay_saved.dataset_hash,
    )
    _assert_audit_parents(chemical_audit, replay_units)
    _assert_chemical_candidate_audit(
        chemical_audit,
        training_audit,
        budgets,
        replay_units,
        scientific,
    )
    _assert_uncertainty_pair(budgets, chemical_audit)
    _assert_metric_budget_linkage(metrics, budgets, replay_units, plan)
    for row in budgets.to_dict(orient="records"):
        key = (row["evaluation_unit"], row["control_id"])
        if audited_training_hashes.get(key) != row["augmented_training_hash"]:
            raise ValueError("Phase 11 audited training hash mismatch.")
    predictions = pd.read_csv(root / "predictions.csv")
    _assert_prediction_replay(
        predictions,
        metrics,
        replay_units,
        scientific["dataset_path"],
        plan["metrics"],
        manifest,
    )
    expected_paired = _paired_comparisons(metrics, budgets)
    observed_paired = pd.read_csv(root / "paired_comparisons.csv")
    pd.testing.assert_frame_equal(
        observed_paired,
        expected_paired,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    expected_summary = _summarize(metrics, expected_paired)
    observed_summary = pd.read_csv(root / "summary.csv")
    pd.testing.assert_frame_equal(
        observed_summary,
        expected_summary,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    metadata = pd.read_csv(root / "control_metadata.csv")
    _assert_control_metadata(metadata, plan["controls"])
    return manifest


def _resolve_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset")
    splits = config.get("splits")
    augmentation = config.get("augmentation")
    features = config.get("features")
    model = config.get("model")
    if not isinstance(dataset, Mapping) or not dataset.get("path"):
        raise ValueError("Phase 11 requires dataset.path.")
    if (
        not isinstance(splits, Mapping)
        or not splits.get("canonical_directory")
        or not isinstance(splits.get("random_seeds"), list)
        or not splits["random_seeds"]
        or not isinstance(splits.get("random_fractions"), list)
        or not splits["random_fractions"]
    ):
        raise ValueError("Phase 11 requires saved random seeds and fractions.")
    if tuple(config.get("controls", ())) != CONTROL_IDS:
        raise ValueError("Phase 11 requires the exact 13 controls in frozen order.")
    if not isinstance(augmentation, Mapping):
        raise ValueError("Phase 11 requires augmentation configuration.")
    expected_augmentation = {
        "nominal_added_multiplier",
        "n_yield_strata",
        "mixup_alpha",
        "self_training_teacher",
        "chemical_teacher_models",
        "raw_teacher_std_threshold",
    }
    if set(augmentation) != expected_augmentation:
        raise ValueError("Phase 11 augmentation configuration schema mismatch.")
    multiplier = float(augmentation["nominal_added_multiplier"])
    strata = augmentation["n_yield_strata"]
    alpha = float(augmentation["mixup_alpha"])
    threshold = float(augmentation["raw_teacher_std_threshold"])
    teachers = augmentation["chemical_teacher_models"]
    if (
        not np.isfinite(multiplier)
        or multiplier < 0
        or not isinstance(strata, int)
        or isinstance(strata, bool)
        or strata <= 0
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
        or not np.isfinite(threshold)
        or threshold < 0
        or not isinstance(teachers, list)
        or not teachers
        or any(not isinstance(item, str) or not item for item in teachers)
    ):
        raise ValueError("Phase 11 augmentation values are invalid.")
    if not isinstance(features, Mapping):
        raise ValueError("Phase 11 requires canonical features.")
    resolved_features = resolve_corrected_feature_config(
        features, required_kind="bh_role_separated"
    )
    if not isinstance(model, Mapping) or set(model) != {"name", "params"}:
        raise ValueError("Phase 11 model requires exactly name and params.")
    if not isinstance(model["params"], Mapping):
        raise ValueError("Phase 11 model.params must be a mapping.")
    metrics = config.get("metrics")
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(item not in _METRICS for item in metrics)
        or len(set(metrics)) != len(metrics)
    ):
        raise ValueError("Phase 11 metrics are invalid.")
    seeds = tuple(int(item) for item in splits["random_seeds"])
    fractions = tuple(float(item) for item in splits["random_fractions"])
    if len(set(seeds)) != len(seeds) or len(set(fractions)) != len(fractions):
        raise ValueError("Phase 11 seeds and fractions must be unique.")
    return {
        "dataset_path": Path(dataset["path"]),
        "canonical_split_directory": Path(splits["canonical_directory"]),
        "random_seeds": seeds,
        "random_fractions": fractions,
        "augmentation": {
            "nominal_added_multiplier": multiplier,
            "n_yield_strata": strata,
            "mixup_alpha": alpha,
            "self_training_teacher": json.loads(
                json.dumps(augmentation["self_training_teacher"])
            ),
            "chemical_teacher_models": list(teachers),
            "raw_teacher_std_threshold": threshold,
        },
        "features": resolved_features,
        "model": {
            "name": str(model["name"]),
            "params": dict(model["params"]),
        },
        "metrics": tuple(metrics),
        "base_seed": int(config.get("base_seed", 0)),
        "output_directory": Path(
            config.get("output", {}).get(
                "directory", "results/corrected_augmentation_controls_phase11"
            )
        ),
    }


def _control_plan_records() -> list[dict[str, Any]]:
    simple = [
        {
            "control_number": number,
            "control_id": control_id,
            "control_name": name,
            "family": (
                "nonchemical_feature_control"
                if control_id
                in {
                    "nearest_neighbor_pseudo_labeling",
                    "self_training",
                    "feature_mixup",
                }
                else "resampling_or_weight_control"
            ),
        }
        for number, (control_id, name) in enumerate(
            SIMPLE_CONTROL_SPECS, start=1
        )
    ]
    return simple + [
        {**asdict(spec), "family": "canonical_chemical_transfer"}
        for spec in CHEMICAL_CONTROL_SPECS
    ]


def _assert_control_result(
    result: Any,
    *,
    unit: EvaluationSplitUnit,
    nominal_budget: int,
) -> None:
    X = np.asarray(result.X)
    y = np.asarray(result.y).reshape(-1)
    if (
        X.ndim != 2
        or len(X) != len(y)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
        or result.requested_added_count != nominal_budget
        or result.effective_added_count != len(y) - len(unit.train_source_ids)
        or result.effective_added_count < 0
        or result.effective_added_count > nominal_budget
    ):
        raise ValueError("Phase 11 control output violates its frozen budget.")
    weights = result.sample_weight
    if weights is not None:
        weights = np.asarray(weights).reshape(-1)
        if (
            len(weights) != len(y)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("Phase 11 control weights are invalid.")
    expected_total = (
        float(np.asarray(weights, dtype=np.float64).sum())
        if weights is not None
        else float(len(y))
    )
    if not np.isclose(
        result.total_sample_weight, expected_total, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Phase 11 total sample weight is inconsistent.")
    if (
        result.control_id == "sample_reweighting"
        and (
            result.effective_added_count != 0
            or not np.isclose(
                result.total_sample_weight,
                len(unit.train_source_ids) + nominal_budget,
                rtol=0.0,
                atol=1e-5,
            )
        )
    ):
        raise ValueError("Sample reweighting does not match the nominal budget.")
    if hasattr(result, "budget_underfill_count") and (
        int(result.budget_underfill_count)
        != nominal_budget - result.effective_added_count
        or bool(result.budget_underfill_reason)
        != bool(result.budget_underfill_count)
    ):
        raise ValueError("Chemical control underfill metadata is inconsistent.")


def _assert_paired_gate(
    metrics: pd.DataFrame,
    budgets: pd.DataFrame,
    units: tuple[EvaluationSplitUnit, ...],
    plan: Mapping[str, Any],
) -> None:
    common = (
        "exact_split_hash",
        "aggregate_assignment_hash",
        "feature_metadata_hash",
        "model_protocol_hash",
        "budget_protocol_hash",
        "model_seed",
        "plan_hash",
    )
    unit_by_name = {unit.evaluation_unit: unit for unit in units}
    expected_provenance = {
        "model_protocol_hash": plan["model_protocol_hash"],
        "budget_protocol_hash": plan["budget_protocol_hash"],
        "feature_metadata_hash": plan["feature_metadata_hash"],
        "plan_hash": plan["plan_hash"],
    }
    for unit_name, rows in metrics.groupby("evaluation_unit", sort=False):
        unit = unit_by_name.get(unit_name)
        if unit is None:
            raise ValueError("Phase 11 metric references an unknown split unit.")
        if (
            set(rows["control_id"]) != set(CONTROL_IDS)
            or any(
                not rows[column].notna().all()
                or rows[column].nunique(dropna=False) != 1
                for column in common
            )
            or not rows["test_evaluation_count"].eq(1).all()
            or rows["test_used_for_selection"].map(_strict_bool).any()
            or set(rows["metric"]) != set(plan["metrics"])
        ):
            raise ValueError(f"Unpaired Phase 11 metrics for {unit_name}.")
        counts = rows.groupby(["control_id", "metric"]).size()
        if not counts.eq(1).all():
            raise ValueError(f"Repeated Phase 11 metric for {unit_name}.")
        if (
            set(rows["exact_split_hash"]) != {unit.exact_split_hash}
            or set(rows["aggregate_assignment_hash"])
            != {unit.aggregate_assignment_hash}
        ):
            raise ValueError("Phase 11 metric split linkage mismatch.")
        if any(
            set(rows[column]) != {expected}
            for column, expected in expected_provenance.items()
        ):
            raise ValueError("Phase 11 metric frozen provenance mismatch.")
    for unit_name, rows in budgets.groupby("evaluation_unit", sort=False):
        unit = unit_by_name.get(unit_name)
        expected_model_seed = int(
            plan["resolved_scientific_config"]["base_seed"]
        ) + (int(stable_hash(unit_name)[:12], 16) % 1_000_000)
        expected_nominal_budget = int(
            np.ceil(
                float(
                    plan["resolved_scientific_config"]["augmentation"][
                        "nominal_added_multiplier"
                    ]
                )
                * len(unit.train_source_ids)
            )
        )
        if (
            unit is None
            or len(rows) != len(CONTROL_IDS)
            or set(rows["control_id"]) != set(CONTROL_IDS)
            or rows["control_id"].duplicated().any()
            or any(
                not rows[column].notna().all()
                or rows[column].nunique(dropna=False) != 1
                for column in common
            )
            or not rows["policy_selection_budget"].eq(1).all()
            or rows["budget_backfill_performed"].map(_strict_bool).any()
            or rows["used_validation_or_test_sources"].map(_strict_bool).any()
            or set(rows["exact_split_hash"]) != {unit.exact_split_hash}
            or set(rows["aggregate_assignment_hash"])
            != {unit.aggregate_assignment_hash}
            or set(rows["model_seed"]) != {expected_model_seed}
            or set(rows["nominal_added_sample_count"])
            != {expected_nominal_budget}
        ):
            raise ValueError(f"Unpaired Phase 11 budgets for {unit_name}.")
        if any(
            set(rows[column]) != {expected}
            for column, expected in expected_provenance.items()
        ):
            raise ValueError("Phase 11 budget frozen provenance mismatch.")
        for row in rows.to_dict(orient="records"):
            nominal = int(row["nominal_added_sample_count"])
            effective = int(row["effective_added_sample_count"])
            budget_effective = float(row["budget_effective_units"])
            applicable = _strict_bool(
                row["augmentation_budget_applicable"]
            )
            expected_measure = (
                "not_applicable"
                if row["control_id"] == "real_only"
                else (
                    "added_weight"
                    if row["control_id"] == "sample_reweighting"
                    else "added_rows"
                )
            )
            expected_uncertainty = (
                _RAW_UNCERTAINTY_SEMANTICS
                if row["control_id"]
                in {
                    "typed_transfer_without_uncertainty_filtering",
                    "typed_transfer_with_uncertainty_filtering",
                }
                else None
            )
            n_real = len(unit.train_source_ids)
            total_weight = float(row["total_sample_weight"])
            effective_weight = float(row["effective_added_weight"])
            expected_effective_units = (
                0.0
                if expected_measure == "not_applicable"
                else (
                    effective_weight
                    if expected_measure == "added_weight"
                    else float(effective)
                )
            )
            expected_underfill = (
                max(0, nominal - int(round(budget_effective)))
                if applicable
                else 0
            )
            reason_present = _nonempty_value(
                row["budget_underfill_reason"]
            )
            if (
                effective < 0
                or effective > nominal
                or row["budget_measure"] != expected_measure
                or applicable != (expected_measure != "not_applicable")
                or (
                    expected_uncertainty is None
                    and _nonempty_value(row["uncertainty_semantics"])
                )
                or (
                    expected_uncertainty is not None
                    and row["uncertainty_semantics"]
                    != expected_uncertainty
                )
                or not np.isfinite(budget_effective)
                or budget_effective < 0
                or budget_effective > nominal
                or not np.isclose(
                    budget_effective,
                    expected_effective_units,
                    rtol=0.0,
                    atol=1e-9,
                )
                or int(row["final_train_row_count"]) != n_real + effective
                or not np.isclose(
                    float(row["real_sample_weight"]),
                    n_real,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.isclose(
                    float(row["synthetic_or_added_sample_weight"]),
                    effective_weight,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.isclose(
                    total_weight,
                    n_real + effective_weight,
                    rtol=0.0,
                    atol=1e-9,
                )
                or (
                    expected_measure == "added_rows"
                    and (
                        not np.isclose(
                            effective_weight,
                            effective,
                            rtol=0.0,
                            atol=1e-9,
                        )
                        or not np.isclose(
                            total_weight,
                            n_real + effective,
                            rtol=0.0,
                            atol=1e-9,
                        )
                    )
                )
                or (
                    expected_measure == "added_weight"
                    and (
                        effective != 0
                        or not np.isclose(
                            effective_weight,
                            nominal,
                            rtol=0.0,
                            atol=1e-6,
                        )
                    )
                )
                or (
                    expected_measure == "not_applicable"
                    and (
                        effective != 0
                        or not np.isclose(
                            effective_weight, 0.0, rtol=0.0, atol=1e-12
                        )
                    )
                )
                or int(row["budget_underfill_count"]) != expected_underfill
                or reason_present != (expected_underfill > 0)
                or not np.isclose(
                    float(row["budget_utilization"]),
                    (
                        budget_effective / nominal
                        if applicable and nominal
                        else (1.0 if applicable else 0.0)
                    ),
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError("Phase 11 budget arithmetic mismatch.")


def _assert_prediction_replay(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    units: tuple[EvaluationSplitUnit, ...],
    dataset_path: str | Path,
    metric_names: list[str],
    manifest: Mapping[str, Any],
) -> None:
    expected_columns = (
        "evaluation_unit",
        "control_id",
        "source_row_id",
        "prediction",
    )
    if tuple(predictions.columns) != expected_columns:
        raise ValueError("Phase 11 prediction schema mismatch.")
    predictions["source_row_id"] = predictions["source_row_id"].astype(str)
    expected_count = sum(len(unit.test_source_ids) for unit in units) * len(
        CONTROL_IDS
    )
    if (
        len(predictions) != expected_count
        or int(manifest["prediction_row_count"]) != expected_count
        or not np.isfinite(predictions["prediction"].to_numpy(float)).all()
    ):
        raise ValueError("Phase 11 prediction coverage mismatch.")
    outcomes = pd.read_csv(
        dataset_path,
        usecols=["source_row_id", "yield"],
        dtype={"source_row_id": str},
    )
    outcome_by_source = dict(
        zip(outcomes["source_row_id"], outcomes["yield"], strict=True)
    )
    for unit in units:
        for control_id in CONTROL_IDS:
            rows = predictions.loc[
                predictions["evaluation_unit"].eq(unit.evaluation_unit)
                & predictions["control_id"].eq(control_id)
            ].sort_values("source_row_id", kind="mergesort")
            if (
                len(rows) != len(unit.test_source_ids)
                or rows["source_row_id"].duplicated().any()
                or tuple(rows["source_row_id"]) != unit.test_source_ids
            ):
                raise ValueError("Phase 11 prediction membership mismatch.")
            y_true = np.asarray(
                [outcome_by_source[source] for source in rows["source_row_id"]],
                dtype=np.float32,
            )
            y_pred = rows["prediction"].to_numpy(float)
            for metric_name in metric_names:
                expected = float(_METRICS[metric_name](y_true, y_pred))
                observed = metrics.loc[
                    metrics["evaluation_unit"].eq(unit.evaluation_unit)
                    & metrics["control_id"].eq(control_id)
                    & metrics["metric"].eq(metric_name),
                    "value",
                ]
                if len(observed) != 1 or not np.isclose(
                    float(observed.iloc[0]),
                    expected,
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise ValueError("Phase 11 prediction metric replay mismatch.")


def _paired_comparisons(
    metrics: pd.DataFrame, budgets: pd.DataFrame
) -> pd.DataFrame:
    values = metrics[
        [
            "evaluation_unit",
            "seed",
            "train_fraction",
            "control_id",
            "metric",
            "value",
        ]
    ]
    budget_values = budgets[
        [
            "evaluation_unit",
            "control_id",
            "effective_added_sample_count",
            "total_sample_weight",
        ]
    ]
    rows: list[dict[str, Any]] = []
    for comparator_id in COMPARATOR_IDS:
        comparator = values.loc[
            values["control_id"].eq(comparator_id)
        ].rename(
            columns={
                "control_id": "comparator_id",
                "value": "comparator_value",
            }
        )
        merged = values.merge(
            comparator[
                [
                    "evaluation_unit",
                    "metric",
                    "comparator_id",
                    "comparator_value",
                ]
            ],
            on=["evaluation_unit", "metric"],
            how="inner",
            validate="many_to_one",
        )
        merged = merged.merge(
            budget_values.rename(
                columns={
                    "effective_added_sample_count": "control_added_count",
                    "total_sample_weight": "control_total_weight",
                }
            ),
            on=["evaluation_unit", "control_id"],
            validate="many_to_one",
        ).merge(
            budget_values.loc[
                budget_values["control_id"].eq(comparator_id)
            ].rename(
                columns={
                    "control_id": "comparator_id",
                    "effective_added_sample_count": "comparator_added_count",
                    "total_sample_weight": "comparator_total_weight",
                }
            ),
            on=["evaluation_unit", "comparator_id"],
            validate="many_to_one",
        )
        for row in merged.to_dict(orient="records"):
            delta = float(row["value"]) - float(row["comparator_value"])
            improvement = (
                -delta if _LOWER_IS_BETTER[row["metric"]] else delta
            )
            rows.append(
                {
                    "evaluation_unit": row["evaluation_unit"],
                    "seed": row["seed"],
                    "train_fraction": row["train_fraction"],
                    "control_id": row["control_id"],
                    "comparator_id": row["comparator_id"],
                    "metric": row["metric"],
                    "control_value": row["value"],
                    "comparator_value": row["comparator_value"],
                    "delta_control_minus_comparator": delta,
                    "improvement_over_comparator": improvement,
                    "control_added_count": row["control_added_count"],
                    "comparator_added_count": row[
                        "comparator_added_count"
                    ],
                    "control_total_weight": row["control_total_weight"],
                    "comparator_total_weight": row[
                        "comparator_total_weight"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _summarize(
    metrics: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    absolute = (
        metrics.groupby(
            ["train_fraction", "control_id", "metric"], dropna=False
        )["value"]
        .agg(mean="mean", median="median", std="std", count="count")
        .reset_index()
    )
    improvements = (
        paired.groupby(
            [
                "train_fraction",
                "control_id",
                "comparator_id",
                "metric",
            ],
            dropna=False,
        )["improvement_over_comparator"]
        .agg(
            mean_improvement="mean",
            median_improvement="median",
            improvement_std="std",
            win_count=lambda values: int((values > 0).sum()),
            pair_count="count",
        )
        .reset_index()
    )
    merged = improvements.merge(
        absolute,
        on=["train_fraction", "control_id", "metric"],
        how="left",
        validate="many_to_one",
    )
    return merged[
        [
            "train_fraction",
            "control_id",
            "comparator_id",
            "metric",
            "mean",
            "median",
            "std",
            "count",
            "mean_improvement",
            "median_improvement",
            "improvement_std",
            "win_count",
            "pair_count",
        ]
    ]


def _assert_uncertainty_pair(
    budgets: pd.DataFrame, chemical_audit: pd.DataFrame
) -> None:
    unfiltered_id = "typed_transfer_without_uncertainty_filtering"
    filtered_id = "typed_transfer_with_uncertainty_filtering"
    for unit_name, rows in budgets.loc[
        budgets["control_id"].isin([unfiltered_id, filtered_id])
    ].groupby("evaluation_unit", sort=False):
        by_control = rows.set_index("control_id")
        if (
            len(by_control) != 2
            or str(by_control.loc[unfiltered_id, "prefilter_pool_hash"])
            != str(by_control.loc[filtered_id, "prefilter_pool_hash"])
        ):
            raise ValueError(
                f"Phase 11 uncertainty controls lack a common pool for {unit_name}."
            )
        if chemical_audit.empty:
            continue
        unit_audit = chemical_audit.loc[
            chemical_audit["evaluation_unit"].eq(unit_name)
        ]
        unfiltered = set(
            unit_audit.loc[
                unit_audit["control_id"].eq(unfiltered_id)
                & unit_audit["accepted"].map(_strict_bool),
                "canonical_reaction_hash",
            ].dropna()
        )
        filtered = set(
            unit_audit.loc[
                unit_audit["control_id"].eq(filtered_id)
                & unit_audit["accepted"].map(_strict_bool),
                "canonical_reaction_hash",
            ].dropna()
        )
        if not filtered.issubset(unfiltered):
            raise ValueError("Phase 11 uncertainty acceptance is not a subset.")


def _assert_audit_parents(
    audit: pd.DataFrame, units: tuple[EvaluationSplitUnit, ...]
) -> None:
    if audit.empty:
        return
    unit_by_name = {unit.evaluation_unit: unit for unit in units}
    if "evaluation_unit" not in audit or "control_id" not in audit:
        raise ValueError("Phase 11 audit lacks run context.")
    for unit_name, rows in audit.groupby("evaluation_unit", sort=False):
        unit = unit_by_name.get(unit_name)
        if unit is None:
            raise ValueError("Phase 11 audit references an unknown unit.")
        allowed = set(unit.train_source_ids)
        for column in (
            "source_row_id",
            "donor_row_id",
            "label_source_row_id",
        ):
            if column in rows:
                parents = set(rows[column].dropna().astype(str))
                if not parents <= allowed:
                    raise ValueError(
                        f"Phase 11 audit contains non-training {column}."
                    )


def _contextualize_training_audit(
    result: Any,
    *,
    unit: EvaluationSplitUnit,
    control_id: str,
) -> pd.DataFrame:
    audit = _contextualize_audit(
        result.row_audit, unit=unit, control_id=control_id
    )
    audit["training_feature_hash"] = None
    audit["training_label"] = np.nan
    audit["training_sample_weight"] = np.nan
    weights = (
        np.asarray(result.sample_weight, dtype=float)
        if result.sample_weight is not None
        else np.ones(len(result.y), dtype=float)
    )
    output_rows = audit.loc[audit["output_row_index"].notna()]
    indices = pd.to_numeric(
        output_rows["output_row_index"], errors="raise"
    ).to_numpy(float)
    if (
        not np.equal(indices, np.floor(indices)).all()
        or (indices < 0).any()
        or (indices >= len(result.y)).any()
        or len(indices) != len(set(indices.astype(int)))
    ):
        raise ValueError("Phase 11 control audit has invalid output indices.")
    for row_index, output_index in zip(
        output_rows.index, indices.astype(int), strict=True
    ):
        audit.loc[row_index, "training_feature_hash"] = _row_feature_hash(
            result.X[output_index]
        )
        audit.loc[row_index, "training_label"] = float(
            result.y[output_index]
        )
        audit.loc[row_index, "training_sample_weight"] = float(
            weights[output_index]
        )
    return audit


def _assert_training_audit(
    audit: pd.DataFrame,
    budgets: pd.DataFrame,
    units: tuple[EvaluationSplitUnit, ...],
    scientific: Mapping[str, Any],
    dataset_hash: str,
) -> dict[tuple[str, str], str]:
    if audit.empty:
        raise ValueError("Phase 11 training audit cannot be empty.")
    required = {
        "evaluation_unit",
        "control_id",
        "candidate_id",
        "output_row_index",
        "is_real",
        "is_added",
        "accepted",
        "source_row_id",
        "training_feature_hash",
        "training_label",
        "training_sample_weight",
    }
    if not required <= set(audit):
        raise ValueError("Phase 11 training audit schema is incomplete.")
    expected_groups = {
        (unit.evaluation_unit, control_id)
        for unit in units
        for control_id in CONTROL_IDS
    }
    observed_groups = set(
        audit[["evaluation_unit", "control_id"]].itertuples(
            index=False, name=None
        )
    )
    if observed_groups != expected_groups:
        raise ValueError("Phase 11 training audit control coverage mismatch.")
    budget_by_key = budgets.set_index(
        ["evaluation_unit", "control_id"], drop=False
    )
    unit_by_name = {unit.evaluation_unit: unit for unit in units}
    canonical = pd.read_csv(scientific["dataset_path"])
    if sha256_file(scientific["dataset_path"]) != dataset_hash:
        raise ValueError("Phase 11 canonical dataset changed during validation.")
    canonical["source_row_id"] = canonical["source_row_id"].astype(str)
    identity_only = canonical.copy()
    identity_only["yield"] = 0.0
    X_all, _, _, _ = build_feature_matrix_with_metadata(
        identity_only, scientific["features"]
    )
    X_all = np.asarray(X_all, dtype=np.float32)
    positions = {
        source_id: position
        for position, source_id in enumerate(canonical["source_row_id"])
    }
    outcomes = dict(
        zip(
            canonical["source_row_id"],
            pd.to_numeric(canonical["yield"], errors="raise").astype(
                np.float32
            ),
            strict=True,
        )
    )
    hashes: dict[tuple[str, str], str] = {}
    for key, rows in audit.groupby(
        ["evaluation_unit", "control_id"], sort=False
    ):
        unit = unit_by_name[key[0]]
        budget = budget_by_key.loc[key]
        is_real = rows["is_real"].map(_strict_bool)
        accepted = rows["accepted"].map(_strict_bool)
        measured = rows.loc[is_real]
        if (
            len(measured) != len(unit.train_source_ids)
            or set(measured["source_row_id"].astype(str))
            != set(unit.train_source_ids)
            or not accepted.loc[measured.index].all()
            or measured["is_added"].map(_strict_bool).any()
        ):
            raise ValueError("Phase 11 measured training audit mismatch.")
        for row in measured.to_dict(orient="records"):
            source_id = str(row["source_row_id"])
            if (
                row["training_feature_hash"]
                != _row_feature_hash(X_all[positions[source_id]])
                or not np.isclose(
                    float(row["training_label"]),
                    float(outcomes[source_id]),
                    rtol=0.0,
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    "Phase 11 measured training audit is not canonical-grounded."
                )
        output = rows.loc[rows["output_row_index"].notna()].copy()
        output_indices = pd.to_numeric(
            output["output_row_index"], errors="raise"
        ).to_numpy(float)
        expected_count = int(budget["final_train_row_count"])
        if (
            len(output) != expected_count
            or not np.equal(output_indices, np.floor(output_indices)).all()
            or set(output_indices.astype(int)) != set(range(expected_count))
            or not accepted.loc[output.index].all()
            or rows.loc[
                rows["output_row_index"].isna(), "is_real"
            ].map(_strict_bool).any()
        ):
            raise ValueError("Phase 11 output-row audit reconstruction mismatch.")
        added_output = output.loc[~output["is_real"].map(_strict_bool)]
        if len(added_output) != int(budget["effective_added_sample_count"]):
            raise ValueError("Phase 11 added-row audit count mismatch.")
        if (
            not added_output.empty
            and not added_output["is_added"].map(_strict_bool).all()
        ):
            raise ValueError("Phase 11 output synthetic rows are not marked added.")
        feature_hashes = output["training_feature_hash"]
        labels = pd.to_numeric(output["training_label"], errors="raise")
        weights = pd.to_numeric(
            output["training_sample_weight"], errors="raise"
        )
        if (
            feature_hashes.isna().any()
            or not feature_hashes.astype(str).map(_valid_hash).all()
            or not np.isfinite(labels.to_numpy(float)).all()
            or not np.isfinite(weights.to_numpy(float)).all()
            or (weights < 0).any()
            or not np.isclose(
                float(weights.sum()),
                float(budget["total_sample_weight"]),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError("Phase 11 audited output values are invalid.")
        if key[1] != "sample_reweighting" and not np.isclose(
            weights.to_numpy(float), 1.0, rtol=0.0, atol=1e-12
        ).all():
            raise ValueError("Phase 11 non-reweighting control has altered weights.")
        hashes[key] = _audited_training_hash(rows)
        if key[1] in SIMPLE_CONTROL_IDS:
            train_positions = np.asarray(
                [positions[source_id] for source_id in unit.train_source_ids],
                dtype=int,
            )
            expected_result = build_simple_augmentation_control(
                key[1],
                X_all[train_positions],
                np.asarray(
                    [outcomes[source_id] for source_id in unit.train_source_ids],
                    dtype=np.float32,
                ),
                unit.train_source_ids,
                added_count=int(budget["nominal_added_sample_count"]),
                random_state=int(budget["model_seed"]),
                n_yield_strata=int(
                    scientific["augmentation"]["n_yield_strata"]
                ),
                mixup_alpha=float(
                    scientific["augmentation"]["mixup_alpha"]
                ),
                teacher_model_config=scientific["augmentation"][
                    "self_training_teacher"
                ],
            )
            expected_audit = _contextualize_training_audit(
                expected_result, unit=unit, control_id=key[1]
            )
            if _audited_training_hash(expected_audit) != hashes[key]:
                raise ValueError("Phase 11 simple control replay mismatch.")
        elif not added_output.empty:
            if (
                "feature_hash" not in added_output
                or "synthetic_label" not in added_output
                or not added_output["training_feature_hash"]
                .eq(added_output["feature_hash"])
                .all()
                or not np.isclose(
                    pd.to_numeric(
                        added_output["training_label"], errors="raise"
                    ).to_numpy(float),
                    pd.to_numeric(
                        added_output["synthetic_label"], errors="raise"
                    ).to_numpy(float),
                    rtol=0.0,
                    atol=1e-7,
                ).all()
            ):
                raise ValueError(
                    "Phase 11 chemical training outputs mismatch candidate audit."
                )
    return hashes


def _audited_training_hash(audit: pd.DataFrame) -> str:
    output = audit.loc[audit["output_row_index"].notna()].copy()
    output["output_row_index"] = pd.to_numeric(
        output["output_row_index"], errors="raise"
    ).astype(int)
    output = output.sort_values("output_row_index", kind="mergesort")
    records = [
        {
            "output_row_index": int(row["output_row_index"]),
            "training_feature_hash": str(row["training_feature_hash"]),
            "training_label": format(float(row["training_label"]), ".9g"),
            "training_sample_weight": format(
                float(row["training_sample_weight"]), ".9g"
            ),
        }
        for row in output.to_dict(orient="records")
    ]
    return stable_hash(records)


def _assert_chemical_candidate_audit(
    audit: pd.DataFrame,
    training_audit: pd.DataFrame,
    budgets: pd.DataFrame,
    units: tuple[EvaluationSplitUnit, ...],
    scientific: Mapping[str, Any],
) -> None:
    if audit.empty:
        raise ValueError("Phase 11 chemical candidate audit cannot be empty.")
    required = {
        "evaluation_unit",
        "control_id",
        "candidate_id",
        "source_row_id",
        "donor_row_id",
        "canonical_reaction_key",
        "canonical_reaction_hash",
        "feature_hash",
        "source_identical",
        "already_measured",
        "duplicate_synthetic",
        "feature_duplicate_synthetic",
        "chemical_parse_valid",
        "rejection_reason",
        "role_change_valid",
        "accepted",
        "kept",
        "requested_roles",
        "actual_changed_roles",
        "unchanged_requested_roles",
        "unexpected_changed_roles",
        "change_mask",
        "role_change_requirement",
        "synthetic_label",
        "reaction_smiles",
        *{
            f"canonical_{role}_smiles"
            for role in (
                "reactant_1",
                "reactant_2",
                "catalyst",
                "ligand",
                "base",
                "solvent_or_additive",
                "product",
            )
        },
    }
    if not required <= set(audit):
        raise ValueError("Phase 11 chemical candidate audit schema is incomplete.")
    expected_groups = {
        (unit.evaluation_unit, control_id)
        for unit in units
        for control_id in CHEMICAL_CONTROL_IDS
    }
    observed_groups = set(
        audit[["evaluation_unit", "control_id"]].itertuples(
            index=False, name=None
        )
    )
    if observed_groups != expected_groups:
        raise ValueError("Phase 11 chemical audit control coverage mismatch.")
    canonical = pd.read_csv(scientific["dataset_path"])
    canonical["source_row_id"] = canonical["source_row_id"].astype(str)
    measured_keys = set(canonical["canonical_reaction_key"].astype(str))
    source_by_id = canonical.set_index("source_row_id", drop=False)
    budget_by_key = budgets.set_index(
        ["evaluation_unit", "control_id"], drop=False
    )
    for key, rows in audit.groupby(
        ["evaluation_unit", "control_id"], sort=False
    ):
        budget = budget_by_key.loc[key]
        accepted_mask = rows["accepted"].map(_strict_bool)
        kept_mask = rows["kept"].map(_strict_bool)
        accepted = rows.loc[accepted_mask]
        kept = rows.loc[kept_mask]
        training_candidates = training_audit.loc[
            training_audit["evaluation_unit"].eq(key[0])
            & training_audit["control_id"].eq(key[1])
            & training_audit["candidate_id"].notna()
        ].copy()
        if (
            len(training_candidates) != len(rows)
            or training_candidates["candidate_id"].duplicated().any()
            or rows["candidate_id"].duplicated().any()
            or set(
                pd.to_numeric(
                    training_candidates["candidate_id"], errors="raise"
                ).astype(int)
            )
            != set(
                pd.to_numeric(rows["candidate_id"], errors="raise").astype(
                    int
                )
            )
        ):
            raise ValueError("Phase 11 chemical/training audit coverage mismatch.")
        training_candidates = training_candidates.set_index(
            pd.to_numeric(
                training_candidates["candidate_id"], errors="raise"
            ).astype(int)
        )
        candidate_rows = rows.set_index(
            pd.to_numeric(rows["candidate_id"], errors="raise").astype(int)
        )
        for field in (
            "canonical_reaction_key",
            "canonical_reaction_hash",
            "feature_hash",
            "source_row_id",
            "donor_row_id",
            "accepted",
            "kept",
            "synthetic_label",
        ):
            left = training_candidates[field]
            right = candidate_rows[field].reindex(left.index)
            if field == "synthetic_label":
                equal = np.isclose(
                    pd.to_numeric(left, errors="coerce").to_numpy(float),
                    pd.to_numeric(right, errors="coerce").to_numpy(float),
                    rtol=0.0,
                    atol=1e-7,
                    equal_nan=True,
                ).all()
            elif field in {"accepted", "kept"}:
                equal = left.map(_strict_bool).eq(
                    right.map(_strict_bool)
                ).all()
            else:
                equal = left.fillna("").astype(str).eq(
                    right.fillna("").astype(str)
                ).all()
            if not equal:
                raise ValueError(
                    f"Phase 11 chemical/training audit mismatch: {field}."
                )
        if (
            len(kept) != int(budget["effective_added_sample_count"])
            or (kept_mask & ~accepted_mask).any()
        ):
            raise ValueError("Phase 11 chemical kept-count audit mismatch.")
        if not accepted.empty:
            spec = next(
                item
                for item in _control_plan_records()
                if item["control_id"] == key[1]
            )
            requested_roles = tuple(spec["requested_roles"])
            invalid = (
                ~accepted["chemical_parse_valid"].map(_strict_bool)
                | accepted["source_identical"].map(_strict_bool)
                | accepted["already_measured"].map(_strict_bool)
                | accepted["duplicate_synthetic"].map(_strict_bool)
                | accepted["feature_duplicate_synthetic"].map(_strict_bool)
                | ~accepted["role_change_valid"].map(_strict_bool)
                | accepted["rejection_reason"].map(_nonempty_value)
                | accepted["canonical_reaction_key"].isna()
                | accepted["canonical_reaction_hash"].isna()
                | accepted["feature_hash"].isna()
            )
            if invalid.any():
                raise ValueError("Phase 11 accepted chemistry violates identity gates.")
            feature_records: list[dict[str, Any]] = []
            for row in accepted.to_dict(orient="records"):
                canonical_roles: dict[str, str] = {}
                for role in (
                    "reactant_1",
                    "reactant_2",
                    "catalyst",
                    "ligand",
                    "base",
                    "solvent_or_additive",
                    "product",
                ):
                    canonical_column = f"canonical_{role}_smiles"
                    molecule = canonicalize_smiles(
                        str(row[canonical_column]), isomeric=True
                    )
                    if (
                        not molecule.parse_valid
                        or molecule.canonical_smiles
                        != str(row[canonical_column])
                    ):
                        raise ValueError(
                            "Phase 11 accepted role is not valid canonical chemistry."
                        )
                    canonical_roles[role] = molecule.canonical_smiles
                identity = build_canonical_reaction_identity(
                    {
                        "all_required_roles_parse_valid": True,
                        **{
                            f"canonical_{role}_smiles": value
                            for role, value in canonical_roles.items()
                        },
                    }
                )
                if (
                    identity["key"] != row["canonical_reaction_key"]
                    or identity["hash"] != row["canonical_reaction_hash"]
                ):
                    raise ValueError(
                        "Phase 11 candidate canonical identity replay mismatch."
                    )
                source = source_by_id.loc[str(row["source_row_id"])]
                changed_roles = tuple(
                    role
                    for role in (
                        "reactant_1",
                        "reactant_2",
                        "catalyst",
                        "ligand",
                        "base",
                        "solvent_or_additive",
                        "product",
                    )
                    if canonical_roles[role]
                    != str(source[f"canonical_{role}_smiles"])
                )
                unchanged_requested = tuple(
                    role
                    for role in requested_roles
                    if role not in changed_roles
                )
                unexpected = tuple(
                    role
                    for role in changed_roles
                    if role not in requested_roles
                )
                if (
                    changed_roles == ()
                    or unchanged_requested
                    or unexpected
                    or row["requested_roles"] != "|".join(requested_roles)
                    or row["actual_changed_roles"]
                    != "|".join(changed_roles)
                    or _nonempty_value(row["unchanged_requested_roles"])
                    or _nonempty_value(row["unexpected_changed_roles"])
                    or str(row["change_mask"]).zfill(7)
                    != "".join(
                        "1" if role in changed_roles else "0"
                        for role in (
                            "reactant_1",
                            "reactant_2",
                            "catalyst",
                            "ligand",
                            "base",
                            "solvent_or_additive",
                            "product",
                        )
                    )
                    or row["role_change_requirement"] != "all"
                ):
                    raise ValueError(
                        "Phase 11 candidate role-change replay mismatch."
                    )
                feature_records.append(
                    {
                        "yield": 0.0,
                        "reaction_smiles": row["reaction_smiles"],
                        **{
                            ROLE_TO_COLUMN[role]: value
                            for role, value in canonical_roles.items()
                        },
                    }
                )
            candidate_X, _, _, _ = build_feature_matrix_with_metadata(
                pd.DataFrame(feature_records), scientific["features"]
            )
            replayed_feature_hashes = [
                configured_feature_hash(vector) for vector in candidate_X
            ]
            if replayed_feature_hashes != list(accepted["feature_hash"]):
                raise ValueError("Phase 11 candidate feature hash replay mismatch.")
            if (
                set(accepted["canonical_reaction_key"].astype(str))
                & measured_keys
                or accepted["canonical_reaction_key"].duplicated().any()
                or accepted["feature_hash"].duplicated().any()
            ):
                raise ValueError(
                    "Phase 11 accepted chemistry duplicates measured/synthetic data."
                )
            for key_value, hash_value in zip(
                accepted["canonical_reaction_key"],
                accepted["canonical_reaction_hash"],
                strict=True,
            ):
                if hashlib.sha256(str(key_value).encode()).hexdigest() != str(
                    hash_value
                ):
                    raise ValueError("Phase 11 canonical reaction hash mismatch.")
        expected_accepted_hash = _chemical_accepted_candidate_hash(rows)
        if expected_accepted_hash != str(budget["accepted_candidate_hash"]):
            raise ValueError("Phase 11 accepted-candidate hash mismatch.")
        if key[1] in {
            "typed_transfer_without_uncertainty_filtering",
            "typed_transfer_with_uncertainty_filtering",
        }:
            if _chemical_pool_hash(rows) != str(
                budget["prefilter_pool_hash"]
            ):
                raise ValueError("Phase 11 chemical prefilter pool hash mismatch.")


def _chemical_accepted_candidate_hash(audit: pd.DataFrame) -> str:
    accepted = audit.loc[audit["accepted"].map(_strict_bool)]
    identities = sorted(
        (
            str(row["canonical_reaction_key"]),
            str(row["source_row_id"]),
            str(row["donor_row_id"]),
        )
        for row in accepted.to_dict(orient="records")
    )
    payload = json.dumps(identities, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _chemical_pool_hash(audit: pd.DataFrame) -> str:
    fields = (
        "canonical_reaction_key",
        "canonical_reaction_hash",
        "source_row_id",
        "donor_row_id",
    )
    records = (
        audit.loc[:, fields].fillna("").astype(str).to_dict("records")
    )
    records.sort(key=lambda row: tuple(row[field] for field in fields))
    payload = json.dumps(
        {
            "schema": "phase11-chemical-prefilter-pool-v1",
            "candidates": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _assert_metric_budget_linkage(
    metrics: pd.DataFrame,
    budgets: pd.DataFrame,
    units: tuple[EvaluationSplitUnit, ...],
    plan: Mapping[str, Any],
) -> None:
    budget_by_key = budgets.set_index(
        ["evaluation_unit", "control_id"], drop=False
    )
    unit_by_name = {unit.evaluation_unit: unit for unit in units}
    for (unit_name, control_id), rows in metrics.groupby(
        ["evaluation_unit", "control_id"], sort=False
    ):
        budget = budget_by_key.loc[(unit_name, control_id)]
        unit = unit_by_name[unit_name]
        expected_fit_hash = stable_hash(
            {
                "model_protocol_hash": plan["model_protocol_hash"],
                "model_seed": int(budget["model_seed"]),
                "exact_split_hash": unit.exact_split_hash,
                "control_config_hash": budget["control_config_hash"],
                "augmented_training_hash": budget[
                    "augmented_training_hash"
                ],
            }
        )
        try:
            resolved_control = json.loads(
                budget["resolved_control_config"]
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Phase 11 resolved control config is invalid.") from exc
        if (
            stable_hash(resolved_control) != budget["control_config_hash"]
            or budget["fit_instance_hash"] != expected_fit_hash
            or not _valid_hash(str(budget["augmented_training_hash"]))
        ):
            raise ValueError("Phase 11 fit/config hash replay mismatch.")
        _assert_frozen_control_config(
            resolved_control,
            control_id=control_id,
            unit=unit,
            budget=budget,
            plan=plan,
        )
        expected = {
            "n_train_real": len(unit.train_source_ids),
            "n_train_final": int(budget["final_train_row_count"]),
            "n_test": len(unit.test_source_ids),
            "nominal_added_sample_count": int(
                budget["nominal_added_sample_count"]
            ),
            "effective_added_sample_count": int(
                budget["effective_added_sample_count"]
            ),
            "total_sample_weight": float(budget["total_sample_weight"]),
            "exact_split_hash": budget["exact_split_hash"],
            "aggregate_assignment_hash": budget[
                "aggregate_assignment_hash"
            ],
            "feature_metadata_hash": budget["feature_metadata_hash"],
            "model_protocol_hash": budget["model_protocol_hash"],
            "budget_protocol_hash": budget["budget_protocol_hash"],
            "model_seed": int(budget["model_seed"]),
            "fit_instance_hash": budget["fit_instance_hash"],
        }
        for field, value in expected.items():
            observed = rows[field]
            if isinstance(value, float):
                equal = np.isclose(
                    observed.to_numpy(float), value, rtol=0.0, atol=1e-9
                ).all()
            else:
                equal = observed.eq(value).all()
            if not equal:
                raise ValueError(
                    f"Phase 11 metric-budget linkage mismatch: {field}."
                )


def _assert_frozen_control_config(
    resolved: Mapping[str, Any],
    *,
    control_id: str,
    unit: EvaluationSplitUnit,
    budget: pd.Series,
    plan: Mapping[str, Any],
) -> None:
    scientific = plan["resolved_scientific_config"]
    nominal = int(budget["nominal_added_sample_count"])
    seed = int(budget["model_seed"])
    if control_id in SIMPLE_CONTROL_IDS:
        expected = {
            "control_id": control_id,
            "nominal_added_budget": nominal,
            "random_state": seed,
            "n_yield_strata": scientific["augmentation"]["n_yield_strata"],
            "mixup_alpha": scientific["augmentation"]["mixup_alpha"],
            "self_training_teacher": scientific["augmentation"][
                "self_training_teacher"
            ],
        }
        if resolved != expected:
            raise ValueError("Phase 11 simple control config is not frozen.")
        return
    if set(resolved) != {
        "spec",
        "generator_config",
        "nominal_added_budget",
        "seed",
        "max_teacher_std",
        "teacher_models",
    }:
        raise ValueError("Phase 11 chemical control config schema mismatch.")
    plan_spec = next(
        item for item in plan["controls"] if item["control_id"] == control_id
    )
    expected_spec = {
        key: value for key, value in plan_spec.items() if key != "family"
    }
    expected_threshold = (
        scientific["augmentation"]["raw_teacher_std_threshold"]
        if control_id == "typed_transfer_with_uncertainty_filtering"
        else None
    )
    generator = resolved["generator_config"]
    expected_multiplier = (
        float(np.nextafter(nominal / len(unit.train_source_ids), 0.0))
        if nominal
        else 0.0
    )
    source_cap = max(
        3, math.ceil(nominal / len(unit.train_source_ids)) * 3
    )
    teachers = scientific["augmentation"]["chemical_teacher_models"]
    if expected_spec["generator_family"] == "anonymous":
        expected_generator_object: (
            ConditionTransferConfig | RoleAwareConditionTransferConfig
        ) = ConditionTransferConfig(
            donor_strategy=expected_spec["donor_strategy"],
            synthetic_multiplier=expected_multiplier,
            n_neighbors=max(1, len(unit.train_source_ids) - 1),
            label_strategy=expected_spec["label_strategy"],
            teacher_models=list(teachers),
            max_teacher_std=None,
            min_similarity=None,
            high_yield_threshold=70.0,
            clip_y_min=0.0,
            clip_y_max=100.0,
            candidates_per_real=source_cap,
            random_state=seed,
            donor_similarity_n_bits=256,
            donor_similarity_radius=2,
            donor_similarity_backend="rdkit",
            role_change_requirement="all",
            fallback_policy="reject",
            max_candidates_per_source=source_cap,
            requested_roles=tuple(expected_spec["requested_roles"]),
        )
    else:
        expected_generator_object = RoleAwareConditionTransferConfig(
            role_transfer_mode=expected_spec["role_transfer_mode"],
            donor_strategy=expected_spec["donor_strategy"],
            label_strategy=expected_spec["label_strategy"],
            synthetic_multiplier=expected_multiplier,
            max_candidates_per_source=source_cap,
            teacher_models=list(teachers),
            max_teacher_std=expected_threshold,
            min_similarity=None,
            clip_y_min=0.0,
            clip_y_max=100.0,
            random_state=seed,
            donor_similarity_n_bits=256,
            donor_similarity_radius=2,
            donor_similarity_backend="rdkit",
            role_change_requirement="all",
            fallback_policy="reject",
        )
    expected_generator = json.loads(
        json.dumps(asdict(expected_generator_object))
    )
    if (
        resolved["spec"] != expected_spec
        or int(resolved["nominal_added_budget"]) != nominal
        or int(resolved["seed"]) != seed
        or resolved["teacher_models"]
        != scientific["augmentation"]["chemical_teacher_models"]
        or resolved["max_teacher_std"] != expected_threshold
        or generator != expected_generator
    ):
        raise ValueError("Phase 11 chemical control config is not frozen.")


def _contextualize_audit(
    audit: pd.DataFrame,
    *,
    unit: EvaluationSplitUnit,
    control_id: str,
) -> pd.DataFrame:
    result = audit.copy()
    if "evaluation_unit" in result or "control_id" in result:
        raise ValueError("Phase 11 control audit already contains run context.")
    result.insert(0, "control_id", control_id)
    result.insert(0, "evaluation_unit", unit.evaluation_unit)
    return result


def _assert_split_unit_table(
    path: Path, expected_rows: list[dict[str, Any]]
) -> None:
    observed = pd.read_csv(path)
    if (
        len(observed) != len(expected_rows)
        or observed["evaluation_unit"].duplicated().any()
    ):
        raise ValueError("Phase 11 split-unit table coverage mismatch.")
    observed = observed.set_index("evaluation_unit", drop=False)
    for expected in expected_rows:
        row = observed.loc[expected["evaluation_unit"]]
        for field in (
            "n_train",
            "n_validation",
            "n_test",
            "n_excluded",
            "train_source_id_hash",
            "validation_source_id_hash",
            "test_source_id_hash",
            "excluded_source_id_hash",
            "dataset_hash",
            "exact_split_hash",
            "aggregate_assignment_hash",
            "canonical_split_dependency_hash",
            "canonicalization_version",
            "split_schema_version",
        ):
            if field.startswith("n_"):
                equal = int(row[field]) == int(expected[field])
            else:
                equal = str(row[field]) == str(expected[field])
            if not equal:
                raise ValueError(
                    f"Phase 11 split-unit mismatch: "
                    f"{expected['evaluation_unit']}/{field}."
                )


def _fit_control_estimator(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Any:
    """Fit controls strictly: weighted methods may never fall back unweighted."""
    if sample_weight is None:
        return train_model(estimator, X, y)
    try:
        parameters = inspect.signature(estimator.fit).parameters.values()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Phase 11 cannot verify estimator sample_weight support."
        ) from exc
    if not any(parameter.name == "sample_weight" for parameter in parameters):
        raise ValueError(
            f"Phase 11 model {type(estimator).__name__} lacks sample_weight support."
        )
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError as exc:
        raise ValueError(
            f"Phase 11 model {type(estimator).__name__} rejected sample_weight."
        ) from exc
    return estimator


def _assert_control_metadata(
    metadata: pd.DataFrame, plan_controls: list[dict[str, Any]]
) -> None:
    expected_names = {
        item["control_id"]: item["control_name"] for item in plan_controls
    }
    if (
        tuple(metadata["control_id"]) != CONTROL_IDS
        or tuple(metadata["control_name"])
        != tuple(expected_names[control_id] for control_id in CONTROL_IDS)
    ):
        raise ValueError("Augmentation-control metadata identity mismatch.")
    feature_controls = {
        "nearest_neighbor_pseudo_labeling",
        "self_training",
        "feature_mixup",
    }
    for row in metadata.to_dict(orient="records"):
        applicable = _strict_bool(row["chemical_identity_applicable"])
        expected_limitation = (
            _NONCHEMICAL_DESCRIPTION
            if row["control_id"] in feature_controls
            else None
        )
        if (
            applicable == (row["control_id"] in feature_controls)
            or (
                expected_limitation is None
                and _nonempty_value(row["nonchemical_semantics"])
            )
            or (
                expected_limitation is not None
                and row["nonchemical_semantics"] != expected_limitation
            )
        ):
            raise ValueError("Augmentation-control metadata semantics mismatch.")


def _row_feature_hash(row: np.ndarray) -> str:
    value = configured_feature_hash(np.asarray(row, dtype=np.float32))
    if value is None:
        raise ValueError("Phase 11 cannot hash a non-finite training feature row.")
    return value


def _valid_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_config(
    config: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return json.loads(json.dumps(config))
    loaded = yaml.safe_load(Path(config).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("Phase 11 config must be a mapping.")
    return loaded


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Invalid artifact boolean value: {value!r}.")


def _nonempty_value(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scikit-learn", "rdkit", "xgboost"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
