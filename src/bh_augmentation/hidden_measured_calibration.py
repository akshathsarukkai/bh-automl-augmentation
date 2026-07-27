"""Validation-calibrated uncertainty and hidden-measured transfer experiments."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_saved_random_split_unit,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.hidden_condition_probes import (
    SUPPORTED_PROBE_DONOR_STRATEGIES,
    build_hidden_condition_probes_from_context,
    prepare_hidden_condition_probe_context,
)
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.uncertainty.calibration import (
    CalibrationDataRoles,
    ValidationCalibrationCandidate,
    build_selective_prediction_curve,
    interval_calibration_metrics,
    select_validation_policy,
)
from bh_augmentation.uncertainty.estimators import (
    SUPPORTED_UNCERTAINTY_METHODS,
    EstimatorAudit,
    FittedUncertaintyEstimator,
    UncertaintyEstimatorConfig,
    fit_uncertainty_estimator,
)
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
)

PHASE12_SCHEMA_VERSION = "bh-hidden-measured-calibration-v1"
PARTITION_PROTOCOL = "saved-nested-train-global-condition-holdout-v1"
VALIDATION_PROTOCOL = "saved-validation-condition-group-disjoint-halves-v1"
PROBE_ROLE_SETS = (
    ("ligand",),
    ("base",),
    ("solvent_or_additive",),
    ("ligand", "base"),
    ("ligand", "solvent_or_additive"),
    ("base", "solvent_or_additive"),
)
UNSUPPORTED_PROBE_ROLE_SETS = (
    ("catalyst",),
    ("ligand", "base", "solvent_or_additive"),
    ("catalyst", "ligand", "base", "solvent_or_additive"),
)
_OUTPUT_NAMES = (
    "calibration_plan.json",
    "split_units.csv",
    "partition_assignments.csv",
    "fit_audit.csv",
    "probe_provenance.csv",
    "probe_exclusions.csv",
    "calibration_predictions.csv",
    "calibration_metrics.csv",
    "selective_prediction_curves.csv",
    "frozen_uncertainty_policies.json",
    "hidden_measured_predictions.csv",
    "hidden_measured_metrics.csv",
    "hidden_stratified_metrics.csv",
    "filtering_evidence.csv",
)


@dataclass(frozen=True, slots=True)
class CalibrationHoldoutUnit:
    """One saved split/fraction with one globally removed condition group."""

    evaluation_unit: str
    base_evaluation_unit: str
    seed: int
    train_fraction: float
    held_condition_key: str
    teacher_fit_source_ids: tuple[str, ...]
    hidden_source_ids: tuple[str, ...]
    interval_calibration_source_ids: tuple[str, ...]
    policy_validation_source_ids: tuple[str, ...]
    held_condition_validation_excluded_source_ids: tuple[str, ...]
    outer_test_source_ids: tuple[str, ...]
    nested_excluded_source_ids: tuple[str, ...]
    exact_split_hash: str
    aggregate_split_hash: str
    source_id_split_hash: str

    @property
    def record(self) -> dict[str, Any]:
        """Return a complete immutable membership record."""
        payload = {
            "evaluation_unit": self.evaluation_unit,
            "base_evaluation_unit": self.base_evaluation_unit,
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "held_condition_key": self.held_condition_key,
            "partition_protocol": PARTITION_PROTOCOL,
            "validation_protocol": VALIDATION_PROTOCOL,
            "teacher_fit_count": len(self.teacher_fit_source_ids),
            "hidden_count": len(self.hidden_source_ids),
            "interval_calibration_count": len(
                self.interval_calibration_source_ids
            ),
            "policy_validation_count": len(
                self.policy_validation_source_ids
            ),
            "held_condition_validation_excluded_count": len(
                self.held_condition_validation_excluded_source_ids
            ),
            "outer_test_count": len(self.outer_test_source_ids),
            "nested_excluded_count": len(self.nested_excluded_source_ids),
            "teacher_fit_source_id_hash": stable_hash(
                list(self.teacher_fit_source_ids)
            ),
            "hidden_source_id_hash": stable_hash(list(self.hidden_source_ids)),
            "interval_calibration_source_id_hash": stable_hash(
                list(self.interval_calibration_source_ids)
            ),
            "policy_validation_source_id_hash": stable_hash(
                list(self.policy_validation_source_ids)
            ),
            "held_condition_validation_excluded_source_id_hash": stable_hash(
                list(self.held_condition_validation_excluded_source_ids)
            ),
            "outer_test_source_id_hash": stable_hash(
                list(self.outer_test_source_ids)
            ),
            "nested_excluded_source_id_hash": stable_hash(
                list(self.nested_excluded_source_ids)
            ),
            "exact_split_hash": self.exact_split_hash,
            "aggregate_split_hash": self.aggregate_split_hash,
            "source_id_split_hash": self.source_id_split_hash,
            "outer_test_labels_accessible_during_search": False,
            "hidden_labels_accessible_during_search": False,
        }
        payload["holdout_unit_hash"] = stable_hash(payload)
        return payload


def run_hidden_measured_calibration(
    config: str | Path | Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run validation-only uncertainty selection then hidden outcome scoring."""
    contract = _resolve_contract(_load_config(config))
    saved = load_saved_canonical_split_identities(
        contract["dataset_path"],
        contract["canonical_split_directory"],
        requested_seeds=contract["random_seeds"],
        requested_fractions=contract["random_fractions"],
    )
    base_units = tuple(
        build_saved_random_split_unit(saved, seed=seed, train_fraction=fraction)
        for seed in contract["random_seeds"]
        for fraction in contract["random_fractions"]
    )
    by_seed_fraction = {
        (int(unit.seed), float(unit.train_fraction)): unit for unit in base_units
    }
    panels = _select_condition_panels(
        saved.canonical,
        by_seed_fraction,
        seeds=contract["random_seeds"],
        fractions=contract["random_fractions"],
        n_groups=contract["hidden_experiment"]["n_condition_groups"],
        minimum_hidden_rows=contract["hidden_experiment"][
            "minimum_hidden_rows"
        ],
        salt=contract["hidden_experiment"]["panel_salt"],
    )
    holdout_units = tuple(
        _build_holdout_unit(
            unit,
            saved.canonical,
            held_condition_key=held_key,
            validation_salt=contract["hidden_experiment"][
                "validation_partition_salt"
            ],
        )
        for unit in base_units
        for held_key in panels[int(unit.seed)]
    )
    if not holdout_units:
        raise ValueError("Phase 12 has no calibration holdout units.")

    feature_config = resolve_corrected_feature_config(
        contract["features"], required_kind="bh_role_separated"
    )
    identity_frame = saved.canonical.copy()
    identity_frame["yield"] = 0.0
    X_all, _, feature_names, feature_metadata = (
        build_feature_matrix_with_metadata(identity_frame, feature_config)
    )
    X_all = np.asarray(X_all, dtype=np.float32)
    feature_contract = feature_contract_record(feature_metadata, feature_names)
    positions = {
        source_id: position
        for position, source_id in enumerate(
            saved.canonical["source_row_id"].astype(str)
        )
    }
    by_source = saved.canonical.set_index("source_row_id", drop=False)

    partition_rows = _partition_rows(holdout_units)
    probe_rows, exclusion_rows = _build_probe_plan(
        holdout_units,
        by_source,
        role_sets=contract["hidden_experiment"]["requested_role_sets"],
        donor_strategies=contract["hidden_experiment"]["donor_strategies"],
        seed=contract["base_seed"],
        similarity_n_bits=contract["hidden_experiment"][
            "similarity_n_bits"
        ],
        similarity_radius=contract["hidden_experiment"][
            "similarity_radius"
        ],
    )
    probed_ids_by_unit = _probed_source_ids(holdout_units, probe_rows)
    context_edges_by_unit = {
        unit.evaluation_unit: _tertile_edges(
            [
                float(record["substrate_similarity"])
                for record in probe_rows
                if record["evaluation_unit"] == unit.evaluation_unit
            ]
        )
        for unit in holdout_units
    }
    scientific_config = {
        **contract,
        "dataset_path": str(contract["dataset_path"]),
        "canonical_split_directory": str(
            contract["canonical_split_directory"]
        ),
        "features": feature_config,
    }
    scientific_config.pop("output_directory", None)
    plan = {
        "schema_version": PHASE12_SCHEMA_VERSION,
        "status": "frozen_before_any_label_loading",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "canonical_split_artifact_hashes": saved.artifact_hashes,
        "split_units": [unit.audit_record for unit in base_units],
        "condition_panels": {
            str(seed): list(groups) for seed, groups in panels.items()
        },
        "holdout_units": [unit.record for unit in holdout_units],
        "probe_coverage": [
            {
                "evaluation_unit": unit.evaluation_unit,
                "hidden_target_denominator": len(unit.hidden_source_ids),
                "exactly_reconstructed_target_count": len(
                    probed_ids_by_unit[unit.evaluation_unit]
                ),
                "reconstruction_coverage": len(
                    probed_ids_by_unit[unit.evaluation_unit]
                )
                / len(unit.hidden_source_ids),
                "probed_source_id_hash": stable_hash(
                    list(probed_ids_by_unit[unit.evaluation_unit])
                ),
            }
            for unit in holdout_units
        ],
        "descriptive_context_similarity_bin_edges": context_edges_by_unit,
        "feature_contract": feature_contract,
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "methods": list(SUPPORTED_UNCERTAINTY_METHODS),
        "data_roles": asdict(CalibrationDataRoles()),
        "hidden_probe_semantics": (
            "canonical measured-duplicate evaluation probes only; "
            "eligible_for_training=false; normal synthetic measured-duplicate "
            "rejection remains unchanged"
        ),
        "threshold_selection": {
            "selection_data_role": "policy_validation",
            "nominal_coverage": contract["uncertainty"]["coverage"],
            "max_calibration_error": contract["uncertainty"][
                "max_calibration_error"
            ],
            "min_retained_fraction": contract["uncertainty"][
                "min_retained_fraction"
            ],
            "configured_raw_threshold_used": False,
        },
        "resolved_scientific_config": scientific_config,
        "config_hash": stable_hash(scientific_config),
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metrics_generated": False,
    }
    plan["probe_plan_hash"] = stable_hash(
        _stable_records(probe_rows, "probe_id")
    )
    plan["probe_exclusion_plan_hash"] = stable_hash(
        _stable_records(exclusion_rows, "exclusion_id")
    )
    plan["partition_plan_hash"] = stable_hash(
        _stable_records(partition_rows, "partition_id")
    )
    plan["plan_hash"] = stable_hash(plan)

    output = Path(
        output_directory
        if output_directory is not None
        else contract["output_directory"]
    )
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 12 output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        name.removesuffix(".json").removesuffix(".csv"): output / name
        for name in _OUTPUT_NAMES
    }
    paths["output_directory"] = output
    paths["manifest"] = output / "manifest.json"
    _write_json(paths["calibration_plan"], plan)
    pd.DataFrame([unit.audit_record for unit in base_units]).to_csv(
        paths["split_units"], index=False
    )
    pd.DataFrame(partition_rows).to_csv(
        paths["partition_assignments"], index=False
    )
    pd.DataFrame(probe_rows).to_csv(paths["probe_provenance"], index=False)
    pd.DataFrame(exclusion_rows).to_csv(paths["probe_exclusions"], index=False)

    fitted: dict[
        tuple[str, str], FittedUncertaintyEstimator
    ] = {}
    policy_prediction_by_key: dict[tuple[str, str], Any] = {}
    fit_rows: list[dict[str, Any]] = []
    calibration_prediction_rows: list[dict[str, Any]] = []
    calibration_metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    for unit in holdout_units:
        search_ids = tuple(
            sorted(
                unit.teacher_fit_source_ids
                + unit.interval_calibration_source_ids
                + unit.policy_validation_source_ids
            )
        )
        outcomes = _read_allowed_outcomes(
            contract["dataset_path"],
            saved.canonical["source_row_id"].astype(str).tolist(),
            search_ids,
        )
        train_X, train_y = _features_and_labels(
            unit.teacher_fit_source_ids, X_all, positions, outcomes
        )
        interval_X, interval_y = _features_and_labels(
            unit.interval_calibration_source_ids, X_all, positions, outcomes
        )
        policy_X, policy_y = _features_and_labels(
            unit.policy_validation_source_ids, X_all, positions, outcomes
        )
        candidates: list[ValidationCalibrationCandidate] = []
        for method in SUPPORTED_UNCERTAINTY_METHODS:
            estimator_config = _estimator_config(
                method,
                contract["uncertainty"]["method_configs"][method],
                coverage=contract["uncertainty"]["coverage"],
                base_seed=contract["base_seed"],
                evaluation_unit=unit.evaluation_unit,
            )
            estimator = fit_uncertainty_estimator(
                train_X,
                train_y,
                train_source_ids=unit.teacher_fit_source_ids,
                calibration_features=interval_X,
                calibration_labels=interval_y,
                calibration_source_ids=unit.interval_calibration_source_ids,
                config=estimator_config,
            )
            fitted[(unit.evaluation_unit, method)] = estimator
            fit_rows.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "held_condition_key": unit.held_condition_key,
                    **asdict(estimator.audit),
                    "audit_hash": estimator.audit.audit_hash,
                    "resolved_config": json.dumps(
                        asdict(estimator_config),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "hidden_source_id_hash": stable_hash(
                        list(unit.hidden_source_ids)
                    ),
                    "outer_test_source_id_hash": stable_hash(
                        list(unit.outer_test_source_ids)
                    ),
                    "hidden_or_test_fit_contamination": False,
                    "plan_hash": plan["plan_hash"],
                }
            )
            interval_prediction = estimator.predict(
                interval_X,
                source_ids=unit.interval_calibration_source_ids,
            )
            policy_prediction = estimator.predict(
                policy_X,
                source_ids=unit.policy_validation_source_ids,
            )
            policy_prediction_by_key[
                (unit.evaluation_unit, method)
            ] = policy_prediction
            calibration_prediction_rows.extend(
                _prediction_rows(
                    unit,
                    method,
                    "interval_calibration",
                    unit.interval_calibration_source_ids,
                    interval_y,
                    interval_prediction,
                    estimator.config.config_hash,
                    plan["plan_hash"],
                )
            )
            calibration_prediction_rows.extend(
                _prediction_rows(
                    unit,
                    method,
                    "policy_validation",
                    unit.policy_validation_source_ids,
                    policy_y,
                    policy_prediction,
                    estimator.config.config_hash,
                    plan["plan_hash"],
                )
            )
            metrics = interval_calibration_metrics(
                policy_y,
                policy_prediction.point,
                policy_prediction.interval_lower,
                policy_prediction.interval_upper,
                policy_prediction.calibrated_uncertainty,
                nominal_coverage=contract["uncertainty"]["coverage"],
            )
            curve = build_selective_prediction_curve(
                policy_y,
                policy_prediction.point,
                policy_prediction.calibrated_uncertainty,
                unit.policy_validation_source_ids,
            )
            candidates.append(
                ValidationCalibrationCandidate(
                    method_id=method,
                    method_config_hash=estimator.config.config_hash,
                    metrics=metrics,
                    selective_curve=curve,
                    data_roles=CalibrationDataRoles(),
                )
            )
            calibration_metric_rows.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "method": method,
                    **asdict(metrics),
                    "selection_data_role": "policy_validation",
                    "estimator_audit_hash": estimator.audit.audit_hash,
                    "method_config_hash": estimator.config.config_hash,
                    "plan_hash": plan["plan_hash"],
                }
            )
            curve_rows.extend(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "method": method,
                    "method_config_hash": estimator.config.config_hash,
                    **asdict(point),
                    "selection_data_role": "policy_validation",
                    "plan_hash": plan["plan_hash"],
                }
                for point in curve
            )
        selection = select_validation_policy(
            candidates,
            nominal_coverage=contract["uncertainty"]["coverage"],
            max_calibration_error=contract["uncertainty"][
                "max_calibration_error"
            ],
            min_retained_fraction=contract["uncertainty"][
                "min_retained_fraction"
            ],
        )
        selected_prediction = policy_prediction_by_key[
            (unit.evaluation_unit, selection.method_id)
        ]
        uncertainty_edges = _tertile_edges(
            selected_prediction.calibrated_uncertainty
        )
        support_edges = _tertile_edges(
            policy_prediction_by_key[
                (unit.evaluation_unit, "distance_aware_ridge")
            ].raw_score
        )
        payload = {
            "schema_version": PHASE12_SCHEMA_VERSION,
            "status": "frozen_before_hidden_outcome_loading",
            "evaluation_unit": unit.evaluation_unit,
            "holdout_unit_hash": unit.record["holdout_unit_hash"],
            "selection": asdict(selection),
            "estimator_audit_hash": fitted[
                (unit.evaluation_unit, selection.method_id)
            ].audit.audit_hash,
            "validation_prediction_hash": selected_prediction.prediction_hash,
            "uncertainty_bin_edges": uncertainty_edges,
            "support_distance_bin_edges": support_edges,
            "context_similarity_bin_edges": context_edges_by_unit[
                unit.evaluation_unit
            ],
            "context_similarity_bins_are_descriptive": True,
            "plan_hash": plan["plan_hash"],
            "selection_data_roles": ["policy_validation"],
            "hidden_labels_accessed": False,
            "outer_test_labels_accessed": False,
        }
        payload["frozen_policy_hash"] = stable_hash(payload)
        policy_records.append(payload)

    pd.DataFrame(fit_rows).to_csv(paths["fit_audit"], index=False)
    pd.DataFrame(calibration_prediction_rows).to_csv(
        paths["calibration_predictions"], index=False
    )
    pd.DataFrame(calibration_metric_rows).to_csv(
        paths["calibration_metrics"], index=False
    )
    pd.DataFrame(curve_rows).to_csv(
        paths["selective_prediction_curves"], index=False
    )
    policy_document = {
        "schema_version": PHASE12_SCHEMA_VERSION,
        "status": "frozen_before_hidden_outcome_loading",
        "plan_hash": plan["plan_hash"],
        "policy_count": len(policy_records),
        "policies": policy_records,
        "hidden_labels_accessed": False,
        "outer_test_labels_accessed": False,
    }
    policy_document["policy_document_hash"] = stable_hash(policy_document)
    _write_json(paths["frozen_uncertainty_policies"], policy_document)

    policy_by_unit = {
        record["evaluation_unit"]: record for record in policy_records
    }
    hidden_rows: list[dict[str, Any]] = []
    filtering_rows: list[dict[str, Any]] = []
    hidden_metric_rows: list[dict[str, Any]] = []
    for unit in holdout_units:
        evaluated_source_ids = probed_ids_by_unit[unit.evaluation_unit]
        hidden_outcomes = _read_allowed_outcomes(
            contract["dataset_path"],
            saved.canonical["source_row_id"].astype(str).tolist(),
            evaluated_source_ids,
        )
        hidden_positions = np.asarray(
            [positions[source_id] for source_id in evaluated_source_ids],
            dtype=int,
        )
        hidden_X = X_all[hidden_positions]
        hidden_y = np.asarray(
            [hidden_outcomes[source_id] for source_id in evaluated_source_ids],
            dtype=float,
        )
        support_X = X_all[
            np.asarray(
                [positions[source_id] for source_id in unit.teacher_fit_source_ids],
                dtype=int,
            )
        ]
        support_distance = _nearest_cosine_distance(hidden_X, support_X)
        policy = policy_by_unit[unit.evaluation_unit]
        selected_method = policy["selection"]["method_id"]
        cutoff = float(policy["selection"]["uncertainty_cutoff"])
        for method in SUPPORTED_UNCERTAINTY_METHODS:
            estimator = fitted[(unit.evaluation_unit, method)]
            prediction = estimator.predict(
                hidden_X, source_ids=evaluated_source_ids
            )
            absolute_error = np.abs(hidden_y - prediction.point)
            covered = (hidden_y >= prediction.interval_lower) & (
                hidden_y <= prediction.interval_upper
            )
            for index, source_id in enumerate(evaluated_source_ids):
                is_selected = method == selected_method
                accepted = bool(
                    is_selected
                    and prediction.calibrated_uncertainty[index] <= cutoff
                )
                hidden_rows.append(
                    {
                        "evaluation_unit": unit.evaluation_unit,
                        "seed": unit.seed,
                        "train_fraction": unit.train_fraction,
                        "held_condition_key": unit.held_condition_key,
                        "hidden_row_id": source_id,
                        "canonical_reaction_key": by_source.loc[
                            source_id, "canonical_reaction_key"
                        ],
                        "method": method,
                        "method_config_hash": estimator.config.config_hash,
                        "estimator_audit_hash": estimator.audit.audit_hash,
                        "point_prediction": float(prediction.point[index]),
                        "interval_lower": float(
                            prediction.interval_lower[index]
                        ),
                        "interval_upper": float(
                            prediction.interval_upper[index]
                        ),
                        "calibrated_uncertainty": float(
                            prediction.calibrated_uncertainty[index]
                        ),
                        "nearest_visible_support_distance": float(
                            support_distance[index]
                        ),
                        "measured_yield": float(hidden_y[index]),
                        "signed_error": float(
                            prediction.point[index] - hidden_y[index]
                        ),
                        "absolute_error": float(absolute_error[index]),
                        "interval_covered": bool(covered[index]),
                        "selected_by_validation": is_selected,
                        "accepted_by_frozen_filter": accepted,
                        "frozen_policy_hash": policy["frozen_policy_hash"],
                        "hidden_outcome_data_role": "hidden_measured_evaluation",
                        "eligible_for_training": False,
                        "outer_test_data_used": False,
                        "plan_hash": plan["plan_hash"],
                    }
                )
                if is_selected:
                    filtering_rows.append(
                        {
                            "evaluation_unit": unit.evaluation_unit,
                            "hidden_row_id": source_id,
                            "method": method,
                            "calibrated_uncertainty": float(
                                prediction.calibrated_uncertainty[index]
                            ),
                            "frozen_uncertainty_cutoff": cutoff,
                            "accepted_by_frozen_filter": accepted,
                            "filter_rule": (
                                "calibrated_uncertainty <= "
                                "validation_selected_cutoff"
                            ),
                            "calibration_evidence_hash": stable_hash(
                                {
                                    "estimator_audit_hash": (
                                        estimator.audit.audit_hash
                                    ),
                                    "validation_prediction_hash": policy[
                                        "validation_prediction_hash"
                                    ],
                                    "selection": policy["selection"],
                                }
                            ),
                            "frozen_policy_hash": policy[
                                "frozen_policy_hash"
                            ],
                            "eligible_for_training": False,
                            "plan_hash": plan["plan_hash"],
                        }
                    )
            metrics = interval_calibration_metrics(
                hidden_y,
                prediction.point,
                prediction.interval_lower,
                prediction.interval_upper,
                prediction.calibrated_uncertainty,
                nominal_coverage=contract["uncertainty"]["coverage"],
            )
            hidden_metric_rows.append(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "seed": unit.seed,
                    "train_fraction": unit.train_fraction,
                    "held_condition_key": unit.held_condition_key,
                    "method": method,
                    "rmse": float(
                        np.sqrt(np.mean((hidden_y - prediction.point) ** 2))
                    ),
                    "mae": float(np.mean(np.abs(hidden_y - prediction.point))),
                    "median_absolute_error": float(
                        np.median(np.abs(hidden_y - prediction.point))
                    ),
                    "mean_signed_bias": float(
                        np.mean(prediction.point - hidden_y)
                    ),
                    **asdict(metrics),
                    "unique_target_count": len(hidden_y),
                    "hidden_target_denominator": len(unit.hidden_source_ids),
                    "exactly_reconstructed_target_count": len(
                        evaluated_source_ids
                    ),
                    "reconstruction_coverage": (
                        len(evaluated_source_ids) / len(unit.hidden_source_ids)
                    ),
                    "evaluation_scope": (
                        "exactly_reconstructed_hidden_targets_only"
                    ),
                    "hidden_outcomes_used_for_selection": False,
                    "outer_test_data_used": False,
                    "plan_hash": plan["plan_hash"],
                }
            )

    hidden_frame = pd.DataFrame(hidden_rows)
    filtering_frame = pd.DataFrame(filtering_rows)
    hidden_metrics = pd.DataFrame(hidden_metric_rows)
    hidden_frame.to_csv(paths["hidden_measured_predictions"], index=False)
    hidden_metrics.to_csv(paths["hidden_measured_metrics"], index=False)
    pd.DataFrame(filtering_rows).to_csv(paths["filtering_evidence"], index=False)
    stratified = _stratified_hidden_metrics(
        hidden_frame,
        pd.DataFrame(probe_rows),
        policy_by_unit,
        holdout_units,
    )
    stratified.to_csv(paths["hidden_stratified_metrics"], index=False)

    output_hashes = {
        path.name: sha256_file(path)
        for key, path in paths.items()
        if key not in {"output_directory", "manifest"}
    }
    manifest = {
        "schema_version": PHASE12_SCHEMA_VERSION,
        "status": "corrected_revalidation_complete",
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "feature_metadata_hash": feature_contract["feature_metadata_hash"],
        "config_hash": plan["config_hash"],
        "plan_hash": plan["plan_hash"],
        "policy_document_hash": policy_document["policy_document_hash"],
        "resolved_scientific_config": scientific_config,
        "dependency_versions": _dependency_versions(),
        "command": _command_record(config, output),
        "seeds": list(contract["random_seeds"]),
        "fractions": list(contract["random_fractions"]),
        "holdout_unit_count": len(holdout_units),
        "method_count": len(SUPPORTED_UNCERTAINTY_METHODS),
        "calibrated_methods_passing_selection": sorted(
            {record["selection"]["method_id"] for record in policy_records}
        ),
        "partition_row_count": len(partition_rows),
        "probe_row_count": len(probe_rows),
        "probe_exclusion_row_count": len(exclusion_rows),
        "fit_audit_row_count": len(fit_rows),
        "calibration_prediction_row_count": len(
            calibration_prediction_rows
        ),
        "calibration_metric_row_count": len(calibration_metric_rows),
        "selective_curve_row_count": len(curve_rows),
        "hidden_prediction_row_count": len(hidden_frame),
        "hidden_target_denominator_unit_sum": sum(
            len(unit.hidden_source_ids) for unit in holdout_units
        ),
        "exactly_reconstructed_target_unit_sum": sum(
            len(probed_ids_by_unit[unit.evaluation_unit])
            for unit in holdout_units
        ),
        "hidden_metric_row_count": len(hidden_metrics),
        "hidden_stratified_metric_row_count": len(stratified),
        "filtering_evidence_row_count": len(filtering_frame),
        "selection_data_roles": ["policy_validation"],
        "hidden_labels_accessed_during_search": False,
        "hidden_labels_accessed_after_policy_freeze": True,
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metrics_generated": False,
        "test_evaluated": False,
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    _write_json(paths["manifest"], manifest)
    validate_hidden_measured_calibration(output)
    return paths


def validate_hidden_measured_calibration(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Semantically replay Phase 12 artifacts without refitting estimators."""
    root = Path(output_directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed_manifest_hash = manifest.pop("manifest_hash", None)
    if (
        manifest.get("schema_version") != PHASE12_SCHEMA_VERSION
        or manifest.get("status") != "corrected_revalidation_complete"
        or claimed_manifest_hash != stable_hash(manifest)
        or manifest.get("hidden_labels_accessed_during_search") is not False
        or manifest.get("hidden_labels_accessed_after_policy_freeze") is not True
        or manifest.get("outer_test_labels_accessed") is not False
        or manifest.get("outer_test_predictions_generated") is not False
        or manifest.get("outer_test_metrics_generated") is not False
        or manifest.get("test_evaluated") is not False
        or manifest.get("selection_data_roles") != ["policy_validation"]
    ):
        raise ValueError("Invalid Phase 12 completion manifest.")
    manifest["manifest_hash"] = claimed_manifest_hash
    expected_names = set(_OUTPUT_NAMES)
    if set(manifest["output_hashes"]) != expected_names:
        raise ValueError("Phase 12 output hash coverage mismatch.")
    for name, expected in manifest["output_hashes"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Phase 12 output hash mismatch: {name}.")

    plan = json.loads((root / "calibration_plan.json").read_text())
    claimed_plan_hash = plan.pop("plan_hash", None)
    if (
        plan.get("schema_version") != PHASE12_SCHEMA_VERSION
        or plan.get("status") != "frozen_before_any_label_loading"
        or claimed_plan_hash != stable_hash(plan)
        or plan.get("outer_test_labels_accessed") is not False
        or plan.get("outer_test_predictions_generated") is not False
        or plan.get("outer_test_metrics_generated") is not False
        or plan.get("methods") != list(SUPPORTED_UNCERTAINTY_METHODS)
        or plan.get("data_roles")
        != _jsonable(asdict(CalibrationDataRoles()))
        or plan.get("threshold_selection", {}).get(
            "configured_raw_threshold_used"
        )
        is not False
    ):
        raise ValueError("Invalid frozen Phase 12 plan.")
    plan["plan_hash"] = claimed_plan_hash
    if (
        manifest["plan_hash"] != claimed_plan_hash
        or manifest["config_hash"] != plan["config_hash"]
        or manifest["resolved_scientific_config"]
        != plan["resolved_scientific_config"]
    ):
        raise ValueError("Phase 12 manifest-plan provenance mismatch.")
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
    ):
        raise ValueError("Phase 12 canonical dependency mismatch.")
    base_units = tuple(
        build_saved_random_split_unit(saved, seed=seed, train_fraction=fraction)
        for seed in scientific["random_seeds"]
        for fraction in scientific["random_fractions"]
    )
    panels = {
        int(seed): tuple(groups)
        for seed, groups in plan["condition_panels"].items()
    }
    replay_units_by_seed_fraction = {
        (int(unit.seed), float(unit.train_fraction)): unit
        for unit in base_units
    }
    expected_panels = _select_condition_panels(
        saved.canonical,
        replay_units_by_seed_fraction,
        seeds=scientific["random_seeds"],
        fractions=scientific["random_fractions"],
        n_groups=scientific["hidden_experiment"]["n_condition_groups"],
        minimum_hidden_rows=scientific["hidden_experiment"][
            "minimum_hidden_rows"
        ],
        salt=scientific["hidden_experiment"]["panel_salt"],
    )
    if panels != expected_panels:
        raise ValueError("Phase 12 hidden condition panel replay mismatch.")
    expected_holdouts = tuple(
        _build_holdout_unit(
            unit,
            saved.canonical,
            held_condition_key=held_key,
            validation_salt=scientific["hidden_experiment"][
                "validation_partition_salt"
            ],
        )
        for unit in base_units
        for held_key in panels[int(unit.seed)]
    )
    expected_records = [unit.record for unit in expected_holdouts]
    if stable_hash(expected_records) != stable_hash(plan["holdout_units"]):
        raise ValueError("Phase 12 holdout membership replay mismatch.")
    read_options = {"float_precision": "round_trip"}
    partitions = pd.read_csv(
        root / "partition_assignments.csv", **read_options
    )
    if stable_hash(
        _stable_records(partitions.to_dict("records"), "partition_id")
    ) != plan["partition_plan_hash"]:
        raise ValueError("Phase 12 partition artifact mismatch.")
    _assert_partition_semantics(partitions, expected_holdouts)

    by_source = saved.canonical.set_index("source_row_id", drop=False)
    expected_probes, expected_exclusions = _build_probe_plan(
        expected_holdouts,
        by_source,
        role_sets=scientific["hidden_experiment"]["requested_role_sets"],
        donor_strategies=scientific["hidden_experiment"]["donor_strategies"],
        seed=scientific["base_seed"],
        similarity_n_bits=scientific["hidden_experiment"][
            "similarity_n_bits"
        ],
        similarity_radius=scientific["hidden_experiment"][
            "similarity_radius"
        ],
    )
    expected_probed_ids = _probed_source_ids(
        expected_holdouts, expected_probes
    )
    expected_probe_coverage = [
        {
            "evaluation_unit": unit.evaluation_unit,
            "hidden_target_denominator": len(unit.hidden_source_ids),
            "exactly_reconstructed_target_count": len(
                expected_probed_ids[unit.evaluation_unit]
            ),
            "reconstruction_coverage": len(
                expected_probed_ids[unit.evaluation_unit]
            )
            / len(unit.hidden_source_ids),
            "probed_source_id_hash": stable_hash(
                list(expected_probed_ids[unit.evaluation_unit])
            ),
        }
        for unit in expected_holdouts
    ]
    expected_context_edges = {
        unit.evaluation_unit: _tertile_edges(
            [
                float(record["substrate_similarity"])
                for record in expected_probes
                if record["evaluation_unit"] == unit.evaluation_unit
            ]
        )
        for unit in expected_holdouts
    }
    if (
        not _records_close(
            expected_probe_coverage, plan.get("probe_coverage", [])
        )
        or plan.get("descriptive_context_similarity_bin_edges")
        != expected_context_edges
    ):
        raise ValueError("Phase 12 probe coverage plan replay mismatch.")
    probes = pd.read_csv(root / "probe_provenance.csv", **read_options)
    exclusions = pd.read_csv(root / "probe_exclusions.csv", **read_options)
    if (
        stable_hash(_csv_stable_records(probes, "probe_id"))
        != plan["probe_plan_hash"]
        or stable_hash(_csv_stable_records(exclusions, "exclusion_id"))
        != plan["probe_exclusion_plan_hash"]
        or stable_hash(_csv_stable_records(probes, "probe_id"))
        != stable_hash(_stable_records(expected_probes, "probe_id"))
        or stable_hash(_csv_stable_records(exclusions, "exclusion_id"))
        != stable_hash(_stable_records(expected_exclusions, "exclusion_id"))
    ):
        raise ValueError("Phase 12 canonical probe replay mismatch.")

    fit_audit = pd.read_csv(root / "fit_audit.csv", **read_options)
    calibration_predictions = pd.read_csv(
        root / "calibration_predictions.csv", **read_options
    )
    calibration_metrics = pd.read_csv(
        root / "calibration_metrics.csv", **read_options
    )
    curves = pd.read_csv(
        root / "selective_prediction_curves.csv", **read_options
    )
    policy_document = json.loads(
        (root / "frozen_uncertainty_policies.json").read_text()
    )
    hidden = pd.read_csv(
        root / "hidden_measured_predictions.csv", **read_options
    )
    hidden_metrics = pd.read_csv(
        root / "hidden_measured_metrics.csv", **read_options
    )
    filtering = pd.read_csv(
        root / "filtering_evidence.csv", **read_options
    )
    stratified = pd.read_csv(
        root / "hidden_stratified_metrics.csv", **read_options
    )
    _assert_row_counts(
        manifest,
        partition=partitions,
        probe=probes,
        probe_exclusion=exclusions,
        fit_audit=fit_audit,
        calibration_prediction=calibration_predictions,
        calibration_metric=calibration_metrics,
        selective_curve=curves,
        hidden_prediction=hidden,
        hidden_metric=hidden_metrics,
        hidden_stratified_metric=stratified,
        filtering_evidence=filtering,
    )
    validation_fitted, validation_X_all, validation_positions = (
        _refit_uncertainty_estimators(
            expected_holdouts,
            saved.canonical,
            scientific,
            expected_feature_metadata_hash=manifest[
                "feature_metadata_hash"
            ],
        )
    )
    _assert_fit_audit(
        fit_audit,
        expected_holdouts,
        claimed_plan_hash,
        scientific,
        validation_fitted,
    )
    replayed_policies = _replay_calibration_and_policies(
        calibration_predictions,
        calibration_metrics,
        curves,
        fit_audit,
        expected_holdouts,
        scientific,
        claimed_plan_hash,
        saved.canonical["source_row_id"].astype(str).tolist(),
        plan["descriptive_context_similarity_bin_edges"],
        validation_fitted,
        validation_X_all,
        validation_positions,
    )
    if stable_hash(replayed_policies) != stable_hash(
        policy_document["policies"]
    ):
        raise ValueError("Phase 12 validation-only frozen policy replay mismatch.")
    claimed_document_hash = policy_document.pop("policy_document_hash", None)
    if (
        policy_document.get("status")
        != "frozen_before_hidden_outcome_loading"
        or policy_document.get("hidden_labels_accessed") is not False
        or policy_document.get("outer_test_labels_accessed") is not False
        or policy_document.get("plan_hash") != claimed_plan_hash
        or policy_document.get("policy_count") != len(expected_holdouts)
        or claimed_document_hash != stable_hash(policy_document)
    ):
        raise ValueError("Invalid Phase 12 frozen policy document.")
    policy_document["policy_document_hash"] = claimed_document_hash
    if manifest["policy_document_hash"] != claimed_document_hash:
        raise ValueError("Phase 12 frozen policy linkage mismatch.")
    _assert_hidden_predictions(
        hidden,
        hidden_metrics,
        filtering,
        expected_holdouts,
        replayed_policies,
        scientific,
        claimed_plan_hash,
        saved.canonical,
        expected_probed_ids,
        validation_fitted,
        validation_X_all,
        validation_positions,
    )
    expected_stratified = _stratified_hidden_metrics(
        hidden,
        probes,
        {
            record["evaluation_unit"]: record for record in replayed_policies
        },
        expected_holdouts,
    )
    if not _records_close(
        _frame_records(expected_stratified), _frame_records(stratified)
    ):
        raise ValueError("Phase 12 hidden stratified metric replay mismatch.")
    return manifest


def _resolve_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "dataset",
            "splits",
            "hidden_experiment",
            "uncertainty",
            "features",
            "base_seed",
            "output",
        },
        "Phase 12 config",
    )
    dataset = _mapping(raw["dataset"], "dataset")
    splits = _mapping(raw["splits"], "splits")
    hidden = _mapping(raw["hidden_experiment"], "hidden_experiment")
    uncertainty = _mapping(raw["uncertainty"], "uncertainty")
    output = _mapping(raw["output"], "output")
    _exact_keys(dataset, {"path"}, "dataset")
    _exact_keys(
        splits,
        {"canonical_directory", "random_seeds", "random_fractions"},
        "splits",
    )
    _exact_keys(
        hidden,
        {
            "n_condition_groups",
            "minimum_hidden_rows",
            "panel_salt",
            "validation_partition_salt",
            "requested_role_sets",
            "donor_strategies",
            "similarity_n_bits",
            "similarity_radius",
        },
        "hidden_experiment",
    )
    _exact_keys(
        uncertainty,
        {
            "coverage",
            "max_calibration_error",
            "min_retained_fraction",
            "methods",
        },
        "uncertainty",
    )
    _exact_keys(output, {"directory"}, "output")
    seeds = _unique_integers(splits["random_seeds"], "random_seeds")
    fractions = _unique_fractions(
        splits["random_fractions"], "random_fractions"
    )
    role_sets = _role_sets(hidden["requested_role_sets"])
    donor_strategies = tuple(hidden["donor_strategies"])
    if (
        donor_strategies != SUPPORTED_PROBE_DONOR_STRATEGIES
        or len(set(donor_strategies)) != len(donor_strategies)
    ):
        raise ValueError(
            "Phase 12 donor_strategies must exactly preserve "
            f"{SUPPORTED_PROBE_DONOR_STRATEGIES}."
        )
    method_configs = _mapping(uncertainty["methods"], "uncertainty.methods")
    if tuple(method_configs) != SUPPORTED_UNCERTAINTY_METHODS:
        raise ValueError(
            "Phase 12 uncertainty methods must preserve the exact supported order."
        )
    resolved_methods: dict[str, dict[str, Any]] = {}
    allowed_config_fields = {
        field.name for field in fields(UncertaintyEstimatorConfig)
    } - {"method", "coverage", "random_state"}
    for method, value in method_configs.items():
        overrides = dict(_mapping(value, f"uncertainty.methods.{method}"))
        unknown = sorted(set(overrides) - allowed_config_fields)
        if unknown:
            raise ValueError(
                f"Unknown estimator keys for {method}: {unknown}."
            )
        if "xgboost_seeds" in overrides:
            overrides["xgboost_seeds"] = tuple(overrides["xgboost_seeds"])
        UncertaintyEstimatorConfig(
            method=method,
            coverage=float(uncertainty["coverage"]),
            random_state=0,
            **overrides,
        )
        resolved_methods[method] = overrides
    coverage = _open_unit(uncertainty["coverage"], "coverage")
    max_error = _closed_unit(
        uncertainty["max_calibration_error"], "max_calibration_error"
    )
    min_retained = _open_unit(
        uncertainty["min_retained_fraction"], "min_retained_fraction"
    )
    base_seed = _integer(raw["base_seed"], "base_seed")
    return {
        "dataset_path": str(dataset["path"]),
        "canonical_split_directory": str(splits["canonical_directory"]),
        "random_seeds": seeds,
        "random_fractions": fractions,
        "hidden_experiment": {
            "n_condition_groups": _positive_integer(
                hidden["n_condition_groups"], "n_condition_groups"
            ),
            "minimum_hidden_rows": _positive_integer(
                hidden["minimum_hidden_rows"], "minimum_hidden_rows"
            ),
            "panel_salt": _nonempty_string(hidden["panel_salt"], "panel_salt"),
            "validation_partition_salt": _nonempty_string(
                hidden["validation_partition_salt"],
                "validation_partition_salt",
            ),
            "requested_role_sets": [list(values) for values in role_sets],
            "donor_strategies": list(donor_strategies),
            "similarity_n_bits": _positive_integer(
                hidden["similarity_n_bits"], "similarity_n_bits"
            ),
            "similarity_radius": _nonnegative_integer(
                hidden["similarity_radius"], "similarity_radius"
            ),
        },
        "uncertainty": {
            "coverage": coverage,
            "max_calibration_error": max_error,
            "min_retained_fraction": min_retained,
            "method_configs": resolved_methods,
        },
        "features": dict(_mapping(raw["features"], "features")),
        "base_seed": base_seed,
        "output_directory": str(output["directory"]),
    }


def _select_condition_panels(
    canonical: pd.DataFrame,
    units: Mapping[tuple[int, float], EvaluationSplitUnit],
    *,
    seeds: Sequence[int],
    fractions: Sequence[float],
    n_groups: int,
    minimum_hidden_rows: int,
    salt: str,
) -> dict[int, tuple[str, ...]]:
    by_source = canonical.set_index("source_row_id", drop=False)
    minimum_fraction = min(float(value) for value in fractions)
    panels: dict[int, tuple[str, ...]] = {}
    for seed in seeds:
        unit = units[(int(seed), minimum_fraction)]
        rows = by_source.loc[list(unit.train_source_ids)]
        counts = rows.groupby("canonical_condition_key").size()
        eligible = [
            str(key)
            for key, count in counts.items()
            if int(count) >= minimum_hidden_rows
        ]
        ordered = sorted(
            eligible,
            key=lambda key: (stable_hash([salt, int(seed), key]), key),
        )
        if len(ordered) < n_groups:
            raise ValueError(
                "Insufficient chemistry-only hidden condition groups at the "
                f"minimum fraction for seed={seed}: required={n_groups}, "
                f"observed={len(ordered)}."
            )
        panels[int(seed)] = tuple(ordered[:n_groups])
    return panels


def _build_holdout_unit(
    unit: EvaluationSplitUnit,
    canonical: pd.DataFrame,
    *,
    held_condition_key: str,
    validation_salt: str,
) -> CalibrationHoldoutUnit:
    if unit.seed is None or unit.train_fraction is None:
        raise ValueError("Phase 12 requires saved random split units.")
    by_source = canonical.set_index("source_row_id", drop=False)
    train = by_source.loc[list(unit.train_source_ids)]
    hidden_ids = tuple(
        sorted(
            train.loc[
                train["canonical_condition_key"].eq(held_condition_key),
                "source_row_id",
            ].astype(str)
        )
    )
    if not hidden_ids:
        raise ValueError("Held condition has no hidden rows in training subset.")
    teacher_ids = tuple(sorted(set(unit.train_source_ids) - set(hidden_ids)))
    validation = by_source.loc[list(unit.validation_source_ids)]
    held_validation_ids = tuple(
        sorted(
            validation.loc[
                validation["canonical_condition_key"].eq(
                    held_condition_key
                ),
                "source_row_id",
            ].astype(str)
        )
    )
    validation = validation.loc[
        validation["canonical_condition_key"].ne(held_condition_key)
    ]
    group_keys = sorted(
        validation["canonical_condition_key"].astype(str).unique(),
        key=lambda key: (stable_hash([validation_salt, int(unit.seed), key]), key),
    )
    if len(group_keys) < 2:
        raise ValueError("Insufficient validation condition groups.")
    split = len(group_keys) // 2
    interval_groups = set(group_keys[:split])
    policy_groups = set(group_keys[split:])
    interval_ids = tuple(
        sorted(
            validation.loc[
                validation["canonical_condition_key"].isin(interval_groups),
                "source_row_id",
            ].astype(str)
        )
    )
    policy_ids = tuple(
        sorted(
            validation.loc[
                validation["canonical_condition_key"].isin(policy_groups),
                "source_row_id",
            ].astype(str)
        )
    )
    if not teacher_ids or not interval_ids or not policy_ids:
        raise ValueError("Phase 12 partition contains an empty scientific role.")
    memberships = [
        set(teacher_ids),
        set(hidden_ids),
        set(interval_ids),
        set(policy_ids),
        set(held_validation_ids),
        set(unit.test_source_ids),
        set(unit.excluded_source_ids),
    ]
    for index, left in enumerate(memberships):
        if any(left & right for right in memberships[index + 1 :]):
            raise ValueError("Phase 12 holdout partitions overlap.")
    source_hash = stable_hash(
        {
            "teacher_fit": list(teacher_ids),
            "hidden": list(hidden_ids),
            "interval_calibration": list(interval_ids),
            "policy_validation": list(policy_ids),
            "held_condition_validation_excluded": list(
                held_validation_ids
            ),
            "outer_test": list(unit.test_source_ids),
            "nested_excluded": list(unit.excluded_source_ids),
        }
    )
    short_group = stable_hash(held_condition_key)[:12]
    name = f"{unit.evaluation_unit}:condition={short_group}"
    return CalibrationHoldoutUnit(
        evaluation_unit=name,
        base_evaluation_unit=unit.evaluation_unit,
        seed=int(unit.seed),
        train_fraction=float(unit.train_fraction),
        held_condition_key=held_condition_key,
        teacher_fit_source_ids=teacher_ids,
        hidden_source_ids=hidden_ids,
        interval_calibration_source_ids=interval_ids,
        policy_validation_source_ids=policy_ids,
        held_condition_validation_excluded_source_ids=held_validation_ids,
        outer_test_source_ids=unit.test_source_ids,
        nested_excluded_source_ids=unit.excluded_source_ids,
        exact_split_hash=unit.exact_split_hash,
        aggregate_split_hash=unit.aggregate_assignment_hash,
        source_id_split_hash=source_hash,
    )


def _partition_rows(
    units: Sequence[CalibrationHoldoutUnit],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_memberships = (
        ("teacher_fit", "teacher_fit_source_ids"),
        ("hidden_measured_evaluation", "hidden_source_ids"),
        ("interval_calibration", "interval_calibration_source_ids"),
        ("policy_validation", "policy_validation_source_ids"),
        (
            "held_condition_validation_excluded",
            "held_condition_validation_excluded_source_ids",
        ),
        ("outer_test_redacted", "outer_test_source_ids"),
        ("nested_training_excluded", "nested_excluded_source_ids"),
    )
    for unit in units:
        for role, field_name in role_memberships:
            for source_id in getattr(unit, field_name):
                payload = {
                    "evaluation_unit": unit.evaluation_unit,
                    "source_row_id": source_id,
                    "data_role": role,
                    "held_condition_key": unit.held_condition_key,
                    "seed": unit.seed,
                    "train_fraction": unit.train_fraction,
                }
                payload["partition_id"] = stable_hash(payload)
                rows.append(payload)
    return sorted(rows, key=lambda row: row["partition_id"])


def _build_probe_plan(
    units: Sequence[CalibrationHoldoutUnit],
    by_source: pd.DataFrame,
    *,
    role_sets: Sequence[Sequence[str]],
    donor_strategies: Sequence[str],
    seed: int,
    similarity_n_bits: int,
    similarity_radius: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probes: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for unit in units:
        first_probe_index = len(probes)
        train = by_source.loc[list(unit.teacher_fit_source_ids)].copy()
        hidden = by_source.loc[list(unit.hidden_source_ids)].copy()
        hidden = hidden.drop(columns=["yield"], errors="ignore")
        context = prepare_hidden_condition_probe_context(
            train,
            hidden,
            similarity_n_bits=similarity_n_bits,
            similarity_radius=similarity_radius,
        )
        for roles in role_sets:
            for donor_strategy in donor_strategies:
                result = build_hidden_condition_probes_from_context(
                    context,
                    requested_roles=roles,
                    donor_strategy=donor_strategy,
                    seed=_derived_seed(
                        seed,
                        unit.evaluation_unit,
                        "|".join(roles),
                        donor_strategy,
                    ),
                )
                for record in result.probes.to_dict("records"):
                    payload = {
                        "evaluation_unit": unit.evaluation_unit,
                        **record,
                        "hidden_labels_accessed": False,
                        "outer_test_data_used": False,
                    }
                    payload["probe_id"] = stable_hash(payload)
                    probes.append(payload)
                for record in result.exclusions.to_dict("records"):
                    payload = {
                        "evaluation_unit": unit.evaluation_unit,
                        **record,
                        "hidden_labels_accessed": False,
                        "outer_test_data_used": False,
                    }
                    payload["exclusion_id"] = stable_hash(payload)
                    exclusions.append(payload)
        unit_probed_ids = {
            str(record["hidden_row_id"])
            for record in probes[first_probe_index:]
        }
        for hidden_row_id in sorted(
            set(unit.hidden_source_ids) - unit_probed_ids
        ):
            payload = {
                "evaluation_unit": unit.evaluation_unit,
                "hidden_row_id": hidden_row_id,
                "held_condition_key": unit.held_condition_key,
                "requested_roles": "any_feasible_role_set",
                "donor_strategy": "any_configured_strategy",
                "exclusion_reason": (
                    "no_exact_transfer_probe_under_any_configured_policy"
                ),
                "hidden_labels_accessed": False,
                "outer_test_data_used": False,
            }
            payload["exclusion_id"] = stable_hash(payload)
            exclusions.append(payload)
    return (
        sorted(probes, key=lambda row: row["probe_id"]),
        sorted(exclusions, key=lambda row: row["exclusion_id"]),
    )


def _probed_source_ids(
    units: Sequence[CalibrationHoldoutUnit],
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Return exact-reconstructed hidden targets, never the full holdout."""
    records_by_unit: dict[str, set[str]] = {
        unit.evaluation_unit: set() for unit in units
    }
    hidden_by_unit = {
        unit.evaluation_unit: set(unit.hidden_source_ids) for unit in units
    }
    for record in probes:
        unit_name = str(record["evaluation_unit"])
        if unit_name not in records_by_unit:
            raise ValueError("Probe references an unknown calibration unit.")
        source_id = str(record["hidden_row_id"])
        if source_id not in hidden_by_unit[unit_name]:
            raise ValueError("Probe references a non-hidden target.")
        records_by_unit[unit_name].add(source_id)
    result = {
        unit_name: tuple(sorted(source_ids))
        for unit_name, source_ids in records_by_unit.items()
    }
    if any(not source_ids for source_ids in result.values()):
        raise ValueError(
            "Every hidden-condition unit must reconstruct at least one target."
        )
    return result


def _prediction_rows(
    unit: CalibrationHoldoutUnit,
    method: str,
    partition: str,
    source_ids: Sequence[str],
    y_true: np.ndarray,
    prediction: Any,
    method_config_hash: str,
    plan_hash: str,
) -> list[dict[str, Any]]:
    return [
        {
            "evaluation_unit": unit.evaluation_unit,
            "seed": unit.seed,
            "train_fraction": unit.train_fraction,
            "held_condition_key": unit.held_condition_key,
            "source_row_id": source_id,
            "data_role": partition,
            "method": method,
            "method_config_hash": method_config_hash,
            "estimator_audit_hash": prediction.estimator_audit_hash,
            "point_prediction": float(prediction.point[index]),
            "raw_uncertainty_score": float(prediction.raw_score[index]),
            "raw_lower": float(prediction.raw_lower[index]),
            "raw_upper": float(prediction.raw_upper[index]),
            "interval_lower": float(prediction.interval_lower[index]),
            "interval_upper": float(prediction.interval_upper[index]),
            "calibrated_uncertainty": float(
                prediction.calibrated_uncertainty[index]
            ),
            "measured_yield": float(y_true[index]),
            "absolute_error": float(
                abs(prediction.point[index] - y_true[index])
            ),
            "interval_covered": bool(
                prediction.interval_lower[index]
                <= y_true[index]
                <= prediction.interval_upper[index]
            ),
            "prediction_hash": prediction.prediction_hash,
            "hidden_outcomes_present": False,
            "outer_test_outcomes_present": False,
            "plan_hash": plan_hash,
        }
        for index, source_id in enumerate(source_ids)
    ]


def _assert_prediction_frame_matches(
    rows: pd.DataFrame,
    expected: Any,
    source_ids: Sequence[str],
    measured_yields: Sequence[float],
) -> None:
    """Ground every persisted prediction field in a deterministic refit."""
    if tuple(rows["source_row_id"].astype(str)) != tuple(source_ids):
        raise ValueError("Phase 12 prediction source membership mismatch.")
    expected_arrays = {
        "point_prediction": expected.point,
        "raw_uncertainty_score": expected.raw_score,
        "raw_lower": expected.raw_lower,
        "raw_upper": expected.raw_upper,
        "interval_lower": expected.interval_lower,
        "interval_upper": expected.interval_upper,
        "calibrated_uncertainty": expected.calibrated_uncertainty,
    }
    for column, values in expected_arrays.items():
        if not np.allclose(
            rows[column].to_numpy(float),
            np.asarray(values, dtype=float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                f"Phase 12 refitted prediction mismatch: {column}."
            )
    measured = np.asarray(measured_yields, dtype=float)
    point = np.asarray(expected.point, dtype=float)
    covered = (
        (measured >= np.asarray(expected.interval_lower, dtype=float))
        & (measured <= np.asarray(expected.interval_upper, dtype=float))
    )
    if (
        set(rows["prediction_hash"]) != {expected.prediction_hash}
        or set(rows["estimator_audit_hash"])
        != {expected.estimator_audit_hash}
        or not np.allclose(
            rows["absolute_error"].to_numpy(float),
            np.abs(point - measured),
            rtol=0.0,
            atol=1e-10,
        )
        or not np.array_equal(
            rows["interval_covered"].map(_strict_bool).to_numpy(),
            covered,
        )
    ):
        raise ValueError("Phase 12 refitted prediction provenance mismatch.")


def _replay_calibration_and_policies(
    predictions: pd.DataFrame,
    metrics_table: pd.DataFrame,
    curves_table: pd.DataFrame,
    fit_audit: pd.DataFrame,
    units: Sequence[CalibrationHoldoutUnit],
    scientific: Mapping[str, Any],
    plan_hash: str,
    canonical_source_ids: Sequence[str],
    context_edges_by_unit: Mapping[str, Sequence[float]],
    expected_estimators: Mapping[
        tuple[str, str], FittedUncertaintyEstimator
    ],
    X_all: np.ndarray,
    positions: Mapping[str, int],
) -> list[dict[str, Any]]:
    expected_prediction_count = sum(
        (
            len(unit.interval_calibration_source_ids)
            + len(unit.policy_validation_source_ids)
        )
        * len(SUPPORTED_UNCERTAINTY_METHODS)
        for unit in units
    )
    if (
        len(predictions) != expected_prediction_count
        or set(predictions["plan_hash"]) != {plan_hash}
        or predictions["hidden_outcomes_present"].map(_strict_bool).any()
        or predictions["outer_test_outcomes_present"].map(_strict_bool).any()
    ):
        raise ValueError("Phase 12 calibration prediction coverage mismatch.")
    fit_by_key = fit_audit.set_index(["evaluation_unit", "method"])
    records: list[dict[str, Any]] = []
    for unit in units:
        allowed_ids = tuple(
            sorted(
                unit.interval_calibration_source_ids
                + unit.policy_validation_source_ids
            )
        )
        grounded_outcomes = _read_allowed_outcomes(
            scientific["dataset_path"], canonical_source_ids, allowed_ids
        )
        candidates: list[ValidationCalibrationCandidate] = []
        prediction_by_method: dict[str, pd.DataFrame] = {}
        for method in SUPPORTED_UNCERTAINTY_METHODS:
            method_rows = predictions.loc[
                predictions["evaluation_unit"].eq(unit.evaluation_unit)
                & predictions["method"].eq(method)
            ]
            interval_rows = method_rows.loc[
                method_rows["data_role"].eq("interval_calibration")
            ].sort_values("source_row_id", kind="mergesort")
            policy_rows = method_rows.loc[
                method_rows["data_role"].eq("policy_validation")
            ].sort_values("source_row_id", kind="mergesort")
            if (
                tuple(interval_rows["source_row_id"])
                != unit.interval_calibration_source_ids
                or tuple(policy_rows["source_row_id"])
                != unit.policy_validation_source_ids
                or set(method_rows["method_config_hash"])
                != {fit_by_key.loc[(unit.evaluation_unit, method), "config_hash"]}
                or set(method_rows["estimator_audit_hash"])
                != {fit_by_key.loc[(unit.evaluation_unit, method), "audit_hash"]}
            ):
                raise ValueError("Phase 12 calibration row provenance mismatch.")
            for row in method_rows.to_dict("records"):
                if not math.isclose(
                    float(row["measured_yield"]),
                    grounded_outcomes[str(row["source_row_id"])],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Phase 12 calibration outcome is not canonical-grounded."
                    )
            estimator = expected_estimators[
                (unit.evaluation_unit, method)
            ]
            interval_expected = estimator.predict(
                X_all[
                    np.asarray(
                        [
                            positions[source_id]
                            for source_id in unit.interval_calibration_source_ids
                        ],
                        dtype=int,
                    )
                ],
                source_ids=unit.interval_calibration_source_ids,
            )
            policy_expected = estimator.predict(
                X_all[
                    np.asarray(
                        [
                            positions[source_id]
                            for source_id in unit.policy_validation_source_ids
                        ],
                        dtype=int,
                    )
                ],
                source_ids=unit.policy_validation_source_ids,
            )
            _assert_prediction_frame_matches(
                interval_rows,
                interval_expected,
                unit.interval_calibration_source_ids,
                [
                    grounded_outcomes[source_id]
                    for source_id in unit.interval_calibration_source_ids
                ],
            )
            _assert_prediction_frame_matches(
                policy_rows,
                policy_expected,
                unit.policy_validation_source_ids,
                [
                    grounded_outcomes[source_id]
                    for source_id in unit.policy_validation_source_ids
                ],
            )
            replayed_metrics = interval_calibration_metrics(
                policy_rows["measured_yield"].to_numpy(float),
                policy_rows["point_prediction"].to_numpy(float),
                policy_rows["interval_lower"].to_numpy(float),
                policy_rows["interval_upper"].to_numpy(float),
                policy_rows["calibrated_uncertainty"].to_numpy(float),
                nominal_coverage=scientific["uncertainty"]["coverage"],
            )
            curve = build_selective_prediction_curve(
                policy_rows["measured_yield"].to_numpy(float),
                policy_rows["point_prediction"].to_numpy(float),
                policy_rows["calibrated_uncertainty"].to_numpy(float),
                tuple(policy_rows["source_row_id"]),
            )
            observed_metric = metrics_table.loc[
                metrics_table["evaluation_unit"].eq(unit.evaluation_unit)
                & metrics_table["method"].eq(method)
            ]
            observed_curve = curves_table.loc[
                curves_table["evaluation_unit"].eq(unit.evaluation_unit)
                & curves_table["method"].eq(method)
            ].sort_values("uncertainty_cutoff", kind="mergesort")
            if (
                len(observed_metric) != 1
                or not _mapping_close(
                    asdict(replayed_metrics),
                    observed_metric.iloc[0].to_dict(),
                )
                or not _records_close(
                    [asdict(point) for point in curve],
                    observed_curve.to_dict("records"),
                )
            ):
                raise ValueError("Phase 12 calibration metric replay mismatch.")
            candidates.append(
                ValidationCalibrationCandidate(
                    method_id=method,
                    method_config_hash=str(
                        fit_by_key.loc[
                            (unit.evaluation_unit, method), "config_hash"
                        ]
                    ),
                    metrics=replayed_metrics,
                    selective_curve=curve,
                    data_roles=CalibrationDataRoles(),
                )
            )
            prediction_by_method[method] = policy_rows
        selection = select_validation_policy(
            candidates,
            nominal_coverage=scientific["uncertainty"]["coverage"],
            max_calibration_error=scientific["uncertainty"][
                "max_calibration_error"
            ],
            min_retained_fraction=scientific["uncertainty"][
                "min_retained_fraction"
            ],
        )
        selected_rows = prediction_by_method[selection.method_id]
        payload = {
            "schema_version": PHASE12_SCHEMA_VERSION,
            "status": "frozen_before_hidden_outcome_loading",
            "evaluation_unit": unit.evaluation_unit,
            "holdout_unit_hash": unit.record["holdout_unit_hash"],
            "selection": asdict(selection),
            "estimator_audit_hash": str(
                fit_by_key.loc[
                    (unit.evaluation_unit, selection.method_id), "audit_hash"
                ]
            ),
            "validation_prediction_hash": str(
                selected_rows["prediction_hash"].iloc[0]
            ),
            "uncertainty_bin_edges": _tertile_edges(
                selected_rows["calibrated_uncertainty"].to_numpy(float)
            ),
            "support_distance_bin_edges": _tertile_edges(
                _policy_support_distances(unit, predictions, fit_audit)
            ),
            "context_similarity_bin_edges": list(
                context_edges_by_unit[unit.evaluation_unit]
            ),
            "context_similarity_bins_are_descriptive": True,
            "plan_hash": plan_hash,
            "selection_data_roles": ["policy_validation"],
            "hidden_labels_accessed": False,
            "outer_test_labels_accessed": False,
        }
        payload["frozen_policy_hash"] = stable_hash(payload)
        records.append(payload)
    return records


def _assert_hidden_predictions(
    hidden: pd.DataFrame,
    metrics_table: pd.DataFrame,
    filtering: pd.DataFrame,
    units: Sequence[CalibrationHoldoutUnit],
    policies: Sequence[Mapping[str, Any]],
    scientific: Mapping[str, Any],
    plan_hash: str,
    canonical: pd.DataFrame,
    probed_ids_by_unit: Mapping[str, Sequence[str]],
    expected_estimators: Mapping[
        tuple[str, str], FittedUncertaintyEstimator
    ],
    X_all: np.ndarray,
    positions: Mapping[str, int],
) -> None:
    policies_by_unit = {
        record["evaluation_unit"]: record for record in policies
    }
    expected_count = sum(
        len(probed_ids_by_unit[unit.evaluation_unit])
        * len(SUPPORTED_UNCERTAINTY_METHODS)
        for unit in units
    )
    if (
        len(hidden) != expected_count
        or set(hidden["plan_hash"]) != {plan_hash}
        or hidden["eligible_for_training"].map(_strict_bool).any()
        or hidden["outer_test_data_used"].map(_strict_bool).any()
        or not hidden["hidden_outcome_data_role"]
        .eq("hidden_measured_evaluation")
        .all()
    ):
        raise ValueError("Phase 12 hidden prediction coverage mismatch.")
    for unit in units:
        policy = policies_by_unit[unit.evaluation_unit]
        evaluated_source_ids = tuple(
            probed_ids_by_unit[unit.evaluation_unit]
        )
        grounded_outcomes = _read_allowed_outcomes(
            scientific["dataset_path"],
            canonical["source_row_id"].astype(str).tolist(),
            evaluated_source_ids,
        )
        canonical_by_source = canonical.set_index("source_row_id")
        hidden_X = X_all[
            np.asarray(
                [positions[source_id] for source_id in evaluated_source_ids],
                dtype=int,
            )
        ]
        expected_support_distance = _nearest_cosine_distance(
            hidden_X,
            X_all[
                np.asarray(
                    [
                        positions[source_id]
                        for source_id in unit.teacher_fit_source_ids
                    ],
                    dtype=int,
                )
            ],
        )
        for method in SUPPORTED_UNCERTAINTY_METHODS:
            rows = hidden.loc[
                hidden["evaluation_unit"].eq(unit.evaluation_unit)
                & hidden["method"].eq(method)
            ].sort_values("hidden_row_id", kind="mergesort")
            if (
                tuple(rows["hidden_row_id"]) != evaluated_source_ids
                or rows["canonical_reaction_key"].duplicated().any()
                or set(rows["frozen_policy_hash"])
                != {policy["frozen_policy_hash"]}
            ):
                raise ValueError("Phase 12 hidden target membership mismatch.")
            for row in rows.to_dict("records"):
                source_id = str(row["hidden_row_id"])
                if (
                    not math.isclose(
                        float(row["measured_yield"]),
                        grounded_outcomes[source_id],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or row["canonical_reaction_key"]
                    != canonical_by_source.loc[
                        source_id, "canonical_reaction_key"
                    ]
                ):
                    raise ValueError(
                        "Phase 12 hidden outcome or identity is not canonical-grounded."
                    )
            expected_estimator = expected_estimators[
                (unit.evaluation_unit, method)
            ]
            expected_prediction = expected_estimator.predict(
                hidden_X, source_ids=evaluated_source_ids
            )
            expected_y = np.asarray(
                [
                    grounded_outcomes[source_id]
                    for source_id in evaluated_source_ids
                ],
                dtype=float,
            )
            expected_selected = method == policy["selection"]["method_id"]
            expected_covered = (
                expected_y >= expected_prediction.interval_lower
            ) & (expected_y <= expected_prediction.interval_upper)
            if (
                set(rows["method_config_hash"])
                != {expected_estimator.config.config_hash}
                or set(rows["estimator_audit_hash"])
                != {expected_estimator.audit.audit_hash}
                or not np.allclose(
                    rows["point_prediction"].to_numpy(float),
                    expected_prediction.point,
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.allclose(
                    rows["interval_lower"].to_numpy(float),
                    expected_prediction.interval_lower,
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.allclose(
                    rows["interval_upper"].to_numpy(float),
                    expected_prediction.interval_upper,
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.allclose(
                    rows["calibrated_uncertainty"].to_numpy(float),
                    expected_prediction.calibrated_uncertainty,
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.allclose(
                    rows["nearest_visible_support_distance"].to_numpy(float),
                    expected_support_distance,
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.allclose(
                    rows["signed_error"].to_numpy(float),
                    expected_prediction.point - expected_y,
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.allclose(
                    rows["absolute_error"].to_numpy(float),
                    np.abs(expected_prediction.point - expected_y),
                    rtol=0.0,
                    atol=1e-10,
                )
                or not np.array_equal(
                    rows["interval_covered"].map(_strict_bool).to_numpy(),
                    expected_covered,
                )
                or not rows["selected_by_validation"]
                .map(_strict_bool)
                .eq(expected_selected)
                .all()
                or not np.array_equal(
                    rows["accepted_by_frozen_filter"]
                    .map(_strict_bool)
                    .to_numpy(),
                    (
                        expected_selected
                        & (
                            expected_prediction.calibrated_uncertainty
                            <= float(
                                policy["selection"]["uncertainty_cutoff"]
                            )
                        )
                    ),
                )
            ):
                raise ValueError(
                    "Phase 12 hidden prediction does not match deterministic refit."
                )
            replayed = interval_calibration_metrics(
                rows["measured_yield"].to_numpy(float),
                rows["point_prediction"].to_numpy(float),
                rows["interval_lower"].to_numpy(float),
                rows["interval_upper"].to_numpy(float),
                rows["calibrated_uncertainty"].to_numpy(float),
                nominal_coverage=scientific["uncertainty"]["coverage"],
            )
            observed = metrics_table.loc[
                metrics_table["evaluation_unit"].eq(unit.evaluation_unit)
                & metrics_table["method"].eq(method)
            ]
            if (
                len(observed) != 1
                or not _mapping_close(
                    asdict(replayed), observed.iloc[0].to_dict()
                )
                or not math.isclose(
                    float(observed["rmse"].iloc[0]),
                    float(
                        np.sqrt(
                            np.mean(
                                (
                                    rows["measured_yield"].to_numpy(float)
                                    - rows["point_prediction"].to_numpy(float)
                                )
                                ** 2
                            )
                        )
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                or int(observed["unique_target_count"].iloc[0])
                != len(evaluated_source_ids)
                or int(
                    observed["hidden_target_denominator"].iloc[0]
                )
                != len(unit.hidden_source_ids)
                or int(
                    observed[
                        "exactly_reconstructed_target_count"
                    ].iloc[0]
                )
                != len(evaluated_source_ids)
                or not math.isclose(
                    float(observed["reconstruction_coverage"].iloc[0]),
                    len(evaluated_source_ids) / len(unit.hidden_source_ids),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or observed["evaluation_scope"].iloc[0]
                != "exactly_reconstructed_hidden_targets_only"
            ):
                raise ValueError("Phase 12 hidden metric replay mismatch.")
        selected = hidden.loc[
            hidden["evaluation_unit"].eq(unit.evaluation_unit)
            & hidden["method"].eq(policy["selection"]["method_id"])
        ].sort_values("hidden_row_id", kind="mergesort")
        evidence = filtering.loc[
            filtering["evaluation_unit"].eq(unit.evaluation_unit)
        ].sort_values("hidden_row_id", kind="mergesort")
        expected_evidence_hash = stable_hash(
            {
                "estimator_audit_hash": policy["estimator_audit_hash"],
                "validation_prediction_hash": policy[
                    "validation_prediction_hash"
                ],
                "selection": policy["selection"],
            }
        )
        expected_acceptance = (
            selected["calibrated_uncertainty"].to_numpy(float)
            <= float(policy["selection"]["uncertainty_cutoff"])
        )
        if (
            tuple(evidence["hidden_row_id"]) != evaluated_source_ids
            or set(evidence["method"])
            != {policy["selection"]["method_id"]}
            or set(evidence["frozen_policy_hash"])
            != {policy["frozen_policy_hash"]}
            or set(evidence["plan_hash"]) != {plan_hash}
            or set(evidence["filter_rule"])
            != {
                "calibrated_uncertainty <= validation_selected_cutoff"
            }
            or set(evidence["calibration_evidence_hash"])
            != {expected_evidence_hash}
            or evidence["eligible_for_training"].map(_strict_bool).any()
            or not np.allclose(
                evidence["calibrated_uncertainty"].to_numpy(float),
                selected["calibrated_uncertainty"].to_numpy(float),
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                evidence["frozen_uncertainty_cutoff"].to_numpy(float),
                float(policy["selection"]["uncertainty_cutoff"]),
                rtol=0.0,
                atol=1e-12,
            )
            or not np.array_equal(
                evidence["accepted_by_frozen_filter"]
                .map(_strict_bool)
                .to_numpy(),
                expected_acceptance,
            )
        ):
            raise ValueError("Phase 12 frozen filtering linkage mismatch.")


def _assert_partition_semantics(
    rows: pd.DataFrame, units: Sequence[CalibrationHoldoutUnit]
) -> None:
    expected = pd.DataFrame(_partition_rows(units))
    if stable_hash(_frame_records(rows)) != stable_hash(
        _frame_records(expected)
    ):
        raise ValueError("Phase 12 partition semantics mismatch.")


def _assert_fit_audit(
    audit: pd.DataFrame,
    units: Sequence[CalibrationHoldoutUnit],
    plan_hash: str,
    scientific: Mapping[str, Any],
    expected_estimators: Mapping[
        tuple[str, str], FittedUncertaintyEstimator
    ],
) -> None:
    expected_keys = {
        (unit.evaluation_unit, method)
        for unit in units
        for method in SUPPORTED_UNCERTAINTY_METHODS
    }
    observed_keys = set(
        audit[["evaluation_unit", "method"]].itertuples(
            index=False, name=None
        )
    )
    if (
        observed_keys != expected_keys
        or set(audit["plan_hash"]) != {plan_hash}
        or audit["hidden_or_test_fit_contamination"].map(_strict_bool).any()
        or audit["audit_hash"].duplicated().any()
    ):
        raise ValueError("Phase 12 fit audit coverage mismatch.")
    by_unit = {unit.evaluation_unit: unit for unit in units}
    for row in audit.to_dict("records"):
        unit = by_unit[row["evaluation_unit"]]
        if (
            row["train_source_id_hash"]
            != stable_hash(list(unit.teacher_fit_source_ids))
            or row["calibration_source_id_hash"]
            != stable_hash(list(unit.interval_calibration_source_ids))
            or row["hidden_source_id_hash"]
            != stable_hash(list(unit.hidden_source_ids))
            or row["outer_test_source_id_hash"]
            != stable_hash(list(unit.outer_test_source_ids))
            or int(row["train_size"]) != len(unit.teacher_fit_source_ids)
            or int(row["calibration_size"])
            != len(unit.interval_calibration_source_ids)
        ):
            raise ValueError("Phase 12 fit membership provenance mismatch.")
        tuple_fields = (
            "model_names",
            "model_seeds",
            "bootstrap_sample_hashes",
        )
        parsed = dict(row)
        for field_name in tuple_fields:
            try:
                parsed[field_name] = tuple(
                    ast.literal_eval(str(parsed[field_name]))
                )
            except (SyntaxError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid serialized estimator audit field {field_name}."
                ) from exc
        audit_fields = {field.name for field in fields(EstimatorAudit)}
        reconstructed = EstimatorAudit(
            **{field_name: parsed[field_name] for field_name in audit_fields}
        )
        expected_audit = expected_estimators[
            (unit.evaluation_unit, str(row["method"]))
        ].audit
        if (
            reconstructed.audit_hash != row["audit_hash"]
            or reconstructed.audit_hash != expected_audit.audit_hash
            or not _mapping_close(
                asdict(expected_audit), asdict(reconstructed)
            )
        ):
            raise ValueError("Phase 12 estimator audit hash mismatch.")
        try:
            resolved = json.loads(str(row["resolved_config"]))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid resolved uncertainty estimator config.") from exc
        if "xgboost_seeds" in resolved:
            resolved["xgboost_seeds"] = tuple(resolved["xgboost_seeds"])
        resolved_config = UncertaintyEstimatorConfig(**resolved)
        expected_config = _estimator_config(
            str(row["method"]),
            scientific["uncertainty"]["method_configs"][str(row["method"])],
            coverage=scientific["uncertainty"]["coverage"],
            base_seed=scientific["base_seed"],
            evaluation_unit=unit.evaluation_unit,
        )
        if (
            resolved_config.config_hash != row["config_hash"]
            or resolved_config.method != row["method"]
            or resolved_config.config_hash != expected_config.config_hash
        ):
            raise ValueError("Phase 12 resolved estimator config mismatch.")


def _stratified_hidden_metrics(
    hidden: pd.DataFrame,
    probes: pd.DataFrame,
    policies: Mapping[str, Mapping[str, Any]],
    units: Sequence[CalibrationHoldoutUnit],
) -> pd.DataFrame:
    if hidden.empty:
        return pd.DataFrame(
            columns=[
                "evaluation_unit",
                "method",
                "stratifier",
                "level",
                "unique_target_count",
                "rmse",
                "mae",
                "median_absolute_error",
                "mean_uncertainty",
            ]
        )
    selected = hidden.loc[hidden["selected_by_validation"].map(_strict_bool)].copy()
    units_by_name = {unit.evaluation_unit: unit for unit in units}
    rows: list[dict[str, Any]] = []
    for unit_name, unit_rows in selected.groupby("evaluation_unit", sort=False):
        policy = policies[unit_name]
        uncertainty_edges = policy["uncertainty_bin_edges"]
        support_edges = policy["support_distance_bin_edges"]
        unit_rows["uncertainty_bin"] = [
            _bin_label(value, uncertainty_edges)
            for value in unit_rows["calibrated_uncertainty"].to_numpy(float)
        ]
        unit_rows["support_distance_bin"] = [
            _bin_label(value, support_edges)
            for value in unit_rows[
                "nearest_visible_support_distance"
            ].to_numpy(float)
        ]
        provenance = probes.loc[
            probes["evaluation_unit"].eq(unit_name)
        ][
            [
                "hidden_row_id",
                "requested_roles",
                "donor_strategy",
                "substrate_similarity",
            ]
        ].copy()
        if not provenance.empty:
            context_edges = policy["context_similarity_bin_edges"]
            provenance["context_similarity_bin"] = [
                _bin_label(value, context_edges)
                for value in provenance["substrate_similarity"].to_numpy(float)
            ]
        joined = provenance.merge(
            unit_rows,
            on="hidden_row_id",
            how="inner",
            validate="many_to_one",
        )
        dimensions = {
            "transferred_role": (joined, "requested_roles"),
            "donor_strategy": (joined, "donor_strategy"),
            "context_similarity": (joined, "context_similarity_bin"),
            "calibrated_uncertainty": (unit_rows, "uncertainty_bin"),
            "training_fraction": (unit_rows, "train_fraction"),
            "support_distance": (unit_rows, "support_distance_bin"),
        }
        for stratifier, (frame, column) in dimensions.items():
            if frame.empty or column not in frame:
                continue
            for level, group in frame.groupby(column, sort=True, dropna=False):
                unique = group.drop_duplicates("hidden_row_id")
                unit = units_by_name[unit_name]
                errors = unique["point_prediction"].to_numpy(float) - unique[
                    "measured_yield"
                ].to_numpy(float)
                rows.append(
                    {
                        "evaluation_unit": unit_name,
                        "method": policy["selection"]["method_id"],
                        "stratifier": stratifier,
                        "level": str(level),
                        "unique_target_count": len(unique),
                        "unit_target_count": len(unique),
                        "unique_source_target_count": int(
                            unique["hidden_row_id"].nunique()
                        ),
                        "rmse": float(np.sqrt(np.mean(errors**2))),
                        "mae": float(np.mean(np.abs(errors))),
                        "median_absolute_error": float(
                            np.median(np.abs(errors))
                        ),
                        "mean_uncertainty": float(
                            unique["calibrated_uncertainty"].mean()
                        ),
                        "duplicate_provenance_count": len(group) - len(unique),
                        "hidden_target_denominator": len(
                            unit.hidden_source_ids
                        ),
                        "exactly_reconstructed_target_count": len(unit_rows),
                        "reconstruction_coverage": (
                            len(unit_rows) / len(unit.hidden_source_ids)
                        ),
                        "inferential_unit": "unique_hidden_target",
                        "interpretation": (
                            "descriptive transfer-feasibility cohort"
                            if stratifier
                            in {
                                "transferred_role",
                                "donor_strategy",
                                "context_similarity",
                            }
                            else "descriptive selected-method error cohort"
                        ),
                        "hidden_outcomes_used_for_selection": False,
                    }
                )
    selected["unit_target_id"] = (
        selected["evaluation_unit"].astype(str)
        + "::"
        + selected["hidden_row_id"].astype(str)
    )
    for fraction, group in selected.groupby(
        "train_fraction", sort=True, dropna=False
    ):
        unique = group.drop_duplicates("unit_target_id")
        errors = unique["point_prediction"].to_numpy(float) - unique[
            "measured_yield"
        ].to_numpy(float)
        denominator = sum(
            len(unit.hidden_source_ids)
            for unit in units
            if math.isclose(
                float(unit.train_fraction),
                float(fraction),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        rows.append(
            {
                "evaluation_unit": "aggregate",
                "method": "validation_selected_per_unit",
                "stratifier": "training_fraction",
                "level": str(fraction),
                "unique_target_count": len(unique),
                "unit_target_count": len(unique),
                "unique_source_target_count": int(
                    unique["hidden_row_id"].nunique()
                ),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "median_absolute_error": float(
                    np.median(np.abs(errors))
                ),
                "mean_uncertainty": float(
                    unique["calibrated_uncertainty"].mean()
                ),
                "duplicate_provenance_count": 0,
                "hidden_target_denominator": denominator,
                "exactly_reconstructed_target_count": len(unique),
                "reconstruction_coverage": len(unique) / denominator,
                "inferential_unit": "evaluation_unit_hidden_target",
                "interpretation": (
                    "descriptive aggregate; repeated source IDs across units "
                    "are not treated as independent"
                ),
                "hidden_outcomes_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_unit", "stratifier", "level"],
        kind="mergesort",
    )


def _refit_uncertainty_estimators(
    units: Sequence[CalibrationHoldoutUnit],
    canonical: pd.DataFrame,
    scientific: Mapping[str, Any],
    *,
    expected_feature_metadata_hash: str,
) -> tuple[
    dict[tuple[str, str], FittedUncertaintyEstimator],
    np.ndarray,
    dict[str, int],
]:
    """Refit declared models so self-consistent forged artifacts cannot pass."""
    identity_frame = canonical.copy()
    identity_frame["yield"] = 0.0
    feature_config = resolve_corrected_feature_config(
        scientific["features"], required_kind="bh_role_separated"
    )
    X_all, _, feature_names, feature_metadata = (
        build_feature_matrix_with_metadata(identity_frame, feature_config)
    )
    contract = feature_contract_record(feature_metadata, feature_names)
    if contract["feature_metadata_hash"] != expected_feature_metadata_hash:
        raise ValueError("Phase 12 validation feature contract mismatch.")
    X_all = np.asarray(X_all, dtype=np.float32)
    source_ids = canonical["source_row_id"].astype(str).tolist()
    positions = {
        source_id: position
        for position, source_id in enumerate(source_ids)
    }
    fitted: dict[tuple[str, str], FittedUncertaintyEstimator] = {}
    for unit in units:
        allowed_ids = tuple(
            sorted(
                unit.teacher_fit_source_ids
                + unit.interval_calibration_source_ids
            )
        )
        outcomes = _read_allowed_outcomes(
            scientific["dataset_path"], source_ids, allowed_ids
        )
        train_X, train_y = _features_and_labels(
            unit.teacher_fit_source_ids, X_all, positions, outcomes
        )
        interval_X, interval_y = _features_and_labels(
            unit.interval_calibration_source_ids,
            X_all,
            positions,
            outcomes,
        )
        for method in SUPPORTED_UNCERTAINTY_METHODS:
            config = _estimator_config(
                method,
                scientific["uncertainty"]["method_configs"][method],
                coverage=scientific["uncertainty"]["coverage"],
                base_seed=scientific["base_seed"],
                evaluation_unit=unit.evaluation_unit,
            )
            fitted[(unit.evaluation_unit, method)] = (
                fit_uncertainty_estimator(
                    train_X,
                    train_y,
                    train_source_ids=unit.teacher_fit_source_ids,
                    calibration_features=interval_X,
                    calibration_labels=interval_y,
                    calibration_source_ids=(
                        unit.interval_calibration_source_ids
                    ),
                    config=config,
                )
            )
    return fitted, X_all, positions


def _features_and_labels(
    source_ids: Sequence[str],
    X_all: np.ndarray,
    positions: Mapping[str, int],
    outcomes: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([positions[source_id] for source_id in source_ids], dtype=int)
    return (
        X_all[indices],
        np.asarray([outcomes[source_id] for source_id in source_ids], dtype=float),
    )


def _read_allowed_outcomes(
    dataset_path: str | Path,
    canonical_source_ids: Sequence[str],
    allowed_source_ids: Sequence[str],
) -> dict[str, float]:
    allowed = set(allowed_source_ids)
    if len(allowed) != len(tuple(allowed_source_ids)) or not allowed:
        raise ValueError("Allowed outcome source IDs must be nonempty and unique.")
    positions = {
        source_id: index for index, source_id in enumerate(canonical_source_ids)
    }
    missing = sorted(allowed - set(positions))
    if missing:
        raise ValueError(f"Unknown allowed outcome IDs: {missing[:5]}.")
    allowed_positions = {positions[source_id] for source_id in allowed}
    frame = pd.read_csv(
        dataset_path,
        usecols=["source_row_id", "yield"],
        dtype={"source_row_id": str},
        skiprows=lambda row_number: (
            row_number > 0 and row_number - 1 not in allowed_positions
        ),
    )
    if set(frame["source_row_id"]) != allowed or len(frame) != len(allowed):
        raise ValueError("Selective outcome loading did not match allowed source IDs.")
    values = pd.to_numeric(frame["yield"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Allowed outcomes must be finite.")
    return dict(zip(frame["source_row_id"], values, strict=True))


def _estimator_config(
    method: str,
    overrides: Mapping[str, Any],
    *,
    coverage: float,
    base_seed: int,
    evaluation_unit: str,
) -> UncertaintyEstimatorConfig:
    resolved = dict(overrides)
    if "xgboost_seeds" in resolved:
        offsets = tuple(int(value) for value in resolved["xgboost_seeds"])
        derived = _derived_seed(base_seed, evaluation_unit, method)
        resolved["xgboost_seeds"] = tuple(derived + value for value in offsets)
    return UncertaintyEstimatorConfig(
        method=method,
        coverage=coverage,
        random_state=_derived_seed(base_seed, evaluation_unit, method),
        **resolved,
    )


def _nearest_cosine_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=float)
    s = np.asarray(support, dtype=float)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    s_norm = np.linalg.norm(s, axis=1, keepdims=True)
    similarities = (q / np.maximum(q_norm, 1e-12)) @ (
        s / np.maximum(s_norm, 1e-12)
    ).T
    return 1.0 - np.max(np.clip(similarities, -1.0, 1.0), axis=1)


def _policy_support_distances(
    unit: CalibrationHoldoutUnit,
    predictions: pd.DataFrame,
    fit_audit: pd.DataFrame,
) -> np.ndarray:
    rows = predictions.loc[
        predictions["evaluation_unit"].eq(unit.evaluation_unit)
        & predictions["method"].eq("distance_aware_ridge")
        & predictions["data_role"].eq("policy_validation")
    ].sort_values("source_row_id", kind="mergesort")
    if tuple(rows["source_row_id"]) != unit.policy_validation_source_ids:
        raise ValueError("Distance support rows do not match policy validation IDs.")
    return rows["raw_uncertainty_score"].to_numpy(float)


def _tertile_edges(values: Sequence[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Binning values must be finite and one-dimensional.")
    return [float(np.quantile(array, value)) for value in (1 / 3, 2 / 3)]


def _bin_label(value: float, edges: Sequence[float]) -> str:
    if value <= edges[0]:
        return "low"
    if value <= edges[1]:
        return "medium"
    return "high"


def _role_sets(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("requested_role_sets must be a nonempty list.")
    result = tuple(tuple(item) for item in value)
    expected = PROBE_ROLE_SETS + UNSUPPORTED_PROBE_ROLE_SETS
    if result != expected:
        raise ValueError(
            "requested_role_sets must preserve the six feasible and three "
            "explicitly unsupported scientific controls in canonical order."
        )
    return result


def _load_config(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("Phase 12 config must be a mapping.")
    return dict(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys mismatch: expected={sorted(expected)}, "
            f"observed={sorted(value)}."
        )


def _unique_integers(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    output = [_integer(item, name) for item in value]
    if len(output) != len(set(output)):
        raise ValueError(f"{name} must not contain duplicates.")
    return output


def _unique_fractions(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    output = [_open_closed_unit(item, name) for item in value]
    if output != sorted(output) or len(output) != len(set(output)):
        raise ValueError(f"{name} must be unique and increasing.")
    return output


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    output = _integer(value, name)
    if output < 1:
        raise ValueError(f"{name} must be positive.")
    return output


def _nonnegative_integer(value: Any, name: str) -> int:
    output = _integer(value, name)
    if output < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return output


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _open_unit(value: Any, name: str) -> float:
    output = float(value)
    if not math.isfinite(output) or not 0 < output < 1:
        raise ValueError(f"{name} must be in (0, 1).")
    return output


def _closed_unit(value: Any, name: str) -> float:
    output = float(value)
    if not math.isfinite(output) or not 0 <= output <= 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return output


def _open_closed_unit(value: Any, name: str) -> float:
    output = float(value)
    if not math.isfinite(output) or not 0 < output <= 1:
        raise ValueError(f"{name} must be in (0, 1].")
    return output


def _derived_seed(base_seed: int, *parts: str) -> int:
    return base_seed + int(stable_hash(list(parts))[:12], 16) % 1_000_000


def _stable_records(
    records: Sequence[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    return [
        _jsonable(record)
        for record in sorted(records, key=lambda record: str(record[key]))
    ]


def _csv_stable_records(frame: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    return _stable_records(_frame_records(frame), key)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonable(record) for record in frame.to_dict("records")]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _mapping_close(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    atol: float = 1e-10,
) -> bool:
    for key, expected_value in expected.items():
        if key not in observed:
            return False
        observed_value = observed[key]
        if expected_value is None:
            if not (
                observed_value is None
                or (isinstance(observed_value, float) and math.isnan(observed_value))
            ):
                return False
        elif isinstance(expected_value, (float, int)):
            if not math.isclose(
                float(expected_value),
                float(observed_value),
                rel_tol=0.0,
                abs_tol=atol,
            ):
                return False
        elif expected_value != observed_value:
            return False
    return True


def _records_close(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> bool:
    return len(expected) == len(observed) and all(
        _mapping_close(left, right) for left, right in zip(expected, observed, strict=True)
    )


def _strict_bool(value: Any) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Invalid serialized boolean: {value!r}.")


def _assert_row_counts(
    manifest: Mapping[str, Any], **tables: pd.DataFrame
) -> None:
    for stem, table in tables.items():
        key = f"{stem}_row_count"
        if int(manifest.get(key, -1)) != len(table):
            raise ValueError(f"Phase 12 row count mismatch: {stem}.")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scikit-learn", "scipy", "rdkit", "xgboost")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


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


def _command_record(config: Any, output: Path) -> str:
    config_value = str(config) if not isinstance(config, Mapping) else "<mapping>"
    return (
        "scripts/run_hidden_measured_calibration.py "
        f"--config {config_value} --output-directory {output}"
    )
