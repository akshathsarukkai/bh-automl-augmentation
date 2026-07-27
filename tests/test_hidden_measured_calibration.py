"""End-to-end contracts for Phase 12 hidden-measured calibration artifacts."""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
import pytest
import yaml

import bh_augmentation.hidden_measured_calibration as phase12
from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.representation_splits import (
    build_saved_random_split_unit,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

pytest.importorskip("rdkit")
pytest.importorskip("xgboost")

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    ROOT / "configs/corrected_hidden_measured_calibration_phase12_smoke.yaml"
)


@pytest.fixture(scope="module")
def completed_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("phase12") / "corrected-phase12-bundle"
    phase12.run_hidden_measured_calibration(
        SMOKE_CONFIG, output_directory=output
    )
    return output


def _copy_bundle(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "corrected-phase12-tampered"
    shutil.copytree(source, target)
    return target


def _rehash_output(root: Path, changed_name: str) -> None:
    _rehash_outputs(root, [changed_name])


def _rehash_outputs(root: Path, changed_names: list[str]) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for changed_name in changed_names:
        manifest["output_hashes"][changed_name] = sha256_file(
            root / changed_name
        )
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _replay_inputs(
    root: Path,
) -> tuple[dict, dict, object, tuple[object, ...]]:
    plan = json.loads((root / "calibration_plan.json").read_text())
    scientific = plan["resolved_scientific_config"]
    saved = load_saved_canonical_split_identities(
        scientific["dataset_path"],
        scientific["canonical_split_directory"],
        requested_seeds=scientific["random_seeds"],
        requested_fractions=scientific["random_fractions"],
    )
    panels = {
        int(seed): tuple(groups)
        for seed, groups in plan["condition_panels"].items()
    }
    base_units = tuple(
        build_saved_random_split_unit(saved, seed=seed, train_fraction=fraction)
        for seed in scientific["random_seeds"]
        for fraction in scientific["random_fractions"]
    )
    holdout_units = tuple(
        phase12._build_holdout_unit(
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
    return plan, scientific, saved, holdout_units


def _rewrite_hidden_metrics_as_oracle(root: Path, hidden: pd.DataFrame) -> None:
    path = root / "hidden_measured_metrics.csv"
    original = pd.read_csv(path, float_precision="round_trip")
    plan = json.loads((root / "calibration_plan.json").read_text())
    nominal_coverage = plan["resolved_scientific_config"]["uncertainty"][
        "coverage"
    ]
    records = []
    for (evaluation_unit, method), rows in hidden.groupby(
        ["evaluation_unit", "method"], sort=False
    ):
        record = (
            original.loc[
                original["evaluation_unit"].eq(evaluation_unit)
                & original["method"].eq(method)
            ]
            .iloc[0]
            .to_dict()
        )
        metrics = phase12.interval_calibration_metrics(
            rows["measured_yield"].to_numpy(float),
            rows["point_prediction"].to_numpy(float),
            rows["interval_lower"].to_numpy(float),
            rows["interval_upper"].to_numpy(float),
            rows["calibrated_uncertainty"].to_numpy(float),
            nominal_coverage=nominal_coverage,
        )
        record.update(
            rmse=0.0,
            mae=0.0,
            median_absolute_error=0.0,
            mean_signed_bias=0.0,
            **asdict(metrics),
        )
        records.append(record)
    pd.DataFrame(records, columns=original.columns).to_csv(path, index=False)


def _rewrite_policy_validation_as_oracle(root: Path) -> list[dict]:
    plan, scientific, _saved, holdout_units = _replay_inputs(root)
    prediction_path = root / "calibration_predictions.csv"
    predictions = pd.read_csv(prediction_path, float_precision="round_trip")
    policy_mask = predictions["data_role"].eq("policy_validation")
    predictions.loc[policy_mask, "point_prediction"] = predictions.loc[
        policy_mask, "measured_yield"
    ]
    predictions.loc[policy_mask, "absolute_error"] = 0.0
    predictions.loc[policy_mask, "interval_covered"] = (
        predictions.loc[policy_mask, "measured_yield"]
        >= predictions.loc[policy_mask, "interval_lower"]
    ) & (
        predictions.loc[policy_mask, "measured_yield"]
        <= predictions.loc[policy_mask, "interval_upper"]
    )
    predictions.to_csv(prediction_path, index=False)

    metric_path = root / "calibration_metrics.csv"
    original_metrics = pd.read_csv(metric_path, float_precision="round_trip")
    metric_records = []
    curve_records = []
    metrics_by_key = {}
    curves_by_key = {}
    for unit in holdout_units:
        for method in phase12.SUPPORTED_UNCERTAINTY_METHODS:
            rows = predictions.loc[
                predictions["evaluation_unit"].eq(unit.evaluation_unit)
                & predictions["method"].eq(method)
                & predictions["data_role"].eq("policy_validation")
            ].sort_values("source_row_id", kind="mergesort")
            metrics = phase12.interval_calibration_metrics(
                rows["measured_yield"].to_numpy(float),
                rows["point_prediction"].to_numpy(float),
                rows["interval_lower"].to_numpy(float),
                rows["interval_upper"].to_numpy(float),
                rows["calibrated_uncertainty"].to_numpy(float),
                nominal_coverage=scientific["uncertainty"]["coverage"],
            )
            metrics_by_key[(unit.evaluation_unit, method)] = metrics
            record = (
                original_metrics.loc[
                    original_metrics["evaluation_unit"].eq(unit.evaluation_unit)
                    & original_metrics["method"].eq(method)
                ]
                .iloc[0]
                .to_dict()
            )
            record.update(asdict(metrics))
            metric_records.append(record)
            curve = phase12.build_selective_prediction_curve(
                rows["measured_yield"].to_numpy(float),
                rows["point_prediction"].to_numpy(float),
                rows["calibrated_uncertainty"].to_numpy(float),
                tuple(rows["source_row_id"]),
            )
            curves_by_key[(unit.evaluation_unit, method)] = curve
            curve_records.extend(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "method": method,
                    "method_config_hash": rows["method_config_hash"].iloc[0],
                    **asdict(point),
                    "selection_data_role": "policy_validation",
                    "plan_hash": plan["plan_hash"],
                }
                for point in curve
            )
    pd.DataFrame(
        metric_records, columns=original_metrics.columns
    ).to_csv(metric_path, index=False)
    curve_path = root / "selective_prediction_curves.csv"
    curve_columns = pd.read_csv(curve_path, nrows=0).columns
    pd.DataFrame(curve_records, columns=curve_columns).to_csv(
        curve_path, index=False
    )

    fit_audit = pd.read_csv(root / "fit_audit.csv", float_precision="round_trip")
    fit_by_key = fit_audit.set_index(["evaluation_unit", "method"])
    policies = []
    for unit in holdout_units:
        candidates = tuple(
            phase12.ValidationCalibrationCandidate(
                method_id=method,
                method_config_hash=str(
                    fit_by_key.loc[
                        (unit.evaluation_unit, method), "config_hash"
                    ]
                ),
                metrics=metrics_by_key[(unit.evaluation_unit, method)],
                selective_curve=curves_by_key[
                    (unit.evaluation_unit, method)
                ],
                data_roles=phase12.CalibrationDataRoles(),
            )
            for method in phase12.SUPPORTED_UNCERTAINTY_METHODS
        )
        selection = phase12.select_validation_policy(
            candidates,
            nominal_coverage=scientific["uncertainty"]["coverage"],
            max_calibration_error=scientific["uncertainty"][
                "max_calibration_error"
            ],
            min_retained_fraction=scientific["uncertainty"][
                "min_retained_fraction"
            ],
        )
        selected_rows = predictions.loc[
            predictions["evaluation_unit"].eq(unit.evaluation_unit)
            & predictions["method"].eq(selection.method_id)
            & predictions["data_role"].eq("policy_validation")
        ].sort_values("source_row_id", kind="mergesort")
        distance_rows = predictions.loc[
            predictions["evaluation_unit"].eq(unit.evaluation_unit)
            & predictions["method"].eq("distance_aware_ridge")
            & predictions["data_role"].eq("policy_validation")
        ].sort_values("source_row_id", kind="mergesort")
        payload = {
            "schema_version": phase12.PHASE12_SCHEMA_VERSION,
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
            "uncertainty_bin_edges": phase12._tertile_edges(
                selected_rows["calibrated_uncertainty"].to_numpy(float)
            ),
            "support_distance_bin_edges": phase12._tertile_edges(
                distance_rows["raw_uncertainty_score"].to_numpy(float)
            ),
            "context_similarity_bin_edges": list(
                plan["descriptive_context_similarity_bin_edges"][
                    unit.evaluation_unit
                ]
            ),
            "context_similarity_bins_are_descriptive": True,
            "plan_hash": plan["plan_hash"],
            "selection_data_roles": ["policy_validation"],
            "hidden_labels_accessed": False,
            "outer_test_labels_accessed": False,
        }
        payload["frozen_policy_hash"] = stable_hash(payload)
        policies.append(payload)
    policy_document = {
        "schema_version": phase12.PHASE12_SCHEMA_VERSION,
        "status": "frozen_before_hidden_outcome_loading",
        "plan_hash": plan["plan_hash"],
        "policy_count": len(policies),
        "policies": policies,
        "hidden_labels_accessed": False,
        "outer_test_labels_accessed": False,
    }
    policy_document["policy_document_hash"] = stable_hash(policy_document)
    policy_path = root / "frozen_uncertainty_policies.json"
    policy_path.write_text(
        json.dumps(policy_document, indent=2, sort_keys=True) + "\n"
    )

    hidden_path = root / "hidden_measured_predictions.csv"
    hidden = pd.read_csv(hidden_path, float_precision="round_trip")
    filtering_records = []
    for policy in policies:
        evaluation_unit = policy["evaluation_unit"]
        method = policy["selection"]["method_id"]
        cutoff = float(policy["selection"]["uncertainty_cutoff"])
        unit_mask = hidden["evaluation_unit"].eq(evaluation_unit)
        hidden.loc[unit_mask, "frozen_policy_hash"] = policy[
            "frozen_policy_hash"
        ]
        hidden.loc[unit_mask, "selected_by_validation"] = hidden.loc[
            unit_mask, "method"
        ].eq(method)
        hidden.loc[unit_mask, "accepted_by_frozen_filter"] = hidden.loc[
            unit_mask, "selected_by_validation"
        ] & (hidden.loc[unit_mask, "calibrated_uncertainty"] <= cutoff)
        evidence_hash = stable_hash(
            {
                "estimator_audit_hash": policy["estimator_audit_hash"],
                "validation_prediction_hash": policy[
                    "validation_prediction_hash"
                ],
                "selection": policy["selection"],
            }
        )
        selected = hidden.loc[
            unit_mask & hidden["method"].eq(method)
        ].sort_values("hidden_row_id", kind="mergesort")
        filtering_records.extend(
            {
                "evaluation_unit": evaluation_unit,
                "hidden_row_id": row.hidden_row_id,
                "method": method,
                "calibrated_uncertainty": row.calibrated_uncertainty,
                "frozen_uncertainty_cutoff": cutoff,
                "accepted_by_frozen_filter": (
                    row.calibrated_uncertainty <= cutoff
                ),
                "filter_rule": (
                    "calibrated_uncertainty <= validation_selected_cutoff"
                ),
                "calibration_evidence_hash": evidence_hash,
                "frozen_policy_hash": policy["frozen_policy_hash"],
                "eligible_for_training": False,
                "plan_hash": plan["plan_hash"],
            }
            for row in selected.itertuples(index=False)
        )
    hidden.to_csv(hidden_path, index=False)
    filtering_path = root / "filtering_evidence.csv"
    filtering_columns = pd.read_csv(filtering_path, nrows=0).columns
    pd.DataFrame(filtering_records, columns=filtering_columns).to_csv(
        filtering_path, index=False
    )
    stratified_path = root / "hidden_stratified_metrics.csv"
    probes = pd.read_csv(
        root / "probe_provenance.csv", float_precision="round_trip"
    )
    stratified = phase12._stratified_hidden_metrics(
        hidden,
        probes,
        {policy["evaluation_unit"]: policy for policy in policies},
        holdout_units,
    )
    stratified.to_csv(stratified_path, index=False)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_document_hash"] = policy_document["policy_document_hash"]
    manifest["calibrated_methods_passing_selection"] = sorted(
        {policy["selection"]["method_id"] for policy in policies}
    )
    manifest["hidden_stratified_metric_row_count"] = len(stratified)
    for changed_path in (
        prediction_path,
        metric_path,
        curve_path,
        policy_path,
        hidden_path,
        filtering_path,
        stratified_path,
    ):
        manifest["output_hashes"][changed_path.name] = sha256_file(changed_path)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return policies


def test_smoke_bundle_passes_public_validator(completed_bundle: Path) -> None:
    manifest = phase12.validate_hidden_measured_calibration(completed_bundle)
    plan = json.loads((completed_bundle / "calibration_plan.json").read_text())
    partitions = pd.read_csv(
        completed_bundle / "partition_assignments.csv"
    )
    probes = pd.read_csv(completed_bundle / "probe_provenance.csv")
    hidden = pd.read_csv(
        completed_bundle / "hidden_measured_predictions.csv"
    )

    assert manifest["holdout_unit_count"] == 1
    assert manifest["method_count"] == 6
    assert manifest["outer_test_labels_accessed"] is False
    assert manifest["hidden_labels_accessed_during_search"] is False
    assert manifest["hidden_labels_accessed_after_policy_freeze"] is True
    assert manifest["calibrated_methods_passing_selection"]
    assert len(partitions) == 3955
    held_validation_count = plan["holdout_units"][0][
        "held_condition_validation_excluded_count"
    ]
    assert (
        partitions["data_role"]
        .eq("held_condition_validation_excluded")
        .sum()
        == held_validation_count
    )
    assert set(hidden["hidden_row_id"]) == set(probes["hidden_row_id"])
    assert plan["probe_coverage"][0]["hidden_target_denominator"] >= plan[
        "probe_coverage"
    ][0]["exactly_reconstructed_target_count"]


def test_holdout_unit_retains_source_identity_column() -> None:
    config = yaml.safe_load(SMOKE_CONFIG.read_text())
    saved = load_saved_canonical_split_identities(
        config["dataset"]["path"],
        config["splits"]["canonical_directory"],
        requested_seeds=[0],
        requested_fractions=[0.2],
    )
    unit = build_saved_random_split_unit(saved, seed=0, train_fraction=0.2)
    held_key = (
        saved.canonical.set_index("source_row_id", drop=False)
        .loc[list(unit.train_source_ids), "canonical_condition_key"]
        .iloc[0]
    )

    holdout = phase12._build_holdout_unit(
        unit,
        saved.canonical,
        held_condition_key=held_key,
        validation_salt="fixture-validation-salt",
    )

    assert holdout.hidden_source_ids
    assert not (
        set(holdout.hidden_source_ids) & set(holdout.teacher_fit_source_ids)
    )


def test_selective_outcome_loader_materializes_only_allowed_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical.csv"
    pd.DataFrame(
        {
            "source_row_id": ["a", "b", "c"],
            "yield": [1.0, 999.0, 3.0],
        }
    ).to_csv(path, index=False)

    assert phase12._read_allowed_outcomes(
        path, ["a", "b", "c"], ["c", "a"]
    ) == {"a": 1.0, "c": 3.0}


def test_validator_rejects_fit_membership_attack_after_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "fit_audit.csv"
    rows = pd.read_csv(path)
    rows.loc[0, "train_source_id_hash"] = "0" * 64
    rows.to_csv(path, index=False)
    _rehash_output(root, path.name)

    with pytest.raises(ValueError, match="fit membership provenance"):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_hidden_outcome_attack_after_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "hidden_measured_predictions.csv"
    rows = pd.read_csv(path)
    rows.loc[0, "measured_yield"] += 17.0
    rows.to_csv(path, index=False)
    _rehash_output(root, path.name)

    with pytest.raises(ValueError, match="canonical-grounded"):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_filter_cutoff_attack_after_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "filtering_evidence.csv"
    rows = pd.read_csv(path)
    rows.loc[0, "frozen_uncertainty_cutoff"] += 100.0
    rows.loc[0, "accepted_by_frozen_filter"] = True
    rows.to_csv(path, index=False)
    _rehash_output(root, path.name)

    with pytest.raises(ValueError, match="filtering linkage"):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_policy_selection_attack_after_global_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "frozen_uncertainty_policies.json"
    document = json.loads(path.read_text())
    policy = document["policies"][0]
    policy["selection"]["uncertainty_cutoff"] += 1.0
    policy.pop("frozen_policy_hash")
    policy["frozen_policy_hash"] = stable_hash(policy)
    document.pop("policy_document_hash")
    document["policy_document_hash"] = stable_hash(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    _rehash_output(root, path.name)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_document_hash"] = document["policy_document_hash"]
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="frozen policy replay"):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_hidden_label_oracle_predictions_after_full_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    hidden_path = root / "hidden_measured_predictions.csv"
    hidden = pd.read_csv(hidden_path, float_precision="round_trip")
    hidden["point_prediction"] = hidden["measured_yield"]
    hidden["signed_error"] = 0.0
    hidden["absolute_error"] = 0.0
    hidden["interval_covered"] = (
        hidden["measured_yield"] >= hidden["interval_lower"]
    ) & (hidden["measured_yield"] <= hidden["interval_upper"])
    hidden.to_csv(hidden_path, index=False)
    _rewrite_hidden_metrics_as_oracle(root, hidden)

    policy_document = json.loads(
        (root / "frozen_uncertainty_policies.json").read_text()
    )
    _plan, _scientific, _saved, holdout_units = _replay_inputs(root)
    stratified_path = root / "hidden_stratified_metrics.csv"
    stratified = phase12._stratified_hidden_metrics(
        hidden,
        pd.read_csv(root / "probe_provenance.csv", float_precision="round_trip"),
        {
            policy["evaluation_unit"]: policy
            for policy in policy_document["policies"]
        },
        holdout_units,
    )
    stratified.to_csv(stratified_path, index=False)
    _rehash_outputs(
        root,
        [
            hidden_path.name,
            "hidden_measured_metrics.csv",
            stratified_path.name,
        ],
    )

    with pytest.raises(ValueError):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_policy_validation_label_oracle_after_full_rehash(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    policies = _rewrite_policy_validation_as_oracle(root)

    assert all(
        policy["selection"]["validation_rmse"] == 0.0 for policy in policies
    )
    with pytest.raises(ValueError):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_self_consistent_fit_data_hash_forgery(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    policy_document = json.loads(
        (root / "frozen_uncertainty_policies.json").read_text()
    )
    selected_methods = {
        policy["selection"]["method_id"]
        for policy in policy_document["policies"]
    }
    audit_path = root / "fit_audit.csv"
    audit = pd.read_csv(audit_path, float_precision="round_trip")
    attacked_index = audit.index[
        ~audit["method"].isin(selected_methods)
    ][0]
    old_audit_hash = audit.loc[attacked_index, "audit_hash"]
    audit.loc[attacked_index, "train_data_hash"] = "forged-training-data-hash"
    parsed = audit.loc[attacked_index].to_dict()
    for field_name in (
        "model_names",
        "model_seeds",
        "bootstrap_sample_hashes",
    ):
        parsed[field_name] = tuple(ast.literal_eval(str(parsed[field_name])))
    reconstructed = phase12.EstimatorAudit(
        **{
            field.name: parsed[field.name]
            for field in fields(phase12.EstimatorAudit)
        }
    )
    new_audit_hash = reconstructed.audit_hash
    audit.loc[attacked_index, "audit_hash"] = new_audit_hash
    audit.to_csv(audit_path, index=False)

    prediction_path = root / "calibration_predictions.csv"
    predictions = pd.read_csv(
        prediction_path, float_precision="round_trip"
    )
    predictions.loc[
        predictions["estimator_audit_hash"].eq(old_audit_hash),
        "estimator_audit_hash",
    ] = new_audit_hash
    predictions.to_csv(prediction_path, index=False)
    _rehash_outputs(root, [audit_path.name, prediction_path.name])

    with pytest.raises(ValueError):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_hidden_derived_and_provenance_field_forgery(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "hidden_measured_predictions.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    rows["signed_error"] = 12345.0
    rows["absolute_error"] = 12345.0
    rows["interval_covered"] = False
    rows["accepted_by_frozen_filter"] = ~rows[
        "accepted_by_frozen_filter"
    ].map(phase12._strict_bool)
    rows["method_config_hash"] = "forged-config-hash"
    rows["estimator_audit_hash"] = "forged-audit-hash"
    rows.to_csv(path, index=False)
    _rehash_output(root, path.name)

    with pytest.raises(ValueError):
        phase12.validate_hidden_measured_calibration(root)


def test_validator_rejects_calibration_derived_field_forgery(
    completed_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(completed_bundle, tmp_path)
    path = root / "calibration_predictions.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    rows["absolute_error"] = 0.0
    rows["interval_covered"] = True
    rows["raw_lower"] = -9999.0
    rows["raw_upper"] = 9999.0
    rows.to_csv(path, index=False)
    _rehash_output(root, path.name)

    with pytest.raises(ValueError):
        phase12.validate_hidden_measured_calibration(root)
