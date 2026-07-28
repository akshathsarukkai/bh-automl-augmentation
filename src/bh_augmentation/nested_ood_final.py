"""One-time outer evaluation of frozen nested group-aware OOD policies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.evaluation_registry import (
    EvaluationIdentity,
    EvaluationRegistry,
    default_evaluation_registry_root,
)
from bh_augmentation.evaluation.nested_ood import (
    NESTED_OOD_SCHEMA_VERSION,
    aggregate_inner_ood_metrics,
    build_nested_group_ood_contract,
    build_nested_ood_all_outer_plan,
)
from bh_augmentation.evaluation.nested_ood_artifacts import (
    validate_nested_ood_search_artifacts,
)
from bh_augmentation.models.predict import predict_model
from bh_augmentation.nested_ood_common import (
    TARGET_COLUMNS,
    assert_fresh_output,
    build_partition,
    create_fresh_output,
    fit_policy,
    metric_value,
    resolve_contract,
    scientific_projection,
)
from bh_augmentation.nested_ood_search import NESTED_OOD_SEARCH_SCHEMA_VERSION
from bh_augmentation.policy_search import current_commit
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

NESTED_OOD_FINAL_SCHEMA_VERSION = "bh-nested-ood-final-v1"
# Anchored to the checkout root so the outer-test claim cannot be bypassed by
# launching the runner from a different working directory.
EVALUATION_REGISTRY_DIRECTORY = default_evaluation_registry_root()


def run_nested_ood_final(
    config_path: str | Path,
    *,
    frozen_policies_path: str | Path,
    search_manifest_path: str | Path,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Refit frozen policies on all outer groups and evaluate each outer group once."""
    config = load_config(config_path)
    resolved = resolve_contract(config)
    saved = load_saved_canonical_splits(
        resolved["dataset_path"],
        resolved["split_directory"],
    )
    config_hash = stable_hash(scientific_projection(config))
    commit = current_commit()
    search_manifest = _load_search_manifest(search_manifest_path)
    frozen = _load_frozen_policies(frozen_policies_path)
    expected_selection_contract = {
        "metric": resolved["selection_metric"],
        "weighting": resolved["selection_weighting"],
        "lower_is_better": resolved["lower_is_better"],
    }
    validated_search = validate_nested_ood_search_artifacts(
        saved.canonical,
        search_manifest_path=search_manifest_path,
        frozen_policies_path=frozen_policies_path,
        target_columns={
            target: TARGET_COLUMNS[target] for target in resolved["targets"]
        },
        expected_policies=resolved["policies"],
        expected_metrics=resolved["metrics"],
        expected_selection_metric=resolved["selection_metric"],
        expected_selection_weighting=resolved["selection_weighting"],
        expected_lower_is_better=resolved["lower_is_better"],
        expected_binding={
            "dataset_hash": saved.dataset_hash,
            "canonical_split_dependency_hash": saved.aggregate_split_hash,
            "canonicalization_version": saved.manifest[
                "canonicalization_version"
            ],
            "split_schema_version": saved.manifest["split_schema_version"],
            "config_hash": config_hash,
            "commit_hash": commit,
        },
    )
    if (
        frozen["candidate_policies"] != resolved["policies"]
        or frozen["selection_contract"] != expected_selection_contract
        or search_manifest.get("candidate_policies") != resolved["policies"]
        or search_manifest.get("selection_contract")
        != expected_selection_contract
    ):
        raise ValueError("Nested OOD candidate budget or selection contract mismatch.")
    _verify_search_binding(
        search_manifest,
        frozen,
        frozen_path=Path(frozen_policies_path),
        saved=saved,
        config_hash=config_hash,
        commit=commit,
    )
    expected_units = {
        _unit(target, outer_group)
        for target in resolved["targets"]
        for outer_group in sorted(
            saved.canonical[TARGET_COLUMNS[target]].astype(str).unique()
        )
    }
    frozen_by_unit = {row["evaluation_unit"]: row for row in frozen["policies"]}
    if validated_search.frozen_by_unit != frozen_by_unit:
        raise ValueError("Validated frozen policies differ from the loaded document.")
    expected_pairs = {
        (target, outer_group, TARGET_COLUMNS[target])
        for target in resolved["targets"]
        for outer_group in sorted(
            saved.canonical[TARGET_COLUMNS[target]].astype(str).unique()
        )
    }
    actual_pairs = {
        (row.get("target"), row.get("outer_group"), row.get("group_column"))
        for row in frozen["policies"]
    }
    if (
        len(frozen_by_unit) != len(frozen["policies"])
        or set(frozen_by_unit) != expected_units
        or actual_pairs != expected_pairs
        or any(
            row["evaluation_unit"] != _unit(row["target"], row["outer_group"])
            for row in frozen["policies"]
        )
    ):
        raise ValueError("Frozen nested OOD units do not match configured outer folds.")
    expected_plans = {
        target: _target_plan(saved.canonical, target) for target in resolved["targets"]
    }
    if search_manifest.get("target_plans") != expected_plans:
        raise ValueError("Nested OOD whole-target plan hash mismatch.")
    _replay_search(
        frozen_by_unit,
        inner_metrics_path=Path(search_manifest_path).parent / "inner_metrics.csv",
        canonical=saved.canonical,
        resolved=resolved,
        expected_plans=expected_plans,
    )
    raw_output = (
        output_directory
        if output_directory is not None
        else config.get("output", {}).get(
            "nested_ood_final_directory",
            "results/corrected_nested_ood_final_example_only",
        )
    )
    output_path = assert_fresh_output(raw_output)
    output = create_fresh_output(output_path)
    folds_directory = output / "fold_manifests"
    folds_directory.mkdir()
    registry = EvaluationRegistry(EVALUATION_REGISTRY_DIRECTORY)
    outer_rows: list[dict[str, Any]] = []
    fold_hashes: dict[str, str] = {}
    registry_paths: dict[str, Path] = {}
    prediction_batches_this_invocation: dict[str, int] = {}
    unresolved: list[dict[str, str]] = []
    for unit in sorted(expected_units):
        selection = frozen_by_unit[unit]
        target = selection["target"]
        outer_group = selection["outer_group"]
        group_column = TARGET_COLUMNS[target]
        contract = build_nested_group_ood_contract(
            _identity_frame(saved.canonical, group_column),
            group_column=group_column,
            outer_group=outer_group,
        )
        _verify_frozen_unit(
            selection,
            contract=contract,
            saved=saved,
            config_hash=config_hash,
            commit=commit,
            target_plan_hash=expected_plans[target]["plan_hash"],
        )
        train_frame = _rows_by_id(saved.canonical, contract.outer_train_source_ids)
        X_train, y_train, train_feature = build_partition(
            train_frame,
            resolved["feature_config"],
        )
        if train_feature["feature_metadata_hash"] != selection["feature_metadata_hash"]:
            raise ValueError("Frozen nested OOD feature metadata hash mismatch.")
        policy = selection["policy"]
        matching_policies = [
            (index, candidate)
            for index, candidate in enumerate(resolved["policies"])
            if candidate["policy_id"] == policy["policy_id"]
        ]
        if len(matching_policies) != 1 or matching_policies[0][1] != policy:
            raise ValueError("Frozen nested OOD policy config mismatch.")
        policy_index = matching_policies[0][0]
        model_seed = resolved["seed"] + 9_000_000 + policy_index
        model = fit_policy(policy, X_train, y_train, seed=model_seed)
        identity = EvaluationIdentity(
            dataset_hash=saved.dataset_hash,
            split_or_search_manifest_hash=search_manifest["search_manifest_hash"],
            frozen_policy_hash=selection["frozen_policy_hash"],
            evaluation_unit=unit,
        )
        registry_paths[unit] = registry.path_for(identity)
        record = registry.inspect(identity)
        reused = record is not None and record.status == "metrics_complete"
        if reused:
            unit_rows = _validated_registry_metrics(
                record.metrics_payload,
                selection=selection,
                metrics=resolved["metrics"],
                n_refit=len(train_frame),
                n_test=len(contract.outer_test_source_ids),
            )
            prediction_batches_this_invocation[unit] = 0
        elif record is not None:
            unresolved.append({"evaluation_unit": unit, "status": record.status})
            prediction_batches_this_invocation[unit] = 0
            _write_unresolved_fold(
                folds_directory,
                unit=unit,
                contract=contract,
                selection=selection,
                registry_status=record.status,
                registry_path=registry.path_for(identity),
                fold_hashes=fold_hashes,
            )
            continue
        else:
            registry.reserve(identity)
            try:
                test_frame = _rows_by_id(
                    saved.canonical,
                    contract.outer_test_source_ids,
                )
                X_test, y_test, test_feature = build_partition(
                    test_frame,
                    resolved["feature_config"],
                )
                if train_feature != test_feature:
                    raise ValueError(
                        "Nested OOD refit/test feature contracts differ."
                    )
                prediction = np.asarray(predict_model(model, X_test), dtype=float)
                if not np.isfinite(prediction).all():
                    raise ValueError("Nested OOD outer prediction is non-finite.")
                registry.mark_prediction_complete(
                    identity,
                    prediction_hash=stable_hash(prediction.tolist()),
                )
                unit_rows = _outer_metric_rows(
                    unit=unit,
                    target=target,
                    group_column=group_column,
                    outer_group=outer_group,
                    selection=selection,
                    model_seed=model_seed,
                    y_test=y_test,
                    prediction=prediction,
                    n_refit=len(train_frame),
                    n_test=len(test_frame),
                    metrics=resolved["metrics"],
                )
                record = registry.mark_metrics_complete(
                    identity,
                    metrics_payload=unit_rows,
                )
                unit_rows = list(record.metrics_payload or [])
                prediction_batches_this_invocation[unit] = 1
            except Exception as exc:
                current = registry.inspect(identity)
                if current is not None and current.status in {
                    "reserved",
                    "prediction_complete",
                }:
                    registry.mark_failed_or_uncertain(
                        identity,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                unresolved.append(
                    {"evaluation_unit": unit, "status": "failed_or_uncertain"}
                )
                prediction_batches_this_invocation[unit] = 0
                _write_unresolved_fold(
                    folds_directory,
                    unit=unit,
                    contract=contract,
                    selection=selection,
                    registry_status="failed_or_uncertain",
                    registry_path=registry.path_for(identity),
                    fold_hashes=fold_hashes,
                )
                continue
        outer_rows.extend(unit_rows)
        registry_hash = sha256_file(registry.path_for(identity))
        fold_payload = {
            "schema_version": NESTED_OOD_SCHEMA_VERSION,
            "status": "complete",
            "evaluation_unit": unit,
            "contract": contract.audit_record,
            "frozen_selection": selection,
            "refit_source_id_hash": stable_hash(contract.outer_train_source_ids),
            "test_source_id_hash": stable_hash(contract.outer_test_source_ids),
            "outer_test_prediction_batches": 1,
            "outer_test_prediction_batches_this_invocation": (
                prediction_batches_this_invocation[unit]
            ),
            "outer_test_metric_evaluations": len(resolved["metrics"]),
            "outer_metric_rows_hash": stable_hash(unit_rows),
            "registry_status": "metrics_complete",
            "registry_record_hash": registry_hash,
            "registry_metrics_reused": reused,
        }
        fold_document = {
            "payload": fold_payload,
            "payload_hash": stable_hash(fold_payload),
        }
        fold_path = folds_directory / f"{unit}.json"
        fold_path.write_text(json.dumps(fold_document, indent=2, sort_keys=True) + "\n")
        fold_hashes[fold_path.name] = sha256_file(fold_path)

    paths = {
        "directory": output,
        "outer_metrics": output / "outer_metrics.csv",
        "outer_summary": output / "outer_summary.csv",
        "final_manifest": output / "final_manifest.json",
    }
    outer_frame = pd.DataFrame(outer_rows)
    outer_frame.to_csv(paths["outer_metrics"], index=False)
    if unresolved:
        incomplete_payload = {
            "schema_version": NESTED_OOD_FINAL_SCHEMA_VERSION,
            "status": "incomplete",
            "completed_evaluation_units": sorted(
                set(expected_units)
                - {row["evaluation_unit"] for row in unresolved}
            ),
            "unresolved_evaluation_units": unresolved,
            "outer_metrics_hash": sha256_file(paths["outer_metrics"]),
            "fold_manifest_hashes": fold_hashes,
        }
        incomplete_payload["incomplete_manifest_hash"] = stable_hash(
            incomplete_payload
        )
        incomplete_path = output / "incomplete_manifest.json"
        incomplete_path.write_text(
            json.dumps(incomplete_payload, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(
            "Nested OOD final evaluation is incomplete; uncertain units were "
            "not reevaluated. See incomplete_manifest.json."
        )
    _outer_summary(outer_frame).to_csv(paths["outer_summary"], index=False)
    registry_hashes = {
        registry_paths[unit].name: sha256_file(registry_paths[unit])
        for unit in sorted(expected_units)
    }
    final_payload = {
        "schema_version": NESTED_OOD_FINAL_SCHEMA_VERSION,
        "status": "complete",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "config_hash": config_hash,
        "commit_hash": commit,
        "search_manifest_hash": search_manifest["search_manifest_hash"],
        "frozen_document_hash": frozen["frozen_document_hash"],
        "expected_evaluation_units": sorted(expected_units),
        "completed_evaluation_units": sorted(expected_units),
        "per_unit_test_prediction_batches": {
            unit: 1 for unit in sorted(expected_units)
        },
        "per_unit_test_prediction_batches_this_invocation": (
            prediction_batches_this_invocation
        ),
        "selection_data_roles": ["inner_train", "inner_validation"],
        "refit_data_roles": ["outer_train"],
        "test_evaluated": True,
        "outer_metrics_hash": sha256_file(paths["outer_metrics"]),
        "outer_summary_hash": sha256_file(paths["outer_summary"]),
        "evaluation_registry_directory": str(EVALUATION_REGISTRY_DIRECTORY),
        "registry_record_hashes": registry_hashes,
        "fold_manifest_hashes": fold_hashes,
    }
    final_payload["final_manifest_hash"] = stable_hash(final_payload)
    paths["final_manifest"].write_text(
        json.dumps(final_payload, indent=2, sort_keys=True) + "\n"
    )
    return paths


def _load_search_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    recorded = value.get("search_manifest_hash")
    payload = dict(value)
    payload.pop("search_manifest_hash", None)
    if (
        value.get("schema_version") != NESTED_OOD_SEARCH_SCHEMA_VERSION
        or value.get("status") != "complete"
        or stable_hash(payload) != recorded
        or value.get("selection_data_roles")
        != ["inner_train", "inner_validation"]
        or value.get("outer_test_labels_accessed") is not False
        or value.get("outer_test_predictions_generated") is not False
        or value.get("outer_test_metric_evaluations") != 0
        or value.get("test_evaluated") is not False
    ):
        raise ValueError("Nested OOD search manifest is invalid or incomplete.")
    for name, expected in value.get("output_hashes", {}).items():
        artifact = Path(path).parent / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("Nested OOD search artifact hash mismatch.")
    for name, expected in value.get("fold_manifest_hashes", {}).items():
        artifact = Path(path).parent / "fold_manifests" / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("Nested OOD search fold-manifest hash mismatch.")
    return value


def _load_frozen_policies(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    recorded = value.get("frozen_document_hash")
    payload = dict(value)
    payload.pop("frozen_document_hash", None)
    if (
        value.get("schema_version") != NESTED_OOD_SEARCH_SCHEMA_VERSION
        or value.get("status") != "frozen"
        or not isinstance(value.get("candidate_policies"), list)
        or not isinstance(value.get("selection_contract"), dict)
        or not isinstance(value.get("policies"), list)
        or stable_hash(payload) != recorded
    ):
        raise ValueError("Frozen nested OOD policy document is invalid.")
    for row in value["policies"]:
        frozen_hash = row.get("frozen_policy_hash")
        unhashed = dict(row)
        unhashed.pop("frozen_policy_hash", None)
        if stable_hash(unhashed) != frozen_hash:
            raise ValueError("Frozen nested OOD policy hash mismatch.")
    return value


def _verify_search_binding(
    search: dict[str, Any],
    frozen: dict[str, Any],
    *,
    frozen_path: Path,
    saved: Any,
    config_hash: str,
    commit: str,
) -> None:
    if (
        search["dataset_hash"] != saved.dataset_hash
        or search["canonical_split_dependency_hash"] != saved.aggregate_split_hash
        or search["config_hash"] != config_hash
        or search["commit_hash"] != commit
        or search["output_hashes"].get(frozen_path.name) != sha256_file(frozen_path)
        or search.get("candidate_policies") != frozen.get("candidate_policies")
        or search.get("selection_contract") != frozen.get("selection_contract")
    ):
        raise ValueError("Nested OOD search/final scientific binding mismatch.")
    units = search.get("evaluation_units")
    if units != [row["evaluation_unit"] for row in frozen["policies"]]:
        raise ValueError("Frozen policy units do not match search manifest.")


def _verify_frozen_unit(
    selection: dict[str, Any],
    *,
    contract: Any,
    saved: Any,
    config_hash: str,
    commit: str,
    target_plan_hash: str,
) -> None:
    if (
        selection["outer_assignment_hash"] != contract.outer_assignment_hash
        or selection["aggregate_assignment_hash"] != contract.aggregate_assignment_hash
        or selection["dataset_hash"] != saved.dataset_hash
        or selection["canonical_split_dependency_hash"] != saved.aggregate_split_hash
        or selection["config_hash"] != config_hash
        or selection["commit_hash"] != commit
        or selection["target_plan_hash"] != target_plan_hash
    ):
        raise ValueError("Frozen nested OOD unit binding mismatch.")


def _target_plan(canonical: pd.DataFrame, target: str) -> dict[str, Any]:
    group_column = TARGET_COLUMNS[target]
    identity = _identity_frame(canonical, group_column)
    groups = sorted(identity[group_column].astype(str).unique())
    return build_nested_ood_all_outer_plan(
        identity,
        (
            build_nested_group_ood_contract(
                identity,
                group_column=group_column,
                outer_group=outer_group,
            )
            for outer_group in groups
        ),
        group_column=group_column,
    ).audit_record


def _replay_search(
    frozen_by_unit: dict[str, dict[str, Any]],
    *,
    inner_metrics_path: Path,
    canonical: pd.DataFrame,
    resolved: dict[str, Any],
    expected_plans: dict[str, dict[str, Any]],
) -> None:
    metrics = pd.read_csv(inner_metrics_path)
    expected_policy_hashes = {
        policy["policy_id"]: policy["policy_hash"] for policy in resolved["policies"]
    }
    expected_policies = {
        policy["policy_id"]: policy for policy in resolved["policies"]
    }
    for unit, selection in frozen_by_unit.items():
        if (
            selection["selection_metric"] != resolved["selection_metric"]
            or selection["selection_weighting"] != resolved["selection_weighting"]
            or selection["lower_is_better"] != resolved["lower_is_better"]
            or selection["evaluation_unit"]
            != _unit(selection["target"], selection["outer_group"])
            or selection["group_column"] != TARGET_COLUMNS[selection["target"]]
        ):
            raise ValueError("Frozen nested OOD selection contract mismatch.")
        contract = build_nested_group_ood_contract(
            _identity_frame(canonical, selection["group_column"]),
            group_column=selection["group_column"],
            outer_group=selection["outer_group"],
        )
        rows = metrics.loc[metrics["evaluation_unit"].eq(unit)].copy()
        expected_count = (
            contract.inner_fold_count
            * len(resolved["policies"])
            * len(resolved["metrics"])
        )
        if len(rows) != expected_count or rows.duplicated(
            ["policy_id", "metric", "inner_fold_index"]
        ).any():
            raise ValueError("Nested OOD inner policy/fold coverage is incomplete.")
        if set(rows["policy_id"]) != set(expected_policy_hashes):
            raise ValueError("Nested OOD inner policy coverage mismatch.")
        if set(rows["metric"]) != set(resolved["metrics"]):
            raise ValueError("Nested OOD inner metric coverage mismatch.")
        if any(
            not rows.loc[rows["policy_id"].eq(policy_id), "policy_hash"].eq(
                policy_hash
            ).all()
            for policy_id, policy_hash in expected_policy_hashes.items()
        ):
            raise ValueError("Nested OOD inner policy hash mismatch.")
        if (
            not rows["outer_group"].eq(selection["outer_group"]).all()
            or not rows["outer_assignment_hash"].eq(
                contract.outer_assignment_hash
            ).all()
            or not rows["aggregate_assignment_hash"].eq(
                contract.aggregate_assignment_hash
            ).all()
        ):
            raise ValueError("Nested OOD inner outer-fold hash mismatch.")
        fold_hashes = {
            fold.fold_index: fold.assignment_hash for fold in contract.inner_folds
        }
        if any(
            not rows.loc[rows["inner_fold_index"].eq(index), "inner_assignment_hash"]
            .eq(fold_hash)
            .all()
            for index, fold_hash in fold_hashes.items()
        ):
            raise ValueError("Nested OOD inner fold assignment hash mismatch.")
        aggregate = aggregate_inner_ood_metrics(
            contract,
            rows,
            expected_group_keys=[
                {
                    "policy_id": policy["policy_id"],
                    "policy_hash": policy["policy_hash"],
                    "metric": metric,
                }
                for policy in resolved["policies"]
                for metric in resolved["metrics"]
            ],
            group_columns=("policy_id", "policy_hash", "metric"),
            weighting=selection["selection_weighting"],
        )
        candidates = aggregate.loc[
            aggregate["metric"].eq(selection["selection_metric"])
        ].sort_values(
            ["value", "policy_hash", "policy_id"],
            ascending=[selection["lower_is_better"], True, True],
            kind="mergesort",
        )
        selected = candidates.iloc[0]
        if (
            selected["policy_id"] != selection["policy"]["policy_id"]
            or selection["policy"]
            != expected_policies.get(selection["policy"]["policy_id"])
            or not math.isclose(
                float(selected["value"]),
                float(selection["selection_value"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or selection["target_plan_hash"]
            != expected_plans[selection["target"]]["plan_hash"]
        ):
            raise ValueError("Frozen nested OOD selection is not reproducible.")


def _outer_metric_rows(
    *,
    unit: str,
    target: str,
    group_column: str,
    outer_group: str,
    selection: dict[str, Any],
    model_seed: int,
    y_test: np.ndarray,
    prediction: np.ndarray,
    n_refit: int,
    n_test: int,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    policy = selection["policy"]
    return [
        {
            "evaluation_unit": unit,
            "target": target,
            "group_column": group_column,
            "outer_group": outer_group,
            "policy_id": policy["policy_id"],
            "policy_hash": policy["policy_hash"],
            "frozen_policy_hash": selection["frozen_policy_hash"],
            "model_seed": model_seed,
            "metric": metric,
            "value": metric_value(
                metric,
                y_test,
                prediction,
                context=(
                    f"target={target!r}, outer_group={outer_group!r}, "
                    f"policy={policy['policy_id']!r}"
                ),
            ),
            "n_refit": n_refit,
            "n_test": n_test,
            "outer_test_prediction_batch": 1,
            "outer_test_metric_evaluation_count": 1,
        }
        for metric in metrics
    ]


def _validated_registry_metrics(
    payload: list[dict[str, Any]] | None,
    *,
    selection: dict[str, Any],
    metrics: tuple[str, ...],
    n_refit: int,
    n_test: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) != len(metrics):
        raise ValueError("Completed registry metric payload has invalid row count.")
    if {row.get("metric") for row in payload} != set(metrics):
        raise ValueError("Completed registry metric payload has invalid metrics.")
    for row in payload:
        if (
            row.get("evaluation_unit") != selection["evaluation_unit"]
            or row.get("target") != selection["target"]
            or row.get("group_column") != selection["group_column"]
            or row.get("outer_group") != selection["outer_group"]
            or row.get("policy_id") != selection["policy"]["policy_id"]
            or row.get("policy_hash") != selection["policy"]["policy_hash"]
            or row.get("frozen_policy_hash") != selection["frozen_policy_hash"]
            or row.get("n_refit") != n_refit
            or row.get("n_test") != n_test
            or row.get("outer_test_prediction_batch") != 1
            or row.get("outer_test_metric_evaluation_count") != 1
        ):
            raise ValueError("Completed registry metric payload binding mismatch.")
    return list(payload)


def _write_unresolved_fold(
    directory: Path,
    *,
    unit: str,
    contract: Any,
    selection: dict[str, Any],
    registry_status: str,
    registry_path: Path,
    fold_hashes: dict[str, str],
) -> None:
    payload = {
        "schema_version": NESTED_OOD_SCHEMA_VERSION,
        "status": "incomplete",
        "evaluation_unit": unit,
        "contract": contract.audit_record,
        "frozen_selection": selection,
        "registry_status": registry_status,
        "registry_record_hash": sha256_file(registry_path),
        "outer_test_prediction_batches_this_invocation": 0,
    }
    document = {"payload": payload, "payload_hash": stable_hash(payload)}
    path = directory / f"{unit}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    fold_hashes[path.name] = sha256_file(path)


def _outer_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, metric), group in metrics.groupby(
        ["target", "metric"],
        sort=True,
    ):
        for weighting in ("group_weighted", "sample_weighted"):
            if weighting == "group_weighted":
                value = float(group["value"].mean())
            else:
                value = float(
                    (group["value"] * group["n_test"]).sum()
                    / group["n_test"].sum()
                )
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "weighting": weighting,
                    "value": value,
                    "n_outer_groups": len(group),
                    "n_test_samples": int(group["n_test"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _rows_by_id(frame: pd.DataFrame, source_ids: tuple[str, ...]) -> pd.DataFrame:
    ids = set(source_ids)
    result = frame.loc[frame["source_row_id"].astype(str).isin(ids)].copy()
    if len(result) != len(ids):
        raise ValueError("Nested OOD source-ID materialization is incomplete.")
    return result


def _identity_frame(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return frame[
        ["source_row_id", "canonical_reaction_key", group_column]
    ].copy()


def _unit(target: str, outer_group: str) -> str:
    return f"{target}-{stable_hash(outer_group)[:16]}"
