"""Validation-only nested group-aware OOD policy search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.nested_ood import (
    NESTED_OOD_SCHEMA_VERSION,
    NestedGroupOODContract,
    aggregate_inner_ood_metrics,
    build_nested_group_ood_contract,
    build_nested_ood_all_outer_plan,
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
from bh_augmentation.policy_search import current_commit
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

NESTED_OOD_SEARCH_SCHEMA_VERSION = "bh-nested-ood-search-v1"


def run_nested_ood_search(
    config_path: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Select and freeze one policy per target and outer held-out group."""
    config = load_config(config_path)
    resolved = resolve_contract(config)
    raw_output = (
        output_directory
        if output_directory is not None
        else config.get("output", {}).get(
            "nested_ood_search_directory",
            "results/corrected_nested_ood_search_example_only",
        )
    )
    output_path = assert_fresh_output(raw_output)
    saved = load_saved_canonical_splits(
        resolved["dataset_path"],
        resolved["split_directory"],
    )
    config_hash = stable_hash(scientific_projection(config))
    commit = current_commit()
    group_rows: list[dict[str, Any]] = []
    definition_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    fold_manifests: list[dict[str, Any]] = []
    target_plans: dict[str, dict[str, Any]] = {}

    for target in resolved["targets"]:
        group_column = TARGET_COLUMNS[target]
        identity = _identity_frame(saved.canonical, group_column)
        target_groups = sorted(identity[group_column].astype(str).unique())
        for row in identity.to_dict(orient="records"):
            group_rows.append(
                {
                    "target": target,
                    "group_column": group_column,
                    "source_row_id": str(row["source_row_id"]),
                    "canonical_reaction_key": str(row["canonical_reaction_key"]),
                    "group_value": str(row[group_column]),
                }
            )
        for outer_index, outer_group in enumerate(target_groups):
            contract = build_nested_group_ood_contract(
                identity,
                group_column=group_column,
                outer_group=outer_group,
            )
            outer_train = _rows_by_id(
                saved.canonical,
                contract.outer_train_source_ids,
            )
            result = _search_outer(
                outer_train=outer_train,
                contract=contract,
                target=target,
                outer_index=outer_index,
                resolved=resolved,
                dataset_hash=saved.dataset_hash,
                split_dependency_hash=saved.aggregate_split_hash,
                config_hash=config_hash,
                commit=commit,
            )
            definition_rows.extend(result["definitions"])
            inner_rows.extend(result["inner_metrics"])
            aggregate_rows.extend(result["aggregates"])
            frozen_rows.append(result["frozen"])
            fold_manifests.append(result["fold_manifest"])
        target_plan = build_nested_ood_all_outer_plan(
            identity,
            (
                build_nested_group_ood_contract(
                    identity,
                    group_column=group_column,
                    outer_group=outer_group,
                )
                for outer_group in target_groups
            ),
            group_column=group_column,
        ).audit_record
        target_plans[target] = target_plan
        for row in frozen_rows:
            if row["target"] == target:
                row.pop("frozen_policy_hash")
                row["target_plan_hash"] = target_plan["plan_hash"]
                row["frozen_policy_hash"] = stable_hash(row)

    output = create_fresh_output(output_path)
    fold_directory = output / "fold_manifests"
    fold_directory.mkdir()
    paths = {
        "directory": output,
        "group_assignments": output / "group_assignments.csv",
        "fold_definitions": output / "fold_definitions.csv",
        "inner_metrics": output / "inner_metrics.csv",
        "aggregate_metrics": output / "aggregate_metrics.csv",
        "frozen_policies": output / "frozen_policies.json",
        "search_manifest": output / "search_manifest.json",
    }
    pd.DataFrame(group_rows).to_csv(paths["group_assignments"], index=False)
    pd.DataFrame(definition_rows).to_csv(paths["fold_definitions"], index=False)
    pd.DataFrame(inner_rows).to_csv(paths["inner_metrics"], index=False)
    pd.DataFrame(aggregate_rows).to_csv(paths["aggregate_metrics"], index=False)
    frozen_document = {
        "schema_version": NESTED_OOD_SEARCH_SCHEMA_VERSION,
        "status": "frozen",
        "candidate_policies": resolved["policies"],
        "selection_contract": {
            "metric": resolved["selection_metric"],
            "weighting": resolved["selection_weighting"],
            "lower_is_better": resolved["lower_is_better"],
        },
        "policies": frozen_rows,
    }
    frozen_document["frozen_document_hash"] = stable_hash(frozen_document)
    paths["frozen_policies"].write_text(
        json.dumps(frozen_document, indent=2, sort_keys=True) + "\n"
    )
    fold_hashes = {}
    for record in fold_manifests:
        path = fold_directory / f"{record['unit']}.json"
        document = dict(record)
        document["payload_hash"] = stable_hash(document["payload"])
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        fold_hashes[path.name] = sha256_file(path)
    manifest_payload = {
        "schema_version": NESTED_OOD_SEARCH_SCHEMA_VERSION,
        "status": "complete",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "canonicalization_version": saved.manifest["canonicalization_version"],
        "split_schema_version": saved.manifest["split_schema_version"],
        "config_hash": config_hash,
        "commit_hash": commit,
        "targets": list(resolved["targets"]),
        "target_plans": target_plans,
        "evaluation_units": [row["evaluation_unit"] for row in frozen_rows],
        "candidate_policies": resolved["policies"],
        "selection_contract": frozen_document["selection_contract"],
        "selection_data_roles": ["inner_train", "inner_validation"],
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metric_evaluations": 0,
        "test_evaluated": False,
        "output_hashes": {
            path.name: sha256_file(path)
            for name, path in paths.items()
            if name not in {"directory", "search_manifest"}
        },
        "fold_manifest_hashes": fold_hashes,
    }
    manifest_payload["search_manifest_hash"] = stable_hash(manifest_payload)
    paths["search_manifest"].write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    )
    return paths


def _search_outer(
    *,
    outer_train: pd.DataFrame,
    contract: NestedGroupOODContract,
    target: str,
    outer_index: int,
    resolved: dict[str, Any],
    dataset_hash: str,
    split_dependency_hash: str,
    config_hash: str,
    commit: str,
) -> dict[str, Any]:
    """Search one outer fold with no labeled object beyond outer training."""
    if set(outer_train["source_row_id"].astype(str)) != set(
        contract.outer_train_source_ids
    ):
        raise ValueError("Labeled outer-training frame does not match its contract.")
    unit = _unit(target, contract.outer_group)
    definitions = []
    metric_rows = []
    unit_feature_hash: str | None = None
    for inner_fold in contract.inner_folds:
        definitions.append(
            {
                "evaluation_unit": unit,
                "target": target,
                "group_column": contract.group_column,
                "outer_group": contract.outer_group,
                "outer_assignment_hash": contract.outer_assignment_hash,
                "aggregate_assignment_hash": contract.aggregate_assignment_hash,
                "inner_fold_index": inner_fold.fold_index,
                "inner_validation_group": inner_fold.validation_group,
                "n_inner_train": inner_fold.train_size,
                "n_inner_validation": inner_fold.validation_size,
                "inner_assignment_hash": inner_fold.assignment_hash,
                "all_overlaps_zero": inner_fold.overlap_audit["all_overlaps_zero"],
            }
        )
        train_frame = _rows_by_id(outer_train, inner_fold.train_source_ids)
        validation_frame = _rows_by_id(
            outer_train,
            inner_fold.validation_source_ids,
        )
        X_train, y_train, train_feature = build_partition(
            train_frame,
            resolved["feature_config"],
        )
        X_validation, y_validation, validation_feature = build_partition(
            validation_frame,
            resolved["feature_config"],
        )
        if train_feature != validation_feature:
            raise ValueError("Nested OOD inner feature contracts differ.")
        feature_hash = str(train_feature["feature_metadata_hash"])
        if unit_feature_hash is None:
            unit_feature_hash = feature_hash
        elif unit_feature_hash != feature_hash:
            raise ValueError("Nested OOD feature contract changed across folds.")
        for policy_index, policy in enumerate(resolved["policies"]):
            model_seed = (
                resolved["seed"]
                + outer_index * 100_000
                + inner_fold.fold_index * 1_000
                + policy_index
            )
            model = fit_policy(policy, X_train, y_train, seed=model_seed)
            prediction = predict_model(model, X_validation)
            for metric in resolved["metrics"]:
                metric_rows.append(
                    {
                        "evaluation_unit": unit,
                        "target": target,
                        "group_column": contract.group_column,
                        "outer_group": contract.outer_group,
                        "outer_assignment_hash": contract.outer_assignment_hash,
                        "aggregate_assignment_hash": (
                            contract.aggregate_assignment_hash
                        ),
                        "inner_assignment_hash": inner_fold.assignment_hash,
                        "policy_id": policy["policy_id"],
                        "policy_hash": policy["policy_hash"],
                        "model_seed": model_seed,
                        "metric": metric,
                        "inner_fold_index": inner_fold.fold_index,
                        "inner_validation_group": inner_fold.validation_group,
                        "value": metric_value(
                            metric,
                            y_validation,
                            prediction,
                            context=(
                                f"target={target!r}, "
                                f"outer_group={contract.outer_group!r}, "
                                f"inner_group={inner_fold.validation_group!r}, "
                                f"policy={policy['policy_id']!r}"
                            ),
                        ),
                        "n_validation_samples": inner_fold.validation_size,
                        "outer_test_labels_accessed": False,
                        "outer_test_predictions_generated": False,
                    }
                )
    unit_frame = pd.DataFrame(metric_rows)
    expected_keys = [
        {
            "policy_id": policy["policy_id"],
            "policy_hash": policy["policy_hash"],
            "metric": metric,
        }
        for policy in resolved["policies"]
        for metric in resolved["metrics"]
    ]
    aggregates = []
    unit_aggregates = []
    for weighting in ("group_weighted", "sample_weighted"):
        aggregate = aggregate_inner_ood_metrics(
            contract,
            unit_frame,
            expected_group_keys=expected_keys,
            group_columns=("policy_id", "policy_hash", "metric"),
            weighting=weighting,
        )
        aggregate.insert(0, "group_column", contract.group_column)
        aggregate.insert(0, "target", target)
        aggregate.insert(0, "evaluation_unit", unit)
        unit_aggregates.append(aggregate)
        aggregates.extend(aggregate.to_dict(orient="records"))
    combined = pd.concat(unit_aggregates, ignore_index=True)
    candidates = combined.loc[
        combined["weighting"].eq(resolved["selection_weighting"])
        & combined["metric"].eq(resolved["selection_metric"])
    ].sort_values(
        ["value", "policy_hash", "policy_id"],
        ascending=[resolved["lower_is_better"], True, True],
        kind="mergesort",
    )
    selected = candidates.iloc[0]
    selected_policy = next(
        policy
        for policy in resolved["policies"]
        if policy["policy_id"] == selected["policy_id"]
    )
    frozen = {
        "evaluation_unit": unit,
        "target": target,
        "group_column": contract.group_column,
        "outer_group": contract.outer_group,
        "outer_assignment_hash": contract.outer_assignment_hash,
        "aggregate_assignment_hash": contract.aggregate_assignment_hash,
        "feature_metadata_hash": unit_feature_hash,
        "policy": selected_policy,
        "selection_metric": resolved["selection_metric"],
        "selection_weighting": resolved["selection_weighting"],
        "selection_value": float(selected["value"]),
        "lower_is_better": resolved["lower_is_better"],
        "dataset_hash": dataset_hash,
        "canonical_split_dependency_hash": split_dependency_hash,
        "config_hash": config_hash,
        "commit_hash": commit,
    }
    frozen["frozen_policy_hash"] = stable_hash(frozen)
    fold_payload = {
        "schema_version": NESTED_OOD_SCHEMA_VERSION,
        "status": "frozen",
        "contract": contract.audit_record,
        "selection": frozen,
        "inner_metrics_hash": stable_hash(metric_rows),
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metric_evaluations": 0,
    }
    return {
        "definitions": definitions,
        "inner_metrics": metric_rows,
        "aggregates": aggregates,
        "frozen": frozen,
        "fold_manifest": {"unit": unit, "payload": fold_payload},
    }


def _identity_frame(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Return the label-free identity table accepted by split contracts."""
    return frame[
        ["source_row_id", "canonical_reaction_key", group_column]
    ].copy()


def _rows_by_id(frame: pd.DataFrame, source_ids: tuple[str, ...]) -> pd.DataFrame:
    ids = set(source_ids)
    result = frame.loc[frame["source_row_id"].astype(str).isin(ids)].copy()
    if len(result) != len(ids):
        raise ValueError("Nested OOD source-ID materialization is incomplete.")
    return result


def _unit(target: str, outer_group: str) -> str:
    return f"{target}-{stable_hash(outer_group)[:16]}"
