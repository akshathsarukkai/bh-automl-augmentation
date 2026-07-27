"""Tests for independent persisted nested-OOD search evidence validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import bh_augmentation.evaluation.nested_ood_artifacts as artifact_module
from bh_augmentation.evaluation.nested_ood import (
    aggregate_inner_ood_metrics,
    build_nested_group_ood_contract,
    build_nested_ood_all_outer_plan,
)
from bh_augmentation.evaluation.nested_ood_artifacts import (
    validate_nested_ood_search_artifacts,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": [f"row-{index:02d}" for index in range(6)],
            "canonical_reaction_key": [
                f"reaction-{index:02d}" for index in range(6)
            ],
            "canonical_product_key": [
                "product-a",
                "product-a",
                "product-b",
                "product-b",
                "product-c",
                "product-c",
            ],
            "yield": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        }
    )


def _policy(policy_id: str, alpha: float) -> dict[str, Any]:
    policy = {
        "policy_id": policy_id,
        "method": "real_only",
        "model": "ridge",
        "params": {"alpha": alpha},
    }
    policy["policy_hash"] = stable_hash(policy)
    return policy


def _unit(outer_group: str) -> str:
    return f"product_key-{stable_hash(outer_group)[:16]}"


def _write_valid_artifacts(tmp_path: Path) -> dict[str, Any]:
    canonical = _canonical_frame()
    directory = tmp_path / "search"
    fold_directory = directory / "fold_manifests"
    fold_directory.mkdir(parents=True)
    target = "product_key"
    group_column = "canonical_product_key"
    groups = sorted(canonical[group_column].unique())
    policies = [_policy("ridge-01", 0.1), _policy("ridge-1", 1.0)]
    metrics = ["rmse", "mae"]
    contracts = [
        build_nested_group_ood_contract(
            canonical,
            group_column=group_column,
            outer_group=group,
        )
        for group in groups
    ]
    plan = build_nested_ood_all_outer_plan(
        canonical,
        iter(contracts),
        group_column=group_column,
    )
    binding = {
        "dataset_hash": stable_hash("dataset"),
        "canonical_split_dependency_hash": stable_hash("canonical split"),
        "canonicalization_version": "bh-rdkit-role-canonicalization-v1",
        "split_schema_version": "bh-canonical-grouped-splits-v1",
        "config_hash": stable_hash("config"),
        "commit_hash": "a" * 40,
    }
    group_rows = [
        {
            "target": target,
            "group_column": group_column,
            "source_row_id": row["source_row_id"],
            "canonical_reaction_key": row["canonical_reaction_key"],
            "group_value": row[group_column],
        }
        for row in canonical.to_dict(orient="records")
    ]
    definition_rows = []
    inner_rows = []
    aggregate_rows = []
    frozen_rows = []
    fold_documents = {}
    expected_keys = [
        {
            "policy_id": policy["policy_id"],
            "policy_hash": policy["policy_hash"],
            "metric": metric,
        }
        for policy in policies
        for metric in metrics
    ]
    for outer_index, contract in enumerate(contracts):
        unit = _unit(contract.outer_group)
        unit_rows = []
        for fold in contract.inner_folds:
            definition_rows.append(
                {
                    "evaluation_unit": unit,
                    "target": target,
                    "group_column": group_column,
                    "outer_group": contract.outer_group,
                    "outer_assignment_hash": contract.outer_assignment_hash,
                    "aggregate_assignment_hash": (
                        contract.aggregate_assignment_hash
                    ),
                    "inner_fold_index": fold.fold_index,
                    "inner_validation_group": fold.validation_group,
                    "n_inner_train": fold.train_size,
                    "n_inner_validation": fold.validation_size,
                    "inner_assignment_hash": fold.assignment_hash,
                    "all_overlaps_zero": True,
                }
            )
            for policy_index, policy in enumerate(policies):
                for metric_index, metric in enumerate(metrics):
                    row = {
                        "evaluation_unit": unit,
                        "target": target,
                        "group_column": group_column,
                        "outer_group": contract.outer_group,
                        "outer_assignment_hash": contract.outer_assignment_hash,
                        "aggregate_assignment_hash": (
                            contract.aggregate_assignment_hash
                        ),
                        "inner_assignment_hash": fold.assignment_hash,
                        "policy_id": policy["policy_id"],
                        "policy_hash": policy["policy_hash"],
                        "model_seed": (
                            outer_index * 100_000
                            + fold.fold_index * 1_000
                            + policy_index
                        ),
                        "metric": metric,
                        "inner_fold_index": fold.fold_index,
                        "inner_validation_group": fold.validation_group,
                        "value": float(
                            1
                            + policy_index * 10
                            + fold.fold_index
                            + metric_index
                        ),
                        "n_validation_samples": fold.validation_size,
                        "outer_test_labels_accessed": False,
                        "outer_test_predictions_generated": False,
                    }
                    inner_rows.append(row)
                    unit_rows.append(row)
        unit_frame = pd.DataFrame(unit_rows)
        unit_aggregates = []
        for weighting in ("group_weighted", "sample_weighted"):
            aggregate = aggregate_inner_ood_metrics(
                contract,
                unit_frame,
                expected_group_keys=expected_keys,
                weighting=weighting,
            )
            aggregate.insert(0, "group_column", group_column)
            aggregate.insert(0, "target", target)
            aggregate.insert(0, "evaluation_unit", unit)
            unit_aggregates.append(aggregate)
            aggregate_rows.extend(aggregate.to_dict(orient="records"))
        combined = pd.concat(unit_aggregates, ignore_index=True)
        selected = combined.loc[
            combined["weighting"].eq("group_weighted")
            & combined["metric"].eq("rmse")
        ].sort_values(["value", "policy_hash", "policy_id"]).iloc[0]
        selected_policy = next(
            policy
            for policy in policies
            if policy["policy_id"] == selected["policy_id"]
        )
        frozen = {
            "evaluation_unit": unit,
            "target": target,
            "group_column": group_column,
            "outer_group": contract.outer_group,
            "outer_assignment_hash": contract.outer_assignment_hash,
            "aggregate_assignment_hash": contract.aggregate_assignment_hash,
            "feature_metadata_hash": stable_hash("features"),
            "policy": selected_policy,
            "selection_metric": "rmse",
            "selection_weighting": "group_weighted",
            "selection_value": float(selected["value"]),
            "lower_is_better": True,
            "dataset_hash": binding["dataset_hash"],
            "canonical_split_dependency_hash": binding[
                "canonical_split_dependency_hash"
            ],
            "config_hash": binding["config_hash"],
            "commit_hash": binding["commit_hash"],
            "target_plan_hash": plan.plan_hash,
        }
        frozen["frozen_policy_hash"] = stable_hash(frozen)
        frozen_rows.append(frozen)
        payload = {
            "schema_version": "bh-nested-group-ood-v1",
            "status": "frozen",
            "contract": contract.audit_record,
            "selection": frozen,
            "inner_metrics_hash": stable_hash(unit_rows),
            "outer_test_labels_accessed": False,
            "outer_test_predictions_generated": False,
            "outer_test_metric_evaluations": 0,
        }
        document = {"unit": unit, "payload": payload}
        document["payload_hash"] = stable_hash(payload)
        fold_documents[f"{unit}.json"] = document

    paths = {
        "group_assignments.csv": directory / "group_assignments.csv",
        "fold_definitions.csv": directory / "fold_definitions.csv",
        "inner_metrics.csv": directory / "inner_metrics.csv",
        "aggregate_metrics.csv": directory / "aggregate_metrics.csv",
        "frozen_policies.json": directory / "frozen_policies.json",
    }
    pd.DataFrame(group_rows).to_csv(paths["group_assignments.csv"], index=False)
    pd.DataFrame(definition_rows).to_csv(paths["fold_definitions.csv"], index=False)
    pd.DataFrame(inner_rows).to_csv(paths["inner_metrics.csv"], index=False)
    pd.DataFrame(aggregate_rows).to_csv(paths["aggregate_metrics.csv"], index=False)
    frozen_document = {
        "schema_version": "bh-nested-ood-search-v1",
        "status": "frozen",
        "candidate_policies": policies,
        "selection_contract": {
            "metric": "rmse",
            "weighting": "group_weighted",
            "lower_is_better": True,
        },
        "policies": frozen_rows,
    }
    frozen_document["frozen_document_hash"] = stable_hash(frozen_document)
    paths["frozen_policies.json"].write_text(
        json.dumps(frozen_document, indent=2, sort_keys=True) + "\n"
    )
    fold_hashes = {}
    for name, document in fold_documents.items():
        path = fold_directory / name
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        fold_hashes[name] = sha256_file(path)
    manifest = {
        "schema_version": "bh-nested-ood-search-v1",
        "status": "complete",
        **binding,
        "targets": [target],
        "target_plans": {target: plan.audit_record},
        "evaluation_units": [row["evaluation_unit"] for row in frozen_rows],
        "candidate_policies": policies,
        "selection_contract": frozen_document["selection_contract"],
        "selection_data_roles": ["inner_train", "inner_validation"],
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metric_evaluations": 0,
        "test_evaluated": False,
        "output_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "fold_manifest_hashes": fold_hashes,
    }
    manifest["search_manifest_hash"] = stable_hash(manifest)
    manifest_path = directory / "search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "canonical": canonical,
        "directory": directory,
        "manifest_path": manifest_path,
        "frozen_path": paths["frozen_policies.json"],
        "binding": binding,
        "policies": policies,
        "metrics": metrics,
    }


def _validate(artifacts: dict[str, Any]) -> object:
    return validate_nested_ood_search_artifacts(
        artifacts["canonical"],
        search_manifest_path=artifacts["manifest_path"],
        frozen_policies_path=artifacts["frozen_path"],
        target_columns={"product_key": "canonical_product_key"},
        expected_policies=artifacts["policies"],
        expected_metrics=artifacts["metrics"],
        expected_selection_metric="rmse",
        expected_selection_weighting="group_weighted",
        expected_lower_is_better=True,
        expected_binding=artifacts["binding"],
    )


def _rehash_output(artifacts: dict[str, Any], name: str) -> None:
    manifest = json.loads(artifacts["manifest_path"].read_text())
    manifest["output_hashes"][name] = sha256_file(
        artifacts["directory"] / name
    )
    manifest.pop("search_manifest_hash")
    manifest["search_manifest_hash"] = stable_hash(manifest)
    artifacts["manifest_path"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _rehash_fold(artifacts: dict[str, Any], name: str) -> None:
    path = artifacts["directory"] / "fold_manifests" / name
    manifest = json.loads(artifacts["manifest_path"].read_text())
    manifest["fold_manifest_hashes"][name] = sha256_file(path)
    manifest.pop("search_manifest_hash")
    manifest["search_manifest_hash"] = stable_hash(manifest)
    artifacts["manifest_path"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_validates_complete_recomputed_search_evidence(tmp_path: Path) -> None:
    artifacts = _write_valid_artifacts(tmp_path)

    validated = _validate(artifacts)

    assert set(validated.frozen_by_unit) == set(
        validated.manifest["evaluation_units"]
    )
    assert set(validated.aggregate_metrics["weighting"]) == {
        "group_weighted",
        "sample_weighted",
    }
    changed = validated.group_assignments
    changed.loc[:, "group_value"] = "changed"
    assert not validated.group_assignments["group_value"].eq("changed").all()


def test_rejects_unknown_search_and_fold_manifest_keys(tmp_path: Path) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    manifest = json.loads(artifacts["manifest_path"].read_text())
    manifest["unknown"] = "value"
    manifest.pop("search_manifest_hash")
    manifest["search_manifest_hash"] = stable_hash(manifest)
    artifacts["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="key-set mismatch"):
        _validate(artifacts)

    artifacts = _write_valid_artifacts(tmp_path / "fold")
    name = next(
        iter(json.loads(artifacts["manifest_path"].read_text())["fold_manifest_hashes"])
    )
    path = artifacts["directory"] / "fold_manifests" / name
    document = json.loads(path.read_text())
    document["payload"]["unknown"] = "value"
    document["payload_hash"] = stable_hash(document["payload"])
    path.write_text(json.dumps(document))
    _rehash_fold(artifacts, name)
    with pytest.raises(ValueError, match="key-set mismatch"):
        _validate(artifacts)


def test_rejects_rehashed_group_identity_and_fold_definition_tampering(
    tmp_path: Path,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    group_path = artifacts["directory"] / "group_assignments.csv"
    groups = pd.read_csv(group_path)
    groups.loc[0, "group_value"] = "laundered-product"
    groups.to_csv(group_path, index=False)
    _rehash_output(artifacts, group_path.name)
    with pytest.raises(ValueError, match="group assignments"):
        _validate(artifacts)

    artifacts = _write_valid_artifacts(tmp_path / "fold-definition")
    definition_path = artifacts["directory"] / "fold_definitions.csv"
    definitions = pd.read_csv(definition_path)
    definitions.loc[0, "n_inner_train"] += 1
    definitions.to_csv(definition_path, index=False)
    _rehash_output(artifacts, definition_path.name)
    with pytest.raises(ValueError, match="fold definitions"):
        _validate(artifacts)


def test_rejects_plausibly_rehashed_aggregate_metrics_not_supported_by_inner_rows(
    tmp_path: Path,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    aggregate_path = artifacts["directory"] / "aggregate_metrics.csv"
    aggregates = pd.read_csv(aggregate_path)
    aggregates.loc[0, "value"] += 0.25
    aggregates.to_csv(aggregate_path, index=False)
    _rehash_output(artifacts, aggregate_path.name)

    with pytest.raises(ValueError, match="aggregate metrics"):
        _validate(artifacts)


def test_rejects_rehashed_fold_contract_and_selection_inconsistency(
    tmp_path: Path,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    manifest = json.loads(artifacts["manifest_path"].read_text())
    name = next(iter(manifest["fold_manifest_hashes"]))
    path = artifacts["directory"] / "fold_manifests" / name
    document = json.loads(path.read_text())
    document["payload"]["contract"]["n_outer_train"] += 1
    document["payload_hash"] = stable_hash(document["payload"])
    path.write_text(json.dumps(document))
    _rehash_fold(artifacts, name)
    with pytest.raises(ValueError, match="Fold manifest is inconsistent"):
        _validate(artifacts)

    artifacts = _write_valid_artifacts(tmp_path / "selection")
    manifest = json.loads(artifacts["manifest_path"].read_text())
    name = next(iter(manifest["fold_manifest_hashes"]))
    path = artifacts["directory"] / "fold_manifests" / name
    document = json.loads(path.read_text())
    document["payload"]["selection"]["selection_value"] += 1.0
    document["payload_hash"] = stable_hash(document["payload"])
    path.write_text(json.dumps(document))
    _rehash_fold(artifacts, name)
    with pytest.raises(ValueError, match="Fold manifest is inconsistent"):
        _validate(artifacts)


def test_rejects_missing_or_unexpected_required_artifact_hash_keys(
    tmp_path: Path,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    manifest = json.loads(artifacts["manifest_path"].read_text())
    manifest["output_hashes"].pop("aggregate_metrics.csv")
    manifest["output_hashes"]["unexpected.csv"] = stable_hash("unexpected")
    manifest.pop("search_manifest_hash")
    manifest["search_manifest_hash"] = stable_hash(manifest)
    artifacts["manifest_path"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="key-set mismatch"):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "selection_contract",
            {
                "metric": "mae",
                "weighting": "group_weighted",
                "lower_is_better": True,
            },
        ),
        ("test_evaluated", True),
    ],
)
def test_search_manifest_enforces_expected_selection_contract_and_no_test_status(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    manifest = json.loads(artifacts["manifest_path"].read_text())
    manifest[field] = value
    manifest.pop("search_manifest_hash")
    manifest["search_manifest_hash"] = stable_hash(manifest)
    artifacts["manifest_path"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest is invalid"):
        _validate(artifacts)


def test_candidate_policy_order_is_trusted_only_when_exact_in_both_documents(
    tmp_path: Path,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    manifest = json.loads(artifacts["manifest_path"].read_text())
    manifest["candidate_policies"] = list(
        reversed(manifest["candidate_policies"])
    )
    manifest.pop("search_manifest_hash")
    manifest["search_manifest_hash"] = stable_hash(manifest)
    artifacts["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest is invalid"):
        _validate(artifacts)

    artifacts = _write_valid_artifacts(tmp_path / "frozen")
    frozen_path = artifacts["frozen_path"]
    frozen = json.loads(frozen_path.read_text())
    frozen["candidate_policies"] = list(reversed(frozen["candidate_policies"]))
    frozen.pop("frozen_document_hash")
    frozen["frozen_document_hash"] = stable_hash(frozen)
    frozen_path.write_text(json.dumps(frozen))
    _rehash_output(artifacts, frozen_path.name)
    with pytest.raises(ValueError, match="policy document is invalid"):
        _validate(artifacts)


def test_contract_and_plan_rebuilds_receive_label_free_identity_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _write_valid_artifacts(tmp_path)
    artifacts["canonical"]["yield"] = artifacts["canonical"]["yield"].astype(
        object
    )
    artifacts["canonical"].loc[:, "yield"] = [
        object() for _ in range(len(artifacts["canonical"]))
    ]
    observed_columns: list[tuple[str, ...]] = []
    production_contract = artifact_module.build_nested_group_ood_contract
    production_plan = artifact_module.build_nested_ood_all_outer_plan

    def inspect_contract(
        frame: pd.DataFrame,
        *,
        group_column: str,
        outer_group: str,
    ) -> object:
        observed_columns.append(tuple(frame.columns))
        return production_contract(
            frame,
            group_column=group_column,
            outer_group=outer_group,
        )

    def inspect_plan(
        frame: pd.DataFrame,
        contracts: object,
        *,
        group_column: str,
    ) -> object:
        observed_columns.append(tuple(frame.columns))
        return production_plan(
            frame,
            contracts,  # type: ignore[arg-type]
            group_column=group_column,
        )

    monkeypatch.setattr(
        artifact_module,
        "build_nested_group_ood_contract",
        inspect_contract,
    )
    monkeypatch.setattr(
        artifact_module,
        "build_nested_ood_all_outer_plan",
        inspect_plan,
    )

    _validate(artifacts)

    assert observed_columns
    assert set(observed_columns) == {
        (
            "source_row_id",
            "canonical_reaction_key",
            "canonical_product_key",
        )
    }
