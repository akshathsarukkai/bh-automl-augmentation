"""Strict persisted-artifact validation for nested group-aware OOD search."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.evaluation.nested_ood import (
    NESTED_OOD_SCHEMA_VERSION,
    NestedGroupOODContract,
    aggregate_inner_ood_metrics,
    build_nested_group_ood_contract,
    build_nested_ood_all_outer_plan,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

SEARCH_SCHEMA_VERSION = "bh-nested-ood-search-v1"
SEARCH_ARTIFACT_NAMES = {
    "group_assignments.csv",
    "fold_definitions.csv",
    "inner_metrics.csv",
    "aggregate_metrics.csv",
    "frozen_policies.json",
}
SEARCH_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "dataset_hash",
    "canonical_split_dependency_hash",
    "canonicalization_version",
    "split_schema_version",
    "config_hash",
    "commit_hash",
    "targets",
    "target_plans",
    "evaluation_units",
    "candidate_policies",
    "selection_contract",
    "selection_data_roles",
    "outer_test_labels_accessed",
    "outer_test_predictions_generated",
    "outer_test_metric_evaluations",
    "test_evaluated",
    "output_hashes",
    "fold_manifest_hashes",
    "search_manifest_hash",
}
FROZEN_DOCUMENT_KEYS = {
    "schema_version",
    "status",
    "candidate_policies",
    "selection_contract",
    "policies",
    "frozen_document_hash",
}
SELECTION_CONTRACT_KEYS = {"metric", "weighting", "lower_is_better"}
FROZEN_POLICY_KEYS = {
    "evaluation_unit",
    "target",
    "group_column",
    "outer_group",
    "outer_assignment_hash",
    "aggregate_assignment_hash",
    "feature_metadata_hash",
    "policy",
    "selection_metric",
    "selection_weighting",
    "selection_value",
    "lower_is_better",
    "dataset_hash",
    "canonical_split_dependency_hash",
    "config_hash",
    "commit_hash",
    "target_plan_hash",
    "frozen_policy_hash",
}
POLICY_KEYS = {
    "policy_id",
    "method",
    "model",
    "params",
    "policy_hash",
}
FOLD_DOCUMENT_KEYS = {"unit", "payload", "payload_hash"}
FOLD_PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "contract",
    "selection",
    "inner_metrics_hash",
    "outer_test_labels_accessed",
    "outer_test_predictions_generated",
    "outer_test_metric_evaluations",
}
CONTRACT_AUDIT_KEYS = {
    "schema_version",
    "group_column",
    "outer_group",
    "n_outer_train",
    "n_outer_test",
    "expected_inner_group_count",
    "observed_inner_fold_count",
    "expected_inner_groups",
    "outer_assignment_hash",
    "per_inner_fold_assignment_hashes",
    "aggregate_assignment_hash",
    "all_inner_overlaps_zero",
}
BINDING_KEYS = {
    "dataset_hash",
    "canonical_split_dependency_hash",
    "canonicalization_version",
    "split_schema_version",
    "config_hash",
    "commit_hash",
}
FROZEN_BINDING_KEYS = {
    "dataset_hash",
    "canonical_split_dependency_hash",
    "config_hash",
    "commit_hash",
}
GROUP_ASSIGNMENT_COLUMNS = (
    "target",
    "group_column",
    "source_row_id",
    "canonical_reaction_key",
    "group_value",
)
FOLD_DEFINITION_COLUMNS = (
    "evaluation_unit",
    "target",
    "group_column",
    "outer_group",
    "outer_assignment_hash",
    "aggregate_assignment_hash",
    "inner_fold_index",
    "inner_validation_group",
    "n_inner_train",
    "n_inner_validation",
    "inner_assignment_hash",
    "all_overlaps_zero",
)
INNER_METRIC_COLUMNS = (
    "evaluation_unit",
    "target",
    "group_column",
    "outer_group",
    "outer_assignment_hash",
    "aggregate_assignment_hash",
    "inner_assignment_hash",
    "policy_id",
    "policy_hash",
    "model_seed",
    "metric",
    "inner_fold_index",
    "inner_validation_group",
    "value",
    "n_validation_samples",
    "outer_test_labels_accessed",
    "outer_test_predictions_generated",
)
AGGREGATE_METRIC_COLUMNS = (
    "evaluation_unit",
    "target",
    "group_column",
    "policy_id",
    "policy_hash",
    "metric",
    "weighting",
    "value",
    "n_inner_folds",
    "n_inner_validation_samples",
    "outer_group",
    "outer_assignment_hash",
    "aggregate_assignment_hash",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedNestedOODSearchArtifacts:
    """Defensively copied search evidence safe for final-evaluation replay."""

    manifest: Mapping[str, Any]
    frozen_policies: tuple[Mapping[str, Any], ...]
    target_plan_hashes: Mapping[str, str]
    _group_assignments: pd.DataFrame = field(repr=False, compare=False)
    _fold_definitions: pd.DataFrame = field(repr=False, compare=False)
    _inner_metrics: pd.DataFrame = field(repr=False, compare=False)
    _aggregate_metrics: pd.DataFrame = field(repr=False, compare=False)

    @property
    def group_assignments(self) -> pd.DataFrame:
        """Return validated compact canonical identities."""
        return self._group_assignments.copy(deep=True)

    @property
    def fold_definitions(self) -> pd.DataFrame:
        """Return validated compact nested fold definitions."""
        return self._fold_definitions.copy(deep=True)

    @property
    def inner_metrics(self) -> pd.DataFrame:
        """Return provenance-bound inner validation metrics."""
        return self._inner_metrics.copy(deep=True)

    @property
    def aggregate_metrics(self) -> pd.DataFrame:
        """Return exactly recomputed aggregate metrics."""
        return self._aggregate_metrics.copy(deep=True)

    @property
    def frozen_by_unit(self) -> dict[str, dict[str, Any]]:
        """Return defensive frozen-policy mappings keyed by evaluation unit."""
        return {
            str(row["evaluation_unit"]): _json_copy(row)
            for row in self.frozen_policies
        }


def validate_nested_ood_search_artifacts(
    canonical: pd.DataFrame,
    *,
    search_manifest_path: str | Path,
    frozen_policies_path: str | Path,
    target_columns: Mapping[str, str],
    expected_policies: Sequence[Mapping[str, Any]],
    expected_metrics: Sequence[str],
    expected_selection_metric: str,
    expected_selection_weighting: str,
    expected_lower_is_better: bool,
    expected_binding: Mapping[str, str],
) -> ValidatedNestedOODSearchArtifacts:
    """Validate all persisted search evidence without trusting saved summaries."""
    manifest_path = Path(search_manifest_path)
    frozen_path = Path(frozen_policies_path)
    if frozen_path.parent != manifest_path.parent or frozen_path.name != "frozen_policies.json":
        raise ValueError(
            "Frozen policies must be the required artifact beside search_manifest.json."
        )
    binding = _strict_mapping(
        expected_binding,
        keys=BINDING_KEYS,
        name="expected scientific binding",
    )
    targets = _validate_target_columns(target_columns, canonical)
    policies = _validate_expected_policies(expected_policies)
    metrics = _validate_expected_metrics(expected_metrics)
    if expected_selection_metric not in metrics:
        raise ValueError("Expected selection metric must be among expected metrics.")
    if expected_selection_weighting not in {
        "group_weighted",
        "sample_weighted",
    }:
        raise ValueError("Expected selection weighting is unsupported.")
    if not isinstance(expected_lower_is_better, bool):
        raise ValueError("Expected lower_is_better must be boolean.")
    expected_selection_contract = {
        "metric": expected_selection_metric,
        "weighting": expected_selection_weighting,
        "lower_is_better": expected_lower_is_better,
    }

    manifest = _load_json_object(manifest_path, name="search manifest")
    _require_exact_keys(manifest, SEARCH_MANIFEST_KEYS, name="search manifest")
    recorded_manifest_hash = manifest["search_manifest_hash"]
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("search_manifest_hash")
    if (
        manifest["schema_version"] != SEARCH_SCHEMA_VERSION
        or manifest["status"] != "complete"
        or stable_hash(unhashed_manifest) != recorded_manifest_hash
        or manifest["targets"] != list(targets)
        or manifest["candidate_policies"] != list(policies)
        or manifest["selection_contract"] != expected_selection_contract
        or manifest["selection_data_roles"]
        != ["inner_train", "inner_validation"]
        or manifest["outer_test_labels_accessed"] is not False
        or manifest["outer_test_predictions_generated"] is not False
        or manifest["outer_test_metric_evaluations"] != 0
        or manifest["test_evaluated"] is not False
    ):
        raise ValueError("Nested OOD search manifest is invalid or incomplete.")
    if any(manifest[key] != binding[key] for key in BINDING_KEYS):
        raise ValueError("Nested OOD search scientific binding mismatch.")
    output_hashes = _strict_hash_mapping(
        manifest["output_hashes"],
        expected_keys=SEARCH_ARTIFACT_NAMES,
        name="search output hashes",
    )
    for name, expected_hash in output_hashes.items():
        path = manifest_path.parent / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Nested OOD search artifact hash mismatch: {name}.")

    frozen_document = _load_json_object(frozen_path, name="frozen policy document")
    frozen_rows = _validate_frozen_document(
        frozen_document,
        targets=targets,
        target_columns=target_columns,
        policies=policies,
        selection_contract=expected_selection_contract,
        selection_metric=expected_selection_metric,
        selection_weighting=expected_selection_weighting,
        lower_is_better=expected_lower_is_better,
        binding=binding,
    )
    units = tuple(str(row["evaluation_unit"]) for row in frozen_rows)
    if len(units) != len(set(units)) or manifest["evaluation_units"] != list(units):
        raise ValueError("Search manifest evaluation units do not match frozen rows.")

    group_assignments = _read_csv_exact(
        manifest_path.parent / "group_assignments.csv",
        columns=GROUP_ASSIGNMENT_COLUMNS,
        name="group assignments",
    )
    _validate_group_assignments(
        canonical,
        group_assignments,
        targets=targets,
        target_columns=target_columns,
    )
    contracts, expected_definitions, plans = _rebuild_contracts(
        canonical,
        frozen_rows=frozen_rows,
        targets=targets,
        target_columns=target_columns,
    )
    if manifest["target_plans"] != {
        target: plans[target].audit_record for target in targets
    }:
        raise ValueError("Persisted nested OOD target plans are not reproducible.")
    for row in frozen_rows:
        if row["target_plan_hash"] != plans[str(row["target"])].plan_hash:
            raise ValueError("Frozen nested OOD target plan hash mismatch.")

    fold_definitions = _read_csv_exact(
        manifest_path.parent / "fold_definitions.csv",
        columns=FOLD_DEFINITION_COLUMNS,
        name="fold definitions",
    )
    _compare_records_exact(
        fold_definitions,
        pd.DataFrame(expected_definitions, columns=FOLD_DEFINITION_COLUMNS),
        sort_columns=("target", "evaluation_unit", "inner_fold_index"),
        name="fold definitions",
    )
    inner_metrics = _read_csv_exact(
        manifest_path.parent / "inner_metrics.csv",
        columns=INNER_METRIC_COLUMNS,
        name="inner metrics",
    )
    if (
        not _all_exact_bool(inner_metrics["outer_test_labels_accessed"], False)
        or not _all_exact_bool(
            inner_metrics["outer_test_predictions_generated"],
            False,
        )
    ):
        raise ValueError("Inner metrics report forbidden outer-test access.")

    expected_keys = [
        {
            "policy_id": policy["policy_id"],
            "policy_hash": policy["policy_hash"],
            "metric": metric,
        }
        for policy in policies
        for metric in metrics
    ]
    recomputed_aggregates: list[pd.DataFrame] = []
    aggregate_by_unit: dict[str, pd.DataFrame] = {}
    frozen_by_unit = {
        str(row["evaluation_unit"]): row for row in frozen_rows
    }
    for unit in units:
        rows = inner_metrics.loc[
            inner_metrics["evaluation_unit"].astype(str).eq(unit)
        ].copy()
        selection = frozen_by_unit[unit]
        contract = contracts[unit]
        if (
            rows.empty
            or not rows["target"].astype(str).eq(selection["target"]).all()
            or not rows["group_column"]
            .astype(str)
            .eq(selection["group_column"])
            .all()
        ):
            raise ValueError("Inner metric unit metadata is incomplete or inconsistent.")
        unit_aggregates = []
        for weighting in ("group_weighted", "sample_weighted"):
            aggregate = aggregate_inner_ood_metrics(
                contract,
                rows,
                expected_group_keys=expected_keys,
                group_columns=("policy_id", "policy_hash", "metric"),
                weighting=weighting,
            )
            aggregate.insert(0, "group_column", selection["group_column"])
            aggregate.insert(0, "target", selection["target"])
            aggregate.insert(0, "evaluation_unit", unit)
            unit_aggregates.append(aggregate)
            recomputed_aggregates.append(aggregate)
        aggregate_by_unit[unit] = pd.concat(unit_aggregates, ignore_index=True)
        _validate_selection(
            selection,
            aggregate_by_unit[unit],
            expected_policies=policies,
        )

    aggregate_metrics = _read_csv_exact(
        manifest_path.parent / "aggregate_metrics.csv",
        columns=AGGREGATE_METRIC_COLUMNS,
        name="aggregate metrics",
    )
    recomputed = pd.concat(recomputed_aggregates, ignore_index=True)
    _compare_records_exact(
        aggregate_metrics,
        recomputed,
        sort_columns=(
            "target",
            "evaluation_unit",
            "policy_hash",
            "metric",
            "weighting",
        ),
        name="aggregate metrics",
    )
    _validate_fold_manifests(
        manifest_path.parent / "fold_manifests",
        recorded_hashes=manifest["fold_manifest_hashes"],
        units=units,
        frozen_by_unit=frozen_by_unit,
        contracts=contracts,
        inner_metrics=inner_metrics,
    )
    return ValidatedNestedOODSearchArtifacts(
        manifest=_json_copy(manifest),
        frozen_policies=tuple(_json_copy(row) for row in frozen_rows),
        target_plan_hashes={
            target: plans[target].plan_hash for target in targets
        },
        _group_assignments=group_assignments.copy(deep=True),
        _fold_definitions=fold_definitions.copy(deep=True),
        _inner_metrics=inner_metrics.copy(deep=True),
        _aggregate_metrics=aggregate_metrics.copy(deep=True),
    )


def _validate_frozen_document(
    document: Mapping[str, Any],
    *,
    targets: tuple[str, ...],
    target_columns: Mapping[str, str],
    policies: tuple[dict[str, Any], ...],
    selection_contract: Mapping[str, Any],
    selection_metric: str,
    selection_weighting: str,
    lower_is_better: bool,
    binding: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    _require_exact_keys(
        document,
        FROZEN_DOCUMENT_KEYS,
        name="frozen policy document",
    )
    unhashed = dict(document)
    recorded = unhashed.pop("frozen_document_hash")
    if (
        document["schema_version"] != SEARCH_SCHEMA_VERSION
        or document["status"] != "frozen"
        or document["candidate_policies"] != list(policies)
        or document["selection_contract"] != selection_contract
        or not isinstance(document["policies"], list)
        or stable_hash(unhashed) != recorded
    ):
        raise ValueError("Frozen nested OOD policy document is invalid.")
    _strict_mapping(
        document["selection_contract"],
        keys=SELECTION_CONTRACT_KEYS,
        name="frozen selection contract",
    )
    expected_policy_by_id = {
        policy["policy_id"]: policy for policy in policies
    }
    rows: list[dict[str, Any]] = []
    for value in document["policies"]:
        row = _strict_mapping(value, keys=FROZEN_POLICY_KEYS, name="frozen policy")
        policy = _strict_mapping(
            row["policy"],
            keys=POLICY_KEYS,
            name="selected policy",
        )
        unhashed_row = dict(row)
        recorded_row_hash = unhashed_row.pop("frozen_policy_hash")
        if (
            stable_hash(unhashed_row) != recorded_row_hash
            or row["target"] not in targets
            or row["group_column"] != target_columns[row["target"]]
            or policy != expected_policy_by_id.get(policy["policy_id"])
            or row["selection_metric"] != selection_metric
            or row["selection_weighting"] != selection_weighting
            or row["lower_is_better"] is not lower_is_better
            or not _is_finite_number(row["selection_value"])
            or any(
                row[key] != binding[key] for key in FROZEN_BINDING_KEYS
            )
        ):
            raise ValueError("Frozen nested OOD policy row is invalid.")
        rows.append(_json_copy(row))
    return tuple(rows)


def _validate_group_assignments(
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    target_columns: Mapping[str, str],
) -> None:
    required = {"source_row_id", "canonical_reaction_key", *target_columns.values()}
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Canonical data is missing identity columns: {missing}.")
    canonical_frame = canonical.copy()
    for column in required:
        if canonical_frame[column].isna().any():
            raise ValueError(f"Canonical identity column {column} contains missing values.")
        canonical_frame[column] = canonical_frame[column].astype(str)
    if canonical_frame["source_row_id"].duplicated().any():
        raise ValueError("Canonical source_row_id values must be unique.")
    expected_rows = []
    for target in targets:
        group_column = target_columns[target]
        for row in canonical_frame[
            ["source_row_id", "canonical_reaction_key", group_column]
        ].to_dict(orient="records"):
            expected_rows.append(
                {
                    "target": target,
                    "group_column": group_column,
                    "source_row_id": row["source_row_id"],
                    "canonical_reaction_key": row["canonical_reaction_key"],
                    "group_value": row[group_column],
                }
            )
    _compare_records_exact(
        assignments,
        pd.DataFrame(expected_rows, columns=GROUP_ASSIGNMENT_COLUMNS),
        sort_columns=("target", "source_row_id"),
        name="group assignments",
    )


def _identity_frame(
    canonical: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Return the only label-free columns accepted by assignment rebuilds."""
    required = ["source_row_id", "canonical_reaction_key", group_column]
    missing = sorted(set(required) - set(canonical))
    if missing:
        raise ValueError(f"Canonical data is missing identity columns: {missing}.")
    return canonical[required].copy(deep=True)


def _rebuild_contracts(
    canonical: pd.DataFrame,
    *,
    frozen_rows: tuple[dict[str, Any], ...],
    targets: tuple[str, ...],
    target_columns: Mapping[str, str],
) -> tuple[
    dict[str, NestedGroupOODContract],
    list[dict[str, Any]],
    dict[str, Any],
]:
    contracts: dict[str, NestedGroupOODContract] = {}
    definition_rows: list[dict[str, Any]] = []
    plans: dict[str, Any] = {}
    rows_by_target = {
        target: [row for row in frozen_rows if row["target"] == target]
        for target in targets
    }
    for target in targets:
        group_column = target_columns[target]
        identity = _identity_frame(canonical, group_column)
        expected_groups = sorted(identity[group_column].astype(str).unique())
        observed_groups = [str(row["outer_group"]) for row in rows_by_target[target]]
        if observed_groups != expected_groups:
            raise ValueError(
                f"Frozen rows do not cover sorted outer groups for target {target!r}."
            )
        target_contracts = []
        for row in rows_by_target[target]:
            unit = str(row["evaluation_unit"])
            if unit in contracts:
                raise ValueError("Duplicate frozen nested OOD evaluation unit.")
            contract = build_nested_group_ood_contract(
                identity,
                group_column=group_column,
                outer_group=str(row["outer_group"]),
            )
            if (
                row["outer_assignment_hash"] != contract.outer_assignment_hash
                or row["aggregate_assignment_hash"]
                != contract.aggregate_assignment_hash
            ):
                raise ValueError("Frozen assignment hashes do not match canonical data.")
            contracts[unit] = contract
            target_contracts.append(contract)
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
                        "all_overlaps_zero": fold.overlap_audit[
                            "all_overlaps_zero"
                        ],
                    }
                )
        plans[target] = build_nested_ood_all_outer_plan(
            identity,
            iter(target_contracts),
            group_column=group_column,
        )
    return contracts, definition_rows, plans


def _validate_selection(
    selection: Mapping[str, Any],
    aggregates: pd.DataFrame,
    *,
    expected_policies: tuple[dict[str, Any], ...],
) -> None:
    candidates = aggregates.loc[
        aggregates["weighting"].eq(selection["selection_weighting"])
        & aggregates["metric"].eq(selection["selection_metric"])
    ].sort_values(
        ["value", "policy_hash", "policy_id"],
        ascending=[selection["lower_is_better"], True, True],
        kind="mergesort",
    )
    if candidates.empty:
        raise ValueError("Frozen selection has no reproducible aggregate candidate.")
    winner = candidates.iloc[0]
    expected_by_id = {
        policy["policy_id"]: policy for policy in expected_policies
    }
    if (
        winner["policy_id"] != selection["policy"]["policy_id"]
        or winner["policy_hash"] != selection["policy"]["policy_hash"]
        or selection["policy"]
        != expected_by_id.get(selection["policy"]["policy_id"])
        or not math.isclose(
            float(winner["value"]),
            float(selection["selection_value"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Frozen nested OOD selection is not exactly reproducible.")


def _validate_fold_manifests(
    directory: Path,
    *,
    recorded_hashes: Any,
    units: tuple[str, ...],
    frozen_by_unit: Mapping[str, Mapping[str, Any]],
    contracts: Mapping[str, NestedGroupOODContract],
    inner_metrics: pd.DataFrame,
) -> None:
    expected_names = {f"{unit}.json" for unit in units}
    hashes = _strict_hash_mapping(
        recorded_hashes,
        expected_keys=expected_names,
        name="fold manifest hashes",
    )
    if not directory.is_dir() or {
        path.name for path in directory.iterdir() if path.is_file()
    } != expected_names:
        raise ValueError("Fold manifest directory contents are incomplete or unexpected.")
    for unit in units:
        name = f"{unit}.json"
        path = directory / name
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"Fold manifest hash mismatch: {name}.")
        document = _load_json_object(path, name="fold manifest")
        _require_exact_keys(document, FOLD_DOCUMENT_KEYS, name="fold manifest")
        payload = _strict_mapping(
            document["payload"],
            keys=FOLD_PAYLOAD_KEYS,
            name="fold manifest payload",
        )
        contract_audit = _strict_mapping(
            payload["contract"],
            keys=CONTRACT_AUDIT_KEYS,
            name="fold manifest contract",
        )
        unit_rows = inner_metrics.loc[
            inner_metrics["evaluation_unit"].astype(str).eq(unit)
        ].to_dict(orient="records")
        if (
            document["unit"] != unit
            or stable_hash(payload) != document["payload_hash"]
            or payload["schema_version"] != NESTED_OOD_SCHEMA_VERSION
            or payload["status"] != "frozen"
            or contract_audit != contracts[unit].audit_record
            or payload["selection"] != frozen_by_unit[unit]
            or payload["inner_metrics_hash"] != stable_hash(unit_rows)
            or payload["outer_test_labels_accessed"] is not False
            or payload["outer_test_predictions_generated"] is not False
            or payload["outer_test_metric_evaluations"] != 0
        ):
            raise ValueError(f"Fold manifest is inconsistent: {name}.")


def _validate_target_columns(
    target_columns: Mapping[str, str],
    canonical: pd.DataFrame,
) -> tuple[str, ...]:
    if not isinstance(target_columns, Mapping) or not target_columns:
        raise ValueError("target_columns must be a non-empty mapping.")
    targets = tuple(target_columns)
    if any(
        not isinstance(target, str)
        or not target
        or not isinstance(column, str)
        or not column
        or column not in canonical
        for target, column in target_columns.items()
    ):
        raise ValueError("target_columns contains an invalid target or canonical column.")
    return targets


def _validate_expected_policies(
    expected_policies: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(expected_policies, Sequence) or isinstance(
        expected_policies,
        (str, bytes),
    ) or not expected_policies:
        raise ValueError("expected_policies must be a non-empty sequence.")
    policies = tuple(
        _strict_mapping(policy, keys=POLICY_KEYS, name="expected policy")
        for policy in expected_policies
    )
    if len({policy["policy_id"] for policy in policies}) != len(policies):
        raise ValueError("Expected policy IDs must be unique.")
    for policy in policies:
        unhashed = dict(policy)
        recorded = unhashed.pop("policy_hash")
        if stable_hash(unhashed) != recorded:
            raise ValueError("Expected policy hash mismatch.")
    return tuple(_json_copy(policy) for policy in policies)


def _validate_expected_metrics(expected_metrics: Sequence[str]) -> tuple[str, ...]:
    if (
        not isinstance(expected_metrics, Sequence)
        or isinstance(expected_metrics, (str, bytes))
        or not expected_metrics
        or any(not isinstance(metric, str) or not metric for metric in expected_metrics)
        or len(expected_metrics) != len(set(expected_metrics))
    ):
        raise ValueError("expected_metrics must contain unique non-empty names.")
    return tuple(expected_metrics)


def _read_csv_exact(
    path: Path,
    *,
    columns: tuple[str, ...],
    name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) != columns or frame.empty:
        raise ValueError(
            f"Persisted {name} schema/content mismatch: "
            f"expected={list(columns)}, observed={list(frame.columns)}."
        )
    return frame


def _compare_records_exact(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: tuple[str, ...],
    name: str,
) -> None:
    actual_records = (
        actual.sort_values(list(sort_columns), kind="mergesort")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    expected_records = (
        expected.sort_values(list(sort_columns), kind="mergesort")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    if stable_hash(actual_records) != stable_hash(expected_records):
        raise ValueError(f"Persisted {name} do not exactly match recomputed evidence.")


def _strict_hash_mapping(
    value: Any,
    *,
    expected_keys: set[str],
    name: str,
) -> dict[str, str]:
    mapping = _strict_mapping(value, keys=expected_keys, name=name)
    if any(not isinstance(item, str) or not item for item in mapping.values()):
        raise ValueError(f"{name} values must be non-empty strings.")
    return dict(mapping)


def _strict_mapping(
    value: Any,
    *,
    keys: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    _require_exact_keys(value, keys, name=name)
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} key-set mismatch: expected={sorted(expected)}, "
            f"observed={sorted(value)}."
        )


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load valid {name}: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return value


def _all_exact_bool(values: pd.Series, expected: bool) -> bool:
    return all(
        isinstance(value, bool) and value is expected
        for value in values.tolist()
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))
