"""Deterministic nested group-aware OOD assignment and aggregation contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.utils.corrected_runs import stable_hash

NESTED_OOD_SCHEMA_VERSION = "bh-nested-group-ood-v1"
OUTER_ASSIGNMENT_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "group_column",
    "group_value",
    "outer_group",
    "outer_split",
)
INNER_ASSIGNMENT_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "group_column",
    "group_value",
    "outer_group",
    "inner_fold_index",
    "inner_validation_group",
    "inner_split",
)
_WEIGHTINGS = {"group_weighted", "sample_weighted"}


@dataclass(frozen=True, slots=True)
class NestedOODInnerFold:
    """Immutable membership and overlap audit for one inner OOD fold."""

    fold_index: int
    validation_group: str
    train_source_ids: tuple[str, ...]
    validation_source_ids: tuple[str, ...]
    train_size: int
    validation_size: int
    assignment_hash: str
    source_id_overlap_count: int
    group_overlap_count: int
    canonical_reaction_key_overlap_count: int

    @property
    def overlap_audit(self) -> dict[str, Any]:
        """Return complete group-disjointness metadata for the inner fold."""
        return {
            "inner_fold_index": self.fold_index,
            "inner_validation_group": self.validation_group,
            "n_inner_train": self.train_size,
            "n_inner_validation": self.validation_size,
            "source_id_overlap_count": self.source_id_overlap_count,
            "group_overlap_count": self.group_overlap_count,
            "canonical_reaction_key_overlap_count": (
                self.canonical_reaction_key_overlap_count
            ),
            "all_overlaps_zero": (
                self.source_id_overlap_count == 0
                and self.group_overlap_count == 0
                and self.canonical_reaction_key_overlap_count == 0
            ),
            "assignment_hash": self.assignment_hash,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class NestedGroupOODContract:
    """Validated assignments for one outer held-out group and its inner LOGO."""

    group_column: str
    outer_group: str
    outer_train_source_ids: tuple[str, ...]
    outer_test_source_ids: tuple[str, ...]
    expected_inner_groups: tuple[str, ...]
    inner_folds: tuple[NestedOODInnerFold, ...]
    outer_assignment_hash: str
    aggregate_assignment_hash: str
    _outer_assignments: pd.DataFrame = field(repr=False, compare=False)
    _inner_assignments: pd.DataFrame = field(repr=False, compare=False)
    _inner_overlap_audit: pd.DataFrame = field(repr=False, compare=False)

    @property
    def inner_fold_count(self) -> int:
        """Return the number of validated inner held-out-group folds."""
        return len(self.inner_folds)

    @property
    def outer_assignments(self) -> pd.DataFrame:
        """Return a defensive copy of outer train/test assignments."""
        return self._outer_assignments.copy(deep=True)

    @property
    def inner_assignments(self) -> pd.DataFrame:
        """Return a defensive copy of search-only inner assignments."""
        return self._inner_assignments.copy(deep=True)

    @property
    def inner_overlap_audit(self) -> pd.DataFrame:
        """Return a defensive copy of inner group-overlap checks."""
        return self._inner_overlap_audit.copy(deep=True)

    @property
    def per_inner_fold_assignment_hashes(self) -> dict[str, str]:
        """Return stable hashes keyed by inner validation group."""
        return {
            fold.validation_group: fold.assignment_hash
            for fold in self.inner_folds
        }

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return concise reproducibility metadata for a fold manifest."""
        return {
            "schema_version": NESTED_OOD_SCHEMA_VERSION,
            "group_column": self.group_column,
            "outer_group": self.outer_group,
            "n_outer_train": len(self.outer_train_source_ids),
            "n_outer_test": len(self.outer_test_source_ids),
            "expected_inner_group_count": len(self.expected_inner_groups),
            "observed_inner_fold_count": self.inner_fold_count,
            "expected_inner_groups": list(self.expected_inner_groups),
            "outer_assignment_hash": self.outer_assignment_hash,
            "per_inner_fold_assignment_hashes": (
                self.per_inner_fold_assignment_hashes
            ),
            "aggregate_assignment_hash": self.aggregate_assignment_hash,
            "all_inner_overlaps_zero": bool(
                self._inner_overlap_audit["all_overlaps_zero"].all()
            ),
        }


@dataclass(frozen=True, slots=True)
class NestedOODOuterFoldDefinition:
    """Compact immutable definition of one outer OOD fold."""

    outer_fold_index: int
    outer_group: str
    outer_train_size: int
    outer_test_size: int
    expected_inner_group_count: int
    aggregate_assignment_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compact fold definition."""
        return {
            "outer_fold_index": self.outer_fold_index,
            "outer_group": self.outer_group,
            "outer_train_size": self.outer_train_size,
            "outer_test_size": self.outer_test_size,
            "expected_inner_group_count": self.expected_inner_group_count,
            "aggregate_assignment_hash": self.aggregate_assignment_hash,
        }


@dataclass(frozen=True, slots=True)
class NestedOODAllOuterPlan:
    """Compact validated plan covering every outer group exactly once."""

    group_column: str
    canonical_source_id_hash: str
    expected_outer_groups: tuple[str, ...]
    fold_definitions: tuple[NestedOODOuterFoldDefinition, ...]
    plan_hash: str

    @property
    def outer_fold_count(self) -> int:
        """Return the number of planned outer folds."""
        return len(self.fold_definitions)

    @property
    def per_outer_aggregate_assignment_hashes(self) -> dict[str, str]:
        """Return nested assignment hashes keyed by outer group."""
        return {
            fold.outer_group: fold.aggregate_assignment_hash
            for fold in self.fold_definitions
        }

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return compact all-outer reproducibility metadata."""
        return {
            "schema_version": NESTED_OOD_SCHEMA_VERSION,
            "group_column": self.group_column,
            "canonical_source_id_hash": self.canonical_source_id_hash,
            "expected_outer_group_count": len(self.expected_outer_groups),
            "observed_outer_fold_count": self.outer_fold_count,
            "expected_outer_groups": list(self.expected_outer_groups),
            "fold_definitions": [
                fold.to_dict() for fold in self.fold_definitions
            ],
            "per_outer_aggregate_assignment_hashes": (
                self.per_outer_aggregate_assignment_hashes
            ),
            "plan_hash": self.plan_hash,
        }


def build_nested_group_ood_contract(
    canonical: pd.DataFrame,
    *,
    group_column: str,
    outer_group: str,
) -> NestedGroupOODContract:
    """Build inner LOGO search assignments for one outer held-out group."""
    frame = _normalize_canonical(canonical, group_column=group_column)
    normalized_outer_group = _source_text(outer_group, name="outer_group")
    groups = sorted(frame[group_column].unique())
    if normalized_outer_group not in groups:
        raise ValueError(
            f"Outer OOD group is absent from {group_column}: "
            f"{normalized_outer_group!r}."
        )
    inner_groups = [group for group in groups if group != normalized_outer_group]
    if len(inner_groups) < 2:
        raise ValueError(
            "Nested group-aware OOD requires at least three total groups so "
            "every inner fold has non-empty group-disjoint training data."
        )
    outer = frame[
        ["source_row_id", "canonical_reaction_key", group_column]
    ].copy()
    outer = outer.rename(columns={group_column: "group_value"})
    outer.insert(2, "group_column", group_column)
    outer["outer_group"] = normalized_outer_group
    outer["outer_split"] = np.where(
        outer["group_value"].eq(normalized_outer_group),
        "test",
        "train",
    )
    outer = outer[list(OUTER_ASSIGNMENT_COLUMNS)]

    outer_train = frame.loc[
        ~frame[group_column].eq(normalized_outer_group)
    ].copy()
    inner_records: list[dict[str, Any]] = []
    for fold_index, inner_validation_group in enumerate(inner_groups):
        for row in outer_train.to_dict(orient="records"):
            inner_records.append(
                {
                    "source_row_id": row["source_row_id"],
                    "canonical_reaction_key": row[
                        "canonical_reaction_key"
                    ],
                    "group_column": group_column,
                    "group_value": row[group_column],
                    "outer_group": normalized_outer_group,
                    "inner_fold_index": fold_index,
                    "inner_validation_group": inner_validation_group,
                    "inner_split": (
                        "valid"
                        if row[group_column] == inner_validation_group
                        else "train"
                    ),
                }
            )
    inner = pd.DataFrame(inner_records, columns=INNER_ASSIGNMENT_COLUMNS)
    return validate_nested_group_ood_assignments(
        frame,
        outer,
        inner,
        group_column=group_column,
        outer_group=normalized_outer_group,
    )


def validate_nested_group_ood_assignments(
    canonical: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    *,
    group_column: str,
    outer_group: str,
) -> NestedGroupOODContract:
    """Validate persisted nested assignments and return an immutable contract."""
    frame = _normalize_canonical(canonical, group_column=group_column)
    normalized_outer_group = _source_text(outer_group, name="outer_group")
    outer = _normalize_assignment_frame(
        outer_assignments,
        columns=OUTER_ASSIGNMENT_COLUMNS,
        integer_column=None,
    )
    inner = _normalize_assignment_frame(
        inner_assignments,
        columns=INNER_ASSIGNMENT_COLUMNS,
        integer_column="inner_fold_index",
    )
    groups = sorted(frame[group_column].unique())
    if normalized_outer_group not in groups:
        raise ValueError(
            f"Outer OOD group is absent from {group_column}: "
            f"{normalized_outer_group!r}."
        )
    expected_inner_groups = tuple(
        group for group in groups if group != normalized_outer_group
    )
    if len(expected_inner_groups) < 2:
        raise ValueError(
            "Nested group-aware OOD requires at least three total groups."
        )
    _validate_outer_assignments(
        frame,
        outer,
        group_column=group_column,
        outer_group=normalized_outer_group,
    )
    folds = _validate_inner_assignments(
        frame,
        inner,
        group_column=group_column,
        outer_group=normalized_outer_group,
        expected_inner_groups=expected_inner_groups,
    )
    normalized_outer = outer.sort_values(
        "source_row_id",
        kind="mergesort",
    ).reset_index(drop=True)
    normalized_inner = inner.sort_values(
        ["inner_fold_index", "source_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    outer_hash = stable_hash(
        {
            "schema_version": NESTED_OOD_SCHEMA_VERSION,
            "group_column": group_column,
            "outer_group": normalized_outer_group,
            "assignments": normalized_outer.to_dict(orient="records"),
        }
    )
    aggregate_hash = stable_hash(
        {
            "schema_version": NESTED_OOD_SCHEMA_VERSION,
            "group_column": group_column,
            "outer_group": normalized_outer_group,
            "outer_assignment_hash": outer_hash,
            "per_inner_fold_assignment_hashes": [
                fold.assignment_hash for fold in folds
            ],
        }
    )
    overlap = pd.DataFrame([fold.overlap_audit for fold in folds])
    return NestedGroupOODContract(
        group_column=group_column,
        outer_group=normalized_outer_group,
        outer_train_source_ids=tuple(
            sorted(
                outer.loc[
                    outer["outer_split"].eq("train"),
                    "source_row_id",
                ]
            )
        ),
        outer_test_source_ids=tuple(
            sorted(
                outer.loc[
                    outer["outer_split"].eq("test"),
                    "source_row_id",
                ]
            )
        ),
        expected_inner_groups=expected_inner_groups,
        inner_folds=tuple(folds),
        outer_assignment_hash=outer_hash,
        aggregate_assignment_hash=aggregate_hash,
        _outer_assignments=normalized_outer.copy(deep=True),
        _inner_assignments=normalized_inner.copy(deep=True),
        _inner_overlap_audit=overlap.copy(deep=True),
    )


def build_nested_ood_all_outer_plan(
    canonical: pd.DataFrame,
    contracts: Iterable[NestedGroupOODContract],
    *,
    group_column: str,
) -> NestedOODAllOuterPlan:
    """Validate streamed outer contracts and retain compact fold definitions."""
    frame = _normalize_canonical(canonical, group_column=group_column)
    expected_outer_groups = tuple(sorted(frame[group_column].unique()))
    definitions_by_group: dict[str, NestedOODOuterFoldDefinition] = {}
    for contract in contracts:
        if not isinstance(contract, NestedGroupOODContract):
            raise ValueError(
                "All-outer nested OOD plans require NestedGroupOODContract values."
            )
        if contract.group_column != group_column:
            raise ValueError("Nested OOD contract group_column does not match plan.")
        if contract.outer_group in definitions_by_group:
            raise ValueError(
                f"Duplicate nested OOD outer group: {contract.outer_group!r}."
            )
        if contract.outer_group not in expected_outer_groups:
            raise ValueError(
                f"Unexpected nested OOD outer group: {contract.outer_group!r}."
            )
        validated = validate_nested_group_ood_assignments(
            frame,
            contract.outer_assignments,
            contract.inner_assignments,
            group_column=group_column,
            outer_group=contract.outer_group,
        )
        if validated.aggregate_assignment_hash != contract.aggregate_assignment_hash:
            raise ValueError(
                "Nested OOD contract aggregate assignment hash is stale."
            )
        definitions_by_group[contract.outer_group] = (
            NestedOODOuterFoldDefinition(
                outer_fold_index=-1,
                outer_group=contract.outer_group,
                outer_train_size=len(validated.outer_train_source_ids),
                outer_test_size=len(validated.outer_test_source_ids),
                expected_inner_group_count=len(
                    validated.expected_inner_groups
                ),
                aggregate_assignment_hash=(
                    validated.aggregate_assignment_hash
                ),
            )
        )
    observed_groups = set(definitions_by_group)
    if observed_groups != set(expected_outer_groups):
        missing = sorted(set(expected_outer_groups) - observed_groups)
        unexpected = sorted(observed_groups - set(expected_outer_groups))
        raise ValueError(
            "All-outer nested OOD plan must contain every outer group exactly "
            f"once: missing={missing}, unexpected={unexpected}."
        )
    definitions = tuple(
        NestedOODOuterFoldDefinition(
            outer_fold_index=fold_index,
            outer_group=outer_group,
            outer_train_size=definitions_by_group[
                outer_group
            ].outer_train_size,
            outer_test_size=definitions_by_group[outer_group].outer_test_size,
            expected_inner_group_count=definitions_by_group[
                outer_group
            ].expected_inner_group_count,
            aggregate_assignment_hash=definitions_by_group[
                outer_group
            ].aggregate_assignment_hash,
        )
        for fold_index, outer_group in enumerate(expected_outer_groups)
    )
    canonical_source_id_hash = stable_hash(
        list(frame["source_row_id"])
    )
    plan_payload = _all_outer_plan_payload(
        group_column=group_column,
        canonical_source_id_hash=canonical_source_id_hash,
        expected_outer_groups=expected_outer_groups,
        fold_definitions=definitions,
    )
    plan = NestedOODAllOuterPlan(
        group_column=group_column,
        canonical_source_id_hash=canonical_source_id_hash,
        expected_outer_groups=expected_outer_groups,
        fold_definitions=definitions,
        plan_hash=stable_hash(plan_payload),
    )
    validate_nested_ood_all_outer_plan(frame, plan)
    return plan


def validate_nested_ood_all_outer_plan(
    canonical: pd.DataFrame,
    plan: NestedOODAllOuterPlan,
) -> None:
    """Validate a compact all-outer plan without expanding assignments."""
    if not isinstance(plan, NestedOODAllOuterPlan):
        raise ValueError("Nested OOD all-outer plan has an unsupported type.")
    frame = _normalize_canonical(
        canonical,
        group_column=plan.group_column,
    )
    expected_groups = tuple(sorted(frame[plan.group_column].unique()))
    if plan.expected_outer_groups != expected_groups:
        raise ValueError(
            "Nested OOD all-outer plan groups do not match canonical data."
        )
    if len(plan.fold_definitions) != len(expected_groups):
        raise ValueError(
            "Nested OOD all-outer plan fold count does not match group count."
        )
    observed_groups = tuple(
        fold.outer_group for fold in plan.fold_definitions
    )
    if observed_groups != expected_groups or len(set(observed_groups)) != len(
        observed_groups
    ):
        raise ValueError(
            "Nested OOD all-outer folds must enumerate every sorted group once."
        )
    n_rows = len(frame)
    for fold_index, fold in enumerate(plan.fold_definitions):
        expected_test_size = int(
            frame[plan.group_column].eq(fold.outer_group).sum()
        )
        if fold.outer_fold_index != fold_index:
            raise ValueError(
                "Nested OOD outer fold indices must be deterministic and zero-based."
            )
        if (
            fold.outer_test_size != expected_test_size
            or fold.outer_train_size != n_rows - expected_test_size
            or fold.expected_inner_group_count != len(expected_groups) - 1
            or not fold.aggregate_assignment_hash
        ):
            raise ValueError(
                "Nested OOD compact outer fold definition does not match "
                "canonical support."
            )
    canonical_source_id_hash = stable_hash(
        list(frame["source_row_id"])
    )
    if plan.canonical_source_id_hash != canonical_source_id_hash:
        raise ValueError(
            "Nested OOD all-outer canonical source-ID hash mismatch."
        )
    expected_hash = stable_hash(
        _all_outer_plan_payload(
            group_column=plan.group_column,
            canonical_source_id_hash=plan.canonical_source_id_hash,
            expected_outer_groups=plan.expected_outer_groups,
            fold_definitions=plan.fold_definitions,
        )
    )
    if plan.plan_hash != expected_hash:
        raise ValueError("Nested OOD all-outer plan hash mismatch.")


def aggregate_inner_ood_metrics(
    contract: NestedGroupOODContract,
    metrics: pd.DataFrame,
    *,
    expected_group_keys: pd.DataFrame | Iterable[Mapping[str, Any]],
    group_columns: tuple[str, ...] = ("policy_id", "policy_hash", "metric"),
    weighting: str = "group_weighted",
    value_column: str = "value",
    sample_count_column: str = "n_validation_samples",
) -> pd.DataFrame:
    """Aggregate complete inner OOD fold metrics with declared weighting."""
    if weighting not in _WEIGHTINGS:
        raise ValueError(
            f"Unsupported inner OOD weighting {weighting!r}; "
            f"supported={sorted(_WEIGHTINGS)}."
        )
    if (
        not group_columns
        or len(group_columns) != len(set(group_columns))
        or any(not isinstance(column, str) or not column for column in group_columns)
    ):
        raise ValueError(
            "group_columns must contain unique non-empty column names."
        )
    required = {
        *group_columns,
        "inner_fold_index",
        "inner_validation_group",
        "outer_group",
        "outer_assignment_hash",
        "inner_assignment_hash",
        "aggregate_assignment_hash",
        value_column,
        sample_count_column,
    }
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("Inner OOD metrics must be a non-empty DataFrame.")
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Inner OOD metrics are missing columns: {missing}.")
    expected_keys = _normalize_expected_group_keys(
        expected_group_keys,
        group_columns=group_columns,
    )
    actual_keys = {
        tuple(row[column] for column in group_columns)
        for row in metrics[list(group_columns)].drop_duplicates().to_dict(
            orient="records"
        )
    }
    missing_keys = sorted(expected_keys - actual_keys, key=repr)
    unexpected_keys = sorted(actual_keys - expected_keys, key=repr)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Inner OOD metric candidate/metric keys differ from the declared "
            f"search grid: missing={missing_keys}, unexpected={unexpected_keys}."
        )
    rows: list[dict[str, Any]] = []
    expected_by_index = {
        fold.fold_index: fold for fold in contract.inner_folds
    }
    groupby_key: str | list[str] = (
        group_columns[0]
        if len(group_columns) == 1
        else list(group_columns)
    )
    for key, candidate_rows in metrics.groupby(
        groupby_key,
        sort=True,
        dropna=False,
    ):
        key_values = (key,) if len(group_columns) == 1 else tuple(key)
        if any(pd.isna(value) for value in key_values):
            raise ValueError("Inner OOD metric grouping values must not be missing.")
        if (
            len(candidate_rows) != contract.inner_fold_count
            or candidate_rows["inner_fold_index"].duplicated().any()
        ):
            raise ValueError(
                "Every aggregated policy/metric must contain exactly one row "
                "for each inner OOD fold."
            )
        values: list[float] = []
        sample_counts: list[int] = []
        for row in candidate_rows.to_dict(orient="records"):
            fold_index = row["inner_fold_index"]
            if (
                not isinstance(fold_index, Integral)
                or isinstance(fold_index, bool)
                or int(fold_index) not in expected_by_index
            ):
                raise ValueError(
                    "Inner OOD metric fold indices do not match the contract."
                )
            fold = expected_by_index[int(fold_index)]
            if row["inner_validation_group"] != fold.validation_group:
                raise ValueError(
                    "Inner OOD metric validation group does not match its fold."
                )
            if (
                row["outer_group"] != contract.outer_group
                or row["outer_assignment_hash"]
                != contract.outer_assignment_hash
                or row["inner_assignment_hash"] != fold.assignment_hash
                or row["aggregate_assignment_hash"]
                != contract.aggregate_assignment_hash
            ):
                raise ValueError(
                    "Inner OOD metric assignment provenance does not match "
                    "the active nested OOD contract."
                )
            value = _finite_number(row[value_column], name=value_column)
            sample_count = row[sample_count_column]
            if (
                not isinstance(sample_count, Integral)
                or isinstance(sample_count, bool)
                or int(sample_count) != fold.validation_size
            ):
                raise ValueError(
                    "Inner OOD metric validation sample count does not match "
                    "the assignment contract."
                )
            values.append(value)
            sample_counts.append(int(sample_count))
        if weighting == "group_weighted":
            aggregate_value = float(np.mean(values))
        else:
            aggregate_value = float(np.average(values, weights=sample_counts))
        result = dict(zip(group_columns, key_values, strict=True))
        result.update(
            {
                "weighting": weighting,
                "value": aggregate_value,
                "n_inner_folds": contract.inner_fold_count,
                "n_inner_validation_samples": sum(sample_counts),
                "outer_group": contract.outer_group,
                "outer_assignment_hash": contract.outer_assignment_hash,
                "aggregate_assignment_hash": (
                    contract.aggregate_assignment_hash
                ),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows).sort_values(
        list(group_columns),
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_canonical(
    canonical: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    if not isinstance(group_column, str) or not group_column.strip():
        raise ValueError("Nested OOD group_column must be a non-empty string.")
    if not isinstance(canonical, pd.DataFrame) or canonical.empty:
        raise ValueError("Nested OOD canonical input must be a non-empty DataFrame.")
    required = {"source_row_id", "canonical_reaction_key", group_column}
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Nested OOD canonical input is missing columns: {missing}.")
    frame = canonical.copy(deep=True)
    for column in required:
        if frame[column].isna().any():
            raise ValueError(f"Nested OOD column {column} contains missing values.")
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Nested OOD column {column} contains empty values.")
    if frame["source_row_id"].duplicated().any():
        raise ValueError("Nested OOD source_row_id values must be unique.")
    if frame[group_column].nunique() < 3:
        raise ValueError(
            "Nested group-aware OOD requires at least three total groups."
        )
    reaction_group_counts = frame.groupby(
        "canonical_reaction_key",
        sort=False,
    )[group_column].nunique()
    if reaction_group_counts.max() != 1:
        raise ValueError(
            "A canonical reaction key maps to multiple nested OOD groups."
        )
    return frame.sort_values("source_row_id", kind="mergesort").reset_index(
        drop=True
    )


def _normalize_assignment_frame(
    assignments: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    integer_column: str | None,
) -> pd.DataFrame:
    if not isinstance(assignments, pd.DataFrame) or assignments.empty:
        raise ValueError("Nested OOD assignments must be a non-empty DataFrame.")
    if tuple(assignments.columns) != columns:
        raise ValueError(
            "Nested OOD assignment schema mismatch: "
            f"expected={list(columns)}, observed={list(assignments.columns)}."
        )
    frame = assignments.copy(deep=True)
    for column in columns:
        if column == integer_column:
            continue
        if frame[column].isna().any():
            raise ValueError(
                f"Nested OOD assignment column {column} contains missing values."
            )
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(
                f"Nested OOD assignment column {column} contains empty values."
            )
    if integer_column is not None:
        if not frame[integer_column].map(
            lambda value: isinstance(value, Integral)
            and not isinstance(value, bool)
        ).all():
            raise ValueError(
                f"Nested OOD {integer_column} values must be exact integers."
            )
        frame[integer_column] = frame[integer_column].astype(int)
    return frame


def _normalize_expected_group_keys(
    expected_group_keys: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    group_columns: tuple[str, ...],
) -> set[tuple[Any, ...]]:
    if isinstance(expected_group_keys, pd.DataFrame):
        if tuple(expected_group_keys.columns) != group_columns:
            raise ValueError(
                "Expected inner OOD group-key schema mismatch: "
                f"expected={list(group_columns)}, "
                f"observed={list(expected_group_keys.columns)}."
            )
        records = expected_group_keys.to_dict(orient="records")
    else:
        if isinstance(expected_group_keys, (str, bytes, Mapping)):
            raise ValueError(
                "expected_group_keys must be a DataFrame or iterable of mappings."
            )
        records = list(expected_group_keys)
        if any(
            not isinstance(record, Mapping)
            or set(record) != set(group_columns)
            for record in records
        ):
            raise ValueError(
                "Each expected inner OOD group key must contain exactly "
                f"{list(group_columns)}."
            )
    if not records:
        raise ValueError("Expected inner OOD group keys must not be empty.")
    normalized: list[tuple[Any, ...]] = []
    for record in records:
        values = tuple(record[column] for column in group_columns)
        if any(pd.isna(value) for value in values):
            raise ValueError(
                "Expected inner OOD group-key values must not be missing."
            )
        try:
            hash(values)
        except TypeError as exc:
            raise ValueError(
                "Expected inner OOD group-key values must be hashable scalars."
            ) from exc
        normalized.append(values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Expected inner OOD group keys contain duplicates.")
    return set(normalized)


def _all_outer_plan_payload(
    *,
    group_column: str,
    canonical_source_id_hash: str,
    expected_outer_groups: tuple[str, ...],
    fold_definitions: tuple[NestedOODOuterFoldDefinition, ...],
) -> dict[str, Any]:
    return {
        "schema_version": NESTED_OOD_SCHEMA_VERSION,
        "group_column": group_column,
        "canonical_source_id_hash": canonical_source_id_hash,
        "expected_outer_groups": list(expected_outer_groups),
        "fold_definitions": [
            fold.to_dict() for fold in fold_definitions
        ],
    }


def _validate_outer_assignments(
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    group_column: str,
    outer_group: str,
) -> None:
    if assignments["source_row_id"].duplicated().any():
        raise ValueError("Nested OOD outer assignments contain duplicate source IDs.")
    expected_ids = set(canonical["source_row_id"])
    if set(assignments["source_row_id"]) != expected_ids:
        raise ValueError("Nested OOD outer assignments do not exactly cover all rows.")
    if set(assignments["group_column"]) != {group_column}:
        raise ValueError("Nested OOD outer assignment group_column mismatch.")
    if set(assignments["outer_group"]) != {outer_group}:
        raise ValueError("Nested OOD outer assignment outer_group mismatch.")
    if not assignments["outer_split"].isin({"train", "test"}).all():
        raise ValueError("Nested OOD outer_split supports only train and test.")
    canonical_by_id = canonical.set_index("source_row_id")
    assigned_by_id = assignments.set_index("source_row_id")
    if not assigned_by_id["canonical_reaction_key"].sort_index().equals(
        canonical_by_id["canonical_reaction_key"].sort_index()
    ) or not assigned_by_id["group_value"].sort_index().equals(
        canonical_by_id[group_column].sort_index()
    ):
        raise ValueError(
            "Nested OOD outer assignment identities do not match canonical data."
        )
    expected_test_ids = set(
        canonical.loc[canonical[group_column].eq(outer_group), "source_row_id"]
    )
    observed_test_ids = set(
        assignments.loc[assignments["outer_split"].eq("test"), "source_row_id"]
    )
    observed_train_ids = set(
        assignments.loc[assignments["outer_split"].eq("train"), "source_row_id"]
    )
    if (
        observed_test_ids != expected_test_ids
        or observed_train_ids != expected_ids - expected_test_ids
    ):
        raise ValueError(
            "Nested OOD outer assignments do not hold out exactly outer_group."
        )


def _validate_inner_assignments(
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    group_column: str,
    outer_group: str,
    expected_inner_groups: tuple[str, ...],
) -> list[NestedOODInnerFold]:
    if not assignments["inner_split"].isin({"train", "valid"}).all():
        raise ValueError("Nested OOD inner_split supports only train and valid.")
    if set(assignments["group_column"]) != {group_column}:
        raise ValueError("Nested OOD inner assignment group_column mismatch.")
    if set(assignments["outer_group"]) != {outer_group}:
        raise ValueError("Nested OOD inner assignment outer_group mismatch.")
    outer_test_ids = set(
        canonical.loc[canonical[group_column].eq(outer_group), "source_row_id"]
    )
    contaminated = set(assignments["source_row_id"]) & outer_test_ids
    if contaminated:
        raise ValueError(
            "Outer held-out group entered inner OOD search assignments: "
            f"{sorted(contaminated)}."
        )
    observed_fold_metadata = assignments[
        ["inner_fold_index", "inner_validation_group"]
    ].drop_duplicates()
    if len(observed_fold_metadata) != len(expected_inner_groups):
        raise ValueError(
            "Expected inner group count does not match observed inner fold count."
        )
    observed_groups = list(
        observed_fold_metadata.sort_values("inner_fold_index", kind="mergesort")[
            "inner_validation_group"
        ]
    )
    if observed_groups != list(expected_inner_groups):
        raise ValueError(
            "Every remaining group must appear exactly once as inner validation "
            "in deterministic sorted order."
        )
    observed_indices = list(
        observed_fold_metadata.sort_values("inner_fold_index")[
            "inner_fold_index"
        ]
    )
    if observed_indices != list(range(len(expected_inner_groups))):
        raise ValueError(
            "Nested OOD inner fold indices must be a complete zero-based sequence."
        )
    outer_train = canonical.loc[
        ~canonical[group_column].eq(outer_group)
    ].copy()
    outer_train_ids = set(outer_train["source_row_id"])
    canonical_by_id = outer_train.set_index("source_row_id")
    folds: list[NestedOODInnerFold] = []
    for fold_index, validation_group in enumerate(expected_inner_groups):
        fold = assignments.loc[
            assignments["inner_fold_index"].eq(fold_index)
        ].copy()
        if set(fold["inner_validation_group"]) != {validation_group}:
            raise ValueError("Nested OOD inner validation fold metadata mismatch.")
        if fold["source_row_id"].duplicated().any():
            raise ValueError("Nested OOD inner fold contains duplicate source IDs.")
        if set(fold["source_row_id"]) != outer_train_ids:
            raise ValueError(
                "Every inner OOD fold must exactly cover outer-training rows."
            )
        assigned_by_id = fold.set_index("source_row_id")
        if not assigned_by_id["canonical_reaction_key"].sort_index().equals(
            canonical_by_id["canonical_reaction_key"].sort_index()
        ) or not assigned_by_id["group_value"].sort_index().equals(
            canonical_by_id[group_column].sort_index()
        ):
            raise ValueError(
                "Nested OOD inner assignment identities do not match canonical data."
            )
        expected_valid_ids = set(
            outer_train.loc[
                outer_train[group_column].eq(validation_group),
                "source_row_id",
            ]
        )
        valid_ids = set(
            fold.loc[fold["inner_split"].eq("valid"), "source_row_id"]
        )
        train_ids = set(
            fold.loc[fold["inner_split"].eq("train"), "source_row_id"]
        )
        if (
            valid_ids != expected_valid_ids
            or train_ids != outer_train_ids - expected_valid_ids
        ):
            raise ValueError(
                "Nested OOD inner assignments do not hold out exactly the "
                "declared validation group."
            )
        train = outer_train.loc[outer_train["source_row_id"].isin(train_ids)]
        valid = outer_train.loc[outer_train["source_row_id"].isin(valid_ids)]
        source_overlap = train_ids & valid_ids
        group_overlap = set(train[group_column]) & set(valid[group_column])
        reaction_overlap = set(train["canonical_reaction_key"]) & set(
            valid["canonical_reaction_key"]
        )
        if source_overlap or group_overlap or reaction_overlap:
            raise ValueError("Nested OOD inner train/validation overlap is nonzero.")
        hash_rows = (
            fold[list(INNER_ASSIGNMENT_COLUMNS)]
            .sort_values("source_row_id", kind="mergesort")
            .to_dict(orient="records")
        )
        assignment_hash = stable_hash(
            {
                "schema_version": NESTED_OOD_SCHEMA_VERSION,
                "outer_group": outer_group,
                "inner_fold_index": fold_index,
                "inner_validation_group": validation_group,
                "assignments": hash_rows,
            }
        )
        folds.append(
            NestedOODInnerFold(
                fold_index=fold_index,
                validation_group=validation_group,
                train_source_ids=tuple(sorted(train_ids)),
                validation_source_ids=tuple(sorted(valid_ids)),
                train_size=len(train_ids),
                validation_size=len(valid_ids),
                assignment_hash=assignment_hash,
                source_id_overlap_count=len(source_overlap),
                group_overlap_count=len(group_overlap),
                canonical_reaction_key_overlap_count=len(reaction_overlap),
            )
        )
    return folds


def _source_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result
