"""Strict canonical leave-one-group-out assignment contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import pandas as pd

from bh_augmentation.utils.corrected_runs import stable_hash

LOGO_SCHEMA_VERSION = "bh-canonical-logo-v1"
LOGO_SPLIT_METHOD = "leave_one_group_out"
LOGO_TARGET_COLUMNS: Mapping[str, str] = {
    "product_key": "canonical_product_key",
    "reactant_key": "canonical_substrate_key",
}
LOGO_ASSIGNMENT_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "logo_target",
    "group_column",
    "group_value",
    "fold_index",
    "fold_group",
    "split_method",
    "outer_split",
)


@dataclass(frozen=True, slots=True)
class CanonicalLOGOFold:
    """Immutable membership and audit metadata for one canonical LOGO fold."""

    fold_index: int
    fold_group: str
    group_size: int
    train_size: int
    test_size: int
    train_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]
    assignment_hash: str
    source_id_overlap_count: int
    group_overlap_count: int
    canonical_reaction_key_overlap_count: int

    @property
    def overlap_audit(self) -> dict[str, Any]:
        """Return the complete train/test overlap audit for this fold."""
        return {
            "fold_index": self.fold_index,
            "fold_group": self.fold_group,
            "train_size": self.train_size,
            "test_size": self.test_size,
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
class CanonicalLOGOContract:
    """Validated immutable assignment contract for one logical LOGO target."""

    target: str
    group_column: str
    expected_group_count: int
    folds: tuple[CanonicalLOGOFold, ...]
    group_sizes: tuple[tuple[str, int], ...]
    aggregate_assignment_hash: str
    _assignments: pd.DataFrame = field(repr=False, compare=False)
    _overlap_audit: pd.DataFrame = field(repr=False, compare=False)

    @property
    def split_method(self) -> str:
        """Return the only supported outer split method."""
        return LOGO_SPLIT_METHOD

    @property
    def fold_count(self) -> int:
        """Return the observed number of validated folds."""
        return len(self.folds)

    @property
    def assignments(self) -> pd.DataFrame:
        """Return a defensive copy of canonical fold assignments."""
        return self._assignments.copy(deep=True)

    @property
    def overlap_audit(self) -> pd.DataFrame:
        """Return a defensive copy of per-fold overlap checks."""
        return self._overlap_audit.copy(deep=True)

    @property
    def per_fold_assignment_hashes(self) -> dict[str, str]:
        """Return stable hashes keyed by held-out canonical group."""
        return {
            fold.fold_group: fold.assignment_hash for fold in self.folds
        }

    @property
    def group_size_records(self) -> list[dict[str, Any]]:
        """Return held-out group sizes in deterministic fold order."""
        return [
            {
                "fold_index": fold_index,
                "fold_group": group,
                "group_size": size,
            }
            for fold_index, (group, size) in enumerate(self.group_sizes)
        ]

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return concise assignment metadata suitable for a run manifest."""
        return {
            "schema_version": LOGO_SCHEMA_VERSION,
            "split_method": LOGO_SPLIT_METHOD,
            "target": self.target,
            "group_column": self.group_column,
            "expected_group_count": self.expected_group_count,
            "observed_fold_count": self.fold_count,
            "group_sizes": self.group_size_records,
            "per_fold_assignment_hashes": self.per_fold_assignment_hashes,
            "aggregate_assignment_hash": self.aggregate_assignment_hash,
            "all_overlaps_zero": bool(
                self._overlap_audit["all_overlaps_zero"].all()
            ),
        }


def build_canonical_logo_contract(
    canonical: pd.DataFrame,
    *,
    target: str,
) -> CanonicalLOGOContract:
    """Build and independently validate every canonical LOGO assignment."""
    group_column = resolve_logo_group_column(target)
    frame = _normalize_canonical(canonical, group_column=group_column)
    groups = sorted(frame[group_column].unique())
    records: list[dict[str, Any]] = []
    for fold_index, fold_group in enumerate(groups):
        for row in frame.to_dict(orient="records"):
            records.append(
                {
                    "source_row_id": row["source_row_id"],
                    "canonical_reaction_key": row[
                        "canonical_reaction_key"
                    ],
                    "logo_target": target,
                    "group_column": group_column,
                    "group_value": row[group_column],
                    "fold_index": fold_index,
                    "fold_group": fold_group,
                    "split_method": LOGO_SPLIT_METHOD,
                    "outer_split": (
                        "test"
                        if row[group_column] == fold_group
                        else "train"
                    ),
                }
            )
    assignments = pd.DataFrame(records, columns=LOGO_ASSIGNMENT_COLUMNS)
    return validate_canonical_logo_assignments(
        frame,
        assignments,
        target=target,
    )


def validate_canonical_logo_assignments(
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    target: str,
) -> CanonicalLOGOContract:
    """Validate supplied assignments and return an immutable LOGO contract."""
    group_column = resolve_logo_group_column(target)
    frame = _normalize_canonical(canonical, group_column=group_column)
    assignment_frame = _normalize_assignments(assignments)
    groups = sorted(frame[group_column].unique())
    expected_group_count = len(groups)
    observed_folds = assignment_frame[
        ["fold_index", "fold_group"]
    ].drop_duplicates()
    if len(observed_folds) != expected_group_count:
        raise ValueError(
            "Canonical LOGO expected group count does not match observed fold "
            f"count: expected={expected_group_count}, "
            f"observed={len(observed_folds)}."
        )
    fold_groups = list(
        observed_folds.sort_values("fold_index", kind="mergesort")[
            "fold_group"
        ]
    )
    if len(fold_groups) != len(set(fold_groups)):
        raise ValueError("Canonical LOGO fold_group values must be unique.")
    if set(fold_groups) != set(groups):
        omitted = sorted(set(groups) - set(fold_groups))
        unexpected = sorted(set(fold_groups) - set(groups))
        raise ValueError(
            "Every canonical group must appear exactly once as outer test: "
            f"omitted={omitted}, unexpected={unexpected}."
        )
    if fold_groups != groups:
        raise ValueError(
            "Canonical LOGO fold order must follow sorted canonical groups."
        )
    if list(observed_folds.sort_values("fold_index")["fold_index"]) != list(
        range(expected_group_count)
    ):
        raise ValueError(
            "Canonical LOGO fold_index values must be a complete deterministic "
            "zero-based sequence."
        )

    expected_ids = set(frame["source_row_id"])
    canonical_by_id = frame.set_index("source_row_id")
    folds: list[CanonicalLOGOFold] = []
    overlap_rows: list[dict[str, Any]] = []
    for fold_index, fold_group in enumerate(groups):
        fold = assignment_frame.loc[
            assignment_frame["fold_index"].eq(fold_index)
        ].copy()
        _validate_fold_metadata(
            fold,
            target=target,
            group_column=group_column,
            fold_group=fold_group,
        )
        if fold["source_row_id"].duplicated().any():
            raise ValueError(
                f"Canonical LOGO fold {fold_index} contains duplicate source IDs."
            )
        if set(fold["source_row_id"]) != expected_ids:
            missing = sorted(expected_ids - set(fold["source_row_id"]))
            unexpected = sorted(set(fold["source_row_id"]) - expected_ids)
            raise ValueError(
                "Every canonical LOGO fold must cover every source row exactly "
                f"once: missing={missing}, unexpected={unexpected}."
            )
        keyed = fold.set_index("source_row_id")
        if not keyed["canonical_reaction_key"].sort_index().equals(
            canonical_by_id["canonical_reaction_key"].sort_index()
        ):
            raise ValueError(
                "Canonical LOGO assignment reaction keys do not match the "
                "canonical dataset."
            )
        if not keyed["group_value"].sort_index().equals(
            canonical_by_id[group_column].sort_index()
        ):
            raise ValueError(
                "Canonical LOGO assignment group values do not match the "
                "canonical dataset."
            )
        expected_test_ids = set(
            frame.loc[frame[group_column].eq(fold_group), "source_row_id"]
        )
        observed_test_ids = set(
            fold.loc[fold["outer_split"].eq("test"), "source_row_id"]
        )
        observed_train_ids = set(
            fold.loc[fold["outer_split"].eq("train"), "source_row_id"]
        )
        if (
            observed_test_ids != expected_test_ids
            or observed_train_ids != expected_ids - expected_test_ids
        ):
            raise ValueError(
                "Assignments labeled leave_one_group_out do not hold out "
                f"exactly fold_group={fold_group!r}."
            )
        train = frame.loc[frame["source_row_id"].isin(observed_train_ids)]
        test = frame.loc[frame["source_row_id"].isin(observed_test_ids)]
        source_overlap = observed_train_ids & observed_test_ids
        group_overlap = set(train[group_column]) & set(test[group_column])
        reaction_overlap = set(train["canonical_reaction_key"]) & set(
            test["canonical_reaction_key"]
        )
        if source_overlap or group_overlap or reaction_overlap:
            raise ValueError(
                "Canonical LOGO train/test overlap is nonzero: "
                f"source_ids={sorted(source_overlap)}, "
                f"groups={sorted(group_overlap)}, "
                f"canonical_reaction_keys={sorted(reaction_overlap)}."
            )
        hash_rows = (
            fold[list(LOGO_ASSIGNMENT_COLUMNS)]
            .sort_values("source_row_id", kind="mergesort")
            .to_dict(orient="records")
        )
        assignment_hash = stable_hash(
            {
                "schema_version": LOGO_SCHEMA_VERSION,
                "target": target,
                "fold_index": fold_index,
                "fold_group": fold_group,
                "assignments": hash_rows,
            }
        )
        validated_fold = CanonicalLOGOFold(
            fold_index=fold_index,
            fold_group=fold_group,
            group_size=len(test),
            train_size=len(train),
            test_size=len(test),
            train_source_ids=tuple(sorted(observed_train_ids)),
            test_source_ids=tuple(sorted(observed_test_ids)),
            assignment_hash=assignment_hash,
            source_id_overlap_count=len(source_overlap),
            group_overlap_count=len(group_overlap),
            canonical_reaction_key_overlap_count=len(reaction_overlap),
        )
        folds.append(validated_fold)
        overlap_rows.append(validated_fold.overlap_audit)

    group_sizes = tuple(
        (group, int(frame[group_column].eq(group).sum())) for group in groups
    )
    aggregate_hash = stable_hash(
        {
            "schema_version": LOGO_SCHEMA_VERSION,
            "split_method": LOGO_SPLIT_METHOD,
            "target": target,
            "group_column": group_column,
            "group_sizes": group_sizes,
            "per_fold_assignment_hashes": [
                fold.assignment_hash for fold in folds
            ],
        }
    )
    normalized_assignments = assignment_frame.sort_values(
        ["fold_index", "source_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    overlap_audit = pd.DataFrame(overlap_rows)
    return CanonicalLOGOContract(
        target=target,
        group_column=group_column,
        expected_group_count=expected_group_count,
        folds=tuple(folds),
        group_sizes=group_sizes,
        aggregate_assignment_hash=aggregate_hash,
        _assignments=normalized_assignments.copy(deep=True),
        _overlap_audit=overlap_audit.copy(deep=True),
    )


def resolve_logo_group_column(target: str) -> str:
    """Resolve a supported logical target to its canonical identity column."""
    if target not in LOGO_TARGET_COLUMNS:
        raise ValueError(
            "Unsupported canonical LOGO target: "
            f"{target!r}; supported={sorted(LOGO_TARGET_COLUMNS)}."
        )
    return LOGO_TARGET_COLUMNS[target]


def _normalize_canonical(
    canonical: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    if not isinstance(canonical, pd.DataFrame) or canonical.empty:
        raise ValueError("Canonical LOGO input must be a non-empty DataFrame.")
    required = {
        "source_row_id",
        "canonical_reaction_key",
        group_column,
    }
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(
            f"Canonical LOGO input is missing required columns: {missing}."
        )
    frame = canonical.copy(deep=True)
    for column in required:
        if frame[column].isna().any():
            raise ValueError(
                f"Canonical LOGO column {column} contains missing values."
            )
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(
                f"Canonical LOGO column {column} contains empty values."
            )
    if frame["source_row_id"].duplicated().any():
        raise ValueError("Canonical LOGO source_row_id values must be unique.")
    groups = sorted(frame[group_column].unique())
    if len(groups) < 2:
        raise ValueError(
            "Canonical leave-one-group-out requires at least two groups."
        )
    reaction_group_counts = frame.groupby(
        "canonical_reaction_key",
        sort=False,
    )[group_column].nunique()
    if reaction_group_counts.max() != 1:
        raise ValueError(
            "A canonical reaction key maps to multiple declared LOGO groups."
        )
    return frame.sort_values("source_row_id", kind="mergesort").reset_index(
        drop=True
    )


def _normalize_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(assignments, pd.DataFrame) or assignments.empty:
        raise ValueError(
            "Canonical LOGO assignments must be a non-empty DataFrame."
        )
    if tuple(assignments.columns) != LOGO_ASSIGNMENT_COLUMNS:
        raise ValueError(
            "Canonical LOGO assignment schema mismatch: "
            f"expected={list(LOGO_ASSIGNMENT_COLUMNS)}, "
            f"observed={list(assignments.columns)}."
        )
    frame = assignments.copy(deep=True)
    text_columns = [
        column
        for column in LOGO_ASSIGNMENT_COLUMNS
        if column != "fold_index"
    ]
    for column in text_columns:
        if frame[column].isna().any():
            raise ValueError(
                f"Canonical LOGO assignment column {column} contains missing values."
            )
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(
                f"Canonical LOGO assignment column {column} contains empty values."
            )
    if not frame["fold_index"].map(
        lambda value: isinstance(value, Integral) and not isinstance(value, bool)
    ).all():
        raise ValueError(
            "Canonical LOGO assignment fold_index values must be exact integers."
        )
    frame["fold_index"] = frame["fold_index"].astype(int)
    if not frame["outer_split"].isin({"train", "test"}).all():
        raise ValueError(
            "Canonical LOGO outer_split supports only train and test."
        )
    return frame


def _validate_fold_metadata(
    fold: pd.DataFrame,
    *,
    target: str,
    group_column: str,
    fold_group: str,
) -> None:
    if fold.empty:
        raise ValueError("Canonical LOGO fold is missing assignment rows.")
    expected = {
        "logo_target": target,
        "group_column": group_column,
        "fold_group": fold_group,
        "split_method": LOGO_SPLIT_METHOD,
    }
    for column, value in expected.items():
        observed = set(fold[column])
        if observed != {value}:
            raise ValueError(
                f"Canonical LOGO fold metadata mismatch for {column}: "
                f"expected={value!r}, observed={sorted(observed)}."
            )
