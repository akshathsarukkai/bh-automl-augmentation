"""Canonical condition-combination, ligand, and base OOD split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    stable_json,
)
from bh_augmentation.utils.corrected_runs import stable_hash

CONDITION_OOD_SCHEMA_VERSION = "bh-canonical-condition-ood-v1"
CONDITION_COMBINATION_KEY_SCHEMA_VERSION = "bh-condition-combination-v1"
CONDITION_ROLES = ("catalyst", "ligand", "base", "solvent_or_additive")
CONDITION_ROLE_COLUMNS = tuple(
    f"canonical_{role}_smiles" for role in CONDITION_ROLES
)
CONDITION_OOD_TARGET_COLUMNS = {
    "condition_combination": "condition_combination_key",
    "ligand": "canonical_ligand_smiles",
    "base": "canonical_base_smiles",
}


@dataclass(frozen=True, slots=True)
class FoldSupportCriteria:
    """Minimum evidence required for a scientifically meaningful OOD fold."""

    min_test_samples: int = 1
    min_train_samples: int = 2
    min_train_groups: int = 2
    min_test_substrates: int = 1
    min_train_substrates: int = 1

    def __post_init__(self) -> None:
        for name in (
            "min_test_samples",
            "min_train_samples",
            "min_train_groups",
            "min_test_substrates",
            "min_train_substrates",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    def to_dict(self) -> dict[str, int]:
        """Serialize criteria without depending on dataclass internals."""
        return {
            name: getattr(self, name)
            for name in (
                "min_test_samples",
                "min_train_samples",
                "min_train_groups",
                "min_test_substrates",
                "min_train_substrates",
            )
        }


@dataclass(frozen=True, slots=True)
class ConditionOODFold:
    """One viable or explicitly excluded condition-side OOD fold."""

    fold_index: int
    heldout_group: str
    train_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]
    train_size: int
    test_size: int
    train_group_count: int
    train_substrate_count: int
    test_substrate_count: int
    group_overlap_count: int
    canonical_reaction_key_overlap_count: int
    exclusion_reason: str | None
    assignment_hash: str

    @property
    def included(self) -> bool:
        """Return whether this fold passed every declared support criterion."""
        return self.exclusion_reason is None

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return complete fold support and overlap metadata."""
        return {
            "fold_index": self.fold_index,
            "heldout_group": self.heldout_group,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "train_group_count": self.train_group_count,
            "train_substrate_count": self.train_substrate_count,
            "test_substrate_count": self.test_substrate_count,
            "group_overlap_count": self.group_overlap_count,
            "canonical_reaction_key_overlap_count": (
                self.canonical_reaction_key_overlap_count
            ),
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            "assignment_hash": self.assignment_hash,
        }


@dataclass(frozen=True, slots=True)
class ConditionOODPlan:
    """Deterministic support-filtered holdout plan for one grouping target."""

    target: str
    group_column: str
    criteria: FoldSupportCriteria
    folds: tuple[ConditionOODFold, ...]
    plan_hash: str

    @property
    def included_folds(self) -> tuple[ConditionOODFold, ...]:
        """Return viable folds in stable group order."""
        return tuple(fold for fold in self.folds if fold.included)

    @property
    def excluded_folds(self) -> tuple[ConditionOODFold, ...]:
        """Return unsupported folds with explicit reasons."""
        return tuple(fold for fold in self.folds if not fold.included)

    @property
    def audit_frame(self) -> pd.DataFrame:
        """Return a defensive tabular fold audit."""
        return pd.DataFrame([fold.audit_record for fold in self.folds])

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return compact reproducibility metadata."""
        return {
            "schema_version": CONDITION_OOD_SCHEMA_VERSION,
            "target": self.target,
            "group_column": self.group_column,
            "criteria": self.criteria.to_dict(),
            "observed_group_count": len(self.folds),
            "included_group_count": len(self.included_folds),
            "excluded_group_count": len(self.excluded_folds),
            "folds": [fold.audit_record for fold in self.folds],
            "plan_hash": self.plan_hash,
        }


def canonical_condition_combination_key(row: dict[str, Any]) -> str:
    """Build a versioned key from four canonical condition identities.

    The role names are explicitly supplied in canonical reaction-role order.
    No raw SMILES, fingerprint, or feature-vector identity is accepted.
    """
    values = {}
    for role, column in zip(
        CONDITION_ROLES,
        CONDITION_ROLE_COLUMNS,
        strict=True,
    ):
        value = row.get(column)
        if value is None or bool(pd.isna(value)) or not str(value).strip():
            raise ValueError(
                f"Cannot build condition combination key: invalid {column}."
            )
        values[role] = str(value).strip()
    return stable_json(
        {
            "schema": CONDITION_COMBINATION_KEY_SCHEMA_VERSION,
            **values,
        }
    )


def add_condition_combination_keys(canonical: pd.DataFrame) -> pd.DataFrame:
    """Return canonical rows with independently reconstructed condition keys."""
    frame = _validate_canonical(canonical)
    result = frame.copy()
    result["condition_combination_key"] = [
        canonical_condition_combination_key(row)
        for row in result.to_dict(orient="records")
    ]
    return result


def build_condition_ood_plan(
    canonical: pd.DataFrame,
    *,
    target: str,
    criteria: FoldSupportCriteria | None = None,
) -> ConditionOODPlan:
    """Build leave-one-combination/ligand/base-out folds with support audits."""
    if target not in CONDITION_OOD_TARGET_COLUMNS:
        raise ValueError(
            f"Unsupported condition OOD target {target!r}; "
            f"supported={sorted(CONDITION_OOD_TARGET_COLUMNS)}."
        )
    support = criteria or FoldSupportCriteria()
    frame = add_condition_combination_keys(canonical)
    group_column = CONDITION_OOD_TARGET_COLUMNS[target]
    groups = sorted(frame[group_column].unique())
    folds = []
    for fold_index, heldout_group in enumerate(groups):
        test = frame.loc[frame[group_column].eq(heldout_group)]
        train = frame.loc[~frame[group_column].eq(heldout_group)]
        train_groups = set(train[group_column])
        test_groups = set(test[group_column])
        train_keys = set(train["canonical_reaction_key"])
        test_keys = set(test["canonical_reaction_key"])
        train_substrates = set(train["canonical_substrate_key"])
        test_substrates = set(test["canonical_substrate_key"])
        reasons = _support_reasons(
            support,
            train_size=len(train),
            test_size=len(test),
            train_group_count=len(train_groups),
            train_substrate_count=len(train_substrates),
            test_substrate_count=len(test_substrates),
        )
        group_overlap = len(train_groups & test_groups)
        reaction_overlap = len(train_keys & test_keys)
        if group_overlap:
            reasons.append(f"declared_group_overlap(observed={group_overlap})")
        if reaction_overlap:
            reasons.append(
                f"canonical_reaction_key_overlap(observed={reaction_overlap})"
            )
        assignment_payload = {
            "schema_version": CONDITION_OOD_SCHEMA_VERSION,
            "target": target,
            "heldout_group": heldout_group,
            "train_source_ids": sorted(train["source_row_id"].astype(str)),
            "test_source_ids": sorted(test["source_row_id"].astype(str)),
        }
        folds.append(
            ConditionOODFold(
                fold_index=fold_index,
                heldout_group=heldout_group,
                train_source_ids=tuple(assignment_payload["train_source_ids"]),
                test_source_ids=tuple(assignment_payload["test_source_ids"]),
                train_size=len(train),
                test_size=len(test),
                train_group_count=len(train_groups),
                train_substrate_count=len(train_substrates),
                test_substrate_count=len(test_substrates),
                group_overlap_count=group_overlap,
                canonical_reaction_key_overlap_count=reaction_overlap,
                exclusion_reason="; ".join(reasons) if reasons else None,
                assignment_hash=stable_hash(assignment_payload),
            )
        )
    plan_payload = {
        "schema_version": CONDITION_OOD_SCHEMA_VERSION,
        "target": target,
        "group_column": group_column,
        "criteria": support.to_dict(),
        "fold_assignment_hashes": [fold.assignment_hash for fold in folds],
        "exclusion_reasons": [fold.exclusion_reason for fold in folds],
    }
    return ConditionOODPlan(
        target=target,
        group_column=group_column,
        criteria=support,
        folds=tuple(folds),
        plan_hash=stable_hash(plan_payload),
    )


def _validate_canonical(canonical: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(canonical, pd.DataFrame) or canonical.empty:
        raise ValueError("Condition OOD requires a non-empty canonical DataFrame.")
    required = {
        "source_row_id",
        "canonical_reaction_key",
        "canonical_substrate_key",
        "canonicalization_version",
        *CONDITION_ROLE_COLUMNS,
    }
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Condition OOD canonical rows are missing columns: {missing}.")
    frame = canonical.copy()
    if frame["source_row_id"].isna().any() or frame["source_row_id"].duplicated().any():
        raise ValueError("Condition OOD source_row_id values must be unique and non-missing.")
    versions = set(frame["canonicalization_version"].astype(str))
    if versions != {CANONICALIZATION_VERSION}:
        raise ValueError(
            "Condition OOD canonicalization version mismatch: "
            f"expected={CANONICALIZATION_VERSION!r}, observed={sorted(versions)}."
        )
    for column in (
        "canonical_reaction_key",
        "canonical_substrate_key",
        *CONDITION_ROLE_COLUMNS,
    ):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Condition OOD canonical identity column {column} is invalid.")
    return frame


def _support_reasons(
    criteria: FoldSupportCriteria,
    *,
    train_size: int,
    test_size: int,
    train_group_count: int,
    train_substrate_count: int,
    test_substrate_count: int,
) -> list[str]:
    checks = (
        ("test_samples", test_size, criteria.min_test_samples),
        ("train_samples", train_size, criteria.min_train_samples),
        ("train_groups", train_group_count, criteria.min_train_groups),
        ("test_substrates", test_substrate_count, criteria.min_test_substrates),
        ("train_substrates", train_substrate_count, criteria.min_train_substrates),
    )
    return [
        f"{name}_below_minimum(observed={observed},required={required})"
        for name, observed, required in checks
        if observed < required
    ]
