"""Leakage-resistant inner splits for supervised-AE policy selection."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bh_augmentation.utils.corrected_runs import stable_hash

AE_INNER_SPLIT_SCHEMA_VERSION = "bh-ae-inner-split-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class AEInnerSplit:
    """An immutable, validated measured-data boundary for AE policy search.

    DataFrames are copied into stable source-ID order on construction and are
    exposed through copy-returning properties. This prevents callers from
    mutating the validated partitions after their membership hashes have been
    computed.
    """

    seed: int
    valid_size: float
    source_id_column: str
    group_column: str
    minimum_rows: int
    minimum_groups: int
    forbidden_outer_valid_source_ids: tuple[str, ...]
    forbidden_outer_test_source_ids: tuple[str, ...]
    _eligible_outer_train: pd.DataFrame = field(repr=False, compare=False)
    _ae_train: pd.DataFrame = field(repr=False, compare=False)
    _internal_validation: pd.DataFrame = field(repr=False, compare=False)
    eligible_source_ids: tuple[str, ...] = field(init=False)
    ae_train_source_ids: tuple[str, ...] = field(init=False)
    internal_validation_source_ids: tuple[str, ...] = field(init=False)
    eligible_source_id_hash: str = field(init=False)
    ae_train_source_id_hash: str = field(init=False)
    internal_validation_source_id_hash: str = field(init=False)
    split_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Normalize, validate, and hash the complete split boundary."""
        _validate_split_parameters(
            seed=self.seed,
            valid_size=self.valid_size,
            minimum_rows=self.minimum_rows,
            minimum_groups=self.minimum_groups,
        )
        eligible = _normalize_measured_frame(
            self._eligible_outer_train,
            name="eligible_outer_train",
            source_id_column=self.source_id_column,
            group_column=self.group_column,
        )
        ae_train = _normalize_measured_frame(
            self._ae_train,
            name="ae_train",
            source_id_column=self.source_id_column,
            group_column=self.group_column,
        )
        internal_validation = _normalize_measured_frame(
            self._internal_validation,
            name="internal_validation",
            source_id_column=self.source_id_column,
            group_column=self.group_column,
        )
        forbidden_valid = _normalize_source_ids(
            self.forbidden_outer_valid_source_ids,
            name="forbidden_outer_valid_source_ids",
        )
        forbidden_test = _normalize_source_ids(
            self.forbidden_outer_test_source_ids,
            name="forbidden_outer_test_source_ids",
        )
        object.__setattr__(self, "_eligible_outer_train", eligible)
        object.__setattr__(self, "_ae_train", ae_train)
        object.__setattr__(self, "_internal_validation", internal_validation)
        object.__setattr__(
            self,
            "forbidden_outer_valid_source_ids",
            forbidden_valid,
        )
        object.__setattr__(
            self,
            "forbidden_outer_test_source_ids",
            forbidden_test,
        )

        _validate_partition_membership(
            eligible,
            ae_train,
            internal_validation,
            source_id_column=self.source_id_column,
            group_column=self.group_column,
            minimum_rows=self.minimum_rows,
            minimum_groups=self.minimum_groups,
            forbidden_outer_valid_source_ids=forbidden_valid,
            forbidden_outer_test_source_ids=forbidden_test,
        )
        eligible_ids = tuple(eligible[self.source_id_column])
        ae_train_ids = tuple(ae_train[self.source_id_column])
        internal_valid_ids = tuple(internal_validation[self.source_id_column])
        object.__setattr__(self, "eligible_source_ids", eligible_ids)
        object.__setattr__(self, "ae_train_source_ids", ae_train_ids)
        object.__setattr__(
            self,
            "internal_validation_source_ids",
            internal_valid_ids,
        )
        eligible_hash = stable_hash(list(eligible_ids))
        ae_train_hash = stable_hash(list(ae_train_ids))
        internal_valid_hash = stable_hash(list(internal_valid_ids))
        object.__setattr__(self, "eligible_source_id_hash", eligible_hash)
        object.__setattr__(self, "ae_train_source_id_hash", ae_train_hash)
        object.__setattr__(
            self,
            "internal_validation_source_id_hash",
            internal_valid_hash,
        )
        membership = eligible[
            [self.source_id_column, self.group_column]
        ].copy()
        valid_id_set = set(internal_valid_ids)
        membership["inner_partition"] = membership[self.source_id_column].map(
            lambda source_id: (
                "internal_validation" if source_id in valid_id_set else "ae_train"
            )
        )
        split_payload = {
            "schema_version": AE_INNER_SPLIT_SCHEMA_VERSION,
            "seed": self.seed,
            "valid_size": float(self.valid_size),
            "source_id_column": self.source_id_column,
            "group_column": self.group_column,
            "minimum_rows": self.minimum_rows,
            "minimum_groups": self.minimum_groups,
            "membership": membership.to_dict(orient="records"),
            "eligible_source_id_hash": eligible_hash,
            "ae_train_source_id_hash": ae_train_hash,
            "internal_validation_source_id_hash": internal_valid_hash,
            "forbidden_outer_valid_source_id_hash": stable_hash(
                list(forbidden_valid)
            ),
            "forbidden_outer_test_source_id_hash": stable_hash(
                list(forbidden_test)
            ),
        }
        object.__setattr__(self, "split_hash", stable_hash(split_payload))

    @classmethod
    def from_partitions(
        cls,
        eligible_outer_train: pd.DataFrame,
        ae_train: pd.DataFrame,
        internal_validation: pd.DataFrame,
        *,
        seed: int,
        valid_size: float,
        forbidden_outer_valid_source_ids: Iterable[str] = (),
        forbidden_outer_test_source_ids: Iterable[str] = (),
        source_id_column: str = "source_row_id",
        group_column: str = "canonical_reaction_key",
        minimum_rows: int = 4,
        minimum_groups: int = 2,
    ) -> AEInnerSplit:
        """Construct and validate an explicit split supplied by a runner."""
        return cls(
            seed=seed,
            valid_size=valid_size,
            source_id_column=source_id_column,
            group_column=group_column,
            minimum_rows=minimum_rows,
            minimum_groups=minimum_groups,
            forbidden_outer_valid_source_ids=tuple(
                forbidden_outer_valid_source_ids
            ),
            forbidden_outer_test_source_ids=tuple(
                forbidden_outer_test_source_ids
            ),
            _eligible_outer_train=eligible_outer_train,
            _ae_train=ae_train,
            _internal_validation=internal_validation,
        )

    @property
    def eligible_outer_train(self) -> pd.DataFrame:
        """Return a defensive copy of all measured rows eligible for refit."""
        return self._eligible_outer_train.copy(deep=True)

    @property
    def ae_train(self) -> pd.DataFrame:
        """Return a defensive copy of measured rows visible during AE fitting."""
        return self._ae_train.copy(deep=True)

    @property
    def internal_validation(self) -> pd.DataFrame:
        """Return a defensive copy of untouched measured validation rows."""
        return self._internal_validation.copy(deep=True)

    @property
    def source_id_hashes(self) -> dict[str, str]:
        """Return stable hashes for each measured-data role."""
        return {
            "eligible_outer_train": self.eligible_source_id_hash,
            "ae_train": self.ae_train_source_id_hash,
            "internal_validation": self.internal_validation_source_id_hash,
        }

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return concise reproducibility metadata for a scientific manifest."""
        return {
            "schema_version": AE_INNER_SPLIT_SCHEMA_VERSION,
            "seed": self.seed,
            "valid_size": float(self.valid_size),
            "source_id_column": self.source_id_column,
            "group_column": self.group_column,
            "minimum_rows": self.minimum_rows,
            "minimum_groups": self.minimum_groups,
            "n_eligible_outer_train": len(self.eligible_source_ids),
            "n_ae_train": len(self.ae_train_source_ids),
            "n_internal_validation": len(
                self.internal_validation_source_ids
            ),
            "source_id_hashes": self.source_id_hashes,
            "split_hash": self.split_hash,
        }


def make_ae_inner_split(
    measured_outer_train: pd.DataFrame,
    *,
    valid_size: float,
    seed: int,
    forbidden_outer_valid_source_ids: Iterable[str] = (),
    forbidden_outer_test_source_ids: Iterable[str] = (),
    source_id_column: str = "source_row_id",
    group_column: str = "canonical_reaction_key",
    minimum_rows: int = 4,
    minimum_groups: int = 2,
) -> AEInnerSplit:
    """Create a deterministic, row-order-invariant grouped inner split."""
    _validate_split_parameters(
        seed=seed,
        valid_size=valid_size,
        minimum_rows=minimum_rows,
        minimum_groups=minimum_groups,
    )
    eligible = _normalize_measured_frame(
        measured_outer_train,
        name="measured_outer_train",
        source_id_column=source_id_column,
        group_column=group_column,
    )
    forbidden_valid = _normalize_source_ids(
        forbidden_outer_valid_source_ids,
        name="forbidden_outer_valid_source_ids",
    )
    forbidden_test = _normalize_source_ids(
        forbidden_outer_test_source_ids,
        name="forbidden_outer_test_source_ids",
    )
    _reject_forbidden_contamination(
        set(eligible[source_id_column]),
        forbidden_valid,
        forbidden_test,
    )
    n_rows = len(eligible)
    n_groups = eligible[group_column].nunique()
    if n_rows < minimum_rows:
        raise ValueError(
            "AE inner split has insufficient measured-row support: "
            f"requires at least {minimum_rows}, observed {n_rows}."
        )
    if n_groups < minimum_groups:
        raise ValueError(
            "AE inner split has insufficient canonical-group support: "
            f"requires at least {minimum_groups}, observed {n_groups}."
        )

    group_sizes = (
        eligible.groupby(group_column, sort=True)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    group_sizes["_rank"] = group_sizes[group_column].map(
        lambda group: stable_hash(
            {
                "schema_version": AE_INNER_SPLIT_SCHEMA_VERSION,
                "seed": seed,
                "canonical_group": group,
            }
        )
    )
    traversal = group_sizes.sort_values(
        ["_rank", group_column],
        kind="mergesort",
    )
    target_valid_rows = max(1, int(math.floor(n_rows * float(valid_size))))
    validation_groups: list[str] = []
    validation_rows = 0
    for row in traversal.to_dict(orient="records"):
        if len(validation_groups) >= n_groups - 1:
            break
        validation_groups.append(str(row[group_column]))
        validation_rows += int(row["n_rows"])
        if validation_rows >= target_valid_rows:
            break
    validation_group_set = set(validation_groups)
    internal_validation = eligible.loc[
        eligible[group_column].isin(validation_group_set)
    ].copy()
    ae_train = eligible.loc[
        ~eligible[group_column].isin(validation_group_set)
    ].copy()
    return AEInnerSplit.from_partitions(
        eligible,
        ae_train,
        internal_validation,
        seed=seed,
        valid_size=valid_size,
        forbidden_outer_valid_source_ids=forbidden_valid,
        forbidden_outer_test_source_ids=forbidden_test,
        source_id_column=source_id_column,
        group_column=group_column,
        minimum_rows=minimum_rows,
        minimum_groups=minimum_groups,
    )


def _validate_split_parameters(
    *,
    seed: int,
    valid_size: float,
    minimum_rows: int,
    minimum_groups: int,
) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("AE inner split seed must be an integer.")
    if (
        isinstance(valid_size, bool)
        or not isinstance(valid_size, (int, float))
        or not math.isfinite(float(valid_size))
        or not 0 < float(valid_size) < 1
    ):
        raise ValueError("AE inner split valid_size must be finite and in (0, 1).")
    if (
        not isinstance(minimum_rows, int)
        or isinstance(minimum_rows, bool)
        or minimum_rows < 2
    ):
        raise ValueError("AE inner split minimum_rows must be an integer of at least 2.")
    if (
        not isinstance(minimum_groups, int)
        or isinstance(minimum_groups, bool)
        or minimum_groups < 2
    ):
        raise ValueError(
            "AE inner split minimum_groups must be an integer of at least 2."
        )


def _normalize_measured_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    source_id_column: str,
    group_column: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"AE inner split {name} must be a pandas DataFrame.")
    missing = sorted({source_id_column, group_column} - set(frame))
    if missing:
        raise ValueError(
            f"AE inner split {name} is missing required columns: {missing}."
        )
    if frame.empty:
        raise ValueError(f"AE inner split {name} must not be empty.")
    normalized = frame.copy(deep=True)
    if normalized[source_id_column].isna().any():
        raise ValueError(f"AE inner split {name} contains missing source IDs.")
    if normalized[group_column].isna().any():
        raise ValueError(f"AE inner split {name} contains missing canonical groups.")
    normalized[source_id_column] = normalized[source_id_column].astype(str)
    normalized[group_column] = normalized[group_column].astype(str)
    if normalized[source_id_column].str.strip().eq("").any():
        raise ValueError(f"AE inner split {name} contains empty source IDs.")
    if normalized[group_column].str.strip().eq("").any():
        raise ValueError(f"AE inner split {name} contains empty canonical groups.")
    if normalized[source_id_column].duplicated().any():
        raise ValueError(f"AE inner split {name} contains duplicate source IDs.")
    return normalized.sort_values(source_id_column, kind="mergesort").reset_index(
        drop=True
    )


def _normalize_source_ids(
    values: Iterable[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of source IDs, not a string.")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain non-empty string source IDs.")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} contains duplicate source IDs.")
    return tuple(sorted(normalized))


def _reject_forbidden_contamination(
    eligible_ids: set[str],
    forbidden_valid: tuple[str, ...],
    forbidden_test: tuple[str, ...],
) -> None:
    valid_set = set(forbidden_valid)
    test_set = set(forbidden_test)
    outer_overlap = valid_set & test_set
    if outer_overlap:
        raise ValueError(
            "Outer validation and test source IDs overlap: "
            f"{sorted(outer_overlap)}."
        )
    contaminated_valid = eligible_ids & valid_set
    contaminated_test = eligible_ids & test_set
    if contaminated_valid or contaminated_test:
        raise ValueError(
            "Forbidden outer rows entered eligible AE training data: "
            f"outer_valid={sorted(contaminated_valid)}, "
            f"outer_test={sorted(contaminated_test)}."
        )


def _validate_partition_membership(
    eligible: pd.DataFrame,
    ae_train: pd.DataFrame,
    internal_validation: pd.DataFrame,
    *,
    source_id_column: str,
    group_column: str,
    minimum_rows: int,
    minimum_groups: int,
    forbidden_outer_valid_source_ids: tuple[str, ...],
    forbidden_outer_test_source_ids: tuple[str, ...],
) -> None:
    eligible_ids = set(eligible[source_id_column])
    ae_train_ids = set(ae_train[source_id_column])
    internal_valid_ids = set(internal_validation[source_id_column])
    _reject_forbidden_contamination(
        eligible_ids,
        forbidden_outer_valid_source_ids,
        forbidden_outer_test_source_ids,
    )
    if len(eligible) < minimum_rows:
        raise ValueError(
            "AE inner split has insufficient measured-row support: "
            f"requires at least {minimum_rows}, observed {len(eligible)}."
        )
    n_groups = eligible[group_column].nunique()
    if n_groups < minimum_groups:
        raise ValueError(
            "AE inner split has insufficient canonical-group support: "
            f"requires at least {minimum_groups}, observed {n_groups}."
        )
    overlap = ae_train_ids & internal_valid_ids
    if overlap:
        raise ValueError(
            "AE training and internal-validation source IDs overlap: "
            f"{sorted(overlap)}."
        )
    if ae_train_ids | internal_valid_ids != eligible_ids:
        missing = eligible_ids - (ae_train_ids | internal_valid_ids)
        unexpected = (ae_train_ids | internal_valid_ids) - eligible_ids
        raise ValueError(
            "AE inner partitions do not exactly cover eligible outer training rows: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    if tuple(ae_train.columns) != tuple(eligible.columns) or tuple(
        internal_validation.columns
    ) != tuple(eligible.columns):
        raise ValueError("AE inner partition schemas do not match eligible rows.")
    _validate_partition_rows(
        eligible,
        ae_train,
        source_ids=ae_train_ids,
        source_id_column=source_id_column,
        name="ae_train",
    )
    _validate_partition_rows(
        eligible,
        internal_validation,
        source_ids=internal_valid_ids,
        source_id_column=source_id_column,
        name="internal_validation",
    )
    train_groups = set(ae_train[group_column])
    valid_groups = set(internal_validation[group_column])
    group_overlap = train_groups & valid_groups
    if group_overlap:
        raise ValueError(
            "Canonical reaction groups cross AE train/internal validation: "
            f"{sorted(group_overlap)}."
        )


def _validate_partition_rows(
    eligible: pd.DataFrame,
    partition: pd.DataFrame,
    *,
    source_ids: set[str],
    source_id_column: str,
    name: str,
) -> None:
    expected = eligible.loc[eligible[source_id_column].isin(source_ids)]
    expected = expected.sort_values(source_id_column, kind="mergesort").reset_index(
        drop=True
    )
    if not partition.equals(expected):
        raise ValueError(
            f"AE inner split {name} rows do not exactly match eligible measured rows."
        )
