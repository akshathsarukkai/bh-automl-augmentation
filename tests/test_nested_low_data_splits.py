"""Tests for cumulative group-safe low-data training subsets."""

from __future__ import annotations

import pandas as pd
from tests.canonical_test_utils import grouped_assignment_frame

from bh_augmentation.data.canonical_splits import (
    build_grouped_outer_assignments,
    build_nested_low_data_assignments,
)


def _low_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = build_grouped_outer_assignments(
        grouped_assignment_frame(),
        seed=3,
        train_size=0.8,
        valid_size=0.1,
        test_size=0.1,
    )
    low = build_nested_low_data_assignments(
        outer,
        train_fractions=[0.01, 0.05, 0.1, 0.2, 1.0],
    )
    return outer, low


def test_low_data_subsets_are_nested_and_keep_complete_groups() -> None:
    _, low = _low_assignments()
    previous_rows: set[str] = set()
    for fraction in sorted(low["train_fraction"].unique()):
        subset = low.loc[
            low["train_fraction"].eq(fraction)
            & low["included_in_training_subset"]
        ]
        current_rows = set(subset["source_row_id"])
        assert previous_rows.issubset(current_rows)
        previous_rows = current_rows
        included_by_group = low.loc[low["train_fraction"].eq(fraction)].groupby(
            "canonical_reaction_key"
        )["included_in_training_subset"].nunique()
        assert included_by_group.max() == 1


def test_validation_and_test_rows_are_fixed_across_fractions() -> None:
    outer, low = _low_assignments()
    expected_valid = set(outer.loc[outer["outer_split"] == "valid", "source_row_id"])
    expected_test = set(outer.loc[outer["outer_split"] == "test", "source_row_id"])
    for _, fraction_rows in low.groupby("train_fraction"):
        assert set(
            fraction_rows.loc[fraction_rows["outer_split"] == "valid", "source_row_id"]
        ) == expected_valid
        assert set(
            fraction_rows.loc[fraction_rows["outer_split"] == "test", "source_row_id"]
        ) == expected_test
        assert not fraction_rows.loc[
            fraction_rows["outer_split"].isin(["valid", "test"]),
            "included_in_training_subset",
        ].any()


def test_full_fraction_contains_entire_outer_training_pool() -> None:
    outer, low = _low_assignments()
    full = low.loc[low["train_fraction"].eq(1.0)]
    included = set(full.loc[full["included_in_training_subset"], "source_row_id"])
    expected = set(outer.loc[outer["outer_split"] == "train", "source_row_id"])
    assert included == expected
