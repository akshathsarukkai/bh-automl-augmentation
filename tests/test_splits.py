"""Tests for deterministic train/validation/test split helpers."""

import pandas as pd
import pytest

from bh_augmentation.data.split_data import (
    heldout_group_split,
    leave_one_group_out_splits,
    low_data_split,
    random_split,
)


def _sample_df(n_rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(n_rows)],
            "group": [f"aryl_{index // 5}" for index in range(n_rows)],
            "yield": [float(index % 100) for index in range(n_rows)],
        },
        index=[f"row_{index:03d}" for index in range(n_rows)],
    )


def _assert_no_index_overlap(splits: dict[str, pd.DataFrame]) -> None:
    train_index = set(splits["train"].index)
    valid_index = set(splits["valid"].index)
    test_index = set(splits["test"].index)

    assert train_index.isdisjoint(valid_index)
    assert train_index.isdisjoint(test_index)
    assert valid_index.isdisjoint(test_index)


def test_random_split_sizes_are_approximately_correct() -> None:
    """Random splits should respect requested fractions for deterministic data."""
    df = _sample_df(100)

    splits = random_split(df, train_size=0.8, valid_size=0.1, test_size=0.1, seed=7)

    assert set(splits) == {"train", "valid", "test"}
    assert len(splits["train"]) == 80
    assert len(splits["valid"]) == 10
    assert len(splits["test"]) == 10
    _assert_no_index_overlap(splits)


def test_random_split_is_deterministic() -> None:
    """Using the same seed should return the same split indices."""
    df = _sample_df(30)

    first = random_split(df, seed=123)
    second = random_split(df, seed=123)

    assert first["train"].index.tolist() == second["train"].index.tolist()
    assert first["valid"].index.tolist() == second["valid"].index.tolist()
    assert first["test"].index.tolist() == second["test"].index.tolist()


def test_random_split_validates_sizes() -> None:
    """Split fractions should be validated before sampling."""
    df = _sample_df(20)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        random_split(df, train_size=0.7, valid_size=0.2, test_size=0.2)


def test_low_data_split_uses_requested_fraction_of_training_data() -> None:
    """Low-data split should subsample only the training pool."""
    df = _sample_df(100)

    splits = low_data_split(df, train_fraction=0.25, valid_size=0.1, test_size=0.1, seed=7)

    assert len(splits["train"]) == 20
    assert len(splits["valid"]) == 10
    assert len(splits["test"]) == 10
    _assert_no_index_overlap(splits)


def test_low_data_split_validates_train_fraction() -> None:
    """Invalid low-data train fractions should raise helpful errors."""
    df = _sample_df(20)

    with pytest.raises(ValueError, match="train_fraction"):
        low_data_split(df, train_fraction=0)


def test_heldout_group_split_prevents_group_leakage() -> None:
    """Held-out test groups should not appear in train or validation."""
    df = _sample_df(100)

    splits = heldout_group_split(
        df,
        group_column="group",
        heldout_fraction=0.2,
        valid_fraction=0.1,
        seed=42,
    )

    train_groups = set(splits["train"]["group"])
    valid_groups = set(splits["valid"]["group"])
    test_groups = set(splits["test"]["group"])
    assert test_groups.isdisjoint(train_groups)
    assert test_groups.isdisjoint(valid_groups)
    assert len(test_groups) == 4
    assert len(splits["test"]) == 20
    _assert_no_index_overlap(splits)


def test_heldout_group_split_validates_group_column() -> None:
    """Missing group columns should raise a clear ValueError."""
    df = _sample_df(20)

    with pytest.raises(ValueError, match="Available columns: reaction_id, group, yield"):
        heldout_group_split(df, group_column="missing")


def test_heldout_group_split_accepts_arbitrary_existing_column() -> None:
    """Any existing dataframe column should be usable as a held-out group."""
    df = _sample_df(30)
    df["product_key"] = [f"product_{index // 5}" for index in range(len(df))]

    splits = heldout_group_split(
        df,
        group_column="product_key",
        heldout_fraction=0.2,
        valid_fraction=0.1,
        seed=42,
    )

    train_groups = set(splits["train"]["product_key"])
    test_groups = set(splits["test"]["product_key"])
    assert train_groups.isdisjoint(test_groups)


def test_leave_one_group_out_creates_one_leak_free_fold_per_group() -> None:
    """LOGO should hold every group out exactly once."""
    df = _sample_df(30)

    folds = leave_one_group_out_splits(
        df,
        group_column="group",
        valid_fraction=0.1,
        seed=42,
    )

    assert len(folds) == df["group"].nunique()
    assert {heldout for heldout, _ in folds} == set(df["group"])
    for heldout, splits in folds:
        assert set(splits["test"]["group"]) == {heldout}
        assert heldout not in set(splits["train"]["group"])
        assert heldout not in set(splits["valid"]["group"])
        _assert_no_index_overlap(splits)


def test_leave_one_group_out_rejects_missing_group_column() -> None:
    """LOGO missing-column errors should list available columns."""
    df = _sample_df(20)

    with pytest.raises(ValueError, match="Available columns: reaction_id, group, yield"):
        leave_one_group_out_splits(df, group_column="not_a_column")
