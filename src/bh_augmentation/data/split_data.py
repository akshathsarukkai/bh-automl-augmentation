"""Deterministic train/validation/test split helpers."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

SplitDict = dict[str, pd.DataFrame]
GroupFold = tuple[object, SplitDict]
SPLIT_KEYS: Final[tuple[str, str, str]] = ("train", "valid", "test")


def random_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> SplitDict:
    """Create a deterministic random train/validation/test split."""
    _validate_non_empty(df)
    _validate_split_sizes(train_size, valid_size, test_size)

    shuffled_indices = _shuffled_index_array(df.index, seed)
    n_train, n_valid, _ = _split_counts(
        n_rows=len(df),
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )

    train_indices = shuffled_indices[:n_train]
    valid_indices = shuffled_indices[n_train : n_train + n_valid]
    test_indices = shuffled_indices[n_train + n_valid :]
    return _make_split_dict(df, train_indices, valid_indices, test_indices)


def low_data_split(
    df: pd.DataFrame,
    train_fraction: float,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> SplitDict:
    """Create a split using only a fraction of the available training rows.

    Validation and test sizes are computed first. The remaining rows are the
    full training pool, and `train_fraction` selects a deterministic subset of
    that pool.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be greater than 0 and at most 1.")
    if not 0 < valid_size < 1 or not 0 < test_size < 1:
        raise ValueError("valid_size and test_size must both be between 0 and 1.")
    if valid_size + test_size >= 1:
        raise ValueError("valid_size + test_size must be less than 1.")

    full_train_size = 1.0 - valid_size - test_size
    full_split = random_split(
        df,
        train_size=full_train_size,
        valid_size=valid_size,
        test_size=test_size,
        seed=seed,
    )

    train_pool = full_split["train"]
    n_low_train = max(1, int(math.floor(len(train_pool) * train_fraction)))
    low_train_indices = _shuffled_index_array(train_pool.index, seed + 1)[:n_low_train]
    return {
        "train": train_pool.loc[low_train_indices].copy(),
        "valid": full_split["valid"],
        "test": full_split["test"],
    }


def subset_train_split(
    splits: SplitDict,
    train_fraction: float,
    seed: int = 42,
) -> SplitDict:
    """Return a copy of splits with a deterministic subset of training rows.

    Validation and test splits are copied unchanged. This is useful for
    low-data experiments where validation/test should remain fixed across
    different training fractions.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be greater than 0 and at most 1.")
    for key in SPLIT_KEYS:
        if key not in splits:
            raise ValueError(f"Missing split key: {key}")

    train = splits["train"]
    _validate_non_empty(train)
    n_train = max(1, int(math.floor(len(train) * train_fraction)))
    train_indices = _shuffled_index_array(train.index, seed)[:n_train]
    return {
        "train": train.loc[train_indices].copy(),
        "valid": splits["valid"].copy(),
        "test": splits["test"].copy(),
    }


def heldout_group_split(
    df: pd.DataFrame,
    group_column: str,
    heldout_fraction: float = 0.2,
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> SplitDict:
    """Create a split where test groups are held out from training.

    The test split is selected by group. Validation rows are sampled from the
    remaining rows, so validation may share groups with training but the heldout
    test groups never appear in train or validation.
    """
    _validate_non_empty(df)
    if group_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Group column is missing from DataFrame: {group_column}. "
            f"Available columns: {available}."
        )
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be between 0 and 1.")
    if not 0 <= valid_fraction < 1:
        raise ValueError("valid_fraction must be at least 0 and less than 1.")

    groups = pd.Series(df[group_column].dropna().unique())
    if groups.empty:
        raise ValueError(f"Group column has no non-missing groups: {group_column}")

    shuffled_groups = _shuffle_values(groups.to_numpy(), seed)
    n_test_groups = max(1, int(math.ceil(len(shuffled_groups) * heldout_fraction)))
    if n_test_groups >= len(shuffled_groups):
        raise ValueError("heldout_fraction leaves no groups available for training.")

    test_groups = set(shuffled_groups[:n_test_groups].tolist())
    test_mask = df[group_column].isin(test_groups)
    test = df.loc[test_mask].copy()
    remaining = df.loc[~test_mask].copy()
    if remaining.empty:
        raise ValueError("Held-out group split produced an empty train/valid pool.")

    n_valid = int(math.floor(len(remaining) * valid_fraction))
    shuffled_remaining_indices = _shuffled_index_array(remaining.index, seed + 1)
    valid_indices = shuffled_remaining_indices[:n_valid]
    train_indices = shuffled_remaining_indices[n_valid:]
    if len(train_indices) == 0:
        raise ValueError("valid_fraction leaves no rows available for training.")

    return {
        "train": remaining.loc[train_indices].copy(),
        "valid": remaining.loc[valid_indices].copy(),
        "test": test,
    }


def leave_one_group_out_splits(
    df: pd.DataFrame,
    group_column: str,
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> list[GroupFold]:
    """Create one deterministic train/valid/test fold per group value."""
    _validate_non_empty(df)
    if group_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Group column is missing from DataFrame: {group_column}. "
            f"Available columns: {available}."
        )
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")
    if df[group_column].isna().any():
        raise ValueError(f"Group column contains missing values: {group_column}")

    groups = sorted(df[group_column].unique().tolist(), key=str)
    if len(groups) < 2:
        raise ValueError("Leave-one-group-out requires at least two unique groups.")

    folds: list[GroupFold] = []
    for fold_index, heldout_group in enumerate(groups):
        test_mask = df[group_column] == heldout_group
        test = df.loc[test_mask].copy()
        train_valid = df.loc[~test_mask].copy()

        n_valid = max(1, int(math.floor(len(train_valid) * valid_fraction)))
        shuffled_indices = _shuffled_index_array(train_valid.index, seed + fold_index)
        valid_indices = shuffled_indices[:n_valid]
        train_indices = shuffled_indices[n_valid:]
        if len(train_indices) == 0:
            raise ValueError("valid_fraction leaves no rows available for training.")

        folds.append(
            (
                heldout_group,
                {
                    "train": train_valid.loc[train_indices].copy(),
                    "valid": train_valid.loc[valid_indices].copy(),
                    "test": test,
                },
            )
        )
    return folds


def _validate_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")


def _validate_split_sizes(train_size: float, valid_size: float, test_size: float) -> None:
    sizes = {
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
    }
    for name, size in sizes.items():
        if not 0 < size < 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    total = train_size + valid_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_size, valid_size, and test_size must sum to 1.0; "
            f"got {total:.6f}."
        )


def _split_counts(
    n_rows: int,
    train_size: float,
    valid_size: float,
    test_size: float,
) -> tuple[int, int, int]:
    n_train = int(math.floor(n_rows * train_size))
    n_valid = int(math.floor(n_rows * valid_size))
    n_test = n_rows - n_train - n_valid
    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError(
            "Split sizes produced an empty split; use more rows or adjust sizes."
        )
    return n_train, n_valid, n_test


def _make_split_dict(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_indices: np.ndarray,
) -> SplitDict:
    return {
        "train": df.loc[train_indices].copy(),
        "valid": df.loc[valid_indices].copy(),
        "test": df.loc[test_indices].copy(),
    }


def _shuffled_index_array(index: pd.Index, seed: int) -> np.ndarray:
    return _shuffle_values(index.to_numpy(), seed)


def _shuffle_values(values: np.ndarray, seed: int) -> np.ndarray:
    shuffled = np.array(values, copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    return shuffled
