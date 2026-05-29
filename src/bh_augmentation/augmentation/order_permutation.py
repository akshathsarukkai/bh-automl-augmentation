"""Safe reaction component order-permutation augmentation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def permute_reaction_components(
    df: pd.DataFrame,
    component_columns: Sequence[str],
    n_permutations: int = 1,
    seed: int = 42,
    include_original: bool = True,
) -> pd.DataFrame:
    """Create augmented rows by permuting explicitly allowed component columns.

    This function does not infer chemical exchangeability. The caller must pass
    only columns that are semantically safe to permute for the experiment.
    Yield labels and all non-permuted columns are preserved.
    """
    if n_permutations < 0:
        raise ValueError("n_permutations must be greater than or equal to 0.")
    if len(component_columns) < 2 and n_permutations > 0:
        raise ValueError("At least two component_columns are required for permutation.")

    missing_columns = [column for column in component_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing component columns for permutation: {missing_columns}")

    source_ids = _source_reaction_ids(df)
    frames: list[pd.DataFrame] = []

    if include_original:
        original = df.copy()
        original["is_augmented"] = False
        original["augmentation_type"] = "original"
        original["source_reaction_id"] = source_ids
        frames.append(original)

    rng = np.random.default_rng(seed)
    component_columns = list(component_columns)
    for permutation_index in range(n_permutations):
        augmented = df.copy()
        augmented["is_augmented"] = True
        augmented["augmentation_type"] = "order_permutation"
        augmented["source_reaction_id"] = source_ids
        if "reaction_id" in augmented.columns:
            augmented["reaction_id"] = [
                f"{source_id}_perm_{permutation_index + 1}" for source_id in source_ids
            ]

        for row_position, row_index in enumerate(df.index):
            permutation = _non_identity_permutation(
                len(component_columns),
                rng,
                offset=permutation_index + row_position,
            )
            original_values = df.loc[row_index, component_columns].to_numpy(copy=True)
            permuted_values = original_values[permutation]
            for column, value in zip(component_columns, permuted_values, strict=True):
                augmented.at[row_index, column] = value

        frames.append(augmented)

    if not frames:
        empty = df.iloc[0:0].copy()
        empty["is_augmented"] = pd.Series(dtype=bool)
        empty["augmentation_type"] = pd.Series(dtype=object)
        empty["source_reaction_id"] = pd.Series(dtype=object)
        return empty

    return pd.concat(frames, ignore_index=True)


def _non_identity_permutation(
    n_items: int,
    rng: np.random.Generator,
    offset: int,
) -> np.ndarray:
    """Return a deterministic non-identity permutation when possible."""
    permutation = rng.permutation(n_items)
    identity = np.arange(n_items)
    if np.array_equal(permutation, identity):
        permutation = np.roll(identity, (offset % (n_items - 1)) + 1)
    return permutation


def _source_reaction_ids(df: pd.DataFrame) -> pd.Series:
    if "reaction_id" in df.columns:
        return df["reaction_id"].astype(str)
    return pd.Series(df.index.astype(str), index=df.index)
