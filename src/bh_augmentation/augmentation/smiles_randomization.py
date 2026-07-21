"""Legacy component-column randomized SMILES augmentation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd


def randomize_smiles(smiles: str, seed: int | None = None) -> str:
    """Return a randomized non-canonical SMILES string.

    RDKit is imported lazily because it is an optional dependency. If RDKit is
    unavailable, the SMILES is invalid, or the molecule cannot be randomized,
    this function returns the original input unchanged and emits a warning.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        warnings.warn("Invalid SMILES value; returning unchanged.", stacklevel=2)
        return smiles

    try:
        from rdkit import Chem
    except ImportError:
        warnings.warn(
            "RDKit is not installed; returning SMILES unchanged.",
            stacklevel=2,
        )
        return smiles

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        warnings.warn(
            f"RDKit could not parse SMILES; returning unchanged: {smiles}",
            stacklevel=2,
        )
        return smiles

    try:
        n_atoms = molecule.GetNumAtoms()
        if n_atoms <= 1:
            return Chem.MolToSmiles(molecule, canonical=False)

        rng = np.random.default_rng(seed)
        atom_order = rng.permutation(n_atoms).tolist()
        randomized_molecule = Chem.RenumberAtoms(molecule, atom_order)
        return Chem.MolToSmiles(randomized_molecule, canonical=False)
    except Exception as exc:  # pragma: no cover - defensive around RDKit internals.
        warnings.warn(
            f"Could not randomize SMILES; returning unchanged: {smiles}. Error: {exc}",
            stacklevel=2,
        )
        return smiles


def augment_randomized_smiles(
    df: pd.DataFrame,
    smiles_columns: Sequence[str],
    n_augments: int = 1,
    seed: int = 42,
    include_original: bool = True,
) -> pd.DataFrame:
    """Create label-preserving rows with randomized component SMILES.

    The returned DataFrame includes `source_reaction_id` and `is_augmented`.
    Original rows are included by default with `is_augmented=False`.
    Augmented rows preserve all non-SMILES values, including `yield`, and their
    `source_reaction_id` links back to the original `reaction_id` when present.
    """
    if n_augments < 0:
        raise ValueError("n_augments must be greater than or equal to 0.")
    missing_columns = [column for column in smiles_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing SMILES columns for augmentation: {missing_columns}")

    source_ids = _source_reaction_ids(df)
    frames: list[pd.DataFrame] = []

    if include_original:
        original = df.copy()
        original["source_reaction_id"] = source_ids
        original["is_augmented"] = False
        frames.append(original)

    for augment_index in range(n_augments):
        augmented = df.copy()
        augmented["source_reaction_id"] = source_ids
        augmented["is_augmented"] = True
        if "reaction_id" in augmented.columns:
            augmented["reaction_id"] = [
                f"{source_id}_aug_{augment_index + 1}" for source_id in source_ids
            ]

        for row_position, row_index in enumerate(df.index):
            for column_position, column in enumerate(smiles_columns):
                value = df.at[row_index, column]
                smiles_seed = seed + (augment_index * 1_000_003) + (row_position * 997) + column_position
                augmented.at[row_index, column] = randomize_smiles(value, seed=smiles_seed)

        frames.append(augmented)

    if not frames:
        empty = df.iloc[0:0].copy()
        empty["source_reaction_id"] = pd.Series(dtype=object)
        empty["is_augmented"] = pd.Series(dtype=bool)
        return empty

    return pd.concat(frames, ignore_index=True)


def _source_reaction_ids(df: pd.DataFrame) -> pd.Series:
    if "reaction_id" in df.columns:
        return df["reaction_id"].astype(str)
    return pd.Series(df.index.astype(str), index=df.index)
