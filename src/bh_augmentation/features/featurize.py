"""Featurization helpers for molecular components and reaction conditions."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a Morgan fingerprint bit vector for a SMILES string.

    Invalid or missing SMILES values return an all-zero fingerprint and emit a
    warning. RDKit is imported lazily so the package can be used for non-RDKit
    workflows until this function is called.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        warnings.warn("Invalid SMILES encountered; returning zero fingerprint.", stacklevel=2)
        return np.zeros(n_bits, dtype=np.float32)

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for Morgan fingerprints. Install RDKit with "
            "`conda install -c conda-forge rdkit` or an equivalent package "
            "manager command."
        ) from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        warnings.warn(
            f"Invalid SMILES encountered; returning zero fingerprint: {smiles}",
            stacklevel=2,
        )
        return np.zeros(n_bits, dtype=np.float32)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius,
        nBits=n_bits,
    )
    array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array.astype(np.float32)


def component_fingerprint_features(
    df: pd.DataFrame,
    smiles_columns: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Build concatenated Morgan fingerprints for component SMILES columns."""
    if not smiles_columns:
        return np.empty((len(df), 0), dtype=np.float32)

    column_features: list[np.ndarray] = []
    for column in smiles_columns:
        if column not in df.columns:
            raise ValueError(f"Missing SMILES column for featurization: {column}")

        fingerprints = [
            morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
            for smiles in df[column].fillna("")
        ]
        column_features.append(np.vstack(fingerprints).astype(np.float32))

    return np.hstack(column_features).astype(np.float32)


def one_hot_condition_features(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode categorical condition columns deterministically.

    Categories are sorted lexicographically within each column. Missing values
    are encoded as `"UNKNOWN"`.
    """
    if not categorical_columns:
        return np.empty((len(df), 0), dtype=np.float32), []

    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in categorical_columns:
        if column not in df.columns:
            raise ValueError(f"Missing categorical column for featurization: {column}")

        values = df[column].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")
        categories = sorted(values.unique().tolist())
        encoded = np.zeros((len(df), len(categories)), dtype=np.float32)
        category_to_index = {category: index for index, category in enumerate(categories)}
        for row_index, value in enumerate(values):
            encoded[row_index, category_to_index[value]] = 1.0

        arrays.append(encoded)
        names.extend([f"{column}__{category}" for category in categories])

    return np.hstack(arrays).astype(np.float32), names


def build_feature_matrix(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a feature matrix, target vector, and feature names.

    The target column is `yield`. Supported feature configuration keys are:
    `smiles_columns`, `categorical_columns`, `radius`, and `n_bits`.
    """
    if "yield" not in df.columns:
        raise ValueError("Target column is required for feature matrix construction: yield")

    smiles_columns = list(feature_config.get("smiles_columns", []))
    categorical_columns = list(feature_config.get("categorical_columns", []))
    radius = int(feature_config.get("radius", 2))
    n_bits = int(feature_config.get("n_bits", 2048))

    fingerprint_features = component_fingerprint_features(
        df,
        smiles_columns=smiles_columns,
        radius=radius,
        n_bits=n_bits,
    )
    fingerprint_names = [
        f"{column}__morgan_{bit_index}"
        for column in smiles_columns
        for bit_index in range(n_bits)
    ]

    condition_features, condition_names = one_hot_condition_features(
        df,
        categorical_columns=categorical_columns,
    )

    X = np.hstack([fingerprint_features, condition_features]).astype(np.float32)
    y = pd.to_numeric(df["yield"], errors="coerce").to_numpy(dtype=np.float32)
    feature_names = fingerprint_names + condition_names
    return X, y, feature_names
