"""Tests for molecular component featurization."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.features.featurize import (
    build_feature_matrix,
    component_fingerprint_features,
    morgan_fingerprint,
    one_hot_condition_features,
)


@contextmanager
def fake_rdkit_modules() -> Iterator[None]:
    """Install a minimal fake RDKit implementation for unit tests."""
    old_modules = {
        name: sys.modules.get(name)
        for name in ["rdkit", "rdkit.Chem", "rdkit.Chem.AllChem", "rdkit.DataStructs"]
    }

    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    all_chem = types.ModuleType("rdkit.Chem.AllChem")
    data_structs = types.ModuleType("rdkit.DataStructs")

    def mol_from_smiles(smiles: str) -> str | None:
        if smiles == "INVALID":
            return None
        return smiles

    def get_morgan_fingerprint_as_bit_vect(
        molecule: str,
        radius: int,
        nBits: int,
    ) -> np.ndarray:
        fingerprint = np.zeros(nBits, dtype=np.int8)
        fingerprint[(len(molecule) + radius) % nBits] = 1
        fingerprint[(sum(ord(char) for char in molecule) + radius) % nBits] = 1
        return fingerprint

    def convert_to_numpy_array(fingerprint: np.ndarray, array: np.ndarray) -> None:
        array[:] = fingerprint

    chem.MolFromSmiles = mol_from_smiles
    all_chem.GetMorganFingerprintAsBitVect = get_morgan_fingerprint_as_bit_vect
    data_structs.ConvertToNumpyArray = convert_to_numpy_array
    chem.AllChem = all_chem
    rdkit.Chem = chem
    rdkit.DataStructs = data_structs

    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = chem
    sys.modules["rdkit.Chem.AllChem"] = all_chem
    sys.modules["rdkit.DataStructs"] = data_structs
    try:
        yield
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_morgan_fingerprint_shape_and_dtype() -> None:
    """Morgan fingerprints should have deterministic shape and dtype."""
    with fake_rdkit_modules():
        fingerprint = morgan_fingerprint("CCO", radius=2, n_bits=16)

    assert fingerprint.shape == (16,)
    assert fingerprint.dtype == np.float32
    assert fingerprint.sum() == 2


def test_morgan_fingerprint_invalid_smiles_returns_zeros() -> None:
    """Invalid SMILES should warn and return an all-zero fingerprint."""
    with fake_rdkit_modules(), pytest.warns(UserWarning, match="Invalid SMILES"):
        fingerprint = morgan_fingerprint("INVALID", n_bits=16)

    np.testing.assert_array_equal(fingerprint, np.zeros(16, dtype=np.float32))


def test_component_fingerprint_features_concatenate_columns() -> None:
    """Component fingerprints should concatenate each requested SMILES column."""
    df = pd.DataFrame(
        {
            "aryl_halide_smiles": ["c1ccccc1", "INVALID"],
            "amine_smiles": ["N", "CCO"],
        }
    )

    with fake_rdkit_modules(), pytest.warns(UserWarning, match="Invalid SMILES"):
        features = component_fingerprint_features(
            df,
            smiles_columns=["aryl_halide_smiles", "amine_smiles"],
            n_bits=8,
        )

    assert features.shape == (2, 16)
    np.testing.assert_array_equal(features[1, :8], np.zeros(8, dtype=np.float32))


def test_one_hot_condition_features_are_deterministic() -> None:
    """One-hot features should sort categories and expose feature names."""
    df = pd.DataFrame({"solvent": ["toluene", "DMF", None], "base": ["K3PO4", "K3PO4", ""]})

    features, names = one_hot_condition_features(df, ["solvent", "base"])

    assert names == [
        "solvent__DMF",
        "solvent__UNKNOWN",
        "solvent__toluene",
        "base__K3PO4",
        "base__UNKNOWN",
    ]
    assert features.shape == (3, 5)
    np.testing.assert_array_equal(features.sum(axis=1), np.array([2.0, 2.0, 2.0]))


def test_build_feature_matrix_returns_X_y_and_feature_names() -> None:
    """The full feature matrix should combine fingerprints and conditions."""
    df = pd.DataFrame(
        {
            "aryl_halide_smiles": ["CCO", "c1ccccc1"],
            "amine_smiles": ["N", "INVALID"],
            "solvent": ["DMF", "toluene"],
            "yield": [10.0, 20.0],
        }
    )
    feature_config = {
        "smiles_columns": ["aryl_halide_smiles", "amine_smiles"],
        "categorical_columns": ["solvent"],
        "n_bits": 8,
        "radius": 2,
    }

    with fake_rdkit_modules(), pytest.warns(UserWarning, match="Invalid SMILES"):
        X, y, feature_names = build_feature_matrix(df, feature_config)

    assert X.shape == (2, 18)
    np.testing.assert_array_equal(y, np.array([10.0, 20.0], dtype=np.float32))
    assert len(feature_names) == 18
    assert feature_names[:2] == [
        "aryl_halide_smiles__morgan_0",
        "aryl_halide_smiles__morgan_1",
    ]
    assert feature_names[-2:] == ["solvent__DMF", "solvent__toluene"]
