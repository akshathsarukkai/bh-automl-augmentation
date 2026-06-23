"""Tests for active reaction features and isolated legacy feature helpers."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.features.featurize as featurize_module
from bh_augmentation.features.diagnostics import diagnose_smiles_featurization
from bh_augmentation.features.featurize import (
    build_feature_matrix,
    component_fingerprint_features,
    extract_reaction_parts,
    extract_smiles_tokens,
    morgan_fingerprint,
    one_hot_condition_features,
    reaction_smiles_fingerprint,
    split_reaction_to_molecule_smiles,
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
        if smiles in {"INVALID", "UNKNOWN"} or "?" in smiles or ">" in smiles or "{" in smiles:
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


def test_build_feature_matrix_without_rdkit_uses_hash_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base CI should not require RDKit for lightweight runner tests."""
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.delitem(sys.modules, "rdkit.Chem", raising=False)
    monkeypatch.delitem(sys.modules, "rdkit.Chem.AllChem", raising=False)
    monkeypatch.delitem(sys.modules, "rdkit.DataStructs", raising=False)
    df = pd.DataFrame(
        {
            "reaction_smiles": ["CCBr.N>>CCN", "c1ccccc1Br.N>>c1ccccc1N"],
            "yield": [70.0, 80.0],
        }
    )

    X, y, names = build_feature_matrix(df, {"kind": "reaction_morgan_sum", "n_bits": 32})

    assert X.shape == (2, 32)
    assert y.tolist() == [70.0, 80.0]
    assert len(names) == 32
    assert np.all(np.any(X != 0, axis=1))

    featurize_module._WARNED_HASH_FINGERPRINT_FALLBACK = False
    with pytest.warns(UserWarning, match="hash fingerprints"):
        direct = morgan_fingerprint("CCO", n_bits=32)
    assert direct.sum() > 0


def test_legacy_component_fingerprint_features_concatenate_columns() -> None:
    """The legacy helper should concatenate explicitly requested columns."""
    df = pd.DataFrame(
        {
            "aryl_halide_smiles": ["c1ccccc1", "INVALID"],
            "amine_smiles": ["N", "CCO"],
        }
    )

    with fake_rdkit_modules():
        features = component_fingerprint_features(
            df,
            smiles_columns=["aryl_halide_smiles", "amine_smiles"],
            n_bits=8,
        )

    assert features.shape == (2, 16)
    np.testing.assert_array_equal(features[1, :8], np.zeros(8, dtype=np.float32))


def test_reaction_smiles_fingerprint_splits_reaction_sections() -> None:
    """Reaction strings should not be fingerprinted as one invalid molecule."""
    with fake_rdkit_modules():
        fingerprint = reaction_smiles_fingerprint("CCO.N>O.Cl>CCN", n_bits=8)

    assert fingerprint.shape == (8,)
    assert fingerprint.sum() > 0


def test_split_reaction_to_molecule_smiles_handles_common_formats() -> None:
    """Reaction splitting should support arrows and plain molecule SMILES."""
    assert split_reaction_to_molecule_smiles("CCBr.NCC>>CCNCC") == [
        "CCBr",
        "NCC",
        "CCNCC",
    ]
    assert split_reaction_to_molecule_smiles("A.B>C.D>E") == ["A", "B", "C", "D", "E"]
    assert split_reaction_to_molecule_smiles("CCO") == ["CCO"]
    assert split_reaction_to_molecule_smiles(None) == []


def test_extract_smiles_tokens_handles_dict_style_tdc_record() -> None:
    """Dict-like TDC records should yield only molecular values."""
    value = "{'product': 'CCNCC', 'catalyst': '', 'reactant': 'CCBr.NCC'}"

    assert extract_smiles_tokens(value) == ["CCBr", "NCC", "CCNCC"]


def test_extract_smiles_tokens_keeps_existing_formats() -> None:
    """Single molecule and reaction SMILES formats should still work."""
    assert extract_smiles_tokens("CCO") == ["CCO"]
    assert extract_smiles_tokens("CCBr.NCC>>CCNCC") == ["CCBr", "NCC", "CCNCC"]


def test_extract_smiles_tokens_malformed_dict_falls_back() -> None:
    """Malformed dict-like strings should not crash."""
    assert extract_smiles_tokens("{'product': 'CCO',") == []


def test_extract_reaction_parts_handles_dict_object() -> None:
    """Dict records should preserve reactant, agent, and product roles."""
    value = {
        "reactant": "CCBr.N",
        "catalyst": "O",
        "product": "CCN",
    }

    assert extract_reaction_parts(value) == {
        "reactants": ["CCBr", "N"],
        "agents": ["O"],
        "products": ["CCN"],
    }


def test_extract_reaction_parts_handles_serialized_dict() -> None:
    """Serialized dict records should be parsed with literal_eval."""
    value = "{'product': 'CCN', 'catalyst': '', 'reactant': 'CCBr.N'}"

    assert extract_reaction_parts(value) == {
        "reactants": ["CCBr", "N"],
        "agents": [],
        "products": ["CCN"],
    }


def test_extract_reaction_parts_handles_reaction_smiles() -> None:
    """Reaction SMILES should be split into role-specific sections."""
    assert extract_reaction_parts("CCBr.N>O.Cl>CCN") == {
        "reactants": ["CCBr", "N"],
        "agents": ["O", "Cl"],
        "products": ["CCN"],
    }
    assert extract_reaction_parts("CCBr.N>>CCN") == {
        "reactants": ["CCBr", "N"],
        "agents": [],
        "products": ["CCN"],
    }


def test_reaction_smiles_fingerprint_handles_single_molecule_smiles() -> None:
    """Plain molecule SMILES should still produce nonzero fingerprints."""
    with fake_rdkit_modules():
        fingerprint = reaction_smiles_fingerprint("CCO", n_bits=8)

    assert fingerprint.shape == (8,)
    assert fingerprint.sum() > 0


@pytest.mark.parametrize(
    "reaction_smiles",
    ["CCBr.NCC>>CCNCC", "c1ccccc1Br.N>>c1ccccc1N"],
)
def test_reaction_smiles_fingerprint_common_reaction_strings_nonzero(
    reaction_smiles: str,
) -> None:
    """Common reaction-SMILES strings should produce nonzero fingerprints."""
    with fake_rdkit_modules():
        fingerprint = reaction_smiles_fingerprint(reaction_smiles, n_bits=8)

    assert fingerprint.shape == (8,)
    assert fingerprint.sum() > 0


def test_reaction_smiles_fingerprint_malformed_string_does_not_crash() -> None:
    """Malformed reaction strings should produce zero or partial features."""
    with fake_rdkit_modules():
        fingerprint = reaction_smiles_fingerprint("???>>CCO", n_bits=8)

    assert fingerprint.shape == (8,)


def test_reaction_smiles_fingerprint_dict_style_string_nonzero() -> None:
    """Dict-like TDC reaction records should fingerprint extracted molecules."""
    value = "{'product': 'CCNCC', 'catalyst': '', 'reactant': 'CCBr.NCC'}"

    with fake_rdkit_modules():
        fingerprint = reaction_smiles_fingerprint(value, n_bits=8)

    assert fingerprint.shape == (8,)
    assert fingerprint.sum() > 0


def test_component_features_use_reaction_smiles_section_fingerprints() -> None:
    """reaction_smiles should use split molecule fingerprints."""
    df = pd.DataFrame({"reaction_smiles": ["CCO.N>O.Cl>CCN", "N>>CCN"]})

    with fake_rdkit_modules():
        features = component_fingerprint_features(df, ["reaction_smiles"], n_bits=8)

    assert features.shape == (2, 8)
    assert features.sum() > 0


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


def test_legacy_build_feature_matrix_path_returns_names() -> None:
    """The legacy explicit-column path should remain isolated and deterministic."""
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

    with fake_rdkit_modules():
        X, y, feature_names = build_feature_matrix(df, feature_config)

    assert X.shape == (2, 18)
    np.testing.assert_array_equal(y, np.array([10.0, 20.0], dtype=np.float32))
    assert len(feature_names) == 18
    assert feature_names[:2] == [
        "aryl_halide_smiles__morgan_0",
        "aryl_halide_smiles__morgan_1",
    ]
    assert feature_names[-2:] == ["solvent__DMF", "solvent__toluene"]


def test_legacy_reaction_smiles_alias_width() -> None:
    """The legacy explicit reaction column should retain its prior width."""
    df = pd.DataFrame(
        {
            "reaction_smiles": ["CCO.N>O>CCN"],
            "yield": [75.0],
        }
    )

    with fake_rdkit_modules():
        X, y, feature_names = build_feature_matrix(
            df,
            {"smiles_columns": ["reaction_smiles"], "n_bits": 8},
        )

    assert X.shape == (1, 8)
    assert len(feature_names) == 8
    assert feature_names[0] == "reaction_smiles__morgan_0"
    np.testing.assert_array_equal(y, np.array([75.0], dtype=np.float32))


def test_reaction_smiles_alias_warns_and_uses_reaction_morgan_sum() -> None:
    """The legacy reaction_smiles kind should map to reaction_morgan_sum."""
    df = pd.DataFrame(
        {
            "reaction_smiles": ["CCBr.NCC>>CCNCC"],
            "yield": [88.0],
        }
    )

    with fake_rdkit_modules(), pytest.warns(DeprecationWarning, match="reaction_morgan_sum"):
        X, _, _ = build_feature_matrix(
            df,
            {"kind": "reaction_smiles", "n_bits": 8},
        )

    assert X.shape == (1, 8)
    assert X.sum() > 0


def test_reaction_plus_components_alias_does_not_pass_full_dict_to_morgan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy alias should route through role parsing, never raw dict SMILES."""
    dict_record = "{'product': 'CCN', 'catalyst': '', 'reactant': 'CCBr.N'}"
    df = pd.DataFrame(
        {
            "reaction_smiles": [dict_record],
            "yield": [80.0],
        }
    )
    calls: list[str] = []

    def fake_morgan(
        smiles: str,
        radius: int = 2,
        n_bits: int = 2048,
        warn_invalid: bool = True,
    ) -> np.ndarray:
        calls.append(smiles)
        fingerprint = np.zeros(n_bits, dtype=np.float32)
        if smiles != "UNKNOWN":
            fingerprint[0] = 1.0
        return fingerprint

    monkeypatch.setattr(featurize_module, "morgan_fingerprint", fake_morgan)

    with pytest.warns(DeprecationWarning, match="reaction_role_concat_delta"):
        X, _, names = build_feature_matrix(
            df,
            {"kind": "reaction_plus_components", "n_bits": 8},
        )

    assert dict_record not in calls
    assert {"CCBr", "N", "CCN"}.issubset(set(calls))
    assert X.shape == (1, 32)
    assert len(names) == 32
    assert X.sum() != 0


@pytest.mark.parametrize("kind", ["fp_concat", "fp_plus_conditions", "categorical_conditions"])
def test_removed_feature_kinds_raise_clear_error(kind: str) -> None:
    """Removed legacy named modes should direct users to canonical modes."""
    df = pd.DataFrame(
        {
            "reaction_smiles": ["CCBr.N>>CCN"],
            "yield": [80.0],
        }
    )

    with pytest.raises(ValueError, match="has been removed"):
        build_feature_matrix(df, {"kind": kind, "n_bits": 8})


@pytest.mark.parametrize(
    ("kind", "width"),
    [
        ("reaction_morgan_sum", 8),
        ("reaction_role_concat", 24),
        ("reaction_role_concat_delta", 32),
    ],
)
def test_reaction_feature_ablation_shapes_and_nonzero_features(
    kind: str,
    width: int,
) -> None:
    """Each reaction ablation should have its documented deterministic width."""
    df = pd.DataFrame(
        {
            "reaction_smiles": [
                "{'product': 'CCN', 'catalyst': 'O', 'reactant': 'CCBr.N'}",
                "c1ccccc1Br.N>>c1ccccc1N",
            ],
            "yield": [70.0, 80.0],
        }
    )

    with fake_rdkit_modules():
        X, y, feature_names = build_feature_matrix(
            df,
            {"kind": kind, "n_bits": 8, "radius": 2},
        )

    assert X.shape == (2, width)
    assert np.all(np.any(X != 0, axis=1))
    assert len(feature_names) == width
    np.testing.assert_array_equal(y, np.array([70.0, 80.0], dtype=np.float32))


def test_reaction_feature_ablation_malformed_record_does_not_crash() -> None:
    """Malformed serialized records should produce zeros without crashing."""
    df = pd.DataFrame(
        {
            "reaction_smiles": ["{'product': 'CCO',"],
            "yield": [50.0],
        }
    )

    with fake_rdkit_modules():
        X, _, _ = build_feature_matrix(
            df,
            {"kind": "reaction_role_concat_delta", "n_bits": 8},
        )

    assert X.shape == (1, 32)
    assert X.sum() == 0


def test_diagnose_smiles_featurization_reports_split_success() -> None:
    """Diagnostics should show reaction strings fail whole parse but split successfully."""
    dict_record = "{'product': 'CCNCC', 'catalyst': '', 'reactant': 'CCBr.NCC'}"
    df = pd.DataFrame({"reaction_smiles": ["CCBr.NCC>>CCNCC", "CCO", dict_record, None]})

    with fake_rdkit_modules():
        summary = diagnose_smiles_featurization(df, n_examples=4)

    assert summary["total_rows_checked"] == 4
    assert summary["missing_strings"] == 1
    assert summary["strings_containing_gt"] == 1
    assert summary["strings_containing_double_gt"] == 1
    assert summary["strings_parseable_as_whole"] == 1
    assert summary["strings_parseable_after_splitting"] == 3
    assert summary["dict_like_records"] == 1
    assert summary["dict_literal_eval_successes"] == 1
    assert summary["extracted_molecule_tokens"] == 7
    assert summary["rows_producing_nonzero_fingerprints"] == 3
    assert repr("CCBr.NCC>>CCNCC") in summary["first_examples_that_fail_whole_string_parsing"]
    assert ["CCBr", "NCC", "CCNCC"] in summary["first_extracted_token_lists"]
