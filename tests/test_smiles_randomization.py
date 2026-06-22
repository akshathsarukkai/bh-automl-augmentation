"""Legacy tests for component-column randomized SMILES augmentation."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import pytest

from bh_augmentation.augmentation.smiles_randomization import (
    augment_randomized_smiles,
    randomize_smiles,
)


class FakeMolecule:
    """Minimal molecule object for testing RDKit-backed control flow."""

    def __init__(self, smiles: str) -> None:
        self.smiles = smiles

    def GetNumAtoms(self) -> int:
        return len(self.smiles)


@contextmanager
def fake_rdkit_modules() -> Iterator[None]:
    """Install a small fake RDKit module for deterministic tests."""
    old_modules = {
        name: sys.modules.get(name)
        for name in ["rdkit", "rdkit.Chem"]
    }
    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")

    def mol_from_smiles(smiles: str) -> FakeMolecule | None:
        if smiles == "INVALID":
            return None
        return FakeMolecule(smiles)

    def renumber_atoms(molecule: FakeMolecule, atom_order: list[int]) -> FakeMolecule:
        return FakeMolecule("".join(molecule.smiles[index] for index in atom_order))

    def mol_to_smiles(molecule: FakeMolecule, canonical: bool = False) -> str:
        return molecule.smiles

    chem.MolFromSmiles = mol_from_smiles
    chem.RenumberAtoms = renumber_atoms
    chem.MolToSmiles = mol_to_smiles
    rdkit.Chem = chem
    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = chem
    try:
        yield
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": ["rxn_1", "rxn_2"],
            "aryl_halide_smiles": ["CCO", "NCC"],
            "amine_smiles": ["N", "INVALID"],
            "yield": [75.0, 20.0],
        }
    )


def test_randomize_smiles_is_deterministic_with_seed() -> None:
    """A fixed seed should produce the same randomized SMILES."""
    with fake_rdkit_modules():
        first = randomize_smiles("CCO", seed=123)
        second = randomize_smiles("CCO", seed=123)

    assert first == second


def test_randomize_smiles_invalid_smiles_returns_unchanged() -> None:
    """Invalid SMILES should not crash randomization."""
    with fake_rdkit_modules(), pytest.warns(UserWarning, match="could not parse"):
        randomized = randomize_smiles("INVALID", seed=123)

    assert randomized == "INVALID"


def test_randomize_smiles_without_rdkit_returns_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing RDKit should keep the package usable."""
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.delitem(sys.modules, "rdkit.Chem", raising=False)

    with pytest.warns(UserWarning, match="RDKit is not installed"):
        randomized = randomize_smiles("CCO", seed=123)

    assert randomized == "CCO"


def test_augment_randomized_smiles_expected_row_count_and_labels() -> None:
    """Augmentation should include originals and preserve yield labels."""
    df = _sample_df()

    with fake_rdkit_modules(), pytest.warns(UserWarning, match="could not parse"):
        augmented = augment_randomized_smiles(
            df,
            smiles_columns=["aryl_halide_smiles", "amine_smiles"],
            n_augments=2,
            seed=7,
        )

    assert len(augmented) == 6
    assert augmented["is_augmented"].tolist() == [False, False, True, True, True, True]
    assert augmented["source_reaction_id"].tolist() == [
        "rxn_1",
        "rxn_2",
        "rxn_1",
        "rxn_2",
        "rxn_1",
        "rxn_2",
    ]
    assert augmented["yield"].tolist() == [75.0, 20.0, 75.0, 20.0, 75.0, 20.0]


def test_augment_randomized_smiles_is_deterministic() -> None:
    """A fixed seed should produce identical augmented DataFrames."""
    df = _sample_df()

    with fake_rdkit_modules(), pytest.warns(UserWarning, match="could not parse"):
        first = augment_randomized_smiles(df, ["aryl_halide_smiles", "amine_smiles"], seed=9)
    with fake_rdkit_modules(), pytest.warns(UserWarning, match="could not parse"):
        second = augment_randomized_smiles(df, ["aryl_halide_smiles", "amine_smiles"], seed=9)

    pd.testing.assert_frame_equal(first, second)


def test_augment_randomized_smiles_can_exclude_original_rows() -> None:
    """include_original=False should return only augmented rows."""
    df = _sample_df()

    with fake_rdkit_modules(), pytest.warns(UserWarning, match="could not parse"):
        augmented = augment_randomized_smiles(
            df,
            smiles_columns=["aryl_halide_smiles", "amine_smiles"],
            n_augments=1,
            seed=7,
            include_original=False,
        )

    assert len(augmented) == 2
    assert augmented["is_augmented"].tolist() == [True, True]


def test_augment_randomized_smiles_validates_columns() -> None:
    """Missing SMILES columns should raise a helpful error."""
    with pytest.raises(ValueError, match="Missing SMILES columns"):
        augment_randomized_smiles(_sample_df(), ["missing"])
