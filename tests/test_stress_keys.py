"""Tests for held-out stress-test key generation."""

from pathlib import Path

import pandas as pd

from bh_augmentation.data.make_stress_dataset import make_stress_dataset
from bh_augmentation.data.stress_keys import add_stress_group_keys


def test_add_stress_group_keys_uses_parsed_columns_order_invariantly() -> None:
    """Parsed product/reactant columns should produce deterministic keys."""
    data = pd.DataFrame(
        {
            "product_smiles": ["CCN", "CCN"],
            "reactant_1_smiles": ["CCBr", "N"],
            "reactant_2_smiles": ["N", "CCBr"],
            "reaction_smiles": ["CCBr.N>>CCN", "N.CCBr>>CCN"],
        }
    )

    keyed = add_stress_group_keys(data)

    assert keyed["product_key"].tolist() == ["CCN", "CCN"]
    assert keyed["reactant_key"].tolist() == ["CCBr.N", "CCBr.N"]


def test_add_stress_group_keys_falls_back_to_reaction_smiles() -> None:
    """Reaction strings should supply keys when parsed columns are unavailable."""
    data = pd.DataFrame({"reaction_smiles": ["CCBr.N>O>CCN"]})

    keyed = add_stress_group_keys(data)

    assert keyed.loc[0, "product_key"] == "CCN"
    assert keyed.loc[0, "reactant_key"] == "CCBr.N"


def test_make_stress_dataset_writes_keyed_csv(tmp_path: Path) -> None:
    """The stress dataset maker should write all rows and both key columns."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "product_smiles": ["CCN", "CO"],
            "reactant_1_smiles": ["CCBr", "CCl"],
            "reactant_2_smiles": ["N", "O"],
            "reaction_smiles": ["CCBr.N>>CCN", "CCl.O>>CO"],
            "yield": [70.0, 40.0],
        }
    ).to_csv(input_path, index=False)

    result = make_stress_dataset(input_path, output_path)
    saved = pd.read_csv(output_path)

    assert len(result) == 2
    assert output_path.exists()
    assert {"product_key", "reactant_key"}.issubset(saved.columns)
