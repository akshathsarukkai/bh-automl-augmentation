"""Tests for reaction data loading and cleaning."""

from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.data.clean_data import (
    ACTIVE_REQUIRED_SCHEMA,
    DEPRECATED_COMPONENT_COLUMNS,
    NORMALIZED_SCHEMA,
    clean_buchwald_hartwig,
    count_unknown_components,
    normalize_tdc_buchwald_hartwig,
)
from bh_augmentation.data.load_data import load_reaction_csv

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_bh.csv"


def test_load_reaction_csv_reads_fixture() -> None:
    """CSV loading should return the raw fixture rows."""
    df = load_reaction_csv(FIXTURE_PATH)

    assert len(df) == 7
    assert "Yield (%)" in df.columns


def test_load_reaction_csv_raises_for_missing_path(tmp_path: Path) -> None:
    """Missing CSV paths should raise a helpful error."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Reaction CSV does not exist"):
        load_reaction_csv(missing_path)


def test_clean_buchwald_hartwig_normalizes_schema() -> None:
    """Cleaning should normalize columns to the expected schema."""
    raw = load_reaction_csv(FIXTURE_PATH)
    cleaned = clean_buchwald_hartwig(raw)

    assert set(ACTIVE_REQUIRED_SCHEMA).issubset(cleaned.columns)
    assert not {"product_smiles", "reactant_1_smiles", "reactant_2_smiles"} & set(
        cleaned.columns
    )
    assert cleaned.attrs["original_row_count"] == 7


def test_clean_buchwald_hartwig_drops_bad_missing_and_out_of_range_yields() -> None:
    """Invalid yields should be dropped instead of clipped."""
    raw = load_reaction_csv(FIXTURE_PATH)
    cleaned = clean_buchwald_hartwig(raw)

    assert cleaned["yield"].tolist() == [81.5, 45.0, 0.0]
    assert cleaned.attrs["dropped_invalid_yield_count"] == 4
    assert cleaned["yield"].between(0, 100, inclusive="both").all()


def test_clean_buchwald_hartwig_does_not_materialize_optional_fields() -> None:
    """Absent optional helpers should remain absent from active cleaned data."""
    raw = pd.DataFrame(
        {
            "Reaction SMILES": ["Brc1ccccc1.NCC>>c1ccccc1NCC"],
            "Yield (%)": [50],
        }
    )

    cleaned = clean_buchwald_hartwig(raw)

    assert cleaned.loc[0, "reaction_id"] == "rxn_000001"
    assert set(cleaned.columns) == set(ACTIVE_REQUIRED_SCHEMA)
    assert cleaned.loc[0, "reaction_smiles"] == "Brc1ccccc1.NCC>>c1ccccc1NCC"


def test_clean_buchwald_hartwig_drops_missing_reaction_smiles() -> None:
    raw = pd.DataFrame(
        {
            "reaction_id": ["valid", "missing"],
            "reaction_smiles": ["CCBr.N>>CCN", None],
            "yield": [50.0, 60.0],
        }
    )

    cleaned = clean_buchwald_hartwig(raw)

    assert cleaned["reaction_id"].tolist() == ["valid"]
    assert cleaned.attrs["dropped_missing_reaction_count"] == 1


def test_clean_buchwald_hartwig_preserves_helper_columns() -> None:
    """Cleaning should retain experiment helper columns after normalized fields."""
    raw = pd.DataFrame(
        {
            "reaction_id": ["rxn_1"],
            "reaction_smiles": ["CCBr.N>>CCN"],
            "yield": [75.0],
            "product_key": ["CCN"],
        }
    )

    cleaned = clean_buchwald_hartwig(raw)

    assert cleaned.loc[0, "product_key"] == "CCN"


def test_normalize_tdc_buchwald_hartwig_maps_common_tdc_columns() -> None:
    """TDC Drug_ID/Drug/Y columns should map into the normalized schema."""
    raw = pd.DataFrame(
        {
            "Drug_ID": ["tdc_1", "tdc_2", "tdc_3", "tdc_4"],
            "Drug": ["BrC.N>>CN", "ClC.N>>CN", "", None],
            "Y": ["72.5", "bad", "55", "80"],
        }
    )

    with pytest.warns(UserWarning, match="More than 50% of molecular component fields"):
        normalized = normalize_tdc_buchwald_hartwig(raw)

    assert list(normalized.columns) == NORMALIZED_SCHEMA
    assert len(normalized) == 1
    assert normalized.loc[0, "reaction_id"] == "tdc_1"
    assert normalized.loc[0, "reaction_smiles"] == "BrC.N>>CN"
    assert normalized.loc[0, "yield"] == 72.5
    assert not set(DEPRECATED_COMPONENT_COLUMNS) & set(normalized.columns)
    assert normalized.loc[0, "solvent"] == "UNKNOWN"
    assert normalized.loc[0, "temperature"] == "UNKNOWN"
    assert normalized.attrs["tdc_adapter"]["mapped_reaction_smiles_column"] == "Drug"
    assert normalized.attrs["tdc_adapter"]["mapped_yield_column"] == "Y"
    assert normalized.attrs["tdc_adapter"]["rows_before"] == 4
    assert normalized.attrs["tdc_adapter"]["rows_after"] == 1
    assert count_unknown_components(normalized)["product_smiles"] == 1


def test_clean_buchwald_hartwig_handles_tdc_style_dataframe() -> None:
    """The generic cleaner should run the TDC adapter before normal cleaning."""
    raw = pd.DataFrame(
        {
            "Drug_ID": ["tdc_1", "tdc_2"],
            "Drug": ["BrC.N>>CN", "ClC.N>>CN"],
            "Y": [72.5, 30.0],
        }
    )

    with pytest.warns(UserWarning, match="More than 50% of molecular component fields"):
        cleaned = clean_buchwald_hartwig(raw)

    assert len(cleaned) == 2
    assert cleaned["reaction_id"].tolist() == ["tdc_1", "tdc_2"]
    assert cleaned["reaction_smiles"].tolist() == ["BrC.N>>CN", "ClC.N>>CN"]
    assert not set(DEPRECATED_COMPONENT_COLUMNS) & set(cleaned.columns)


def test_normalize_tdc_buchwald_hartwig_maps_reaction_dict_records() -> None:
    """Actual TDC Reaction_ID/Reaction/Y records should keep nonzero rows."""
    raw = pd.DataFrame(
        {
            "Reaction_ID": ["reactions_1", "reactions_2"],
            "Reaction": [
                {"product": "CCN", "catalyst": "", "reactant": "CCBr.N"},
                {"product": "c1ccccc1N", "catalyst": "", "reactant": "c1ccccc1Br.N"},
            ],
            "Y": [0.10, 0.75],
        }
    )

    normalized = normalize_tdc_buchwald_hartwig(raw)

    assert len(normalized) == 2
    assert normalized["reaction_id"].tolist() == ["reactions_1", "reactions_2"]
    assert normalized["reaction_smiles"].tolist() == [
        "CCBr.N>>CCN",
        "c1ccccc1Br.N>>c1ccccc1N",
    ]
    assert normalized["yield"].tolist() == [10.0, 75.0]
    assert normalized["reactant_1_smiles"].tolist() == ["CCBr", "c1ccccc1Br"]
    assert normalized["reactant_2_smiles"].tolist() == ["N", "N"]
    assert normalized["product_smiles"].tolist() == ["CCN", "c1ccccc1N"]
    assert not set(DEPRECATED_COMPONENT_COLUMNS) & set(normalized.columns)
    assert normalized.attrs["tdc_adapter"]["detected_id_column"] == "Reaction_ID"
    assert normalized.attrs["tdc_adapter"]["detected_reaction_column"] == "Reaction"
    assert normalized.attrs["tdc_adapter"]["detected_yield_column"] == "Y"
    assert normalized.attrs["tdc_adapter"]["yield_was_fraction_scaled"] is True
    assert normalized.attrs["tdc_adapter"]["final_saved_row_count"] == 2


def test_normalize_tdc_buchwald_hartwig_maps_reaction_dict_strings() -> None:
    """Serialized TDC reaction dicts should expose generic molecular fields."""
    raw = pd.DataFrame(
        {
            "Reaction_ID": ["reactions_1"],
            "Reaction": ["{'product': 'CCN', 'catalyst': '', 'reactant': 'CCBr.N'}"],
            "Y": [0.10],
            "product_key": ["CCN"],
        }
    )

    normalized = normalize_tdc_buchwald_hartwig(raw)

    assert len(normalized) == 1
    assert normalized.loc[0, "reaction_id"] == "reactions_1"
    assert normalized.loc[0, "reactant_1_smiles"] == "CCBr"
    assert normalized.loc[0, "reactant_2_smiles"] == "N"
    assert normalized.loc[0, "product_smiles"] == "CCN"
    assert normalized.loc[0, "reaction_smiles"] == "CCBr.N>>CCN"
    assert normalized.loc[0, "yield"] == 10.0
    assert normalized.loc[0, "product_key"] == "CCN"
