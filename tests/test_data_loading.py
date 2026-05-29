"""Tests for reaction data loading and cleaning."""

from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.data.clean_data import NORMALIZED_SCHEMA, clean_buchwald_hartwig
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

    assert list(cleaned.columns) == NORMALIZED_SCHEMA
    assert cleaned.attrs["original_row_count"] == 7


def test_clean_buchwald_hartwig_drops_bad_missing_and_out_of_range_yields() -> None:
    """Invalid yields should be dropped instead of clipped."""
    raw = load_reaction_csv(FIXTURE_PATH)
    cleaned = clean_buchwald_hartwig(raw)

    assert cleaned["yield"].tolist() == [81.5, 45.0, 0.0]
    assert cleaned.attrs["dropped_invalid_yield_count"] == 4
    assert cleaned["yield"].between(0, 100, inclusive="both").all()


def test_clean_buchwald_hartwig_fills_optional_missing_fields() -> None:
    """Missing optional categorical values should use UNKNOWN."""
    raw = pd.DataFrame(
        {
            "Aryl Halide SMILES": ["Brc1ccccc1"],
            "Amine SMILES": ["NCC"],
            "Yield (%)": [50],
        }
    )

    cleaned = clean_buchwald_hartwig(raw)

    assert cleaned.loc[0, "reaction_id"] == "rxn_000001"
    assert cleaned.loc[0, "ligand_smiles"] == "UNKNOWN"
    assert cleaned.loc[0, "base_smiles"] == "UNKNOWN"
    assert cleaned.loc[0, "additive_smiles"] == "UNKNOWN"
    assert cleaned.loc[0, "solvent"] == "UNKNOWN"
    assert cleaned.loc[0, "reaction_smiles"] == "UNKNOWN"
