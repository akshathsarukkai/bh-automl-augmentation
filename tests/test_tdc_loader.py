"""Tests for the optional TDC Buchwald-Hartwig loader."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.data.load_data import (
    TDC_BUCHWALD_HARTWIG_NAME,
    load_tdc_buchwald_hartwig,
    save_tdc_buchwald_hartwig,
)


def test_load_tdc_buchwald_hartwig_raises_helpful_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing TDC should produce an installation-focused ImportError."""
    monkeypatch.setitem(sys.modules, "tdc", None)
    monkeypatch.delitem(sys.modules, "tdc.single_pred", raising=False)

    with pytest.raises(ImportError, match="python -m pip install PyTDC"):
        load_tdc_buchwald_hartwig()


def test_load_tdc_buchwald_hartwig_uses_mocked_tdc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TDC loader should call the Yields dataset without real downloads."""
    expected = pd.DataFrame(
        {
            "reaction_smiles": ["BrC.N>>CN"],
            "yield": [72.0],
        }
    )
    calls: list[str] = []

    class FakeYields:
        def __init__(self, name: str) -> None:
            calls.append(name)

        def get_data(self) -> pd.DataFrame:
            return expected

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.Yields = FakeYields
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    loaded = load_tdc_buchwald_hartwig()

    pd.testing.assert_frame_equal(loaded, expected)
    assert calls == [TDC_BUCHWALD_HARTWIG_NAME]


def test_save_tdc_buchwald_hartwig_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving should write the mocked TDC data without internet access."""
    tdc_data = pd.DataFrame({"Drug_ID": ["tdc_1"], "Drug": ["ClC.N>>CN"], "Y": [55.0]})

    class FakeYields:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_data(self) -> pd.DataFrame:
            return tdc_data

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.Yields = FakeYields
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    output_path = tmp_path / "nested" / "tdc_bh.csv"
    saved = save_tdc_buchwald_hartwig(output_path)

    assert saved.loc[0, "reaction_id"] == "tdc_1"
    assert saved.loc[0, "reaction_smiles"] == "ClC.N>>CN"
    assert saved.loc[0, "yield"] == 55.0
    assert "aryl_halide_smiles" not in saved.columns
    reloaded = pd.read_csv(output_path)
    assert reloaded.loc[0, "reaction_id"] == "tdc_1"
    assert reloaded.loc[0, "reaction_smiles"] == "ClC.N>>CN"
    assert reloaded.loc[0, "yield"] == 55.0


def test_save_tdc_buchwald_hartwig_keeps_valid_tdc_rows_after_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TDC save should not overwrite processed output with an empty CSV."""
    tdc_data = pd.DataFrame(
        {
            "Drug_ID": ["tdc_1", "tdc_2", "tdc_3", "tdc_4"],
            "Drug": ["BrC.N>>CN", "ClC.N>>CN", "", "IC.N>>CN"],
            "Y": [72.0, "bad", 55.0, 105.0],
        }
    )

    class FakeYields:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_data(self) -> pd.DataFrame:
            return tdc_data

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.Yields = FakeYields
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    output_path = tmp_path / "processed" / "bh_clean.csv"
    saved = save_tdc_buchwald_hartwig(output_path)

    assert len(saved) == 1
    assert saved.loc[0, "reaction_id"] == "tdc_1"
    assert saved.loc[0, "reaction_smiles"] == "BrC.N>>CN"
    assert saved.loc[0, "yield"] == 72.0
    assert "aryl_halide_smiles" not in saved.columns
    assert saved.attrs["tdc_adapter"]["rows_before"] == 4
    assert saved.attrs["tdc_adapter"]["rows_after_reaction_smiles_filter"] == 3
    assert saved.attrs["tdc_adapter"]["rows_after_numeric_yield_filter"] == 2
    assert saved.attrs["tdc_adapter"]["rows_after_yield_range_filter"] == 1

    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == 1
    assert reloaded.loc[0, "reaction_id"] == "tdc_1"


def test_save_tdc_buchwald_hartwig_keeps_actual_reaction_dict_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving should support TDC Reaction_ID/Reaction/Y dict-style records."""
    tdc_data = pd.DataFrame(
        {
            "Reaction_ID": ["reactions_1", "reactions_2"],
            "Reaction": [
                {"product": "CCN", "catalyst": "", "reactant": "CCBr.N"},
                {"product": "c1ccccc1N", "catalyst": "", "reactant": "c1ccccc1Br.N"},
            ],
            "Y": [0.10, 0.75],
        }
    )

    class FakeYields:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_data(self) -> pd.DataFrame:
            return tdc_data

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.Yields = FakeYields
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    output_path = tmp_path / "processed" / "bh_clean.csv"
    saved = save_tdc_buchwald_hartwig(output_path)

    assert len(saved) == 2
    assert saved["reaction_id"].tolist() == ["reactions_1", "reactions_2"]
    assert saved["yield"].tolist() == [10.0, 75.0]
    assert saved["reaction_smiles"].tolist() == [
        "CCBr.N>>CCN",
        "c1ccccc1Br.N>>c1ccccc1N",
    ]
    assert saved["reactant_1_smiles"].tolist() == ["CCBr", "c1ccccc1Br"]
    assert saved["reactant_2_smiles"].tolist() == ["N", "N"]
    assert saved["product_smiles"].tolist() == ["CCN", "c1ccccc1N"]
    assert "base_smiles" not in saved.columns
    assert saved.attrs["tdc_adapter"]["detected_reaction_column"] == "Reaction"
    assert saved.attrs["tdc_adapter"]["final_saved_row_count"] == 2

    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == 2
    assert reloaded["yield"].tolist() == [10.0, 75.0]
