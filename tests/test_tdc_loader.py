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
    expected = pd.DataFrame({"reaction_smiles": ["ClC.N>>CN"], "yield": [55.0]})

    class FakeYields:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_data(self) -> pd.DataFrame:
            return expected

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.Yields = FakeYields
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    output_path = tmp_path / "nested" / "tdc_bh.csv"
    saved = save_tdc_buchwald_hartwig(output_path)

    pd.testing.assert_frame_equal(saved, expected)
    reloaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(reloaded, expected)
