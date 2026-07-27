"""Tests for selective measured-outcome access."""

from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.data.outcome_access import read_allowed_outcomes


def test_reads_only_allowed_rows(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    pd.DataFrame(
        {
            "source_row_id": ["a", "b", "c"],
            "yield": [1.0, 999.0, 3.0],
        }
    ).to_csv(path, index=False)

    assert read_allowed_outcomes(path, ["a", "b", "c"], ["c", "a"]) == {
        "a": 1.0,
        "c": 3.0,
    }


def test_rejects_unknown_or_duplicate_allowed_ids(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    pd.DataFrame(
        {"source_row_id": ["a", "b"], "yield": [1.0, 2.0]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unique"):
        read_allowed_outcomes(path, ["a", "b"], ["a", "a"])
    with pytest.raises(ValueError, match="Unknown"):
        read_allowed_outcomes(path, ["a", "b"], ["c"])


def test_rejects_nonfinite_allowed_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    pd.DataFrame(
        {"source_row_id": ["a", "b"], "yield": [1.0, float("inf")]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        read_allowed_outcomes(path, ["a", "b"], ["b"])
