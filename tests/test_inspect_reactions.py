"""Tests for raw reaction string inspection utilities."""

from pathlib import Path

import pandas as pd

from bh_augmentation.data.inspect_reactions import (
    detect_reaction_columns,
    main,
    summarize_reaction_string_format,
)


def test_summarize_reaction_string_format_counts_expected_patterns() -> None:
    """Inspector should report basic reaction string format counts."""
    df = pd.DataFrame(
        {
            "reaction_smiles": [
                "Clc1ccccc1.N>>c1ccccc1N",
                "BrC.N>base>CN",
                "",
                None,
                "text_without_arrow",
            ]
        }
    )

    summary = summarize_reaction_string_format(df)

    assert summary["total_rows"] == 5
    assert summary["missing_reaction_strings"] == 2
    assert summary["strings_containing_gt"] == 2
    assert summary["strings_with_exactly_2_gt"] == 2
    assert summary["strings_containing_dot"] == 2
    assert summary["strings_containing_obvious_halogens"] == 2
    assert summary["strings_containing_nitrogen"] == 2
    assert summary["unique_reaction_strings"] == 3
    assert summary["first_10_raw_reaction_strings"][0] == repr("Clc1ccccc1.N>>c1ccccc1N")


def test_summarize_reaction_string_format_handles_strings_without_gt() -> None:
    """Strings without reaction arrows should be reported, not parsed."""
    df = pd.DataFrame({"Drug": ["abc", "def.ghi", "NCC"]})

    summary = summarize_reaction_string_format(df, "Drug")

    assert summary["strings_containing_gt"] == 0
    assert summary["strings_with_exactly_2_gt"] == 0
    assert summary["strings_containing_dot"] == 1


def test_detect_reaction_columns_finds_tdc_drug_column() -> None:
    """Inspector should detect common TDC reaction columns."""
    df = pd.DataFrame({"Drug_ID": ["a"], "Drug": ["BrC.N>>CN"], "Y": [70]})

    assert detect_reaction_columns(df) == ["Drug"]


def test_inspect_reactions_cli_prints_columns_and_summary(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """CLI should inspect likely reaction columns from a CSV."""
    csv_path = tmp_path / "tdc.csv"
    pd.DataFrame({"Drug_ID": ["a"], "Drug": ["BrC.N>>CN"], "Y": [70]}).to_csv(
        csv_path,
        index=False,
    )
    monkeypatch.setattr("sys.argv", ["inspect_reactions", "--input", str(csv_path)])

    main()

    output = capsys.readouterr().out
    assert "columns:" in output
    assert "likely reaction columns: ['Drug']" in output
    assert "strings_with_exactly_2_gt" in output
