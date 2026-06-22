"""Inspect raw reaction string formats before preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LIKELY_REACTION_COLUMNS = ["reaction_smiles", "Drug", "drug", "X", "rxn", "reaction"]


def summarize_reaction_string_format(
    df: pd.DataFrame,
    reaction_col: str = "reaction_smiles",
) -> dict[str, object]:
    """Summarize raw reaction string format for a candidate reaction column."""
    if reaction_col not in df.columns:
        raise ValueError(f"Reaction column is missing: {reaction_col}")

    values = df[reaction_col]
    non_missing = values.dropna().astype(str)
    stripped = non_missing.str.strip()
    non_empty = stripped[stripped != ""]
    return {
        "total_rows": int(len(df)),
        "missing_reaction_strings": int(len(df) - len(non_empty)),
        "strings_containing_gt": int(non_empty.str.contains(">", regex=False).sum()),
        "strings_with_exactly_2_gt": int(non_empty.str.count(">").eq(2).sum()),
        "strings_containing_dot": int(non_empty.str.contains(".", regex=False).sum()),
        "strings_containing_obvious_halogens": int(
            non_empty.str.contains("Cl|Br|I|F", regex=True).sum()
        ),
        "strings_containing_nitrogen": int(non_empty.str.contains("N|n", regex=True).sum()),
        "unique_reaction_strings": int(non_empty.nunique()),
        "first_10_raw_reaction_strings": [repr(value) for value in values.head(10).tolist()],
    }


def detect_reaction_columns(df: pd.DataFrame) -> list[str]:
    """Return likely reaction-string columns present in a DataFrame."""
    return [column for column in LIKELY_REACTION_COLUMNS if column in df.columns]


def main() -> None:
    """CLI entry point for raw reaction string inspection."""
    parser = argparse.ArgumentParser(description="Inspect reaction string columns in a CSV.")
    parser.add_argument("--input", required=True, help="Path to raw reaction CSV.")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    reaction_columns = detect_reaction_columns(df)
    if not reaction_columns:
        print("likely reaction columns: []")
        return

    print(f"likely reaction columns: {reaction_columns}")
    for column in reaction_columns:
        print(f"\n[{column}]")
        summary = summarize_reaction_string_format(df, column)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("first 10 values:")
        for value in df[column].head(10).tolist():
            print(repr(value))


if __name__ == "__main__":
    main()
