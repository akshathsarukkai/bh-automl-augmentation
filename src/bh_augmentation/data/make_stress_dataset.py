"""Create a processed dataset with held-out stress-test group keys."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.stress_keys import add_stress_group_keys


def make_stress_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Add stress-test keys to a processed CSV and save the result."""
    data = add_stress_group_keys(clean_buchwald_hartwig(load_reaction_csv(input_path)))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return data


def main() -> None:
    """Run the stress-dataset CLI."""
    parser = argparse.ArgumentParser(description="Create held-out stress-test group keys.")
    parser.add_argument("--input", required=True, help="Input processed CSV.")
    parser.add_argument("--output", required=True, help="Output stress-test CSV.")
    args = parser.parse_args()

    data = make_stress_dataset(args.input, args.output)
    print(f"Saved {len(data)} rows to {args.output}")
    for column in ["product_key", "reactant_key"]:
        print(f"{column} unique groups: {data[column].nunique(dropna=False)}")
        print(f"{column} top group sizes:")
        print(data[column].value_counts(dropna=False).head().to_string())


if __name__ == "__main__":
    main()
