#!/usr/bin/env python
"""Inspect dataset-specific BH condition recovery on a processed CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DISPLAY_COLUMNS = [
    "reaction_id",
    "reaction_smiles",
    "recovered_reactant_1_smiles",
    "recovered_reactant_2_smiles",
    "recovered_catalyst_smiles",
    "recovered_ligand_smiles",
    "recovered_base_smiles",
    "recovered_solvent_or_additive_smiles",
    "recovered_temperature",
    "recovered_product_smiles",
    "condition_parse_status",
    "role_validation_status",
]

UNIQUE_COLUMNS = [
    "recovered_catalyst_smiles",
    "recovered_ligand_smiles",
    "recovered_base_smiles",
    "recovered_solvent_or_additive_smiles",
    "recovered_temperature",
]


def main() -> None:
    from bh_augmentation.data.bh_condition_reader import (
        augment_bh_dataframe,
        summarize_recovery,
    )

    parser = argparse.ArgumentParser(description="Inspect processed BH condition recovery.")
    parser.add_argument("--input", default="data/processed/bh_clean_stress.csv")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument(
        "--output",
        default="results/condition_reader_preview/parsed_conditions_preview.csv",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show-repaired", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    original_columns = list(df.columns)
    if args.nrows is not None:
        df = df.head(args.nrows).copy()

    augmented = augment_bh_dataframe(df, strict=args.strict)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_path, index=False)

    print(f"Input: {args.input}")
    print(f"Shape: {augmented.shape}")
    print(f"Original columns: {original_columns}")
    print("Parse status counts:")
    print(augmented["condition_parse_status"].value_counts(dropna=False).to_string())
    print("Role validation status counts:")
    print(augmented["role_validation_status"].value_counts(dropna=False).to_string())
    n_repaired = int((augmented["role_validation_status"] == "repaired").sum())
    n_ambiguous = int((augmented["role_validation_status"] == "ambiguous").sum())
    n_invalid = int((augmented["role_validation_status"] == "invalid_or_unresolved").sum())
    print(f"Repaired rows: {n_repaired}")
    print(f"Ambiguous rows: {n_ambiguous}")
    print(f"Invalid/unresolved rows: {n_invalid}")
    print("Unique recovered value counts after repair:")
    for column in UNIQUE_COLUMNS:
        print(f"{column}: {augmented[column].nunique(dropna=False)}")
    print("Recovery summary:")
    print(summarize_recovery(augmented).to_string(index=False))
    if args.show_repaired > 0:
        repaired_examples = augmented.loc[augmented["role_validation_status"] == "repaired"]
        ambiguous_examples = augmented.loc[augmented["role_validation_status"] == "ambiguous"]
        example_columns = [
            "reaction_id",
            "condition_block_smiles",
            "repaired_condition_order_smiles",
            "role_validation_notes",
        ]
        example_columns = [column for column in example_columns if column in augmented]
        print("Repaired examples:")
        if repaired_examples.empty:
            print("(none)")
        else:
            print(repaired_examples[example_columns].head(args.show_repaired).to_string(index=False))
        print("Ambiguous examples:")
        if ambiguous_examples.empty:
            print("(none)")
        else:
            print(ambiguous_examples[example_columns].head(args.show_repaired).to_string(index=False))
    print("Preview:")
    print(augmented[[column for column in DISPLAY_COLUMNS if column in augmented]].head(10).to_string(index=False))
    print(f"Saved preview CSV to {output_path}")


if __name__ == "__main__":
    main()
