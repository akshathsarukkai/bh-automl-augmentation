"""Diagnostics for SMILES and reaction-SMILES featurization."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.features.featurize import (
    extract_smiles_tokens,
    reaction_smiles_fingerprint,
)


def diagnose_smiles_featurization(
    df: pd.DataFrame,
    smiles_column: str = "reaction_smiles",
    n_examples: int = 20,
) -> dict[str, object]:
    """Diagnose whether reaction strings can produce nonzero RDKit features."""
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column is missing: {smiles_column}")

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SMILES featurization diagnostics. Install it with "
            "`python -m pip install rdkit`."
        ) from exc

    values = df[smiles_column].head(n_examples)
    missing = 0
    containing_gt = 0
    containing_double_gt = 0
    containing_dot = 0
    whole_parseable = 0
    split_parseable = 0
    nonzero = 0
    dict_like = 0
    dict_literal_eval_success = 0
    extracted_token_count = 0
    fail_whole_examples: list[str] = []
    split_success_examples: list[str] = []
    extracted_token_examples: list[list[str]] = []

    for value in values:
        if not isinstance(value, str) or not value.strip():
            missing += 1
            continue
        text = value.strip()
        containing_gt += int(">" in text)
        containing_double_gt += int(">>" in text)
        containing_dot += int("." in text)
        is_dict_like = text.startswith("{") and text.endswith("}")
        dict_like += int(is_dict_like)
        if is_dict_like:
            try:
                dict_literal_eval_success += int(isinstance(ast.literal_eval(text), dict))
            except (SyntaxError, ValueError):
                pass

        whole_ok = Chem.MolFromSmiles(text) is not None
        whole_parseable += int(whole_ok)
        if not whole_ok and len(fail_whole_examples) < 5:
            fail_whole_examples.append(repr(text))

        molecules = extract_smiles_tokens(text)
        extracted_token_count += len(molecules)
        if molecules and len(extracted_token_examples) < 5:
            extracted_token_examples.append(molecules)
        split_ok = any(Chem.MolFromSmiles(smiles) is not None for smiles in molecules)
        split_parseable += int(split_ok)
        if split_ok and len(split_success_examples) < 5:
            split_success_examples.append(repr(text))

        fingerprint = reaction_smiles_fingerprint(text)
        nonzero += int(np.any(fingerprint > 0))

    return {
        "total_rows_checked": int(len(values)),
        "missing_strings": missing,
        "strings_containing_gt": containing_gt,
        "strings_containing_double_gt": containing_double_gt,
        "strings_containing_dot": containing_dot,
        "strings_parseable_as_whole": whole_parseable,
        "strings_parseable_after_splitting": split_parseable,
        "dict_like_records": dict_like,
        "dict_literal_eval_successes": dict_literal_eval_success,
        "extracted_molecule_tokens": extracted_token_count,
        "rows_producing_nonzero_fingerprints": nonzero,
        "first_examples_that_fail_whole_string_parsing": fail_whole_examples,
        "first_examples_that_succeed_after_component_splitting": split_success_examples,
        "first_extracted_token_lists": extracted_token_examples,
    }


def main() -> None:
    """CLI entry point for SMILES featurization diagnostics."""
    parser = argparse.ArgumentParser(description="Diagnose reaction-SMILES featurization.")
    parser.add_argument("--input", required=True, help="Path to processed CSV.")
    parser.add_argument("--column", default="reaction_smiles", help="SMILES column to inspect.")
    parser.add_argument("--n-examples", type=int, default=20, help="Number of rows to inspect.")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    summary = diagnose_smiles_featurization(df, args.column, args.n_examples)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
