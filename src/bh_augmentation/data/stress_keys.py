"""Helpers for constructing held-out stress-test group keys."""

from __future__ import annotations

import pandas as pd

from bh_augmentation.features.featurize import extract_reaction_parts


def add_stress_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic product and order-invariant reactant group keys."""
    result = df.copy()

    product_keys: list[str] = []
    reactant_keys: list[str] = []
    for _, row in result.iterrows():
        parts = extract_reaction_parts(row.get("reaction_smiles", ""))

        product = parts["products"]
        if not product:
            product = _known_values([row.get("product_smiles")])
        product_keys.append(".".join(product) if product else "UNKNOWN")

        reactants = _known_values(
            [row.get("reactant_1_smiles"), row.get("reactant_2_smiles")]
        )
        if not reactants:
            reactants = parts["reactants"]
        reactant_keys.append(".".join(sorted(reactants)) if reactants else "UNKNOWN")

    result["product_key"] = product_keys
    result["reactant_key"] = reactant_keys
    return result


def _known_values(values: list[object]) -> list[str]:
    known: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != "UNKNOWN":
            known.extend(part.strip() for part in text.split(".") if part.strip())
    return known
