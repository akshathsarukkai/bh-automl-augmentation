"""Tests for the reaction-smiles-only active modeling path."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.data.clean_data import (
    DEPRECATED_COMPONENT_COLUMNS,
    clean_buchwald_hartwig,
)
from bh_augmentation.features.featurize import build_feature_matrix


CONFIG_DIR = Path(__file__).parents[1] / "configs"
ACTIVE_FEATURE_KINDS = {
    "reaction_morgan_sum",
    "reaction_role_concat",
    "reaction_role_concat_delta",
}
CONFIG_FEATURE_KINDS = {
    *ACTIVE_FEATURE_KINDS,
    "role_separated_conditions",
    "role_separated_conditions_delta",
}


def test_cleaning_drops_fully_unknown_deprecated_components_and_keeps_keys() -> None:
    """Fully UNKNOWN legacy components should not survive active preprocessing."""
    data = pd.DataFrame(
        {
            "reaction_id": ["r1"],
            "reaction_smiles": ["CCBr.N>>CCN"],
            "yield": [75.0],
            "product_key": ["CCN"],
            "reactant_key": ["CCBr.N"],
            **{column: ["UNKNOWN"] for column in DEPRECATED_COMPONENT_COLUMNS},
        }
    )

    cleaned = clean_buchwald_hartwig(data)

    assert not set(DEPRECATED_COMPONENT_COLUMNS) & set(cleaned.columns)
    assert cleaned.loc[0, "reaction_id"] == "r1"
    assert cleaned.loc[0, "reaction_smiles"] == "CCBr.N>>CCN"
    assert cleaned.loc[0, "yield"] == 75.0
    assert cleaned.loc[0, "product_key"] == "CCN"
    assert cleaned.loc[0, "reactant_key"] == "CCBr.N"


def test_canonical_features_require_only_reaction_smiles() -> None:
    """Every active feature mode should be nonzero without component columns."""
    data = pd.DataFrame(
        {
            "reaction_smiles": ["CCBr.N>>CCN", "c1ccccc1Br.N>>c1ccccc1N"],
            "yield": [70.0, 80.0],
        }
    )

    for kind in sorted(ACTIVE_FEATURE_KINDS):
        X, _, _ = build_feature_matrix(data, {"kind": kind, "n_bits": 64})
        assert X.shape[0] == len(data)
        assert np.all(np.any(X != 0, axis=1))


def test_active_configs_do_not_reference_deprecated_component_columns() -> None:
    """Shipped experiment configs should use reaction-derived features only."""
    forbidden = set(DEPRECATED_COMPONENT_COLUMNS)
    for path in CONFIG_DIR.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert not any(column in text for column in forbidden), path

        config = yaml.safe_load(text)
        features = config.get("features", {})
        configured_kinds = [features.get("kind")]
        configured_kinds.extend(item.get("kind") for item in features.get("compare", []))
        for kind in filter(None, configured_kinds):
            assert kind in CONFIG_FEATURE_KINDS, (path, kind)
