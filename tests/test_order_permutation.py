"""Legacy tests for explicit component-column order permutation."""

import pandas as pd
import pytest

from bh_augmentation.augmentation.order_permutation import permute_reaction_components


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": ["rxn_1", "rxn_2"],
            "component_a": ["A1", "A2"],
            "component_b": ["B1", "B2"],
            "component_c": ["C1", "C2"],
            "solvent": ["DMF", "THF"],
            "yield": [80.0, 25.0],
        }
    )


def test_permute_reaction_components_expected_row_count() -> None:
    """Original rows plus requested permutations should be returned."""
    df = _sample_df()

    augmented = permute_reaction_components(
        df,
        component_columns=["component_a", "component_b"],
        n_permutations=2,
        seed=7,
    )

    assert len(augmented) == 6
    assert augmented["is_augmented"].tolist() == [False, False, True, True, True, True]
    assert augmented["augmentation_type"].tolist() == [
        "original",
        "original",
        "order_permutation",
        "order_permutation",
        "order_permutation",
        "order_permutation",
    ]


def test_permute_reaction_components_preserves_yield_and_source_ids() -> None:
    """Augmented rows should preserve labels and link to original reactions."""
    df = _sample_df()

    augmented = permute_reaction_components(
        df,
        component_columns=["component_a", "component_b"],
        n_permutations=1,
        seed=11,
    )

    assert augmented["yield"].tolist() == [80.0, 25.0, 80.0, 25.0]
    assert augmented["source_reaction_id"].tolist() == ["rxn_1", "rxn_2", "rxn_1", "rxn_2"]
    assert augmented.loc[2:, "reaction_id"].tolist() == ["rxn_1_perm_1", "rxn_2_perm_1"]


def test_permute_reaction_components_only_permutates_explicit_columns() -> None:
    """Columns not listed by the caller should remain unchanged."""
    df = _sample_df()

    augmented = permute_reaction_components(
        df,
        component_columns=["component_a", "component_b"],
        n_permutations=1,
        seed=13,
        include_original=False,
    )

    assert augmented["component_c"].tolist() == df["component_c"].tolist()
    assert augmented["solvent"].tolist() == df["solvent"].tolist()
    assert augmented["component_a"].tolist() == df["component_b"].tolist()
    assert augmented["component_b"].tolist() == df["component_a"].tolist()


def test_permute_reaction_components_is_deterministic() -> None:
    """Using the same seed should return identical augmented data."""
    df = _sample_df()

    first = permute_reaction_components(df, ["component_a", "component_b", "component_c"], seed=5)
    second = permute_reaction_components(df, ["component_a", "component_b", "component_c"], seed=5)

    pd.testing.assert_frame_equal(first, second)


def test_permute_reaction_components_validates_explicit_columns() -> None:
    """Missing or insufficient component columns should raise helpful errors."""
    df = _sample_df()

    with pytest.raises(ValueError, match="Missing component columns"):
        permute_reaction_components(df, ["component_a", "missing"])

    with pytest.raises(ValueError, match="At least two"):
        permute_reaction_components(df, ["component_a"])


def test_permute_reaction_components_can_exclude_originals() -> None:
    """include_original=False should return only augmented rows."""
    df = _sample_df()

    augmented = permute_reaction_components(
        df,
        component_columns=["component_a", "component_b"],
        n_permutations=1,
        include_original=False,
    )

    assert len(augmented) == 2
    assert augmented["is_augmented"].tolist() == [True, True]
