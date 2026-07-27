from __future__ import annotations

import pandas as pd
import pytest

from bh_augmentation.evaluation.chemical_ood_artifacts import (
    FAMILY_ORDER,
    ChemicalOODConfig,
    _assert_exact_canonical_mapping,
    _assert_nearest_replay,
    _read_bool,
    _support_reasons,
)


def test_phase9_family_order_is_scientifically_declared_order() -> None:
    assert FAMILY_ORDER == (
        "electrophile_scaffold",
        "amine_scaffold",
        "reaction_fingerprint_cluster",
        "maximum_similarity_bounded",
        "condition_combination",
        "ligand",
        "base",
    )


def test_chemical_ood_config_rejects_unknown_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        ChemicalOODConfig.from_mapping({"invented": 1})
    with pytest.raises(ValueError, match="strictly between"):
        ChemicalOODConfig(maximum_similarity_bound=1.0)
    with pytest.raises(ValueError, match="integer"):
        ChemicalOODConfig(min_test_rows=0)
    with pytest.raises(ValueError, match="seed"):
        ChemicalOODConfig(seed=True)


def test_scaffold_support_failures_are_explicit() -> None:
    assert _support_reasons(
        n_train_rows=99,
        n_test_rows=9,
        n_train_groups=1,
        min_train_rows=100,
        min_test_rows=10,
        min_train_groups=2,
    ) == [
        "train_rows_below_minimum(observed=99,required=100)",
        "test_rows_below_minimum(observed=9,required=10)",
        "train_groups_below_minimum(observed=1,required=2)",
    ]


def test_serialized_boolean_parser_never_uses_truthiness() -> None:
    assert _read_bool(True)
    assert _read_bool("True")
    assert not _read_bool(False)
    assert not _read_bool("False")
    with pytest.raises(ValueError, match="Invalid"):
        _read_bool("yes")


def test_exact_source_to_canonical_key_mapping_cannot_be_permuted() -> None:
    canonical = pd.DataFrame(
        {
            "source_row_id": ["a", "b"],
            "canonical_reaction_key": ["key-a", "key-b"],
        }
    )
    assignments = pd.DataFrame(
        {
            "family": ["base", "base"],
            "source_row_id": ["a", "b"],
            "canonical_reaction_key": ["key-b", "key-a"],
        }
    )
    with pytest.raises(ValueError, match="exact source-to-canonical-key"):
        _assert_exact_canonical_mapping(assignments, canonical)


def test_nearest_similarity_replay_rejects_fabricated_values_and_stale_hash() -> None:
    expected = pd.DataFrame(
        {
            "canonical_reaction_key": ["test"],
            "nearest_train_canonical_reaction_key": ["train"],
            "nearest_train_tanimoto": [0.75],
        }
    )
    observed = expected.copy()
    observed["nearest_train_tanimoto"] = 0.123
    observed["query_hash"] = "stale"
    with pytest.raises(ValueError, match="nearest-training semantic"):
        _assert_nearest_replay(observed, expected, "correct")
