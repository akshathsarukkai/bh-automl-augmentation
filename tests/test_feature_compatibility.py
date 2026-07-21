"""Negative tests for semantic feature compatibility checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_dataframe
from bh_augmentation.features.compatibility import (
    FeatureBlock,
    FeatureMetadata,
    assert_feature_compatibility,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata


def test_equal_width_with_different_names_is_rejected() -> None:
    with pytest.raises(ValueError, match="Feature name mismatch"):
        _assert(["a", "b"], ["a", "c"], _metadata(), _metadata())


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"radius": 3}, "radius"),
        ({"role_ordering": ("ligand", "reactant_1")}, "role_ordering"),
        ({"fingerprint_backend": "rdkit"}, "fingerprint_backend"),
    ],
)
def test_metadata_mismatch_is_rejected(changed: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _assert(["a", "b"], ["a", "b"], _metadata(), replace(_metadata(), **changed))


def test_missing_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="metadata is required"):
        assert_feature_compatibility(np.zeros((1, 2)), ["a", "b"], np.zeros((1, 2)), ["a", "b"])


def test_delta_metadata_exactly_matches_vector_layout() -> None:
    frame = reaction_roles_dataframe(
        [ReactionRoles("CCBr", "CN", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "CCNC")],
        yields=[50.0],
    )
    X, _, names, metadata = build_feature_matrix_with_metadata(
        frame,
        {"kind": "bh_role_separated_delta", "n_bits": 8, "radius": 2, "fingerprint_backend": "hash"},
    )

    assert metadata.total_width == X.shape[1] == len(names) == 80
    assert [block.name for block in metadata.block_slices] == [
        "reactant_1", "reactant_2", "catalyst", "ligand", "base",
        "solvent_or_additive", "product", "delta_product_minus_reactant_1",
        "delta_product_minus_reactant_2", "delta_product_minus_reactant_pair",
    ]
    assert [(block.start, block.stop) for block in metadata.block_slices] == [
        (index * 8, (index + 1) * 8) for index in range(10)
    ]


def test_categorical_metadata_has_stable_named_slices() -> None:
    frame = pd.DataFrame(
        {
            "molecule": ["CC", "CCC"],
            "solvent": ["A", "B"],
            "yield": [1.0, 2.0],
        }
    )
    X, _, _, metadata = build_feature_matrix_with_metadata(
        frame,
        {
            "smiles_columns": ["molecule"],
            "categorical_columns": ["solvent"],
            "n_bits": 4,
            "fingerprint_backend": "hash",
        },
    )

    assert metadata.total_width == X.shape[1] == 6
    assert metadata.block_slices == (
        FeatureBlock("smiles:molecule", 0, 4),
        FeatureBlock("categorical:solvent", 4, 6),
    )


def _assert(real_names, synthetic_names, real_metadata, synthetic_metadata) -> None:
    assert_feature_compatibility(
        np.zeros((1, 2)), real_names, np.zeros((1, 2)), synthetic_names,
        real_metadata=real_metadata, synthetic_metadata=synthetic_metadata,
    )


def _metadata() -> FeatureMetadata:
    return FeatureMetadata(
        representation_kind="bh_role_separated",
        n_bits=1,
        radius=2,
        fingerprint_backend="hash",
        role_ordering=("reactant_1", "ligand"),
        block_slices=(FeatureBlock("reactant_1", 0, 1), FeatureBlock("ligand", 1, 2)),
        total_width=2,
    )
