"""Tests for canonical representation names and temporary legacy aliases."""

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_dataframe
from bh_augmentation.features.featurize import build_feature_matrix, canonical_feature_kind


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("reaction_role_concat", "reaction_section_concat"),
        ("reaction_role_concat_delta", "reaction_section_concat_delta"),
        ("role_separated_conditions", "bh_role_separated"),
        ("role_separated_conditions_delta", "bh_role_separated_delta"),
    ],
)
def test_legacy_aliases_warn_and_resolve(legacy: str, canonical: str) -> None:
    with pytest.warns(DeprecationWarning, match=canonical):
        assert canonical_feature_kind(legacy) == canonical


def test_three_section_alias_preserves_legacy_vector_values() -> None:
    frame = pd.DataFrame({"reaction_smiles": ["CCBr.CN>>CCNC"], "yield": [50.0]})
    canonical, _, _ = build_feature_matrix(
        frame, {"kind": "reaction_section_concat", "n_bits": 32, "radius": 2, "fingerprint_backend": "hash"}
    )
    with pytest.warns(DeprecationWarning):
        legacy, _, _ = build_feature_matrix(
            frame, {"kind": "reaction_role_concat", "n_bits": 32, "radius": 2, "fingerprint_backend": "hash"}
        )
    np.testing.assert_array_equal(legacy, canonical)


def test_role_alias_preserves_canonical_vector_values() -> None:
    frame = reaction_roles_dataframe([_roles()], yields=[50.0])
    canonical, _, canonical_names = build_feature_matrix(
        frame, {"kind": "bh_role_separated", "n_bits": 32, "radius": 2, "fingerprint_backend": "hash"}
    )
    with pytest.warns(DeprecationWarning):
        legacy, _, legacy_names = build_feature_matrix(
            frame, {"kind": "role_separated_conditions", "n_bits": 32, "radius": 2, "fingerprint_backend": "hash"}
        )
    np.testing.assert_array_equal(legacy, canonical)
    assert legacy_names == canonical_names


def _roles() -> ReactionRoles:
    return ReactionRoles("CCBr", "CN", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "CCNC")
