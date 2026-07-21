"""Bit-level identity tests for measured and reconstructed role features."""

import importlib.util

import numpy as np
import pytest

from bh_augmentation.augmentation.condition_transfer import build_anonymous_condition_transfer_roles
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    build_role_transferred_reaction_smiles,
)
from bh_augmentation.data.reaction_roles import (
    ReactionRoles,
    reaction_roles_dataframe,
    reaction_roles_from_row,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata


@pytest.mark.parametrize("kind", ["bh_role_separated", "bh_role_separated_delta"])
@pytest.mark.parametrize("n_bits", [16, 64])
@pytest.mark.parametrize("radius", [1, 2])
def test_measured_equals_unchanged_synthetic_paths(kind: str, n_bits: int, radius: int) -> None:
    roles = _roles()
    config = {"kind": kind, "n_bits": n_bits, "radius": radius, "fingerprint_backend": "hash"}
    measured = reaction_roles_dataframe([roles], yields=[50.0])
    X_real, _, real_names, real_metadata = build_feature_matrix_with_metadata(measured, config)

    anonymous_roles = build_anonymous_condition_transfer_roles(measured.iloc[0], measured.iloc[0])
    anonymous = reaction_roles_dataframe([anonymous_roles], yields=[50.0])
    X_anonymous, _, anonymous_names, anonymous_metadata = build_feature_matrix_with_metadata(anonymous, config)

    role_record = build_role_transferred_reaction_smiles(
        measured.iloc[0], measured.iloc[0], "all_transferable_roles"
    )
    role_aware = reaction_roles_dataframe(
        [reaction_roles_from_row(role_record)], yields=[50.0]
    )
    X_role, _, role_names, role_metadata = build_feature_matrix_with_metadata(role_aware, config)

    np.testing.assert_array_equal(X_real, X_anonymous)
    np.testing.assert_array_equal(X_real, X_role)
    assert real_names == anonymous_names == role_names
    assert real_metadata == anonymous_metadata == role_metadata


@pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="RDKit is unavailable")
def test_identity_with_explicit_rdkit_backend() -> None:
    roles = _roles()
    frame = reaction_roles_dataframe([roles], yields=[50.0])
    config = {"kind": "bh_role_separated", "n_bits": 32, "radius": 2, "fingerprint_backend": "rdkit"}

    X_measured, _, names, metadata = build_feature_matrix_with_metadata(frame, config)
    reconstructed = reaction_roles_dataframe(
        [build_anonymous_condition_transfer_roles(frame.iloc[0], frame.iloc[0])], yields=[50.0]
    )
    X_reconstructed, _, reconstructed_names, reconstructed_metadata = build_feature_matrix_with_metadata(
        reconstructed, config
    )

    np.testing.assert_array_equal(X_measured, X_reconstructed)
    assert names == reconstructed_names
    assert metadata == reconstructed_metadata


def _roles() -> ReactionRoles:
    return ReactionRoles("c1ccccc1Br", "CN", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "c1ccccc1NC")
