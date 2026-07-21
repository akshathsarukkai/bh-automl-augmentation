"""Feature-block locality tests for explicit role transfer modes."""

import numpy as np
import pytest

from bh_augmentation.augmentation.role_aware_condition_transfer import (
    _role_similarity_matrix,
    build_role_transferred_reaction_smiles,
)
from bh_augmentation.data.reaction_roles import (
    ReactionRoles,
    reaction_roles_dataframe,
    reaction_roles_from_row,
)
from bh_augmentation.features.compatibility import block_slice
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    morgan_fingerprint,
)


@pytest.mark.parametrize(
    ("mode", "changed_roles"),
    [
        ("ligand_only", {"ligand"}),
        ("base_only", {"base"}),
        ("solvent_or_additive_only", {"solvent_or_additive"}),
        ("catalyst_only", {"catalyst"}),
        ("ligand_base", {"ligand", "base"}),
        ("all_transferable_roles", {"catalyst", "ligand", "base", "solvent_or_additive"}),
    ],
)
def test_transfer_changes_only_requested_role_blocks(mode: str, changed_roles: set[str]) -> None:
    source = reaction_roles_dataframe([_source()], yields=[50.0])
    donor = reaction_roles_dataframe([_donor()], yields=[80.0])
    synthetic_record = build_role_transferred_reaction_smiles(source.iloc[0], donor.iloc[0], mode)
    synthetic = reaction_roles_dataframe([reaction_roles_from_row(synthetic_record)], yields=[60.0])
    config = {"kind": "bh_role_separated", "n_bits": 256, "radius": 2, "fingerprint_backend": "hash"}

    X_source, _, source_names, source_metadata = build_feature_matrix_with_metadata(source, config)
    X_synthetic, _, synthetic_names, synthetic_metadata = build_feature_matrix_with_metadata(synthetic, config)

    assert source_names == synthetic_names
    assert source_metadata == synthetic_metadata
    for role in source_metadata.role_ordering:
        role_slice = block_slice(source_metadata, role)
        equal = np.array_equal(X_source[:, role_slice], X_synthetic[:, role_slice])
        assert equal is (role not in changed_roles), role


def test_identity_transfer_changes_no_blocks() -> None:
    source = reaction_roles_dataframe([_source()], yields=[50.0])
    record = build_role_transferred_reaction_smiles(source.iloc[0], source.iloc[0], "all_transferable_roles")
    synthetic = reaction_roles_dataframe([reaction_roles_from_row(record)], yields=[50.0])
    config = {"kind": "bh_role_separated", "n_bits": 64, "radius": 1, "fingerprint_backend": "hash"}

    X_source, _, names, metadata = build_feature_matrix_with_metadata(source, config)
    X_synthetic, _, synthetic_names, synthetic_metadata = build_feature_matrix_with_metadata(synthetic, config)

    np.testing.assert_array_equal(X_source, X_synthetic)
    assert names == synthetic_names
    assert metadata == synthetic_metadata


def test_explicit_donor_similarity_preserves_legacy_summed_role_ranking() -> None:
    frame = reaction_roles_dataframe(
        [_source(), _donor(), ReactionRoles("CCCl", "CCN", "[Pd]", "P(C)(C)C", "N1CCCCC1", "CCO", "CCNC")],
        yields=[10.0, 20.0, 30.0],
    )
    roles = ["catalyst", "ligand", "base", "solvent_or_additive"]
    actual = _role_similarity_matrix(
        frame,
        roles,
        n_bits=256,
        radius=2,
        backend="hash",
    )

    vectors = []
    role_columns = {
        "catalyst": "recovered_catalyst_smiles",
        "ligand": "recovered_ligand_smiles",
        "base": "recovered_base_smiles",
        "solvent_or_additive": "recovered_solvent_or_additive_smiles",
    }
    for _, row in frame.iterrows():
        vector = sum(
            (
                morgan_fingerprint(
                    row[role_columns[role]],
                    radius=2,
                    n_bits=256,
                    warn_invalid=False,
                    backend="hash",
                )
                for role in roles
            ),
            np.zeros(256, dtype=np.float32),
        )
        vectors.append(vector)
    matrix = np.vstack(vectors)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    legacy = normalized @ normalized.T

    np.testing.assert_array_equal(actual, legacy)
    assert np.argsort(-actual, axis=1).tolist() == np.argsort(-legacy, axis=1).tolist()


def _source() -> ReactionRoles:
    return ReactionRoles("c1ccccc1Br", "CN", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "c1ccccc1NC")


def _donor() -> ReactionRoles:
    return ReactionRoles("CCBr", "CCN", "[Ni]", "P(CC)(CC)CC", "N1CCCCC1", "CCCO", "CCNC")
