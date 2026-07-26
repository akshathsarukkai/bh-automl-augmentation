"""Tests for canonical seven-role reaction identities."""

from __future__ import annotations

import json

from tests.canonical_test_utils import measured_role_frame

from bh_augmentation.data.canonicalize_roles import (
    REACTION_KEY_SCHEMA_VERSION,
    build_canonical_reaction_identity,
    canonicalize_reaction_roles_dataframe,
    stable_json,
)


def test_equivalent_raw_encodings_produce_same_reaction_key() -> None:
    canonical = canonicalize_reaction_roles_dataframe(
        measured_role_frame().iloc[:2],
        source_file_hash="a" * 64,
    )
    assert canonical["canonical_reaction_key"].nunique() == 1
    assert canonical["canonical_reaction_hash"].nunique() == 1


def test_changing_one_role_changes_key() -> None:
    frame = measured_role_frame().iloc[[0, 4]]
    canonical = canonicalize_reaction_roles_dataframe(frame, source_file_hash="b" * 64)
    assert canonical["canonical_reaction_hash"].nunique() == 2


def test_swapping_reactant_roles_changes_key() -> None:
    canonical = canonicalize_reaction_roles_dataframe(
        measured_role_frame().iloc[:1],
        source_file_hash="c" * 64,
    )
    row = canonical.iloc[0].to_dict()
    original = build_canonical_reaction_identity(row)
    row["canonical_reactant_1_smiles"], row["canonical_reactant_2_smiles"] = (
        row["canonical_reactant_2_smiles"],
        row["canonical_reactant_1_smiles"],
    )
    swapped = build_canonical_reaction_identity(row)
    assert original["hash"] != swapped["hash"]


def test_dataframe_column_order_does_not_change_key() -> None:
    frame = measured_role_frame()
    reversed_frame = frame[list(reversed(frame.columns))]
    first = canonicalize_reaction_roles_dataframe(frame, source_file_hash="d" * 64)
    second = canonicalize_reaction_roles_dataframe(reversed_frame, source_file_hash="d" * 64)
    assert first["canonical_reaction_key"].tolist() == second["canonical_reaction_key"].tolist()


def test_invalid_role_prevents_valid_key() -> None:
    row = {
        "canonical_reactant_1_smiles": "CCBr",
        "canonical_reactant_2_smiles": "CN",
        "canonical_catalyst_smiles": None,
        "canonical_ligand_smiles": "CP(C)C",
        "canonical_base_smiles": "[OH-].[Na+]",
        "canonical_solvent_or_additive_smiles": "CCO",
        "canonical_product_smiles": "CCN",
    }
    assert build_canonical_reaction_identity(row) == {"key": None, "hash": None}


def test_key_serialization_is_stable_and_versioned() -> None:
    first = stable_json({"reactant_1": "CCBr", "schema": REACTION_KEY_SCHEMA_VERSION})
    second = stable_json({"schema": REACTION_KEY_SCHEMA_VERSION, "reactant_1": "CCBr"})
    assert first == second
    assert json.loads(first)["schema"] == REACTION_KEY_SCHEMA_VERSION


def test_source_row_ids_follow_original_positions_after_reordering() -> None:
    frame = measured_role_frame().copy()
    canonical = canonicalize_reaction_roles_dataframe(frame, source_file_hash="e" * 64)
    reordered = frame.sample(frac=1.0, random_state=4)
    second = canonicalize_reaction_roles_dataframe(reordered, source_file_hash="e" * 64)
    first_ids = canonical.set_index("reaction_id")["source_row_id"].to_dict()
    second_ids = second.set_index("reaction_id")["source_row_id"].to_dict()
    assert first_ids == second_ids


def test_explicit_invalidity_flag_prevents_identity() -> None:
    canonical = canonicalize_reaction_roles_dataframe(
        measured_role_frame().iloc[:1],
        source_file_hash="f" * 64,
    )
    row = canonical.iloc[0].to_dict()
    row["base_parse_valid"] = False
    assert build_canonical_reaction_identity(row) == {"key": None, "hash": None}
