"""Tests for the canonical Buchwald-Hartwig role schema."""

import pandas as pd
import pytest

from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    CANONICAL_ROLE_NAMES,
    ReactionRoles,
    ensure_reaction_role_columns,
    reaction_roles_dataframe,
    reaction_roles_from_row,
    reaction_roles_to_record,
)


def test_canonical_role_order_and_round_trip() -> None:
    roles = _roles()
    record = reaction_roles_to_record(roles)

    assert CANONICAL_ROLE_NAMES == (
        "reactant_1",
        "reactant_2",
        "catalyst",
        "ligand",
        "base",
        "solvent_or_additive",
        "product",
    )
    assert tuple(record) == (*CANONICAL_ROLE_COLUMNS, "reaction_smiles")
    assert reaction_roles_from_row(record) == roles


def test_ensure_roles_recovers_from_six_token_reaction_and_retains_index() -> None:
    roles = _roles()
    frame = pd.DataFrame({"reaction_smiles": [roles.reaction_smiles()], "yield": [50.0]}, index=[17])

    recovered = ensure_reaction_role_columns(frame)

    assert recovered.index.tolist() == [17]
    assert reaction_roles_from_row(recovered.loc[17]) == roles


def test_missing_required_role_reports_row_and_column() -> None:
    record = reaction_roles_to_record(_roles())
    record["recovered_ligand_smiles"] = "UNKNOWN"
    frame = pd.DataFrame([{**record, "yield": 1.0}], index=[42])

    with pytest.raises(ValueError, match=r"row 42.*recovered_ligand_smiles"):
        ensure_reaction_role_columns(frame, parse_if_missing=False)


def test_empty_role_dataframe_is_safe_and_has_canonical_columns() -> None:
    frame = reaction_roles_dataframe([])

    assert frame.empty
    assert list(frame.columns) == [*CANONICAL_ROLE_COLUMNS, "reaction_smiles", "yield"]


def _roles() -> ReactionRoles:
    return ReactionRoles(
        reactant_1="c1ccccc1Br",
        reactant_2="CN",
        catalyst="[Pd]",
        ligand="P(C)(C)C",
        base="N(C)(C)C",
        solvent_or_additive="CCO",
        product="c1ccccc1NC",
    )
