"""Tests for conservative canonical BH substrate-scaffold definitions."""

from __future__ import annotations

import pandas as pd
import pytest

from bh_augmentation.data.canonicalize_roles import canonicalize_smiles
from bh_augmentation.evaluation.scaffold_ood import (
    AMINE_SMARTS,
    ELECTROPHILE_SMARTS,
    build_scaffold_ood_assignments,
)

pytest.importorskip("rdkit")


def _canon(smiles: str) -> str:
    result = canonicalize_smiles(smiles, isomeric=True)
    assert result.parse_valid
    assert result.canonical_smiles is not None
    return result.canonical_smiles


def _frame(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_row_id": f"row-{index:02d}",
                "canonical_reaction_key": f"reaction-{index:02d}",
                "canonical_reactant_1_smiles": _canon(reactant_1),
                "canonical_reactant_2_smiles": _canon(reactant_2),
            }
            for index, (reactant_1, reactant_2) in enumerate(pairs)
        ]
    )


def test_role_basis_is_explicit_and_independent_of_reactant_position() -> None:
    frame = _frame(
        [
            ("Brc1ccccc1", "NC1CCCCC1"),
            ("N1CCCCC1", "Brc1ccc2ccccc2c1"),
        ]
    )

    electrophiles = build_scaffold_ood_assignments(
        frame,
        target="electrophile_scaffold",
    ).assignments
    amines = build_scaffold_ood_assignments(
        frame,
        target="amine_scaffold",
    ).assignments

    assert ELECTROPHILE_SMARTS == "[#6]-[Cl,Br,I]"
    assert AMINE_SMARTS == "[N;H1,H2,H3;+0,+1]"
    assert electrophiles["assigned_role"].tolist() == [
        "reactant_1",
        "reactant_2",
    ]
    assert amines["assigned_role"].tolist() == [
        "reactant_2",
        "reactant_1",
    ]
    assert electrophiles["included"].all()
    assert amines["included"].all()
    assert electrophiles["bemis_murcko_scaffold_smiles"].tolist() == [
        "c1ccccc1",
        "c1ccc2ccccc2c1",
    ]
    assert amines["bemis_murcko_scaffold_smiles"].tolist() == [
        "C1CCCCC1",
        "C1CCNCC1",
    ]


def test_scaffold_holdout_contains_exactly_the_intended_molecules() -> None:
    frame = _frame(
        [
            ("Brc1ccccc1", "NC1CCCCC1"),
            ("Brc1ccccc1C", "Nc1ccccc1"),
            ("Brc1ccc2ccccc2c1", "N1CCCCC1"),
        ]
    )
    assignments = build_scaffold_ood_assignments(
        frame,
        target="electrophile_scaffold",
    )
    benzene_key = assignments.included.loc[
        assignments.included["bemis_murcko_scaffold_smiles"].eq("c1ccccc1"),
        "scaffold_key",
    ].iloc[0]

    holdout = assignments.holdout(benzene_key)

    assert holdout.test_source_ids == ("row-00", "row-01")
    assert holdout.train_source_ids == ("row-02",)
    assert not set(holdout.train_source_ids) & set(holdout.test_source_ids)
    assert holdout.split_hash


def test_scaffold_holdout_rejects_insufficient_group_support() -> None:
    assignments = build_scaffold_ood_assignments(
        _frame([("Brc1ccccc1", "NC1CCCCC1")]),
        target="electrophile_scaffold",
    )

    with pytest.raises(ValueError, match="Insufficient included"):
        assignments.holdout(assignments.included_scaffold_groups[0])


def test_empty_scaffold_is_excluded_without_invalidating_role_assignment() -> None:
    frame = _frame([("Brc1ccccc1", "CN")])

    amines = build_scaffold_ood_assignments(
        frame,
        target="amine_scaffold",
    ).assignments
    electrophiles = build_scaffold_ood_assignments(
        frame,
        target="electrophile_scaffold",
    ).assignments

    assert amines.loc[0, "chemical_parse_valid"]
    assert amines.loc[0, "role_assignment_valid"]
    assert not amines.loc[0, "included"]
    assert amines.loc[0, "exclusion_reason"] == "empty_bemis_murcko_scaffold"
    assert electrophiles.loc[0, "included"]


@pytest.mark.parametrize(
    ("pair", "reason"),
    [
        (
            ("O=S(=O)(Oc1ccccc1)C(F)(F)F", "NC1CCCCC1"),
            "no_unique_carbon_halogen_electrophile",
        ),
        (
            ("Brc1ccccc1", "BrCCN"),
            "ambiguous_carbon_halogen_electrophile",
        ),
        (
            ("Brc1ccccc1", "CCO"),
            "complementary_reactant_not_amine_like",
        ),
    ],
)
def test_unsupported_or_ambiguous_role_chemistry_has_explicit_exclusion(
    pair: tuple[str, str],
    reason: str,
) -> None:
    assignment = build_scaffold_ood_assignments(
        _frame([pair]),
        target="electrophile_scaffold",
    ).assignments.iloc[0]

    assert not assignment["included"]
    assert assignment["exclusion_reason"] == reason


def test_invalid_and_noncanonical_role_identities_are_explicitly_excluded() -> None:
    invalid = _frame([("Brc1ccccc1", "NC1CCCCC1")])
    invalid.loc[0, "canonical_reactant_1_smiles"] = "not-a-smiles"
    invalid_assignment = build_scaffold_ood_assignments(
        invalid,
        target="electrophile_scaffold",
    ).assignments.iloc[0]
    assert not invalid_assignment["chemical_parse_valid"]
    assert (
        invalid_assignment["exclusion_reason"]
        == "invalid_reactant_1_canonical_identity"
    )

    noncanonical = _frame([("Brc1ccccc1", "NC1CCCCC1")])
    noncanonical.loc[0, "canonical_reactant_1_smiles"] = "c1ccccc1Br"
    noncanonical_assignment = build_scaffold_ood_assignments(
        noncanonical,
        target="electrophile_scaffold",
    ).assignments.iloc[0]
    assert (
        noncanonical_assignment["exclusion_reason"]
        == "noncanonical_reactant_1_identity"
    )


def test_stereochemistry_and_charge_are_distinct_before_scaffold_derivation() -> None:
    frame = _frame(
        [
            ("Brc1ccccc1", "N[C@H](C)c1ccccc1"),
            ("Brc1ccccc1", "N[C@@H](C)c1ccccc1"),
            ("Brc1ccccc1", "[NH2+][C@H](C)c1ccccc1"),
        ]
    )

    assignments = build_scaffold_ood_assignments(
        frame,
        target="amine_scaffold",
    ).assignments

    assert assignments["included"].all()
    assert assignments["assigned_canonical_smiles"].nunique() == 3
    assert assignments["molecule_identity_hash"].nunique() == 3
    assert assignments["assigned_canonical_smiles"].str.contains("@").all()
    assert assignments["assigned_canonical_smiles"].str.contains(
        r"\+",
        regex=True,
    ).sum() == 1
    assert assignments["bemis_murcko_scaffold_smiles"].eq("c1ccccc1").all()
    assert assignments["scaffold_key"].nunique() == 1


def test_assignments_and_hashes_are_row_order_invariant_and_defensive() -> None:
    frame = _frame(
        [
            ("Brc1ccccc1", "NC1CCCCC1"),
            ("Brc1ccc2ccccc2c1", "N1CCCCC1"),
            ("BrC1CCCCC1", "Nc1ccccc1"),
        ]
    )
    first = build_scaffold_ood_assignments(
        frame,
        target="electrophile_scaffold",
    )
    second = build_scaffold_ood_assignments(
        frame.sample(frac=1.0, random_state=22),
        target="electrophile_scaffold",
    )

    assert first.assignment_hash == second.assignment_hash
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    changed = first.assignments
    changed.loc[:, "included"] = False
    assert first.assignments["included"].all()


def test_input_identity_and_target_validation_is_strict() -> None:
    frame = _frame([("Brc1ccccc1", "NC1CCCCC1")])
    with pytest.raises(ValueError, match="missing columns"):
        build_scaffold_ood_assignments(
            frame.drop(columns="canonical_reaction_key"),
            target="electrophile_scaffold",
        )
    with pytest.raises(ValueError, match="must be unique"):
        build_scaffold_ood_assignments(
            pd.concat([frame, frame], ignore_index=True),
            target="electrophile_scaffold",
        )
    with pytest.raises(ValueError, match="Unsupported scaffold OOD target"):
        build_scaffold_ood_assignments(
            frame,
            target="generic_scaffold",
        )
