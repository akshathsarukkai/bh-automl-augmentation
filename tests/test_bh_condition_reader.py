"""Tests for processed Buchwald-Hartwig condition role recovery."""

from __future__ import annotations

import pandas as pd
import pytest

from bh_augmentation.data.bh_condition_reader import (
    NOT_RECOVERABLE,
    PARSED_COLUMNS,
    RECOVERED_COLUMNS,
    ROLE_VALIDATION_COLUMNS,
    augment_bh_dataframe,
    classify_bh_component,
    parse_bh_reaction_smiles,
    recover_condition_fields,
    summarize_recovery,
)


def test_valid_six_token_reaction_smiles_parses_correctly() -> None:
    parsed = parse_bh_reaction_smiles("A.B.Cat.Lig.Base.Solv>>P")

    assert parsed is not None
    assert parsed["condition_parse_status"] == "ok"
    assert parsed["parsed_reactant_1_smiles"] == "A"
    assert parsed["parsed_reactant_2_smiles"] == "B"
    assert parsed["parsed_catalyst_smiles"] == "Cat"
    assert parsed["parsed_ligand_smiles"] == "Lig"
    assert parsed["parsed_base_smiles"] == "Base"
    assert parsed["parsed_solvent_or_additive_smiles"] == "Solv"
    assert parsed["parsed_product_smiles"] == "P"
    assert parsed["condition_block_smiles"] == "Cat.Lig.Base.Solv"


def test_malformed_missing_separator_returns_malformed_status() -> None:
    parsed = parse_bh_reaction_smiles("A.B.Cat.Lig.Base.Solv")

    assert parsed is not None
    assert parsed["condition_parse_status"] == "malformed"
    assert ">>" in str(parsed["condition_parse_error"])


def test_wrong_left_token_count_returns_unexpected_status() -> None:
    parsed = parse_bh_reaction_smiles("A.B.Cat>>P")

    assert parsed is not None
    assert parsed["condition_parse_status"] == "unexpected_left_token_count"
    assert "expected 6" in str(parsed["condition_parse_error"])


def test_existing_known_catalyst_is_preserved_over_parsed_catalyst() -> None:
    row = _row(catalyst_smiles="ExistingCatalyst")

    recovered = recover_condition_fields(row)

    assert recovered["parsed_catalyst_smiles"] == "Cat"
    assert recovered["recovered_catalyst_smiles"] == "ExistingCatalyst"


def test_unknown_catalyst_is_recovered_from_token_two() -> None:
    row = _row(catalyst_smiles="UNKNOWN")

    recovered = recover_condition_fields(row)

    assert recovered["recovered_catalyst_smiles"] == "Cat"


def test_unknown_temperature_is_not_hallucinated() -> None:
    row = _row(temperature="UNKNOWN")

    recovered = recover_condition_fields(row)

    assert recovered["recovered_temperature"] == NOT_RECOVERABLE


def test_product_mismatch_is_flagged() -> None:
    row = _row(product_smiles="DifferentProduct")

    recovered = recover_condition_fields(row)

    assert "product_mismatch" in str(recovered["condition_parse_status"])
    assert "product_mismatch" in str(recovered["condition_parse_error"])


def test_strict_true_raises_on_mismatch() -> None:
    row = _chemical_row(product_smiles="DifferentProduct")

    with pytest.raises(ValueError, match="product_mismatch"):
        recover_condition_fields(row, strict=True)


def test_augment_bh_dataframe_adds_expected_columns() -> None:
    df = pd.DataFrame([_row(), _row(reaction_smiles="A.B.Cat>>P")])

    augmented = augment_bh_dataframe(df)

    for column in [*PARSED_COLUMNS, *RECOVERED_COLUMNS, *ROLE_VALIDATION_COLUMNS]:
        assert column in augmented.columns
    assert augmented.loc[0, "condition_parse_status"] == "ok"
    assert augmented.loc[1, "condition_parse_status"] == "unexpected_left_token_count"


def test_summarize_recovery_returns_counts() -> None:
    df = pd.DataFrame([_row(), _row(reaction_smiles="bad")])
    augmented = augment_bh_dataframe(df)

    summary = summarize_recovery(augmented)

    assert {"section", "field", "value"} <= set(summary.columns)
    status_rows = summary.loc[summary["section"] == "parse_status"]
    assert set(status_rows["field"]) == {"ok", "malformed"}
    unique_rows = summary.loc[summary["section"] == "unique_recovered_values"]
    assert "recovered_catalyst_smiles" in set(unique_rows["field"])


def test_correctly_ordered_condition_block_validates() -> None:
    recovered = recover_condition_fields(_chemical_row())

    assert recovered["role_validation_status"] == "valid"
    assert recovered["recovered_catalyst_smiles"] == _PD_CATALYST
    assert recovered["recovered_ligand_smiles"] == _PHOSPHINE_LIGAND
    assert recovered["recovered_base_smiles"] == _PHOSPHATE_BASE
    assert recovered["recovered_solvent_or_additive_smiles"] == _DIOXANE_SOLVENT


def test_swapped_catalyst_and_ligand_are_repaired() -> None:
    row = _chemical_row(reaction_smiles=f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_PHOSPHINE_LIGAND}.{_PD_CATALYST}.{_PHOSPHATE_BASE}.{_DIOXANE_SOLVENT}>>{_PRODUCT}")

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] == "repaired"
    assert recovered["recovered_catalyst_smiles"] == _PD_CATALYST
    assert recovered["recovered_ligand_smiles"] == _PHOSPHINE_LIGAND


def test_swapped_base_and_solvent_are_repaired() -> None:
    row = _chemical_row(reaction_smiles=f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_PD_CATALYST}.{_PHOSPHINE_LIGAND}.{_DIOXANE_SOLVENT}.{_PHOSPHATE_BASE}>>{_PRODUCT}")

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] == "repaired"
    assert recovered["recovered_base_smiles"] == _PHOSPHATE_BASE
    assert recovered["recovered_solvent_or_additive_smiles"] == _DIOXANE_SOLVENT


def test_ambiguous_two_phosphorus_tokens_are_not_repaired() -> None:
    second_ligand_like_token = "P(C)(C)C"
    row = _chemical_row(reaction_smiles=f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_PD_CATALYST}.{_PHOSPHINE_LIGAND}.{second_ligand_like_token}.{_DIOXANE_SOLVENT}>>{_PRODUCT}")

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] == "ambiguous"


def test_palladium_chloride_is_catalyst_not_electrophile() -> None:
    roles = classify_bh_component(_PD_CATALYST)

    assert "catalyst" in roles
    assert "electrophile" not in roles


def test_aryl_bromide_is_electrophile_only_among_reactive_roles() -> None:
    roles = classify_bh_component(_ELECTROPHILE)

    assert "electrophile" in roles
    assert "catalyst" not in roles
    assert "base" not in roles
    assert "ligand" not in roles


def test_amine_is_nucleophile_unless_known_standalone_base() -> None:
    amine_roles = classify_bh_component(_NUCLEOPHILE)
    known_base_roles = classify_bh_component("NEt3")

    assert "nucleophile" in amine_roles
    assert "base" not in amine_roles
    assert "base" in known_base_roles
    assert "nucleophile" not in known_base_roles


def test_organic_guanidine_superbase_classifies_as_base() -> None:
    roles = classify_bh_component(_GUANIDINE_BASE)

    assert "base" in roles
    assert "nucleophile" not in roles


def test_phosphazene_superbase_classifies_as_base() -> None:
    roles = classify_bh_component(_PHOSPHAZENE_BASE)

    assert "base" in roles
    assert "nucleophile" not in roles


def test_normal_aniline_reactant_stays_nucleophile_not_base() -> None:
    roles = classify_bh_component("Cc1ccc(N)cc1")

    assert "nucleophile" in roles
    assert "base" not in roles


def test_phosphine_ligand_without_phosphazene_motif_is_not_base() -> None:
    roles = classify_bh_component(_PHOSPHINE_LIGAND)

    assert "ligand" in roles
    assert "base" not in roles


def test_guanidine_condition_block_no_longer_ambiguous() -> None:
    row = _chemical_row(
        reaction_smiles=(
            f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_DATASET_PD_CATALYST}."
            f"{_DATASET_PHOSPHINE_LIGAND}.{_GUANIDINE_BASE}.{_OXAZOLE_ADDITIVE}>>{_PRODUCT}"
        )
    )

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] in {"valid", "repaired"}
    assert recovered["recovered_base_smiles"] == _GUANIDINE_BASE


def test_n_containing_condition_additive_uses_position_fallback() -> None:
    row = _chemical_row(
        reaction_smiles=(
            f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_DATASET_PD_CATALYST}."
            f"{_DATASET_PHOSPHINE_LIGAND}.{_PHOSPHAZENE_BASE}.{_N_CONTAINING_ADDITIVE}>>{_PRODUCT}"
        )
    )

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] == "valid"
    assert recovered["recovered_solvent_or_additive_smiles"] == _N_CONTAINING_ADDITIVE
    assert "remaining_condition_token_assigned_to_solvent_or_additive" in str(
        recovered["role_validation_notes"]
    )


def test_reactant_token_one_normal_amine_still_classifies_as_nucleophile() -> None:
    parsed = parse_bh_reaction_smiles(
        f"{_ELECTROPHILE}.Cc1ccc(N)cc1.{_PD_CATALYST}.{_PHOSPHINE_LIGAND}."
        f"{_PHOSPHATE_BASE}.{_DIOXANE_SOLVENT}>>{_PRODUCT}"
    )

    assert parsed is not None
    roles = classify_bh_component(str(parsed["parsed_reactant_2_smiles"]))
    assert "nucleophile" in roles
    assert "base" not in roles


def test_palladium_remaining_condition_token_is_not_solvent_fallback() -> None:
    row = _chemical_row(
        reaction_smiles=(
            f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_PD_CATALYST}.{_PHOSPHINE_LIGAND}."
            f"{_PHOSPHATE_BASE}.{_DATASET_PD_CATALYST}>>{_PRODUCT}"
        )
    )

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] == "ambiguous"
    assert "remaining_condition_token_assigned_to_solvent_or_additive" not in str(
        recovered["role_validation_notes"]
    )


def test_missing_base_with_two_additive_like_tokens_stays_ambiguous() -> None:
    row = _chemical_row(
        reaction_smiles=(
            f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_DATASET_PD_CATALYST}."
            f"{_DATASET_PHOSPHINE_LIGAND}.{_DIOXANE_SOLVENT}.{_N_CONTAINING_ADDITIVE}>>{_PRODUCT}"
        )
    )

    recovered = recover_condition_fields(row)

    assert recovered["role_validation_status"] == "ambiguous"
    assert "remaining_condition_token_assigned_to_solvent_or_additive" not in str(
        recovered["role_validation_notes"]
    )


def test_unknown_or_empty_components_classify_unknown() -> None:
    assert classify_bh_component("UNKNOWN") == {"unknown"}
    assert classify_bh_component("") == {"unknown"}


def test_strict_true_raises_on_ambiguous_role_validation() -> None:
    row = _chemical_row(reaction_smiles=f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_PD_CATALYST}.{_PHOSPHINE_LIGAND}.P(C)(C)C.{_DIOXANE_SOLVENT}>>{_PRODUCT}")

    with pytest.raises(ValueError, match="ambiguous_or_missing"):
        recover_condition_fields(row, strict=True)


_ELECTROPHILE = "Brc1ccccc1"
_NUCLEOPHILE = "Nc1ccccc1"
_PD_CATALYST = "Cl[Pd]Cl"
_PHOSPHINE_LIGAND = "P(c1ccccc1)(c1ccccc1)c1ccccc1"
_PHOSPHATE_BASE = "K3PO4"
_DIOXANE_SOLVENT = "O1CCOCC1"
_PRODUCT = "c1ccc(Nc2ccccc2)cc1"
_DATASET_PD_CATALYST = "O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F"
_DATASET_PHOSPHINE_LIGAND = "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1"
_GUANIDINE_BASE = "CN(C)C(=NC(C)(C)C)N(C)C"
_PHOSPHAZENE_BASE = "CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C"
_OXAZOLE_ADDITIVE = "c1ccc(-c2ccno2)cc1"
_N_CONTAINING_ADDITIVE = "c1ccc(CN(Cc2ccccc2)c2ccon2)cc1"


def _chemical_row(**overrides: object) -> dict[str, object]:
    row = _row(
        reaction_smiles=f"{_ELECTROPHILE}.{_NUCLEOPHILE}.{_PD_CATALYST}.{_PHOSPHINE_LIGAND}.{_PHOSPHATE_BASE}.{_DIOXANE_SOLVENT}>>{_PRODUCT}",
        reactant_1_smiles=_ELECTROPHILE,
        reactant_2_smiles=_NUCLEOPHILE,
        product_smiles=_PRODUCT,
    )
    row.update(overrides)
    return row


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "reaction_id": "rxn_1",
        "reaction_smiles": "A.B.Cat.Lig.Base.Solv>>P",
        "reactant_1_smiles": "A",
        "reactant_2_smiles": "B",
        "product_smiles": "P",
        "catalyst_smiles": "UNKNOWN",
        "solvent": "UNKNOWN",
        "temperature": "UNKNOWN",
        "yield": 50.0,
    }
    row.update(overrides)
    return row
