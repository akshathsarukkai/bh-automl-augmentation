"""Scientific contracts for the canonical Phase 10 representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ROLE_TO_COLUMN,
)
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    morgan_fingerprint,
)

pytest.importorskip("rdkit")

N_BITS = 64
SUBSTRATE_ROLES = ("reactant_1", "reactant_2")
CONDITION_ROLES = ("catalyst", "ligand", "base", "solvent_or_additive")
PRODUCT_FREE_ROLES = (*SUBSTRATE_ROLES, *CONDITION_ROLES)

ROLE_SMILES = {
    "reactant_1": "Brc1ccccc1",
    "reactant_2": "CN",
    "catalyst": "[Pd]",
    "ligand": "P(C)(C)C",
    "base": "CC(C)(C)[O-].[Na+]",
    "solvent_or_additive": "CCO",
    "product": "CNc1ccccc1",
}

EXPECTED_BLOCKS = {
    "bh_flat_reaction_sum": ("all_roles_sum",),
    "bh_reaction_section_concat": ("substrate_sum", "condition_sum", "product"),
    "bh_role_separated": CANONICAL_ROLE_NAMES,
    "bh_role_separated_delta": (
        *CANONICAL_ROLE_NAMES,
        "delta_product_minus_reactant_1",
        "delta_product_minus_reactant_2",
        "delta_product_minus_reactant_pair",
    ),
    "bh_substrate_only": SUBSTRATE_ROLES,
    "bh_condition_only": CONDITION_ROLES,
    "bh_product_aware": CANONICAL_ROLE_NAMES,
    "bh_product_free": PRODUCT_FREE_ROLES,
}

EXPECTED_ROLE_ORDERING = {
    "bh_flat_reaction_sum": CANONICAL_ROLE_NAMES,
    "bh_reaction_section_concat": CANONICAL_ROLE_NAMES,
    "bh_role_separated": CANONICAL_ROLE_NAMES,
    "bh_role_separated_delta": CANONICAL_ROLE_NAMES,
    "bh_substrate_only": SUBSTRATE_ROLES,
    "bh_condition_only": CONDITION_ROLES,
    "bh_product_aware": CANONICAL_ROLE_NAMES,
    "bh_product_free": PRODUCT_FREE_ROLES,
}


def _frame(**role_overrides: object) -> pd.DataFrame:
    roles = {**ROLE_SMILES, **role_overrides}
    record = {
        "yield": 73.0,
        "reaction_smiles": "this field must not define canonical role features",
    }
    record.update({ROLE_TO_COLUMN[role]: value for role, value in roles.items()})
    return pd.DataFrame([record])


def _config(kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "radius": 2,
        "n_bits": N_BITS,
        "fingerprint_backend": "rdkit",
        "categorical_columns": [],
    }


def _role_fingerprints() -> dict[str, np.ndarray]:
    return {
        role: morgan_fingerprint(
            smiles,
            radius=2,
            n_bits=N_BITS,
            backend="rdkit",
        )
        for role, smiles in ROLE_SMILES.items()
    }


def _expected_vector(kind: str) -> np.ndarray:
    fingerprints = _role_fingerprints()
    if kind == "bh_flat_reaction_sum":
        return sum(
            (fingerprints[role] for role in CANONICAL_ROLE_NAMES),
            start=np.zeros(N_BITS, dtype=np.float32),
        )
    if kind == "bh_reaction_section_concat":
        substrate = fingerprints["reactant_1"] + fingerprints["reactant_2"]
        condition = sum(
            (fingerprints[role] for role in CONDITION_ROLES),
            start=np.zeros(N_BITS, dtype=np.float32),
        )
        return np.concatenate([substrate, condition, fingerprints["product"]])
    if kind == "bh_role_separated_delta":
        product = fingerprints["product"]
        reactant_1 = fingerprints["reactant_1"]
        reactant_2 = fingerprints["reactant_2"]
        return np.concatenate(
            [
                *(fingerprints[role] for role in CANONICAL_ROLE_NAMES),
                product - reactant_1,
                product - reactant_2,
                product - reactant_1 - reactant_2,
            ]
        )
    roles = EXPECTED_ROLE_ORDERING[kind]
    return np.concatenate([fingerprints[role] for role in roles])


@pytest.mark.parametrize("kind", tuple(EXPECTED_BLOCKS))
def test_role_derived_representations_match_exact_formulas_and_metadata(
    kind: str,
) -> None:
    X, y, names, metadata = build_feature_matrix_with_metadata(_frame(), _config(kind))

    expected = _expected_vector(kind).astype(np.float32)
    expected_blocks = EXPECTED_BLOCKS[kind]
    assert X.shape == (1, len(expected))
    np.testing.assert_array_equal(X[0], expected)
    np.testing.assert_array_equal(y, np.array([73.0], dtype=np.float32))
    assert names == [
        f"{block}__morgan_{bit}"
        for block in expected_blocks
        for bit in range(N_BITS)
    ]
    assert metadata.representation_kind == kind
    assert metadata.role_ordering == tuple(EXPECTED_ROLE_ORDERING[kind])
    assert metadata.total_width == len(expected_blocks) * N_BITS
    assert tuple(block.name for block in metadata.block_slices) == expected_blocks
    assert tuple(
        (block.start, block.stop) for block in metadata.block_slices
    ) == tuple(
        (index * N_BITS, (index + 1) * N_BITS)
        for index in range(len(expected_blocks))
    )


def test_product_aware_is_feature_identical_but_metadata_distinct_control() -> None:
    separated = build_feature_matrix_with_metadata(
        _frame(), _config("bh_role_separated")
    )
    product_aware = build_feature_matrix_with_metadata(
        _frame(), _config("bh_product_aware")
    )

    np.testing.assert_array_equal(separated[0], product_aware[0])
    assert separated[2] == product_aware[2]
    assert separated[3].role_ordering == product_aware[3].role_ordering
    assert separated[3].block_slices == product_aware[3].block_slices
    assert separated[3].representation_kind == "bh_role_separated"
    assert product_aware[3].representation_kind == "bh_product_aware"


def test_each_restricted_representation_is_invariant_to_excluded_roles() -> None:
    baseline_substrate = build_feature_matrix_with_metadata(
        _frame(), _config("bh_substrate_only")
    )[0]
    changed_substrate_exclusions = _frame(
        catalyst="[Pt]",
        ligand="P(c1ccccc1)(c1ccccc1)c1ccccc1",
        base="[OH-].[K+]",
        solvent_or_additive="CC#N",
        product="CC(=O)Oc1ccccc1C(=O)O",
    )
    substrate_after = build_feature_matrix_with_metadata(
        changed_substrate_exclusions, _config("bh_substrate_only")
    )[0]
    np.testing.assert_array_equal(baseline_substrate, substrate_after)

    baseline_condition = build_feature_matrix_with_metadata(
        _frame(), _config("bh_condition_only")
    )[0]
    changed_condition_exclusions = _frame(
        reactant_1="Clc1ncccc1",
        reactant_2="NCCN",
        product="CC(=O)Oc1ccccc1C(=O)O",
    )
    condition_after = build_feature_matrix_with_metadata(
        changed_condition_exclusions, _config("bh_condition_only")
    )[0]
    np.testing.assert_array_equal(baseline_condition, condition_after)


def test_product_free_never_reads_product_or_reaction_fields() -> None:
    baseline = _frame().drop(columns=["reaction_smiles", ROLE_TO_COLUMN["product"]])
    adversarial = baseline.assign(
        reaction_smiles=object(),
        **{ROLE_TO_COLUMN["product"]: "definitely-not-a-smiles"},
    )

    X_missing, _, names_missing, metadata_missing = build_feature_matrix_with_metadata(
        baseline, _config("bh_product_free")
    )
    X_adversarial, _, names_adversarial, metadata_adversarial = (
        build_feature_matrix_with_metadata(adversarial, _config("bh_product_free"))
    )

    np.testing.assert_array_equal(X_missing, X_adversarial)
    assert names_missing == names_adversarial
    assert metadata_missing == metadata_adversarial


def test_product_perturbation_changes_product_aware_but_not_product_free() -> None:
    baseline = _frame()
    changed_product = _frame(product="O=C(O)c1ccccc1")

    product_free_before = build_feature_matrix_with_metadata(
        baseline, _config("bh_product_free")
    )[0]
    product_free_after = build_feature_matrix_with_metadata(
        changed_product, _config("bh_product_free")
    )[0]
    product_aware_before = build_feature_matrix_with_metadata(
        baseline, _config("bh_product_aware")
    )[0]
    product_aware_after = build_feature_matrix_with_metadata(
        changed_product, _config("bh_product_aware")
    )[0]

    np.testing.assert_array_equal(product_free_before, product_free_after)
    assert not np.array_equal(product_aware_before, product_aware_after)


def test_included_invalid_role_is_rejected_in_rdkit_mode() -> None:
    invalid_ligand = _frame(ligand="definitely-not-a-smiles")
    with pytest.raises(ValueError, match="role 'ligand'"):
        build_feature_matrix_with_metadata(
            invalid_ligand, _config("bh_condition_only")
        )

    invalid_product = _frame(product="definitely-not-a-smiles")
    build_feature_matrix_with_metadata(invalid_product, _config("bh_product_free"))
    with pytest.raises(ValueError, match="role 'product'"):
        build_feature_matrix_with_metadata(
            invalid_product, _config("bh_product_aware")
        )


def test_product_aware_requires_product_but_product_free_does_not() -> None:
    no_product = _frame().drop(columns=[ROLE_TO_COLUMN["product"]])
    build_feature_matrix_with_metadata(no_product, _config("bh_product_free"))

    with pytest.raises(
        ValueError,
        match="recovered_product_smiles",
    ):
        build_feature_matrix_with_metadata(no_product, _config("bh_product_aware"))
