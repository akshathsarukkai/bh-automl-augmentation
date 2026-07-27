from __future__ import annotations

import json

import pandas as pd
import pytest

from bh_augmentation.data.canonicalize_roles import CANONICALIZATION_VERSION
from bh_augmentation.evaluation.condition_ood import (
    CONDITION_COMBINATION_KEY_SCHEMA_VERSION,
    FoldSupportCriteria,
    add_condition_combination_keys,
    build_condition_ood_plan,
    canonical_condition_combination_key,
)


def _canonical_fixture() -> pd.DataFrame:
    conditions = [
        ("[Pd]", "P(C)(C)C", "O=C([O-])[O-]", "CCO"),
        ("[Pd]", "P(C)(C)C", "O=C([O-])[O-]", "CCO"),
        ("[Pd]", "P(CC)(CC)CC", "O=C([O-])[O-]", "CCO"),
        ("[Pd]", "P(CC)(CC)CC", "O=P([O-])([O-])O", "O1CCOCC1"),
        ("[Ni]", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "O=P([O-])([O-])O", "CCO"),
        ("[Ni]", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "O=P([O-])([O-])O", "CCO"),
    ]
    rows = []
    for index, (catalyst, ligand, base, solvent) in enumerate(conditions):
        rows.append(
            {
                "source_row_id": f"row-{index}",
                "canonical_reaction_key": f"reaction-{index}",
                "canonical_substrate_key": f"substrate-{index % 3}",
                "canonicalization_version": CANONICALIZATION_VERSION,
                "canonical_catalyst_smiles": catalyst,
                "canonical_ligand_smiles": ligand,
                "canonical_base_smiles": base,
                "canonical_solvent_or_additive_smiles": solvent,
            }
        )
    return pd.DataFrame(rows)


def test_condition_combination_key_is_role_explicit_and_column_order_independent() -> None:
    row = _canonical_fixture().iloc[0].to_dict()
    reverse = dict(reversed(list(row.items())))

    first = canonical_condition_combination_key(row)
    second = canonical_condition_combination_key(reverse)

    assert first == second
    payload = json.loads(first)
    assert payload["schema"] == CONDITION_COMBINATION_KEY_SCHEMA_VERSION
    assert [
        payload[role]
        for role in ("catalyst", "ligand", "base", "solvent_or_additive")
    ] == ["[Pd]", "P(C)(C)C", "O=C([O-])[O-]", "CCO"]


def test_condition_combination_holdout_excludes_the_exact_four_role_context() -> None:
    keyed = add_condition_combination_keys(_canonical_fixture())
    plan = build_condition_ood_plan(
        keyed,
        target="condition_combination",
        criteria=FoldSupportCriteria(min_train_groups=2),
    )

    assert plan.included_folds
    for fold in plan.included_folds:
        train = keyed.loc[keyed["source_row_id"].isin(fold.train_source_ids)]
        test = keyed.loc[keyed["source_row_id"].isin(fold.test_source_ids)]
        assert set(test["condition_combination_key"]) == {fold.heldout_group}
        assert fold.heldout_group not in set(train["condition_combination_key"])
        assert fold.group_overlap_count == 0
        assert fold.canonical_reaction_key_overlap_count == 0


@pytest.mark.parametrize(
    ("target", "column"),
    [
        ("ligand", "canonical_ligand_smiles"),
        ("base", "canonical_base_smiles"),
    ],
)
def test_ligand_and_base_logo_hold_out_the_declared_canonical_identity(
    target: str,
    column: str,
) -> None:
    frame = _canonical_fixture()
    plan = build_condition_ood_plan(
        frame,
        target=target,
        criteria=FoldSupportCriteria(
            min_test_samples=1,
            min_train_samples=2,
            min_train_groups=1,
        ),
    )

    assert plan.included_folds
    for fold in plan.included_folds:
        train = frame.loc[frame["source_row_id"].isin(fold.train_source_ids)]
        test = frame.loc[frame["source_row_id"].isin(fold.test_source_ids)]
        assert set(test[column]) == {fold.heldout_group}
        assert fold.heldout_group not in set(train[column])
        assert fold.group_overlap_count == 0


def test_unsupported_small_groups_are_excluded_with_exact_reasons() -> None:
    frame = _canonical_fixture()
    criteria = FoldSupportCriteria(
        min_test_samples=2,
        min_train_samples=5,
        min_train_groups=2,
        min_test_substrates=2,
        min_train_substrates=3,
    )

    plan = build_condition_ood_plan(
        frame,
        target="condition_combination",
        criteria=criteria,
    )
    singleton = next(
        fold
        for fold in plan.excluded_folds
        if fold.test_size == 1
    )

    assert singleton.exclusion_reason == (
        "test_samples_below_minimum(observed=1,required=2); "
        "test_substrates_below_minimum(observed=1,required=2)"
    )
    assert not singleton.included
    assert plan.audit_frame.loc[
        plan.audit_frame["fold_index"].eq(singleton.fold_index),
        "exclusion_reason",
    ].item() == singleton.exclusion_reason


def test_keys_reject_raw_or_missing_condition_identity() -> None:
    row = _canonical_fixture().iloc[0].to_dict()
    row["canonical_ligand_smiles"] = None

    with pytest.raises(ValueError, match="canonical_ligand_smiles"):
        canonical_condition_combination_key(row)
