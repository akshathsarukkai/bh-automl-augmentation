"""Suzuki-Miyaura fixture: canonical identity, grouped splits, typed transfer.

All Suzuki data used here is the SYNTHETIC fixture from ``tests/suzuki_fixture.py``.
No empirical Suzuki-Miyaura claim is made or implied. The point of these tests is
that the *general* infrastructure -- canonical identity, grouped splits, transfer
eligibility, and the evaluation protocol -- is reused unchanged, and only the
adapter is family-specific.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from bh_augmentation.augmentation.family_condition_transfer import (
    assert_condition_transfer_invariants,
    build_condition_transfer_candidates,
)
from bh_augmentation.data import canonical_splits
from bh_augmentation.data.canonical_splits import (
    build_grouped_outer_assignments,
    build_nested_low_data_assignments,
)
from bh_augmentation.data.reaction_family import (
    SUZUKI_MIYAURA_ROLE_NAMES,
    suzuki_miyaura_eligibility,
)
from bh_augmentation.evaluation import policy_protocol
from suzuki_fixture import suzuki_fixture_frame, write_suzuki_fixture_csv


@pytest.fixture(scope="module")
def fixture_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("suzuki_fixture")
    return write_suzuki_fixture_csv(directory)


@pytest.fixture(scope="module")
def adapter(fixture_paths):
    from suzuki_fixture import suzuki_fixture_adapter

    return suzuki_fixture_adapter(fixture_paths)


@pytest.fixture(scope="module")
def canonical(adapter):
    return adapter.canonicalize_dataframe(
        suzuki_fixture_frame(),
        source_file_hash="c" * 64,
        source_row_positions=range(len(suzuki_fixture_frame())),
    )


# ----------------------------------------------------------------------
# Canonicalization
# ----------------------------------------------------------------------
def test_every_fixture_row_parses_and_is_family_eligible(canonical) -> None:
    assert canonical["all_required_roles_parse_valid"].all()
    assert canonical["family_eligible"].all()
    assert canonical["family_eligibility_reason"].isna().all()
    assert canonical["canonical_reaction_key"].notna().all()


def test_reaction_keys_are_family_scoped_and_deterministic(adapter, canonical) -> None:
    keys = canonical["canonical_reaction_key"].tolist()
    assert all("suzuki-seven-role-v1" in key for key in keys)
    for role in SUZUKI_MIYAURA_ROLE_NAMES:
        assert all(f'"{role}"' in key for key in keys)

    repeated = adapter.canonicalize_dataframe(
        suzuki_fixture_frame(),
        source_file_hash="c" * 64,
        source_row_positions=range(len(canonical)),
    )
    assert repeated["canonical_reaction_key"].tolist() == keys
    assert repeated["canonical_reaction_hash"].tolist() == (
        canonical["canonical_reaction_hash"].tolist()
    )


def test_equivalent_smiles_writings_collapse_to_one_canonical_reaction(canonical) -> None:
    equivalent = canonical.loc[
        canonical["reaction_id"].eq("synthetic_suzuki_equivalent_smiles")
    ]
    replicate = canonical.loc[
        canonical["reaction_id"].eq("synthetic_suzuki_replicate_0")
    ]
    assert len(equivalent) == 1
    assert len(replicate) == 1
    assert (
        equivalent["smiles_organoboron"].iloc[0] != replicate["smiles_organoboron"].iloc[0]
    )
    assert (
        equivalent["canonical_organoboron_smiles"].iloc[0]
        == replicate["canonical_organoboron_smiles"].iloc[0]
    )
    assert (
        equivalent["canonical_reaction_key"].iloc[0]
        == replicate["canonical_reaction_key"].iloc[0]
    )


# ----------------------------------------------------------------------
# Family eligibility hook
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"organoboron": "c1ccccc1"}, "organoboron_contains_no_boron"),
        ({"organohalide": "Fc1ccccc1"}, "organohalide_has_no_leaving_group"),
        ({"catalyst": "[Cu]"}, "catalyst_is_not_palladium_or_nickel"),
        (
            {"product": "Brc1ccccc1"},
            "product_identical_to_substrate",
        ),
    ],
)
def test_suzuki_eligibility_rejects_non_suzuki_rows(changes, reason: str) -> None:
    roles = {
        "organohalide": "Brc1ccccc1",
        "organoboron": "OB(O)c1ccccc1",
        "catalyst": "[Pd]",
        "ligand": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
        "base": "O=C([O-])[O-].[K+].[K+]",
        "solvent_or_additive": "C1CCOC1",
        "product": "c1ccc(-c2ccccc2)cc1",
    }
    assert suzuki_miyaura_eligibility(roles) is None
    assert suzuki_miyaura_eligibility({**roles, **changes}) == reason


def test_triflate_pseudohalide_is_an_eligible_electrophile() -> None:
    roles = {
        "organohalide": "O=S(=O)(Oc1ccccc1)C(F)(F)F",
        "organoboron": "OB(O)c1ccccc1",
        "catalyst": "[Pd]",
        "ligand": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
        "base": "O=C([O-])[O-].[K+].[K+]",
        "solvent_or_additive": "C1CCOC1",
        "product": "c1ccc(-c2ccccc2)cc1",
    }
    assert suzuki_miyaura_eligibility(roles) is None


def test_eligibility_reason_is_recorded_rather_than_silently_dropping_rows(adapter) -> None:
    frame = suzuki_fixture_frame(n_rows=8)
    frame.loc[0, "smiles_organoboron"] = "c1ccccc1"
    canonical = adapter.canonicalize_dataframe(
        frame,
        source_file_hash="d" * 64,
        source_row_positions=range(len(frame)),
    )
    assert len(canonical) == len(frame)
    assert canonical.loc[0, "family_eligibility_reason"] == "organoboron_contains_no_boron"
    assert not bool(canonical.loc[0, "family_eligible"])
    assert bool(canonical.loc[0, "all_required_roles_parse_valid"])


# ----------------------------------------------------------------------
# Grouped splits: the EXISTING splitter, unchanged
# ----------------------------------------------------------------------
def test_grouped_splits_reuse_the_existing_buchwald_hartwig_splitter() -> None:
    assert build_grouped_outer_assignments is canonical_splits.build_grouped_outer_assignments
    source = inspect.getsource(canonical_splits.build_grouped_outer_assignments)
    for token in ("reactant_1", "amine", "aryl_halide", "buchwald", "suzuki"):
        assert token not in source.lower()


def test_identical_canonical_reactions_never_cross_outer_splits(canonical) -> None:
    outer = build_grouped_outer_assignments(
        canonical,
        seed=0,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )
    groups = {
        split: set(outer.loc[outer["outer_split"].eq(split), "canonical_reaction_key"])
        for split in ("train", "valid", "test")
    }
    assert not groups["train"] & groups["valid"]
    assert not groups["train"] & groups["test"]
    assert not groups["valid"] & groups["test"]
    assert outer["source_row_id"].nunique() == len(outer)

    duplicated_key = (
        canonical.groupby("canonical_reaction_key").size().sort_values().index[-1]
    )
    replicate_splits = set(
        outer.loc[outer["canonical_reaction_key"].eq(duplicated_key), "outer_split"]
    )
    assert len(replicate_splits) == 1

    low = build_nested_low_data_assignments(outer, train_fractions=[0.5, 1.0])
    small = set(
        low.loc[
            low["train_fraction"].eq(0.5) & low["included_in_training_subset"],
            "source_row_id",
        ]
    )
    large = set(
        low.loc[
            low["train_fraction"].eq(1.0) & low["included_in_training_subset"],
            "source_row_id",
        ]
    )
    assert small <= large


# ----------------------------------------------------------------------
# Typed condition transfer
# ----------------------------------------------------------------------
def _training_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    outer = build_grouped_outer_assignments(
        canonical,
        seed=0,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )
    frame = canonical.copy()
    frame["outer_split"] = frame["source_row_id"].map(
        outer.set_index("source_row_id")["outer_split"]
    )
    return frame.loc[frame["outer_split"].eq("train")].reset_index(drop=True)


def test_transfer_changes_only_condition_roles_and_preserves_substrate_identity(
    adapter,
    canonical,
) -> None:
    train = _training_frame(canonical)
    candidates = build_condition_transfer_candidates(
        train,
        adapter,
        seed=7,
        max_candidates_per_source=3,
        measured_canonical_keys=adapter.measured_canonical_keys(canonical),
    )
    assert_condition_transfer_invariants(candidates, adapter)
    accepted = candidates.loc[candidates["accepted"].astype(bool)]
    assert not accepted.empty

    allowed = set(adapter.transferable_roles)
    for _, row in accepted.iterrows():
        changed = {role for role in str(row["changed_roles"]).split("|") if role}
        assert changed
        assert changed <= allowed
        assert "organohalide" not in changed
        assert "organoboron" not in changed
        assert "product" not in changed
        assert row["canonical_substrate_key"] == row["source_substrate_key"]
        assert row["canonical_product_key"] == row["source_product_key"]


def test_candidate_duplicating_a_measured_canonical_reaction_is_rejected(
    adapter,
    canonical,
) -> None:
    # Two measured rows sharing substrates, catalyst and product, differing only
    # in the transferable condition roles. Transferring either row's conditions
    # onto the other exactly reconstructs a reaction that was already measured.
    frame = pd.DataFrame(
        [
            {
                "reaction_id": "measured_a",
                "smiles_organohalide": "Brc1ccccc1",
                "smiles_organoboron": "OB(O)c1ccccc1",
                "smiles_catalyst": "[Pd]",
                "smiles_ligand": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                "smiles_base": "O=C([O-])[O-].[K+].[K+]",
                "smiles_solvent_or_additive": "C1CCOC1",
                "smiles_product": "c1ccc(-c2ccccc2)cc1",
                "yield": 41.0,
            },
            {
                "reaction_id": "measured_b",
                "smiles_organohalide": "Brc1ccccc1",
                "smiles_organoboron": "OB(O)c1ccccc1",
                "smiles_catalyst": "[Pd]",
                "smiles_ligand": "CC(C)P(C(C)C)C(C)C",
                "smiles_base": "O=P([O-])([O-])[O-].[K+].[K+].[K+]",
                "smiles_solvent_or_additive": "CC(C)O",
                "smiles_product": "c1ccc(-c2ccccc2)cc1",
                "yield": 63.0,
            },
        ]
    )
    measured = adapter.canonicalize_dataframe(
        frame,
        source_file_hash="e" * 64,
        source_row_positions=range(len(frame)),
    )
    candidates = build_condition_transfer_candidates(
        measured,
        adapter,
        seed=0,
        max_candidates_per_source=1,
        measured_canonical_keys=adapter.measured_canonical_keys(measured),
    )
    assert len(candidates) == 2
    assert candidates["rejection_reason"].eq("already_measured").all()
    assert not candidates["accepted"].astype(bool).any()

    # Chemical identity comes from canonical keys, not from feature equality.
    train = _training_frame(canonical)
    all_measured = adapter.measured_canonical_keys(canonical)
    natural = build_condition_transfer_candidates(
        train,
        adapter,
        seed=3,
        max_candidates_per_source=4,
        measured_canonical_keys=all_measured,
    )
    accepted_keys = natural.loc[
        natural["accepted"].astype(bool), "canonical_reaction_key"
    ]
    assert not accepted_keys.isin(all_measured).any()


def test_transfer_refuses_to_read_validation_or_test_rows(adapter, canonical) -> None:
    outer = build_grouped_outer_assignments(
        canonical,
        seed=0,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )
    frame = canonical.copy()
    frame["outer_split"] = frame["source_row_id"].map(
        outer.set_index("source_row_id")["outer_split"]
    )
    with pytest.raises(ValueError, match="only read training rows"):
        build_condition_transfer_candidates(frame, adapter, seed=0)


def test_transfer_rejects_roles_outside_the_declared_transferable_set(
    adapter,
    canonical,
) -> None:
    train = _training_frame(canonical)
    with pytest.raises(ValueError, match="non-empty subset"):
        build_condition_transfer_candidates(
            train,
            adapter,
            transferable_roles=["organoboron"],
        )


# ----------------------------------------------------------------------
# Reuse: the protocol layer knows nothing about any reaction family
# ----------------------------------------------------------------------
def test_policy_protocol_contains_no_family_specific_vocabulary() -> None:
    source = inspect.getsource(policy_protocol).lower()
    for token in (
        "reactant_1",
        "organohalide",
        "organoboron",
        "aryl halide",
        "amine",
        "suzuki",
    ):
        assert token not in source
