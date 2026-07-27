"""Scientific contracts for hidden-measured condition probes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

import bh_augmentation.hidden_condition_probes as probes_module
from bh_augmentation.data.canonicalize_roles import (
    canonicalize_reaction_roles_dataframe,
)
from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_to_record
from bh_augmentation.hidden_condition_probes import (
    build_hidden_condition_probes,
    build_hidden_condition_probes_from_context,
    prepare_hidden_condition_probe_context,
)

pytest.importorskip("rdkit")


def _canonical_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    target = ReactionRoles(
        reactant_1="Brc1ccccc1",
        reactant_2="CN",
        catalyst="[Pd]",
        ligand="P(C)(C)C",
        base="[OH-].[Na+]",
        solvent_or_additive="CCO",
        product="CNc1ccccc1",
    )
    rows = [
        target,
        ReactionRoles(
            **{
                **target.__dict__,
                "ligand": "P(CC)(CC)CC",
            }
        ),
        ReactionRoles(
            **{
                **target.__dict__,
                "ligand": "P(CC)(CC)CC",
                "base": "[OH-].[K+]",
            }
        ),
        ReactionRoles(
            reactant_1="Clc1ccccc1",
            reactant_2="CN",
            catalyst="[Pd]",
            ligand=target.ligand,
            base=target.base,
            solvent_or_additive="CCN",
            product="CNc1ccccc1",
        ),
        ReactionRoles(
            reactant_1="Brc1ccccn1",
            reactant_2="CN",
            catalyst="[Pd]",
            ligand=target.ligand,
            base=target.base,
            solvent_or_additive="CC#N",
            product="CNc1ccccn1",
        ),
    ]
    raw = pd.DataFrame([reaction_roles_to_record(roles) for roles in rows])
    canonical = canonicalize_reaction_roles_dataframe(
        raw,
        source_file_hash="fixture-hash",
    )
    hidden = canonical.iloc[[0]].drop(columns=["yield"], errors="ignore").copy()
    train = canonical.iloc[1:].copy()
    return train, hidden


@pytest.mark.parametrize("strategy", ["random", "nearest_substrate"])
@pytest.mark.parametrize(
    ("requested_roles", "expected_mode"),
    [
        (("ligand",), "ligand_only"),
        (("ligand", "base"), "ligand_base"),
    ],
)
def test_supported_probes_exactly_recreate_hidden_identity(
    strategy: str,
    requested_roles: tuple[str, ...],
    expected_mode: str,
) -> None:
    train, hidden = _canonical_fixture()

    result = build_hidden_condition_probes(
        train,
        hidden,
        requested_roles=requested_roles,
        donor_strategy=strategy,
        seed=11,
        similarity_n_bits=64,
    )

    assert len(result.probes) == 1
    assert result.exclusions.empty
    probe = result.probes.iloc[0]
    assert probe["role_transfer_mode"] == expected_mode
    assert probe["donor_strategy"] == strategy
    assert probe["canonical_reaction_key"] == hidden.iloc[0]["canonical_reaction_key"]
    assert probe["canonical_reaction_hash"] == hidden.iloc[0]["canonical_reaction_hash"]
    assert bool(probe["canonical_identity_exact_hidden_match"])
    assert probe["source_row_id"] in set(train["source_row_id"])
    assert probe["donor_row_id"] in set(train["source_row_id"])
    assert bool(probe["source_train_only"])
    assert bool(probe["donor_train_only"])
    assert not bool(probe["eligible_for_training"])
    assert result.metadata["hidden_outcomes_accessible"] is False
    assert result.metadata["eligible_for_training"] is False
    with pytest.raises(FrozenInstanceError):
        result.seed = 3  # type: ignore[misc]


def test_random_strategy_is_seeded_and_reproducible() -> None:
    train, hidden = _canonical_fixture()
    first = build_hidden_condition_probes(
        train.sample(frac=1.0, random_state=1),
        hidden,
        requested_roles=("ligand", "base"),
        donor_strategy="random",
        seed=7,
    )
    second = build_hidden_condition_probes(
        train.sample(frac=1.0, random_state=2),
        hidden,
        requested_roles=("base", "ligand"),
        donor_strategy="random",
        seed=7,
    )

    assert first.requested_roles == second.requested_roles == ("ligand", "base")
    assert first.probes["source_row_id"].tolist() == second.probes["source_row_id"].tolist()
    assert first.probes["donor_row_id"].tolist() == second.probes["donor_row_id"].tolist()


@pytest.mark.parametrize("strategy", ["random", "nearest_substrate"])
@pytest.mark.parametrize("requested_roles", [("ligand",), ("ligand", "base")])
def test_prepared_context_is_equivalent_to_frame_api(
    strategy: str,
    requested_roles: tuple[str, ...],
) -> None:
    train, hidden = _canonical_fixture()
    direct = build_hidden_condition_probes(
        train,
        hidden,
        requested_roles=requested_roles,
        donor_strategy=strategy,
        seed=17,
        similarity_n_bits=64,
        similarity_radius=2,
    )
    context = prepare_hidden_condition_probe_context(
        train,
        hidden,
        similarity_n_bits=64,
        similarity_radius=2,
    )
    prepared = build_hidden_condition_probes_from_context(
        context,
        requested_roles=tuple(reversed(requested_roles)),
        donor_strategy=strategy,
        seed=17,
    )

    assert prepared.held_condition_key == direct.held_condition_key
    assert prepared.requested_roles == direct.requested_roles
    assert prepared.metadata == direct.metadata
    pd.testing.assert_frame_equal(prepared.probes, direct.probes)
    pd.testing.assert_frame_equal(prepared.exclusions, direct.exclusions)


def test_prepared_context_reuses_canonical_rows_and_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, hidden = _canonical_fixture()
    canonicalize_calls = 0
    fingerprint_calls = 0
    original_canonicalize = probes_module.canonicalize_synthetic_roles
    original_fingerprint = probes_module._substrate_fingerprint

    def counted_canonicalize(roles: ReactionRoles) -> object:
        nonlocal canonicalize_calls
        canonicalize_calls += 1
        return original_canonicalize(roles)

    def counted_fingerprint(
        roles: ReactionRoles,
        *,
        n_bits: int,
        radius: int,
    ) -> object:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return original_fingerprint(roles, n_bits=n_bits, radius=radius)

    monkeypatch.setattr(
        probes_module, "canonicalize_synthetic_roles", counted_canonicalize
    )
    monkeypatch.setattr(
        probes_module, "_substrate_fingerprint", counted_fingerprint
    )
    context = prepare_hidden_condition_probe_context(
        train,
        hidden,
        similarity_n_bits=64,
    )
    preparation_call_count = len(train) + len(hidden)
    assert canonicalize_calls == preparation_call_count
    assert fingerprint_calls == preparation_call_count

    role_sets = (
        ("ligand",),
        ("base",),
        ("solvent_or_additive",),
        ("ligand", "base"),
        ("ligand", "solvent_or_additive"),
        ("base", "solvent_or_additive"),
        ("catalyst",),
        ("ligand", "base", "solvent_or_additive"),
        ("catalyst", "ligand", "base", "solvent_or_additive"),
    )
    generated_probe_count = 0
    donor_choices: list[tuple[str, str]] = []
    for roles in role_sets:
        for strategy in ("random", "nearest_substrate"):
            result = build_hidden_condition_probes_from_context(
                context,
                requested_roles=roles,
                donor_strategy=strategy,
                seed=23,
            )
            generated_probe_count += len(result.probes)
            donor_choices.extend(
                result.probes[["source_row_id", "donor_row_id"]].itertuples(
                    index=False, name=None
                )
            )

    assert donor_choices
    assert fingerprint_calls == preparation_call_count
    assert canonicalize_calls == preparation_call_count + generated_probe_count

    repeated = build_hidden_condition_probes_from_context(
        context,
        requested_roles=("base", "ligand"),
        donor_strategy="random",
        seed=23,
    )
    first = build_hidden_condition_probes_from_context(
        context,
        requested_roles=("ligand", "base"),
        donor_strategy="random",
        seed=23,
    )
    pd.testing.assert_frame_equal(repeated.probes, first.probes)


@pytest.mark.parametrize(
    ("requested_roles", "reason"),
    [
        (("catalyst",), "unsupported_invariant_catalyst"),
        (
            ("ligand", "base", "solvent_or_additive"),
            "unsupported_triple_variable_role_transfer_global_holdout",
        ),
        (
            ("catalyst", "ligand", "base", "solvent_or_additive"),
            "unsupported_full_condition_transfer_global_holdout",
        ),
    ],
)
def test_unsupported_transfers_are_explicit_exclusions(
    requested_roles: tuple[str, ...],
    reason: str,
) -> None:
    train, hidden = _canonical_fixture()

    result = build_hidden_condition_probes(
        train,
        hidden,
        requested_roles=requested_roles,
        donor_strategy="random",
        seed=0,
    )

    assert result.probes.empty
    assert result.exclusions["exclusion_reason"].tolist() == [reason]
    assert result.metadata["n_exclusions"] == 1


def test_hidden_yield_is_architecturally_inaccessible() -> None:
    train, hidden = _canonical_fixture()
    hidden["yield"] = 99.0

    with pytest.raises(ValueError, match="must not expose hidden yield"):
        build_hidden_condition_probes(
            train,
            hidden,
            requested_roles=("ligand",),
            donor_strategy="random",
            seed=0,
        )


def test_global_condition_holdout_is_required() -> None:
    train, hidden = _canonical_fixture()
    leaked = pd.concat([train, hidden], ignore_index=True)

    with pytest.raises(ValueError, match="Global condition holdout violated"):
        build_hidden_condition_probes(
            leaked,
            hidden,
            requested_roles=("ligand",),
            donor_strategy="nearest_substrate",
            seed=0,
        )


def test_tampered_canonical_condition_identity_is_rejected() -> None:
    train, hidden = _canonical_fixture()
    hidden.loc[:, "canonical_condition_key"] = "tampered-held-key"

    with pytest.raises(ValueError, match="inconsistent canonical condition identity"):
        build_hidden_condition_probes(
            train,
            hidden,
            requested_roles=("ligand",),
            donor_strategy="random",
            seed=0,
        )


def test_missing_exact_source_is_recorded_not_fabricated() -> None:
    train, hidden = _canonical_fixture()
    target_substrate = hidden.iloc[0]["canonical_substrate_key"]
    train = train.loc[train["canonical_substrate_key"].ne(target_substrate)]

    result = build_hidden_condition_probes(
        train,
        hidden,
        requested_roles=("ligand", "base"),
        donor_strategy="random",
        seed=0,
    )

    assert result.probes.empty
    assert result.exclusions["exclusion_reason"].tolist() == [
        "no_exact_train_source"
    ]
