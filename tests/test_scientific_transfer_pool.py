"""Focused contracts for immutable scientific transfer pools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation.scientific_transfer_pool as pool_module
from bh_augmentation.augmentation.condition_transfer import ConditionTransferConfig
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
)
from bh_augmentation.augmentation.scientific_transfer_pool import (
    build_anonymous_transfer_pool,
    build_strict_context_matched_typed_transfer_pool,
)
from bh_augmentation.augmentation.synthetic_identity import measured_canonical_keys
from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_dataframe
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata

pytest.importorskip("rdkit")


def _training_bundle() -> dict[str, object]:
    contexts = (
        ("Brc1ccccc1", "NCC", "NCCc1ccccc1"),
        ("Brc1ccncc1", "NC", "CNc1ccncc1"),
    )
    conditions = (
        ("[Pd]", "P(C)(C)C", "[Na+].[OH-]", "O"),
        ("[Ni]", "P(CC)(CC)CC", "[K+].[OH-]", "CO"),
    )
    roles = [
        ReactionRoles(
            reactant_1=reactant_1,
            reactant_2=reactant_2,
            catalyst=catalyst,
            ligand=ligand,
            base=base,
            solvent_or_additive=solvent,
            product=product,
        )
        for reactant_1, reactant_2, product in contexts
        for catalyst, ligand, base, solvent in conditions
    ]
    frame = reaction_roles_dataframe(roles, yields=[20.0, 40.0, 60.0, 80.0])
    frame.insert(0, "source_row_id", [f"train-{index}" for index in range(len(frame))])
    feature_config = {
        "kind": "bh_role_separated",
        "n_bits": 32,
        "radius": 2,
        "fingerprint_backend": "rdkit",
        "categorical_columns": [],
    }
    X, y, names, metadata = build_feature_matrix_with_metadata(frame, feature_config)
    return {
        "training_frame": frame,
        "training_features": X,
        "training_labels": y,
        "feature_config": feature_config,
        "feature_names": names,
        "feature_metadata": metadata,
        "global_measured_identity_keys": measured_canonical_keys(frame),
    }


def _anonymous_config(**overrides: object) -> ConditionTransferConfig:
    values: dict[str, object] = {
        "donor_strategy": "random",
        "synthetic_multiplier": 0.5,
        "n_neighbors": 3,
        "label_strategy": "average_label",
        "teacher_models": ["ridge"],
        "max_teacher_std": None,
        "min_similarity": None,
        "high_yield_threshold": 70.0,
        "clip_y_min": 0.0,
        "clip_y_max": 100.0,
        "candidates_per_real": 2,
        "random_state": 7,
        "donor_similarity_n_bits": 32,
        "donor_similarity_radius": 2,
        "donor_similarity_backend": "rdkit",
        "role_change_requirement": "all",
        "fallback_policy": "reject",
        "max_candidates_per_source": 2,
        "requested_roles": ("ligand", "base", "solvent_or_additive"),
    }
    values.update(overrides)
    return ConditionTransferConfig(**values)


def _typed_config(**overrides: object) -> RoleAwareConditionTransferConfig:
    values: dict[str, object] = {
        "role_transfer_mode": "ligand_base",
        "donor_strategy": "same_substrate_different_role",
        "label_strategy": "average_source_donor_label",
        "synthetic_multiplier": 0.5,
        "max_candidates_per_source": 2,
        "teacher_models": ["ridge"],
        "max_teacher_std": None,
        "min_similarity": None,
        "random_state": 7,
        "donor_similarity_n_bits": 32,
        "donor_similarity_radius": 2,
        "donor_similarity_backend": "rdkit",
        "role_change_requirement": "all",
        "fallback_policy": "reject",
    }
    values.update(overrides)
    return RoleAwareConditionTransferConfig(**values)


def test_anonymous_pool_is_canonical_bounded_and_defensively_immutable() -> None:
    bundle = _training_bundle()
    pool = build_anonymous_transfer_pool(**bundle, config=_anonymous_config())

    assert pool.pool_kind == "anonymous"
    assert pool.selected_count > 0
    assert pool.accepted_count >= pool.selected_count
    assert pool.features.shape == (
        pool.selected_count,
        bundle["feature_metadata"].total_width,
    )
    assert set(pool.audit["source_row_id"]) <= set(
        bundle["training_frame"]["source_row_id"]
    )
    assert set(pool.audit["donor_row_id"]) <= set(
        bundle["training_frame"]["source_row_id"]
    )
    accepted = pool.accepted_audit
    assert accepted["chemical_parse_valid"].all()
    assert accepted["role_change_valid"].all()
    assert not (
        set(accepted["canonical_reaction_key"])
        & set(bundle["global_measured_identity_keys"])
    )
    assert not accepted["canonical_reaction_key"].duplicated().any()
    assert not accepted["feature_hash"].duplicated().any()

    rows = pool.rows
    features = pool.features
    labels = pool.labels
    audit = pool.audit
    rows.loc[:, "yield"] = -999.0
    features[:] = -999.0
    labels[:] = -999.0
    audit.loc[:, "source_row_id"] = "outside"
    assert not pool.rows["yield"].eq(-999.0).any()
    assert not np.equal(pool.features, -999.0).any()
    assert not np.equal(pool.labels, -999.0).any()
    assert "outside" not in set(pool.audit["source_row_id"])


def test_strict_typed_pool_changes_all_requested_roles_in_exact_context() -> None:
    bundle = _training_bundle()
    pool = build_strict_context_matched_typed_transfer_pool(
        **bundle, config=_typed_config()
    )

    assert pool.pool_kind == "strict_context_matched_typed"
    assert pool.selected_count > 0
    accepted = pool.accepted_audit
    assert set(accepted["requested_roles"]) == {"ligand|base"}
    assert set(accepted["actual_changed_roles"]) == {"ligand|base"}
    assert not accepted["unchanged_requested_roles"].astype(bool).any()
    assert not accepted["unexpected_changed_roles"].astype(bool).any()
    assert set(accepted["change_mask"]) == {"0001100"}
    assert pool.generator_metadata["donor_strategy"] == (
        "same_substrate_different_role"
    )
    assert set(accepted["donor_fallback_level"]) == {
        "strict_same_reactant_key"
    }
    assert not accepted["fallback_used"].astype(bool).any()


def test_pool_regeneration_has_identical_scientific_hashes_and_content() -> None:
    bundle = _training_bundle()
    first = build_anonymous_transfer_pool(**bundle, config=_anonymous_config())
    second = build_anonymous_transfer_pool(**bundle, config=_anonymous_config())

    assert first.pool_hash == second.pool_hash
    assert first.candidate_audit_hash == second.candidate_audit_hash
    assert first.accepted_identity_hash == second.accepted_identity_hash
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)
    pd.testing.assert_frame_equal(first.rows, second.rows)
    pd.testing.assert_frame_equal(first.audit, second.audit)


@pytest.mark.parametrize("kind", ["anonymous", "typed"])
def test_explicit_zero_candidate_pool_is_valid_and_does_not_fallback(kind: str) -> None:
    bundle = _training_bundle()
    if kind == "anonymous":
        pool = build_anonymous_transfer_pool(
            **bundle,
            config=_anonymous_config(synthetic_multiplier=0.0),
        )
    else:
        pool = build_strict_context_matched_typed_transfer_pool(
            **bundle,
            config=_typed_config(synthetic_multiplier=0.0),
        )

    assert pool.candidate_count == 0
    assert pool.accepted_count == 0
    assert pool.selected_count == 0
    assert pool.features.shape == (
        0,
        bundle["feature_metadata"].total_width,
    )
    assert pool.generator_metadata["fallback_policy"] == "reject"


@pytest.mark.parametrize(
    ("builder", "config", "message"),
    [
        (
            build_anonymous_transfer_pool,
            _anonymous_config(role_change_requirement="any"),
            "all requested roles",
        ),
        (
            build_anonymous_transfer_pool,
            _anonymous_config(fallback_policy="random"),
            "fallback_policy='reject'",
        ),
        (
            build_anonymous_transfer_pool,
            _anonymous_config(donor_similarity_backend="hash"),
            "RDKit donor similarity",
        ),
        (
            build_strict_context_matched_typed_transfer_pool,
            _typed_config(donor_strategy="random"),
            "Strict context-matched",
        ),
    ],
)
def test_pool_rejects_unsafe_or_mislabeled_config(
    builder: object,
    config: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        builder(**_training_bundle(), config=config)


def test_pool_rejects_poisoned_features_and_incomplete_global_identity_contract() -> None:
    poisoned = _training_bundle()
    poisoned["training_features"] = np.asarray(
        poisoned["training_features"]
    ).copy()
    poisoned["training_features"][0, 0] = 1.0 - poisoned["training_features"][0, 0]
    with pytest.raises(ValueError, match="do not match canonical identities"):
        build_anonymous_transfer_pool(**poisoned, config=_anonymous_config())

    incomplete = _training_bundle()
    incomplete["global_measured_identity_keys"] = {
        next(iter(incomplete["global_measured_identity_keys"]))
    }
    with pytest.raises(ValueError, match="omits training reactions"):
        build_anonymous_transfer_pool(**incomplete, config=_anonymous_config())


def test_adapter_detects_generator_parent_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _training_bundle()
    production = pool_module.generate_condition_transfer_examples

    def leaking_generator(*args: object, **kwargs: object) -> dict[str, object]:
        result = production(*args, **kwargs)
        result["candidate_df"] = result["candidate_df"].copy()
        result["candidate_df"].loc[0, "source_row_id"] = "validation-row"
        return result

    monkeypatch.setattr(
        pool_module, "generate_condition_transfer_examples", leaking_generator
    )
    with pytest.raises(ValueError, match="outside provided training IDs"):
        build_anonymous_transfer_pool(**bundle, config=_anonymous_config())
