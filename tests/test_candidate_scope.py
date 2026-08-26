"""Adversarial tests for the typed candidate-eligibility scope.

These tests are written to break the contract, not to demonstrate it.  Each one
either mutates information the protocol says must not matter and proves the
output is byte-identical, or supplies information the protocol says must never
be reachable and proves the code refuses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.augmentation.candidate_scope import (
    ALREADY_MEASURED_REASON,
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    OBSERVED_IN_LABELED_TRAIN_REASON,
    OBSERVED_ONLY_LOW_DATA,
    QUARANTINED_REASON,
    CandidateScopePolicy,
    CandidateScopeViolation,
    globally_unmeasured_scope,
    legacy_scope_from_measured_identity_keys,
    observed_only_scope,
    resolve_generation_scope,
    resolved_candidate_scope_mode,
)

# --------------------------------------------------------------------- policy


def test_observed_only_scope_structurally_cannot_carry_global_identities() -> None:
    """The low-data rule must be unable to consult complete-dataset membership.

    This is enforced by the type, not by a convention: there is no code path
    that can put a global key set into an observed-only policy.
    """
    with pytest.raises(CandidateScopeViolation, match="must not carry global"):
        CandidateScopePolicy(
            mode=OBSERVED_ONLY_LOW_DATA,
            observed_identity_keys=("observed",),
            global_identity_keys=("hidden-outer-train",),
        )


def test_prospective_scope_requires_the_complete_universe() -> None:
    """A novelty claim without the measured universe is not a novelty claim."""
    with pytest.raises(CandidateScopeViolation, match="requires the complete measured"):
        CandidateScopePolicy(
            mode=GLOBALLY_UNMEASURED_PROSPECTIVE,
            observed_identity_keys=("observed",),
        )


def test_full_dataset_membership_cannot_suppress_an_unobserved_candidate() -> None:
    """A candidate measured elsewhere but unobserved stays eligible under low data."""
    scope = observed_only_scope(labeled_train_identity_keys=["observed"])
    assert scope.rejection_reason_for("hidden-outer-train") is None
    assert "hidden-outer-train" not in scope.rejection_identity_keys()


def test_identity_in_labeled_training_is_rejected() -> None:
    """Chemistry the simulated learner already holds is never a useful target."""
    scope = observed_only_scope(labeled_train_identity_keys=["observed"])
    assert scope.rejection_reason_for("observed") == OBSERVED_IN_LABELED_TRAIN_REASON


def test_validation_and_test_identities_are_quarantined_not_treated_as_observed() -> None:
    """Held-out identities are excluded from training without being called observed."""
    scope = observed_only_scope(
        labeled_train_identity_keys=["observed"],
        quarantine_identity_keys=["valid-key", "test-key"],
    )
    assert scope.rejection_reason_for("valid-key") == QUARANTINED_REASON
    assert scope.rejection_reason_for("test-key") == QUARANTINED_REASON
    assert "valid-key" not in scope.observed_identity_keys


def test_prospective_mode_still_rejects_every_historically_measured_identity() -> None:
    """Fixing low-data eligibility must not weaken the prospective safeguard."""
    scope = globally_unmeasured_scope(
        labeled_train_identity_keys=["observed"],
        global_identity_keys=["observed", "hidden-outer-train", "valid-key", "test-key"],
    )
    for key in ("observed", "hidden-outer-train", "valid-key", "test-key"):
        assert scope.rejection_reason_for(key) == ALREADY_MEASURED_REASON


def test_observed_chemistry_outranks_quarantine_under_a_row_level_split() -> None:
    """An identity in both train and test is rejected as observed, counted once."""
    scope = observed_only_scope(
        labeled_train_identity_keys=["shared"],
        quarantine_identity_keys=["shared", "test-only"],
    )
    assert scope.rejection_reason_for("shared") == OBSERVED_IN_LABELED_TRAIN_REASON
    assert scope.quarantined_identity_keys() == frozenset({"test-only"})
    assert scope.quarantine_overlap_with_observed_count == 1


def test_scope_record_records_that_global_membership_was_never_consulted() -> None:
    """Provenance must distinguish 'no global keys' from 'never looked'."""
    observed = observed_only_scope(labeled_train_identity_keys=["a"])
    assert observed.scope_record()["global_identity_hash"] == "not_consulted"
    assert observed.consults_complete_dataset is False


def test_assert_excludes_catches_hidden_identities_reaching_generation() -> None:
    """The runner-side guard must fire when hidden chemistry becomes ineligible."""
    scope = observed_only_scope(labeled_train_identity_keys=["a", "hidden"])
    with pytest.raises(CandidateScopeViolation, match="Hidden outer-training"):
        scope.assert_excludes(["hidden"], description="Hidden outer-training identities")


def test_two_eligibility_rules_cannot_apply_to_one_generation_call() -> None:
    """Supplying both a policy and the legacy keyword is a contradiction."""
    scope = observed_only_scope(labeled_train_identity_keys=["a"])
    with pytest.raises(CandidateScopeViolation, match="not both"):
        resolve_generation_scope(
            labeled_train_identity_keys=["a"],
            candidate_scope=scope,
            measured_identity_keys=["b"],
        )


def test_scope_must_cover_the_training_frame_it_is_used_with() -> None:
    """A policy built from a different subset would silently mis-scope the run."""
    scope = observed_only_scope(labeled_train_identity_keys=["a"])
    with pytest.raises(CandidateScopeViolation, match="omits identities"):
        resolve_generation_scope(
            labeled_train_identity_keys=["a", "b"],
            candidate_scope=scope,
            measured_identity_keys=(),
        )


def test_legacy_keyword_reproduces_the_historical_rejection_reason() -> None:
    """Re-running a pre-audit configuration must reproduce its audit verbatim."""
    legacy = legacy_scope_from_measured_identity_keys(
        labeled_train_identity_keys=["observed"],
        measured_identity_keys=["observed", "elsewhere"],
    )
    assert legacy.mode == GLOBALLY_UNMEASURED_PROSPECTIVE
    assert legacy.rejection_reason_for("observed") == ALREADY_MEASURED_REASON
    assert legacy.rejection_reason_for("elsewhere") == ALREADY_MEASURED_REASON

    observed_only_legacy = legacy_scope_from_measured_identity_keys(
        labeled_train_identity_keys=["observed"],
        measured_identity_keys=(),
    )
    assert observed_only_legacy.mode == OBSERVED_ONLY_LOW_DATA
    # Legacy provenance keeps the historical string even in observed-only mode.
    assert observed_only_legacy.rejection_reason_for("observed") == ALREADY_MEASURED_REASON


@pytest.mark.parametrize(
    ("config", "default", "expected"),
    [
        ({}, OBSERVED_ONLY_LOW_DATA, OBSERVED_ONLY_LOW_DATA),
        ({}, GLOBALLY_UNMEASURED_PROSPECTIVE, GLOBALLY_UNMEASURED_PROSPECTIVE),
        (
            {"candidate_scope": {"mode": GLOBALLY_UNMEASURED_PROSPECTIVE}},
            OBSERVED_ONLY_LOW_DATA,
            GLOBALLY_UNMEASURED_PROSPECTIVE,
        ),
    ],
)
def test_configured_mode_resolution(config: dict, default: str, expected: str) -> None:
    """Each runner supplies its own default; an explicit mode always wins."""
    assert resolved_candidate_scope_mode(config, default=default) == expected


def test_unknown_configured_mode_is_refused() -> None:
    """A typo in a config must not silently fall back to a different protocol."""
    with pytest.raises(CandidateScopeViolation, match="candidate_scope.mode"):
        resolved_candidate_scope_mode({"candidate_scope": {"mode": "observed_only"}})
    with pytest.raises(CandidateScopeViolation, match="Unknown candidate_scope keys"):
        resolved_candidate_scope_mode({"candidate_scope": {"scope": "whatever"}})


# ------------------------------------------------------------- identity gate


def _candidate_frame(keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_reaction_key": keys,
            "canonical_reaction_hash": [f"hash-{key}" for key in keys],
            "accepted": [True] * len(keys),
        }
    )


def test_accepted_pool_never_contains_a_quarantined_identity() -> None:
    """A quarantined match must be recorded and excluded, not silently dropped."""
    from bh_augmentation.augmentation.synthetic_identity import (
        _assert_accepted_scope_invariants,
    )

    accepted = _candidate_frame(["k1"]).assign(
        candidate_scope_mode=OBSERVED_ONLY_LOW_DATA,
        observed_in_labeled_train=False,
        quarantined_held_out_identity=True,
        globally_measured_outside_labeled_train=pd.NA,
    )
    with pytest.raises(AssertionError, match="quarantined held-out identity"):
        _assert_accepted_scope_invariants(accepted)


def test_observed_only_audit_must_not_record_complete_dataset_membership() -> None:
    """Recording global membership at generation time is itself the violation."""
    from bh_augmentation.augmentation.synthetic_identity import (
        _assert_accepted_scope_invariants,
    )

    accepted = _candidate_frame(["k1"]).assign(
        candidate_scope_mode=OBSERVED_ONLY_LOW_DATA,
        observed_in_labeled_train=False,
        quarantined_held_out_identity=False,
        globally_measured_outside_labeled_train=False,
    )
    with pytest.raises(AssertionError, match="forbids consulting"):
        _assert_accepted_scope_invariants(accepted)


def test_one_pool_cannot_mix_two_eligibility_modes() -> None:
    """Mixed-mode pools would make every downstream count uninterpretable."""
    from bh_augmentation.augmentation.synthetic_identity import (
        _assert_accepted_scope_invariants,
    )

    accepted = _candidate_frame(["k1", "k2"]).assign(
        candidate_scope_mode=[OBSERVED_ONLY_LOW_DATA, GLOBALLY_UNMEASURED_PROSPECTIVE],
        observed_in_labeled_train=False,
        quarantined_held_out_identity=False,
        globally_measured_outside_labeled_train=pd.NA,
    )
    with pytest.raises(AssertionError, match="mixes candidate-scope modes"):
        _assert_accepted_scope_invariants(accepted)


def test_stored_and_live_canonicalization_disagreement_fails_loudly() -> None:
    """Two identity vocabularies would accept everything; that must not be quiet.

    If the installed RDKit canonicalizes differently from the version that built
    the dataset, the partition's stored keys and the generator's live keys stop
    intersecting.  Eligibility would then reject nothing -- including candidates
    identical to observed training reactions -- with no visible symptom.
    """
    from bh_augmentation.augmentation.synthetic_identity import (
        assert_stored_identities_match_roles,
    )
    from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_to_record

    roles = ReactionRoles("CCBr", "CN", "[Pd]", "CP(C)C", "[Na+].[OH-]", "CCO", "CCNC")
    frame = pd.DataFrame(
        [
            {
                **reaction_roles_to_record(roles),
                "source_row_id": "row-a",
                "canonical_reaction_key": "a-stale-key-from-another-rdkit",
            }
        ]
    )
    with pytest.raises(CandidateScopeViolation, match="two different identity vocabularies"):
        assert_stored_identities_match_roles(frame, description="Labeled training partition")


def test_matching_stored_identities_pass_the_vocabulary_guard() -> None:
    """The guard must not fire on a frame the installed RDKit agrees with."""
    from bh_augmentation.augmentation.synthetic_identity import (
        assert_stored_identities_match_roles,
        canonicalize_synthetic_roles,
    )
    from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_to_record

    roles = ReactionRoles("CCBr", "CN", "[Pd]", "CP(C)C", "[Na+].[OH-]", "CCO", "CCNC")
    identity = canonicalize_synthetic_roles(roles)
    frame = pd.DataFrame(
        [
            {
                **reaction_roles_to_record(roles),
                "source_row_id": "row-a",
                "canonical_reaction_key": identity.canonical_reaction_key,
            }
        ]
    )
    assert_stored_identities_match_roles(frame, description="Labeled training partition")


def test_feature_matrix_and_candidate_row_counts_must_agree() -> None:
    """A misaligned audit would attach one candidate's identity to another's row."""
    from bh_augmentation.augmentation.synthetic_identity import audit_candidate_identities

    with pytest.raises(ValueError, match="same row count"):
        audit_candidate_identities(
            _candidate_frame(["k1", "k2"]),
            np.zeros((1, 2), dtype=np.float32),
            source_rows=[],
            candidate_scope=observed_only_scope(labeled_train_identity_keys=["a"]),
        )


def test_identity_gate_refuses_two_simultaneous_rules() -> None:
    """The gate itself must reject an ambiguous eligibility contract."""
    from bh_augmentation.augmentation.synthetic_identity import audit_candidate_identities

    with pytest.raises(CandidateScopeViolation, match="not both"):
        audit_candidate_identities(
            _candidate_frame(["k1"]),
            np.zeros((1, 2), dtype=np.float32),
            source_rows=[],
            measured_keys=["k1"],
            candidate_scope=observed_only_scope(labeled_train_identity_keys=["a"]),
        )
