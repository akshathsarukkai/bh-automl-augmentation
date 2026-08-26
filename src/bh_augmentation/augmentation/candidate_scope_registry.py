"""Declared candidate-scope semantics for every generation call site.

This module is the machine-readable answer to "which scientific question does
this call site's candidate-eligibility rule belong to?".  Every module that can
reach the canonical identity gate is classified into exactly one of the four
labels in :data:`~bh_augmentation.augmentation.candidate_scope.CANDIDATE_SCOPE_CLASSIFICATIONS`,
together with the mechanism that enforces the classification and a short
rationale.

The registry is *declared* here and *verified* by ``scripts/audit_candidate_scope.py``,
which walks the source tree and fails when a module's real wiring stops matching
its declaration.  Keeping the two apart is deliberate: a hand-written table
rots, and a purely derived table cannot express intent.  Together they make the
intent explicit and the drift detectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bh_augmentation.augmentation.candidate_scope import (
    CANDIDATE_SCOPE_CLASSIFICATIONS,
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    NOT_APPLICABLE,
    OBSERVED_ONLY_LOW_DATA,
    REPRESENTATION_AUGMENTATION,
    CandidateScopeViolation,
)

CANDIDATE_SCOPE_REGISTRY_VERSION = "bh-candidate-scope-registry-v1"

#: How a call site's eligibility rule is supplied.
EXPLICIT_POLICY = "explicit_candidate_scope_policy"
CONFIGURED_POLICY = "configured_candidate_scope_mode"
CONDUIT = "conduit_forwards_callers_policy"
GLOBAL_UNIVERSE = "complete_measured_universe"
NO_IDENTITY_GATE = "no_chemical_identity_gate"
NULL_IDENTITY = "identity_not_evaluable_recorded_as_null"
INERT = "no_generation_performed"


@dataclass(frozen=True, slots=True)
class ScopeDeclaration:
    """One module's declared candidate-scope semantics."""

    module: str
    classification: str
    mechanism: str
    rationale: str
    #: Result families this module's eligibility rule can affect.
    affects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.classification not in CANDIDATE_SCOPE_CLASSIFICATIONS:
            raise CandidateScopeViolation(
                f"{self.module}: unknown classification {self.classification!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize one declaration for the machine-readable audit."""
        return {
            "module": self.module,
            "classification": self.classification,
            "mechanism": self.mechanism,
            "rationale": self.rationale,
            "affects": list(self.affects),
        }


_DECLARATIONS = (
    # ------------------------------------------------------------------ core
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/synthetic_identity.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "The shared identity gate. It applies whichever CandidateScopePolicy "
            "it is handed and has no eligibility opinion of its own."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/candidate_scope.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale="Defines the taxonomy and the typed policy; performs no generation.",
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/condition_transfer.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "Anonymous generator. Resolves the caller's policy and folds in the "
            "labeled training frame's own identities; the scientific choice "
            "belongs to the caller."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/role_aware_condition_transfer.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale="Typed generator; same conduit contract as the anonymous generator.",
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/condition_recombine.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "Recombination generator. The resolved policy also bounds the "
            "generation budget, so the scope decides pool size as well as "
            "acceptance."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/ensemble_filter.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale="Wraps the recombination generator and forwards its caller's rule.",
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/scientific_transfer_pool.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "Pool builder. Accepts either an explicit policy or the historical "
            "global key contract, and binds the resolved mode into the pool hash."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/chemical_controls.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "Builds one chemical control from whichever rule its caller declares, "
            "and records the resolved mode in the control's frozen config."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/family_condition_transfer.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "External-family typed transfer. Rejects against whatever key set the "
            "adapter caller supplies; defaults to the training frame's identities."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/search.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale="Dispatcher; propagates the callee's semantics unchanged.",
    ),
    # -------------------------------------------------- low-data augmentation
    ScopeDeclaration(
        module="src/bh_augmentation/policy_search.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale=(
            "Frozen-policy search. Reads candidate_scope.mode and binds it into "
            "the scientific config hash, so the declared rule is part of the "
            "frozen policy identity. Defaults to the historical prospective rule "
            "so a pre-audit config reproduces exactly."
        ),
        affects=("observed_only_condition_transfer", "globally_unmeasured_condition_transfer"),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/final_evaluation.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale=(
            "Refit and single outer-test batch. Must resolve the same rule the "
            "search froze, or the refit pool would differ from the selected one."
        ),
        affects=("observed_only_condition_transfer", "globally_unmeasured_condition_transfer"),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/ae_policy_search.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale="Joint AE + transfer search on the same low-data protocol.",
        affects=("supervised_ae_condition_transfer",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/ae_final_evaluation.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale="AE refit; resolves the same rule the AE search froze.",
        affects=("supervised_ae_condition_transfer",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation_control_benchmark.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale=(
            "Phase 11 matched-budget controls. Asks whether chemical augmentation "
            "helps a low-data learner, so it defaults to observed-only and the "
            "post-hoc validator applies the same declared rule."
        ),
        affects=("augmentation_controls_matched_budget",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/corrected_condition_transfer.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale="Matched corrected anonymous/typed transfer at low data fractions.",
        affects=(
            "corrected_anonymous_condition_transfer",
            "corrected_role_aware_condition_transfer",
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/run_condition_transfer.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale="Legacy anonymous transfer runner on a low-data protocol.",
        affects=("condition_transfer_legacy",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/run_role_aware_condition_transfer.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale="Legacy typed transfer runner on a low-data protocol.",
        affects=("role_aware_condition_transfer_legacy",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/run_condition_transfer_supervised_ae.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale="Supervised-AE runner whose training pool depends on transferred candidates.",
        affects=("condition_transfer_supervised_ae",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/run_external_family_validation.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=CONFIGURED_POLICY,
        rationale=(
            "External reaction-family adapter. Low-data augmentation benefit on a "
            "second family; eligibility follows the labeled training partition. "
            "The empirical datasets remain blocked, so only the semantics change."
        ),
        affects=("external_family_validation",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/candidate_scope_reanalysis.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=EXPLICIT_POLICY,
        rationale=(
            "Development reanalysis. Runs both modes side by side and freezes each "
            "pool before the withheld-cell oracle reads any hidden yield."
        ),
        affects=(
            "observed_only_condition_transfer",
            "globally_unmeasured_condition_transfer",
            "withheld_cell_transfer_oracle",
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/redesigned_ae_benchmark.py",
        classification=OBSERVED_ONLY_LOW_DATA,
        mechanism=GLOBAL_UNIVERSE,
        rationale=(
            "Phase 14/15. Already correct before this audit: its key set is the "
            "phase's own visible rows (inner policy-fit rows during search, saved "
            "training rows during placement and final), never the complete "
            "dataset, and a hash guard hard-fails if a pool consumes anything "
            "else. The `global_measured_identity_keys` parameter name is "
            "historical; the value passed is observed-only."
        ),
        affects=("redesigned_supervised_ae", "anonymous_transfer_without_ae",
                 "typed_transfer_without_ae"),
    ),
    # ------------------------------------------------- prospective discovery
    ScopeDeclaration(
        module="src/bh_augmentation/prospective_package.py",
        classification=GLOBALLY_UNMEASURED_PROSPECTIVE,
        mechanism=GLOBAL_UNIVERSE,
        rationale=(
            "Phase 18 prospective discovery. A proposed reaction must never have "
            "been measured anywhere in the complete historical dataset, so global "
            "membership is the correct and intended eligibility rule."
        ),
        affects=("prospective_discovery_package",),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/recommendation_simulation.py",
        classification=GLOBALLY_UNMEASURED_PROSPECTIVE,
        mechanism=GLOBAL_UNIVERSE,
        rationale=(
            "Recommendation simulation ranks reactions for prospective execution; "
            "a recommendation that is already measured is not a recommendation."
        ),
        affects=("recommendation_simulation",),
    ),
    ScopeDeclaration(
        module="scripts/run_phase1_identity_smoke.py",
        classification=GLOBALLY_UNMEASURED_PROSPECTIVE,
        mechanism=GLOBAL_UNIVERSE,
        rationale=(
            "Determinism smoke for the identity gate. It rejects against the "
            "complete measured dataset, so its recorded zero-accepted result is "
            "a statement about GLOBAL discovery headroom on a near-complete "
            "factorial matrix -- not a statement about how much a low-data "
            "learner could generate. Retained under its executed protocol."
        ),
        affects=("phase_01_identity_smoke",),
    ),
    ScopeDeclaration(
        module="scripts/run_phase2_role_change_smoke.py",
        classification=GLOBALLY_UNMEASURED_PROSPECTIVE,
        mechanism=GLOBAL_UNIVERSE,
        rationale=(
            "Role-change contract smoke under the same global rule; its "
            "null-eligibility finding is a global-novelty statement."
        ),
        affects=("phase_02_role_change_smoke",),
    ),
    ScopeDeclaration(
        module="scripts/run_phase3_ranking_smoke.py",
        classification=GLOBALLY_UNMEASURED_PROSPECTIVE,
        mechanism=GLOBAL_UNIVERSE,
        rationale=(
            "Candidate-ranking smoke under the same global rule; its zero-"
            "accepted result is a global-novelty statement."
        ),
        affects=("phase_03_ranking_smoke",),
    ),
    # ------------------------------------------ representation augmentations
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/smiles_randomization.py",
        classification=REPRESENTATION_AUGMENTATION,
        mechanism=NO_IDENTITY_GATE,
        rationale=(
            "RDKit atom renumbering produces a different SMILES string for the "
            "same molecule and copies the measured label. Canonicalization is an "
            "involution over this transform, so an identity gate would reject "
            "100% of the output by definition. The gate must not apply."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/order_permutation.py",
        classification=REPRESENTATION_AUGMENTATION,
        mechanism=NO_IDENTITY_GATE,
        rationale=(
            "Reaction-component reordering within a role. Identity-preserving only "
            "while the caller passes same-role columns; the module documents that "
            "it does not infer chemical exchangeability and refuses an empty "
            "column list rather than guessing."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/simple_controls.py",
        classification=REPRESENTATION_AUGMENTATION,
        mechanism=NULL_IDENTITY,
        rationale=(
            "Duplication and reweighting controls reuse measured chemistry by "
            "design; the feature-space controls (nearest-neighbour pseudo-"
            "labelling, self-training, feature mixup) create coordinates with no "
            "seven-role identity and record chemical_identity_applicable=False."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/latent_interpolation.py",
        classification=REPRESENTATION_AUGMENTATION,
        mechanism=NULL_IDENTITY,
        rationale=(
            "A latent coordinate has no seven-role reaction identity. Chemical "
            "duplicate questions are recorded as null, not false: they are not "
            "answerable, which is a different statement from 'answered no'."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/augmentation/utility_guided_feature_gan.py",
        classification=REPRESENTATION_AUGMENTATION,
        mechanism=NULL_IDENTITY,
        rationale=(
            "Generated feature coordinates carry no asserted chemical identity; "
            "feature-hash deduplication is applied, chemical-identity rejection is "
            "not."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/representation_benchmark.py",
        classification=NOT_APPLICABLE,
        mechanism=INERT,
        rationale="Compares feature encodings with no augmentation at all.",
    ),
    # ------------------------------------------------------ diagnostics only
    ScopeDeclaration(
        module="src/bh_augmentation/hidden_measured_calibration.py",
        classification=NOT_APPLICABLE,
        mechanism=INERT,
        rationale=(
            "Phase 12 pseudo-label accuracy and interval calibration. Its withheld "
            "cells are a condition group carved out of the labeled training subset "
            "rather than the hidden outer-training partition, so it measures a "
            "proxy population; the withheld-cell oracle covers the population this "
            "audit is about."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/evaluation/withheld_cell_oracle.py",
        classification=NOT_APPLICABLE,
        mechanism=INERT,
        rationale=(
            "Reads hidden outer-training yields only after candidates and pseudo-"
            "labels are frozen and hash-verified. Performs no generation."
        ),
    ),
    ScopeDeclaration(
        module="src/bh_augmentation/evaluation/low_data_partitions.py",
        classification=NOT_APPLICABLE,
        mechanism=CONDUIT,
        rationale=(
            "Defines the four partitions and builds typed policies from them. Has "
            "no field that can carry hidden outer-training identities into "
            "generation."
        ),
    ),
)

#: Declarations keyed by repository-relative module path.
CANDIDATE_SCOPE_DECLARATIONS: dict[str, ScopeDeclaration] = {
    declaration.module: declaration for declaration in _DECLARATIONS
}


def declaration_for(module: str) -> ScopeDeclaration | None:
    """Return the declared semantics for one repository-relative module path."""
    return CANDIDATE_SCOPE_DECLARATIONS.get(str(module))


def registry_record() -> dict[str, Any]:
    """Return the complete machine-readable declaration set."""
    return {
        "schema_version": CANDIDATE_SCOPE_REGISTRY_VERSION,
        "classifications": list(CANDIDATE_SCOPE_CLASSIFICATIONS),
        "declarations": [
            declaration.to_dict()
            for declaration in sorted(_DECLARATIONS, key=lambda item: item.module)
        ],
    }
