"""Explicit, typed candidate-eligibility scope for synthetic reaction generation.

Two scientifically different experiments in this repository generate synthetic
reactions from a labeled training subset, and they must not share one candidate
identity rule:

``observed_only_low_data``
    Simulate a learner that has observed only a nested low-data labeled subset.
    A generated reaction is a legitimate synthetic target whenever the simulated
    learner has not observed it.  Whether the reaction happens to exist somewhere
    else in the complete historical dataset is *unknowable* to that learner, so
    complete-dataset membership must never decide eligibility.  Structurally,
    a policy in this mode is forbidden from carrying global identity keys at all.

``globally_unmeasured_prospective``
    Propose reactions that have genuinely never been measured in the complete
    historical dataset.  A candidate whose canonical identity occurs anywhere in
    the measured universe is ineligible.  This is the correct rule for Phase-18
    prospective discovery and for the historical prospective control family.

Two further taxonomy labels exist for the repository-wide call-site audit but
never construct a generation policy:

``representation_augmentation``
    The augmentation re-represents an *existing* reaction (SMILES randomization,
    role/reaction-order permutation, latent or feature-space interpolation).
    Chemical-identity rejection is semantically inapplicable: the chemistry is
    deliberately unchanged, or the object generated is a feature coordinate that
    carries no asserted chemical identity at all.

``not_applicable``
    The call site has no candidate-eligibility semantics.

Quarantine is a separate concept from eligibility.  Validation and outer-test
identities are partition membership, not label information, and the inductive
benchmark requires that a generated candidate matching one of them never enters
student training.  Such candidates are recorded in the candidate audit with an
explicit quarantine reason rather than being silently dropped.

Hidden outer-training identities are deliberately *absent* from this module's
data model.  There is no field that can carry them, so no generation path can
consult them; they are joined only after generation and pseudo-labels are frozen
(see :mod:`bh_augmentation.evaluation.withheld_cell_oracle`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from bh_augmentation.utils.corrected_runs import stable_hash

CANDIDATE_SCOPE_SCHEMA_VERSION = "bh-candidate-scope-v1"

#: Eligibility is decided against the labeled low-data subset only.
OBSERVED_ONLY_LOW_DATA = "observed_only_low_data"
#: Eligibility is decided against the complete measured universe.
GLOBALLY_UNMEASURED_PROSPECTIVE = "globally_unmeasured_prospective"
#: Audit-only label: the family re-represents existing chemistry.
REPRESENTATION_AUGMENTATION = "representation_augmentation"
#: Audit-only label: the call site has no candidate-eligibility semantics.
NOT_APPLICABLE = "not_applicable"

#: The complete classification taxonomy used by the repository-wide audit.
CANDIDATE_SCOPE_CLASSIFICATIONS = (
    OBSERVED_ONLY_LOW_DATA,
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    REPRESENTATION_AUGMENTATION,
    NOT_APPLICABLE,
)
#: Only these two classifications construct a generation-time policy.
GENERATION_SCOPE_MODES = (OBSERVED_ONLY_LOW_DATA, GLOBALLY_UNMEASURED_PROSPECTIVE)

#: Provenance marker for a policy the caller declared explicitly.
EXPLICIT_PROVENANCE = "explicit"
#: Provenance marker for the pre-audit ``measured_identity_keys`` keyword path.
LEGACY_PROVENANCE = "legacy_measured_identity_keys"
_PROVENANCES = frozenset({EXPLICIT_PROVENANCE, LEGACY_PROVENANCE})

#: Rejection reason recorded when the simulated learner already observed the identity.
OBSERVED_IN_LABELED_TRAIN_REASON = "observed_in_labeled_train"
#: Rejection reason recorded under the historical/global eligibility rule.
ALREADY_MEASURED_REASON = "already_measured"
#: Rejection reason recorded for held-out-partition identity matches.
QUARANTINED_REASON = "quarantined_held_out_identity"

#: Default description of why quarantined identities may not train the student.
DEFAULT_QUARANTINE_ROLE = "validation_or_outer_test_partition_identity"

_NOT_CONSULTED = "not_consulted"


class CandidateScopeViolation(ValueError):
    """Raised when a candidate-scope contract would be broken."""


@dataclass(frozen=True, slots=True)
class CandidateScopePolicy:
    """One immutable, explicitly typed candidate-eligibility rule.

    Attributes
    ----------
    mode:
        One of :data:`GENERATION_SCOPE_MODES`.
    observed_identity_keys:
        Canonical reaction keys the simulated learner has actually observed --
        that is, the labeled training subset.  Always part of the rejection set.
    global_identity_keys:
        Canonical reaction keys of the complete measured universe.  Permitted
        **only** in :data:`GLOBALLY_UNMEASURED_PROSPECTIVE` mode; supplying them
        in observed-only mode raises :class:`CandidateScopeViolation`, which is
        what makes "the algorithm never consults complete-dataset membership"
        a structural guarantee rather than a convention.
    quarantine_identity_keys:
        Validation and outer-test identities.  Matches are recorded in the
        candidate audit with :data:`QUARANTINED_REASON` and kept out of student
        training; they are never counted as chemistry the learner has observed.
    """

    mode: str
    observed_identity_keys: tuple[str, ...]
    global_identity_keys: tuple[str, ...] = ()
    quarantine_identity_keys: tuple[str, ...] = ()
    quarantine_role: str = DEFAULT_QUARANTINE_ROLE
    provenance: str = EXPLICIT_PROVENANCE
    schema_version: str = CANDIDATE_SCOPE_SCHEMA_VERSION
    #: Identities that were requested as quarantined but are already observed.
    #: Nonzero only under a split protocol that is not group-safe.
    quarantine_overlap_with_observed_count: int = 0

    def __post_init__(self) -> None:
        if self.mode not in GENERATION_SCOPE_MODES:
            raise CandidateScopeViolation(
                "Candidate scope mode must be one of "
                f"{sorted(GENERATION_SCOPE_MODES)}; observed {self.mode!r}."
            )
        if self.provenance not in _PROVENANCES:
            raise CandidateScopeViolation(
                f"Unknown candidate scope provenance {self.provenance!r}."
            )
        if self.schema_version != CANDIDATE_SCOPE_SCHEMA_VERSION:
            raise CandidateScopeViolation("Candidate scope schema version mismatch.")
        for name in (
            "observed_identity_keys",
            "global_identity_keys",
            "quarantine_identity_keys",
        ):
            object.__setattr__(self, name, _normalized_keys(getattr(self, name), name))
        if not isinstance(self.quarantine_role, str) or not self.quarantine_role.strip():
            raise CandidateScopeViolation("quarantine_role must be a non-empty string.")

        if self.mode == OBSERVED_ONLY_LOW_DATA and self.global_identity_keys:
            raise CandidateScopeViolation(
                "An observed-only low-data candidate scope must not carry global "
                "measured identities: complete-dataset membership is unknowable to "
                "the simulated learner and may only be joined after candidate "
                "generation and pseudo-labels are frozen."
            )
        if self.mode == GLOBALLY_UNMEASURED_PROSPECTIVE and not self.global_identity_keys:
            raise CandidateScopeViolation(
                "A globally-unmeasured prospective candidate scope requires the "
                "complete measured identity universe."
            )
        # Observed chemistry outranks quarantine. Under a grouped split the two
        # sets are disjoint by construction, but the legacy row-level splitters
        # can place the same canonical identity in both train and test. Such an
        # identity is already ineligible because the learner observed it, so
        # normalizing it out of the quarantine set keeps the two channels
        # meaningful instead of double-counting one rejection. Callers whose
        # split protocol *guarantees* disjointness assert that themselves --
        # see LowDataEvaluationUnit, which is built from grouped assignments.
        overlap = set(self.observed_identity_keys) & set(self.quarantine_identity_keys)
        if overlap:
            object.__setattr__(
                self,
                "quarantine_identity_keys",
                tuple(sorted(set(self.quarantine_identity_keys) - overlap)),
            )
        object.__setattr__(self, "quarantine_overlap_with_observed_count", len(overlap))

    @property
    def consults_complete_dataset(self) -> bool:
        """Whether eligibility depends on the complete measured universe."""
        return self.mode == GLOBALLY_UNMEASURED_PROSPECTIVE

    def rejection_identity_keys(self) -> frozenset[str]:
        """Return every identity whose reuse makes a candidate ineligible."""
        if self.mode == OBSERVED_ONLY_LOW_DATA:
            return frozenset(self.observed_identity_keys)
        return frozenset(self.observed_identity_keys) | frozenset(self.global_identity_keys)

    def quarantined_identity_keys(self) -> frozenset[str]:
        """Return identities that must never reach student training."""
        return frozenset(self.quarantine_identity_keys)

    def rejection_reason_for(self, key: str | None) -> str | None:
        """Return the eligibility rejection reason for one canonical key.

        A policy reconstructed from the legacy ``measured_identity_keys``
        keyword deliberately keeps reporting the historical
        ``"already_measured"`` string so that re-running any pre-audit
        configuration reproduces its candidate audit byte for byte.  Explicitly
        declared policies report the precise reason instead.
        """
        if key is None:
            return None
        legacy = self.provenance == LEGACY_PROVENANCE
        if key in set(self.observed_identity_keys):
            if legacy or self.mode != OBSERVED_ONLY_LOW_DATA:
                return ALREADY_MEASURED_REASON
            return OBSERVED_IN_LABELED_TRAIN_REASON
        if self.consults_complete_dataset and key in set(self.global_identity_keys):
            return ALREADY_MEASURED_REASON
        if key in set(self.quarantine_identity_keys):
            return QUARANTINED_REASON
        return None

    def with_quarantine(self, keys: Iterable[str]) -> CandidateScopePolicy:
        """Return a copy carrying an additional quarantined identity set."""
        merged = set(self.quarantine_identity_keys) | set(_normalized_keys(keys, "keys"))
        return CandidateScopePolicy(
            mode=self.mode,
            observed_identity_keys=self.observed_identity_keys,
            global_identity_keys=self.global_identity_keys,
            quarantine_identity_keys=tuple(sorted(merged)),
            quarantine_role=self.quarantine_role,
            provenance=self.provenance,
        )

    @property
    def observed_identity_hash(self) -> str:
        """Hash the exact labeled-training identity set."""
        return stable_hash(
            {"schema": CANDIDATE_SCOPE_SCHEMA_VERSION, "keys": list(self.observed_identity_keys)}
        )

    @property
    def global_identity_hash(self) -> str:
        """Hash the global universe, or record that it was never consulted."""
        if not self.consults_complete_dataset:
            return _NOT_CONSULTED
        return stable_hash(
            {"schema": CANDIDATE_SCOPE_SCHEMA_VERSION, "keys": list(self.global_identity_keys)}
        )

    @property
    def quarantine_identity_hash(self) -> str:
        """Hash the exact quarantined identity set."""
        return stable_hash(
            {
                "schema": CANDIDATE_SCOPE_SCHEMA_VERSION,
                "keys": list(self.quarantine_identity_keys),
            }
        )

    @property
    def scope_hash(self) -> str:
        """Hash the complete scope contract for run provenance."""
        return stable_hash(self.scope_record())

    def scope_record(self) -> dict[str, Any]:
        """Return machine-readable provenance for manifests and audits."""
        return {
            "schema_version": self.schema_version,
            "candidate_scope_mode": self.mode,
            "candidate_scope_provenance": self.provenance,
            "consults_complete_dataset": self.consults_complete_dataset,
            "observed_identity_count": len(self.observed_identity_keys),
            "observed_identity_hash": self.observed_identity_hash,
            "global_identity_count": (
                len(self.global_identity_keys) if self.consults_complete_dataset else 0
            ),
            "global_identity_hash": self.global_identity_hash,
            "quarantine_identity_count": len(self.quarantine_identity_keys),
            "quarantine_identity_hash": self.quarantine_identity_hash,
            "quarantine_role": self.quarantine_role,
            "quarantine_overlap_with_observed_count": (
                self.quarantine_overlap_with_observed_count
            ),
        }

    def assert_excludes(self, keys: Iterable[str], *, description: str) -> None:
        """Hard-fail if any forbidden identity reached this generation scope.

        Used by leakage tests and by runners to prove that hidden outer-training
        identities were never handed to candidate generation.
        """
        forbidden = set(_normalized_keys(keys, description))
        contaminated = sorted(forbidden & self.rejection_identity_keys())
        if contaminated:
            raise CandidateScopeViolation(
                f"{description} reached the candidate generation scope: "
                f"{contaminated[:3]} ({len(contaminated)} identities)."
            )


def observed_only_scope(
    *,
    labeled_train_identity_keys: Iterable[str],
    quarantine_identity_keys: Iterable[str] = (),
    quarantine_role: str = DEFAULT_QUARANTINE_ROLE,
) -> CandidateScopePolicy:
    """Build the low-data protocol scope: only observed chemistry is ineligible."""
    return CandidateScopePolicy(
        mode=OBSERVED_ONLY_LOW_DATA,
        observed_identity_keys=tuple(labeled_train_identity_keys),
        quarantine_identity_keys=tuple(quarantine_identity_keys),
        quarantine_role=quarantine_role,
    )


def globally_unmeasured_scope(
    *,
    labeled_train_identity_keys: Iterable[str],
    global_identity_keys: Iterable[str],
    quarantine_identity_keys: Iterable[str] = (),
    quarantine_role: str = DEFAULT_QUARANTINE_ROLE,
) -> CandidateScopePolicy:
    """Build the prospective-discovery scope: any measured identity is ineligible."""
    return CandidateScopePolicy(
        mode=GLOBALLY_UNMEASURED_PROSPECTIVE,
        observed_identity_keys=tuple(labeled_train_identity_keys),
        global_identity_keys=tuple(global_identity_keys),
        quarantine_identity_keys=tuple(quarantine_identity_keys),
        quarantine_role=quarantine_role,
    )


def legacy_scope_from_measured_identity_keys(
    *,
    labeled_train_identity_keys: Iterable[str],
    measured_identity_keys: Iterable[str],
) -> CandidateScopePolicy:
    """Reconstruct the exact pre-audit rule from the legacy keyword argument.

    Every historical caller reached the identity gate through
    ``measured_keys = measured_canonical_keys(df_train, additional_keys=...)``.
    When the caller supplied no additional keys the effective rule was already
    observed-only; when it supplied the complete canonical universe the
    effective rule was global novelty.  Reconstructing the mode here keeps
    historical runs bit-identical while making the semantics explicit in
    provenance instead of implicit in a keyword argument.
    """
    observed = _normalized_keys(labeled_train_identity_keys, "labeled_train_identity_keys")
    additional = _normalized_keys(measured_identity_keys, "measured_identity_keys")
    extra = tuple(sorted(set(additional) - set(observed)))
    if not extra:
        return CandidateScopePolicy(
            mode=OBSERVED_ONLY_LOW_DATA,
            observed_identity_keys=observed,
            provenance=LEGACY_PROVENANCE,
        )
    return CandidateScopePolicy(
        mode=GLOBALLY_UNMEASURED_PROSPECTIVE,
        observed_identity_keys=observed,
        global_identity_keys=tuple(sorted(set(observed) | set(additional))),
        provenance=LEGACY_PROVENANCE,
    )


def resolve_generation_scope(
    *,
    labeled_train_identity_keys: Iterable[str],
    candidate_scope: CandidateScopePolicy | None,
    measured_identity_keys: Iterable[str],
) -> CandidateScopePolicy:
    """Resolve exactly one candidate scope for a generator invocation.

    Generators accept either the explicit ``candidate_scope`` policy or the
    legacy ``measured_identity_keys`` iterable, never both.  In either case the
    labeled training frame's own identities are folded in, because a candidate
    identical to a reaction the learner has already observed is never a useful
    synthetic target under either protocol.
    """
    observed = _normalized_keys(labeled_train_identity_keys, "labeled_train_identity_keys")
    legacy = _normalized_keys(measured_identity_keys, "measured_identity_keys")
    if candidate_scope is None:
        return legacy_scope_from_measured_identity_keys(
            labeled_train_identity_keys=observed,
            measured_identity_keys=legacy,
        )
    if legacy:
        raise CandidateScopeViolation(
            "Pass either candidate_scope or measured_identity_keys, not both: "
            "two eligibility rules cannot apply to one generation call."
        )
    if not isinstance(candidate_scope, CandidateScopePolicy):
        raise CandidateScopeViolation(
            "candidate_scope must be a CandidateScopePolicy instance."
        )
    missing = set(observed) - set(candidate_scope.observed_identity_keys)
    if missing:
        raise CandidateScopeViolation(
            "The candidate scope omits identities present in the training frame "
            f"handed to generation: {sorted(missing)[:3]} "
            f"({len(missing)} identities). The scope must be built from the same "
            "labeled subset the generator sees."
        )
    return candidate_scope


def resolved_candidate_scope_mode(
    config: Mapping[str, Any],
    *,
    default: str = OBSERVED_ONLY_LOW_DATA,
) -> str:
    """Read and validate a config's declared ``candidate_scope.mode``.

    The default belongs to the caller, not to this function: a low-data
    augmentation runner defaults to ``observed_only_low_data`` while a
    prospective-discovery runner defaults to ``globally_unmeasured_prospective``.
    Making each runner state its own default is what stops one implicit rule
    from silently spanning two different scientific questions.
    """
    if default not in GENERATION_SCOPE_MODES:
        raise CandidateScopeViolation(f"Unsupported default candidate scope {default!r}.")
    requested = config.get("candidate_scope", {})
    if not isinstance(requested, Mapping):
        raise CandidateScopeViolation("candidate_scope must be a mapping.")
    unknown = sorted(set(requested) - {"mode", "modes"})
    if unknown:
        raise CandidateScopeViolation(
            f"Unknown candidate_scope keys: {unknown}."
        )
    mode = str(requested.get("mode", default))
    if mode not in GENERATION_SCOPE_MODES:
        raise CandidateScopeViolation(
            f"candidate_scope.mode must be one of {sorted(GENERATION_SCOPE_MODES)}; "
            f"observed {mode!r}."
        )
    return mode


def scope_classification_of(mode: str) -> str:
    """Map a generation mode to its audit classification label."""
    if mode not in CANDIDATE_SCOPE_CLASSIFICATIONS:
        raise CandidateScopeViolation(f"Unknown candidate scope classification {mode!r}.")
    return mode


def _normalized_keys(value: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise CandidateScopeViolation(f"{name} must be an iterable of keys, not a string.")
    keys: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise CandidateScopeViolation(
                f"{name} must contain only non-empty canonical reaction keys."
            )
        keys.add(item)
    return tuple(sorted(keys))
