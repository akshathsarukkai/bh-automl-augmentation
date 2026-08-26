"""Immutable four-way partitioning of one low-data evaluation unit.

A low-data evaluation unit is one ``(seed, train_fraction)`` slice of a saved
canonical grouped split.  Four partitions are defined once and never redefined
downstream:

``labeled_train``
    The nested low-data subset the simulated learner is allowed to observe:
    outer split ``train`` **and** ``included_in_training_subset``.
``hidden_outer_train``
    Rows in the same outer training partition that the low-data subset excluded.
    The learner has not observed them.  Their identities are legitimate synthetic
    targets and their measured yields are oracle labels for retrospective
    diagnostics only.
``validation``
    The untouched policy-selection partition.
``test``
    The untouched final-evaluation partition.

The frames this module hands out carry **no** ``yield`` column for
``hidden_outer_train`` or ``test``.  Reading those outcomes requires an explicit,
separate call to :meth:`LowDataEvaluationUnit.hidden_outer_train_oracle` or
:meth:`LowDataEvaluationUnit.outer_test_outcomes`, each of which goes through
:func:`bh_augmentation.data.outcome_access.read_allowed_outcomes` so the read is
narrow, deliberate, and greppable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    OBSERVED_ONLY_LOW_DATA,
    CandidateScopePolicy,
    CandidateScopeViolation,
    globally_unmeasured_scope,
    observed_only_scope,
)
from bh_augmentation.augmentation.synthetic_identity import (
    assert_stored_identities_match_roles,
    measured_canonical_keys,
)
from bh_augmentation.data.outcome_access import read_allowed_outcomes
from bh_augmentation.data.saved_canonical_splits import SavedCanonicalSplits
from bh_augmentation.utils.corrected_runs import stable_hash

LOW_DATA_PARTITION_SCHEMA_VERSION = "bh-low-data-partitions-v1"

_LABEL_COLUMNS = ("yield",)


class LowDataPartitionError(ValueError):
    """Raised when a low-data partition contract is violated."""


@dataclass(frozen=True, slots=True)
class LowDataEvaluationUnit:
    """The four immutable partitions of one low-data evaluation unit."""

    evaluation_unit: str
    seed: int
    train_fraction: float
    dataset_path: Path
    dataset_hash: str
    labeled_train: pd.DataFrame = field(repr=False)
    hidden_outer_train: pd.DataFrame = field(repr=False)
    validation: pd.DataFrame = field(repr=False)
    test: pd.DataFrame = field(repr=False)
    canonical_source_row_ids: tuple[str, ...] = field(repr=False, default=())
    schema_version: str = LOW_DATA_PARTITION_SCHEMA_VERSION

    # ---------------------------------------------------------------- identities

    @property
    def labeled_train_identity_keys(self) -> tuple[str, ...]:
        """Canonical identities the simulated learner has observed."""
        return _identities(self.labeled_train)

    @property
    def hidden_outer_train_identity_keys(self) -> tuple[str, ...]:
        """Canonical identities withheld from the simulated learner.

        Never hand these to candidate generation.  They exist so the oracle
        diagnostic and the leakage tests can name what must stay hidden.
        """
        return _identities(self.hidden_outer_train)

    @property
    def validation_identity_keys(self) -> tuple[str, ...]:
        """Canonical identities of the policy-selection partition."""
        return _identities(self.validation)

    @property
    def test_identity_keys(self) -> tuple[str, ...]:
        """Canonical identities of the final-evaluation partition."""
        return _identities(self.test)

    @property
    def quarantine_identity_keys(self) -> tuple[str, ...]:
        """Validation and outer-test identities, which never train the student."""
        return tuple(sorted(set(self.validation_identity_keys) | set(self.test_identity_keys)))

    @property
    def global_identity_keys(self) -> tuple[str, ...]:
        """Every canonical identity in the complete measured dataset."""
        return tuple(
            sorted(
                set(self.labeled_train_identity_keys)
                | set(self.hidden_outer_train_identity_keys)
                | set(self.validation_identity_keys)
                | set(self.test_identity_keys)
            )
        )

    # -------------------------------------------------------------------- scopes

    def _assert_group_safe(self) -> None:
        """Fail if a canonical identity crosses this unit's partitions.

        Saved canonical splits are grouped on ``canonical_reaction_key``, so
        every identity belongs to exactly one partition.  An overlap here is a
        broken split, not a tolerable quirk, and it would silently make the
        quarantine channel meaningless.
        """
        observed = set(self.labeled_train_identity_keys)
        for name, keys in (
            ("hidden_outer_train", self.hidden_outer_train_identity_keys),
            ("validation", self.validation_identity_keys),
            ("test", self.test_identity_keys),
        ):
            overlap = observed & set(keys)
            if overlap:
                raise LowDataPartitionError(
                    f"Canonical identities cross labeled_train and {name} in "
                    f"{self.evaluation_unit}: {sorted(overlap)[:3]}."
                )

    def observed_only_candidate_scope(self) -> CandidateScopePolicy:
        """Build the low-data protocol scope for this unit.

        The returned policy structurally cannot carry hidden outer-training or
        global identities: eligibility is decided against ``labeled_train``
        alone, and validation/test identities are quarantined rather than
        treated as chemistry the learner has seen.
        """
        self._assert_group_safe()
        return observed_only_scope(
            labeled_train_identity_keys=self.labeled_train_identity_keys,
            quarantine_identity_keys=self.quarantine_identity_keys,
        )

    def globally_unmeasured_candidate_scope(self) -> CandidateScopePolicy:
        """Build the prospective-novelty control scope for this unit."""
        self._assert_group_safe()
        return globally_unmeasured_scope(
            labeled_train_identity_keys=self.labeled_train_identity_keys,
            global_identity_keys=self.global_identity_keys,
            quarantine_identity_keys=self.quarantine_identity_keys,
        )

    def candidate_scope(self, mode: str) -> CandidateScopePolicy:
        """Build the scope named by a config's ``candidate_scope.mode``."""
        from bh_augmentation.augmentation.candidate_scope import (
            GLOBALLY_UNMEASURED_PROSPECTIVE,
            OBSERVED_ONLY_LOW_DATA,
            CandidateScopeViolation,
        )

        if mode == OBSERVED_ONLY_LOW_DATA:
            return self.observed_only_candidate_scope()
        if mode == GLOBALLY_UNMEASURED_PROSPECTIVE:
            return self.globally_unmeasured_candidate_scope()
        raise CandidateScopeViolation(
            f"{mode!r} does not name a generation-time candidate scope."
        )

    # ------------------------------------------------------------ outcome access

    def hidden_outer_train_oracle(self) -> pd.DataFrame:
        """Read hidden outer-training yields for retrospective diagnostics only.

        These are oracle labels.  They must never influence candidate
        generation, candidate ranking, teacher fitting, policy selection,
        hyperparameter selection, synthetic weights, or student fitting.
        """
        return self._read_outcomes(self.hidden_outer_train, "hidden outer-training")

    def outer_test_outcomes(self) -> pd.DataFrame:
        """Read outer-test yields, which only final evaluation may consume."""
        return self._read_outcomes(self.test, "outer-test")

    def _read_outcomes(self, frame: pd.DataFrame, description: str) -> pd.DataFrame:
        if frame.empty:
            raise LowDataPartitionError(
                f"The {description} partition of {self.evaluation_unit} is empty."
            )
        allowed = [str(value) for value in frame["source_row_id"]]
        outcomes = read_allowed_outcomes(
            self.dataset_path,
            self.canonical_source_row_ids,
            allowed,
        )
        result = frame.loc[:, ["source_row_id", "canonical_reaction_key"]].copy()
        result["source_row_id"] = result["source_row_id"].astype(str)
        result["yield"] = result["source_row_id"].map(outcomes).astype(float)
        if result["yield"].isna().any():
            raise LowDataPartitionError(
                f"Selective {description} outcome read returned incomplete yields."
            )
        return result

    # ---------------------------------------------------------------- provenance

    def partition_record(self) -> dict[str, Any]:
        """Return machine-readable provenance for this unit's four partitions."""
        return {
            "schema_version": self.schema_version,
            "evaluation_unit": self.evaluation_unit,
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "dataset_hash": self.dataset_hash,
            "labeled_train_row_count": int(len(self.labeled_train)),
            "hidden_outer_train_row_count": int(len(self.hidden_outer_train)),
            "validation_row_count": int(len(self.validation)),
            "test_row_count": int(len(self.test)),
            "labeled_train_identity_count": len(self.labeled_train_identity_keys),
            "hidden_outer_train_identity_count": len(self.hidden_outer_train_identity_keys),
            "labeled_train_identity_hash": stable_hash(
                list(self.labeled_train_identity_keys)
            ),
            "hidden_outer_train_identity_hash": stable_hash(
                list(self.hidden_outer_train_identity_keys)
            ),
            "quarantine_identity_hash": stable_hash(list(self.quarantine_identity_keys)),
            "labeled_train_source_id_hash": stable_hash(
                sorted(str(value) for value in self.labeled_train["source_row_id"])
            ),
        }


def build_low_data_evaluation_unit(
    saved: SavedCanonicalSplits,
    *,
    seed: int,
    train_fraction: float,
    dataset_path: str | Path,
    evaluation_unit: str | None = None,
) -> LowDataEvaluationUnit:
    """Materialize the four immutable partitions of one low-data unit.

    ``hidden_outer_train`` and ``test`` are returned without a ``yield`` column
    so that no downstream stage can consume those outcomes by accident.
    """
    seed_value = int(seed)
    fraction = float(train_fraction)
    rows = saved.low_data_assignments.loc[
        saved.low_data_assignments["seed"].eq(seed_value)
        & saved.low_data_assignments["train_fraction"].eq(fraction)
    ]
    if len(rows) != len(saved.canonical):
        raise LowDataPartitionError(
            f"Saved split slice for seed={seed_value} fraction={fraction} is incomplete."
        )

    outer_train = rows.loc[rows["outer_split"].eq("train")]
    labeled_ids = set(outer_train.loc[outer_train["included_in_training_subset"], "source_row_id"])
    hidden_ids = set(
        outer_train.loc[~outer_train["included_in_training_subset"], "source_row_id"]
    )
    validation_ids = set(rows.loc[rows["outer_split"].eq("valid"), "source_row_id"])
    test_ids = set(rows.loc[rows["outer_split"].eq("test"), "source_row_id"])

    _assert_disjoint(
        {
            "labeled_train": labeled_ids,
            "hidden_outer_train": hidden_ids,
            "validation": validation_ids,
            "test": test_ids,
        }
    )
    if not labeled_ids:
        raise LowDataPartitionError(
            f"Low-data unit seed={seed_value} fraction={fraction} has no labeled training rows."
        )

    canonical = saved.canonical
    canonical_ids = tuple(str(value) for value in canonical["source_row_id"])
    unit = LowDataEvaluationUnit(
        evaluation_unit=(
            evaluation_unit
            if evaluation_unit is not None
            else f"canonical-random:seed={seed_value}:fraction={_format_fraction(fraction)}"
        ),
        seed=seed_value,
        train_fraction=fraction,
        dataset_path=Path(dataset_path),
        dataset_hash=saved.dataset_hash,
        labeled_train=_slice(canonical, labeled_ids, keep_labels=True),
        hidden_outer_train=_slice(canonical, hidden_ids, keep_labels=False),
        validation=_slice(canonical, validation_ids, keep_labels=True),
        test=_slice(canonical, test_ids, keep_labels=False),
        canonical_source_row_ids=canonical_ids,
    )
    # The labeled subset is the eligibility rejection set, so its stored
    # identities must be the same vocabulary the generator will produce.
    assert_stored_identities_match_roles(
        unit.labeled_train,
        description=f"Labeled training partition of {unit.evaluation_unit}",
    )
    if unit.labeled_train["yield"].isna().any():
        raise LowDataPartitionError("Labeled low-data training outcomes are missing.")
    if unit.validation["yield"].isna().any():
        raise LowDataPartitionError("Validation outcomes are missing.")
    return unit


def scope_from_split_frames(
    mode: str,
    *,
    splits: Mapping[str, pd.DataFrame],
    global_identity_keys: Iterable[str] = (),
) -> CandidateScopePolicy:
    """Build a candidate scope from a runner's materialized split frames.

    Runners that already hold ``{"train", "valid", "test"}`` frames -- the
    :meth:`SavedCanonicalSplits.materialize_variants` shape -- can obtain a
    typed policy without re-deriving partitions.  ``train`` is the labeled
    low-data subset, and ``valid``/``test`` become the quarantine set.  Rows
    that appear in none of the three are the hidden outer-training population;
    they are deliberately not represented here, which is what makes them
    generatable under the observed-only protocol.
    """
    assert_stored_identities_match_roles(
        splits["train"],
        description="Labeled training split handed to candidate generation",
    )
    observed = _identities(splits["train"])
    quarantine = tuple(
        sorted(set(_identities(splits["valid"])) | set(_identities(splits["test"])))
    )
    if mode == OBSERVED_ONLY_LOW_DATA:
        return observed_only_scope(
            labeled_train_identity_keys=observed,
            quarantine_identity_keys=quarantine,
        )
    if mode == GLOBALLY_UNMEASURED_PROSPECTIVE:
        return globally_unmeasured_scope(
            labeled_train_identity_keys=observed,
            global_identity_keys=global_identity_keys,
            quarantine_identity_keys=quarantine,
        )
    raise CandidateScopeViolation(f"{mode!r} does not name a generation-time scope.")


def _slice(canonical: pd.DataFrame, ids: Iterable[str], *, keep_labels: bool) -> pd.DataFrame:
    wanted = {str(value) for value in ids}
    frame = canonical.loc[canonical["source_row_id"].astype(str).isin(wanted)].copy()
    if not keep_labels:
        frame = frame.drop(columns=[c for c in _LABEL_COLUMNS if c in frame])
    return frame.reset_index(drop=True)


def _identities(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return one partition's canonical identities.

    Canonical dataset frames already carry ``canonical_reaction_key``; the
    legacy runners operate on a cleaned frame that does not, so fall back to the
    shared canonicalization rather than requiring callers to precompute it.
    """
    if frame.empty:
        return ()
    if (
        "canonical_reaction_key" in frame
        and not frame["canonical_reaction_key"].isna().any()
    ):
        return tuple(
            sorted({str(value) for value in frame["canonical_reaction_key"] if str(value)})
        )
    return tuple(sorted(measured_canonical_keys(frame)))


def _assert_disjoint(partitions: dict[str, set[str]]) -> None:
    names = sorted(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = partitions[left] & partitions[right]
            if overlap:
                raise LowDataPartitionError(
                    f"Partitions {left} and {right} share source rows: {sorted(overlap)[:3]}."
                )


def _format_fraction(fraction: float) -> str:
    return f"{fraction:g}"
