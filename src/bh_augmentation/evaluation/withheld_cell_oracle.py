"""Leakage-safe retrospective oracle for withheld-cell reconstruction.

This diagnostic answers one question:

    Can condition transfer reconstruct measured reactions that were unavailable
    to the simulated low-data learner?

It is a *retrospective* measurement, not a predictive benchmark, and it is
deliberately structured so that the hidden measured yields it consumes cannot
flow backwards into anything that produced the candidates.

The mechanism is a two-stage contract:

1. :class:`FrozenCandidateBundle` is constructed from a candidate audit *after*
   generation and pseudo-labeling are complete.  It stores an immutable copy of
   the candidate identities and their pseudo-labels together with hashes of
   both, and it refuses to be built from a frame that carries a measured-yield
   column at all.
2. :func:`withheld_cell_oracle_diagnostic` is the only function that ever reads
   hidden outer-training yields.  It re-verifies the bundle hashes before the
   join, returns metrics, and mutates nothing.

Because the bundle is frozen and hash-verified, an oracle metric can never
become an input to candidate generation, candidate ranking, teacher fitting,
policy selection, hyperparameter selection, synthetic weights, or student
fitting: by the time any hidden yield is legible, the artifacts those stages
consume are already sealed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.evaluation.metrics import mae, rmse, spearman_corr
from bh_augmentation.utils.corrected_runs import stable_hash

WITHHELD_CELL_ORACLE_SCHEMA_VERSION = "bh-withheld-cell-oracle-v1"

#: Columns a candidate audit must supply before it can be frozen.
REQUIRED_BUNDLE_COLUMNS = (
    "canonical_reaction_key",
    "synthetic_label",
    "accepted",
)
#: Optional columns the diagnostic uses when present.
OPTIONAL_BUNDLE_COLUMNS = (
    "nearest_training_support_distance",
    "support_distance_metric",
    "actual_changed_roles",
    "candidate_scope_mode",
    "kept",
)
#: Columns that must never appear in a frozen bundle -- they would mean a true
#: measured outcome had already been joined onto the candidates.
FORBIDDEN_BUNDLE_COLUMNS = ("yield", "true_yield", "measured_yield", "oracle_yield")


class OracleContractViolation(ValueError):
    """Raised when the leakage-safe oracle contract would be broken."""


@dataclass(frozen=True, slots=True)
class FrozenCandidateBundle:
    """Candidate identities and pseudo-labels sealed before any oracle join."""

    evaluation_unit: str
    train_fraction: float
    candidate_scope_mode: str
    generated_candidate_count: int
    candidate_identity_hash: str
    pseudo_label_hash: str
    frozen_hash: str
    schema_version: str = WITHHELD_CELL_ORACLE_SCHEMA_VERSION
    _rows: pd.DataFrame = field(repr=False, compare=False, default_factory=pd.DataFrame)

    @classmethod
    def freeze(
        cls,
        candidate_df: pd.DataFrame,
        *,
        evaluation_unit: str,
        train_fraction: float,
        candidate_scope_mode: str,
        accepted_only: bool = True,
    ) -> FrozenCandidateBundle:
        """Seal one candidate pool's identities and pseudo-labels.

        ``accepted_only`` keeps the pool that actually reached the student.  The
        rejected rows are not lost -- they stay in the run's candidate audit,
        which is where the quarantine and eligibility record belongs -- but the
        oracle measures the reconstruction quality of the pool that was used.
        """
        if not isinstance(candidate_df, pd.DataFrame):
            raise OracleContractViolation("A frozen bundle requires a candidate DataFrame.")
        present = [column for column in FORBIDDEN_BUNDLE_COLUMNS if column in candidate_df]
        if present:
            raise OracleContractViolation(
                "Candidate frames handed to the oracle must not already carry measured "
                f"outcomes; observed columns: {present}."
            )
        missing = [column for column in REQUIRED_BUNDLE_COLUMNS if column not in candidate_df]
        if missing:
            raise OracleContractViolation(
                f"Candidate audit is missing fields required to freeze: {missing}."
            )
        rows = candidate_df.copy()
        if accepted_only:
            rows = rows.loc[rows["accepted"].astype(bool)]
        keep = [
            column
            for column in (*REQUIRED_BUNDLE_COLUMNS, *OPTIONAL_BUNDLE_COLUMNS)
            if column in rows
        ]
        rows = rows.loc[:, keep].reset_index(drop=True)
        rows["canonical_reaction_key"] = rows["canonical_reaction_key"].astype("string")
        labels = pd.to_numeric(rows["synthetic_label"], errors="coerce")
        if labels.isna().any() or not np.isfinite(labels.to_numpy(float)).all():
            raise OracleContractViolation(
                "Every frozen candidate requires a finite pseudo-label."
            )
        rows["synthetic_label"] = labels.astype(float)

        identity_hash = stable_hash(
            {
                "schema": WITHHELD_CELL_ORACLE_SCHEMA_VERSION,
                "keys": [str(value) for value in rows["canonical_reaction_key"]],
            }
        )
        label_hash = stable_hash(
            {
                "schema": WITHHELD_CELL_ORACLE_SCHEMA_VERSION,
                "labels": [float(value) for value in rows["synthetic_label"]],
            }
        )
        frozen_hash = stable_hash(
            {
                "schema": WITHHELD_CELL_ORACLE_SCHEMA_VERSION,
                "evaluation_unit": str(evaluation_unit),
                "train_fraction": float(train_fraction),
                "candidate_scope_mode": str(candidate_scope_mode),
                "candidate_identity_hash": identity_hash,
                "pseudo_label_hash": label_hash,
                "generated_candidate_count": int(len(rows)),
            }
        )
        return cls(
            evaluation_unit=str(evaluation_unit),
            train_fraction=float(train_fraction),
            candidate_scope_mode=str(candidate_scope_mode),
            generated_candidate_count=int(len(rows)),
            candidate_identity_hash=identity_hash,
            pseudo_label_hash=label_hash,
            frozen_hash=frozen_hash,
            _rows=rows,
        )

    def verify(self) -> None:
        """Recompute both content hashes and refuse a mutated bundle."""
        rows = self._rows
        identity_hash = stable_hash(
            {
                "schema": WITHHELD_CELL_ORACLE_SCHEMA_VERSION,
                "keys": [str(value) for value in rows["canonical_reaction_key"]],
            }
        )
        label_hash = stable_hash(
            {
                "schema": WITHHELD_CELL_ORACLE_SCHEMA_VERSION,
                "labels": [float(value) for value in rows["synthetic_label"]],
            }
        )
        if identity_hash != self.candidate_identity_hash:
            raise OracleContractViolation("Frozen candidate identities were mutated.")
        if label_hash != self.pseudo_label_hash:
            raise OracleContractViolation("Frozen candidate pseudo-labels were mutated.")
        if int(len(rows)) != self.generated_candidate_count:
            raise OracleContractViolation("Frozen candidate row count changed.")

    def frozen_rows(self) -> pd.DataFrame:
        """Return a defensive copy of the sealed candidate rows."""
        self.verify()
        return self._rows.copy()

    def provenance(self) -> dict[str, Any]:
        """Return the bundle's machine-readable freeze record."""
        return {
            "schema_version": self.schema_version,
            "evaluation_unit": self.evaluation_unit,
            "train_fraction": self.train_fraction,
            "candidate_scope_mode": self.candidate_scope_mode,
            "generated_candidate_count": self.generated_candidate_count,
            "candidate_identity_hash": self.candidate_identity_hash,
            "pseudo_label_hash": self.pseudo_label_hash,
            "frozen_hash": self.frozen_hash,
        }


def withheld_cell_oracle_diagnostic(
    bundle: FrozenCandidateBundle,
    *,
    hidden_outer_train: pd.DataFrame,
    labeled_train_identity_keys: Iterable[str] = (),
    high_yield_threshold: float = 80.0,
    top_k_fractions: Sequence[float] = (0.1, 0.2),
    stratify: bool = True,
) -> dict[str, Any]:
    """Join frozen candidates onto hidden outer-training rows and score them.

    ``hidden_outer_train`` must contain ``canonical_reaction_key`` and ``yield``
    for rows that belong to the outer training partition but were excluded from
    the labeled low-data subset.  Those yields are oracle labels: they are read
    here and nowhere else.

    Replicate measurements of one canonical identity are reduced to their mean
    before scoring, because a canonical identity is the unit a generated
    candidate can address -- scoring against individual replicates would charge
    the pseudo-label for experimental replicate spread it cannot predict.
    """
    bundle.verify()
    _assert_hidden_frame(hidden_outer_train)
    labeled_keys = {str(value) for value in labeled_train_identity_keys}

    hidden = hidden_outer_train.loc[:, ["canonical_reaction_key", "yield"]].copy()
    hidden["canonical_reaction_key"] = hidden["canonical_reaction_key"].astype(str)
    hidden["yield"] = pd.to_numeric(hidden["yield"], errors="raise").astype(float)
    if not np.isfinite(hidden["yield"].to_numpy(float)).all():
        raise OracleContractViolation("Hidden outer-training yields must be finite.")
    contaminated = labeled_keys & set(hidden["canonical_reaction_key"])
    if contaminated:
        raise OracleContractViolation(
            "Hidden outer-training rows overlap the labeled training subset: "
            f"{sorted(contaminated)[:3]}."
        )
    hidden_by_identity = (
        hidden.groupby("canonical_reaction_key", as_index=False)["yield"]
        .mean()
        .rename(columns={"yield": "hidden_measured_yield"})
    )

    candidates = bundle.frozen_rows()
    candidates["canonical_reaction_key"] = candidates["canonical_reaction_key"].astype(str)
    # One pseudo-label per canonical identity: the pool already deduplicates
    # accepted identities, but averaging keeps the join total even if a caller
    # freezes a pool that does not.
    aggregations: dict[str, Any] = {"synthetic_label": "mean"}
    if "nearest_training_support_distance" in candidates:
        aggregations["nearest_training_support_distance"] = "mean"
    if "support_distance_metric" in candidates:
        aggregations["support_distance_metric"] = "first"
    if "actual_changed_roles" in candidates:
        aggregations["actual_changed_roles"] = "first"
    unique_candidates = (
        candidates.groupby("canonical_reaction_key", as_index=False).agg(aggregations)
    )

    joined = unique_candidates.merge(
        hidden_by_identity,
        on="canonical_reaction_key",
        how="inner",
        validate="one_to_one",
    )

    hidden_identity_count = int(len(hidden_by_identity))
    unique_candidate_count = int(len(unique_candidates))
    overlap_count = int(len(joined))
    record: dict[str, Any] = {
        "schema_version": WITHHELD_CELL_ORACLE_SCHEMA_VERSION,
        **bundle.provenance(),
        "high_yield_threshold": float(high_yield_threshold),
        "hidden_outer_train_row_count": int(len(hidden)),
        "hidden_outer_train_identity_count": hidden_identity_count,
        "unique_canonical_candidate_count": unique_candidate_count,
        "hidden_overlap_count": overlap_count,
        "hidden_overlap_fraction_of_candidates": _ratio(overlap_count, unique_candidate_count),
        "hidden_cell_coverage_fraction": _ratio(overlap_count, hidden_identity_count),
        **_score_block(joined, high_yield_threshold, top_k_fractions),
    }
    record.update(_support_distance_stats(joined))

    if stratify and "actual_changed_roles" in joined:
        strata = []
        for change_type, group in joined.groupby(
            joined["actual_changed_roles"].fillna("").astype(str), sort=True
        ):
            strata.append(
                {
                    "transferred_change_type": change_type or "none",
                    "overlap_count": int(len(group)),
                    **_score_block(group, high_yield_threshold, top_k_fractions),
                    **_support_distance_stats(group),
                }
            )
        record["by_transferred_change_type"] = strata
    return record


def oracle_records_to_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Flatten oracle records into one tabular artifact without the strata."""
    flat = [
        {key: value for key, value in record.items() if key != "by_transferred_change_type"}
        for record in records
    ]
    return pd.DataFrame(flat)


def oracle_strata_to_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the per-change-type strata of many oracle records."""
    rows: list[dict[str, Any]] = []
    for record in records:
        for stratum in record.get("by_transferred_change_type", []):
            rows.append(
                {
                    "evaluation_unit": record["evaluation_unit"],
                    "train_fraction": record["train_fraction"],
                    "candidate_scope_mode": record["candidate_scope_mode"],
                    **stratum,
                }
            )
    return pd.DataFrame(rows)


def _assert_hidden_frame(hidden_outer_train: pd.DataFrame) -> None:
    if not isinstance(hidden_outer_train, pd.DataFrame):
        raise OracleContractViolation("hidden_outer_train must be a DataFrame.")
    missing = sorted({"canonical_reaction_key", "yield"} - set(hidden_outer_train))
    if missing:
        raise OracleContractViolation(
            f"hidden_outer_train is missing required columns: {missing}."
        )


def _score_block(
    joined: pd.DataFrame,
    high_yield_threshold: float,
    top_k_fractions: Sequence[float],
) -> dict[str, Any]:
    """Score one joined candidate/oracle table."""
    if joined.empty:
        block: dict[str, Any] = {
            "pseudo_label_mae": math.nan,
            "pseudo_label_rmse": math.nan,
            "pseudo_label_spearman": math.nan,
            "pseudo_label_bias": math.nan,
            "high_yield_prevalence": math.nan,
            "high_yield_precision": math.nan,
            "high_yield_recall": math.nan,
            "high_yield_enrichment": math.nan,
        }
        for fraction in top_k_fractions:
            block[f"top_{_pct(fraction)}_enrichment"] = math.nan
            block[f"top_{_pct(fraction)}_mean_true_yield"] = math.nan
        return block

    truth = joined["hidden_measured_yield"].to_numpy(float)
    predicted = joined["synthetic_label"].to_numpy(float)
    is_high = truth >= float(high_yield_threshold)
    predicted_high = predicted >= float(high_yield_threshold)
    prevalence = float(is_high.mean())
    precision = float(is_high[predicted_high].mean()) if predicted_high.any() else math.nan
    recall = float(predicted_high[is_high].mean()) if is_high.any() else math.nan
    block = {
        "pseudo_label_mae": float(mae(truth, predicted)),
        "pseudo_label_rmse": float(rmse(truth, predicted)),
        "pseudo_label_spearman": (
            float(spearman_corr(truth, predicted)) if len(truth) >= 3 else math.nan
        ),
        "pseudo_label_bias": float(np.mean(predicted - truth)),
        "high_yield_prevalence": prevalence,
        "high_yield_precision": precision,
        "high_yield_recall": recall,
        "high_yield_enrichment": (
            precision / prevalence
            if predicted_high.any() and prevalence > 0.0
            else math.nan
        ),
    }
    order = np.argsort(-predicted, kind="mergesort")
    for fraction in top_k_fractions:
        k = max(1, int(math.ceil(float(fraction) * len(predicted))))
        top = order[:k]
        top_prevalence = float(is_high[top].mean())
        block[f"top_{_pct(fraction)}_enrichment"] = (
            top_prevalence / prevalence if prevalence > 0.0 else math.nan
        )
        block[f"top_{_pct(fraction)}_mean_true_yield"] = float(truth[top].mean())
    return block


def _support_distance_stats(joined: pd.DataFrame) -> dict[str, Any]:
    """Summarize distance to the nearest labeled training reaction.

    The generators do not share one distance metric: anonymous transfer records
    a cosine distance in [0, 2] while typed transfer records an unbounded
    Euclidean distance. Converting both to ``1 - distance`` and calling the
    result a similarity produces a number that is meaningless when the two are
    mixed, so the distance is reported as-is under its declared metric and a
    bounded similarity is derived only where that conversion is defined.
    """
    empty: dict[str, Any] = {
        "support_distance_metric": None,
        "nearest_labeled_training_distance_mean": math.nan,
        "nearest_labeled_training_distance_median": math.nan,
        "nearest_labeled_training_similarity_mean": math.nan,
        "nearest_labeled_training_similarity_median": math.nan,
    }
    if joined.empty or "nearest_training_support_distance" not in joined:
        return empty
    distances = pd.to_numeric(
        joined["nearest_training_support_distance"], errors="coerce"
    ).dropna()
    if distances.empty:
        return empty

    metrics = (
        set(joined["support_distance_metric"].dropna().astype(str))
        if "support_distance_metric" in joined
        else set()
    )
    metric = metrics.pop() if len(metrics) == 1 else ("mixed" if metrics else None)
    values = distances.to_numpy(float)
    record: dict[str, Any] = {
        "support_distance_metric": metric,
        "nearest_labeled_training_distance_mean": float(np.mean(values)),
        "nearest_labeled_training_distance_median": float(np.median(values)),
        "nearest_labeled_training_similarity_mean": math.nan,
        "nearest_labeled_training_similarity_median": math.nan,
    }
    if metric == "cosine_distance":
        similarity = 1.0 - values
        record["nearest_labeled_training_similarity_mean"] = float(np.mean(similarity))
        record["nearest_labeled_training_similarity_median"] = float(
            np.median(similarity)
        )
    return record


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _pct(fraction: float) -> str:
    value = float(fraction) * 100.0
    return f"{int(value)}pct" if value.is_integer() else f"{value:g}pct".replace(".", "p")
