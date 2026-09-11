"""Candidate-pool accounting shared by the reanalysis and the frozen-policy protocol.

Preregistration section 9 of ``PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md``
requires the accepted-synthetic-row count per evaluation unit under both
eligibility rules, so the size of the treatment is visible alongside its
effect.  The function below is the single definition of that accounting; the
development reanalysis and the policy-search sidecars both call it, so a count
means the same thing wherever it appears.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bh_augmentation.augmentation.candidate_scope import CandidateScopePolicy

COUNTED_REJECTION_REASONS = (
    "observed_in_labeled_train",
    "already_measured",
    "quarantined_held_out_identity",
    "source_identical",
    "duplicate_synthetic",
)


def pool_statistics(
    candidate_df: pd.DataFrame,
    scope: CandidateScopePolicy,
) -> dict[str, Any]:
    """Summarize one generated candidate pool under its declared eligibility rule.

    The returned mapping is JSON-serializable and carries the scope record, so
    a count can never be read without the rule that produced it.
    """
    if candidate_df.empty:
        return {
            "generated_candidate_count": 0,
            "accepted_candidate_count": 0,
            "unique_accepted_identity_count": 0,
            "selected_candidate_count": 0,
            **{f"rejected_{reason}": 0 for reason in COUNTED_REJECTION_REASONS},
            "rejected_other": 0,
            **scope.scope_record(),
        }
    accepted = candidate_df.loc[candidate_df["accepted"].astype(bool)]
    reasons = candidate_df["rejection_reason"].fillna("").astype(str)
    counted = {
        reason: int(reasons.eq(reason).sum()) for reason in COUNTED_REJECTION_REASONS
    }
    return {
        "generated_candidate_count": int(len(candidate_df)),
        "accepted_candidate_count": int(len(accepted)),
        "unique_accepted_identity_count": int(
            accepted["canonical_reaction_key"].astype(str).nunique()
        ),
        "selected_candidate_count": (
            int(candidate_df["kept"].astype(bool).sum())
            if "kept" in candidate_df
            else int(len(accepted))
        ),
        **{f"rejected_{reason}": count for reason, count in counted.items()},
        "rejected_other": int(reasons.ne("").sum() - sum(counted.values())),
        **scope.scope_record(),
    }
