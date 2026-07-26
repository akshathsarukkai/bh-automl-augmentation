"""Shared construction of candidate-level synthetic audit artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
)

CANDIDATE_AUDIT_CONTEXT_FIELDS = (
    "transfer_kind",
    "seed",
    "train_fraction",
    "policy_id",
)


def build_candidate_audit_frame(
    candidate_df: pd.DataFrame,
    *,
    transfer_kind: str,
    seed: int,
    train_fraction: float,
    policy_id: str,
) -> pd.DataFrame:
    """Attach run context without narrowing the generator's candidate audit."""
    missing = [
        field for field in REQUIRED_SYNTHETIC_AUDIT_FIELDS if field not in candidate_df
    ]
    if missing:
        raise ValueError(
            "Synthetic candidate audit is missing required fields: " + ", ".join(missing)
        )

    result = candidate_df.copy()
    context = {
        "transfer_kind": str(transfer_kind),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "policy_id": str(policy_id),
    }
    for position, field in enumerate(CANDIDATE_AUDIT_CONTEXT_FIELDS):
        if field in result:
            raise ValueError(
                f"Synthetic candidate audit unexpectedly already contains {field!r}."
            )
        result.insert(position, field, context[field])
    return result


def combine_candidate_audit_frames(
    frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Combine policy audits while preserving a useful empty artifact schema."""
    materialized = list(frames)
    if materialized:
        result = pd.concat(materialized, ignore_index=True, sort=False)
    else:
        result = pd.DataFrame(
            columns=[
                *CANDIDATE_AUDIT_CONTEXT_FIELDS,
                *REQUIRED_SYNTHETIC_AUDIT_FIELDS,
            ]
        )
    missing = [
        field for field in REQUIRED_SYNTHETIC_AUDIT_FIELDS if field not in result
    ]
    if missing:  # pragma: no cover - guarded by build_candidate_audit_frame
        raise AssertionError(
            "Combined candidate audit lost required fields: " + ", ".join(missing)
        )
    return result
