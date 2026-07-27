"""Selective access to measured outcomes after scientific plans are frozen."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def read_allowed_outcomes(
    dataset_path: str | Path,
    canonical_source_ids: Sequence[str],
    allowed_source_ids: Sequence[str],
) -> dict[str, float]:
    """Read only explicitly allowed outcome rows from a canonical CSV."""
    canonical_ids = tuple(canonical_source_ids)
    allowed_ids = tuple(allowed_source_ids)
    allowed = set(allowed_ids)
    if len(allowed) != len(allowed_ids) or not allowed:
        raise ValueError("Allowed outcome source IDs must be nonempty and unique.")
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("Canonical source IDs must be unique.")
    positions = {
        source_id: index for index, source_id in enumerate(canonical_ids)
    }
    missing = sorted(allowed - set(positions))
    if missing:
        raise ValueError(f"Unknown allowed outcome IDs: {missing[:5]}.")
    allowed_positions = {positions[source_id] for source_id in allowed}
    frame = pd.read_csv(
        dataset_path,
        usecols=["source_row_id", "yield"],
        dtype={"source_row_id": str},
        skiprows=lambda row_number: (
            row_number > 0 and row_number - 1 not in allowed_positions
        ),
    )
    observed_ids = tuple(frame["source_row_id"].astype(str))
    if set(observed_ids) != allowed or len(observed_ids) != len(allowed):
        raise ValueError("Selective outcome read did not return exactly allowed rows.")
    yields = pd.to_numeric(frame["yield"], errors="raise")
    if yields.isna().any() or not np.isfinite(yields.to_numpy(float)).all():
        raise ValueError("Allowed outcome rows contain non-finite yields.")
    return dict(zip(observed_ids, yields.astype(float), strict=True))
