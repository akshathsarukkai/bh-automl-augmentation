"""Shared registry preventing invalid historical results from silent reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

STATUS_DOCUMENT = "RESULT_STATUS.md"

INVALID_RESULT_FAMILIES = {
    "role_aware_condition_transfer_v2": (
        "Measured and synthetic rows used incompatible feature-block semantics."
    ),
    "role_aware_condition_transfer": (
        "Legacy role-aware results predate canonical seven-role feature compatibility checks."
    ),
    "condition_transfer_matched_comparison": (
        "This comparison table contains invalid historical role-aware results."
    ),
}

INVALID_HISTORICAL_PATHS = {
    ("results", "stress", "logo_product"): (
        "historical_logo_product",
        "The saved product LOGO family does not establish a verified "
        "leave-one-group-out split contract.",
    ),
    ("results", "stress", "logo_reactant"): (
        "historical_logo_reactant",
        "The saved reactant LOGO family does not establish a verified "
        "leave-one-group-out split contract.",
    ),
    ("results", "baseline", "stress_logo_product_metrics.csv"): (
        "historical_logo_product_baseline",
        "The saved product stress-LOGO baseline lacks the fold assignments and "
        "overlap audit required to verify leave-one-group-out evaluation.",
    ),
    ("results", "baseline", "stress_logo_reactant_metrics.csv"): (
        "historical_logo_reactant_baseline",
        "The saved reactant stress-LOGO baseline lacks the fold assignments and "
        "overlap audit required to verify leave-one-group-out evaluation.",
    ),
}

INVALID_ROW_MARKERS = (
    "role_aware_condition_transfer",
    "role-aware condition transfer",
)


class InvalidResultError(ValueError):
    """Raised when invalid historical evidence is loaded without an override."""


def assert_result_directory_allowed(
    path: str | Path,
    *,
    allow_invalid: bool = False,
) -> None:
    """Reject known-invalid historical result directories by default."""
    if allow_invalid:
        return
    normalized_parts = tuple(part.lower() for part in Path(path).parts)
    for marker, (family, reason) in INVALID_HISTORICAL_PATHS.items():
        if _contains_path_marker(normalized_parts, marker):
            raise InvalidResultError(_message(family, reason, path))
    for family, reason in INVALID_RESULT_FAMILIES.items():
        matching_parts = [part for part in normalized_parts if family in part]
        if matching_parts:
            if all(part.startswith("corrected_") for part in matching_parts):
                continue
            raise InvalidResultError(_message(family, reason, path))


def assert_result_table_allowed(
    table: pd.DataFrame,
    *,
    source: str | Path = "in-memory table",
    allow_invalid: bool = False,
) -> None:
    """Reject tables containing rows from invalid role-aware result families."""
    if allow_invalid or table.empty:
        return
    text_columns = [
        column
        for column in table.columns
        if table[column].dtype == object or isinstance(table[column].dtype, pd.StringDtype)
    ]
    for column in text_columns:
        values = table[column].astype(str).str.lower()
        for marker in INVALID_ROW_MARKERS:
            invalid_rows = values.str.contains(marker, regex=False, na=False)
            if "result_status" in table.columns:
                corrected = table["result_status"].astype(str).str.lower().eq(
                    "corrected_revalidation"
                )
                invalid_rows &= ~corrected
            if invalid_rows.any():
                raise InvalidResultError(
                    _message(
                        "role_aware_condition_transfer_v2",
                        f"Column {column!r} contains invalid role-aware result rows.",
                        source,
                    )
                )


def read_result_csv(
    path: str | Path,
    *,
    allow_invalid: bool = False,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a result CSV only after directory and row-level validity checks."""
    assert_result_directory_allowed(path, allow_invalid=allow_invalid)
    table = pd.read_csv(path, **read_csv_kwargs)
    assert_result_table_allowed(table, source=path, allow_invalid=allow_invalid)
    return table


def _message(family: str, reason: str, path: str | Path) -> str:
    return (
        f"Result family '{family}' is invalid and cannot be loaded by default from {path}. "
        f"{reason} See {STATUS_DOCUMENT}. Pass allow_invalid=True only for explicit "
        "historical inspection; do not use these rows in benchmark claims."
    )


def _contains_path_marker(parts: tuple[str, ...], marker: tuple[str, ...]) -> bool:
    """Return whether an exact, contiguous historical path marker is present."""
    marker_length = len(marker)
    return any(
        parts[start : start + marker_length] == marker
        for start in range(len(parts) - marker_length + 1)
    )
