"""Strict loading and validation of saved canonical grouped split assignments."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.data.canonical_splits import (
    OUTER_SPLITS,
    SPLIT_SCHEMA_VERSION,
    split_assignment_hash,
)
from bh_augmentation.data.canonicalize_roles import CANONICALIZATION_VERSION
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

ASSIGNMENT_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "seed",
    "outer_split",
    "outer_group_order",
    "train_fraction",
    "included_in_training_subset",
)
_INTEGER_PATTERN = re.compile(r"-?[0-9]+")
_BOOLEAN_VALUES = {"True": True, "False": False}
_REQUIRED_MANIFEST_FIELDS = {
    "canonical_dataset_hash",
    "canonical_rows_read",
    "canonical_subset_hash",
    "canonicalization_version",
    "dataset_nrows",
    "group_column",
    "n_groups",
    "n_rows",
    "seeds",
    "split_hash",
    "split_hashes",
    "split_schema_version",
    "train_fractions",
}


@dataclass(frozen=True)
class SavedCanonicalSplits:
    """Validated canonical data and its authoritative saved assignments."""

    canonical: pd.DataFrame
    outer_assignments: pd.DataFrame
    low_data_assignments: pd.DataFrame
    manifest: dict[str, Any]
    dataset_hash: str
    aggregate_split_hash: str
    per_seed_split_hashes: dict[int, str]
    split_directory: Path
    artifact_hashes: dict[str, str]

    @property
    def audit_metadata(self) -> dict[str, Any]:
        """Return immutable split provenance suitable for a run manifest."""
        return {
            "canonical_dataset_hash": self.dataset_hash,
            "canonical_subset_hash": str(self.manifest["canonical_subset_hash"]),
            "canonicalization_version": CANONICALIZATION_VERSION,
            "split_schema_version": SPLIT_SCHEMA_VERSION,
            "split_aggregate_hash": self.aggregate_split_hash,
            "per_seed_split_hashes": {
                str(seed): value for seed, value in sorted(self.per_seed_split_hashes.items())
            },
            "split_directory": str(self.split_directory),
            "split_artifact_hashes": dict(sorted(self.artifact_hashes.items())),
        }

    def source_id_split_hash(self, *, seed: int, train_fraction: float) -> str:
        """Hash exact source-ID membership for one saved split slice."""
        rows = self._fraction_rows(seed=seed, train_fraction=train_fraction)
        resolved_fraction = float(rows["train_fraction"].iloc[0])
        payload = {
            "schema": "bh-canonical-split-source-ids-v1",
            "seed": int(seed),
            "train_fraction": resolved_fraction,
            "train": sorted(
                rows.loc[rows["included_in_training_subset"], "source_row_id"].tolist()
            ),
            "valid": sorted(rows.loc[rows["outer_split"].eq("valid"), "source_row_id"].tolist()),
            "test": sorted(rows.loc[rows["outer_split"].eq("test"), "source_row_id"].tolist()),
        }
        return stable_hash(payload)

    def audit_record(self, *, seed: int, train_fraction: float) -> dict[str, Any]:
        """Return authoritative aggregate, seed, and fraction-level hashes."""
        seed_value = int(seed)
        if seed_value not in self.per_seed_split_hashes:
            raise ValueError(f"Requested seed is absent from saved assignments: {seed_value}.")
        resolved_fraction = _resolve_fraction(
            float(train_fraction),
            [float(value) for value in self.manifest["train_fractions"]],
        )
        return {
            **self.audit_metadata,
            "seed": seed_value,
            "train_fraction": resolved_fraction,
            "per_seed_split_hash": self.per_seed_split_hashes[seed_value],
            "source_id_split_hash": self.source_id_split_hash(
                seed=seed_value,
                train_fraction=resolved_fraction,
            ),
        }

    def materialize_variants(
        self,
        frame: pd.DataFrame | None = None,
        *,
        seed: int,
        train_fractions: list[float] | tuple[float, ...] | None = None,
    ) -> list[tuple[float, dict[str, pd.DataFrame]]]:
        """Materialize saved split slices by source ID without resetting indices."""
        scientific_frame = self.canonical if frame is None else frame
        _validate_materialization_frame(scientific_frame, self.canonical)
        fractions = (
            [float(value) for value in self.manifest["train_fractions"]]
            if train_fractions is None
            else [float(value) for value in train_fractions]
        )
        variants: list[tuple[float, dict[str, pd.DataFrame]]] = []
        for requested_fraction in fractions:
            resolved_fraction = _resolve_fraction(
                requested_fraction,
                [float(value) for value in self.manifest["train_fractions"]],
            )
            rows = self._fraction_rows(
                seed=int(seed),
                train_fraction=resolved_fraction,
            )
            train_ids = set(rows.loc[rows["included_in_training_subset"], "source_row_id"])
            valid_ids = set(rows.loc[rows["outer_split"].eq("valid"), "source_row_id"])
            test_ids = set(rows.loc[rows["outer_split"].eq("test"), "source_row_id"])
            variants.append(
                (
                    resolved_fraction,
                    {
                        "train": scientific_frame.loc[
                            scientific_frame["source_row_id"].isin(train_ids)
                        ].copy(),
                        "valid": scientific_frame.loc[
                            scientific_frame["source_row_id"].isin(valid_ids)
                        ].copy(),
                        "test": scientific_frame.loc[
                            scientific_frame["source_row_id"].isin(test_ids)
                        ].copy(),
                    },
                )
            )
        return variants

    def _fraction_rows(self, *, seed: int, train_fraction: float) -> pd.DataFrame:
        if int(seed) not in self.per_seed_split_hashes:
            raise ValueError(f"Requested seed is absent from saved assignments: {int(seed)}.")
        fraction = _resolve_fraction(
            float(train_fraction),
            [float(value) for value in self.manifest["train_fractions"]],
        )
        rows = self.low_data_assignments.loc[
            self.low_data_assignments["seed"].eq(int(seed))
            & self.low_data_assignments["train_fraction"].eq(fraction)
        ]
        if len(rows) != len(self.canonical):
            raise AssertionError("Validated split slice unexpectedly lost canonical rows.")
        return rows


def load_saved_canonical_splits(
    dataset_path: str | Path,
    split_directory: str | Path,
    *,
    requested_seeds: list[int] | tuple[int, ...] | None = None,
    requested_fractions: list[float] | tuple[float, ...] | None = None,
) -> SavedCanonicalSplits:
    """Load saved assignments and reject any provenance or semantic mismatch."""
    dataset = Path(dataset_path)
    directory = Path(split_directory)
    manifest_path = directory / "split_manifest.json"
    manifest = _load_manifest(manifest_path)
    dataset_hash = sha256_file(dataset)
    if dataset_hash != manifest["canonical_dataset_hash"]:
        raise ValueError("Canonical dataset hash does not match the saved split manifest.")

    nrows = manifest["dataset_nrows"]
    if nrows is not None and (not isinstance(nrows, int) or isinstance(nrows, bool) or nrows <= 0):
        raise ValueError("split manifest dataset_nrows must be null or a positive integer.")
    canonical = pd.read_csv(dataset, nrows=nrows)
    _validate_canonical_dataset(canonical, manifest)
    outer = _read_assignments(directory / "outer_split_assignments.csv")
    low = _read_assignments(directory / "low_data_subset_assignments.csv")
    _validate_assignments(canonical, outer, low, manifest)

    seeds = [int(value) for value in manifest["seeds"]]
    fractions = [float(value) for value in manifest["train_fractions"]]
    _validate_requests(
        seeds,
        fractions,
        requested_seeds=requested_seeds,
        requested_fractions=requested_fractions,
    )

    computed_seed_hashes = {
        seed: split_assignment_hash(
            outer.loc[outer["seed"].eq(seed)],
            low.loc[low["seed"].eq(seed)],
        )
        for seed in seeds
    }
    expected_seed_hashes = {int(seed): value for seed, value in manifest["split_hashes"].items()}
    if computed_seed_hashes != expected_seed_hashes:
        raise ValueError("Saved canonical per-seed split hash mismatch.")
    aggregate_hash = stable_hash({str(seed): value for seed, value in computed_seed_hashes.items()})
    if aggregate_hash != manifest["split_hash"]:
        raise ValueError("Saved canonical aggregate split hash mismatch.")

    return SavedCanonicalSplits(
        canonical=canonical,
        outer_assignments=outer,
        low_data_assignments=low,
        manifest=manifest,
        dataset_hash=dataset_hash,
        aggregate_split_hash=aggregate_hash,
        per_seed_split_hashes=computed_seed_hashes,
        split_directory=directory,
        artifact_hashes={
            "low_data_subset_assignments.csv": sha256_file(
                directory / "low_data_subset_assignments.csv"
            ),
            "outer_split_assignments.csv": sha256_file(
                directory / "outer_split_assignments.csv"
            ),
            "split_manifest.json": sha256_file(manifest_path),
        },
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid saved canonical split manifest: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError("Saved canonical split manifest must contain a JSON object.")
    missing = sorted(_REQUIRED_MANIFEST_FIELDS - set(value))
    if missing:
        raise ValueError(f"Saved canonical split manifest is missing fields: {missing}.")
    if value["split_schema_version"] != SPLIT_SCHEMA_VERSION:
        raise ValueError("Saved canonical split schema version mismatch.")
    if value["canonicalization_version"] != CANONICALIZATION_VERSION:
        raise ValueError("Saved canonicalization version mismatch.")
    if value["group_column"] != "canonical_reaction_key":
        raise ValueError("Saved canonical splits must group by canonical_reaction_key.")
    if not isinstance(value["seeds"], list) or not value["seeds"]:
        raise ValueError("Saved canonical split manifest seeds must be a non-empty list.")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in value["seeds"]) or len(
        value["seeds"]
    ) != len(set(value["seeds"])):
        raise ValueError("Saved canonical split manifest seeds must be unique integers.")
    if not isinstance(value["train_fractions"], list) or not value["train_fractions"]:
        raise ValueError("Saved train_fractions must be a non-empty list.")
    fractions = [_strict_manifest_fraction(value) for value in value["train_fractions"]]
    if fractions != sorted(set(fractions)) or 1.0 not in fractions:
        raise ValueError("Saved train_fractions must be unique, sorted, and include 1.0.")
    if not isinstance(value["split_hashes"], dict):
        raise ValueError("Saved split_hashes must be a mapping.")
    if set(value["split_hashes"]) != {str(seed) for seed in value["seeds"]}:
        raise ValueError("Saved split_hashes keys do not exactly match manifest seeds.")
    return value


def _strict_manifest_fraction(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Saved train fractions must be numeric.")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Saved train fractions must be finite and in (0, 1].")
    return fraction


def _read_assignments(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(raw.columns) != ASSIGNMENT_COLUMNS:
        raise ValueError(
            f"Saved assignment schema mismatch for {path.name}: "
            f"expected {list(ASSIGNMENT_COLUMNS)}, observed {list(raw.columns)}."
        )
    result = raw.copy()
    for column in ("source_row_id", "canonical_reaction_key"):
        if result[column].eq("").any():
            raise ValueError(f"Saved assignment column {column} contains empty values.")
    result["seed"] = _parse_integer_column(result["seed"], "seed", minimum=None)
    result["outer_group_order"] = _parse_integer_column(
        result["outer_group_order"], "outer_group_order", minimum=0
    )
    if not result["outer_split"].isin(OUTER_SPLITS).all():
        raise ValueError("Saved assignment outer_split contains an unsupported value.")
    result["train_fraction"] = _parse_fraction_column(result["train_fraction"])
    if not result["included_in_training_subset"].isin(_BOOLEAN_VALUES).all():
        raise ValueError(
            "Saved included_in_training_subset values must be exactly 'True' or 'False'."
        )
    result["included_in_training_subset"] = result["included_in_training_subset"].map(
        _BOOLEAN_VALUES
    )
    return result


def _parse_integer_column(
    values: pd.Series,
    name: str,
    *,
    minimum: int | None,
) -> pd.Series:
    if not values.map(lambda value: bool(_INTEGER_PATTERN.fullmatch(value))).all():
        raise ValueError(f"Saved assignment column {name} must contain exact integers.")
    parsed = values.map(int)
    if minimum is not None and (parsed < minimum).any():
        raise ValueError(f"Saved assignment column {name} must be at least {minimum}.")
    return parsed


def _parse_fraction_column(values: pd.Series) -> pd.Series:
    try:
        parsed = values.map(float)
    except ValueError as exc:
        raise ValueError("Saved train_fraction values must be numeric.") from exc
    if not parsed.map(math.isfinite).all() or not parsed.between(0, 1, inclusive="right").all():
        raise ValueError("Saved train_fraction values must be finite and in (0, 1].")
    return parsed


def _validate_canonical_dataset(canonical: pd.DataFrame, manifest: dict[str, Any]) -> None:
    required = {
        "source_row_id",
        "canonical_reaction_key",
        "canonicalization_version",
        "all_required_roles_parse_valid",
    }
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Canonical dataset is missing saved-split columns: {missing}.")
    if canonical.empty:
        raise ValueError("Canonical dataset for saved splits is empty.")
    if (
        canonical["source_row_id"].isna().any()
        or canonical["source_row_id"].astype(str).eq("").any()
    ):
        raise ValueError("Canonical dataset contains missing source_row_id values.")
    if canonical["source_row_id"].duplicated().any():
        raise ValueError("Canonical dataset source_row_id values must be unique.")
    if (
        canonical["canonical_reaction_key"].isna().any()
        or canonical["canonical_reaction_key"].astype(str).eq("").any()
    ):
        raise ValueError("Canonical dataset contains missing canonical reaction keys.")
    versions = set(canonical["canonicalization_version"].dropna().astype(str))
    if versions != {CANONICALIZATION_VERSION}:
        raise ValueError("Canonical dataset canonicalization version mismatch.")
    if not canonical["all_required_roles_parse_valid"].map(_strict_dataframe_bool).all():
        raise ValueError(
            "Saved scientific splits require all canonical molecular roles to be valid."
        )

    expected_rows = _strict_manifest_count(manifest["n_rows"], "n_rows")
    rows_read = _strict_manifest_count(manifest["canonical_rows_read"], "canonical_rows_read")
    if len(canonical) != expected_rows or len(canonical) != rows_read:
        raise ValueError("Canonical dataset row count does not match the split manifest.")
    expected_groups = _strict_manifest_count(manifest["n_groups"], "n_groups")
    observed_groups = canonical["canonical_reaction_key"].nunique()
    if observed_groups != expected_groups:
        raise ValueError("Canonical reaction group count does not match the split manifest.")
    subset_hash = stable_hash(
        canonical[["source_row_id", "canonical_reaction_key"]]
        .sort_values("source_row_id", kind="stable")
        .to_dict(orient="records")
    )
    if subset_hash != manifest["canonical_subset_hash"]:
        raise ValueError("Canonical dataset subset hash does not match the split manifest.")


def _strict_dataframe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item") and isinstance(value.item(), bool):
        return bool(value)
    raise ValueError("Canonical role-validity values must be booleans.")


def _strict_manifest_count(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Saved split manifest {name} must be a positive integer.")
    return value


def _validate_assignments(
    canonical: pd.DataFrame,
    outer: pd.DataFrame,
    low: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    seeds = [int(value) for value in manifest["seeds"]]
    fractions = [float(value) for value in manifest["train_fractions"]]
    canonical_ids = set(canonical["source_row_id"].astype(str))
    canonical_keys = canonical.set_index("source_row_id")["canonical_reaction_key"].astype(str)

    if outer.duplicated(["seed", "source_row_id"]).any():
        raise ValueError("Outer assignments contain duplicate seed/source_row_id rows.")
    if low.duplicated(["seed", "train_fraction", "source_row_id"]).any():
        raise ValueError("Low-data assignments contain duplicate seed/fraction/source_row_id rows.")
    if set(outer["seed"]) != set(seeds) or set(low["seed"]) != set(seeds):
        raise ValueError("Assignment seeds do not exactly match the split manifest.")
    if set(low["train_fraction"]) != set(fractions):
        raise ValueError("Low-data fractions do not exactly match the split manifest.")

    outer_expected_rows = len(canonical) * len(seeds)
    low_expected_rows = outer_expected_rows * len(fractions)
    if len(outer) != outer_expected_rows or len(low) != low_expected_rows:
        raise ValueError("Saved assignments do not provide exact canonical row coverage.")

    for seed in seeds:
        outer_seed = outer.loc[outer["seed"].eq(seed)]
        if set(outer_seed["source_row_id"]) != canonical_ids:
            raise ValueError(
                "Outer assignments do not exactly cover canonical source_row_id values."
            )
        if set(outer_seed["outer_split"]) != set(OUTER_SPLITS):
            raise ValueError(
                "Every saved seed must contain non-empty train, valid, and test splits."
            )
        if not outer_seed["train_fraction"].eq(1.0).all():
            raise ValueError("Outer assignment train_fraction must be exactly 1.0.")
        expected_included = outer_seed["outer_split"].eq("train")
        if not outer_seed["included_in_training_subset"].equals(expected_included):
            raise ValueError("Outer assignment inclusion must identify the complete train split.")

        keyed = outer_seed.set_index("source_row_id")["canonical_reaction_key"]
        if not keyed.sort_index().equals(canonical_keys.sort_index()):
            raise ValueError("Outer assignment canonical keys do not match the canonical dataset.")
        _validate_group_assignments(outer_seed, canonical, seed)

        previous_ids: set[str] = set()
        outer_by_id = outer_seed.set_index("source_row_id")
        for fraction in fractions:
            fraction_rows = low.loc[low["seed"].eq(seed) & low["train_fraction"].eq(fraction)]
            if set(fraction_rows["source_row_id"]) != canonical_ids:
                raise ValueError(
                    "Low-data assignments do not exactly cover canonical source_row_id values."
                )
            low_by_id = fraction_rows.set_index("source_row_id")
            for column in (
                "canonical_reaction_key",
                "outer_split",
                "outer_group_order",
            ):
                if not low_by_id[column].sort_index().equals(outer_by_id[column].sort_index()):
                    raise ValueError(
                        f"Low-data assignment {column} does not agree with outer assignments."
                    )
            included = fraction_rows["included_in_training_subset"]
            if (included & ~fraction_rows["outer_split"].eq("train")).any():
                raise ValueError("Validation or test rows entered a low-data training subset.")
            group_inclusion = fraction_rows.groupby("canonical_reaction_key", sort=False)[
                "included_in_training_subset"
            ].nunique()
            if group_inclusion.max() != 1:
                raise ValueError("A canonical group is only partially included in a subset.")
            included_ids = set(fraction_rows.loc[included, "source_row_id"])
            if not previous_ids.issubset(included_ids):
                raise ValueError("Saved low-data training subsets are not cumulative.")
            previous_ids = included_ids
            if fraction == 1.0:
                expected_train_ids = set(
                    outer_seed.loc[outer_seed["outer_split"].eq("train"), "source_row_id"]
                )
                if included_ids != expected_train_ids:
                    raise ValueError("The saved 1.0 subset is not the complete outer train split.")


def _validate_group_assignments(
    assignments: pd.DataFrame,
    canonical: pd.DataFrame,
    seed: int,
) -> None:
    by_group = assignments.groupby("canonical_reaction_key", sort=False)
    if by_group["outer_split"].nunique().max() != 1:
        raise ValueError(f"Canonical reaction groups cross outer splits for seed {seed}.")
    if by_group["outer_group_order"].nunique().max() != 1:
        raise ValueError(f"Canonical groups have inconsistent traversal order for seed {seed}.")
    group_orders = by_group["outer_group_order"].first()
    if set(group_orders) != set(range(canonical["canonical_reaction_key"].nunique())):
        raise ValueError(
            f"Canonical outer group order is not a complete permutation for seed {seed}."
        )


def _validate_requests(
    seeds: list[int],
    fractions: list[float],
    *,
    requested_seeds: list[int] | tuple[int, ...] | None,
    requested_fractions: list[float] | tuple[float, ...] | None,
) -> None:
    if requested_seeds is not None:
        missing_seeds = sorted(set(map(int, requested_seeds)) - set(seeds))
        if missing_seeds:
            raise ValueError(f"Requested saved split seeds are absent: {missing_seeds}.")
    if requested_fractions is not None:
        for fraction in requested_fractions:
            _resolve_fraction(float(fraction), fractions)


def _resolve_fraction(requested: float, available: list[float]) -> float:
    matches = [
        fraction
        for fraction in available
        if math.isclose(requested, fraction, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"Requested train fraction is absent from saved assignments: {requested}.")
    return matches[0]


def _validate_materialization_frame(
    frame: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    if "source_row_id" not in frame:
        raise ValueError("Materialization frame is missing source_row_id.")
    if frame["source_row_id"].isna().any() or frame["source_row_id"].duplicated().any():
        raise ValueError(
            "Materialization frame source_row_id values must be unique and non-missing."
        )
    if set(frame["source_row_id"].astype(str)) != set(canonical["source_row_id"].astype(str)):
        raise ValueError(
            "Materialization frame does not exactly cover canonical source_row_id values."
        )
    if "canonical_reaction_key" not in frame:
        raise ValueError("Materialization frame is missing canonical_reaction_key.")
    observed = (
        frame.assign(source_row_id=frame["source_row_id"].astype(str))
        .set_index("source_row_id")["canonical_reaction_key"]
        .astype(str)
        .sort_index()
    )
    expected = (
        canonical.assign(source_row_id=canonical["source_row_id"].astype(str))
        .set_index("source_row_id")["canonical_reaction_key"]
        .astype(str)
        .sort_index()
    )
    if not observed.equals(expected):
        raise ValueError(
            "Materialization frame canonical keys do not match source_row_id assignments."
        )
