"""Deterministic group-safe outer splits and nested low-data subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    build_canonical_reaction_identity,
    source_row_id,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    prepare_fresh_output_directory,
    sha256_file,
    stable_hash,
    write_json,
)

SPLIT_SCHEMA_VERSION = "bh-canonical-grouped-splits-v1"
OUTER_SPLITS = ("train", "valid", "test")


def build_grouped_outer_assignments(
    df: pd.DataFrame,
    *,
    seed: int,
    train_size: float,
    valid_size: float,
    test_size: float,
    group_column: str = "canonical_reaction_key",
) -> pd.DataFrame:
    """Assign complete canonical groups to one deterministic outer split."""
    _validate_split_input(df, group_column, train_size, valid_size, test_size)
    group_sizes = df.groupby(group_column, sort=True).size()
    ordered_groups = np.array(group_sizes.index.tolist(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered_groups)

    group_to_split = _assign_groups_to_outer_splits(
        ordered_groups,
        group_sizes,
        proportions={"train": train_size, "valid": valid_size, "test": test_size},
    )
    group_order = {group: position for position, group in enumerate(ordered_groups)}

    assignments = df[["source_row_id", group_column]].copy()
    assignments["seed"] = int(seed)
    assignments["outer_split"] = assignments[group_column].map(group_to_split)
    assignments["outer_group_order"] = assignments[group_column].map(group_order).astype(int)
    if assignments["outer_split"].isna().any():
        raise AssertionError("At least one canonical group was not assigned.")
    return assignments


def build_nested_low_data_assignments(
    outer_assignments: pd.DataFrame,
    *,
    train_fractions: list[float],
    group_column: str = "canonical_reaction_key",
) -> pd.DataFrame:
    """Create cumulative complete-group prefixes of one outer training order."""
    fractions = sorted({float(value) for value in train_fractions})
    if not fractions or fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("train_fractions must contain values greater than 0 and at most 1.")
    train = outer_assignments.loc[outer_assignments["outer_split"] == "train"]
    if train.empty:
        raise ValueError("Outer assignments contain no training rows.")
    group_table = (
        train.groupby(group_column, sort=False)
        .agg(n_rows=("source_row_id", "size"), group_order=("outer_group_order", "first"))
        .sort_values("group_order", kind="stable")
    )
    ordered_groups = group_table.index.tolist()
    cumulative_rows = group_table["n_rows"].cumsum().to_numpy()
    n_train_rows = len(train)

    records: list[pd.DataFrame] = []
    previous_included: set[str] = set()
    for fraction in fractions:
        if np.isclose(fraction, 1.0):
            included_groups = set(ordered_groups)
        else:
            target = max(1, int(np.ceil(n_train_rows * fraction)))
            prefix_stop = int(np.searchsorted(cumulative_rows, target, side="left")) + 1
            included_groups = set(ordered_groups[:prefix_stop])
        if not previous_included.issubset(included_groups):
            raise AssertionError("Nested low-data group prefixes are not cumulative.")
        previous_included = included_groups

        assignment = outer_assignments.copy()
        assignment["train_fraction"] = fraction
        assignment["included_in_training_subset"] = (
            assignment["outer_split"].eq("train")
            & assignment[group_column].isin(included_groups)
        )
        records.append(assignment)
    return pd.concat(records, ignore_index=True)


def split_assignment_hash(
    outer_assignments: pd.DataFrame,
    low_data_assignments: pd.DataFrame,
) -> str:
    """Hash exact ordered row/group assignments rather than aggregate counts."""
    outer_columns = [
        "source_row_id",
        "canonical_reaction_key",
        "seed",
        "outer_split",
        "outer_group_order",
    ]
    low_columns = [
        "source_row_id",
        "canonical_reaction_key",
        "seed",
        "outer_split",
        "train_fraction",
        "included_in_training_subset",
    ]
    outer = (
        outer_assignments[outer_columns]
        .sort_values(["seed", "source_row_id"], kind="stable")
        .to_dict(orient="records")
    )
    low = (
        low_data_assignments[low_columns]
        .sort_values(["seed", "train_fraction", "source_row_id"], kind="stable")
        .to_dict(orient="records")
    )
    return stable_hash({"schema": SPLIT_SCHEMA_VERSION, "outer": outer, "low_data": low})


def run_canonical_grouped_splits(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Build and persist canonical group-safe assignments for configured seeds."""
    _validate_split_config(config)
    canonical_path = Path(config["dataset"]["path"])
    source_path = Path(config["dataset"]["source_path"])
    output_path = Path(config["output"]["directory"])
    if output_path.exists():
        raise FileExistsError(
            f"Canonical split directory already exists and will not be reused: {output_path}"
        )
    output_dir = prepare_fresh_output_directory(output_path)
    paths = _split_paths(output_dir)
    canonical = pd.read_csv(canonical_path, nrows=config["dataset"].get("nrows"))
    _validate_canonical_dataset_provenance(canonical, source_path)
    if not canonical["all_required_roles_parse_valid"].astype(bool).all():
        raise ValueError("Canonical grouped splitting requires every included row to be valid.")
    if canonical["canonical_reaction_key"].isna().any():
        raise ValueError("Canonical grouped splitting found missing canonical_reaction_key values.")

    split_config = config["splits"]
    group_column = split_config["group_column"]
    outer_frames = []
    low_frames = []
    summaries = []
    overlap_records = []
    split_hashes: dict[str, str] = {}
    for seed_value in split_config["seeds"]:
        seed = int(seed_value)
        outer = build_grouped_outer_assignments(
            canonical,
            seed=seed,
            train_size=float(split_config["train_size"]),
            valid_size=float(split_config["valid_size"]),
            test_size=float(split_config["test_size"]),
            group_column=group_column,
        )
        low = build_nested_low_data_assignments(
            outer,
            train_fractions=list(split_config["train_fractions"]),
            group_column=group_column,
        )
        split_hashes[str(seed)] = split_assignment_hash(outer, low)
        outer_frames.append(outer)
        low_frames.append(low)
        summaries.extend(_split_summary(outer, low, split_config))
        overlap_records.append(_split_overlap_record(outer, low))

    outer_all = pd.concat(outer_frames, ignore_index=True)
    outer_all["train_fraction"] = 1.0
    outer_all["included_in_training_subset"] = outer_all["outer_split"].eq("train")
    low_all = pd.concat(low_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    overlap = pd.DataFrame(overlap_records)
    if not overlap["all_outer_group_overlaps_zero"].all() or not overlap["all_subsets_nested"].all():
        raise AssertionError("Canonical split overlap or nested-subset invariant failed.")

    outer_all.to_csv(paths["outer_assignments"], index=False)
    low_all.to_csv(paths["low_data_assignments"], index=False)
    summary.to_csv(paths["summary"], index=False)
    overlap.to_csv(paths["overlap_audit"], index=False)

    manifest = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "source_dataset_path": str(source_path),
        "source_dataset_hash": sha256_file(source_path),
        "canonical_dataset_path": str(canonical_path),
        "canonical_dataset_hash": sha256_file(canonical_path),
        "canonical_rows_read": len(canonical),
        "dataset_nrows": config["dataset"].get("nrows"),
        "canonical_subset_hash": stable_hash(
            canonical[["source_row_id", "canonical_reaction_key"]]
            .sort_values("source_row_id", kind="stable")
            .to_dict(orient="records")
        ),
        "resolved_config": config,
        "config_hash": stable_hash(config),
        "config_path": str(config_path),
        "group_column": group_column,
        "seed": (
            int(split_config["seeds"][0])
            if len(split_config["seeds"]) == 1
            else None
        ),
        "seeds": [int(seed) for seed in split_config["seeds"]],
        "requested_proportions": {
            "train": float(split_config["train_size"]),
            "valid": float(split_config["valid_size"]),
            "test": float(split_config["test_size"]),
        },
        "observed_proportions": _observed_outer_proportions(outer_all),
        "train_fractions": [float(value) for value in split_config["train_fractions"]],
        "split_hash": stable_hash(split_hashes),
        "split_hashes": split_hashes,
        "n_rows": len(canonical),
        "n_groups": int(canonical[group_column].nunique()),
        "historical_results_loaded": False,
    }
    write_json(paths["manifest"], manifest)
    return paths


def _prefix_group_count(
    ordered_groups: np.ndarray,
    group_sizes: pd.Series,
    *,
    target_rows: float,
    min_remaining_groups: int,
) -> int:
    max_count = len(ordered_groups) - min_remaining_groups
    if max_count <= 0:
        raise ValueError("Too few canonical groups to create three non-empty outer splits.")
    cumulative = 0
    count = 0
    for group in ordered_groups[:max_count]:
        size = int(group_sizes.loc[group])
        if count and abs(cumulative - target_rows) <= abs(cumulative + size - target_rows):
            break
        cumulative += size
        count += 1
    return max(1, count)


def _assign_groups_to_outer_splits(
    shuffled_groups: np.ndarray,
    group_sizes: pd.Series,
    *,
    proportions: dict[str, float],
) -> dict[object, str]:
    """Allocate shuffled groups while avoiding large-group split imbalance."""
    if int(group_sizes.max()) == 1:
        n_rows = int(group_sizes.sum())
        train_count = _prefix_group_count(
            shuffled_groups,
            group_sizes,
            target_rows=proportions["train"] * n_rows,
            min_remaining_groups=2,
        )
        remaining = shuffled_groups[train_count:]
        valid_count = _prefix_group_count(
            remaining,
            group_sizes,
            target_rows=proportions["valid"] * n_rows,
            min_remaining_groups=1,
        )
        return {
            group: (
                "train"
                if position < train_count
                else "valid"
                if position < train_count + valid_count
                else "test"
            )
            for position, group in enumerate(shuffled_groups)
        }

    random_order = {group: position for position, group in enumerate(shuffled_groups)}
    allocation_order = sorted(
        shuffled_groups.tolist(),
        key=lambda group: (-int(group_sizes.loc[group]), random_order[group]),
    )
    targets = {
        split: proportions[split] * float(group_sizes.sum())
        for split in OUTER_SPLITS
    }
    assigned_rows = dict.fromkeys(OUTER_SPLITS, 0)
    assignment: dict[object, str] = {}
    for group in allocation_order:
        split = max(
            OUTER_SPLITS,
            key=lambda name: (targets[name] - assigned_rows[name]) / targets[name],
        )
        assignment[group] = split
        assigned_rows[split] += int(group_sizes.loc[group])
    if set(assignment.values()) != set(OUTER_SPLITS):
        raise ValueError("Canonical groups could not create three non-empty outer splits.")
    return assignment


def _split_summary(
    outer: pd.DataFrame,
    low: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    seed = int(outer["seed"].iloc[0])
    n_rows = len(outer)
    records: list[dict[str, Any]] = []
    requested = {
        "train": float(config["train_size"]),
        "valid": float(config["valid_size"]),
        "test": float(config["test_size"]),
    }
    for split in OUTER_SPLITS:
        rows = outer.loc[outer["outer_split"] == split]
        records.append(
            {
                "seed": seed,
                "summary_type": "outer_split",
                "outer_split": split,
                "requested_fraction": requested[split],
                "observed_fraction": len(rows) / n_rows,
                "n_rows": len(rows),
                "n_groups": rows["canonical_reaction_key"].nunique(),
            }
        )
    train_pool_rows = int(outer["outer_split"].eq("train").sum())
    for fraction, subset in low.groupby("train_fraction", sort=True):
        included = subset.loc[subset["included_in_training_subset"]]
        records.append(
            {
                "seed": seed,
                "summary_type": "low_data_subset",
                "outer_split": "train",
                "requested_fraction": float(fraction),
                "observed_fraction": len(included) / train_pool_rows,
                "n_rows": len(included),
                "n_groups": included["canonical_reaction_key"].nunique(),
            }
        )
    return records


def _split_overlap_record(outer: pd.DataFrame, low: pd.DataFrame) -> dict[str, Any]:
    groups = {
        split: set(outer.loc[outer["outer_split"] == split, "canonical_reaction_key"])
        for split in OUTER_SPLITS
    }
    fractions = sorted(low["train_fraction"].unique().tolist())
    subset_ids = {
        fraction: set(
            low.loc[
                low["train_fraction"].eq(fraction)
                & low["included_in_training_subset"],
                "source_row_id",
            ]
        )
        for fraction in fractions
    }
    nested = all(
        subset_ids[left].issubset(subset_ids[right])
        for left, right in zip(fractions, fractions[1:], strict=False)
    )
    return {
        "seed": int(outer["seed"].iloc[0]),
        "train_valid_group_overlap": len(groups["train"] & groups["valid"]),
        "train_test_group_overlap": len(groups["train"] & groups["test"]),
        "valid_test_group_overlap": len(groups["valid"] & groups["test"]),
        "all_outer_group_overlaps_zero": not (
            groups["train"] & groups["valid"]
            or groups["train"] & groups["test"]
            or groups["valid"] & groups["test"]
        ),
        "all_rows_assigned_once": outer["source_row_id"].nunique() == len(outer),
        "all_subsets_nested": nested,
        "validation_row_hash": stable_hash(
            sorted(outer.loc[outer["outer_split"] == "valid", "source_row_id"])
        ),
        "test_row_hash": stable_hash(
            sorted(outer.loc[outer["outer_split"] == "test", "source_row_id"])
        ),
    }


def _observed_outer_proportions(assignments: pd.DataFrame) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    for seed, group in assignments.groupby("seed", sort=True):
        records[str(seed)] = {
            split: float(group["outer_split"].eq(split).mean())
            for split in OUTER_SPLITS
        }
    return records


def _validate_split_input(
    df: pd.DataFrame,
    group_column: str,
    train_size: float,
    valid_size: float,
    test_size: float,
) -> None:
    required = {"source_row_id", group_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Grouped split input is missing columns: {missing}.")
    if df.empty:
        raise ValueError("Cannot split an empty canonical dataset.")
    if df["source_row_id"].duplicated().any():
        raise ValueError("source_row_id values must be unique.")
    if df[group_column].isna().any():
        raise ValueError(f"Group column contains missing values: {group_column}.")
    for name, value in {
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1.")
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("Outer train/valid/test proportions must sum to 1.0.")


def _validate_split_config(config: dict[str, Any]) -> None:
    split_config = config.get("splits", {})
    if split_config.get("group_column") != "canonical_reaction_key":
        raise ValueError("Batch 3 grouped splits require group_column: canonical_reaction_key.")
    if not config.get("scientific_run", False):
        raise ValueError("Canonical grouped split construction requires scientific_run: true.")
    if config.get("allow_hash_fingerprint_fallback", True):
        raise ValueError("allow_hash_fingerprint_fallback must be false.")
    canonicalization = config.get("canonicalization", {})
    if canonicalization.get("backend") != "rdkit":
        raise ValueError("Canonical grouped splits require canonicalization.backend: rdkit.")
    if not canonicalization.get("preserve_role_order", False):
        raise ValueError("canonicalization.preserve_role_order must be true.")
    if canonicalization.get("isomeric_smiles") is not True:
        raise ValueError("Canonical grouped splits require isomeric_smiles: true.")


def _validate_canonical_dataset_provenance(
    canonical: pd.DataFrame,
    source_path: Path,
) -> None:
    required = {
        "source_row_id",
        "source_row_position",
        "canonicalization_version",
    }
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ValueError(f"Canonical dataset is missing provenance columns: {missing}.")
    versions = set(canonical["canonicalization_version"].dropna().astype(str))
    if versions != {CANONICALIZATION_VERSION}:
        raise ValueError(
            "Canonical dataset version mismatch: "
            f"expected {CANONICALIZATION_VERSION!r}, observed {sorted(versions)!r}."
        )
    source_hash = sha256_file(source_path)
    positions = pd.to_numeric(canonical["source_row_position"], errors="raise").astype(int)
    expected_ids = [
        source_row_id(source_hash, position)
        for position in positions.tolist()
    ]
    if canonical["source_row_id"].tolist() != expected_ids:
        raise ValueError(
            "Canonical dataset source_row_id values do not match the configured "
            "source-file hash and saved original row positions."
        )
    identity_records = [
        build_canonical_reaction_identity(row)
        for row in canonical.to_dict(orient="records")
    ]
    expected_keys = [record["key"] for record in identity_records]
    expected_hashes = [record["hash"] for record in identity_records]
    if canonical["canonical_reaction_key"].tolist() != expected_keys:
        raise ValueError("Saved canonical_reaction_key values do not match canonical roles.")
    if canonical["canonical_reaction_hash"].tolist() != expected_hashes:
        raise ValueError("Saved canonical_reaction_hash values do not match canonical roles.")


def _split_paths(directory: Path) -> dict[str, Path]:
    return {
        "outer_assignments": directory / "outer_split_assignments.csv",
        "low_data_assignments": directory / "low_data_subset_assignments.csv",
        "summary": directory / "split_summary.csv",
        "overlap_audit": directory / "split_overlap_audit.csv",
        "manifest": directory / "split_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    paths = run_canonical_grouped_splits(load_config(args.config), config_path=args.config)
    print(f"Grouped split assignments: {paths['outer_assignments']}")
    print(f"Split manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
