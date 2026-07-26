"""CLI and reusable audit for canonical seven-role Buchwald-Hartwig data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    canonicalize_reaction_roles_dataframe,
)
from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES, ROLE_TO_COLUMN
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    build_run_manifest,
    feature_contract_record,
    prepare_fresh_output_directory,
    sha256_file,
    write_json,
)

EXACT_DUPLICATE_METADATA_COLUMNS = (
    "product_smiles",
    "reactant_1_smiles",
    "reactant_2_smiles",
    "catalyst_smiles",
    "solvent",
    "temperature",
    "product_key",
    "reactant_key",
)


def run_canonical_data_audit(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Path]:
    """Canonicalize source rows and write non-destructive duplicate audits."""
    _validate_audit_config(config)
    source_path = Path(config["dataset"]["path"])
    canonical_path = Path(config["output"]["canonical_dataset_path"])
    output_dir = Path(config["output"]["directory"])
    if canonical_path.exists():
        raise FileExistsError(
            f"Canonical dataset already exists and will not be overwritten: {canonical_path}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Canonical audit directory already exists and will not be reused: {output_dir}"
        )
    output_dir = prepare_fresh_output_directory(output_dir)
    paths = _audit_paths(output_dir)

    source_hash = sha256_file(source_path)
    frame = pd.read_csv(source_path, nrows=config["dataset"].get("nrows"))
    canonical = canonicalize_reaction_roles_dataframe(
        frame,
        source_file_hash=source_hash,
        isomeric=bool(config["canonicalization"].get("isomeric_smiles", True)),
        source_row_positions=range(len(frame)),
    )
    valid = canonical["all_required_roles_parse_valid"].astype(bool)
    invalid_molecules = _invalid_molecule_rows(canonical)
    invalid_reactions = canonical.loc[~valid].copy()

    canonical_valid = canonical.loc[valid].copy()
    group_stats = canonical_reaction_group_statistics(canonical_valid)
    exact_duplicates = exact_duplicate_rows(canonical)
    canonical_duplicates = group_stats.loc[group_stats["replicate_count"] > 1].copy()
    replicate_groups = canonical_duplicates.copy()
    yield_conflicts = group_stats.loc[group_stats["n_unique_yields"] > 1].copy()
    threshold = float(
        config.get("duplicate_audit", {}).get("yield_conflict_reporting_threshold", 10.0)
    )
    yield_conflicts["exceeds_reporting_threshold"] = (
        yield_conflicts["yield_range"] >= threshold
    )

    feature_duplicates, feature_contract = feature_duplicate_audit(
        canonical_valid,
        config["features"],
    )
    role_counts = canonical_role_value_counts(canonical)
    summary = canonicalization_summary(
        canonical,
        group_stats=group_stats,
        exact_duplicates=exact_duplicates,
        feature_duplicates=feature_duplicates,
    )
    summary["yield_conflict_reporting_threshold"] = threshold
    summary["exclusion_policy"] = config["duplicate_audit"].get("exclusion_policy", "none")

    invalid_molecules.to_csv(paths["invalid_molecules"], index=False)
    invalid_reactions.to_csv(paths["invalid_reactions"], index=False)
    exact_duplicates.to_csv(paths["exact_duplicate_rows"], index=False)
    canonical_duplicates.to_csv(paths["canonical_duplicate_groups"], index=False)
    replicate_groups.to_csv(paths["replicate_groups"], index=False)
    yield_conflicts.to_csv(paths["yield_conflict_groups"], index=False)
    feature_duplicates.to_csv(paths["feature_duplicate_groups"], index=False)
    role_counts.to_csv(paths["canonical_role_value_counts"], index=False)
    pd.DataFrame([summary]).to_csv(paths["summary_csv"], index=False)
    write_json(paths["summary_json"], summary)
    paths["report"].write_text(_audit_report(summary), encoding="utf-8")

    construction_failed = bool(
        config.get("scientific_run", False) and invalid_reactions.shape[0]
    )
    canonical_hash = None
    if not construction_failed:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(canonical_path, index=False)
        canonical_hash = sha256_file(canonical_path)
    manifest = build_run_manifest(
        config=config,
        config_path=config_path,
        dataset_path=source_path,
        dataset_hash=source_hash,
        output_directory=output_dir,
        split_hashes={},
        feature_metadata_hash=feature_contract["feature_metadata_hash"],
    )
    manifest.update(
        {
            "canonical_dataset_path": str(canonical_path),
            "canonical_dataset_hash": canonical_hash,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "rows_removed": 0,
            "yields_aggregated": False,
            "construction_failed": construction_failed,
        }
    )
    write_json(paths["run_manifest"], manifest)
    if construction_failed:
        raise ValueError(
            f"Scientific canonical dataset construction found {len(invalid_reactions)} rows "
            "with invalid required molecules. A complete audit was written, but no canonical "
            "dataset was created. Define and document an exclusion policy before proceeding."
        )
    return {**paths, "canonical_dataset": canonical_path}


def canonical_reaction_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize measured replicates without removing or aggregating rows."""
    columns = [
        "canonical_reaction_key",
        "canonical_reaction_hash",
        "replicate_count",
        "yield_count",
        "yield_mean",
        "yield_median",
        "yield_std",
        "yield_min",
        "yield_max",
        "yield_range",
        "n_unique_yields",
        "source_row_ids",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for key, group in df.groupby("canonical_reaction_key", sort=True, dropna=False):
        yields = pd.to_numeric(group["yield"], errors="coerce")
        records.append(
            {
                "canonical_reaction_key": key,
                "canonical_reaction_hash": group["canonical_reaction_hash"].iloc[0],
                "replicate_count": len(group),
                "yield_count": int(yields.count()),
                "yield_mean": float(yields.mean()),
                "yield_median": float(yields.median()),
                "yield_std": float(yields.std(ddof=0)),
                "yield_min": float(yields.min()),
                "yield_max": float(yields.max()),
                "yield_range": float(yields.max() - yields.min()),
                "n_unique_yields": int(yields.nunique(dropna=True)),
                "source_row_ids": json.dumps(sorted(group["source_row_id"].tolist())),
            }
        )
    return pd.DataFrame(records, columns=columns)


def exact_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows duplicated in raw chemistry, yield, and source metadata."""
    signature = _exact_duplicate_signature(df)
    if df.empty:
        return df.copy()
    duplicate_mask = df.duplicated(subset=signature, keep=False)
    return df.loc[duplicate_mask].sort_values(signature, kind="stable")


def feature_duplicate_audit(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hash canonical role-separated vectors and report equality/collisions."""
    columns = [
        "feature_vector_hash",
        "n_rows",
        "n_canonical_reactions",
        "collision_type",
        "canonical_reaction_hashes",
        "source_row_ids",
    ]
    if df.empty:
        empty_frame = df.copy()
        empty_frame["yield"] = pd.Series(dtype=float)
        _, _, names, metadata = build_feature_matrix_with_metadata(
            empty_frame,
            feature_config,
        )
        return pd.DataFrame(columns=columns), feature_contract_record(metadata, names)

    feature_frame = df.copy()
    for role, recovered_column in ROLE_TO_COLUMN.items():
        feature_frame[recovered_column] = feature_frame[f"canonical_{role}_smiles"]
    X, _, names, metadata = build_feature_matrix_with_metadata(feature_frame, feature_config)
    hashes = [
        hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()
        for row in X
    ]
    work = df[["source_row_id", "canonical_reaction_hash"]].copy()
    work["feature_vector_hash"] = hashes
    records: list[dict[str, Any]] = []
    for vector_hash, group in work.groupby("feature_vector_hash", sort=True):
        if len(group) <= 1:
            continue
        canonical_hashes = sorted(group["canonical_reaction_hash"].unique().tolist())
        records.append(
            {
                "feature_vector_hash": vector_hash,
                "n_rows": len(group),
                "n_canonical_reactions": len(canonical_hashes),
                "collision_type": (
                    "same_canonical_reaction"
                    if len(canonical_hashes) == 1
                    else "different_canonical_reactions_same_feature_vector"
                ),
                "canonical_reaction_hashes": json.dumps(canonical_hashes),
                "source_row_ids": json.dumps(sorted(group["source_row_id"].tolist())),
            }
        )
    return pd.DataFrame(records, columns=columns), feature_contract_record(metadata, names)


def canonical_role_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid/invalid and unique canonical identity counts by role."""
    records = []
    for role in CANONICAL_ROLE_NAMES:
        valid = df[f"{role}_parse_valid"].astype(bool)
        records.append(
            {
                "role": role,
                "n_valid": int(valid.sum()),
                "n_invalid": int((~valid).sum()),
                "n_unique_canonical_values": int(
                    df.loc[valid, f"canonical_{role}_smiles"].nunique()
                ),
            }
        )
    return pd.DataFrame(records)


def canonicalization_summary(
    df: pd.DataFrame,
    *,
    group_stats: pd.DataFrame,
    exact_duplicates: pd.DataFrame,
    feature_duplicates: pd.DataFrame,
) -> dict[str, Any]:
    """Build stable headline counts for the canonical data audit."""
    valid = df["all_required_roles_parse_valid"].astype(bool)
    invalid_by_role = {
        role: int((~df[f"{role}_parse_valid"].astype(bool)).sum())
        for role in CANONICAL_ROLE_NAMES
    }
    collision_mask = feature_duplicates.get(
        "collision_type", pd.Series(dtype=str)
    ).eq("different_canonical_reactions_same_feature_vector")
    return {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "source_row_count": len(df),
        "valid_canonical_row_count": int(valid.sum()),
        "invalid_row_count": int((~valid).sum()),
        "invalid_count_by_role": invalid_by_role,
        "n_unique_canonical_reactions": len(group_stats),
        "n_exact_duplicate_rows": len(exact_duplicates),
        "n_exact_duplicate_groups": _duplicate_signature_group_count(exact_duplicates),
        "n_canonical_duplicate_groups": int((group_stats["replicate_count"] > 1).sum()),
        "n_replicate_groups": int((group_stats["replicate_count"] > 1).sum()),
        "n_yield_conflict_groups": int((group_stats["n_unique_yields"] > 1).sum()),
        "yield_range_distribution": _yield_range_distribution(group_stats),
        "n_feature_duplicate_groups": len(feature_duplicates),
        "n_feature_hash_collision_groups": int(collision_mask.sum()),
        "rows_removed": 0,
        "yields_aggregated": False,
    }


def _invalid_molecule_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for role in CANONICAL_ROLE_NAMES:
            if not bool(row[f"{role}_parse_valid"]):
                records.append(
                    {
                        "source_row_id": row["source_row_id"],
                        "source_row_position": row["source_row_position"],
                        "role": role,
                        "raw_smiles": row[f"raw_{role}_smiles"],
                        "parse_error": row[f"{role}_parse_error"],
                    }
                )
    return pd.DataFrame(
        records,
        columns=[
            "source_row_id",
            "source_row_position",
            "role",
            "raw_smiles",
            "parse_error",
        ],
    )


def _duplicate_signature_group_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    signature = _exact_duplicate_signature(df)
    return int(df.groupby(signature, dropna=False).ngroups)


def _exact_duplicate_signature(df: pd.DataFrame) -> list[str]:
    base = [
        *[f"raw_{role}_smiles" for role in CANONICAL_ROLE_NAMES],
        "reaction_smiles",
        "yield",
    ]
    return [*base, *[column for column in EXACT_DUPLICATE_METADATA_COLUMNS if column in df]]


def _yield_range_distribution(group_stats: pd.DataFrame) -> dict[str, float]:
    ranges = group_stats["yield_range"] if not group_stats.empty else pd.Series(dtype=float)
    return {
        "min": float(ranges.min()) if not ranges.empty else 0.0,
        "median": float(ranges.median()) if not ranges.empty else 0.0,
        "mean": float(ranges.mean()) if not ranges.empty else 0.0,
        "max": float(ranges.max()) if not ranges.empty else 0.0,
    }


def _audit_report(summary: dict[str, Any]) -> str:
    invalid_lines = "\n".join(
        f"  - {role}: {count}"
        for role, count in sorted(summary["invalid_count_by_role"].items())
    )
    yield_ranges = summary["yield_range_distribution"]
    return (
        "# Canonical Buchwald-Hartwig Data Audit\n\n"
        f"- Source rows: {summary['source_row_count']}\n"
        f"- Valid canonical rows: {summary['valid_canonical_row_count']}\n"
        f"- Invalid rows: {summary['invalid_row_count']}\n"
        f"- Invalid molecules by role:\n{invalid_lines}\n"
        f"- Unique canonical reactions: {summary['n_unique_canonical_reactions']}\n"
        f"- Exact duplicate groups: {summary['n_exact_duplicate_groups']}\n"
        f"- Canonical duplicate groups: {summary['n_canonical_duplicate_groups']}\n"
        f"- Replicate groups: {summary['n_replicate_groups']}\n"
        f"- Yield-conflict groups: {summary['n_yield_conflict_groups']}\n"
        "- Yield-range distribution across canonical groups: "
        f"min={yield_ranges['min']:.6g}, median={yield_ranges['median']:.6g}, "
        f"mean={yield_ranges['mean']:.6g}, max={yield_ranges['max']:.6g}\n"
        f"- Feature duplicate groups: {summary['n_feature_duplicate_groups']}\n"
        f"- Feature-hash collision groups: {summary['n_feature_hash_collision_groups']}\n"
        "- Rows removed: 0\n"
        "- Yields aggregated: false\n\n"
        "Canonical molecular equivalence, feature-vector equality, experimental "
        "replication, and yield conflict are reported separately and must not be "
        "interpreted as interchangeable concepts.\n"
    )


def _validate_audit_config(config: dict[str, Any]) -> None:
    canonicalization = config.get("canonicalization", {})
    if canonicalization.get("backend") != "rdkit":
        raise ValueError("Canonical scientific data construction requires backend: rdkit.")
    if not canonicalization.get("preserve_role_order", False):
        raise ValueError("canonicalization.preserve_role_order must be true.")
    if canonicalization.get("isomeric_smiles") is not True:
        raise ValueError("Scientific canonicalization requires isomeric_smiles: true.")
    if config.get("allow_hash_fingerprint_fallback", True):
        raise ValueError("allow_hash_fingerprint_fallback must be false.")
    expected_policy = "audit_then_error_for_scientific_dataset"
    if config.get("invalid_smiles_policy") != expected_policy:
        raise ValueError(f"invalid_smiles_policy must be {expected_policy!r}.")
    if config.get("duplicate_audit", {}).get("exclusion_policy", "none") != "none":
        raise ValueError("Batch 3 supports duplicate_audit.exclusion_policy: none only.")
    features = config.get("features", {})
    if features.get("kind") != "bh_role_separated":
        raise ValueError("Feature-identity audit requires features.kind: bh_role_separated.")
    if features.get("fingerprint_backend") != "rdkit":
        raise ValueError("Feature-identity audit requires fingerprint_backend: rdkit.")


def _audit_paths(directory: Path) -> dict[str, Path]:
    return {
        "summary_json": directory / "canonicalization_summary.json",
        "summary_csv": directory / "canonicalization_summary.csv",
        "invalid_molecules": directory / "invalid_molecules.csv",
        "invalid_reactions": directory / "invalid_reactions.csv",
        "exact_duplicate_rows": directory / "exact_duplicate_rows.csv",
        "canonical_duplicate_groups": directory / "canonical_duplicate_groups.csv",
        "replicate_groups": directory / "replicate_groups.csv",
        "yield_conflict_groups": directory / "yield_conflict_groups.csv",
        "feature_duplicate_groups": directory / "feature_duplicate_groups.csv",
        "canonical_role_value_counts": directory / "canonical_role_value_counts.csv",
        "report": directory / "data_audit_report.md",
        "run_manifest": directory / "run_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = run_canonical_data_audit(config, config_path=args.config)
    print(f"Canonical dataset: {paths['canonical_dataset']}")
    print(f"Audit directory: {Path(paths['run_manifest']).parent}")


if __name__ == "__main__":
    main()
