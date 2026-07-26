"""Canonical-data smoke test for deterministic ranking and source caps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_RANKING_FIELDS,
    REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
    measured_canonical_keys,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata


def run_smoke(output_directory: str | Path) -> dict[str, Path]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)

    dataset_path = Path("data/processed/bh_canonical_roles_v1.csv")
    assignment_path = Path(
        "results/corrected_canonical_splits/outer_split_assignments.csv"
    )
    split_manifest_path = Path("results/corrected_canonical_splits/split_manifest.json")
    measured = pd.read_csv(dataset_path)
    assignments = pd.read_csv(assignment_path)
    train_ids = set(
        assignments.loc[
            (assignments["seed"] == 0)
            & assignments["outer_split"].eq("train"),
            "source_row_id",
        ]
    )
    train = (
        measured.loc[measured["source_row_id"].isin(train_ids)]
        .sort_values("source_row_id", kind="stable")
        .head(96)
        .copy()
    )
    feature_config = {
        "kind": "bh_role_separated",
        "n_bits": 16,
        "radius": 2,
        "fingerprint_backend": "rdkit",
        "categorical_columns": [],
    }
    measured_keys = measured_canonical_keys(measured)
    first = _generate(
        train,
        feature_config=feature_config,
        measured_keys=measured_keys,
        min_similarity=None,
    )
    permuted = _generate(
        train.sample(frac=1.0, random_state=37),
        feature_config=feature_config,
        measured_keys=measured_keys,
        min_similarity=None,
    )
    required = {
        *REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
        *REQUIRED_SYNTHETIC_RANKING_FIELDS,
    }
    if not required <= set(first):
        raise AssertionError(f"Smoke audit is missing fields: {sorted(required - set(first))}")
    if first.empty:
        raise AssertionError("Canonical typed smoke produced no candidate proposals.")
    counts = first.groupby("source_row_id").size()
    if counts.max() > 2:
        raise AssertionError("Canonical typed smoke violated its per-source cap.")
    if not np.isfinite(first["nearest_training_support_distance"]).all():
        raise AssertionError("Canonical typed smoke has invalid support distances.")
    if not first["calibrated_uncertainty"].isna().all():
        raise AssertionError("Phase 3 must not claim calibrated uncertainty.")
    if not first["uncertainty_rank_basis"].str.endswith("phase12_pending").all():
        raise AssertionError("Uncalibrated ranking proxy is not labeled transparently.")
    comparison_columns = [
        "source_row_id",
        "donor_row_id",
        "canonical_reaction_hash",
        "feature_hash",
        "rejection_reason",
        *REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
        *REQUIRED_SYNTHETIC_RANKING_FIELDS,
    ]
    if _records(first, comparison_columns) != _records(permuted, comparison_columns):
        raise AssertionError("Input traversal order changed canonical candidate ranking.")

    thresholded = _generate(
        train,
        feature_config=feature_config,
        measured_keys=measured_keys,
        min_similarity=2.0,
    )
    if not thresholded.empty:
        raise AssertionError("Impossible final-donor threshold generated candidates.")

    audit_path = output / "ranked_candidate_audit.csv"
    _serializable(first).to_csv(audit_path, index=False)
    split_manifest = json.loads(split_manifest_path.read_text())
    accepted = first.loc[first["accepted"].astype(bool)]
    manifest = {
        "schema_version": "phase-03-ranking-smoke-v1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "split_assignment_path": str(assignment_path),
        "split_assignment_sha256": _sha256_file(assignment_path),
        "split_aggregate_hash": split_manifest["split_hash"],
        "seed": 31,
        "n_train_sources": len(train),
        "max_candidates_per_source": 2,
        "max_observed_candidates_per_source": int(counts.max()),
        "n_candidates": len(first),
        "n_accepted": len(accepted),
        "n_kept": int(first["kept"].astype(bool).sum()),
        "input_permutation_invariant": True,
        "impossible_threshold_generated_zero": True,
        "calibrated_uncertainty_available": False,
        "uncertainty_rank_basis": sorted(
            first["uncertainty_rank_basis"].astype(str).unique().tolist()
        ),
        "support_distance_metric": sorted(
            first["support_distance_metric"].astype(str).unique().tolist()
        ),
        "candidate_audit_sha256": _sha256_file(audit_path),
        "warning": (
            "The dense canonical HTE grid produced no novel accepted transfers; "
            "proposal ranking, source caps, identities, and support fields remain "
            "fully audited."
            if accepted.empty
            else None
        ),
    }
    manifest_path = output / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"audit": audit_path, "manifest": manifest_path}


def _generate(
    train: pd.DataFrame,
    *,
    feature_config: dict[str, object],
    measured_keys: set[str],
    min_similarity: float | None,
) -> pd.DataFrame:
    X, y, names, metadata = build_feature_matrix_with_metadata(train, feature_config)
    config = RoleAwareConditionTransferConfig(
        role_transfer_mode="ligand_base",
        donor_strategy="random",
        label_strategy="source_label",
        synthetic_multiplier=0.1,
        max_candidates_per_source=2,
        teacher_models=["ridge"],
        max_teacher_std=None,
        min_similarity=min_similarity,
        random_state=31,
        max_resample_attempts=4,
        donor_similarity_n_bits=64,
        donor_similarity_backend="rdkit",
        role_change_requirement="all",
        fallback_policy="reject",
    )
    result = generate_role_aware_condition_transfer_examples(
        train,
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        config,
        feature_config=feature_config,
        real_feature_names=names,
        real_feature_metadata=metadata,
        measured_identity_keys=measured_keys,
    )
    return result["candidate_df"]


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    ordered = frame.sort_values(
        ["canonical_reaction_key", "source_row_id", "donor_row_id"],
        kind="mergesort",
    )
    return ordered[columns].replace({np.nan: None}).to_dict(orient="records")


def _serializable(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=["reaction_roles", "canonical_reaction_roles"],
        errors="ignore",
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory")
    args = parser.parse_args()
    for name, path in run_smoke(args.output_directory).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
