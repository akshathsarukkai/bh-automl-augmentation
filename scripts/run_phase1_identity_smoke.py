"""Tiny deterministic smoke test for canonical synthetic-candidate auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    assert_accepted_identity_invariants,
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
    train_ids = assignments.loc[
        (assignments["seed"] == 0) & assignments["outer_split"].eq("train"),
        "source_row_id",
    ]
    train = (
        measured.loc[measured["source_row_id"].isin(set(train_ids))]
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
    X_train, y_train, feature_names, feature_metadata = (
        build_feature_matrix_with_metadata(train, feature_config)
    )
    config = ConditionTransferConfig(
        donor_strategy="random",
        synthetic_multiplier=0.1,
        n_neighbors=3,
        label_strategy="average_label",
        teacher_models=["ridge"],
        max_teacher_std=None,
        min_similarity=None,
        high_yield_threshold=70.0,
        clip_y_min=0.0,
        clip_y_max=100.0,
        candidates_per_real=3,
        random_state=19,
        donor_similarity_n_bits=64,
        donor_similarity_backend="rdkit",
    )
    # Candidate scope: globally_unmeasured_prospective. This smoke deliberately
    # rejects against the COMPLETE measured dataset, not against the rows the
    # training slice observed. Its zero-accepted result therefore measures global
    # discovery headroom on a near-complete factorial matrix; it is not evidence
    # about how much chemistry a low-data learner could legitimately generate.
    # See src/bh_augmentation/augmentation/candidate_scope.py.
    all_measured_keys = measured_canonical_keys(measured)
    kwargs = {
        "feature_config": feature_config,
        "real_feature_names": feature_names,
        "real_feature_metadata": feature_metadata,
        "measured_identity_keys": all_measured_keys,
    }
    first = generate_condition_transfer_examples(
        train,
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        config,
        **kwargs,
    )
    second = generate_condition_transfer_examples(
        train,
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        config,
        **kwargs,
    )
    first_audit = first["candidate_df"]
    second_audit = second["candidate_df"]
    assert_accepted_identity_invariants(first_audit)
    identity_columns = [
        "canonical_reaction_hash",
        "feature_hash",
        "accepted",
        "kept",
    ]
    if first_audit[identity_columns].to_dict(orient="records") != second_audit[
        identity_columns
    ].to_dict(orient="records"):
        raise AssertionError("Deterministic regeneration changed candidate hashes.")
    accepted = first_audit.loc[first_audit["accepted"].astype(bool)]
    if accepted["already_measured"].astype(bool).any():
        raise AssertionError("Smoke output contains measured chemistry.")
    if accepted["canonical_reaction_key"].duplicated().any():
        raise AssertionError("Smoke output contains duplicate synthetic chemistry.")
    if accepted["feature_hash"].duplicated().any():
        raise AssertionError("Smoke output contains duplicate synthetic features.")
    if not set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(first_audit):
        raise AssertionError("Smoke output is missing required candidate audit fields.")

    audit_path = output / "candidate_identity_audit.csv"
    serializable = first_audit.drop(
        columns=["reaction_roles", "canonical_reaction_roles"],
        errors="ignore",
    )
    serializable.to_csv(audit_path, index=False)
    split_manifest = json.loads(split_manifest_path.read_text())
    manifest = {
        "schema_version": "phase-01-identity-smoke-v1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "split_assignment_path": str(assignment_path),
        "split_assignment_sha256": _sha256_file(assignment_path),
        "split_aggregate_hash": split_manifest["split_hash"],
        "seed": 19,
        "n_train_sources": len(train),
        "n_candidates": len(first_audit),
        "n_accepted": len(accepted),
        "n_kept": int(first_audit["kept"].astype(bool).sum()),
        "rejection_reason_counts": {
            str(reason): int(count)
            for reason, count in first_audit["rejection_reason"]
            .fillna("accepted")
            .value_counts()
            .sort_index()
            .items()
        },
        "accepted_canonical_hashes": accepted[
            "canonical_reaction_hash"
        ].tolist(),
        "accepted_feature_hashes": accepted["feature_hash"].tolist(),
        "deterministic_regeneration": True,
        "warning": (
            "Canonical HTE subset produced no novel accepted condition transfers; "
            "all candidate rejection identities were still validated."
            if accepted.empty
            else None
        ),
        "candidate_audit_sha256": _sha256_file(audit_path),
    }
    manifest_path = output / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"audit": audit_path, "manifest": manifest_path}


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
    paths = run_smoke(args.output_directory)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
