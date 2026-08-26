"""Canonical-data smoke test for exact role changes and declared fallbacks."""

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
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import (
    assert_accepted_role_change_invariants,
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
    # Candidate scope: globally_unmeasured_prospective. This smoke deliberately
    # rejects against the COMPLETE measured dataset, not against the rows the
    # training slice observed. Its zero-accepted result therefore measures global
    # discovery headroom on a near-complete factorial matrix; it is not evidence
    # about how much chemistry a low-data learner could legitimately generate.
    # See src/bh_augmentation/augmentation/candidate_scope.py.
    all_measured_keys = measured_canonical_keys(measured)
    anonymous = _run_anonymous_strict(
        train,
        feature_config=feature_config,
        measured_keys=all_measured_keys,
    )
    assert_accepted_role_change_invariants(anonymous)
    accepted = anonymous.loc[anonymous["accepted"].astype(bool)]
    if not accepted.empty:
        if not accepted["role_change_valid"].astype(bool).all():
            raise AssertionError("Strict anonymous smoke accepted an invalid role change.")
        if accepted["unchanged_requested_roles"].fillna("").astype(str).str.len().gt(0).any():
            raise AssertionError("Strict anonymous smoke left a requested role unchanged.")
        if accepted["unexpected_changed_roles"].fillna("").astype(str).str.len().gt(0).any():
            raise AssertionError("Strict anonymous smoke changed an unrequested role.")

    pair = _fallback_pair(train)
    rejected = _run_typed_fallback(
        pair,
        feature_config=feature_config,
        measured_keys=all_measured_keys,
        fallback_policy="reject",
    )
    if not rejected.empty:
        raise AssertionError("Declared reject fallback unexpectedly generated a candidate.")
    random_fallback = _run_typed_fallback(
        pair,
        feature_config=feature_config,
        measured_keys=all_measured_keys,
        fallback_policy="random",
    )
    if random_fallback.empty:
        raise AssertionError("Declared random fallback generated no auditable candidate.")
    if set(random_fallback["donor_fallback_level"]) != {"fallback_random"}:
        raise AssertionError("Typed smoke used a fallback other than declared random.")
    if not random_fallback["fallback_used"].astype(bool).all():
        raise AssertionError("Typed smoke failed to mark its declared fallback.")
    if not random_fallback["role_change_valid"].astype(bool).all():
        raise AssertionError("Typed fallback violated exact role-change semantics.")

    anonymous_path = output / "anonymous_strict_audit.csv"
    typed_path = output / "typed_fallback_audit.csv"
    _serializable(anonymous).to_csv(anonymous_path, index=False)
    _serializable(random_fallback).to_csv(typed_path, index=False)
    split_manifest = json.loads(split_manifest_path.read_text())
    manifest = {
        "schema_version": "phase-02-role-change-smoke-v1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "split_assignment_path": str(assignment_path),
        "split_assignment_sha256": _sha256_file(assignment_path),
        "split_aggregate_hash": split_manifest["split_hash"],
        "n_anonymous_sources": len(train),
        "n_anonymous_candidates": len(anonymous),
        "n_anonymous_accepted": len(accepted),
        "anonymous_role_change_requirement": "all",
        "anonymous_fallback_policy": "reject",
        "anonymous_requested_role_cardinalities": {
            role: int(train[f"canonical_{role}_smiles"].nunique(dropna=False))
            for role in [
                "catalyst",
                "ligand",
                "base",
                "solvent_or_additive",
            ]
        },
        "anonymous_limitation": (
            "No strict full-condition donor is eligible because at least one "
            "requested role is invariant in this canonical HTE subset."
            if anonymous.empty
            else None
        ),
        "typed_pair_source_row_ids": pair["source_row_id"].astype(str).tolist(),
        "n_typed_fallback_candidates": len(random_fallback),
        "typed_fallback_policy": "random",
        "strict_reject_generated_zero": True,
        "anonymous_audit_sha256": _sha256_file(anonymous_path),
        "typed_audit_sha256": _sha256_file(typed_path),
    }
    manifest_path = output / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "anonymous_audit": anonymous_path,
        "typed_audit": typed_path,
        "manifest": manifest_path,
    }


def _run_anonymous_strict(
    train: pd.DataFrame,
    *,
    feature_config: dict[str, object],
    measured_keys: set[str],
) -> pd.DataFrame:
    X, y, names, metadata = build_feature_matrix_with_metadata(train, feature_config)
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
        random_state=23,
        donor_similarity_n_bits=64,
        donor_similarity_backend="rdkit",
        role_change_requirement="all",
        fallback_policy="reject",
    )
    result = generate_condition_transfer_examples(
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


def _run_typed_fallback(
    pair: pd.DataFrame,
    *,
    feature_config: dict[str, object],
    measured_keys: set[str],
    fallback_policy: str,
) -> pd.DataFrame:
    X, y, names, metadata = build_feature_matrix_with_metadata(pair, feature_config)
    config = RoleAwareConditionTransferConfig(
        role_transfer_mode="ligand_base",
        donor_strategy="same_nontransferred_roles",
        label_strategy="source_label",
        synthetic_multiplier=0.5,
        max_candidates_per_source=1,
        teacher_models=["ridge"],
        max_teacher_std=None,
        min_similarity=None,
        random_state=29,
        max_resample_attempts=4,
        donor_similarity_n_bits=64,
        donor_similarity_backend="rdkit",
        role_change_requirement="all",
        fallback_policy=fallback_policy,
    )
    result = generate_role_aware_condition_transfer_examples(
        pair,
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        config,
        feature_config=feature_config,
        real_feature_names=names,
        real_feature_metadata=metadata,
        measured_identity_keys=measured_keys,
    )
    return result["candidate_df"]


def _fallback_pair(train: pd.DataFrame) -> pd.DataFrame:
    rows = train.reset_index(drop=True)
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            ligand_and_base_change = all(
                rows.loc[left, role] != rows.loc[right, role]
                for role in ["canonical_ligand_smiles", "canonical_base_smiles"]
            )
            nontransferred_context_changes = any(
                rows.loc[left, role] != rows.loc[right, role]
                for role in [
                    "canonical_catalyst_smiles",
                    "canonical_solvent_or_additive_smiles",
                ]
            )
            if ligand_and_base_change and nontransferred_context_changes:
                return rows.loc[[left, right]].copy()
    raise ValueError("Canonical smoke subset has no suitable deterministic fallback pair.")


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
