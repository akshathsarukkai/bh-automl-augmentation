"""One-time outer-test evaluation of a frozen joint supervised-AE policy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.ae_policy_search import (
    AE_REFIT_PROTOCOL_SCHEMA_VERSION,
    AE_SEARCH_MANIFEST_SCHEMA_VERSION,
    _evaluation_unit,
    _frozen_training_protocol,
    _resolve_ae_search_config,
    _resolved_joint_ae_policies,
    _transfer_key_from_policy,
    scientific_ae_config_projection,
)
from bh_augmentation.augmentation.candidate_scope import (
    CandidateScopePolicy,
    observed_only_scope,
)
from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.ae_policy_protocol import make_ae_inner_split
from bh_augmentation.evaluation.policy_protocol import (
    FinalEvaluationInputs,
    FrozenPolicyEnvelope,
    ResolvedPolicy,
    ScientificBinding,
    evaluate_frozen_policy_once,
    load_frozen_policy,
)
from bh_augmentation.features.compatibility import assert_feature_compatibility
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.policy_search import (
    LabeledPartition,
    _build_partition,
    _materialize_search_frames,
    _metric_rows,
    _resolve_candidate_scope,
    _resolve_run_contract,
    current_commit,
)
from bh_augmentation.representations.supervised_autoencoder import (
    SupervisedAEConfig,
    encode_with_supervised_autoencoder,
    fit_supervised_autoencoder,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    resolve_corrected_feature_config,
    stable_hash,
)

AE_FINAL_MANIFEST_SCHEMA_VERSION = "bh-joint-ae-final-evaluation-manifest-v1"
AE_EVALUATION_CLAIM_SCHEMA_VERSION = "bh-joint-ae-final-evaluation-claim-v1"

_SEARCH_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "payload",
    "search_manifest_hash",
    "frozen_policy_hash",
}
_SEARCH_PAYLOAD_FIELDS = {
    "run_type",
    "scientific_binding",
    "scientific_config",
    "evaluation_unit",
    "inner_split",
    "candidate_policy_hashes",
    "feasible_policy_hashes",
    "selected_policy_hash",
    "selection_metric",
    "lower_is_better",
    "selection_data_roles",
    "outer_test_labels_accessed",
    "outer_test_predictions_generated",
    "outer_test_metric_evaluations",
    "test_evaluated",
    "artifact_hashes",
}
_SEARCH_ARTIFACT_FILES = {
    "search_metrics": "search_metrics.csv",
    "search_exclusions": "search_exclusions.csv",
    "ae_training_audit": "ae_training_audit.csv",
    "synthetic_transfer_audit": "synthetic_transfer_audit.csv",
    "synthetic_candidate_audit": "synthetic_candidate_audit.csv",
}


@dataclass(frozen=True)
class _RefittedAEPipeline:
    artifacts: dict[str, Any]
    downstream: Any
    n_synthetic: int
    synthetic_parent_ids: tuple[str, ...]
    synthetic_identity: dict[str, Any]
    refit_history_epochs: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        latent = encode_with_supervised_autoencoder(self.artifacts, X)
        return predict_model(self.downstream, latent)


@dataclass(frozen=True)
class _RuntimeFeasibility:
    policies: tuple[ResolvedPolicy, ...]
    synthetic_candidate_audit_sha256: str
    synthetic_transfer_audit_sha256: str


def run_ae_final_evaluation_command(
    config_path: str | Path,
    *,
    frozen_policy_path: str | Path,
    search_manifest_path: str | Path,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Verify, refit the frozen joint tuple, and evaluate outer test once."""
    config = load_config(config_path)
    dataset_path, split_directory, seed, fraction = _resolve_run_contract(config)
    saved = load_saved_canonical_splits(
        dataset_path,
        split_directory,
        requested_seeds=[seed],
        requested_fractions=[fraction],
    )
    assignment_rows = saved.low_data_assignments.loc[
        saved.low_data_assignments["seed"].eq(seed)
        & saved.low_data_assignments["train_fraction"].eq(fraction)
    ].copy()
    memberships = _validated_assignment_memberships(assignment_rows)
    refit_frame = saved.canonical.loc[
        saved.canonical["source_row_id"].astype(str).isin(memberships["train"])
    ].copy()
    if set(refit_frame["source_row_id"].astype(str)) != memberships["train"]:
        raise ValueError("Saved training-subset rows are incomplete in the canonical dataset.")

    feature_config = resolve_corrected_feature_config(
        config.get("features", {}),
        required_kind="bh_role_separated",
    )
    candidate_scope = _resolve_candidate_scope(
        config,
        saved,
        seed=seed,
        train_fraction=fraction,
        dataset_path=dataset_path,
    )
    refit, refit_contract = _build_partition(
        refit_frame,
        feature_config,
        candidate_scope=candidate_scope,
    )
    audit = saved.audit_record(seed=seed, train_fraction=fraction)
    expected_binding = ScientificBinding(
        dataset_hash=saved.dataset_hash,
        split_hash=str(audit["per_seed_split_hash"]),
        split_aggregate_hash=saved.aggregate_split_hash,
        per_seed_split_hash=str(audit["per_seed_split_hash"]),
        source_id_split_hash=str(audit["source_id_split_hash"]),
        canonicalization_version=str(audit["canonicalization_version"]),
        split_schema_version=str(audit["split_schema_version"]),
        feature_metadata_hash=str(refit_contract["feature_metadata_hash"]),
        config_hash=stable_hash(scientific_ae_config_projection(config)),
        commit_hash=current_commit(),
    )
    frozen = load_frozen_policy(
        frozen_policy_path,
        expected_binding=expected_binding,
    )
    search_config = _resolve_ae_search_config(config)
    search_context = _rebuild_search_context(
        saved=saved,
        assignment_rows=assignment_rows,
        memberships=memberships,
        feature_config=feature_config,
        candidate_scope=candidate_scope,
        search_config=search_config,
        seed=seed,
        fraction=fraction,
    )
    runtime_policies = _resolved_joint_ae_policies(
        search_config,
        seed=seed,
        fraction=fraction,
        input_dim=refit.X.shape[1],
        inner_split_hash=search_context["inner_split"].split_hash,
    )
    runtime_feasibility = _runtime_feasible_policies(
        runtime_policies,
        search_context["ae_fit"],
        forbidden_source_ids=memberships["valid"] | memberships["test"],
    )
    search_manifest = _load_ae_search_manifest(search_manifest_path)
    _verify_ae_search_artifacts(
        search_manifest,
        search_manifest_path=Path(search_manifest_path),
        frozen=frozen,
        expected_binding=expected_binding,
        config=config,
        search_config=search_config,
        runtime_policies=runtime_policies,
        runtime_feasibility=runtime_feasibility,
        runtime_inner_split=search_context["inner_split"],
        seed=seed,
        fraction=fraction,
    )

    output = Path(
        output_directory
        if output_directory is not None
        else config.get("output", {}).get(
            "ae_final_directory", "results/corrected_joint_ae_final_evaluation"
        )
    )
    evaluation_unit = _evaluation_unit(seed, fraction)
    global_claim_path, global_claim_id = _claim_global_evaluation(
        Path(search_manifest_path),
        evaluation_unit=evaluation_unit,
        frozen_policy_hash=frozen.frozen_policy_hash,
        binding=expected_binding,
        output_directory=output,
    )
    try:
        claim_path = _claim_evaluation(
            output,
            evaluation_unit=evaluation_unit,
            frozen_policy_hash=frozen.frozen_policy_hash,
            binding=expected_binding,
            global_claim_id=global_claim_id,
        )
    except Exception:
        _update_global_claim(
            global_claim_path,
            status="output_claim_failed",
            outer_test_attempts=0,
            outer_test_labels_accessed=False,
            outer_test_prediction_batches=0,
        )
        raise
    try:
        pipeline = _refit_frozen_ae_pipeline(
            frozen,
            refit,
            allowed_source_ids=memberships["train"],
            forbidden_source_ids=memberships["valid"] | memberships["test"],
        )
    except Exception as exc:
        failure = {
            "refit_error_type": type(exc).__name__,
            "refit_error": str(exc),
            "outer_test_attempts": 0,
            "outer_test_labels_accessed": False,
            "outer_test_prediction_batches": 0,
        }
        _update_claim(claim_path, status="refit_failed", **failure)
        _update_global_claim(
            global_claim_path,
            status="refit_failed",
            **failure,
        )
        raise
    _update_claim(
        claim_path,
        status="refit_complete",
        outer_test_prediction_batches=0,
        n_measured_refit=len(refit.y),
        n_synthetic_refit=pipeline.n_synthetic,
        synthetic_identity=pipeline.synthetic_identity,
    )
    _update_global_claim(
        global_claim_path,
        status="refit_complete",
        outer_test_prediction_batches=0,
        n_measured_refit=len(refit.y),
        n_synthetic_refit=pipeline.n_synthetic,
        synthetic_identity=pipeline.synthetic_identity,
    )

    # Outer-test features and labels are deliberately materialized only after
    # the complete representation and downstream refit has succeeded.
    _update_claim(
        claim_path,
        status="outer_test_attempt_reserved",
        outer_test_attempts=1,
        outer_test_labels_accessed=True,
        outer_test_prediction_batches=0,
    )
    _update_global_claim(
        global_claim_path,
        status="outer_test_attempt_reserved",
        outer_test_attempts=1,
        outer_test_labels_accessed=True,
        outer_test_prediction_batches=0,
    )
    test_frame = saved.canonical.loc[
        saved.canonical["source_row_id"].astype(str).isin(memberships["test"])
    ].copy()
    outer_test, test_contract = _build_partition(
        test_frame,
        feature_config,
        candidate_scope=observed_only_scope(
            labeled_train_identity_keys=(
                str(value)
                for value in test_frame["canonical_reaction_key"].dropna()
            ),
        ),
    )
    if refit_contract != test_contract:
        raise ValueError("Refit and outer-test feature contracts differ.")
    if set(outer_test.source_row_ids) != memberships["test"]:
        raise ValueError("Outer-test rows are incomplete in the canonical dataset.")

    leakage_audit = _derived_leakage_audit(
        refit_ids=set(refit.source_row_ids),
        valid_ids=memberships["valid"],
        test_ids=set(outer_test.source_row_ids),
        parent_ids=set(pipeline.synthetic_parent_ids),
        expected_refit_ids=memberships["train"],
    )
    if leakage_audit["leakage_detected"]:
        _update_claim(claim_path, status="leakage_detected", leakage_audit=leakage_audit)
        _update_global_claim(
            global_claim_path,
            status="leakage_detected",
            leakage_audit=leakage_audit,
        )
        raise ValueError("Derived final-evaluation leakage audit failed.")

    inputs = FinalEvaluationInputs(
        refit=refit,
        outer_test=outer_test,
        binding=expected_binding,
        evaluation_unit=evaluation_unit,
    )

    def outer_evaluator(
        final_inputs: FinalEvaluationInputs,
        policy: FrozenPolicyEnvelope,
    ) -> pd.DataFrame:
        del policy
        predictions = pipeline.predict(final_inputs.outer_test.X)
        _update_claim(
            claim_path,
            status="outer_test_prediction_complete",
            outer_test_attempts=1,
            outer_test_labels_accessed=True,
            outer_test_prediction_batches=1,
        )
        _update_global_claim(
            global_claim_path,
            status="outer_test_prediction_complete",
            outer_test_attempts=1,
            outer_test_labels_accessed=True,
            outer_test_prediction_batches=1,
        )
        return _metric_rows(
            final_inputs.outer_test.y,
            predictions,
            search_config["metrics"],
            split="test",
        )

    result = evaluate_frozen_policy_once(
        inputs,
        frozen,
        outer_evaluator,
        evaluated_units=set(),
    )
    resolved = frozen.resolved_policy.config
    metrics = result.metrics.assign(
        seed=seed,
        train_fraction=fraction,
        transfer_method=str(resolved["transfer_method"]),
        latent_dim=int(resolved["latent_dim"]),
        synthetic_supervised_weight=float(
            resolved["synthetic_supervised_weight"]
        ),
        synthetic_reconstruction_weight=float(
            resolved["synthetic_reconstruction_weight"]
        ),
        downstream_model=str(resolved["downstream_model"]),
        n_measured_refit=len(refit.y),
        n_synthetic_refit=pipeline.n_synthetic,
        refit_epochs=int(frozen.training_protocol["refit_epochs"]),
        test_prediction_batch=1,
    )
    metrics_bytes = metrics.to_csv(index=False).encode("utf-8")
    metrics_hash = hashlib.sha256(metrics_bytes).hexdigest()
    manifest_payload = {
        "run_type": "joint_ae_final_evaluation",
        "scientific_binding": expected_binding.to_dict(),
        "evaluation_unit": evaluation_unit,
        "global_claim_id": global_claim_id,
        "frozen_policy_hash": frozen.frozen_policy_hash,
        "selected_policy_hash": frozen.resolved_policy.policy_hash,
        "search_manifest_hash": frozen.search_manifest_hash,
        "training_protocol": dict(frozen.training_protocol),
        "inner_split_hash": search_context["inner_split"].split_hash,
        "selection_data_roles": [
            "ae_fit",
            "ae_internal_validation",
            "saved_validation",
        ],
        "refit_data_roles": ["all_saved_training_subset_rows"],
        "selected_transfer_method": str(resolved["transfer_method"]),
        "selected_latent_dim": int(resolved["latent_dim"]),
        "selected_downstream_model": str(resolved["downstream_model"]),
        "n_measured_refit": len(refit.y),
        "n_synthetic_refit": pipeline.n_synthetic,
        "refit_epochs_requested": int(frozen.training_protocol["refit_epochs"]),
        "refit_history_epochs": pipeline.refit_history_epochs,
        "refit_source_id_hash": stable_hash(sorted(refit.source_row_ids)),
        "outer_validation_source_id_hash": stable_hash(
            sorted(memberships["valid"])
        ),
        "outer_test_source_id_hash": stable_hash(
            sorted(outer_test.source_row_ids)
        ),
        "synthetic_parent_source_id_hash": stable_hash(
            sorted(pipeline.synthetic_parent_ids)
        ),
        "synthetic_identity": pipeline.synthetic_identity,
        "leakage_audit": leakage_audit,
        "outer_test_labels_accessed": True,
        "test_evaluated": True,
        "outer_test_attempts": 1,
        "outer_test_prediction_batches": 1,
        "per_unit_test_evaluation_counts": {evaluation_unit: 1},
        "final_test_metrics_sha256": metrics_hash,
    }
    manifest = {
        "schema_version": AE_FINAL_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "payload": manifest_payload,
        "final_evaluation_manifest_hash": stable_hash(manifest_payload),
    }
    paths = {
        "directory": output,
        "final_test_metrics": output / "final_test_metrics.csv",
        "final_evaluation_manifest": output / "final_evaluation_manifest.json",
        "evaluation_claim": claim_path,
    }
    paths["final_test_metrics"].write_bytes(metrics_bytes)
    paths["final_evaluation_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _update_claim(
        claim_path,
        status="complete",
        outer_test_prediction_batches=1,
        outer_test_attempts=1,
        outer_test_labels_accessed=True,
        final_test_metrics_sha256=metrics_hash,
        final_evaluation_manifest_hash=manifest[
            "final_evaluation_manifest_hash"
        ],
        leakage_detected=leakage_audit["leakage_detected"],
    )
    _update_global_claim(
        global_claim_path,
        status="complete",
        outer_test_prediction_batches=1,
        outer_test_attempts=1,
        outer_test_labels_accessed=True,
        final_test_metrics_sha256=metrics_hash,
        final_evaluation_manifest_hash=manifest[
            "final_evaluation_manifest_hash"
        ],
        leakage_detected=leakage_audit["leakage_detected"],
    )
    paths["global_evaluation_claim"] = global_claim_path
    return paths


def _validated_assignment_memberships(
    assignment_rows: pd.DataFrame,
) -> dict[str, set[str]]:
    required = {
        "source_row_id",
        "outer_split",
        "included_in_training_subset",
    }
    missing = sorted(required - set(assignment_rows))
    if missing:
        raise ValueError(f"Saved split assignments are missing fields: {missing}.")
    if assignment_rows.empty:
        raise ValueError("Saved split assignments are empty.")
    if assignment_rows["source_row_id"].astype(str).duplicated().any():
        raise ValueError("Saved split assignments contain duplicate source IDs.")
    ids = assignment_rows["source_row_id"].astype(str)
    train = set(
        ids.loc[assignment_rows["included_in_training_subset"].astype(bool)]
    )
    valid = set(ids.loc[assignment_rows["outer_split"].eq("valid")])
    test = set(ids.loc[assignment_rows["outer_split"].eq("test")])
    if not train:
        raise ValueError("Saved split assignments contain no eligible training rows.")
    if not test:
        raise ValueError("Saved split assignments contain no outer-test rows.")
    if train & valid or train & test or valid & test:
        raise ValueError("Saved train/validation/test source-ID overlap detected.")
    nontrain_included = assignment_rows.loc[
        assignment_rows["included_in_training_subset"].astype(bool)
        & ~assignment_rows["outer_split"].eq("train")
    ]
    if not nontrain_included.empty:
        raise ValueError("A validation/test row is included in the training subset.")
    return {"train": train, "valid": valid, "test": test}


def _rebuild_search_context(
    *,
    saved: Any,
    assignment_rows: pd.DataFrame,
    memberships: dict[str, set[str]],
    feature_config: dict[str, Any],
    candidate_scope: CandidateScopePolicy,
    search_config: dict[str, Any],
    seed: int,
    fraction: float,
) -> dict[str, Any]:
    del assignment_rows
    train_frame, policy_validation_frame = _materialize_search_frames(
        saved,
        seed=seed,
        train_fraction=fraction,
    )
    settings = search_config["ae_settings"]
    inner_split = make_ae_inner_split(
        train_frame,
        valid_size=float(settings["internal_valid_size"]),
        seed=seed + int(search_config["inner_split_seed_offset"]),
        forbidden_outer_valid_source_ids=sorted(memberships["valid"]),
        forbidden_outer_test_source_ids=sorted(memberships["test"]),
        minimum_rows=int(settings["minimum_inner_rows"]),
        minimum_groups=int(settings["minimum_inner_groups"]),
    )
    ae_fit, fit_contract = _build_partition(
        inner_split.ae_train,
        feature_config,
        candidate_scope=candidate_scope,
    )
    internal_validation, internal_contract = _build_partition(
        inner_split.internal_validation,
        feature_config,
        candidate_scope=candidate_scope,
    )
    policy_validation, policy_contract = _build_partition(
        policy_validation_frame,
        feature_config,
        candidate_scope=candidate_scope,
    )
    if fit_contract != internal_contract or fit_contract != policy_contract:
        raise ValueError("Runtime joint AE search feature contracts differ.")
    return {
        "inner_split": inner_split,
        "ae_fit": ae_fit,
        "internal_validation": internal_validation,
        "policy_validation": policy_validation,
    }


def _runtime_feasible_policies(
    policies: list[ResolvedPolicy],
    ae_fit: LabeledPartition,
    *,
    forbidden_source_ids: set[str],
) -> _RuntimeFeasibility:
    feasible_transfer_keys: set[str] = set()
    representatives: dict[str, ResolvedPolicy] = {}
    transfer_audits: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for policy in policies:
        representatives.setdefault(_transfer_key_from_policy(policy), policy)
    for transfer_key, policy in sorted(representatives.items()):
        method = str(policy.config["transfer_method"])
        if method == "real_only":
            feasible_transfer_keys.add(transfer_key)
            transfer_audits.append(
                {
                    "transfer_key": transfer_key,
                    "transfer_method": method,
                    "n_accepted_synthetic": 0,
                    "selection_feasible": True,
                    "rejection_reason": "",
                    "parent_source_id_hash": stable_hash([]),
                    "parent_ids_ae_fit_only": True,
                }
            )
            continue
        generation = _generate_frozen_transfer(
            policy,
            ae_fit,
            allowed_source_ids=set(ae_fit.source_row_ids),
            forbidden_source_ids=forbidden_source_ids,
        )
        parent_ids = _validate_synthetic_parent_ids(
            generation,
            allowed_source_ids=set(ae_fit.source_row_ids),
            forbidden_source_ids=forbidden_source_ids,
        )
        n_synthetic = len(generation["synthetic_y"])
        if n_synthetic > 0:
            feasible_transfer_keys.add(transfer_key)
        metadata = dict(generation["metadata"])
        transfer_audits.append(
            {
                "transfer_key": transfer_key,
                "transfer_method": method,
                **metadata,
                "n_accepted_synthetic": n_synthetic,
                "selection_feasible": n_synthetic > 0,
                "rejection_reason": (
                    "" if n_synthetic > 0 else "zero_accepted_synthetic"
                ),
                "parent_source_id_hash": stable_hash(sorted(parent_ids)),
                "parent_ids_ae_fit_only": True,
            }
        )
        candidate = generation["candidate_df"].copy()
        candidate.insert(0, "transfer_key", transfer_key)
        candidate_frames.append(candidate)
    feasible = [
        policy
        for policy in policies
        if _transfer_key_from_policy(policy) in feasible_transfer_keys
    ]
    if not feasible:
        raise ValueError("No runtime-feasible joint AE policies remain.")
    candidate_audit = (
        pd.concat(candidate_frames, ignore_index=True, sort=False)
        if candidate_frames
        else pd.DataFrame(columns=["transfer_key"])
    )
    transfer_audit = pd.DataFrame(transfer_audits)
    return _RuntimeFeasibility(
        policies=tuple(feasible),
        synthetic_candidate_audit_sha256=_dataframe_sha256(candidate_audit),
        synthetic_transfer_audit_sha256=_dataframe_sha256(transfer_audit),
    )


def _synthetic_identity_record(generation: dict[str, Any]) -> dict[str, Any]:
    candidate = generation.get("candidate_df")
    if not isinstance(candidate, pd.DataFrame):
        raise ValueError("Synthetic generator did not return a candidate audit.")
    synthetic_y = np.asarray(generation.get("synthetic_y"), dtype=np.float64).reshape(-1)
    if not np.isfinite(synthetic_y).all():
        raise ValueError("Synthetic labels must be finite.")
    if candidate.empty:
        if len(synthetic_y):
            raise ValueError("Synthetic labels exist without candidate provenance.")
        used = candidate.copy()
    else:
        required = {
            "canonical_reaction_hash",
            "feature_hash",
            "synthetic_label",
        }
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"Synthetic identity audit is missing fields: {missing}.")
        if "kept" in candidate:
            used = candidate.loc[candidate["kept"].astype(bool)].copy()
        elif "accepted" in candidate:
            used = candidate.loc[candidate["accepted"].astype(bool)].copy()
        else:
            raise ValueError("Synthetic identity audit lacks a kept/accepted marker.")
        if len(used) != len(synthetic_y):
            raise ValueError("Used synthetic identity rows do not match synthetic labels.")
        if used[
            ["canonical_reaction_hash", "feature_hash", "synthetic_label"]
        ].isna().any().any():
            raise ValueError("Used synthetic rows contain incomplete identity or label fields.")
        audit_labels = pd.to_numeric(
            used["synthetic_label"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(audit_labels).all()
            or not np.array_equal(audit_labels, synthetic_y)
        ):
            raise ValueError("Candidate-audit synthetic labels differ from training labels.")
    canonical_hashes = tuple(used.get("canonical_reaction_hash", pd.Series(dtype=str)).astype(str))
    feature_hashes = tuple(used.get("feature_hash", pd.Series(dtype=str)).astype(str))
    if len(canonical_hashes) != len(set(canonical_hashes)):
        raise ValueError("Used synthetic canonical reaction hashes are not unique.")
    if len(feature_hashes) != len(set(feature_hashes)):
        raise ValueError("Used synthetic feature hashes are not unique.")
    metadata = _manifest_json_value(generation.get("metadata", {}))
    return {
        "candidate_count": len(candidate),
        "used_synthetic_count": len(used),
        "accepted_canonical_reaction_hashes": list(canonical_hashes),
        "accepted_canonical_reaction_hashes_hash": stable_hash(
            list(canonical_hashes)
        ),
        "accepted_feature_hashes": list(feature_hashes),
        "accepted_feature_hashes_hash": stable_hash(list(feature_hashes)),
        "synthetic_label_hash": stable_hash([float(value) for value in synthetic_y]),
        "candidate_audit_sha256": _dataframe_sha256(candidate),
        "accepted_audit_sha256": _dataframe_sha256(used),
        "generation_metadata": metadata,
        "generation_metadata_hash": stable_hash(metadata),
    }


def _empty_synthetic_identity() -> dict[str, Any]:
    metadata = {"transfer_method": "real_only"}
    return {
        "candidate_count": 0,
        "used_synthetic_count": 0,
        "accepted_canonical_reaction_hashes": [],
        "accepted_canonical_reaction_hashes_hash": stable_hash([]),
        "accepted_feature_hashes": [],
        "accepted_feature_hashes_hash": stable_hash([]),
        "synthetic_label_hash": stable_hash([]),
        "candidate_audit_sha256": _dataframe_sha256(pd.DataFrame()),
        "accepted_audit_sha256": _dataframe_sha256(pd.DataFrame()),
        "generation_metadata": metadata,
        "generation_metadata_hash": stable_hash(metadata),
    }


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _manifest_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _manifest_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("Generation metadata must be JSON serializable.")


def _generate_frozen_transfer(
    policy: ResolvedPolicy,
    partition: LabeledPartition,
    *,
    allowed_source_ids: set[str],
    forbidden_source_ids: set[str],
) -> dict[str, Any]:
    if set(partition.source_row_ids) != allowed_source_ids:
        raise ValueError("Synthetic source pool does not match the eligible real rows.")
    if allowed_source_ids & forbidden_source_ids:
        raise ValueError("Synthetic source pool overlaps forbidden validation/test rows.")
    transfer = ConditionTransferConfig(**dict(policy.config["transfer_config"]))
    generation = generate_condition_transfer_examples(
        partition.frame,
        partition.X,
        partition.y,
        transfer,
        feature_config=partition.feature_config,
        real_feature_names=list(partition.feature_names),
        real_feature_metadata=partition.feature_metadata,
        candidate_scope=partition.candidate_scope,
    )
    _validate_synthetic_parent_ids(
        generation,
        allowed_source_ids=allowed_source_ids,
        forbidden_source_ids=forbidden_source_ids,
    )
    return generation


def _validate_synthetic_parent_ids(
    generation: dict[str, Any],
    *,
    allowed_source_ids: set[str],
    forbidden_source_ids: set[str],
) -> tuple[str, ...]:
    candidate = generation.get("candidate_df")
    if not isinstance(candidate, pd.DataFrame):
        raise ValueError("Synthetic generator did not return candidate provenance.")
    if candidate.empty:
        return ()
    required = {"source_row_id", "donor_row_id"}
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"Synthetic parent provenance is missing fields: {missing}.")
    if candidate[list(required)].isna().any().any():
        raise ValueError("Synthetic parent provenance contains missing source IDs.")
    parent_ids = set(candidate["source_row_id"].astype(str)) | set(
        candidate["donor_row_id"].astype(str)
    )
    outside = parent_ids - allowed_source_ids
    forbidden = parent_ids & forbidden_source_ids
    if outside or forbidden:
        raise ValueError(
            "Synthetic parent leakage detected: "
            f"outside_source_pool={sorted(outside)}, "
            f"forbidden_validation_or_test={sorted(forbidden)}."
        )
    return tuple(sorted(parent_ids))


def _load_ae_search_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Joint AE search manifest contains invalid JSON.") from exc
    if not isinstance(manifest, dict) or set(manifest) != _SEARCH_MANIFEST_FIELDS:
        raise ValueError("Joint AE search manifest schema mismatch.")
    if (
        manifest["schema_version"] != AE_SEARCH_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "complete"
    ):
        raise ValueError("Final evaluation requires a completed joint AE search.")
    payload = manifest["payload"]
    if not isinstance(payload, dict) or set(payload) != _SEARCH_PAYLOAD_FIELDS:
        raise ValueError("Joint AE search manifest payload schema mismatch.")
    if stable_hash(payload) != manifest["search_manifest_hash"]:
        raise ValueError("Joint AE search manifest hash mismatch.")
    if (
        payload["run_type"] != "joint_ae_policy_search"
        or payload["selection_data_roles"]
        != ["ae_fit", "ae_internal_validation", "saved_validation"]
        or payload["outer_test_labels_accessed"] is not False
        or payload["outer_test_predictions_generated"] is not False
        or payload["outer_test_metric_evaluations"] != 0
        or payload["test_evaluated"] is not False
    ):
        raise ValueError("Joint AE search manifest reports forbidden test access.")
    if set(payload["artifact_hashes"]) != set(_SEARCH_ARTIFACT_FILES):
        raise ValueError("Joint AE search artifact hash schema mismatch.")
    return manifest


def _verify_ae_search_artifacts(
    manifest: dict[str, Any],
    *,
    search_manifest_path: Path,
    frozen: FrozenPolicyEnvelope,
    expected_binding: ScientificBinding,
    config: dict[str, Any],
    search_config: dict[str, Any],
    runtime_policies: list[ResolvedPolicy],
    runtime_feasibility: _RuntimeFeasibility,
    runtime_inner_split: Any,
    seed: int,
    fraction: float,
) -> None:
    payload = manifest["payload"]
    if ScientificBinding.from_dict(payload["scientific_binding"]) != expected_binding:
        raise ValueError("Joint AE search scientific binding mismatch.")
    scientific_config = scientific_ae_config_projection(config)
    if payload["scientific_config"] != scientific_config:
        raise ValueError("Joint AE search scientific config mismatch.")
    if stable_hash(payload["scientific_config"]) != expected_binding.config_hash:
        raise ValueError("Joint AE search scientific config hash mismatch.")
    if manifest["search_manifest_hash"] != frozen.search_manifest_hash:
        raise ValueError("Frozen policy search-manifest hash mismatch.")
    if manifest["frozen_policy_hash"] != frozen.frozen_policy_hash:
        raise ValueError("Search manifest frozen-policy hash mismatch.")
    if payload["selected_policy_hash"] != frozen.resolved_policy.policy_hash:
        raise ValueError("Search manifest selected-policy hash mismatch.")
    if payload["evaluation_unit"] != _evaluation_unit(seed, fraction):
        raise ValueError("Joint AE search evaluation unit mismatch.")
    if payload["inner_split"] != runtime_inner_split.audit_record:
        raise ValueError("Runtime AE inner split differs from frozen search.")
    if frozen.training_protocol.get("schema_version") != AE_REFIT_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Frozen policy declares an unsupported training protocol.")
    expected_protocol = _frozen_training_protocol(
        runtime_inner_split,
        frozen.resolved_policy,
    )
    if dict(frozen.training_protocol) != expected_protocol:
        raise ValueError("Frozen policy declares an unsupported training protocol.")
    if frozen.training_protocol["inner_split_hash"] != runtime_inner_split.split_hash:
        raise ValueError("Frozen training protocol inner split hash mismatch.")

    runtime_hashes = sorted(policy.policy_hash for policy in runtime_policies)
    feasible_hashes = sorted(
        policy.policy_hash for policy in runtime_feasibility.policies
    )
    if payload["candidate_policy_hashes"] != runtime_hashes:
        raise ValueError("Runtime joint AE candidate policies differ from search.")
    if payload["feasible_policy_hashes"] != feasible_hashes:
        raise ValueError("Runtime joint AE feasible policies differ from search.")
    if (
        runtime_feasibility.synthetic_candidate_audit_sha256
        != payload["artifact_hashes"]["synthetic_candidate_audit"]
    ):
        raise ValueError(
            "Runtime synthetic candidate identities differ from hash-bound search."
        )
    if (
        runtime_feasibility.synthetic_transfer_audit_sha256
        != payload["artifact_hashes"]["synthetic_transfer_audit"]
    ):
        raise ValueError(
            "Runtime synthetic labels or transfer audit differ from hash-bound search."
        )
    runtime_by_id = {
        policy.policy_id: policy for policy in runtime_feasibility.policies
    }
    if frozen.resolved_policy.policy_id not in runtime_by_id:
        raise ValueError("Frozen joint AE policy is not runtime feasible.")
    if (
        runtime_by_id[frozen.resolved_policy.policy_id]
        != frozen.resolved_policy
    ):
        raise ValueError("Frozen joint AE policy differs from runtime resolution.")

    for name, filename in _SEARCH_ARTIFACT_FILES.items():
        artifact_path = search_manifest_path.parent / filename
        if not artifact_path.is_file():
            raise ValueError(f"Joint AE search artifact is missing: {filename}.")
        observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if observed != payload["artifact_hashes"][name]:
            raise ValueError(f"Joint AE search artifact hash mismatch: {filename}.")
    metrics = pd.read_csv(search_manifest_path.parent / "search_metrics.csv")
    _replay_validation_winner(
        metrics,
        frozen=frozen,
        payload=payload,
        runtime_by_id=runtime_by_id,
        selection_metric=str(search_config["selection_metric"]),
        lower_is_better=bool(search_config["lower_is_better"]),
        inner_split_hash=runtime_inner_split.split_hash,
    )


def _replay_validation_winner(
    metrics: pd.DataFrame,
    *,
    frozen: FrozenPolicyEnvelope,
    payload: dict[str, Any],
    runtime_by_id: dict[str, ResolvedPolicy],
    selection_metric: str,
    lower_is_better: bool,
    inner_split_hash: str,
) -> None:
    required = {
        "split",
        "metric",
        "value",
        "policy_id",
        "policy_hash",
        "selected_policy",
        "inner_split_hash",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Joint AE search metrics are missing fields: {missing}.")
    if metrics.empty or not metrics["split"].eq("valid").all():
        raise ValueError("Joint AE search metrics must contain validation rows only.")
    if not metrics["inner_split_hash"].eq(inner_split_hash).all():
        raise ValueError("Joint AE search metrics inner split hash mismatch.")
    if set(metrics["policy_id"]) != set(runtime_by_id):
        raise ValueError("Joint AE search metric policy IDs differ from feasible policies.")
    expected_hashes = metrics["policy_id"].map(
        {
            policy_id: policy.policy_hash
            for policy_id, policy in runtime_by_id.items()
        }
    )
    if not metrics["policy_hash"].eq(expected_hashes).all():
        raise ValueError("Joint AE search metric policy hash mismatch.")
    selected = _strict_boolean_series(metrics["selected_policy"])
    selected_ids = set(metrics.loc[selected, "policy_id"])
    if len(selected_ids) != 1:
        raise ValueError("Joint AE search metrics must select exactly one policy.")
    candidates = metrics.loc[metrics["metric"].eq(selection_metric)].copy()
    if len(candidates) != len(runtime_by_id) or candidates["policy_id"].duplicated().any():
        raise ValueError("Selection metric must appear exactly once per feasible AE policy.")
    candidates["value"] = pd.to_numeric(candidates["value"], errors="coerce")
    if candidates["value"].isna().any() or not candidates["value"].map(math.isfinite).all():
        raise ValueError("Joint AE validation selection values must be finite.")
    winner = candidates.sort_values(
        ["value", "policy_hash", "policy_id"],
        ascending=[lower_is_better, True, True],
        kind="mergesort",
    ).iloc[0]
    winner_policy = runtime_by_id[str(winner["policy_id"])]
    expected_selected = metrics["policy_id"].eq(winner_policy.policy_id)
    if (
        selected_ids != {winner_policy.policy_id}
        or not selected.eq(expected_selected).all()
    ):
        raise ValueError("Recorded policy is not the deterministic validation winner.")
    if (
        payload["selection_metric"] != selection_metric
        or payload["lower_is_better"] != lower_is_better
        or frozen.selection_metric != selection_metric
        or frozen.lower_is_better != lower_is_better
        or frozen.resolved_policy != winner_policy
        or not math.isclose(
            frozen.selection_value,
            float(winner["value"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Frozen policy does not match the deterministic validation winner.")


def _strict_boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    text = values.astype(str)
    if not text.isin({"True", "False"}).all():
        raise ValueError("Joint AE selected_policy values must be boolean.")
    return text.eq("True")


def _refit_frozen_ae_pipeline(
    frozen: FrozenPolicyEnvelope,
    refit: LabeledPartition,
    *,
    allowed_source_ids: set[str],
    forbidden_source_ids: set[str],
) -> _RefittedAEPipeline:
    resolved = frozen.resolved_policy.config
    if resolved.get("method") != "condition_transfer_supervised_ae":
        raise ValueError("Frozen policy is not a joint supervised-AE policy.")
    X_train = np.asarray(refit.X, dtype=np.float32)
    y_train = np.asarray(refit.y, dtype=np.float32)
    yield_weights = np.ones(len(y_train), dtype=np.float32)
    reconstruction_weights = np.ones(len(y_train), dtype=np.float32)
    n_synthetic = 0
    parent_ids: tuple[str, ...] = ()
    synthetic_identity = _empty_synthetic_identity()
    transfer_method = str(resolved["transfer_method"])
    if transfer_method == "anonymous":
        generation = _generate_frozen_transfer(
            frozen.resolved_policy,
            refit,
            allowed_source_ids=allowed_source_ids,
            forbidden_source_ids=forbidden_source_ids,
        )
        parent_ids = _validate_synthetic_parent_ids(
            generation,
            allowed_source_ids=allowed_source_ids,
            forbidden_source_ids=forbidden_source_ids,
        )
        synthetic_identity = _synthetic_identity_record(generation)
        X_synthetic = np.asarray(generation["X_synthetic"], dtype=np.float32)
        y_synthetic = np.asarray(generation["synthetic_y"], dtype=np.float32)
        if len(y_synthetic) == 0:
            raise ValueError("Frozen augmentation produced zero synthetic refit rows.")
        if len(X_synthetic) != len(y_synthetic):
            raise ValueError("Synthetic refit feature and label row counts differ.")
        assert_feature_compatibility(
            refit.X,
            refit.feature_names,
            X_synthetic,
            generation["feature_names"],
            real_metadata=refit.feature_metadata,
            synthetic_metadata=generation["feature_metadata"],
        )
        n_synthetic = len(y_synthetic)
        X_train = np.vstack([X_train, X_synthetic]).astype(np.float32)
        y_train = np.concatenate([y_train, y_synthetic]).astype(np.float32)
        yield_weights = np.concatenate(
            [
                yield_weights,
                np.full(
                    n_synthetic,
                    float(resolved["synthetic_supervised_weight"]),
                    dtype=np.float32,
                ),
            ]
        )
        reconstruction_weights = np.concatenate(
            [
                reconstruction_weights,
                np.full(
                    n_synthetic,
                    float(resolved["synthetic_reconstruction_weight"]),
                    dtype=np.float32,
                ),
            ]
        )
    elif transfer_method != "real_only":
        raise ValueError(f"Unsupported frozen AE transfer method: {transfer_method}.")

    settings = dict(resolved["ae_settings"])
    refit_epochs = int(frozen.training_protocol["refit_epochs"])
    ae_config = SupervisedAEConfig(
        input_dim=int(settings["input_dim"]),
        latent_dim=int(resolved["latent_dim"]),
        hidden_dims=[int(value) for value in settings["hidden_dims"]],
        dropout=float(settings["dropout"]),
        reconstruction_weight=float(settings["reconstruction_weight"]),
        yield_weight=float(settings["yield_weight"]),
        latent_l2_weight=float(settings["latent_l2_weight"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        batch_size=int(settings["batch_size"]),
        max_epochs=refit_epochs,
        patience=int(settings["patience"]),
        random_state=int(settings["random_state"]),
        device=str(settings["device"]),
    )
    artifacts = fit_supervised_autoencoder(
        X_train,
        y_train,
        None,
        None,
        ae_config,
        sample_weight=yield_weights,
        reconstruction_sample_weight=reconstruction_weights,
    )
    history = artifacts["history"]
    if len(history) != refit_epochs:
        raise ValueError("AE refit did not execute the predefined epoch count.")
    latent_train = encode_with_supervised_autoencoder(artifacts, X_train)
    downstream = get_model(
        str(resolved["downstream_model"]),
        seed=int(resolved["seed"]),
        **dict(resolved["downstream_params"]),
    )
    downstream = train_model(
        downstream,
        latent_train,
        y_train,
        sample_weight=yield_weights,
    )
    return _RefittedAEPipeline(
        artifacts=artifacts,
        downstream=downstream,
        n_synthetic=n_synthetic,
        synthetic_parent_ids=parent_ids,
        synthetic_identity=synthetic_identity,
        refit_history_epochs=len(history),
    )


def _derived_leakage_audit(
    *,
    refit_ids: set[str],
    valid_ids: set[str],
    test_ids: set[str],
    parent_ids: set[str],
    expected_refit_ids: set[str],
) -> dict[str, Any]:
    missing_refit = expected_refit_ids - refit_ids
    unexpected_refit = refit_ids - expected_refit_ids
    refit_valid_overlap = refit_ids & valid_ids
    refit_test_overlap = refit_ids & test_ids
    parent_outside_refit = parent_ids - refit_ids
    parent_valid_overlap = parent_ids & valid_ids
    parent_test_overlap = parent_ids & test_ids
    leakage = bool(
        missing_refit
        or unexpected_refit
        or refit_valid_overlap
        or refit_test_overlap
        or parent_outside_refit
        or parent_valid_overlap
        or parent_test_overlap
    )
    return {
        "missing_expected_refit_count": len(missing_refit),
        "unexpected_refit_count": len(unexpected_refit),
        "refit_validation_overlap_count": len(refit_valid_overlap),
        "refit_test_overlap_count": len(refit_test_overlap),
        "synthetic_parent_outside_refit_count": len(parent_outside_refit),
        "synthetic_parent_validation_overlap_count": len(parent_valid_overlap),
        "synthetic_parent_test_overlap_count": len(parent_test_overlap),
        "leakage_detected": leakage,
    }


def _claim_evaluation(
    output: Path,
    *,
    evaluation_unit: str,
    frozen_policy_hash: str,
    binding: ScientificBinding,
    global_claim_id: str,
) -> Path:
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to reuse joint-AE final output directory: {output}"
        ) from exc
    claim_path = output / "evaluation_claim.json"
    claim = {
        "schema_version": AE_EVALUATION_CLAIM_SCHEMA_VERSION,
        "status": "claimed",
        "evaluation_unit": evaluation_unit,
        "frozen_policy_hash": frozen_policy_hash,
        "global_claim_id": global_claim_id,
        "scientific_binding": binding.to_dict(),
        "outer_test_attempts": 0,
        "outer_test_labels_accessed": False,
        "outer_test_prediction_batches": 0,
    }
    temporary = output / ".evaluation_claim.json.tmp"
    temporary.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n")
    temporary.replace(claim_path)
    return claim_path


def _claim_global_evaluation(
    search_manifest_path: Path,
    *,
    evaluation_unit: str,
    frozen_policy_hash: str,
    binding: ScientificBinding,
    output_directory: Path,
) -> tuple[Path, str]:
    """Atomically reserve one test attempt across all possible output paths."""
    claim_id = stable_hash(
        {
            "evaluation_unit": evaluation_unit,
            "frozen_policy_hash": frozen_policy_hash,
            "scientific_binding": binding.to_dict(),
        }
    )
    claim_root = (
        search_manifest_path.parent / ".outer_test_evaluation_claims"
    )
    claim_root.mkdir(parents=True, exist_ok=True)
    token_directory = claim_root / claim_id
    try:
        token_directory.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            "Outer-test evaluation unit and frozen policy were already claimed."
        ) from exc
    claim_path = token_directory / "evaluation_claim.json"
    claim = {
        "schema_version": AE_EVALUATION_CLAIM_SCHEMA_VERSION,
        "status": "claimed",
        "global_claim_id": claim_id,
        "evaluation_unit": evaluation_unit,
        "frozen_policy_hash": frozen_policy_hash,
        "scientific_binding": binding.to_dict(),
        "output_directory": str(output_directory),
        "outer_test_attempts": 0,
        "outer_test_labels_accessed": False,
        "outer_test_prediction_batches": 0,
    }
    temporary = token_directory / ".evaluation_claim.json.tmp"
    temporary.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n")
    temporary.replace(claim_path)
    return claim_path, claim_id


def _update_claim(claim_path: Path, *, status: str, **updates: Any) -> None:
    claim = json.loads(claim_path.read_text())
    claim.update({"status": status, **updates})
    temporary = claim_path.parent / ".evaluation_claim.json.tmp"
    temporary.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n")
    temporary.replace(claim_path)


def _update_global_claim(
    claim_path: Path,
    *,
    status: str,
    **updates: Any,
) -> None:
    _update_claim(claim_path, status=status, **updates)
