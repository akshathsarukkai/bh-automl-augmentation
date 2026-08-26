"""One-time outer-test evaluation of a hash-bound frozen real-only policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.augmentation.candidate_scope import observed_only_scope
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.evaluation_registry import (
    EvaluationAlreadyClaimedError,
    EvaluationIdentity,
    EvaluationRegistry,
    default_evaluation_registry_root,
)
from bh_augmentation.evaluation.policy_protocol import (
    FinalEvaluationInputs,
    FrozenPolicyEnvelope,
    ScientificBinding,
    evaluate_frozen_policy_once,
    load_frozen_policy,
)
from bh_augmentation.models.predict import predict_model
from bh_augmentation.policy_search import (
    REAL_ONLY_TRAINING_PROTOCOL,
    _build_partition,
    _evaluation_unit,
    _fit_resolved_model,
    _metric_rows,
    _resolve_candidate_scope,
    _resolve_run_contract,
    _resolved_model_policies,
    _selection_contract,
    current_commit,
    load_search_manifest,
    scientific_config_projection,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    resolve_corrected_feature_config,
    stable_hash,
)

FINAL_MANIFEST_SCHEMA_VERSION = "bh-final-evaluation-manifest-v1"


def run_final_evaluation_command(
    config_path: str | Path,
    *,
    frozen_policy_path: str | Path,
    search_manifest_path: str | Path,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Verify all bindings, refit, and evaluate one outer-test batch."""
    config = load_config(config_path)
    search_manifest = load_search_manifest(search_manifest_path)
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
    ]
    train_ids = set(
        assignment_rows.loc[
            assignment_rows["included_in_training_subset"], "source_row_id"
        ]
    )
    refit_frame = saved.canonical.loc[
        saved.canonical["source_row_id"].isin(train_ids)
    ].copy()
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
        config_hash=stable_hash(scientific_config_projection(config)),
        commit_hash=current_commit(),
    )
    frozen = load_frozen_policy(
        frozen_policy_path,
        expected_binding=expected_binding,
    )
    if dict(frozen.training_protocol) != REAL_ONLY_TRAINING_PROTOCOL:
        raise ValueError("Frozen policy declares an unsupported training protocol.")
    _verify_search_artifacts(
        search_manifest,
        search_manifest_path=Path(search_manifest_path),
        frozen=frozen,
        expected_binding=expected_binding,
        config=config,
        seed=seed,
        fraction=fraction,
    )
    evaluation_unit = _evaluation_unit(seed, fraction)
    if search_manifest["payload"]["evaluation_unit"] != evaluation_unit:
        raise ValueError("Frozen search evaluation unit does not match final config.")
    output = Path(
        output_directory
        if output_directory is not None
        else config.get("output", {}).get(
            "final_directory", "results/corrected_final_evaluation"
        )
    )
    test_ids = set(
        assignment_rows.loc[
            assignment_rows["outer_split"].eq("test"), "source_row_id"
        ]
    )
    metric_names, _, _ = _selection_contract(config)
    claim_path = _claim_evaluation(
        output,
        evaluation_unit=evaluation_unit,
        frozen_policy_hash=frozen.frozen_policy_hash,
        binding=expected_binding,
    )
    # Reserve the repository-global outer-test identity before the refit, so a
    # crash between refit and prediction can never leave the unit re-runnable.
    # The local directory claim above is filesystem-scoped; this one is not, and
    # it is what makes "never reuse a consumed outer-test identity" a mechanism
    # rather than a discipline.
    registry, registry_identity = _reserve_outer_test_identity(
        config,
        evaluation_unit=evaluation_unit,
        dataset_hash=saved.dataset_hash,
        search_manifest_hash=str(frozen.search_manifest_hash),
        frozen_policy_hash=frozen.frozen_policy_hash,
    )
    try:
        model = _fit_resolved_model(frozen.resolved_policy, refit)
    except Exception as exc:  # noqa: BLE001 - the registry must record any doubt
        if registry is not None and registry_identity is not None:
            registry.mark_failed_or_uncertain(
                registry_identity,
                reason=f"refit_failed:{type(exc).__name__}",
            )
        raise
    _update_claim(
        claim_path,
        status="refit_complete",
        outer_test_prediction_batches=0,
    )
    test_frame = saved.canonical.loc[
        saved.canonical["source_row_id"].isin(test_ids)
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
        predictions = predict_model(model, final_inputs.outer_test.X)
        return _metric_rows(
            final_inputs.outer_test.y,
            predictions,
            metric_names,
            split="test",
        )

    evaluated_units: set[str] = set()
    result = evaluate_frozen_policy_once(
        inputs,
        frozen,
        outer_evaluator,
        evaluated_units=evaluated_units,
    )
    metrics = result.metrics.assign(
        seed=seed,
        train_fraction=fraction,
        method=str(frozen.resolved_policy.config["method"]),
        model=str(frozen.resolved_policy.config["model"]),
        n_refit=len(refit.y),
        n_test=len(outer_test.y),
        test_prediction_batch=1,
    )
    metrics_bytes = metrics.to_csv(index=False).encode("utf-8")
    metrics_hash = hashlib.sha256(metrics_bytes).hexdigest()
    if registry is not None and registry_identity is not None:
        registry.mark_prediction_complete(
            registry_identity,
            prediction_hash=metrics_hash,
        )
        registry.mark_metrics_complete(
            registry_identity,
            metrics_payload=[
                {
                    "split": str(row["split"]),
                    "metric": str(row["metric"]),
                    "value": float(row["value"]),
                }
                for row in metrics.to_dict(orient="records")
            ],
        )
    manifest_payload = {
        "run_type": "final_evaluation",
        "scientific_binding": expected_binding.to_dict(),
        "evaluation_unit": evaluation_unit,
        "frozen_policy_hash": frozen.frozen_policy_hash,
        "selected_method": str(frozen.resolved_policy.config["method"]),
        "search_manifest_hash": frozen.search_manifest_hash,
        "selection_data_roles": ["train", "valid"],
        "refit_data_roles": ["train"],
        "outer_test_labels_accessed": True,
        "test_evaluated": True,
        "outer_test_prediction_batches": 1,
        "per_unit_test_evaluation_counts": {evaluation_unit: 1},
        "refit_source_id_hash": stable_hash(sorted(refit.source_row_ids)),
        "test_source_id_hash": stable_hash(sorted(outer_test.source_row_ids)),
        "final_test_metrics_sha256": metrics_hash,
    }
    manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
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
        final_test_metrics_sha256=metrics_hash,
        final_evaluation_manifest_hash=manifest["final_evaluation_manifest_hash"],
    )
    return paths


def _verify_search_artifacts(
    search_manifest: dict[str, Any],
    *,
    search_manifest_path: Path,
    frozen: FrozenPolicyEnvelope,
    expected_binding: ScientificBinding,
    config: dict[str, Any],
    seed: int,
    fraction: float,
) -> None:
    payload = search_manifest["payload"]
    if ScientificBinding.from_dict(payload["scientific_binding"]) != expected_binding:
        raise ValueError("Search manifest scientific binding mismatch.")
    if stable_hash(payload["scientific_config"]) != expected_binding.config_hash:
        raise ValueError("Search manifest scientific config hash mismatch.")
    if search_manifest["search_manifest_hash"] != frozen.search_manifest_hash:
        raise ValueError("Frozen policy search-manifest hash mismatch.")
    if search_manifest["frozen_policy_hash"] != frozen.frozen_policy_hash:
        raise ValueError("Search manifest frozen-policy hash mismatch.")
    if payload["selected_policy_hash"] != frozen.resolved_policy.policy_hash:
        raise ValueError("Search manifest selected-policy hash mismatch.")
    metrics_path = search_manifest_path.parent / "search_metrics.csv"
    if not metrics_path.is_file():
        raise ValueError("Search metrics artifact is missing.")
    metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if metrics_hash != payload["search_metrics_sha256"]:
        raise ValueError("Search metrics artifact hash mismatch.")
    search_metrics = pd.read_csv(metrics_path)
    runtime_selection = _selection_contract(config)
    if (
        payload["selection_metric"] != runtime_selection[1]
        or payload["lower_is_better"] != runtime_selection[2]
        or frozen.selection_metric != runtime_selection[1]
        or frozen.lower_is_better != runtime_selection[2]
    ):
        raise ValueError("Search selection protocol differs from runtime config.")
    _replay_policy_selection(
        search_metrics,
        frozen=frozen,
        manifest_payload=payload,
        runtime_policies=_resolved_model_policies(
            config,
            seed=seed,
            fraction=fraction,
        ),
        selection_metric=runtime_selection[1],
        lower_is_better=runtime_selection[2],
    )


def _replay_policy_selection(
    metrics: pd.DataFrame,
    *,
    frozen: FrozenPolicyEnvelope,
    manifest_payload: dict[str, Any],
    runtime_policies: list[Any],
    selection_metric: str,
    lower_is_better: bool,
) -> None:
    required = {"split", "metric", "value", "policy_id", "policy_hash", "selected_policy"}
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Search metrics schema is missing fields: {missing}.")
    if metrics.empty or not metrics["split"].eq("valid").all():
        raise ValueError("Search metrics must contain validation rows only.")
    runtime_by_id = {policy.policy_id: policy for policy in runtime_policies}
    runtime_hashes = [policy.policy_hash for policy in runtime_policies]
    if manifest_payload["candidate_policy_hashes"] != runtime_hashes:
        raise ValueError("Runtime candidate policies differ from frozen search candidates.")
    if set(metrics["policy_id"]) != set(runtime_by_id):
        raise ValueError("Search metrics policy IDs differ from runtime candidates.")
    expected_hashes = metrics["policy_id"].map(
        {policy_id: policy.policy_hash for policy_id, policy in runtime_by_id.items()}
    )
    if not metrics["policy_hash"].eq(expected_hashes).all():
        raise ValueError("Search metrics policy hash mismatch.")
    selected = metrics["selected_policy"]
    if not selected.isin([True, False]).all():
        raise ValueError("Search selected_policy values must be boolean.")
    selected_ids = set(metrics.loc[selected.astype(bool), "policy_id"])
    if len(selected_ids) != 1:
        raise ValueError("Search metrics must identify exactly one selected policy.")
    candidates = metrics.loc[metrics["metric"].eq(selection_metric)].copy()
    if len(candidates) != len(runtime_policies) or candidates["policy_id"].duplicated().any():
        raise ValueError("Selection metric must appear exactly once per candidate policy.")
    candidates["value"] = pd.to_numeric(candidates["value"], errors="coerce")
    if candidates["value"].isna().any() or not candidates["value"].map(math.isfinite).all():
        raise ValueError("Search selection values must be finite.")
    winner = candidates.sort_values(
        ["value", "policy_hash", "policy_id"],
        ascending=[lower_is_better, True, True],
        kind="mergesort",
    ).iloc[0]
    winner_policy = runtime_by_id[str(winner["policy_id"])]
    expected_selected = metrics["policy_id"].eq(winner_policy.policy_id)
    if (
        selected_ids != {winner_policy.policy_id}
        or not selected.astype(bool).eq(expected_selected).all()
    ):
        raise ValueError("Recorded selected policy is not the deterministic validation winner.")
    if (
        frozen.resolved_policy != winner_policy
        or frozen.selection_metric != selection_metric
        or frozen.lower_is_better != lower_is_better
        or not math.isclose(
            frozen.selection_value,
            float(winner["value"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or manifest_payload["selected_policy_hash"] != winner_policy.policy_hash
    ):
        raise ValueError("Frozen policy does not match the recomputed validation winner.")


def _reserve_outer_test_identity(
    config: dict[str, Any],
    *,
    evaluation_unit: str,
    dataset_hash: str,
    search_manifest_hash: str,
    frozen_policy_hash: str,
) -> tuple[EvaluationRegistry | None, EvaluationIdentity | None]:
    """Reserve this unit in the repository-global registry, if one is declared.

    A run declares its family with ``evaluation_registry.family``; the family
    name is part of the identity, so two scientifically different families may
    evaluate the same split without either one silently re-using the other's
    claim, while the same family cannot evaluate the same unit twice.
    """
    declared = config.get("evaluation_registry", {})
    if not isinstance(declared, dict):
        raise ValueError("evaluation_registry must be a mapping.")
    family = declared.get("family")
    if family is None:
        return None, None
    if not isinstance(family, str) or not family.strip():
        raise ValueError("evaluation_registry.family must be a non-empty string.")
    identity = EvaluationIdentity(
        dataset_hash=dataset_hash,
        split_or_search_manifest_hash=search_manifest_hash,
        frozen_policy_hash=frozen_policy_hash,
        evaluation_unit=f"{evaluation_unit}|family={family}",
    )
    registry = EvaluationRegistry(default_evaluation_registry_root())
    if registry.inspect(identity) is not None:
        raise EvaluationAlreadyClaimedError(
            "Outer-test evaluation identity was already reserved; reevaluation "
            f"is prohibited: {identity.evaluation_unit}."
        )
    registry.reserve(identity)
    return registry, identity


def _claim_evaluation(
    output: Path,
    *,
    evaluation_unit: str,
    frozen_policy_hash: str,
    binding: ScientificBinding,
) -> Path:
    """Atomically claim a fresh output path before refit or test access."""
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to reuse policy-protocol output directory: {output}"
        ) from exc
    claim_path = output / "evaluation_claim.json"
    claim = {
        "schema_version": "bh-final-evaluation-claim-v1",
        "status": "claimed",
        "evaluation_unit": evaluation_unit,
        "frozen_policy_hash": frozen_policy_hash,
        "scientific_binding": binding.to_dict(),
        "outer_test_prediction_batches": 0,
    }
    temporary = output / ".evaluation_claim.json.tmp"
    temporary.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n")
    temporary.replace(claim_path)
    return claim_path


def _update_claim(claim_path: Path, *, status: str, **updates: Any) -> None:
    claim = json.loads(claim_path.read_text())
    claim.update({"status": status, **updates})
    temporary = claim_path.parent / ".evaluation_claim.json.tmp"
    temporary.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n")
    temporary.replace(claim_path)
