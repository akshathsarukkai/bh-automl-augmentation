"""Validation-only model-policy search on canonical saved BH splits."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    CandidateScopePolicy,
    CandidateScopeViolation,
)
from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.pool_accounting import pool_statistics
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    clear_role_aware_teacher_cache,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.data.saved_canonical_splits import (
    SavedCanonicalSplits,
    load_saved_canonical_splits,
)
from bh_augmentation.evaluation.low_data_partitions import (
    build_low_data_evaluation_unit,
)
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.evaluation.policy_protocol import (
    PolicySearchInputs,
    ResolvedPolicy,
    ScientificBinding,
    freeze_policy,
    run_policy_search,
)
from bh_augmentation.features.compatibility import FeatureMetadata, assert_feature_compatibility
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    stable_hash,
)

SEARCH_MANIFEST_SCHEMA_VERSION = "bh-policy-search-manifest-v1"
REAL_ONLY_TRAINING_PROTOCOL = {
    "method": "real_only_or_training_only_typed_transfer",
    "refit_rows": "saved_canonical_training_subset",
    "validation_rows": "selection_only_not_refit",
    "synthetic_rows": "training_sources_and_donors_only",
    "outer_test_prediction_batches": 1,
}
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
    "candidate_policy_hashes",
    "selected_policy_hash",
    "selection_metric",
    "lower_is_better",
    "selection_data_roles",
    "outer_test_labels_accessed",
    "outer_test_predictions_generated",
    "outer_test_metric_evaluations",
    "test_evaluated",
    "search_metrics_sha256",
    "supported_methods",
}
_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman_corr}

DEGENERATE_UNIT_SCHEMA_VERSION = "bh-degenerate-search-unit-v1"
DEGENERATE_UNIT_STATUS = "degenerate_no_frozen_policy"
POOL_ACCOUNTING_SCHEMA_VERSION = "bh-search-pool-accounting-v1"
DEGENERATE_POOL_MESSAGE = (
    "Typed augmentation policy produced zero accepted synthetic rows and "
    "cannot be credited as augmentation."
)


class DegenerateAugmentationPool(ValueError):
    """A typed augmentation policy accepted zero synthetic rows.

    Under the frozen-policy protocol such a policy is *not* fitted as plain
    real-only training and silently counted as augmentation: an augmented arm
    that is identical to its comparator by construction must be reported as
    degenerate (preregistration section 6), never scored as a treatment.  The
    exception carries the pool accounting so the caller can record exactly how
    many candidates were generated and why each was rejected.
    """

    def __init__(self, pool: dict[str, Any] | None = None) -> None:
        super().__init__(DEGENERATE_POOL_MESSAGE)
        self.pool_statistics = dict(pool or {})


@dataclass(frozen=True)
class LabeledPartition:
    """Features and labels for one explicitly allowed scientific partition."""

    X: np.ndarray
    y: np.ndarray
    source_row_ids: tuple[str, ...]
    frame: pd.DataFrame
    feature_config: dict[str, Any]
    feature_names: tuple[str, ...]
    feature_metadata: FeatureMetadata
    candidate_scope: CandidateScopePolicy


def run_policy_search_command(
    config_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    record_degenerate: bool = False,
) -> dict[str, Path]:
    """Search real-only model policies without materializing outer-test labels.

    With ``record_degenerate`` the typed policies' candidate pools are generated
    once before the search and their accounting is written next to the search
    outputs as ``pool_accounting.json``.  If *every* typed policy accepts zero
    synthetic rows the unit is degenerate: ``degenerate_unit.json`` is written
    instead of a frozen policy, nothing is searched, no outer-test identity is
    claimed, and the returned mapping carries ``degenerate_unit`` rather than
    ``frozen_policy``.  The flag lives on the command line only, so the
    scientific configuration and every hash derived from it are unchanged.
    """
    config = load_config(config_path)
    dataset_path, split_directory, seed, fraction = _resolve_run_contract(config)
    saved = load_saved_canonical_splits(
        dataset_path,
        split_directory,
        requested_seeds=[seed],
        requested_fractions=[fraction],
    )
    saved = _redact_outer_test_outcomes(saved, seed=seed, train_fraction=fraction)
    candidate_scope = _resolve_candidate_scope(
        config,
        saved,
        seed=seed,
        train_fraction=fraction,
        dataset_path=dataset_path,
    )
    train_frame, validation_frame = _materialize_search_frames(
        saved, seed=seed, train_fraction=fraction
    )
    feature_config = resolve_corrected_feature_config(
        config.get("features", {}),
        required_kind="bh_role_separated",
    )
    train, train_contract = _build_partition(
        train_frame,
        feature_config,
        candidate_scope=candidate_scope,
    )
    validation, validation_contract = _build_partition(
        validation_frame,
        feature_config,
        candidate_scope=candidate_scope,
    )
    if train_contract != validation_contract:
        raise ValueError("Train and validation feature contracts differ.")

    config_projection = scientific_config_projection(config)
    audit = saved.audit_record(seed=seed, train_fraction=fraction)
    evaluation_unit = _evaluation_unit(seed, fraction)
    binding = ScientificBinding(
        dataset_hash=saved.dataset_hash,
        split_hash=str(audit["per_seed_split_hash"]),
        split_aggregate_hash=saved.aggregate_split_hash,
        per_seed_split_hash=str(audit["per_seed_split_hash"]),
        source_id_split_hash=str(audit["source_id_split_hash"]),
        canonicalization_version=str(audit["canonicalization_version"]),
        split_schema_version=str(audit["split_schema_version"]),
        feature_metadata_hash=str(train_contract["feature_metadata_hash"]),
        config_hash=stable_hash(config_projection),
        commit_hash=current_commit(),
    )
    inputs = PolicySearchInputs(train=train, validation=validation, binding=binding)
    policies = _resolved_model_policies(config, seed=seed, fraction=fraction)
    metric_names, selection_metric, lower_is_better = _selection_contract(config)
    output = Path(
        output_directory
        if output_directory is not None
        else config.get("output", {}).get(
            "search_directory", "results/corrected_policy_search"
        )
    )

    pool_accounting: list[dict[str, Any]] | None = None
    if record_degenerate:
        pool_accounting = _account_policy_pools(policies, train)
        typed = [record for record in pool_accounting if record["method"] != "real_only"]
        degenerate = [
            record
            for record in typed
            if record["pool_statistics"]["accepted_candidate_count"] == 0
        ]
        if typed and len(degenerate) == len(typed):
            _create_fresh_directory(output)
            payload = {
                "run_type": "policy_search_degenerate_pool",
                "scientific_binding": binding.to_dict(),
                "scientific_config": config_projection,
                "evaluation_unit": evaluation_unit,
                "seed": seed,
                "train_fraction": fraction,
                "family": _declared_registry_family(config),
                "candidate_scope_mode": candidate_scope.mode,
                "candidate_policy_hashes": [policy.policy_hash for policy in policies],
                "per_policy": pool_accounting,
                "reason": DEGENERATE_POOL_MESSAGE,
                "n_train": int(len(train.y)),
                "n_validation": int(len(validation.y)),
                "outer_test_labels_accessed": False,
                "outer_test_predictions_generated": False,
                "test_evaluated": False,
                "frozen_policy_written": False,
                "registry_claimed": False,
            }
            record = {
                "schema_version": DEGENERATE_UNIT_SCHEMA_VERSION,
                "status": DEGENERATE_UNIT_STATUS,
                "payload": payload,
                "degenerate_unit_hash": stable_hash(payload),
            }
            degenerate_path = output / "degenerate_unit.json"
            degenerate_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            return {"directory": output, "degenerate_unit": degenerate_path}
        if degenerate:
            raise DegenerateAugmentationPool(degenerate[0]["pool_statistics"])

    def validation_evaluator(
        search_inputs: PolicySearchInputs,
        policy: ResolvedPolicy,
    ) -> pd.DataFrame:
        model = _fit_resolved_model(policy, search_inputs.train)
        predictions = predict_model(model, search_inputs.validation.X)
        return _metric_rows(
            search_inputs.validation.y,
            predictions,
            metric_names,
            split="valid",
        )

    result = run_policy_search(
        inputs,
        policies,
        validation_evaluator,
        selection_metric=selection_metric,
        lower_is_better=lower_is_better,
    )
    search_metrics = result.search_metrics.assign(
        evaluation_unit=evaluation_unit,
        seed=seed,
        train_fraction=fraction,
        method=result.search_metrics["policy_id"].map(
            {policy.policy_id: policy.config["method"] for policy in policies}
        ),
        model=result.search_metrics["policy_id"].map(
            {policy.policy_id: policy.config["model"] for policy in policies}
        ),
        n_train=len(train.y),
        n_validation=len(validation.y),
    )
    search_metrics_bytes = search_metrics.to_csv(index=False).encode("utf-8")
    search_metrics_hash = hashlib.sha256(search_metrics_bytes).hexdigest()
    manifest_payload = {
        "run_type": "policy_search",
        "scientific_binding": binding.to_dict(),
        "scientific_config": config_projection,
        "evaluation_unit": evaluation_unit,
        "candidate_policy_hashes": [policy.policy_hash for policy in policies],
        "selected_policy_hash": result.selected_policy.policy_hash,
        "selection_metric": selection_metric,
        "lower_is_better": lower_is_better,
        "selection_data_roles": ["train", "valid"],
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metric_evaluations": 0,
        "test_evaluated": False,
        "search_metrics_sha256": search_metrics_hash,
        "supported_methods": ["real_only", "anonymous", "role_aware"],
    }
    search_manifest_hash = stable_hash(manifest_payload)
    frozen = freeze_policy(
        result,
        training_protocol=REAL_ONLY_TRAINING_PROTOCOL,
        search_manifest_hash=search_manifest_hash,
    )
    manifest = {
        "schema_version": SEARCH_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "payload": manifest_payload,
        "search_manifest_hash": search_manifest_hash,
        "frozen_policy_hash": frozen.frozen_policy_hash,
    }

    _create_fresh_directory(output)
    paths = {
        "directory": output,
        "frozen_policy": output / "frozen_policy.json",
        "search_metrics": output / "search_metrics.csv",
        "search_manifest": output / "search_manifest.json",
    }
    paths["search_metrics"].write_bytes(search_metrics_bytes)
    frozen.write_json(paths["frozen_policy"])
    paths["search_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if pool_accounting is not None:
        accounting_payload = {
            "evaluation_unit": evaluation_unit,
            "seed": seed,
            "train_fraction": fraction,
            "family": _declared_registry_family(config),
            "candidate_scope_mode": candidate_scope.mode,
            "search_manifest_hash": search_manifest_hash,
            "frozen_policy_hash": frozen.frozen_policy_hash,
            "per_policy": pool_accounting,
        }
        paths["pool_accounting"] = output / "pool_accounting.json"
        paths["pool_accounting"].write_text(
            json.dumps(
                {
                    "schema_version": POOL_ACCOUNTING_SCHEMA_VERSION,
                    "payload": accounting_payload,
                    "pool_accounting_hash": stable_hash(accounting_payload),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return paths


def load_degenerate_unit(path: str | Path) -> dict[str, Any]:
    """Strictly load and hash-verify a degenerate-unit record."""
    try:
        record = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Degenerate-unit record contains invalid JSON.") from exc
    expected_fields = {"schema_version", "status", "payload", "degenerate_unit_hash"}
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise ValueError("Degenerate-unit record schema mismatch.")
    if (
        record["schema_version"] != DEGENERATE_UNIT_SCHEMA_VERSION
        or record["status"] != DEGENERATE_UNIT_STATUS
    ):
        raise ValueError("Unsupported degenerate-unit record.")
    payload = record["payload"]
    if stable_hash(payload) != record["degenerate_unit_hash"]:
        raise ValueError("Degenerate-unit record hash mismatch.")
    if (
        payload.get("outer_test_labels_accessed") is not False
        or payload.get("test_evaluated") is not False
        or payload.get("frozen_policy_written") is not False
        or payload.get("registry_claimed") is not False
    ):
        raise ValueError("Degenerate-unit record reports forbidden activity.")
    typed = [entry for entry in payload["per_policy"] if entry["method"] != "real_only"]
    if not typed or any(
        entry["pool_statistics"]["accepted_candidate_count"] != 0 for entry in typed
    ):
        raise ValueError("Degenerate-unit record is not degenerate.")
    return record


def _declared_registry_family(config: dict[str, Any]) -> str | None:
    declared = config.get("evaluation_registry", {})
    if not isinstance(declared, dict):
        raise ValueError("evaluation_registry must be a mapping.")
    family = declared.get("family")
    return None if family is None else str(family)


def _account_policy_pools(
    policies: list[ResolvedPolicy],
    partition: LabeledPartition,
) -> list[dict[str, Any]]:
    """Generate each typed policy's candidate pool once and summarize it.

    Generation is deterministic under the policy's ``random_state``, so the
    pool the search later regenerates is the pool accounted for here.  Real-only
    policies generate nothing and are recorded with an empty accounting.
    """
    records = []
    for policy in policies:
        generated = _generate_synthetic(policy, partition)
        if generated is None:
            statistics = pool_statistics(pd.DataFrame(), partition.candidate_scope)
        else:
            statistics = pool_statistics(generated["candidate_df"], partition.candidate_scope)
        records.append(
            {
                "policy_id": policy.policy_id,
                "policy_hash": policy.policy_hash,
                "method": str(policy.config.get("method", "real_only")),
                "model": str(policy.config["model"]),
                "pool_statistics": statistics,
            }
        )
    return records


def scientific_config_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the output-independent scientific configuration bound at search."""
    keys = (
        "candidate_scope",
        "dataset",
        "splits",
        "low_data",
        "features",
        "metrics",
        "policy_search",
    )
    return json.loads(
        json.dumps({key: config[key] for key in keys if key in config}, sort_keys=True)
    )


def load_search_manifest(path: str | Path) -> dict[str, Any]:
    """Strictly load and hash-verify a completed validation-only search manifest."""
    try:
        manifest = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Search manifest contains invalid JSON.") from exc
    if not isinstance(manifest, dict) or set(manifest) != _SEARCH_MANIFEST_FIELDS:
        raise ValueError("Search manifest schema mismatch.")
    if (
        manifest["schema_version"] != SEARCH_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "complete"
    ):
        raise ValueError("Final evaluation requires a completed policy search.")
    payload = manifest["payload"]
    if not isinstance(payload, dict) or set(payload) != _SEARCH_PAYLOAD_FIELDS:
        raise ValueError("Search manifest payload schema mismatch.")
    if stable_hash(payload) != manifest["search_manifest_hash"]:
        raise ValueError("Search manifest hash mismatch.")
    if (
        payload.get("run_type") != "policy_search"
        or payload.get("selection_data_roles") != ["train", "valid"]
        or payload.get("outer_test_labels_accessed") is not False
        or payload.get("outer_test_predictions_generated") is not False
        or payload.get("outer_test_metric_evaluations") != 0
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError("Search manifest reports forbidden outer-test access.")
    return manifest


def current_commit() -> str:
    """Return the commit bound to scientific policy selection."""
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Scientific runs require an available Git commit.") from exc
    if not value:
        raise ValueError("Scientific runs require a non-empty Git commit.")
    return value


def _resolve_run_contract(config: dict[str, Any]) -> tuple[Path, Path, int, float]:
    dataset_path = config.get("dataset", {}).get("path")
    splits = config.get("splits", {})
    if not dataset_path:
        raise ValueError("Policy search requires dataset.path.")
    if splits.get("method") != "canonical_saved" or not splits.get("directory"):
        raise ValueError("Policy search requires canonical_saved split assignments.")
    seeds = config.get("seeds", [config.get("seed", 42)])
    fractions = config.get("low_data", {}).get("train_fractions", [1.0])
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise ValueError("The v1 frozen-policy protocol requires exactly one seed.")
    if not isinstance(fractions, list) or len(fractions) != 1:
        raise ValueError("The v1 frozen-policy protocol requires exactly one train fraction.")
    return Path(dataset_path), Path(splits["directory"]), int(seeds[0]), float(fractions[0])


def _materialize_search_frames(
    saved: SavedCanonicalSplits,
    *,
    seed: int,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = saved.low_data_assignments.loc[
        saved.low_data_assignments["seed"].eq(seed)
        & saved.low_data_assignments["train_fraction"].eq(train_fraction)
    ]
    if len(rows) != len(saved.canonical):
        raise ValueError("Requested saved split slice is incomplete.")
    train_ids = set(rows.loc[rows["included_in_training_subset"], "source_row_id"])
    validation_ids = set(rows.loc[rows["outer_split"].eq("valid"), "source_row_id"])
    test_ids = set(rows.loc[rows["outer_split"].eq("test"), "source_row_id"])
    if train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Outer-test rows entered a search-visible partition.")
    train = saved.canonical.loc[saved.canonical["source_row_id"].isin(train_ids)].copy()
    validation = saved.canonical.loc[
        saved.canonical["source_row_id"].isin(validation_ids)
    ].copy()
    if train["yield"].isna().any() or validation["yield"].isna().any():
        raise ValueError("Search-visible train/validation outcomes are missing.")
    return (
        train,
        validation,
    )


def _redact_outer_test_outcomes(
    saved: SavedCanonicalSplits,
    *,
    seed: int,
    train_fraction: float,
) -> SavedCanonicalSplits:
    """Replace outer-test outcomes with NaN before search-facing materialization."""
    rows = saved.low_data_assignments.loc[
        saved.low_data_assignments["seed"].eq(seed)
        & saved.low_data_assignments["train_fraction"].eq(train_fraction)
    ]
    test_ids = set(rows.loc[rows["outer_split"].eq("test"), "source_row_id"])
    redacted = saved.canonical.copy()
    redacted.loc[redacted["source_row_id"].isin(test_ids), "yield"] = np.nan
    if not redacted.loc[redacted["source_row_id"].isin(test_ids), "yield"].isna().all():
        raise AssertionError("Outer-test outcomes were not fully redacted.")
    return replace(saved, canonical=redacted)


def _resolve_candidate_scope(
    config: dict[str, Any],
    saved: SavedCanonicalSplits,
    *,
    seed: int,
    train_fraction: float,
    dataset_path: Path,
) -> CandidateScopePolicy:
    """Resolve the declared candidate-eligibility rule for this search.

    Absent a ``candidate_scope`` block the historical rule applies: eligibility
    is decided against the complete measured universe.  That rule answers a
    prospective-novelty question, so a low-data augmentation experiment must
    declare ``mode: observed_only_low_data`` explicitly.
    """
    requested = config.get("candidate_scope", {})
    if not isinstance(requested, dict):
        raise CandidateScopeViolation("candidate_scope must be a mapping.")
    mode = str(requested.get("mode", GLOBALLY_UNMEASURED_PROSPECTIVE))
    unit = build_low_data_evaluation_unit(
        saved,
        seed=seed,
        train_fraction=train_fraction,
        dataset_path=dataset_path,
    )
    return unit.candidate_scope(mode)


def _build_partition(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    *,
    candidate_scope: CandidateScopePolicy,
) -> tuple[LabeledPartition, dict[str, Any]]:
    X, y, names, metadata = build_feature_matrix_with_metadata(frame, feature_config)
    contract = feature_contract_record(metadata, names)
    return (
        LabeledPartition(
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(y, dtype=np.float32),
            source_row_ids=tuple(frame["source_row_id"].astype(str)),
            frame=frame.copy(),
            feature_config=dict(feature_config),
            feature_names=tuple(names),
            feature_metadata=metadata,
            candidate_scope=candidate_scope,
        ),
        contract,
    )


def _resolved_model_policies(
    config: dict[str, Any],
    *,
    seed: int,
    fraction: float,
) -> list[ResolvedPolicy]:
    raw = config.get("policy_search", {}).get("model_policies")
    if not isinstance(raw, list) or not raw:
        raise ValueError("policy_search.model_policies must be a non-empty list.")
    policies = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each model policy must be a mapping.")
        method = str(item.get("method", "real_only"))
        expected = {"policy_id", "model", "params"}
        if method != "real_only":
            expected |= {"method", "transfer_config"}
        elif "method" in item:
            expected |= {"method"}
        if set(item) != expected:
            raise ValueError(f"Policy schema mismatch for method {method!r}.")
        if not isinstance(item["params"], dict):
            raise ValueError("Real-only model policy params must be a mapping.")
        resolved_transfer: dict[str, Any] | None = None
        if method == "anonymous":
            resolved_transfer = asdict(ConditionTransferConfig(**item["transfer_config"]))
        elif method == "role_aware":
            resolved_transfer = asdict(
                RoleAwareConditionTransferConfig(**item["transfer_config"])
            )
        elif method != "real_only":
            raise ValueError(f"Unsupported policy method: {method}.")
        if method != "real_only" and (
            resolved_transfer["donor_similarity_backend"] != "rdkit"
            or resolved_transfer["role_change_requirement"] != "all"
            or resolved_transfer["fallback_policy"] != "reject"
        ):
            raise ValueError(
                "Typed scientific policies require RDKit similarity, "
                "role_change_requirement='all', and fallback_policy='reject'."
            )
        resolved_config = {
            "method": method,
            "model": str(item["model"]),
            "model_params": item["params"],
            "seed": seed,
            "train_fraction": fraction,
        }
        if resolved_transfer is not None:
            resolved_config["transfer_config"] = resolved_transfer
        policies.append(
            ResolvedPolicy(
                policy_id=str(item["policy_id"]),
                config=resolved_config,
            )
        )
    return policies


def _selection_contract(config: dict[str, Any]) -> tuple[list[str], str, bool]:
    metrics = config.get("metrics", ["rmse", "mae"])
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(metric not in _METRICS for metric in metrics)
    ):
        raise ValueError(f"Unsupported policy-search metrics: {metrics!r}.")
    search = config.get("policy_search", {})
    selection_metric = str(search.get("selection_metric", "rmse"))
    if selection_metric not in metrics:
        raise ValueError("Selection metric must be included in metrics.")
    lower_is_better = search.get("lower_is_better", True)
    if not isinstance(lower_is_better, bool):
        raise ValueError("policy_search.lower_is_better must be boolean.")
    return list(metrics), selection_metric, lower_is_better


def _generate_synthetic(
    policy: ResolvedPolicy,
    partition: LabeledPartition,
) -> dict[str, Any] | None:
    """Generate a typed policy's candidate pool from the partition alone.

    Returns ``None`` for real-only policies.  Every generator call receives the
    partition's declared candidate scope, so eligibility is decided by the rule
    the configuration bound, never by an implicit default.
    """
    resolved = policy.config
    method = resolved.get("method")
    if method == "anonymous":
        transfer = ConditionTransferConfig(**dict(resolved["transfer_config"]))
        return generate_condition_transfer_examples(
            partition.frame,
            partition.X,
            partition.y,
            transfer,
            feature_config=partition.feature_config,
            real_feature_names=list(partition.feature_names),
            real_feature_metadata=partition.feature_metadata,
            candidate_scope=partition.candidate_scope,
        )
    if method == "role_aware":
        clear_role_aware_teacher_cache()
        transfer = RoleAwareConditionTransferConfig(**dict(resolved["transfer_config"]))
        return generate_role_aware_condition_transfer_examples(
            partition.frame,
            partition.X,
            partition.y,
            transfer,
            feature_config=partition.feature_config,
            real_feature_names=list(partition.feature_names),
            real_feature_metadata=partition.feature_metadata,
            candidate_scope=partition.candidate_scope,
        )
    if method not in (None, "real_only"):
        raise ValueError(f"Unsupported resolved policy method: {method}.")
    return None


def _fit_resolved_model(policy: ResolvedPolicy, partition: LabeledPartition) -> Any:
    resolved = policy.config
    X_train = partition.X
    y_train = partition.y
    generated = _generate_synthetic(policy, partition)
    if generated is not None:
        X_train, y_train = _augment_training(partition, generated)
    model = get_model(
        str(resolved["model"]),
        seed=int(resolved["seed"]),
        **dict(resolved["model_params"]),
    )
    return train_model(model, X_train, y_train)


def _augment_training(
    partition: LabeledPartition,
    generated: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    X_synthetic = np.asarray(generated["X_synthetic"], dtype=np.float32)
    y_synthetic = np.asarray(generated["synthetic_y"], dtype=np.float32).reshape(-1)
    assert_feature_compatibility(
        partition.X,
        partition.feature_names,
        X_synthetic,
        generated["feature_names"],
        real_metadata=partition.feature_metadata,
        synthetic_metadata=generated["feature_metadata"],
    )
    if len(y_synthetic) == 0:
        raise DegenerateAugmentationPool(
            pool_statistics(generated["candidate_df"], partition.candidate_scope)
        )
    if len(X_synthetic) != len(y_synthetic):
        raise ValueError("Synthetic feature and label counts differ.")
    return (
        np.vstack([partition.X, X_synthetic]).astype(np.float32),
        np.concatenate([partition.y, y_synthetic]).astype(np.float32),
    )


def _metric_rows(
    y_true: np.ndarray,
    predictions: np.ndarray,
    metrics: list[str],
    *,
    split: str,
) -> pd.DataFrame:
    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite {split} predictions.")
    return pd.DataFrame(
        [
            {"split": split, "metric": metric, "value": _METRICS[metric](y_true, predictions)}
            for metric in metrics
        ]
    )


def _evaluation_unit(seed: int, fraction: float) -> str:
    return f"seed={seed}|train_fraction={fraction:.12g}|comparison=model_policy_search"


def _create_fresh_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to reuse policy-protocol output directory: {path}")
    path.mkdir(parents=True)
