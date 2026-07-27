"""Validation-only joint policy search for condition-transfer supervised AEs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import measured_canonical_keys
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.ae_policy_protocol import (
    AEInnerSplit,
    make_ae_inner_split,
)
from bh_augmentation.evaluation.policy_protocol import (
    PolicySearchInputs,
    ResolvedPolicy,
    ScientificBinding,
    freeze_policy,
    run_policy_search,
)
from bh_augmentation.features.compatibility import assert_feature_compatibility
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.policy_search import (
    LabeledPartition,
    _build_partition,
    _create_fresh_directory,
    _materialize_search_frames,
    _metric_rows,
    _redact_outer_test_outcomes,
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

AE_SEARCH_MANIFEST_SCHEMA_VERSION = "bh-joint-ae-policy-search-manifest-v1"
AE_REFIT_PROTOCOL_SCHEMA_VERSION = "bh-joint-ae-refit-protocol-v1"
_SUPPORTED_METRICS = {"rmse", "mae", "r2", "spearman"}
_AE_SETTING_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [128],
    "dropout": 0.0,
    "reconstruction_weight": 1.0,
    "yield_weight": 1.0,
    "latent_l2_weight": 0.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": 32,
    "max_epochs": 200,
    "patience": 20,
    "device": "auto",
    "internal_valid_size": 0.2,
    "minimum_inner_rows": 4,
    "minimum_inner_groups": 2,
    "refit_epochs": 50,
}


@dataclass(frozen=True)
class AEPolicySearchContext:
    """The three measured partitions visible to joint AE policy search."""

    ae_fit: LabeledPartition
    internal_validation: LabeledPartition
    policy_validation: LabeledPartition
    inner_split: AEInnerSplit
    forbidden_source_ids: frozenset[str]
    outer_validation_source_ids: frozenset[str]
    outer_test_source_ids: frozenset[str]


@dataclass(frozen=True)
class JointAEFitResult:
    """Validation metrics and derived training audit for one joint tuple."""

    metrics: pd.DataFrame
    audit: dict[str, Any]


def run_ae_policy_search_command(
    config_path: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Jointly select and freeze an AE policy without exposing outer-test outcomes."""
    config = load_config(config_path)
    dataset_path, split_directory, seed, fraction = _resolve_run_contract(config)
    output = Path(
        output_directory
        if output_directory is not None
        else config.get("output", {}).get(
            "ae_search_directory", "results/corrected_joint_ae_policy_search"
        )
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to reuse policy-protocol output directory: {output}"
        )

    saved = _redact_outer_test_outcomes(
        load_saved_canonical_splits(
            dataset_path,
            split_directory,
            requested_seeds=[seed],
            requested_fractions=[fraction],
        ),
        seed=seed,
        train_fraction=fraction,
    )
    train_frame, policy_validation_frame = _materialize_search_frames(
        saved,
        seed=seed,
        train_fraction=fraction,
    )
    assignment_rows = saved.low_data_assignments.loc[
        saved.low_data_assignments["seed"].eq(seed)
        & saved.low_data_assignments["train_fraction"].eq(fraction)
    ]
    outer_valid_ids = tuple(
        sorted(
            assignment_rows.loc[
                assignment_rows["outer_split"].eq("valid"), "source_row_id"
            ].astype(str)
        )
    )
    outer_test_ids = tuple(
        sorted(
            assignment_rows.loc[
                assignment_rows["outer_split"].eq("test"), "source_row_id"
            ].astype(str)
        )
    )
    search_config = _resolve_ae_search_config(config)
    inner_split = make_ae_inner_split(
        train_frame,
        valid_size=float(search_config["ae_settings"]["internal_valid_size"]),
        seed=seed + int(search_config["inner_split_seed_offset"]),
        forbidden_outer_valid_source_ids=outer_valid_ids,
        forbidden_outer_test_source_ids=outer_test_ids,
        minimum_rows=int(search_config["ae_settings"]["minimum_inner_rows"]),
        minimum_groups=int(search_config["ae_settings"]["minimum_inner_groups"]),
    )

    feature_config = resolve_corrected_feature_config(
        config.get("features", {}),
        required_kind="bh_role_separated",
    )
    global_identity_keys = frozenset(measured_canonical_keys(saved.canonical))
    ae_fit, fit_contract = _build_partition(
        inner_split.ae_train,
        feature_config,
        measured_identity_keys=global_identity_keys,
    )
    internal_validation, internal_contract = _build_partition(
        inner_split.internal_validation,
        feature_config,
        measured_identity_keys=global_identity_keys,
    )
    policy_validation, policy_contract = _build_partition(
        policy_validation_frame,
        feature_config,
        measured_identity_keys=global_identity_keys,
    )
    if fit_contract != internal_contract or fit_contract != policy_contract:
        raise ValueError("Joint AE search partitions have incompatible feature contracts.")
    context = AEPolicySearchContext(
        ae_fit=ae_fit,
        internal_validation=internal_validation,
        policy_validation=policy_validation,
        inner_split=inner_split,
        forbidden_source_ids=frozenset((*outer_valid_ids, *outer_test_ids)),
        outer_validation_source_ids=frozenset(outer_valid_ids),
        outer_test_source_ids=frozenset(outer_test_ids),
    )

    audit = saved.audit_record(seed=seed, train_fraction=fraction)
    scientific_config = scientific_ae_config_projection(config)
    binding = ScientificBinding(
        dataset_hash=saved.dataset_hash,
        split_hash=str(audit["per_seed_split_hash"]),
        split_aggregate_hash=saved.aggregate_split_hash,
        per_seed_split_hash=str(audit["per_seed_split_hash"]),
        source_id_split_hash=str(audit["source_id_split_hash"]),
        canonicalization_version=str(audit["canonicalization_version"]),
        split_schema_version=str(audit["split_schema_version"]),
        feature_metadata_hash=str(fit_contract["feature_metadata_hash"]),
        config_hash=stable_hash(scientific_config),
        commit_hash=current_commit(),
    )
    all_policies = _resolved_joint_ae_policies(
        search_config,
        seed=seed,
        fraction=fraction,
        input_dim=ae_fit.X.shape[1],
        inner_split_hash=inner_split.split_hash,
    )
    generations, transfer_audits, candidate_audit = _prepare_transfer_generations(
        all_policies,
        context,
    )
    feasible_policies, exclusions = _partition_feasible_policies(
        all_policies,
        generations,
    )
    if not feasible_policies:
        raise ValueError("No feasible joint AE policies remain after transfer generation.")

    inputs = PolicySearchInputs(
        train=context,
        validation=policy_validation,
        binding=binding,
    )
    fit_audits: dict[str, dict[str, Any]] = {}

    def validation_evaluator(
        search_inputs: PolicySearchInputs,
        policy: ResolvedPolicy,
    ) -> pd.DataFrame:
        if search_inputs.train is not context:
            raise ValueError("Joint AE evaluator received an unexpected search context.")
        result = _fit_joint_ae_policy(
            policy,
            context,
            generations[_transfer_key_from_policy(policy)],
            metric_names=search_config["metrics"],
        )
        fit_audits[policy.policy_id] = result.audit
        return result.metrics

    search_result = run_policy_search(
        inputs,
        feasible_policies,
        validation_evaluator,
        selection_metric=str(search_config["selection_metric"]),
        lower_is_better=bool(search_config["lower_is_better"]),
    )
    search_metrics = search_result.search_metrics.assign(
        evaluation_unit=_evaluation_unit(seed, fraction),
        seed=seed,
        train_fraction=fraction,
        inner_split_hash=inner_split.split_hash,
    )
    training_audit = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                **record,
            }
            for policy_id, record in sorted(fit_audits.items())
        ]
    )
    exclusions_df = pd.DataFrame(exclusions)
    if exclusions_df.empty:
        exclusions_df = pd.DataFrame(
            columns=["policy_id", "policy_hash", "transfer_key", "rejection_reason"]
        )
    transfer_audit_df = pd.DataFrame(transfer_audits)

    artifact_bytes = {
        "search_metrics": search_metrics.to_csv(index=False).encode("utf-8"),
        "search_exclusions": exclusions_df.to_csv(index=False).encode("utf-8"),
        "ae_training_audit": training_audit.to_csv(index=False).encode("utf-8"),
        "synthetic_transfer_audit": transfer_audit_df.to_csv(index=False).encode(
            "utf-8"
        ),
        "synthetic_candidate_audit": candidate_audit.to_csv(index=False).encode(
            "utf-8"
        ),
    }
    artifact_hashes = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in artifact_bytes.items()
    }
    manifest_payload = {
        "run_type": "joint_ae_policy_search",
        "scientific_binding": binding.to_dict(),
        "scientific_config": scientific_config,
        "evaluation_unit": _evaluation_unit(seed, fraction),
        "inner_split": inner_split.audit_record,
        "candidate_policy_hashes": [
            policy.policy_hash
            for policy in sorted(all_policies, key=lambda item: item.policy_hash)
        ],
        "feasible_policy_hashes": [
            policy.policy_hash
            for policy in sorted(feasible_policies, key=lambda item: item.policy_hash)
        ],
        "selected_policy_hash": search_result.selected_policy.policy_hash,
        "selection_metric": search_result.selection_metric,
        "lower_is_better": search_result.lower_is_better,
        "selection_data_roles": [
            "ae_fit",
            "ae_internal_validation",
            "saved_validation",
        ],
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metric_evaluations": 0,
        "test_evaluated": False,
        "artifact_hashes": artifact_hashes,
    }
    search_manifest_hash = stable_hash(manifest_payload)
    frozen = freeze_policy(
        search_result,
        training_protocol=_frozen_training_protocol(
            inner_split,
            search_result.selected_policy,
        ),
        search_manifest_hash=search_manifest_hash,
    )
    manifest = {
        "schema_version": AE_SEARCH_MANIFEST_SCHEMA_VERSION,
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
        "search_exclusions": output / "search_exclusions.csv",
        "ae_training_audit": output / "ae_training_audit.csv",
        "synthetic_transfer_audit": output / "synthetic_transfer_audit.csv",
        "synthetic_candidate_audit": output / "synthetic_candidate_audit.csv",
    }
    for name, value in artifact_bytes.items():
        paths[name].write_bytes(value)
    frozen.write_json(paths["frozen_policy"])
    paths["search_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return paths


def scientific_ae_config_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the output-independent configuration bound to joint AE search."""
    keys = (
        "dataset",
        "splits",
        "low_data",
        "features",
        "metrics",
        "ae_policy_search",
    )
    return json.loads(
        json.dumps(
            {key: config[key] for key in keys if key in config},
            sort_keys=True,
            allow_nan=False,
        )
    )


def _resolve_ae_search_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("ae_policy_search")
    if not isinstance(raw, dict):
        raise ValueError("ae_policy_search must be a mapping.")
    allowed = {
        "transfer_policies",
        "latent_dims",
        "synthetic_supervised_weights",
        "synthetic_reconstruction_weights",
        "downstream_models",
        "ae_settings",
        "selection_metric",
        "lower_is_better",
        "inner_split_seed_offset",
    }
    if set(raw) - allowed:
        raise ValueError(
            f"Unknown ae_policy_search keys: {sorted(set(raw) - allowed)}."
        )
    metrics = config.get("metrics", ["rmse", "mae"])
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(metric not in _SUPPORTED_METRICS for metric in metrics)
    ):
        raise ValueError(f"Unsupported joint AE search metrics: {metrics!r}.")
    selection_metric = str(raw.get("selection_metric", "rmse"))
    if selection_metric not in metrics:
        raise ValueError("Joint AE selection metric must be included in metrics.")
    lower_is_better = raw.get("lower_is_better", True)
    if not isinstance(lower_is_better, bool):
        raise ValueError("ae_policy_search.lower_is_better must be boolean.")

    transfer_policies = raw.get("transfer_policies")
    if not isinstance(transfer_policies, list) or not transfer_policies:
        raise ValueError("ae_policy_search.transfer_policies must be non-empty.")
    downstream_models = raw.get("downstream_models")
    if not isinstance(downstream_models, list) or not downstream_models:
        raise ValueError("ae_policy_search.downstream_models must be non-empty.")
    latent_dims = _positive_int_list(raw.get("latent_dims"), "latent_dims")
    supervised_weights = _nonnegative_float_list(
        raw.get("synthetic_supervised_weights"),
        "synthetic_supervised_weights",
    )
    reconstruction_weights = _nonnegative_float_list(
        raw.get("synthetic_reconstruction_weights"),
        "synthetic_reconstruction_weights",
    )
    inner_split_seed_offset = raw.get("inner_split_seed_offset", 20_000)
    if not isinstance(inner_split_seed_offset, int) or isinstance(
        inner_split_seed_offset, bool
    ):
        raise ValueError("ae_policy_search.inner_split_seed_offset must be an integer.")
    ae_settings = {**_AE_SETTING_DEFAULTS, **dict(raw.get("ae_settings", {}))}
    if set(ae_settings) != set(_AE_SETTING_DEFAULTS):
        raise ValueError("Unknown joint AE setting.")
    _validate_ae_settings(ae_settings)
    return {
        "metrics": list(metrics),
        "selection_metric": selection_metric,
        "lower_is_better": lower_is_better,
        "transfer_policies": transfer_policies,
        "downstream_models": downstream_models,
        "latent_dims": latent_dims,
        "synthetic_supervised_weights": supervised_weights,
        "synthetic_reconstruction_weights": reconstruction_weights,
        "ae_settings": ae_settings,
        "inner_split_seed_offset": inner_split_seed_offset,
    }


def _resolved_joint_ae_policies(
    search_config: dict[str, Any],
    *,
    seed: int,
    fraction: float,
    input_dim: int,
    inner_split_hash: str,
) -> list[ResolvedPolicy]:
    """Resolve the full joint Cartesian budget in traversal-independent order."""
    transfers = [
        _resolve_transfer_policy(item, seed=seed)
        for item in search_config["transfer_policies"]
    ]
    models = [
        _resolve_downstream_model(item)
        for item in search_config["downstream_models"]
    ]
    policies: dict[str, ResolvedPolicy] = {}
    for transfer in transfers:
        method = transfer["method"]
        supervised_weights = (
            [0.0]
            if method == "real_only"
            else search_config["synthetic_supervised_weights"]
        )
        reconstruction_weights = (
            [0.0]
            if method == "real_only"
            else search_config["synthetic_reconstruction_weights"]
        )
        for latent_dim in search_config["latent_dims"]:
            for supervised_weight in supervised_weights:
                for reconstruction_weight in reconstruction_weights:
                    for model in models:
                        resolved = {
                            "method": "condition_transfer_supervised_ae",
                            "transfer_method": method,
                            "transfer_config": transfer.get("transfer_config"),
                            "latent_dim": latent_dim,
                            "synthetic_supervised_weight": supervised_weight,
                            "synthetic_reconstruction_weight": reconstruction_weight,
                            "downstream_model": model["name"],
                            "downstream_params": model["params"],
                            "ae_settings": {
                                **search_config["ae_settings"],
                                "input_dim": input_dim,
                                "random_state": seed,
                            },
                            "seed": seed,
                            "train_fraction": fraction,
                            "inner_split_hash": inner_split_hash,
                        }
                        policy_id = f"joint-ae-{stable_hash(resolved)[:16]}"
                        policy = ResolvedPolicy(policy_id=policy_id, config=resolved)
                        if policy.policy_hash in policies:
                            raise ValueError("Joint AE search contains a duplicate tuple.")
                        policies[policy.policy_hash] = policy
    return [policies[key] for key in sorted(policies)]


def _resolve_transfer_policy(item: Any, *, seed: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each joint AE transfer policy must be a mapping.")
    method = item.get("method")
    if method == "real_only":
        if set(item) != {"method"}:
            raise ValueError("real_only transfer policy accepts no transfer_config.")
        return {"method": "real_only"}
    if method != "anonymous" or set(item) != {"method", "transfer_config"}:
        raise ValueError("Joint AE search supports real_only and anonymous transfer.")
    raw_config = item["transfer_config"]
    if not isinstance(raw_config, dict):
        raise ValueError("Anonymous transfer_config must be a mapping.")
    transfer = ConditionTransferConfig(**raw_config)
    if transfer.random_state != seed:
        raise ValueError(
            "Anonymous transfer random_state must equal the scientific run seed."
        )
    resolved = asdict(transfer)
    if (
        resolved["role_change_requirement"] != "all"
        or resolved["fallback_policy"] != "reject"
        or resolved["donor_similarity_backend"] != "rdkit"
    ):
        raise ValueError(
            "Scientific anonymous AE policies require exact all-role changes, "
            "reject fallback, and RDKit donor similarity."
        )
    return {"method": "anonymous", "transfer_config": resolved}


def _resolve_downstream_model(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": item, "params": {}}
    if (
        not isinstance(item, dict)
        or set(item) != {"name", "params"}
        or not isinstance(item["params"], dict)
    ):
        raise ValueError("Downstream models require exactly name and params.")
    return {"name": str(item["name"]), "params": dict(item["params"])}


def _prepare_transfer_generations(
    policies: list[ResolvedPolicy],
    context: AEPolicySearchContext,
) -> tuple[dict[str, dict[str, Any] | None], list[dict[str, Any]], pd.DataFrame]:
    generations: dict[str, dict[str, Any] | None] = {}
    audits: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    by_key = {
        _transfer_key_from_policy(policy): policy
        for policy in policies
    }
    for transfer_key in sorted(by_key):
        policy = by_key[transfer_key]
        method = str(policy.config["transfer_method"])
        if method == "real_only":
            generations[transfer_key] = None
            audits.append(
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
        transfer = ConditionTransferConfig(
            **dict(policy.config["transfer_config"])
        )
        result = generate_condition_transfer_examples(
            context.ae_fit.frame,
            context.ae_fit.X,
            context.ae_fit.y,
            transfer,
            feature_config=context.ae_fit.feature_config,
            real_feature_names=list(context.ae_fit.feature_names),
            real_feature_metadata=context.ae_fit.feature_metadata,
            measured_identity_keys=context.ae_fit.measured_identity_keys,
        )
        parent_ids = _validate_generated_parent_ids(
            result,
            context,
        )
        generations[transfer_key] = result
        metadata = dict(result["metadata"])
        n_synthetic = len(result["synthetic_y"])
        audits.append(
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
        candidate = result["candidate_df"].copy()
        candidate.insert(0, "transfer_key", transfer_key)
        candidate_frames.append(candidate)
    candidate_audit = (
        pd.concat(candidate_frames, ignore_index=True, sort=False)
        if candidate_frames
        else pd.DataFrame(columns=["transfer_key"])
    )
    return generations, audits, candidate_audit


def _validate_generated_parent_ids(
    generation: dict[str, Any],
    context: AEPolicySearchContext,
) -> set[str]:
    candidate = generation.get("candidate_df")
    if not isinstance(candidate, pd.DataFrame):
        raise ValueError("Synthetic generator did not return candidate provenance.")
    if candidate.empty:
        return set()
    required = {"source_row_id", "donor_row_id"}
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"Synthetic parent provenance is missing fields: {missing}.")
    if candidate[list(required)].isna().any().any():
        raise ValueError("Synthetic parent provenance contains missing source IDs.")
    parent_ids = set(candidate["source_row_id"].astype(str)) | set(
        candidate["donor_row_id"].astype(str)
    )
    allowed_ids = set(context.inner_split.ae_train_source_ids)
    outside = parent_ids - allowed_ids
    forbidden = parent_ids & context.forbidden_source_ids
    internal = parent_ids & set(context.inner_split.internal_validation_source_ids)
    if outside or forbidden or internal:
        raise ValueError(
            "Synthetic parent leakage detected: "
            f"outside_ae_fit={sorted(outside)}, "
            f"internal_validation={sorted(internal)}, "
            f"outer_forbidden={sorted(forbidden)}."
        )
    return parent_ids


def _partition_feasible_policies(
    policies: list[ResolvedPolicy],
    generations: dict[str, dict[str, Any] | None],
) -> tuple[list[ResolvedPolicy], list[dict[str, Any]]]:
    feasible: list[ResolvedPolicy] = []
    exclusions: list[dict[str, Any]] = []
    for policy in policies:
        transfer_key = _transfer_key_from_policy(policy)
        generation = generations[transfer_key]
        if (
            policy.config["transfer_method"] == "anonymous"
            and generation is not None
            and len(generation["synthetic_y"]) == 0
        ):
            exclusions.append(
                {
                    "policy_id": policy.policy_id,
                    "policy_hash": policy.policy_hash,
                    "transfer_key": transfer_key,
                    "rejection_reason": "zero_accepted_synthetic",
                }
            )
        else:
            feasible.append(policy)
    return feasible, exclusions


def _fit_joint_ae_policy(
    policy: ResolvedPolicy,
    context: AEPolicySearchContext,
    generation: dict[str, Any] | None,
    *,
    metric_names: list[str],
) -> JointAEFitResult:
    boundary_audit = _validate_search_context_boundaries(context)
    resolved = policy.config
    X_train = np.asarray(context.ae_fit.X, dtype=np.float32)
    y_train = np.asarray(context.ae_fit.y, dtype=np.float32)
    yield_weights = np.ones(len(y_train), dtype=np.float32)
    reconstruction_weights = np.ones(len(y_train), dtype=np.float32)
    n_synthetic = 0
    if resolved["transfer_method"] == "anonymous":
        if generation is None or len(generation["synthetic_y"]) == 0:
            raise ValueError("Infeasible typed AE policy reached model fitting.")
        _validate_generated_parent_ids(generation, context)
        X_synthetic = np.asarray(generation["X_synthetic"], dtype=np.float32)
        y_synthetic = np.asarray(generation["synthetic_y"], dtype=np.float32)
        assert_feature_compatibility(
            context.ae_fit.X,
            context.ae_fit.feature_names,
            X_synthetic,
            generation["feature_names"],
            real_metadata=context.ae_fit.feature_metadata,
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

    settings = dict(resolved["ae_settings"])
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
        max_epochs=int(settings["max_epochs"]),
        patience=int(settings["patience"]),
        random_state=int(settings["random_state"]),
        device=str(settings["device"]),
    )
    artifacts = fit_supervised_autoencoder(
        X_train,
        y_train,
        context.internal_validation.X,
        context.internal_validation.y,
        ae_config,
        sample_weight=yield_weights,
        reconstruction_sample_weight=reconstruction_weights,
    )
    Z_train = encode_with_supervised_autoencoder(artifacts, X_train)
    Z_validation = encode_with_supervised_autoencoder(
        artifacts,
        context.policy_validation.X,
    )
    downstream = get_model(
        str(resolved["downstream_model"]),
        seed=int(resolved["seed"]),
        **dict(resolved["downstream_params"]),
    )
    downstream = train_model(
        downstream,
        Z_train,
        y_train,
        sample_weight=yield_weights,
    )
    predictions = predict_model(downstream, Z_validation)
    metrics = _metric_rows(
        context.policy_validation.y,
        predictions,
        metric_names,
        split="valid",
    ).assign(
        transfer_method=resolved["transfer_method"],
        latent_dim=resolved["latent_dim"],
        synthetic_supervised_weight=resolved["synthetic_supervised_weight"],
        synthetic_reconstruction_weight=resolved[
            "synthetic_reconstruction_weight"
        ],
        downstream_model=resolved["downstream_model"],
        n_measured_ae_fit=len(context.ae_fit.y),
        n_synthetic_train=n_synthetic,
        ae_best_epoch=int(artifacts["best_epoch"]),
        ae_best_internal_validation_loss=float(
            artifacts["best_validation_loss"]
        ),
        inner_split_hash=context.inner_split.split_hash,
    )
    audit = {
        "policy_hash": policy.policy_hash,
        "inner_split_hash": context.inner_split.split_hash,
        "ae_fit_source_id_hash": context.inner_split.ae_train_source_id_hash,
        "internal_validation_source_id_hash": (
            context.inner_split.internal_validation_source_id_hash
        ),
        "policy_validation_source_id_hash": stable_hash(
            sorted(context.policy_validation.source_row_ids)
        ),
        "n_measured_ae_fit": len(context.ae_fit.y),
        "n_internal_validation": len(context.internal_validation.y),
        "n_policy_validation": len(context.policy_validation.y),
        "n_synthetic_train": n_synthetic,
        "ae_best_epoch": int(artifacts["best_epoch"]),
        "ae_best_internal_validation_loss": float(
            artifacts["best_validation_loss"]
        ),
        **boundary_audit,
    }
    return JointAEFitResult(metrics=metrics, audit=audit)


def _validate_search_context_boundaries(
    context: AEPolicySearchContext,
) -> dict[str, int]:
    """Derive and enforce every measured-row overlap relevant to model fitting."""
    ae_fit_ids = set(context.ae_fit.source_row_ids)
    internal_ids = set(context.internal_validation.source_row_ids)
    policy_validation_ids = set(context.policy_validation.source_row_ids)
    outer_valid_ids = set(context.outer_validation_source_ids)
    outer_test_ids = set(context.outer_test_source_ids)
    expected_ae_fit_ids = set(context.inner_split.ae_train_source_ids)
    expected_internal_ids = set(
        context.inner_split.internal_validation_source_ids
    )
    overlaps = {
        "ae_fit_internal_validation_overlap_count": len(
            ae_fit_ids & internal_ids
        ),
        "ae_fit_policy_validation_overlap_count": len(
            ae_fit_ids & policy_validation_ids
        ),
        "ae_fit_outer_validation_overlap_count": len(
            ae_fit_ids & outer_valid_ids
        ),
        "ae_fit_outer_test_overlap_count": len(ae_fit_ids & outer_test_ids),
        "internal_policy_validation_overlap_count": len(
            internal_ids & policy_validation_ids
        ),
        "internal_outer_validation_overlap_count": len(
            internal_ids & outer_valid_ids
        ),
        "internal_outer_test_overlap_count": len(
            internal_ids & outer_test_ids
        ),
        "policy_validation_outer_test_overlap_count": len(
            policy_validation_ids & outer_test_ids
        ),
        "downstream_internal_validation_overlap_count": len(
            ae_fit_ids & internal_ids
        ),
        "downstream_policy_validation_overlap_count": len(
            ae_fit_ids & policy_validation_ids
        ),
        "downstream_outer_test_overlap_count": len(
            ae_fit_ids & outer_test_ids
        ),
    }
    membership_mismatch = (
        ae_fit_ids != expected_ae_fit_ids
        or internal_ids != expected_internal_ids
        or policy_validation_ids != outer_valid_ids
        or context.forbidden_source_ids != frozenset(
            outer_valid_ids | outer_test_ids
        )
    )
    if membership_mismatch or any(overlaps.values()):
        raise ValueError(
            "Joint AE search partition contamination detected: "
            f"membership_mismatch={membership_mismatch}, overlaps={overlaps}."
        )
    return overlaps


def _transfer_key_from_policy(policy: ResolvedPolicy) -> str:
    return stable_hash(
        {
            "transfer_method": policy.config["transfer_method"],
            "transfer_config": policy.config.get("transfer_config"),
        }
    )


def _frozen_training_protocol(
    inner_split: AEInnerSplit,
    selected_policy: ResolvedPolicy,
) -> dict[str, Any]:
    settings = dict(selected_policy.config["ae_settings"])
    return {
        "schema_version": AE_REFIT_PROTOCOL_SCHEMA_VERSION,
        "development_rows": "saved_training_subset_split_by_canonical_group",
        "ae_fit_rows": "inner_ae_fit_only",
        "early_stopping_rows": "untouched_real_inner_validation_only",
        "policy_selection_rows": "saved_validation_only",
        "synthetic_rows": "ae_fit_sources_and_donors_only",
        "refit_rows": "all_saved_training_subset_rows",
        "refit_validation_rows": "none",
        "refit_epoch_contract": "predefined_resolved_epoch_count",
        "refit_epochs": int(settings["refit_epochs"]),
        "inner_split_hash": inner_split.split_hash,
        "outer_test_prediction_batches": 1,
    }


def _evaluation_unit(seed: int, fraction: float) -> str:
    return (
        f"seed={seed}|train_fraction={fraction:.12g}|"
        "comparison=joint_condition_transfer_supervised_ae"
    )


def _positive_int_list(value: Any, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"ae_policy_search.{name} must contain positive integers.")
    return sorted(set(value))


def _nonnegative_float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"ae_policy_search.{name} must be non-empty.")
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not np.isfinite(float(item))
            or float(item) < 0
        ):
            raise ValueError(
                f"ae_policy_search.{name} must contain finite nonnegative values."
            )
        result.append(float(item))
    return sorted(set(result))


def _validate_ae_settings(settings: dict[str, Any]) -> None:
    _positive_int_list(settings["hidden_dims"], "ae_settings.hidden_dims")
    for name in (
        "dropout",
        "reconstruction_weight",
        "yield_weight",
        "latent_l2_weight",
        "learning_rate",
        "weight_decay",
        "internal_valid_size",
    ):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"Joint AE setting {name} must be nonnegative.")
    if not 0 < float(settings["internal_valid_size"]) < 1:
        raise ValueError("Joint AE internal_valid_size must be in (0, 1).")
    for name in (
        "batch_size",
        "max_epochs",
        "patience",
        "minimum_inner_rows",
        "minimum_inner_groups",
        "refit_epochs",
    ):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Joint AE setting {name} must be a positive integer.")
    if int(settings["refit_epochs"]) > int(settings["max_epochs"]):
        raise ValueError("Joint AE refit_epochs cannot exceed development max_epochs.")
    if not isinstance(settings["device"], str) or not settings["device"].strip():
        raise ValueError("Joint AE device must be a non-empty string.")
