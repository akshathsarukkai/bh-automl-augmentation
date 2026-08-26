"""Matched corrected execution path for anonymous and role-aware transfer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    CandidateScopePolicy,
    resolved_candidate_scope_mode,
)
from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    ROLE_TRANSFER_MODES,
    RoleAwareConditionTransferConfig,
    clear_role_aware_teacher_cache,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import measured_canonical_keys
from bh_augmentation.data.bh_condition_reader import parse_bh_reaction_smiles
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.low_data_partitions import scope_from_split_frames
from bh_augmentation.features.compatibility import assert_feature_compatibility
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_supervised_ae_latent_baseline import (
    _compute_metric,
    _parse_model_config,
    _resolve_model_configs,
    _resolve_seeds,
    _split_positions,
    _validate_metrics,
)
from bh_augmentation.utils.corrected_runs import (
    CORRECTED_ROLE_FEATURE_KIND,
    CORRECTED_STATUS,
    assert_training_only_parents,
    build_run_manifest,
    canonical_split_audit_record,
    feature_contract_record,
    load_corrected_bh_dataframe,
    prepare_fresh_output_directory,
    resolve_corrected_feature_config,
    sha256_file,
    write_json,
)
from bh_augmentation.utils.seed import set_global_seed
from bh_augmentation.utils.synthetic_audits import (
    build_candidate_audit_frame,
    combine_candidate_audit_frames,
)

Policy = ConditionTransferConfig | RoleAwareConditionTransferConfig
PolicyFactory = Callable[[dict[str, Any], int, pd.DataFrame], list[Policy]]


def run_corrected_condition_transfer(
    config_path: str | Path,
    config: dict[str, Any],
    *,
    transfer_kind: str,
    policy_factory: PolicyFactory,
) -> dict[str, Path]:
    """Run one corrected transfer family with validation-only policy selection."""
    if transfer_kind not in {"anonymous", "role_aware"}:
        raise ValueError(f"Unknown corrected transfer kind: {transfer_kind}")
    metrics_requested = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metrics_requested)
    if "rmse" not in metrics_requested:
        raise ValueError("Corrected policy selection requires validation RMSE.")
    feature_config = resolve_corrected_feature_config(
        config.get("features", {}), required_kind=CORRECTED_ROLE_FEATURE_KIND
    )
    dataset_path = Path(_dataset_path(config))
    seeds = _resolve_seeds(config)
    train_fractions = _corrected_train_fractions(config)
    split_directory = _corrected_split_directory(config)
    saved_splits = load_saved_canonical_splits(
        dataset_path,
        split_directory,
        requested_seeds=seeds,
        requested_fractions=train_fractions,
    )
    dataset_hash = sha256_file(dataset_path)
    frame = load_corrected_bh_dataframe(dataset_path)
    # Candidate eligibility is a declared scientific choice. This is a low-data
    # augmentation-benefit runner, so its default is observed_only_low_data: a
    # candidate is ineligible only when the simulated learner has actually
    # observed that chemistry. Rejecting against the complete historical dataset
    # instead answers a prospective-novelty question, which is available by
    # declaring candidate_scope.mode: globally_unmeasured_prospective.
    candidate_scope_mode = resolved_candidate_scope_mode(config)
    complete_measured_identity_keys = measured_canonical_keys(frame)
    output_dir = prepare_fresh_output_directory(config["output"]["directory"])
    paths = _output_paths(output_dir, transfer_kind)

    X, y, feature_names, feature_metadata = build_feature_matrix_with_metadata(
        frame, feature_config
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    feature_contract = feature_contract_record(feature_metadata, feature_names)
    feature_metadata_hash = str(feature_contract["feature_metadata_hash"])
    role_counts = _role_value_counts(frame)

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidate_audit_frames: list[pd.DataFrame] = []
    compatibility_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_hashes: dict[str, str] = {}
    selection_exclusions: list[dict[str, Any]] = []

    for seed in seeds:
        set_global_seed(seed)
        for train_fraction, splits in saved_splits.materialize_variants(
            frame,
            seed=seed,
            train_fractions=train_fractions,
        ):
            fraction = float(train_fraction)
            split_record = canonical_split_audit_record(
                seed,
                fraction,
                splits,
                dataset_hash=dataset_hash,
                saved_audit=saved_splits.audit_record(
                    seed=seed,
                    train_fraction=fraction,
                ),
            )
            split_rows.append(split_record)
            split_hashes[_split_key(seed, fraction)] = str(split_record["split_hash"])
            unit_candidate_scope = scope_from_split_frames(
                candidate_scope_mode,
                splits=splits,
                global_identity_keys=complete_measured_identity_keys,
            )
            train_idx = _split_positions(splits["train"])
            valid_idx = _split_positions(splits["valid"])
            test_idx = _split_positions(splits["test"])
            X_train, y_train = X[train_idx], y[train_idx]
            train_frame = frame.loc[train_idx].copy()
            if transfer_kind == "role_aware":
                clear_role_aware_teacher_cache()

            model_configs = _resolve_model_configs(config.get("models", ["xgboost"]))
            for model_config in model_configs:
                model_name, model_kwargs = _parse_model_config(model_config)
                model = train_model(
                    get_model(model_name, seed=seed, **model_kwargs), X_train, y_train
                )
                baseline = _base_record(
                    seed=seed,
                    fraction=fraction,
                    model_name=model_name,
                    representation="bh_role_separated_real_only",
                    n_real=len(train_idx),
                    n_synthetic=0,
                    split_record=split_record,
                    feature_width=X.shape[1],
                    feature_metadata_hash=feature_metadata_hash,
                    dataset_hash=dataset_hash,
                )
                metric_rows.extend(
                    _evaluate_model(
                        model,
                        X,
                        y,
                        {"valid": valid_idx, "test": test_idx},
                        metrics_requested,
                        baseline,
                    )
                )

            eligible: dict[str, list[dict[str, Any]]] = {
                _parse_model_config(item)[0]: [] for item in model_configs
            }
            policies = policy_factory(config, seed, role_counts)
            for policy_index, policy in enumerate(policies):
                policy_id = _policy_id(transfer_kind, seed, fraction, policy_index, policy)
                result = _generate(
                    transfer_kind,
                    train_frame,
                    X_train,
                    y_train,
                    policy,
                    feature_config,
                    feature_names,
                    feature_metadata,
                    unit_candidate_scope,
                )
                candidate_audit_frames.append(
                    build_candidate_audit_frame(
                        result["candidate_df"],
                        transfer_kind=transfer_kind,
                        seed=seed,
                        train_fraction=fraction,
                        policy_id=policy_id,
                    )
                )
                assert_training_only_parents(result["candidate_df"], splits)
                assert_feature_compatibility(
                    X_train,
                    feature_names,
                    result["X_synthetic"],
                    result["feature_names"],
                    real_metadata=feature_metadata,
                    synthetic_metadata=result["feature_metadata"],
                )
                policy_audit = _policy_audit(
                    transfer_kind,
                    policy,
                    policy_id,
                    result,
                    seed,
                    fraction,
                    split_record,
                    feature_metadata_hash,
                    dataset_hash,
                )
                audit_rows.append(policy_audit)
                compatibility_rows.append(
                    {
                        "seed": seed,
                        "train_fraction": fraction,
                        "policy_id": policy_id,
                        "compatible": True,
                        "real_feature_metadata_hash": feature_metadata_hash,
                        "synthetic_feature_metadata_hash": feature_contract_record(
                            result["feature_metadata"], result["feature_names"]
                        )["feature_metadata_hash"],
                        "feature_name_hash": feature_contract["feature_name_hash"],
                        "feature_width": X.shape[1],
                        "split_hash": split_record["split_hash"],
                        "dataset_hash": dataset_hash,
                    }
                )
                n_synthetic = int(len(result["synthetic_y"]))
                if n_synthetic <= 0:
                    continue
                if not np.isfinite(result["X_synthetic"]).all() or not np.isfinite(
                    result["synthetic_y"]
                ).all():
                    continue

                X_synthetic = np.asarray(result["X_synthetic"], dtype=np.float32)
                y_synthetic = np.asarray(result["synthetic_y"], dtype=np.float32)
                X_augmented = np.vstack([X_train, X_synthetic]).astype(np.float32)
                y_augmented = np.concatenate([y_train, y_synthetic]).astype(np.float32)
                for model_config in model_configs:
                    model_name, model_kwargs = _parse_model_config(model_config)
                    model = train_model(
                        get_model(model_name, seed=seed, **model_kwargs),
                        X_augmented,
                        y_augmented,
                    )
                    predictions = predict_model(model, X[valid_idx])
                    if not np.isfinite(predictions).all():
                        continue
                    record = {
                        **_base_record(
                            seed=seed,
                            fraction=fraction,
                            model_name=model_name,
                            representation=f"corrected_{transfer_kind}_condition_transfer",
                            n_real=len(train_idx),
                            n_synthetic=n_synthetic,
                            split_record=split_record,
                            feature_width=X.shape[1],
                            feature_metadata_hash=feature_metadata_hash,
                            dataset_hash=dataset_hash,
                        ),
                        "policy_id": policy_id,
                        **_policy_fields(policy_audit),
                    }
                    rows = _prediction_rows(
                        y[valid_idx], predictions, metrics_requested, record, "valid"
                    )
                    metric_rows.extend(rows)
                    valid_rmse = next(
                        row["value"] for row in rows if row["metric"] == "rmse"
                    )
                    if np.isfinite(valid_rmse):
                        eligible[model_name].append(
                            {
                                "model": model,
                                "record": record,
                                "valid_rmse": float(valid_rmse),
                            }
                        )

            for model_name, candidates in eligible.items():
                if not candidates:
                    selection_exclusions.append(
                        {
                            "seed": seed,
                            "train_fraction": fraction,
                            "model": model_name,
                            "transfer_kind": transfer_kind,
                            "selection_status": "no_eligible_nonzero_synthetic_policy",
                            "reason": (
                                "All generated policies were empty, rejected, or non-finite; "
                                "no augmentation test metric was evaluated."
                            ),
                            "split_hash": split_record["split_hash"],
                            "source_id_split_hash": split_record["source_id_split_hash"],
                            "dataset_hash": dataset_hash,
                        }
                    )
                    continue
                selected = min(candidates, key=lambda item: item["valid_rmse"])
                selected_record = dict(selected["record"])
                selected_record["selected_policy"] = True
                selected_rows.append(
                    {
                        **selected_record,
                        "valid_rmse": selected["valid_rmse"],
                    }
                )
                for row in metric_rows:
                    if (
                        row.get("seed") == seed
                        and row.get("train_fraction") == fraction
                        and row.get("model") == model_name
                        and row.get("policy_id") == selected_record["policy_id"]
                        and row.get("split") == "valid"
                    ):
                        row["selected_policy"] = True
                test_predictions = predict_model(selected["model"], X[test_idx])
                if not np.isfinite(test_predictions).all():
                    raise ValueError(
                        f"Selected {transfer_kind} policy produced non-finite test predictions."
                    )
                metric_rows.extend(
                    _prediction_rows(
                        y[test_idx],
                        test_predictions,
                        metrics_requested,
                        selected_record,
                        "test",
                    )
                )

    policy_metrics = pd.DataFrame(metric_rows)
    selected_policies = pd.DataFrame(selected_rows)
    if selected_policies.empty:
        selected_policies = pd.DataFrame(
            columns=[
                *[
                    column
                    for column in policy_metrics.columns
                    if column not in {"split", "metric", "value"}
                ],
                "valid_rmse",
            ]
        )
    selection_exclusions_frame = pd.DataFrame(
        selection_exclusions,
        columns=[
            "seed",
            "train_fraction",
            "model",
            "transfer_kind",
            "selection_status",
            "reason",
            "split_hash",
            "source_id_split_hash",
            "dataset_hash",
        ],
    )
    selected_policy_metrics = policy_metrics.loc[
        policy_metrics["selected_policy"].fillna(False).astype(bool)
    ].copy()
    if (
        not selected_policies.empty
        and (selected_policies["n_synthetic_train"].astype(int) <= 0).any()
    ):
        raise ValueError("A zero-synthetic corrected policy was selected.")
    if selected_policy_metrics["value"].isna().any():
        raise ValueError("Selected corrected policy metrics contain NaN values.")
    summary = _summary(policy_metrics, selected_policy_metrics)

    policy_metrics.to_csv(paths["policy_metrics"], index=False)
    selected_policies.to_csv(paths["selected_policies"], index=False)
    selected_policy_metrics.to_csv(paths["selected_policy_metrics"], index=False)
    pd.DataFrame(audit_rows).to_csv(paths["audit"], index=False)
    pd.DataFrame(compatibility_rows).to_csv(paths["feature_compatibility_audit"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(split_rows).to_csv(paths["split_audit"], index=False)
    selection_exclusions_frame.to_csv(paths["selection_exclusions"], index=False)
    combine_candidate_audit_frames(candidate_audit_frames).to_csv(
        paths["candidate_audit"], index=False
    )
    if transfer_kind == "role_aware":
        role_counts.to_csv(paths["role_value_counts"], index=False)
    manifest = build_run_manifest(
        config=config,
        config_path=config_path,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        output_directory=output_dir,
        split_hashes=split_hashes,
        feature_metadata_hash=feature_metadata_hash,
        canonical_split_contract={
            **saved_splits.audit_metadata,
            "split_directory": str(split_directory),
        },
    )
    manifest["transfer_kind"] = transfer_kind
    manifest["selection_exclusion_count"] = len(selection_exclusions)
    manifest["execution_protocol"] = "development_combined_search_and_test_pre_phase5"
    manifest["confirmatory_selection_eligible"] = False
    write_json(paths["run_manifest"], manifest)
    return paths


def select_corrected_policy_rows(validation_rows: pd.DataFrame) -> pd.DataFrame:
    """Select finite, nonzero-synthetic policies using validation RMSE only."""
    required = {"seed", "train_fraction", "model", "split", "metric", "value", "n_synthetic_train"}
    missing = sorted(required - set(validation_rows.columns))
    if missing:
        raise ValueError(f"Policy selection rows are missing columns: {missing}")
    eligible = validation_rows.loc[
        validation_rows["split"].eq("valid")
        & validation_rows["metric"].eq("rmse")
        & (validation_rows["n_synthetic_train"].fillna(0).astype(float) > 0)
        & np.isfinite(validation_rows["value"].to_numpy(dtype=float))
    ].copy()
    return (
        eligible.sort_values("value", kind="mergesort")
        .groupby(["seed", "train_fraction", "model"], as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def _generate(
    transfer_kind: str,
    train_frame: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    policy: Policy,
    feature_config: dict[str, Any],
    feature_names: list[str],
    feature_metadata: Any,
    candidate_scope: CandidateScopePolicy,
) -> dict[str, Any]:
    kwargs = {
        "feature_config": feature_config,
        "real_feature_names": feature_names,
        "real_feature_metadata": feature_metadata,
        "candidate_scope": candidate_scope,
    }
    if transfer_kind == "anonymous":
        if not isinstance(policy, ConditionTransferConfig):
            raise TypeError("Anonymous corrected run received a role-aware policy.")
        return generate_condition_transfer_examples(
            train_frame, X_train, y_train, policy, **kwargs
        )
    if not isinstance(policy, RoleAwareConditionTransferConfig):
        raise TypeError("Role-aware corrected run received an anonymous policy.")
    return generate_role_aware_condition_transfer_examples(
        train_frame, X_train, y_train, policy, **kwargs
    )


def _policy_audit(
    transfer_kind: str,
    policy: Policy,
    policy_id: str,
    result: dict[str, Any],
    seed: int,
    fraction: float,
    split_record: Mapping[str, Any],
    feature_metadata_hash: str,
    dataset_hash: str,
) -> dict[str, Any]:
    metadata = dict(result["metadata"])
    requested_roles = (
        ["catalyst", "ligand", "base", "solvent_or_additive"]
        if transfer_kind == "anonymous"
        else ROLE_TRANSFER_MODES[policy.role_transfer_mode]
    )
    changed_columns = {
        "catalyst": "changed_catalyst",
        "ligand": "changed_ligand",
        "base": "changed_base",
        "solvent_or_additive": "changed_solvent_or_additive",
    }
    synthetic = result["synthetic_df"].copy()
    if transfer_kind == "anonymous" and not synthetic.empty:
        synthetic = _annotate_anonymous_changes(synthetic, requested_roles, policy_id)
    actual_changed = [
        role
        for role, column in changed_columns.items()
        if column in synthetic and synthetic[column].fillna(False).astype(bool).any()
    ]
    effective_mode = str(metadata.get("effective_role_transfer_mode", ""))
    effective_roles = (
        requested_roles
        if transfer_kind == "anonymous"
        else ROLE_TRANSFER_MODES.get(effective_mode, requested_roles)
    )
    accepted = result["candidate_df"].loc[
        result["candidate_df"].get("accepted", False).eq(True)
    ].copy()
    if transfer_kind == "anonymous" and not accepted.empty:
        accepted = _annotate_anonymous_changes(accepted, requested_roles, policy_id)
    if not accepted.empty:
        changed_any = np.zeros(len(accepted), dtype=bool)
        for role in effective_roles:
            column = changed_columns[role]
            if column in accepted:
                changed_any |= accepted[column].fillna(False).to_numpy(dtype=bool)
        if not changed_any.all():
            raise ValueError(
                f"Accepted policy {policy_id} contains a candidate with no actual transferred-role change."
            )
        changed_fraction = float(changed_any.mean())
    else:
        changed_fraction = 0.0
    return {
        **metadata,
        "seed": seed,
        "train_fraction": fraction,
        "policy_id": policy_id,
        "requested_roles": "|".join(requested_roles),
        "actual_changed_roles": "|".join(actual_changed),
        "changed_any_transferred_role_fraction": changed_fraction,
        "feature_metadata_hash": feature_metadata_hash,
        "split_hash": split_record["split_hash"],
        "dataset_hash": dataset_hash,
        "used_validation_or_test_parents": False,
        "result_status": CORRECTED_STATUS,
    }


def _annotate_anonymous_changes(
    frame: pd.DataFrame,
    requested_roles: list[str],
    policy_id: str,
) -> pd.DataFrame:
    result = frame.copy()
    source_fields = {
        "catalyst": "parsed_catalyst_smiles",
        "ligand": "parsed_ligand_smiles",
        "base": "parsed_base_smiles",
        "solvent_or_additive": "parsed_solvent_or_additive_smiles",
    }
    synthetic_fields = {
        "catalyst": "recovered_catalyst_smiles",
        "ligand": "recovered_ligand_smiles",
        "base": "recovered_base_smiles",
        "solvent_or_additive": "recovered_solvent_or_additive_smiles",
    }
    parsed_sources = [
        parse_bh_reaction_smiles(value)
        for value in result["source_reaction_smiles"].astype(str)
    ]
    if any(item is None for item in parsed_sources):
        raise ValueError(f"Could not parse an accepted anonymous source for {policy_id}.")
    for role in requested_roles:
        result[f"changed_{role}"] = [
            str(row[synthetic_fields[role]]) != str(parsed[source_fields[role]])
            for (_, row), parsed in zip(result.iterrows(), parsed_sources, strict=True)
        ]
    return result


def _policy_fields(audit: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "donor_strategy",
        "label_strategy",
        "synthetic_multiplier",
        "n_neighbors",
        "min_similarity",
        "max_teacher_std",
        "role_transfer_mode",
        "effective_role_transfer_mode",
        "requested_roles",
        "actual_changed_roles",
        "n_candidates_generated",
        "n_candidates_accepted",
        "filter_acceptance_rate",
        "mean_donor_similarity",
        "mean_teacher_std",
        "changed_any_transferred_role_fraction",
    ]
    return {field: audit.get(field, np.nan) for field in fields}


def _base_record(
    *,
    seed: int,
    fraction: float,
    model_name: str,
    representation: str,
    n_real: int,
    n_synthetic: int,
    split_record: Mapping[str, Any],
    feature_width: int,
    feature_metadata_hash: str,
    dataset_hash: str,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "train_fraction": fraction,
        "representation": representation,
        "model": model_name,
        "policy_id": "",
        "n_real_train": n_real,
        "n_synthetic_train": n_synthetic,
        "n_train": n_real,
        "n_valid": split_record["n_valid"],
        "n_test": split_record["n_test"],
        "feature_width": feature_width,
        "feature_metadata_hash": feature_metadata_hash,
        "split_hash": split_record["split_hash"],
        "dataset_hash": dataset_hash,
        "selected_policy": False,
        "result_status": CORRECTED_STATUS,
    }


def _evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    splits: Mapping[str, np.ndarray],
    metric_names: list[str],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, indices in splits.items():
        predictions = predict_model(model, X[indices])
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite baseline predictions on {split_name}.")
        rows.extend(_prediction_rows(y[indices], predictions, metric_names, record, split_name))
    return rows


def _prediction_rows(
    y_true: np.ndarray,
    predictions: np.ndarray,
    metric_names: list[str],
    record: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "split": split,
            "metric": metric,
            "value": _compute_metric(metric, y_true, predictions),
        }
        for metric in metric_names
    ]


def _summary(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = policy_metrics.loc[
        policy_metrics["representation"].eq("bh_role_separated_real_only")
    ]
    report = pd.concat([baseline, selected_policy_metrics], ignore_index=True)
    return (
        report.groupby(
            ["train_fraction", "representation", "model", "split", "metric"],
            dropna=False,
        )["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def _role_value_counts(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "catalyst": "recovered_catalyst_smiles",
        "ligand": "recovered_ligand_smiles",
        "base": "recovered_base_smiles",
        "solvent_or_additive": "recovered_solvent_or_additive_smiles",
    }
    return pd.DataFrame(
        [
            {
                "role": role,
                "column": column,
                "n_unique": int(frame[column].nunique(dropna=False)),
                "invariant": int(frame[column].nunique(dropna=False)) <= 1,
            }
            for role, column in columns.items()
        ]
    )


def _policy_id(
    kind: str,
    seed: int,
    fraction: float,
    index: int,
    policy: Policy,
) -> str:
    return f"{kind}|seed={seed}|fraction={fraction}|policy={index}|{asdict(policy)}"


def _output_paths(directory: Path, kind: str) -> dict[str, Path]:
    audit_name = "synthetic_audit.csv" if kind == "anonymous" else "role_transfer_audit.csv"
    candidate_audit_name = (
        "synthetic_candidate_audit.csv"
        if kind == "anonymous"
        else "role_transfer_candidate_audit.csv"
    )
    paths = {
        "directory": directory,
        "policy_metrics": directory / "policy_metrics.csv",
        "selected_policies": directory / "selected_policies.csv",
        "selected_policy_metrics": directory / "selected_policy_metrics.csv",
        "audit": directory / audit_name,
        "candidate_audit": directory / candidate_audit_name,
        "feature_compatibility_audit": directory / "feature_compatibility_audit.csv",
        "summary": directory / "summary.csv",
        "split_audit": directory / "split_audit.csv",
        "selection_exclusions": directory / "selection_exclusions.csv",
        "role_value_counts": directory / "role_value_counts.csv",
        "run_manifest": directory / "run_manifest.json",
    }
    paths.update(
        {
            "policy_metrics_path": paths["policy_metrics"],
            "selected_policies_path": paths["selected_policies"],
            "selected_policy_metrics_path": paths["selected_policy_metrics"],
            "summary_path": paths["summary"],
            "synthetic_audit_path": paths["audit"],
            "role_transfer_audit_path": paths["audit"],
            "candidate_audit_path": paths["candidate_audit"],
            "synthetic_candidate_audit_path": paths["candidate_audit"],
            "role_transfer_candidate_audit_path": paths["candidate_audit"],
            "role_value_counts_path": paths["role_value_counts"],
        }
    )
    return paths


def _dataset_path(config: Mapping[str, Any]) -> str:
    path = config.get("dataset", {}).get("path")
    if not path:
        raise ValueError("Config must define dataset.path.")
    return str(path)


def _corrected_split_directory(config: Mapping[str, Any]) -> Path:
    split_config = config.get("splits", {})
    if split_config.get("method") != "canonical_saved":
        raise ValueError("Corrected scientific runs require splits.method='canonical_saved'.")
    directory = split_config.get("directory")
    if not directory:
        raise ValueError("Corrected scientific runs require splits.directory.")
    return Path(directory)


def _corrected_train_fractions(config: Mapping[str, Any]) -> list[float]:
    low_data = config.get("low_data", {})
    if not low_data.get("enabled", False):
        return [1.0]
    fractions = low_data.get("train_fractions")
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("Corrected low_data.train_fractions must be a non-empty list.")
    return [float(value) for value in fractions]


def _split_key(seed: int, fraction: float) -> str:
    return f"seed={seed}|train_fraction={fraction:.12g}"
