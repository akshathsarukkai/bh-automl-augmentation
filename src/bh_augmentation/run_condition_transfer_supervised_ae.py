"""Run the matched anonymous condition-transfer plus supervised-AE experiment."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import measured_canonical_keys
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.reaction_roles import ensure_reaction_role_columns
from bh_augmentation.features.compatibility import FeatureMetadata, assert_feature_compatibility
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    normalize_feature_config,
)
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.representations.supervised_autoencoder import (
    SupervisedAEConfig,
    encode_with_supervised_autoencoder,
    fit_supervised_autoencoder,
)
from bh_augmentation.run_condition_transfer import _iter_condition_transfer_policies
from bh_augmentation.run_supervised_ae_latent_baseline import (
    _compute_metric,
    _create_random_splits,
    _create_split_variants,
    _get_dataset_path,
    _parse_model_config,
    _resolve_feature_config,
    _resolve_model_configs,
    _resolve_seeds,
    _split_positions,
    _validate_metrics,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed
from bh_augmentation.utils.synthetic_audits import (
    build_candidate_audit_frame,
    combine_candidate_audit_frames,
)


def run_condition_transfer_supervised_ae(config_path: str | Path) -> dict[str, Path]:
    """Run validation-selected anonymous transfer before weighted supervised AE fitting."""
    config = load_config(config_path)
    seeds = _resolve_seeds(config)
    metric_names = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metric_names)
    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = ensure_reaction_role_columns(clean_buchwald_hartwig(raw_df), parse_if_missing=True)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run the hybrid experiment.")
    complete_measured_identity_keys = measured_canonical_keys(df)

    configured_features = {"kind": "bh_role_separated", **dict(config.get("features", {}))}
    feature_config = normalize_feature_config(_resolve_feature_config(configured_features))
    if feature_config["kind"] not in {"bh_role_separated", "bh_role_separated_delta"}:
        raise ValueError(
            "Corrected condition-transfer + supervised-AE runs require "
            "bh_role_separated features; chemical block semantics are not inferred "
            "from neural-network input width."
        )
    X, y, feature_names, feature_metadata = build_feature_matrix_with_metadata(
        df,
        feature_config,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    paths = _resolve_output_paths(config)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    print(
        f"Loaded {len(df)} rows with {X.shape[1]} reaction-role features; "
        f"seeds={seeds}, latent_dims={_latent_dims(config)}.",
        flush=True,
    )

    metrics: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    ae_audit: list[dict[str, object]] = []
    synthetic_audit: list[dict[str, object]] = []
    candidate_audit_frames: list[pd.DataFrame] = []

    for seed in seeds:
        set_global_seed(seed)
        print(f"Starting seed={seed}.", flush=True)
        if bool(config.get("evaluate_full_data_reference", True)):
            outer = _create_random_splits(df, dict(config.get("splits", {})), seed)
            _append_full_data_reference(
                metrics,
                X,
                y,
                outer,
                config,
                seed,
                metric_names,
            )

        for train_fraction, splits in _create_split_variants(df, config, seed):
            train_indices = _split_positions(splits["train"])
            valid_indices = _split_positions(splits["valid"])
            test_indices = _split_positions(splits["test"])
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            train_df = df.loc[train_indices].copy()
            print(
                f"seed={seed} fraction={float(train_fraction):.2f} "
                f"n_real_train={len(train_indices)}: selecting anonymous transfer.",
                flush=True,
            )

            metrics.extend(
                _evaluate_original_baselines(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                    y_test,
                    config,
                    seed,
                    float(train_fraction),
                    metric_names,
                )
            )

            condition_candidates = _evaluate_condition_transfer_candidates(
                train_df=train_df,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                config=config,
                seed=seed,
                train_fraction=float(train_fraction),
                metric_names=metric_names,
                valid_indices=valid_indices,
                test_indices=test_indices,
                feature_config=feature_config,
                feature_names=feature_names,
                feature_metadata=feature_metadata,
                complete_measured_identity_keys=complete_measured_identity_keys,
            )
            for candidate in condition_candidates:
                metrics.extend(candidate["metric_records"])
                synthetic_audit.append(candidate["audit"])
                candidate_audit_frames.append(candidate["candidate_audit"])
            selected_condition = _select_condition_transfer_candidate(condition_candidates)
            if selected_condition is not None:
                selected_condition["audit"]["selected_condition_transfer_policy"] = True
                metrics.extend(
                    _evaluate_fitted_model(
                        selected_condition["model"],
                        X_test,
                        y_test,
                        metric_names,
                        selected_condition["record_base"],
                        split="test",
                    )
                )
                print(
                    f"seed={seed} fraction={float(train_fraction):.2f}: selected "
                    f"{selected_condition['record_base']['condition_transfer_policy_id']} "
                    f"with n_synthetic={len(selected_condition['generation_result']['synthetic_y'])}.",
                    flush=True,
                )

            ae_only_candidates: list[dict[str, Any]] = []
            hybrid_candidates: list[dict[str, Any]] = []
            for latent_dim in _latent_dims(config):
                real_candidate = _fit_ae_candidate(
                    X_real=X_train,
                    y_real=y_train,
                    X_synthetic=None,
                    y_synthetic=None,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    latent_dim=latent_dim,
                    synthetic_example_weight=0.0,
                    condition_candidate=None,
                    config=config,
                    seed=seed,
                    train_fraction=float(train_fraction),
                    metric_names=metric_names,
                    representation=f"supervised_ae_{latent_dim}_real_only",
                    feature_names=feature_names,
                    feature_metadata=feature_metadata,
                )
                ae_audit.append(real_candidate["audit"])
                metrics.extend(real_candidate["metric_records"])
                ae_only_candidates.extend(real_candidate["model_candidates"])

                if selected_condition is None:
                    continue
                result = selected_condition["generation_result"]
                if len(result["synthetic_y"]) == 0:
                    continue
                for synthetic_weight in _synthetic_example_weights(config):
                    hybrid_candidate = _fit_ae_candidate(
                        X_real=X_train,
                        y_real=y_train,
                        X_synthetic=np.asarray(result["X_synthetic"], dtype=np.float32),
                        y_synthetic=np.asarray(result["synthetic_y"], dtype=np.float32),
                        X_valid=X_valid,
                        y_valid=y_valid,
                        latent_dim=latent_dim,
                        synthetic_example_weight=synthetic_weight,
                        condition_candidate=selected_condition,
                        config=config,
                        seed=seed,
                        train_fraction=float(train_fraction),
                        metric_names=metric_names,
                        representation=f"supervised_ae_{latent_dim}_condition_transfer",
                        feature_names=feature_names,
                        feature_metadata=feature_metadata,
                    )
                    ae_audit.append(hybrid_candidate["audit"])
                    metrics.extend(hybrid_candidate["metric_records"])
                    hybrid_candidates.extend(hybrid_candidate["model_candidates"])

            selected_ae_only = _select_model_candidates(ae_only_candidates)
            for candidate in selected_ae_only:
                candidate["record_base"]["ae_only_selected"] = True
                metrics.extend(
                    _evaluate_fitted_model(
                        candidate["model"],
                        candidate["X_test_latent_factory"](X_test),
                        y_test,
                        metric_names,
                        candidate["record_base"],
                        split="test",
                    )
                )

            selected_hybrid = _select_model_candidates(hybrid_candidates)
            for candidate in selected_hybrid:
                candidate["record_base"]["selected_policy"] = True
                selected_rows.append(
                    {
                        **candidate["record_base"],
                        "valid_rmse": candidate["valid_rmse"],
                    }
                )
                metrics.extend(
                    _evaluate_fitted_model(
                        candidate["model"],
                        candidate["X_test_latent_factory"](X_test),
                        y_test,
                        metric_names,
                        candidate["record_base"],
                        split="test",
                    )
                )
            print(
                f"seed={seed} fraction={float(train_fraction):.2f}: completed "
                f"{len(ae_only_candidates)} AE-only and {len(hybrid_candidates)} hybrid model candidates.",
                flush=True,
            )

    policy_metrics = pd.DataFrame(metrics)
    selected_policies = pd.DataFrame(selected_rows)
    policy_metrics = _mark_selected_hybrid_metrics(policy_metrics, selected_policies)
    selected_policy_metrics = policy_metrics.loc[policy_metrics["selected_policy"]].copy()
    summary = _summarize(policy_metrics)
    ae_audit_df = pd.DataFrame(ae_audit)
    synthetic_audit_df = pd.DataFrame(synthetic_audit)
    candidate_audit_df = combine_candidate_audit_frames(candidate_audit_frames)

    comparisons = _build_all_comparisons(policy_metrics, selected_policy_metrics)
    _write_outputs(
        policy_metrics,
        selected_policies,
        selected_policy_metrics,
        ae_audit_df,
        synthetic_audit_df,
        candidate_audit_df,
        summary,
        comparisons,
        paths,
    )
    _print_completion_summary(policy_metrics, selected_policies, comparisons)
    return paths


def combine_real_and_synthetic_training_data(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synthetic: np.ndarray | None,
    y_synthetic: np.ndarray | None,
    real_example_weight: float,
    synthetic_example_weight: float,
    synthetic_reconstruction_weight: float = 1.0,
    *,
    real_feature_names: list[str] | None = None,
    synthetic_feature_names: list[str] | None = None,
    real_feature_metadata: FeatureMetadata | None = None,
    synthetic_feature_metadata: FeatureMetadata | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine training-only rows and return labels, weights, and origin markers."""
    X_real_array = np.asarray(X_real, dtype=np.float32)
    y_real_array = np.asarray(y_real, dtype=np.float32).reshape(-1)
    if len(X_real_array) != len(y_real_array):
        raise ValueError("Real feature and label row counts must match.")
    if X_synthetic is None or y_synthetic is None or len(X_synthetic) == 0:
        n_real = len(X_real_array)
        return (
            X_real_array.copy(),
            y_real_array.copy(),
            np.full(n_real, float(real_example_weight), dtype=np.float32),
            np.ones(n_real, dtype=np.float32),
            np.zeros(n_real, dtype=bool),
        )
    X_synth_array = np.asarray(X_synthetic, dtype=np.float32)
    y_synth_array = np.asarray(y_synthetic, dtype=np.float32).reshape(-1)
    if X_synth_array.shape[1] != X_real_array.shape[1] or len(X_synth_array) != len(y_synth_array):
        raise ValueError("Synthetic features and labels must align with the real feature matrix.")
    assert_feature_compatibility(
        X_real_array,
        real_feature_names or [],
        X_synth_array,
        synthetic_feature_names or [],
        real_metadata=real_feature_metadata,
        synthetic_metadata=synthetic_feature_metadata,
    )
    return (
        np.vstack([X_real_array, X_synth_array]).astype(np.float32),
        np.concatenate([y_real_array, y_synth_array]).astype(np.float32),
        np.concatenate(
            [
                np.full(len(X_real_array), float(real_example_weight)),
                np.full(len(X_synth_array), float(synthetic_example_weight)),
            ]
        ).astype(np.float32),
        np.concatenate(
            [
                np.ones(len(X_real_array)),
                np.full(len(X_synth_array), float(synthetic_reconstruction_weight)),
            ]
        ).astype(np.float32),
        np.concatenate(
            [np.zeros(len(X_real_array), dtype=bool), np.ones(len(X_synth_array), dtype=bool)]
        ),
    )


def select_hybrid_policies(validation_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select finite, non-empty hybrid policies using validation RMSE only."""
    candidates = validation_metrics.loc[
        validation_metrics["representation"].astype(str).str.endswith("_condition_transfer")
        & (validation_metrics["split"] == "valid")
        & (validation_metrics["metric"] == "rmse")
        & (validation_metrics["n_synthetic_train"] > 0)
        & (validation_metrics["ae_training_status"] == "ok")
        & np.isfinite(pd.to_numeric(validation_metrics["value"], errors="coerce"))
        & np.isfinite(pd.to_numeric(validation_metrics["ae_reconstruction_loss"], errors="coerce"))
        & np.isfinite(pd.to_numeric(validation_metrics["ae_supervised_loss"], errors="coerce"))
    ].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values("value", kind="mergesort")
        .groupby(["seed", "train_fraction", "downstream_model"], dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def best_parent_rmse_by_seed(
    anonymous: pd.DataFrame,
    ae_only: pd.DataFrame,
) -> pd.DataFrame:
    """Choose the lower parent RMSE independently for every seed and fraction."""
    keys = ["seed", "train_fraction"]
    left = anonymous[keys + ["value"]].rename(columns={"value": "anonymous_rmse"})
    right = ae_only[keys + ["value"]].rename(columns={"value": "ae_only_rmse"})
    merged = left.merge(right, on=keys, how="inner")
    merged["best_parent"] = np.where(
        merged["anonymous_rmse"] <= merged["ae_only_rmse"],
        "anonymous_condition_transfer",
        "supervised_ae_only",
    )
    merged["best_parent_rmse"] = merged[["anonymous_rmse", "ae_only_rmse"]].min(axis=1)
    return merged


def _evaluate_condition_transfer_candidates(
    *,
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: dict[str, Any],
    seed: int,
    train_fraction: float,
    metric_names: list[str],
    valid_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_config: dict[str, Any],
    feature_names: list[str],
    feature_metadata: FeatureMetadata,
    complete_measured_identity_keys: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    train_index_set = set(train_df.index.tolist())
    for policy_index, policy in enumerate(_anonymous_policies(config, seed)):
        result = generate_condition_transfer_examples(
            train_df,
            X_train,
            y_train,
            policy,
            feature_config=feature_config,
            real_feature_names=feature_names,
            real_feature_metadata=feature_metadata,
            measured_identity_keys=complete_measured_identity_keys,
        )
        metadata = dict(result["metadata"])
        policy_id = _condition_policy_id(policy_index, policy)
        synthetic_df = result["synthetic_df"]
        parent_indices = set(synthetic_df.get("source_index", pd.Series(dtype=object))) | set(
            synthetic_df.get("donor_index", pd.Series(dtype=object))
        )
        n_synthetic = len(result["synthetic_y"])
        audit = {
            "seed": seed,
            "train_fraction": train_fraction,
            "condition_transfer_policy_id": policy_id,
            **asdict(policy),
            **metadata,
            "source_and_donor_indices_train_only": parent_indices <= train_index_set,
            "used_validation_or_test_parents": bool(
                parent_indices & (set(valid_indices.tolist()) | set(test_indices.tolist()))
            ),
            "selected_condition_transfer_policy": False,
        }
        candidate_audit = build_candidate_audit_frame(
            result["candidate_df"],
            transfer_kind="anonymous_supervised_ae",
            seed=seed,
            train_fraction=train_fraction,
            policy_id=policy_id,
        )
        if n_synthetic == 0:
            audit["selection_eligible"] = False
            candidates.append(
                {
                    "audit": audit,
                    "candidate_audit": candidate_audit,
                    "metric_records": [],
                    "generation_result": result,
                }
            )
            continue
        X_aug, y_aug, _, _, _ = combine_real_and_synthetic_training_data(
            X_train,
            y_train,
            result["X_synthetic"],
            result["synthetic_y"],
            1.0,
            1.0,
            real_feature_names=feature_names,
            synthetic_feature_names=result["feature_names"],
            real_feature_metadata=feature_metadata,
            synthetic_feature_metadata=result["feature_metadata"],
        )
        model_name, model_kwargs = _condition_selection_model(config)
        model = train_model(get_model(model_name, seed=seed, **model_kwargs), X_aug, y_aug)
        predictions = predict_model(model, X_valid)
        base = _record_base(
            seed=seed,
            train_fraction=train_fraction,
            n_real_train=len(X_train),
            n_synthetic_train=n_synthetic,
            representation="anonymous_condition_transfer_bh_role_separated",
            downstream_model=model_name,
            condition_transfer_policy_id=policy_id,
            policy=policy,
        )
        records = _prediction_records(y_valid, predictions, metric_names, base, "valid")
        rmse = next(record["value"] for record in records if record["metric"] == "rmse")
        audit["selection_eligible"] = bool(np.isfinite(rmse))
        candidates.append(
            {
                "audit": audit,
                "candidate_audit": candidate_audit,
                "metric_records": records,
                "generation_result": result,
                "model": model,
                "record_base": base,
                "valid_rmse": rmse,
            }
        )
    return candidates


def _select_condition_transfer_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("audit", {}).get("selection_eligible", False)
        and candidate.get("audit", {}).get("n_synthetic_train", 0) > 0
        and np.isfinite(candidate.get("valid_rmse", np.nan))
    ]
    return min(eligible, key=lambda item: float(item["valid_rmse"])) if eligible else None


def _fit_ae_candidate(
    *,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synthetic: np.ndarray | None,
    y_synthetic: np.ndarray | None,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    latent_dim: int,
    synthetic_example_weight: float,
    condition_candidate: dict[str, Any] | None,
    config: dict[str, Any],
    seed: int,
    train_fraction: float,
    metric_names: list[str],
    representation: str,
    feature_names: list[str],
    feature_metadata: FeatureMetadata,
) -> dict[str, Any]:
    ae_config = config.get("supervised_autoencoder", {})
    real_weight = float(ae_config.get("real_example_weight", 1.0))
    reconstruction_weight = float(ae_config.get("synthetic_reconstruction_weight", 1.0))
    X_combined, y_combined, yield_weights, reconstruction_weights, synthetic_mask = (
        combine_real_and_synthetic_training_data(
            X_real,
            y_real,
            X_synthetic,
            y_synthetic,
            real_weight,
            synthetic_example_weight,
            reconstruction_weight,
            real_feature_names=feature_names,
            synthetic_feature_names=(
                condition_candidate["generation_result"]["feature_names"]
                if condition_candidate is not None
                else None
            ),
            real_feature_metadata=feature_metadata,
            synthetic_feature_metadata=(
                condition_candidate["generation_result"]["feature_metadata"]
                if condition_candidate is not None
                else None
            ),
        )
    )
    policy_object = None
    condition_policy_id = ""
    if condition_candidate is not None:
        condition_policy_id = str(condition_candidate["record_base"]["condition_transfer_policy_id"])
        policy_object = condition_candidate["record_base"]
    hybrid_policy_id = (
        f"seed={seed}|fraction={train_fraction}|condition={condition_policy_id or 'none'}|"
        f"latent={latent_dim}|synthetic_weight={synthetic_example_weight}"
    )
    audit_base = {
        "seed": seed,
        "train_fraction": train_fraction,
        "hybrid_policy_id": hybrid_policy_id,
        "condition_transfer_policy_id": condition_policy_id,
        "representation": representation,
        "latent_dim": latent_dim,
        "n_real_train": len(X_real),
        "n_synthetic_train": int(synthetic_mask.sum()),
        "real_example_weight": real_weight,
        "synthetic_example_weight": synthetic_example_weight,
        "synthetic_reconstruction_weight": reconstruction_weight,
        "used_outer_valid_for_ae_training": False,
        "used_outer_test_for_ae_training": False,
        "synthetic_label_sum": float(np.asarray(y_synthetic).sum()) if y_synthetic is not None else 0.0,
    }
    try:
        split = _make_weighted_internal_ae_split(
            X_combined,
            y_combined,
            yield_weights,
            reconstruction_weights,
            synthetic_mask,
            valid_size=float(ae_config.get("internal_valid_size", 0.2)),
            seed=seed + 10_000 + latent_dim,
        )
        training_config = SupervisedAEConfig(
            input_dim=X_combined.shape[1],
            latent_dim=latent_dim,
            hidden_dims=[int(dim) for dim in ae_config.get("hidden_dims", [1024, 256])],
            dropout=float(ae_config.get("dropout", 0.1)),
            reconstruction_weight=float(ae_config.get("reconstruction_weight", 1.0)),
            yield_weight=float(ae_config.get("yield_weight", 1.0)),
            latent_l2_weight=float(ae_config.get("latent_l2_weight", 0.0)),
            learning_rate=float(ae_config.get("learning_rate", 0.001)),
            weight_decay=float(ae_config.get("weight_decay", 0.0)),
            batch_size=int(ae_config.get("batch_size", 32)),
            max_epochs=int(ae_config.get("max_epochs", 200)),
            patience=int(ae_config.get("patience", 20)),
            random_state=seed,
            device=str(ae_config.get("device", "auto")),
        )
        artifacts = fit_supervised_autoencoder(
            split["X_train"],
            split["y_train"],
            split["X_valid"],
            split["y_valid"],
            training_config,
            sample_weight=split["yield_weight"],
            reconstruction_sample_weight=split["reconstruction_weight"],
        )
        loss_row = _best_history_row(artifacts)
        losses_finite = np.isfinite(
            [loss_row["train_reconstruction_loss"], loss_row["train_yield_loss"]]
        ).all()
        if not losses_finite:
            raise ValueError("AE training produced non-finite losses.")
        audit = {
            **audit_base,
            "ae_training_status": "ok",
            "ae_training_error": "",
            "ae_best_epoch": int(artifacts["best_epoch"]),
            "ae_best_internal_valid_loss": float(artifacts["best_validation_loss"]),
            "ae_reconstruction_loss": float(loss_row["train_reconstruction_loss"]),
            "ae_supervised_loss": float(loss_row["train_yield_loss"]),
            "n_internal_ae_train": len(split["X_train"]),
            "n_internal_ae_valid": 0 if split["X_valid"] is None else len(split["X_valid"]),
            "n_synthetic_in_internal_valid": 0,
        }
        Z_combined = encode_with_supervised_autoencoder(artifacts, X_combined)
        Z_valid = encode_with_supervised_autoencoder(artifacts, X_valid)
        record_base = {
            **_record_base(
                seed=seed,
                train_fraction=train_fraction,
                n_real_train=len(X_real),
                n_synthetic_train=int(synthetic_mask.sum()),
                representation=representation,
                downstream_model="",
                condition_transfer_policy_id=condition_policy_id,
                policy=policy_object,
            ),
            "hybrid_policy_id": hybrid_policy_id,
            "synthetic_example_weight": synthetic_example_weight,
            "latent_dim": latent_dim,
            "ae_reconstruction_loss": audit["ae_reconstruction_loss"],
            "ae_supervised_loss": audit["ae_supervised_loss"],
            "ae_best_epoch": audit["ae_best_epoch"],
            "ae_best_internal_valid_loss": audit["ae_best_internal_valid_loss"],
            "ae_training_status": "ok",
        }
        records: list[dict[str, object]] = []
        model_candidates: list[dict[str, Any]] = []
        for model_config in _resolve_model_configs(config.get("models", ["ridge", "xgboost"])):
            model_name, model_kwargs = _parse_model_config(model_config)
            model = train_model(
                get_model(model_name, seed=seed, **model_kwargs),
                Z_combined,
                y_combined,
                sample_weight=yield_weights,
            )
            predictions = predict_model(model, Z_valid)
            model_base = {**record_base, "downstream_model": model_name, "model": model_name}
            model_records = _prediction_records(y_valid, predictions, metric_names, model_base, "valid")
            records.extend(model_records)
            valid_rmse = next(item["value"] for item in model_records if item["metric"] == "rmse")
            if np.isfinite(predictions).all() and np.isfinite(valid_rmse):
                model_candidates.append(
                    {
                        "model": model,
                        "valid_rmse": valid_rmse,
                        "record_base": model_base,
                        "X_test_latent_factory": lambda X_test, a=artifacts: encode_with_supervised_autoencoder(a, X_test),
                    }
                )
        return {"audit": audit, "metric_records": records, "model_candidates": model_candidates}
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {
            "audit": {
                **audit_base,
                "ae_training_status": "failed",
                "ae_training_error": str(exc),
                "ae_best_epoch": 0,
                "ae_best_internal_valid_loss": np.nan,
                "ae_reconstruction_loss": np.nan,
                "ae_supervised_loss": np.nan,
            },
            "metric_records": [],
            "model_candidates": [],
        }


def _make_weighted_internal_ae_split(
    X: np.ndarray,
    y: np.ndarray,
    yield_weight: np.ndarray,
    reconstruction_weight: np.ndarray,
    synthetic_mask: np.ndarray,
    valid_size: float,
    seed: int,
) -> dict[str, np.ndarray | None]:
    real_indices = np.flatnonzero(~synthetic_mask)
    synthetic_indices = np.flatnonzero(synthetic_mask)
    if len(real_indices) < 4 or valid_size <= 0:
        train_indices = np.arange(len(X))
        valid_indices = np.empty(0, dtype=int)
    else:
        rng = np.random.default_rng(seed)
        shuffled = real_indices.copy()
        rng.shuffle(shuffled)
        n_valid = min(max(1, int(math.floor(len(real_indices) * valid_size))), len(real_indices) - 1)
        valid_indices = shuffled[:n_valid]
        train_indices = np.concatenate([shuffled[n_valid:], synthetic_indices])
        rng.shuffle(train_indices)
    return {
        "X_train": X[train_indices],
        "y_train": y[train_indices],
        "yield_weight": yield_weight[train_indices],
        "reconstruction_weight": reconstruction_weight[train_indices],
        "X_valid": X[valid_indices] if len(valid_indices) else None,
        "y_valid": y[valid_indices] if len(valid_indices) else None,
    }


def _select_model_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for model_name in sorted({item["record_base"]["downstream_model"] for item in candidates}):
        eligible = [
            item
            for item in candidates
            if item["record_base"]["downstream_model"] == model_name
            and item["record_base"]["n_synthetic_train"] >= 0
            and np.isfinite(item["valid_rmse"])
        ]
        if eligible:
            selected.append(min(eligible, key=lambda item: float(item["valid_rmse"])))
    return selected


def _anonymous_policies(config: dict[str, Any], seed: int) -> list[ConditionTransferConfig]:
    policies = _iter_condition_transfer_policies(config, seed)
    allowed_pairs = {
        (str(item["donor_strategy"]), str(item["label_strategy"]))
        for item in config.get("condition_transfer", {}).get("policy_pairs", [])
    }
    if allowed_pairs:
        policies = [
            policy for policy in policies if (policy.donor_strategy, policy.label_strategy) in allowed_pairs
        ]
    max_policies = config.get("condition_transfer", {}).get("max_policies")
    if max_policies is not None:
        policies = policies[: max(0, int(max_policies))]
    return policies


def _condition_selection_model(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    selection_model = config.get("condition_transfer", {}).get("selection_model")
    if selection_model is None:
        selection_model = next(
            (
                item
                for item in _resolve_model_configs(config.get("models", ["xgboost"]))
                if _parse_model_config(item)[0] == "xgboost"
            ),
            "xgboost",
        )
    return _parse_model_config(selection_model)


def _evaluate_original_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    seed: int,
    train_fraction: float,
    metric_names: list[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model_config in _resolve_model_configs(config.get("models", ["ridge", "xgboost"])):
        model_name, kwargs = _parse_model_config(model_config)
        model = train_model(get_model(model_name, seed=seed, **kwargs), X_train, y_train)
        base = _record_base(
            seed=seed,
            train_fraction=train_fraction,
            n_real_train=len(X_train),
            n_synthetic_train=0,
            representation="bh_role_separated_real_only",
            downstream_model=model_name,
        )
        records.extend(_evaluate_fitted_model(model, X_valid, y_valid, metric_names, base, "valid"))
        records.extend(_evaluate_fitted_model(model, X_test, y_test, metric_names, base, "test"))
    return records


def _append_full_data_reference(
    records: list[dict[str, object]],
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, pd.DataFrame],
    config: dict[str, Any],
    seed: int,
    metric_names: list[str],
) -> None:
    model_name, kwargs = _condition_selection_model(config)
    train_idx = _split_positions(splits["train"])
    model = train_model(get_model(model_name, seed=seed, **kwargs), X[train_idx], y[train_idx])
    base = _record_base(
        seed=seed,
        train_fraction=1.0,
        n_real_train=len(train_idx),
        n_synthetic_train=0,
        representation="full_data_bh_role_separated",
        downstream_model=model_name,
    )
    for split_name in ["valid", "test"]:
        idx = _split_positions(splits[split_name])
        records.extend(_evaluate_fitted_model(model, X[idx], y[idx], metric_names, base, split_name))


def _evaluate_fitted_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_names: list[str],
    base: dict[str, object],
    split: str,
) -> list[dict[str, object]]:
    return _prediction_records(y, predict_model(model, X), metric_names, base, split)


def _prediction_records(
    y: np.ndarray,
    predictions: np.ndarray,
    metric_names: list[str],
    base: dict[str, object],
    split: str,
) -> list[dict[str, object]]:
    if not np.isfinite(predictions).all():
        return []
    return [
        {**base, "split": split, "metric": metric, "value": _compute_metric(metric, y, predictions)}
        for metric in metric_names
    ]


def _record_base(
    *,
    seed: int,
    train_fraction: float,
    n_real_train: int,
    n_synthetic_train: int,
    representation: str,
    downstream_model: str,
    condition_transfer_policy_id: str = "",
    policy: ConditionTransferConfig | dict[str, Any] | None = None,
) -> dict[str, object]:
    if isinstance(policy, ConditionTransferConfig):
        policy_values: dict[str, Any] = asdict(policy)
    elif isinstance(policy, dict):
        policy_values = policy
    else:
        policy_values = {}
    return {
        "seed": seed,
        "train_fraction": train_fraction,
        "n_real_train": n_real_train,
        "n_synthetic_train": n_synthetic_train,
        "representation": representation,
        "condition_transfer_policy_id": condition_transfer_policy_id,
        "donor_strategy": policy_values.get("donor_strategy", ""),
        "label_strategy": policy_values.get("label_strategy", ""),
        "synthetic_multiplier": policy_values.get("synthetic_multiplier", np.nan),
        "synthetic_example_weight": np.nan,
        "latent_dim": np.nan,
        "use_latent_interpolation": False,
        "downstream_model": downstream_model,
        "model": downstream_model,
        "hybrid_policy_id": "",
        "ae_reconstruction_loss": np.nan,
        "ae_supervised_loss": np.nan,
        "ae_best_epoch": np.nan,
        "ae_best_internal_valid_loss": np.nan,
        "ae_training_status": "not_applicable",
        "ae_only_selected": False,
        "condition_transfer_selected": representation.startswith(
            "anonymous_condition_transfer_"
        ),
        "selected_policy": False,
    }


def _best_history_row(artifacts: dict[str, Any]) -> pd.Series:
    history = artifacts["history"]
    best_epoch = int(artifacts["best_epoch"])
    matched = history.loc[history["epoch"] == best_epoch]
    return matched.iloc[0] if not matched.empty else history.iloc[-1]


def _latent_dims(config: dict[str, Any]) -> list[int]:
    return [int(value) for value in config.get("supervised_autoencoder", {}).get("latent_dims", [16, 64])]


def _synthetic_example_weights(config: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for value in config.get("supervised_autoencoder", {}).get(
            "synthetic_example_weights", [0.25, 0.5, 1.0]
        )
    ]


def _condition_policy_id(index: int, policy: ConditionTransferConfig) -> str:
    return (
        f"anonymous_{index}|donor={policy.donor_strategy}|label={policy.label_strategy}|"
        f"mult={policy.synthetic_multiplier}|sim={policy.min_similarity}|std={policy.max_teacher_std}"
    )


def _mark_selected_hybrid_metrics(
    metrics: pd.DataFrame,
    selected_policies: pd.DataFrame,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    metrics = metrics.copy()
    metrics["selected_policy"] = metrics.get("selected_policy", False).fillna(False).astype(bool)
    if selected_policies.empty:
        return metrics
    keys = selected_policies[["seed", "train_fraction", "downstream_model", "hybrid_policy_id"]].drop_duplicates()
    keys["_selected"] = True
    marked = metrics.merge(keys, on=["seed", "train_fraction", "downstream_model", "hybrid_policy_id"], how="left")
    marked["selected_policy"] = marked["selected_policy"] | marked["_selected"].eq(True)
    return marked.drop(columns="_selected")


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "train_fraction",
        "representation",
        "downstream_model",
        "latent_dim",
        "synthetic_example_weight",
        "split",
        "metric",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=[*columns, "mean", "std", "count"])
    return metrics.groupby(columns, dropna=False)["value"].agg(mean="mean", std="std", count="count").reset_index()


def _build_all_comparisons(
    metrics: pd.DataFrame,
    selected_hybrid: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    hybrid = selected_hybrid.loc[
        (selected_hybrid["split"] == "test") & (selected_hybrid["downstream_model"] == "xgboost")
    ].copy()
    parents = {
        "real_only": metrics.loc[
            (metrics["representation"] == "bh_role_separated_real_only")
            & (metrics["downstream_model"] == "xgboost")
            & (metrics["split"] == "test")
        ],
        "anonymous_transfer": metrics.loc[
            metrics["representation"].astype(str).str.startswith(
                "anonymous_condition_transfer_"
            )
            & metrics["condition_transfer_selected"].astype(bool)
            & (metrics["split"] == "test")
        ],
        "ae_only": metrics.loc[
            metrics["representation"].astype(str).str.endswith("_real_only")
            & metrics["ae_only_selected"].astype(bool)
            & (metrics["downstream_model"] == "xgboost")
            & (metrics["split"] == "test")
        ],
    }
    output = {name: _comparison_frame(hybrid, parent, name) for name, parent in parents.items()}
    output["best_parent"] = _comparison_vs_best_parent(hybrid, parents["anonymous_transfer"], parents["ae_only"])
    return output


def _comparison_frame(
    hybrid: pd.DataFrame,
    parent: pd.DataFrame,
    parent_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["seed", "train_fraction", "metric"]
    hybrid_values = hybrid[keys + ["value", "hybrid_policy_id", "latent_dim", "synthetic_example_weight"]].rename(
        columns={"value": "hybrid_value"}
    )
    parent_values = parent[keys + ["value"]].drop_duplicates(keys).rename(columns={"value": "parent_value"})
    by_seed = hybrid_values.merge(parent_values, on=keys, how="inner")
    by_seed["parent"] = parent_name
    by_seed["delta"] = by_seed["hybrid_value"] - by_seed["parent_value"]
    by_seed["hybrid_better"] = np.where(
        by_seed["metric"].isin(["rmse", "mae"]),
        by_seed["delta"] < 0,
        by_seed["delta"] > 0,
    )
    return by_seed, _comparison_summary(by_seed)


def _comparison_vs_best_parent(
    hybrid: pd.DataFrame,
    anonymous: pd.DataFrame,
    ae_only: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rmse_anon = anonymous.loc[anonymous["metric"] == "rmse", ["seed", "train_fraction", "value"]]
    rmse_ae = ae_only.loc[ae_only["metric"] == "rmse", ["seed", "train_fraction", "value"]]
    choices = best_parent_rmse_by_seed(rmse_anon, rmse_ae)
    parent_all = pd.concat(
        [
            anonymous.assign(_parent_choice="anonymous_condition_transfer"),
            ae_only.assign(_parent_choice="supervised_ae_only"),
        ],
        ignore_index=True,
    ).merge(choices[["seed", "train_fraction", "best_parent"]], on=["seed", "train_fraction"])
    parent_all = parent_all.loc[parent_all["_parent_choice"] == parent_all["best_parent"]]
    by_seed, _ = _comparison_frame(hybrid, parent_all, "best_parent_per_seed")
    by_seed = by_seed.merge(
        choices[["seed", "train_fraction", "best_parent", "best_parent_rmse"]],
        on=["seed", "train_fraction"],
        how="left",
    )
    return by_seed, _comparison_summary(by_seed)


def _comparison_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    if by_seed.empty:
        return pd.DataFrame(
            columns=["train_fraction", "metric", "mean_delta", "std_delta", "count", "n_hybrid_better", "paired_t_pvalue", "wilcoxon_pvalue"]
        )
    rows: list[dict[str, object]] = []
    for (fraction, metric), group in by_seed.groupby(["train_fraction", "metric"], dropna=False):
        hybrid = group["hybrid_value"].to_numpy(dtype=float)
        parent = group["parent_value"].to_numpy(dtype=float)
        rows.append(
            {
                "train_fraction": fraction,
                "metric": metric,
                "mean_delta": float(group["delta"].mean()),
                "std_delta": float(group["delta"].std()),
                "count": len(group),
                "n_hybrid_better": int(group["hybrid_better"].sum()),
                "paired_t_pvalue": _paired_t_pvalue(hybrid, parent),
                "wilcoxon_pvalue": _wilcoxon_pvalue(hybrid, parent),
            }
        )
    return pd.DataFrame(rows)


def _paired_t_pvalue(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    return float(ttest_rel(left, right, nan_policy="omit").pvalue)


def _wilcoxon_pvalue(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.allclose(left, right):
        return float("nan")
    try:
        return float(wilcoxon(left, right).pvalue)
    except ValueError:
        return float("nan")


def _write_outputs(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    ae_audit: pd.DataFrame,
    synthetic_audit: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    paths: dict[str, Path],
) -> None:
    frames = {
        "policy_metrics_path": metrics,
        "selected_hybrid_policies_path": selected,
        "selected_hybrid_policy_metrics_path": selected_metrics,
        "ae_training_audit_path": ae_audit,
        "synthetic_training_audit_path": synthetic_audit,
        "synthetic_candidate_audit_path": candidate_audit,
        "summary_path": summary,
    }
    for name, (by_seed, comparison_summary) in comparisons.items():
        frames[f"hybrid_vs_{name}_by_seed_path"] = by_seed
        frames[f"hybrid_vs_{name}_summary_path"] = comparison_summary
    for key, frame in frames.items():
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[key], index=False)


def _resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output = config.get("output", {})
    directory = Path(output.get("directory", "results/condition_transfer_supervised_ae_xgboost"))
    names = {
        "policy_metrics_path": "policy_metrics.csv",
        "selected_hybrid_policies_path": "selected_hybrid_policies.csv",
        "selected_hybrid_policy_metrics_path": "selected_hybrid_policy_metrics.csv",
        "ae_training_audit_path": "ae_training_audit.csv",
        "synthetic_training_audit_path": "synthetic_training_audit.csv",
        "synthetic_candidate_audit_path": "synthetic_candidate_audit.csv",
        "summary_path": "summary.csv",
    }
    for parent in ["real_only", "anonymous_transfer", "ae_only", "best_parent"]:
        names[f"hybrid_vs_{parent}_by_seed_path"] = f"hybrid_vs_{parent}_by_seed.csv"
        names[f"hybrid_vs_{parent}_summary_path"] = f"hybrid_vs_{parent}_summary.csv"
    return {
        "directory": directory,
        **{key: Path(output.get(key, directory / filename)) for key, filename in names.items()},
    }


def _print_completion_summary(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    comparisons: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    test = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["metric"] == "rmse")
        & (metrics["downstream_model"] == "xgboost")
    ]
    print("\nMATCHED HYBRID TEST RMSE")
    for representation in [
        "bh_role_separated_real_only",
        "anonymous_condition_transfer_bh_role_separated",
        "supervised_ae_16_real_only",
        "supervised_ae_64_real_only",
        "supervised_ae_16_condition_transfer",
        "supervised_ae_64_condition_transfer",
        "full_data_bh_role_separated",
    ]:
        rows = test.loc[test["representation"] == representation]
        if not rows.empty:
            print(f"{representation}: {rows.groupby('train_fraction')['value'].mean().to_dict()}")
    if not selected.empty:
        columns = [
            "seed",
            "train_fraction",
            "downstream_model",
            "latent_dim",
            "synthetic_example_weight",
            "condition_transfer_policy_id",
            "valid_rmse",
        ]
        print("\nSELECTED HYBRID POLICIES")
        print(selected[columns].to_string(index=False))
    best_summary = comparisons["best_parent"][1]
    if not best_summary.empty:
        print("\nHYBRID DELTA VS BETTER PARENT (negative RMSE is better)")
        print(best_summary.loc[best_summary["metric"] == "rmse"].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run condition-transfer plus supervised-AE experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    args = parser.parse_args()
    paths = run_condition_transfer_supervised_ae(args.config)
    print(f"Saved hybrid metrics to {paths['policy_metrics_path']}")


if __name__ == "__main__":
    main()
