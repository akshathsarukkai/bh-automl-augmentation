"""Run chemically constrained condition-transfer augmentation experiments."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import measured_canonical_keys
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.reaction_roles import ensure_reaction_role_columns
from bh_augmentation.features.compatibility import assert_feature_compatibility
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
    normalize_feature_config,
)
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_supervised_ae_latent_baseline import (
    _compute_metric,
    _create_split_variants,
    _parse_model_config,
    _resolve_feature_config,
    _resolve_model_configs,
    _resolve_seeds,
    _validate_metrics,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed
from bh_augmentation.utils.synthetic_audits import (
    build_candidate_audit_frame,
    combine_candidate_audit_frames,
)


def run_condition_transfer(config_path: str | Path) -> dict[str, Path]:
    """Run real-only and condition-transfer augmentation policies."""
    config = load_config(config_path)
    if config.get("corrected_revalidation", {}).get("enabled", False):
        from bh_augmentation.corrected_condition_transfer import (
            run_corrected_condition_transfer,
        )

        return run_corrected_condition_transfer(
            config_path,
            config,
            transfer_kind="anonymous",
            policy_factory=lambda current, seed, _role_counts: (
                _iter_condition_transfer_policies(current, seed)
            ),
        )
    seeds = _resolve_seeds(config)
    metric_names = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metric_names)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = ensure_reaction_role_columns(clean_buchwald_hartwig(raw_df), parse_if_missing=True)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run condition transfer.")
    complete_measured_identity_keys = measured_canonical_keys(df)

    configured_features = {"kind": "bh_role_separated", **dict(config.get("features", {}))}
    feature_config = normalize_feature_config(_resolve_feature_config(configured_features))
    supported_kinds = {
        "bh_role_separated",
        "bh_role_separated_delta",
        "reaction_section_concat",
        "reaction_section_concat_delta",
    }
    if feature_config["kind"] not in supported_kinds:
        raise ValueError(
            "Anonymous condition transfer requires bh_role_separated features for "
            "scientific runs, or an explicitly configured reaction_section_concat "
            "legacy/development representation."
        )
    X, y, feature_names, feature_metadata = build_feature_matrix_with_metadata(
        df,
        feature_config,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    output_paths = _resolve_output_paths(config)
    output_paths["directory"].mkdir(parents=True, exist_ok=True)

    metric_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    candidate_audit_frames: list[pd.DataFrame] = []
    for seed in seeds:
        set_global_seed(seed)
        for train_fraction, splits in _create_split_variants(df, config, seed):
            train_indices = _split_positions(splits["train"])
            valid_indices = _split_positions(splits["valid"])
            test_indices = _split_positions(splits["test"])
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            train_df = df.loc[train_indices].copy()
            n_real_train = int(len(train_indices))

            metric_records.extend(
                _evaluate_models(
                    X_train,
                    y_train,
                    {"valid": (X_valid, y_valid), "test": (X_test, y_test)},
                    config,
                    seed=seed,
                    train_fraction=float(train_fraction),
                    n_real_train=n_real_train,
                    representation=f"{feature_config['kind']}_real_only",
                    policy_metadata=_empty_policy_metadata(),
                    metric_names=metric_names,
                    policy_id=f"seed={seed}|frac={train_fraction}|real_only",
                )
            )

            for policy_index, policy in enumerate(_iter_condition_transfer_policies(config, seed)):
                policy_id = _policy_id(seed, float(train_fraction), policy_index, policy)
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
                metadata.update(
                    {
                        "seed": seed,
                        "train_fraction": float(train_fraction),
                        "policy_id": policy_id,
                        "used_validation_or_test_parents": False,
                    }
                )
                audit_records.append(metadata)
                candidate_audit_frames.append(
                    build_candidate_audit_frame(
                        result["candidate_df"],
                        transfer_kind="anonymous",
                        seed=seed,
                        train_fraction=float(train_fraction),
                        policy_id=policy_id,
                    )
                )

                synthetic_df = result["synthetic_df"]
                synthetic_y = result["synthetic_y"]
                if len(synthetic_df) == 0:
                    X_augmented = X_train
                    y_augmented = y_train
                else:
                    assert_feature_compatibility(
                        X_train,
                        feature_names,
                        result["X_synthetic"],
                        result["feature_names"],
                        real_metadata=feature_metadata,
                        synthetic_metadata=result["feature_metadata"],
                    )
                    X_synthetic = np.asarray(result["X_synthetic"], dtype=np.float32)
                    X_augmented = np.vstack([X_train, X_synthetic.astype(np.float32)])
                    y_augmented = np.concatenate([y_train, synthetic_y]).astype(float)

                metric_records.extend(
                    _evaluate_models(
                        X_augmented,
                        y_augmented,
                        {"valid": (X_valid, y_valid), "test": (X_test, y_test)},
                        config,
                        seed=seed,
                        train_fraction=float(train_fraction),
                        n_real_train=n_real_train,
                        representation=f"anonymous_condition_transfer_{feature_config['kind']}",
                        policy_metadata=metadata,
                        metric_names=metric_names,
                        policy_id=policy_id,
                    )
                )

    policy_metrics = pd.DataFrame(metric_records)
    policy_metrics["selected_policy"] = False
    selected_policies = _select_policies(policy_metrics, config)
    policy_metrics = _mark_selected_policy_metrics(policy_metrics, selected_policies)
    selected_policy_metrics = policy_metrics.loc[policy_metrics["selected_policy"]].copy()
    audit = pd.DataFrame(audit_records)
    summary = _summarize_for_reporting(policy_metrics, selected_policy_metrics)
    rf_delta, rf_delta_summary = _save_vs_original_rf(
        policy_metrics,
        selected_policy_metrics,
        output_paths,
    )
    same_model_delta, same_model_delta_summary = _save_vs_real_only_same_model(
        policy_metrics,
        selected_policy_metrics,
        output_paths,
    )

    candidate_audit = combine_candidate_audit_frames(candidate_audit_frames)
    _write_outputs(
        policy_metrics,
        selected_policies,
        selected_policy_metrics,
        summary,
        audit,
        candidate_audit,
        output_paths,
    )
    _print_test_summary(summary)
    _print_selected_policy_table(selected_policies, selected_policy_metrics)
    _print_delta_summary("CONDITION TRANSFER DELTAS VS ORIGINAL 6144D RF", rf_delta_summary)
    _print_delta_summary(
        "CONDITION TRANSFER DELTAS VS REAL-ONLY SAME MODEL",
        same_model_delta_summary,
    )
    return output_paths


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_condition_transfer`."""
    parser = argparse.ArgumentParser(description="Run condition-transfer augmentation.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    paths = run_condition_transfer(args.config)
    print(f"Saved condition-transfer metrics to {paths['policy_metrics_path']}")


def _evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    evaluation_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seed: int,
    train_fraction: float,
    n_real_train: int,
    representation: str,
    policy_metadata: dict[str, Any],
    metric_names: list[str],
    policy_id: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model_config in _resolve_model_configs(config.get("models", ["ridge", "random_forest"])):
        model_name, model_kwargs = _parse_model_config(model_config)
        try:
            model = get_model(model_name, seed=seed, **model_kwargs)
        except ImportError:
            continue
        fitted_model = train_model(model, X_train, y_train)
        for split_name, (X_split, y_split) in evaluation_splits.items():
            predictions = predict_model(fitted_model, X_split)
            for metric_name in metric_names:
                records.append(
                    {
                        "seed": seed,
                        "train_fraction": train_fraction,
                        "n_real_train": n_real_train,
                        "representation": representation,
                        "model": model_name,
                        "policy_id": policy_id,
                        **_policy_metric_fields(policy_metadata),
                        "split": split_name,
                        "metric": metric_name,
                        "value": _compute_metric(metric_name, y_split, predictions),
                    }
                )
    return records


def _iter_condition_transfer_policies(
    config: dict[str, Any],
    seed: int,
) -> list[ConditionTransferConfig]:
    transfer = config.get("condition_transfer", {})
    if not transfer.get("enabled", True):
        return []
    policies: list[ConditionTransferConfig] = []
    donor_strategies = _as_list(transfer.get("donor_strategies", ["random"]))
    label_strategies = _as_list(transfer.get("label_strategies", ["teacher_ensemble"]))
    multipliers = _as_list(transfer.get("synthetic_multipliers", [1.0]))
    n_neighbors_values = _as_list(transfer.get("n_neighbors", [5]))
    min_similarities = _as_list(transfer.get("min_similarities", [None]))
    max_teacher_stds = _as_list(transfer.get("max_teacher_stds", [None]))
    for donor_strategy, label_strategy, multiplier, n_neighbors in product(
        donor_strategies,
        label_strategies,
        multipliers,
        n_neighbors_values,
    ):
        min_values = min_similarities if donor_strategy != "random" else [None]
        std_values = max_teacher_stds if label_strategy == "uncertainty_filtered_teacher" else [None]
        for min_similarity, max_teacher_std in product(min_values, std_values):
            policies.append(
                ConditionTransferConfig(
                    donor_strategy=str(donor_strategy),
                    synthetic_multiplier=float(multiplier),
                    n_neighbors=int(n_neighbors),
                    label_strategy=str(label_strategy),
                    teacher_models=[str(model) for model in transfer.get("teacher_models", ["ridge", "random_forest"])],
                    max_teacher_std=_none_or_float(max_teacher_std),
                    min_similarity=_none_or_float(min_similarity),
                    high_yield_threshold=float(transfer.get("high_yield_threshold", 70.0)),
                    clip_y_min=float(transfer.get("clip_y_min", 0.0)),
                    clip_y_max=float(transfer.get("clip_y_max", 100.0)),
                    candidates_per_real=int(transfer.get("candidates_per_real", 20)),
                    random_state=seed,
                    donor_similarity_n_bits=int(transfer.get("donor_similarity_n_bits", 2048)),
                    donor_similarity_radius=int(transfer.get("donor_similarity_radius", 2)),
                    donor_similarity_backend=str(
                        transfer.get("donor_similarity_backend", "auto")
                    ),
                )
            )
    return policies


def _select_policies(policy_metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config.get("selection", {})
    split = str(selection.get("split", "valid"))
    metric = str(selection.get("metric", "rmse"))
    lower_is_better = bool(selection.get("lower_is_better", True))
    candidates = policy_metrics.loc[
        policy_metrics["representation"].astype(str).str.startswith(
            "anonymous_condition_transfer_"
        )
        & (policy_metrics["split"] == split)
        & (policy_metrics["metric"] == metric)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values("value", ascending=lower_is_better, kind="mergesort")
    selected = candidates.groupby(
        ["seed", "train_fraction", "model"],
        dropna=False,
        as_index=False,
    ).head(1)
    return selected.rename(columns={"value": f"{split}_{metric}"}).reset_index(drop=True)


def _mark_selected_policy_metrics(
    policy_metrics: pd.DataFrame,
    selected_policies: pd.DataFrame,
) -> pd.DataFrame:
    if selected_policies.empty:
        return policy_metrics
    key_columns = ["seed", "train_fraction", "model", "policy_id"]
    selected_keys = selected_policies[key_columns].drop_duplicates().copy()
    selected_keys["selected_policy_marker"] = True
    marked = policy_metrics.drop(columns=["selected_policy"]).merge(
        selected_keys,
        on=key_columns,
        how="left",
    )
    marked["selected_policy"] = marked["selected_policy_marker"].eq(True)
    return marked.drop(columns=["selected_policy_marker"])


def _save_vs_original_rf(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_test = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"].copy()
    baseline = policy_metrics.loc[
        (policy_metrics["split"] == "test")
        & policy_metrics["representation"].astype(str).str.endswith("_real_only")
        & (policy_metrics["model"] == "random_forest"),
        ["seed", "train_fraction", "metric", "value"],
    ].rename(columns={"value": "baseline_value"})
    by_seed = selected_test.merge(baseline, on=["seed", "train_fraction", "metric"], how="left")
    by_seed["delta"] = by_seed["value"] - by_seed["baseline_value"]
    by_seed["delta_interpretation"] = np.where(
        by_seed["metric"].isin(["mae", "rmse"]),
        "negative_better_positive_worse",
        "positive_better_negative_worse",
    )
    by_seed.to_csv(output_paths["condition_transfer_vs_original_rf_by_seed_path"], index=False)
    summary = _delta_summary(by_seed)
    summary.to_csv(output_paths["condition_transfer_vs_original_rf_summary_path"], index=False)
    return by_seed, summary


def _save_vs_real_only_same_model(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_test = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"].copy()
    baseline = policy_metrics.loc[
        (policy_metrics["split"] == "test")
        & policy_metrics["representation"].astype(str).str.endswith("_real_only"),
        ["seed", "train_fraction", "model", "metric", "value"],
    ].rename(columns={"value": "baseline_value"})
    by_seed = selected_test.merge(
        baseline,
        on=["seed", "train_fraction", "model", "metric"],
        how="left",
    )
    by_seed["delta"] = by_seed["value"] - by_seed["baseline_value"]
    by_seed["delta_interpretation"] = np.where(
        by_seed["metric"].isin(["mae", "rmse"]),
        "negative_better_positive_worse",
        "positive_better_negative_worse",
    )
    by_seed.to_csv(output_paths["condition_transfer_vs_real_only_same_model_by_seed_path"], index=False)
    summary = _delta_summary(by_seed)
    summary.to_csv(output_paths["condition_transfer_vs_real_only_same_model_summary_path"], index=False)
    return by_seed, summary


def _delta_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["train_fraction", "model", "metric", "delta_interpretation"]
    if by_seed.empty:
        return pd.DataFrame(columns=[*group_columns, "mean", "std", "count"])
    return by_seed.groupby(group_columns, dropna=False)["delta"].agg(mean="mean", std="std", count="count").reset_index()


def _summarize_for_reporting(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    report_metrics = pd.concat(
        [
            policy_metrics.loc[
                policy_metrics["representation"].astype(str).str.endswith("_real_only")
            ],
            selected_policy_metrics,
        ],
        ignore_index=True,
    )
    group_columns = [
        "train_fraction",
        "representation",
        "donor_strategy",
        "label_strategy",
        "model",
        "split",
        "metric",
    ]
    return report_metrics.groupby(group_columns, dropna=False)["value"].agg(mean="mean", std="std", count="count").reset_index()


def _write_outputs(
    policy_metrics: pd.DataFrame,
    selected_policies: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    for key in [
        "policy_metrics_path",
        "selected_policies_path",
        "selected_policy_metrics_path",
        "summary_path",
        "synthetic_audit_path",
        "synthetic_candidate_audit_path",
    ]:
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
    policy_metrics.to_csv(output_paths["policy_metrics_path"], index=False)
    selected_policies.to_csv(output_paths["selected_policies_path"], index=False)
    selected_policy_metrics.to_csv(output_paths["selected_policy_metrics_path"], index=False)
    summary.to_csv(output_paths["summary_path"], index=False)
    audit.to_csv(output_paths["synthetic_audit_path"], index=False)
    candidate_audit.to_csv(
        output_paths["synthetic_candidate_audit_path"], index=False
    )


def _print_test_summary(summary: pd.DataFrame) -> None:
    test_summary = summary.loc[summary["split"] == "test"]
    if test_summary.empty:
        return
    table = test_summary.pivot_table(
        index=["train_fraction", "representation", "donor_strategy", "label_strategy", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    columns = [
        column
        for column in [
            "train_fraction",
            "representation",
            "donor_strategy",
            "label_strategy",
            "model",
            "mae",
            "rmse",
            "r2",
            "spearman",
        ]
        if column in table.columns
    ]
    print("\nTEST MEAN METRICS ACROSS SEEDS")
    print(table[columns].to_string(index=False))


def _print_selected_policy_table(
    selected_policies: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
) -> None:
    if selected_policies.empty:
        return
    test_metrics = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"]
    pivot = test_metrics.pivot_table(
        index=["seed", "train_fraction", "model", "policy_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    table = selected_policies.merge(
        pivot,
        on=["seed", "train_fraction", "model", "policy_id"],
        how="left",
    )
    columns = [
        column
        for column in [
            "train_fraction",
            "model",
            "donor_strategy",
            "label_strategy",
            "synthetic_multiplier",
            "valid_rmse",
            "rmse",
            "r2",
            "spearman",
        ]
        if column in table.columns
    ]
    print("\nSELECTED CONDITION-TRANSFER POLICIES")
    print(table[columns].rename(columns={"rmse": "test_rmse", "r2": "test_r2", "spearman": "test_spearman"}).to_string(index=False))


def _print_delta_summary(title: str, delta_summary: pd.DataFrame) -> None:
    if delta_summary.empty:
        return
    table = delta_summary.pivot_table(
        index=["train_fraction", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    columns = [column for column in ["train_fraction", "model", "mae", "rmse", "r2", "spearman"] if column in table.columns]
    print(f"\n{title}")
    print("MAE/RMSE: negative is better. R2/Spearman: positive is better.")
    print(table[columns].to_string(index=False))


def _policy_metric_fields(metadata: dict[str, Any]) -> dict[str, object]:
    fields = _empty_policy_metadata()
    fields.update({key: metadata.get(key, fields[key]) for key in fields})
    return fields


def _empty_policy_metadata() -> dict[str, object]:
    return {
        "donor_strategy": "",
        "label_strategy": "",
        "synthetic_multiplier": np.nan,
        "n_neighbors": np.nan,
        "min_similarity": np.nan,
        "max_teacher_std": np.nan,
        "n_candidates_generated": 0,
        "n_candidates_accepted": 0,
        "n_synthetic_train": 0,
        "filter_acceptance_rate": np.nan,
        "mean_donor_similarity": np.nan,
        "mean_teacher_std": np.nan,
        "mean_synthetic_yield": np.nan,
        "std_synthetic_yield": np.nan,
    }


def _policy_id(
    seed: int,
    train_fraction: float,
    policy_index: int,
    policy: ConditionTransferConfig,
) -> str:
    return (
        f"seed={seed}|frac={train_fraction}|policy={policy_index}|"
        f"donor={policy.donor_strategy}|label={policy.label_strategy}|"
        f"mult={policy.synthetic_multiplier}|k={policy.n_neighbors}|"
        f"sim={policy.min_similarity}|std={policy.max_teacher_std}"
    )


def _resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output = config.get("output", {})
    directory = Path(output.get("directory", "results/condition_transfer_corrected"))
    return {
        "directory": directory,
        "policy_metrics_path": Path(output.get("policy_metrics_path", directory / "policy_metrics.csv")),
        "selected_policies_path": Path(output.get("selected_policies_path", directory / "selected_policies.csv")),
        "selected_policy_metrics_path": Path(output.get("selected_policy_metrics_path", directory / "selected_policy_metrics.csv")),
        "summary_path": Path(output.get("summary_path", directory / "summary.csv")),
        "synthetic_audit_path": Path(output.get("synthetic_audit_path", directory / "synthetic_audit.csv")),
        "synthetic_candidate_audit_path": Path(
            output.get(
                "synthetic_candidate_audit_path",
                directory / "synthetic_candidate_audit.csv",
            )
        ),
        "condition_transfer_vs_original_rf_by_seed_path": Path(
            output.get("condition_transfer_vs_original_rf_by_seed_path", directory / "condition_transfer_vs_original_rf_by_seed.csv")
        ),
        "condition_transfer_vs_original_rf_summary_path": Path(
            output.get("condition_transfer_vs_original_rf_summary_path", directory / "condition_transfer_vs_original_rf_summary.csv")
        ),
        "condition_transfer_vs_real_only_same_model_by_seed_path": Path(
            output.get(
                "condition_transfer_vs_real_only_same_model_by_seed_path",
                directory / "condition_transfer_vs_real_only_same_model_by_seed.csv",
            )
        ),
        "condition_transfer_vs_real_only_same_model_summary_path": Path(
            output.get(
                "condition_transfer_vs_real_only_same_model_summary_path",
                directory / "condition_transfer_vs_real_only_same_model_summary.csv",
            )
        ),
    }


def _get_dataset_path(config: dict[str, Any]) -> str | Path:
    path = config.get("dataset", {}).get("path")
    if not path:
        raise ValueError("Config must define dataset.path.")
    return path


def _split_positions(split: pd.DataFrame) -> np.ndarray:
    return split.index.to_numpy(dtype=int)


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _none_or_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return float(value)


if __name__ == "__main__":
    main()
