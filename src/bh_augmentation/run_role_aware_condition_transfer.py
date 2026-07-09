"""Run role-aware Buchwald-Hartwig condition-transfer experiments."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.data.bh_condition_reader import RECOVERED_COLUMNS, augment_bh_dataframe
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_supervised_ae_latent_baseline import (
    _compute_metric,
    _create_split_variants,
    _parse_model_config,
    _resolve_model_configs,
    _resolve_seeds,
    _validate_metrics,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed


def run_role_aware_condition_transfer(config_path: str | Path) -> dict[str, Path]:
    """Run role-aware condition-transfer policies and save reports."""
    config = load_config(config_path)
    seeds = _resolve_seeds(config)
    metric_names = list(config.get("metrics", ["rmse", "mae", "r2", "spearman"]))
    _validate_metrics(metric_names)
    output_paths = _resolve_output_paths(config)
    output_paths["directory"].mkdir(parents=True, exist_ok=True)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = _prepare_condition_dataframe(clean_buchwald_hartwig(raw_df))
    if df.empty:
        raise ValueError("No rows remain after condition-role parsing.")
    print(f"Rows used after condition-role parsing: {len(df)}")
    print("Role validation status counts:")
    print(df["role_validation_status"].value_counts(dropna=False).to_string())

    feature_config = {"kind": "reaction_role_concat", **dict(config.get("features", {}))}
    feature_config["kind"] = "reaction_role_concat"
    X, y, _ = build_feature_matrix(df, feature_config)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    metric_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
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
                    representation="original_6144",
                    policy_metadata=_empty_policy_metadata(),
                    metric_names=metric_names,
                    policy_id=f"seed={seed}|frac={train_fraction}|real_only",
                )
            )

            for policy_index, policy in enumerate(_iter_role_transfer_policies(config, seed)):
                policy_id = _policy_id(seed, float(train_fraction), policy_index, policy)
                result = generate_role_aware_condition_transfer_examples(train_df, X_train, y_train, policy)
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

                synthetic_df = result["synthetic_df"]
                if len(synthetic_df) == 0:
                    X_augmented = X_train
                    y_augmented = y_train
                else:
                    X_augmented = np.vstack([X_train, np.asarray(result["X_synthetic"], dtype=np.float32)])
                    y_augmented = np.concatenate([y_train, np.asarray(result["synthetic_y"], dtype=float)])

                metric_records.extend(
                    _evaluate_models(
                        X_augmented,
                        y_augmented,
                        {"valid": (X_valid, y_valid), "test": (X_test, y_test)},
                        config,
                        seed=seed,
                        train_fraction=float(train_fraction),
                        n_real_train=n_real_train,
                        representation="role_aware_condition_transfer",
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
    summary = _summarize(policy_metrics, selected_policy_metrics)
    same_model_delta, same_model_summary = _save_vs_real_only_same_model(policy_metrics, selected_policy_metrics, output_paths)
    original_rf_delta, original_rf_summary = _save_vs_original_rf(policy_metrics, selected_policy_metrics, output_paths)

    _write_outputs(policy_metrics, selected_policies, selected_policy_metrics, summary, audit, output_paths)
    _print_test_summary(summary)
    _print_selected_policy_table(selected_policies, selected_policy_metrics)
    _print_delta_summary("ROLE-AWARE CONDITION TRANSFER DELTAS VS REAL-ONLY SAME MODEL", same_model_summary)
    _print_delta_summary("ROLE-AWARE CONDITION TRANSFER DELTAS VS ORIGINAL 6144D RF", original_rf_summary)
    _print_winning_modes(selected_policies)
    return output_paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run role-aware condition-transfer augmentation.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    paths = run_role_aware_condition_transfer(args.config)
    print(f"Saved role-aware condition-transfer metrics to {paths['policy_metrics_path']}")


def _prepare_condition_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not set(RECOVERED_COLUMNS).issubset(df.columns) or "role_validation_status" not in df.columns:
        df = augment_bh_dataframe(df)
    valid_status = {"valid", "repaired", "valid_position_fallback"}
    mask = df["condition_parse_status"].eq("ok") & df["role_validation_status"].isin(valid_status)
    excluded = int((~mask).sum())
    if excluded:
        print(f"Excluded {excluded} rows with invalid condition-role parsing.")
    return df.loc[mask].reset_index(drop=True)


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
    for model_config in _resolve_model_configs(config.get("models", ["xgboost"])):
        model_name, model_kwargs = _parse_model_config(model_config)
        model = get_model(model_name, seed=seed, **model_kwargs)
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


def _iter_role_transfer_policies(config: dict[str, Any], seed: int) -> list[RoleAwareConditionTransferConfig]:
    transfer = config.get("role_aware_condition_transfer", {})
    if not transfer.get("enabled", True):
        return []
    policies: list[RoleAwareConditionTransferConfig] = []
    for mode, donor_strategy, label_strategy, multiplier in product(
        _as_list(transfer.get("role_transfer_modes", ["ligand_only"])),
        _as_list(transfer.get("donor_strategies", ["random"])),
        _as_list(transfer.get("label_strategies", ["uncertainty_filtered_teacher"])),
        _as_list(transfer.get("synthetic_multipliers", [0.5])),
    ):
        min_values = _as_list(transfer.get("min_similarities", [None])) if donor_strategy != "random" else [None]
        std_values = (
            _as_list(transfer.get("max_teacher_stds", [None]))
            if label_strategy == "uncertainty_filtered_teacher"
            else [None]
        )
        for min_similarity, max_teacher_std in product(min_values, std_values):
            policies.append(
                RoleAwareConditionTransferConfig(
                    role_transfer_mode=str(mode),
                    donor_strategy=str(donor_strategy),
                    label_strategy=str(label_strategy),
                    synthetic_multiplier=float(multiplier),
                    max_candidates_per_source=int(transfer.get("max_candidates_per_source", 5)),
                    teacher_models=[str(model) for model in transfer.get("teacher_models", ["ridge", "random_forest"])],
                    max_teacher_std=_none_or_float(max_teacher_std),
                    min_similarity=_none_or_float(min_similarity),
                    clip_y_min=float(transfer.get("clip_y_min", 0.0)),
                    clip_y_max=float(transfer.get("clip_y_max", 100.0)),
                    random_state=seed,
                    max_resample_attempts=int(transfer.get("max_resample_attempts", 10)),
                )
            )
    max_policies = transfer.get("max_policies")
    if max_policies is not None:
        max_count = int(max_policies)
        if max_count <= 0 or len(policies) <= max_count:
            return policies
        indices = np.linspace(0, len(policies) - 1, max_count, dtype=int)
        return [policies[index] for index in sorted(set(indices.tolist()))]
    return policies


def _select_policies(policy_metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config.get("selection", {})
    split = str(selection.get("split", "valid"))
    metric = str(selection.get("metric", "rmse"))
    lower_is_better = bool(selection.get("lower_is_better", True))
    candidates = policy_metrics.loc[
        (policy_metrics["representation"] == "role_aware_condition_transfer")
        & (policy_metrics["split"] == split)
        & (policy_metrics["metric"] == metric)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values("value", ascending=lower_is_better, kind="mergesort")
    selected = candidates.groupby(["seed", "train_fraction", "model"], dropna=False, as_index=False).head(1)
    return selected.rename(columns={"value": f"{split}_{metric}"}).reset_index(drop=True)


def _mark_selected_policy_metrics(policy_metrics: pd.DataFrame, selected_policies: pd.DataFrame) -> pd.DataFrame:
    if selected_policies.empty:
        return policy_metrics
    key_columns = ["seed", "train_fraction", "model", "policy_id"]
    selected_keys = selected_policies[key_columns].drop_duplicates().copy()
    selected_keys["selected_policy_marker"] = True
    marked = policy_metrics.drop(columns=["selected_policy"]).merge(selected_keys, on=key_columns, how="left")
    marked["selected_policy"] = marked["selected_policy_marker"].eq(True)
    return marked.drop(columns=["selected_policy_marker"])


def _summarize(policy_metrics: pd.DataFrame, selected_policy_metrics: pd.DataFrame) -> pd.DataFrame:
    report_metrics = pd.concat(
        [
            policy_metrics.loc[policy_metrics["representation"] == "original_6144"],
            selected_policy_metrics,
        ],
        ignore_index=True,
    )
    group_columns = [
        "train_fraction",
        "representation",
        "role_transfer_mode",
        "donor_strategy",
        "label_strategy",
        "model",
        "split",
        "metric",
    ]
    return report_metrics.groupby(group_columns, dropna=False)["value"].agg(mean="mean", std="std", count="count").reset_index()


def _save_vs_real_only_same_model(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_test = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"].copy()
    baseline = policy_metrics.loc[
        (policy_metrics["split"] == "test") & (policy_metrics["representation"] == "original_6144"),
        ["seed", "train_fraction", "model", "metric", "value"],
    ].rename(columns={"value": "baseline_value"})
    by_seed = selected_test.merge(baseline, on=["seed", "train_fraction", "model", "metric"], how="left")
    by_seed["delta"] = by_seed["value"] - by_seed["baseline_value"]
    by_seed["delta_interpretation"] = np.where(by_seed["metric"].isin(["mae", "rmse"]), "negative_better", "positive_better")
    by_seed.to_csv(output_paths["role_transfer_vs_real_only_same_model_by_seed_path"], index=False)
    summary = _delta_summary(by_seed)
    summary.to_csv(output_paths["role_transfer_vs_real_only_same_model_summary_path"], index=False)
    return by_seed, summary


def _save_vs_original_rf(
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_test = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"].copy()
    baseline = policy_metrics.loc[
        (policy_metrics["split"] == "test")
        & (policy_metrics["representation"] == "original_6144")
        & (policy_metrics["model"] == "random_forest"),
        ["seed", "train_fraction", "metric", "value"],
    ].rename(columns={"value": "baseline_value"})
    by_seed = selected_test.merge(baseline, on=["seed", "train_fraction", "metric"], how="left")
    by_seed["delta"] = by_seed["value"] - by_seed["baseline_value"]
    by_seed["delta_interpretation"] = np.where(by_seed["metric"].isin(["mae", "rmse"]), "negative_better", "positive_better")
    by_seed.to_csv(output_paths["role_transfer_vs_original_rf_by_seed_path"], index=False)
    summary = _delta_summary(by_seed)
    summary.to_csv(output_paths["role_transfer_vs_original_rf_summary_path"], index=False)
    return by_seed, summary


def _delta_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["train_fraction", "model", "metric", "delta_interpretation"]
    if by_seed.empty:
        return pd.DataFrame(columns=[*group_columns, "mean", "std", "count"])
    return by_seed.groupby(group_columns, dropna=False)["delta"].agg(mean="mean", std="std", count="count").reset_index()


def _write_outputs(
    policy_metrics: pd.DataFrame,
    selected_policies: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    for key in [
        "policy_metrics_path",
        "selected_policies_path",
        "selected_policy_metrics_path",
        "summary_path",
        "role_transfer_audit_path",
    ]:
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)
    policy_metrics.to_csv(output_paths["policy_metrics_path"], index=False)
    selected_policies.to_csv(output_paths["selected_policies_path"], index=False)
    selected_policy_metrics.to_csv(output_paths["selected_policy_metrics_path"], index=False)
    summary.to_csv(output_paths["summary_path"], index=False)
    audit.to_csv(output_paths["role_transfer_audit_path"], index=False)


def _print_test_summary(summary: pd.DataFrame) -> None:
    test_summary = summary.loc[summary["split"] == "test"]
    if test_summary.empty:
        return
    table = test_summary.pivot_table(
        index=["train_fraction", "role_transfer_mode", "donor_strategy", "label_strategy", "model"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    columns = [column for column in ["train_fraction", "role_transfer_mode", "donor_strategy", "label_strategy", "model", "mae", "rmse", "r2", "spearman"] if column in table.columns]
    print("\nTEST MEAN METRICS ACROSS SEEDS")
    print(table[columns].to_string(index=False))


def _print_selected_policy_table(selected_policies: pd.DataFrame, selected_policy_metrics: pd.DataFrame) -> None:
    if selected_policies.empty:
        return
    test_metrics = selected_policy_metrics.loc[selected_policy_metrics["split"] == "test"]
    pivot = test_metrics.pivot_table(
        index=["seed", "train_fraction", "model", "policy_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    table = selected_policies.merge(pivot, on=["seed", "train_fraction", "model", "policy_id"], how="left")
    columns = [column for column in ["seed", "train_fraction", "model", "role_transfer_mode", "donor_strategy", "label_strategy", "synthetic_multiplier", "valid_rmse", "rmse", "r2", "spearman"] if column in table.columns]
    print("\nSELECTED ROLE-AWARE CONDITION-TRANSFER POLICIES")
    print(table[columns].rename(columns={"rmse": "test_rmse", "r2": "test_r2", "spearman": "test_spearman"}).to_string(index=False))


def _print_delta_summary(title: str, delta_summary: pd.DataFrame) -> None:
    if delta_summary.empty:
        return
    table = delta_summary.pivot_table(index=["train_fraction", "model"], columns="metric", values="mean", aggfunc="first").reset_index()
    columns = [column for column in ["train_fraction", "model", "mae", "rmse", "r2", "spearman"] if column in table.columns]
    print(f"\n{title}")
    print("MAE/RMSE: negative is better. R2/Spearman: positive is better.")
    print(table[columns].to_string(index=False))


def _print_winning_modes(selected_policies: pd.DataFrame) -> None:
    if selected_policies.empty or "role_transfer_mode" not in selected_policies:
        return
    print("\nSELECTED ROLE_TRANSFER_MODE COUNTS")
    print(selected_policies["role_transfer_mode"].value_counts().to_string())


def _policy_metric_fields(metadata: dict[str, Any]) -> dict[str, object]:
    fields = _empty_policy_metadata()
    fields.update({key: metadata.get(key, fields[key]) for key in fields})
    return fields


def _empty_policy_metadata() -> dict[str, object]:
    return {
        "role_transfer_mode": "",
        "donor_strategy": "",
        "label_strategy": "",
        "synthetic_multiplier": np.nan,
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


def _policy_id(seed: int, train_fraction: float, policy_index: int, policy: RoleAwareConditionTransferConfig) -> str:
    return (
        f"seed={seed}|frac={train_fraction}|policy={policy_index}|"
        f"mode={policy.role_transfer_mode}|donor={policy.donor_strategy}|"
        f"label={policy.label_strategy}|mult={policy.synthetic_multiplier}|"
        f"sim={policy.min_similarity}|std={policy.max_teacher_std}"
    )


def _resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output = config.get("output", {})
    directory = Path(output.get("directory", "results/role_aware_condition_transfer"))
    return {
        "directory": directory,
        "policy_metrics_path": Path(output.get("policy_metrics_path", directory / "policy_metrics.csv")),
        "selected_policies_path": Path(output.get("selected_policies_path", directory / "selected_policies.csv")),
        "selected_policy_metrics_path": Path(output.get("selected_policy_metrics_path", directory / "selected_policy_metrics.csv")),
        "summary_path": Path(output.get("summary_path", directory / "summary.csv")),
        "role_transfer_audit_path": Path(output.get("role_transfer_audit_path", directory / "role_transfer_audit.csv")),
        "role_transfer_vs_real_only_same_model_by_seed_path": Path(output.get("role_transfer_vs_real_only_same_model_by_seed_path", directory / "role_transfer_vs_real_only_same_model_by_seed.csv")),
        "role_transfer_vs_real_only_same_model_summary_path": Path(output.get("role_transfer_vs_real_only_same_model_summary_path", directory / "role_transfer_vs_real_only_same_model_summary.csv")),
        "role_transfer_vs_original_rf_by_seed_path": Path(output.get("role_transfer_vs_original_rf_by_seed_path", directory / "role_transfer_vs_original_rf_by_seed.csv")),
        "role_transfer_vs_original_rf_summary_path": Path(output.get("role_transfer_vs_original_rf_summary_path", directory / "role_transfer_vs_original_rf_summary.csv")),
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
