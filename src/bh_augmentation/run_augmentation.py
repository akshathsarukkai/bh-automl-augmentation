"""Command-line runner for safe augmentation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.order_permutation import permute_reaction_components
from bh_augmentation.augmentation.smiles_randomization import augment_randomized_smiles
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.reporting.make_report import save_metrics_csv
from bh_augmentation.run_baseline import (
    _compute_metric,
    _create_split_variants,
    _get_dataset_path,
    _parse_model_config,
    _resolve_feature_config,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed


def run_augmentation(config_path: str | Path) -> Path:
    """Run no-augmentation and configured safe-augmentation experiments."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = clean_buchwald_hartwig(raw_df).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run augmentation experiment.")

    split_variants = _create_split_variants(df, config, seed)
    feature_config = _resolve_feature_config(config.get("features", {}), df)
    augmentation_config = config.get("augmentation", {})

    model_configs = config.get("models", ["ridge"])
    metric_names = config.get("metrics", ["rmse", "mae", "r2"])
    records: list[dict[str, object]] = []

    for train_fraction, splits in split_variants:
        train_variants = _build_training_variants(
            splits["train"],
            augmentation_config,
            feature_config,
            seed,
        )

        for variant_name, train_df in train_variants.items():
            train_df = _with_augmentation_metadata(train_df, is_augmented_default=False)
            valid_df = _with_augmentation_metadata(splits["valid"], is_augmented_default=False)
            test_df = _with_augmentation_metadata(splits["test"], is_augmented_default=False)

            combined_df = pd.concat(
                [
                    train_df.assign(__split="train"),
                    valid_df.assign(__split="valid"),
                    test_df.assign(__split="test"),
                ],
                ignore_index=True,
            )
            X, y, _ = build_feature_matrix(combined_df, feature_config)
            train_mask = combined_df["__split"].to_numpy() == "train"

            for model_config in model_configs:
                model_name, model_kwargs = _parse_model_config(model_config)
                model = get_model(model_name, seed=seed, **model_kwargs)
                fitted_model = train_model(model, X[train_mask], y[train_mask])

                for split_name in ["valid", "test"]:
                    eval_mask = combined_df["__split"].to_numpy() == split_name
                    predictions = predict_model(fitted_model, X[eval_mask])
                    y_true = y[eval_mask]
                    eval_augmented_rows = int(combined_df.loc[eval_mask, "is_augmented"].sum())
                    for metric_name in metric_names:
                        records.append(
                            {
                                "train_fraction": train_fraction,
                                "augmentation": variant_name,
                                "model": model_name,
                                "split": split_name,
                                "metric": metric_name,
                                "value": _compute_metric(metric_name, y_true, predictions),
                                "train_rows": int(train_mask.sum()),
                                "augmented_train_rows": int(train_df["is_augmented"].sum()),
                                "eval_rows": int(eval_mask.sum()),
                                "eval_augmented_rows": eval_augmented_rows,
                            }
                        )

    return save_metrics_csv(records, _get_metrics_output_path(config))


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_augmentation`."""
    parser = argparse.ArgumentParser(description="Run safe augmentation comparisons.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    output_path = run_augmentation(args.config)
    print(f"Saved safe augmentation metrics to {output_path}")


def _build_training_variants(
    train_df: pd.DataFrame,
    augmentation_config: dict[str, Any],
    feature_config: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    variants = {"none": _with_augmentation_metadata(train_df, is_augmented_default=False)}

    smiles_config = augmentation_config.get("smiles_randomization", {})
    order_config = augmentation_config.get("order_permutation", {})
    combined_config = augmentation_config.get("combined", {})

    if smiles_config.get("enabled", False):
        variants["randomized_smiles"] = _apply_smiles_randomization(
            train_df,
            smiles_config,
            feature_config,
            seed,
        )
    if order_config.get("enabled", False):
        variants["order_permutation"] = _apply_order_permutation(
            train_df,
            order_config,
            seed,
        )
    if combined_config.get("enabled", False):
        combined = train_df
        if smiles_config.get("enabled", False):
            combined = _apply_smiles_randomization(combined, smiles_config, feature_config, seed)
        if order_config.get("enabled", False):
            order_augmented = _apply_order_permutation(combined, order_config, seed)
            combined = pd.concat([combined, order_augmented[order_augmented["is_augmented"]]], ignore_index=True)
        variants["combined_safe"] = _with_augmentation_metadata(combined, is_augmented_default=False)

    return variants


def _apply_smiles_randomization(
    train_df: pd.DataFrame,
    smiles_config: dict[str, Any],
    feature_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows_to_augment = _select_augmentation_rows(
        train_df,
        float(smiles_config.get("ratio", 1.0)),
        int(smiles_config.get("random_state", seed)),
    )
    untouched = train_df.drop(index=rows_to_augment.index)
    augmented = augment_randomized_smiles(
        rows_to_augment,
        smiles_columns=smiles_config.get("smiles_columns", feature_config.get("smiles_columns", [])),
        n_augments=int(smiles_config.get("n_augments", smiles_config.get("n_variants", 1))),
        seed=int(smiles_config.get("random_state", seed)),
        include_original=True,
    )
    return _with_augmentation_metadata(
        pd.concat([untouched, augmented], ignore_index=True),
        is_augmented_default=False,
    )


def _apply_order_permutation(
    train_df: pd.DataFrame,
    order_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows_to_augment = _select_augmentation_rows(
        train_df,
        float(order_config.get("ratio", 1.0)),
        int(order_config.get("random_state", seed)),
    )
    untouched = train_df.drop(index=rows_to_augment.index)
    augmented = permute_reaction_components(
        rows_to_augment,
        component_columns=order_config.get("component_columns", []),
        n_permutations=int(order_config.get("n_permutations", order_config.get("max_permutations", 1))),
        seed=int(order_config.get("random_state", seed)),
        include_original=True,
    )
    return _with_augmentation_metadata(
        pd.concat([untouched, augmented], ignore_index=True),
        is_augmented_default=False,
    )


def _select_augmentation_rows(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    if not 0 <= ratio <= 1:
        raise ValueError("Augmentation ratio must be between 0 and 1.")
    if ratio == 0 or df.empty:
        return df.iloc[0:0].copy()

    n_rows = max(1, int(np.floor(len(df) * ratio)))
    sampled_indices = df.sample(n=n_rows, random_state=seed).index
    return df.loc[sampled_indices].copy()


def _with_augmentation_metadata(
    df: pd.DataFrame,
    is_augmented_default: bool,
) -> pd.DataFrame:
    result = df.copy()
    if "is_augmented" not in result.columns:
        result["is_augmented"] = is_augmented_default
    result["is_augmented"] = result["is_augmented"].fillna(is_augmented_default).astype(bool)
    if "augmentation_type" not in result.columns:
        result["augmentation_type"] = "none" if not is_augmented_default else "safe_augmentation"
    if "source_reaction_id" not in result.columns:
        if "reaction_id" in result.columns:
            result["source_reaction_id"] = result["reaction_id"].astype(str)
        else:
            result["source_reaction_id"] = result.index.astype(str)
    return result


def _get_metrics_output_path(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    return Path(output_config.get("metrics_path", "results/augmentation/safe_aug_metrics.csv"))


if __name__ == "__main__":
    main()
