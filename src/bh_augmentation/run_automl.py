"""Command-line runner for compact Optuna AutoML search."""

from __future__ import annotations

import argparse
from pathlib import Path

from bh_augmentation.automl.search import run_automl_search
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.run_baseline import _create_splits, _get_dataset_path, _resolve_feature_config
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed


def run_automl(config_path: str | Path) -> dict[str, Path]:
    """Run compact AutoML search from a YAML config."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = clean_buchwald_hartwig(raw_df).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after cleaning; cannot run AutoML.")

    splits = _create_splits(df, config.get("splits", {}), seed)
    feature_config = _resolve_feature_config(config.get("features", {}), df)
    return run_automl_search(
        splits=splits,
        base_feature_config=feature_config,
        automl_config=config.get("automl", {}),
        output_config=config.get("output", {}),
        seed=seed,
    )


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_automl`."""
    parser = argparse.ArgumentParser(description="Run compact Optuna AutoML search.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    outputs = run_automl(args.config)
    print(f"Saved AutoML trials to {outputs['trials']}")
    print(f"Saved AutoML best config to {outputs['best_config']}")
    print(f"Saved AutoML final test metrics to {outputs['final_test_metrics']}")


if __name__ == "__main__":
    main()
