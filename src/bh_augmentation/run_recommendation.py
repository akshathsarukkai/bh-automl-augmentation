"""Command-line runner for simulated reaction recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.evaluation.recommendation import simulate_single_round_recommendation
from bh_augmentation.reporting.make_report import save_metrics_csv
from bh_augmentation.run_baseline import _get_dataset_path
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed


def run_recommendation(config_path: str | Path) -> Path:
    """Run a single-round recommendation simulation and save top-k metrics."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    raw_df = load_reaction_csv(_get_dataset_path(config))
    df = clean_buchwald_hartwig(raw_df)
    recommendation_config = config.get("recommendation", {})
    results = simulate_single_round_recommendation(
        dataframe=df,
        model_config=_get_model_config(config),
        feature_config=config.get("features", {}),
        seed_size=recommendation_config.get("seed_size"),
        seed_fraction=recommendation_config.get("seed_fraction", 0.2),
        candidate_pool=None,
        k=int(recommendation_config.get("k", 5)),
        high_yield_threshold=float(recommendation_config.get("high_yield_threshold", 70.0)),
        augmentation_config=config.get("augmentation"),
        seed=seed,
    )
    return save_metrics_csv(results.to_dict(orient="records"), _get_output_path(config))


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_recommendation`."""
    parser = argparse.ArgumentParser(description="Run simulated reaction recommendation.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    output_path = run_recommendation(args.config)
    print(f"Saved recommendation metrics to {output_path}")


def _get_model_config(config: dict[str, Any]) -> str | dict[str, Any]:
    recommendation_config = config.get("recommendation", {})
    if "model" in recommendation_config:
        return recommendation_config["model"]
    models = config.get("models", ["ridge"])
    if not models:
        raise ValueError("At least one model must be configured.")
    return models[0]


def _get_output_path(config: dict[str, Any]) -> Path:
    output_config = config.get("output", {})
    return Path(output_config.get("recommendation_metrics_path", "results/recommendation/topk_metrics.csv"))


if __name__ == "__main__":
    main()
