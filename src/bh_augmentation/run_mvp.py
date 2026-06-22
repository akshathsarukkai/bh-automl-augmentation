"""Run the complete MVP workflow from one configuration file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.run_augmentation import run_augmentation
from bh_augmentation.run_automl import run_automl
from bh_augmentation.run_baseline import run_baseline
from bh_augmentation.run_recommendation import run_recommendation
from bh_augmentation.run_report import run_report
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed


def run_mvp(config_path: str | Path) -> dict[str, Path]:
    """Run the baseline, low-data, augmentation, recommendation, and report workflow."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    output_dir = Path(config.get("output", {}).get("results_dir", "results/mvp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    prepared_data_path = _prepare_dataset(config, output_dir)
    base_runner_config = _base_runner_config(config, prepared_data_path)

    baseline_config_path = _write_yaml(
        config_dir / "baseline.yaml",
        _baseline_config(base_runner_config, output_dir),
    )
    low_data_config_path = _write_yaml(
        config_dir / "low_data.yaml",
        _low_data_config(base_runner_config, output_dir),
    )
    augmentation_config_path = _write_yaml(
        config_dir / "augmentation.yaml",
        _augmentation_config(base_runner_config, output_dir),
    )
    recommendation_config_path = _write_yaml(
        config_dir / "recommendation.yaml",
        _recommendation_config(base_runner_config, output_dir),
    )

    outputs = {
        "prepared_data": prepared_data_path,
        "baseline_metrics": run_baseline(baseline_config_path),
        "low_data_metrics": run_baseline(low_data_config_path),
        "augmentation_metrics": run_augmentation(augmentation_config_path),
        "recommendation_metrics": run_recommendation(recommendation_config_path),
    }

    if bool(config.get("automl", {}).get("enabled", False)):
        automl_config_path = _write_yaml(
            config_dir / "automl.yaml",
            _automl_config(base_runner_config, output_dir, config),
        )
        automl_outputs = run_automl(automl_config_path)
        outputs.update({f"automl_{key}": value for key, value in automl_outputs.items()})

    outputs["report"] = run_report(output_dir, output_dir / "final_report.md")
    return outputs


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_mvp`."""
    parser = argparse.ArgumentParser(description="Run the complete MVP workflow.")
    parser.add_argument("--config", required=True, help="Path to an MVP YAML config file.")
    args = parser.parse_args()

    outputs = run_mvp(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _prepare_dataset(config: dict[str, Any], output_dir: Path) -> Path:
    data_config = config.get("dataset", {})
    use_fixture = bool(data_config.get("use_fixture", False))
    source_path = Path(
        data_config.get(
            "fixture_path" if use_fixture else "path",
            "tests/fixtures/sample_bh.csv",
        )
    )
    raw = load_reaction_csv(source_path)
    cleaned = clean_buchwald_hartwig(raw)
    if cleaned.empty:
        raise ValueError("No valid rows remain after cleaning MVP input data.")

    if use_fixture:
        cleaned = _expand_fixture_rows(
            cleaned,
            min_rows=int(data_config.get("fixture_min_rows", 12)),
        )

    prepared_path = Path(
        data_config.get("prepared_path", output_dir / "data" / "mvp_clean.csv")
    )
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(prepared_path, index=False)
    return prepared_path


def _expand_fixture_rows(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    frames = []
    repeat = 0
    while sum(len(frame) for frame in frames) < min_rows:
        copy = df.copy()
        copy["reaction_id"] = copy["reaction_id"].astype(str) + f"_mvp_{repeat}"
        copy["yield"] = (copy["yield"] + repeat).clip(0, 100)
        frames.append(copy)
        repeat += 1
    return pd.concat(frames, ignore_index=True).iloc[:min_rows].copy()


def _base_runner_config(config: dict[str, Any], data_path: Path) -> dict[str, Any]:
    return {
        "seed": int(config.get("seed", 42)),
        "dataset": {
            "name": config.get("dataset", {}).get("name", "mvp_dataset"),
            "path": str(data_path),
        },
        "splits": config.get(
            "splits",
            {"method": "random", "train_size": 0.6, "valid_size": 0.2, "test_size": 0.2},
        ),
        "features": config.get(
            "features",
            {
                "kind": "reaction_role_concat_delta",
                "n_bits": 2048,
                "radius": 2,
            },
        ),
        "models": config.get("models", ["ridge"]),
        "metrics": config.get("metrics", ["rmse", "mae"]),
    }


def _baseline_config(base: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = dict(base)
    config["output"] = {
        "metrics_path": str(output_dir / "baseline" / "baseline_metrics.csv"),
        "split_metadata_path": str(output_dir / "baseline" / "split_metadata.csv"),
    }
    return config


def _low_data_config(base: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = dict(base)
    config["low_data"] = {
        "enabled": True,
        "train_fractions": [0.5, 1.0],
    }
    config["output"] = {
        "metrics_path": str(output_dir / "low_data" / "baseline_metrics.csv"),
        "split_metadata_path": str(output_dir / "low_data" / "split_metadata.csv"),
    }
    return config


def _augmentation_config(base: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = dict(base)
    config["augmentation"] = {}
    config["output"] = {
        "metrics_path": str(output_dir / "augmentation" / "safe_aug_metrics.csv"),
        "split_metadata_path": str(output_dir / "augmentation" / "split_metadata.csv"),
    }
    return config


def _recommendation_config(base: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = dict(base)
    config["recommendation"] = {
        "seed_fraction": 0.25,
        "k": 2,
        "high_yield_threshold": 70,
    }
    config["augmentation"] = {}
    config["output"] = {
        "recommendation_metrics_path": str(output_dir / "recommendation" / "topk_metrics.csv")
    }
    return config


def _automl_config(base: dict[str, Any], output_dir: Path, mvp_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    config["automl"] = mvp_config.get("automl", {})
    config["output"] = {
        "trial_table_path": str(output_dir / "automl" / "trials.csv"),
        "best_config_path": str(output_dir / "automl" / "best_config.json"),
        "final_test_metrics_path": str(output_dir / "automl" / "final_test_metrics.csv"),
    }
    return config


def _write_yaml(path: Path, config: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
