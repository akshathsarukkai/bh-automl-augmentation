"""Tests for held-out group split runner support."""

from pathlib import Path

import pandas as pd

from bh_augmentation.run_augmentation import run_augmentation
from bh_augmentation.run_baseline import run_baseline


def _write_group_fixture(path: Path) -> None:
    rows = []
    products = [f"product_{index}" for index in range(5)]
    for product_index, product_key in enumerate(products):
        for repeat in range(4):
            row_index = product_index * 4 + repeat
            rows.append(
                {
                    "reaction_id": f"rxn_{row_index:03d}",
                    "reaction_smiles": (
                        f"CC{'C' * product_index}Br.N."
                        f"{'O' if repeat % 2 else 'Cl'}.{'C' if repeat < 2 else 'P'}"
                        f">>CC{'C' * product_index}N"
                    ),
                    "product_key": product_key,
                    "reactant_key": f"reactant_{repeat}",
                    "yield": float((row_index * 5) % 100),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _heldout_config(
    data_path: Path,
    metrics_path: Path,
    split_metadata_path: Path,
    group_column: str = "product_key",
) -> str:
    return f"""
seed: 17
dataset:
  name: synthetic_bh
  path: {data_path}
splits:
  method: heldout_group
  group_column: {group_column}
  heldout_fraction: 0.2
  valid_fraction: 0.1
features:
  kind: reaction_morgan_sum
  n_bits: 8
models:
  - ridge
metrics:
  - rmse
output:
  metrics_path: {metrics_path}
  split_metadata_path: {split_metadata_path}
"""


def test_baseline_runner_heldout_group_split_has_no_group_leakage(
    tmp_path: Path,
) -> None:
    """Baseline runner should keep held-out test groups out of training."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "baseline_metrics.csv"
    split_metadata_path = tmp_path / "baseline_split_metadata.csv"
    config_path = tmp_path / "baseline.yaml"
    _write_group_fixture(data_path)
    config_path.write_text(
        _heldout_config(data_path, metrics_path, split_metadata_path),
        encoding="utf-8",
    )

    run_baseline(config_path)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["split_method"]) == {"heldout_group"}
    assert set(metrics["group_column"]) == {"product_key"}

    metadata = pd.read_csv(split_metadata_path)
    train_groups = set(metadata.loc[metadata["split"] == "train", "group_value"])
    test_groups = set(metadata.loc[metadata["split"] == "test", "group_value"])
    assert train_groups.isdisjoint(test_groups)


def test_augmentation_runner_heldout_group_split_has_no_group_leakage(
    tmp_path: Path,
) -> None:
    """Augmentation runner should keep held-out test groups out of training."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "augmentation_metrics.csv"
    split_metadata_path = tmp_path / "augmentation_split_metadata.csv"
    config_path = tmp_path / "augmentation.yaml"
    _write_group_fixture(data_path)
    config_path.write_text(
        _heldout_config(data_path, metrics_path, split_metadata_path)
        + """
augmentation:
  condition_recombine_pseudolabel:
    enabled: true
    synthetic_multiplier: 0.5
    max_synthetic_rows: 20
    teacher_model: ridge
    min_neighbor_similarity: 0.0
    random_state: 17
""",
        encoding="utf-8",
    )

    run_augmentation(config_path)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["split_method"]) == {"heldout_group"}
    assert set(metrics["group_column"]) == {"product_key"}
    assert (metrics["eval_augmented_rows"] == 0).all()

    metadata = pd.read_csv(split_metadata_path)
    train_groups = set(metadata.loc[metadata["split"] == "train", "group_value"])
    test_groups = set(metadata.loc[metadata["split"] == "test", "group_value"])
    assert train_groups.isdisjoint(test_groups)


def test_baseline_runner_accepts_custom_heldout_group_column(tmp_path: Path) -> None:
    """Custom helper columns should survive cleaning and drive held-out splits."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "product_metrics.csv"
    split_metadata_path = tmp_path / "product_split_metadata.csv"
    config_path = tmp_path / "product.yaml"
    _write_group_fixture(data_path)
    config_path.write_text(
        _heldout_config(
            data_path,
            metrics_path,
            split_metadata_path,
            group_column="reactant_key",
        ),
        encoding="utf-8",
    )

    run_baseline(config_path)

    metrics = pd.read_csv(metrics_path)
    metadata = pd.read_csv(split_metadata_path)
    assert set(metrics["group_column"]) == {"reactant_key"}
    train_groups = set(metadata.loc[metadata["split"] == "train", "group_value"])
    test_groups = set(metadata.loc[metadata["split"] == "test", "group_value"])
    assert train_groups.isdisjoint(test_groups)
