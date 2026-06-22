"""Tests for low-data experiment runner support."""

from pathlib import Path

import pandas as pd

from bh_augmentation.run_augmentation import run_augmentation
from bh_augmentation.run_baseline import run_baseline


def _write_fixture_csv(path: Path, n_rows: int = 20) -> None:
    pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(n_rows)],
            "reaction_smiles": [
                f"CC{'C' * (index % 4)}Br.N."
                f"{'O' if index % 2 else 'Cl'}.{'C' if index % 3 else 'P'}"
                f">>CC{'C' * (index % 4)}N"
                for index in range(n_rows)
            ],
            "product_key": [f"product_{index % 4}" for index in range(n_rows)],
            "reactant_key": [f"reactant_{index % 5}" for index in range(n_rows)],
            "yield": [float((index * 7) % 100) for index in range(n_rows)],
        }
    ).to_csv(path, index=False)


def _base_config(data_path: Path, metrics_path: Path) -> str:
    return f"""
seed: 13
dataset:
  name: synthetic_bh
  path: {data_path}
splits:
  method: random
  train_size: 0.6
  valid_size: 0.2
  test_size: 0.2
low_data:
  enabled: true
  train_fractions:
    - 0.5
    - 1.0
features:
  kind: reaction_morgan_sum
  n_bits: 8
models:
  - ridge
metrics:
  - rmse
output:
  metrics_path: {metrics_path}
"""


def test_baseline_runner_writes_metrics_for_each_low_data_fraction(tmp_path: Path) -> None:
    """Baseline metrics should include one row per split for each train fraction."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "baseline_metrics.csv"
    config_path = tmp_path / "baseline.yaml"
    _write_fixture_csv(data_path)
    config_path.write_text(_base_config(data_path, metrics_path), encoding="utf-8")

    run_baseline(config_path)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["train_fraction"]) == {0.5, 1.0}
    assert set(metrics["split_method"]) == {"random"}
    assert set(metrics["model"]) == {"ridge"}
    assert set(metrics["split"]) == {"valid", "test"}
    assert len(metrics) == 4


def test_augmentation_runner_writes_metrics_for_each_low_data_fraction(
    tmp_path: Path,
) -> None:
    """Augmentation runner should preserve real validation/test across fractions."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "safe_aug_metrics.csv"
    config_path = tmp_path / "augmentation.yaml"
    _write_fixture_csv(data_path)
    config_path.write_text(
        _base_config(data_path, metrics_path)
        + """
augmentation:
  condition_recombine_pseudolabel:
    enabled: true
    synthetic_multiplier: 0.5
    max_synthetic_rows: 20
    teacher_model: ridge
    min_neighbor_similarity: 0.0
    random_state: 13
""",
        encoding="utf-8",
    )

    run_augmentation(config_path)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["train_fraction"]) == {0.5, 1.0}
    assert set(metrics["split_method"]) == {"random"}
    assert set(metrics["augmentation"]) == {
        "none",
        "condition_recombine_pseudolabel",
    }
    assert set(metrics["split"]) == {"valid", "test"}
    assert (metrics["eval_augmented_rows"] == 0).all()
    assert len(metrics) == 8
