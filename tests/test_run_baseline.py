"""Tests for the baseline experiment runner."""

from pathlib import Path

import pandas as pd

from bh_augmentation.run_baseline import run_baseline


def test_run_baseline_creates_metrics_csv(tmp_path: Path) -> None:
    """The baseline runner should work on a tiny local fixture dataset."""
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "baseline.yaml"
    metrics_path = tmp_path / "results" / "baseline_metrics.csv"

    pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(10)],
            "reaction_smiles": [
                f"CC{'C' * (index % 3)}Br.N.O.C>>CC{'C' * (index % 3)}N"
                for index in range(10)
            ],
            "product_key": [f"product_{index % 3}" for index in range(10)],
            "reactant_key": [f"reactant_{index % 4}" for index in range(10)],
            "yield": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    ).to_csv(data_path, index=False)

    config_path.write_text(
        f"""
seed: 7
dataset:
  name: synthetic_bh
  path: {data_path}
splits:
  method: random
  train_size: 0.6
  valid_size: 0.2
  test_size: 0.2
features:
  kind: reaction_morgan_sum
  n_bits: 8
models:
  - ridge
metrics:
  - rmse
  - mae
output:
  metrics_path: {metrics_path}
""",
        encoding="utf-8",
    )

    output_path = run_baseline(config_path)

    assert output_path == metrics_path
    assert metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert list(metrics.columns) == [
        "train_fraction",
        "split_method",
        "group_column",
        "feature_kind",
        "n_features",
        "model",
        "split",
        "metric",
        "value",
    ]
    assert metrics["train_fraction"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert set(metrics["split_method"]) == {"random"}
    assert set(metrics["split"]) == {"valid", "test"}
    assert set(metrics["metric"]) == {"rmse", "mae"}
    assert set(metrics["model"]) == {"ridge"}
    assert set(metrics["feature_kind"]) == {"reaction_morgan_sum"}
    assert set(metrics["n_features"]) == {8}
    assert len(metrics) == 4


def test_run_baseline_compares_reaction_feature_modes(tmp_path: Path) -> None:
    """The baseline runner should compare feature modes on shared splits."""
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "baseline_compare.yaml"
    metrics_path = tmp_path / "results" / "baseline_metrics.csv"

    pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(12)],
            "reaction_smiles": [
                f"CC{'C' * (index % 3)}Br.N>O>CC{'C' * (index % 3)}N"
                for index in range(12)
            ],
            "yield": [float(index * 8) for index in range(12)],
        }
    ).to_csv(data_path, index=False)

    config_path.write_text(
        f"""
seed: 7
dataset:
  path: {data_path}
splits:
  method: random
  train_size: 0.6
  valid_size: 0.2
  test_size: 0.2
features:
  compare:
    - kind: reaction_morgan_sum
      n_bits: 8
    - kind: reaction_role_concat
      n_bits: 8
    - kind: reaction_role_concat_delta
      n_bits: 8
models:
  - ridge
metrics:
  - rmse
output:
  metrics_path: {metrics_path}
""",
        encoding="utf-8",
    )

    run_baseline(config_path)
    metrics = pd.read_csv(metrics_path)

    assert set(metrics["feature_kind"]) == {
        "reaction_morgan_sum",
        "reaction_role_concat",
        "reaction_role_concat_delta",
    }
    assert set(zip(metrics["feature_kind"], metrics["n_features"])) == {
        ("reaction_morgan_sum", 8),
        ("reaction_role_concat", 24),
        ("reaction_role_concat_delta", 32),
    }
    assert len(metrics) == 6
