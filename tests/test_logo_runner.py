"""Tests for leave-one-group-out baseline experiments."""

from pathlib import Path

import pandas as pd

from bh_augmentation.run_baseline import run_baseline


def test_logo_runner_saves_fold_and_summary_metrics(tmp_path: Path) -> None:
    """LOGO should record each held-out group and aggregate mean/std metrics."""
    data_path = tmp_path / "stress.csv"
    config_path = tmp_path / "logo.yaml"
    metrics_path = tmp_path / "fold_metrics.csv"
    summary_path = tmp_path / "summary_metrics.csv"
    metadata_path = tmp_path / "split_metadata.csv"

    rows = []
    for group_index in range(3):
        for repeat in range(6):
            rows.append(
                {
                    "reaction_id": f"rxn_{group_index}_{repeat}",
                    "reaction_smiles": f"CC{'C' * group_index}Br.N>>CC{'C' * group_index}N",
                    "product_key": f"product_{group_index}",
                    "yield": float(group_index * 20 + repeat),
                }
            )
    pd.DataFrame(rows).to_csv(data_path, index=False)

    config_path.write_text(
        f"""
seed: 7
dataset:
  path: {data_path}
splits:
  method: leave_one_group_out
  group_column: product_key
  valid_fraction: 0.1
features:
  compare:
    - kind: reaction_morgan_sum
      n_bits: 8
    - kind: reaction_role_concat
      n_bits: 8
models:
  include:
    - ridge
metrics:
  - rmse
output:
  metrics_path: {metrics_path}
  summary_metrics_path: {summary_path}
  split_metadata_path: {metadata_path}
""",
        encoding="utf-8",
    )

    run_baseline(config_path)

    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    metadata = pd.read_csv(metadata_path)

    assert set(metrics["heldout_group_value"]) == {
        "product_0",
        "product_1",
        "product_2",
    }
    assert len(metrics) == 12
    assert {"mean", "std"}.issubset(summary.columns)
    assert len(summary) == 4

    for heldout in metrics["heldout_group_value"].unique():
        fold = metadata[metadata["heldout_group_value"] == heldout]
        assert set(fold.loc[fold["split"] == "test", "group_value"]) == {heldout}
        assert heldout not in set(fold.loc[fold["split"] == "train", "group_value"])
        assert heldout not in set(fold.loc[fold["split"] == "valid", "group_value"])
