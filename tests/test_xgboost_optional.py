"""Optional XGBoost integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.run_condition_transfer import run_condition_transfer


def test_condition_transfer_tiny_xgboost_config_runs_when_installed(tmp_path: Path) -> None:
    """A runner config requesting only XGBoost should write XGBoost metric rows."""
    pytest.importorskip("xgboost")

    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "condition_transfer_xgboost.yaml"
    output_dir = tmp_path / "results"
    _tiny_reactions(24).to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 0
seeds: [0]
dataset:
  path: {data_path}
splits:
  method: random
  train_size: 0.6
  valid_size: 0.2
  test_size: 0.2
low_data:
  enabled: true
  train_fractions: [0.5]
features:
  kind: reaction_role_concat
  n_bits: 8
  radius: 2
models:
  - xgboost
metrics:
  - rmse
  - mae
condition_transfer:
  enabled: true
  donor_strategies: [random]
  label_strategies: [average_label]
  synthetic_multipliers: [0.5]
  n_neighbors: [3]
  min_similarities: [null]
  max_teacher_stds: [null]
  high_yield_threshold: 70.0
  clip_y_min: 0.0
  clip_y_max: 100.0
  candidates_per_real: 3
  teacher_models: [ridge, random_forest]
selection:
  split: valid
  metric: rmse
  lower_is_better: true
output:
  directory: {output_dir}
  policy_metrics_path: {output_dir / "policy_metrics.csv"}
  selected_policies_path: {output_dir / "selected_policies.csv"}
  selected_policy_metrics_path: {output_dir / "selected_policy_metrics.csv"}
  summary_path: {output_dir / "summary.csv"}
  synthetic_audit_path: {output_dir / "synthetic_audit.csv"}
  condition_transfer_vs_original_rf_by_seed_path: {output_dir / "condition_transfer_vs_original_rf_by_seed.csv"}
  condition_transfer_vs_original_rf_summary_path: {output_dir / "condition_transfer_vs_original_rf_summary.csv"}
  condition_transfer_vs_real_only_same_model_by_seed_path: {output_dir / "condition_transfer_vs_real_only_same_model_by_seed.csv"}
  condition_transfer_vs_real_only_same_model_summary_path: {output_dir / "condition_transfer_vs_real_only_same_model_summary.csv"}
""",
        encoding="utf-8",
    )

    paths = run_condition_transfer(config_path)

    metrics = pd.read_csv(paths["policy_metrics_path"])
    assert set(metrics["model"]) == {"xgboost"}
    assert {"original_6144", "condition_transfer"} <= set(metrics["representation"])
    assert set(metrics["split"]) == {"valid", "test"}


def _tiny_reactions(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(n_rows)],
            "reaction_smiles": [
                f"CC{'C' * (index % 4)}Br.N.O.Cl>>CC{'C' * (index % 4)}N"
                for index in range(n_rows)
            ],
            "yield": [float((index * 11) % 101) for index in range(n_rows)],
        }
    )
