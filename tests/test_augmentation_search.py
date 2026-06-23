"""Tests for validation-selected augmentation policy search."""

from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_recombine import (
    assign_synthetic_sample_weights,
)
from bh_augmentation.augmentation.search import (
    build_augmentation_policy_grid,
    select_best_policy,
)
from bh_augmentation.run_augmentation import run_augmentation


def test_build_policy_grid_has_expected_size_and_unique_ids() -> None:
    policies = build_augmentation_policy_grid(
        {
            "synthetic_multipliers": [0.25, 0.5, 1.0],
            "min_neighbor_similarities": [0.7, 0.8, 0.9],
            "teacher_models": ["ridge", "random_forest"],
            "random_states": [0, 1, 2, 3, 4],
            "max_synthetic_rows": 3000,
        }
    )

    assert len(policies) == 90
    assert len({policy["policy_id"] for policy in policies}) == 90


def test_large_policy_grid_enforces_similarity_safety_constraints() -> None:
    policies = build_augmentation_policy_grid(
        {
            "synthetic_multipliers": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
            "min_neighbor_similarities": [0.7, 0.8, 0.85, 0.9, 0.95],
            "teacher_models": ["ridge", "random_forest"],
            "random_states": [0, 1, 2, 3, 4],
            "max_synthetic_rows": 6000,
        }
    )

    assert len(policies) == 270
    assert {policy["synthetic_multiplier"] for policy in policies} == {
        0.25,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
    }
    assert all(
        policy["min_neighbor_similarity"] >= 0.8
        for policy in policies
        if policy["synthetic_multiplier"] >= 2.0
    )
    assert all(
        policy["min_neighbor_similarity"] >= 0.85
        for policy in policies
        if policy["synthetic_multiplier"] >= 3.0
    )


def test_ensemble_policy_grid_expands_uncertainty_parameters() -> None:
    policies = build_augmentation_policy_grid(
        {
            "method": "condition_recombine_ensemble_filter",
            "synthetic_multipliers": [0.5, 1.0, 1.5],
            "min_neighbor_similarities": [0.75, 0.8, 0.85, 0.9],
            "max_teacher_stds": [8.0, 12.0, 16.0],
            "max_prediction_ranges": [25.0, 35.0, 50.0],
            "random_states": [0, 1, 2, 3, 4],
        }
    )

    assert len(policies) == 540
    assert len({policy["policy_id"] for policy in policies}) == 540
    assert all("__std_" in policy["policy_id"] for policy in policies)
    assert all("__range_" in policy["policy_id"] for policy in policies)


def test_utility_guided_feature_gan_policy_grid_expands_parameters() -> None:
    policies = build_augmentation_policy_grid(
        {
            "method": "utility_guided_feature_gan",
            "synthetic_multipliers": [0.25, 0.5, 1.0],
            "noise_dims": [32, 64],
            "hidden_dims": [128, 256],
            "svd_components": [64, 128],
            "n_epochs": [100, 200],
            "batch_sizes": [32],
            "learning_rates": [0.0002],
            "gradient_penalties": [10.0],
            "utility_rounds": [3, 5],
            "random_states": [0, 1, 2],
            "max_synthetic_rows": 1000,
            "utility_config": {"filters": {"min_nearest_similarity": 0.6}},
        }
    )

    assert len(policies) == 288
    assert len({policy["policy_id"] for policy in policies}) == 288
    assert all("__noise_" in policy["policy_id"] for policy in policies)
    assert all("__svd_" in policy["policy_id"] for policy in policies)
    assert all(policy["augmentation_method"] == "utility_guided_feature_gan" for policy in policies)


def test_similarity_weighting_assigns_real_and_clipped_synthetic_weights() -> None:
    data = pd.DataFrame(
        {
            "is_synthetic": [False, True, True, True],
            "nearest_train_similarity": [np.nan, 0.1, 0.6, 0.95],
        }
    )

    weighted = assign_synthetic_sample_weights(
        data,
        {
            "enabled": True,
            "real_weight": 1.0,
            "synthetic_weight_mode": "nearest_similarity",
            "min_synthetic_weight": 0.25,
            "max_synthetic_weight": 0.85,
        },
    )

    assert weighted["sample_weight"].tolist() == [1.0, 0.25, 0.6, 0.85]


def test_select_best_policy_uses_validation_rmse_and_deterministic_ties() -> None:
    metrics = pd.DataFrame(
        [
            _selection_row("p1", "valid", 5.0, 0.5, 0.8, "ridge", 1),
            _selection_row("p2", "valid", 4.0, 1.0, 0.9, "ridge", 0),
            _selection_row("test_winner", "test", 0.0, 0.25, 0.9, "ridge", 0),
        ]
    )
    assert select_best_policy(metrics)["policy_id"] == "p2"

    tied = pd.DataFrame(
        [
            _selection_row("larger", "valid", 4.0, 0.5, 0.9, "ridge", 0),
            _selection_row("lower_sim", "valid", 4.0, 0.25, 0.7, "ridge", 0),
            _selection_row("winner", "valid", 4.0, 0.25, 0.9, "ridge", 0),
            _selection_row("later_teacher", "valid", 4.0, 0.25, 0.9, "random_forest", 0),
        ]
    )
    assert select_best_policy(tied)["policy_id"] == "later_teacher"


def test_policy_search_runner_writes_validation_selected_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "search.yaml"
    search_path = tmp_path / "policy_search_metrics.csv"
    selected_path = tmp_path / "selected_policies.csv"
    metrics_path = tmp_path / "selected_policy_metrics.csv"
    split_path = tmp_path / "split_metadata.csv"
    pd.DataFrame(
        {
            "reaction_id": [f"r{index}" for index in range(24)],
            "reaction_smiles": [
                f"{'C' * (index + 2)}Br.N.{'O' if index % 2 else 'Cl'}.C"
                f">>{'C' * (index + 2)}N"
                for index in range(24)
            ],
            "yield": [float((index * 11) % 100) for index in range(24)],
        }
    ).to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 9
dataset:
  path: {data_path}
splits:
  method: random
  train_size: 0.6
  valid_size: 0.2
  test_size: 0.2
low_data:
  enabled: true
  train_fractions: [0.5, 1.0]
features:
  kind: reaction_morgan_sum
  n_bits: 8
models:
  - ridge
metrics:
  - rmse
augmentation_search:
  enabled: true
  selection_metric: rmse
  lower_is_better: true
  synthetic_multipliers: [0.5, 1.0]
  min_neighbor_similarities: [0.0]
  teacher_models: [ridge]
  random_states: [1, 2]
  max_synthetic_rows: 20
  synthetic_weighting:
    enabled: true
    real_weight: 1.0
    synthetic_weight_mode: nearest_similarity
    min_synthetic_weight: 0.25
    max_synthetic_weight: 0.85
output:
  metrics_path: {metrics_path}
  search_metrics_path: {search_path}
  selected_policies_path: {selected_path}
  split_metadata_path: {split_path}
""",
        encoding="utf-8",
    )

    output = run_augmentation(config_path)

    assert output == metrics_path
    assert search_path.exists()
    assert selected_path.exists()
    assert metrics_path.exists()
    search_metrics = pd.read_csv(search_path)
    selected = pd.read_csv(selected_path)
    selected_metrics = pd.read_csv(metrics_path)
    assert search_metrics["policy_id"].nunique() == 4
    assert set(search_metrics["split"]) == {"valid"}
    assert search_metrics.groupby("train_fraction")["selected_policy"].sum().eq(1).all()
    assert len(selected) == 2
    assert set(selected["train_fraction"]) == {0.5, 1.0}
    assert selected_metrics["selected_policy"].all()
    assert set(selected_metrics["split"]) == {"valid", "test"}
    assert selected_metrics["synthetic_weighting_enabled"].all()
    assert selected_metrics["mean_synthetic_weight"].between(0.25, 0.85).all()


def test_ensemble_policy_search_runner_writes_selected_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "ensemble.yaml"
    search_path = tmp_path / "search.csv"
    selected_path = tmp_path / "selected.csv"
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "reaction_id": [f"r{index}" for index in range(20)],
            "reaction_smiles": [
                f"{'C' * (index + 2)}Br.N.{'O' if index % 2 else 'Cl'}.C"
                f">>{'C' * (index + 2)}N"
                for index in range(20)
            ],
            "yield": [float((index * 9) % 100) for index in range(20)],
        }
    ).to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 3
dataset:
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
augmentation:
  condition_recombine_ensemble_filter:
    enabled: true
    synthetic_multiplier: 0.5
    max_synthetic_rows: 20
    min_neighbor_similarity: 0.0
    teachers:
      - model: ridge
        random_state: 0
      - model: random_forest
        random_state: 0
    uncertainty_filter:
      enabled: true
      max_teacher_std: 100.0
      max_prediction_range: 200.0
    pseudo_label:
      mode: ensemble_mean
augmentation_search:
  enabled: true
  method: condition_recombine_ensemble_filter
  synthetic_multipliers: [0.5]
  min_neighbor_similarities: [0.0]
  max_teacher_stds: [50.0, 100.0]
  max_prediction_ranges: [200.0]
  random_states: [1]
output:
  metrics_path: {metrics_path}
  search_metrics_path: {search_path}
  selected_policies_path: {selected_path}
""",
        encoding="utf-8",
    )

    run_augmentation(config_path)
    search = pd.read_csv(search_path)
    selected = pd.read_csv(selected_path)
    metrics = pd.read_csv(metrics_path)

    assert search["policy_id"].nunique() == 2
    assert set(search["split"]) == {"valid"}
    assert len(selected) == 1
    assert set(metrics["split"]) == {"valid", "test"}
    assert metrics["selected_policy"].all()
    assert {
        "n_candidates_generated",
        "acceptance_rate",
        "mean_teacher_std",
        "mean_prediction_range",
        "mean_nearest_train_similarity",
    }.issubset(metrics.columns)


def _selection_row(
    policy_id: str,
    split: str,
    value: float,
    multiplier: float,
    similarity: float,
    teacher: str,
    random_state: int,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "split": split,
        "metric": "rmse",
        "value": value,
        "synthetic_multiplier": multiplier,
        "min_neighbor_similarity": similarity,
        "teacher_model": teacher,
        "augmentation_random_state": random_state,
    }
