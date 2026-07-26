"""Tests for utility-guided feature-space generator augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.augmentation.utility_guided_feature_gan import (
    build_condition_matrix,
    compute_batch_reward,
    fit_feature_transform,
    run_utility_guided_feature_gan_augmentation,
    score_and_filter_feature_candidates,
    select_elite_batches,
    update_yield_bin_policy,
)
from bh_augmentation.evaluation.metrics import rmse
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.run_augmentation import run_augmentation


@pytest.fixture(autouse=True)
def _require_torch() -> None:
    pytest.importorskip("torch")


def test_transform_and_generated_output_shapes_with_clipped_labels() -> None:
    df = _tiny_reactions(18)
    feature_config = {"kind": "reaction_role_concat", "n_bits": 16, "radius": 2}
    config = _tiny_gan_config(
        {
            "filters": {"enabled": False},
            "synthetic_multiplier": 0.5,
            "max_synthetic_rows": 8,
        }
    )

    result, metadata = run_utility_guided_feature_gan_augmentation(
        df,
        df.iloc[:3],
        feature_config,
        "ridge",
        config,
        random_state=7,
    )

    assert result["X_train_augmented"].shape[1] == 48
    assert result["X_train_augmented"].shape[0] > len(df)
    assert result["y_train_augmented"].min() >= 0.0
    assert result["y_train_augmented"].max() <= 100.0
    assert metadata["n_generator_train"] + metadata["n_reward_valid"] == len(df)
    assert metadata["n_synthetic_train"] > 0
    audit = result["candidate_audit"]
    assert len(audit) == metadata["n_candidates_generated"]
    assert audit["identity_classification"].eq("feature_space_nonchemical").all()
    assert audit["source_id_semantics"].eq(
        "nearest_training_support_not_generator_parent"
    ).all()
    assert not audit["chemical_parse_valid"].any()
    assert audit.loc[audit["accepted"], "feature_hash"].is_unique
    assert metadata["development_control_only"] is True


def test_feature_candidate_audit_deduplicates_without_claiming_chemistry() -> None:
    X_real = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y_real = np.array([20.0, 80.0], dtype=np.float32)
    teacher = train_model(get_model("ridge"), X_real, y_real)
    candidates = pd.DataFrame(
        {
            "feature_vector": [
                np.array([0.4, 0.6], dtype=np.float32),
                np.array([0.4, 0.6], dtype=np.float32),
                np.array([0.7, 0.3], dtype=np.float32),
            ],
            "yield_bin": [0, 0, 1],
        }
    )

    accepted, audit = score_and_filter_feature_candidates(
        candidates,
        X_real,
        [teacher],
        {"filters": {"enabled": False}},
        source_row_ids=["row-a", "row-b"],
        return_audit=True,
    )

    assert len(accepted) == 2
    assert audit["feature_duplicate_synthetic"].tolist() == [False, True, False]
    assert audit["rejection_reason"].tolist() == [
        None,
        "feature_duplicate_synthetic",
        None,
    ]
    assert audit["source_row_id"].isin(["row-a", "row-b"]).all()
    assert audit["donor_row_id"].isna().all()
    assert audit["canonical_reaction_key"].isna().all()
    assert audit["canonical_reaction_hash"].isna().all()
    assert not audit["chemical_parse_valid"].any()
    assert audit["identity_classification"].eq("feature_space_nonchemical").all()
    assert not audit["scientific_candidate_eligible"].any()
    assert accepted["feature_hash"].is_unique


def test_latent_mode_trains_student_on_generated_latent_vectors() -> None:
    df = _tiny_reactions(18)
    feature_config = {"kind": "reaction_role_concat", "n_bits": 16, "radius": 2}
    config = _tiny_gan_config(
        {
            "representation": {
                "use_dim_reduction": True,
                "method": "truncated_svd",
                "n_components": 6,
                "reconstruct_to_original_dim": False,
                "train_student_in_latent_space": True,
            },
            "filters": {"enabled": False},
            "synthetic_multiplier": 0.5,
            "max_synthetic_rows": 8,
        }
    )

    result, metadata = run_utility_guided_feature_gan_augmentation(
        df,
        df.iloc[:3],
        feature_config,
        "ridge",
        config,
        random_state=7,
    )

    assert metadata["train_student_in_latent_space"] is True
    assert metadata["original_n_features"] == 48
    assert result["X_train_augmented"].shape[1] == metadata["latent_dim"]
    assert metadata["student_n_features"] == metadata["latent_dim"]
    synthetic = result["synthetic_metadata"]
    assert not synthetic.empty
    assert len(synthetic.loc[0, "feature_vector"]) == metadata["latent_dim"]


def test_svd_transform_is_fit_on_train_features_only() -> None:
    X_train = np.eye(8, 12, dtype=np.float32)
    X_heldout = np.ones((2, 12), dtype=np.float32)

    transform = fit_feature_transform(
        X_train,
        {"use_dim_reduction": True, "method": "truncated_svd", "n_components": 4},
        random_state=1,
    )

    assert transform.transformer is not None
    assert transform.transform(X_train).shape == (8, 4)
    assert transform.inverse_transform(transform.transform(X_heldout)).shape == X_heldout.shape
    assert transform.transformer.n_components == 4


def test_batch_reward_is_baseline_minus_augmented_rmse() -> None:
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    X_valid = np.array([[4.0], [5.0]], dtype=np.float32)
    y_valid = np.array([4.0, 5.0], dtype=np.float32)
    synthetic = pd.DataFrame(
        {
            "feature_vector": [np.array([4.0], dtype=np.float32)],
            "yield": [4.0],
        }
    )
    baseline_model = train_model(get_model("ridge"), X_train, y_train)
    baseline = rmse(y_valid, predict_model(baseline_model, X_valid))

    reward = compute_batch_reward(X_train, y_train, X_valid, y_valid, synthetic, baseline)

    X_aug = np.vstack([X_train, np.array([[4.0]], dtype=np.float32)])
    y_aug = np.concatenate([y_train, np.array([4.0], dtype=np.float32)])
    augmented_model = train_model(get_model("ridge"), X_aug, y_aug)
    expected = baseline - rmse(y_valid, predict_model(augmented_model, X_valid))
    assert np.isclose(reward, expected)


def test_cross_entropy_elites_update_yield_bin_policy() -> None:
    batches = [
        (0.1, pd.DataFrame({"yield_bin": [0, 0, 1]})),
        (0.5, pd.DataFrame({"yield_bin": [2, 2, 2]})),
        (0.3, pd.DataFrame({"yield_bin": [1, 2, 2]})),
    ]

    elites = select_elite_batches(batches, elite_fraction=0.34)
    probabilities = update_yield_bin_policy(elites, n_bins=3)

    assert len(elites) == 2
    assert probabilities[2] > probabilities[0]
    assert np.isclose(probabilities.sum(), 1.0)


def test_zero_accepted_candidates_returns_real_training_only() -> None:
    df = _tiny_reactions(14)
    config = _tiny_gan_config(
        {
            "filters": {
                "enabled": True,
                "min_nearest_similarity": 1.1,
                "remove_near_duplicates": False,
                "max_teacher_std": 100.0,
                "max_prediction_range": 200.0,
            },
            "synthetic_multiplier": 1.0,
        }
    )

    result, metadata = run_utility_guided_feature_gan_augmentation(
        df,
        df.iloc[:3],
        {"kind": "reaction_morgan_sum", "n_bits": 16, "radius": 2},
        "ridge",
        config,
        random_state=2,
    )

    assert result["X_train_augmented"].shape[0] == len(df)
    assert metadata["n_synthetic_train"] == 0
    assert metadata["n_candidates_accepted"] == 0


def test_conditioning_supports_train_only_product_and_reactant_keys() -> None:
    df = _tiny_reactions(6)
    y = df["yield"].to_numpy(dtype=float)

    matrix, info = build_condition_matrix(
        df,
        y,
        {
            "n_yield_bins": 5,
            "include_yield_value": True,
            "include_product_key": True,
            "include_reactant_key": True,
        },
    )

    assert matrix.shape[0] == len(df)
    assert set(info["categories"]) == {"product_key", "reactant_key"}
    assert matrix.shape[1] == 5 + 1 + len(info["categories"]["product_key"]) + len(info["categories"]["reactant_key"])


def test_utility_gan_runner_writes_policy_search_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "utility_gan.yaml"
    search_path = tmp_path / "policy_search_metrics.csv"
    selected_path = tmp_path / "selected_policies.csv"
    metrics_path = tmp_path / "selected_policy_metrics.csv"
    split_path = tmp_path / "split_metadata.csv"
    _tiny_reactions(22).to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 5
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
  utility_guided_feature_gan:
    enabled: true
    synthetic_multiplier: 0.25
    max_synthetic_rows: 8
    random_state: 5
    max_effective_rounds: 1
    representation:
      use_dim_reduction: true
      method: truncated_svd
      n_components: 4
    conditioning:
      n_yield_bins: 3
      include_yield_value: true
    model:
      noise_dim: 4
      hidden_dim: 8
      n_epochs: 1
      max_effective_epochs: 1
      batch_size: 4
      critic_steps: 1
      learning_rate: 0.001
      gradient_penalty: 1.0
    pseudo_label:
      teachers:
        - model: ridge
          random_state: 0
    filters:
      enabled: false
    utility_guidance:
      inner_valid_size: 0.25
      n_rounds: 1
      candidates_per_round: 12
      batch_sizes: [4]
      elite_fraction: 0.5
augmentation_search:
  enabled: true
  method: utility_guided_feature_gan
  selection_metric: rmse
  lower_is_better: true
  synthetic_multipliers: [0.25, 0.5]
  noise_dims: [4]
  hidden_dims: [8]
  svd_components: [4]
  n_epochs: [1]
  batch_sizes: [4]
  learning_rates: [0.001]
  gradient_penalties: [1.0]
  utility_rounds: [1]
  random_states: [0]
  max_synthetic_rows: 8
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
    search = pd.read_csv(search_path)
    selected = pd.read_csv(selected_path)
    metrics = pd.read_csv(metrics_path)
    assert search["policy_id"].nunique() == 2
    assert set(search["split"]) == {"valid"}
    assert len(selected) == 1
    assert set(metrics["split"]) == {"valid", "test"}
    assert metrics["selected_policy"].all()
    assert {
        "n_generator_train",
        "n_reward_valid",
        "n_candidates_generated",
        "n_candidates_accepted",
        "best_inner_reward",
        "outer_valid_rmse",
        "noise_dim",
        "hidden_dim",
        "svd_components",
        "train_student_in_latent_space",
        "latent_dim",
        "original_n_features",
        "student_n_features",
    }.issubset(metrics.columns)


def test_latent_mode_runner_writes_student_latent_dimensions(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "utility_gan_latent.yaml"
    search_path = tmp_path / "policy_search_metrics.csv"
    selected_path = tmp_path / "selected_policies.csv"
    metrics_path = tmp_path / "selected_policy_metrics.csv"
    _tiny_reactions(24).to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 6
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
  utility_guided_feature_gan:
    enabled: true
    synthetic_multiplier: 0.25
    max_synthetic_rows: 8
    random_state: 6
    max_effective_rounds: 1
    representation:
      use_dim_reduction: true
      method: truncated_svd
      n_components: 4
      reconstruct_to_original_dim: false
      train_student_in_latent_space: true
    conditioning:
      n_yield_bins: 3
      include_yield_value: true
    model:
      noise_dim: 4
      hidden_dim: 8
      n_epochs: 1
      max_effective_epochs: 1
      batch_size: 4
      critic_steps: 1
      learning_rate: 0.001
      gradient_penalty: 1.0
    pseudo_label:
      teachers:
        - model: ridge
          random_state: 0
    filters:
      enabled: false
    utility_guidance:
      inner_valid_size: 0.25
      n_rounds: 1
      candidates_per_round: 12
      batch_sizes: [4]
      elite_fraction: 0.5
augmentation_search:
  enabled: true
  method: utility_guided_feature_gan
  selection_metric: rmse
  lower_is_better: true
  synthetic_multipliers: [0.25, 0.5]
  noise_dims: [4]
  hidden_dims: [8]
  svd_components: [4]
  n_epochs: [1]
  batch_sizes: [4]
  learning_rates: [0.001]
  gradient_penalties: [1.0]
  utility_rounds: [1]
  random_states: [0]
  max_synthetic_rows: 8
output:
  metrics_path: {metrics_path}
  search_metrics_path: {search_path}
  selected_policies_path: {selected_path}
""",
        encoding="utf-8",
    )

    run_augmentation(config_path)

    metrics = pd.read_csv(metrics_path)
    search = pd.read_csv(search_path)
    assert search["policy_id"].nunique() == 2
    assert metrics["train_student_in_latent_space"].all()
    assert metrics["student_n_features"].eq(metrics["latent_dim"]).all()
    assert metrics["student_n_features"].lt(metrics["original_n_features"]).all()
    assert set(metrics["split"]) == {"valid", "test"}


def _tiny_reactions(n_rows: int) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        chain = "C" * (index % 5 + 2)
        condition = "O" if index % 2 else "Cl"
        product = f"{chain}N"
        rows.append(
            {
                "reaction_id": f"r{index}",
                "reaction_smiles": f"{chain}Br.N.{condition}.C>>{product}",
                "product_key": product,
                "reactant_key": f"{chain}Br|N",
                "yield": float((index * 13 + 7) % 100),
            }
        )
    return pd.DataFrame(rows)


def _tiny_gan_config(overrides: dict[str, object] | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "synthetic_multiplier": 0.5,
        "max_synthetic_rows": 6,
        "max_effective_rounds": 1,
        "representation": {
            "use_dim_reduction": True,
            "method": "truncated_svd",
            "n_components": 6,
        },
        "conditioning": {
            "n_yield_bins": 3,
            "include_yield_value": True,
        },
        "model": {
            "noise_dim": 4,
            "hidden_dim": 8,
            "n_epochs": 1,
            "max_effective_epochs": 1,
            "batch_size": 4,
            "critic_steps": 1,
            "learning_rate": 0.001,
            "gradient_penalty": 1.0,
        },
        "pseudo_label": {
            "clip_min": 0.0,
            "clip_max": 100.0,
            "teachers": [{"model": "ridge", "random_state": 0}],
        },
        "filters": {
            "enabled": False,
        },
        "utility_guidance": {
            "inner_valid_size": 0.25,
            "n_rounds": 1,
            "candidates_per_round": 12,
            "batch_sizes": [4],
            "elite_fraction": 0.5,
        },
        "diversity": {"max_synthetic_per_nearest_real": 5},
    }
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[arg-type]
        else:
            target[key] = value
