"""Tests for supervised-AE latent interpolation augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.augmentation.latent_interpolation import (
    LatentInterpolationConfig,
    generate_latent_interpolations,
)
from bh_augmentation.representations.supervised_autoencoder import SupervisedAutoencoder
from bh_augmentation.run_supervised_ae_latent_interpolation import (
    run_supervised_ae_latent_interpolation,
)

pytest.importorskip("torch")


def test_latent_interpolation_shapes_and_train_parent_scope() -> None:
    z_train, y_train = _latent_data()
    result = generate_latent_interpolations(
        z_train,
        y_train,
        _ae_artifacts(latent_dim=z_train.shape[1]),
        _config(label_strategy="mixup_label", synthetic_multiplier=0.5),
    )

    assert result["z_synthetic"].shape == (6, z_train.shape[1])
    assert result["y_synthetic"].shape == (6,)
    candidates = result["candidate_df"]
    kept = candidates.loc[candidates["kept"]]
    assert not kept.empty
    assert kept["parent_i"].between(0, len(z_train) - 1).all()
    assert kept["parent_j"].between(0, len(z_train) - 1).all()


def test_mixup_label_is_between_parent_labels() -> None:
    z_train, y_train = _latent_data()
    result = generate_latent_interpolations(
        z_train,
        y_train,
        _ae_artifacts(latent_dim=z_train.shape[1]),
        _config(label_strategy="mixup_label", synthetic_multiplier=1.0),
    )

    kept = result["candidate_df"].loc[result["candidate_df"]["kept"]]
    parent_min = np.minimum(y_train[kept["parent_i"].to_numpy()], y_train[kept["parent_j"].to_numpy()])
    parent_max = np.maximum(y_train[kept["parent_i"].to_numpy()], y_train[kept["parent_j"].to_numpy()])
    labels = kept["y_synthetic"].to_numpy()
    assert np.all(labels >= parent_min)
    assert np.all(labels <= parent_max)


@pytest.mark.parametrize("label_strategy", ["ae_yield_head", "teacher_ensemble", "blended"])
def test_latent_interpolation_label_strategies_are_finite(label_strategy: str) -> None:
    z_train, y_train = _latent_data()
    result = generate_latent_interpolations(
        z_train,
        y_train,
        _ae_artifacts(latent_dim=z_train.shape[1]),
        _config(label_strategy=label_strategy, synthetic_multiplier=0.5),
    )

    assert len(result["y_synthetic"]) > 0
    assert np.isfinite(result["y_synthetic"]).all()
    if label_strategy in {"teacher_ensemble", "blended"}:
        kept = result["candidate_df"].loc[result["candidate_df"]["kept"]]
        assert np.isfinite(kept["teacher_mean"]).all()
        assert np.isfinite(kept["teacher_std"]).all()


def test_latent_interpolation_filters_reduce_or_preserve_candidates_and_clip_labels() -> None:
    z_train, y_train = _latent_data()
    y_train = y_train * 4.0 - 80.0
    artifacts = _ae_artifacts(latent_dim=z_train.shape[1])
    base = generate_latent_interpolations(
        z_train,
        y_train,
        artifacts,
        _config(label_strategy="teacher_ensemble", synthetic_multiplier=1.0),
    )
    std_filtered = generate_latent_interpolations(
        z_train,
        y_train,
        artifacts,
        _config(label_strategy="teacher_ensemble", synthetic_multiplier=1.0, max_teacher_std=0.0),
    )
    sim_filtered = generate_latent_interpolations(
        z_train,
        y_train,
        artifacts,
        _config(label_strategy="mixup_label", synthetic_multiplier=1.0, min_neighbor_similarity=1.01),
    )

    assert std_filtered["metadata"]["n_candidates_accepted"] <= base["metadata"]["n_candidates_accepted"]
    assert sim_filtered["metadata"]["n_candidates_accepted"] <= base["metadata"]["n_candidates_accepted"]
    assert np.all(base["y_synthetic"] >= 0.0)
    assert np.all(base["y_synthetic"] <= 100.0)


def test_supervised_ae_latent_interpolation_runner_writes_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "interpolation.yaml"
    output_dir = tmp_path / "results"
    _tiny_reactions(40).to_csv(data_path, index=False)
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
  - ridge
  - name: random_forest
    params:
      n_estimators: 5
metrics:
  - rmse
  - mae
  - r2
  - spearman
supervised_autoencoder:
  latent_dims: [4]
  hidden_dims: [8]
  dropout: 0.0
  reconstruction_weight: 1.0
  yield_weight: 1.0
  latent_l2_weight: 0.0
  learning_rate: 0.001
  weight_decay: 0.0
  batch_size: 4
  max_epochs: 2
  patience: 1
  internal_valid_size: 0.25
  device: cpu
latent_interpolation:
  enabled: true
  synthetic_multipliers: [0.5]
  n_neighbors: [3]
  alpha_min: 0.2
  alpha_max: 0.8
  alpha_distributions: ["uniform"]
  label_strategies: ["mixup_label", "teacher_ensemble"]
  teacher_models: ["ridge", "random_forest"]
  teacher_blend_weights: [0.5]
  min_neighbor_similarities: [null]
  max_teacher_stds: [null]
  clip_y_min: 0.0
  clip_y_max: 100.0
  candidates_per_real: 3
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
  audit_path: {output_dir / "synthetic_audit.csv"}
  interpolation_vs_original_rf_by_seed_path: {output_dir / "interpolation_vs_original_rf_by_seed.csv"}
  interpolation_vs_ae_only_by_seed_path: {output_dir / "interpolation_vs_ae_only_by_seed.csv"}
  interpolation_vs_original_rf_summary_path: {output_dir / "interpolation_vs_original_rf_summary.csv"}
  interpolation_vs_ae_only_summary_path: {output_dir / "interpolation_vs_ae_only_summary.csv"}
""",
        encoding="utf-8",
    )

    paths = run_supervised_ae_latent_interpolation(config_path)

    policy_metrics = pd.read_csv(paths["policy_metrics_path"])
    selected_metrics = pd.read_csv(paths["selected_policy_metrics_path"])
    audit = pd.read_csv(paths["audit_path"])
    assert paths["policy_metrics_path"].exists()
    assert paths["selected_policy_metrics_path"].exists()
    assert paths["selected_policies_path"].exists()
    assert paths["audit_path"].exists()
    assert "original_6144" in set(policy_metrics["representation"])
    assert "supervised_ae_4" in set(policy_metrics["representation"])
    assert "supervised_ae_interpolation_4" in set(policy_metrics["representation"])
    assert set(policy_metrics["split"]) == {"valid", "test"}
    assert set(selected_metrics["split"]) == {"valid", "test"}
    assert not audit["used_validation_or_test_parents"].any()
    assert (audit["n_synthetic_train"] > 0).any()


def _config(
    label_strategy: str,
    synthetic_multiplier: float = 0.5,
    min_neighbor_similarity: float | None = None,
    max_teacher_std: float | None = None,
) -> LatentInterpolationConfig:
    return LatentInterpolationConfig(
        synthetic_multiplier=synthetic_multiplier,
        n_neighbors=3,
        alpha_min=0.2,
        alpha_max=0.8,
        alpha_distribution="uniform",
        label_strategy=label_strategy,
        teacher_models=["ridge", "random_forest"],
        teacher_blend_weight=0.5,
        min_neighbor_similarity=min_neighbor_similarity,
        max_teacher_std=max_teacher_std,
        clip_y_min=0.0,
        clip_y_max=100.0,
        random_state=3,
        candidates_per_real=6,
    )


def _latent_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    z = rng.normal(size=(12, 5)).astype(np.float32)
    y = np.linspace(5.0, 95.0, len(z), dtype=np.float32)
    return z, y


def _ae_artifacts(latent_dim: int) -> dict[str, object]:
    return {
        "model": SupervisedAutoencoder(
            input_dim=latent_dim,
            latent_dim=latent_dim,
            hidden_dims=[],
            dropout=0.0,
        ),
        "y_mean": 50.0,
        "y_std": 10.0,
        "device": "cpu",
    }


def _tiny_reactions(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(n_rows)],
            "reaction_smiles": [
                f"CC{'C' * (index % 4)}Br.N>O>CC{'C' * (index % 4)}N"
                for index in range(n_rows)
            ],
            "yield": [float((index * 9) % 101) for index in range(n_rows)],
        }
    )

