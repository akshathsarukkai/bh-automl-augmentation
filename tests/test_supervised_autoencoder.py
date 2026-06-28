"""Tests for supervised autoencoder representation learning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bh_augmentation.representations.supervised_autoencoder import (
    SupervisedAEConfig,
    SupervisedAutoencoder,
    encode_with_supervised_autoencoder,
    fit_supervised_autoencoder,
    predict_yield_with_supervised_autoencoder,
)
from bh_augmentation.run_supervised_ae_latent_baseline import (
    fit_svd_on_train_only,
    make_internal_ae_split,
    run_supervised_ae_latent_baseline,
)

pytest.importorskip("torch")


def test_supervised_autoencoder_forward_shapes() -> None:
    model = SupervisedAutoencoder(input_dim=12, latent_dim=4, hidden_dims=[8], dropout=0.0)
    X = torch.randn(5, 12)

    z, x_reconstructed, y_pred_scaled = model(X)

    assert z.shape == (5, 4)
    assert x_reconstructed.shape == (5, 12)
    assert y_pred_scaled.shape == (5,)


def test_supervised_autoencoder_training_returns_artifacts() -> None:
    X, y = _synthetic_matrix(n_rows=18, n_features=10)
    config = SupervisedAEConfig(
        input_dim=10,
        latent_dim=3,
        hidden_dims=[8],
        batch_size=6,
        max_epochs=4,
        patience=2,
        random_state=7,
        device="cpu",
    )

    artifacts = fit_supervised_autoencoder(X[:14], y[:14], X[14:], y[14:], config)

    assert artifacts["model"] is not None
    assert np.isfinite(artifacts["y_mean"])
    assert np.isfinite(artifacts["y_std"])
    assert artifacts["y_std"] > 0
    assert not artifacts["history"].empty
    assert "train_total_loss" in artifacts["history"].columns
    assert artifacts["best_epoch"] >= 1


def test_supervised_autoencoder_encoding_and_prediction_shapes() -> None:
    X, y = _synthetic_matrix(n_rows=16, n_features=9)
    config = SupervisedAEConfig(
        input_dim=9,
        latent_dim=2,
        hidden_dims=[6],
        batch_size=4,
        max_epochs=3,
        patience=2,
        random_state=11,
        device="cpu",
    )
    artifacts = fit_supervised_autoencoder(X[:12], y[:12], X[12:], y[12:], config)

    encoded = encode_with_supervised_autoencoder(artifacts, X[12:])
    predictions = predict_yield_with_supervised_autoencoder(artifacts, X[12:])

    assert encoded.shape == (4, 2)
    assert not hasattr(encoded, "requires_grad")
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_representation_helpers_use_train_rows_only() -> None:
    X_train = np.arange(60, dtype=np.float32).reshape(10, 6)
    y_train = np.arange(10, dtype=np.float32)
    X_valid = np.full((3, 6), 999.0, dtype=np.float32)
    X_test = np.full((3, 6), -999.0, dtype=np.float32)

    X_ae_train, _, X_ae_valid, _ = make_internal_ae_split(
        X_train,
        y_train,
        valid_size=0.2,
        seed=3,
    )
    assert X_ae_valid is not None
    ae_rows = np.vstack([X_ae_train, X_ae_valid])
    assert set(map(tuple, ae_rows)) <= set(map(tuple, X_train))
    assert not set(map(tuple, ae_rows)) & set(map(tuple, X_valid))
    assert not set(map(tuple, ae_rows)) & set(map(tuple, X_test))

    svd, actual_components = fit_svd_on_train_only(
        X_train,
        requested_components=4,
        random_state=5,
    )
    assert actual_components == 4
    assert svd.transform(X_valid).shape == (3, 4)
    assert svd.transform(X_test).shape == (3, 4)


def test_supervised_ae_latent_runner_writes_expected_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "supervised_ae_tiny.yaml"
    metrics_path = tmp_path / "results" / "metrics.csv"
    summary_path = tmp_path / "results" / "summary.csv"
    history_dir = tmp_path / "results" / "history"
    ae_by_seed_path = tmp_path / "results" / "ae_vs_original_rf_by_seed.csv"
    ae_summary_path = tmp_path / "results" / "ae_vs_original_rf_summary.csv"
    _tiny_reactions(36).to_csv(data_path, index=False)
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
svd_baseline:
  enabled: true
  components: [4]
supervised_autoencoder:
  enabled: true
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
output:
  metrics_path: {metrics_path}
  summary_path: {summary_path}
  history_dir: {history_dir}
  ae_vs_original_rf_by_seed_path: {ae_by_seed_path}
  ae_vs_original_rf_summary_path: {ae_summary_path}
""",
        encoding="utf-8",
    )

    paths = run_supervised_ae_latent_baseline(config_path)

    assert paths["metrics_path"] == metrics_path
    assert metrics_path.exists()
    assert summary_path.exists()
    assert ae_by_seed_path.exists()
    assert ae_summary_path.exists()
    assert list(history_dir.glob("*.csv"))
    metrics = pd.read_csv(metrics_path)
    assert {"original_6144", "svd_4", "supervised_ae_4"} <= set(metrics["representation"])
    assert "supervised_ae_yield_head_4" in set(metrics["representation"])
    assert set(metrics["split"]) == {"valid", "test"}
    assert "ae_yield_head" in set(metrics["model"])


def _synthetic_matrix(n_rows: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    weights = rng.normal(size=n_features)
    y = (X @ weights + rng.normal(scale=0.1, size=n_rows)).astype(np.float32)
    return X, y


def _tiny_reactions(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(n_rows)],
            "reaction_smiles": [
                f"CC{'C' * (index % 4)}Br.N>O>CC{'C' * (index % 4)}N"
                for index in range(n_rows)
            ],
            "product_key": [f"product_{index % 3}" for index in range(n_rows)],
            "reactant_key": [f"reactant_{index % 4}" for index in range(n_rows)],
            "yield": [float((index * 7) % 101) for index in range(n_rows)],
        }
    )

