"""Tests for the matched anonymous-transfer plus supervised-AE experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
)
from bh_augmentation.features.compatibility import coordinate_feature_contract
from bh_augmentation.representations.supervised_autoencoder import (
    SupervisedAEConfig,
    _compute_losses,
    encode_with_supervised_autoencoder,
    fit_supervised_autoencoder,
)
from bh_augmentation.run_condition_transfer_supervised_ae import (
    _make_weighted_internal_ae_split,
    best_parent_rmse_by_seed,
    combine_real_and_synthetic_training_data,
    run_condition_transfer_supervised_ae,
    select_hybrid_policies,
)


def test_combined_training_contains_real_and_synthetic_labels_and_weights() -> None:
    X_real = np.arange(12, dtype=np.float32).reshape(3, 4)
    y_real = np.array([10.0, 20.0, 30.0])
    X_synthetic = np.full((2, 4), 99.0, dtype=np.float32)
    y_synthetic = np.array([41.0, 42.0])

    X, y, yield_weight, reconstruction_weight, synthetic = (
        combine_real_and_synthetic_training_data(
            X_real,
            y_real,
            X_synthetic,
            y_synthetic,
            real_example_weight=1.0,
            synthetic_example_weight=0.25,
            synthetic_reconstruction_weight=0.5,
            **_compatibility_kwargs(4),
        )
    )

    assert X.shape == (5, 4)
    assert np.array_equal(y[-2:], y_synthetic)
    assert np.allclose(yield_weight, [1.0, 1.0, 1.0, 0.25, 0.25])
    assert np.allclose(reconstruction_weight[-2:], 0.5)
    assert synthetic.tolist() == [False, False, False, True, True]


def test_real_only_combination_is_unchanged_without_synthetic_rows() -> None:
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    combined = combine_real_and_synthetic_training_data(X, y, None, None, 1.0, 0.5)

    assert np.array_equal(combined[0], X)
    assert np.array_equal(combined[1], y)
    assert np.all(combined[2] == 1.0)
    assert not combined[4].any()


def test_synthetic_weight_changes_supervised_loss_contribution() -> None:
    config = SupervisedAEConfig(input_dim=1, latent_dim=1, hidden_dims=[])
    X = torch.zeros((2, 1))
    y = torch.zeros(2)
    z = torch.zeros((2, 1))
    reconstructed = torch.zeros((2, 1))
    predictions = torch.tensor([0.0, 2.0])

    unweighted = _compute_losses(X, y, z, reconstructed, predictions, config)
    weighted = _compute_losses(
        X,
        y,
        z,
        reconstructed,
        predictions,
        config,
        yield_sample_weight=torch.tensor([1.0, 0.1]),
    )

    assert weighted["yield_loss"] < unweighted["yield_loss"]


def test_internal_ae_validation_contains_real_rows_only() -> None:
    X_real = np.arange(24, dtype=np.float32).reshape(6, 4)
    X_synthetic = np.full((2, 4), 999.0, dtype=np.float32)
    combined = combine_real_and_synthetic_training_data(
        X_real,
        np.arange(6),
        X_synthetic,
        np.array([50.0, 60.0]),
        1.0,
        0.5,
        **_compatibility_kwargs(4),
    )

    split = _make_weighted_internal_ae_split(
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        combined[4],
        valid_size=0.34,
        seed=3,
    )

    assert split["X_valid"] is not None
    assert not np.any(np.asarray(split["X_valid"]) == 999.0)
    assert sum(np.all(row == 999.0) for row in np.asarray(split["X_train"])) == 2


def test_hybrid_selection_uses_validation_rmse_and_rejects_zero_synthetic() -> None:
    rows = [
        _selection_row("policy_valid_best", "valid", 4.0, 3),
        _selection_row("policy_test_best", "valid", 7.0, 3),
        _selection_row("policy_test_best", "test", 1.0, 3),
        _selection_row("zero_synthetic", "valid", 0.5, 0),
    ]

    selected = select_hybrid_policies(pd.DataFrame(rows))

    assert selected.iloc[0]["hybrid_policy_id"] == "policy_valid_best"


def test_best_parent_is_chosen_per_seed_before_aggregation() -> None:
    anonymous = pd.DataFrame(
        {"seed": [0, 1], "train_fraction": [0.1, 0.1], "value": [5.0, 20.0]}
    )
    ae_only = pd.DataFrame(
        {"seed": [0, 1], "train_fraction": [0.1, 0.1], "value": [10.0, 6.0]}
    )

    result = best_parent_rmse_by_seed(anonymous, ae_only)

    assert result.sort_values("seed")["best_parent_rmse"].tolist() == [5.0, 6.0]
    assert result.sort_values("seed")["best_parent"].tolist() == [
        "anonymous_condition_transfer",
        "supervised_ae_only",
    ]


def test_weighted_ae_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    y = rng.normal(size=12).astype(np.float32)
    config = SupervisedAEConfig(
        input_dim=6,
        latent_dim=3,
        hidden_dims=[5],
        max_epochs=2,
        batch_size=4,
        random_state=11,
        device="cpu",
    )
    weights = np.linspace(0.25, 1.0, len(X), dtype=np.float32)

    first = fit_supervised_autoencoder(X, y, None, None, config, sample_weight=weights)
    second = fit_supervised_autoencoder(X, y, None, None, config, sample_weight=weights)

    assert np.allclose(
        encode_with_supervised_autoencoder(first, X),
        encode_with_supervised_autoencoder(second, X),
    )


def test_hybrid_runner_tiny_writes_outputs_and_leakage_audits(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "hybrid.yaml"
    _tiny_reactions(54).to_csv(data_path, index=False)
    config_path.write_text(_tiny_config(data_path, output_dir), encoding="utf-8")

    paths = run_condition_transfer_supervised_ae(config_path)

    metrics = pd.read_csv(paths["policy_metrics_path"])
    selected = pd.read_csv(paths["selected_hybrid_policy_metrics_path"])
    ae_audit = pd.read_csv(paths["ae_training_audit_path"])
    synthetic_audit = pd.read_csv(paths["synthetic_training_audit_path"])
    candidate_audit = pd.read_csv(paths["synthetic_candidate_audit_path"])
    assert {"seed", "train_fraction"} <= set(metrics.columns)
    assert {"valid", "test"} <= set(metrics["split"])
    assert any(metrics["representation"].astype(str).str.endswith("_condition_transfer"))
    assert not selected.empty
    assert not ae_audit["used_outer_valid_for_ae_training"].any()
    assert not ae_audit["used_outer_test_for_ae_training"].any()
    assert (ae_audit["n_synthetic_in_internal_valid"].fillna(0) == 0).all()
    assert synthetic_audit["source_and_donor_indices_train_only"].all()
    assert not synthetic_audit["used_validation_or_test_parents"].any()
    assert set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(candidate_audit)
    assert not candidate_audit[["source_row_id", "donor_row_id"]].isna().any().any()
    for required in [
        "selected_hybrid_policies_path",
        "summary_path",
        "hybrid_vs_best_parent_by_seed_path",
    ]:
        assert paths[required].exists()


def _selection_row(policy: str, split: str, value: float, n_synthetic: int) -> dict[str, object]:
    return {
        "seed": 0,
        "train_fraction": 0.1,
        "representation": "supervised_ae_16_condition_transfer",
        "downstream_model": "xgboost",
        "hybrid_policy_id": policy,
        "split": split,
        "metric": "rmse",
        "value": value,
        "n_synthetic_train": n_synthetic,
        "ae_training_status": "ok",
        "ae_reconstruction_loss": 1.0,
        "ae_supervised_loss": 1.0,
    }


def _tiny_reactions(n_rows: int) -> pd.DataFrame:
    ligands = ["P(C)(C)C", "P(CC)(CC)CC", "P(c1ccccc1)(C)C"]
    bases = ["N(C)(C)C", "C[NH2+]C", "O=C(O)O"]
    solvents = ["CCO", "CCCO", "CCOC"]
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{i}" for i in range(n_rows)],
            "reaction_smiles": [
                f"{'C' * (2 + i % 4)}Br.CN.[Pd].{ligands[i % 3]}.{bases[(i // 3) % 3]}.{solvents[(i // 9) % 3]}>>{'C' * (2 + i % 4)}N"
                for i in range(n_rows)
            ],
            "product_key": [f"p_{i % 4}" for i in range(n_rows)],
            "reactant_key": [f"r_{i % 6}" for i in range(n_rows)],
            "yield": [float((i * 13) % 101) for i in range(n_rows)],
        }
    )


def _tiny_config(data_path: Path, output_dir: Path) -> str:
    return f"""
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
  kind: bh_role_separated
  n_bits: 8
  radius: 2
  fingerprint_backend: hash
models: [ridge]
metrics: [rmse, mae, r2, spearman]
condition_transfer:
  enabled: true
  role_change_requirement: any
  fallback_policy: reject
  selection_model: ridge
  donor_strategies: [random]
  label_strategies: [teacher_ensemble]
  policy_pairs:
    - {{donor_strategy: random, label_strategy: teacher_ensemble}}
  synthetic_multipliers: [0.5]
  n_neighbors: [3]
  min_similarities: [null]
  max_teacher_stds: [null]
  candidates_per_real: 5
  teacher_models: [ridge]
supervised_autoencoder:
  latent_dims: [4]
  hidden_dims: [8]
  max_epochs: 2
  patience: 1
  batch_size: 4
  internal_valid_size: 0.2
  device: cpu
  synthetic_example_weights: [0.5]
evaluate_full_data_reference: false
output:
  directory: {output_dir}
"""


def _compatibility_kwargs(width: int) -> dict[str, object]:
    names, metadata = coordinate_feature_contract("test_feature_space", width)
    return {
        "real_feature_names": names,
        "synthetic_feature_names": names,
        "real_feature_metadata": metadata,
        "synthetic_feature_metadata": metadata,
    }
