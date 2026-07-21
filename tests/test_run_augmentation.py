"""Tests for the safe augmentation experiment runner."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.run_augmentation as runner_module
from bh_augmentation.run_augmentation import run_augmentation


def test_run_augmentation_applies_augmentation_only_to_train(tmp_path: Path) -> None:
    """Validation and test rows should remain unaugmented real data."""
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "augmentation.yaml"
    metrics_path = tmp_path / "results" / "safe_aug_metrics.csv"

    pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(12)],
            "reaction_smiles": [
                f"CC{'C' * (index % 3)}Br.N."
                f"{'O' if index % 2 else 'Cl'}.{'C' if index % 4 else 'P'}"
                f">>CC{'C' * (index % 3)}N"
                for index in range(12)
            ],
            "product_key": [f"product_{index % 3}" for index in range(12)],
            "reactant_key": [f"reactant_{index % 4}" for index in range(12)],
            "yield": [10, 20, 30, 40, 50, 60, 70, 80, 90, 35, 45, 55],
        }
    ).to_csv(data_path, index=False)

    config_path.write_text(
        f"""
seed: 3
dataset:
  name: synthetic_bh
  path: {data_path}
splits:
  method: random
  train_size: 0.5
  valid_size: 0.25
  test_size: 0.25
features:
  kind: reaction_role_concat
  n_bits: 16
models:
  - ridge
metrics:
  - rmse
augmentation:
  condition_recombine_pseudolabel:
    enabled: true
    synthetic_multiplier: 0.5
    max_synthetic_rows: 10
    teacher_model: ridge
    min_neighbor_similarity: 0.0
    random_state: 3
output:
  metrics_path: {metrics_path}
""",
        encoding="utf-8",
    )

    output_path = run_augmentation(config_path)

    assert output_path == metrics_path
    assert metrics_path.exists()

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["augmentation"]) == {
        "none",
        "condition_recombine_pseudolabel",
    }
    assert set(metrics["train_fraction"]) == {1.0}
    assert set(metrics["split_method"]) == {"random"}
    assert set(metrics["feature_kind"]) == {"reaction_section_concat"}
    assert set(metrics["n_features"]) == {48}
    assert set(metrics["split"]) == {"valid", "test"}
    assert (metrics["eval_augmented_rows"] == 0).all()

    augmented_metrics = metrics[metrics["augmentation"] != "none"]
    assert (augmented_metrics["augmented_train_rows"] > 0).all()

    none_metrics = metrics[metrics["augmentation"] == "none"]
    assert (none_metrics["augmented_train_rows"] == 0).all()


def test_run_augmentation_records_condition_recombine_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "augmentation.yaml"
    metrics_path = tmp_path / "condition_metrics.csv"
    pd.DataFrame(
        {
            "reaction_id": [f"r{index}" for index in range(12)],
            "reaction_smiles": [
                f"CCBr.N.O.C>>{['CCN', 'CCCN', 'CCCCN'][index % 3]}"
                for index in range(12)
            ],
            "yield": np.linspace(10, 90, 12),
        }
    ).to_csv(data_path, index=False)

    def fake_recombine(
        train_df: pd.DataFrame,
        feature_config: dict[str, object],
        **kwargs: object,
    ) -> pd.DataFrame:
        real = train_df.assign(
            is_synthetic=False,
            is_augmented=False,
            augmentation_method="none",
            pseudo_label=False,
        )
        synthetic = train_df.iloc[[0]].copy()
        synthetic["reaction_id"] = "synthetic_1"
        synthetic["reaction_smiles"] = "CCBr.N.Br.C>>CCN"
        synthetic["yield"] = 55.0
        synthetic["is_synthetic"] = True
        synthetic["is_augmented"] = True
        synthetic["augmentation_method"] = "condition_recombine_pseudolabel"
        synthetic["pseudo_label"] = True
        return pd.concat([real, synthetic], ignore_index=True)

    monkeypatch.setattr(runner_module, "condition_recombine_pseudolabel", fake_recombine)
    config_path.write_text(
        f"""
seed: 5
dataset:
  path: {data_path}
splits:
  method: random
  train_size: 0.5
  valid_size: 0.25
  test_size: 0.25
features:
  kind: reaction_morgan_sum
  n_bits: 16
models:
  - ridge
metrics:
  - rmse
augmentation:
  condition_recombine_pseudolabel:
    enabled: true
    synthetic_multiplier: 1.0
    max_synthetic_rows: 10
    teacher_model: random_forest
    min_neighbor_similarity: 0.3
output:
  metrics_path: {metrics_path}
""",
        encoding="utf-8",
    )

    run_augmentation(config_path)
    metrics = pd.read_csv(metrics_path)
    augmented = metrics[
        metrics["augmentation_method"] == "condition_recombine_pseudolabel"
    ]

    assert not augmented.empty
    assert (augmented["n_synthetic_train"] == 1).all()
    assert (augmented["n_real_train"] == 6).all()
    assert (augmented["eval_augmented_rows"] == 0).all()
