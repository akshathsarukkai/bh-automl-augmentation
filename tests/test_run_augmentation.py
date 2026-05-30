"""Tests for the safe augmentation experiment runner."""

from pathlib import Path

import pandas as pd

from bh_augmentation.run_augmentation import run_augmentation


def test_run_augmentation_applies_augmentation_only_to_train(tmp_path: Path) -> None:
    """Validation and test rows should remain unaugmented real data."""
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "augmentation.yaml"
    metrics_path = tmp_path / "results" / "safe_aug_metrics.csv"

    pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(12)],
            "aryl_halide_smiles": ["CCO", "CCN", "CCC", "CNC"] * 3,
            "amine_smiles": ["N", "CN", "CCN"] * 4,
            "ligand_smiles": ["P"] * 12,
            "base_smiles": ["O"] * 12,
            "additive_smiles": ["Cl"] * 12,
            "solvent": ["DMF", "THF", "toluene"] * 4,
            "temperature": [80, 90, 100] * 4,
            "reaction_smiles": ["CCO.N>>CCN"] * 12,
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
  smiles_columns: []
  categorical_columns:
    - solvent
models:
  - ridge
metrics:
  - rmse
augmentation:
  smiles_randomization:
    enabled: true
    ratio: 0.5
    n_variants: 1
    random_state: 3
    smiles_columns:
      - aryl_halide_smiles
      - amine_smiles
  order_permutation:
    enabled: true
    ratio: 0.5
    max_permutations: 1
    random_state: 3
    component_columns:
      - aryl_halide_smiles
      - amine_smiles
  combined:
    enabled: true
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
        "randomized_smiles",
        "order_permutation",
        "combined_safe",
    }
    assert set(metrics["train_fraction"]) == {1.0}
    assert set(metrics["split_method"]) == {"random"}
    assert set(metrics["split"]) == {"valid", "test"}
    assert (metrics["eval_augmented_rows"] == 0).all()

    augmented_metrics = metrics[metrics["augmentation"] != "none"]
    assert (augmented_metrics["augmented_train_rows"] > 0).all()

    none_metrics = metrics[metrics["augmentation"] == "none"]
    assert (none_metrics["augmented_train_rows"] == 0).all()
