"""Tests for held-out group split runner support."""

from pathlib import Path

import pandas as pd

from bh_augmentation.run_augmentation import run_augmentation
from bh_augmentation.run_baseline import run_baseline


def _write_group_fixture(path: Path) -> None:
    rows = []
    ligands = ["ligand_a", "ligand_b", "ligand_c", "ligand_d", "ligand_e"]
    for ligand_index, ligand in enumerate(ligands):
        for repeat in range(4):
            row_index = ligand_index * 4 + repeat
            rows.append(
                {
                    "reaction_id": f"rxn_{row_index:03d}",
                    "aryl_halide_smiles": ["CCO", "CCN", "CCC", "CNC"][repeat],
                    "amine_smiles": ["N", "CN", "CCN", "NC"][repeat],
                    "ligand_smiles": ligand,
                    "base_smiles": "base_a",
                    "additive_smiles": "additive_a",
                    "solvent": ["DMF", "THF"][repeat % 2],
                    "temperature": 80 + repeat,
                    "reaction_smiles": "CCO.N>>CCN",
                    "yield": float((row_index * 5) % 100),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _heldout_config(data_path: Path, metrics_path: Path, split_metadata_path: Path) -> str:
    return f"""
seed: 17
dataset:
  name: synthetic_bh
  path: {data_path}
splits:
  method: heldout_group
  group_column: ligand_smiles
  heldout_fraction: 0.2
  valid_fraction: 0.1
features:
  smiles_columns: []
  categorical_columns:
    - solvent
models:
  - ridge
metrics:
  - rmse
output:
  metrics_path: {metrics_path}
  split_metadata_path: {split_metadata_path}
"""


def test_baseline_runner_heldout_group_split_has_no_group_leakage(
    tmp_path: Path,
) -> None:
    """Baseline runner should keep held-out test groups out of training."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "baseline_metrics.csv"
    split_metadata_path = tmp_path / "baseline_split_metadata.csv"
    config_path = tmp_path / "baseline.yaml"
    _write_group_fixture(data_path)
    config_path.write_text(
        _heldout_config(data_path, metrics_path, split_metadata_path),
        encoding="utf-8",
    )

    run_baseline(config_path)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["split_method"]) == {"heldout_group"}
    assert set(metrics["group_column"]) == {"ligand_smiles"}

    metadata = pd.read_csv(split_metadata_path)
    train_groups = set(metadata.loc[metadata["split"] == "train", "group_value"])
    test_groups = set(metadata.loc[metadata["split"] == "test", "group_value"])
    assert train_groups.isdisjoint(test_groups)


def test_augmentation_runner_heldout_group_split_has_no_group_leakage(
    tmp_path: Path,
) -> None:
    """Augmentation runner should keep held-out test groups out of training."""
    data_path = tmp_path / "clean_bh.csv"
    metrics_path = tmp_path / "augmentation_metrics.csv"
    split_metadata_path = tmp_path / "augmentation_split_metadata.csv"
    config_path = tmp_path / "augmentation.yaml"
    _write_group_fixture(data_path)
    config_path.write_text(
        _heldout_config(data_path, metrics_path, split_metadata_path)
        + """
augmentation:
  order_permutation:
    enabled: true
    ratio: 0.5
    max_permutations: 1
    random_state: 17
    component_columns:
      - aryl_halide_smiles
      - amine_smiles
""",
        encoding="utf-8",
    )

    run_augmentation(config_path)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["split_method"]) == {"heldout_group"}
    assert set(metrics["group_column"]) == {"ligand_smiles"}
    assert (metrics["eval_augmented_rows"] == 0).all()

    metadata = pd.read_csv(split_metadata_path)
    train_groups = set(metadata.loc[metadata["split"] == "train", "group_value"])
    test_groups = set(metadata.loc[metadata["split"] == "test", "group_value"])
    assert train_groups.isdisjoint(test_groups)
