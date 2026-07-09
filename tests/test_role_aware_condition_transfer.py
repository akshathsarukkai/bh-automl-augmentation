"""Tests for role-aware condition-transfer augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation.role_aware_condition_transfer as role_transfer_module
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    build_role_transferred_reaction_smiles,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.data.bh_condition_reader import (
    parse_bh_reaction_smiles,
    recover_condition_fields,
)
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.run_role_aware_condition_transfer import (
    _select_policies,
    run_role_aware_condition_transfer,
)


def test_role_transfer_construction_preserves_source_reactants_and_product() -> None:
    synthetic = build_role_transferred_reaction_smiles(_row(0), _row(1), "ligand_only")

    assert synthetic["synthetic_reactant_1_smiles"] == _row(0)["recovered_reactant_1_smiles"]
    assert synthetic["synthetic_reactant_2_smiles"] == _row(0)["recovered_reactant_2_smiles"]
    assert synthetic["synthetic_product_smiles"] == _row(0)["recovered_product_smiles"]
    assert _row(1)["recovered_product_smiles"] not in synthetic["synthetic_reaction_smiles"]


def test_ligand_only_changes_only_ligand() -> None:
    synthetic = build_role_transferred_reaction_smiles(_row(0), _row(1), "ligand_only")

    assert synthetic["changed_ligand"]
    assert not synthetic["changed_catalyst"]
    assert not synthetic["changed_base"]
    assert not synthetic["changed_solvent_or_additive"]


def test_base_only_changes_only_base() -> None:
    synthetic = build_role_transferred_reaction_smiles(_row(0), _row(1), "base_only")

    assert synthetic["changed_base"]
    assert not synthetic["changed_catalyst"]
    assert not synthetic["changed_ligand"]
    assert not synthetic["changed_solvent_or_additive"]


def test_solvent_only_changes_only_solvent() -> None:
    synthetic = build_role_transferred_reaction_smiles(_row(0), _row(1), "solvent_or_additive_only")

    assert synthetic["changed_solvent_or_additive"]
    assert not synthetic["changed_catalyst"]
    assert not synthetic["changed_ligand"]
    assert not synthetic["changed_base"]


def test_full_condition_block_changes_all_condition_roles() -> None:
    synthetic = build_role_transferred_reaction_smiles(_row(0), _row(1), "full_condition_block")

    assert synthetic["changed_catalyst"]
    assert synthetic["changed_ligand"]
    assert synthetic["changed_base"]
    assert synthetic["changed_solvent_or_additive"]


def test_identical_synthetic_reactions_are_skipped() -> None:
    train = pd.DataFrame([_row(0), _row(2)])
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(role_transfer_mode="ligand_only", donor_strategy="random", synthetic_multiplier=1.0),
    )

    assert result["metadata"]["n_identical_skipped"] > 0
    assert result["synthetic_df"].empty


def test_donor_rows_are_train_only_and_valid_test_are_not_used() -> None:
    train = _train_df()
    train.index = [10, 11, 12, 13, 14, 15]
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(role_transfer_mode="full_condition_block", donor_strategy="random"),
    )

    synthetic = result["synthetic_df"]
    assert not synthetic.empty
    train_indices = set(train.index)
    assert set(synthetic["source_index"]) <= train_indices
    assert set(synthetic["donor_index"]) <= train_indices
    assert set(synthetic["source_index"]).isdisjoint({0, 1, 2})
    assert set(synthetic["donor_index"]).isdisjoint({0, 1, 2})


def test_uncertainty_filtered_teacher_rejects_high_std(monkeypatch: pytest.MonkeyPatch) -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    def fake_teacher_predictions(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, list[str]]:
        n = len(args[2])
        return np.full(n, 50.0), np.full(n, 99.0), ["fake"]

    monkeypatch.setattr(role_transfer_module, "_teacher_predictions", fake_teacher_predictions)
    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="full_condition_block",
            donor_strategy="random",
            label_strategy="uncertainty_filtered_teacher",
            max_teacher_std=10.0,
        ),
    )

    assert result["metadata"]["n_teacher_uncertainty_rejected"] > 0
    assert result["synthetic_df"].empty


def test_selected_policy_uses_validation_rmse_not_test_rmse() -> None:
    metrics = pd.DataFrame(
        [
            _metric_row("p1", "valid", "rmse", 1.0),
            _metric_row("p1", "test", "rmse", 100.0),
            _metric_row("p2", "valid", "rmse", 2.0),
            _metric_row("p2", "test", "rmse", 1.0),
        ]
    )

    selected = _select_policies(metrics, {"selection": {"split": "valid", "metric": "rmse", "lower_is_better": True}})

    assert selected.iloc[0]["policy_id"] == "p1"


def test_generated_synthetic_reaction_has_six_left_tokens_and_passes_parser() -> None:
    synthetic = build_role_transferred_reaction_smiles(_row(0), _row(1), "full_condition_block")

    parsed = parse_bh_reaction_smiles(synthetic["synthetic_reaction_smiles"])
    recovered = recover_condition_fields({"reaction_smiles": synthetic["synthetic_reaction_smiles"]})

    assert parsed is not None
    assert parsed["condition_parse_status"] == "ok"
    assert len(str(synthetic["synthetic_reaction_smiles"]).split(">>")[0].split(".")) == 6
    assert recovered["condition_parse_status"] == "ok"


def test_audit_metadata_contains_changed_role_fractions() -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(role_transfer_mode="full_condition_block", donor_strategy="random"),
    )

    metadata = result["metadata"]
    for column in [
        "changed_catalyst_fraction",
        "changed_ligand_fraction",
        "changed_base_fraction",
        "changed_solvent_fraction",
    ]:
        assert column in metadata


def test_runner_smoke_tiny_config_outputs(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "role_transfer.yaml"
    output_dir = tmp_path / "results"
    _train_df(36).to_csv(data_path, index=False)
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
  n_bits: 16
  radius: 2
models:
  - name: xgboost
    params:
      n_estimators: 2
      max_depth: 2
metrics: [rmse, mae, r2, spearman]
role_aware_condition_transfer:
  enabled: true
  role_transfer_modes: [ligand_only, full_condition_block]
  donor_strategies: [random]
  label_strategies: [teacher_ensemble]
  synthetic_multipliers: [0.5]
  max_candidates_per_source: 2
  max_teacher_stds: [10.0]
  min_similarities: [0.0]
  teacher_models: [ridge]
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
  role_transfer_audit_path: {output_dir / "role_transfer_audit.csv"}
  role_transfer_vs_real_only_same_model_by_seed_path: {output_dir / "role_transfer_vs_real_only_same_model_by_seed.csv"}
  role_transfer_vs_real_only_same_model_summary_path: {output_dir / "role_transfer_vs_real_only_same_model_summary.csv"}
  role_transfer_vs_original_rf_by_seed_path: {output_dir / "role_transfer_vs_original_rf_by_seed.csv"}
  role_transfer_vs_original_rf_summary_path: {output_dir / "role_transfer_vs_original_rf_summary.csv"}
""",
        encoding="utf-8",
    )

    paths = run_role_aware_condition_transfer(config_path)

    policy_metrics = pd.read_csv(paths["policy_metrics_path"])
    selected_metrics = pd.read_csv(paths["selected_policy_metrics_path"])
    audit = pd.read_csv(paths["role_transfer_audit_path"])
    assert "role_transfer_mode" in policy_metrics.columns
    assert "role_transfer_mode" in audit.columns
    assert "original_6144" in set(policy_metrics["representation"])
    assert "role_aware_condition_transfer" in set(policy_metrics["representation"])
    assert set(policy_metrics["split"]) == {"valid", "test"}
    assert set(selected_metrics["split"]) == {"valid", "test"}
    assert {"changed_ligand_fraction", "changed_base_fraction"} <= set(audit.columns)


def _metric_row(policy_id: str, split: str, metric: str, value: float) -> dict[str, object]:
    return {
        "seed": 0,
        "train_fraction": 0.1,
        "model": "xgboost",
        "policy_id": policy_id,
        "representation": "role_aware_condition_transfer",
        "role_transfer_mode": "ligand_only",
        "donor_strategy": "random",
        "label_strategy": "teacher_ensemble",
        "split": split,
        "metric": metric,
        "value": value,
    }


def _config(**overrides: object) -> RoleAwareConditionTransferConfig:
    config = RoleAwareConditionTransferConfig(
        role_transfer_mode="ligand_only",
        donor_strategy="random",
        label_strategy="teacher_ensemble",
        synthetic_multiplier=0.5,
        max_candidates_per_source=4,
        teacher_models=["ridge", "random_forest"],
        max_teacher_std=None,
        min_similarity=None,
        random_state=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _feature_config() -> dict[str, object]:
    return {"kind": "reaction_role_concat", "n_bits": 16, "radius": 2}


def _train_df(n: int = 6) -> pd.DataFrame:
    rows = [_row(index % 6) for index in range(n)]
    return pd.DataFrame(rows)


def _row(index: int) -> dict[str, object]:
    values = [
        ("Brc1ccccc1", "Nc1ccccc1", "Cl[Pd]Cl", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "CCN=P(N(C)C)(N(C)C)", "O1CCOCC1", "c1ccc(Nc2ccccc2)cc1", 10.0),
        ("Clc1ccccc1", "Nc1ccccc1", "O[Pd]1ccccc1", "CC(C)c1cc(P(C2CCCCC2)C2CCCCC2)ccc1", "CN(C)C(=NC(C)(C)C)N(C)C", "c1ccc(-c2ccno2)cc1", "CCNc1ccccc1", 80.0),
        ("Brc1ccccc1", "Nc1ccccc1", "Cl[Pd]Cl", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "CCN=P(N(C)C)(N(C)C)", "O1CCOCC1", "c1ccc(Nc2ccccc2)cc1", 50.0),
        ("Ic1ccccc1", "Nc1ccccc1", "Cl[Pd]Cl", "CC(C)c1cc(P(C2CCCCC2)C2CCCCC2)ccc1", "K3PO4", "c1ccccc1", "c1ccc(Nc2ccccc2)cc1", 70.0),
        ("Brc1ccncc1", "Cc1ccc(N)cc1", "O[Pd]1ccccc1", "P(C)(C)C", "K2CO3", "COC", "Cc1ccc(Nc2ccncc2)cc1", 30.0),
        ("Clc1ccncc1", "Cc1ccc(N)cc1", "Cl[Pd]Cl", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "CN(C)C(=NC(C)(C)C)N(C)C", "CCO", "Cc1ccc(Nc2ccncc2)cc1", 90.0),
    ][index]
    r1, r2, catalyst, ligand, base, solvent, product, y_value = values
    return {
        "reaction_id": f"r{index}",
        "reaction_smiles": f"{r1}.{r2}.{catalyst}.{ligand}.{base}.{solvent}>>{product}",
        "yield": y_value,
        "recovered_reactant_1_smiles": r1,
        "recovered_reactant_2_smiles": r2,
        "recovered_catalyst_smiles": catalyst,
        "recovered_ligand_smiles": ligand,
        "recovered_base_smiles": base,
        "recovered_solvent_or_additive_smiles": solvent,
        "recovered_product_smiles": product,
        "condition_parse_status": "ok",
        "role_validation_status": "valid",
    }
