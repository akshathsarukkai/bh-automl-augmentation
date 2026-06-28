"""Tests for condition-transfer augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    build_condition_transfer_reaction,
    generate_condition_transfer_examples,
    parse_condition_transfer_reaction,
)
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.run_condition_transfer import run_condition_transfer


def test_parse_condition_transfer_reaction_sections() -> None:
    parsed = parse_condition_transfer_reaction("A.B.C.D.E>>P")

    assert parsed is not None
    assert parsed.left_tokens == ["A", "B", "C", "D", "E"]
    assert parsed.substrates == ("A", "B")
    assert parsed.condition_tokens == ("C", "D", "E")
    assert parsed.substrate_block == "A.B"
    assert parsed.condition_block == "C.D.E"
    assert parsed.product == "P"


def test_parse_condition_transfer_reaction_rejects_invalid_values() -> None:
    assert parse_condition_transfer_reaction("A.B>>P") is None
    assert parse_condition_transfer_reaction("A.B.C") is None
    assert parse_condition_transfer_reaction("A.B.C>>") is None


def test_build_condition_transfer_reaction_preserves_source_parts() -> None:
    source = parse_condition_transfer_reaction("A.B.C1.D1>>P")
    donor = parse_condition_transfer_reaction("A2.B2.C2.D2.E2>>P2")

    assert source is not None
    assert donor is not None
    synthetic = build_condition_transfer_reaction(source, donor)

    assert synthetic == "A.B.C2.D2.E2>>P"
    assert "P2" not in synthetic


def test_condition_transfer_parent_indices_are_train_only() -> None:
    train = _tiny_train_df().copy()
    train.index = [10, 11, 12, 13, 14, 15]
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(donor_strategy="nearest_reaction", label_strategy="average_label"),
    )

    synthetic = result["synthetic_df"]
    assert not synthetic.empty
    train_indices = set(train.index)
    valid_test_indices = {0, 1, 2, 3}
    assert set(synthetic["source_index"]) <= train_indices
    assert set(synthetic["donor_index"]) <= train_indices
    assert set(synthetic["source_index"]).isdisjoint(valid_test_indices)
    assert set(synthetic["donor_index"]).isdisjoint(valid_test_indices)


def test_condition_transfer_donor_strategies_do_not_select_self() -> None:
    train = _tiny_train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())
    for strategy in ["random", "nearest_reaction"]:
        result = generate_condition_transfer_examples(
            train,
            X,
            y,
            _config(donor_strategy=strategy, label_strategy="average_label"),
        )
        kept = result["candidate_df"].loc[result["candidate_df"]["kept"]]
        assert not kept.empty
        assert (kept["source_position"] != kept["donor_position"]).all()


def test_high_yield_nearest_prefers_high_yield_donors_when_available() -> None:
    train = _tiny_train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="high_yield_nearest",
            label_strategy="donor_label",
            high_yield_threshold=70.0,
        ),
    )

    synthetic = result["synthetic_df"]
    assert not synthetic.empty
    assert (synthetic["donor_yield"] >= 70.0).all()


def test_condition_transfer_label_strategies() -> None:
    train = _tiny_train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    for strategy in ["source_label", "donor_label", "average_label", "teacher_ensemble"]:
        result = generate_condition_transfer_examples(
            train,
            X,
            y,
            _config(donor_strategy="random", label_strategy=strategy),
        )
        synthetic = result["synthetic_df"]
        assert not synthetic.empty
        if strategy == "source_label":
            assert np.allclose(synthetic["yield"], synthetic["source_yield"])
        elif strategy == "donor_label":
            assert np.allclose(synthetic["yield"], synthetic["donor_yield"])
        elif strategy == "average_label":
            expected = 0.5 * (synthetic["source_yield"] + synthetic["donor_yield"])
            assert np.allclose(synthetic["yield"], expected)
        else:
            assert np.isfinite(synthetic["yield"]).all()
            assert np.isfinite(synthetic["teacher_mean"]).all()
            assert np.isfinite(synthetic["teacher_std"]).all()


def test_uncertainty_filtered_teacher_respects_max_teacher_std() -> None:
    train = _tiny_train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())
    base = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(donor_strategy="random", label_strategy="teacher_ensemble"),
    )
    filtered = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="random",
            label_strategy="uncertainty_filtered_teacher",
            max_teacher_std=0.0,
        ),
    )

    assert filtered["metadata"]["n_candidates_accepted"] <= base["metadata"]["n_candidates_accepted"]
    if not filtered["synthetic_df"].empty:
        assert (filtered["synthetic_df"]["teacher_std"] <= 0.0).all()


def test_condition_transfer_filtering_clips_and_removes_existing_duplicates() -> None:
    train = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2"],
            "reaction_smiles": ["A.B.C>>P", "A.B.D>>P"],
            "yield": [-20.0, 140.0],
        }
    )
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="random",
            label_strategy="average_label",
            synthetic_multiplier=2.0,
            candidates_per_real=20,
        ),
    )

    assert result["metadata"]["existing_real_duplicate_count"] > 0
    assert result["synthetic_df"].empty
    labels = result["candidate_df"]["synthetic_label"].dropna()
    assert labels.between(0.0, 100.0).all()


def test_condition_transfer_runner_tiny_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "reactions.csv"
    config_path = tmp_path / "condition_transfer.yaml"
    output_dir = tmp_path / "results"
    _tiny_train_df(40).to_csv(data_path, index=False)
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
condition_transfer:
  enabled: true
  donor_strategies: [random, nearest_reaction]
  label_strategies: [teacher_ensemble, average_label]
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

    policy_metrics = pd.read_csv(paths["policy_metrics_path"])
    selected = pd.read_csv(paths["selected_policy_metrics_path"])
    audit = pd.read_csv(paths["synthetic_audit_path"])
    assert paths["selected_policies_path"].exists()
    assert paths["summary_path"].exists()
    assert "original_6144" in set(policy_metrics["representation"])
    assert "condition_transfer" in set(policy_metrics["representation"])
    assert set(policy_metrics["split"]) == {"valid", "test"}
    assert set(selected["split"]) == {"valid", "test"}
    assert not audit["used_validation_or_test_parents"].any()
    assert (audit["n_synthetic_train"] > 0).any()


def _feature_config() -> dict[str, object]:
    return {"kind": "reaction_role_concat", "n_bits": 8, "radius": 2}


def _config(
    donor_strategy: str,
    label_strategy: str,
    synthetic_multiplier: float = 1.0,
    candidates_per_real: int = 10,
    high_yield_threshold: float = 70.0,
    max_teacher_std: float | None = None,
) -> ConditionTransferConfig:
    return ConditionTransferConfig(
        donor_strategy=donor_strategy,
        synthetic_multiplier=synthetic_multiplier,
        n_neighbors=3,
        label_strategy=label_strategy,
        teacher_models=["ridge", "random_forest"],
        max_teacher_std=max_teacher_std,
        min_similarity=None,
        high_yield_threshold=high_yield_threshold,
        clip_y_min=0.0,
        clip_y_max=100.0,
        candidates_per_real=candidates_per_real,
        random_state=4,
    )


def _tiny_train_df(n_rows: int = 6) -> pd.DataFrame:
    base = [
        ("r1", "CCBr.N.O.Cl>>CCN", 20.0),
        ("r2", "CCCl.N.Br.C>>CCN", 85.0),
        ("r3", "CCCBr.N.O.C>>CCCN", 75.0),
        ("r4", "CCCCl.N.Br.O>>CCCN", 35.0),
        ("r5", "CCBr.CN.O.C>>CCNC", 90.0),
        ("r6", "CCCl.CN.Br.Cl>>CCNC", 45.0),
    ]
    rows = [base[index % len(base)] for index in range(n_rows)]
    return pd.DataFrame(
        {
            "reaction_id": [f"{reaction_id}_{index}" for index, (reaction_id, _, _) in enumerate(rows)],
            "reaction_smiles": [reaction for _, reaction, _ in rows],
            "yield": [float((yield_value + index) % 101) for index, (_, _, yield_value) in enumerate(rows)],
        }
    )

