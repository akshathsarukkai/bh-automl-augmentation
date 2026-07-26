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
)
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    generate_role_aware_condition_transfer_examples as _generate_role_aware_examples,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
)
from bh_augmentation.data.bh_condition_reader import (
    parse_bh_reaction_smiles,
    recover_condition_fields,
)
from bh_augmentation.features.featurize import (
    build_feature_matrix,
    build_feature_matrix_with_metadata,
)
from bh_augmentation.run_role_aware_condition_transfer import (
    _iter_role_transfer_policies,
    _role_value_counts,
    _select_policies,
    run_role_aware_condition_transfer,
)


def generate_role_aware_condition_transfer_examples(df, X, y, config):
    _, _, names, metadata = build_feature_matrix_with_metadata(df, _feature_config())
    return _generate_role_aware_examples(
        df,
        X,
        y,
        config,
        feature_config=_feature_config(),
        real_feature_names=names,
        real_feature_metadata=metadata,
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


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"role_change_requirement": "sometimes"},
            "role_change_requirement must be one of",
        ),
        ({"fallback_policy": "hidden_chain"}, "fallback_policy must be one of"),
        (
            {"max_candidates_per_source": 0},
            "max_candidates_per_source must be greater than zero",
        ),
    ],
)
def test_invalid_role_change_configuration_is_rejected(
    override: dict[str, object],
    message: str,
) -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    with pytest.raises(ValueError, match=message):
        generate_role_aware_condition_transfer_examples(
            train,
            X,
            y,
            _config(**override),
        )


def test_ligand_base_all_rejects_donor_that_only_changes_ligand() -> None:
    train = pd.DataFrame([_row(0), _row(1)])
    train.loc[1, "recovered_base_smiles"] = train.loc[
        0, "recovered_base_smiles"
    ]
    X, y, _ = build_feature_matrix(train, _feature_config())

    strict = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="ligand_base",
            donor_strategy="random",
            label_strategy="source_label",
            role_change_requirement="all",
            fallback_policy="reject",
        ),
    )
    permissive = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="ligand_base",
            donor_strategy="random",
            label_strategy="source_label",
            role_change_requirement="any",
            fallback_policy="reject",
        ),
    )

    assert strict["candidate_df"].empty
    assert not permissive["candidate_df"].empty
    accepted = permissive["candidate_df"].loc[
        permissive["candidate_df"]["accepted"]
    ]
    assert not accepted.empty
    assert accepted["role_change_valid"].astype(bool).all()
    assert set(accepted["unchanged_requested_roles"]) == {"base"}
    assert set(accepted["unexpected_changed_roles"]) == {""}


def test_context_strategy_uses_only_the_declared_fallback() -> None:
    train = pd.DataFrame([_row(0), _row(1)])
    X, y, _ = build_feature_matrix(train, _feature_config())
    common = {
        "role_transfer_mode": "ligand_only",
        "donor_strategy": "same_nontransferred_roles",
        "label_strategy": "source_label",
        "role_change_requirement": "all",
    }

    rejected = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(**common, fallback_policy="reject"),
    )
    random_fallback = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(**common, fallback_policy="random"),
    )

    assert rejected["candidate_df"].empty
    assert not random_fallback["candidate_df"].empty
    assert set(random_fallback["candidate_df"]["donor_fallback_level"]) == {
        "fallback_random"
    }
    assert random_fallback["candidate_df"]["fallback_used"].astype(bool).all()


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


def test_invariant_catalyst_is_detected() -> None:
    counts = _role_value_counts(_invariant_catalyst_df())

    catalyst = counts.set_index("role").loc["catalyst"]
    assert catalyst["n_unique"] == 1
    assert bool(catalyst["invariant"])


def test_catalyst_modes_are_made_catalyst_free_when_excluding_invariant_roles() -> None:
    counts = _role_value_counts(_invariant_catalyst_df())
    config = {
        "role_aware_condition_transfer": {
            "exclude_invariant_roles": True,
            "role_transfer_modes": ["catalyst_only", "full_condition_block"],
            "donor_strategies": ["random"],
            "label_strategies": ["source_label"],
            "synthetic_multipliers": [0.5],
            "teacher_models": ["ridge"],
        }
    }

    policies = _iter_role_transfer_policies(config, seed=0, role_value_counts=counts)

    assert len(policies) == 1
    assert policies[0].role_transfer_mode == "full_condition_block"
    assert policies[0].effective_role_transfer_mode == "ligand_base_solvent_or_additive"
    assert policies[0].invariant_roles_excluded == "catalyst"


def test_role_aware_factory_preserves_random_similarity_threshold() -> None:
    counts = _role_value_counts(_train_df())
    config = {
        "role_aware_condition_transfer": {
            "exclude_invariant_roles": False,
            "role_transfer_modes": ["ligand_only"],
            "donor_strategies": ["random"],
            "label_strategies": ["source_label"],
            "synthetic_multipliers": [0.5],
            "min_similarities": [0.73],
            "teacher_models": ["ridge"],
        }
    }

    policies = _iter_role_transfer_policies(config, seed=0, role_value_counts=counts)

    assert len(policies) == 1
    assert policies[0].min_similarity == pytest.approx(0.73)


def test_ligand_base_solvent_mode_changes_at_least_one_variable_role() -> None:
    train = _invariant_catalyst_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="ligand_base_solvent_or_additive",
            donor_strategy="diverse_role_value",
            label_strategy="source_label",
        ),
    )

    synthetic = result["synthetic_df"]
    assert not synthetic.empty
    changed = synthetic[["changed_ligand", "changed_base", "changed_solvent_or_additive"]].astype(bool)
    assert changed.any(axis=1).all()
    assert not synthetic["changed_catalyst"].astype(bool).any()


def test_zero_synthetic_policies_are_not_eligible_for_selection() -> None:
    metrics = pd.DataFrame(
        [
            _metric_row("zero", "valid", "rmse", 1.0, n_synthetic_train=0),
            _metric_row("zero", "test", "rmse", 1.0, n_synthetic_train=0),
            _metric_row("nonzero", "valid", "rmse", 5.0, n_synthetic_train=3),
            _metric_row("nonzero", "test", "rmse", 5.0, n_synthetic_train=3),
            _metric_row(
                "real",
                "valid",
                "rmse",
                10.0,
                representation="bh_role_separated_real_only",
                n_synthetic_train=0,
            ),
        ]
    )

    selected = _select_policies(
        metrics,
        {
            "selection": {"split": "valid", "metric": "rmse", "lower_is_better": True},
            "role_aware_condition_transfer": {"allow_zero_synthetic_policy_selection": False},
        },
    )

    assert selected.iloc[0]["policy_id"] == "nonzero"


@pytest.mark.parametrize(
    ("mode", "same_columns", "different_column"),
    [
        ("ligand_only", ["synthetic_base_smiles", "synthetic_solvent_or_additive_smiles"], "synthetic_ligand_smiles"),
        ("base_only", ["synthetic_ligand_smiles", "synthetic_solvent_or_additive_smiles"], "synthetic_base_smiles"),
        ("solvent_or_additive_only", ["synthetic_ligand_smiles", "synthetic_base_smiles"], "synthetic_solvent_or_additive_smiles"),
    ],
)
def test_same_nontransferred_roles_keeps_context_when_possible(
    mode: str,
    same_columns: list[str],
    different_column: str,
) -> None:
    train = _same_context_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(role_transfer_mode=mode, donor_strategy="same_nontransferred_roles", label_strategy="source_label"),
    )

    kept = result["candidate_df"].loc[result["candidate_df"]["kept"]]
    assert not kept.empty
    assert set(kept["donor_fallback_level"]) == {"strict_same_nontransferred_roles"}
    for _, row in kept.iterrows():
        source = train.iloc[int(row["source_position"])]
        for column in same_columns:
            source_column = column.replace("synthetic_", "recovered_")
            assert row[column] == source[source_column]
        assert row[different_column] != source[different_column.replace("synthetic_", "recovered_")]


def test_audit_contains_invariant_and_fallback_fields() -> None:
    train = _invariant_catalyst_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="full_condition_block",
            effective_role_transfer_mode="ligand_base_solvent_or_additive",
            invariant_roles_excluded="catalyst",
            n_unique_catalysts=1,
            n_unique_ligands=3,
            n_unique_bases=3,
            n_unique_solvents=3,
            donor_strategy="same_nontransferred_roles",
            label_strategy="source_label",
        ),
    )

    metadata = result["metadata"]
    for column in [
        "n_unique_catalysts",
        "invariant_roles_excluded",
        "effective_role_transfer_mode",
        "donor_fallback_level",
        "n_zero_synthetic_policy_skipped",
        "n_same_context_donors_found",
        "n_role_changed_candidates_found",
        "changed_any_transferred_role_fraction",
    ]:
        assert column in metadata


def test_generation_enforces_the_exact_per_source_candidate_cap() -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="full_condition_block",
            donor_strategy="random",
            label_strategy="source_label",
            max_candidates_per_source=2,
            teacher_models=["ridge"],
        ),
    )

    generated_counts = result["candidate_df"].groupby("source_position").size()
    assert not generated_counts.empty
    assert generated_counts.le(2).all()
    assert generated_counts.eq(2).all()


def test_seeded_generation_and_selection_are_row_permutation_invariant() -> None:
    train = _train_df()
    permuted = train.sample(frac=1.0, random_state=91).reset_index(drop=True)
    config = _config(
        role_transfer_mode="full_condition_block",
        donor_strategy="random",
        label_strategy="source_label",
        synthetic_multiplier=1.0,
        max_candidates_per_source=3,
        teacher_models=["ridge"],
        random_state=17,
    )

    X, y, _ = build_feature_matrix(train, _feature_config())
    permuted_X, permuted_y, _ = build_feature_matrix(permuted, _feature_config())
    first = generate_role_aware_condition_transfer_examples(train, X, y, config)
    second = generate_role_aware_condition_transfer_examples(
        permuted,
        permuted_X,
        permuted_y,
        config,
    )

    first_keys = first["synthetic_df"]["canonical_reaction_key"].tolist()
    second_keys = second["synthetic_df"]["canonical_reaction_key"].tolist()
    assert first_keys
    assert first_keys == second_keys


@pytest.mark.parametrize("donor_strategy", sorted(role_transfer_module.DONOR_STRATEGIES))
def test_similarity_threshold_applies_to_every_donor_strategy(
    donor_strategy: str,
) -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy=donor_strategy,
            label_strategy="source_label",
            min_similarity=2.0,
            fallback_policy="random",
            teacher_models=["ridge"],
        ),
    )

    assert result["candidate_df"].empty
    assert result["synthetic_df"].empty


@pytest.mark.parametrize("fallback_policy", sorted(role_transfer_module.FALLBACK_POLICIES))
def test_similarity_threshold_applies_to_every_declared_fallback(
    fallback_policy: str,
) -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="nearest_condition",
            label_strategy="source_label",
            min_similarity=2.0,
            fallback_policy=fallback_policy,
            teacher_models=["ridge"],
        ),
    )

    assert result["candidate_df"].empty
    assert result["synthetic_df"].empty


def test_phase3_candidate_metrics_and_ranking_fields_are_audited() -> None:
    train = _train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_role_aware_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            role_transfer_mode="full_condition_block",
            donor_strategy="random",
            label_strategy="source_label",
            max_candidates_per_source=3,
            teacher_models=["ridge"],
        ),
    )

    candidates = result["candidate_df"]
    required_metrics = {
        "substrate_similarity",
        "product_similarity",
        "nontransferred_role_similarity",
        "condition_similarity",
        "overall_similarity",
        "nearest_training_support_distance",
        "calibrated_uncertainty",
        "uncertainty_rank_value",
        "uncertainty_rank_basis",
        "relevant_context_similarity",
        "diversity_contribution",
        "out_of_support_distance",
        "candidate_rank",
    }
    assert required_metrics <= set(candidates)
    finite_metrics = required_metrics - {
        "calibrated_uncertainty",
        "uncertainty_rank_basis",
    }
    assert np.isfinite(
        candidates[list(finite_metrics)].to_numpy(dtype=float)
    ).all()
    assert candidates["calibrated_uncertainty"].isna().all()
    assert set(candidates["uncertainty_rank_basis"]) == {
        "teacher_std_proxy_phase12_pending"
    }
    assert sorted(candidates["candidate_rank"].astype(int)) == list(
        range(1, len(candidates) + 1)
    )
    kept = candidates.loc[candidates["kept"]].sort_values("candidate_rank")
    expected = (
        candidates.loc[candidates["accepted"]]
        .sort_values("candidate_rank")
        .head(len(kept))
    )
    assert kept["canonical_reaction_key"].tolist() == expected[
        "canonical_reaction_key"
    ].tolist()


def test_candidate_ranking_is_independent_of_generation_traversal() -> None:
    candidates = pd.DataFrame(
        {
            "canonical_reaction_key": [
                "uncertainty",
                "context",
                "diversity",
                "support",
                "canonical_b",
                "canonical_a",
            ],
            "source_row_id": [f"s{index}" for index in range(6)],
            "donor_row_id": [f"d{index}" for index in range(6)],
            "calibrated_uncertainty": [np.nan] * 6,
            "uncertainty_rank_value": [0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
            "relevant_context_similarity": [1.0, 0.9, 0.8, 0.8, 0.8, 0.8],
            "diversity_contribution": [1.0, 0.0, 0.9, 0.8, 0.8, 0.8],
            "out_of_support_distance": [0.0, 1.0, 1.0, 0.1, 0.2, 0.2],
        }
    )

    ranked = role_transfer_module._rank_candidates(candidates)
    permuted = role_transfer_module._rank_candidates(
        candidates.sample(frac=1.0, random_state=4)
    )

    expected = {
        "context": 1,
        "diversity": 2,
        "support": 3,
        "canonical_a": 4,
        "canonical_b": 5,
        "uncertainty": 6,
    }
    assert ranked.set_index("canonical_reaction_key")["candidate_rank"].astype(
        int
    ).to_dict() == expected
    assert permuted.set_index("canonical_reaction_key")[
        "candidate_rank"
    ].astype(int).to_dict() == expected


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
  kind: bh_role_separated
  n_bits: 16
  radius: 2
  fingerprint_backend: hash
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
  exclude_invariant_roles: true
  allow_zero_synthetic_policy_selection: false
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
  role_value_counts_path: {output_dir / "role_value_counts.csv"}
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
    candidate_audit = pd.read_csv(paths["role_transfer_candidate_audit_path"])
    assert "role_transfer_mode" in policy_metrics.columns
    assert "role_transfer_mode" in audit.columns
    assert paths["role_value_counts_path"].exists()
    assert "bh_role_separated_real_only" in set(policy_metrics["representation"])
    assert "bh_role_separated_condition_transfer" in set(
        policy_metrics["representation"]
    )
    assert set(policy_metrics["split"]) == {"valid", "test"}
    assert set(selected_metrics["split"]) == {"valid", "test"}
    assert {"changed_ligand_fraction", "changed_base_fraction"} <= set(audit.columns)
    assert set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(candidate_audit)
    assert not candidate_audit[["source_row_id", "donor_row_id"]].isna().any().any()


def _metric_row(
    policy_id: str,
    split: str,
    metric: str,
    value: float,
    representation: str = "bh_role_separated_condition_transfer",
    n_synthetic_train: int = 1,
) -> dict[str, object]:
    return {
        "seed": 0,
        "train_fraction": 0.1,
        "model": "xgboost",
        "policy_id": policy_id,
        "representation": representation,
        "role_transfer_mode": "ligand_only",
        "donor_strategy": "random",
        "label_strategy": "teacher_ensemble",
        "split": split,
        "metric": metric,
        "value": value,
        "n_synthetic_train": n_synthetic_train,
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
    return {
        "kind": "bh_role_separated",
        "n_bits": 16,
        "radius": 2,
        "fingerprint_backend": "hash",
    }


def _train_df(n: int = 6) -> pd.DataFrame:
    rows = [_row(index % 6) for index in range(n)]
    return pd.DataFrame(rows)


def _invariant_catalyst_df() -> pd.DataFrame:
    rows = []
    for index in range(6):
        row = _row(index)
        row["recovered_catalyst_smiles"] = "Cl[Pd]Cl"
        row["reaction_smiles"] = (
            f"{row['recovered_reactant_1_smiles']}.{row['recovered_reactant_2_smiles']}."
            f"{row['recovered_catalyst_smiles']}.{row['recovered_ligand_smiles']}."
            f"{row['recovered_base_smiles']}.{row['recovered_solvent_or_additive_smiles']}>>"
            f"{row['recovered_product_smiles']}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _same_context_df() -> pd.DataFrame:
    catalyst = "Cl[Pd]Cl"
    product = "c1ccc(Nc2ccccc2)cc1"
    reactants = [
        "Brc1ccccc1",
        "Brc1ccc(C)cc1",
        "Brc1ccc(F)cc1",
        "Brc1ccc(Cl)cc1",
        "Brc1ccc(OC)cc1",
        "Brc1ccc(C#N)cc1",
        "Brc1ccncc1",
        "Brc1ccc(C(F)(F)F)cc1",
    ]
    values = [
        (ligand, base, solvent, float(index * 10 + 10))
        for index, (ligand, base, solvent) in enumerate(
            (ligand, base, solvent)
            for ligand in ["P(C)(C)C", "P(CC)(CC)CC"]
            for base in ["N(C)(C)C", "N1CCCCC1"]
            for solvent in ["CCO", "CCCO"]
        )
    ]
    rows = []
    for index, (ligand, base, solvent, y_value) in enumerate(values):
        r1 = reactants[index]
        r2 = "Nc1ccccc1"
        rows.append(
            {
                "reaction_id": f"context_{index}",
                "reaction_smiles": f"{r1}.{r2}.{catalyst}.{ligand}.{base}.{solvent}>>{product}",
                "yield": y_value,
                "reactant_key": f"{r1}.{r2}",
                "product_key": product,
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
        )
    return pd.DataFrame(rows)


def _row(index: int) -> dict[str, object]:
    values = [
        ("Brc1ccccc1", "Nc1ccccc1", "Cl[Pd]Cl", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "CCN=P(N(C)C)(N(C)C)", "O1CCOCC1", "c1ccc(Nc2ccccc2)cc1", 10.0),
        ("Clc1ccccc1", "Nc1ccccc1", "[Pd]", "CC(C)c1cc(P(C2CCCCC2)C2CCCCC2)ccc1", "CN(C)C(=NC(C)(C)C)N(C)C", "CCCO", "CCNc1ccccc1", 80.0),
        ("Brc1ccccc1", "Nc1ccccc1", "Cl[Pd]Cl", "P(c1ccccc1)(c1ccccc1)c1ccccc1", "CCN=P(N(C)C)(N(C)C)", "O1CCOCC1", "c1ccc(Nc2ccccc2)cc1", 50.0),
        ("Ic1ccccc1", "Nc1ccccc1", "Cl[Pd]Cl", "CC(C)c1cc(P(C2CCCCC2)C2CCCCC2)ccc1", "O=P(O)(O)O", "c1ccccc1", "c1ccc(Nc2ccccc2)cc1", 70.0),
        ("Brc1ccncc1", "Cc1ccc(N)cc1", "[Pd]", "P(C)(C)C", "O=C(O)O", "COC", "Cc1ccc(Nc2ccncc2)cc1", 30.0),
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
