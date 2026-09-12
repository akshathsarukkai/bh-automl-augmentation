"""Tests for condition-transfer augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bh_augmentation.augmentation.condition_transfer import (
    ANONYMOUS_TRANSFER_ROLES,
    ConditionTransferConfig,
    _select_donor_position,
    _validate_config,
    build_condition_transfer_reaction,
    parse_condition_transfer_reaction,
)
from bh_augmentation.augmentation.condition_transfer import (
    generate_condition_transfer_examples as _generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
)
from bh_augmentation.data.reaction_roles import ReactionRoles, ensure_reaction_role_columns
from bh_augmentation.features.featurize import (
    build_feature_matrix as _build_feature_matrix,
)
from bh_augmentation.features.featurize import (
    build_feature_matrix_with_metadata,
)
from bh_augmentation.run_condition_transfer import (
    _iter_condition_transfer_policies,
    run_condition_transfer,
)


def build_feature_matrix(df: pd.DataFrame, config: dict[str, object]):
    return _build_feature_matrix(ensure_reaction_role_columns(df), config)


def generate_condition_transfer_examples(df, X, y, config):
    enriched = ensure_reaction_role_columns(df)
    _, _, names, metadata = build_feature_matrix_with_metadata(enriched, _feature_config())
    return _generate_condition_transfer_examples(
        enriched,
        X,
        y,
        config,
        feature_config=_feature_config(),
        real_feature_names=names,
        real_feature_metadata=metadata,
    )


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


def test_strict_condition_transfer_changes_every_requested_role_only() -> None:
    train = _strict_role_change_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="random",
            label_strategy="average_label",
            role_change_requirement="all",
            fallback_policy="reject",
        ),
    )

    kept = result["candidate_df"].loc[result["candidate_df"]["kept"]]
    assert not kept.empty
    assert kept["role_change_valid"].all()
    assert set(kept["requested_roles"]) == {"|".join(ANONYMOUS_TRANSFER_ROLES)}
    assert set(kept["actual_changed_roles"]) == {"|".join(ANONYMOUS_TRANSFER_ROLES)}
    assert not kept["unchanged_requested_roles"].any()
    assert not kept["unexpected_changed_roles"].any()
    assert set(kept["change_mask"]) == {"0011110"}
    assert set(kept["role_change_requirement"]) == {"all"}


def test_strict_variable_role_transfer_preserves_invariant_catalyst_and_is_permutation_deterministic(
) -> None:
    train = _strict_role_change_df().copy()
    train["reaction_smiles"] = train["reaction_smiles"].str.replace(
        "[Pt]", "[Pd]", regex=False
    ).str.replace("[Ni]", "[Pd]", regex=False)
    X, y, _ = build_feature_matrix(train, _feature_config())
    canonical_roles = ("ligand", "base", "solvent_or_additive")
    permuted_roles = ("solvent_or_additive", "ligand", "base")

    canonical = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="random",
            label_strategy="average_label",
            role_change_requirement="all",
            fallback_policy="reject",
            requested_roles=canonical_roles,
        ),
    )
    permuted = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="random",
            label_strategy="average_label",
            role_change_requirement="all",
            fallback_policy="reject",
            requested_roles=permuted_roles,
        ),
    )

    kept = canonical["candidate_df"].loc[canonical["candidate_df"]["kept"]]
    assert not kept.empty
    expected_roles = "|".join(canonical_roles)
    assert set(kept["requested_roles"]) == {expected_roles}
    assert set(kept["actual_changed_roles"]) == {expected_roles}
    assert not kept["unchanged_requested_roles"].any()
    assert not kept["unexpected_changed_roles"].any()
    assert set(kept["change_mask"]) == {"0001110"}
    assert set(kept["recovered_catalyst_smiles"]) == {"[Pd]"}
    assert canonical["metadata"]["requested_roles"] == expected_roles
    assert permuted["metadata"]["requested_roles"] == expected_roles
    deterministic_columns = [
        "source_row_id",
        "donor_row_id",
        "canonical_reaction_key",
        "canonical_reaction_hash",
        "feature_hash",
        "requested_roles",
        "actual_changed_roles",
        "change_mask",
        "rejection_reason",
        "kept",
    ]
    pd.testing.assert_frame_equal(
        canonical["candidate_df"].loc[:, deterministic_columns],
        permuted["candidate_df"].loc[:, deterministic_columns],
    )


def test_any_role_change_accepts_partial_condition_changes_and_audits_them() -> None:
    train = _tiny_train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="random",
            label_strategy="average_label",
            role_change_requirement="any",
        ),
    )

    kept = result["candidate_df"].loc[result["candidate_df"]["kept"]]
    assert not kept.empty
    assert kept["role_change_valid"].all()
    assert kept["actual_changed_roles"].str.len().gt(0).all()
    assert kept["unchanged_requested_roles"].str.len().gt(0).any()
    assert not kept["unexpected_changed_roles"].any()


def test_declared_fallback_policies_select_only_the_named_behavior() -> None:
    roles = [
        ReactionRoles(
            "CCBr", "N", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "CCN"
        ),
        ReactionRoles(
            "CCCl", "N", "[Pt]", "P(CC)(CC)CC", "N1CCCCC1", "CCCO", "CCN"
        ),
        ReactionRoles(
            "CCCBr", "CN", "[Ni]", "P(C)(C)CC", "CCN(CC)CC", "CO", "CCCN"
        ),
    ]
    reaction_similarity = np.array(
        [[-np.inf, 0.3, 0.9], [0.3, -np.inf, 0.1], [0.9, 0.1, -np.inf]]
    )
    substrate_similarity = np.array(
        [[-np.inf, 0.2, 0.8], [0.2, -np.inf, 0.1], [0.8, 0.1, -np.inf]]
    )
    y = np.array([10.0, 20.0, 30.0])

    def select(fallback_policy: str, seed: int = 9):
        return _select_donor_position(
            source_position=0,
            valid_positions=[0, 1, 2],
            y_train=y,
            reaction_similarity=reaction_similarity,
            substrate_similarity=substrate_similarity,
            canonical_roles=roles,
            config=_config(
                donor_strategy="high_yield_nearest",
                label_strategy="average_label",
                high_yield_threshold=100.0,
                role_change_requirement="all",
                fallback_policy=fallback_policy,
            ),
            rng=np.random.default_rng(seed),
        )

    reject_donor, reject_similarity, reject_used = select("reject")
    assert reject_donor is None
    assert np.isnan(reject_similarity)
    assert reject_used is True
    assert select("same_product") == (1, 0.3, True)
    assert select("nearest_substrate") == (2, 0.8, True)
    expected_random = int(np.random.default_rng(9).choice([1, 2]))
    assert select("random")[0] == expected_random
    assert select("random")[2] is True


def test_strict_reject_never_silently_becomes_random_fallback() -> None:
    train = _strict_role_change_df()
    X, y, _ = build_feature_matrix(train, _feature_config())

    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            donor_strategy="high_yield_nearest",
            label_strategy="average_label",
            high_yield_threshold=101.0,
            role_change_requirement="all",
            fallback_policy="reject",
        ),
    )

    assert result["synthetic_df"].empty
    assert result["metadata"]["fallback_success_count"] == 0


def test_random_and_random_fallback_apply_final_donor_similarity_threshold() -> None:
    roles = [
        ReactionRoles("CCBr", "N", "[Pd]", "P(C)(C)C", "N(C)(C)C", "CCO", "CCN"),
        ReactionRoles("CCCl", "N", "[Pt]", "P(CC)(CC)CC", "N1CCCCC1", "CCCO", "CCN"),
        ReactionRoles("CCCBr", "CN", "[Ni]", "P(C)(C)CC", "CCN(CC)CC", "CO", "CCCN"),
    ]
    reaction_similarity = np.array(
        [[-np.inf, 0.3, 0.9], [0.3, -np.inf, 0.1], [0.9, 0.1, -np.inf]]
    )
    substrate_similarity = reaction_similarity.copy()
    y = np.array([10.0, 20.0, 30.0])
    random_config = _config("random", "average_label", min_similarity=0.8)

    assert _select_donor_position(
        0,
        [0, 1, 2],
        y,
        reaction_similarity,
        substrate_similarity,
        roles,
        random_config,
        np.random.default_rng(1),
    ) == (2, 0.9, False)

    fallback_config = _config(
        "high_yield_nearest",
        "average_label",
        high_yield_threshold=100.0,
        fallback_policy="random",
        min_similarity=0.8,
    )
    assert _select_donor_position(
        0,
        [0, 1, 2],
        y,
        reaction_similarity,
        substrate_similarity,
        roles,
        fallback_config,
        np.random.default_rng(1),
    ) == (2, 0.9, True)


@pytest.mark.parametrize("donor_strategy", ["random", "nearest_reaction"])
def test_candidate_generation_is_permutation_stable_and_source_capped(
    donor_strategy: str,
) -> None:
    train = _tiny_train_df()
    permuted = train.sample(frac=1.0, random_state=17).reset_index(drop=True)

    def generate(frame: pd.DataFrame) -> dict[str, object]:
        X, y, _ = build_feature_matrix(frame, _feature_config())
        return generate_condition_transfer_examples(
            frame,
            X,
            y,
            _config(
                donor_strategy,
                "average_label",
                candidates_per_real=8,
                max_candidates_per_source=2,
            ),
        )

    original_result = generate(train)
    comparison_columns = [
        "source_reaction_smiles",
        "donor_reaction_smiles",
        "canonical_reaction_key",
        "candidate_rank",
        "kept",
    ]
    # Several fixture rows tie on substrate and product similarity, so the
    # donor choice must not depend on how a BLAS product rounds a tie under a
    # different row order. Check more than one permutation.
    for permutation_seed in (17, 3, 29, 101):
        permuted = train.sample(frac=1.0, random_state=permutation_seed).reset_index(drop=True)
        permuted_result = generate(permuted)
        pd.testing.assert_frame_equal(
            original_result["candidate_df"][comparison_columns].reset_index(drop=True),
            permuted_result["candidate_df"][comparison_columns].reset_index(drop=True),
        )

    candidates = original_result["candidate_df"]
    generated_counts = candidates.groupby("source_row_id").size()
    assert generated_counts.le(2).all()
    assert candidates["source_candidate_ordinal"].le(2).all()
    assert candidates["generated_per_source"].le(2).all()
    assert (candidates["max_candidates_per_source"] == 2).all()
    assert original_result["metadata"]["max_generated_per_source"] <= 2
    assert original_result["metadata"]["source_cap_violation_count"] == 0


def test_audited_candidates_have_phase3_ranking_and_support_fields() -> None:
    train = _tiny_train_df()
    X, y, _ = build_feature_matrix(train, _feature_config())
    result = generate_condition_transfer_examples(
        train,
        X,
        y,
        _config(
            "random",
            "average_label",
            synthetic_multiplier=0.5,
            candidates_per_real=4,
        ),
    )
    candidates = result["candidate_df"]
    required = {
        "substrate_similarity",
        "product_similarity",
        "nontransferred_role_similarity",
        "condition_similarity",
        "overall_similarity",
        "nearest_training_support_distance",
        "calibrated_uncertainty",
        "relevant_context_similarity",
        "diversity_contribution",
        "out_of_support_distance",
        "candidate_rank",
    }
    assert required <= set(candidates)
    assert candidates["calibrated_uncertainty"].isna().all()
    assert candidates["uncertainty_rank_value"].notna().all()
    assert candidates["uncertainty_rank_basis"].str.endswith("phase12_pending").all()
    assert np.allclose(
        candidates["nearest_training_support_distance"],
        candidates["out_of_support_distance"],
    )
    kept = candidates.loc[candidates["kept"]]
    assert sorted(kept["candidate_rank"].tolist()) == list(range(1, len(kept) + 1))


def test_policy_factory_propagates_role_requirement_and_fallback_exactly() -> None:
    policies = _iter_condition_transfer_policies(
        {
            "condition_transfer": {
                "donor_strategies": ["nearest_reaction"],
                "label_strategies": ["average_label"],
                "role_change_requirement": "any",
                "fallback_policy": "same_product",
            }
        },
        seed=7,
    )

    assert len(policies) == 1
    assert policies[0].role_change_requirement == "any"
    assert policies[0].fallback_policy == "same_product"


def test_policy_factory_defaults_primary_semantics_to_strict_reject() -> None:
    policies = _iter_condition_transfer_policies(
        {
            "condition_transfer": {
                "donor_strategies": ["random"],
                "label_strategies": ["average_label"],
            }
        },
        seed=7,
    )

    assert len(policies) == 1
    assert policies[0].role_change_requirement == "all"
    assert policies[0].fallback_policy == "reject"


def test_policy_factory_preserves_random_similarity_threshold() -> None:
    policies = _iter_condition_transfer_policies(
        {
            "condition_transfer": {
                "donor_strategies": ["random"],
                "label_strategies": ["average_label"],
                "min_similarities": [0.75],
            }
        },
        seed=7,
    )

    assert len(policies) == 1
    assert policies[0].min_similarity == 0.75


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role_change_requirement", "sometimes"),
        ("fallback_policy", "high_yield"),
    ],
)
def test_role_change_and_fallback_config_reject_unsupported_values(
    field: str,
    value: str,
) -> None:
    config = _config(donor_strategy="random", label_strategy="average_label")
    setattr(config, field, value)

    with pytest.raises(ValueError, match=field):
        _validate_config(config)


@pytest.mark.parametrize(
    ("requested_roles", "message"),
    [
        ([], "nonempty"),
        (["ligand", "ligand"], "duplicates"),
        (["product"], "unsupported"),
        ("ligand", "nonempty tuple or list"),
    ],
)
def test_requested_roles_reject_invalid_or_ambiguous_values(
    requested_roles: object,
    message: str,
) -> None:
    config = _config(donor_strategy="random", label_strategy="average_label")
    config.requested_roles = requested_roles  # type: ignore[assignment]

    with pytest.raises(ValueError, match=message):
        _validate_config(config)


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
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCBr.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCN",
            ],
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
  kind: bh_role_separated
  n_bits: 8
  radius: 2
  fingerprint_backend: hash
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
  role_change_requirement: any
  fallback_policy: reject
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
    candidate_audit = pd.read_csv(paths["synthetic_candidate_audit_path"])
    assert paths["selected_policies_path"].exists()
    assert paths["summary_path"].exists()
    assert "bh_role_separated_real_only" in set(policy_metrics["representation"])
    assert "anonymous_condition_transfer_bh_role_separated" in set(
        policy_metrics["representation"]
    )
    assert set(policy_metrics["split"]) == {"valid", "test"}
    assert set(selected["split"]) == {"valid", "test"}
    assert not audit["used_validation_or_test_parents"].any()
    assert (audit["n_synthetic_train"] > 0).any()
    assert set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(candidate_audit)
    assert not candidate_audit[["source_row_id", "donor_row_id"]].isna().any().any()


def _feature_config() -> dict[str, object]:
    return {"kind": "bh_role_separated", "n_bits": 8, "radius": 2, "fingerprint_backend": "hash"}


def _config(
    donor_strategy: str,
    label_strategy: str,
    synthetic_multiplier: float = 1.0,
    candidates_per_real: int = 10,
    high_yield_threshold: float = 70.0,
    max_teacher_std: float | None = None,
    role_change_requirement: str = "any",
    fallback_policy: str = "reject",
    min_similarity: float | None = None,
    max_candidates_per_source: int | None = None,
    requested_roles: tuple[str, ...] | list[str] | None = None,
) -> ConditionTransferConfig:
    return ConditionTransferConfig(
        donor_strategy=donor_strategy,
        synthetic_multiplier=synthetic_multiplier,
        n_neighbors=3,
        label_strategy=label_strategy,
        teacher_models=["ridge", "random_forest"],
        max_teacher_std=max_teacher_std,
        min_similarity=min_similarity,
        high_yield_threshold=high_yield_threshold,
        clip_y_min=0.0,
        clip_y_max=100.0,
        candidates_per_real=candidates_per_real,
        random_state=4,
        role_change_requirement=role_change_requirement,
        fallback_policy=fallback_policy,
        max_candidates_per_source=max_candidates_per_source,
        requested_roles=requested_roles,
    )


def _tiny_train_df(n_rows: int = 6) -> pd.DataFrame:
    base = [
        ("r1", "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN", 20.0),
        ("r2", "CCCl.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCN", 85.0),
        ("r3", "CCCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCOC>>CCCN", 75.0),
        ("r4", "CCCCl.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCO>>CCCN", 35.0),
        ("r5", "CCBr.CN.[Pd].P(C)(C)C.N1CCCCC1.CCCO>>CCNC", 90.0),
        ("r6", "CCCl.CN.[Pd].P(CC)(CC)CC.N(C)(C)C.CCOC>>CCNC", 45.0),
    ]
    rows = [base[index % len(base)] for index in range(n_rows)]
    return pd.DataFrame(
        {
            "reaction_id": [f"{reaction_id}_{index}" for index, (reaction_id, _, _) in enumerate(rows)],
            "reaction_smiles": [reaction for _, reaction, _ in rows],
            "yield": [float((yield_value + index) % 101) for index, (_, _, yield_value) in enumerate(rows)],
        }
    )


def _strict_role_change_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": ["strict_1", "strict_2", "strict_3"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCCl.N.[Pt].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCN",
                "CCCBr.CN.[Ni].P(C)(C)CC.CCN(CC)CC.CO>>CCCNC",
            ],
            "yield": [20.0, 50.0, 80.0],
        }
    )
