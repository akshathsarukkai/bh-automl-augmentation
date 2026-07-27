"""Scientific contracts for the matched Phase 11 augmentation benchmark."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation_control_benchmark as benchmark
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ROLE_TO_COLUMN,
    ReactionRoles,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

EXPECTED_CONTROLS = (
    "real_only",
    "exact_duplication",
    "random_oversampling",
    "yield_stratified_oversampling",
    "sample_reweighting",
    "nearest_neighbor_pseudo_labeling",
    "self_training",
    "feature_mixup",
    "anonymous_condition_transfer",
    "random_typed_transfer",
    "strict_context_matched_typed_transfer",
    "typed_transfer_without_uncertainty_filtering",
    "typed_transfer_with_uncertainty_filtering",
)
EXPECTED_COMPARATORS = (
    "real_only",
    "exact_duplication",
    "random_oversampling",
    "yield_stratified_oversampling",
)


def _unit(name: str = "random_seed_0_fraction_1") -> SimpleNamespace:
    return SimpleNamespace(
        evaluation_unit=name,
        seed=0,
        train_fraction=1.0,
        train_source_ids=("train-a", "train-b"),
        test_source_ids=("test-a", "test-b"),
        exact_split_hash="exact-split",
        aggregate_assignment_hash="aggregate-split",
        audit_record={"evaluation_unit": name},
    )


def _tables(
    *,
    unit: SimpleNamespace | None = None,
    metric_names: tuple[str, ...] = ("rmse", "r2"),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    split = _unit() if unit is None else unit
    base_seed = 0
    unit_model_seed = base_seed + (
        int(stable_hash(split.evaluation_unit)[:12], 16) % 1_000_000
    )
    common = {
        "exact_split_hash": split.exact_split_hash,
        "aggregate_assignment_hash": split.aggregate_assignment_hash,
        "feature_metadata_hash": "feature-hash",
        "model_protocol_hash": "model-hash",
        "budget_protocol_hash": "budget-hash",
        "model_seed": unit_model_seed,
        "plan_hash": "plan-hash",
    }
    metric_rows = []
    for control_number, control_id in enumerate(EXPECTED_CONTROLS):
        for metric_name in metric_names:
            metric_rows.append(
                {
                    "evaluation_unit": split.evaluation_unit,
                    "seed": split.seed,
                    "train_fraction": split.train_fraction,
                    "control_id": control_id,
                    "metric": metric_name,
                    "value": (
                        float(control_number + 1)
                        if metric_name in {"rmse", "mae"}
                        else float(control_number) / 20.0
                    ),
                    "test_evaluation_count": 1,
                    "test_used_for_selection": False,
                    **common,
                }
            )
    budget_rows = []
    for control_id in EXPECTED_CONTROLS:
        is_real_only = control_id == "real_only"
        is_reweighting = control_id == "sample_reweighting"
        budget_rows.append(
            {
                "evaluation_unit": split.evaluation_unit,
                "seed": split.seed,
                "train_fraction": split.train_fraction,
                "control_id": control_id,
                "policy_selection_budget": 1,
                "nominal_added_sample_count": 5,
                "augmentation_budget_applicable": not is_real_only,
                "budget_measure": (
                    "not_applicable"
                    if is_real_only
                    else ("added_weight" if is_reweighting else "added_rows")
                ),
                "budget_effective_units": 0.0 if is_real_only else 5.0,
                "effective_added_sample_count": (
                    0 if is_real_only or is_reweighting else 5
                ),
                "effective_added_weight": 0.0 if is_real_only else 5.0,
                "final_train_row_count": (
                    2 if is_real_only or is_reweighting else 7
                ),
                "real_sample_weight": 2.0,
                "synthetic_or_added_sample_weight": (
                    0.0 if is_real_only else 5.0
                ),
                "budget_underfill_count": 0,
                "budget_underfill_reason": None,
                "budget_utilization": 0.0 if is_real_only else 1.0,
                "budget_backfill_performed": False,
                "used_validation_or_test_sources": False,
                "total_sample_weight": 2.0 if is_real_only else 7.0,
                "prefilter_pool_hash": "shared-pool",
                "uncertainty_semantics": (
                    benchmark._RAW_UNCERTAINTY_SEMANTICS
                    if control_id
                    in {
                        "typed_transfer_without_uncertainty_filtering",
                        "typed_transfer_with_uncertainty_filtering",
                    }
                    else None
                ),
                **common,
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(budget_rows),
        {
            "metrics": list(metric_names),
            "model_protocol_hash": "model-hash",
            "budget_protocol_hash": "budget-hash",
            "feature_metadata_hash": "feature-hash",
            "plan_hash": "plan-hash",
            "resolved_scientific_config": {
                "base_seed": base_seed,
                "augmentation": {"nominal_added_multiplier": 2.5},
            },
        },
    )


def test_exact_thirteen_control_contract_and_order_are_frozen() -> None:
    assert benchmark.CONTROL_IDS == EXPECTED_CONTROLS
    assert benchmark.COMPARATOR_IDS == EXPECTED_COMPARATORS
    assert tuple(
        record["control_id"] for record in benchmark._control_plan_records()
    ) == EXPECTED_CONTROLS
    assert tuple(
        record["control_number"] for record in benchmark._control_plan_records()
    ) == tuple(range(1, 14))

    reordered = list(EXPECTED_CONTROLS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    config = {
        "dataset": {"path": "canonical.csv"},
        "splits": {
            "canonical_directory": "canonical-splits",
            "random_seeds": [0],
            "random_fractions": [1.0],
        },
        "controls": reordered,
        "augmentation": {},
    }
    with pytest.raises(ValueError, match="exact 13 controls"):
        benchmark._resolve_contract(config)


@pytest.mark.parametrize(
    "field",
    ("exact_split_hash", "model_protocol_hash", "budget_protocol_hash"),
)
def test_paired_gate_rejects_method_specific_provenance_hashes(
    field: str,
) -> None:
    metrics, budgets, plan = _tables()
    metrics.loc[metrics["control_id"].eq("feature_mixup"), field] = "other-hash"

    with pytest.raises(ValueError, match="Unpaired Phase 11 metrics"):
        benchmark._assert_paired_gate(metrics, budgets, (_unit(),), plan)


def test_paired_gate_rejects_missing_method_hash_and_test_selection() -> None:
    metrics, budgets, plan = _tables()
    metrics.loc[
        metrics["control_id"].eq("feature_mixup"), "model_protocol_hash"
    ] = np.nan
    with pytest.raises(ValueError, match="Unpaired Phase 11 metrics"):
        benchmark._assert_paired_gate(metrics, budgets, (_unit(),), plan)

    metrics, budgets, plan = _tables()
    metrics.loc[
        metrics["control_id"].eq("feature_mixup"), "test_used_for_selection"
    ] = True
    with pytest.raises(ValueError, match="Unpaired Phase 11 metrics"):
        benchmark._assert_paired_gate(metrics, budgets, (_unit(),), plan)


def test_budget_gate_accepts_recorded_underfill_without_backfill() -> None:
    metrics, budgets, plan = _tables()
    row = budgets["control_id"].eq("anonymous_condition_transfer")
    budgets.loc[row, "budget_effective_units"] = 3.0
    budgets.loc[row, "effective_added_sample_count"] = 3
    budgets.loc[row, "budget_underfill_count"] = 2
    budgets.loc[row, "budget_underfill_reason"] = "eligible_pool_exhausted"
    budgets.loc[row, "budget_utilization"] = 0.6
    budgets.loc[row, "effective_added_weight"] = 3.0
    budgets.loc[row, "synthetic_or_added_sample_weight"] = 3.0
    budgets.loc[row, "total_sample_weight"] = 5.0
    budgets.loc[row, "final_train_row_count"] = 5

    benchmark._assert_paired_gate(metrics, budgets, (_unit(),), plan)


def test_budget_gate_links_effective_units_to_effective_added_rows() -> None:
    metrics, budgets, plan = _tables()
    row = budgets["control_id"].eq("anonymous_condition_transfer")
    budgets.loc[row, "budget_effective_units"] = 4.0
    budgets.loc[row, "budget_underfill_count"] = 1
    budgets.loc[row, "budget_underfill_reason"] = "eligible_pool_exhausted"
    budgets.loc[row, "budget_utilization"] = 0.8

    with pytest.raises(ValueError, match="budget arithmetic"):
        benchmark._assert_paired_gate(metrics, budgets, (_unit(),), plan)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("effective_added_sample_count", 6),
        ("budget_underfill_count", 1),
        ("budget_utilization", 0.8),
        ("budget_backfill_performed", True),
    ),
)
def test_budget_gate_rejects_invalid_arithmetic_or_backfill(
    field: str,
    value: object,
) -> None:
    metrics, budgets, plan = _tables()
    budgets.loc[
        budgets["control_id"].eq("anonymous_condition_transfer"), field
    ] = value

    with pytest.raises(ValueError, match="Phase 11"):
        benchmark._assert_paired_gate(metrics, budgets, (_unit(),), plan)


class _NoSampleWeight:
    def fit(self, X: np.ndarray, y: np.ndarray) -> _NoSampleWeight:
        return self


class _KwargsButRejectsSampleWeight:
    def fit(
        self, X: np.ndarray, y: np.ndarray, **kwargs: object
    ) -> _KwargsButRejectsSampleWeight:
        if "sample_weight" in kwargs:
            raise TypeError("sample_weight is unsupported")
        return self


@pytest.mark.parametrize(
    "estimator",
    (_NoSampleWeight(), _KwargsButRejectsSampleWeight()),
)
def test_sample_weight_unsupported_is_a_hard_failure(estimator: object) -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        benchmark._fit_control_estimator(
            estimator,
            np.ones((2, 1), dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )


def test_paired_comparisons_include_required_controls_and_metric_direction() -> None:
    metrics, budgets, _ = _tables()
    paired = benchmark._paired_comparisons(metrics, budgets)

    assert tuple(dict.fromkeys(paired["comparator_id"])) == EXPECTED_COMPARATORS
    assert len(paired) == len(EXPECTED_CONTROLS) * 4 * 2
    selected = paired.loc[
        paired["control_id"].eq("feature_mixup")
        & paired["comparator_id"].eq("real_only")
    ].set_index("metric")
    assert selected.loc["rmse", "delta_control_minus_comparator"] == 7.0
    assert selected.loc["rmse", "improvement_over_comparator"] == -7.0
    assert selected.loc["r2", "delta_control_minus_comparator"] == pytest.approx(
        0.35
    )
    assert selected.loc["r2", "improvement_over_comparator"] == pytest.approx(
        0.35
    )


def test_summary_keeps_training_fractions_separate() -> None:
    first_metrics, first_budgets, _ = _tables(
        unit=_unit("random_seed_0_fraction_0.5"),
        metric_names=("rmse",),
    )
    first_metrics["train_fraction"] = 0.5
    first_budgets["train_fraction"] = 0.5
    second_metrics, second_budgets, _ = _tables(metric_names=("rmse",))
    second_metrics["value"] += 100.0
    metrics = pd.concat([first_metrics, second_metrics], ignore_index=True)
    budgets = pd.concat([first_budgets, second_budgets], ignore_index=True)

    summary = benchmark._summarize(
        metrics, benchmark._paired_comparisons(metrics, budgets)
    )

    assert set(summary["train_fraction"]) == {0.5, 1.0}
    real_only = summary.loc[
        summary["control_id"].eq("real_only")
        & summary["comparator_id"].eq("real_only")
    ].set_index("train_fraction")
    assert real_only.loc[0.5, "mean"] == 1.0
    assert real_only.loc[1.0, "mean"] == 101.0


def test_prediction_replay_recomputes_metrics_and_membership(tmp_path) -> None:
    split = _unit()
    dataset = tmp_path / "canonical.csv"
    pd.DataFrame(
        {
            "source_row_id": ["test-b", "test-a"],
            "yield": [20.0, 10.0],
        }
    ).to_csv(dataset, index=False)
    predictions = pd.DataFrame(
        [
            {
                "evaluation_unit": split.evaluation_unit,
                "control_id": control_id,
                "source_row_id": source_id,
                "prediction": prediction,
            }
            for control_id in EXPECTED_CONTROLS
            for source_id, prediction in (("test-b", 19.0), ("test-a", 11.0))
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "evaluation_unit": split.evaluation_unit,
                "control_id": control_id,
                "metric": "rmse",
                "value": 1.0,
            }
            for control_id in EXPECTED_CONTROLS
        ]
    )
    manifest = {"prediction_row_count": len(predictions)}

    benchmark._assert_prediction_replay(
        predictions.copy(),
        metrics,
        (split,),
        dataset,
        ["rmse"],
        manifest,
    )

    wrong_membership = predictions.copy()
    wrong_membership.loc[0, "source_row_id"] = "foreign-test"
    with pytest.raises(ValueError, match="membership"):
        benchmark._assert_prediction_replay(
            wrong_membership,
            metrics,
            (split,),
            dataset,
            ["rmse"],
            manifest,
        )

    wrong_metric = metrics.copy()
    wrong_metric.loc[0, "value"] = 2.0
    with pytest.raises(ValueError, match="metric replay"):
        benchmark._assert_prediction_replay(
            predictions.copy(),
            wrong_metric,
            (split,),
            dataset,
            ["rmse"],
            manifest,
        )


def _canonical_fixture_frame() -> pd.DataFrame:
    roles = (
        ReactionRoles("CBr", "N", "[Pd]", "P", "O", "CO", "CN"),
        ReactionRoles("CBr", "N", "[Pd]", "CP", "N", "CCO", "CN"),
        ReactionRoles("CCBr", "N", "[Pd]", "P", "O", "CO", "CCN"),
        ReactionRoles("CCCBr", "N", "[Pd]", "CP", "N", "CCO", "CCCN"),
    )
    records = []
    for source_id, yield_value, raw_roles in zip(
        ("train-a", "train-b", "test-a", "test-b"),
        (10.0, 20.0, 15.0, 25.0),
        roles,
        strict=True,
    ):
        canonical_roles = {
            role: benchmark.canonicalize_smiles(
                getattr(raw_roles, role), isomeric=True
            ).canonical_smiles
            for role in CANONICAL_ROLE_NAMES
        }
        identity = benchmark.build_canonical_reaction_identity(
            {
                "all_required_roles_parse_valid": True,
                **{
                    f"canonical_{role}_smiles": molecule
                    for role, molecule in canonical_roles.items()
                },
            }
        )
        canonical = ReactionRoles(**canonical_roles)
        records.append(
            {
                "source_row_id": source_id,
                "yield": yield_value,
                "reaction_smiles": canonical.reaction_smiles(),
                "canonical_reaction_key": identity["key"],
                "canonical_reaction_hash": identity["hash"],
                **{
                    ROLE_TO_COLUMN[role]: molecule
                    for role, molecule in canonical_roles.items()
                },
                **{
                    f"canonical_{role}_smiles": molecule
                    for role, molecule in canonical_roles.items()
                },
            }
        )
    return pd.DataFrame(records)


def _write_validator_fixture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, SimpleNamespace]:
    root = tmp_path / "benchmark"
    split = _unit()
    dataset = tmp_path / "canonical.csv"
    canonical = _canonical_fixture_frame()
    canonical.to_csv(dataset, index=False)
    replay_saved = SimpleNamespace(
        canonical=canonical,
        dataset_hash=sha256_file(dataset),
        aggregate_split_hash="aggregate-dependency",
    )
    monkeypatch.setattr(
        benchmark,
        "load_saved_canonical_split_identities",
        lambda *args, **kwargs: replay_saved,
    )
    monkeypatch.setattr(
        benchmark, "build_saved_random_split_unit", lambda *args, **kwargs: split
    )
    monkeypatch.setattr(
        benchmark, "_assert_split_unit_table", lambda *args, **kwargs: None
    )
    config = {
        "dataset": {"path": str(dataset)},
        "splits": {
            "canonical_directory": str(tmp_path / "splits"),
            "random_seeds": [0],
            "random_fractions": [1.0],
        },
        "controls": list(EXPECTED_CONTROLS),
        "augmentation": {
            "nominal_added_multiplier": 0.5,
            "n_yield_strata": 2,
            "mixup_alpha": 0.25,
            "self_training_teacher": {
                "name": "ridge",
                "params": {"alpha": 1.0},
            },
            "chemical_teacher_models": ["ridge"],
            "raw_teacher_std_threshold": 100.0,
        },
        "features": {
            "kind": "bh_role_separated",
            "n_bits": 8,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "model": {"name": "ridge", "params": {"alpha": 1.0}},
        "metrics": ["rmse"],
        "base_seed": 0,
    }
    benchmark.run_augmentation_control_benchmark(
        config, output_directory=root
    )
    return root, split


def _attacker_rehash(
    root,
    output_name: str,
    *,
    manifest_updates: dict[str, object] | None = None,
) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_hashes"][output_name] = sha256_file(root / output_name)
    if manifest_updates is not None:
        manifest.update(manifest_updates)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_validator_rejects_semantic_budget_tamper_after_attacker_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    benchmark.validate_augmentation_control_benchmark(root)

    budgets = pd.read_csv(root / "budget_audit.csv")
    budgets.loc[0, "effective_added_sample_count"] = 6
    budgets.to_csv(root / "budget_audit.csv", index=False)
    _attacker_rehash(root, "budget_audit.csv")

    with pytest.raises(ValueError, match="budget arithmetic"):
        benchmark.validate_augmentation_control_benchmark(root)


@pytest.mark.parametrize(
    ("audit_name", "erased_control", "manifest_count", "error"),
    (
        (
            "training_audit.csv",
            "feature_mixup",
            "training_audit_row_count",
            "training audit control coverage",
        ),
        (
            "chemical_candidate_audit.csv",
            "anonymous_condition_transfer",
            "chemical_candidate_audit_row_count",
            "chemical audit control coverage",
        ),
    ),
)
def test_validator_rejects_audit_erasure_after_attacker_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    audit_name: str,
    erased_control: str,
    manifest_count: str,
    error: str,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    audit = pd.read_csv(root / audit_name)
    audit = audit.loc[~audit["control_id"].eq(erased_control)]
    audit.to_csv(root / audit_name, index=False)
    _attacker_rehash(
        root,
        audit_name,
        manifest_updates={manifest_count: len(audit)},
    )

    with pytest.raises(ValueError, match=error):
        benchmark.validate_augmentation_control_benchmark(root)


def test_validator_rejects_metric_budget_provenance_tamper_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    metrics = pd.read_csv(root / "metrics.csv")
    metrics.loc[
        metrics["control_id"].eq("feature_mixup"), "fit_instance_hash"
    ] = hashlib.sha256(b"forged-fit-instance").hexdigest()
    metrics.to_csv(root / "metrics.csv", index=False)
    _attacker_rehash(root, "metrics.csv")

    with pytest.raises(ValueError, match="metric-budget linkage"):
        benchmark.validate_augmentation_control_benchmark(root)


def test_audited_training_hash_survives_csv_roundtrip(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    audit = pd.read_csv(root / "training_audit.csv")
    budgets = pd.read_csv(root / "budget_audit.csv").set_index(
        ["evaluation_unit", "control_id"]
    )

    for key, rows in audit.groupby(
        ["evaluation_unit", "control_id"], sort=False
    ):
        assert benchmark._audited_training_hash(rows) == budgets.loc[
            key, "augmented_training_hash"
        ]


def test_validator_rejects_uncertainty_description_bypass_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    plan_path = root / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    plan.pop("plan_hash")
    plan["uncertainty_filter_semantics"] = (
        "calibrated uncertainty suitable for scientific claims"
    )
    plan["plan_hash"] = stable_hash(plan)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    _attacker_rehash(
        root,
        "benchmark_plan.json",
        manifest_updates={"plan_hash": plan["plan_hash"]},
    )

    with pytest.raises(ValueError, match="frozen plan mismatch"):
        benchmark.validate_augmentation_control_benchmark(root)


def test_validator_rejects_raw_uncertainty_semantics_bypass_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    budgets = pd.read_csv(root / "budget_audit.csv")
    budgets.loc[
        budgets["control_id"].eq(
            "typed_transfer_without_uncertainty_filtering"
        ),
        "uncertainty_semantics",
    ] = "calibrated_conformal_interval"
    budgets.to_csv(root / "budget_audit.csv", index=False)
    _attacker_rehash(root, "budget_audit.csv")

    with pytest.raises(ValueError, match="budget arithmetic"):
        benchmark.validate_augmentation_control_benchmark(root)


def test_validator_rejects_chemical_similarity_config_bypass_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, split = _write_validator_fixture(tmp_path, monkeypatch)
    plan = json.loads((root / "benchmark_plan.json").read_text())
    budgets = pd.read_csv(root / "budget_audit.csv")
    control = budgets["control_id"].eq(
        "strict_context_matched_typed_transfer"
    )
    budget_index = budgets.index[control][0]
    resolved = json.loads(
        budgets.loc[budget_index, "resolved_control_config"]
    )
    resolved["generator_config"]["min_similarity"] = 0.5
    control_hash = stable_hash(resolved)
    fit_hash = stable_hash(
        {
            "model_protocol_hash": plan["model_protocol_hash"],
            "model_seed": int(budgets.loc[budget_index, "model_seed"]),
            "exact_split_hash": split.exact_split_hash,
            "control_config_hash": control_hash,
            "augmented_training_hash": budgets.loc[
                budget_index, "augmented_training_hash"
            ],
        }
    )
    budgets.loc[budget_index, "resolved_control_config"] = json.dumps(
        resolved, sort_keys=True
    )
    budgets.loc[budget_index, "control_config_hash"] = control_hash
    budgets.loc[budget_index, "fit_instance_hash"] = fit_hash
    budgets.to_csv(root / "budget_audit.csv", index=False)
    metrics = pd.read_csv(root / "metrics.csv")
    metrics.loc[control, "fit_instance_hash"] = fit_hash
    metrics.to_csv(root / "metrics.csv", index=False)
    _attacker_rehash(root, "budget_audit.csv")
    _attacker_rehash(root, "metrics.csv")

    with pytest.raises(ValueError, match="chemical control config"):
        benchmark.validate_augmentation_control_benchmark(root)


def test_validator_rejects_canonical_candidate_identity_bypass_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    chemical = pd.read_csv(root / "chemical_candidate_audit.csv")
    candidate = chemical["control_id"].eq("random_typed_transfer") & chemical[
        "accepted"
    ]
    candidate_index = chemical.index[candidate][0]
    candidate_id = int(chemical.loc[candidate_index, "candidate_id"])
    forged_key = chemical.loc[candidate_index, "canonical_reaction_key"] + " "
    chemical.loc[candidate_index, "canonical_reaction_key"] = forged_key
    chemical.loc[
        candidate_index, "canonical_reaction_hash"
    ] = hashlib.sha256(forged_key.encode()).hexdigest()
    chemical.to_csv(root / "chemical_candidate_audit.csv", index=False)

    training = pd.read_csv(root / "training_audit.csv")
    training_candidate = training["control_id"].eq(
        "random_typed_transfer"
    ) & pd.to_numeric(training["candidate_id"], errors="coerce").eq(
        candidate_id
    )
    training.loc[
        training_candidate, "canonical_reaction_key"
    ] = forged_key
    training.loc[
        training_candidate, "canonical_reaction_hash"
    ] = hashlib.sha256(forged_key.encode()).hexdigest()
    training.to_csv(root / "training_audit.csv", index=False)

    budgets = pd.read_csv(root / "budget_audit.csv")
    budget = budgets["control_id"].eq("random_typed_transfer")
    budgets.loc[
        budget, "accepted_candidate_hash"
    ] = benchmark._chemical_accepted_candidate_hash(
        chemical.loc[chemical["control_id"].eq("random_typed_transfer")]
    )
    budgets.to_csv(root / "budget_audit.csv", index=False)
    _attacker_rehash(root, "chemical_candidate_audit.csv")
    _attacker_rehash(root, "training_audit.csv")
    _attacker_rehash(root, "budget_audit.csv")

    with pytest.raises(ValueError, match="canonical identity replay"):
        benchmark.validate_augmentation_control_benchmark(root)


def test_validator_rejects_role_change_audit_bypass_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    chemical = pd.read_csv(root / "chemical_candidate_audit.csv")
    candidate = chemical["control_id"].eq("random_typed_transfer") & chemical[
        "accepted"
    ]
    chemical.loc[candidate, "actual_changed_roles"] = "ligand"
    chemical.to_csv(root / "chemical_candidate_audit.csv", index=False)
    _attacker_rehash(root, "chemical_candidate_audit.csv")

    with pytest.raises(ValueError, match="role-change replay"):
        benchmark.validate_augmentation_control_benchmark(root)


def test_validator_rejects_nonchemical_disclaimer_erasure_after_rehash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _write_validator_fixture(tmp_path, monkeypatch)
    metadata = pd.read_csv(root / "control_metadata.csv")
    metadata.loc[
        metadata["control_id"].eq("feature_mixup"),
        "nonchemical_semantics",
    ] = ""
    metadata.to_csv(root / "control_metadata.csv", index=False)
    _attacker_rehash(root, "control_metadata.csv")

    with pytest.raises(ValueError, match="metadata semantics"):
        benchmark.validate_augmentation_control_benchmark(root)
