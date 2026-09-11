"""Tests for the preregistered observed-only confirmatory analysis.

The constants are asserted against ``PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md``
as committed at ``c6aab9e`` -- read out of git, not from the working tree -- so
the analysis cannot drift from the frozen document without this file failing.
The end-to-end cases build a synthetic confirmatory tree with the real artifact
schemas and the real hash functions.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bh_augmentation.observed_only_confirmatory_analysis import (
    ARM_FAMILIES,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPARATOR_FAMILY,
    MAXIMUM_DEGENERATE_UNITS,
    MINIMUM_IMPROVED_UNITS_FOR_POSITIVE,
    MINIMUM_INCLUDED_UNITS,
    PLANNED_SEEDS,
    POOL_COUNT_FIELDS,
    PRACTICAL_MINIMUM_EFFECT,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT,
    PRIMARY_FAMILY,
    PRIMARY_METRIC,
    PRIMARY_SCOPE_MODE,
    PRIMARY_TRAIN_FRACTION,
    SECONDARY_CONTROL_FAMILY,
    SECONDARY_CONTROL_SCOPE_MODE,
    ObservedOnlyConfirmatoryAnalysisError,
    analyze_observed_only_confirmation,
    evaluation_unit,
    interpretation_table_verdict,
    load_arm_unit,
    preregistration_plan_hash,
    render_observed_only_report,
)
from bh_augmentation.policy_search import (
    DEGENERATE_POOL_MESSAGE,
    DEGENERATE_UNIT_SCHEMA_VERSION,
    DEGENERATE_UNIT_STATUS,
)
from bh_augmentation.utils.corrected_runs import stable_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "corrected_observed_only_transfer"
DATASET_HASH = "df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365"


def _preregistration_text() -> str:
    return subprocess.run(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:{PREREGISTRATION_DOCUMENT}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


# ---------------------------------------------------------------- fixtures


def _binding(seed: int, family: str) -> dict:
    return {
        "canonicalization_version": "bh-rdkit-isomeric-v1",
        "commit_hash": "0" * 40,
        "config_hash": stable_hash({"family": family}),
        "dataset_hash": DATASET_HASH,
        "feature_metadata_hash": "f" * 64,
        "per_seed_split_hash": stable_hash({"seed": seed}),
        "source_id_split_hash": "5" * 64,
        "split_aggregate_hash": "a" * 64,
        "split_hash": stable_hash({"seed": seed}),
        "split_schema_version": "bh-canonical-grouped-splits-v1",
    }


def _scientific_config(family: str) -> dict:
    mode = (
        SECONDARY_CONTROL_SCOPE_MODE
        if family == SECONDARY_CONTROL_FAMILY
        else PRIMARY_SCOPE_MODE
    )
    return {
        "candidate_scope": {"mode": mode},
        "dataset": {"path": "data/processed/bh_canonical_roles_v1.csv"},
        "low_data": {"train_fractions": [PRIMARY_TRAIN_FRACTION]},
    }


def write_complete_unit(
    root: Path, family: str, seed: int, rmse: float, *, policies: int = 3
) -> Path:
    unit_dir = root / family / f"seed_{seed}"
    (unit_dir / "search").mkdir(parents=True)
    (unit_dir / "final").mkdir()
    unit = evaluation_unit(seed)
    frozen_hash = stable_hash({"frozen": family, "seed": seed})
    search_hash = stable_hash({"search": family, "seed": seed})
    method = "real_only" if family == COMPARATOR_FAMILY else "anonymous"
    (unit_dir / "search" / "frozen_policy.json").write_text(
        json.dumps(
            {
                "binding": _binding(seed, family),
                "frozen_policy_hash": frozen_hash,
                "policy_id": f"{method}-xgb-1",
                "search_manifest_hash": search_hash,
                "schema_version": "bh-frozen-policy-v1",
                "status": "frozen",
            }
        )
    )
    (unit_dir / "search" / "search_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bh-policy-search-manifest-v1",
                "status": "complete",
                "payload": {
                    "evaluation_unit": unit,
                    "candidate_policy_hashes": ["1" * 64] * policies,
                    "scientific_binding": _binding(seed, family),
                    "scientific_config": _scientific_config(family),
                },
                "search_manifest_hash": search_hash,
                "frozen_policy_hash": frozen_hash,
            }
        )
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "metric": metric,
                "value": rmse if metric == "rmse" else 0.5,
                "evaluation_unit": unit,
                "frozen_policy_hash": frozen_hash,
                "seed": seed,
                "train_fraction": PRIMARY_TRAIN_FRACTION,
                "method": method,
                "model": "xgboost",
                "n_refit": 159,
                "n_test": 396,
                "test_prediction_batch": 1,
            }
            for metric in ("rmse", "mae")
        ]
    ).to_csv(unit_dir / "final" / "final_test_metrics.csv", index=False)
    (unit_dir / "final" / "evaluation_claim.json").write_text(
        json.dumps(
            {
                "evaluation_unit": unit,
                "frozen_policy_hash": frozen_hash,
                "outer_test_prediction_batches": 1,
                "schema_version": "bh-final-evaluation-claim-v1",
                "scientific_binding": _binding(seed, family),
                "status": "complete",
            }
        )
    )
    (unit_dir / "final" / "final_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bh-final-evaluation-manifest-v1",
                "status": "complete",
                "payload": {
                    "evaluation_unit": unit,
                    "frozen_policy_hash": frozen_hash,
                    "outer_test_prediction_batches": 1,
                    "per_unit_test_evaluation_counts": {unit: 1},
                    "selected_method": method,
                },
                "final_evaluation_manifest_hash": "e" * 64,
            }
        )
    )
    return unit_dir


def write_degenerate_unit(root: Path, family: str, seed: int, *, generated: int = 318) -> Path:
    unit_dir = root / family / f"seed_{seed}"
    (unit_dir / "search").mkdir(parents=True)
    mode = (
        SECONDARY_CONTROL_SCOPE_MODE
        if family == SECONDARY_CONTROL_FAMILY
        else PRIMARY_SCOPE_MODE
    )
    stats = {field: 0 for field in POOL_COUNT_FIELDS}
    stats["generated_candidate_count"] = generated
    stats["rejected_already_measured"] = generated
    stats["candidate_scope_mode"] = mode
    payload = {
        "run_type": "policy_search_degenerate_pool",
        "scientific_binding": _binding(seed, family),
        "scientific_config": _scientific_config(family),
        "evaluation_unit": evaluation_unit(seed),
        "seed": seed,
        "train_fraction": PRIMARY_TRAIN_FRACTION,
        "family": family,
        "candidate_scope_mode": mode,
        "candidate_policy_hashes": ["2" * 64] * 3,
        "per_policy": [
            {
                "policy_id": f"anonymous-xgb-{index}",
                "policy_hash": "2" * 64,
                "method": "anonymous",
                "model": "xgboost",
                "pool_statistics": dict(stats),
            }
            for index in range(3)
        ],
        "reason": DEGENERATE_POOL_MESSAGE,
        "n_train": 159,
        "n_validation": 395,
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "test_evaluated": False,
        "frozen_policy_written": False,
        "registry_claimed": False,
    }
    (unit_dir / "search" / "degenerate_unit.json").write_text(
        json.dumps(
            {
                "schema_version": DEGENERATE_UNIT_SCHEMA_VERSION,
                "status": DEGENERATE_UNIT_STATUS,
                "payload": payload,
                "degenerate_unit_hash": stable_hash(payload),
            }
        )
    )
    return unit_dir


def write_pool_accounting(
    root: Path,
    *,
    observed_accepted: dict[int, int],
    global_accepted: dict[int, int],
    generated: int = 318,
    with_oracle: bool = True,
    outer_test_accessed: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode, accepted_by_seed in (
        (PRIMARY_SCOPE_MODE, observed_accepted),
        (SECONDARY_CONTROL_SCOPE_MODE, global_accepted),
    ):
        for kind in ("anonymous", "typed"):
            for seed, accepted in accepted_by_seed.items():
                row = {
                    "evaluation_unit": f"canonical-random:seed={seed}:fraction=0.05",
                    "seed": seed,
                    "train_fraction": PRIMARY_TRAIN_FRACTION,
                    "result_family": (
                        PRIMARY_FAMILY if mode == PRIMARY_SCOPE_MODE else SECONDARY_CONTROL_FAMILY
                    ),
                    "candidate_scope_mode": mode,
                    "transfer_kind": kind,
                }
                row.update({field: 0 for field in POOL_COUNT_FIELDS})
                row["generated_candidate_count"] = generated
                row["accepted_candidate_count"] = accepted
                row["unique_accepted_identity_count"] = accepted
                rejected = generated - accepted
                if mode == PRIMARY_SCOPE_MODE:
                    row["rejected_quarantined_held_out_identity"] = rejected
                else:
                    row["rejected_already_measured"] = rejected
                rows.append(row)
    pd.DataFrame(rows).to_csv(root / "candidate_pool_statistics.csv", index=False)
    (root / "leakage_contracts.json").write_text(
        json.dumps(
            {
                "outer_test_labels_accessed": outer_test_accessed,
                "evaluation_registry_claimed": False,
                "hidden_outer_train_read_stage": "after_candidate_and_pseudo_label_freeze",
                "hidden_outer_train_influences": [],
                "frozen_bundle_hashes": ["9" * 64],
            }
        )
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_hash": "m" * 64,
                "dataset_hash": DATASET_HASH,
                "split_directory": "results/corrected_canonical_splits_phase15",
            }
        )
    )
    if with_oracle:
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "transfer_kind": kind,
                    "train_fraction": PRIMARY_TRAIN_FRACTION,
                    "candidate_scope_mode": PRIMARY_SCOPE_MODE,
                    "generated_candidate_count": generated,
                    "unique_canonical_candidate_count": generated,
                    "hidden_overlap_count": generated - 1,
                    "hidden_overlap_fraction_of_candidates": 0.99,
                    "hidden_cell_coverage_fraction": 0.04,
                    "pseudo_label_mae": 11.0 + 0.1 * seed,
                    "pseudo_label_rmse": 16.0,
                    "pseudo_label_spearman": 0.77,
                    "pseudo_label_bias": 0.1,
                    "high_yield_precision": float("nan"),
                    "high_yield_recall": 0.2,
                    "high_yield_enrichment": 1.5,
                }
                for seed in observed_accepted
                for kind in ("anonymous", "typed")
            ]
        ).to_csv(root / "withheld_cell_oracle.csv", index=False)
    return root


def build_confirmatory_tree(
    tmp_path: Path,
    *,
    deltas: dict[int, float],
    control_complete_seeds: set[int] = frozenset({6}),
    degenerate_primary_seeds: set[int] = frozenset(),
    missing_primary_seeds: set[int] = frozenset(),
) -> tuple[Path, Path]:
    run_root = tmp_path / "corrected_run"
    confirmatory = run_root / "confirmatory"
    observed_accepted: dict[int, int] = {}
    global_accepted: dict[int, int] = {}
    for seed in PLANNED_SEEDS:
        comparator_rmse = 14.0 + 0.1 * (seed - 6)
        write_complete_unit(confirmatory, COMPARATOR_FAMILY, seed, comparator_rmse)
        if seed in degenerate_primary_seeds:
            write_degenerate_unit(confirmatory, PRIMARY_FAMILY, seed)
            observed_accepted[seed] = 0
        elif seed in missing_primary_seeds:
            observed_accepted[seed] = 300
        else:
            write_complete_unit(confirmatory, PRIMARY_FAMILY, seed, comparator_rmse - deltas[seed])
            observed_accepted[seed] = 300 + seed
        if seed in control_complete_seeds:
            write_complete_unit(confirmatory, SECONDARY_CONTROL_FAMILY, seed, comparator_rmse + 2.0)
            global_accepted[seed] = 1
        else:
            write_degenerate_unit(confirmatory, SECONDARY_CONTROL_FAMILY, seed)
            global_accepted[seed] = 0
    pool = write_pool_accounting(
        tmp_path / "corrected_pool",
        observed_accepted=observed_accepted,
        global_accepted=global_accepted,
    )
    return run_root, pool


# ------------------------------------------------------------------- tests


def test_constants_match_the_frozen_preregistration() -> None:
    text = _preregistration_text()
    assert "**Seeds 6 through 14 at training fraction 0.05**" in text
    assert PLANNED_SEEDS == tuple(range(6, 15))
    assert PRIMARY_TRAIN_FRACTION == 0.05
    assert "fixed bootstrap seed **1601**" in text
    assert BOOTSTRAP_SEED == 1601
    assert "10,000 replicates" in text
    assert BOOTSTRAP_REPLICATES == 10_000
    assert "95% percentile bootstrap" in text
    assert BOOTSTRAP_ALPHA == 0.05
    assert "**1.0 RMSE**" in text
    assert PRACTICAL_MINIMUM_EFFECT == 1.0
    assert "If more than 2 of 9 units are degenerate" in text
    assert MAXIMUM_DEGENERATE_UNITS == 2
    assert "If fewer than 7 units remain after exclusions" in text
    assert MINIMUM_INCLUDED_UNITS == 7
    assert "at least 7 of *n* units improve" in text
    assert MINIMUM_IMPROVED_UNITS_FOR_POSITIVE == 7
    assert "Outer-test RMSE (lower is better)" in text
    assert PRIMARY_METRIC == "rmse"
    assert "`candidate_scope.mode: observed_only_low_data`" in text
    assert PRIMARY_SCOPE_MODE == "observed_only_low_data"
    assert "`candidate_scope.mode: globally_unmeasured_prospective`" in text
    assert SECONDARY_CONTROL_SCOPE_MODE == "globally_unmeasured_prospective"
    assert "| Registry family | `observed_only_condition_transfer` |" in text
    assert PRIMARY_FAMILY == "observed_only_condition_transfer"
    assert "Mean paired RMSE reduction: comparator RMSE minus augmented RMSE" in text
    assert "Seed 5 at fraction 0.05 is deliberately excluded" in text
    assert 5 not in PLANNED_SEEDS


def test_plan_hash_is_the_frozen_document() -> None:
    import hashlib

    frozen = subprocess.run(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:{PREREGISTRATION_DOCUMENT}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert preregistration_plan_hash(ROOT) == hashlib.sha256(frozen).hexdigest()
    # The working-tree document must still be the frozen one.
    assert (ROOT / PREREGISTRATION_DOCUMENT).read_bytes() == frozen


def test_confirmatory_configs_encode_the_preregistered_design() -> None:
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) == len(ARM_FAMILIES) * len(PLANNED_SEEDS)
    budgets = set()
    for path in paths:
        config = yaml.safe_load(path.read_text())
        family = config["evaluation_registry"]["family"]
        (seed,) = config["seeds"]
        assert path.name == f"{family}_seed_{seed}.yaml"
        assert seed in PLANNED_SEEDS
        assert config["low_data"]["train_fractions"] == [PRIMARY_TRAIN_FRACTION]
        assert config["splits"]["directory"] == "results/corrected_canonical_splits_phase15"
        assert config["dataset"]["path"] == "data/processed/bh_canonical_roles_v1.csv"
        expected_mode = (
            SECONDARY_CONTROL_SCOPE_MODE
            if family == SECONDARY_CONTROL_FAMILY
            else PRIMARY_SCOPE_MODE
        )
        assert config["candidate_scope"]["mode"] == expected_mode
        policies = config["policy_search"]["model_policies"]
        budgets.add(len(policies))
        for policy in policies:
            assert policy["model"] == "xgboost"
            if family == COMPARATOR_FAMILY:
                assert policy["method"] == "real_only"
            else:
                assert policy["method"] == "anonymous"
                assert policy["transfer_config"]["random_state"] == seed
                assert policy["transfer_config"]["requested_roles"] == ["ligand", "base"]
    # Section 5: equal search budgets for every arm.
    assert budgets == {3}


def test_end_to_end_accounts_for_all_27_arm_units(tmp_path: Path) -> None:
    deltas = {seed: -0.4 + 0.05 * (seed - 6) for seed in PLANNED_SEEDS}
    run_root, pool = build_confirmatory_tree(tmp_path, deltas=deltas)

    payload = analyze_observed_only_confirmation(run_root, pool)

    accounting = payload["accounting"]
    assert accounting["planned_unit_count"] == 9
    assert accounting["included_unit_count"] == 9
    assert accounting["degenerate_excluded_unit_count"] == 0
    assert accounting["all_planned_seeds_accounted_for"] is True
    assert accounting["planned_arm_unit_pairs"] == 27
    assert accounting["all_arm_unit_pairs_accounted_for"] is True
    assert accounting["arm_unit_counts"][SECONDARY_CONTROL_FAMILY] == {
        "complete": 1,
        "degenerate": 8,
        "missing": 0,
    }
    assert payload["primary"]["mean_paired_rmse_reduction"] == pytest.approx(
        sum(deltas.values()) / 9
    )
    assert payload["primary"]["bootstrap"]["seed"] == BOOTSTRAP_SEED
    assert payload["primary"]["bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES
    assert payload["verdict"]["verdict"] == "null"
    assert payload["verdict"]["rule"] == "section_7_interpretation_table"
    assert payload["secondary"]["label"] == "secondary"
    assert payload["run"]["equal_arm_search_budget"] is True
    assert payload["run"]["dataset_hash"] == DATASET_HASH
    sizes = payload["treatment_size"]["accepted_synthetic_rows_by_seed"]
    assert sizes[PRIMARY_SCOPE_MODE]["7"] == 307
    assert sizes[SECONDARY_CONTROL_SCOPE_MODE]["7"] == 0
    assert sizes[SECONDARY_CONTROL_SCOPE_MODE]["6"] == 1
    control = payload["secondary_control_arm"]
    assert control["unit_status_counts"] == {"complete": 1, "degenerate": 8, "missing": 0}
    assert control["per_unit_rows"][1]["status"] == "degenerate"
    assert control["per_unit_rows"][1]["rejected_already_measured"] == 318
    oracle = payload["oracle"]
    assert oracle["label"] == "secondary_non_selecting"
    assert oracle["by_transfer_kind"]["anonymous"]["seeds"] == list(PLANNED_SEEDS)
    assert oracle["by_transfer_kind"]["anonymous"]["mean"]["high_yield_precision"] is None

    # Deterministic and self-hashing.
    again = analyze_observed_only_confirmation(run_root, pool)
    assert again == payload
    without_hash = {key: value for key, value in payload.items() if key != "analysis_hash"}
    assert stable_hash(without_hash) == payload["analysis_hash"]

    report = render_observed_only_report(payload)
    assert "NULL" in report
    for seed in PLANNED_SEEDS:
        assert f"| {seed} |" in report
    assert "degenerate" in report
    assert "secondary" in report.lower()
    assert "not OOD evidence" in report


def test_degenerate_primary_units_are_excluded_and_reported(tmp_path: Path) -> None:
    deltas = dict.fromkeys(PLANNED_SEEDS, 0.2)
    run_root, pool = build_confirmatory_tree(
        tmp_path, deltas=deltas, degenerate_primary_seeds={9, 12}
    )
    payload = analyze_observed_only_confirmation(run_root, pool)
    accounting = payload["accounting"]
    assert accounting["degenerate_excluded_seeds"] == [9, 12]
    assert accounting["included_unit_count"] == 7
    assert accounting["all_planned_seeds_accounted_for"] is True
    assert payload["verdict"]["rule"] == "section_7_interpretation_table"
    rows = {row["seed"]: row for row in payload["per_unit_rows"]}
    assert rows[9]["status"] == "excluded_degenerate_pool"
    assert rows[9]["augmented_rmse"] is None
    assert rows[9]["accepted_synthetic_rows_observed_only"] == 0


def test_more_than_two_degenerate_units_is_inconclusive_by_section_6(tmp_path: Path) -> None:
    deltas = dict.fromkeys(PLANNED_SEEDS, 3.0)
    run_root, pool = build_confirmatory_tree(
        tmp_path, deltas=deltas, degenerate_primary_seeds={7, 8, 9}
    )
    payload = analyze_observed_only_confirmation(run_root, pool)
    assert payload["verdict"]["verdict"] == "inconclusive"
    assert payload["verdict"]["rule"] == "section_6_degenerate_pool_override"


def test_missing_arm_is_recorded_not_dropped(tmp_path: Path) -> None:
    deltas = dict.fromkeys(PLANNED_SEEDS, 0.1)
    run_root, pool = build_confirmatory_tree(
        tmp_path, deltas=deltas, missing_primary_seeds={14}
    )
    payload = analyze_observed_only_confirmation(run_root, pool)
    accounting = payload["accounting"]
    assert accounting["failed_or_missing_seeds"] == [14]
    assert accounting["included_unit_count"] == 8
    assert accounting["all_planned_seeds_accounted_for"] is True
    assert accounting["arm_unit_counts"][PRIMARY_FAMILY]["missing"] == 1
    assert payload["verdict"]["verdict"] == "null"


def test_pool_accounting_disagreement_is_an_error(tmp_path: Path) -> None:
    deltas = dict.fromkeys(PLANNED_SEEDS, 0.1)
    run_root, pool = build_confirmatory_tree(tmp_path, deltas=deltas)

    frame = pd.read_csv(pool / "candidate_pool_statistics.csv")
    mask = (
        frame["candidate_scope_mode"].eq(PRIMARY_SCOPE_MODE)
        & frame["transfer_kind"].eq("anonymous")
        & frame["seed"].eq(8)
    )
    frame.loc[mask, "accepted_candidate_count"] = 0
    frame.to_csv(pool / "candidate_pool_statistics.csv", index=False)
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="zero accepted rows"):
        analyze_observed_only_confirmation(run_root, pool)

    run_root, pool = build_confirmatory_tree(tmp_path / "second", deltas=deltas)
    frame = pd.read_csv(pool / "candidate_pool_statistics.csv")
    mask = (
        frame["candidate_scope_mode"].eq(SECONDARY_CONTROL_SCOPE_MODE)
        & frame["transfer_kind"].eq("anonymous")
        & frame["seed"].eq(9)
    )
    frame.loc[mask, "generated_candidate_count"] = 5
    frame.to_csv(pool / "candidate_pool_statistics.csv", index=False)
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="generated"):
        analyze_observed_only_confirmation(run_root, pool)


def test_pool_accounting_that_touched_an_outer_test_is_refused(tmp_path: Path) -> None:
    deltas = dict.fromkeys(PLANNED_SEEDS, 0.1)
    run_root, pool = build_confirmatory_tree(tmp_path, deltas=deltas)
    contracts = json.loads((pool / "leakage_contracts.json").read_text())
    contracts["outer_test_labels_accessed"] = True
    (pool / "leakage_contracts.json").write_text(json.dumps(contracts))
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="outer-test access"):
        analyze_observed_only_confirmation(run_root, pool)


def test_inconsistent_unit_artifacts_raise(tmp_path: Path) -> None:
    confirmatory = tmp_path / "confirmatory"
    unit_dir = write_complete_unit(confirmatory, PRIMARY_FAMILY, 6, 14.0)
    claim_path = unit_dir / "final" / "evaluation_claim.json"
    claim = json.loads(claim_path.read_text())
    claim["outer_test_prediction_batches"] = 2
    claim_path.write_text(json.dumps(claim))
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="not once"):
        load_arm_unit(confirmatory, PRIMARY_FAMILY, 6)

    unit_dir = write_complete_unit(confirmatory, PRIMARY_FAMILY, 7, 14.0)
    frozen_path = unit_dir / "search" / "frozen_policy.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["frozen_policy_hash"] = "d" * 64
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="frozen_policy.json"):
        load_arm_unit(confirmatory, PRIMARY_FAMILY, 7)

    # A degenerate record filed under the wrong family is refused.
    write_degenerate_unit(confirmatory, SECONDARY_CONTROL_FAMILY, 8)
    wrong = confirmatory / PRIMARY_FAMILY / "seed_8" / "search"
    wrong.mkdir(parents=True)
    (wrong / "degenerate_unit.json").write_bytes(
        (confirmatory / SECONDARY_CONTROL_FAMILY / "seed_8" / "search" / "degenerate_unit.json")
        .read_bytes()
    )
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="declares family"):
        load_arm_unit(confirmatory, PRIMARY_FAMILY, 8)

    # Artifacts without a result or a degenerate record are an unrecognised state.
    (confirmatory / PRIMARY_FAMILY / "seed_9" / "search").mkdir(parents=True)
    (confirmatory / PRIMARY_FAMILY / "seed_9" / "search" / "search_metrics.csv").write_text("x")
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError, match="unrecognised"):
        load_arm_unit(confirmatory, PRIMARY_FAMILY, 9)

    # An empty directory is simply missing (the state the crash left behind).
    (confirmatory / PRIMARY_FAMILY / "seed_10").mkdir(parents=True)
    assert load_arm_unit(confirmatory, PRIMARY_FAMILY, 10)["status"] == "missing"


def _verdict(**kwargs):
    defaults = dict(
        included_unit_count=9,
        improved_unit_count=2,
        degenerate_unit_count=0,
        practical_minimum_effect=PRACTICAL_MINIMUM_EFFECT,
        maximum_degenerate_units=MAXIMUM_DEGENERATE_UNITS,
        minimum_included_units=MINIMUM_INCLUDED_UNITS,
        minimum_improved_units_for_positive=MINIMUM_IMPROVED_UNITS_FOR_POSITIVE,
        planned_unit_count=len(PLANNED_SEEDS),
        degenerate_override_rule="section_6_degenerate_pool_override",
        underpowered_override_rule="section_6_underpowered_override",
        table_rule="section_7_interpretation_table",
        error_type=ObservedOnlyConfirmatoryAnalysisError,
    )
    defaults.update(kwargs)
    return interpretation_table_verdict(**defaults)


def test_interval_excluding_zero_inside_the_margin_is_null_not_negative() -> None:
    verdict = _verdict(mean_effect=-0.38, ci_lower=-0.75, ci_upper=-0.05)
    assert verdict["verdict"] == "null"
    assert verdict["conditions"]["negative"] is False


def test_negative_wide_and_underpowered_outcomes() -> None:
    assert _verdict(mean_effect=-1.4, ci_lower=-2.0, ci_upper=-0.9)["verdict"] == "negative"
    assert _verdict(mean_effect=0.1, ci_lower=-1.2, ci_upper=1.3)["verdict"] == "inconclusive"
    positive = _verdict(mean_effect=1.3, ci_lower=0.4, ci_upper=2.1, improved_unit_count=8)
    assert positive["verdict"] == "positive"
    not_consistent = _verdict(
        mean_effect=1.3, ci_lower=0.4, ci_upper=2.1, improved_unit_count=6
    )
    assert not_consistent["verdict"] == "undetermined_by_preregistered_table"
    underpowered = _verdict(
        mean_effect=1.3, ci_lower=0.4, ci_upper=2.1, included_unit_count=6, degenerate_unit_count=2
    )
    assert underpowered["rule"] == "section_6_underpowered_override"
    too_degenerate = _verdict(mean_effect=1.3, ci_lower=0.4, ci_upper=2.1, degenerate_unit_count=3)
    assert too_degenerate["rule"] == "section_6_degenerate_pool_override"
    with pytest.raises(ObservedOnlyConfirmatoryAnalysisError):
        _verdict(mean_effect=None, ci_lower=None, ci_upper=None)
