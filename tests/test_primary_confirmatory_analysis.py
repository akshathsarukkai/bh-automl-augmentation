"""Tests for the preregistered Phase 15 primary confirmatory analysis.

These tests pin the analysis to ``PRIMARY_EXPERIMENT.md`` as committed at
``13fbfde``: the arms, metric, fraction, seed list, margin, interval method,
exclusion rules and interpretation table are all asserted against the frozen
document rather than against whatever the code currently does.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bh_augmentation.primary_confirmatory_analysis import (
    AUGMENTED_METHOD_FAMILY,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPARATOR_METHOD_FAMILY,
    MAXIMUM_DEGENERATE_UNITS,
    MINIMUM_IMPROVED_UNITS_FOR_POSITIVE,
    MINIMUM_INCLUDED_UNITS,
    PLANNED_SEEDS,
    PRACTICAL_MINIMUM_EFFECT,
    PREREGISTRATION_COMMIT,
    PRIMARY_METRIC,
    PRIMARY_TRAIN_FRACTION,
    PrimaryConfirmatoryAnalysisError,
    accepted_synthetic_rows,
    analyze_primary_confirmation,
    paired_deltas,
    preregistered_verdict,
    render_confirmatory_report,
    seed_cluster_percentile_interval,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_CONFIG = ROOT / "configs/primary_confirmatory_phase15.yaml"
SMOKE_CONFIG = ROOT / "configs/primary_confirmatory_phase15_smoke.yaml"
SPLIT_CONFIG = ROOT / "configs/canonical_grouped_splits_phase15.yaml"
CANONICAL_DATASET_HASH = (
    "df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365"
)


def _preregistration_text() -> str:
    return subprocess.run(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:PRIMARY_EXPERIMENT.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _metric_rows(rmse_by_seed: dict[int, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for seed, (augmented, comparator) in rmse_by_seed.items():
        for family, value in (
            (AUGMENTED_METHOD_FAMILY, augmented),
            (COMPARATOR_METHOD_FAMILY, comparator),
        ):
            rows.append(
                {
                    "evaluation_unit": f"canonical-random:seed={seed}:fraction=0.2",
                    "seed": seed,
                    "train_fraction": PRIMARY_TRAIN_FRACTION,
                    "method_family": family,
                    "policy_id": f"{family}:0",
                    "policy_hash": "a" * 64,
                    "metric": PRIMARY_METRIC,
                    "value": value,
                    "n_samples": 396,
                    "data_role": "test",
                    "prediction_hash": "b" * 64,
                    "state_hash": "c" * 64,
                    "plan_hash": "d" * 64,
                    "frozen_policy_hash": "e" * 64,
                    "test_evaluation_count": 1,
                    "test_used_for_selection_or_retention": False,
                }
            )
    return pd.DataFrame(rows)


def _pool_rows(accepted_by_seed: dict[int, int]) -> pd.DataFrame:
    rows = []
    for seed, accepted in accepted_by_seed.items():
        for phase, accepted_value in (("search", 200), ("placement", accepted)):
            rows.append(
                {
                    "evaluation_unit": f"canonical-random:seed={seed}:fraction=0.2",
                    "phase": phase,
                    "pool_kind": "anonymous",
                    "parent_count": 507,
                    "candidate_count": 507,
                    "accepted_count": accepted_value,
                    "selected_count": accepted_value,
                    "degenerate_pool": accepted_value == 0,
                }
            )
    return pd.DataFrame(rows)


def test_constants_match_the_frozen_preregistration() -> None:
    text = _preregistration_text()
    assert "**10 fresh seeds, 5 through 14**" in text
    assert PLANNED_SEEDS == (5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    assert "**0.2** (primary)" in text
    assert PRIMARY_TRAIN_FRACTION == 0.2
    assert "Outer-test RMSE (lower is better)" in text
    assert PRIMARY_METRIC == "rmse"
    assert "**1.0 RMSE**" in text
    assert PRACTICAL_MINIMUM_EFFECT == 1.0
    assert "10 000 replicates, fixed seed 1501" in text
    assert BOOTSTRAP_REPLICATES == 10_000
    assert BOOTSTRAP_SEED == 1501
    assert "95% percentile bootstrap" in text
    assert BOOTSTRAP_ALPHA == 0.05
    assert "more than 3 of 10 units are degenerate" in text
    assert MAXIMUM_DEGENERATE_UNITS == 3
    assert "fewer than 7 units remain after exclusions" in text
    assert MINIMUM_INCLUDED_UNITS == 7
    assert "at least 7 of *n* units improve" in text
    assert MINIMUM_IMPROVED_UNITS_FOR_POSITIVE == 7
    assert (
        "comparator RMSE minus augmented RMSE" in text
        or "comparator minus augmented" in text
    )


def test_phase15_configs_encode_the_preregistered_design() -> None:
    splits = yaml.safe_load(SPLIT_CONFIG.read_text())
    assert splits["splits"]["seeds"] == list(PLANNED_SEEDS)
    assert splits["splits"]["group_column"] == "canonical_reaction_key"
    assert splits["output"]["directory"] == (
        "results/corrected_canonical_splits_phase15"
    )

    config = yaml.safe_load(CONFIRMATORY_CONFIG.read_text())
    assert config["splits"]["seeds"] == list(PLANNED_SEEDS)
    assert config["splits"]["fractions"] == [PRIMARY_TRAIN_FRACTION]
    assert config["splits"]["canonical_directory"] == (
        "results/corrected_canonical_splits_phase15"
    )
    assert config["features"]["n_bits"] == 2048
    assert config["selection_metric"] == PRIMARY_METRIC

    smoke = yaml.safe_load(SMOKE_CONFIG.read_text())
    assert smoke["splits"]["canonical_directory"] == (
        "results/corrected_canonical_splits_phase15"
    )
    assert set(smoke["splits"]["seeds"]) <= set(PLANNED_SEEDS)


def test_phase15_split_manifest_matches_the_preregistered_dataset() -> None:
    import json

    manifest_path = (
        ROOT / "results/corrected_canonical_splits_phase15/split_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("Phase 15 split artifacts have not been built here.")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["canonical_dataset_hash"] == CANONICAL_DATASET_HASH
    assert manifest["seeds"] == list(PLANNED_SEEDS)
    overlap = pd.read_csv(
        ROOT / "results/corrected_canonical_splits_phase15/split_overlap_audit.csv"
    )
    assert overlap["all_outer_group_overlaps_zero"].all()
    assert overlap["all_subsets_nested"].all()
    assert sorted(overlap["seed"].tolist()) == sorted(PLANNED_SEEDS)


def test_paired_delta_is_comparator_minus_augmented() -> None:
    frame = _metric_rows({seed: (10.0, 11.5) for seed in PLANNED_SEEDS})
    records = paired_deltas(frame)
    assert set(records) == set(PLANNED_SEEDS)
    for record in records.values():
        assert record["paired_delta_rmse_reduction"] == pytest.approx(1.5)
        assert record["both_arms_evaluated"] is True


def test_missing_arm_is_recorded_not_dropped() -> None:
    frame = _metric_rows({seed: (10.0, 11.0) for seed in PLANNED_SEEDS})
    frame = frame.loc[
        ~(
            (frame["seed"] == 7)
            & (frame["method_family"] == AUGMENTED_METHOD_FAMILY)
        )
    ]
    records = paired_deltas(frame)
    assert set(records) == set(PLANNED_SEEDS)
    assert records[7]["both_arms_evaluated"] is False
    assert records[7]["paired_delta_rmse_reduction"] is None


def test_repeated_outer_test_evaluation_is_rejected() -> None:
    frame = _metric_rows({seed: (10.0, 11.0) for seed in PLANNED_SEEDS})
    frame.loc[frame["seed"] == 5, "test_evaluation_count"] = 2
    with pytest.raises(PrimaryConfirmatoryAnalysisError, match="exactly once"):
        paired_deltas(frame)


def test_only_the_placement_pool_counts_as_the_treatment() -> None:
    pools = accepted_synthetic_rows(_pool_rows({seed: 340 for seed in PLANNED_SEEDS}))
    for record in pools.values():
        assert record["accepted_synthetic_rows"] == 340
        assert record["degenerate_pool"] is False


def test_zero_accepted_rows_is_degenerate() -> None:
    accepted = dict.fromkeys(PLANNED_SEEDS, 340)
    accepted[9] = 0
    pools = accepted_synthetic_rows(_pool_rows(accepted))
    assert pools[9]["degenerate_pool"] is True
    assert pools[9]["accepted_synthetic_rows"] == 0


def test_bootstrap_is_deterministic_and_seeded() -> None:
    deltas = [0.4, -0.2, 0.1, 0.0, -0.5, 0.3, 0.2, -0.1, 0.05, 0.15]
    first = seed_cluster_percentile_interval(deltas)
    second = seed_cluster_percentile_interval(deltas)
    assert first == second
    assert first["replicates"] == BOOTSTRAP_REPLICATES
    assert first["seed"] == BOOTSTRAP_SEED
    assert first["cluster_count"] == len(deltas)
    assert first["mean"] == pytest.approx(float(np.mean(deltas)))
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]
    shifted = seed_cluster_percentile_interval([value + 100.0 for value in deltas])
    assert shifted["mean"] == pytest.approx(first["mean"] + 100.0)


def test_interval_excluding_zero_inside_the_margin_is_null_not_positive() -> None:
    outcome = preregistered_verdict(
        mean_effect=0.2,
        ci_lower=0.05,
        ci_upper=0.35,
        included_unit_count=10,
        improved_unit_count=9,
        degenerate_unit_count=0,
    )
    assert outcome["verdict"] == "null"
    assert outcome["conditions"]["positive"] is False


def test_positive_requires_margin_interval_and_consistency() -> None:
    assert (
        preregistered_verdict(
            mean_effect=1.5,
            ci_lower=0.8,
            ci_upper=2.4,
            included_unit_count=10,
            improved_unit_count=9,
            degenerate_unit_count=0,
        )["verdict"]
        == "positive"
    )
    # Same interval, insufficient consistency: no predefined row applies.
    assert (
        preregistered_verdict(
            mean_effect=1.5,
            ci_lower=0.8,
            ci_upper=2.4,
            included_unit_count=10,
            improved_unit_count=6,
            degenerate_unit_count=0,
        )["verdict"]
        == "undetermined_by_preregistered_table"
    )


def test_negative_and_wide_interval_outcomes() -> None:
    assert (
        preregistered_verdict(
            mean_effect=-1.4,
            ci_lower=-2.5,
            ci_upper=-0.4,
            included_unit_count=10,
            improved_unit_count=1,
            degenerate_unit_count=0,
        )["verdict"]
        == "negative"
    )
    assert (
        preregistered_verdict(
            mean_effect=0.1,
            ci_lower=-1.6,
            ci_upper=1.9,
            included_unit_count=10,
            improved_unit_count=5,
            degenerate_unit_count=0,
        )["verdict"]
        == "inconclusive"
    )


def test_section_5_overrides_precede_the_interpretation_table() -> None:
    degenerate = preregistered_verdict(
        mean_effect=2.0,
        ci_lower=1.5,
        ci_upper=2.5,
        included_unit_count=6,
        improved_unit_count=6,
        degenerate_unit_count=4,
    )
    assert degenerate["verdict"] == "inconclusive"
    assert degenerate["rule"] == "section_5_degenerate_pool_override"
    underpowered = preregistered_verdict(
        mean_effect=2.0,
        ci_lower=1.5,
        ci_upper=2.5,
        included_unit_count=6,
        improved_unit_count=6,
        degenerate_unit_count=2,
    )
    assert underpowered["verdict"] == "inconclusive"
    assert underpowered["rule"] == "section_5_underpowered_override"


def test_end_to_end_analysis_accounts_for_every_planned_seed(
    tmp_path: Path,
) -> None:
    import json

    rmse = {
        seed: (10.0 + 0.01 * index, 10.0 + 0.01 * index + 0.05)
        for index, seed in enumerate(PLANNED_SEEDS)
    }
    accepted = dict.fromkeys(PLANNED_SEEDS, 340)
    accepted[14] = 0
    _metric_rows(rmse).to_csv(tmp_path / "final_test_metrics.csv", index=False)
    _pool_rows(accepted).to_csv(tmp_path / "pool_summary.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_hash": "f" * 64,
                "plan_hash": "0" * 64,
                "config_hash": "1" * 64,
                "dataset_hash": CANONICAL_DATASET_HASH,
                "canonical_split_hash": "2" * 64,
                "feature_metadata_hash": "3" * 64,
                "git_commit": "4" * 40,
                "retention_placement": "secondary_ablation",
                "search_budget_by_family": {
                    AUGMENTED_METHOD_FAMILY: 3,
                    COMPARATOR_METHOD_FAMILY: 3,
                },
            }
        )
    )
    payload = analyze_primary_confirmation(tmp_path)
    accounting = payload["accounting"]
    assert accounting["planned_unit_count"] == 10
    assert accounting["included_unit_count"] == 9
    assert accounting["degenerate_excluded_unit_count"] == 1
    assert accounting["failed_or_missing_unit_count"] == 0
    assert accounting["all_planned_seeds_accounted_for"] is True
    assert accounting["degenerate_excluded_seeds"] == [14]
    assert payload["primary"]["mean_paired_rmse_reduction"] == pytest.approx(0.05)
    assert payload["primary"]["improved_unit_count"] == 9
    assert payload["run"]["equal_arm_search_budget"] is True
    assert payload["verdict"]["verdict"] == "null"
    assert payload["secondary"]["label"] == "secondary"
    assert analyze_primary_confirmation(tmp_path) == payload

    report = render_confirmatory_report(payload)
    assert "NULL" in report
    assert "excluded_degenerate_pool" in report
    for seed in PLANNED_SEEDS:
        assert f"| {seed} |" in report
