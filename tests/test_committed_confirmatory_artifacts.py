"""Re-derive both preregistered confirmatory verdicts from committed artifacts.

Every other result in this repository lives under gitignored ``results/``. The
two confirmatory bundles are the exception: their per-unit outer-test metrics,
frozen policies, claims, degenerate records, pool accounting and analysis
outputs are force-tracked so that a fresh clone can recompute each verdict from
the same files the analysis scripts read, compare the recomputed analysis hash
with the committed one, and verify every manifest.  These tests do exactly
that.  They must never skip: the inputs are committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bh_augmentation.observed_only_confirmatory_analysis import (
    PLANNED_SEEDS as OBSERVED_ONLY_SEEDS,
)
from bh_augmentation.observed_only_confirmatory_analysis import (
    analyze_observed_only_confirmation,
    render_observed_only_report,
)
from bh_augmentation.primary_confirmatory_analysis import (
    analyze_primary_confirmation,
    render_confirmatory_report,
)
from bh_augmentation.utils.scientific_manifest import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
PHASE15_BUNDLE = (
    ROOT
    / "results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1"
)
OBSERVED_ONLY_RUN = ROOT / "results/corrected_candidate_scope_reanalysis_20260903T194209Z"
OBSERVED_ONLY_POOL = ROOT / "results/corrected_observed_only_transfer_pool_accounting_v1"
OBSERVED_ONLY_ANALYSIS = OBSERVED_ONLY_RUN / "summary/observed_only_confirmatory_analysis"


def test_phase15_verdict_is_rederived_from_committed_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committed = json.loads((PHASE15_BUNDLE / "confirmatory_analysis.json").read_text())
    # The payload records the bundle path as given, so recompute from the
    # repository root with the committed spelling.
    monkeypatch.chdir(ROOT)
    payload = analyze_primary_confirmation(committed["run"]["directory"])
    assert payload["analysis_hash"] == committed["analysis_hash"]
    assert payload["verdict"]["verdict"] == "null"
    assert payload == committed
    report = (PHASE15_BUNDLE / "confirmatory_report.md").read_text()
    assert render_confirmatory_report(payload) == report
    verification = verify_manifest(PHASE15_BUNDLE, filename="confirmatory_analysis_manifest.json")
    assert verification.verified_output_count >= 2, verification.to_dict()


def test_observed_only_verdict_is_rederived_from_committed_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committed = json.loads((OBSERVED_ONLY_ANALYSIS / "confirmatory_analysis.json").read_text())
    monkeypatch.chdir(ROOT)
    payload_relative = analyze_observed_only_confirmation(
        committed["run"]["run_root"], committed["run"]["pool_accounting"]["directory"]
    )
    assert payload_relative["analysis_hash"] == committed["analysis_hash"]
    assert payload_relative == committed
    payload = analyze_observed_only_confirmation(OBSERVED_ONLY_RUN, OBSERVED_ONLY_POOL)
    assert payload["verdict"] == committed["verdict"]
    assert payload["primary"] == committed["primary"]
    accounting = committed["accounting"]
    assert accounting["all_planned_seeds_accounted_for"] is True
    assert accounting["all_arm_unit_pairs_accounted_for"] is True
    assert accounting["planned_arm_unit_pairs"] == 27
    assert accounting["included_seeds"] == list(OBSERVED_ONLY_SEEDS)
    report = (OBSERVED_ONLY_ANALYSIS / "confirmatory_report.md").read_text()
    assert render_observed_only_report(payload_relative) == report
    verification = verify_manifest(OBSERVED_ONLY_ANALYSIS)
    assert verification.verified_output_count == 2, verification.to_dict()


def test_observed_only_control_arm_degeneracy_is_committed_evidence() -> None:
    committed = json.loads((OBSERVED_ONLY_ANALYSIS / "confirmatory_analysis.json").read_text())
    control = committed["secondary_control_arm"]
    counts = control["unit_status_counts"]
    assert counts["complete"] + counts["degenerate"] == len(OBSERVED_ONLY_SEEDS)
    assert counts["missing"] == 0
    for row in control["per_unit_rows"]:
        if row["status"] == "degenerate":
            assert row["accepted_synthetic_rows"] == 0
            assert row["rmse"] is None
            record_path = (
                OBSERVED_ONLY_RUN
                / "confirmatory"
                / control["family"]
                / f"seed_{row['seed']}"
                / "search"
                / "degenerate_unit.json"
            )
            assert record_path.is_file()
            assert json.loads(record_path.read_text())["degenerate_unit_hash"] == (
                row["degenerate_unit_hash"]
            )


# ------------------------------------------------ exploratory fraction sweep

SWEEP_ANALYSIS = ROOT / "results/corrected_exploratory_fraction_sweep_analysis_v1"
SWEEP_POOL = ROOT / "results/corrected_exploratory_fraction_sweep_pool_accounting_v1"
SWEEP_RUNS = {
    0.01: "results/corrected_exploratory_fraction_sweep_f0p01",
    0.10: "results/corrected_exploratory_fraction_sweep_f0p10",
}


def test_exploratory_sweep_is_rederived_and_carries_no_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.run_exploratory_fraction_sweep_analysis import declaration_commit

    from bh_augmentation.observed_only_confirmatory_analysis import exploratory_sweep_design

    monkeypatch.chdir(ROOT)
    summary = json.loads((SWEEP_ANALYSIS / "fraction_sweep_summary.json").read_text())
    commit = declaration_commit(summary["declaration"]["document"])
    assert summary["declaration"]["commit"] == commit
    rows = {row["train_fraction"]: row for row in summary["rows"]}
    assert set(rows) == {0.01, 0.05, 0.1}
    assert rows[0.05]["label"] == "confirmatory" and rows[0.05]["verdict"] == "null"
    for fraction, run_root in SWEEP_RUNS.items():
        slug = f"{fraction:.2f}".replace(".", "p")
        committed = json.loads((SWEEP_ANALYSIS / f"analysis_f{slug}.json").read_text())
        design = exploratory_sweep_design(fraction, declaration_commit=commit)
        payload = analyze_observed_only_confirmation(
            run_root, str(SWEEP_POOL.relative_to(ROOT)), design=design
        )
        assert payload["analysis_hash"] == committed["analysis_hash"]
        assert payload == committed
        assert "preregistration" not in payload
        assert payload["verdict"]["verdict"] == "not_applicable_exploratory"
        assert payload["accounting"]["all_arm_unit_pairs_accounted_for"] is True
        assert payload["accounting"]["planned_arm_unit_pairs"] == 18
        assert rows[fraction]["label"] == "exploratory"
        assert rows[fraction]["analysis_hash"] == committed["analysis_hash"]
        assert (SWEEP_ANALYSIS / f"report_f{slug}.md").read_text() == (
            render_observed_only_report(payload)
        )
    verification = verify_manifest(SWEEP_ANALYSIS)
    assert verification.verified_output_count == 6, verification.to_dict()
