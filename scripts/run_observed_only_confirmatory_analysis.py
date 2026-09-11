#!/usr/bin/env python
"""Compute and persist the preregistered observed-only confirmatory analysis.

The analysis takes no scientific parameters: the arms, metric, fraction, seed
list, margin, interval method, exclusion rules and interpretation table are
frozen in ``PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md`` (commit ``c6aab9e``)
and encoded as constants in
:mod:`bh_augmentation.observed_only_confirmatory_analysis`.

Inputs are the confirmatory run root (holding ``confirmatory/<family>/seed_<n>``)
and the validation-only pool accounting bundle that regenerates each unit's
candidate pool under both eligibility rules.  Outputs are written into a fresh
directory and sealed with a scientific manifest whose ``plan_hash`` is the
SHA-256 of the preregistration exactly as committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bh_augmentation.observed_only_confirmatory_analysis import (
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT,
    analyze_observed_only_confirmation,
    preregistration_plan_hash,
    render_observed_only_report,
)
from bh_augmentation.utils.corrected_runs import stable_hash
from bh_augmentation.utils.scientific_manifest import (
    build_scientific_manifest,
    verify_manifest,
    write_scientific_manifest,
)

ANALYSIS_OUTPUTS = ("confirmatory_analysis.json", "confirmatory_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run root holding confirmatory/<family>/seed_<n>/{search,final}.",
    )
    parser.add_argument(
        "--pool-accounting-directory",
        type=Path,
        required=True,
        help="Validation-only candidate-scope reanalysis bundle for seeds 6-14 at 0.05.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Default: <run-root>/summary/observed_only_confirmatory_analysis",
    )
    args = parser.parse_args()
    run_root: Path = args.run_root
    output: Path = args.output_directory or (
        run_root / "summary" / "observed_only_confirmatory_analysis"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to write into an existing directory: {output}")

    payload = analyze_observed_only_confirmation(run_root, args.pool_accounting_directory)
    report = render_observed_only_report(payload)

    output.mkdir(parents=True)
    (output / "confirmatory_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output / "confirmatory_report.md").write_text(report)

    run = payload["run"]
    manifest = build_scientific_manifest(
        status="observed_only_confirmatory_analysis_complete",
        output_directory=output,
        outputs=ANALYSIS_OUTPUTS,
        dataset_path=run["dataset_path"] or "unknown",
        dataset_hash=run["dataset_hash"] or "unknown",
        split_hash=run["split_aggregate_hash"] or "unknown",
        feature_metadata_hash=run["feature_metadata_hash"] or "unknown",
        config_hash=stable_hash(payload["preregistration"]),
        plan_hash=preregistration_plan_hash(),
        resolved_scientific_config=payload["preregistration"],
        row_counts={
            "per_unit_rows": len(payload["per_unit_rows"]),
            "included_units": payload["accounting"]["included_unit_count"],
            "paired_deltas": len(payload["primary"]["paired_deltas"]),
            "secondary_control_rows": len(payload["secondary_control_arm"]["per_unit_rows"]),
        },
        command=(
            "python scripts/run_observed_only_confirmatory_analysis.py "
            f"--run-root {run_root} --pool-accounting-directory "
            f"{args.pool_accounting_directory} --output-directory {output}"
        ),
        extra={
            "analysis_hash": payload["analysis_hash"],
            "verdict": payload["verdict"]["verdict"],
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_document": PREREGISTRATION_DOCUMENT,
            "pool_accounting_manifest_hash": run["pool_accounting"]["manifest_hash"],
        },
    )
    write_scientific_manifest(output, manifest)
    verification = verify_manifest(output)
    print(f"Verdict: {payload['verdict']['verdict']}")
    print(f"Analysis: {output / 'confirmatory_analysis.json'}")
    print(f"Report: {output / 'confirmatory_report.md'}")
    print(f"Manifest verified: {verification.to_dict()}")


if __name__ == "__main__":
    main()
