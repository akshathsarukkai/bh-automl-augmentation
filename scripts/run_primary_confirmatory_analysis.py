#!/usr/bin/env python
"""Compute and persist the preregistered Phase 15 primary confirmatory analysis.

The analysis takes no scientific parameters: the metric, fraction, arms, margin,
seed list, interval method and interpretation table are frozen in
``PRIMARY_EXPERIMENT.md`` and encoded as constants in
:mod:`bh_augmentation.primary_confirmatory_analysis`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bh_augmentation.primary_confirmatory_analysis import (
    analyze_primary_confirmation,
    render_confirmatory_report,
)
from bh_augmentation.utils.scientific_manifest import (
    build_scientific_manifest,
    verify_manifest,
    write_scientific_manifest,
)

ANALYSIS_OUTPUTS = ("confirmatory_analysis.json", "confirmatory_report.md")
MANIFEST_FILENAME = "confirmatory_analysis_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Completed Phase 14 runner bundle holding the confirmatory run.",
    )
    args = parser.parse_args()
    root: Path = args.run_directory

    payload = analyze_primary_confirmation(root)
    report = render_confirmatory_report(payload)

    for name in ANALYSIS_OUTPUTS:
        if (root / name).exists():
            raise FileExistsError(f"Refusing to overwrite: {root / name}")
    (root / "confirmatory_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (root / "confirmatory_report.md").write_text(report)

    run_manifest = json.loads((root / "manifest.json").read_text())
    manifest = build_scientific_manifest(
        status="primary_confirmatory_analysis_complete",
        output_directory=root,
        outputs=ANALYSIS_OUTPUTS,
        dataset_path=payload["preregistration"]["document"],
        dataset_hash=run_manifest["dataset_hash"],
        split_hash=run_manifest["canonical_split_hash"],
        feature_metadata_hash=run_manifest["feature_metadata_hash"],
        config_hash=run_manifest["config_hash"],
        plan_hash=run_manifest["plan_hash"],
        resolved_scientific_config=payload["preregistration"],
        row_counts={
            "per_unit_rows": len(payload["per_unit_rows"]),
            "included_units": payload["accounting"]["included_unit_count"],
            "paired_deltas": len(payload["primary"]["paired_deltas"]),
        },
        command=(
            "python scripts/run_primary_confirmatory_analysis.py "
            f"--run-directory {root}"
        ),
        extra={
            "analysis_hash": payload["analysis_hash"],
            "run_manifest_hash": run_manifest["manifest_hash"],
            "verdict": payload["verdict"]["verdict"],
        },
    )
    write_scientific_manifest(root, manifest, filename=MANIFEST_FILENAME)
    verify_manifest(root, filename=MANIFEST_FILENAME)
    print(
        json.dumps(
            {
                "run_directory": str(root),
                "verdict": payload["verdict"]["verdict"],
                "analysis_hash": payload["analysis_hash"],
                "confirmatory_manifest_hash": manifest["manifest_hash"],
                "report": str(root / "confirmatory_report.md"),
                "analysis": str(root / "confirmatory_analysis.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
