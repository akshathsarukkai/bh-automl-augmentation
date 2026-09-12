#!/usr/bin/env python
"""Analyse the exploratory training-fraction sweep and place it beside the confirmatory point.

This is the post hoc, descriptive analysis declared in
``EXPLORATORY_FRACTION_SWEEP.md``.  It reuses the observed-only confirmatory
loaders, accounting, estimand and bootstrap with the sweep's declared design
(fresh bootstrap seed 1701, no secondary control arm, **no interpretation table
and no verdict**), once per fraction, then writes one summary table that puts
the fractions next to the committed confirmatory result at 0.05 -- clearly
labelled exploratory versus confirmatory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bh_augmentation.observed_only_confirmatory_analysis import (
    EXPLORATORY_SWEEP_DOCUMENT,
    analyze_observed_only_confirmation,
    exploratory_sweep_design,
    render_observed_only_report,
)
from bh_augmentation.utils.corrected_runs import stable_hash
from bh_augmentation.utils.scientific_manifest import (
    build_scientific_manifest,
    verify_manifest,
    write_scientific_manifest,
)

CONFIRMATORY_ANALYSIS = Path(
    "results/corrected_candidate_scope_reanalysis_20260903T194209Z/summary/"
    "observed_only_confirmatory_analysis/confirmatory_analysis.json"
)


def declaration_commit(document: str) -> str:
    """The first commit that added the declaration document."""
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", document],
        check=True,
        capture_output=True,
        text=True,
    )
    commits = [line for line in result.stdout.splitlines() if line.strip()]
    if not commits:
        raise SystemExit(
            f"{document} has not been committed; commit the declaration before analysing."
        )
    return commits[-1]


def fraction_slug(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        nargs=2,
        action="append",
        metavar=("PATH", "FRACTION"),
        required=True,
        help="A sweep run root and its training fraction; repeat once per fraction.",
    )
    parser.add_argument("--pool-accounting-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--confirmatory-analysis",
        type=Path,
        default=CONFIRMATORY_ANALYSIS,
        help="Committed confirmatory analysis JSON used as the 0.05 reference row.",
    )
    args = parser.parse_args()
    output: Path = args.output_directory
    if output.exists():
        raise FileExistsError(f"Refusing to write into an existing directory: {output}")
    commit = declaration_commit(EXPLORATORY_SWEEP_DOCUMENT)

    output.mkdir(parents=True)
    per_fraction = []
    outputs = []
    for run_root, fraction_text in args.run_root:
        fraction = float(fraction_text)
        design = exploratory_sweep_design(fraction, declaration_commit=commit)
        payload = analyze_observed_only_confirmation(
            run_root, args.pool_accounting_directory, design=design
        )
        report = render_observed_only_report(payload)
        slug = fraction_slug(fraction)
        (output / f"analysis_f{slug}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        (output / f"report_f{slug}.md").write_text(report)
        outputs.extend([f"analysis_f{slug}.json", f"report_f{slug}.md"])
        per_fraction.append((fraction, payload))

    confirmatory = json.loads(args.confirmatory_analysis.read_text())
    rows = []
    for fraction, payload in per_fraction:
        rows.append(_summary_row(fraction, payload, label="exploratory"))
    rows.append(_summary_row(confirmatory["preregistration"]["train_fraction"], confirmatory, label="confirmatory"))
    rows.sort(key=lambda row: row["train_fraction"])
    summary = {
        "schema_version": "bh-exploratory-fraction-sweep-summary-v1",
        "declaration": {"document": EXPLORATORY_SWEEP_DOCUMENT, "commit": commit},
        "confirmatory_reference": {
            "path": str(args.confirmatory_analysis),
            "analysis_hash": confirmatory["analysis_hash"],
            "verdict": confirmatory["verdict"]["verdict"],
        },
        "rows": rows,
        "note": (
            "Exploratory rows carry no verdict and cannot be promoted to a claim. The "
            "confirmatory row is the committed preregistered result, included for scale."
        ),
    }
    summary["summary_hash"] = stable_hash(summary)
    (output / "fraction_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "fraction_sweep_summary.md").write_text(_render_summary(summary))
    outputs.extend(["fraction_sweep_summary.json", "fraction_sweep_summary.md"])

    reference = per_fraction[0][1]["run"]
    manifest = build_scientific_manifest(
        status="exploratory_fraction_sweep_analysis_complete",
        output_directory=output,
        outputs=outputs,
        dataset_path=reference["dataset_path"] or "unknown",
        dataset_hash=reference["dataset_hash"] or "unknown",
        split_hash=reference["split_aggregate_hash"] or "unknown",
        feature_metadata_hash=reference["feature_metadata_hash"] or "unknown",
        config_hash=stable_hash([payload["declaration"] for _, payload in per_fraction]),
        plan_hash=_document_hash(EXPLORATORY_SWEEP_DOCUMENT, commit),
        resolved_scientific_config={"declaration": summary["declaration"], "fractions": [f for f, _ in per_fraction]},
        row_counts={"fractions": len(per_fraction), "summary_rows": len(rows)},
        command=" ".join(["python scripts/run_exploratory_fraction_sweep_analysis.py"] + [
            f"--run-root {root} {fraction}" for root, fraction in args.run_root
        ] + [f"--pool-accounting-directory {args.pool_accounting_directory}", f"--output-directory {output}"]),
        extra={"summary_hash": summary["summary_hash"], "exploratory": True, "verdict": "not_applicable_exploratory"},
    )
    write_scientific_manifest(output, manifest)
    verification = verify_manifest(output)
    print(_render_summary(summary))
    print(f"Manifest verified: {verification.verified_output_count} outputs -> {output}")


def _summary_row(fraction: float, payload: dict, *, label: str) -> dict:
    primary = payload["primary"]
    bootstrap = primary["bootstrap"] or {}
    accounting = payload["accounting"]
    sizes = payload["treatment_size"]["accepted_synthetic_rows_by_seed"]
    observed = [v for v in sizes.get("observed_only_low_data", {}).values() if v is not None]
    global_rule = [v for v in sizes.get("globally_unmeasured_prospective", {}).values() if v is not None]
    return {
        "label": label,
        "train_fraction": float(fraction),
        "n_labeled_train": _first_present(payload["per_unit_rows"]),
        "included_units": accounting["included_unit_count"],
        "degenerate_units": accounting["degenerate_excluded_unit_count"],
        "failed_or_missing_units": accounting["failed_or_missing_unit_count"],
        "mean_paired_rmse_reduction": primary["mean_paired_rmse_reduction"],
        "ci_lower": bootstrap.get("ci_lower"),
        "ci_upper": bootstrap.get("ci_upper"),
        "bootstrap_seed": bootstrap.get("seed"),
        "improved_units": primary["improved_unit_count"],
        "worsened_units": primary["worsened_unit_count"],
        "accepted_rows_observed_only_min_max": [min(observed), max(observed)] if observed else None,
        "accepted_rows_globally_unmeasured_min_max": (
            [min(global_rule), max(global_rule)] if global_rule else None
        ),
        "verdict": payload["verdict"]["verdict"],
        "analysis_hash": payload["analysis_hash"],
    }


def _first_present(rows: list[dict]) -> int | None:
    for row in rows:
        if row.get("n_refit") is not None:
            return int(row["n_refit"])
    return None


def _document_hash(document: str, commit: str) -> str:
    import hashlib

    content = subprocess.run(
        ["git", "show", f"{commit}:{document}"], check=True, capture_output=True
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _render_summary(summary: dict) -> str:
    lines = [
        "# Effect of observed-only condition transfer versus training fraction",
        "",
        f"Declared in `{summary['declaration']['document']}` at commit "
        f"`{summary['declaration']['commit'][:7]}`. {summary['note']}",
        "",
        "| Label | Fraction | Included / degenerate / missing | Mean paired RMSE reduction | 95% CI | Improved / worsened | Accepted rows per unit (observed-only) | Accepted rows per unit (global rule) | Verdict |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        ci = (
            f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}] (seed {row['bootstrap_seed']})"
            if row["ci_lower"] is not None
            else "n/a"
        )
        mean = f"{row['mean_paired_rmse_reduction']:+.3f}" if row["mean_paired_rmse_reduction"] is not None else "n/a"
        obs = row["accepted_rows_observed_only_min_max"]
        glob = row["accepted_rows_globally_unmeasured_min_max"]
        lines.append(
            f"| {row['label']} | {row['train_fraction']:g} | {row['included_units']} / "
            f"{row['degenerate_units']} / {row['failed_or_missing_units']} | {mean} | {ci} | "
            f"{row['improved_units']} / {row['worsened_units']} | "
            f"{'n/a' if obs is None else f'{obs[0]}–{obs[1]}'} | "
            f"{'n/a' if glob is None else f'{glob[0]}–{glob[1]}'} | {row['verdict']} |"
        )
    lines.append("")
    lines.append(
        "Positive values favour augmentation. The confirmatory row is decided by its "
        "preregistered interpretation table; the exploratory rows are descriptive only."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
