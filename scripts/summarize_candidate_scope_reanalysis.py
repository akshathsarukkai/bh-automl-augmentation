#!/usr/bin/env python
"""Print the human-readable summary of a candidate-scope reanalysis run.

Factored out of ``scripts/run_lowdata_candidate_scope_reanalysis.sh`` step 9.
It reports which result families the semantics change affected, the candidate
counts under both eligibility rules, the withheld-cell oracle metrics, and the
per-seed confirmatory outcomes -- including degenerate units, which are
reported with their counts rather than treated as failures.

It prints a point estimate only.  The preregistered verdict requires the
seed-1601 bootstrap interval from PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md
section 4, computed by ``scripts/run_observed_only_confirmatory_analysis.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CONTROL_FAMILY = "matched_real_only_control"
PRIMARY_FAMILY = "observed_only_condition_transfer"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def report_affected_families(audit_dir: Path) -> None:
    print()
    print("AFFECTED RESULT FAMILIES (candidate-scope semantics changed the numbers)")
    audit_path = audit_dir / "candidate_scope_call_sites.json"
    if not audit_path.is_file():
        print("  (audit not present)")
        return
    audit = json.loads(audit_path.read_text())
    affected = sorted(
        {
            family
            for entry in audit["modules"]
            if entry["classification"] == "observed_only_low_data"
            for family in entry.get("affects", [])
        }
    )
    for family in affected:
        print(f"  - {family}")
    print(f"  ({audit['call_site_count']} call sites across {audit['module_count']} modules)")

    print()
    print("REUSED UNAFFECTED BASELINES (no candidates generated; historical bundles valid)")
    for family in (
        "real_only / direct_xgboost outer-test baselines",
        "truncated_svd and linear_autoencoder representation controls",
        "Phase 14/15 pools (already observed-only; see the registry declaration)",
        "Phase 18 prospective package (globally_unmeasured_prospective by design)",
    ):
        print(f"  - {family}")


def report_directories(result_root: Path) -> None:
    print()
    print("RESULT DIRECTORIES HOLDING FILES")
    for path in sorted(p for p in result_root.rglob("*") if p.is_dir()):
        if any(child.is_file() for child in path.iterdir()):
            print(f"  {path}")


def report_pool_counts(dev_dir: Path) -> None:
    print()
    print("CANDIDATE COUNTS UNDER BOTH ELIGIBILITY SEMANTICS")
    pool = read_csv(dev_dir / "candidate_pool_statistics.csv")
    if not pool:
        print("  (none recorded)")
        return
    totals: dict = {}
    for row in pool:
        key = (row["candidate_scope_mode"], row["transfer_kind"])
        bucket = totals.setdefault(
            key, {"generated": 0, "accepted": 0, "observed": 0, "measured": 0, "quarantined": 0}
        )
        bucket["generated"] += int(number(row["generated_candidate_count"], 0))
        bucket["accepted"] += int(number(row["accepted_candidate_count"], 0))
        bucket["observed"] += int(number(row["rejected_observed_in_labeled_train"], 0))
        bucket["measured"] += int(number(row["rejected_already_measured"], 0))
        bucket["quarantined"] += int(number(row["rejected_quarantined_held_out_identity"], 0))
    print(
        f"  {'scope':34s} {'kind':10s} {'generated':>10s} {'accepted':>9s} "
        f"{'obs-rej':>8s} {'glob-rej':>9s} {'quar':>6s}"
    )
    for (mode, kind), bucket in sorted(totals.items()):
        print(
            f"  {mode:34s} {kind:10s} {bucket['generated']:10d} {bucket['accepted']:9d} "
            f"{bucket['observed']:8d} {bucket['measured']:9d} {bucket['quarantined']:6d}"
        )


def report_oracle(dev_dir: Path) -> None:
    print()
    print("ORACLE RECONSTRUCTION METRICS (withheld_cell_transfer_oracle, secondary)")
    oracle = [
        row
        for row in read_csv(dev_dir / "withheld_cell_oracle.csv")
        if row.get("candidate_scope_mode") == "observed_only_low_data"
    ]
    if not oracle:
        print("  (none recorded)")
        return
    overlap = sum(int(number(row["hidden_overlap_count"], 0)) for row in oracle)
    unique = sum(int(number(row["unique_canonical_candidate_count"], 0)) for row in oracle)
    scored = [row for row in oracle if number(row["pseudo_label_mae"]) == number(row["pseudo_label_mae"])]

    def mean(field: str) -> float:
        values = [number(row[field]) for row in scored]
        values = [v for v in values if v == v]
        return sum(values) / len(values) if values else float("nan")

    print(f"  units scored           : {len(scored)} of {len(oracle)}")
    print(f"  candidates / overlap   : {unique} / {overlap}")
    print(f"  mean coverage fraction : {mean('hidden_cell_coverage_fraction'):.4f}")
    print(f"  mean pseudo-label MAE  : {mean('pseudo_label_mae'):.3f}")
    print(f"  mean pseudo-label RMSE : {mean('pseudo_label_rmse'):.3f}")
    print(f"  mean Spearman          : {mean('pseudo_label_spearman'):.3f}")
    print(f"  mean bias              : {mean('pseudo_label_bias'):+.3f}")


def report_confirmatory(confirm_root: Path) -> None:
    print()
    print("FINAL AUGMENTATION COMPARISON (point estimate only; not the verdict)")
    outcomes: dict[str, dict[int, float]] = {}
    degenerate: dict[str, list[int]] = {}
    for family_dir in sorted(p for p in confirm_root.iterdir() if p.is_dir()):
        family = family_dir.name
        for unit_dir in sorted(family_dir.glob("seed_*")):
            seed = int(unit_dir.name.split("_", 1)[1])
            metrics_path = unit_dir / "final" / "final_test_metrics.csv"
            degenerate_path = unit_dir / "search" / "degenerate_unit.json"
            if metrics_path.is_file():
                for row in read_csv(metrics_path):
                    if row["metric"] == "rmse" and row["split"] == "test":
                        outcomes.setdefault(family, {})[seed] = number(row["value"])
            elif degenerate_path.is_file():
                degenerate.setdefault(family, []).append(seed)
            else:
                print(f"  INCOMPLETE: {family} seed {seed} has neither a final result "
                      "nor a degenerate record")

    for family in sorted(set(outcomes) | set(degenerate)):
        complete = sorted(outcomes.get(family, {}))
        gone = sorted(degenerate.get(family, []))
        print(f"  {family:44s} complete seeds {complete}  degenerate seeds {gone}")

    control = outcomes.get(CONTROL_FAMILY, {})
    primary = outcomes.get(PRIMARY_FAMILY, {})
    if not control or not primary:
        print("  Confirmatory arms are incomplete; no point estimate is computed.")
        return
    seeds = sorted(set(primary) & set(control))
    deltas = [control[s] - primary[s] for s in seeds]
    n = len(deltas)
    mean_delta = sum(deltas) / n if n else float("nan")
    improved = sum(1 for d in deltas if d > 0)
    print(f"  paired units           : {n}")
    for seed, delta in zip(seeds, deltas, strict=True):
        print(
            f"    seed {seed:>2d}  control {control[seed]:8.4f}  "
            f"augmented {primary[seed]:8.4f}  delta {delta:+8.4f}"
        )
    print(f"  mean paired reduction  : {mean_delta:+.4f} RMSE")
    print(f"  units improved         : {improved} of {n}")
    print()
    print("  The preregistered verdict requires the 95% bootstrap interval from")
    print("  PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md section 4 (seed 1601).")
    print("  Compute it with scripts/run_observed_only_confirmatory_analysis.py")
    print("  before quoting a verdict; the point estimate above is not the verdict.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root: Path = args.run_root
    if not root.is_dir():
        raise SystemExit(f"Run root does not exist: {root}")
    report_affected_families(root / "candidate_scope_audit")
    report_directories(root)
    report_pool_counts(root / "development_reanalysis")
    report_oracle(root / "development_reanalysis")
    confirm_root = root / "confirmatory"
    if confirm_root.is_dir():
        report_confirmatory(confirm_root)
    print()


if __name__ == "__main__":
    sys.exit(main())
