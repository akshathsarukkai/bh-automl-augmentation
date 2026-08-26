#!/usr/bin/env bash
#
# Master reanalysis for the low-data candidate-scope correction.
#
# What this runs, in order:
#   0. Preflight: environment, canonicalization vocabulary, git commit.
#   1. Prerequisite audits and the bounded test suite.
#   2. The candidate-scope call-site audit (fails on declaration drift).
#   3. Development reanalysis: candidate pools under both eligibility rules,
#      plus the withheld-cell oracle diagnostic. Validation only.
#   4. The preregistered confirmatory experiment on fresh evaluation units,
#      through the repository-global evaluation registry.
#   5. Summaries, manifests, hash verification, leakage-contract checks.
#
# What this never does:
#   - push, tag, or otherwise touch a remote;
#   - delete or overwrite any historical output;
#   - write into an existing result directory;
#   - download an external dataset.
#
# Safe under `caffeinate` and `tee`: no interactive prompts, no terminal
# control sequences, line-buffered progress, and a single exit status.
#
# Usage:
#   caffeinate -i bash scripts/run_lowdata_candidate_scope_reanalysis.sh 2>&1 \
#     | tee results/corrected_candidate_scope_reanalysis_run.log
#
# Options (environment variables):
#   PYTHON                 interpreter to use (default: python)
#   RUN_LABEL              suffix for this run's output directories (default: timestamp)
#   SKIP_CONFIRMATORY=1    run everything except the confirmatory experiment
#   SKIP_TESTS=1           skip the bounded test suite (not recommended)

set -euo pipefail

# ---------------------------------------------------------------- constants

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_LABEL="${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="results/corrected_candidate_scope_reanalysis_${RUN_LABEL}"
AUDIT_DIR="${RESULT_ROOT}/candidate_scope_audit"
DEV_DIR="${RESULT_ROOT}/development_reanalysis"
CONFIRM_ROOT="${RESULT_ROOT}/confirmatory"
SUMMARY_DIR="${RESULT_ROOT}/summary"

DEV_CONFIG="configs/corrected_candidate_scope_reanalysis.yaml"
CONFIRM_CONFIG_DIR="configs/corrected_observed_only_transfer"
CONFIRM_SEEDS=(6 7 8 9 10 11 12 13 14)
CONFIRM_FAMILIES=(observed_only_condition_transfer matched_real_only_control globally_unmeasured_condition_transfer)

STEP=0
TOTAL_STEPS=10

# ------------------------------------------------------------------ helpers

log() { printf '%s | %s\n' "$(date -u +%H:%M:%S)" "$*"; }

step() {
    STEP=$((STEP + 1))
    printf '\n'
    printf '================================================================\n'
    printf 'STEP %d/%d  %s\n' "$STEP" "$TOTAL_STEPS" "$*"
    printf '================================================================\n'
}

fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# Refuse to write anywhere that already holds results. Every output directory
# this script creates must be new; historical bundles are never touched.
require_fresh() {
    local path="$1"
    if [[ -e "$path" ]]; then
        fail "Refusing to reuse an existing output path: $path"
    fi
    mkdir -p "$path"
}

# ------------------------------------------------------------------- step 0

step "Preflight"

command -v "$PYTHON" >/dev/null 2>&1 || fail "Interpreter not found: $PYTHON"
git rev-parse HEAD >/dev/null 2>&1 || fail "Scientific runs require a git commit."

log "repository   : $REPO_ROOT"
log "commit       : $(git rev-parse HEAD)"
log "dirty        : $(if [[ -n "$(git status --porcelain)" ]]; then echo yes; else echo no; fi)"
log "interpreter  : $($PYTHON -c 'import sys; print(sys.executable)')"
log "run label    : $RUN_LABEL"
log "result root  : $RESULT_ROOT"

require_fresh "$RESULT_ROOT"

# The canonical dataset stores canonical_reaction_key values produced by a
# specific RDKit. If the installed RDKit canonicalizes differently, the stored
# and live identity vocabularies stop intersecting and the eligibility gate
# would accept every candidate, including ones identical to observed training
# reactions -- with no visible symptom in any metric. Fail here instead.
log "verifying the canonicalization vocabulary matches the dataset..."
$PYTHON - <<'PYCHECK' || fail "Canonicalization vocabulary mismatch (see message above)."
import sys

sys.path.insert(0, "src")
import pandas as pd

from bh_augmentation.augmentation.synthetic_identity import (
    assert_stored_identities_match_roles,
)

frame = pd.read_csv("data/processed/bh_canonical_roles_v1.csv", nrows=64)
assert_stored_identities_match_roles(frame, description="Canonical dataset")
print("canonicalization vocabulary OK")
PYCHECK

# ------------------------------------------------------------------- step 1

step "Lint"

$PYTHON -B -m ruff check . || fail "ruff reported findings."
log "ruff clean"

# ------------------------------------------------------------------- step 2

step "Bounded test suite"

# The Phase 14/15 benchmark module is expensive -- one of its tests takes over
# nine minutes because every isolated fit spawns a fresh interpreter that
# re-imports torch, RDKit and scikit-learn. It is run, not skipped: an earlier
# revision of this script excluded it on a mistaken deadlock diagnosis, and
# excluding a slow test is how coverage quietly disappears.

if [[ "${SKIP_TESTS:-0}" == "1" ]]; then
    log "SKIPPED by SKIP_TESTS=1"
else
    log "running the full suite (expect ~25 minutes; the Phase 14/15 module is slow)"
    $PYTHON -B -m pytest -q -ra || fail "The bounded test suite did not pass."
    log "tests passed"
fi

# ------------------------------------------------------------------- step 3

step "Candidate-scope call-site audit"

require_fresh "$AUDIT_DIR"
$PYTHON -B scripts/audit_candidate_scope.py --output-directory "$AUDIT_DIR" \
    || fail "Candidate-scope declarations do not match the code."

# ------------------------------------------------------------------- step 4

step "Evaluation-registry inspection (before any outer-test access)"

require_fresh "$SUMMARY_DIR"
$PYTHON -B scripts/inspect_evaluation_registry.py list \
    > "${SUMMARY_DIR}/evaluation_registry_before.txt" 2>&1 \
    || fail "Could not inspect the evaluation registry."
log "registry snapshot written to ${SUMMARY_DIR}/evaluation_registry_before.txt"

# Refuse to start if any confirmatory identity is already consumed.
$PYTHON - "$CONFIRM_CONFIG_DIR" <<'PYCHECK' || fail "A confirmatory evaluation identity is already consumed."
import json
import pathlib
import sys

registry_root = pathlib.Path("results/autonomous_execution/evaluation_registry")
consumed = set()
for record_path in registry_root.rglob("*.json"):
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    consumed.add(record.get("identity", {}).get("evaluation_unit"))

wanted = set()
for config_path in sorted(pathlib.Path(sys.argv[1]).glob("*.yaml")):
    text = config_path.read_text()
    family = next(
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().startswith("family:")
    )
    seed = next(
        line.split(":", 1)[1].strip().strip("[]")
        for line in text.splitlines()
        if line.startswith("seeds:")
    )
    wanted.add(
        f"seed={seed}|train_fraction=0.05|comparison=model_policy_search|family={family}"
    )

overlap = sorted(wanted & consumed)
if overlap:
    print("Already-consumed confirmatory identities:", overlap[:5])
    raise SystemExit(1)
print(f"{len(wanted)} confirmatory evaluation identities are unconsumed")
PYCHECK

# ------------------------------------------------------------------- step 5

step "Development reanalysis: candidate pools under both eligibility rules"

# Only calculations whose numbers actually depend on candidate-scope semantics
# are regenerated. No-augmentation baselines are unaffected by the change --
# they generate no candidates -- so their historical bundles are reused rather
# than recomputed; the summary in step 9 lists exactly which.
$PYTHON -B scripts/run_candidate_scope_reanalysis.py \
    --config "$DEV_CONFIG" \
    --output-directory "$DEV_DIR" \
    || fail "The development candidate-scope reanalysis did not complete."

# --------------------------------------------------------------- step 6 / 7

step "Preregistered confirmatory experiment"

if [[ "${SKIP_CONFIRMATORY:-0}" == "1" ]]; then
    log "SKIPPED by SKIP_CONFIRMATORY=1"
else
    require_fresh "$CONFIRM_ROOT"
    for family in "${CONFIRM_FAMILIES[@]}"; do
        for seed in "${CONFIRM_SEEDS[@]}"; do
            config="${CONFIRM_CONFIG_DIR}/${family}_seed_${seed}.yaml"
            [[ -f "$config" ]] || fail "Missing confirmatory config: $config"
            unit_dir="${CONFIRM_ROOT}/${family}/seed_${seed}"
            require_fresh "${unit_dir}"

            log "search : ${family} seed=${seed}"
            $PYTHON -B scripts/run_policy_search.py \
                --config "$config" \
                --output-directory "${unit_dir}/search" \
                || fail "Policy search failed for ${family} seed ${seed}."

            log "final  : ${family} seed=${seed}"
            $PYTHON -B scripts/run_final_evaluation.py \
                --config "$config" \
                --frozen-policy "${unit_dir}/search/frozen_policy.json" \
                --search-manifest "${unit_dir}/search/search_manifest.json" \
                --output-directory "${unit_dir}/final" \
                || fail "Final evaluation failed for ${family} seed ${seed}."
        done
    done
    log "confirmatory experiment complete"
fi

step "Withheld-cell oracle diagnostics"

# The oracle ran inside the development reanalysis, after each candidate pool
# and its pseudo-labels were frozen and hash-verified. Surface it here and fail
# if the frozen-bundle hashes are missing, which would mean the join happened
# without a seal.
$PYTHON - "$DEV_DIR" <<'PYCHECK' || fail "The withheld-cell oracle contract is not satisfied."
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
contracts = json.loads((directory / "leakage_contracts.json").read_text())
if contracts["outer_test_labels_accessed"] or contracts["evaluation_registry_claimed"]:
    raise SystemExit("The development reanalysis touched an outer test.")
if contracts["hidden_outer_train_read_stage"] != "after_candidate_and_pseudo_label_freeze":
    raise SystemExit("Hidden outcomes were read before the candidate freeze.")
if contracts["hidden_outer_train_influences"]:
    raise SystemExit("Hidden outcomes were recorded as influencing a stage.")
if not contracts["frozen_bundle_hashes"]:
    raise SystemExit("No frozen candidate bundle hashes were recorded.")
print(f"oracle contract OK over {len(contracts['frozen_bundle_hashes'])} frozen bundles")
PYCHECK

# ------------------------------------------------------------------- step 8

step "Rebuild summaries and verify hashes"

$PYTHON - "$RESULT_ROOT" "$DEV_DIR" "$CONFIRM_ROOT" "$SUMMARY_DIR" <<'PYSUMMARY' \
    || fail "Summary or hash verification failed."
import hashlib
import json
import pathlib
import sys

result_root, dev_dir, confirm_root, summary_dir = (pathlib.Path(p) for p in sys.argv[1:5])
summary_dir.mkdir(parents=True, exist_ok=True)

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

# Verify every manifest's recorded output hashes against the files on disk.
verified = 0
for manifest_path in sorted(result_root.rglob("manifest.json")):
    manifest = json.loads(manifest_path.read_text())
    for name, expected in (manifest.get("output_hashes") or {}).items():
        target = manifest_path.parent / name
        if not target.is_file():
            raise SystemExit(f"Declared output missing for {manifest_path}: {name}")
        if sha256(target) != expected:
            raise SystemExit(f"Hash mismatch for {name} under {manifest_path.parent}")
        verified += 1
print(f"verified {verified} declared output hashes")

# One index of every file this run produced, with its hash.
index = {
    "run_root": str(result_root),
    "files": {
        str(path.relative_to(result_root)): sha256(path)
        for path in sorted(result_root.rglob("*"))
        if path.is_file()
    },
}
(summary_dir / "run_file_index.json").write_text(
    json.dumps(index, indent=2, sort_keys=True) + "\n"
)
print(f"indexed {len(index['files'])} produced files")
PYSUMMARY

$PYTHON -B scripts/inspect_evaluation_registry.py list \
    > "${SUMMARY_DIR}/evaluation_registry_after.txt" 2>&1 || true

# ------------------------------------------------------------------- step 9

step "Final summary"

$PYTHON - "$RESULT_ROOT" "$AUDIT_DIR" "$DEV_DIR" "$CONFIRM_ROOT" <<'PYREPORT'
import json
import pathlib
import sys

result_root, audit_dir, dev_dir, confirm_root = (pathlib.Path(p) for p in sys.argv[1:5])


def read_csv(path):
    import csv

    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


print()
print("AFFECTED RESULT FAMILIES (candidate-scope semantics changed the numbers)")
audit = json.loads((audit_dir / "candidate_scope_call_sites.json").read_text())
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

print()
print("NEWLY GENERATED RESULT DIRECTORIES")
for path in sorted(p for p in result_root.rglob("*") if p.is_dir()):
    if any(child.is_file() for child in path.iterdir()):
        print(f"  {path}")

print()
print("CANDIDATE COUNTS UNDER BOTH ELIGIBILITY SEMANTICS")
pool = read_csv(dev_dir / "candidate_pool_statistics.csv")
totals = {}
for row in pool:
    key = (row["candidate_scope_mode"], row["transfer_kind"])
    bucket = totals.setdefault(key, {"generated": 0, "accepted": 0, "observed": 0, "measured": 0, "quarantined": 0})
    bucket["generated"] += int(number(row["generated_candidate_count"], 0))
    bucket["accepted"] += int(number(row["accepted_candidate_count"], 0))
    bucket["observed"] += int(number(row["rejected_observed_in_labeled_train"], 0))
    bucket["measured"] += int(number(row["rejected_already_measured"], 0))
    bucket["quarantined"] += int(number(row["rejected_quarantined_held_out_identity"], 0))
print(f"  {'scope':34s} {'kind':10s} {'generated':>10s} {'accepted':>9s} {'obs-rej':>8s} {'glob-rej':>9s} {'quar':>6s}")
for (mode, kind), bucket in sorted(totals.items()):
    print(
        f"  {mode:34s} {kind:10s} {bucket['generated']:10d} {bucket['accepted']:9d} "
        f"{bucket['observed']:8d} {bucket['measured']:9d} {bucket['quarantined']:6d}"
    )

print()
print("ORACLE RECONSTRUCTION METRICS (withheld_cell_transfer_oracle, secondary)")
oracle = [
    row
    for row in read_csv(dev_dir / "withheld_cell_oracle.csv")
    if row.get("candidate_scope_mode") == "observed_only_low_data"
]
if not oracle:
    print("  (none recorded)")
else:
    overlap = sum(int(number(row["hidden_overlap_count"], 0)) for row in oracle)
    unique = sum(int(number(row["unique_canonical_candidate_count"], 0)) for row in oracle)
    scored = [row for row in oracle if number(row["pseudo_label_mae"]) == number(row["pseudo_label_mae"])]
    def mean(field):
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

print()
print("FINAL AUGMENTATION COMPARISON AND CONFIRMATORY VERDICT")
augmented, control = {}, {}
for metrics_path in sorted(confirm_root.rglob("final/final_test_metrics.csv")):
    family = metrics_path.parents[2].name
    for row in read_csv(metrics_path):
        if row["metric"] != "rmse":
            continue
        target = (
            control
            if family == "matched_real_only_control"
            else augmented.setdefault(family, {})
        )
        target[int(row["seed"])] = number(row["value"])

if not control or "observed_only_condition_transfer" not in augmented:
    print("  Confirmatory arms are incomplete; no verdict is computed.")
    print("  VERDICT: not evaluated")
else:
    primary = augmented["observed_only_condition_transfer"]
    seeds = sorted(set(primary) & set(control))
    deltas = [control[s] - primary[s] for s in seeds]
    n = len(deltas)
    mean_delta = sum(deltas) / n if n else float("nan")
    improved = sum(1 for d in deltas if d > 0)
    print(f"  paired units           : {n}")
    for seed, delta in zip(seeds, deltas):
        print(f"    seed {seed:>2d}  control {control[seed]:8.4f}  augmented {primary[seed]:8.4f}  delta {delta:+8.4f}")
    print(f"  mean paired reduction  : {mean_delta:+.4f} RMSE")
    print(f"  units improved         : {improved} of {n}")
    print()
    print("  The preregistered verdict requires the 95% bootstrap interval from")
    print("  PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md section 4 (seed 1601).")
    print("  Compute it with scripts/run_primary_confirmatory_analysis.py before")
    print("  quoting a verdict; the point estimate above is not the verdict.")
PYREPORT

printf '\n'
printf '================================================================\n'
printf 'TEST / LINT STATUS\n'
printf '================================================================\n'
if [[ "${SKIP_TESTS:-0}" == "1" ]]; then
    printf '  tests : SKIPPED (SKIP_TESTS=1)\n'
else
    printf '  tests : PASSED (full suite, nothing excluded)\n'
fi
printf '  lint  : PASSED (ruff)\n'
printf '  audit : PASSED (candidate-scope declarations match the code)\n'
printf '\n'
printf 'All outputs are under %s\n' "$RESULT_ROOT"
printf 'Nothing was pushed. No historical output was deleted or overwritten.\n'
