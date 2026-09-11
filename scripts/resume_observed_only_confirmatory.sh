#!/usr/bin/env bash
#
# Resume an interrupted observed-only confirmatory run.
#
# The 2026-09-03 execution of scripts/run_lowdata_candidate_scope_reanalysis.sh
# completed every primary and comparator unit and then aborted at the first
# globally-unmeasured control unit whose candidate pool accepted zero rows --
# the outcome PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md section 6 says to
# expect and to *report*, not to treat as a failure. This script finishes such a
# run in place:
#
#   * a unit whose final/evaluation_claim.json is complete is left untouched;
#   * a unit whose search/degenerate_unit.json exists is left untouched;
#   * a unit whose directory is absent or empty is searched with
#     --record-degenerate, then finally evaluated only if a frozen policy
#     was written;
#   * any other state is an error, reported and never repaired by deletion.
#
# It then performs the original driver's closing steps: declared-hash
# verification and the produced-file index, the registry snapshot after all
# outer-test access, and the human-readable summary.
#
# It never pushes, never deletes or overwrites an existing file, never writes
# into a completed unit, and never downloads anything.
#
# Usage:
#   RUN_LABEL=20260903T194209Z PYTHON=/path/to/python \
#     bash scripts/resume_observed_only_confirmatory.sh 2>&1 \
#     | tee logs/observed_only_confirmatory_resume_$(date -u +%Y%m%d_%H%M%S).log
#
# Environment:
#   RUN_LABEL   (required) suffix of the run root to resume
#   PYTHON      interpreter (default: python)
#   ALLOW_DIRTY=1  permit a dirty working tree (not recommended: resumed units
#                  bind the current commit hash, which must describe the code)

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${RUN_LABEL:?RUN_LABEL is required (for example 20260903T194209Z)}"
RESULT_ROOT="results/corrected_candidate_scope_reanalysis_${RUN_LABEL}"
CONFIRM_ROOT="${RESULT_ROOT}/confirmatory"
SUMMARY_DIR="${RESULT_ROOT}/summary"
CONFIRM_CONFIG_DIR="configs/corrected_observed_only_transfer"
CONFIRM_SEEDS=(6 7 8 9 10 11 12 13 14)
CONFIRM_FAMILIES=(observed_only_condition_transfer matched_real_only_control globally_unmeasured_condition_transfer)

log() { printf '%s | %s\n' "$(date -u +%H:%M:%S)" "$*"; }
step() { printf '\n================================================================\n%s\n================================================================\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ preflight

step "Preflight"

command -v "$PYTHON" >/dev/null 2>&1 || fail "Interpreter not found: $PYTHON"
git rev-parse HEAD >/dev/null 2>&1 || fail "Scientific runs require a git commit."
[[ -d "$RESULT_ROOT" ]] || fail "Run root does not exist: $RESULT_ROOT"
[[ -d "$CONFIRM_ROOT" ]] || fail "Confirmatory directory does not exist: $CONFIRM_ROOT"
[[ -d "$SUMMARY_DIR" ]] || fail "Summary directory does not exist: $SUMMARY_DIR"

DIRTY="no"
if [[ -n "$(git status --porcelain)" ]]; then
    DIRTY="yes"
    if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
        fail "Working tree is dirty; commit first so the bound commit hash describes the code (or set ALLOW_DIRTY=1)."
    fi
fi

log "repository   : $REPO_ROOT"
log "commit       : $(git rev-parse HEAD)"
log "dirty        : $DIRTY"
log "interpreter  : $($PYTHON -c 'import sys; print(sys.executable)')"
log "run label    : $RUN_LABEL"
log "result root  : $RESULT_ROOT"

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

# --------------------------------------------------------- classify each unit

step "Classify every planned unit"

PENDING=()
for family in "${CONFIRM_FAMILIES[@]}"; do
    for seed in "${CONFIRM_SEEDS[@]}"; do
        unit_dir="${CONFIRM_ROOT}/${family}/seed_${seed}"
        claim="${unit_dir}/final/evaluation_claim.json"
        degenerate="${unit_dir}/search/degenerate_unit.json"
        frozen="${unit_dir}/search/frozen_policy.json"
        if [[ -f "$claim" ]]; then
            status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$claim")"
            [[ "$status" == "complete" ]] || fail "Unit ${family}/seed_${seed} has a non-complete claim (${status}); refusing to touch it."
            log "complete   : ${family} seed=${seed}"
        elif [[ -f "$degenerate" ]]; then
            log "degenerate : ${family} seed=${seed} (already recorded)"
        elif [[ ! -e "$unit_dir" ]] || [[ -z "$(ls -A "$unit_dir")" ]]; then
            log "pending    : ${family} seed=${seed}"
            PENDING+=("${family}:${seed}")
        elif [[ -f "$frozen" ]] && [[ ! -e "${unit_dir}/final" ]]; then
            fail "Unit ${family}/seed_${seed} has a frozen policy but no final evaluation; a search was interrupted before final ran. Inspect by hand -- this script does not decide whether that search is trustworthy."
        else
            fail "Unit ${family}/seed_${seed} is in an unrecognised state: $(ls -A "$unit_dir" | tr '\n' ' ')"
        fi
    done
done
log "${#PENDING[@]} unit(s) pending"

# ----------------------------------------------- registry precheck (pending)

step "Evaluation-registry precheck for pending units"

if [[ ${#PENDING[@]} -gt 0 ]]; then
    $PYTHON - "$CONFIRM_CONFIG_DIR" "${PENDING[@]}" <<'PYCHECK' || fail "A pending confirmatory identity is already consumed."
import json
import pathlib
import sys

config_dir = pathlib.Path(sys.argv[1])
pending = sys.argv[2:]
registry_root = pathlib.Path("results/autonomous_execution/evaluation_registry")
consumed = set()
for record_path in registry_root.rglob("*.json"):
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    consumed.add(record.get("identity", {}).get("evaluation_unit"))

overlap = []
for item in pending:
    family, seed = item.split(":")
    config_path = config_dir / f"{family}_seed_{seed}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"Missing confirmatory config: {config_path}")
    unit = f"seed={seed}|train_fraction=0.05|comparison=model_policy_search|family={family}"
    if unit in consumed:
        overlap.append(unit)
if overlap:
    print("Already-consumed confirmatory identities:", overlap[:5])
    raise SystemExit(1)
print(f"{len(pending)} pending evaluation identities are unconsumed")
PYCHECK
else
    log "nothing pending; skipping"
fi

# ------------------------------------------------------- run pending units

step "Run pending confirmatory units"

for item in "${PENDING[@]}"; do
    family="${item%%:*}"
    seed="${item##*:}"
    config="${CONFIRM_CONFIG_DIR}/${family}_seed_${seed}.yaml"
    unit_dir="${CONFIRM_ROOT}/${family}/seed_${seed}"
    mkdir -p "$unit_dir"
    [[ -e "${unit_dir}/search" ]] && fail "Refusing to reuse ${unit_dir}/search"

    log "search : ${family} seed=${seed}"
    $PYTHON -B scripts/run_policy_search.py \
        --config "$config" \
        --output-directory "${unit_dir}/search" \
        --record-degenerate \
        || fail "Policy search failed for ${family} seed ${seed}."

    if [[ -f "${unit_dir}/search/degenerate_unit.json" ]]; then
        log "degenerate : ${family} seed=${seed} -- recorded; no outer test read"
        continue
    fi
    [[ -f "${unit_dir}/search/frozen_policy.json" ]] || fail "Search for ${family} seed ${seed} produced neither a frozen policy nor a degenerate record."

    log "final  : ${family} seed=${seed}"
    $PYTHON -B scripts/run_final_evaluation.py \
        --config "$config" \
        --frozen-policy "${unit_dir}/search/frozen_policy.json" \
        --search-manifest "${unit_dir}/search/search_manifest.json" \
        --output-directory "${unit_dir}/final" \
        || fail "Final evaluation failed for ${family} seed ${seed}."
done
log "all planned units are complete or recorded as degenerate"

# ------------------------------------------------------- closing steps

step "Verify declared hashes and index produced files"

$PYTHON -B scripts/build_run_file_index.py --run-root "$RESULT_ROOT" \
    || fail "Hash verification or file indexing failed."

step "Evaluation-registry snapshot after all outer-test access"

for suffix in txt json; do
    target="${SUMMARY_DIR}/evaluation_registry_after.${suffix}"
    [[ -e "$target" ]] && fail "Refusing to overwrite ${target}"
done
$PYTHON -B scripts/inspect_evaluation_registry.py list \
    > "${SUMMARY_DIR}/evaluation_registry_after.txt" 2>&1 \
    || fail "Could not inspect the evaluation registry."
$PYTHON -B scripts/inspect_evaluation_registry.py list --json \
    > "${SUMMARY_DIR}/evaluation_registry_after.json" 2>&1 \
    || fail "Could not inspect the evaluation registry (json)."
log "registry snapshots written under ${SUMMARY_DIR}"

step "Summary"

$PYTHON -B scripts/summarize_candidate_scope_reanalysis.py --run-root "$RESULT_ROOT"

printf 'All outputs are under %s\n' "$RESULT_ROOT"
printf 'Nothing was pushed. No historical output was deleted or overwritten.\n'
printf 'Next: scripts/run_observed_only_confirmatory_analysis.py for the preregistered verdict.\n'
