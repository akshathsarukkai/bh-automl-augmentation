#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/role_aware_condition_transfer_v2_xgboost"
CONFIG="configs/role_aware_condition_transfer_v2_xgboost.yaml"
ALLOW_INVALID_RESULTS="${ALLOW_INVALID_RESULTS:-false}"

python - "$OUT" "$ALLOW_INVALID_RESULTS" <<'PY'
import sys

from bh_augmentation.results.status import assert_result_directory_allowed

assert_result_directory_allowed(
    sys.argv[1],
    allow_invalid=sys.argv[2].lower() == "true",
)
PY

echo "============================================================"
echo "ROLE-AWARE CONDITION TRANSFER V2 OVERNIGHT RUN"
echo "Started: $(date)"
echo "Repo: $(pwd)"
echo "Python: $(python --version 2>&1)"
echo "Config: $CONFIG"
echo "============================================================"

required_files=(
  "$CONFIG"
  "src/bh_augmentation/run_role_aware_condition_transfer.py"
  "src/bh_augmentation/augmentation/role_aware_condition_transfer.py"
  "src/bh_augmentation/data/bh_condition_reader.py"
  "tests/test_role_aware_condition_transfer.py"
  "data/processed/bh_clean_stress.csv"
)

for file in "${required_files[@]}"; do
  if [[ ! -e "$file" ]]; then
    echo "ERROR: required file is missing: $file"
    exit 2
  fi
done

echo
echo "Git status before run:"
git status --short || true

# Archive an existing result directory so this run cannot accidentally
# mix fresh output with stale CSV files.
if [[ -d "$OUT" ]]; then
  BACKUP="${OUT}_backup_${STAMP}"
  echo
  echo "Archiving existing output:"
  echo "  $OUT -> $BACKUP"
  mv "$OUT" "$BACKUP"
fi

mkdir -p "$OUT"

echo
echo "============================================================"
echo "STEP 1: TARGETED ROLE-AWARE TESTS"
echo "============================================================"
python -m pytest tests/test_role_aware_condition_transfer.py -q

echo
echo "============================================================"
echo "STEP 2: FULL TEST SUITE"
echo "============================================================"
python -m pytest -q

echo
echo "============================================================"
echo "STEP 3: FULL V2 XGBOOST EXPERIMENT"
echo "============================================================"
PYTHONUNBUFFERED=1 python -m bh_augmentation.run_role_aware_condition_transfer \
  --config "$CONFIG"

echo
echo "============================================================"
echo "STEP 4: VALIDATE EXPECTED OUTPUT FILES"
echo "============================================================"

expected_outputs=(
  "$OUT/policy_metrics.csv"
  "$OUT/selected_policies.csv"
  "$OUT/selected_policy_metrics.csv"
  "$OUT/summary.csv"
  "$OUT/role_transfer_audit.csv"
)

for file in "${expected_outputs[@]}"; do
  if [[ -s "$file" ]]; then
    echo "OK: $file"
  else
    echo "ERROR: expected nonempty output is missing: $file"
    exit 3
  fi
done

echo
echo "============================================================"
echo "STEP 5: BUILD FINAL OVERNIGHT SUMMARIES"
echo "============================================================"

python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

BASE = Path("results/role_aware_condition_transfer_v2_xgboost")
REPORT = BASE / "overnight_report.txt"


def read_csv(name: str, required: bool = True) -> pd.DataFrame:
    path = BASE / name
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def append_section(lines: list[str], title: str, text: str) -> None:
    lines.extend([
        "",
        "=" * 100,
        title,
        "=" * 100,
        text,
    ])


policy = read_csv("policy_metrics.csv")
selected = read_csv("selected_policies.csv")
selected_metrics = read_csv("selected_policy_metrics.csv")
audit = read_csv("role_transfer_audit.csv")
summary = read_csv("summary.csv", required=False)
role_counts = read_csv("role_value_counts.csv", required=False)

for frame in [policy, selected_metrics, summary]:
    if len(frame) and "metric" in frame.columns:
        frame["metric"] = frame["metric"].astype(str).str.strip()

lines: list[str] = []

append_section(
    lines,
    "FILES AND SHAPES",
    "\n".join([
        f"policy_metrics.csv: {policy.shape}",
        f"selected_policies.csv: {selected.shape}",
        f"selected_policy_metrics.csv: {selected_metrics.shape}",
        f"role_transfer_audit.csv: {audit.shape}",
        f"summary.csv: {summary.shape if len(summary) else 'not found'}",
        f"role_value_counts.csv: {role_counts.shape if len(role_counts) else 'not found'}",
    ]),
)

if len(role_counts):
    append_section(
        lines,
        "ROLE VALUE COUNTS",
        role_counts.to_string(index=False),
    )

# ------------------------------------------------------------------
# Selected-policy test metrics
# ------------------------------------------------------------------
selected_index_candidates = [
    "seed",
    "train_fraction",
    "model",
    "role_transfer_mode",
    "effective_role_transfer_mode",
    "donor_strategy",
    "label_strategy",
    "synthetic_multiplier",
]

selected_index = [
    col for col in selected_index_candidates
    if col in selected_metrics.columns
]

selected_test = selected_metrics[
    selected_metrics["split"].astype(str).str.lower().eq("test")
].copy()

selected_wide = (
    selected_test
    .pivot_table(
        index=selected_index,
        columns="metric",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

selected_wide.to_csv(
    BASE / "selected_policy_test_metrics_wide.csv",
    index=False,
)

append_section(
    lines,
    "SELECTED POLICY TEST METRICS BY SEED",
    selected_wide.sort_values(
        [c for c in ["train_fraction", "rmse"] if c in selected_wide.columns]
    ).to_string(index=False),
)

aggregate_group = [
    c for c in [
        "train_fraction",
        "model",
        "role_transfer_mode",
        "effective_role_transfer_mode",
    ]
    if c in selected_wide.columns
]

metric_cols = [
    c for c in ["mae", "rmse", "r2", "spearman"]
    if c in selected_wide.columns
]

selected_aggregate = (
    selected_wide
    .groupby(aggregate_group, dropna=False)[metric_cols]
    .agg(["mean", "std", "count"])
    .reset_index()
)

selected_aggregate.columns = [
    "_".join(str(x) for x in col if str(x))
    if isinstance(col, tuple)
    else str(col)
    for col in selected_aggregate.columns
]

selected_aggregate.to_csv(
    BASE / "selected_policy_test_metrics_summary.csv",
    index=False,
)

append_section(
    lines,
    "SELECTED POLICY TEST SUMMARY",
    selected_aggregate.to_string(index=False),
)

# ------------------------------------------------------------------
# Selected mode counts
# ------------------------------------------------------------------
mode_col = (
    "effective_role_transfer_mode"
    if "effective_role_transfer_mode" in selected.columns
    else "role_transfer_mode"
)

if mode_col in selected.columns:
    mode_counts = selected[mode_col].value_counts(dropna=False)
    append_section(
        lines,
        "SELECTED MODE COUNTS",
        mode_counts.to_string(),
    )

    if "train_fraction" in selected.columns:
        mode_by_fraction = pd.crosstab(
            selected["train_fraction"],
            selected[mode_col],
            dropna=False,
        )
        append_section(
            lines,
            "SELECTED MODES BY FRACTION",
            mode_by_fraction.to_string(),
        )

# ------------------------------------------------------------------
# Fixed-policy paired comparisons against real-only XGBoost
# ------------------------------------------------------------------
real = (
    policy[
        policy["split"].astype(str).str.lower().eq("test")
        & policy["representation"].astype(str).eq("original_6144")
        & policy["model"].astype(str).str.lower().eq("xgboost")
    ]
    .pivot_table(
        index=["seed", "train_fraction", "model"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

real = real.rename(columns={
    "mae": "real_mae",
    "rmse": "real_rmse",
    "r2": "real_r2",
    "spearman": "real_spearman",
})

candidate_rows = policy[
    policy["split"].astype(str).str.lower().eq("test")
    & policy["representation"].astype(str).str.contains(
        "role_aware", case=False, na=False
    )
    & policy["model"].astype(str).str.lower().eq("xgboost")
].copy()

policy_identity_candidates = [
    "seed",
    "train_fraction",
    "representation",
    "model",
    "policy_id",
    "role_transfer_mode",
    "effective_role_transfer_mode",
    "donor_strategy",
    "label_strategy",
    "synthetic_multiplier",
    "min_similarity",
    "max_teacher_std",
]

policy_identity = [
    col for col in policy_identity_candidates
    if col in candidate_rows.columns
]

candidates = (
    candidate_rows
    .pivot_table(
        index=policy_identity,
        columns="metric",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

merged = candidates.merge(
    real,
    on=["seed", "train_fraction", "model"],
    how="left",
)

if merged["real_rmse"].isna().any():
    missing = merged.loc[
        merged["real_rmse"].isna(),
        ["seed", "train_fraction", "model"],
    ].drop_duplicates()
    raise RuntimeError(
        "Some candidate rows could not be matched to a real-only baseline:\n"
        + missing.to_string(index=False)
    )

group_candidates = [
    "train_fraction",
    "role_transfer_mode",
    "effective_role_transfer_mode",
    "donor_strategy",
    "label_strategy",
    "synthetic_multiplier",
    "min_similarity",
    "max_teacher_std",
]

fixed_group = [
    col for col in group_candidates
    if col in merged.columns
]

fixed_rows: list[dict] = []

for keys, sub in merged.groupby(fixed_group, dropna=False):
    if not isinstance(keys, tuple):
        keys = (keys,)

    row = dict(zip(fixed_group, keys))
    row["n_seeds"] = len(sub)

    for metric in ["mae", "rmse", "r2", "spearman"]:
        if metric not in sub.columns or f"real_{metric}" not in sub.columns:
            continue

        candidate_values = sub[metric].astype(float)
        real_values = sub[f"real_{metric}"].astype(float)
        delta = candidate_values - real_values

        row[f"real_{metric}_mean"] = real_values.mean()
        row[f"transfer_{metric}_mean"] = candidate_values.mean()
        row[f"{metric}_delta"] = delta.mean()
        row[f"{metric}_delta_std"] = delta.std()

        if metric in {"mae", "rmse"}:
            row[f"{metric}_better_seeds"] = int((delta < 0).sum())
        else:
            row[f"{metric}_better_seeds"] = int((delta > 0).sum())

        if len(sub) > 1 and not np.allclose(delta, 0):
            row[f"{metric}_paired_t_p"] = float(
                ttest_rel(candidate_values, real_values).pvalue
            )
        else:
            row[f"{metric}_paired_t_p"] = np.nan

    fixed_rows.append(row)

fixed_stats = pd.DataFrame(fixed_rows)

if len(fixed_stats):
    fixed_stats = fixed_stats.sort_values(
        ["train_fraction", "transfer_rmse_mean"]
    )
    fixed_stats.to_csv(
        BASE / "fixed_policy_stats_vs_real_only_xgboost.csv",
        index=False,
    )

    best_fixed = (
        fixed_stats
        .groupby("train_fraction", group_keys=False)
        .head(10)
    )

    append_section(
        lines,
        "BEST FIXED POLICIES VS REAL-ONLY XGBOOST",
        best_fixed.to_string(index=False),
    )

# ------------------------------------------------------------------
# Audit diagnostics
# ------------------------------------------------------------------
audit_group_candidates = [
    "train_fraction",
    "role_transfer_mode",
    "effective_role_transfer_mode",
    "donor_strategy",
    "label_strategy",
    "synthetic_multiplier",
]

audit_group = [
    col for col in audit_group_candidates
    if col in audit.columns
]

audit_numeric_candidates = [
    "n_candidates_generated",
    "n_candidates_accepted",
    "n_synthetic_train",
    "filter_acceptance_rate",
    "n_identical_skipped",
    "n_teacher_uncertainty_rejected",
    "n_zero_synthetic_policy_skipped",
    "n_same_context_donors_found",
    "n_role_changed_candidates_found",
    "changed_any_transferred_role_fraction",
    "changed_catalyst_fraction",
    "changed_ligand_fraction",
    "changed_base_fraction",
    "changed_solvent_fraction",
    "mean_teacher_std",
    "mean_synthetic_yield",
    "std_synthetic_yield",
]

audit_numeric = [
    col for col in audit_numeric_candidates
    if col in audit.columns
]

audit_summary = (
    audit
    .groupby(audit_group, dropna=False)[audit_numeric]
    .mean()
    .reset_index()
)

audit_summary.to_csv(
    BASE / "role_transfer_audit_summary.csv",
    index=False,
)

append_section(
    lines,
    "ROLE TRANSFER AUDIT SUMMARY",
    audit_summary.sort_values(
        [c for c in ["train_fraction", "n_synthetic_train"] if c in audit_summary.columns]
    ).to_string(index=False),
)

if "n_synthetic_train" in audit_summary.columns:
    zero_synthetic = audit_summary[
        audit_summary["n_synthetic_train"].fillna(0).eq(0)
    ]

    append_section(
        lines,
        "ZERO-SYNTHETIC POLICIES",
        (
            zero_synthetic.to_string(index=False)
            if len(zero_synthetic)
            else "None"
        ),
    )

    selected_audit_keys = [
        c for c in [
            "seed",
            "train_fraction",
            "model",
            "role_transfer_mode",
            "effective_role_transfer_mode",
            "donor_strategy",
            "label_strategy",
            "synthetic_multiplier",
        ]
        if c in selected.columns and c in audit.columns
    ]

    if selected_audit_keys:
        selected_with_audit = selected.merge(
            audit,
            on=selected_audit_keys,
            how="left",
            suffixes=("", "_audit"),
        )

        selected_with_audit.to_csv(
            BASE / "selected_policies_with_audit.csv",
            index=False,
        )

        show_cols = [
            c for c in [
                "seed",
                "train_fraction",
                "model",
                "role_transfer_mode",
                "effective_role_transfer_mode",
                "donor_strategy",
                "label_strategy",
                "synthetic_multiplier",
                "n_synthetic_train",
                "n_candidates_generated",
                "n_candidates_accepted",
                "changed_any_transferred_role_fraction",
                "donor_fallback_level",
            ]
            if c in selected_with_audit.columns
        ]

        append_section(
            lines,
            "SELECTED POLICIES WITH AUDIT",
            selected_with_audit[show_cols].to_string(index=False),
        )

# ------------------------------------------------------------------
# Existing delta outputs, if generated by the runner
# ------------------------------------------------------------------
for filename in [
    "role_transfer_vs_real_only_same_model_summary.csv",
    "role_transfer_vs_original_rf_summary.csv",
]:
    frame = read_csv(filename, required=False)
    if len(frame):
        append_section(
            lines,
            filename,
            frame.to_string(index=False),
        )

report_text = "\n".join(lines)
REPORT.write_text(report_text)

print(report_text)
print()
print(f"Saved report: {REPORT}")
print(f"Saved selected summary: {BASE / 'selected_policy_test_metrics_summary.csv'}")
print(f"Saved fixed-policy stats: {BASE / 'fixed_policy_stats_vs_real_only_xgboost.csv'}")
print(f"Saved audit summary: {BASE / 'role_transfer_audit_summary.csv'}")
PY

echo
echo "============================================================"
echo "RUN COMPLETE"
echo "Finished: $(date)"
echo "Main output: $OUT"
echo "Report: $OUT/overnight_report.txt"
echo "============================================================"
