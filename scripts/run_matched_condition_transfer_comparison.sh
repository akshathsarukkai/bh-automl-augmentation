#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROLE_CONFIG="configs/role_aware_condition_transfer_v2_xgboost.yaml"
ANON_CONFIG="configs/condition_transfer_matched_xgboost.yaml"
ANON_OUT="results/condition_transfer_matched_xgboost"
ROLE_OUT="results/role_aware_condition_transfer_v2_xgboost"
COMPARE_OUT="results/condition_transfer_matched_comparison"

mkdir -p "$COMPARE_OUT"

echo "============================================================"
echo "MATCHED ANONYMOUS VS ROLE-AWARE CONDITION TRANSFER"
echo "Started: $(date)"
echo "============================================================"

if [[ ! -f "$ROLE_CONFIG" ]]; then
  echo "ERROR: missing $ROLE_CONFIG"
  exit 1
fi

if [[ ! -f "$ROLE_OUT/selected_policy_metrics.csv" ]]; then
  echo "ERROR: role-aware v2 results are missing."
  echo "Expected: $ROLE_OUT/selected_policy_metrics.csv"
  exit 1
fi

echo
echo "STEP 1: CREATE MATCHED ANONYMOUS CONFIG"

python - <<'PY'
from copy import deepcopy
from pathlib import Path
import yaml

role_path = Path("configs/role_aware_condition_transfer_v2_xgboost.yaml")
out_path = Path("configs/condition_transfer_matched_xgboost.yaml")
anon_out = Path("results/condition_transfer_matched_xgboost")

candidates = [
    Path("configs/condition_transfer_xgboost.yaml"),
    Path("configs/condition_transfer_medium_xgboost.yaml"),
    Path("configs/condition_transfer_20pct_xgboost.yaml"),
]

candidates.extend(
    sorted(
        p for p in Path("configs").glob("*condition_transfer*xgboost*.yaml")
        if "role_aware" not in p.name
        and "matched" not in p.name
        and "tiny" not in p.name
    )
)

seen = set()
candidates = [
    p for p in candidates
    if not (str(p) in seen or seen.add(str(p)))
]

base_path = next((p for p in candidates if p.exists()), None)

if base_path is None:
    raise FileNotFoundError(
        "Could not locate an existing anonymous condition-transfer XGBoost config."
    )

role = yaml.safe_load(role_path.read_text())
anon = yaml.safe_load(base_path.read_text())

print("Anonymous base config:", base_path)
print("Role-aware reference config:", role_path)


def find_first(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_first(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_first(value, key)
            if found is not None:
                return found
    return None


def replace_existing_key(obj, key, value):
    replaced = 0
    if isinstance(obj, dict):
        if key in obj:
            obj[key] = deepcopy(value)
            replaced += 1
        for child in obj.values():
            replaced += replace_existing_key(child, key, value)
    elif isinstance(obj, list):
        for child in obj:
            replaced += replace_existing_key(child, key, value)
    return replaced


def find_xgb_params(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (
                str(key).lower() == "xgboost"
                and isinstance(value, dict)
                and any(
                    parameter in value
                    for parameter in [
                        "n_estimators",
                        "max_depth",
                        "learning_rate",
                        "subsample",
                        "colsample_bytree",
                    ]
                )
            ):
                return deepcopy(value)

        for value in obj.values():
            found = find_xgb_params(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = find_xgb_params(value)
            if found is not None:
                return found

    return None


def replace_xgb_params(obj, params):
    replaced = 0

    if isinstance(obj, dict):
        for key in list(obj):
            value = obj[key]

            if (
                str(key).lower() == "xgboost"
                and isinstance(value, dict)
                and any(
                    parameter in value
                    for parameter in [
                        "n_estimators",
                        "max_depth",
                        "learning_rate",
                        "subsample",
                        "colsample_bytree",
                    ]
                )
            ):
                obj[key] = deepcopy(params)
                replaced += 1
            else:
                replaced += replace_xgb_params(value, params)

    elif isinstance(obj, list):
        for value in obj:
            replaced += replace_xgb_params(value, params)

    return replaced


def rewrite_result_paths(obj):
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            normalized_key = str(key).lower()

            if normalized_key in {
                "output_dir",
                "output_directory",
                "results_dir",
                "result_dir",
            }:
                obj[key] = str(anon_out)

            elif (
                normalized_key == "directory"
                and isinstance(value, str)
                and "result" in value.lower()
            ):
                obj[key] = str(anon_out)

            elif (
                normalized_key.endswith("_path")
                and isinstance(value, str)
            ):
                obj[key] = str(anon_out / Path(value).name)

            else:
                rewrite_result_paths(value)

    elif isinstance(obj, list):
        for value in obj:
            rewrite_result_paths(value)


# Shared experimental settings.
fractions = find_first(role, "train_fractions") or [0.01, 0.05, 0.10, 0.20]
seeds = find_first(role, "seeds") or [0, 1, 2, 3, 4]
models = find_first(role, "models") or ["xgboost"]
multipliers = find_first(role, "synthetic_multipliers") or [0.5, 1.0]
policy_budget = find_first(role, "max_policies")
xgb_params = find_xgb_params(role)

# The completed v2 audit had seven policies per seed/fraction.
if policy_budget is None:
    policy_budget = 7

replace_existing_key(anon, "train_fractions", fractions)
replace_existing_key(anon, "seeds", seeds)
replace_existing_key(anon, "models", models)
replace_existing_key(anon, "synthetic_multipliers", multipliers)
replace_existing_key(anon, "max_policies", policy_budget)

# Add standard locations in case the anonymous config lacks them.
anon["seeds"] = seeds
anon["models"] = ["xgboost"]
anon["max_policies"] = policy_budget

anon.setdefault("low_data", {})
if isinstance(anon["low_data"], dict):
    anon["low_data"]["train_fractions"] = fractions

anon["synthetic_multipliers"] = multipliers

if xgb_params is not None:
    changed = replace_xgb_params(anon, xgb_params)

    if changed == 0:
        anon.setdefault("model_params", {})
        anon["model_params"]["xgboost"] = deepcopy(xgb_params)

rewrite_result_paths(anon)

anon.setdefault("output", {})
if isinstance(anon["output"], dict):
    anon["output"]["directory"] = str(anon_out)

anon["output_dir"] = str(anon_out)

out_path.write_text(yaml.safe_dump(anon, sort_keys=False))

print("Fractions:", fractions)
print("Seeds:", seeds)
print("Synthetic multipliers:", multipliers)
print("Policy budget:", policy_budget)
print("XGBoost params:", xgb_params)
print("Wrote:", out_path)
PY

echo
echo "STEP 2: ARCHIVE OLD MATCHED OUTPUT"

if [[ -d "$ANON_OUT" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  mv "$ANON_OUT" "${ANON_OUT}_backup_${STAMP}"
fi

echo
echo "STEP 3: FIND AND RUN ANONYMOUS CONDITION-TRANSFER RUNNER"

if [[ -f "src/bh_augmentation/run_condition_transfer.py" ]]; then
  python -m bh_augmentation.run_condition_transfer \
    --config "$ANON_CONFIG"

elif [[ -f "src/bh_augmentation/run_condition_transfer_augmentation.py" ]]; then
  python -m bh_augmentation.run_condition_transfer_augmentation \
    --config "$ANON_CONFIG"

elif [[ -f "scripts/run_condition_transfer.py" ]]; then
  python scripts/run_condition_transfer.py \
    --config "$ANON_CONFIG"

elif [[ -f "scripts/run_condition_transfer_augmentation.py" ]]; then
  python scripts/run_condition_transfer_augmentation.py \
    --config "$ANON_CONFIG"

else
  echo "ERROR: could not locate the anonymous condition-transfer runner."
  echo "Candidate files:"
  find src scripts -maxdepth 3 -iname '*condition*transfer*.py' 2>/dev/null || true
  exit 2
fi

echo
echo "STEP 4: VERIFY OUTPUTS"

for file in \
  "$ANON_OUT/policy_metrics.csv" \
  "$ANON_OUT/selected_policy_metrics.csv" \
  "$ROLE_OUT/policy_metrics.csv" \
  "$ROLE_OUT/selected_policy_metrics.csv"
do
  if [[ ! -s "$file" ]]; then
    echo "ERROR: missing or empty $file"
    exit 3
  fi
  echo "OK: $file"
done

echo
echo "STEP 5: PAIRED COMPARISON"

python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

anon_dir = Path("results/condition_transfer_matched_xgboost")
role_dir = Path("results/role_aware_condition_transfer_v2_xgboost")
out_dir = Path("results/condition_transfer_matched_comparison")
out_dir.mkdir(parents=True, exist_ok=True)


def load_selected(directory: Path, prefix: str) -> pd.DataFrame:
    path = directory / "selected_policy_metrics.csv"
    df = pd.read_csv(path)

    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    df = df[
        (df["split"] == "test")
        & (df["model"] == "xgboost")
    ].copy()

    wide = (
        df.pivot_table(
            index=["seed", "train_fraction", "model"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    return wide.rename(columns={
        "mae": f"{prefix}_mae",
        "rmse": f"{prefix}_rmse",
        "r2": f"{prefix}_r2",
        "spearman": f"{prefix}_spearman",
    })


def load_baseline(directory: Path, prefix: str) -> pd.DataFrame:
    path = directory / "policy_metrics.csv"
    df = pd.read_csv(path)

    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    representation = df["representation"].astype(str).str.lower()

    baseline = df[
        (df["split"] == "test")
        & (df["model"] == "xgboost")
        & (
            representation.eq("original_6144")
            | representation.str.contains("real_only", na=False)
            | representation.str.contains("baseline", na=False)
        )
    ].copy()

    if baseline.empty:
        raise RuntimeError(
            f"No real-only baseline rows found in {path}. "
            f"Representations were: {sorted(df['representation'].dropna().unique())}"
        )

    wide = (
        baseline.pivot_table(
            index=["seed", "train_fraction", "model"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    return wide.rename(columns={
        "mae": f"{prefix}_baseline_mae",
        "rmse": f"{prefix}_baseline_rmse",
        "r2": f"{prefix}_baseline_r2",
        "spearman": f"{prefix}_baseline_spearman",
    })


anon = load_selected(anon_dir, "anonymous")
role = load_selected(role_dir, "role_aware")
anon_base = load_baseline(anon_dir, "anonymous")
role_base = load_baseline(role_dir, "role_aware")

paired = (
    anon
    .merge(role, on=["seed", "train_fraction", "model"], how="inner")
    .merge(anon_base, on=["seed", "train_fraction", "model"], how="left")
    .merge(role_base, on=["seed", "train_fraction", "model"], how="left")
)

expected = 5 * 4
if len(paired) != expected:
    print(
        f"WARNING: expected {expected} paired rows but found {len(paired)}."
    )

for metric in ["mae", "rmse", "r2", "spearman"]:
    paired[f"baseline_{metric}_difference"] = (
        paired[f"role_aware_baseline_{metric}"]
        - paired[f"anonymous_baseline_{metric}"]
    )

    paired[f"anonymous_{metric}_delta_vs_baseline"] = (
        paired[f"anonymous_{metric}"]
        - paired[f"anonymous_baseline_{metric}"]
    )

    paired[f"role_aware_{metric}_delta_vs_baseline"] = (
        paired[f"role_aware_{metric}"]
        - paired[f"role_aware_baseline_{metric}"]
    )

    paired[f"role_aware_minus_anonymous_{metric}"] = (
        paired[f"role_aware_{metric}"]
        - paired[f"anonymous_{metric}"]
    )

baseline_max_difference = paired[
    [
        "baseline_mae_difference",
        "baseline_rmse_difference",
        "baseline_r2_difference",
        "baseline_spearman_difference",
    ]
].abs().max()

print("\nMAX ABSOLUTE BASELINE DIFFERENCE")
print(baseline_max_difference.to_string())

if baseline_max_difference.max() > 1e-8:
    print(
        "\nWARNING: baseline rows are not exactly identical. "
        "Check that both runners use the same train/valid/test split logic."
    )
else:
    print("\nPASS: real-only baselines match exactly.")

rows = []

for fraction, sub in paired.groupby("train_fraction"):
    row = {
        "train_fraction": fraction,
        "n_seeds": len(sub),
        "baseline_rmse": sub["anonymous_baseline_rmse"].mean(),
        "anonymous_rmse": sub["anonymous_rmse"].mean(),
        "role_aware_rmse": sub["role_aware_rmse"].mean(),
        "anonymous_delta_vs_baseline": (
            sub["anonymous_rmse"]
            - sub["anonymous_baseline_rmse"]
        ).mean(),
        "role_aware_delta_vs_baseline": (
            sub["role_aware_rmse"]
            - sub["role_aware_baseline_rmse"]
        ).mean(),
        "role_aware_minus_anonymous_rmse": (
            sub["role_aware_rmse"]
            - sub["anonymous_rmse"]
        ).mean(),
        "role_aware_better_seeds": int(
            (sub["role_aware_rmse"] < sub["anonymous_rmse"]).sum()
        ),
    }

    try:
        row["paired_t_p"] = float(
            ttest_rel(
                sub["role_aware_rmse"],
                sub["anonymous_rmse"],
            ).pvalue
        )
    except Exception:
        row["paired_t_p"] = np.nan

    try:
        difference = (
            sub["role_aware_rmse"] - sub["anonymous_rmse"]
        )

        row["wilcoxon_p"] = (
            float(
                wilcoxon(
                    sub["role_aware_rmse"],
                    sub["anonymous_rmse"],
                ).pvalue
            )
            if not np.allclose(difference, 0)
            else np.nan
        )
    except Exception:
        row["wilcoxon_p"] = np.nan

    for metric in ["mae", "r2", "spearman"]:
        row[f"anonymous_{metric}"] = sub[f"anonymous_{metric}"].mean()
        row[f"role_aware_{metric}"] = sub[f"role_aware_{metric}"].mean()
        row[f"role_aware_minus_anonymous_{metric}"] = (
            sub[f"role_aware_{metric}"]
            - sub[f"anonymous_{metric}"]
        ).mean()

    rows.append(row)

summary = pd.DataFrame(rows).sort_values("train_fraction")

paired_path = out_dir / "anonymous_vs_role_aware_by_seed.csv"
summary_path = out_dir / "anonymous_vs_role_aware_summary.csv"

paired.to_csv(paired_path, index=False)
summary.to_csv(summary_path, index=False)

display_columns = [
    "train_fraction",
    "n_seeds",
    "baseline_rmse",
    "anonymous_rmse",
    "role_aware_rmse",
    "anonymous_delta_vs_baseline",
    "role_aware_delta_vs_baseline",
    "role_aware_minus_anonymous_rmse",
    "role_aware_better_seeds",
    "paired_t_p",
    "wilcoxon_p",
]

print("\nANONYMOUS VS ROLE-AWARE CONDITION TRANSFER")
print("Negative role_aware_minus_anonymous_rmse means role-aware is better.")
print(summary[display_columns].to_string(index=False))

print("\nSaved:")
print(paired_path)
print(summary_path)
PY

echo
echo "============================================================"
echo "COMPARISON COMPLETE"
echo "Finished: $(date)"
echo "Summary: $COMPARE_OUT/anonymous_vs_role_aware_summary.csv"
echo "============================================================"
