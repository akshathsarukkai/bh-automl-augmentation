#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETED_TESTS=(
  tests/test_corrected_representation_baselines.py
  tests/test_corrected_condition_transfer_configs.py
  tests/test_corrected_condition_transfer_matching.py
  tests/test_corrected_condition_transfer_comparison.py
  tests/test_corrected_run_manifests.py
)

echo "STEP 1/9: targeted corrected-revalidation tests"
python -m pytest "${TARGETED_TESTS[@]}" -q

echo "STEP 2/9: full test suite"
python -m pytest -q

echo "STEP 3/9: feature-contract report"
python scripts/build_corrected_feature_contract_report.py

echo "STEP 4/9: corrected real-only representation baselines"
python -m bh_augmentation.run_corrected_representation_baselines \
  --config configs/corrected_representation_baselines_xgboost.yaml

echo "STEP 5/9: corrected anonymous condition transfer"
python -m bh_augmentation.run_condition_transfer \
  --config configs/corrected_anonymous_condition_transfer_xgboost.yaml

echo "STEP 6/9: corrected role-aware condition transfer"
python -m bh_augmentation.run_role_aware_condition_transfer \
  --config configs/corrected_role_aware_condition_transfer_xgboost.yaml

echo "STEP 7/9: corrected paired comparison"
python scripts/compare_corrected_condition_transfer.py \
  --anonymous-dir results/corrected_anonymous_condition_transfer_xgboost \
  --role-aware-dir results/corrected_role_aware_condition_transfer_xgboost \
  --output-dir results/corrected_condition_transfer_comparison

echo "STEP 8/9: status-aware data-efficiency report"
python scripts/data_efficiency_comparison.py \
  --output-dir results/corrected_data_efficiency_comparison

echo "STEP 9/9: validate corrected outputs"
python - <<'PY'
from pathlib import Path
import json
import numpy as np

from bh_augmentation.results.status import read_result_csv

required = {
    "results/corrected_feature_contract_report": [
        "feature_contract.json", "feature_contract.csv", "identity_equivalence_summary.csv",
        "role_locality_summary.csv", "README.txt",
    ],
    "results/corrected_representation_baselines_xgboost": [
        "policy_metrics.csv", "summary.csv", "representation_metadata.csv",
        "representation_metadata.json", "split_audit.csv", "run_manifest.json",
    ],
    "results/corrected_anonymous_condition_transfer_xgboost": [
        "policy_metrics.csv", "selected_policies.csv", "selected_policy_metrics.csv",
        "synthetic_audit.csv", "feature_compatibility_audit.csv", "summary.csv",
        "split_audit.csv", "run_manifest.json",
    ],
    "results/corrected_role_aware_condition_transfer_xgboost": [
        "policy_metrics.csv", "selected_policies.csv", "selected_policy_metrics.csv",
        "role_transfer_audit.csv", "role_value_counts.csv", "feature_compatibility_audit.csv",
        "summary.csv", "split_audit.csv", "run_manifest.json",
    ],
    "results/corrected_condition_transfer_comparison": [
        "anonymous_vs_role_aware_by_seed.csv", "anonymous_vs_role_aware_summary.csv",
        "fixed_policy_stats_vs_real_only.csv", "final_corrected_decision_table.csv",
        "comparison_manifest.json", "comparison_report.txt",
    ],
    "results/corrected_data_efficiency_comparison": [
        "data_efficiency_comparison.csv", "data_efficiency_long.csv",
    ],
}
for directory, names in required.items():
    for name in names:
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty corrected output: {path}")

for directory in [
    Path("results/corrected_anonymous_condition_transfer_xgboost"),
    Path("results/corrected_role_aware_condition_transfer_xgboost"),
]:
    selected = read_result_csv(directory / "selected_policy_metrics.csv")
    if (selected["n_synthetic_train"].astype(int) <= 0).any():
        raise SystemExit(f"Zero-synthetic selected policy found in {directory}")
    if not np.isfinite(selected["value"].to_numpy(dtype=float)).all():
        raise SystemExit(f"NaN/inf selected metric found in {directory}")
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("historical_results_loaded") is not False:
        raise SystemExit(f"Historical result access recorded in {directory}")

print("All corrected output checks passed.")
PY

echo "Corrected revalidation completed successfully."
