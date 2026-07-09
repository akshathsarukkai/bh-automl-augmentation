from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from bh_augmentation.features.featurize import build_feature_matrix

OUT = Path("results/data_efficiency_comparison")
OUT.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.01, 0.05, 0.10, 0.20]
SEEDS = [0, 1, 2, 3, 4]

def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).correlation),
    }

def make_xgb(seed):
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )

def read_existing(path):
    path = Path(path)
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

def wide_from_metrics(df, id_cols):
    df = df.copy()
    df["metric"] = df["metric"].astype(str).str.strip()
    return (
        df[df["split"] == "test"]
        .pivot_table(index=id_cols, columns="metric", values="value")
        .reset_index()
    )

# Full-data XGBoost baseline
full_path = OUT / "full_data_xgboost_metrics.csv"

if not full_path.exists():
    df = pd.read_csv("data/processed/bh_clean_stress.csv")
    X_all, y_all, _ = build_feature_matrix(
        df,
        {"kind": "reaction_role_concat", "n_bits": 2048, "radius": 2},
    )

    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()

    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.float32)

    rows = []

    for seed in SEEDS:
        idx = np.arange(len(df))

        train_idx, temp_idx = train_test_split(
            idx,
            train_size=0.8,
            random_state=seed,
            shuffle=True,
        )

        valid_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            random_state=seed,
            shuffle=True,
        )

        model = make_xgb(seed)
        model.fit(X_all[train_idx], y_all[train_idx])

        for split_name, split_idx in [("valid", valid_idx), ("test", test_idx)]:
            pred = model.predict(X_all[split_idx])
            for metric, value in metrics(y_all[split_idx], pred).items():
                rows.append({
                    "seed": seed,
                    "train_fraction": 1.0,
                    "method": "full_data_xgboost",
                    "split": split_name,
                    "metric": metric,
                    "value": value,
                })

    pd.DataFrame(rows).to_csv(full_path, index=False)
    print("Saved", full_path)

# Read condition-transfer outputs
ct_policy_parts = []
for path in [
    "results/condition_transfer_medium_xgboost/policy_metrics.csv",
    "results/condition_transfer_20pct_xgboost/policy_metrics.csv",
]:
    df = read_existing(path)
    if len(df):
        ct_policy_parts.append(df)

ct_selected_parts = []
for path in [
    "results/condition_transfer_medium_xgboost/selected_policy_metrics.csv",
    "results/condition_transfer_20pct_xgboost/selected_policy_metrics.csv",
]:
    df = read_existing(path)
    if len(df):
        ct_selected_parts.append(df)

ct_policy = pd.concat(ct_policy_parts, ignore_index=True) if ct_policy_parts else pd.DataFrame()
ct_selected = pd.concat(ct_selected_parts, ignore_index=True) if ct_selected_parts else pd.DataFrame()

real_xgb = pd.DataFrame()
ct_xgb = pd.DataFrame()

if len(ct_policy):
    real_xgb = wide_from_metrics(
        ct_policy[
            (ct_policy["representation"] == "original_6144")
            & (ct_policy["model"] == "xgboost")
        ],
        ["seed", "train_fraction", "representation", "model"],
    )
    real_xgb["method"] = "real_only_xgboost"

if len(ct_selected):
    ct_xgb = wide_from_metrics(
        ct_selected[
            (ct_selected["representation"] == "condition_transfer")
            & (ct_selected["model"] == "xgboost")
        ],
        [
            "seed",
            "train_fraction",
            "representation",
            "model",
            "donor_strategy",
            "label_strategy",
            "synthetic_multiplier",
        ],
    )
    ct_xgb["method"] = "condition_transfer_xgboost"

# Read supervised AE/interpolation summaries
ae_parts = []
for path in [
    "results/supervised_ae_latent_interpolation_1pct_xgboost/summary.csv",
    "results/supervised_ae_latent_interpolation_confirm_xgboost/summary.csv",
]:
    df = read_existing(path)
    if len(df):
        ae_parts.append(df)

ae_best = pd.DataFrame()

if ae_parts:
    ae_summary = pd.concat(ae_parts, ignore_index=True)
    ae_summary["metric"] = ae_summary["metric"].astype(str).str.strip()

    ae_wide = (
        ae_summary[
            (ae_summary["split"] == "test")
            & (ae_summary["representation"].astype(str).str.contains("supervised_ae", na=False))
        ]
        .pivot_table(
            index=["train_fraction", "representation", "model"],
            columns="metric",
            values="mean",
        )
        .reset_index()
    )

    ae_best = (
        ae_wide.sort_values(["train_fraction", "rmse"])
        .groupby("train_fraction")
        .head(1)
        .copy()
    )
    ae_best["method"] = "best_supervised_ae_or_interpolation"

# Full-data XGBoost summary
full = pd.read_csv(full_path)
full["metric"] = full["metric"].astype(str).str.strip()

full_wide = (
    full[full["split"] == "test"]
    .pivot_table(index=["method"], columns="metric", values="value", aggfunc="mean")
    .reset_index()
)

rows = []

for frac in FRACTIONS:
    sub = real_xgb[np.isclose(real_xgb["train_fraction"], frac)] if len(real_xgb) else pd.DataFrame()
    if len(sub):
        rows.append({
            "train_fraction": frac,
            "method": "real_only_xgboost",
            "details": "original_6144 + xgboost",
            "mae": sub["mae"].mean(),
            "rmse": sub["rmse"].mean(),
            "r2": sub["r2"].mean(),
            "spearman": sub["spearman"].mean(),
        })

    sub = ct_xgb[np.isclose(ct_xgb["train_fraction"], frac)] if len(ct_xgb) else pd.DataFrame()
    if len(sub):
        rows.append({
            "train_fraction": frac,
            "method": "condition_transfer_xgboost",
            "details": "selected by valid RMSE",
            "mae": sub["mae"].mean(),
            "rmse": sub["rmse"].mean(),
            "r2": sub["r2"].mean(),
            "spearman": sub["spearman"].mean(),
        })

    sub = ae_best[np.isclose(ae_best["train_fraction"], frac)] if len(ae_best) else pd.DataFrame()
    if len(sub):
        r = sub.iloc[0]
        rows.append({
            "train_fraction": frac,
            "method": "best_supervised_ae_or_interpolation",
            "details": f"{r['representation']} + {r['model']}",
            "mae": r["mae"],
            "rmse": r["rmse"],
            "r2": r["r2"],
            "spearman": r["spearman"],
        })

    r = full_wide.iloc[0]
    rows.append({
        "train_fraction": frac,
        "method": "full_data_xgboost_reference",
        "details": "100% train split, original_6144 + xgboost",
        "mae": r["mae"],
        "rmse": r["rmse"],
        "r2": r["r2"],
        "spearman": r["spearman"],
    })

table = pd.DataFrame(rows)
table = table.sort_values(["train_fraction", "rmse"])
table.to_csv(OUT / "data_efficiency_comparison.csv", index=False)

print("\nDATA EFFICIENCY COMPARISON")
print(table.to_string(index=False))
print("\nSaved:", OUT / "data_efficiency_comparison.csv")
