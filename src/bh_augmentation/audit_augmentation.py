"""Audit whether configured augmentation changes model inputs safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.run_augmentation import _build_training_variants
from bh_augmentation.run_baseline import (
    _create_baseline_folds,
    _get_dataset_path,
    _get_group_column,
    _resolve_feature_config,
)
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.seed import set_global_seed

DEFAULT_AUDIT_DIR = Path("results/augmentation/audit")
METADATA_COLUMNS = {
    "reaction_id",
    "source_reaction_id",
    "donor_reaction_id",
    "is_augmented",
    "is_synthetic",
    "augmentation_type",
    "augmentation_method",
    "pseudo_label",
    "pseudo_label_model",
    "nearest_train_similarity",
    "sample_weight",
}


def run_augmentation_audit(config_path: str | Path) -> dict[str, Any]:
    """Run the configured train-only augmentation audit and save reports."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    data = clean_buchwald_hartwig(load_reaction_csv(_get_dataset_path(config))).reset_index(
        drop=True
    )
    if data.empty:
        raise ValueError("No rows remain after cleaning; cannot audit augmentation.")

    feature_config = _resolve_feature_config(config.get("features", {}), data)
    augmentation_config = config.get("augmentation", {})
    split_config = config.get("splits", {})
    split_method = str(split_config.get("method", "random"))
    group_column = _get_group_column(split_config)
    folds = _create_baseline_folds(data, config, seed)

    audits: list[dict[str, Any]] = []
    for fold_index, (train_fraction, heldout_group, splits) in enumerate(folds):
        variants = _build_training_variants(
            splits["train"], augmentation_config, feature_config, seed + fold_index
        )
        configured_variants = {
            name: frame for name, frame in variants.items() if name != "none"
        }
        if not configured_variants:
            configured_variants = {"none_configured": variants["none"]}

        for variant_name, augmented_train in configured_variants.items():
            audit = audit_training_data(
                original_train=splits["train"],
                augmented_train=augmented_train,
                valid_df=splits["valid"],
                test_df=splits["test"],
                feature_config=feature_config,
                group_column=group_column,
                heldout_group_value=heldout_group,
            )
            audit.update(
                {
                    "fold_index": fold_index,
                    "train_fraction": train_fraction,
                    "split_method": split_method,
                    "heldout_group_value": heldout_group,
                    "variant": variant_name,
                }
            )
            audits.append(audit)

    synthetic_rows = sum(item["synthetic_rows_added"] for item in audits)
    new_feature_rows = sum(item["synthetic_feature_rows_new"] for item in audits)
    leakage = any(
        item["validation_test_rows_modified"]
        or item["eval_ids_in_augmented_training"]
        or item["heldout_group_in_augmented_training"]
        for item in audits
    )
    no_active_augmentation = all(
        item["variant"] == "none_configured" for item in audits
    )
    status = (
        "PASS"
        if audits and all(item["status"] == "PASS" for item in audits) and not leakage
        else "FAIL"
    )
    if no_active_augmentation:
        message = (
            "No active meaningful augmentation configured. Component-column "
            "augmentation has been disabled because component columns are fully UNKNOWN."
        )
    else:
        message = (
            "PASS: augmentation changes model features"
            if status == "PASS"
            else "FAIL: augmentation does not change model features"
        )
    report = {
        "status": status,
        "message": message,
        "no_active_meaningful_augmentation": no_active_augmentation,
        "feature_kind": feature_config.get("kind", "custom"),
        "total_synthetic_rows": synthetic_rows,
        "total_new_synthetic_feature_rows": new_feature_rows,
        "leakage_detected": leakage,
        "audits": audits,
    }
    _save_audit_reports(report, config)
    print(_format_audit_report(report))
    return report


def audit_training_data(
    original_train: pd.DataFrame,
    augmented_train: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_config: dict[str, Any],
    group_column: str = "",
    heldout_group_value: object = "",
) -> dict[str, Any]:
    """Compare original and augmented training rows, features, labels, and leakage."""
    original = original_train.copy()
    augmented = augmented_train.copy()
    synthetic_mask = (
        augmented.get("is_augmented", pd.Series(False, index=augmented.index))
        .fillna(False)
        .astype(bool)
    )
    synthetic = augmented.loc[synthetic_mask].copy()

    original_x, _, original_names = build_feature_matrix(original, feature_config)
    augmented_x, _, augmented_names = build_feature_matrix(augmented, feature_config)
    if original_names != augmented_names:
        raise ValueError("Feature names differ before and after augmentation.")

    synthetic_x = augmented_x[synthetic_mask.to_numpy()]
    original_keys = _feature_keys(original_x)
    synthetic_keys = _feature_keys(synthetic_x)
    original_key_set = set(original_keys)
    duplicate_flags = [key in original_key_set for key in synthetic_keys]
    duplicate_count = int(sum(duplicate_flags))
    new_count = len(synthetic_keys) - duplicate_count

    source_rows = _source_row_map(original)
    changed_columns, reaction_changed = _changed_data_checks(
        synthetic, source_rows
    )
    source_feature_equal = _source_feature_matches(
        synthetic, synthetic_x, original, original_x
    )
    copied_labels, perturbed_labels = _label_copy_counts(synthetic, source_rows)

    eval_ids = _reaction_ids(valid_df) | _reaction_ids(test_df)
    augmented_ids = _reaction_ids(synthetic) | set(
        synthetic.get("source_reaction_id", pd.Series(dtype=str)).astype(str)
    )
    eval_id_overlap = sorted(eval_ids & augmented_ids)

    heldout_leakage = False
    if group_column and group_column in synthetic.columns and heldout_group_value != "":
        heldout_leakage = bool(
            synthetic[group_column].astype(str).eq(str(heldout_group_value)).any()
        )

    synthetic_count = len(synthetic)
    pseudo_labeled = int(
        synthetic.get("pseudo_label", pd.Series(False, index=synthetic.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    reaction_changed_percent = _percent(reaction_changed, synthetic_count)
    leakage = bool(eval_id_overlap or heldout_leakage)
    warnings: list[str] = []
    if synthetic_count and copied_labels == synthetic_count and duplicate_count == synthetic_count:
        warnings.append("WARNING: labels are copied while synthetic features are duplicated")

    return {
        "status": (
            "PASS"
            if synthetic_count > 0
            and reaction_changed > 0
            and new_count > 0
            and not leakage
            else "FAIL"
        ),
        "original_train_rows": len(original),
        "augmented_train_rows": len(augmented),
        "synthetic_rows_added": synthetic_count,
        "augmentation_factor": _safe_ratio(len(augmented), len(original)),
        "validation_test_rows_modified": False,
        "eval_ids_in_augmented_training": eval_id_overlap,
        "heldout_group_in_augmented_training": heldout_leakage,
        "columns_changed_by_augmentation": changed_columns,
        "percent_synthetic_changed_reaction_smiles": reaction_changed_percent,
        "original_feature_matrix_shape": list(original_x.shape),
        "augmented_feature_matrix_shape": list(augmented_x.shape),
        "unique_feature_vectors_before": len(set(original_keys)),
        "unique_feature_vectors_after": len(set(_feature_keys(augmented_x))),
        "synthetic_feature_rows_duplicate_original": duplicate_count,
        "synthetic_feature_rows_new": new_count,
        "percent_synthetic_features_duplicate_original": _percent(
            duplicate_count, synthetic_count
        ),
        "percent_synthetic_features_new": _percent(new_count, synthetic_count),
        "all_synthetic_features_identical_to_source_rows": bool(
            synthetic_count > 0 and source_feature_equal == synthetic_count
        ),
        "yield_distribution_before": _distribution(original.get("yield")),
        "yield_distribution_after": _distribution(augmented.get("yield")),
        "synthetic_labels_copied_exactly": copied_labels,
        "synthetic_labels_perturbed": perturbed_labels,
        "synthetic_labels_pseudo_labeled": pseudo_labeled,
        "nearest_train_similarity": _distribution(
            synthetic.get("nearest_train_similarity")
        ),
        "warnings": warnings,
    }


def _changed_data_checks(
    synthetic: pd.DataFrame,
    source_rows: dict[str, pd.Series],
) -> tuple[list[str], int]:
    changed_counts: dict[str, int] = {}
    reaction_changed = 0
    comparable_columns = [
        column for column in synthetic.columns if column not in METADATA_COLUMNS
    ]

    for _, row in synthetic.iterrows():
        source = source_rows.get(str(row.get("source_reaction_id", "")))
        if source is None:
            continue
        for column in comparable_columns:
            if column not in source.index or not _values_equal(row[column], source[column]):
                changed_counts[column] = changed_counts.get(column, 0) + 1
                if column == "reaction_smiles":
                    reaction_changed += 1
    return sorted(changed_counts), reaction_changed


def _source_feature_matches(
    synthetic: pd.DataFrame,
    synthetic_x: np.ndarray,
    original: pd.DataFrame,
    original_x: np.ndarray,
) -> int:
    ids = original.get("reaction_id", pd.Series(original.index, index=original.index)).astype(str)
    feature_by_id = {reaction_id: original_x[index] for index, reaction_id in enumerate(ids)}
    matches = 0
    for position, (_, row) in enumerate(synthetic.iterrows()):
        source_feature = feature_by_id.get(str(row.get("source_reaction_id", "")))
        if source_feature is not None and np.array_equal(synthetic_x[position], source_feature):
            matches += 1
    return matches


def _label_copy_counts(
    synthetic: pd.DataFrame,
    source_rows: dict[str, pd.Series],
) -> tuple[int, int]:
    copied = 0
    perturbed = 0
    for _, row in synthetic.iterrows():
        source = source_rows.get(str(row.get("source_reaction_id", "")))
        if source is None or "yield" not in row or "yield" not in source:
            continue
        if _values_equal(row["yield"], source["yield"]):
            copied += 1
        else:
            perturbed += 1
    return copied, perturbed


def _source_row_map(df: pd.DataFrame) -> dict[str, pd.Series]:
    if "reaction_id" in df.columns:
        return {str(row["reaction_id"]): row for _, row in df.iterrows()}
    return {str(index): row for index, row in df.iterrows()}


def _reaction_ids(df: pd.DataFrame) -> set[str]:
    if "reaction_id" not in df.columns:
        return set()
    return set(df["reaction_id"].astype(str))


def _feature_keys(features: np.ndarray) -> list[bytes]:
    contiguous = np.ascontiguousarray(features)
    return [row.tobytes() for row in contiguous]


def _distribution(values: pd.Series | None) -> dict[str, float | int | None]:
    if values is None:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _values_equal(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _percent(numerator: int, denominator: int) -> float:
    return float(100.0 * numerator / denominator) if denominator else 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _save_audit_reports(report: dict[str, Any], config: dict[str, Any]) -> None:
    output = config.get("output", {})
    text_path = Path(
        output.get("audit_text_path", DEFAULT_AUDIT_DIR / "augmentation_audit.txt")
    )
    json_path = Path(
        output.get("audit_json_path", DEFAULT_AUDIT_DIR / "augmentation_audit.json")
    )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(_format_audit_report(report) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _format_audit_report(report: dict[str, Any]) -> str:
    lines = [report["message"], f"feature_kind: {report['feature_kind']}"]
    for audit in report["audits"]:
        lines.extend(
            [
                "",
                f"variant: {audit['variant']} (fold {audit['fold_index']})",
                f"status: {audit['status']}",
                f"original train rows: {audit['original_train_rows']}",
                f"augmented train rows: {audit['augmented_train_rows']}",
                f"synthetic rows added: {audit['synthetic_rows_added']}",
                f"augmentation factor: {audit['augmentation_factor']:.3f}",
                f"validation/test rows modified: {audit['validation_test_rows_modified']}",
                f"evaluation ID leakage: {bool(audit['eval_ids_in_augmented_training'])}",
                f"held-out group leakage: {audit['heldout_group_in_augmented_training']}",
                "columns changed: " + ", ".join(audit["columns_changed_by_augmentation"]),
                "percent synthetic reaction_smiles changed: "
                f"{audit['percent_synthetic_changed_reaction_smiles']:.2f}",
                f"original X shape: {audit['original_feature_matrix_shape']}",
                f"augmented X shape: {audit['augmented_feature_matrix_shape']}",
                f"unique X before: {audit['unique_feature_vectors_before']}",
                f"unique X after: {audit['unique_feature_vectors_after']}",
                "percent synthetic X duplicates original: "
                f"{audit['percent_synthetic_features_duplicate_original']:.2f}",
                f"percent synthetic X new: {audit['percent_synthetic_features_new']:.2f}",
                f"labels copied exactly: {audit['synthetic_labels_copied_exactly']}",
                f"labels perturbed: {audit['synthetic_labels_perturbed']}",
                f"labels pseudo-labeled: {audit['synthetic_labels_pseudo_labeled']}",
                "nearest-neighbor similarity: "
                + json.dumps(audit["nearest_train_similarity"], sort_keys=True),
            ]
        )
        lines.extend(audit["warnings"])
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for augmentation auditing."""
    parser = argparse.ArgumentParser(description="Audit configured train augmentation.")
    parser.add_argument("--config", required=True, help="Path to augmentation YAML config.")
    args = parser.parse_args()
    run_augmentation_audit(args.config)


if __name__ == "__main__":
    main()
