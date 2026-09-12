"""Preregistered confirmatory analysis for observed-only condition transfer.

Every quantity, threshold and decision rule in this module is fixed by
``PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md``, preregistered at commit
``c6aab9e`` before any confirmatory outer-test outcome for this experiment
existed.  The module reads the confirmatory arms produced by the frozen-policy
protocol (:mod:`bh_augmentation.policy_search` and
:mod:`bh_augmentation.final_evaluation`), the degenerate-unit records written for
control units whose candidate pool accepted nothing, and the validation-only
pool accounting that regenerates every unit's candidate pool under both
eligibility rules, and computes the primary estimand mechanically:

* the paired per-seed RMSE reduction, comparator minus augmented, at training
  fraction 0.05 on seeds 6 through 14;
* the mean paired effect with a 95% percentile bootstrap over seeds as
  clusters, 10,000 replicates, fixed seed 1601;
* improvement consistency and the secondary Wilcoxon signed-rank p-value;
* the accepted-synthetic-row count per unit under both eligibility rules, so
  the size of the treatment is visible next to its effect;
* every degenerate, failed and missing unit, for every arm, so all 27 planned
  (unit, arm) pairs are accounted for;
* the withheld-cell oracle metrics, labelled secondary and non-selecting;
* the verdict from the preregistered interpretation table, evaluated as a
  boolean function of the numbers.

Nothing here re-cuts the result by fraction, metric, subgroup or seed subset.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.policy_search import load_degenerate_unit
from bh_augmentation.primary_confirmatory_analysis import (
    interpretation_table_verdict,
    seed_cluster_percentile_interval,
    wilcoxon_secondary,
)
from bh_augmentation.utils.corrected_runs import stable_hash

OBSERVED_ONLY_CONFIRMATORY_SCHEMA_VERSION = "bh-observed-only-confirmatory-analysis-v1"

#: Commit that froze the preregistration before any confirmatory outcome.
PREREGISTRATION_COMMIT = "c6aab9e705613d5b5fca3ffa145f54db4fdaa463"
PREREGISTRATION_DOCUMENT = "PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md"

#: Section 4 of the preregistration.
PRIMARY_FAMILY = "observed_only_condition_transfer"
COMPARATOR_FAMILY = "matched_real_only_control"
SECONDARY_CONTROL_FAMILY = "globally_unmeasured_condition_transfer"
PRIMARY_SCOPE_MODE = "observed_only_low_data"
SECONDARY_CONTROL_SCOPE_MODE = "globally_unmeasured_prospective"
PRIMARY_METRIC = "rmse"
PRIMARY_TRAIN_FRACTION = 0.05
PLANNED_SEEDS: tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12, 13, 14)
PRACTICAL_MINIMUM_EFFECT = 1.0
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 1601
#: The transfer kind the confirmatory arms run; typed transfer is descriptive.
TRANSFER_KIND = "anonymous"

#: Section 6 of the preregistration.
MAXIMUM_DEGENERATE_UNITS = 2
MINIMUM_INCLUDED_UNITS = 7

#: Section 7 of the preregistration.
MINIMUM_IMPROVED_UNITS_FOR_POSITIVE = 7

ARM_FAMILIES = (PRIMARY_FAMILY, COMPARATOR_FAMILY, SECONDARY_CONTROL_FAMILY)
POOL_COUNT_FIELDS = (
    "generated_candidate_count",
    "accepted_candidate_count",
    "unique_accepted_identity_count",
    "rejected_observed_in_labeled_train",
    "rejected_already_measured",
    "rejected_quarantined_held_out_identity",
    "rejected_source_identical",
    "rejected_duplicate_synthetic",
    "rejected_other",
)
ORACLE_FIELDS = (
    "generated_candidate_count",
    "unique_canonical_candidate_count",
    "hidden_overlap_count",
    "hidden_overlap_fraction_of_candidates",
    "hidden_cell_coverage_fraction",
    "pseudo_label_mae",
    "pseudo_label_rmse",
    "pseudo_label_spearman",
    "pseudo_label_bias",
    "high_yield_precision",
    "high_yield_recall",
    "high_yield_enrichment",
)


class ObservedOnlyConfirmatoryAnalysisError(ValueError):
    """Raised when the confirmatory artifacts cannot support the analysis."""


def evaluation_unit(seed: int, fraction: float = PRIMARY_TRAIN_FRACTION) -> str:
    """The frozen-policy protocol's evaluation-unit string for one seed."""
    return f"seed={int(seed)}|train_fraction={fraction:.12g}|comparison=model_policy_search"


def preregistration_plan_hash(repository_root: str | Path = ".") -> str:
    """SHA-256 of the preregistration exactly as frozen at its commit."""
    content = subprocess.run(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:{PREREGISTRATION_DOCUMENT}"],
        cwd=str(repository_root),
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


# --------------------------------------------------------------- arm loading


def load_arm_unit(confirmatory_root: str | Path, family: str, seed: int) -> dict[str, Any]:
    """Load one (arm, seed) unit and classify it.

    ``status`` is ``complete`` (a single hash-verified outer-test evaluation),
    ``degenerate`` (a sealed record that no typed policy accepted a synthetic
    row, so no outer test was read), or ``missing``.  Any inconsistency between
    the search, final and claim artifacts raises rather than being repaired.
    """
    unit_dir = Path(confirmatory_root) / family / f"seed_{int(seed)}"
    unit = evaluation_unit(seed)
    record: dict[str, Any] = {
        "family": family,
        "seed": int(seed),
        "evaluation_unit": unit,
        "directory": str(unit_dir),
    }
    metrics_path = unit_dir / "final" / "final_test_metrics.csv"
    degenerate_path = unit_dir / "search" / "degenerate_unit.json"
    if metrics_path.is_file():
        record.update(_load_complete_unit(unit_dir, family=family, seed=seed, unit=unit))
        record["status"] = "complete"
        return record
    if degenerate_path.is_file():
        record.update(_load_degenerate_arm(degenerate_path, family=family, unit=unit))
        record["status"] = "degenerate"
        return record
    if unit_dir.exists() and any(unit_dir.rglob("*")):
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed} has artifacts but neither a final result nor a "
            "degenerate record; the unit is in an unrecognised state."
        )
    record["status"] = "missing"
    return record


def _load_complete_unit(unit_dir: Path, *, family: str, seed: int, unit: str) -> dict[str, Any]:
    metrics = pd.read_csv(unit_dir / "final" / "final_test_metrics.csv", float_precision="round_trip")
    rows = metrics.loc[
        metrics["split"].astype(str).eq("test") & metrics["metric"].astype(str).eq(PRIMARY_METRIC)
    ]
    if len(rows) != 1:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: expected exactly one outer-test {PRIMARY_METRIC} row, "
            f"found {len(rows)}."
        )
    row = rows.iloc[0]
    if int(row["seed"]) != int(seed) or not math.isclose(
        float(row["train_fraction"]), PRIMARY_TRAIN_FRACTION
    ):
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: metrics row belongs to seed {row['seed']} at fraction "
            f"{row['train_fraction']}."
        )
    if str(row["evaluation_unit"]) != unit:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: evaluation unit {row['evaluation_unit']!r} != {unit!r}."
        )
    claim = json.loads((unit_dir / "final" / "evaluation_claim.json").read_text())
    if claim.get("status") != "complete" or claim.get("evaluation_unit") != unit:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: evaluation claim is not a complete claim for {unit}."
        )
    if int(claim.get("outer_test_prediction_batches", -1)) != 1:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: outer test predicted "
            f"{claim.get('outer_test_prediction_batches')} times, not once."
        )
    frozen_hash = str(row["frozen_policy_hash"])
    if claim.get("frozen_policy_hash") != frozen_hash:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: frozen-policy hash differs between metrics and claim."
        )
    final_manifest = json.loads((unit_dir / "final" / "final_evaluation_manifest.json").read_text())
    payload = final_manifest.get("payload", {})
    counts = payload.get("per_unit_test_evaluation_counts", {})
    if payload.get("frozen_policy_hash") != frozen_hash or counts.get(unit) != 1:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: final manifest does not record exactly one "
            "outer-test evaluation of this frozen policy."
        )
    if payload.get("outer_test_prediction_batches") != 1:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: final manifest records "
            f"{payload.get('outer_test_prediction_batches')} prediction batches."
        )
    frozen = json.loads((unit_dir / "search" / "frozen_policy.json").read_text())
    if frozen.get("frozen_policy_hash") != frozen_hash:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family} seed {seed}: frozen_policy.json hash differs from the evaluated policy."
        )
    binding = dict(frozen.get("binding", {}))
    scientific_binding = dict(claim.get("scientific_binding", {}))
    for key in ("dataset_hash", "config_hash", "split_hash", "feature_metadata_hash"):
        if binding.get(key) != scientific_binding.get(key):
            raise ObservedOnlyConfirmatoryAnalysisError(
                f"{family} seed {seed}: binding field {key!r} differs between search and claim."
            )
    search_manifest = json.loads((unit_dir / "search" / "search_manifest.json").read_text())
    search_payload = search_manifest.get("payload", {})
    if search_payload.get("candidate_scope_mode", None) is None:
        scope_mode = (
            dict(search_payload.get("scientific_config", {}))
            .get("candidate_scope", {})
            .get("mode")
        )
    else:  # pragma: no cover - not a field of the current schema
        scope_mode = search_payload["candidate_scope_mode"]
    return {
        "rmse": float(row["value"]),
        "n_refit": int(row["n_refit"]),
        "n_test": int(row["n_test"]),
        "method": str(row["method"]),
        "model": str(row["model"]),
        "frozen_policy_hash": frozen_hash,
        "selected_policy_id": frozen.get("policy_id"),
        "search_manifest_hash": frozen.get("search_manifest_hash"),
        "candidate_policy_count": len(search_payload.get("candidate_policy_hashes", [])),
        "candidate_scope_mode": scope_mode,
        "commit_hash": binding.get("commit_hash"),
        "dataset_hash": binding.get("dataset_hash"),
        "split_hash": binding.get("split_hash"),
        "split_aggregate_hash": binding.get("split_aggregate_hash"),
        "feature_metadata_hash": binding.get("feature_metadata_hash"),
        "config_hash": binding.get("config_hash"),
        "dataset_path": dict(search_payload.get("scientific_config", {}))
        .get("dataset", {})
        .get("path"),
    }


def _load_degenerate_arm(path: Path, *, family: str, unit: str) -> dict[str, Any]:
    record = load_degenerate_unit(path)
    payload = record["payload"]
    if payload.get("evaluation_unit") != unit:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"{family}: degenerate record at {path} belongs to {payload.get('evaluation_unit')!r}."
        )
    if payload.get("family") != family:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"Degenerate record at {path} declares family {payload.get('family')!r}, "
            f"expected {family!r}."
        )
    typed = [entry for entry in payload["per_policy"] if entry["method"] != "real_only"]
    generated = {int(entry["pool_statistics"]["generated_candidate_count"]) for entry in typed}
    binding = dict(payload.get("scientific_binding", {}))
    return {
        "rmse": None,
        "degenerate_unit_hash": record["degenerate_unit_hash"],
        "candidate_policy_count": len(payload.get("candidate_policy_hashes", [])),
        "candidate_scope_mode": payload.get("candidate_scope_mode"),
        "degenerate_generated_candidate_count": max(generated) if generated else 0,
        "degenerate_generated_counts_by_policy": {
            entry["policy_id"]: int(entry["pool_statistics"]["generated_candidate_count"])
            for entry in typed
        },
        "commit_hash": binding.get("commit_hash"),
        "dataset_hash": binding.get("dataset_hash"),
        "split_hash": binding.get("split_hash"),
        "split_aggregate_hash": binding.get("split_aggregate_hash"),
        "feature_metadata_hash": binding.get("feature_metadata_hash"),
        "config_hash": binding.get("config_hash"),
        "dataset_path": dict(payload.get("scientific_config", {})).get("dataset", {}).get("path"),
    }


def load_arms(confirmatory_root: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load every planned (arm, seed) unit."""
    return {
        family: {seed: load_arm_unit(confirmatory_root, family, seed) for seed in PLANNED_SEEDS}
        for family in ARM_FAMILIES
    }


# ------------------------------------------------------------ pool accounting


def load_pool_accounting(pool_directory: str | Path) -> dict[str, Any]:
    """Load per-seed candidate counts under both rules and the oracle metrics.

    The pool accounting run regenerates each unit's candidate pool from
    ``labeled_train`` alone, under each eligibility rule, and never reads an
    outer test.  Counts are model-independent, so the regenerated pool is the
    pool the confirmatory searches used.
    """
    root = Path(pool_directory)
    pool = pd.read_csv(root / "candidate_pool_statistics.csv", float_precision="round_trip")
    contracts = json.loads((root / "leakage_contracts.json").read_text())
    if contracts.get("outer_test_labels_accessed") is not False or contracts.get(
        "evaluation_registry_claimed"
    ) is not False:
        raise ObservedOnlyConfirmatoryAnalysisError(
            "The pool accounting run reports outer-test access or a registry claim; "
            "it cannot serve as treatment-size accounting."
        )
    at_fraction = pool.loc[
        pool["train_fraction"].astype(float).sub(PRIMARY_TRAIN_FRACTION).abs().lt(1e-12)
    ]
    counts: dict[str, dict[str, dict[int, dict[str, int]]]] = {}
    for _, row in at_fraction.iterrows():
        mode = str(row["candidate_scope_mode"])
        kind = str(row["transfer_kind"])
        seed = int(row["seed"])
        bucket = counts.setdefault(mode, {}).setdefault(kind, {})
        if seed in bucket:
            raise ObservedOnlyConfirmatoryAnalysisError(
                f"Duplicate pool accounting rows for {mode}/{kind} seed {seed}."
            )
        bucket[seed] = {field: int(row[field]) for field in POOL_COUNT_FIELDS}
    oracle_path = root / "withheld_cell_oracle.csv"
    oracle_rows: dict[str, dict[int, dict[str, float | None]]] = {}
    if oracle_path.is_file():
        oracle = pd.read_csv(oracle_path, float_precision="round_trip")
        selected = oracle.loc[
            oracle["candidate_scope_mode"].astype(str).eq(PRIMARY_SCOPE_MODE)
            & oracle["train_fraction"].astype(float).sub(PRIMARY_TRAIN_FRACTION).abs().lt(1e-12)
        ]
        for _, row in selected.iterrows():
            kind = str(row["transfer_kind"])
            oracle_rows.setdefault(kind, {})[int(row["seed"])] = {
                field: _clean_float(row[field]) for field in ORACLE_FIELDS if field in row
            }
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("result_status") != "corrected_revalidation" or manifest.get(
        "historical_results_loaded"
    ) is not False:
        raise ObservedOnlyConfirmatoryAnalysisError(
            "The pool accounting bundle is not a corrected, history-free run."
        )
    return {
        "directory": str(root),
        # The reanalysis manifest carries per-output hashes but no self-hash, so
        # the analysis binds to the SHA-256 of the manifest file as written.
        "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "run_id": manifest.get("run_id"),
        "git_commit": manifest.get("git_commit"),
        "config_hash": manifest.get("config_hash"),
        "dataset_hash": manifest.get("dataset_hash"),
        "split_directory": dict(manifest.get("resolved_config", {}))
        .get("splits", {})
        .get("directory"),
        "leakage_contracts": {
            key: contracts.get(key)
            for key in (
                "outer_test_labels_accessed",
                "evaluation_registry_claimed",
                "hidden_outer_train_read_stage",
                "hidden_outer_train_influences",
            )
        },
        "counts": counts,
        "oracle": oracle_rows,
    }


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


# ------------------------------------------------------------------ analysis


def analyze_observed_only_confirmation(
    run_root: str | Path,
    pool_accounting_directory: str | Path,
) -> dict[str, Any]:
    """Compute the complete preregistered analysis for one confirmatory run."""
    root = Path(run_root)
    arms = load_arms(root / "confirmatory")
    accounting = load_pool_accounting(pool_accounting_directory)
    primary_counts = accounting["counts"].get(PRIMARY_SCOPE_MODE, {}).get(TRANSFER_KIND, {})
    control_counts = (
        accounting["counts"].get(SECONDARY_CONTROL_SCOPE_MODE, {}).get(TRANSFER_KIND, {})
    )
    _cross_check_counts(arms, primary_counts, control_counts)

    unit_rows: list[dict[str, Any]] = []
    for seed in PLANNED_SEEDS:
        augmented = arms[PRIMARY_FAMILY][seed]
        comparator = arms[COMPARATOR_FAMILY][seed]
        both_complete = augmented["status"] == "complete" and comparator["status"] == "complete"
        accepted = primary_counts.get(seed, {}).get("accepted_candidate_count")
        if augmented["status"] == "degenerate":
            status = "excluded_degenerate_pool"
            reason = (
                "Zero accepted synthetic rows makes the augmented arm identical to its "
                "comparator by construction."
            )
        elif not both_complete:
            status = "failed_or_missing"
            reason = (
                f"augmented arm is {augmented['status']}, comparator arm is "
                f"{comparator['status']}; the pair has no outer-test outcome."
            )
        elif accepted is None:
            status = "failed_or_missing"
            reason = "No pool accounting exists for this unit under the observed-only rule."
        else:
            status = "included"
            reason = ""
        delta = (
            float(comparator["rmse"]) - float(augmented["rmse"]) if both_complete else None
        )
        unit_rows.append(
            {
                "seed": int(seed),
                "evaluation_unit": evaluation_unit(seed),
                "comparator_rmse": comparator.get("rmse"),
                "augmented_rmse": augmented.get("rmse"),
                "paired_delta_rmse_reduction": delta,
                "both_arms_evaluated": both_complete,
                "accepted_synthetic_rows_observed_only": accepted,
                "accepted_synthetic_rows_globally_unmeasured": control_counts.get(seed, {}).get(
                    "accepted_candidate_count"
                ),
                "generated_candidates": primary_counts.get(seed, {}).get(
                    "generated_candidate_count"
                ),
                "quarantined_candidates": primary_counts.get(seed, {}).get(
                    "rejected_quarantined_held_out_identity"
                ),
                "augmented_selected_policy_id": augmented.get("selected_policy_id"),
                "comparator_selected_policy_id": comparator.get("selected_policy_id"),
                "augmented_commit_hash": augmented.get("commit_hash"),
                "comparator_commit_hash": comparator.get("commit_hash"),
                "status": status,
                "status_reason": reason,
            }
        )

    included = [row for row in unit_rows if row["status"] == "included"]
    degenerate = [row for row in unit_rows if row["status"] == "excluded_degenerate_pool"]
    failed = [row for row in unit_rows if row["status"] == "failed_or_missing"]
    deltas = [float(row["paired_delta_rmse_reduction"]) for row in included]
    improved = sum(1 for value in deltas if value > 0.0)
    worsened = sum(1 for value in deltas if value < 0.0)
    tied = sum(1 for value in deltas if value == 0.0)

    bootstrap: dict[str, Any] | None = None
    mean_effect: float | None = None
    if deltas:
        bootstrap = seed_cluster_percentile_interval(
            deltas,
            replicates=BOOTSTRAP_REPLICATES,
            alpha=BOOTSTRAP_ALPHA,
            seed=BOOTSTRAP_SEED,
        )
        mean_effect = bootstrap["mean"]

    verdict = interpretation_table_verdict(
        mean_effect=mean_effect,
        ci_lower=bootstrap["ci_lower"] if bootstrap else None,
        ci_upper=bootstrap["ci_upper"] if bootstrap else None,
        included_unit_count=len(included),
        improved_unit_count=improved,
        degenerate_unit_count=len(degenerate),
        practical_minimum_effect=PRACTICAL_MINIMUM_EFFECT,
        maximum_degenerate_units=MAXIMUM_DEGENERATE_UNITS,
        minimum_included_units=MINIMUM_INCLUDED_UNITS,
        minimum_improved_units_for_positive=MINIMUM_IMPROVED_UNITS_FOR_POSITIVE,
        planned_unit_count=len(PLANNED_SEEDS),
        degenerate_override_rule="section_6_degenerate_pool_override",
        underpowered_override_rule="section_6_underpowered_override",
        table_rule="section_7_interpretation_table",
        error_type=ObservedOnlyConfirmatoryAnalysisError,
    )

    control_rows = []
    for seed in PLANNED_SEEDS:
        unit = arms[SECONDARY_CONTROL_FAMILY][seed]
        comparator = arms[COMPARATOR_FAMILY][seed]
        control_rows.append(
            {
                "seed": int(seed),
                "status": unit["status"],
                "rmse": unit.get("rmse"),
                "comparator_rmse": comparator.get("rmse"),
                "delta_vs_comparator": (
                    float(comparator["rmse"]) - float(unit["rmse"])
                    if unit["status"] == "complete" and comparator["status"] == "complete"
                    else None
                ),
                "accepted_synthetic_rows": control_counts.get(seed, {}).get(
                    "accepted_candidate_count"
                ),
                "generated_candidates": control_counts.get(seed, {}).get(
                    "generated_candidate_count"
                ),
                "rejected_already_measured": control_counts.get(seed, {}).get(
                    "rejected_already_measured"
                ),
                "degenerate_unit_hash": unit.get("degenerate_unit_hash"),
                "commit_hash": unit.get("commit_hash"),
            }
        )
    control_statuses = {status: 0 for status in ("complete", "degenerate", "missing")}
    for row in control_rows:
        control_statuses[row["status"]] += 1

    binding_summary = _binding_summary(arms)
    payload: dict[str, Any] = {
        "schema_version": OBSERVED_ONLY_CONFIRMATORY_SCHEMA_VERSION,
        "preregistration": {
            "document": PREREGISTRATION_DOCUMENT,
            "commit": PREREGISTRATION_COMMIT,
            "primary_family": PRIMARY_FAMILY,
            "comparator_family": COMPARATOR_FAMILY,
            "secondary_control_family": SECONDARY_CONTROL_FAMILY,
            "primary_candidate_scope_mode": PRIMARY_SCOPE_MODE,
            "secondary_control_candidate_scope_mode": SECONDARY_CONTROL_SCOPE_MODE,
            "transfer_kind": TRANSFER_KIND,
            "metric": PRIMARY_METRIC,
            "train_fraction": PRIMARY_TRAIN_FRACTION,
            "estimand": "comparator_rmse_minus_augmented_rmse_per_seed",
            "planned_seeds": list(PLANNED_SEEDS),
            "practical_minimum_effect": PRACTICAL_MINIMUM_EFFECT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_alpha": BOOTSTRAP_ALPHA,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "maximum_degenerate_units": MAXIMUM_DEGENERATE_UNITS,
            "minimum_included_units": MINIMUM_INCLUDED_UNITS,
            "minimum_improved_units_for_positive": MINIMUM_IMPROVED_UNITS_FOR_POSITIVE,
        },
        "run": {
            "run_root": str(root),
            "confirmatory_directory": str(root / "confirmatory"),
            **binding_summary,
            "pool_accounting": {
                key: accounting[key]
                for key in (
                    "directory",
                    "manifest_hash",
                    "run_id",
                    "git_commit",
                    "config_hash",
                    "dataset_hash",
                    "split_directory",
                    "leakage_contracts",
                )
            },
            "equal_arm_search_budget": _equal_search_budget(arms),
        },
        "per_unit_rows": unit_rows,
        "accounting": {
            "planned_unit_count": len(PLANNED_SEEDS),
            "included_unit_count": len(included),
            "degenerate_excluded_unit_count": len(degenerate),
            "failed_or_missing_unit_count": len(failed),
            "all_planned_seeds_accounted_for": (
                len(included) + len(degenerate) + len(failed) == len(PLANNED_SEEDS)
            ),
            "included_seeds": [row["seed"] for row in included],
            "degenerate_excluded_seeds": [row["seed"] for row in degenerate],
            "failed_or_missing_seeds": [row["seed"] for row in failed],
            "arm_unit_counts": {
                family: {
                    status: sum(1 for unit in arms[family].values() if unit["status"] == status)
                    for status in ("complete", "degenerate", "missing")
                }
                for family in ARM_FAMILIES
            },
            "planned_arm_unit_pairs": len(ARM_FAMILIES) * len(PLANNED_SEEDS),
            "all_arm_unit_pairs_accounted_for": all(
                sum(
                    1 for unit in arms[family].values() if unit["status"] in
                    ("complete", "degenerate", "missing")
                )
                == len(PLANNED_SEEDS)
                for family in ARM_FAMILIES
            ),
        },
        "primary": {
            "mean_paired_rmse_reduction": mean_effect,
            "bootstrap": bootstrap,
            "improved_unit_count": improved,
            "worsened_unit_count": worsened,
            "tied_unit_count": tied,
            "paired_deltas": deltas,
        },
        "secondary": wilcoxon_secondary(deltas),
        "treatment_size": {
            "source": (
                "validation-only pool accounting that regenerates each unit's candidate "
                "pool from labeled_train under each eligibility rule; the 2026-09-03 "
                "protocol run recorded no per-unit counts"
            ),
            "accepted_synthetic_rows_by_seed": {
                PRIMARY_SCOPE_MODE: {
                    str(seed): primary_counts.get(seed, {}).get("accepted_candidate_count")
                    for seed in PLANNED_SEEDS
                },
                SECONDARY_CONTROL_SCOPE_MODE: {
                    str(seed): control_counts.get(seed, {}).get("accepted_candidate_count")
                    for seed in PLANNED_SEEDS
                },
            },
            "pool_counts_by_rule": {
                mode: {
                    kind: {str(seed): counts for seed, counts in sorted(by_seed.items())}
                    for kind, by_seed in kinds.items()
                }
                for mode, kinds in accounting["counts"].items()
            },
            "totals_by_rule": _pool_totals(accounting["counts"]),
        },
        "secondary_control_arm": {
            "family": SECONDARY_CONTROL_FAMILY,
            "candidate_scope_mode": SECONDARY_CONTROL_SCOPE_MODE,
            "role": (
                "prospective-novelty contrast; reported with accepted-row counts, never "
                "substituted into the primary comparison"
            ),
            "unit_status_counts": control_statuses,
            "per_unit_rows": control_rows,
        },
        "oracle": _oracle_summary(accounting["oracle"]),
        "verdict": verdict,
    }
    payload["analysis_hash"] = stable_hash(payload)
    return payload


def _cross_check_counts(
    arms: Mapping[str, Mapping[int, Mapping[str, Any]]],
    primary_counts: Mapping[int, Mapping[str, int]],
    control_counts: Mapping[int, Mapping[str, int]],
) -> None:
    """Fail if the protocol artifacts and the pool accounting disagree."""
    for seed in PLANNED_SEEDS:
        primary = arms[PRIMARY_FAMILY][seed]
        if primary["status"] == "complete":
            accepted = primary_counts.get(seed, {}).get("accepted_candidate_count")
            if accepted is not None and accepted <= 0:
                raise ObservedOnlyConfirmatoryAnalysisError(
                    f"Pool accounting reports zero accepted rows for {PRIMARY_FAMILY} seed "
                    f"{seed}, but the protocol evaluated it as an augmented unit."
                )
        control = arms[SECONDARY_CONTROL_FAMILY][seed]
        counts = control_counts.get(seed)
        if control["status"] == "degenerate" and counts is not None:
            if counts["accepted_candidate_count"] != 0:
                raise ObservedOnlyConfirmatoryAnalysisError(
                    f"{SECONDARY_CONTROL_FAMILY} seed {seed} is recorded as degenerate but "
                    f"pool accounting accepted {counts['accepted_candidate_count']} rows."
                )
            recorded = control["degenerate_generated_candidate_count"]
            if recorded != counts["generated_candidate_count"]:
                raise ObservedOnlyConfirmatoryAnalysisError(
                    f"{SECONDARY_CONTROL_FAMILY} seed {seed}: degenerate record generated "
                    f"{recorded} candidates, pool accounting generated "
                    f"{counts['generated_candidate_count']}."
                )
        if control["status"] == "complete" and counts is not None:
            if counts["accepted_candidate_count"] <= 0:
                raise ObservedOnlyConfirmatoryAnalysisError(
                    f"{SECONDARY_CONTROL_FAMILY} seed {seed} was evaluated as augmented but "
                    "pool accounting accepted zero rows."
                )


def _binding_summary(arms: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    units = [unit for by_seed in arms.values() for unit in by_seed.values()]
    present = [unit for unit in units if unit["status"] != "missing"]

    def distinct(field: str) -> list[Any]:
        return sorted({unit.get(field) for unit in present if unit.get(field) is not None})

    dataset_hashes = distinct("dataset_hash")
    if len(dataset_hashes) > 1:
        raise ObservedOnlyConfirmatoryAnalysisError(
            f"Confirmatory arms bind different datasets: {dataset_hashes}."
        )
    split_hashes_by_seed: dict[str, list[str]] = {}
    for unit in present:
        split_hashes_by_seed.setdefault(str(unit["seed"]), [])
        if unit.get("split_hash") not in split_hashes_by_seed[str(unit["seed"])]:
            split_hashes_by_seed[str(unit["seed"])].append(unit.get("split_hash"))
    for seed, hashes in split_hashes_by_seed.items():
        if len(hashes) > 1:
            raise ObservedOnlyConfirmatoryAnalysisError(
                f"Arms at seed {seed} bind different splits: {hashes}."
            )
    return {
        "dataset_hash": dataset_hashes[0] if dataset_hashes else None,
        "dataset_path": (distinct("dataset_path") or [None])[0],
        "split_aggregate_hash": (distinct("split_aggregate_hash") or [None])[0],
        "feature_metadata_hash": (distinct("feature_metadata_hash") or [None])[0],
        "split_hash_by_seed": {seed: hashes[0] for seed, hashes in split_hashes_by_seed.items()},
        "config_hash_by_family": {
            family: sorted(
                {
                    unit.get("config_hash")
                    for unit in arms[family].values()
                    if unit.get("config_hash") is not None
                }
            )
            for family in ARM_FAMILIES
        },
        "commit_hashes": distinct("commit_hash"),
    }


def _equal_search_budget(arms: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> bool | None:
    budgets = {
        unit.get("candidate_policy_count")
        for family in (PRIMARY_FAMILY, COMPARATOR_FAMILY)
        for unit in arms[family].values()
        if unit.get("candidate_policy_count") is not None
    }
    if not budgets:
        return None
    return len(budgets) == 1


def _pool_totals(counts: Mapping[str, Mapping[str, Mapping[int, Mapping[str, int]]]]) -> dict:
    totals: dict[str, dict[str, dict[str, int]]] = {}
    for mode, kinds in counts.items():
        for kind, by_seed in kinds.items():
            bucket = totals.setdefault(mode, {}).setdefault(
                kind, dict.fromkeys(POOL_COUNT_FIELDS, 0)
            )
            for row in by_seed.values():
                for field in POOL_COUNT_FIELDS:
                    bucket[field] += int(row[field])
    return totals


def _oracle_summary(oracle: Mapping[str, Mapping[int, Mapping[str, float | None]]]) -> dict:
    summary: dict[str, Any] = {
        "family": "withheld_cell_transfer_oracle",
        "label": "secondary_non_selecting",
        "note": (
            "Measures how well frozen pseudo-labels reconstruct hidden outer-training "
            "cells. It never influenced generation, ranking, teacher fitting, policy "
            "selection, weights or student fitting, and it is not substituted for the "
            "primary outcome."
        ),
        "by_transfer_kind": {},
    }
    for kind, by_seed in sorted(oracle.items()):
        means: dict[str, float | None] = {}
        for field in ORACLE_FIELDS:
            values = [
                row[field]
                for row in by_seed.values()
                if field in row and row[field] is not None
            ]
            means[field] = sum(values) / len(values) if values else None
        summary["by_transfer_kind"][kind] = {
            "seeds": sorted(int(seed) for seed in by_seed),
            "mean": means,
            "per_seed": {str(seed): dict(row) for seed, row in sorted(by_seed.items())},
        }
    return summary


# ------------------------------------------------------------------- report


def render_observed_only_report(payload: Mapping[str, Any]) -> str:
    """Render the human-readable confirmatory report, section 9 in order."""
    prereg = payload["preregistration"]
    run = payload["run"]
    accounting = payload["accounting"]
    primary = payload["primary"]
    secondary = payload["secondary"]
    verdict = payload["verdict"]
    bootstrap = primary["bootstrap"]
    control = payload["secondary_control_arm"]
    treatment = payload["treatment_size"]

    def fmt(value: Any, spec: str = ".6f") -> str:
        return "n/a" if value is None else format(value, spec)

    lines: list[str] = []
    lines.append("# Confirmatory result: observed-only condition transfer")
    lines.append("")
    lines.append(
        f"Preregistered in `{prereg['document']}` at commit `{prereg['commit'][:7]}`, "
        "before any confirmatory outer-test outcome for this experiment existed. Every "
        "threshold below is quoted from that document, not chosen after seeing these numbers."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{str(verdict['verdict']).upper()}** — {verdict['reason']}")
    lines.append("")
    lines.append(f"Computed mechanically by rule `{verdict['rule']}` from the numbers below.")
    lines.append("")
    lines.append("## The question")
    lines.append("")
    lines.append(
        f"Does anonymous condition-transfer augmentation under the `{prereg['primary_candidate_scope_mode']}` "
        f"eligibility rule (`{prereg['primary_family']}`) reduce outer-test "
        f"{str(prereg['metric']).upper()} relative to the matched real-only XGBoost control "
        f"(`{prereg['comparator_family']}`) at training fraction {prereg['train_fraction']} "
        f"on seeds {prereg['planned_seeds'][0]}–{prereg['planned_seeds'][-1]}? "
        "Positive deltas favour augmentation."
    )
    lines.append("")
    lines.append("## 1. Paired per-seed deltas (all planned seeds)")
    lines.append("")
    lines.append(
        "| Seed | Comparator RMSE | Augmented RMSE | Delta (reduction) | "
        "Accepted rows, observed-only | Accepted rows, globally-unmeasured | Status |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["per_unit_rows"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['comparator_rmse'])} | {fmt(row['augmented_rmse'])} | "
            f"{fmt(row['paired_delta_rmse_reduction'], '+.6f')} | "
            f"{fmt(row['accepted_synthetic_rows_observed_only'], 'd')} | "
            f"{fmt(row['accepted_synthetic_rows_globally_unmeasured'], 'd')} | {row['status']} |"
        )
    lines.append("")
    lines.append("## 2. Primary estimand and interval")
    lines.append("")
    if bootstrap is None:
        lines.append("No unit survived exclusion; no effect is estimable.")
    else:
        lines.append(f"- Mean paired RMSE reduction: **{bootstrap['mean']:+.6f}**")
        lines.append(
            f"- 95% percentile bootstrap over seeds as clusters ({bootstrap['replicates']} "
            f"replicates, fixed seed {bootstrap['seed']}): "
            f"**[{bootstrap['ci_lower']:+.6f}, {bootstrap['ci_upper']:+.6f}]**"
        )
        lines.append(f"- Practical minimum effect: {prereg['practical_minimum_effect']} RMSE")
    lines.append("")
    lines.append("## 3. Consistency")
    lines.append("")
    lines.append(
        f"- Units improved: {primary['improved_unit_count']}; worsened: "
        f"{primary['worsened_unit_count']}; tied: {primary['tied_unit_count']} "
        f"(of {accounting['included_unit_count']} included)"
    )
    lines.append("")
    lines.append("## 4. Secondary test")
    lines.append("")
    if secondary["p_value"] is None:
        lines.append(f"- Wilcoxon signed-rank: {secondary['note']}")
    else:
        lines.append(
            f"- Two-sided Wilcoxon signed-rank p-value: {secondary['p_value']:.6f} "
            f"(statistic {secondary['statistic']:.1f}) — **secondary**. Significance alone "
            "never establishes success."
        )
    lines.append("")
    lines.append("## 5. Size of the treatment under both eligibility rules")
    lines.append("")
    lines.append(f"Source: {treatment['source']}.")
    lines.append("")
    lines.append(
        "| Rule | Transfer kind | Generated | Accepted | Rejected: observed in labeled train | "
        "Rejected: already measured | Quarantined (validation/test identity) |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for mode, kinds in sorted(treatment["totals_by_rule"].items()):
        for kind, totals in sorted(kinds.items()):
            lines.append(
                f"| {mode} | {kind} | {totals['generated_candidate_count']} | "
                f"{totals['accepted_candidate_count']} | "
                f"{totals['rejected_observed_in_labeled_train']} | "
                f"{totals['rejected_already_measured']} | "
                f"{totals['rejected_quarantined_held_out_identity']} |"
            )
    lines.append("")
    lines.append("## 6. Degenerate, failed and missing units")
    lines.append("")
    lines.append(f"- Planned units per arm: {accounting['planned_unit_count']}")
    lines.append(
        f"- Primary pair included: {accounting['included_unit_count']} "
        f"(seeds {accounting['included_seeds']})"
    )
    lines.append(
        f"- Primary pair excluded, degenerate pool: {accounting['degenerate_excluded_unit_count']} "
        f"(seeds {accounting['degenerate_excluded_seeds']})"
    )
    lines.append(
        f"- Primary pair failed or missing: {accounting['failed_or_missing_unit_count']} "
        f"(seeds {accounting['failed_or_missing_seeds']})"
    )
    lines.append(f"- All planned seeds accounted for: {accounting['all_planned_seeds_accounted_for']}")
    lines.append("")
    lines.append(
        f"Secondary control arm `{control['family']}` (`{control['candidate_scope_mode']}`): "
        f"{control['role']}."
    )
    lines.append("")
    lines.append("| Seed | Status | Control RMSE | Comparator RMSE | Generated | Accepted | Rejected: already measured |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for row in control["per_unit_rows"]:
        lines.append(
            f"| {row['seed']} | {row['status']} | {fmt(row['rmse'])} | {fmt(row['comparator_rmse'])} | "
            f"{fmt(row['generated_candidates'], 'd')} | {fmt(row['accepted_synthetic_rows'], 'd')} | "
            f"{fmt(row['rejected_already_measured'], 'd')} |"
        )
    lines.append("")
    counts = control["unit_status_counts"]
    lines.append(
        f"Control arm units: {counts['complete']} complete, {counts['degenerate']} degenerate, "
        f"{counts['missing']} missing. Degeneracy here is the control's result: it measures how "
        "completely the historical eligibility rule suppressed the treatment."
    )
    lines.append("")
    lines.append("## 7. Withheld-cell oracle (secondary, non-selecting)")
    lines.append("")
    oracle = payload["oracle"]
    lines.append(oracle["note"])
    lines.append("")
    if not oracle["by_transfer_kind"]:
        lines.append("No oracle rows were recorded at the confirmatory fraction.")
    else:
        lines.append(
            "| Transfer kind | Seeds | Hidden-cell coverage | Pseudo-label MAE | Pseudo-label RMSE | "
            "Spearman | Bias |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for kind, block in oracle["by_transfer_kind"].items():
            mean = block["mean"]
            lines.append(
                f"| {kind} | {len(block['seeds'])} | {fmt(mean.get('hidden_cell_coverage_fraction'), '.4f')} | "
                f"{fmt(mean.get('pseudo_label_mae'), '.3f')} | {fmt(mean.get('pseudo_label_rmse'), '.3f')} | "
                f"{fmt(mean.get('pseudo_label_spearman'), '.3f')} | {fmt(mean.get('pseudo_label_bias'), '+.3f')} |"
            )
    lines.append("")
    lines.append("## Protocol provenance")
    lines.append("")
    lines.append(f"- Run root: `{run['run_root']}`")
    lines.append(f"- Dataset: `{run['dataset_path']}` (`{run['dataset_hash']}`)")
    lines.append(f"- Split aggregate hash: `{run['split_aggregate_hash']}`")
    lines.append(f"- Feature metadata hash: `{run['feature_metadata_hash']}`")
    lines.append(f"- Code commits bound by the arms: {', '.join(f'`{c}`' for c in run['commit_hashes'])}")
    lines.append(
        f"- The two preregistered arms received equal search budgets: {run['equal_arm_search_budget']}"
    )
    lines.append(f"- Pool accounting bundle: `{run['pool_accounting']['directory']}` "
                 f"(manifest `{run['pool_accounting']['manifest_hash']}`)")
    lines.append(f"- Analysis hash: `{payload['analysis_hash']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Training fraction 0.05 only; canonical grouped random splits only; one dataset; "
        "not OOD evidence. The Phase 15 confirmatory result at fraction 0.2 stands under its "
        "own protocol and is neither superseded nor re-cut by this experiment."
    )
    lines.append("")
    return "\n".join(lines)
