"""Preregistered primary confirmatory analysis for Phase 15.

Every quantity, threshold and decision rule in this module is fixed by
``PRIMARY_EXPERIMENT.md``, preregistered at commit ``13fbfde`` before any
confirmatory outer-test outcome existed. The module reads a completed Phase 14
runner bundle and computes the primary estimand mechanically:

* the paired per-seed RMSE reduction, comparator minus augmented, at the
  preregistered training fraction;
* the mean paired effect with a 95% percentile bootstrap over seeds-as-clusters;
* improvement consistency, and the secondary Wilcoxon signed-rank p-value;
* degenerate and failed units, with every planned seed accounted for;
* the verdict from the preregistered interpretation table, evaluated as a
  boolean function of the numbers rather than as a judgement call.

Nothing here re-cuts the result by fraction, metric, subgroup or seed subset.
The analysis inputs are frozen constants, not parameters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bh_augmentation.utils.corrected_runs import stable_hash

PRIMARY_CONFIRMATORY_SCHEMA_VERSION = "bh-primary-confirmatory-analysis-v1"

#: Commit that froze ``PRIMARY_EXPERIMENT.md`` before any confirmatory outcome.
PREREGISTRATION_COMMIT = "13fbfdef7ac0cfd05dda2c1c33bb6a59b4d64fff"
PREREGISTRATION_DOCUMENT = "PRIMARY_EXPERIMENT.md"

#: Section 3 of the preregistration.
AUGMENTED_METHOD_FAMILY = "anonymous_transfer_without_ae"
COMPARATOR_METHOD_FAMILY = "direct_xgboost"
PRIMARY_METRIC = "rmse"
PRIMARY_TRAIN_FRACTION = 0.2
PLANNED_SEEDS: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
PRACTICAL_MINIMUM_EFFECT = 1.0
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 1501

#: Section 5 of the preregistration.
MAXIMUM_DEGENERATE_UNITS = 3
MINIMUM_INCLUDED_UNITS = 7

#: Section 6 of the preregistration.
MINIMUM_IMPROVED_UNITS_FOR_POSITIVE = 7

#: The pool phase whose accepted rows actually train the outer-test model. The
#: ``search`` pools are built from the inner fit rows only and are not the
#: treatment that the confirmed models received.
TREATMENT_POOL_PHASE = "placement"
TREATMENT_POOL_KIND = "anonymous"

_VERDICTS = ("positive", "null", "negative", "inconclusive")


class PrimaryConfirmatoryAnalysisError(ValueError):
    """Raised when a run bundle cannot support the preregistered analysis."""


def paired_deltas(
    final_test_metrics: pd.DataFrame,
    *,
    seeds: Sequence[int] = PLANNED_SEEDS,
) -> dict[int, dict[str, Any]]:
    """Return the per-seed comparator-minus-augmented RMSE reduction.

    A seed with a missing arm is recorded with ``status`` ``failed_or_missing``
    rather than dropped, because Section 5 requires every planned seed to be
    accounted for.
    """
    frame = final_test_metrics
    required = {
        "seed",
        "train_fraction",
        "method_family",
        "policy_id",
        "metric",
        "value",
        "n_samples",
        "data_role",
        "prediction_hash",
        "frozen_policy_hash",
        "test_evaluation_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PrimaryConfirmatoryAnalysisError(
            f"final_test_metrics is missing required columns: {missing}."
        )
    selected = frame.loc[
        (frame["metric"].astype(str) == PRIMARY_METRIC)
        & (frame["data_role"].astype(str) == "test")
        & np.isclose(frame["train_fraction"].astype(float), PRIMARY_TRAIN_FRACTION)
    ]
    records: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rows = selected.loc[selected["seed"].astype(int) == int(seed)]
        arms: dict[str, dict[str, Any]] = {}
        for family in (AUGMENTED_METHOD_FAMILY, COMPARATOR_METHOD_FAMILY):
            matched = rows.loc[rows["method_family"].astype(str) == family]
            if len(matched) == 1:
                row = matched.iloc[0]
                if int(row["test_evaluation_count"]) != 1:
                    raise PrimaryConfirmatoryAnalysisError(
                        "Each (unit, arm) must be evaluated on the outer test "
                        f"exactly once; seed {seed} family {family} records "
                        f"{int(row['test_evaluation_count'])}."
                    )
                arms[family] = {
                    "rmse": float(row["value"]),
                    "policy_id": str(row["policy_id"]),
                    "frozen_policy_hash": str(row["frozen_policy_hash"]),
                    "prediction_hash": str(row["prediction_hash"]),
                    "n_samples": int(row["n_samples"]),
                }
            elif len(matched) > 1:
                raise PrimaryConfirmatoryAnalysisError(
                    f"Duplicate outer-test RMSE rows for seed {seed}, {family}."
                )
        complete = len(arms) == 2
        records[int(seed)] = {
            "seed": int(seed),
            "train_fraction": PRIMARY_TRAIN_FRACTION,
            "augmented_rmse": arms.get(AUGMENTED_METHOD_FAMILY, {}).get("rmse"),
            "comparator_rmse": arms.get(COMPARATOR_METHOD_FAMILY, {}).get("rmse"),
            "paired_delta_rmse_reduction": (
                arms[COMPARATOR_METHOD_FAMILY]["rmse"]
                - arms[AUGMENTED_METHOD_FAMILY]["rmse"]
                if complete
                else None
            ),
            "augmented_policy_id": arms.get(AUGMENTED_METHOD_FAMILY, {}).get(
                "policy_id"
            ),
            "comparator_policy_id": arms.get(COMPARATOR_METHOD_FAMILY, {}).get(
                "policy_id"
            ),
            "augmented_frozen_policy_hash": arms.get(
                AUGMENTED_METHOD_FAMILY, {}
            ).get("frozen_policy_hash"),
            "comparator_frozen_policy_hash": arms.get(
                COMPARATOR_METHOD_FAMILY, {}
            ).get("frozen_policy_hash"),
            "test_row_count": arms.get(AUGMENTED_METHOD_FAMILY, {}).get("n_samples"),
            "both_arms_evaluated": complete,
        }
    return records


def accepted_synthetic_rows(
    pool_summary: pd.DataFrame,
    *,
    seeds: Sequence[int] = PLANNED_SEEDS,
) -> dict[int, dict[str, Any]]:
    """Return the treatment size and degeneracy flag per planned seed."""
    required = {
        "evaluation_unit",
        "phase",
        "pool_kind",
        "parent_count",
        "accepted_count",
        "degenerate_pool",
    }
    missing = sorted(required - set(pool_summary.columns))
    if missing:
        raise PrimaryConfirmatoryAnalysisError(
            f"pool_summary is missing required columns: {missing}."
        )
    rows = pool_summary.loc[
        (pool_summary["phase"].astype(str) == TREATMENT_POOL_PHASE)
        & (pool_summary["pool_kind"].astype(str) == TREATMENT_POOL_KIND)
    ]
    records: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        unit = _evaluation_unit(int(seed))
        matched = rows.loc[rows["evaluation_unit"].astype(str) == unit]
        if len(matched) > 1:
            raise PrimaryConfirmatoryAnalysisError(
                f"Duplicate treatment pool summary rows for {unit}."
            )
        if matched.empty:
            records[int(seed)] = {
                "seed": int(seed),
                "evaluation_unit": unit,
                "parent_count": None,
                "accepted_synthetic_rows": None,
                "degenerate_pool": None,
                "pool_summary_present": False,
            }
            continue
        row = matched.iloc[0]
        accepted = int(row["accepted_count"])
        records[int(seed)] = {
            "seed": int(seed),
            "evaluation_unit": unit,
            "parent_count": int(row["parent_count"]),
            "accepted_synthetic_rows": accepted,
            "degenerate_pool": bool(row["degenerate_pool"]) or accepted == 0,
            "pool_summary_present": True,
        }
    return records


def seed_cluster_percentile_interval(
    deltas: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile bootstrap over paired per-seed deltas, seeds as clusters."""
    values = np.asarray(list(deltas), dtype=float)
    if values.size == 0:
        raise PrimaryConfirmatoryAnalysisError(
            "The seed-cluster bootstrap requires at least one included unit."
        )
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        drawn = generator.choice(values, size=values.size, replace=True)
        samples[replicate] = float(np.mean(drawn))
    lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2], method="linear")
    return {
        "method": "percentile_bootstrap_over_seed_clusters",
        "replicates": int(replicates),
        "alpha": float(alpha),
        "seed": int(seed),
        "cluster_count": int(values.size),
        "mean": float(np.mean(values)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def interpretation_table_verdict(
    *,
    mean_effect: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    included_unit_count: int,
    improved_unit_count: int,
    degenerate_unit_count: int,
    practical_minimum_effect: float,
    maximum_degenerate_units: int,
    minimum_included_units: int,
    minimum_improved_units_for_positive: int,
    planned_unit_count: int,
    degenerate_override_rule: str,
    underpowered_override_rule: str,
    table_rule: str,
    error_type: type[ValueError] | None = None,
) -> dict[str, Any]:
    """Apply a preregistered interpretation table as a pure boolean function.

    Both preregistered experiments in this repository share one decision
    structure: two overrides (too many degenerate units; too few surviving
    units) that force ``inconclusive`` before the table is consulted, then four
    mutually exclusive table conditions on the mean effect and its interval.
    The thresholds and the rule names under which each decision is reported
    are supplied by the caller from its own frozen document, so this function
    encodes the *shape* of the rule and never a threshold.
    """
    error = PrimaryConfirmatoryAnalysisError if error_type is None else error_type
    if degenerate_unit_count > maximum_degenerate_units:
        return {
            "verdict": "inconclusive",
            "rule": degenerate_override_rule,
            "reason": (
                f"{degenerate_unit_count} of {planned_unit_count} units have a "
                "degenerate transfer pool, above the preregistered maximum of "
                f"{maximum_degenerate_units}; the degeneracy is the finding."
            ),
            "conditions": {},
        }
    if included_unit_count < minimum_included_units:
        return {
            "verdict": "inconclusive",
            "rule": underpowered_override_rule,
            "reason": (
                f"Only {included_unit_count} units remain after exclusions, "
                f"below the preregistered minimum of {minimum_included_units}; "
                "the analysis is underpowered and no positive claim is made."
            ),
            "conditions": {},
        }
    if mean_effect is None or ci_lower is None or ci_upper is None:
        raise error("The interpretation table requires a mean effect and an interval.")
    conditions = {
        "positive": (
            mean_effect >= practical_minimum_effect
            and ci_lower > 0.0
            and improved_unit_count >= minimum_improved_units_for_positive
        ),
        "null": (
            -practical_minimum_effect < ci_lower and ci_upper < practical_minimum_effect
        ),
        "negative": (mean_effect <= -practical_minimum_effect and ci_upper < 0.0),
        "inconclusive": (
            ci_upper >= practical_minimum_effect and ci_lower <= -practical_minimum_effect
        ),
    }
    matched = [name for name in _VERDICTS if conditions[name]]
    if len(matched) == 1:
        return {
            "verdict": matched[0],
            "rule": table_rule,
            "reason": _VERDICT_REASONS[matched[0]],
            "conditions": conditions,
        }
    if not matched:
        return {
            "verdict": "undetermined_by_preregistered_table",
            "rule": table_rule,
            "reason": (
                "No condition in the preregistered interpretation table holds "
                "for this interval. The table is reported as it stands; no "
                "post hoc rule is substituted."
            ),
            "conditions": conditions,
        }
    raise error(  # pragma: no cover - unreachable
        f"Preregistered conditions are not mutually exclusive: {matched}."
    )


def preregistered_verdict(
    *,
    mean_effect: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    included_unit_count: int,
    improved_unit_count: int,
    degenerate_unit_count: int,
) -> dict[str, Any]:
    """Apply Section 6 of the Phase 15 preregistration as a pure boolean function.

    Section 5 overrides fire first: more than ``MAXIMUM_DEGENERATE_UNITS``
    degenerate units, or fewer than ``MINIMUM_INCLUDED_UNITS`` surviving units,
    force ``inconclusive`` before the interpretation table is consulted. The
    four table conditions are mutually exclusive by construction, so no
    tie-breaking judgement is ever exercised.
    """
    return interpretation_table_verdict(
        mean_effect=mean_effect,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        included_unit_count=included_unit_count,
        improved_unit_count=improved_unit_count,
        degenerate_unit_count=degenerate_unit_count,
        practical_minimum_effect=PRACTICAL_MINIMUM_EFFECT,
        maximum_degenerate_units=MAXIMUM_DEGENERATE_UNITS,
        minimum_included_units=MINIMUM_INCLUDED_UNITS,
        minimum_improved_units_for_positive=MINIMUM_IMPROVED_UNITS_FOR_POSITIVE,
        planned_unit_count=len(PLANNED_SEEDS),
        degenerate_override_rule="section_5_degenerate_pool_override",
        underpowered_override_rule="section_5_underpowered_override",
        table_rule="section_6_interpretation_table",
    )


_VERDICT_REASONS = {
    "positive": (
        "Mean effect at least the practical minimum, interval excluding zero "
        "from below, and a consistent majority of units improved."
    ),
    "null": (
        "The interval lies entirely inside the practical-equivalence band, so "
        "a practically meaningful benefit and a practically meaningful harm "
        "are both excluded."
    ),
    "negative": (
        "Mean effect at most minus the practical minimum with the interval "
        "entirely below zero: practically meaningful harm."
    ),
    "inconclusive": (
        "The interval admits both a practically meaningful benefit and a "
        "practically meaningful harm at this sample size."
    ),
}


def analyze_primary_confirmation(run_directory: str | Path) -> dict[str, Any]:
    """Compute the complete preregistered primary analysis for one run."""
    root = Path(run_directory)
    read = {"float_precision": "round_trip"}
    final_metrics = pd.read_csv(root / "final_test_metrics.csv", **read)
    pool_summary = pd.read_csv(root / "pool_summary.csv", **read)
    manifest = json.loads((root / "manifest.json").read_text())

    pairs = paired_deltas(final_metrics)
    pools = accepted_synthetic_rows(pool_summary)

    unit_rows: list[dict[str, Any]] = []
    for seed in PLANNED_SEEDS:
        pair = pairs[seed]
        pool = pools[seed]
        if not pair["both_arms_evaluated"]:
            status = "failed_or_missing"
            reason = (
                "One or both preregistered arms have no outer-test RMSE row "
                "for this unit."
            )
        elif pool["degenerate_pool"] is None:
            status = "failed_or_missing"
            reason = "No treatment transfer-pool summary exists for this unit."
        elif pool["degenerate_pool"]:
            status = "excluded_degenerate_pool"
            reason = (
                "Zero accepted synthetic rows makes the augmented arm identical "
                "to its comparator by construction."
            )
        else:
            status = "included"
            reason = ""
        unit_rows.append(
            {
                **pair,
                "evaluation_unit": pool["evaluation_unit"],
                "accepted_synthetic_rows": pool["accepted_synthetic_rows"],
                "transfer_pool_parent_count": pool["parent_count"],
                "degenerate_pool": pool["degenerate_pool"],
                "status": status,
                "status_reason": reason,
            }
        )

    included = [row for row in unit_rows if row["status"] == "included"]
    degenerate = [
        row for row in unit_rows if row["status"] == "excluded_degenerate_pool"
    ]
    failed = [row for row in unit_rows if row["status"] == "failed_or_missing"]
    deltas = [float(row["paired_delta_rmse_reduction"]) for row in included]
    improved = sum(1 for value in deltas if value > 0.0)
    worsened = sum(1 for value in deltas if value < 0.0)
    tied = sum(1 for value in deltas if value == 0.0)

    bootstrap: dict[str, Any] | None = None
    mean_effect: float | None = None
    if deltas:
        bootstrap = seed_cluster_percentile_interval(deltas)
        mean_effect = bootstrap["mean"]

    verdict = preregistered_verdict(
        mean_effect=mean_effect,
        ci_lower=bootstrap["ci_lower"] if bootstrap else None,
        ci_upper=bootstrap["ci_upper"] if bootstrap else None,
        included_unit_count=len(included),
        improved_unit_count=improved,
        degenerate_unit_count=len(degenerate),
    )

    payload: dict[str, Any] = {
        "schema_version": PRIMARY_CONFIRMATORY_SCHEMA_VERSION,
        "preregistration": {
            "document": PREREGISTRATION_DOCUMENT,
            "commit": PREREGISTRATION_COMMIT,
            "augmented_method_family": AUGMENTED_METHOD_FAMILY,
            "comparator_method_family": COMPARATOR_METHOD_FAMILY,
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
            "minimum_improved_units_for_positive": (
                MINIMUM_IMPROVED_UNITS_FOR_POSITIVE
            ),
        },
        "run": {
            "directory": str(root),
            "manifest_hash": manifest.get("manifest_hash"),
            "plan_hash": manifest.get("plan_hash"),
            "config_hash": manifest.get("config_hash"),
            "dataset_hash": manifest.get("dataset_hash"),
            "canonical_split_hash": manifest.get("canonical_split_hash"),
            "git_commit": manifest.get("git_commit"),
            "search_budget_by_family": manifest.get("search_budget_by_family"),
            "equal_arm_search_budget": (
                int(
                    dict(manifest.get("search_budget_by_family", {})).get(
                        AUGMENTED_METHOD_FAMILY, -1
                    )
                )
                == int(
                    dict(manifest.get("search_budget_by_family", {})).get(
                        COMPARATOR_METHOD_FAMILY, -2
                    )
                )
            ),
            "retention_placement_is_phase_14_machinery_not_this_analysis": (
                manifest.get("retention_placement")
            ),
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
        },
        "primary": {
            "mean_paired_rmse_reduction": mean_effect,
            "bootstrap": bootstrap,
            "improved_unit_count": improved,
            "worsened_unit_count": worsened,
            "tied_unit_count": tied,
            "paired_deltas": deltas,
        },
        "secondary": _wilcoxon(deltas),
        "treatment_size": {
            "accepted_synthetic_rows_by_seed": {
                str(row["seed"]): row["accepted_synthetic_rows"]
                for row in unit_rows
            },
            "transfer_pool_parent_count_by_seed": {
                str(row["seed"]): row["transfer_pool_parent_count"]
                for row in unit_rows
            },
        },
        "verdict": verdict,
    }
    payload["analysis_hash"] = stable_hash(payload)
    return payload


def wilcoxon_secondary(deltas: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank on paired deltas, labelled secondary."""
    values = np.asarray(list(deltas), dtype=float)
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return {
            "test": "wilcoxon_signed_rank_two_sided",
            "label": "secondary",
            "statistic": None,
            "p_value": None,
            "note": (
                "Undefined: every included paired delta is exactly zero. "
                "Significance never establishes success in any case."
            ),
        }
    result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    return {
        "test": "wilcoxon_signed_rank_two_sided",
        "label": "secondary",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "note": (
            "Secondary. The interval and the practical margin decide the "
            "outcome; significance alone never establishes success."
        ),
    }


_wilcoxon = wilcoxon_secondary


def _evaluation_unit(seed: int) -> str:
    fraction = PRIMARY_TRAIN_FRACTION
    text = f"{fraction:g}"
    return f"canonical-random:seed={int(seed)}:fraction={text}"


def render_confirmatory_report(payload: Mapping[str, Any]) -> str:
    """Render the human-readable primary confirmatory report."""
    prereg = payload["preregistration"]
    run = payload["run"]
    accounting = payload["accounting"]
    primary = payload["primary"]
    secondary = payload["secondary"]
    verdict = payload["verdict"]
    bootstrap = primary["bootstrap"]

    lines: list[str] = []
    lines.append("# Primary confirmatory result (Phase 15)")
    lines.append("")
    lines.append(
        f"Preregistered in `{prereg['document']}` at commit "
        f"`{prereg['commit'][:7]}`, before any confirmatory outer-test outcome "
        "existed. Every threshold below is quoted from that document, not "
        "chosen after seeing these numbers."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{str(verdict['verdict']).upper()}** — {verdict['reason']}")
    lines.append("")
    lines.append(
        f"Computed mechanically by rule `{verdict['rule']}` from the numbers "
        "below."
    )
    lines.append("")
    lines.append("## The question")
    lines.append("")
    lines.append(
        f"Does `{prereg['augmented_method_family']}` reduce outer-test "
        f"{str(prereg['metric']).upper()} relative to the matched real-only "
        f"`{prereg['comparator_method_family']}` control at training fraction "
        f"{prereg['train_fraction']}? Positive deltas favour augmentation."
    )
    lines.append("")
    lines.append("## Paired per-seed deltas")
    lines.append("")
    lines.append(
        "| Seed | Comparator RMSE | Augmented RMSE | Delta (reduction) | "
        "Accepted synthetic rows | Status |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["per_unit_rows"]:
        comparator = row["comparator_rmse"]
        augmented = row["augmented_rmse"]
        delta = row["paired_delta_rmse_reduction"]
        lines.append(
            "| {seed} | {comparator} | {augmented} | {delta} | {accepted} | "
            "{status} |".format(
                seed=row["seed"],
                comparator="n/a" if comparator is None else f"{comparator:.6f}",
                augmented="n/a" if augmented is None else f"{augmented:.6f}",
                delta="n/a" if delta is None else f"{delta:+.6f}",
                accepted=(
                    "n/a"
                    if row["accepted_synthetic_rows"] is None
                    else row["accepted_synthetic_rows"]
                ),
                status=row["status"],
            )
        )
    lines.append("")
    lines.append("## Primary estimand")
    lines.append("")
    if bootstrap is None:
        lines.append("No unit survived exclusion; no effect is estimable.")
    else:
        lines.append(
            f"- Mean paired RMSE reduction: **{bootstrap['mean']:+.6f}**"
        )
        lines.append(
            "- 95% percentile bootstrap over seeds as clusters "
            f"({bootstrap['replicates']} replicates, fixed seed "
            f"{bootstrap['seed']}): "
            f"**[{bootstrap['ci_lower']:+.6f}, {bootstrap['ci_upper']:+.6f}]**"
        )
        lines.append(
            "- Practical minimum effect: "
            f"{prereg['practical_minimum_effect']} RMSE"
        )
        lines.append(
            f"- Units improved: {primary['improved_unit_count']}; worsened: "
            f"{primary['worsened_unit_count']}; tied: "
            f"{primary['tied_unit_count']}"
        )
    lines.append("")
    lines.append("## Secondary test")
    lines.append("")
    if secondary["p_value"] is None:
        lines.append(f"- Wilcoxon signed-rank: {secondary['note']}")
    else:
        lines.append(
            "- Two-sided Wilcoxon signed-rank p-value: "
            f"{secondary['p_value']:.6f} (statistic "
            f"{secondary['statistic']:.1f}) — **secondary**. "
            "Significance alone never establishes success."
        )
    lines.append("")
    lines.append("## Unit accounting")
    lines.append("")
    lines.append(f"- Planned units: {accounting['planned_unit_count']}")
    lines.append(
        f"- Included: {accounting['included_unit_count']} "
        f"(seeds {accounting['included_seeds']})"
    )
    lines.append(
        f"- Excluded, degenerate transfer pool: "
        f"{accounting['degenerate_excluded_unit_count']} "
        f"(seeds {accounting['degenerate_excluded_seeds']})"
    )
    lines.append(
        f"- Failed or missing: {accounting['failed_or_missing_unit_count']} "
        f"(seeds {accounting['failed_or_missing_seeds']})"
    )
    lines.append(
        "- All planned seeds accounted for: "
        f"{accounting['all_planned_seeds_accounted_for']}"
    )
    lines.append("")
    lines.append("## Protocol provenance")
    lines.append("")
    lines.append(f"- Run directory: `{run['directory']}`")
    lines.append(f"- Run manifest hash: `{run['manifest_hash']}`")
    lines.append(f"- Plan hash: `{run['plan_hash']}`")
    lines.append(f"- Dataset hash: `{run['dataset_hash']}`")
    lines.append(f"- Canonical split hash: `{run['canonical_split_hash']}`")
    lines.append(
        f"- Per-family search budget: `{run['search_budget_by_family']}`"
    )
    lines.append(
        "- The two preregistered arms received equal search budgets: "
        f"{run['equal_arm_search_budget']}"
    )
    lines.append(
        "- The run also computed a Phase 14 AE retention decision "
        f"(`{run['retention_placement_is_phase_14_machinery_not_this_analysis']}`). "
        "That is Phase 14 machinery and forms no part of this analysis."
    )
    lines.append(f"- Analysis hash: `{payload['analysis_hash']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Random-split behaviour is the primary regime. The Phase 9 chemical "
        "OOD regimes are not part of this confirmation. The five other method "
        "families present in the run are descriptive only and enter no "
        "preregistered claim."
    )
    lines.append("")
    return "\n".join(lines)
