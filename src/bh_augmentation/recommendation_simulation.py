"""Retrospective batch recommendation simulation over already-measured reactions.

The simulation answers one question only: given a fixed experimental budget spent
in rounds, does a validated prediction system select reactions that are more
useful than realistic budget-matched baselines?

Design contracts enforced by this module:

* Labels are revealed only for reactions a policy chose to acquire. At round ``t``
  an acquisition function receives exactly the labels revealed in rounds ``< t``
  plus the initial seed pool, and nothing else. :class:`LabelOracle` performs a
  selective on-disk read of the allowed rows only, and
  :class:`AcquisitionContext` raises :class:`FutureLabelAccessError` whenever an
  unacquired label is requested.
* Every strategy inside one evaluation unit shares an identical seed pool,
  candidate pool, budget, round count and batch size.
* Campaign pools contain exactly one row per canonical reaction, so a canonical
  reaction can never appear as both a seed reaction and a later discovery.
* Only the uncertainty machinery that survived Phase 12 may be configured. The
  supervised autoencoder is unavailable and is not referenced anywhere here.
* Pool-wide outcomes are loaded only after every campaign has finished, through
  an explicit gate, so evaluation labels can never re-enter acquisition.

No claim about synthesis feasibility, safety, accessibility, laboratory success,
or prospective validation is made or supported by these artifacts.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from bh_augmentation.data.outcome_access import read_allowed_outcomes
from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.metrics import rmse, spearman_corr
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_saved_random_split_unit,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.uncertainty.calibration import interval_calibration_metrics
from bh_augmentation.uncertainty.estimators import (
    UncertaintyEstimatorConfig,
    fit_uncertainty_estimator,
)
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
)

RECOMMENDATION_SIMULATION_SCHEMA_VERSION = "bh-recommendation-simulation-v1"

#: Uncertainty estimators that passed Phase 12 calibrated selection. Phase 12
#: production froze thirty policies: ``bootstrap_extra_trees`` won twenty-six and
#: ``heterogeneous_disagreement`` won four. No other estimator may be configured.
PHASE12_SURVIVING_UNCERTAINTY_METHODS = (
    "bootstrap_extra_trees",
    "heterogeneous_disagreement",
)

RANDOM_STRATEGY = "random"
INFORMED_STRATEGIES = (
    "greedy_predicted_yield",
    "upper_confidence_bound",
    "expected_improvement",
    "thompson_sampling",
    "diversity_aware",
    "support_distance",
)
ALL_STRATEGIES = (RANDOM_STRATEGY,) + INFORMED_STRATEGIES

ACQUISITION_PROTOCOL = {
    "label_visibility": "seed_pool_plus_rounds_strictly_before_current_round",
    "pool_rows_per_canonical_reaction": 1,
    "matched_across_strategies": [
        "initial_seed_pool",
        "initial_candidate_pool",
        "rounds",
        "batch_size",
        "total_budget",
    ],
    "model_refit_scope": "acquired_labels_only",
    "uncertainty_calibration": "split_conformal_within_acquired_labels",
    "pool_outcome_load": "after_all_campaigns_complete",
    "prospective_validation": False,
}

PERCENTILE_METRICS = (
    "best_yield_discovered",
    "simple_regret",
    "cumulative_regret",
    "cumulative_mean_acquired_yield",
    "top_k_recall",
    "high_yield_hit_count",
    "unique_substrate_keys",
    "unique_condition_keys",
    "batch_mean_support_distance",
)
_HIGHER_IS_BETTER = {
    "best_yield_discovered": True,
    "simple_regret": False,
    "cumulative_regret": False,
    "cumulative_mean_acquired_yield": True,
    "top_k_recall": True,
    "high_yield_hit_count": True,
    "unique_substrate_keys": True,
    "unique_condition_keys": True,
    "batch_mean_support_distance": True,
}

_OUTPUTS = (
    "campaign_plan.json",
    "split_units.csv",
    "campaign_pools.csv",
    "acquisition_audit.csv",
    "acquisitions.csv",
    "uncertainty_diagnostics.csv",
    "round_trajectory.csv",
    "random_distribution.csv",
    "strategy_percentiles.csv",
    "summary.csv",
)
_PLAN_FIELDS = {
    "schema_version",
    "status",
    "dataset_hash",
    "canonical_split_hash",
    "canonical_split_artifact_hashes",
    "feature_contract",
    "feature_metadata_hash",
    "split_units",
    "campaign_pools",
    "strategies",
    "acquisition_protocol",
    "surviving_uncertainty_methods",
    "resolved_scientific_config",
    "config_hash",
    "pool_outcomes_loaded",
    "future_labels_visible_to_acquisition",
    "supervised_autoencoder_used",
    "prospective_validation_claimed",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "git_commit",
    "git_dirty_at_execution",
    "dataset_hash",
    "canonical_split_hash",
    "feature_metadata_hash",
    "config_hash",
    "plan_hash",
    "resolved_scientific_config",
    "dependency_versions",
    "command",
    "evaluation_unit_count",
    "strategy_count",
    "informed_strategy_count",
    "random_replicate_count",
    "rounds",
    "batch_size",
    "seed_pool_size",
    "total_budget",
    "campaign_count",
    "campaign_pool_count",
    "acquisition_audit_row_count",
    "acquisition_row_count",
    "uncertainty_diagnostic_row_count",
    "round_trajectory_row_count",
    "random_distribution_row_count",
    "strategy_percentile_row_count",
    "summary_row_count",
    "uncertainty_method",
    "surviving_uncertainty_methods",
    "acquisition_protocol",
    "future_label_access_events",
    "pool_outcomes_loaded_after_campaigns",
    "supervised_autoencoder_used",
    "prospective_validation_claimed",
    "output_hashes",
}


class FutureLabelAccessError(ValueError):
    """Raised when acquisition code requests a label it may not see."""


class EvaluationOracleGateError(RuntimeError):
    """Raised when pool-wide outcomes are touched at the wrong time."""


@dataclass(slots=True)
class EvaluationOracleGate:
    """Runtime gate separating acquisition from pool-wide outcome scoring."""

    unlocked: bool = False

    def unlock(self) -> None:
        """Permit pool-wide outcome loading once every campaign has finished."""
        self.unlocked = True

    def require_locked(self) -> None:
        """Fail if pool outcomes became available while campaigns still run."""
        if self.unlocked:
            raise EvaluationOracleGateError(
                "Pool-wide outcomes became available during acquisition."
            )

    def require_unlocked(self) -> None:
        """Fail if pool outcomes are requested before campaigns complete."""
        if not self.unlocked:
            raise EvaluationOracleGateError(
                "Pool-wide outcomes may only be loaded after every campaign completes."
            )


@dataclass(slots=True)
class LabelOracle:
    """Reveal measured outcomes only for reactions a policy already acquired.

    Every reveal performs a selective on-disk read restricted to the requested
    source rows, so unacquired outcomes are never materialized in this path.
    """

    dataset_path: Path
    canonical_source_ids: tuple[str, ...]
    allowed_pool_ids: frozenset[str]
    campaign_id: str
    _revealed: dict[str, float] = field(default_factory=dict)
    _reveal_log: list[dict[str, Any]] = field(default_factory=list)
    _last_round: int = -1

    def reveal(self, source_ids: Sequence[str], *, round_index: int) -> dict[str, float]:
        """Reveal outcomes for newly acquired reactions in monotone round order."""
        ids = tuple(str(value) for value in source_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Acquisition reveals must be nonempty and unique.")
        if int(round_index) < self._last_round:
            raise ValueError("Label reveals must follow non-decreasing round order.")
        outside = sorted(set(ids) - self.allowed_pool_ids)
        if outside:
            raise ValueError(f"Reveal requested outside the campaign pool: {outside[:5]}.")
        repeated = sorted(set(ids) & set(self._revealed))
        if repeated:
            raise ValueError(f"Reveal requested for already-acquired rows: {repeated[:5]}.")
        values = read_allowed_outcomes(self.dataset_path, self.canonical_source_ids, ids)
        self._revealed.update(values)
        self._last_round = int(round_index)
        self._reveal_log.append(
            {
                "campaign_id": self.campaign_id,
                "round_index": int(round_index),
                "revealed_count": len(ids),
                "revealed_source_id_hash": stable_hash(sorted(ids)),
                "cumulative_revealed_count": len(self._revealed),
                "cumulative_revealed_source_id_hash": self.revealed_source_id_hash,
            }
        )
        return dict(values)

    @property
    def revealed_source_ids(self) -> tuple[str, ...]:
        """Return every revealed source ID in stable order."""
        return tuple(sorted(self._revealed))

    @property
    def revealed_source_id_hash(self) -> str:
        """Hash the complete revealed membership."""
        return stable_hash(list(self.revealed_source_ids))

    def visible_labels(self) -> dict[str, float]:
        """Return a copy of every outcome revealed so far."""
        return dict(self._revealed)

    def reveal_log(self) -> tuple[dict[str, Any], ...]:
        """Return the immutable reveal history."""
        return tuple(dict(record) for record in self._reveal_log)


@dataclass(frozen=True, slots=True)
class AcquisitionSelection:
    """Candidate-array positions selected for one round plus their scores."""

    positions: tuple[int, ...]
    scores: tuple[float, ...]
    score_kind: str


@dataclass(frozen=True, slots=True)
class AcquisitionContext:
    """Everything an acquisition function is permitted to see at one round.

    The object deliberately carries no candidate outcome. ``label_of`` is the
    only label accessor, and it refuses every source ID the campaign has not
    already acquired.
    """

    evaluation_unit: str
    strategy: str
    replicate: int
    round_index: int
    batch_size: int
    candidate_source_ids: tuple[str, ...]
    candidate_pool_positions: np.ndarray
    labeled_source_ids: tuple[str, ...]
    labeled_values: Mapping[str, float] | None
    similarity: np.ndarray
    nearest_similarity: np.ndarray
    point: np.ndarray | None
    calibrated_uncertainty: np.ndarray | None
    rng: np.random.Generator
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Reject any context whose candidate pool overlaps the labeled pool."""
        if len(self.candidate_source_ids) != len(self.candidate_pool_positions):
            raise ValueError("Candidate identities and positions must align.")
        if len(self.candidate_source_ids) != len(self.nearest_similarity):
            raise ValueError("Candidate identities and support distances must align.")
        overlap = set(self.candidate_source_ids) & set(self.labeled_source_ids)
        if overlap:
            raise FutureLabelAccessError(
                f"Candidate rows are already labeled: {sorted(overlap)[:5]}."
            )
        if self.labeled_values is not None and set(self.labeled_values) != set(
            self.labeled_source_ids
        ):
            raise FutureLabelAccessError(
                "Labeled value mapping does not match the acquired membership."
            )
        if self.batch_size < 1 or self.batch_size > len(self.candidate_source_ids):
            raise ValueError("Batch size must fit inside the remaining candidate pool.")

    @property
    def support_distance(self) -> np.ndarray:
        """Return one minus the nearest cosine similarity to acquired support."""
        return 1.0 - np.asarray(self.nearest_similarity, dtype=np.float64)

    @property
    def consumes_labels(self) -> bool:
        """Return whether this campaign exposes acquired labels to the scorer."""
        return self.labeled_values is not None

    def label_of(self, source_id: str) -> float:
        """Return an acquired label, refusing every unacquired reaction."""
        key = str(source_id)
        if self.labeled_values is None:
            raise FutureLabelAccessError(
                f"Strategy {self.strategy!r} is label-free and may not read outcomes."
            )
        if key not in self.labeled_values:
            raise FutureLabelAccessError(
                f"Acquisition requested an unacquired label at round "
                f"{self.round_index}: {key}."
            )
        return float(self.labeled_values[key])

    @property
    def best_observed_label(self) -> float:
        """Return the best outcome this campaign has already acquired."""
        if self.labeled_values is None:
            raise FutureLabelAccessError(
                f"Strategy {self.strategy!r} is label-free and may not read outcomes."
            )
        return float(max(self.labeled_values.values()))

    @property
    def scorer_input_hash(self) -> str:
        """Hash every input an acquisition function is allowed to consume."""
        return stable_hash(
            {
                "evaluation_unit": self.evaluation_unit,
                "strategy": self.strategy,
                "replicate": int(self.replicate),
                "round_index": int(self.round_index),
                "batch_size": int(self.batch_size),
                "candidate_source_ids": list(self.candidate_source_ids),
                "labeled_source_ids": list(self.labeled_source_ids),
                "labeled_value_hash": (
                    None
                    if self.labeled_values is None
                    else stable_hash(
                        [
                            [source_id, float(self.labeled_values[source_id])]
                            for source_id in self.labeled_source_ids
                        ]
                    )
                ),
                "point_hash": _optional_array_hash(self.point),
                "uncertainty_hash": _optional_array_hash(self.calibrated_uncertainty),
                "nearest_similarity_hash": _optional_array_hash(self.nearest_similarity),
                "params": dict(sorted(self.params.items())),
            }
        )

    def require_model(self) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated point and uncertainty vectors or fail."""
        if self.point is None or self.calibrated_uncertainty is None:
            raise ValueError(f"Strategy {self.strategy!r} requires a fitted model.")
        return self.point, self.calibrated_uncertainty


def _select_random(context: AcquisitionContext) -> AcquisitionSelection:
    positions = context.rng.choice(
        len(context.candidate_source_ids), size=context.batch_size, replace=False
    )
    ordered = tuple(int(value) for value in np.sort(positions))
    return AcquisitionSelection(
        positions=ordered,
        scores=tuple(0.0 for _ in ordered),
        score_kind="uniform_random",
    )


def _select_greedy(context: AcquisitionContext) -> AcquisitionSelection:
    point, _ = context.require_model()
    return _top_positions(point, context.batch_size, "predicted_yield")


def _select_upper_confidence_bound(context: AcquisitionContext) -> AcquisitionSelection:
    point, uncertainty = context.require_model()
    beta = float(context.params["ucb_beta"])
    return _top_positions(point + beta * uncertainty, context.batch_size, "ucb")


def _select_expected_improvement(context: AcquisitionContext) -> AcquisitionSelection:
    point, uncertainty = context.require_model()
    xi = float(context.params["expected_improvement_xi"])
    sigma = np.maximum(uncertainty / float(context.params["coverage_z"]), 1.0e-9)
    gap = point - context.best_observed_label - xi
    z = gap / sigma
    scores = gap * norm.cdf(z) + sigma * norm.pdf(z)
    return _top_positions(scores, context.batch_size, "expected_improvement")


def _select_thompson(context: AcquisitionContext) -> AcquisitionSelection:
    point, uncertainty = context.require_model()
    sigma = np.maximum(uncertainty / float(context.params["coverage_z"]), 1.0e-9)
    draws = point + sigma * context.rng.standard_normal(len(point))
    return _top_positions(draws, context.batch_size, "posterior_draw")


def _select_diversity_aware(context: AcquisitionContext) -> AcquisitionSelection:
    point, _ = context.require_model()
    return _sequential_greedy(
        context,
        base=np.asarray(point, dtype=np.float64),
        penalty=float(context.params["diversity_penalty"]),
        score_kind="diversity_penalized_yield",
    )


def _select_support_distance(context: AcquisitionContext) -> AcquisitionSelection:
    point, _ = context.require_model()
    weight = float(context.params["support_distance_yield_weight"])
    return _sequential_greedy(
        context,
        base=weight * np.asarray(point, dtype=np.float64),
        penalty=1.0,
        score_kind="support_distance",
    )


ACQUISITION_STRATEGIES: dict[str, Callable[[AcquisitionContext], AcquisitionSelection]] = {
    RANDOM_STRATEGY: _select_random,
    "greedy_predicted_yield": _select_greedy,
    "upper_confidence_bound": _select_upper_confidence_bound,
    "expected_improvement": _select_expected_improvement,
    "thompson_sampling": _select_thompson,
    "diversity_aware": _select_diversity_aware,
    "support_distance": _select_support_distance,
}


def _sequential_greedy(
    context: AcquisitionContext,
    *,
    base: np.ndarray,
    penalty: float,
    score_kind: str,
) -> AcquisitionSelection:
    """Select a batch greedily while updating similarity to the growing batch."""
    working = np.asarray(context.nearest_similarity, dtype=np.float64).copy()
    available = np.ones(len(base), dtype=bool)
    positions: list[int] = []
    scores: list[float] = []
    candidate_positions = context.candidate_pool_positions
    for _ in range(context.batch_size):
        current = base - penalty * working
        current[~available] = -np.inf
        chosen = int(np.argmax(current))
        positions.append(chosen)
        scores.append(float(current[chosen]))
        available[chosen] = False
        working = np.maximum(
            working,
            np.asarray(
                context.similarity[candidate_positions, candidate_positions[chosen]],
                dtype=np.float64,
            ),
        )
    order = np.argsort(np.asarray(positions), kind="stable")
    return AcquisitionSelection(
        positions=tuple(int(positions[index]) for index in order),
        scores=tuple(float(scores[index]) for index in order),
        score_kind=score_kind,
    )


def _top_positions(
    scores: np.ndarray, batch_size: int, score_kind: str
) -> AcquisitionSelection:
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Acquisition scores for {score_kind} must be finite.")
    ranked = np.argsort(-values, kind="stable")[:batch_size]
    ordered = np.sort(ranked)
    return AcquisitionSelection(
        positions=tuple(int(value) for value in ordered),
        scores=tuple(float(values[value]) for value in ordered),
        score_kind=score_kind,
    )


@dataclass(frozen=True, slots=True)
class CampaignPool:
    """One immutable, group-separated campaign pool for a single split unit."""

    evaluation_unit: str
    seed: int
    pool_source_ids: tuple[str, ...]
    seed_source_ids: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    dropped_duplicate_group_ids: tuple[str, ...]
    pool_group_count: int
    seed_candidate_group_overlap: int
    rounds: int
    batch_size: int

    def __post_init__(self) -> None:
        """Reject unmatched, overlapping or under-sized campaign pools."""
        if set(self.seed_source_ids) & set(self.candidate_source_ids):
            raise ValueError("Seed and candidate pools overlap.")
        if set(self.seed_source_ids) | set(self.candidate_source_ids) != set(
            self.pool_source_ids
        ):
            raise ValueError("Seed and candidate pools do not partition the campaign pool.")
        if self.pool_group_count != len(self.pool_source_ids):
            raise ValueError("Campaign pools must contain one row per canonical reaction.")
        if self.seed_candidate_group_overlap != 0:
            raise ValueError("A canonical reaction is both a seed and a discovery.")
        if self.rounds * self.batch_size > len(self.candidate_source_ids):
            raise ValueError("Configured budget exceeds the candidate pool size.")

    @property
    def total_budget(self) -> int:
        """Return the number of acquisitions every strategy is allowed to make."""
        return int(self.rounds) * int(self.batch_size)

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return manifest-ready pool provenance."""
        return {
            "evaluation_unit": self.evaluation_unit,
            "seed": int(self.seed),
            "pool_row_count": len(self.pool_source_ids),
            "pool_group_count": int(self.pool_group_count),
            "pool_rows_per_canonical_reaction": 1,
            "pool_source_id_hash": stable_hash(list(self.pool_source_ids)),
            "seed_pool_size": len(self.seed_source_ids),
            "seed_source_id_hash": stable_hash(list(self.seed_source_ids)),
            "candidate_pool_size": len(self.candidate_source_ids),
            "candidate_source_id_hash": stable_hash(list(self.candidate_source_ids)),
            "seed_candidate_group_overlap": int(self.seed_candidate_group_overlap),
            "dropped_duplicate_group_rows": len(self.dropped_duplicate_group_ids),
            "dropped_duplicate_group_hash": stable_hash(
                list(self.dropped_duplicate_group_ids)
            ),
            "rounds": int(self.rounds),
            "batch_size": int(self.batch_size),
            "total_budget": self.total_budget,
        }


def run_recommendation_simulation(
    config: str | Path | Mapping[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run every matched acquisition campaign and write validated artifacts."""
    contract = _resolve_contract(_load_config(config))
    prepared = _prepare(contract)
    plan = _build_plan(contract, prepared)

    output = Path(
        output_directory if output_directory is not None else contract["output_directory"]
    )
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recommendation output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        name.removesuffix(".json").removesuffix(".csv"): output / name for name in _OUTPUTS
    }
    paths["output_directory"] = output
    paths["manifest"] = output / "manifest.json"

    _write_json(paths["campaign_plan"], plan)
    pd.DataFrame([unit.audit_record for unit in prepared["units"]]).to_csv(
        paths["split_units"], index=False
    )
    pd.DataFrame([pool.audit_record for pool in prepared["pools"]]).to_csv(
        paths["campaign_pools"], index=False
    )

    gate = EvaluationOracleGate()
    campaigns = _run_all_campaigns(contract, prepared, plan["plan_hash"], gate)
    outcomes = _score_all_campaigns(contract, prepared, campaigns, plan["plan_hash"], gate)

    pd.DataFrame(campaigns["audit_rows"]).to_csv(paths["acquisition_audit"], index=False)
    pd.DataFrame(outcomes["acquisition_rows"]).to_csv(paths["acquisitions"], index=False)
    pd.DataFrame(outcomes["uncertainty_rows"]).to_csv(
        paths["uncertainty_diagnostics"], index=False
    )
    trajectory = pd.DataFrame(outcomes["trajectory_rows"])
    trajectory.to_csv(paths["round_trajectory"], index=False)
    distribution = _random_distribution(trajectory)
    distribution.to_csv(paths["random_distribution"], index=False)
    percentiles = _strategy_percentiles(trajectory)
    percentiles.to_csv(paths["strategy_percentiles"], index=False)
    summary = _summarize(trajectory, percentiles, contract)
    summary.to_csv(paths["summary"], index=False)

    manifest = _build_manifest(
        contract=contract,
        prepared=prepared,
        plan=plan,
        campaigns=campaigns,
        outcomes=outcomes,
        trajectory=trajectory,
        distribution=distribution,
        percentiles=percentiles,
        summary=summary,
        config=config,
        output=output,
    )
    _write_json(paths["manifest"], manifest)
    validate_recommendation_simulation(output)
    return paths


def validate_recommendation_simulation(directory: str | Path) -> dict[str, Any]:
    """Independently replay the whole simulation from the saved artifacts."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed_manifest_hash = manifest.pop("manifest_hash", None)
    if (
        claimed_manifest_hash != stable_hash(manifest)
        or set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema_version") != RECOMMENDATION_SIMULATION_SCHEMA_VERSION
        or manifest.get("status") != "corrected_revalidation_complete"
        or manifest.get("acquisition_protocol") != ACQUISITION_PROTOCOL
        or manifest.get("surviving_uncertainty_methods")
        != list(PHASE12_SURVIVING_UNCERTAINTY_METHODS)
        or manifest.get("uncertainty_method") not in PHASE12_SURVIVING_UNCERTAINTY_METHODS
        or manifest.get("future_label_access_events") != 0
        or manifest.get("pool_outcomes_loaded_after_campaigns") is not True
        or manifest.get("supervised_autoencoder_used") is not False
        or manifest.get("prospective_validation_claimed") is not False
        or int(manifest.get("random_replicate_count", -1)) < 200
    ):
        raise ValueError("Invalid recommendation simulation completion manifest.")
    manifest["manifest_hash"] = claimed_manifest_hash
    if set(manifest.get("output_hashes", {})) != set(_OUTPUTS):
        raise ValueError("Recommendation simulation output hash coverage mismatch.")
    for name, digest in manifest["output_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Recommendation simulation output hash mismatch: {name}.")

    plan = json.loads((root / "campaign_plan.json").read_text())
    claimed_plan_hash = plan.pop("plan_hash", None)
    if (
        claimed_plan_hash != stable_hash(plan)
        or set(plan) != _PLAN_FIELDS
        or plan.get("schema_version") != RECOMMENDATION_SIMULATION_SCHEMA_VERSION
        or plan.get("status") != "frozen_before_any_outcome_loading"
        or plan.get("acquisition_protocol") != ACQUISITION_PROTOCOL
        or plan.get("pool_outcomes_loaded") is not False
        or plan.get("future_labels_visible_to_acquisition") is not False
        or plan.get("supervised_autoencoder_used") is not False
        or plan.get("prospective_validation_claimed") is not False
        or plan.get("strategies") != list(ALL_STRATEGIES)
        or plan.get("config_hash") != stable_hash(plan.get("resolved_scientific_config"))
    ):
        raise ValueError("Invalid frozen recommendation simulation plan.")
    plan["plan_hash"] = claimed_plan_hash
    if (
        manifest["plan_hash"] != claimed_plan_hash
        or manifest["config_hash"] != plan["config_hash"]
        or manifest["resolved_scientific_config"] != plan["resolved_scientific_config"]
    ):
        raise ValueError("Recommendation simulation manifest-plan mismatch.")

    contract = _contract_from_scientific(plan["resolved_scientific_config"], root)
    prepared = _prepare(contract)
    replay_plan = _build_plan(contract, prepared)
    if replay_plan["plan_hash"] != claimed_plan_hash:
        raise ValueError("Recommendation simulation plan replay mismatch.")
    if (
        manifest["dataset_hash"] != prepared["dataset_hash"]
        or manifest["canonical_split_hash"] != prepared["canonical_split_hash"]
        or manifest["feature_metadata_hash"]
        != prepared["feature_contract"]["feature_metadata_hash"]
    ):
        raise ValueError("Recommendation simulation canonical dependency mismatch.")

    gate = EvaluationOracleGate()
    campaigns = _run_all_campaigns(contract, prepared, claimed_plan_hash, gate)
    outcomes = _score_all_campaigns(
        contract, prepared, campaigns, claimed_plan_hash, gate
    )
    expected = {
        "split_units.csv": pd.DataFrame([u.audit_record for u in prepared["units"]]),
        "campaign_pools.csv": pd.DataFrame(
            [pool.audit_record for pool in prepared["pools"]]
        ),
        "acquisition_audit.csv": pd.DataFrame(campaigns["audit_rows"]),
        "acquisitions.csv": pd.DataFrame(outcomes["acquisition_rows"]),
        "uncertainty_diagnostics.csv": pd.DataFrame(outcomes["uncertainty_rows"]),
        "round_trajectory.csv": pd.DataFrame(outcomes["trajectory_rows"]),
    }
    trajectory = expected["round_trajectory.csv"]
    expected["random_distribution.csv"] = _random_distribution(trajectory)
    expected["strategy_percentiles.csv"] = _strategy_percentiles(trajectory)
    expected["summary.csv"] = _summarize(
        trajectory, expected["strategy_percentiles.csv"], contract
    )
    for name, frame in expected.items():
        observed = pd.read_csv(root / name, float_precision="round_trip")
        if not _records_close(_frame_records(frame), _frame_records(observed)):
            raise ValueError(f"Recommendation simulation replay mismatch: {name}.")

    _assert_matched_conditions(expected["acquisition_audit.csv"])
    _assert_no_future_labels(expected["acquisition_audit.csv"])
    _assert_random_distribution_complete(trajectory, contract["random_replicates"])
    _assert_group_separation(expected["campaign_pools.csv"], expected["acquisitions.csv"])
    _assert_manifest_counts(manifest, expected, contract, prepared)
    return manifest


def _assert_matched_conditions(audit: pd.DataFrame) -> None:
    """Fail unless every strategy in a unit shared identical starting conditions."""
    for unit, rows in audit.groupby("evaluation_unit"):
        initial = rows.loc[rows["round_index"].eq(1)]
        for column in (
            "initial_seed_source_id_hash",
            "initial_candidate_source_id_hash",
            "seed_pool_size",
            "candidate_pool_size",
            "rounds",
            "batch_size",
            "total_budget",
        ):
            if initial[column].nunique() != 1:
                raise ValueError(
                    f"Matched-condition violation in {unit}: {column} differs across "
                    "strategies."
                )
        if rows.groupby(["strategy", "replicate"])["round_index"].max().nunique() != 1:
            raise ValueError(f"Matched-condition violation in {unit}: round counts differ.")
        if rows.groupby(["strategy", "replicate"]).size().nunique() != 1:
            raise ValueError(f"Matched-condition violation in {unit}: budgets differ.")


def _assert_no_future_labels(audit: pd.DataFrame) -> None:
    """Fail unless every acquisition score used only previously acquired labels."""
    if not audit["scorer_visible_label_id_hash"].eq(audit["labeled_before_id_hash"]).all():
        raise ValueError("Acquisition scorer saw labels beyond the acquired history.")
    if (
        not audit["oracle_cumulative_revealed_id_hash"]
        .eq(audit["labeled_before_id_hash"])
        .all()
    ):
        raise ValueError("Revealed outcomes exceed the acquired history at scoring time.")
    if int(audit["future_label_access_events"].sum()) != 0:
        raise ValueError("Future label access recorded during acquisition.")
    if int(audit["candidate_labeled_overlap"].sum()) != 0:
        raise ValueError("Candidate rows overlapped the labeled pool during acquisition.")
    expected = audit["seed_pool_size"] + (audit["round_index"] - 1) * audit["batch_size"]
    if not audit["labeled_before_count"].eq(expected).all():
        raise ValueError("Acquired history size is inconsistent with the round schedule.")
    remaining = audit["candidate_pool_size"] - (audit["round_index"] - 1) * audit[
        "batch_size"
    ]
    if not audit["candidate_before_count"].eq(remaining).all():
        raise ValueError("Candidate pool size is inconsistent with the round schedule.")


def _assert_random_distribution_complete(
    trajectory: pd.DataFrame, replicates: int
) -> None:
    """Fail unless the random baseline is a complete independent distribution."""
    random_rows = trajectory.loc[trajectory["strategy"].eq(RANDOM_STRATEGY)]
    if random_rows.empty:
        raise ValueError("Random baseline distribution is missing.")
    if int(replicates) < 200:
        raise ValueError("Random baseline requires at least 200 independent replicates.")
    counts = random_rows.groupby(["evaluation_unit", "round_index"])["replicate"].nunique()
    if not counts.eq(int(replicates)).all():
        raise ValueError("Random baseline distribution is incomplete.")
    if bool(
        random_rows.duplicated(
            subset=["evaluation_unit", "replicate", "round_index"]
        ).any()
    ):
        raise ValueError("Random baseline distribution contains duplicated replicate rows.")
    distinct = random_rows.loc[random_rows["round_index"].gt(0)].groupby(
        ["evaluation_unit", "round_index"]
    )["selected_source_id_hash"].nunique()
    if not distinct.gt(1).all():
        raise ValueError("Random replicates are not independent draws.")


def _assert_group_separation(pools: pd.DataFrame, acquisitions: pd.DataFrame) -> None:
    """Fail unless canonical reaction identity separates seeds from discoveries."""
    if not pools["seed_candidate_group_overlap"].eq(0).all():
        raise ValueError("A canonical reaction appears as both a seed and a discovery.")
    if not pools["pool_rows_per_canonical_reaction"].eq(1).all():
        raise ValueError("Campaign pools contain repeated canonical reactions.")
    if not pools["pool_group_count"].eq(pools["pool_row_count"]).all():
        raise ValueError("Campaign pool group count does not match its row count.")
    if acquisitions.empty:
        return
    if bool(
        acquisitions.duplicated(
            subset=["evaluation_unit", "strategy", "canonical_reaction_key"]
        ).any()
    ):
        raise ValueError("A canonical reaction was acquired twice in one campaign.")
    if not acquisitions["acquired_reaction_in_seed_pool"].eq(False).all():
        raise ValueError("A seed reaction was recorded as a discovery.")


def _assert_manifest_counts(
    manifest: Mapping[str, Any],
    expected: Mapping[str, pd.DataFrame],
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> None:
    counts = {
        "evaluation_unit_count": len(prepared["units"]),
        "strategy_count": len(ALL_STRATEGIES),
        "informed_strategy_count": len(INFORMED_STRATEGIES),
        "random_replicate_count": contract["random_replicates"],
        "rounds": contract["rounds"],
        "batch_size": contract["batch_size"],
        "seed_pool_size": contract["seed_pool_size"],
        "total_budget": contract["rounds"] * contract["batch_size"],
        "campaign_count": len(prepared["units"])
        * (len(INFORMED_STRATEGIES) + contract["random_replicates"]),
        "campaign_pool_count": len(expected["campaign_pools.csv"]),
        "acquisition_audit_row_count": len(expected["acquisition_audit.csv"]),
        "acquisition_row_count": len(expected["acquisitions.csv"]),
        "uncertainty_diagnostic_row_count": len(expected["uncertainty_diagnostics.csv"]),
        "round_trajectory_row_count": len(expected["round_trajectory.csv"]),
        "random_distribution_row_count": len(expected["random_distribution.csv"]),
        "strategy_percentile_row_count": len(expected["strategy_percentiles.csv"]),
        "summary_row_count": len(expected["summary.csv"]),
    }
    for name, value in counts.items():
        if int(manifest.get(name, -1)) != int(value):
            raise ValueError(f"Recommendation simulation manifest count mismatch: {name}.")


def _prepare(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize split identities, features and matched pools without labels."""
    saved = load_saved_canonical_split_identities(
        contract["dataset_path"],
        contract["canonical_split_directory"],
        requested_seeds=contract["random_seeds"],
        requested_fractions=(contract["pool_train_fraction"],),
    )
    units = tuple(
        build_saved_random_split_unit(
            saved, seed=seed, train_fraction=contract["pool_train_fraction"]
        )
        for seed in contract["random_seeds"]
    )
    if not units:
        raise ValueError("Recommendation simulation has no evaluation units.")
    canonical = saved.canonical.copy()
    canonical["source_row_id"] = canonical["source_row_id"].astype(str)
    canonical_source_ids = tuple(canonical["source_row_id"])
    identity_frame = canonical.copy()
    identity_frame["yield"] = 0.0
    features, _, feature_names, feature_metadata = build_feature_matrix_with_metadata(
        identity_frame, contract["feature_config"]
    )
    features = np.asarray(features, dtype=np.float32)
    if not np.isin(features, (0.0, 1.0)).all():
        raise ValueError(
            "Recommendation simulation requires binary fingerprint features so pool "
            "similarity is exactly reproducible."
        )
    feature_contract = feature_contract_record(feature_metadata, feature_names)
    pools = tuple(_build_pool(unit, canonical, contract) for unit in units)
    return {
        "saved": saved,
        "units": units,
        "pools": pools,
        "canonical": canonical,
        "canonical_source_ids": canonical_source_ids,
        "features": features,
        "feature_contract": feature_contract,
        "dataset_hash": saved.dataset_hash,
        "canonical_split_hash": saved.aggregate_split_hash,
        "canonical_split_artifact_hashes": saved.artifact_hashes,
        "position_of": {
            source_id: index for index, source_id in enumerate(canonical_source_ids)
        },
        "identity": canonical.set_index("source_row_id", drop=False),
    }


def _build_pool(
    unit: EvaluationSplitUnit,
    canonical: pd.DataFrame,
    contract: Mapping[str, Any],
) -> CampaignPool:
    """Build one group-separated campaign pool from a saved training partition."""
    rows = canonical.loc[
        canonical["source_row_id"].isin(set(unit.train_source_ids))
    ].copy()
    rows["canonical_reaction_key"] = rows["canonical_reaction_key"].astype(str)
    rows = rows.sort_values("source_row_id", kind="stable")
    kept = rows.drop_duplicates(subset="canonical_reaction_key", keep="first")
    pool_ids = tuple(sorted(kept["source_row_id"]))
    dropped = tuple(sorted(set(rows["source_row_id"]) - set(pool_ids)))
    if len(pool_ids) <= contract["seed_pool_size"]:
        raise ValueError("Campaign pool is too small for the configured seed pool.")
    key_of = dict(zip(kept["source_row_id"], kept["canonical_reaction_key"], strict=True))
    generator = np.random.default_rng(
        _derived_seed(contract["base_seed"], "seed-pool", unit.evaluation_unit)
    )
    chosen = generator.choice(len(pool_ids), size=contract["seed_pool_size"], replace=False)
    seed_ids = tuple(sorted(pool_ids[int(index)] for index in chosen))
    candidate_ids = tuple(sorted(set(pool_ids) - set(seed_ids)))
    seed_keys = {key_of[source_id] for source_id in seed_ids}
    candidate_keys = {key_of[source_id] for source_id in candidate_ids}
    return CampaignPool(
        evaluation_unit=unit.evaluation_unit,
        seed=int(unit.seed if unit.seed is not None else 0),
        pool_source_ids=pool_ids,
        seed_source_ids=seed_ids,
        candidate_source_ids=candidate_ids,
        dropped_duplicate_group_ids=dropped,
        pool_group_count=len(set(key_of.values())),
        seed_candidate_group_overlap=len(seed_keys & candidate_keys),
        rounds=contract["rounds"],
        batch_size=contract["batch_size"],
    )


def _build_plan(contract: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    scientific = _scientific_config(contract)
    plan = {
        "schema_version": RECOMMENDATION_SIMULATION_SCHEMA_VERSION,
        "status": "frozen_before_any_outcome_loading",
        "dataset_hash": prepared["dataset_hash"],
        "canonical_split_hash": prepared["canonical_split_hash"],
        "canonical_split_artifact_hashes": prepared["canonical_split_artifact_hashes"],
        "feature_contract": prepared["feature_contract"],
        "feature_metadata_hash": prepared["feature_contract"]["feature_metadata_hash"],
        "split_units": [unit.audit_record for unit in prepared["units"]],
        "campaign_pools": [pool.audit_record for pool in prepared["pools"]],
        "strategies": list(ALL_STRATEGIES),
        "acquisition_protocol": ACQUISITION_PROTOCOL,
        "surviving_uncertainty_methods": list(PHASE12_SURVIVING_UNCERTAINTY_METHODS),
        "resolved_scientific_config": scientific,
        "config_hash": stable_hash(scientific),
        "pool_outcomes_loaded": False,
        "future_labels_visible_to_acquisition": False,
        "supervised_autoencoder_used": False,
        "prospective_validation_claimed": False,
    }
    plan["plan_hash"] = stable_hash(plan)
    return plan


def _strategy_params(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ucb_beta": contract["ucb_beta"],
        "expected_improvement_xi": contract["expected_improvement_xi"],
        "diversity_penalty": contract["diversity_penalty"],
        "support_distance_yield_weight": contract["support_distance_yield_weight"],
        "coverage_z": float(norm.ppf(0.5 * (1.0 + contract["coverage"]))),
    }


def _run_all_campaigns(
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    plan_hash: str,
    gate: EvaluationOracleGate,
) -> dict[str, Any]:
    """Run every matched campaign without loading any unacquired outcome."""
    params = _strategy_params(contract)
    audit_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for pool in prepared["pools"]:
        similarity = _pool_similarity(pool, prepared)
        positions = {
            source_id: index for index, source_id in enumerate(pool.pool_source_ids)
        }
        for strategy in INFORMED_STRATEGIES:
            records.append(
                _run_campaign(
                    contract=contract,
                    prepared=prepared,
                    pool=pool,
                    similarity=similarity,
                    pool_positions=positions,
                    strategy=strategy,
                    replicate=0,
                    params=params,
                    plan_hash=plan_hash,
                    audit_rows=audit_rows,
                    gate=gate,
                )
            )
        for replicate in range(contract["random_replicates"]):
            records.append(
                _run_campaign(
                    contract=contract,
                    prepared=prepared,
                    pool=pool,
                    similarity=similarity,
                    pool_positions=positions,
                    strategy=RANDOM_STRATEGY,
                    replicate=replicate,
                    params=params,
                    plan_hash=plan_hash,
                    audit_rows=audit_rows,
                    gate=gate,
                )
            )
    return {"audit_rows": audit_rows, "records": records}


def _run_campaign(
    *,
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    pool: CampaignPool,
    similarity: np.ndarray,
    pool_positions: Mapping[str, int],
    strategy: str,
    replicate: int,
    params: Mapping[str, Any],
    plan_hash: str,
    audit_rows: list[dict[str, Any]],
    gate: EvaluationOracleGate,
) -> dict[str, Any]:
    """Run one campaign, revealing outcomes only for reactions it acquires."""
    campaign_id = f"{pool.evaluation_unit}|{strategy}|replicate={replicate}"
    informed = strategy != RANDOM_STRATEGY
    oracle: LabelOracle | None = None
    labeled_values: dict[str, float] | None = None
    if informed:
        oracle = LabelOracle(
            dataset_path=Path(contract["dataset_path"]),
            canonical_source_ids=prepared["canonical_source_ids"],
            allowed_pool_ids=frozenset(pool.pool_source_ids),
            campaign_id=campaign_id,
        )
        labeled_values = oracle.reveal(pool.seed_source_ids, round_index=0)

    seed_positions = np.asarray(
        [pool_positions[source_id] for source_id in pool.seed_source_ids], dtype=np.int64
    )
    nearest_to_support = similarity[:, seed_positions].max(axis=1)
    acquired: list[str] = list(pool.seed_source_ids)
    remaining: list[str] = list(pool.candidate_source_ids)
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, pool.rounds + 1):
        gate.require_locked()
        labeled_ids = tuple(sorted(acquired))
        candidate_ids = tuple(remaining)
        candidate_positions = np.asarray(
            [pool_positions[source_id] for source_id in candidate_ids], dtype=np.int64
        )
        nearest_similarity = nearest_to_support[candidate_positions]
        model_record: dict[str, Any] | None = None
        if informed and labeled_values is not None:
            model_record = _fit_round_model(
                contract=contract,
                prepared=prepared,
                pool=pool,
                strategy=strategy,
                round_index=round_index,
                labeled_ids=labeled_ids,
                labeled_values=labeled_values,
                candidate_ids=candidate_ids,
            )
        context = AcquisitionContext(
            evaluation_unit=pool.evaluation_unit,
            strategy=strategy,
            replicate=int(replicate),
            round_index=round_index,
            batch_size=pool.batch_size,
            candidate_source_ids=candidate_ids,
            candidate_pool_positions=candidate_positions,
            labeled_source_ids=labeled_ids,
            labeled_values=None if labeled_values is None else dict(labeled_values),
            similarity=similarity,
            nearest_similarity=np.asarray(nearest_similarity, dtype=np.float64),
            point=None if model_record is None else model_record["point"],
            calibrated_uncertainty=(
                None if model_record is None else model_record["calibrated_uncertainty"]
            ),
            rng=np.random.default_rng(
                _derived_seed(
                    contract["base_seed"],
                    "acquisition",
                    pool.evaluation_unit,
                    strategy,
                    f"replicate={replicate}",
                    f"round={round_index}",
                )
            ),
            params=params,
        )
        future_label_events = 0
        try:
            selection = ACQUISITION_STRATEGIES[strategy](context)
        except FutureLabelAccessError:
            future_label_events = 1
            raise
        finally:
            audit_rows.append(
                _audit_row(
                    pool=pool,
                    strategy=strategy,
                    replicate=replicate,
                    round_index=round_index,
                    context=context,
                    oracle=oracle,
                    model_record=model_record,
                    plan_hash=plan_hash,
                    future_label_events=future_label_events,
                )
            )
        if len(set(selection.positions)) != pool.batch_size:
            raise ValueError("Acquisition must select exactly the configured batch size.")
        selected_ids = tuple(candidate_ids[position] for position in selection.positions)
        selected_positions = np.asarray(
            [pool_positions[source_id] for source_id in selected_ids], dtype=np.int64
        )
        support_distance_selected = tuple(
            float(1.0 - nearest_to_support[position]) for position in selected_positions
        )
        if oracle is not None and labeled_values is not None:
            labeled_values.update(oracle.reveal(selected_ids, round_index=round_index))
        rounds.append(
            {
                "round_index": round_index,
                "selected_source_ids": selected_ids,
                "selection_positions": selection.positions,
                "selection_scores": selection.scores,
                "score_kind": selection.score_kind,
                "support_distance_selected": support_distance_selected,
                "model": model_record,
            }
        )
        acquired.extend(selected_ids)
        selected_set = set(selected_ids)
        remaining = [source_id for source_id in remaining if source_id not in selected_set]
        nearest_to_support = np.maximum(
            nearest_to_support, similarity[:, selected_positions].max(axis=1)
        )
    return {
        "evaluation_unit": pool.evaluation_unit,
        "strategy": strategy,
        "replicate": int(replicate),
        "campaign_id": campaign_id,
        "rounds": rounds,
    }


def _fit_round_model(
    *,
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    pool: CampaignPool,
    strategy: str,
    round_index: int,
    labeled_ids: tuple[str, ...],
    labeled_values: Mapping[str, float],
    candidate_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Fit and conformally calibrate one round model on acquired labels only."""
    fit_ids, calibration_ids = _calibration_partition(labeled_ids, contract)
    features = prepared["features"]
    position_of = prepared["position_of"]
    estimator_config = UncertaintyEstimatorConfig(
        method=contract["uncertainty_method"],
        coverage=contract["coverage"],
        random_state=_derived_seed(
            contract["base_seed"],
            "uncertainty",
            pool.evaluation_unit,
            strategy,
            f"round={round_index}",
        ),
        **contract["estimator_params"],
    )
    estimator = fit_uncertainty_estimator(
        features[[position_of[source_id] for source_id in fit_ids]],
        np.asarray([labeled_values[source_id] for source_id in fit_ids], dtype=float),
        train_source_ids=fit_ids,
        calibration_features=features[
            [position_of[source_id] for source_id in calibration_ids]
        ],
        calibration_labels=np.asarray(
            [labeled_values[source_id] for source_id in calibration_ids], dtype=float
        ),
        calibration_source_ids=calibration_ids,
        config=estimator_config,
    )
    prediction = estimator.predict(
        features[[position_of[source_id] for source_id in candidate_ids]],
        source_ids=candidate_ids,
    )
    return {
        "point": np.asarray(prediction.point, dtype=np.float64),
        "calibrated_uncertainty": np.asarray(
            prediction.calibrated_uncertainty, dtype=np.float64
        ),
        "interval_lower": np.asarray(prediction.interval_lower, dtype=np.float64),
        "interval_upper": np.asarray(prediction.interval_upper, dtype=np.float64),
        "fit_source_ids": fit_ids,
        "calibration_source_ids": calibration_ids,
        "estimator_audit_hash": estimator.audit.audit_hash,
        "estimator_config_hash": estimator_config.config_hash,
        "prediction_hash": prediction.prediction_hash,
        "calibration_quantile": float(estimator.audit.calibration_quantile),
        "calibration_quantile_level": float(estimator.audit.calibration_quantile_level),
    }


def _calibration_partition(
    labeled_ids: tuple[str, ...], contract: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split acquired labels into a stable conformal fit/calibration partition.

    Each source ID keeps its role for the whole campaign, so calibration rows
    never migrate into the fitting partition as the acquired set grows.
    """
    fraction = contract["calibration_fraction"]
    salt = str(contract["base_seed"])
    calibration = tuple(
        source_id
        for source_id in labeled_ids
        if _stable_uniform(salt, source_id) < fraction
    )
    calibration_set = set(calibration)
    fit = tuple(source_id for source_id in labeled_ids if source_id not in calibration_set)
    minimum = contract["minimum_calibration_rows"]
    if len(calibration) < minimum or len(fit) < minimum:
        raise ValueError(
            "Conformal calibration partition is too small; increase the seed pool or "
            f"adjust calibration_fraction (fit={len(fit)}, calibration={len(calibration)}, "
            f"minimum={minimum})."
        )
    return fit, calibration


def _stable_uniform(salt: str, source_id: str) -> float:
    return int(stable_hash([salt, source_id])[:12], 16) / float(16**12)


def _pool_similarity(pool: CampaignPool, prepared: Mapping[str, Any]) -> np.ndarray:
    """Return exact cosine similarity between every pair of pool reactions.

    Features are binary, so every dot product is an exact integer representable
    in float32 and the matrix is identical regardless of BLAS reduction order.
    """
    features = prepared["features"]
    position_of = prepared["position_of"]
    matrix = features[[position_of[source_id] for source_id in pool.pool_source_ids]]
    counts = matrix.sum(axis=1, dtype=np.float64)
    norms = np.sqrt(np.maximum(counts, 1.0e-12))
    products = np.asarray(matrix @ matrix.T, dtype=np.float64)
    return np.clip(products / np.outer(norms, norms), 0.0, 1.0)


def _audit_row(
    *,
    pool: CampaignPool,
    strategy: str,
    replicate: int,
    round_index: int,
    context: AcquisitionContext,
    oracle: LabelOracle | None,
    model_record: Mapping[str, Any] | None,
    plan_hash: str,
    future_label_events: int,
) -> dict[str, Any]:
    labeled_hash = stable_hash(list(context.labeled_source_ids))
    return {
        "evaluation_unit": pool.evaluation_unit,
        "strategy": strategy,
        "replicate": int(replicate),
        "round_index": int(round_index),
        "rounds": int(pool.rounds),
        "batch_size": int(pool.batch_size),
        "total_budget": pool.total_budget,
        "seed_pool_size": len(pool.seed_source_ids),
        "candidate_pool_size": len(pool.candidate_source_ids),
        "initial_seed_source_id_hash": stable_hash(list(pool.seed_source_ids)),
        "initial_candidate_source_id_hash": stable_hash(list(pool.candidate_source_ids)),
        "labeled_before_count": len(context.labeled_source_ids),
        "labeled_before_id_hash": labeled_hash,
        "candidate_before_count": len(context.candidate_source_ids),
        "candidate_before_id_hash": stable_hash(list(context.candidate_source_ids)),
        "scorer_visible_label_id_hash": labeled_hash,
        "oracle_cumulative_revealed_id_hash": (
            oracle.revealed_source_id_hash if oracle is not None else labeled_hash
        ),
        "oracle_cumulative_revealed_count": (
            len(oracle.revealed_source_ids)
            if oracle is not None
            else len(context.labeled_source_ids)
        ),
        "labels_consumed_by_acquisition": bool(context.consumes_labels),
        "candidate_labeled_overlap": len(
            set(context.candidate_source_ids) & set(context.labeled_source_ids)
        ),
        "future_label_access_events": int(future_label_events),
        "scorer_input_hash": context.scorer_input_hash,
        "model_fit_source_id_hash": (
            stable_hash(list(model_record["fit_source_ids"]))
            if model_record is not None
            else None
        ),
        "model_calibration_source_id_hash": (
            stable_hash(list(model_record["calibration_source_ids"]))
            if model_record is not None
            else None
        ),
        "estimator_audit_hash": (
            model_record["estimator_audit_hash"] if model_record is not None else None
        ),
        "prediction_hash": (
            model_record["prediction_hash"] if model_record is not None else None
        ),
        "plan_hash": plan_hash,
    }


def _score_all_campaigns(
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    campaigns: Mapping[str, Any],
    plan_hash: str,
    gate: EvaluationOracleGate,
) -> dict[str, Any]:
    """Score completed campaigns after unlocking the pool-outcome gate."""
    gate.unlock()
    acquisition_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    pool_by_unit = {pool.evaluation_unit: pool for pool in prepared["pools"]}
    outcomes_by_unit = {
        pool.evaluation_unit: _load_pool_outcomes(contract, prepared, pool, gate)
        for pool in prepared["pools"]
    }
    for record in campaigns["records"]:
        pool = pool_by_unit[record["evaluation_unit"]]
        trajectory_rows.extend(
            _campaign_trajectory(
                contract=contract,
                pool=pool,
                outcomes=outcomes_by_unit[pool.evaluation_unit],
                record=record,
                plan_hash=plan_hash,
                acquisition_rows=acquisition_rows,
                uncertainty_rows=uncertainty_rows,
            )
        )
    return {
        "acquisition_rows": acquisition_rows,
        "uncertainty_rows": uncertainty_rows,
        "trajectory_rows": trajectory_rows,
        "pool_outcomes_loaded_after_campaigns": True,
    }


def _load_pool_outcomes(
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    pool: CampaignPool,
    gate: EvaluationOracleGate,
) -> dict[str, Any]:
    """Load pool-wide outcomes for evaluation only, after every campaign ends."""
    gate.require_unlocked()
    values = read_allowed_outcomes(
        contract["dataset_path"], prepared["canonical_source_ids"], pool.pool_source_ids
    )
    rows = prepared["identity"].loc[list(pool.pool_source_ids)]
    yields = np.asarray([values[source_id] for source_id in pool.pool_source_ids], float)
    order = np.argsort(-yields, kind="stable")
    top_k = min(contract["top_k"], len(pool.pool_source_ids))
    top_mask = np.zeros(len(pool.pool_source_ids), dtype=bool)
    top_mask[order[:top_k]] = True
    substrate_codes, substrate_total = _codes(rows["canonical_substrate_key"])
    condition_codes, condition_total = _codes(rows["canonical_condition_key"])
    product_codes, _ = _codes(rows["canonical_product_key"])
    return {
        "position_of": {
            source_id: index for index, source_id in enumerate(pool.pool_source_ids)
        },
        "values": values,
        "yields": yields,
        "pool_max": float(yields.max()),
        "top_k": int(top_k),
        "top_mask": top_mask,
        "substrate_codes": substrate_codes,
        "condition_codes": condition_codes,
        "product_codes": product_codes,
        "substrate_key_total": int(substrate_total),
        "condition_key_total": int(condition_total),
        "reaction_keys": rows["canonical_reaction_key"].astype(str).to_numpy(),
        "substrate_keys": rows["canonical_substrate_key"].astype(str).to_numpy(),
        "condition_keys": rows["canonical_condition_key"].astype(str).to_numpy(),
    }


def _codes(values: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    return np.asarray(codes, dtype=np.int64), len(uniques)


def _campaign_trajectory(
    *,
    contract: Mapping[str, Any],
    pool: CampaignPool,
    outcomes: Mapping[str, Any],
    record: Mapping[str, Any],
    plan_hash: str,
    acquisition_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the complete round-by-round outcome trajectory for one campaign."""
    position_of = outcomes["position_of"]
    strategy = record["strategy"]
    replicate = int(record["replicate"])
    seed_ids = frozenset(pool.seed_source_ids)
    acquired = [position_of[source_id] for source_id in pool.seed_source_ids]
    budget_positions: list[int] = []
    rows = [
        _trajectory_row(
            contract=contract,
            pool=pool,
            outcomes=outcomes,
            strategy=strategy,
            replicate=replicate,
            round_index=0,
            acquired=acquired,
            budget_positions=budget_positions,
            batch_positions=[],
            batch_support_distance=(),
            batch_scores=(),
            model=None,
            batch_selection_positions=(),
            selected_source_ids=(),
            plan_hash=plan_hash,
        )
    ]
    for round_record in record["rounds"]:
        selected_ids = tuple(round_record["selected_source_ids"])
        batch_positions = [position_of[source_id] for source_id in selected_ids]
        acquired = acquired + batch_positions
        budget_positions = budget_positions + batch_positions
        if round_record["model"] is not None:
            _extend_model_rows(
                acquisition_rows=acquisition_rows,
                uncertainty_rows=uncertainty_rows,
                contract=contract,
                pool=pool,
                outcomes=outcomes,
                strategy=strategy,
                round_record=round_record,
                batch_positions=batch_positions,
                seed_ids=seed_ids,
                plan_hash=plan_hash,
            )
        rows.append(
            _trajectory_row(
                contract=contract,
                pool=pool,
                outcomes=outcomes,
                strategy=strategy,
                replicate=replicate,
                round_index=int(round_record["round_index"]),
                acquired=acquired,
                budget_positions=budget_positions,
                batch_positions=batch_positions,
                batch_support_distance=tuple(round_record["support_distance_selected"]),
                batch_scores=tuple(round_record["selection_scores"]),
                model=round_record["model"],
                batch_selection_positions=tuple(round_record["selection_positions"]),
                selected_source_ids=selected_ids,
                plan_hash=plan_hash,
            )
        )
    return rows


def _extend_model_rows(
    *,
    acquisition_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
    contract: Mapping[str, Any],
    pool: CampaignPool,
    outcomes: Mapping[str, Any],
    strategy: str,
    round_record: Mapping[str, Any],
    batch_positions: Sequence[int],
    seed_ids: frozenset[str],
    plan_hash: str,
) -> None:
    """Record per-acquisition detail and calibrated-uncertainty diagnostics."""
    model = round_record["model"]
    selection = list(round_record["selection_positions"])
    selected_ids = list(round_record["selected_source_ids"])
    measured = outcomes["yields"][np.asarray(batch_positions, dtype=np.int64)]
    point = np.asarray(model["point"], dtype=float)[selection]
    lower = np.asarray(model["interval_lower"], dtype=float)[selection]
    upper = np.asarray(model["interval_upper"], dtype=float)[selection]
    uncertainty = np.asarray(model["calibrated_uncertainty"], dtype=float)[selection]
    for rank, source_id in enumerate(selected_ids):
        pool_position = batch_positions[rank]
        acquisition_rows.append(
            {
                "evaluation_unit": pool.evaluation_unit,
                "strategy": strategy,
                "round_index": int(round_record["round_index"]),
                "rank_in_batch": rank,
                "source_row_id": source_id,
                "canonical_reaction_key": str(outcomes["reaction_keys"][pool_position]),
                "canonical_substrate_key": str(outcomes["substrate_keys"][pool_position]),
                "canonical_condition_key": str(outcomes["condition_keys"][pool_position]),
                "score_kind": round_record["score_kind"],
                "acquisition_score": float(round_record["selection_scores"][rank]),
                "predicted_yield": float(point[rank]),
                "calibrated_uncertainty": float(uncertainty[rank]),
                "interval_lower": float(lower[rank]),
                "interval_upper": float(upper[rank]),
                "support_distance_before": float(
                    round_record["support_distance_selected"][rank]
                ),
                "measured_yield": float(measured[rank]),
                "measured_within_interval": bool(
                    lower[rank] <= measured[rank] <= upper[rank]
                ),
                "acquired_reaction_in_seed_pool": source_id in seed_ids,
                "plan_hash": plan_hash,
            }
        )
    calibration = interval_calibration_metrics(
        measured, point, lower, upper, uncertainty, nominal_coverage=contract["coverage"]
    )
    candidate_uncertainty = np.asarray(model["calibrated_uncertainty"], dtype=float)
    uncertainty_rows.append(
        {
            "evaluation_unit": pool.evaluation_unit,
            "strategy": strategy,
            "round_index": int(round_record["round_index"]),
            "uncertainty_method": contract["uncertainty_method"],
            "model_fit_rows": len(model["fit_source_ids"]),
            "model_calibration_rows": len(model["calibration_source_ids"]),
            "candidate_count": len(candidate_uncertainty),
            "conformal_quantile": float(model["calibration_quantile"]),
            "conformal_quantile_level": float(model["calibration_quantile_level"]),
            "mean_candidate_uncertainty": float(candidate_uncertainty.mean()),
            "median_candidate_uncertainty": float(np.median(candidate_uncertainty)),
            "mean_selected_uncertainty": float(uncertainty.mean()),
            "mean_selected_prediction": float(point.mean()),
            "nominal_coverage": float(calibration.nominal_coverage),
            "batch_empirical_coverage": float(calibration.empirical_coverage),
            "batch_calibration_error": float(calibration.calibration_error),
            "batch_mean_interval_width": float(calibration.mean_interval_width),
            "batch_uncertainty_error_spearman": (
                None
                if calibration.uncertainty_error_spearman is None
                else float(calibration.uncertainty_error_spearman)
            ),
            "batch_prediction_rmse": float(rmse(measured, point)),
            "batch_prediction_spearman": _finite_or_none(
                float(spearman_corr(measured, point))
            ),
            "estimator_audit_hash": model["estimator_audit_hash"],
            "estimator_config_hash": model["estimator_config_hash"],
            "prediction_hash": model["prediction_hash"],
            "plan_hash": plan_hash,
        }
    )


def _trajectory_row(
    *,
    contract: Mapping[str, Any],
    pool: CampaignPool,
    outcomes: Mapping[str, Any],
    strategy: str,
    replicate: int,
    round_index: int,
    acquired: Sequence[int],
    budget_positions: Sequence[int],
    batch_positions: Sequence[int],
    batch_support_distance: Sequence[float],
    batch_scores: Sequence[float],
    model: Mapping[str, Any] | None,
    batch_selection_positions: Sequence[int],
    selected_source_ids: Sequence[str],
    plan_hash: str,
) -> dict[str, Any]:
    acquired_index = np.asarray(acquired, dtype=np.int64)
    yields = outcomes["yields"]
    acquired_values = yields[acquired_index]
    budget_values = yields[np.asarray(budget_positions, dtype=np.int64)]
    batch_values = yields[np.asarray(batch_positions, dtype=np.int64)]
    best = float(acquired_values.max())
    hits = int((acquired_values >= contract["high_yield_threshold"]).sum())
    top_hits = int(outcomes["top_mask"][acquired_index].sum())
    substrates = int(len(np.unique(outcomes["substrate_codes"][acquired_index])))
    conditions = int(len(np.unique(outcomes["condition_codes"][acquired_index])))
    products = int(len(np.unique(outcomes["product_codes"][acquired_index])))
    selected_uncertainty: float | None = None
    batch_rmse: float | None = None
    batch_spearman: float | None = None
    if model is not None and len(batch_positions):
        selection = list(batch_selection_positions)
        point = np.asarray(model["point"], dtype=float)[selection]
        uncertainty = np.asarray(model["calibrated_uncertainty"], dtype=float)[selection]
        selected_uncertainty = float(uncertainty.mean())
        batch_rmse = float(rmse(batch_values, point))
        batch_spearman = _finite_or_none(float(spearman_corr(batch_values, point)))
    return {
        "evaluation_unit": pool.evaluation_unit,
        "strategy": strategy,
        "replicate": int(replicate),
        "round_index": int(round_index),
        "acquired_total": len(acquired_index),
        "budget_spent": len(budget_positions),
        "pool_size": len(pool.pool_source_ids),
        "pool_best_yield": float(outcomes["pool_max"]),
        "best_yield_discovered": best,
        "simple_regret": float(outcomes["pool_max"] - best),
        "cumulative_regret": (
            float(np.sum(outcomes["pool_max"] - budget_values))
            if len(budget_values)
            else 0.0
        ),
        "cumulative_mean_acquired_yield": (
            float(budget_values.mean()) if len(budget_values) else None
        ),
        "batch_mean_yield": float(batch_values.mean()) if len(batch_values) else None,
        "batch_max_yield": float(batch_values.max()) if len(batch_values) else None,
        "top_k": int(outcomes["top_k"]),
        "top_k_hits": top_hits,
        "top_k_recall": float(top_hits / outcomes["top_k"]),
        "high_yield_threshold": float(contract["high_yield_threshold"]),
        "high_yield_hit_count": hits,
        "high_yield_hit_rate": float(hits / len(acquired_index)),
        "unique_substrate_keys": substrates,
        "unique_condition_keys": conditions,
        "unique_product_keys": products,
        "substrate_coverage_fraction": float(
            substrates / outcomes["substrate_key_total"]
        ),
        "condition_coverage_fraction": float(
            conditions / outcomes["condition_key_total"]
        ),
        "batch_mean_support_distance": (
            float(np.mean(batch_support_distance)) if len(batch_support_distance) else None
        ),
        "batch_max_support_distance": (
            float(np.max(batch_support_distance)) if len(batch_support_distance) else None
        ),
        "mean_candidate_uncertainty": (
            float(np.mean(model["calibrated_uncertainty"])) if model is not None else None
        ),
        "mean_selected_uncertainty": selected_uncertainty,
        "batch_prediction_rmse": batch_rmse,
        "batch_prediction_spearman": batch_spearman,
        "mean_acquisition_score": (
            float(np.mean(batch_scores)) if len(batch_scores) else None
        ),
        "selected_source_id_hash": stable_hash(sorted(selected_source_ids)),
        "acquired_source_id_hash": stable_hash(
            sorted(pool.pool_source_ids[int(position)] for position in acquired_index)
        ),
        "plan_hash": plan_hash,
    }


def _random_distribution(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Summarize the complete random-replicate distribution per unit and round."""
    random_rows = trajectory.loc[trajectory["strategy"].eq(RANDOM_STRATEGY)]
    records: list[dict[str, Any]] = []
    for (unit, round_index), group in random_rows.groupby(
        ["evaluation_unit", "round_index"], sort=True
    ):
        for metric in PERCENTILE_METRICS:
            finite = _finite_values(group[metric])
            base = {
                "evaluation_unit": unit,
                "round_index": int(round_index),
                "metric": metric,
                "higher_is_better": _HIGHER_IS_BETTER[metric],
                "replicate_count": int(len(finite)),
            }
            statistics = (
                "mean",
                "std",
                "minimum",
                "p05",
                "p25",
                "median",
                "p75",
                "p95",
                "maximum",
            )
            if len(finite) == 0:
                records.append({**base, **{name: None for name in statistics}})
                continue
            records.append(
                {
                    **base,
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                    "minimum": float(np.min(finite)),
                    "p05": float(np.quantile(finite, 0.05)),
                    "p25": float(np.quantile(finite, 0.25)),
                    "median": float(np.quantile(finite, 0.50)),
                    "p75": float(np.quantile(finite, 0.75)),
                    "p95": float(np.quantile(finite, 0.95)),
                    "maximum": float(np.max(finite)),
                }
            )
    return pd.DataFrame(records)


def _strategy_percentiles(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Place every informed strategy inside the random-replicate distribution."""
    random_rows = trajectory.loc[trajectory["strategy"].eq(RANDOM_STRATEGY)]
    informed_rows = trajectory.loc[trajectory["strategy"].ne(RANDOM_STRATEGY)]
    reference: dict[tuple[str, int, str], np.ndarray] = {}
    for (unit, round_index), group in random_rows.groupby(
        ["evaluation_unit", "round_index"], sort=True
    ):
        for metric in PERCENTILE_METRICS:
            reference[(unit, int(round_index), metric)] = _finite_values(group[metric])
    records: list[dict[str, Any]] = []
    ordered = informed_rows.sort_values(
        ["evaluation_unit", "strategy", "round_index"], kind="stable"
    )
    for row in ordered.to_dict("records"):
        for metric in PERCENTILE_METRICS:
            observed = _numeric_or_nan(row[metric])
            sample = reference.get(
                (row["evaluation_unit"], int(row["round_index"]), metric),
                np.asarray([], dtype=float),
            )
            percentile: float | None = None
            advantage: float | None = None
            z_score: float | None = None
            reference_mean: float | None = None
            if math.isfinite(observed) and len(sample):
                percentile = float(
                    100.0
                    * (np.sum(sample < observed) + 0.5 * np.sum(sample == observed))
                    / len(sample)
                )
                reference_mean = float(np.mean(sample))
                advantage = float(observed - reference_mean)
                deviation = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
                z_score = float(advantage / deviation) if deviation > 0.0 else None
            records.append(
                {
                    "evaluation_unit": row["evaluation_unit"],
                    "strategy": row["strategy"],
                    "round_index": int(row["round_index"]),
                    "metric": metric,
                    "higher_is_better": _HIGHER_IS_BETTER[metric],
                    "value": float(observed) if math.isfinite(observed) else None,
                    "random_reference_mean": reference_mean,
                    "advantage_over_random_mean": advantage,
                    "random_percentile": percentile,
                    "random_z_score": z_score,
                    "random_replicate_count": int(len(sample)),
                }
            )
    return pd.DataFrame(records)


def _summarize(
    trajectory: pd.DataFrame, percentiles: pd.DataFrame, contract: Mapping[str, Any]
) -> pd.DataFrame:
    """Aggregate final-round outcomes across evaluation units."""
    final_round = int(contract["rounds"])
    final = trajectory.loc[trajectory["round_index"].eq(final_round)]
    final_percentiles = percentiles.loc[percentiles["round_index"].eq(final_round)]
    records: list[dict[str, Any]] = []
    for metric in PERCENTILE_METRICS:
        random_rows = final.loc[final["strategy"].eq(RANDOM_STRATEGY)]
        unit_means = (
            random_rows.assign(
                _value=pd.to_numeric(random_rows[metric], errors="coerce")
            )
            .groupby("evaluation_unit")["_value"]
            .mean()
        )
        records.append(
            {
                "strategy": RANDOM_STRATEGY,
                "metric": metric,
                "higher_is_better": _HIGHER_IS_BETTER[metric],
                "final_round": final_round,
                "evaluation_units": int(unit_means.notna().sum()),
                "mean_value": _finite_or_none(float(unit_means.mean())),
                "std_value": (
                    _finite_or_none(float(unit_means.std(ddof=1)))
                    if len(unit_means) > 1
                    else 0.0
                ),
                "minimum_value": _finite_or_none(float(unit_means.min())),
                "maximum_value": _finite_or_none(float(unit_means.max())),
                "random_reference_mean": _finite_or_none(float(unit_means.mean())),
                "mean_advantage_over_random": 0.0,
                "mean_random_percentile": 50.0,
                "minimum_random_percentile": 50.0,
                "maximum_random_percentile": 50.0,
                "units_at_or_above_95th_percentile": 0,
                "units_at_or_below_5th_percentile": 0,
            }
        )
        for strategy in INFORMED_STRATEGIES:
            subset = final_percentiles.loc[
                final_percentiles["strategy"].eq(strategy)
                & final_percentiles["metric"].eq(metric)
            ]
            values = pd.to_numeric(subset["value"], errors="coerce")
            percentile_values = pd.to_numeric(subset["random_percentile"], errors="coerce")
            advantage = pd.to_numeric(subset["advantage_over_random_mean"], errors="coerce")
            reference = pd.to_numeric(subset["random_reference_mean"], errors="coerce")
            records.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "higher_is_better": _HIGHER_IS_BETTER[metric],
                    "final_round": final_round,
                    "evaluation_units": int(values.notna().sum()),
                    "mean_value": _finite_or_none(float(values.mean())),
                    "std_value": (
                        _finite_or_none(float(values.std(ddof=1)))
                        if len(values) > 1
                        else 0.0
                    ),
                    "minimum_value": _finite_or_none(float(values.min())),
                    "maximum_value": _finite_or_none(float(values.max())),
                    "random_reference_mean": _finite_or_none(float(reference.mean())),
                    "mean_advantage_over_random": _finite_or_none(float(advantage.mean())),
                    "mean_random_percentile": _finite_or_none(
                        float(percentile_values.mean())
                    ),
                    "minimum_random_percentile": _finite_or_none(
                        float(percentile_values.min())
                    ),
                    "maximum_random_percentile": _finite_or_none(
                        float(percentile_values.max())
                    ),
                    "units_at_or_above_95th_percentile": int(
                        (percentile_values >= 95.0).sum()
                    ),
                    "units_at_or_below_5th_percentile": int(
                        (percentile_values <= 5.0).sum()
                    ),
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values(["metric", "strategy"], kind="stable")
        .reset_index(drop=True)
    )


def _build_manifest(
    *,
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    plan: Mapping[str, Any],
    campaigns: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    trajectory: pd.DataFrame,
    distribution: pd.DataFrame,
    percentiles: pd.DataFrame,
    summary: pd.DataFrame,
    config: Any,
    output: Path,
) -> dict[str, Any]:
    audit = pd.DataFrame(campaigns["audit_rows"])
    manifest = {
        "schema_version": RECOMMENDATION_SIMULATION_SCHEMA_VERSION,
        "status": "corrected_revalidation_complete",
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": prepared["dataset_hash"],
        "canonical_split_hash": prepared["canonical_split_hash"],
        "feature_metadata_hash": prepared["feature_contract"]["feature_metadata_hash"],
        "config_hash": plan["config_hash"],
        "plan_hash": plan["plan_hash"],
        "resolved_scientific_config": plan["resolved_scientific_config"],
        "dependency_versions": _dependency_versions(),
        "command": _command_record(config, output),
        "evaluation_unit_count": len(prepared["units"]),
        "strategy_count": len(ALL_STRATEGIES),
        "informed_strategy_count": len(INFORMED_STRATEGIES),
        "random_replicate_count": contract["random_replicates"],
        "rounds": contract["rounds"],
        "batch_size": contract["batch_size"],
        "seed_pool_size": contract["seed_pool_size"],
        "total_budget": contract["rounds"] * contract["batch_size"],
        "campaign_count": len(campaigns["records"]),
        "campaign_pool_count": len(prepared["pools"]),
        "acquisition_audit_row_count": len(audit),
        "acquisition_row_count": len(outcomes["acquisition_rows"]),
        "uncertainty_diagnostic_row_count": len(outcomes["uncertainty_rows"]),
        "round_trajectory_row_count": len(trajectory),
        "random_distribution_row_count": len(distribution),
        "strategy_percentile_row_count": len(percentiles),
        "summary_row_count": len(summary),
        "uncertainty_method": contract["uncertainty_method"],
        "surviving_uncertainty_methods": list(PHASE12_SURVIVING_UNCERTAINTY_METHODS),
        "acquisition_protocol": ACQUISITION_PROTOCOL,
        "future_label_access_events": int(audit["future_label_access_events"].sum()),
        "pool_outcomes_loaded_after_campaigns": bool(
            outcomes["pool_outcomes_loaded_after_campaigns"]
        ),
        "supervised_autoencoder_used": False,
        "prospective_validation_claimed": False,
        "output_hashes": {name: sha256_file(output / name) for name in _OUTPUTS},
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def _scientific_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(contract["dataset_path"]),
        "canonical_split_directory": str(contract["canonical_split_directory"]),
        "random_seeds": list(contract["random_seeds"]),
        "pool_train_fraction": contract["pool_train_fraction"],
        "feature_config": dict(contract["feature_config"]),
        "seed_pool_size": contract["seed_pool_size"],
        "rounds": contract["rounds"],
        "batch_size": contract["batch_size"],
        "random_replicates": contract["random_replicates"],
        "top_k": contract["top_k"],
        "high_yield_threshold": contract["high_yield_threshold"],
        "uncertainty_method": contract["uncertainty_method"],
        "coverage": contract["coverage"],
        "calibration_fraction": contract["calibration_fraction"],
        "minimum_calibration_rows": contract["minimum_calibration_rows"],
        "estimator_params": dict(contract["estimator_params"]),
        "ucb_beta": contract["ucb_beta"],
        "expected_improvement_xi": contract["expected_improvement_xi"],
        "diversity_penalty": contract["diversity_penalty"],
        "support_distance_yield_weight": contract["support_distance_yield_weight"],
        "base_seed": contract["base_seed"],
    }


def _resolve_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "dataset",
            "splits",
            "features",
            "campaign",
            "model",
            "strategies",
            "base_seed",
            "output",
        },
        "config",
    )
    dataset = _mapping(raw["dataset"], "dataset")
    _exact_keys(dataset, {"path"}, "dataset")
    splits = _mapping(raw["splits"], "splits")
    _exact_keys(
        splits, {"canonical_directory", "random_seeds", "pool_train_fraction"}, "splits"
    )
    campaign = _mapping(raw["campaign"], "campaign")
    _exact_keys(
        campaign,
        {
            "seed_pool_size",
            "rounds",
            "batch_size",
            "random_replicates",
            "top_k",
            "high_yield_threshold",
        },
        "campaign",
    )
    model = _mapping(raw["model"], "model")
    _exact_keys(
        model,
        {
            "uncertainty_method",
            "coverage",
            "calibration_fraction",
            "minimum_calibration_rows",
            "estimator_params",
        },
        "model",
    )
    strategies = _mapping(raw["strategies"], "strategies")
    _exact_keys(
        strategies,
        {
            "ucb_beta",
            "expected_improvement_xi",
            "diversity_penalty",
            "support_distance_yield_weight",
        },
        "strategies",
    )
    output = _mapping(raw["output"], "output")
    _exact_keys(output, {"directory"}, "output")

    method = str(model["uncertainty_method"])
    if method not in PHASE12_SURVIVING_UNCERTAINTY_METHODS:
        raise ValueError(
            "model.uncertainty_method must be an estimator that survived Phase 12 "
            f"calibrated selection: {list(PHASE12_SURVIVING_UNCERTAINTY_METHODS)}."
        )
    replicates = _positive_int(campaign["random_replicates"], "campaign.random_replicates")
    if replicates < 200:
        raise ValueError(
            "campaign.random_replicates must be at least 200 so the random baseline "
            "reports a complete distribution."
        )
    return {
        "dataset_path": Path(str(dataset["path"])),
        "canonical_split_directory": Path(str(splits["canonical_directory"])),
        "random_seeds": _unique_ints(splits["random_seeds"], "splits.random_seeds"),
        "pool_train_fraction": _fraction(
            splits["pool_train_fraction"], "splits.pool_train_fraction"
        ),
        "feature_config": resolve_corrected_feature_config(
            _mapping(raw["features"], "features"), required_kind="bh_role_separated"
        ),
        "seed_pool_size": _positive_int(campaign["seed_pool_size"], "campaign.seed_pool_size"),
        "rounds": _positive_int(campaign["rounds"], "campaign.rounds"),
        "batch_size": _positive_int(campaign["batch_size"], "campaign.batch_size"),
        "random_replicates": replicates,
        "top_k": _positive_int(campaign["top_k"], "campaign.top_k"),
        "high_yield_threshold": _nonnegative_float(
            campaign["high_yield_threshold"], "campaign.high_yield_threshold"
        ),
        "uncertainty_method": method,
        "coverage": _open_unit(model["coverage"], "model.coverage"),
        "calibration_fraction": _open_unit(
            model["calibration_fraction"], "model.calibration_fraction"
        ),
        "minimum_calibration_rows": _positive_int(
            model["minimum_calibration_rows"], "model.minimum_calibration_rows"
        ),
        "estimator_params": _estimator_params(model["estimator_params"], method),
        "ucb_beta": _nonnegative_float(strategies["ucb_beta"], "strategies.ucb_beta"),
        "expected_improvement_xi": _nonnegative_float(
            strategies["expected_improvement_xi"], "strategies.expected_improvement_xi"
        ),
        "diversity_penalty": _nonnegative_float(
            strategies["diversity_penalty"], "strategies.diversity_penalty"
        ),
        "support_distance_yield_weight": _nonnegative_float(
            strategies["support_distance_yield_weight"],
            "strategies.support_distance_yield_weight",
        ),
        "base_seed": _int(raw["base_seed"], "base_seed"),
        "output_directory": Path(str(output["directory"])),
    }


def _estimator_params(value: Any, method: str) -> dict[str, Any]:
    params = _mapping(value, "model.estimator_params")
    allowed = {
        "bootstrap_extra_trees": {"bootstrap_members", "bootstrap_trees_per_member"},
        "heterogeneous_disagreement": {"ridge_alpha", "heterogeneous_tree_estimators"},
    }[method]
    _exact_keys(params, allowed, "model.estimator_params")
    resolved: dict[str, Any] = {}
    for name in sorted(allowed):
        if name == "ridge_alpha":
            resolved[name] = _positive_float(params[name], f"model.estimator_params.{name}")
        else:
            resolved[name] = _positive_int(params[name], f"model.estimator_params.{name}")
    return resolved


def _contract_from_scientific(scientific: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "dataset_path": Path(scientific["dataset_path"]),
        "canonical_split_directory": Path(scientific["canonical_split_directory"]),
        "random_seeds": tuple(int(value) for value in scientific["random_seeds"]),
        "pool_train_fraction": float(scientific["pool_train_fraction"]),
        "feature_config": dict(scientific["feature_config"]),
        "seed_pool_size": int(scientific["seed_pool_size"]),
        "rounds": int(scientific["rounds"]),
        "batch_size": int(scientific["batch_size"]),
        "random_replicates": int(scientific["random_replicates"]),
        "top_k": int(scientific["top_k"]),
        "high_yield_threshold": float(scientific["high_yield_threshold"]),
        "uncertainty_method": str(scientific["uncertainty_method"]),
        "coverage": float(scientific["coverage"]),
        "calibration_fraction": float(scientific["calibration_fraction"]),
        "minimum_calibration_rows": int(scientific["minimum_calibration_rows"]),
        "estimator_params": dict(scientific["estimator_params"]),
        "ucb_beta": float(scientific["ucb_beta"]),
        "expected_improvement_xi": float(scientific["expected_improvement_xi"]),
        "diversity_penalty": float(scientific["diversity_penalty"]),
        "support_distance_yield_weight": float(scientific["support_distance_yield_weight"]),
        "base_seed": int(scientific["base_seed"]),
        "output_directory": output,
    }


def _load_config(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Recommendation simulation config must be a mapping.")
    return loaded


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys mismatch: expected={sorted(expected)}, observed={sorted(value)}."
        )


def _unique_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    result = tuple(_int(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values.")
    return result


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _open_unit(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")
    return result


def _fraction(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return result


def _derived_seed(base_seed: int, *parts: str) -> int:
    return int(base_seed) + int(stable_hash(list(parts))[:12], 16) % 1_000_000


def _finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _numeric_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _optional_array_hash(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    return stable_hash([float(value) for value in np.asarray(values, dtype=float)])


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonable(record) for record in frame.to_dict("records")]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records_close(
    expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> bool:
    if len(expected) != len(observed):
        return False
    return all(
        _values_close(_jsonable(left), _jsonable(right))
        for left, right in zip(expected, observed, strict=True)
    )


def _values_close(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)
    return left == right


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dependency in ("numpy", "pandas", "scikit-learn", "scipy", "rdkit"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = "not-installed"
    return versions


def _git_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Scientific runs require an available Git commit.") from exc
    if not value:
        raise ValueError("Scientific runs require a nonempty Git commit.")
    return value


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Unable to determine Git worktree state.") from exc


def _command_record(config: Any, output: Path) -> str:
    return (
        "python -B scripts/run_recommendation_simulation.py "
        f"--config {config} --output-directory {output}"
    )
