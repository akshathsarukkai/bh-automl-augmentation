"""Leakage-resistant protocol primitives for policy search and final evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.utils.corrected_runs import stable_hash

FROZEN_POLICY_SCHEMA_VERSION = "bh-frozen-policy-v1"
_BINDING_FIELDS = (
    "dataset_hash",
    "split_hash",
    "split_aggregate_hash",
    "per_seed_split_hash",
    "source_id_split_hash",
    "canonicalization_version",
    "split_schema_version",
    "feature_metadata_hash",
    "config_hash",
    "commit_hash",
)
_FROZEN_FIELDS = {
    "schema_version",
    "status",
    "binding",
    "resolved_policy",
    "resolved_policy_hash",
    "policy_id",
    "training_protocol",
    "selection_metric",
    "selection_value",
    "lower_is_better",
    "search_manifest_hash",
    "frozen_policy_hash",
}


@dataclass(frozen=True)
class ScientificBinding:
    """Hashes and versions that a frozen policy is scientifically bound to."""

    dataset_hash: str
    split_hash: str
    split_aggregate_hash: str
    per_seed_split_hash: str
    source_id_split_hash: str
    canonicalization_version: str
    split_schema_version: str
    feature_metadata_hash: str
    config_hash: str
    commit_hash: str

    def __post_init__(self) -> None:
        for field in _BINDING_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Scientific binding field {field} must be a non-empty string.")

    def to_dict(self) -> dict[str, str]:
        """Serialize the exact scientific binding schema."""
        return {field: getattr(self, field) for field in _BINDING_FIELDS}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScientificBinding:
        """Parse a strict binding and reject missing or unknown fields."""
        if not isinstance(value, Mapping):
            raise ValueError("Scientific binding must be a mapping.")
        if set(value) != set(_BINDING_FIELDS):
            raise ValueError(
                "Scientific binding schema mismatch: "
                f"expected {sorted(_BINDING_FIELDS)}, observed {sorted(value)}."
            )
        return cls(**{field: value[field] for field in _BINDING_FIELDS})


@dataclass(frozen=True)
class PolicySearchInputs:
    """The only labeled partitions visible to validation policy search."""

    train: Any
    validation: Any
    binding: ScientificBinding


@dataclass(frozen=True)
class ResolvedPolicy:
    """One stable policy identifier and its complete resolved configuration."""

    policy_id: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("Resolved policy_id must be a non-empty string.")
        object.__setattr__(self, "config", _canonical_mapping(self.config, "resolved policy"))

    @property
    def policy_hash(self) -> str:
        """Hash the policy ID and resolved configuration."""
        return stable_hash({"policy_id": self.policy_id, "config": self.config})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved policy."""
        return {"policy_id": self.policy_id, "config": dict(self.config)}


@dataclass(frozen=True)
class PolicySearchResult:
    """Validation-only metrics and the deterministically selected policy."""

    inputs: PolicySearchInputs
    search_metrics: pd.DataFrame
    selected_policy: ResolvedPolicy
    selection_metric: str
    selection_value: float
    lower_is_better: bool


@dataclass(frozen=True)
class FrozenPolicyEnvelope:
    """Hash-verified policy and provenance frozen before outer-test evaluation."""

    binding: ScientificBinding
    resolved_policy: ResolvedPolicy
    training_protocol: Mapping[str, Any]
    selection_metric: str
    selection_value: float
    lower_is_better: bool
    search_manifest_hash: str
    frozen_policy_hash: str
    schema_version: str = FROZEN_POLICY_SCHEMA_VERSION
    status: str = "frozen"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete immutable envelope."""
        return {
            **self._unhashed_payload(),
            "frozen_policy_hash": self.frozen_policy_hash,
        }

    def write_json(self, path: str | Path) -> Path:
        """Write stable human-readable frozen policy JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return output

    def verify(self, *, expected_binding: ScientificBinding | None = None) -> None:
        """Recompute hashes and optionally verify the runtime scientific binding."""
        if self.schema_version != FROZEN_POLICY_SCHEMA_VERSION or self.status != "frozen":
            raise ValueError("Final evaluation requires a policy with status 'frozen'.")
        if expected_binding is not None and self.binding != expected_binding:
            raise ValueError("Frozen policy scientific binding mismatch.")
        if not math.isfinite(self.selection_value):
            raise ValueError("Frozen policy selection value must be finite.")
        expected_hash = stable_hash(self._unhashed_payload())
        if self.frozen_policy_hash != expected_hash:
            raise ValueError("Frozen policy envelope hash mismatch.")

    def _unhashed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "binding": self.binding.to_dict(),
            "resolved_policy": dict(self.resolved_policy.config),
            "resolved_policy_hash": self.resolved_policy.policy_hash,
            "policy_id": self.resolved_policy.policy_id,
            "training_protocol": dict(self.training_protocol),
            "selection_metric": self.selection_metric,
            "selection_value": self.selection_value,
            "lower_is_better": self.lower_is_better,
            "search_manifest_hash": self.search_manifest_hash,
        }


@dataclass(frozen=True)
class FinalEvaluationInputs:
    """Refit and outer-test data, available only to final evaluation."""

    refit: Any
    outer_test: Any
    binding: ScientificBinding
    evaluation_unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_unit, str) or not self.evaluation_unit.strip():
            raise ValueError("Final evaluation_unit must be a non-empty string.")


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Validated outer-test metrics from one unique evaluation unit."""

    evaluation_unit: str
    metrics: pd.DataFrame
    frozen_policy_hash: str


ValidationEvaluator = Callable[
    [PolicySearchInputs, ResolvedPolicy],
    pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any],
]
OuterTestEvaluator = Callable[
    [FinalEvaluationInputs, FrozenPolicyEnvelope],
    pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any],
]


def run_policy_search(
    inputs: PolicySearchInputs,
    policies: Sequence[ResolvedPolicy],
    validation_evaluator: ValidationEvaluator,
    *,
    selection_metric: str = "rmse",
    lower_is_better: bool = True,
) -> PolicySearchResult:
    """Evaluate policies on validation only and select with a stable tie-break."""
    if not isinstance(selection_metric, str) or not selection_metric.strip():
        raise ValueError("Policy search selection_metric must be a non-empty string.")
    if not policies:
        raise ValueError("Policy search requires at least one resolved policy.")
    policy_ids = [policy.policy_id for policy in policies]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("Policy search policy_id values must be unique.")

    frames: list[pd.DataFrame] = []
    by_id = {policy.policy_id: policy for policy in policies}
    for policy in policies:
        frame = _normalize_metric_rows(validation_evaluator(inputs, policy))
        if not frame["split"].eq("valid").all():
            raise ValueError("Policy search evaluator returned a non-validation metric row.")
        if "policy_id" in frame and not frame["policy_id"].eq(policy.policy_id).all():
            raise ValueError("Policy search evaluator returned a mismatched policy_id.")
        frame["policy_id"] = policy.policy_id
        frame["policy_hash"] = policy.policy_hash
        frame["selected_policy"] = False
        frames.append(frame)

    search_metrics = pd.concat(frames, ignore_index=True)
    candidates = search_metrics.loc[search_metrics["metric"].eq(selection_metric)].copy()
    if candidates.empty:
        raise ValueError(f"No validation rows exist for selection metric {selection_metric!r}.")
    candidates = candidates.sort_values(
        ["value", "policy_hash", "policy_id"],
        ascending=[lower_is_better, True, True],
        kind="mergesort",
    )
    selected_row = candidates.iloc[0]
    selected_policy = by_id[str(selected_row["policy_id"])]
    search_metrics.loc[
        search_metrics["policy_id"].eq(selected_policy.policy_id),
        "selected_policy",
    ] = True
    return PolicySearchResult(
        inputs=inputs,
        search_metrics=search_metrics,
        selected_policy=selected_policy,
        selection_metric=selection_metric,
        selection_value=float(selected_row["value"]),
        lower_is_better=bool(lower_is_better),
    )


def freeze_policy(
    result: PolicySearchResult,
    *,
    training_protocol: Mapping[str, Any],
    search_manifest_hash: str,
) -> FrozenPolicyEnvelope:
    """Freeze a selected policy and all scientific bindings before test access."""
    protocol = _canonical_mapping(training_protocol, "training protocol")
    if not protocol:
        raise ValueError("Frozen policy training protocol must not be empty.")
    if not isinstance(search_manifest_hash, str) or not search_manifest_hash.strip():
        raise ValueError("Frozen policy search_manifest_hash must be a non-empty string.")
    provisional = FrozenPolicyEnvelope(
        binding=result.inputs.binding,
        resolved_policy=result.selected_policy,
        training_protocol=protocol,
        selection_metric=result.selection_metric,
        selection_value=result.selection_value,
        lower_is_better=result.lower_is_better,
        search_manifest_hash=search_manifest_hash,
        frozen_policy_hash="pending",
    )
    envelope = FrozenPolicyEnvelope(
        **{
            **provisional.__dict__,
            "frozen_policy_hash": stable_hash(provisional._unhashed_payload()),
        }
    )
    envelope.verify()
    return envelope


def load_frozen_policy(
    source: str | Path | Mapping[str, Any],
    *,
    expected_binding: ScientificBinding | None = None,
) -> FrozenPolicyEnvelope:
    """Load and strictly verify a frozen policy file or mapping."""
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        try:
            payload = json.loads(Path(source).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("Frozen policy file contains invalid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != _FROZEN_FIELDS:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(
            f"Frozen policy schema mismatch: expected {sorted(_FROZEN_FIELDS)}, "
            f"observed {observed}."
        )
    if payload["schema_version"] != FROZEN_POLICY_SCHEMA_VERSION:
        raise ValueError("Frozen policy schema version mismatch.")
    if payload["status"] != "frozen":
        raise ValueError("Final evaluation refuses an unfrozen policy.")
    if not isinstance(payload["lower_is_better"], bool):
        raise ValueError("Frozen lower_is_better must be boolean.")
    if not isinstance(payload["selection_metric"], str) or not payload["selection_metric"].strip():
        raise ValueError("Frozen selection_metric must be a non-empty string.")
    if (
        not isinstance(payload["search_manifest_hash"], str)
        or not payload["search_manifest_hash"].strip()
    ):
        raise ValueError("Frozen search_manifest_hash must be a non-empty string.")
    if isinstance(payload["selection_value"], bool) or not isinstance(
        payload["selection_value"], (int, float)
    ):
        raise ValueError("Frozen selection_value must be numeric.")
    if not isinstance(payload["resolved_policy"], Mapping):
        raise ValueError("Frozen resolved_policy must be a mapping.")
    resolved = ResolvedPolicy(
        policy_id=payload["policy_id"],
        config=payload["resolved_policy"],
    )
    if payload["resolved_policy_hash"] != resolved.policy_hash:
        raise ValueError("Frozen resolved policy hash mismatch.")
    protocol = _canonical_mapping(payload["training_protocol"], "training protocol")
    if not protocol:
        raise ValueError("Frozen policy training protocol must not be empty.")
    envelope = FrozenPolicyEnvelope(
        binding=ScientificBinding.from_dict(payload["binding"]),
        resolved_policy=resolved,
        training_protocol=protocol,
        selection_metric=payload["selection_metric"],
        selection_value=float(payload["selection_value"]),
        lower_is_better=payload["lower_is_better"],
        search_manifest_hash=payload["search_manifest_hash"],
        frozen_policy_hash=payload["frozen_policy_hash"],
        schema_version=payload["schema_version"],
        status=payload["status"],
    )
    envelope.verify(expected_binding=expected_binding)
    return envelope


def evaluate_frozen_policy_once(
    inputs: FinalEvaluationInputs,
    frozen_policy: FrozenPolicyEnvelope,
    outer_evaluator: OuterTestEvaluator,
    *,
    evaluated_units: set[str],
) -> FinalEvaluationResult:
    """Verify bindings and invoke outer evaluation once for a unique unit."""
    frozen_policy.verify(expected_binding=inputs.binding)
    if inputs.evaluation_unit in evaluated_units:
        raise ValueError(
            f"Outer test evaluation unit was already evaluated: {inputs.evaluation_unit}."
        )
    evaluated_units.add(inputs.evaluation_unit)
    metrics = _normalize_metric_rows(outer_evaluator(inputs, frozen_policy))
    if not metrics["split"].eq("test").all():
        raise ValueError("Final evaluator returned a non-test metric row.")
    metrics = metrics.copy()
    metrics["evaluation_unit"] = inputs.evaluation_unit
    metrics["frozen_policy_hash"] = frozen_policy.frozen_policy_hash
    return FinalEvaluationResult(
        evaluation_unit=inputs.evaluation_unit,
        metrics=metrics,
        frozen_policy_hash=frozen_policy.frozen_policy_hash,
    )


def _normalize_metric_rows(
    value: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, Mapping):
        frame = pd.DataFrame([dict(value)])
    else:
        frame = pd.DataFrame([dict(row) for row in value])
    required = {"split", "metric", "value"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Evaluation metric rows are missing columns: {missing}.")
    if frame.empty:
        raise ValueError("Evaluation returned no metric rows.")
    if frame["split"].isna().any() or frame["metric"].isna().any():
        raise ValueError("Evaluation split and metric names must be non-missing.")
    numeric = pd.to_numeric(frame["value"], errors="coerce")
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise ValueError("Evaluation metric values must be finite.")
    frame["value"] = numeric.astype(float)
    return frame


def _canonical_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name.capitalize()} must be a mapping.")
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name.capitalize()} must be JSON serializable.") from exc
