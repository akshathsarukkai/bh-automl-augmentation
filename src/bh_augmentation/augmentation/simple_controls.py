"""Deterministic train-only controls for matched augmentation comparisons.

The public builder accepts only an explicit measured training matrix, labels,
and source identities.  It has no interface for validation, test, excluded,
or external unlabeled rows.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model

SIMPLE_CONTROL_SPECS = (
    ("real_only", "Real-only"),
    ("exact_duplication", "Exact real-row duplication"),
    ("random_oversampling", "Random oversampling"),
    ("yield_stratified_oversampling", "Yield-stratified oversampling"),
    ("sample_reweighting", "Yield-stratified sample reweighting"),
    ("nearest_neighbor_pseudo_labeling", "Nearest-neighbor pseudo-labeling"),
    ("self_training", "Self-training"),
    ("feature_mixup", "Generic feature mixup"),
)
SIMPLE_CONTROL_IDS = tuple(control_id for control_id, _ in SIMPLE_CONTROL_SPECS)
SIMPLE_CONTROL_NAMES = tuple(name for _, name in SIMPLE_CONTROL_SPECS)
_CONTROL_NAME_BY_ID = dict(SIMPLE_CONTROL_SPECS)
_FEATURE_CONTROL_IDS = {
    "nearest_neighbor_pseudo_labeling",
    "self_training",
    "feature_mixup",
}
_AUDIT_COLUMNS = (
    "audit_record_type",
    "output_row_index",
    "row_id",
    "is_real",
    "is_added",
    "accepted",
    "source_row_id",
    "donor_row_id",
    "label_source_row_id",
    "yield_stratum",
    "alpha",
    "label_strategy",
    "feature_hash",
    "feature_duplicate",
    "rejection_reason",
    "chemical_identity_applicable",
    "canonical_reaction_key",
    "canonical_reaction_hash",
)


@dataclass(frozen=True, slots=True)
class SimpleAugmentationControlResult:
    """One fully audited train-only augmentation control."""

    control_id: str
    control_name: str
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray | None
    row_audit: pd.DataFrame
    requested_added_count: int
    effective_added_count: int
    total_sample_weight: float
    chemical_identity_applicable: bool
    nonchemical_semantics: str | None


def build_simple_augmentation_control(
    control_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    source_row_ids: Sequence[str],
    *,
    added_count: int,
    random_state: int = 0,
    n_yield_strata: int = 4,
    mixup_alpha: float = 0.5,
    teacher_model_config: Mapping[str, Any] | None = None,
) -> SimpleAugmentationControlResult:
    """Build one deterministic control from measured training rows only."""
    X, y, source_ids = _normalize_training_inputs(
        X_train,
        y_train,
        source_row_ids,
    )
    budget = _nonnegative_integer(added_count, name="added_count")
    seed = _integer(random_state, name="random_state")
    strata_count = _positive_integer(n_yield_strata, name="n_yield_strata")
    alpha = float(mixup_alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("mixup_alpha must be finite and strictly between zero and one.")
    if control_id not in _CONTROL_NAME_BY_ID:
        raise ValueError(
            f"Unsupported simple augmentation control {control_id!r}; "
            f"expected one of {SIMPLE_CONTROL_IDS}."
        )
    strata = _yield_strata(y, strata_count)

    if control_id == "real_only":
        X_out, y_out = X.copy(), y.copy()
        weights = None
        audit = _base_audit(
            X,
            y,
            source_ids,
            strata,
            chemical_identity_applicable=True,
        )
    elif control_id == "exact_duplication":
        positions = np.arange(budget, dtype=int) % len(X)
        X_out, y_out, audit = _row_resampling_result(
            X,
            y,
            source_ids,
            strata,
            positions,
            control_id=control_id,
        )
        weights = None
    elif control_id == "random_oversampling":
        positions = np.random.default_rng(seed).integers(
            0,
            len(X),
            size=budget,
            dtype=np.int64,
        )
        X_out, y_out, audit = _row_resampling_result(
            X,
            y,
            source_ids,
            strata,
            positions,
            control_id=control_id,
        )
        weights = None
    elif control_id == "yield_stratified_oversampling":
        positions = _stratified_oversampling_positions(
            strata,
            budget,
            random_state=seed,
        )
        X_out, y_out, audit = _row_resampling_result(
            X,
            y,
            source_ids,
            strata,
            positions,
            control_id=control_id,
        )
        weights = None
    elif control_id == "sample_reweighting":
        X_out, y_out = X.copy(), y.copy()
        weights = _stratified_sample_weights(strata, budget)
        audit = _base_audit(
            X,
            y,
            source_ids,
            strata,
            chemical_identity_applicable=True,
        )
        audit["label_strategy"] = "measured_yield_reweighted"
    else:
        candidates, candidate_audit = _convex_candidate_pool(
            X,
            source_ids,
            budget,
            random_state=seed,
            alpha=alpha,
        )
        candidate_y, label_sources, label_strategy = _candidate_labels(
            control_id,
            candidates,
            candidate_audit,
            X,
            y,
            source_ids,
            mixup_alpha=alpha,
            teacher_model_config=teacher_model_config,
            random_state=seed,
        )
        X_out = np.vstack([X, candidates]).astype(np.float32)
        y_out = np.concatenate([y, candidate_y]).astype(np.float32)
        weights = None
        audit = _base_audit(
            X,
            y,
            source_ids,
            strata,
            chemical_identity_applicable=False,
        )
        accepted_audit = _accepted_candidate_audit(
            candidate_audit,
            candidates,
            label_sources,
            label_strategy=label_strategy,
            output_offset=len(X),
        )
        rejected_audit = candidate_audit.loc[
            ~candidate_audit["accepted"]
        ].copy()
        audit = pd.concat(
            [audit, accepted_audit, rejected_audit],
            ignore_index=True,
            sort=False,
        )

    return _package_result(
        control_id,
        X_out,
        y_out,
        weights,
        audit,
        requested_added_count=budget,
        n_real=len(X),
    )


def _normalize_training_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    source_row_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if X.ndim != 2 or not len(X):
        raise ValueError("X_train must be a nonempty dense two-dimensional matrix.")
    if len(X) != len(y):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training features and labels must be finite.")
    if len(source_row_ids) != len(X):
        raise ValueError("source_row_ids must contain one identity per training row.")
    if any(not isinstance(value, str) or not value.strip() for value in source_row_ids):
        raise ValueError("source_row_ids must contain nonempty strings.")
    if len(source_row_ids) != len(set(source_row_ids)):
        raise ValueError("source_row_ids must be unique.")
    order = np.asarray(
        sorted(range(len(X)), key=lambda index: source_row_ids[index]),
        dtype=int,
    )
    return (
        X[order].copy(),
        y[order].copy(),
        tuple(source_row_ids[index] for index in order),
    )


def _yield_strata(y: np.ndarray, requested_count: int) -> np.ndarray:
    count = min(requested_count, len(y))
    quantiles = np.linspace(0.0, 1.0, count + 1)
    edges = np.unique(np.quantile(y, quantiles))
    if len(edges) <= 1:
        return np.zeros(len(y), dtype=int)
    return np.digitize(y, edges[1:-1], right=True).astype(int)


def _stratified_oversampling_positions(
    strata: np.ndarray,
    budget: int,
    *,
    random_state: int,
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(random_state)
    active = tuple(sorted(int(value) for value in np.unique(strata)))
    positions: list[int] = []
    for draw in range(budget):
        candidates = np.flatnonzero(strata == active[draw % len(active)])
        positions.append(int(rng.choice(candidates)))
    return np.asarray(positions, dtype=int)


def _stratified_sample_weights(strata: np.ndarray, budget: int) -> np.ndarray:
    weights = np.ones(len(strata), dtype=np.float64)
    active = tuple(sorted(int(value) for value in np.unique(strata)))
    if budget:
        extra_per_stratum = float(budget) / len(active)
        for stratum in active:
            positions = np.flatnonzero(strata == stratum)
            weights[positions] += extra_per_stratum / len(positions)
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), len(strata) + budget, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("Yield-stratified sample weights are invalid.")
    return weights.astype(np.float32)


def _row_resampling_result(
    X: np.ndarray,
    y: np.ndarray,
    source_ids: tuple[str, ...],
    strata: np.ndarray,
    positions: np.ndarray,
    *,
    control_id: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_out = np.vstack([X, X[positions]]).astype(np.float32)
    y_out = np.concatenate([y, y[positions]]).astype(np.float32)
    audit = _base_audit(
        X,
        y,
        source_ids,
        strata,
        chemical_identity_applicable=True,
    )
    added_rows = [
        _audit_row(
            output_row_index=len(X) + ordinal,
            row_id=f"{control_id}:{ordinal:08d}:{source_ids[position]}",
            is_real=False,
            is_added=True,
            accepted=True,
            source_row_id=source_ids[position],
            donor_row_id=None,
            label_source_row_id=source_ids[position],
            yield_stratum=int(strata[position]),
            alpha=np.nan,
            label_strategy="copied_measured_yield",
            feature_hash=_feature_hash(X[position]),
            feature_duplicate=True,
            rejection_reason=None,
            chemical_identity_applicable=True,
        )
        for ordinal, position in enumerate(positions)
    ]
    if added_rows:
        audit = pd.concat(
            [audit, pd.DataFrame(added_rows, columns=_AUDIT_COLUMNS)],
            ignore_index=True,
        )
    return X_out, y_out, audit


def _convex_candidate_pool(
    X: np.ndarray,
    source_ids: tuple[str, ...],
    budget: int,
    *,
    random_state: int,
    alpha: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    if budget == 0:
        return (
            np.empty((0, X.shape[1]), dtype=np.float32),
            pd.DataFrame(columns=_AUDIT_COLUMNS),
        )
    # Ordered parent pairs allow asymmetric alpha values to contribute both
    # directions. At alpha=0.5 the reverse pair is deliberately encountered as
    # an auditable feature duplicate rather than silently counted twice.
    pairs = list(permutations(range(len(X)), 2))
    rng = np.random.default_rng(random_state)
    if pairs:
        pair_order = rng.permutation(len(pairs))
        pairs = [pairs[int(position)] for position in pair_order]
    real_hashes = {_feature_hash(row) for row in X}
    accepted_hashes: set[str] = set()
    accepted_vectors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for attempt, (left, right) in enumerate(pairs):
        candidate = (
            alpha * X[left] + (1.0 - alpha) * X[right]
        ).astype(np.float32)
        feature_hash = _feature_hash(candidate)
        reason = None
        if feature_hash in real_hashes:
            reason = "feature_identical_to_real"
        elif feature_hash in accepted_hashes:
            reason = "feature_duplicate_candidate"
        accepted = reason is None
        if accepted:
            accepted_hashes.add(feature_hash)
            accepted_vectors.append(candidate)
        records.append(
            _audit_row(
                output_row_index=(
                    len(X) + len(accepted_vectors) - 1 if accepted else pd.NA
                ),
                row_id=f"convex:{attempt:08d}:{source_ids[left]}:{source_ids[right]}",
                is_real=False,
                is_added=True,
                accepted=accepted,
                source_row_id=source_ids[left],
                donor_row_id=source_ids[right],
                label_source_row_id=None,
                yield_stratum=pd.NA,
                alpha=alpha,
                label_strategy=None,
                feature_hash=feature_hash,
                feature_duplicate=reason is not None,
                rejection_reason=reason,
                chemical_identity_applicable=False,
            )
        )
        if len(accepted_vectors) >= budget:
            break
    if len(accepted_vectors) < budget:
        records.append(
            _audit_row(
                output_row_index=pd.NA,
                row_id="convex-pool-underfill",
                is_real=False,
                is_added=False,
                accepted=False,
                source_row_id=None,
                donor_row_id=None,
                label_source_row_id=None,
                yield_stratum=pd.NA,
                alpha=alpha,
                label_strategy=None,
                feature_hash=None,
                feature_duplicate=False,
                rejection_reason="unique_candidate_pool_exhausted",
                chemical_identity_applicable=False,
            )
        )
    candidates = (
        np.vstack(accepted_vectors).astype(np.float32)
        if accepted_vectors
        else np.empty((0, X.shape[1]), dtype=np.float32)
    )
    return candidates, pd.DataFrame(records, columns=_AUDIT_COLUMNS)


def _candidate_labels(
    control_id: str,
    candidates: np.ndarray,
    candidate_audit: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    source_ids: tuple[str, ...],
    *,
    mixup_alpha: float,
    teacher_model_config: Mapping[str, Any] | None,
    random_state: int,
) -> tuple[np.ndarray, tuple[str | None, ...], str]:
    if not len(candidates):
        strategy = {
            "nearest_neighbor_pseudo_labeling": "nearest_real_train_yield_euclidean",
            "self_training": "real_train_only_teacher:not_fitted_empty_pool",
            "feature_mixup": "convex_parent_yield_mixup",
        }[control_id]
        return np.empty(0, dtype=np.float32), (), strategy
    if control_id == "nearest_neighbor_pseudo_labeling":
        labels: list[float] = []
        label_sources: list[str] = []
        for candidate in candidates:
            distances = np.sum((X - candidate) ** 2, axis=1, dtype=np.float64)
            exact_positions = np.flatnonzero(
                np.all(X == candidate, axis=1)
            )
            if len(exact_positions) and len(exact_positions) < len(X):
                distances[exact_positions] = np.inf
            nearest = int(np.argmin(distances))
            labels.append(float(y[nearest]))
            label_sources.append(source_ids[nearest])
        return (
            np.asarray(labels, dtype=np.float32),
            tuple(label_sources),
            "nearest_real_train_yield_euclidean",
        )
    if control_id == "self_training":
        config = _teacher_config(teacher_model_config)
        teacher = train_model(
            get_model(
                config["name"],
                seed=random_state,
                **config["params"],
            ),
            X,
            y,
        )
        predictions = np.asarray(
            predict_model(teacher, candidates),
            dtype=np.float32,
        )
        if not np.isfinite(predictions).all():
            raise ValueError("Self-training teacher produced non-finite predictions.")
        return (
            predictions,
            tuple(None for _ in candidates),
            f"real_train_only_teacher:{config['name']}",
        )
    if control_id == "feature_mixup":
        accepted = candidate_audit.loc[candidate_audit["accepted"]]
        position_by_source = {
            source_id: position for position, source_id in enumerate(source_ids)
        }
        labels = np.asarray(
            [
                mixup_alpha * float(y[position_by_source[row["source_row_id"]]])
                + (1.0 - mixup_alpha)
                * float(y[position_by_source[row["donor_row_id"]]])
                for row in accepted.to_dict(orient="records")
            ],
            dtype=np.float32,
        )
        return (
            labels,
            tuple(None for _ in candidates),
            "convex_parent_yield_mixup",
        )
    raise ValueError(f"Unsupported feature control: {control_id}.")


def _accepted_candidate_audit(
    candidate_audit: pd.DataFrame,
    candidates: np.ndarray,
    label_sources: tuple[str | None, ...],
    *,
    label_strategy: str,
    output_offset: int,
) -> pd.DataFrame:
    accepted = candidate_audit.loc[candidate_audit["accepted"]].copy()
    if len(accepted) != len(candidates) or len(accepted) != len(label_sources):
        raise ValueError("Candidate audit and accepted feature rows differ.")
    accepted["output_row_index"] = np.arange(
        output_offset,
        output_offset + len(accepted),
    )
    accepted["label_source_row_id"] = list(label_sources)
    accepted["label_strategy"] = label_strategy
    return accepted


def _base_audit(
    X: np.ndarray,
    y: np.ndarray,
    source_ids: tuple[str, ...],
    strata: np.ndarray,
    *,
    chemical_identity_applicable: bool,
) -> pd.DataFrame:
    del y
    return pd.DataFrame(
        [
            _audit_row(
                output_row_index=index,
                row_id=f"real:{source_id}",
                is_real=True,
                is_added=False,
                accepted=True,
                source_row_id=source_id,
                donor_row_id=None,
                label_source_row_id=source_id,
                yield_stratum=int(strata[index]),
                alpha=np.nan,
                label_strategy="measured_yield",
                feature_hash=_feature_hash(X[index]),
                feature_duplicate=False,
                rejection_reason=None,
                chemical_identity_applicable=chemical_identity_applicable,
            )
            for index, source_id in enumerate(source_ids)
        ],
        columns=_AUDIT_COLUMNS,
    )


def _audit_row(
    *,
    output_row_index: Any,
    row_id: str,
    is_real: bool,
    is_added: bool,
    accepted: bool,
    source_row_id: str | None,
    donor_row_id: str | None,
    label_source_row_id: str | None,
    yield_stratum: Any,
    alpha: float,
    label_strategy: str | None,
    feature_hash: str | None,
    feature_duplicate: bool,
    rejection_reason: str | None,
    chemical_identity_applicable: bool,
) -> dict[str, Any]:
    return {
        "audit_record_type": "output_row" if accepted else "rejected_candidate",
        "output_row_index": output_row_index,
        "row_id": row_id,
        "is_real": is_real,
        "is_added": is_added,
        "accepted": accepted,
        "source_row_id": source_row_id,
        "donor_row_id": donor_row_id,
        "label_source_row_id": label_source_row_id,
        "yield_stratum": yield_stratum,
        "alpha": alpha,
        "label_strategy": label_strategy,
        "feature_hash": feature_hash,
        "feature_duplicate": feature_duplicate,
        "rejection_reason": rejection_reason,
        "chemical_identity_applicable": chemical_identity_applicable,
        "canonical_reaction_key": None,
        "canonical_reaction_hash": None,
    }


def _teacher_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    config = {"name": "ridge", "params": {}} if value is None else dict(value)
    if set(config) != {"name", "params"}:
        raise ValueError("teacher_model_config must contain exactly 'name' and 'params'.")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("teacher_model_config.name must be a nonempty string.")
    if not isinstance(config["params"], Mapping):
        raise ValueError("teacher_model_config.params must be a mapping.")
    return {"name": config["name"], "params": dict(config["params"])}


def _package_result(
    control_id: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    audit: pd.DataFrame,
    *,
    requested_added_count: int,
    n_real: int,
) -> SimpleAugmentationControlResult:
    X_out = np.asarray(X, dtype=np.float32)
    y_out = np.asarray(y, dtype=np.float32).reshape(-1)
    if (
        X_out.ndim != 2
        or len(X_out) != len(y_out)
        or not np.isfinite(X_out).all()
        or not np.isfinite(y_out).all()
    ):
        raise ValueError("Control output features and labels must be finite and aligned.")
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if (
            len(weights) != len(y_out)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not float(weights.sum()) > 0.0
        ):
            raise ValueError("Control sample weights must be finite and nonnegative.")
    input_sources = set(
        audit.loc[audit["is_real"], "source_row_id"].dropna().astype(str)
    )
    for column in ("source_row_id", "donor_row_id", "label_source_row_id"):
        referenced = set(audit[column].dropna().astype(str))
        if not referenced <= input_sources:
            raise ValueError(f"Control audit contains a foreign {column}.")
    effective_added_count = len(X_out) - n_real
    applicable = control_id not in _FEATURE_CONTROL_IDS
    nonchemical = (
        None
        if applicable
        else (
            "Convex feature coordinates are nonchemical controls; no seven-role "
            "reaction identity, canonical reaction key, or chemical equivalence "
            "is claimed."
        )
    )
    total_weight = (
        float(weights.astype(np.float64).sum())
        if weights is not None
        else float(len(y_out))
    )
    return SimpleAugmentationControlResult(
        control_id=control_id,
        control_name=_CONTROL_NAME_BY_ID[control_id],
        X=X_out,
        y=y_out,
        sample_weight=weights,
        row_audit=audit.loc[:, _AUDIT_COLUMNS].reset_index(drop=True),
        requested_added_count=requested_added_count,
        effective_added_count=effective_added_count,
        total_sample_weight=total_weight,
        chemical_identity_applicable=applicable,
        nonchemical_semantics=nonchemical,
    )


def _feature_hash(row: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(row, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _integer(value: Any, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result
