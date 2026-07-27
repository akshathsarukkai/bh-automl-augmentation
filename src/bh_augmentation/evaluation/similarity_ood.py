"""Deterministic reaction-fingerprint cluster and similarity-bounded OOD splits.

Fingerprints in this module are role-aware similarity features. Fingerprint
equality is never used as a chemical identity relation; canonical reaction keys
remain the only grouping unit used to keep reactions intact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES
from bh_augmentation.utils.corrected_runs import stable_hash

SIMILARITY_OOD_SCHEMA_VERSION = "bh-reaction-similarity-ood-v1"
FINGERPRINT_EQUALITY_SEMANTICS = (
    "feature_collision_only_not_chemical_equivalence"
)
ASSIGNMENT_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "ood_group",
    "ood_split",
)
NEAREST_SIMILARITY_COLUMNS = (
    "source_row_id",
    "canonical_reaction_key",
    "ood_group",
    "nearest_train_canonical_reaction_key",
    "nearest_train_tanimoto",
)
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
GROUP_NEAREST_SIMILARITY_COLUMNS = (
    "canonical_reaction_key",
    "nearest_train_canonical_reaction_key",
    "nearest_train_tanimoto",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReactionFingerprintIndex:
    """Validated role-aware fingerprints for canonical reaction groups."""

    n_bits_per_role: int
    radius: int
    role_order: tuple[str, ...]
    group_keys: tuple[str, ...]
    fingerprint_hashes: Mapping[str, str]
    fingerprint_metadata_hash: str
    _active_bits: Mapping[str, frozenset[int]] = field(
        repr=False,
        compare=False,
    )
    _rdkit_fingerprints: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )

    def tanimoto(self, left_key: str, right_key: str) -> float:
        """Return role-aware Tanimoto similarity for two canonical groups."""
        try:
            left = self._active_bits[left_key]
            right = self._active_bits[right_key]
        except KeyError as exc:
            raise ValueError("Unknown canonical reaction key in similarity query.") from exc
        union = left | right
        if not union:
            raise ValueError("Reaction fingerprint union is empty.")
        return len(left & right) / len(union)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the complete scientific fingerprint contract."""
        return {
            "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
            "fingerprint_type": "role_offset_morgan_bit_vector",
            "n_bits_per_role": self.n_bits_per_role,
            "total_bits": self.n_bits_per_role * len(self.role_order),
            "radius": self.radius,
            "use_chirality": True,
            "role_order": list(self.role_order),
            "identity_grouping": "canonical_reaction_key",
            "fingerprint_equality_semantics": FINGERPRINT_EQUALITY_SEMANTICS,
            "fingerprint_metadata_hash": self.fingerprint_metadata_hash,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class NearestTrainingSimilarityResult:
    """Immutable nearest-training query result for canonical reaction groups."""

    train_keys: tuple[str, ...]
    test_keys: tuple[str, ...]
    query_hash: str
    maximum_similarity: float
    _similarity_quantiles: Mapping[str, float] = field(
        repr=False,
        compare=False,
    )
    _rows: pd.DataFrame = field(repr=False, compare=False)

    @property
    def rows(self) -> pd.DataFrame:
        """Return deterministic group-level nearest-training rows."""
        return self._rows.copy(deep=True)

    @property
    def similarity_quantiles(self) -> dict[str, float]:
        """Return fixed quantiles of group-level nearest similarities."""
        return dict(self._similarity_quantiles)


@dataclass(frozen=True, slots=True, kw_only=True)
class PairwiseReactionSimilarityIndex:
    """Immutable metadata-bound pairwise reaction-similarity matrix."""

    canonical_reaction_keys: tuple[str, ...]
    fingerprint_metadata_hash: str
    fingerprint_table_hash: str
    similarity_matrix_hash: str
    similarity_index_hash: str
    _fingerprint_metadata: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    _matrix: np.ndarray = field(repr=False, compare=False)

    @property
    def matrix(self) -> np.ndarray:
        """Return a read-only copy in ``canonical_reaction_keys`` order."""
        result = self._matrix.copy()
        result.setflags(write=False)
        return result

    @property
    def fingerprint_metadata(self) -> dict[str, Any]:
        """Return a defensive copy of the bound fingerprint contract."""
        return _json_copy(self._fingerprint_metadata)

    def similarity(self, left_key: str, right_key: str) -> float:
        """Return matrix similarity for two canonical reaction keys."""
        positions = self._positions()
        try:
            return float(self._matrix[positions[left_key], positions[right_key]])
        except KeyError as exc:
            raise ValueError(
                "Unknown canonical reaction key in pairwise similarity query."
            ) from exc

    def nearest_training_similarities(
        self,
        *,
        train_keys: set[str] | tuple[str, ...] | list[str],
        test_keys: set[str] | tuple[str, ...] | list[str],
    ) -> NearestTrainingSimilarityResult:
        """Query nearest training groups with deterministic lexical tie breaks."""
        normalized_train = _normalize_query_keys(train_keys, name="train_keys")
        normalized_test = _normalize_query_keys(test_keys, name="test_keys")
        overlap = sorted(set(normalized_train) & set(normalized_test))
        if overlap:
            raise ValueError(
                "Pairwise nearest-training query requires disjoint train/test "
                f"canonical keys; overlap={overlap}."
            )
        known = set(self.canonical_reaction_keys)
        unknown = sorted((set(normalized_train) | set(normalized_test)) - known)
        if unknown:
            raise ValueError(
                f"Pairwise nearest-training query contains unknown keys: {unknown}."
            )
        positions = self._positions()
        train_positions = np.asarray(
            [positions[key] for key in normalized_train],
            dtype=np.int64,
        )
        test_positions = np.asarray(
            [positions[key] for key in normalized_test],
            dtype=np.int64,
        )
        query_matrix = self._matrix[np.ix_(test_positions, train_positions)]
        if (
            query_matrix.shape
            != (len(normalized_test), len(normalized_train))
            or not np.isfinite(query_matrix).all()
        ):
            raise ValueError("Pairwise nearest-training query matrix is invalid.")
        maximum_values = query_matrix.max(axis=1)
        rows = []
        for row_index, test_key in enumerate(normalized_test):
            maximum = maximum_values[row_index]
            # Training keys are sorted lexically, so the first exact maximum is
            # the declared stable tie break.
            nearest_index = int(
                np.flatnonzero(query_matrix[row_index] == maximum)[0]
            )
            rows.append(
                {
                    "canonical_reaction_key": test_key,
                    "nearest_train_canonical_reaction_key": normalized_train[
                        nearest_index
                    ],
                    "nearest_train_tanimoto": float(maximum),
                }
            )
        result_frame = pd.DataFrame(
            rows,
            columns=GROUP_NEAREST_SIMILARITY_COLUMNS,
        )
        quantiles = _similarity_quantiles(
            result_frame["nearest_train_tanimoto"]
        )
        query_payload = {
            "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
            "similarity_index_hash": self.similarity_index_hash,
            "train_keys": list(normalized_train),
            "test_keys": list(normalized_test),
            "rows": result_frame.to_dict(orient="records"),
            "similarity_quantiles": quantiles,
        }
        return NearestTrainingSimilarityResult(
            train_keys=normalized_train,
            test_keys=normalized_test,
            query_hash=stable_hash(query_payload),
            maximum_similarity=float(maximum_values.max()),
            _similarity_quantiles=quantiles,
            _rows=result_frame,
        )

    def _positions(self) -> dict[str, int]:
        return {
            key: index for index, key in enumerate(self.canonical_reaction_keys)
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SimilarityOODSplit:
    """One included or explicitly excluded OOD split."""

    split_name: str
    split_method: str
    status: str
    ood_group: str
    exclusion_reason: str | None
    n_train_rows: int
    n_test_rows: int
    n_train_canonical_groups: int
    n_test_canonical_groups: int
    split_hash: str | None
    similarity_quantiles: Mapping[str, float]
    maximum_test_to_train_similarity: float | None
    configured_similarity_bound: float | None
    _assignments: pd.DataFrame = field(repr=False, compare=False)
    _nearest_train_similarities: pd.DataFrame = field(
        repr=False,
        compare=False,
    )

    @property
    def assignments(self) -> pd.DataFrame:
        """Return a defensive copy of row assignments."""
        return self._assignments.copy(deep=True)

    @property
    def nearest_train_similarities(self) -> pd.DataFrame:
        """Return a defensive copy of test nearest-neighbor similarities."""
        return self._nearest_train_similarities.copy(deep=True)

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return concise serializable evidence for a split manifest."""
        return {
            "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
            "split_name": self.split_name,
            "split_method": self.split_method,
            "status": self.status,
            "ood_group": self.ood_group,
            "exclusion_reason": self.exclusion_reason,
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_train_canonical_groups": self.n_train_canonical_groups,
            "n_test_canonical_groups": self.n_test_canonical_groups,
            "split_hash": self.split_hash,
            "similarity_quantiles": dict(self.similarity_quantiles),
            "maximum_test_to_train_similarity": (
                self.maximum_test_to_train_similarity
            ),
            "configured_similarity_bound": self.configured_similarity_bound,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ReactionFingerprintClusterPlan:
    """Complete deterministic cluster assignment and holdout-fold plan."""

    seed: int
    similarity_cutoff: float
    fingerprint_metadata: Mapping[str, Any]
    cluster_assignment_hash: str
    splits: tuple[SimilarityOODSplit, ...]
    _cluster_assignments: pd.DataFrame = field(repr=False, compare=False)

    @property
    def cluster_assignments(self) -> pd.DataFrame:
        """Return canonical groups and their fingerprint cluster IDs."""
        return self._cluster_assignments.copy(deep=True)

    @property
    def included_splits(self) -> tuple[SimilarityOODSplit, ...]:
        """Return scientifically viable cluster folds."""
        return tuple(split for split in self.splits if split.status == "included")

    @property
    def excluded_splits(self) -> tuple[SimilarityOODSplit, ...]:
        """Return folds excluded with explicit support reasons."""
        return tuple(split for split in self.splits if split.status == "excluded")


def build_reaction_fingerprint_index(
    canonical: pd.DataFrame,
    *,
    n_bits_per_role: int = 2048,
    radius: int = 2,
) -> ReactionFingerprintIndex:
    """Build one role-offset Morgan fingerprint per canonical reaction key."""
    Chem, AllChem, _, DataStructs = _load_rdkit()
    n_bits = _positive_integer(n_bits_per_role, "n_bits_per_role")
    fingerprint_radius = _nonnegative_integer(radius, "radius")
    frame, group_roles = _normalize_canonical_frame(canonical, Chem=Chem)
    active_bits: dict[str, frozenset[int]] = {}
    rdkit_fingerprints: dict[str, Any] = {}
    fingerprint_hashes: dict[str, str] = {}
    for reaction_key in sorted(group_roles):
        bits: set[int] = set()
        role_values = group_roles[reaction_key]
        for role_index, role in enumerate(CANONICAL_ROLE_NAMES):
            molecule = Chem.MolFromSmiles(role_values[role])
            if molecule is None:  # pragma: no cover - normalization already checks
                raise ValueError(
                    f"RDKit failed to parse canonical role {role!r} for "
                    f"reaction {reaction_key!r}."
                )
            role_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                fingerprint_radius,
                nBits=n_bits,
                useChirality=True,
            )
            offset = role_index * n_bits
            bits.update(offset + int(bit) for bit in role_fingerprint.GetOnBits())
        if not bits:
            raise ValueError(
                f"Canonical reaction {reaction_key!r} produced an empty fingerprint."
            )
        active_bits[reaction_key] = frozenset(bits)
        reaction_fingerprint = DataStructs.ExplicitBitVect(
            n_bits * len(CANONICAL_ROLE_NAMES)
        )
        for bit in bits:
            reaction_fingerprint.SetBit(bit)
        rdkit_fingerprints[reaction_key] = reaction_fingerprint
        fingerprint_hashes[reaction_key] = stable_hash(sorted(bits))
    metadata_payload = {
        "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
        "fingerprint_type": "role_offset_morgan_bit_vector",
        "n_bits_per_role": n_bits,
        "radius": fingerprint_radius,
        "use_chirality": True,
        "role_order": list(CANONICAL_ROLE_NAMES),
        "identity_grouping": "canonical_reaction_key",
        "fingerprint_equality_semantics": FINGERPRINT_EQUALITY_SEMANTICS,
    }
    # ``frame`` is intentionally normalized even though only group roles are
    # retained: row/group validation is part of this index's contract.
    if frame.empty:  # pragma: no cover - guarded by normalization
        raise ValueError("Canonical reaction frame is empty.")
    return ReactionFingerprintIndex(
        n_bits_per_role=n_bits,
        radius=fingerprint_radius,
        role_order=tuple(CANONICAL_ROLE_NAMES),
        group_keys=tuple(sorted(active_bits)),
        fingerprint_hashes=dict(sorted(fingerprint_hashes.items())),
        fingerprint_metadata_hash=stable_hash(metadata_payload),
        _active_bits=active_bits,
        _rdkit_fingerprints=rdkit_fingerprints,
    )


def build_pairwise_reaction_similarity_index(
    canonical: pd.DataFrame,
    *,
    n_bits_per_role: int = 2048,
    radius: int = 2,
) -> PairwiseReactionSimilarityIndex:
    """Build a reusable symmetric matrix with RDKit bulk Tanimoto queries."""
    _, _, _, DataStructs = _load_rdkit()
    fingerprint_index = build_reaction_fingerprint_index(
        canonical,
        n_bits_per_role=n_bits_per_role,
        radius=radius,
    )
    keys = fingerprint_index.group_keys
    matrix = np.empty((len(keys), len(keys)), dtype=np.float64)
    fingerprints = [
        fingerprint_index._rdkit_fingerprints[key] for key in keys
    ]
    for left_index, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            fingerprints[: left_index + 1],
        )
        if len(similarities) != left_index + 1:
            raise ValueError("RDKit bulk Tanimoto result has invalid length.")
        matrix[left_index, : left_index + 1] = similarities
        matrix[: left_index + 1, left_index] = similarities
    if (
        not np.isfinite(matrix).all()
        or not np.logical_and(matrix >= 0.0, matrix <= 1.0).all()
        or not np.array_equal(matrix, matrix.T)
        or not np.array_equal(np.diag(matrix), np.ones(len(keys)))
    ):
        raise ValueError(
            "Pairwise reaction-similarity matrix is not finite, symmetric, "
            "bounded, and diagonal-one."
        )
    matrix.setflags(write=False)
    fingerprint_table_hash = stable_hash(
        [
            {
                "canonical_reaction_key": key,
                "reaction_fingerprint_hash": fingerprint_index.fingerprint_hashes[
                    key
                ],
            }
            for key in keys
        ]
    )
    similarity_matrix_hash = hashlib.sha256(
        matrix.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()
    similarity_payload = {
        "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
        "canonical_reaction_keys": list(keys),
        "fingerprint_metadata_hash": (
            fingerprint_index.fingerprint_metadata_hash
        ),
        "fingerprint_table_hash": fingerprint_table_hash,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": "little-endian-float64",
        "similarity_matrix_hash": similarity_matrix_hash,
    }
    return PairwiseReactionSimilarityIndex(
        canonical_reaction_keys=keys,
        fingerprint_metadata_hash=fingerprint_index.fingerprint_metadata_hash,
        fingerprint_table_hash=fingerprint_table_hash,
        similarity_matrix_hash=similarity_matrix_hash,
        similarity_index_hash=stable_hash(similarity_payload),
        _fingerprint_metadata=fingerprint_index.metadata,
        _matrix=matrix,
    )


def build_reaction_fingerprint_cluster_holdouts(
    canonical: pd.DataFrame,
    *,
    n_bits_per_role: int = 2048,
    radius: int = 2,
    similarity_cutoff: float = 0.6,
    seed: int = 0,
    min_train_rows: int = 1,
    min_test_rows: int = 1,
    min_train_canonical_groups: int = 2,
    min_test_canonical_groups: int = 1,
) -> ReactionFingerprintClusterPlan:
    """Cluster canonical reactions and construct one holdout per cluster."""
    _, _, Butina, _ = _load_rdkit()
    frame, _ = _normalize_canonical_frame(canonical)
    index = build_reaction_fingerprint_index(
        frame,
        n_bits_per_role=n_bits_per_role,
        radius=radius,
    )
    cutoff = _unit_interval(similarity_cutoff, "similarity_cutoff")
    normalized_seed = _integer(seed, "seed")
    support = _support_contract(
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        min_train_canonical_groups=min_train_canonical_groups,
        min_test_canonical_groups=min_test_canonical_groups,
    )
    ordered_keys = sorted(
        index.group_keys,
        key=lambda key: (_seeded_rank(normalized_seed, key), key),
    )
    distances = [
        1.0 - index.tanimoto(ordered_keys[left], ordered_keys[right])
        for left in range(1, len(ordered_keys))
        for right in range(left)
    ]
    raw_clusters = Butina.ClusterData(
        distances,
        len(ordered_keys),
        1.0 - cutoff,
        isDistData=True,
        reordering=True,
    )
    member_clusters = [
        tuple(sorted(ordered_keys[index_value] for index_value in cluster))
        for cluster in raw_clusters
    ]
    member_clusters.sort(key=lambda members: (members[0], stable_hash(members)))
    cluster_by_key: dict[str, str] = {}
    cluster_rows = []
    for members in member_clusters:
        cluster_id = f"reaction-fingerprint-cluster-{stable_hash(list(members))[:16]}"
        for reaction_key in members:
            cluster_by_key[reaction_key] = cluster_id
            cluster_rows.append(
                {
                    "canonical_reaction_key": reaction_key,
                    "cluster_id": cluster_id,
                    "reaction_fingerprint_hash": index.fingerprint_hashes[
                        reaction_key
                    ],
                }
            )
    if set(cluster_by_key) != set(index.group_keys):
        raise ValueError("Reaction-fingerprint clustering did not cover every group.")
    cluster_frame = pd.DataFrame(cluster_rows).sort_values(
        ["cluster_id", "canonical_reaction_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    splits = tuple(
        _build_split(
            frame,
            index=index,
            test_keys={
                key for key, assigned_cluster in cluster_by_key.items()
                if assigned_cluster == cluster_id
            },
            ood_group=cluster_id,
            split_name=f"reaction_fingerprint_cluster_holdout__{cluster_id}",
            split_method="reaction_fingerprint_cluster_holdout",
            support=support,
            configured_similarity_bound=None,
        )
        for cluster_id in sorted(set(cluster_by_key.values()))
    )
    return ReactionFingerprintClusterPlan(
        seed=normalized_seed,
        similarity_cutoff=cutoff,
        fingerprint_metadata=index.metadata,
        cluster_assignment_hash=stable_hash(
            cluster_frame.to_dict(orient="records")
        ),
        splits=splits,
        _cluster_assignments=cluster_frame,
    )


def build_maximum_similarity_bounded_split(
    canonical: pd.DataFrame,
    *,
    maximum_similarity: float,
    target_test_fraction: float = 0.2,
    n_bits_per_role: int = 2048,
    radius: int = 2,
    seed: int = 0,
    min_train_rows: int = 1,
    min_test_rows: int = 1,
    min_train_canonical_groups: int = 2,
    min_test_canonical_groups: int = 1,
) -> SimilarityOODSplit:
    """Build a group-safe test set with no similarity above the given bound.

    Canonical groups connected by an edge above the bound must remain on the
    same side of the split. A deterministic component traversal selects whole
    components until the requested test fraction is reached without violating
    minimum training support.
    """
    frame, _ = _normalize_canonical_frame(canonical)
    index = build_reaction_fingerprint_index(
        frame,
        n_bits_per_role=n_bits_per_role,
        radius=radius,
    )
    bound = _unit_interval(maximum_similarity, "maximum_similarity")
    fraction = _open_unit_interval(target_test_fraction, "target_test_fraction")
    normalized_seed = _integer(seed, "seed")
    support = _support_contract(
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        min_train_canonical_groups=min_train_canonical_groups,
        min_test_canonical_groups=min_test_canonical_groups,
    )
    components = _similarity_components(index, bound=bound)
    component_rows = [
        (
            component,
            int(frame["canonical_reaction_key"].isin(component).sum()),
        )
        for component in components
    ]
    component_rows.sort(
        key=lambda item: (
            _seeded_rank(normalized_seed, stable_hash(list(item[0]))),
            item[0],
        )
    )
    target_rows = max(1, math.ceil(len(frame) * fraction))
    test_keys: set[str] = set()
    for component, _ in component_rows:
        proposed = test_keys | set(component)
        remaining_keys = set(index.group_keys) - proposed
        remaining_rows = int(
            frame["canonical_reaction_key"].isin(remaining_keys).sum()
        )
        if (
            len(remaining_keys) < support["min_train_canonical_groups"]
            or remaining_rows < support["min_train_rows"]
        ):
            continue
        test_keys = proposed
        selected_rows = int(
            frame["canonical_reaction_key"].isin(test_keys).sum()
        )
        if (
            selected_rows >= target_rows
            and len(test_keys) >= support["min_test_canonical_groups"]
            and selected_rows >= support["min_test_rows"]
        ):
            break
    split = _build_split(
        frame,
        index=index,
        test_keys=test_keys,
        ood_group=f"maximum-similarity-at-most-{bound:g}",
        split_name=f"maximum_similarity_bounded__{bound:g}",
        split_method="maximum_similarity_bounded",
        support=support,
        configured_similarity_bound=bound,
    )
    if split.status == "included":
        observed = split.maximum_test_to_train_similarity
        if observed is None or observed > bound + 1e-12:
            raise ValueError(
                "Constructed similarity-bounded split violates its configured bound."
            )
    elif len(components) == 1:
        return _replace_exclusion(
            split,
            "impossible_similarity_bound: all canonical reaction groups form "
            "one above-bound connected component, so non-empty group-safe "
            "training and test sets cannot both be constructed.",
        )
    return split


def _build_split(
    frame: pd.DataFrame,
    *,
    index: ReactionFingerprintIndex,
    test_keys: set[str],
    ood_group: str,
    split_name: str,
    split_method: str,
    support: Mapping[str, int],
    configured_similarity_bound: float | None,
) -> SimilarityOODSplit:
    all_keys = set(index.group_keys)
    if not test_keys <= all_keys:
        raise ValueError("Test groups contain unknown canonical reaction keys.")
    train_keys = all_keys - test_keys
    train_rows = int(frame["canonical_reaction_key"].isin(train_keys).sum())
    test_rows = int(frame["canonical_reaction_key"].isin(test_keys).sum())
    reason = _support_exclusion_reason(
        train_rows=train_rows,
        test_rows=test_rows,
        train_groups=len(train_keys),
        test_groups=len(test_keys),
        support=support,
    )
    if reason is not None:
        return _excluded_split(
            split_name=split_name,
            split_method=split_method,
            ood_group=ood_group,
            reason=reason,
            train_rows=train_rows,
            test_rows=test_rows,
            train_groups=len(train_keys),
            test_groups=len(test_keys),
            configured_similarity_bound=configured_similarity_bound,
        )
    assignments = frame[["source_row_id", "canonical_reaction_key"]].copy()
    assignments["ood_group"] = ood_group
    assignments["ood_split"] = assignments["canonical_reaction_key"].map(
        lambda key: "test" if key in test_keys else "train"
    )
    assignments = assignments[list(ASSIGNMENT_COLUMNS)].sort_values(
        "source_row_id",
        kind="mergesort",
    ).reset_index(drop=True)
    if (
        assignments.groupby("canonical_reaction_key")["ood_split"].nunique().max()
        != 1
    ):
        raise ValueError("A canonical reaction group crosses train and test.")
    nearest = _nearest_training_distribution(
        frame,
        index=index,
        train_keys=train_keys,
        test_keys=test_keys,
        ood_group=ood_group,
    )
    maximum = float(nearest["nearest_train_tanimoto"].max())
    quantiles = _similarity_quantiles(nearest["nearest_train_tanimoto"])
    split_payload = {
        "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
        "split_name": split_name,
        "split_method": split_method,
        "ood_group": ood_group,
        "assignments": assignments.to_dict(orient="records"),
        "fingerprint_metadata_hash": index.fingerprint_metadata_hash,
        "configured_similarity_bound": configured_similarity_bound,
    }
    return SimilarityOODSplit(
        split_name=split_name,
        split_method=split_method,
        status="included",
        ood_group=ood_group,
        exclusion_reason=None,
        n_train_rows=train_rows,
        n_test_rows=test_rows,
        n_train_canonical_groups=len(train_keys),
        n_test_canonical_groups=len(test_keys),
        split_hash=stable_hash(split_payload),
        similarity_quantiles=quantiles,
        maximum_test_to_train_similarity=maximum,
        configured_similarity_bound=configured_similarity_bound,
        _assignments=assignments,
        _nearest_train_similarities=nearest,
    )


def _nearest_training_distribution(
    frame: pd.DataFrame,
    *,
    index: ReactionFingerprintIndex,
    train_keys: set[str],
    test_keys: set[str],
    ood_group: str,
) -> pd.DataFrame:
    nearest_by_test: dict[str, tuple[str, float]] = {}
    for test_key in sorted(test_keys):
        ranked = sorted(
            (
                (train_key, index.tanimoto(test_key, train_key))
                for train_key in train_keys
            ),
            key=lambda item: (-item[1], item[0]),
        )
        nearest_by_test[test_key] = ranked[0]
    rows = []
    for row in frame.loc[
        frame["canonical_reaction_key"].isin(test_keys),
        ["source_row_id", "canonical_reaction_key"],
    ].to_dict(orient="records"):
        nearest_key, similarity = nearest_by_test[row["canonical_reaction_key"]]
        rows.append(
            {
                "source_row_id": row["source_row_id"],
                "canonical_reaction_key": row["canonical_reaction_key"],
                "ood_group": ood_group,
                "nearest_train_canonical_reaction_key": nearest_key,
                "nearest_train_tanimoto": float(similarity),
            }
        )
    return pd.DataFrame(rows, columns=NEAREST_SIMILARITY_COLUMNS).sort_values(
        "source_row_id",
        kind="mergesort",
    ).reset_index(drop=True)


def _similarity_components(
    index: ReactionFingerprintIndex,
    *,
    bound: float,
) -> list[tuple[str, ...]]:
    neighbors = {key: set() for key in index.group_keys}
    for left_index, left in enumerate(index.group_keys):
        for right in index.group_keys[:left_index]:
            if index.tanimoto(left, right) > bound + 1e-12:
                neighbors[left].add(right)
                neighbors[right].add(left)
    components = []
    unseen = set(index.group_keys)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            key = stack.pop()
            if key in component:
                continue
            component.add(key)
            stack.extend(sorted(neighbors[key] - component, reverse=True))
        unseen -= component
        components.append(tuple(sorted(component)))
    return sorted(components)


def _normalize_canonical_frame(
    canonical: pd.DataFrame,
    *,
    Chem: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    if not isinstance(canonical, pd.DataFrame) or canonical.empty:
        raise ValueError("Canonical reaction data must be a non-empty DataFrame.")
    if Chem is None:
        Chem, _, _, _ = _load_rdkit()
    role_columns = {
        role: f"canonical_{role}_smiles" for role in CANONICAL_ROLE_NAMES
    }
    required = {
        "source_row_id",
        "canonical_reaction_key",
        *role_columns.values(),
    }
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Canonical reaction data is missing columns: {missing}.")
    ordered_columns = [
        "source_row_id",
        "canonical_reaction_key",
        *(role_columns[role] for role in CANONICAL_ROLE_NAMES),
    ]
    frame = canonical[ordered_columns].copy()
    for column in ordered_columns:
        if frame[column].isna().any():
            raise ValueError(f"Canonical reaction column {column!r} is missing.")
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Canonical reaction column {column!r} is empty.")
    if frame["source_row_id"].duplicated().any():
        raise ValueError("Canonical source_row_id values must be unique.")
    if "all_required_roles_parse_valid" in canonical and not canonical[
        "all_required_roles_parse_valid"
    ].map(lambda value: isinstance(value, bool) and value).all():
        raise ValueError("Canonical reaction data contains invalid molecular roles.")
    group_roles: dict[str, dict[str, str]] = {}
    for reaction_key, rows in frame.groupby("canonical_reaction_key", sort=True):
        signatures = {
            tuple(row[role_columns[role]] for role in CANONICAL_ROLE_NAMES)
            for row in rows.to_dict(orient="records")
        }
        if len(signatures) != 1:
            raise ValueError(
                "One canonical reaction key maps to multiple seven-role identities."
            )
        signature = next(iter(signatures))
        role_values = dict(zip(CANONICAL_ROLE_NAMES, signature, strict=True))
        for role, smiles in role_values.items():
            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError(
                    f"RDKit cannot parse canonical role {role!r} for "
                    f"reaction {reaction_key!r}."
                )
        group_roles[str(reaction_key)] = role_values
    frame = frame.sort_values("source_row_id", kind="mergesort").reset_index(
        drop=True
    )
    return frame, group_roles


def _load_rdkit() -> tuple[Any, Any, Any, Any]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for scientific reaction-similarity OOD splits."
        ) from exc
    return Chem, AllChem, Butina, DataStructs


def _excluded_split(
    *,
    split_name: str,
    split_method: str,
    ood_group: str,
    reason: str,
    train_rows: int,
    test_rows: int,
    train_groups: int,
    test_groups: int,
    configured_similarity_bound: float | None,
) -> SimilarityOODSplit:
    return SimilarityOODSplit(
        split_name=split_name,
        split_method=split_method,
        status="excluded",
        ood_group=ood_group,
        exclusion_reason=reason,
        n_train_rows=train_rows,
        n_test_rows=test_rows,
        n_train_canonical_groups=train_groups,
        n_test_canonical_groups=test_groups,
        split_hash=None,
        similarity_quantiles={},
        maximum_test_to_train_similarity=None,
        configured_similarity_bound=configured_similarity_bound,
        _assignments=pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
        _nearest_train_similarities=pd.DataFrame(
            columns=NEAREST_SIMILARITY_COLUMNS
        ),
    )


def _replace_exclusion(
    split: SimilarityOODSplit,
    reason: str,
) -> SimilarityOODSplit:
    return _excluded_split(
        split_name=split.split_name,
        split_method=split.split_method,
        ood_group=split.ood_group,
        reason=reason,
        train_rows=split.n_train_rows,
        test_rows=split.n_test_rows,
        train_groups=split.n_train_canonical_groups,
        test_groups=split.n_test_canonical_groups,
        configured_similarity_bound=split.configured_similarity_bound,
    )


def _support_contract(**values: int) -> dict[str, int]:
    return {
        name: _positive_integer(value, name) for name, value in values.items()
    }


def _support_exclusion_reason(
    *,
    train_rows: int,
    test_rows: int,
    train_groups: int,
    test_groups: int,
    support: Mapping[str, int],
) -> str | None:
    failures = []
    observed = {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_canonical_groups": train_groups,
        "test_canonical_groups": test_groups,
    }
    minimums = {
        "train_rows": support["min_train_rows"],
        "test_rows": support["min_test_rows"],
        "train_canonical_groups": support["min_train_canonical_groups"],
        "test_canonical_groups": support["min_test_canonical_groups"],
    }
    for name in (
        "train_rows",
        "test_rows",
        "train_canonical_groups",
        "test_canonical_groups",
    ):
        if observed[name] < minimums[name]:
            failures.append(
                f"{name}={observed[name]} below minimum={minimums[name]}"
            )
    if failures:
        return "insufficient_support: " + "; ".join(failures)
    return None


def _similarity_quantiles(values: pd.Series) -> dict[str, float]:
    if values.empty or not values.map(
        lambda value: isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ).all():
        raise ValueError("Nearest-training similarities must be finite.")
    return {
        f"q{int(quantile * 100):02d}": float(values.quantile(quantile))
        for quantile in _QUANTILES
    }


def _normalize_query_keys(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (set, tuple, list)) or not value:
        raise ValueError(f"{name} must be a non-empty set, tuple, or list.")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError(f"{name} must contain non-empty string keys.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate canonical reaction keys.")
    return tuple(sorted(value))


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _seeded_rank(seed: int, value: str) -> str:
    return stable_hash(
        {
            "schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
            "seed": seed,
            "value": value,
        }
    )


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _unit_interval(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return float(value)


def _open_unit_interval(value: Any, name: str) -> float:
    result = _unit_interval(value, name)
    if result in {0.0, 1.0}:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return result
