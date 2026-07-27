"""Immutable split materialization for matched representation benchmarks.

This module deliberately materializes identities, not labeled data frames.
Callers can therefore establish and freeze the complete benchmark plan before
any outer-test outcomes are exposed to modeling code.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.data.canonicalize_roles import CANONICALIZATION_VERSION
from bh_augmentation.data.saved_canonical_splits import (
    SavedCanonicalSplits,
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.chemical_ood_artifacts import (
    ASSIGNMENT_COLUMNS,
    CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION,
    FAMILY_ORDER,
    FOLD_COLUMNS,
    validate_chemical_ood_artifacts,
)
from bh_augmentation.evaluation.logo_protocol import (
    LOGO_SCHEMA_VERSION,
    LOGO_SPLIT_METHOD,
    build_canonical_logo_contract,
    validate_canonical_logo_assignments,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash

REPRESENTATION_SPLIT_SCHEMA_VERSION = "bh-representation-split-unit-v1"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationSplitUnit:
    """One immutable, fully audited representation-evaluation split."""

    evaluation_unit: str
    split_family: str
    split_method: str
    target: str
    fold_index: int | None
    heldout_group: str | None
    seed: int | None
    train_fraction: float | None
    train_source_ids: tuple[str, ...]
    validation_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]
    excluded_source_ids: tuple[str, ...]
    dataset_hash: str
    exact_split_hash: str
    aggregate_assignment_hash: str
    canonical_split_dependency_hash: str
    upstream_split_hash: str | None
    canonicalization_version: str
    split_schema_version: str

    def __post_init__(self) -> None:
        """Reject mutable-looking, ambiguous, or incomplete memberships."""
        for name in (
            "evaluation_unit",
            "split_family",
            "split_method",
            "target",
            "canonicalization_version",
            "split_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.canonicalization_version != CANONICALIZATION_VERSION:
            raise ValueError("Representation split canonicalization version mismatch.")
        for name in (
            "dataset_hash",
            "exact_split_hash",
            "aggregate_assignment_hash",
            "canonical_split_dependency_hash",
        ):
            _require_hash(getattr(self, name), name=name)
        if self.upstream_split_hash is not None:
            _require_hash(self.upstream_split_hash, name="upstream_split_hash")
        if self.fold_index is not None and (
            not isinstance(self.fold_index, int)
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
        ):
            raise ValueError("fold_index must be null or a non-negative integer.")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be null or an integer.")
        if self.train_fraction is not None and (
            not math.isfinite(float(self.train_fraction))
            or not 0.0 < float(self.train_fraction) <= 1.0
        ):
            raise ValueError("train_fraction must be null or in (0, 1].")

        memberships: dict[str, tuple[str, ...]] = {}
        for name in (
            "train_source_ids",
            "validation_source_ids",
            "test_source_ids",
            "excluded_source_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple.")
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings.")
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique, stably sorted IDs.")
            memberships[name] = values
        if not self.train_source_ids or not self.test_source_ids:
            raise ValueError("Every evaluation unit requires nonempty train and test IDs.")
        role_sets = {name: set(values) for name, values in memberships.items()}
        names = tuple(role_sets)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if role_sets[left] & role_sets[right]:
                    raise ValueError(
                        "Representation split source IDs cross partition roles: "
                        f"{left}/{right}."
                    )

    @property
    def all_source_ids(self) -> tuple[str, ...]:
        """Return complete accounted-for membership in stable order."""
        return tuple(
            sorted(
                self.train_source_ids
                + self.validation_source_ids
                + self.test_source_ids
                + self.excluded_source_ids
            )
        )

    @property
    def audit_record(self) -> dict[str, Any]:
        """Return a concise manifest-ready split record."""
        return {
            "representation_split_schema_version": (
                REPRESENTATION_SPLIT_SCHEMA_VERSION
            ),
            "evaluation_unit": self.evaluation_unit,
            "split_family": self.split_family,
            "split_method": self.split_method,
            "target": self.target,
            "fold_index": self.fold_index,
            "heldout_group": self.heldout_group,
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "n_train": len(self.train_source_ids),
            "n_validation": len(self.validation_source_ids),
            "n_test": len(self.test_source_ids),
            "n_excluded": len(self.excluded_source_ids),
            "train_source_id_hash": stable_hash(list(self.train_source_ids)),
            "validation_source_id_hash": stable_hash(
                list(self.validation_source_ids)
            ),
            "test_source_id_hash": stable_hash(list(self.test_source_ids)),
            "excluded_source_id_hash": stable_hash(
                list(self.excluded_source_ids)
            ),
            "dataset_hash": self.dataset_hash,
            "exact_split_hash": self.exact_split_hash,
            "aggregate_assignment_hash": self.aggregate_assignment_hash,
            "canonical_split_dependency_hash": (
                self.canonical_split_dependency_hash
            ),
            "upstream_split_hash": self.upstream_split_hash,
            "canonicalization_version": self.canonicalization_version,
            "split_schema_version": self.split_schema_version,
        }


def build_saved_random_split_unit(
    saved: SavedCanonicalSplits,
    *,
    seed: int,
    train_fraction: float,
) -> EvaluationSplitUnit:
    """Build one exact nested random-split unit from validated assignments."""
    _validate_saved_dependencies(saved)
    audit = saved.audit_record(seed=seed, train_fraction=train_fraction)
    resolved_fraction = float(audit["train_fraction"])
    rows = saved.low_data_assignments.loc[
        saved.low_data_assignments["seed"].eq(int(seed))
        & saved.low_data_assignments["train_fraction"].eq(resolved_fraction)
    ]
    if len(rows) != len(saved.canonical):
        raise ValueError("Saved random split slice does not cover canonical rows.")
    train_ids = _ids(rows.loc[rows["included_in_training_subset"]])
    validation_ids = _ids(rows.loc[rows["outer_split"].eq("valid")])
    test_ids = _ids(rows.loc[rows["outer_split"].eq("test")])
    accounted = set(train_ids) | set(validation_ids) | set(test_ids)
    excluded_ids = tuple(
        sorted(set(saved.canonical["source_row_id"].astype(str)) - accounted)
    )
    return _build_unit(
        canonical=saved.canonical,
        evaluation_unit=(
            f"canonical-random:seed={int(seed)}:"
            f"fraction={resolved_fraction:.12g}"
        ),
        split_family="canonical_random",
        split_method="canonical_saved_nested",
        target="canonical_reaction_key",
        fold_index=None,
        heldout_group=None,
        seed=int(seed),
        train_fraction=resolved_fraction,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        excluded_ids=excluded_ids,
        dataset_hash=saved.dataset_hash,
        exact_split_hash=str(audit["source_id_split_hash"]),
        aggregate_assignment_hash=saved.aggregate_split_hash,
        canonical_split_dependency_hash=saved.aggregate_split_hash,
        upstream_split_hash=str(audit["per_seed_split_hash"]),
        split_schema_version=str(audit["split_schema_version"]),
    )


def build_canonical_logo_split_units(
    saved: SavedCanonicalSplits,
    *,
    target: str,
) -> tuple[EvaluationSplitUnit, ...]:
    """Rebuild product or reactant LOGO units from canonical identities."""
    _validate_saved_dependencies(saved)
    contract = build_canonical_logo_contract(saved.canonical, target=target)
    return _logo_units_from_contract(saved, contract)


def load_corrected_logo_split_units(
    saved: SavedCanonicalSplits,
    *,
    logo_root_directory: str | Path,
    target: str,
) -> tuple[EvaluationSplitUnit, ...]:
    """Load authoritative Phase 7 assignments and match a canonical rebuild.

    Result metrics are intentionally ignored. The completion manifest, target
    manifest, and every output hash they declare are checked before persisted
    assignments are passed to the strict canonical LOGO validator.
    """
    _validate_saved_dependencies(saved)
    root = Path(logo_root_directory)
    completion_path = root / "completion_manifest.json"
    completion = _load_hashed_document(
        completion_path,
        hash_field="completion_payload_hash",
        label="Corrected LOGO completion manifest",
    )
    if (
        completion.get("schema_version") != LOGO_SCHEMA_VERSION
        or completion.get("status") != "complete"
    ):
        raise ValueError("Corrected LOGO completion manifest contract mismatch.")
    expected_targets = completion.get("expected_targets")
    observed_targets = completion.get("observed_targets")
    if (
        not isinstance(expected_targets, list)
        or not isinstance(observed_targets, list)
        or target not in expected_targets
        or target not in observed_targets
    ):
        raise ValueError("Corrected LOGO target is absent from completion manifest.")
    target_outputs = completion.get("target_outputs")
    if not isinstance(target_outputs, dict) or not isinstance(
        target_outputs.get(target), dict
    ):
        raise ValueError("Corrected LOGO target output binding is absent.")
    completion_target = target_outputs[target]
    target_directory = root / target
    manifest_path = target_directory / "manifest.json"
    if sha256_file(manifest_path) != completion_target.get("manifest_hash"):
        raise ValueError("Corrected LOGO target manifest output hash mismatch.")
    _verify_declared_output_hashes(
        target_directory,
        completion_target.get("artifact_hashes"),
        label="completion manifest",
    )

    manifest = _load_hashed_document(
        manifest_path,
        hash_field="manifest_payload_hash",
        label="Corrected LOGO target manifest",
    )
    if (
        manifest.get("schema_version") != LOGO_SCHEMA_VERSION
        or manifest.get("status") != "corrected_revalidation"
        or manifest.get("split_method") != LOGO_SPLIT_METHOD
        or manifest.get("logo_target") != target
    ):
        raise ValueError("Corrected LOGO target manifest contract mismatch.")
    dataset = manifest.get("dataset")
    dependency = manifest.get("canonical_split_dependency")
    if (
        not isinstance(dataset, dict)
        or dataset.get("hash") != saved.dataset_hash
        or dataset.get("n_rows") != len(saved.canonical)
        or not isinstance(dependency, dict)
        or dependency.get("aggregate_split_hash") != saved.aggregate_split_hash
        or dependency.get("canonicalization_version")
        != CANONICALIZATION_VERSION
        or dependency.get("split_schema_version")
        != saved.manifest.get("split_schema_version")
        or dependency.get("artifact_hashes") != saved.artifact_hashes
        or dependency.get("per_seed_split_hashes")
        != {
            str(seed): split_hash
            for seed, split_hash in saved.per_seed_split_hashes.items()
        }
    ):
        raise ValueError("Corrected LOGO canonical data dependency mismatch.")
    _verify_declared_output_hashes(
        target_directory,
        manifest.get("output_hashes"),
        label="target manifest",
    )
    assignments_path = target_directory / "fold_assignments.csv"
    assignments = pd.read_csv(assignments_path)
    persisted = validate_canonical_logo_assignments(
        saved.canonical,
        assignments,
        target=target,
    )
    rebuilt = build_canonical_logo_contract(saved.canonical, target=target)
    if (
        persisted.aggregate_assignment_hash
        != rebuilt.aggregate_assignment_hash
        or persisted.per_fold_assignment_hashes
        != rebuilt.per_fold_assignment_hashes
        or manifest.get("aggregate_assignment_hash")
        != rebuilt.aggregate_assignment_hash
        or manifest.get("per_fold_assignment_hashes")
        != rebuilt.per_fold_assignment_hashes
        or manifest.get("heldout_groups")
        != [fold.fold_group for fold in rebuilt.folds]
        or manifest.get("expected_group_count") != rebuilt.expected_group_count
        or manifest.get("observed_fold_count") != rebuilt.fold_count
        or completion_target.get("aggregate_assignment_hash")
        != rebuilt.aggregate_assignment_hash
        or completion_target.get("fold_count") != rebuilt.fold_count
    ):
        raise ValueError(
            "Persisted corrected LOGO assignments do not equal canonical rebuild."
        )
    expected_fold_counts = completion.get("expected_fold_counts")
    observed_fold_counts = completion.get("observed_fold_counts")
    if (
        not isinstance(expected_fold_counts, dict)
        or not isinstance(observed_fold_counts, dict)
        or expected_fold_counts.get(target) != rebuilt.expected_group_count
        or observed_fold_counts.get(target) != rebuilt.fold_count
    ):
        raise ValueError("Corrected LOGO completion fold counts mismatch.")
    return _logo_units_from_contract(saved, persisted)


def _logo_units_from_contract(
    saved: SavedCanonicalSplits,
    contract: Any,
) -> tuple[EvaluationSplitUnit, ...]:
    """Convert an already validated canonical LOGO contract to units."""
    units = []
    all_ids = set(saved.canonical["source_row_id"].astype(str))
    for fold in contract.folds:
        units.append(
            _build_unit(
                canonical=saved.canonical,
                evaluation_unit=(
                    f"canonical-logo:{contract.target}:fold={fold.fold_index}:"
                    f"{stable_hash(fold.fold_group)[:16]}"
                ),
                split_family="canonical_logo",
                split_method=LOGO_SPLIT_METHOD,
                target=contract.target,
                fold_index=fold.fold_index,
                heldout_group=fold.fold_group,
                seed=None,
                train_fraction=None,
                train_ids=fold.train_source_ids,
                validation_ids=(),
                test_ids=fold.test_source_ids,
                excluded_ids=tuple(
                    sorted(
                        all_ids
                        - set(fold.train_source_ids)
                        - set(fold.test_source_ids)
                    )
                ),
                dataset_hash=saved.dataset_hash,
                exact_split_hash=fold.assignment_hash,
                aggregate_assignment_hash=contract.aggregate_assignment_hash,
                canonical_split_dependency_hash=saved.aggregate_split_hash,
                upstream_split_hash=None,
                split_schema_version=LOGO_SCHEMA_VERSION,
            )
        )
    return tuple(units)


def load_chemical_ood_split_units(
    artifact_directory: str | Path,
    *,
    dataset_path: str | Path,
    canonical_split_directory: str | Path,
    families: tuple[str, ...] | list[str] | None = None,
) -> tuple[EvaluationSplitUnit, ...]:
    """Validate and replay included Phase 9 chemistry-aware OOD folds."""
    manifest = validate_chemical_ood_artifacts(
        artifact_directory,
        dataset_path=dataset_path,
        canonical_split_directory=canonical_split_directory,
    )
    saved = load_saved_canonical_split_identities(
        dataset_path, canonical_split_directory
    )
    _validate_saved_dependencies(saved)
    _validate_chemical_dependencies(manifest, saved)
    requested = _resolve_families(families)
    directory = Path(artifact_directory)
    assignments = pd.read_csv(directory / "group_assignments.csv")
    folds = pd.read_csv(directory / "fold_definitions.csv")
    if tuple(assignments.columns) != ASSIGNMENT_COLUMNS:
        raise ValueError("Chemical OOD assignment schema changed after validation.")
    if tuple(folds.columns) != FOLD_COLUMNS:
        raise ValueError("Chemical OOD fold schema changed after validation.")
    assignments["source_row_id"] = assignments["source_row_id"].astype(str)
    assignments["canonical_reaction_key"] = assignments[
        "canonical_reaction_key"
    ].astype(str)
    _assert_exact_assignment_mapping(assignments, saved.canonical)
    canonical_ids = set(saved.canonical["source_row_id"].astype(str))
    units: list[EvaluationSplitUnit] = []
    for fold in folds.loc[
        folds["status"].eq("included") & folds["family"].isin(requested)
    ].to_dict(orient="records"):
        family = str(fold["family"])
        family_rows = assignments.loc[assignments["family"].eq(family)].copy()
        family_rows = family_rows.loc[
            family_rows["eligible"].map(_serialized_bool)
        ]
        if family == "maximum_similarity_bounded":
            train_ids = _ids(
                family_rows.loc[family_rows["group_value"].eq("train")]
            )
            test_ids = _ids(
                family_rows.loc[family_rows["group_value"].eq("test")]
            )
        else:
            heldout = str(fold["heldout_group"])
            test_ids = _ids(
                family_rows.loc[
                    family_rows["group_value"].astype(str).eq(heldout)
                ]
            )
            train_ids = tuple(sorted(set(_ids(family_rows)) - set(test_ids)))
        excluded_ids = tuple(
            sorted(canonical_ids - set(train_ids) - set(test_ids))
        )
        if (
            len(train_ids) != int(fold["n_train_rows"])
            or len(test_ids) != int(fold["n_test_rows"])
        ):
            raise ValueError(
                "Chemical OOD compact membership counts changed after validation."
            )
        upstream_hash = _optional_hash(fold["upstream_split_hash"])
        heldout_group = str(fold["heldout_group"])
        units.append(
            _build_unit(
                canonical=saved.canonical,
                evaluation_unit=(
                    f"chemical-ood:{family}:fold={int(fold['fold_index'])}:"
                    f"{stable_hash(heldout_group)[:16]}"
                ),
                split_family="chemical_ood",
                split_method="canonical_group_holdout",
                target=family,
                fold_index=int(fold["fold_index"]),
                heldout_group=heldout_group,
                seed=None,
                train_fraction=None,
                train_ids=train_ids,
                validation_ids=(),
                test_ids=test_ids,
                excluded_ids=excluded_ids,
                dataset_hash=saved.dataset_hash,
                exact_split_hash=str(fold["split_hash"]),
                aggregate_assignment_hash=str(manifest["manifest_hash"]),
                canonical_split_dependency_hash=str(
                    manifest["canonical_split_dependency_hash"]
                ),
                upstream_split_hash=upstream_hash,
                split_schema_version=CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION,
            )
        )
    units.sort(key=lambda unit: (requested.index(unit.target), unit.fold_index or 0))
    return tuple(units)


def _build_unit(
    *,
    canonical: pd.DataFrame,
    evaluation_unit: str,
    split_family: str,
    split_method: str,
    target: str,
    fold_index: int | None,
    heldout_group: str | None,
    seed: int | None,
    train_fraction: float | None,
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
    excluded_ids: tuple[str, ...],
    dataset_hash: str,
    exact_split_hash: str,
    aggregate_assignment_hash: str,
    canonical_split_dependency_hash: str,
    upstream_split_hash: str | None,
    split_schema_version: str,
) -> EvaluationSplitUnit:
    identity = _canonical_identity(canonical)
    expected_ids = set(identity)
    roles = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
        "excluded": set(excluded_ids),
    }
    if set().union(*roles.values()) != expected_ids:
        missing = sorted(expected_ids - set().union(*roles.values()))
        unexpected = sorted(set().union(*roles.values()) - expected_ids)
        raise ValueError(
            "Evaluation split does not exactly account for canonical source IDs: "
            f"missing={missing}, unexpected={unexpected}."
        )
    role_names = tuple(roles)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            if roles[left] & roles[right]:
                raise ValueError(
                    f"Evaluation split source IDs cross {left}/{right} roles."
                )
    key_roles = {
        role: {identity[source_id] for source_id in source_ids}
        for role, source_ids in roles.items()
        if role != "excluded"
    }
    key_names = tuple(key_roles)
    for index, left in enumerate(key_names):
        for right in key_names[index + 1 :]:
            if key_roles[left] & key_roles[right]:
                raise ValueError(
                    "A canonical reaction key crosses evaluation split roles: "
                    f"{left}/{right}."
                )
    return EvaluationSplitUnit(
        evaluation_unit=evaluation_unit,
        split_family=split_family,
        split_method=split_method,
        target=target,
        fold_index=fold_index,
        heldout_group=heldout_group,
        seed=seed,
        train_fraction=train_fraction,
        train_source_ids=tuple(sorted(train_ids)),
        validation_source_ids=tuple(sorted(validation_ids)),
        test_source_ids=tuple(sorted(test_ids)),
        excluded_source_ids=tuple(sorted(excluded_ids)),
        dataset_hash=dataset_hash,
        exact_split_hash=exact_split_hash,
        aggregate_assignment_hash=aggregate_assignment_hash,
        canonical_split_dependency_hash=canonical_split_dependency_hash,
        upstream_split_hash=upstream_split_hash,
        canonicalization_version=CANONICALIZATION_VERSION,
        split_schema_version=split_schema_version,
    )


def _validate_saved_dependencies(saved: SavedCanonicalSplits) -> None:
    if not isinstance(saved, SavedCanonicalSplits):
        raise TypeError("saved must be a validated SavedCanonicalSplits instance.")
    if saved.manifest.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise ValueError("Saved split canonicalization dependency mismatch.")
    if saved.dataset_hash != saved.manifest.get("canonical_dataset_hash"):
        raise ValueError("Saved split dataset dependency mismatch.")
    _require_hash(saved.dataset_hash, name="dataset_hash")
    _require_hash(saved.aggregate_split_hash, name="aggregate_split_hash")
    _canonical_identity(saved.canonical)


def _load_hashed_document(
    path: Path,
    *,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid: {path}.") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    payload = dict(document)
    recorded_hash = payload.pop(hash_field, None)
    if recorded_hash != stable_hash(payload):
        raise ValueError(f"{label} payload hash mismatch.")
    return document


def _verify_declared_output_hashes(
    directory: Path,
    declared: Any,
    *,
    label: str,
) -> None:
    if not isinstance(declared, dict) or "fold_assignments.csv" not in declared:
        raise ValueError(
            f"Corrected LOGO {label} must hash fold_assignments.csv."
        )
    for raw_name, expected_hash in declared.items():
        if (
            not isinstance(raw_name, str)
            or Path(raw_name).name != raw_name
            or raw_name in {"", ".", ".."}
        ):
            raise ValueError(f"Corrected LOGO {label} contains an unsafe path.")
        _require_hash(expected_hash, name=f"{label} output hash")
        path = directory / raw_name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(
                f"Corrected LOGO {label} output hash mismatch: {raw_name}."
            )


def _validate_chemical_dependencies(
    manifest: dict[str, Any],
    saved: SavedCanonicalSplits,
) -> None:
    if (
        manifest.get("schema_version")
        != CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION
        or manifest.get("dataset_hash") != saved.dataset_hash
        or manifest.get("canonical_split_dependency_hash")
        != saved.aggregate_split_hash
        or manifest.get("canonicalization_version")
        != CANONICALIZATION_VERSION
    ):
        raise ValueError("Chemical OOD representation split dependency mismatch.")
    _require_hash(manifest.get("manifest_hash"), name="manifest_hash")


def _canonical_identity(canonical: pd.DataFrame) -> dict[str, str]:
    required = {"source_row_id", "canonical_reaction_key"}
    missing = sorted(required - set(canonical))
    if missing:
        raise ValueError(f"Canonical split identities are missing columns: {missing}.")
    frame = canonical.loc[:, ["source_row_id", "canonical_reaction_key"]].copy()
    if frame.isna().any().any():
        raise ValueError("Canonical split identities contain missing values.")
    frame = frame.astype(str)
    if frame.apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Canonical split identities contain empty values.")
    if frame["source_row_id"].duplicated().any():
        raise ValueError("Canonical source_row_id values must be unique.")
    return dict(
        zip(
            frame["source_row_id"],
            frame["canonical_reaction_key"],
            strict=True,
        )
    )


def _resolve_families(
    families: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if families is None:
        return FAMILY_ORDER
    if (
        not isinstance(families, (tuple, list))
        or not families
        or any(not isinstance(value, str) for value in families)
    ):
        raise ValueError("families must be null or a non-empty string sequence.")
    requested = tuple(families)
    if len(requested) != len(set(requested)):
        raise ValueError("families must not contain duplicates.")
    unknown = sorted(set(requested) - set(FAMILY_ORDER))
    if unknown:
        raise ValueError(f"Unsupported chemical OOD families: {unknown}.")
    return tuple(family for family in FAMILY_ORDER if family in requested)


def _assert_exact_assignment_mapping(
    assignments: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    identity = _canonical_identity(canonical)
    for row in assignments.loc[
        :, ["family", "source_row_id", "canonical_reaction_key"]
    ].to_dict(orient="records"):
        source_id = str(row["source_row_id"])
        if identity.get(source_id) != str(row["canonical_reaction_key"]):
            raise ValueError(
                "Chemical OOD exact source-to-canonical-reaction-key mapping "
                f"failed for {row['family']} source {source_id}."
            )


def _ids(frame: pd.DataFrame) -> tuple[str, ...]:
    ids = tuple(sorted(frame["source_row_id"].astype(str)))
    if len(ids) != len(set(ids)):
        raise ValueError("Split membership contains duplicate source IDs.")
    return ids


def _serialized_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid serialized chemical OOD eligibility: {value!r}.")


def _optional_hash(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _require_hash(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hash.")
