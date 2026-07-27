"""Build and validate canonical chemistry-aware OOD split artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.canonicalize_roles import CANONICALIZATION_VERSION
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.condition_ood import (
    CONDITION_OOD_TARGET_COLUMNS,
    FoldSupportCriteria,
    add_condition_combination_keys,
    build_condition_ood_plan,
)
from bh_augmentation.evaluation.scaffold_ood import (
    AMINE_SMARTS,
    ELECTROPHILE_SMARTS,
    SCAFFOLD_SCHEMA_VERSION,
    build_scaffold_ood_assignments,
)
from bh_augmentation.evaluation.similarity_ood import (
    FINGERPRINT_EQUALITY_SEMANTICS,
    SIMILARITY_OOD_SCHEMA_VERSION,
    build_maximum_similarity_bounded_split,
    build_pairwise_reaction_similarity_index,
    build_reaction_fingerprint_cluster_holdouts,
)
from bh_augmentation.utils.corrected_runs import (
    prepare_fresh_output_directory,
    sha256_file,
    stable_hash,
)

CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION = "bh-canonical-chemical-ood-artifacts-v1"
FAMILY_ORDER = (
    "electrophile_scaffold",
    "amine_scaffold",
    "reaction_fingerprint_cluster",
    "maximum_similarity_bounded",
    "condition_combination",
    "ligand",
    "base",
)
ARTIFACT_FILES = (
    "group_assignments.csv",
    "fold_definitions.csv",
    "nearest_train_similarity.csv",
    "similarity_quantiles.csv",
    "fingerprint_contract.json",
)
ASSIGNMENT_COLUMNS = (
    "family",
    "source_row_id",
    "canonical_reaction_key",
    "group_value",
    "eligible",
    "exclusion_reason",
    "assignment_provenance",
)
FOLD_COLUMNS = (
    "family",
    "fold_index",
    "heldout_group",
    "status",
    "exclusion_reason",
    "n_train_rows",
    "n_test_rows",
    "n_train_groups",
    "n_test_groups",
    "group_overlap_count",
    "canonical_reaction_key_overlap_count",
    "split_hash",
    "upstream_split_hash",
    "configured_similarity_bound",
)
NEAREST_COLUMNS = (
    "family",
    "fold_index",
    "heldout_group",
    "source_row_id",
    "canonical_reaction_key",
    "nearest_train_canonical_reaction_key",
    "nearest_train_tanimoto",
    "query_hash",
)
QUANTILE_COLUMNS = (
    "family",
    "fold_index",
    "heldout_group",
    "q00",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q100",
    "maximum_test_to_train_similarity",
    "query_hash",
)
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChemicalOODConfig:
    """Resolved, conservative Phase 9 split configuration."""

    n_bits_per_role: int = 256
    radius: int = 2
    cluster_similarity_cutoff: float = 0.65
    maximum_similarity_bound: float = 0.95
    target_test_fraction: float = 0.2
    seed: int = 0
    min_train_rows: int = 100
    min_test_rows: int = 10
    min_train_groups: int = 2
    min_test_substrates: int = 2
    min_train_substrates: int = 10
    scaffold_min_train_groups: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            "n_bits_per_role",
            "radius",
            "min_train_rows",
            "min_test_rows",
            "min_train_groups",
            "min_test_substrates",
            "min_train_substrates",
            "scaffold_min_train_groups",
        )
        for name in integer_fields:
            value = getattr(self, name)
            minimum = 0 if name == "radius" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in (
            "cluster_similarity_cutoff",
            "maximum_similarity_bound",
            "target_test_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between zero and one.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ChemicalOODConfig:
        """Reject unknown configuration fields."""
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown chemical OOD configuration keys: {unknown}.")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready configuration."""
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def build_chemical_ood_artifacts(
    *,
    dataset_path: str | Path,
    canonical_split_directory: str | Path,
    output_directory: str | Path,
    config: ChemicalOODConfig,
) -> dict[str, Path]:
    """Build all Phase 9 split families and immutable audit artifacts."""
    saved = load_saved_canonical_splits(dataset_path, canonical_split_directory)
    canonical = saved.canonical.copy()
    canonical["source_row_id"] = canonical["source_row_id"].astype(str)
    canonical["canonical_reaction_key"] = canonical[
        "canonical_reaction_key"
    ].astype(str)
    if canonical["canonical_reaction_key"].duplicated().any():
        raise ValueError(
            "Chemical OOD artifacts require one row per canonical reaction key."
        )

    similarity_index = build_pairwise_reaction_similarity_index(
        canonical,
        n_bits_per_role=config.n_bits_per_role,
        radius=config.radius,
    )
    assignments: list[dict[str, Any]] = []
    fold_specs: list[dict[str, Any]] = []
    dependency_hashes: dict[str, str] = {}
    warnings: list[str] = []

    for family in ("electrophile_scaffold", "amine_scaffold"):
        scaffold = build_scaffold_ood_assignments(canonical, target=family)
        dependency_hashes[family] = scaffold.assignment_hash
        frame = scaffold.assignments
        for row in frame.to_dict(orient="records"):
            provenance = {
                key: row[key]
                for key in (
                    "assigned_role",
                    "assigned_canonical_smiles",
                    "molecule_identity_hash",
                    "bemis_murcko_scaffold_smiles",
                    "scaffold_hash",
                )
            }
            assignments.append(
                _assignment_record(
                    family,
                    row["source_row_id"],
                    row["canonical_reaction_key"],
                    row["scaffold_key"],
                    bool(row["included"]),
                    row["exclusion_reason"],
                    provenance,
                )
            )
        groups = tuple(sorted(scaffold.included_scaffold_groups))
        if len(groups) < 2:
            warnings.append(
                f"{family}: insufficient_distinct_scaffold_groups("
                f"observed={len(groups)},required=2)"
            )
            fold_specs.append(
                {
                    "family": family,
                    "fold_index": 0,
                    "heldout_group": "__target_excluded__",
                    "train_keys": (),
                    "test_keys": (),
                    "status": "excluded",
                    "exclusion_reason": (
                        "insufficient_distinct_scaffold_groups"
                        f"(observed={len(groups)},required=2)"
                    ),
                    "n_train_groups": max(0, len(groups) - 1),
                    "n_test_groups": min(1, len(groups)),
                    "upstream_split_hash": None,
                    "configured_similarity_bound": None,
                }
            )
            continue
        if family == "electrophile_scaffold" and len(groups) == 2:
            warnings.append(
                "electrophile_scaffold has only two broad cores; folds are "
                "descriptive fixed-policy benzene-versus-pyridine evidence, "
                "not nested scaffold-selection evidence."
            )
        included = frame.loc[frame["included"]]
        for fold_index, group in enumerate(groups):
            test = included.loc[included["scaffold_key"].eq(group)]
            train = included.loc[~included["scaffold_key"].eq(group)]
            reasons = _support_reasons(
                n_train_rows=len(train),
                n_test_rows=len(test),
                n_train_groups=len(groups) - 1,
                min_train_rows=config.min_train_rows,
                min_test_rows=config.min_test_rows,
                min_train_groups=config.scaffold_min_train_groups,
            )
            holdout = scaffold.holdout(group)
            fold_specs.append(
                {
                    "family": family,
                    "fold_index": fold_index,
                    "heldout_group": group,
                    "train_keys": tuple(
                        sorted(train["canonical_reaction_key"].astype(str))
                    ),
                    "test_keys": tuple(
                        sorted(test["canonical_reaction_key"].astype(str))
                    ),
                    "status": "excluded" if reasons else "included",
                    "exclusion_reason": "; ".join(reasons) if reasons else None,
                    "n_train_groups": len(groups) - 1,
                    "n_test_groups": 1,
                    "upstream_split_hash": holdout.split_hash,
                    "configured_similarity_bound": None,
                }
            )

    cluster = build_reaction_fingerprint_cluster_holdouts(
        canonical,
        n_bits_per_role=config.n_bits_per_role,
        radius=config.radius,
        similarity_cutoff=config.cluster_similarity_cutoff,
        seed=config.seed,
        min_train_rows=config.min_train_rows,
        min_test_rows=config.min_test_rows,
        min_train_canonical_groups=config.min_train_groups,
        min_test_canonical_groups=1,
    )
    dependency_hashes["reaction_fingerprint_cluster"] = (
        cluster.cluster_assignment_hash
    )
    cluster_map = cluster.cluster_assignments
    cluster_by_key = dict(
        zip(
            cluster_map["canonical_reaction_key"],
            cluster_map["cluster_id"],
            strict=True,
        )
    )
    fingerprint_hash_by_key = dict(
        zip(
            cluster_map["canonical_reaction_key"],
            cluster_map["reaction_fingerprint_hash"],
            strict=True,
        )
    )
    for row in canonical.to_dict(orient="records"):
        key = row["canonical_reaction_key"]
        assignments.append(
            _assignment_record(
                "reaction_fingerprint_cluster",
                row["source_row_id"],
                key,
                cluster_by_key[key],
                True,
                None,
                {"reaction_fingerprint_hash": fingerprint_hash_by_key[key]},
            )
        )
    for fold_index, split in enumerate(cluster.splits):
        test_keys = tuple(
            sorted(
                key
                for key, cluster_id in cluster_by_key.items()
                if cluster_id == split.ood_group
            )
        )
        train_keys = tuple(sorted(set(cluster_by_key) - set(test_keys)))
        fold_specs.append(
            {
                "family": "reaction_fingerprint_cluster",
                "fold_index": fold_index,
                "heldout_group": split.ood_group,
                "train_keys": train_keys,
                "test_keys": test_keys,
                "status": split.status,
                "exclusion_reason": split.exclusion_reason,
                "n_train_groups": split.n_train_canonical_groups,
                "n_test_groups": split.n_test_canonical_groups,
                "upstream_split_hash": split.split_hash,
                "configured_similarity_bound": None,
            }
        )

    bounded = build_maximum_similarity_bounded_split(
        canonical,
        maximum_similarity=config.maximum_similarity_bound,
        target_test_fraction=config.target_test_fraction,
        n_bits_per_role=config.n_bits_per_role,
        radius=config.radius,
        seed=config.seed,
        min_train_rows=config.min_train_rows,
        min_test_rows=config.min_test_rows,
        min_train_canonical_groups=config.min_train_groups,
        min_test_canonical_groups=1,
    )
    bounded_assignments = bounded.assignments
    dependency_hashes["maximum_similarity_bounded"] = str(
        bounded.split_hash or stable_hash(bounded.audit_record)
    )
    bounded_split_by_key = dict(
        zip(
            bounded_assignments["canonical_reaction_key"].astype(str),
            bounded_assignments["ood_split"].astype(str),
            strict=True,
        )
    )
    for row in canonical.to_dict(orient="records"):
        key = row["canonical_reaction_key"]
        assignments.append(
            _assignment_record(
                "maximum_similarity_bounded",
                row["source_row_id"],
                key,
                bounded_split_by_key.get(key, "excluded"),
                key in bounded_split_by_key,
                None if key in bounded_split_by_key else bounded.exclusion_reason,
                {"configured_similarity_bound": config.maximum_similarity_bound},
            )
        )
    fold_specs.append(
        {
            "family": "maximum_similarity_bounded",
            "fold_index": 0,
            "heldout_group": bounded.ood_group,
            "train_keys": tuple(
                sorted(
                    bounded_assignments.loc[
                        bounded_assignments["ood_split"].eq("train"),
                        "canonical_reaction_key",
                    ].astype(str)
                )
            ),
            "test_keys": tuple(
                sorted(
                    bounded_assignments.loc[
                        bounded_assignments["ood_split"].eq("test"),
                        "canonical_reaction_key",
                    ].astype(str)
                )
            ),
            "status": bounded.status,
            "exclusion_reason": bounded.exclusion_reason,
            "n_train_groups": bounded.n_train_canonical_groups,
            "n_test_groups": bounded.n_test_canonical_groups,
            "upstream_split_hash": bounded.split_hash,
            "configured_similarity_bound": config.maximum_similarity_bound,
        }
    )

    condition_frame = add_condition_combination_keys(canonical)
    condition_criteria = FoldSupportCriteria(
        min_test_samples=config.min_test_rows,
        min_train_samples=config.min_train_rows,
        min_train_groups=config.min_train_groups,
        min_test_substrates=config.min_test_substrates,
        min_train_substrates=config.min_train_substrates,
    )
    for family in ("condition_combination", "ligand", "base"):
        plan = build_condition_ood_plan(
            canonical,
            target=family,
            criteria=condition_criteria,
        )
        dependency_hashes[family] = plan.plan_hash
        group_column = CONDITION_OOD_TARGET_COLUMNS[family]
        for row in condition_frame.to_dict(orient="records"):
            assignments.append(
                _assignment_record(
                    family,
                    row["source_row_id"],
                    row["canonical_reaction_key"],
                    row[group_column],
                    True,
                    None,
                    {"group_column": group_column},
                )
            )
        key_by_source = dict(
            zip(
                condition_frame["source_row_id"].astype(str),
                condition_frame["canonical_reaction_key"].astype(str),
                strict=True,
            )
        )
        for fold in plan.folds:
            fold_specs.append(
                {
                    "family": family,
                    "fold_index": fold.fold_index,
                    "heldout_group": fold.heldout_group,
                    "train_keys": tuple(
                        sorted(key_by_source[source] for source in fold.train_source_ids)
                    ),
                    "test_keys": tuple(
                        sorted(key_by_source[source] for source in fold.test_source_ids)
                    ),
                    "status": "included" if fold.included else "excluded",
                    "exclusion_reason": fold.exclusion_reason,
                    "n_train_groups": fold.train_group_count,
                    "n_test_groups": 1,
                    "upstream_split_hash": fold.assignment_hash,
                    "configured_similarity_bound": None,
                }
            )

    assignments_frame = pd.DataFrame(
        assignments,
        columns=ASSIGNMENT_COLUMNS,
    ).sort_values(
        ["family", "source_row_id"],
        key=lambda series: (
            series.map({name: index for index, name in enumerate(FAMILY_ORDER)})
            if series.name == "family"
            else series
        ),
        kind="mergesort",
    ).reset_index(drop=True)
    source_by_key = dict(
        zip(
            canonical["canonical_reaction_key"],
            canonical["source_row_id"],
            strict=True,
        )
    )
    fold_rows: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    for spec in fold_specs:
        train_keys = tuple(spec.pop("train_keys"))
        test_keys = tuple(spec.pop("test_keys"))
        split_hash = stable_hash(
            {
                "schema_version": CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION,
                "family": spec["family"],
                "fold_index": spec["fold_index"],
                "heldout_group": spec["heldout_group"],
                "train_keys": list(train_keys),
                "test_keys": list(test_keys),
                "upstream_split_hash": spec["upstream_split_hash"],
                "fingerprint_metadata_hash": (
                    similarity_index.fingerprint_metadata_hash
                ),
            }
        )
        train_set = set(train_keys)
        test_set = set(test_keys)
        fold_rows.append(
            {
                **spec,
                "n_train_rows": len(train_keys),
                "n_test_rows": len(test_keys),
                "group_overlap_count": 0,
                "canonical_reaction_key_overlap_count": len(
                    train_set & test_set
                ),
                "split_hash": split_hash,
            }
        )
        if spec["status"] != "included":
            continue
        nearest = similarity_index.nearest_training_similarities(
            train_keys=train_keys,
            test_keys=test_keys,
        )
        for row in nearest.rows.to_dict(orient="records"):
            nearest_rows.append(
                {
                    "family": spec["family"],
                    "fold_index": spec["fold_index"],
                    "heldout_group": spec["heldout_group"],
                    "source_row_id": source_by_key[
                        row["canonical_reaction_key"]
                    ],
                    **row,
                    "query_hash": nearest.query_hash,
                }
            )
        quantile_rows.append(
            {
                "family": spec["family"],
                "fold_index": spec["fold_index"],
                "heldout_group": spec["heldout_group"],
                **nearest.similarity_quantiles,
                "maximum_test_to_train_similarity": nearest.maximum_similarity,
                "query_hash": nearest.query_hash,
            }
        )

    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    nearest_frame = pd.DataFrame(nearest_rows, columns=NEAREST_COLUMNS)
    quantiles_frame = pd.DataFrame(quantile_rows, columns=QUANTILE_COLUMNS)
    if not nearest_frame.empty:
        nearest_frame = nearest_frame.sort_values(
            ["family", "fold_index", "canonical_reaction_key"],
            kind="mergesort",
        ).reset_index(drop=True)
    if not folds_frame.empty:
        folds_frame = folds_frame.sort_values(
            ["family", "fold_index"], kind="mergesort"
        ).reset_index(drop=True)
    if not quantiles_frame.empty:
        quantiles_frame = quantiles_frame.sort_values(
            ["family", "fold_index"], kind="mergesort"
        ).reset_index(drop=True)

    output = prepare_fresh_output_directory(output_directory)
    assignments_frame.to_csv(output / ARTIFACT_FILES[0], index=False)
    folds_frame.to_csv(output / ARTIFACT_FILES[1], index=False)
    nearest_frame.to_csv(output / ARTIFACT_FILES[2], index=False)
    quantiles_frame.to_csv(output / ARTIFACT_FILES[3], index=False)
    fingerprint_contract = {
        **similarity_index.fingerprint_metadata,
        "similarity_ood_schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
        "fingerprint_equality_semantics": FINGERPRINT_EQUALITY_SEMANTICS,
        "fingerprint_table_hash": similarity_index.fingerprint_table_hash,
        "similarity_matrix_hash": similarity_index.similarity_matrix_hash,
        "similarity_index_hash": similarity_index.similarity_index_hash,
    }
    (output / ARTIFACT_FILES[4]).write_text(
        json.dumps(fingerprint_contract, indent=2, sort_keys=True) + "\n"
    )
    output_hashes = {
        name: sha256_file(output / name) for name in ARTIFACT_FILES
    }
    family_summary = {}
    for family in FAMILY_ORDER:
        family_folds = folds_frame.loc[folds_frame["family"].eq(family)]
        family_summary[family] = {
            "observed_fold_count": len(family_folds),
            "included_fold_count": int(
                family_folds["status"].eq("included").sum()
            ),
            "excluded_fold_count": int(
                family_folds["status"].eq("excluded").sum()
            ),
            "assignment_dependency_hash": dependency_hashes[family],
        }
    manifest = {
        "schema_version": CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "family_order": list(FAMILY_ORDER),
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "canonical_split_artifact_hashes": saved.artifact_hashes,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "dataset_rows": len(canonical),
        "source_row_count": canonical["source_row_id"].nunique(),
        "canonical_reaction_key_count": canonical[
            "canonical_reaction_key"
        ].nunique(),
        "config": config.to_dict(),
        "config_hash": stable_hash(config.to_dict()),
        "git_commit": _git_commit(),
        "group_definitions": {
            "electrophile_scaffold": {
                "schema_version": SCAFFOLD_SCHEMA_VERSION,
                "electrophile_smarts": ELECTROPHILE_SMARTS,
                "amine_smarts": AMINE_SMARTS,
                "scaffold": "direct_non_genericized_bemis_murcko",
            },
            "amine_scaffold": {
                "schema_version": SCAFFOLD_SCHEMA_VERSION,
                "electrophile_smarts": ELECTROPHILE_SMARTS,
                "amine_smarts": AMINE_SMARTS,
                "scaffold": "direct_non_genericized_bemis_murcko",
            },
            "reaction_fingerprint_cluster": {
                "method": "seeded_butina",
                "similarity_cutoff": config.cluster_similarity_cutoff,
                "seed": config.seed,
            },
            "maximum_similarity_bounded": {
                "method": "above_bound_connected_components",
                "maximum_similarity_bound": config.maximum_similarity_bound,
                "target_test_fraction": config.target_test_fraction,
                "seed": config.seed,
            },
            "condition_combination": {
                "canonical_role_order": [
                    "catalyst",
                    "ligand",
                    "base",
                    "solvent_or_additive",
                ]
            },
            "ligand": {"canonical_identity_column": "canonical_ligand_smiles"},
            "base": {"canonical_identity_column": "canonical_base_smiles"},
        },
        "family_summary": family_summary,
        "warnings": warnings,
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_chemical_ood_artifacts(
        output,
        dataset_path=dataset_path,
        canonical_split_directory=canonical_split_directory,
    )
    return {
        "output_directory": output,
        "manifest": output / "manifest.json",
        **{name: output / name for name in ARTIFACT_FILES},
    }


def validate_chemical_ood_artifacts(
    artifact_directory: str | Path,
    *,
    dataset_path: str | Path,
    canonical_split_directory: str | Path,
) -> dict[str, Any]:
    """Reject hash, coverage, membership, overlap, or similarity-audit defects."""
    directory = Path(artifact_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest_hash = manifest.pop("manifest_hash", None)
    if manifest_hash != stable_hash(manifest):
        raise ValueError("Chemical OOD manifest hash mismatch.")
    manifest["manifest_hash"] = manifest_hash
    if (
        manifest.get("schema_version") != CHEMICAL_OOD_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("family_order") != list(FAMILY_ORDER)
    ):
        raise ValueError("Chemical OOD manifest contract mismatch.")
    saved = load_saved_canonical_splits(dataset_path, canonical_split_directory)
    if (
        manifest["dataset_hash"] != saved.dataset_hash
        or manifest["canonical_split_dependency_hash"]
        != saved.aggregate_split_hash
    ):
        raise ValueError("Chemical OOD canonical data dependency mismatch.")
    for name in ARTIFACT_FILES:
        if sha256_file(directory / name) != manifest["output_hashes"].get(name):
            raise ValueError(f"Chemical OOD output hash mismatch: {name}.")
    assignments = pd.read_csv(directory / "group_assignments.csv")
    folds = pd.read_csv(directory / "fold_definitions.csv")
    nearest = pd.read_csv(directory / "nearest_train_similarity.csv")
    quantiles = pd.read_csv(directory / "similarity_quantiles.csv")
    _require_columns(assignments, ASSIGNMENT_COLUMNS, "group assignments")
    _require_columns(folds, FOLD_COLUMNS, "fold definitions")
    _require_columns(nearest, NEAREST_COLUMNS, "nearest similarities")
    _require_columns(quantiles, QUANTILE_COLUMNS, "similarity quantiles")
    canonical_sources = set(saved.canonical["source_row_id"].astype(str))
    canonical_keys = set(saved.canonical["canonical_reaction_key"].astype(str))
    assignments["source_row_id"] = assignments["source_row_id"].astype(str)
    _assert_exact_canonical_mapping(assignments, saved.canonical)
    if set(assignments["family"]) != set(FAMILY_ORDER):
        raise ValueError("Chemical OOD assignment families are incomplete.")
    for family in FAMILY_ORDER:
        family_assignments = assignments.loc[assignments["family"].eq(family)]
        if (
            len(family_assignments) != len(canonical_sources)
            or set(family_assignments["source_row_id"]) != canonical_sources
            or set(family_assignments["canonical_reaction_key"]) != canonical_keys
            or family_assignments["source_row_id"].duplicated().any()
        ):
            raise ValueError(f"Chemical OOD assignment coverage failed: {family}.")
    config = ChemicalOODConfig.from_mapping(manifest["config"])
    semantic = _rebuild_semantic_contract(saved.canonical, config)
    contract = json.loads((directory / "fingerprint_contract.json").read_text())
    if contract != semantic["fingerprint_contract"]:
        raise ValueError("Chemical OOD fingerprint contract replay failed.")
    _assert_group_assignment_replay(assignments, semantic["groups"])
    _assert_fold_replay(folds, semantic["folds"])

    included_folds = folds.loc[folds["status"].eq("included")]
    if not set(folds["status"]) <= {"included", "excluded"}:
        raise ValueError("Chemical OOD fold status is invalid.")
    if (
        (included_folds["n_train_rows"] <= 0).any()
        or (included_folds["n_test_rows"] <= 0).any()
        or (included_folds["canonical_reaction_key_overlap_count"] != 0).any()
        or (included_folds["group_overlap_count"] != 0).any()
    ):
        raise ValueError("Chemical OOD included fold support/overlap audit failed.")
    for fold in folds.to_dict(orient="records"):
        family_assignments = assignments.loc[
            assignments["family"].eq(fold["family"])
            & assignments["eligible"].map(_read_bool)
        ]
        if (
            fold["heldout_group"] == "__target_excluded__"
            and fold["status"] == "excluded"
        ):
            continue
        if fold["family"] == "maximum_similarity_bounded":
            test_keys = set(
                family_assignments.loc[
                    family_assignments["group_value"].eq("test"),
                    "canonical_reaction_key",
                ]
            )
            train_keys = set(
                family_assignments.loc[
                    family_assignments["group_value"].eq("train"),
                    "canonical_reaction_key",
                ]
            )
        else:
            test_keys = set(
                family_assignments.loc[
                    family_assignments["group_value"].eq(
                        str(fold["heldout_group"])
                    ),
                    "canonical_reaction_key",
                ]
            )
            train_keys = (
                set(family_assignments["canonical_reaction_key"]) - test_keys
            )
        if (
            len(train_keys) != int(fold["n_train_rows"])
            or len(test_keys) != int(fold["n_test_rows"])
            or train_keys & test_keys
        ):
            raise ValueError(
                f"Chemical OOD compact membership replay failed: "
                f"{fold['family']} fold {fold['fold_index']}."
            )
        if fold["status"] != "included":
            continue
        evidence = nearest.loc[
            nearest["family"].eq(fold["family"])
            & nearest["fold_index"].eq(fold["fold_index"])
        ]
        if (
            set(evidence["canonical_reaction_key"]) != test_keys
            or not set(evidence["nearest_train_canonical_reaction_key"])
            <= train_keys
            or evidence["canonical_reaction_key"].duplicated().any()
            or not evidence["nearest_train_tanimoto"].between(0.0, 1.0).all()
        ):
            raise ValueError(
                f"Chemical OOD nearest-training evidence failed: "
                f"{fold['family']} fold {fold['fold_index']}."
            )
        replay = semantic["similarity_index"].nearest_training_similarities(
            train_keys=train_keys,
            test_keys=test_keys,
        )
        _assert_nearest_replay(evidence, replay.rows, replay.query_hash)
        summary = quantiles.loc[
            quantiles["family"].eq(fold["family"])
            & quantiles["fold_index"].eq(fold["fold_index"])
        ]
        if len(summary) != 1:
            raise ValueError("Chemical OOD similarity summary coverage failed.")
        expected = np.quantile(
            evidence["nearest_train_tanimoto"].to_numpy(float),
            _QUANTILES,
        )
        observed = summary.loc[
            :,
            ["q00", "q25", "q50", "q75", "q90", "q95", "q100"],
        ].to_numpy(float)[0]
        if not np.allclose(expected, observed, rtol=0.0, atol=1e-12):
            raise ValueError("Chemical OOD similarity quantile replay failed.")
        bound = fold["configured_similarity_bound"]
        if (
            fold["family"] == "maximum_similarity_bounded"
            and not pd.isna(bound)
            and float(summary["maximum_test_to_train_similarity"].iloc[0])
            > float(bound) + 1e-12
        ):
            raise ValueError("Chemical OOD maximum-similarity bound was violated.")
    return manifest


def _assignment_record(
    family: str,
    source_row_id: Any,
    reaction_key: Any,
    group_value: Any,
    eligible: bool,
    exclusion_reason: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "source_row_id": str(source_row_id),
        "canonical_reaction_key": str(reaction_key),
        "group_value": group_value,
        "eligible": bool(eligible),
        "exclusion_reason": exclusion_reason,
        "assignment_provenance": json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), default=str
        ),
    }


def _support_reasons(
    *,
    n_train_rows: int,
    n_test_rows: int,
    n_train_groups: int,
    min_train_rows: int,
    min_test_rows: int,
    min_train_groups: int,
) -> list[str]:
    checks = (
        ("train_rows", n_train_rows, min_train_rows),
        ("test_rows", n_test_rows, min_test_rows),
        ("train_groups", n_train_groups, min_train_groups),
    )
    return [
        f"{name}_below_minimum(observed={observed},required={required})"
        for name, observed, required in checks
        if observed < required
    ]


def _require_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    label: str,
) -> None:
    if tuple(frame.columns) != expected:
        raise ValueError(
            f"Chemical OOD {label} columns mismatch: "
            f"expected={list(expected)}, observed={list(frame.columns)}."
        )


def _read_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid serialized boolean value: {value!r}.")


def _assert_exact_canonical_mapping(
    assignments: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    expected = dict(
        zip(
            canonical["source_row_id"].astype(str),
            canonical["canonical_reaction_key"].astype(str),
            strict=True,
        )
    )
    for row in assignments.loc[
        :,
        ["family", "source_row_id", "canonical_reaction_key"],
    ].to_dict(orient="records"):
        if expected.get(str(row["source_row_id"])) != str(
            row["canonical_reaction_key"]
        ):
            raise ValueError(
                "Chemical OOD exact source-to-canonical-key mapping failed: "
                f"{row['family']} source {row['source_row_id']}."
            )


def _rebuild_semantic_contract(
    canonical: pd.DataFrame,
    config: ChemicalOODConfig,
) -> dict[str, Any]:
    frame = canonical.copy()
    frame["source_row_id"] = frame["source_row_id"].astype(str)
    frame["canonical_reaction_key"] = frame["canonical_reaction_key"].astype(str)
    groups: dict[str, dict[str, tuple[Any, bool, Any]]] = {}
    folds: dict[tuple[str, int], tuple[Any, ...]] = {}
    for family in ("electrophile_scaffold", "amine_scaffold"):
        scaffold = build_scaffold_ood_assignments(frame, target=family)
        groups[family] = {
            str(row["source_row_id"]): (
                row["scaffold_key"],
                bool(row["included"]),
                row["exclusion_reason"],
            )
            for row in scaffold.assignments.to_dict(orient="records")
        }
        scaffold_groups = tuple(sorted(scaffold.included_scaffold_groups))
        if len(scaffold_groups) < 2:
            folds[(family, 0)] = (
                "__target_excluded__",
                "excluded",
                "insufficient_distinct_scaffold_groups"
                f"(observed={len(scaffold_groups)},required=2)",
                0,
                0,
                max(0, len(scaffold_groups) - 1),
                min(1, len(scaffold_groups)),
            )
        else:
            included = scaffold.included
            for fold_index, group in enumerate(scaffold_groups):
                test = included.loc[included["scaffold_key"].eq(group)]
                train = included.loc[~included["scaffold_key"].eq(group)]
                reasons = _support_reasons(
                    n_train_rows=len(train),
                    n_test_rows=len(test),
                    n_train_groups=len(scaffold_groups) - 1,
                    min_train_rows=config.min_train_rows,
                    min_test_rows=config.min_test_rows,
                    min_train_groups=config.scaffold_min_train_groups,
                )
                folds[(family, fold_index)] = (
                    group,
                    "excluded" if reasons else "included",
                    "; ".join(reasons) if reasons else None,
                    len(train),
                    len(test),
                    len(scaffold_groups) - 1,
                    1,
                )
    cluster = build_reaction_fingerprint_cluster_holdouts(
        frame,
        n_bits_per_role=config.n_bits_per_role,
        radius=config.radius,
        similarity_cutoff=config.cluster_similarity_cutoff,
        seed=config.seed,
        min_train_rows=config.min_train_rows,
        min_test_rows=config.min_test_rows,
        min_train_canonical_groups=config.min_train_groups,
        min_test_canonical_groups=1,
    )
    cluster_by_key = dict(
        zip(
            cluster.cluster_assignments["canonical_reaction_key"],
            cluster.cluster_assignments["cluster_id"],
            strict=True,
        )
    )
    groups["reaction_fingerprint_cluster"] = {
        str(row["source_row_id"]): (
            cluster_by_key[str(row["canonical_reaction_key"])],
            True,
            None,
        )
        for row in frame.to_dict(orient="records")
    }
    for fold_index, split in enumerate(cluster.splits):
        n_test = sum(
            cluster_id == split.ood_group
            for cluster_id in cluster_by_key.values()
        )
        folds[("reaction_fingerprint_cluster", fold_index)] = (
            split.ood_group,
            split.status,
            split.exclusion_reason,
            len(cluster_by_key) - n_test,
            n_test,
            split.n_train_canonical_groups,
            split.n_test_canonical_groups,
        )
    bounded = build_maximum_similarity_bounded_split(
        frame,
        maximum_similarity=config.maximum_similarity_bound,
        target_test_fraction=config.target_test_fraction,
        n_bits_per_role=config.n_bits_per_role,
        radius=config.radius,
        seed=config.seed,
        min_train_rows=config.min_train_rows,
        min_test_rows=config.min_test_rows,
        min_train_canonical_groups=config.min_train_groups,
        min_test_canonical_groups=1,
    )
    bounded_by_key = dict(
        zip(
            bounded.assignments["canonical_reaction_key"].astype(str),
            bounded.assignments["ood_split"].astype(str),
            strict=True,
        )
    )
    groups["maximum_similarity_bounded"] = {
        str(row["source_row_id"]): (
            bounded_by_key.get(str(row["canonical_reaction_key"]), "excluded"),
            str(row["canonical_reaction_key"]) in bounded_by_key,
            (
                None
                if str(row["canonical_reaction_key"]) in bounded_by_key
                else bounded.exclusion_reason
            ),
        )
        for row in frame.to_dict(orient="records")
    }
    folds[("maximum_similarity_bounded", 0)] = (
        bounded.ood_group,
        bounded.status,
        bounded.exclusion_reason,
        bounded.n_train_rows,
        bounded.n_test_rows,
        bounded.n_train_canonical_groups,
        bounded.n_test_canonical_groups,
    )
    condition = add_condition_combination_keys(frame)
    criteria = FoldSupportCriteria(
        min_test_samples=config.min_test_rows,
        min_train_samples=config.min_train_rows,
        min_train_groups=config.min_train_groups,
        min_test_substrates=config.min_test_substrates,
        min_train_substrates=config.min_train_substrates,
    )
    for family in ("condition_combination", "ligand", "base"):
        column = CONDITION_OOD_TARGET_COLUMNS[family]
        groups[family] = {
            str(row["source_row_id"]): (row[column], True, None)
            for row in condition.to_dict(orient="records")
        }
        plan = build_condition_ood_plan(frame, target=family, criteria=criteria)
        for fold in plan.folds:
            folds[(family, fold.fold_index)] = (
                fold.heldout_group,
                "included" if fold.included else "excluded",
                fold.exclusion_reason,
                fold.train_size,
                fold.test_size,
                fold.train_group_count,
                1,
            )
    similarity_index = build_pairwise_reaction_similarity_index(
        frame,
        n_bits_per_role=config.n_bits_per_role,
        radius=config.radius,
    )
    fingerprint_contract = {
        **similarity_index.fingerprint_metadata,
        "similarity_ood_schema_version": SIMILARITY_OOD_SCHEMA_VERSION,
        "fingerprint_equality_semantics": FINGERPRINT_EQUALITY_SEMANTICS,
        "fingerprint_table_hash": similarity_index.fingerprint_table_hash,
        "similarity_matrix_hash": similarity_index.similarity_matrix_hash,
        "similarity_index_hash": similarity_index.similarity_index_hash,
    }
    return {
        "groups": groups,
        "folds": folds,
        "similarity_index": similarity_index,
        "fingerprint_contract": fingerprint_contract,
    }


def _assert_group_assignment_replay(
    assignments: pd.DataFrame,
    expected_groups: dict[str, dict[str, tuple[Any, bool, Any]]],
) -> None:
    for row in assignments.to_dict(orient="records"):
        expected_group, expected_eligible, expected_reason = expected_groups[
            str(row["family"])
        ][str(row["source_row_id"])]
        observed_group = row["group_value"]
        groups_equal = (
            pd.isna(observed_group) and expected_group is None
        ) or str(observed_group) == str(expected_group)
        observed_reason = row["exclusion_reason"]
        reasons_equal = (
            pd.isna(observed_reason) and expected_reason is None
        ) or str(observed_reason) == str(expected_reason)
        if (
            not groups_equal
            or _read_bool(row["eligible"]) != expected_eligible
            or not reasons_equal
        ):
            raise ValueError(
                "Chemical OOD group assignment semantic replay failed: "
                f"{row['family']} source {row['source_row_id']}."
            )


def _assert_nearest_replay(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    query_hash: str,
) -> None:
    left = observed.sort_values(
        "canonical_reaction_key", kind="mergesort"
    ).reset_index(drop=True)
    right = expected.sort_values(
        "canonical_reaction_key", kind="mergesort"
    ).reset_index(drop=True)
    if (
        set(left["query_hash"].astype(str)) != {query_hash}
        or left["canonical_reaction_key"].astype(str).tolist()
        != right["canonical_reaction_key"].astype(str).tolist()
        or left["nearest_train_canonical_reaction_key"].astype(str).tolist()
        != right["nearest_train_canonical_reaction_key"].astype(str).tolist()
        or not np.allclose(
            left["nearest_train_tanimoto"].to_numpy(float),
            right["nearest_train_tanimoto"].to_numpy(float),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("Chemical OOD nearest-training semantic replay failed.")


def _assert_fold_replay(
    observed: pd.DataFrame,
    expected: dict[tuple[str, int], tuple[Any, ...]],
) -> None:
    observed_keys = {
        (str(row["family"]), int(row["fold_index"]))
        for row in observed.to_dict(orient="records")
    }
    if observed_keys != set(expected):
        raise ValueError("Chemical OOD fold-definition coverage replay failed.")
    for row in observed.to_dict(orient="records"):
        key = (str(row["family"]), int(row["fold_index"]))
        (
            heldout_group,
            status,
            exclusion_reason,
            n_train,
            n_test,
            n_train_groups,
            n_test_groups,
        ) = expected[key]
        observed_reason = row["exclusion_reason"]
        reasons_equal = (
            pd.isna(observed_reason) and exclusion_reason is None
        ) or str(observed_reason) == str(exclusion_reason)
        if (
            str(row["heldout_group"]) != str(heldout_group)
            or str(row["status"]) != status
            or not reasons_equal
            or int(row["n_train_rows"]) != n_train
            or int(row["n_test_rows"]) != n_test
            or int(row["n_train_groups"]) != n_train_groups
            or int(row["n_test_groups"]) != n_test_groups
        ):
            raise ValueError(
                f"Chemical OOD fold-definition semantic replay failed: {key}."
            )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"
