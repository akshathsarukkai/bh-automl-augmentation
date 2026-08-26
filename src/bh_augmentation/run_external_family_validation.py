"""External reaction-family validation: search phase then final evaluation.

Phase 16 status
---------------
``implementation passed``
``external empirical validation blocked``

This runner is complete and exercised end to end, but only against a clearly
labelled synthetic fixture (``tests/suzuki_fixture.py``). No licensed
Suzuki-Miyaura dataset is present in this repository, so no empirical
Suzuki-Miyaura result is produced or claimed. Acquiring real data requires a
human with network access; see ``docs/EXTERNAL_DATASETS.md``.

Everything family-specific is supplied by a
:class:`~bh_augmentation.data.reaction_family.ReactionFamilyAdapter`. The protocol
itself is the existing, family-agnostic machinery:

* grouped splits -- :func:`bh_augmentation.data.canonical_splits.build_grouped_outer_assignments`
  and :func:`~bh_augmentation.data.canonical_splits.build_nested_low_data_assignments`;
* validation-only search, policy freezing, and single-shot outer-test evaluation
  -- :mod:`bh_augmentation.evaluation.policy_protocol`;
* metrics -- :func:`bh_augmentation.evaluation.metrics.regression_metrics`;
* hashing and manifests -- :mod:`bh_augmentation.utils.corrected_runs`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from bh_augmentation.augmentation.candidate_scope import (
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    resolved_candidate_scope_mode,
)
from bh_augmentation.augmentation.family_condition_transfer import (
    assert_condition_transfer_invariants,
    build_condition_transfer_candidates,
)
from bh_augmentation.data.canonical_splits import (
    SPLIT_SCHEMA_VERSION,
    build_grouped_outer_assignments,
    build_nested_low_data_assignments,
    split_assignment_hash,
)
from bh_augmentation.data.external_dataset_config import (
    EXTERNAL_FAMILY_CONFIG_SCHEMA_VERSION,
    build_adapter_from_config,
    validate_external_family_config,
)
from bh_augmentation.data.reaction_family import ReactionFamilyAdapter
from bh_augmentation.evaluation.metrics import regression_metrics
from bh_augmentation.evaluation.policy_protocol import (
    FinalEvaluationInputs,
    PolicySearchInputs,
    ResolvedPolicy,
    ScientificBinding,
    evaluate_frozen_policy_once,
    freeze_policy,
    run_policy_search,
)
from bh_augmentation.features.family_role_features import family_role_separated_features
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    sha256_file,
    stable_hash,
    write_json,
)

EXTERNAL_VALIDATION_RUN_SCHEMA_VERSION = "external-family-validation-run-v1"
REPORTED_METRICS = ("rmse", "mae", "r2", "spearman")

#: ``(phase, evaluation_unit, source_row_ids)`` observer used by leakage tests.
FitRowObserver = Callable[[str, str, "list[str]"], None]


def run_external_family_validation(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    fit_row_observer: FitRowObserver | None = None,
) -> dict[str, Path]:
    """Run search and final evaluation for one external reaction family."""
    resolved = validate_external_family_config(config)
    adapter = build_adapter_from_config(resolved)

    dataset_path = Path(str(resolved["dataset"]["path"]))
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"External {adapter.family_name} dataset is not present: {dataset_path}. "
            "This repository ships no licensed dataset for this family; obtain it "
            "manually and record its provenance, then rerun. See "
            "docs/EXTERNAL_DATASETS.md."
        )
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != adapter.provenance.raw_file_sha256:
        raise ValueError(
            "Dataset file hash does not match the declared provenance "
            f"raw_file_sha256: observed {dataset_hash}, declared "
            f"{adapter.provenance.raw_file_sha256}."
        )

    output_dir = _prepare_output_directory(resolved["output"]["directory"])
    frame = pd.read_csv(dataset_path, nrows=resolved["dataset"].get("nrows"))
    yield_column = str(resolved["dataset"].get("yield_column", "yield"))
    if yield_column not in frame.columns:
        raise ValueError(f"Dataset is missing the yield column {yield_column!r}.")

    canonical = adapter.canonicalize_dataframe(
        frame,
        source_file_hash=dataset_hash,
        isomeric=bool(resolved["canonicalization"].get("isomeric_smiles", True)),
        source_row_positions=range(len(frame)),
    )
    canonical["yield"] = pd.to_numeric(canonical[yield_column], errors="coerce")
    eligibility_audit = canonical[
        [
            "source_row_id",
            "source_row_position",
            "all_required_roles_parse_valid",
            "family_eligible",
            "family_eligibility_reason",
        ]
    ].copy()
    usable = canonical.loc[
        canonical["family_eligible"].astype(bool) & canonical["yield"].notna()
    ].reset_index(drop=True)
    if len(usable) < 12:
        raise ValueError(
            "External family validation needs at least 12 eligible measured rows; "
            f"got {len(usable)}."
        )

    features, feature_names, feature_metadata = family_role_separated_features(
        usable,
        adapter,
        n_bits=int(resolved["features"]["n_bits"]),
        radius=int(resolved["features"].get("radius", 2)),
        fingerprint_backend=str(resolved["features"]["fingerprint_backend"]),
    )
    feature_contract = feature_contract_record(feature_metadata, feature_names)

    split_config = resolved["splits"]
    transfer_config = dict(resolved.get("condition_transfer", {}) or {})
    selection_metric = str(resolved["selection"]["metric"])
    lower_is_better = bool(resolved["selection"].get("lower_is_better", True))
    policies = _resolved_policies(transfer_config, adapter)

    search_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    overlap_records: list[dict[str, Any]] = []
    frozen_records: list[dict[str, Any]] = []
    evaluated_units: set[str] = set()
    split_hashes: dict[str, str] = {}

    for seed_value in split_config["seeds"]:
        seed = int(seed_value)
        outer = build_grouped_outer_assignments(
            usable,
            seed=seed,
            train_size=float(split_config["train_size"]),
            valid_size=float(split_config["valid_size"]),
            test_size=float(split_config["test_size"]),
            group_column="canonical_reaction_key",
        )
        low = build_nested_low_data_assignments(
            outer,
            train_fractions=[float(v) for v in split_config.get("train_fractions", [1.0])],
            group_column="canonical_reaction_key",
        )
        split_hashes[str(seed)] = split_assignment_hash(outer, low)
        overlap_records.append(_overlap_record(seed, outer))

        for fraction, subset in low.groupby("train_fraction", sort=True):
            unit = f"family={adapter.family_id}|seed={seed}|train_fraction={float(fraction)}"
            candidate_scope_mode = resolved_candidate_scope_mode(resolved)
            partitions = _partitions(usable, outer, subset)
            binding = ScientificBinding(
                dataset_hash=dataset_hash,
                split_hash=split_hashes[str(seed)],
                split_aggregate_hash=stable_hash(split_hashes),
                per_seed_split_hash=split_hashes[str(seed)],
                source_id_split_hash=stable_hash(
                    {
                        name: sorted(part["source_row_id"].astype(str).tolist())
                        for name, part in partitions.items()
                    }
                ),
                canonicalization_version=adapter.canonicalization_version,
                split_schema_version=SPLIT_SCHEMA_VERSION,
                feature_metadata_hash=str(feature_contract["feature_metadata_hash"]),
                config_hash=stable_hash(resolved),
                commit_hash=stable_hash(adapter.provenance_record()),
            )
            context = _UnitContext(
                adapter=adapter,
                features=features,
                feature_names=feature_names,
                feature_metadata=feature_metadata,
                partitions=partitions,
                # External-family validation is a low-data augmentation-benefit
                # experiment, so candidate eligibility follows the same rule as
                # the in-repo low-data runners: a candidate is ineligible only
                # when the training partition the learner sees already contains
                # that chemistry. `usable` spans train, valid and test, so using
                # it here would decide eligibility from partitions the learner
                # never observed. Declare `candidate_scope.mode:
                # globally_unmeasured_prospective` for a novelty-claim run.
                measured_keys=_family_eligibility_keys(
                    candidate_scope_mode,
                    adapter=adapter,
                    partitions=partitions,
                    usable=usable,
                ),
                seed=seed,
                unit=unit,
                fit_row_observer=fit_row_observer,
                candidate_frames=candidate_frames,
                feature_config=dict(resolved["features"]),
            )

            search_result = run_policy_search(
                PolicySearchInputs(
                    train=partitions["train"],
                    validation=partitions["valid"],
                    binding=binding,
                ),
                policies,
                context.validation_evaluator,
                selection_metric=selection_metric,
                lower_is_better=lower_is_better,
            )
            search_metrics = search_result.search_metrics.copy()
            search_metrics["evaluation_unit"] = unit
            search_frames.append(search_metrics)

            frozen = freeze_policy(
                search_result,
                training_protocol={
                    "refit_rows": "outer_train_subset_only",
                    "augmentation_scope": "training_rows_only",
                    "teacher_fit_scope": "training_rows_only",
                    "test_access": "single_shot_after_freeze",
                },
                search_manifest_hash=stable_hash(
                    {
                        "unit": unit,
                        "search": search_metrics.drop(columns=["selected_policy"]).to_dict(
                            orient="records"
                        ),
                    }
                ),
            )
            frozen_path = output_dir / f"frozen_policy__{_slug(unit)}.json"
            frozen.write_json(frozen_path)
            frozen_records.append(
                {
                    "evaluation_unit": unit,
                    "policy_id": frozen.resolved_policy.policy_id,
                    "frozen_policy_hash": frozen.frozen_policy_hash,
                    "selection_metric": frozen.selection_metric,
                    "selection_value": frozen.selection_value,
                    "frozen_policy_path": frozen_path.name,
                }
            )

            final_result = evaluate_frozen_policy_once(
                FinalEvaluationInputs(
                    refit=partitions["train"],
                    outer_test=partitions["test"],
                    binding=binding,
                    evaluation_unit=unit,
                ),
                frozen,
                context.outer_test_evaluator,
                evaluated_units=evaluated_units,
            )
            final_frames.append(final_result.metrics)

    paths = _output_paths(output_dir)
    pd.concat(search_frames, ignore_index=True).to_csv(paths["search_metrics"], index=False)
    pd.concat(final_frames, ignore_index=True).to_csv(paths["final_metrics"], index=False)
    pd.DataFrame(overlap_records).to_csv(paths["split_overlap_audit"], index=False)
    eligibility_audit.to_csv(paths["eligibility_audit"], index=False)
    candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(columns=["evaluation_unit", "accepted"])
    )
    candidates.to_csv(paths["candidate_audit"], index=False)

    manifest = {
        "run_schema_version": EXTERNAL_VALIDATION_RUN_SCHEMA_VERSION,
        "config_schema_version": EXTERNAL_FAMILY_CONFIG_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_hash": stable_hash(resolved),
        "resolved_config": resolved,
        "reaction_family": adapter.provenance_record(),
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "n_source_rows": int(len(frame)),
        "n_eligible_rows": int(len(usable)),
        "feature_contract": feature_contract,
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "split_hashes": split_hashes,
        "split_aggregate_hash": stable_hash(split_hashes),
        "frozen_policies": frozen_records,
        "evaluation_units": sorted(evaluated_units),
        "n_test_evaluations": len(evaluated_units),
        "synthetic_candidates_accepted": int(
            candidates["accepted"].astype(bool).sum() if len(candidates) else 0
        ),
        "empirical_status": "synthetic_fixture_or_user_supplied_dataset",
        "historical_results_loaded": False,
    }
    manifest["output_file_hashes"] = {
        path.name: sha256_file(path)
        for key, path in sorted(paths.items())
        if key != "manifest" and path.exists()
    }
    manifest["output_hash"] = stable_hash(manifest["output_file_hashes"])
    write_json(paths["manifest"], manifest)
    return paths


def _family_eligibility_keys(
    mode: str,
    *,
    adapter: ReactionFamilyAdapter,
    partitions: Mapping[str, pd.DataFrame],
    usable: pd.DataFrame,
) -> set[str]:
    """Resolve the identity universe an external-family run rejects against."""
    if mode == GLOBALLY_UNMEASURED_PROSPECTIVE:
        return set(adapter.measured_canonical_keys(usable))
    return set(adapter.measured_canonical_keys(partitions["train"]))


class _UnitContext:
    """Per-unit fitting closure shared by the search and final phases."""

    def __init__(
        self,
        *,
        adapter: ReactionFamilyAdapter,
        features: np.ndarray,
        feature_names: Sequence[str],
        feature_metadata: Any,
        partitions: Mapping[str, pd.DataFrame],
        measured_keys: set[str],
        seed: int,
        unit: str,
        fit_row_observer: FitRowObserver | None,
        candidate_frames: list[pd.DataFrame],
        feature_config: Mapping[str, Any],
    ) -> None:
        self.adapter = adapter
        self.features = features
        self.feature_names = list(feature_names)
        self.feature_metadata = feature_metadata
        self.partitions = dict(partitions)
        self.measured_keys = set(measured_keys)
        self.seed = int(seed)
        self.unit = unit
        self.fit_row_observer = fit_row_observer
        self.candidate_frames = candidate_frames
        self.feature_config = dict(feature_config)

    def validation_evaluator(self, inputs: PolicySearchInputs, policy: ResolvedPolicy):
        """Fit on training rows only and score the validation partition."""
        model = self._fit(policy, phase="search")
        return self._metric_rows(model, self.partitions["valid"], "valid")

    def outer_test_evaluator(self, inputs: FinalEvaluationInputs, frozen: Any):
        """Refit under the frozen policy and score the outer test partition once."""
        model = self._fit(frozen.resolved_policy, phase="final")
        return self._metric_rows(model, self.partitions["test"], "test")

    # ------------------------------------------------------------------
    def _fit(self, policy: ResolvedPolicy, *, phase: str) -> Ridge:
        train = self.partitions["train"]
        if self.fit_row_observer is not None:
            self.fit_row_observer(
                phase, self.unit, train["source_row_id"].astype(str).tolist()
            )
        X_train = self.features[train["feature_position"].to_numpy(dtype=int)]
        y_train = train["yield"].to_numpy(dtype=float)
        alpha = float(policy.config.get("ridge_alpha", 1.0))
        multiplier = float(policy.config.get("synthetic_multiplier", 0.0))
        if multiplier > 0:
            X_synth, y_synth = self._synthetic_rows(policy, phase=phase)
            if len(y_synth):
                limit = max(1, int(round(multiplier * len(y_train))))
                X_train = np.vstack([X_train, X_synth[:limit]])
                y_train = np.concatenate([y_train, y_synth[:limit]])
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def _synthetic_rows(
        self,
        policy: ResolvedPolicy,
        *,
        phase: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        train = self.partitions["train"]
        candidates = build_condition_transfer_candidates(
            train,
            self.adapter,
            seed=self.seed,
            max_candidates_per_source=int(policy.config.get("max_candidates_per_source", 2)),
            transferable_roles=tuple(policy.config.get("transferable_roles", ())) or None,
            measured_canonical_keys=self.measured_keys,
        )
        assert_condition_transfer_invariants(candidates, self.adapter)
        audit = candidates.copy()
        audit["evaluation_unit"] = self.unit
        audit["policy_id"] = policy.policy_id
        audit["phase"] = phase
        self.candidate_frames.append(
            audit[
                [
                    "evaluation_unit",
                    "policy_id",
                    "phase",
                    "source_row_id",
                    "donor_row_id",
                    "canonical_reaction_key",
                    "changed_roles",
                    "rejection_reason",
                    "accepted",
                ]
            ]
        )
        accepted = candidates.loc[candidates["accepted"].astype(bool)]
        if accepted.empty:
            return np.empty((0, self.features.shape[1]), dtype=np.float32), np.empty(0)

        X_synth, names, metadata = family_role_separated_features(
            accepted,
            self.adapter,
            n_bits=int(self.feature_config["n_bits"]),
            radius=int(self.feature_config.get("radius", 2)),
            fingerprint_backend=str(self.feature_config["fingerprint_backend"]),
        )
        if names != self.feature_names or metadata != self.feature_metadata:
            raise ValueError(
                "Synthetic feature semantics differ from the measured representation."
            )
        # Teacher labels come from training rows only.
        teacher = Ridge(alpha=float(policy.config.get("teacher_alpha", 1.0)))
        teacher.fit(
            self.features[train["feature_position"].to_numpy(dtype=int)],
            train["yield"].to_numpy(dtype=float),
        )
        y_synth = np.clip(teacher.predict(X_synth), 0.0, 100.0)
        return X_synth, y_synth

    def _metric_rows(self, model: Ridge, partition: pd.DataFrame, split: str):
        X = self.features[partition["feature_position"].to_numpy(dtype=int)]
        y_true = partition["yield"].to_numpy(dtype=float)
        values = regression_metrics(y_true, model.predict(X))
        return [
            {"split": split, "metric": name, "value": float(values[name])}
            for name in REPORTED_METRICS
            if np.isfinite(values[name])
        ]


def _resolved_policies(
    transfer_config: Mapping[str, Any],
    adapter: ReactionFamilyAdapter,
) -> list[ResolvedPolicy]:
    base = {
        "ridge_alpha": 1.0,
        "teacher_alpha": 1.0,
        "synthetic_multiplier": 0.0,
        "transferable_roles": [],
        "max_candidates_per_source": 0,
    }
    policies = [ResolvedPolicy("no_augmentation", dict(base))]
    if not transfer_config.get("enabled", False):
        return policies
    roles = [
        str(value)
        for value in transfer_config.get("transferable_roles", adapter.transferable_roles)
    ]
    max_candidates = int(transfer_config.get("max_candidates_per_source", 2))
    for multiplier in transfer_config.get("synthetic_multipliers", []):
        value = float(multiplier)
        if value <= 0:
            continue
        policies.append(
            ResolvedPolicy(
                f"typed_condition_transfer|multiplier={value:g}",
                {
                    **base,
                    "synthetic_multiplier": value,
                    "transferable_roles": roles,
                    "max_candidates_per_source": max_candidates,
                },
            )
        )
    return policies


def _partitions(
    usable: pd.DataFrame,
    outer: pd.DataFrame,
    low_subset: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frame = usable.copy()
    frame["feature_position"] = np.arange(len(frame), dtype=int)
    assignment = outer.set_index("source_row_id")["outer_split"]
    frame["outer_split"] = frame["source_row_id"].map(assignment)
    included = set(
        low_subset.loc[
            low_subset["included_in_training_subset"].astype(bool), "source_row_id"
        ].astype(str)
    )
    train = frame.loc[
        frame["outer_split"].eq("train") & frame["source_row_id"].astype(str).isin(included)
    ].reset_index(drop=True)
    valid = frame.loc[frame["outer_split"].eq("valid")].reset_index(drop=True)
    test = frame.loc[frame["outer_split"].eq("test")].reset_index(drop=True)
    for name, part in {"train": train, "valid": valid, "test": test}.items():
        if part.empty:
            raise ValueError(f"Partition {name!r} is empty for this evaluation unit.")
    keys = {
        name: set(part["canonical_reaction_key"].astype(str))
        for name, part in {"train": train, "valid": valid, "test": test}.items()
    }
    if keys["train"] & keys["valid"] or keys["train"] & keys["test"] or keys["valid"] & keys["test"]:
        raise AssertionError("A canonical reaction key crosses train/valid/test partitions.")
    return {"train": train, "valid": valid, "test": test}


def _overlap_record(seed: int, outer: pd.DataFrame) -> dict[str, Any]:
    groups = {
        split: set(outer.loc[outer["outer_split"].eq(split), "canonical_reaction_key"])
        for split in ("train", "valid", "test")
    }
    return {
        "seed": int(seed),
        "train_valid_group_overlap": len(groups["train"] & groups["valid"]),
        "train_test_group_overlap": len(groups["train"] & groups["test"]),
        "valid_test_group_overlap": len(groups["valid"] & groups["test"]),
        "all_outer_group_overlaps_zero": not (
            groups["train"] & groups["valid"]
            or groups["train"] & groups["test"]
            or groups["valid"] & groups["test"]
        ),
        "all_rows_assigned_once": outer["source_row_id"].nunique() == len(outer),
    }


def _prepare_output_directory(path: str | Path) -> Path:
    directory = Path(path)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty external validation directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _output_paths(directory: Path) -> dict[str, Path]:
    return {
        "search_metrics": directory / "search_metrics.csv",
        "final_metrics": directory / "final_test_metrics.csv",
        "split_overlap_audit": directory / "split_overlap_audit.csv",
        "eligibility_audit": directory / "family_eligibility_audit.csv",
        "candidate_audit": directory / "synthetic_candidate_audit.csv",
        "manifest": directory / "run_manifest.json",
    }


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    paths = run_external_family_validation(load_config(args.config), config_path=args.config)
    print(f"Search metrics: {paths['search_metrics']}")
    print(f"Final test metrics: {paths['final_metrics']}")
    print(f"Run manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
