"""Matched no-augmentation representation benchmark across random and OOD splits."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bh_augmentation.data.saved_canonical_splits import (
    load_saved_canonical_split_identities,
)
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.evaluation.representation_splits import (
    EvaluationSplitUnit,
    build_saved_random_split_unit,
    load_chemical_ood_split_units,
    load_corrected_logo_split_units,
)
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    sha256_file,
    stable_hash,
)

REPRESENTATION_BENCHMARK_SCHEMA_VERSION = "bh-no-augmentation-representations-v1"
REPRESENTATION_SPECS = (
    ("flat_reaction_sum", "bh_flat_reaction_sum"),
    ("reaction_section_concatenation", "bh_reaction_section_concat"),
    ("seven_role_blocks", "bh_role_separated"),
    ("seven_blocks_plus_deltas", "bh_role_separated_delta"),
    ("substrate_only", "bh_substrate_only"),
    ("condition_only", "bh_condition_only"),
    ("product_aware", "bh_product_aware"),
    ("product_free", "bh_product_free"),
)
REPRESENTATION_IDS = tuple(item[0] for item in REPRESENTATION_SPECS)
_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman_corr}
_OUTPUTS = (
    "benchmark_plan.json",
    "split_units.csv",
    "representation_metadata.csv",
    "predictions.csv",
    "metrics.csv",
    "summary.csv",
)


def run_representation_benchmark(
    config: str | Path | Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Evaluate every predefined representation without test-based selection."""
    resolved = _load_config(config)
    contract = _resolve_contract(resolved)
    saved = load_saved_canonical_split_identities(
        contract["dataset_path"],
        contract["canonical_split_directory"],
        requested_seeds=contract["random_seeds"],
        requested_fractions=contract["random_fractions"],
    )
    units = _build_units(saved, contract)
    if not units:
        raise ValueError("Representation benchmark has no evaluation units.")
    model_protocol = {
        "name": contract["model"]["name"],
        "params": contract["model"]["params"],
        "fit_protocol": "real_only_train_partition_no_validation_or_test",
        "seed_derivation": "base_seed_plus_stable_evaluation_unit_hash_mod_1e6",
    }
    model_protocol_hash = stable_hash(model_protocol)
    scientific_config = {
        "dataset_path": str(contract["dataset_path"]),
        "canonical_split_directory": str(contract["canonical_split_directory"]),
        "logo_artifact_directory": str(contract["logo_artifact_directory"]),
        "chemical_ood_directory": str(contract["chemical_ood_directory"]),
        "random_seeds": list(contract["random_seeds"]),
        "random_fractions": list(contract["random_fractions"]),
        "logo_targets": list(contract["logo_targets"]),
        "chemical_ood_families": list(contract["chemical_ood_families"]),
        "feature_config": contract["feature_config"],
        "model": contract["model"],
        "metrics": list(contract["metrics"]),
        "base_seed": contract["base_seed"],
    }
    plan = {
        "schema_version": REPRESENTATION_BENCHMARK_SCHEMA_VERSION,
        "status": "frozen_before_label_loading",
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "representations": [
            {"benchmark_id": benchmark_id, "feature_kind": kind}
            for benchmark_id, kind in REPRESENTATION_SPECS
        ],
        "representation_selection": "none_all_predefined_methods_reported",
        "model_protocol": model_protocol,
        "model_protocol_hash": model_protocol_hash,
        "resolved_scientific_config": scientific_config,
        "config_hash": stable_hash(scientific_config),
        "metrics": list(contract["metrics"]),
        "split_units": [unit.audit_record for unit in units],
        "test_used_for_selection": False,
        "test_evaluations_planned_per_unit_representation": 1,
        "product_aware_semantics": (
            "intentional seven-role-block duplicate control of seven_role_blocks; "
            "not independent evidence"
        ),
    }
    plan["plan_hash"] = stable_hash(plan)

    output = Path(
        output_directory
        if output_directory is not None
        else contract["output_directory"]
    )
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite representation output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths = {name.removesuffix(".json").removesuffix(".csv"): output / name for name in _OUTPUTS}
    paths["output_directory"] = output
    paths["manifest"] = output / "manifest.json"
    paths["benchmark_plan"].write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame([unit.audit_record for unit in units]).to_csv(
        paths["split_units"], index=False
    )

    canonical = pd.read_csv(contract["dataset_path"])
    if sha256_file(contract["dataset_path"]) != saved.dataset_hash:
        raise ValueError("Canonical dataset changed after benchmark plan freeze.")
    identity_columns = list(saved.canonical.columns)
    if not canonical.loc[:, identity_columns].equals(
        saved.canonical.reset_index(drop=True)
    ):
        raise ValueError("Canonical identities changed after benchmark plan freeze.")
    if "yield" not in canonical:
        raise ValueError("Canonical labels unavailable after benchmark plan freeze.")
    canonical["source_row_id"] = canonical["source_row_id"].astype(str)
    by_source = canonical.set_index("source_row_id", drop=False)
    source_positions = {
        source_id: position
        for position, source_id in enumerate(canonical["source_row_id"])
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    metadata_seen: dict[str, dict[str, Any]] = {}
    identity_only = canonical.copy()
    identity_only["yield"] = 0.0
    for benchmark_id, feature_kind in REPRESENTATION_SPECS:
        feature_config = {
            **contract["feature_config"],
            "kind": feature_kind,
        }
        X, _, names, metadata = build_feature_matrix_with_metadata(
            identity_only,
            feature_config,
        )
        X = np.asarray(X, dtype=np.float32)
        train_contract = feature_contract_record(metadata, names)
        metadata_seen[benchmark_id] = train_contract
        for unit in units:
            train = by_source.loc[
                list(unit.train_source_ids)
            ].reset_index(drop=True)
            test = by_source.loc[list(unit.test_source_ids)].reset_index(drop=True)
            train_positions = np.asarray(
                [
                    source_positions[source_id]
                    for source_id in unit.train_source_ids
                ],
                dtype=int,
            )
            test_positions = np.asarray(
                [
                    source_positions[source_id]
                    for source_id in unit.test_source_ids
                ],
                dtype=int,
            )
            y_train = pd.to_numeric(
                train["yield"], errors="raise"
            ).to_numpy(dtype=np.float32)
            model_seed = contract["base_seed"] + (
                int(stable_hash(unit.evaluation_unit)[:12], 16) % 1_000_000
            )
            fit_instance_hash = stable_hash(
                {
                    "model_protocol_hash": model_protocol_hash,
                    "model_seed": model_seed,
                    "exact_split_hash": unit.exact_split_hash,
                }
            )
            estimator = train_model(
                get_model(
                    contract["model"]["name"],
                    seed=model_seed,
                    **contract["model"]["params"],
                ),
                X[train_positions],
                y_train,
            )
            prediction = np.asarray(
                predict_model(estimator, X[test_positions]),
                dtype=float,
            )
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Non-finite predictions for {unit.evaluation_unit}/{benchmark_id}."
                )
            prediction_rows.extend(
                {
                    "evaluation_unit": unit.evaluation_unit,
                    "benchmark_id": benchmark_id,
                    "source_row_id": source_id,
                    "prediction": float(value),
                }
                for source_id, value in zip(
                    unit.test_source_ids,
                    prediction,
                    strict=True,
                )
            )
            evidence_class = _evidence_class(unit)
            y_test = pd.to_numeric(test["yield"], errors="raise").to_numpy(
                dtype=np.float32
            )
            for metric_name in contract["metrics"]:
                value = float(
                    _METRICS[metric_name](
                        np.asarray(y_test, dtype=float),
                        prediction,
                    )
                )
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite {metric_name} for "
                        f"{unit.evaluation_unit}/{benchmark_id}."
                    )
                metric_rows.append(
                    {
                        "evaluation_unit": unit.evaluation_unit,
                        "evidence_class": evidence_class,
                        "split_family": unit.split_family,
                        "split_target": unit.target,
                        "fold_index": unit.fold_index,
                        "heldout_group": unit.heldout_group,
                        "seed": unit.seed,
                        "train_fraction": unit.train_fraction,
                        "benchmark_id": benchmark_id,
                        "feature_kind": feature_kind,
                        "metric": metric_name,
                        "value": value,
                        "n_train": len(train),
                        "n_test": len(test),
                        "exact_split_hash": unit.exact_split_hash,
                        "aggregate_assignment_hash": (
                            unit.aggregate_assignment_hash
                        ),
                        "model_protocol_hash": model_protocol_hash,
                        "model_seed": model_seed,
                        "fit_instance_hash": fit_instance_hash,
                        "feature_metadata_hash": train_contract[
                            "feature_metadata_hash"
                        ],
                        "test_evaluation_count": 1,
                        "test_used_for_selection": False,
                        "plan_hash": plan["plan_hash"],
                    }
                )

    for benchmark_id, feature_kind in REPRESENTATION_SPECS:
        record = metadata_seen[benchmark_id]
        metadata_rows.append(
            {
                "benchmark_id": benchmark_id,
                "feature_kind": feature_kind,
                "representation_kind": record["representation_kind"],
                "feature_width": record["total_width"],
                "role_ordering": "|".join(record["role_ordering"]),
                "block_names": "|".join(
                    block["name"] for block in record["block_slices"]
                ),
                "feature_name_hash": record["feature_name_hash"],
                "feature_metadata_hash": record["feature_metadata_hash"],
                "product_information_included": benchmark_id not in {
                    "substrate_only",
                    "condition_only",
                    "product_free",
                },
                "product_aware_duplicate_control": benchmark_id == "product_aware",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metadata = pd.DataFrame(metadata_rows)
    summary = _summarize(metrics)
    metadata.to_csv(paths["representation_metadata"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    _assert_paired_metrics(metrics)
    output_hashes = {name: sha256_file(output / name) for name in _OUTPUTS}
    manifest = {
        "schema_version": REPRESENTATION_BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": saved.dataset_hash,
        "canonical_split_dependency_hash": saved.aggregate_split_hash,
        "plan_hash": plan["plan_hash"],
        "model_protocol_hash": model_protocol_hash,
        "config_hash": plan["config_hash"],
        "resolved_scientific_config": scientific_config,
        "command": " ".join(sys.argv),
        "dependency_versions": _dependency_versions(),
        "logo_artifact_directory": str(contract["logo_artifact_directory"]),
        "chemical_ood_directory": str(contract["chemical_ood_directory"]),
        "evaluation_unit_count": len(units),
        "representation_count": len(REPRESENTATION_SPECS),
        "metric_row_count": len(metrics),
        "prediction_row_count": len(predictions),
        "selection_performed": False,
        "test_used_for_selection": False,
        "product_aware_duplicate_control": True,
        "output_hashes": output_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_representation_benchmark(output)
    return paths


def validate_representation_benchmark(directory: str | Path) -> dict[str, Any]:
    """Validate immutable hashes and complete paired benchmark coverage."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed = manifest.pop("manifest_hash", None)
    if claimed != stable_hash(manifest):
        raise ValueError("Representation benchmark manifest hash mismatch.")
    manifest["manifest_hash"] = claimed
    if (
        manifest.get("schema_version") != REPRESENTATION_BENCHMARK_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("selection_performed") is not False
        or manifest.get("test_used_for_selection") is not False
        or int(manifest.get("representation_count", -1))
        != len(REPRESENTATION_IDS)
    ):
        raise ValueError("Representation benchmark manifest contract mismatch.")
    for name, digest in manifest["output_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Representation benchmark output hash mismatch: {name}.")
    plan = json.loads((root / "benchmark_plan.json").read_text())
    plan_hash = plan.pop("plan_hash", None)
    if plan_hash != stable_hash(plan):
        raise ValueError("Representation benchmark plan hash mismatch.")
    plan["plan_hash"] = plan_hash
    if (
        plan.get("status") != "frozen_before_label_loading"
        or tuple(item["benchmark_id"] for item in plan["representations"])
        != REPRESENTATION_IDS
        or plan.get("model_protocol_hash")
        != stable_hash(plan.get("model_protocol"))
        or plan.get("config_hash")
        != stable_hash(plan.get("resolved_scientific_config"))
    ):
        raise ValueError("Representation benchmark frozen plan mismatch.")
    scientific = plan["resolved_scientific_config"]
    replay_saved = load_saved_canonical_split_identities(
        scientific["dataset_path"],
        scientific["canonical_split_directory"],
        requested_seeds=scientific["random_seeds"],
        requested_fractions=scientific["random_fractions"],
    )
    replay_contract = {
        "dataset_path": Path(scientific["dataset_path"]),
        "canonical_split_directory": Path(
            scientific["canonical_split_directory"]
        ),
        "logo_artifact_directory": Path(scientific["logo_artifact_directory"]),
        "chemical_ood_directory": Path(scientific["chemical_ood_directory"]),
        "random_seeds": tuple(scientific["random_seeds"]),
        "random_fractions": tuple(scientific["random_fractions"]),
        "logo_targets": tuple(scientific["logo_targets"]),
        "chemical_ood_families": tuple(scientific["chemical_ood_families"]),
    }
    replay_units = _build_units(replay_saved, replay_contract)
    if stable_hash([unit.audit_record for unit in replay_units]) != stable_hash(
        plan["split_units"]
    ):
        raise ValueError("Representation frozen split plan upstream replay failed.")
    if (
        plan["dataset_hash"] != replay_saved.dataset_hash
        or plan["canonical_split_dependency_hash"]
        != replay_saved.aggregate_split_hash
        or
        manifest["plan_hash"] != plan_hash
        or manifest["model_protocol_hash"] != plan["model_protocol_hash"]
        or manifest["dataset_hash"] != plan["dataset_hash"]
        or manifest["canonical_split_dependency_hash"]
        != plan["canonical_split_dependency_hash"]
        or int(manifest["evaluation_unit_count"]) != len(replay_units)
    ):
        raise ValueError("Representation top-level provenance linkage mismatch.")
    metrics = pd.read_csv(root / "metrics.csv")
    if (
        len(metrics) != int(manifest["metric_row_count"])
        or len(metrics)
        != len(plan["split_units"])
        * len(REPRESENTATION_IDS)
        * len(plan["metrics"])
        or set(metrics["metric"]) != set(plan["metrics"])
        or set(metrics["plan_hash"]) != {plan_hash}
        or set(metrics["model_protocol_hash"])
        != {plan["model_protocol_hash"]}
        or manifest["config_hash"] != plan["config_hash"]
        or manifest["resolved_scientific_config"]
        != plan["resolved_scientific_config"]
    ):
        raise ValueError("Representation metric count/plan linkage mismatch.")
    _assert_paired_metrics(metrics)
    _assert_split_unit_table(root / "split_units.csv", plan["split_units"])
    metadata = pd.read_csv(root / "representation_metadata.csv")
    if tuple(metadata["benchmark_id"]) != REPRESENTATION_IDS:
        raise ValueError("Representation metadata coverage/order mismatch.")
    canonical = pd.read_csv(scientific["dataset_path"], nrows=1)
    if sha256_file(scientific["dataset_path"]) != plan["dataset_hash"]:
        raise ValueError("Representation benchmark canonical dataset mismatch.")
    canonical["yield"] = 0.0
    expected_metadata: dict[str, dict[str, Any]] = {}
    for benchmark_id, feature_kind in REPRESENTATION_SPECS:
        X, _, names, feature_metadata = build_feature_matrix_with_metadata(
            canonical,
            {**scientific["feature_config"], "kind": feature_kind},
        )
        if len(X) != 1:
            raise ValueError("Representation metadata replay shape mismatch.")
        expected_metadata[benchmark_id] = feature_contract_record(
            feature_metadata, names
        )
    for row in metadata.to_dict(orient="records"):
        expected = expected_metadata[row["benchmark_id"]]
        if (
            row["feature_kind"] != expected["representation_kind"]
            or int(row["feature_width"]) != int(expected["total_width"])
            or row["feature_name_hash"] != expected["feature_name_hash"]
            or row["feature_metadata_hash"] != expected["feature_metadata_hash"]
            or str(row["role_ordering"])
            != "|".join(expected["role_ordering"])
            or str(row["block_names"])
            != "|".join(block["name"] for block in expected["block_slices"])
        ):
            raise ValueError("Representation metadata semantic replay failed.")
    product_free = metadata.set_index("benchmark_id").loc["product_free"]
    if (
        bool(product_free["product_information_included"])
        or "product" in str(product_free["role_ordering"])
        or "product" in str(product_free["block_names"])
        or "delta" in str(product_free["block_names"])
    ):
        raise ValueError("Product-free representation contains product information.")
    feature_hashes = dict(
        zip(
            metadata["benchmark_id"],
            metadata["feature_metadata_hash"],
            strict=True,
        )
    )
    unit_by_name = {unit.evaluation_unit: unit for unit in replay_units}
    kind_by_id = dict(REPRESENTATION_SPECS)
    for row in metrics.to_dict(orient="records"):
        unit = unit_by_name.get(row["evaluation_unit"])
        if unit is None:
            raise ValueError("Representation metric references unknown split unit.")
        expected_seed = int(plan["resolved_scientific_config"]["base_seed"]) + (
            int(stable_hash(row["evaluation_unit"])[:12], 16) % 1_000_000
        )
        expected_fit_hash = stable_hash(
            {
                "model_protocol_hash": plan["model_protocol_hash"],
                "model_seed": expected_seed,
                "exact_split_hash": unit.exact_split_hash,
            }
        )
        if (
            row["exact_split_hash"] != unit.exact_split_hash
            or int(row["n_train"]) != len(unit.train_source_ids)
            or int(row["n_test"]) != len(unit.test_source_ids)
            or int(row["model_seed"]) != expected_seed
            or row["fit_instance_hash"] != expected_fit_hash
            or row["feature_metadata_hash"]
            != feature_hashes[row["benchmark_id"]]
            or row["feature_kind"] != kind_by_id[row["benchmark_id"]]
            or row["evidence_class"] != _evidence_class(unit)
            or row["split_family"] != unit.split_family
            or row["split_target"] != unit.target
            or not _same_nullable(row["fold_index"], unit.fold_index)
            or not _same_nullable(row["heldout_group"], unit.heldout_group)
            or not _same_nullable(row["seed"], unit.seed)
            or not _same_nullable(row["train_fraction"], unit.train_fraction)
            or row["aggregate_assignment_hash"]
            != unit.aggregate_assignment_hash
        ):
            raise ValueError("Representation metric semantic linkage mismatch.")
    predictions = pd.read_csv(root / "predictions.csv")
    _assert_prediction_replay(
        predictions,
        metrics,
        replay_units,
        scientific["dataset_path"],
        plan["metrics"],
        manifest,
    )
    observed_summary = pd.read_csv(root / "summary.csv")
    expected_summary = _summarize(metrics)
    pd.testing.assert_frame_equal(
        observed_summary,
        expected_summary,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    paired = metrics.loc[
        metrics["benchmark_id"].isin(["seven_role_blocks", "product_aware"])
    ].pivot_table(
        index=["evaluation_unit", "metric"],
        columns="benchmark_id",
        values="value",
    )
    if not np.allclose(
        paired["seven_role_blocks"],
        paired["product_aware"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Intentional product-aware duplicate control diverged.")
    return manifest


def _assert_paired_metrics(metrics: pd.DataFrame) -> None:
    required = {
        "evaluation_unit",
        "benchmark_id",
        "metric",
        "exact_split_hash",
        "model_protocol_hash",
        "model_seed",
        "test_evaluation_count",
        "test_used_for_selection",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Representation metrics missing fields: {missing}.")
    for unit, rows in metrics.groupby("evaluation_unit", sort=False):
        if (
            set(rows["benchmark_id"]) != set(REPRESENTATION_IDS)
            or rows["exact_split_hash"].nunique() != 1
            or rows["model_protocol_hash"].nunique() != 1
            or rows["model_seed"].nunique() != 1
            or not rows["test_evaluation_count"].eq(1).all()
            or rows["test_used_for_selection"].map(_false_value).any()
        ):
            raise ValueError(f"Unpaired representation evidence for {unit}.")
        counts = rows.groupby(["benchmark_id", "metric"]).size()
        if not counts.eq(1).all():
            raise ValueError(f"Repeated representation test metric for {unit}.")


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep random training fractions separate while aggregating OOD folds."""
    return (
        metrics.groupby(
            [
                "evidence_class",
                "split_target",
                "train_fraction",
                "benchmark_id",
                "metric",
            ],
            dropna=False,
        )["value"]
        .agg(mean="mean", median="median", std="std", count="count")
        .reset_index()
    )


def _assert_split_unit_table(
    path: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    observed = pd.read_csv(path)
    if len(observed) != len(expected_rows) or observed[
        "evaluation_unit"
    ].duplicated().any():
        raise ValueError("Representation split-unit table coverage mismatch.")
    observed_by_name = observed.set_index("evaluation_unit", drop=False)
    fields = (
        "split_family",
        "split_method",
        "target",
        "n_train",
        "n_validation",
        "n_test",
        "n_excluded",
        "train_source_id_hash",
        "validation_source_id_hash",
        "test_source_id_hash",
        "excluded_source_id_hash",
        "dataset_hash",
        "exact_split_hash",
        "aggregate_assignment_hash",
        "canonical_split_dependency_hash",
        "canonicalization_version",
        "split_schema_version",
    )
    for expected in expected_rows:
        name = expected["evaluation_unit"]
        if name not in observed_by_name.index:
            raise ValueError("Representation split unit is missing from table.")
        row = observed_by_name.loc[name]
        for field in fields:
            left = row[field]
            right = expected[field]
            if field.startswith("n_"):
                equal = int(left) == int(right)
            else:
                equal = str(left) == str(right)
            if not equal:
                raise ValueError(
                    f"Representation split-unit semantic mismatch: {name}/{field}."
                )


def _assert_prediction_replay(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    units: tuple[EvaluationSplitUnit, ...],
    dataset_path: str,
    metric_names: list[str],
    manifest: dict[str, Any],
) -> None:
    expected_columns = (
        "evaluation_unit",
        "benchmark_id",
        "source_row_id",
        "prediction",
    )
    if tuple(predictions.columns) != expected_columns:
        raise ValueError("Representation prediction schema mismatch.")
    predictions["source_row_id"] = predictions["source_row_id"].astype(str)
    expected_count = sum(len(unit.test_source_ids) for unit in units) * len(
        REPRESENTATION_IDS
    )
    if (
        len(predictions) != expected_count
        or int(manifest["prediction_row_count"]) != expected_count
        or not np.isfinite(predictions["prediction"].to_numpy(float)).all()
    ):
        raise ValueError("Representation prediction coverage mismatch.")
    outcomes = pd.read_csv(
        dataset_path,
        usecols=["source_row_id", "yield"],
        dtype={"source_row_id": str},
    )
    outcome_by_source = dict(
        zip(outcomes["source_row_id"], outcomes["yield"], strict=True)
    )
    for unit in units:
        for benchmark_id in REPRESENTATION_IDS:
            rows = predictions.loc[
                predictions["evaluation_unit"].eq(unit.evaluation_unit)
                & predictions["benchmark_id"].eq(benchmark_id)
            ].sort_values("source_row_id", kind="mergesort")
            if (
                len(rows) != len(unit.test_source_ids)
                or rows["source_row_id"].duplicated().any()
                or tuple(rows["source_row_id"]) != unit.test_source_ids
            ):
                raise ValueError(
                    "Representation prediction test-membership replay failed."
                )
            y_true = np.asarray(
                [outcome_by_source[source] for source in rows["source_row_id"]],
                dtype=np.float32,
            )
            y_pred = rows["prediction"].to_numpy(float)
            for metric_name in metric_names:
                expected = float(_METRICS[metric_name](y_true, y_pred))
                metric = metrics.loc[
                    metrics["evaluation_unit"].eq(unit.evaluation_unit)
                    & metrics["benchmark_id"].eq(benchmark_id)
                    & metrics["metric"].eq(metric_name),
                    "value",
                ]
                if len(metric) != 1 or not np.isclose(
                    float(metric.iloc[0]),
                    expected,
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise ValueError("Representation metric prediction replay failed.")
    duplicate = predictions.loc[
        predictions["benchmark_id"].isin(
            ["seven_role_blocks", "product_aware"]
        )
    ].pivot(
        index=["evaluation_unit", "source_row_id"],
        columns="benchmark_id",
        values="prediction",
    )
    if not np.array_equal(
        duplicate["seven_role_blocks"].to_numpy(float),
        duplicate["product_aware"].to_numpy(float),
    ):
        raise ValueError("Product-aware duplicate prediction control diverged.")


def _same_nullable(left: Any, right: Any) -> bool:
    if right is None:
        return bool(pd.isna(left))
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    return str(left) == str(right)


def _build_units(saved: Any, contract: dict[str, Any]) -> tuple[EvaluationSplitUnit, ...]:
    units: list[EvaluationSplitUnit] = []
    for seed in contract["random_seeds"]:
        for fraction in contract["random_fractions"]:
            units.append(
                build_saved_random_split_unit(
                    saved,
                    seed=seed,
                    train_fraction=fraction,
                )
            )
    for target in contract["logo_targets"]:
        units.extend(
            load_corrected_logo_split_units(
                saved,
                logo_root_directory=contract["logo_artifact_directory"],
                target=target,
            )
        )
    units.extend(
        load_chemical_ood_split_units(
            contract["chemical_ood_directory"],
            dataset_path=contract["dataset_path"],
            canonical_split_directory=contract["canonical_split_directory"],
            families=contract["chemical_ood_families"],
        )
    )
    names = [unit.evaluation_unit for unit in units]
    if len(names) != len(set(names)):
        raise ValueError("Representation evaluation unit IDs are not unique.")
    return tuple(units)


def _resolve_contract(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "dataset",
        "splits",
        "representations",
        "features",
        "model",
        "metrics",
        "base_seed",
        "output",
    }
    if set(config) != expected:
        raise ValueError(
            f"Representation config keys mismatch: "
            f"missing={sorted(expected-set(config))}, "
            f"unknown={sorted(set(config)-expected)}."
        )
    reps = config["representations"]
    if not isinstance(reps, list) or tuple(reps) != REPRESENTATION_IDS:
        raise ValueError(
            "representations must contain the exact eight predefined IDs in order."
        )
    splits = _mapping(config["splits"], "splits")
    split_keys = {
        "canonical_directory",
        "random_seeds",
        "random_fractions",
        "logo_targets",
        "logo_artifact_directory",
        "chemical_ood_directory",
        "chemical_ood_families",
    }
    if set(splits) != split_keys:
        raise ValueError("Representation splits schema mismatch.")
    metrics = config["metrics"]
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(name not in _METRICS for name in metrics)
        or len(metrics) != len(set(metrics))
    ):
        raise ValueError("Representation metrics are invalid or duplicated.")
    feature_config = _mapping(config["features"], "features")
    if (
        set(feature_config) != {"n_bits", "radius", "fingerprint_backend"}
        or feature_config["fingerprint_backend"] != "rdkit"
    ):
        raise ValueError("Representation features require explicit RDKit settings.")
    model = _mapping(config["model"], "model")
    if set(model) != {"name", "params"} or not isinstance(model["params"], dict):
        raise ValueError("Representation model schema mismatch.")
    dataset = _mapping(config["dataset"], "dataset")
    output = _mapping(config["output"], "output")
    if set(dataset) != {"path"} or set(output) != {"directory"}:
        raise ValueError("Representation dataset/output schema mismatch.")
    random_seeds = splits["random_seeds"]
    random_fractions = splits["random_fractions"]
    logo_targets = splits["logo_targets"]
    chemical_families = splits["chemical_ood_families"]
    if (
        not isinstance(random_seeds, list)
        or not random_seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in random_seeds)
        or not isinstance(random_fractions, list)
        or not random_fractions
        or not isinstance(logo_targets, list)
        or tuple(logo_targets) != ("product_key", "reactant_key")
        or not isinstance(chemical_families, list)
        or not chemical_families
    ):
        raise ValueError("Representation split selections are invalid.")
    base_seed = config["base_seed"]
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer.")
    return {
        "dataset_path": Path(dataset["path"]),
        "canonical_split_directory": Path(splits["canonical_directory"]),
        "random_seeds": tuple(random_seeds),
        "random_fractions": tuple(float(value) for value in random_fractions),
        "logo_targets": tuple(logo_targets),
        "logo_artifact_directory": Path(splits["logo_artifact_directory"]),
        "chemical_ood_directory": Path(splits["chemical_ood_directory"]),
        "chemical_ood_families": tuple(chemical_families),
        "feature_config": dict(feature_config),
        "model": {"name": str(model["name"]), "params": dict(model["params"])},
        "metrics": tuple(metrics),
        "base_seed": base_seed,
        "output_directory": Path(output["directory"]),
    }


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return json.loads(json.dumps(dict(config)))
    value = yaml.safe_load(Path(config).read_text())
    if not isinstance(value, dict):
        raise ValueError("Representation benchmark config must be a mapping.")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _evidence_class(unit: EvaluationSplitUnit) -> str:
    if unit.split_family == "canonical_random":
        return "nested_random"
    if unit.target == "product_key":
        return "product_logo_ood"
    if unit.target == "reactant_key":
        return "reactant_logo_ood"
    return "chemical_ood"


def _false_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value) == "False":
        return False
    if str(value) == "True":
        return True
    raise ValueError(f"Invalid boolean evidence value: {value!r}.")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "rdkit", "xgboost"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions
