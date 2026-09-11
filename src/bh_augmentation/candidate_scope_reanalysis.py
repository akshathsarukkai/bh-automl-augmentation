"""Development reanalysis of candidate-scope semantics for low-data transfer.

This runner answers two development questions on validation data only.  It never
materializes an outer-test outcome and never claims an entry in the
repository-global evaluation registry; the preregistered confirmatory experiment
does that separately through
:mod:`bh_augmentation.policy_search` and :mod:`bh_augmentation.final_evaluation`.

The two questions are:

``observed_only_condition_transfer`` vs ``globally_unmeasured_condition_transfer``
    How large is the candidate pool, and how does downstream validation
    performance move, when eligibility is decided against the labeled low-data
    subset instead of the complete measured universe?  This matters because the
    canonical Buchwald-Hartwig matrix is nearly complete globally: at a 1%
    training fraction almost every chemically sensible condition transfer is
    "already measured" somewhere in the dataset, even though the simulated
    learner has observed almost none of it.

``withheld_cell_transfer_oracle``
    Of the candidates generated under the observed-only rule, how many land on
    reactions that were measured but hidden from the learner, and how accurate
    are the pseudo-labels on those reconstructed cells?

Information flow is one-directional by construction: candidates are generated
and pseudo-labeled from ``labeled_train`` only, sealed into a
:class:`~bh_augmentation.evaluation.withheld_cell_oracle.FrozenCandidateBundle`,
and only then joined against hidden outer-training yields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.candidate_scope import (
    GENERATION_SCOPE_MODES,
    GLOBALLY_UNMEASURED_PROSPECTIVE,
    OBSERVED_ONLY_LOW_DATA,
    CandidateScopePolicy,
    CandidateScopeViolation,
)
from bh_augmentation.augmentation.condition_transfer import (
    ConditionTransferConfig,
    generate_condition_transfer_examples,
)
from bh_augmentation.augmentation.pool_accounting import pool_statistics
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    RoleAwareConditionTransferConfig,
    clear_role_aware_teacher_cache,
    generate_role_aware_condition_transfer_examples,
)
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.low_data_partitions import (
    build_low_data_evaluation_unit,
)
from bh_augmentation.evaluation.metrics import mae, r2, rmse, spearman_corr
from bh_augmentation.evaluation.withheld_cell_oracle import (
    FrozenCandidateBundle,
    oracle_records_to_frame,
    oracle_strata_to_frame,
    withheld_cell_oracle_diagnostic,
)
from bh_augmentation.features.compatibility import assert_feature_compatibility
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.utils.corrected_runs import (
    CORRECTED_ROLE_FEATURE_KIND,
    build_run_manifest,
    feature_contract_record,
    prepare_fresh_output_directory,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
    write_json,
)
from bh_augmentation.utils.seed import set_global_seed
from bh_augmentation.utils.synthetic_audits import (
    build_candidate_audit_frame,
    combine_candidate_audit_frames,
)

CANDIDATE_SCOPE_REANALYSIS_SCHEMA_VERSION = "bh-candidate-scope-reanalysis-v1"

#: Result family names this runner produces.
OBSERVED_ONLY_FAMILY = "observed_only_condition_transfer"
GLOBALLY_UNMEASURED_FAMILY = "globally_unmeasured_condition_transfer"
WITHHELD_CELL_ORACLE_FAMILY = "withheld_cell_transfer_oracle"

_TRANSFER_KINDS = ("anonymous", "typed")
_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman_corr}
_FAMILY_BY_MODE = {
    OBSERVED_ONLY_LOW_DATA: OBSERVED_ONLY_FAMILY,
    GLOBALLY_UNMEASURED_PROSPECTIVE: GLOBALLY_UNMEASURED_FAMILY,
}


def run_candidate_scope_reanalysis(
    config_path: str | Path,
    config: Mapping[str, Any],
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run the validation-only candidate-scope reanalysis and oracle diagnostic."""
    resolved = dict(config)
    dataset_path = Path(resolved["dataset"]["path"])
    split_directory = Path(resolved["splits"]["directory"])
    seeds = [int(value) for value in resolved["splits"]["seeds"]]
    fractions = [float(value) for value in resolved["splits"]["train_fractions"]]
    modes = _resolved_modes(resolved)
    feature_config = resolve_corrected_feature_config(
        resolved.get("features", {}),
        required_kind=CORRECTED_ROLE_FEATURE_KIND,
    )
    metrics_requested = [str(value) for value in resolved.get("metrics", ["rmse", "mae"])]
    _validate_metrics(metrics_requested)
    model_specs = _resolved_models(resolved)
    oracle_config = dict(resolved.get("oracle", {}))
    high_yield_threshold = float(oracle_config.get("high_yield_threshold", 80.0))
    top_k_fractions = tuple(
        float(value) for value in oracle_config.get("top_k_fractions", (0.1, 0.2))
    )

    saved = load_saved_canonical_splits(
        dataset_path,
        split_directory,
        requested_seeds=seeds,
        requested_fractions=fractions,
    )
    dataset_hash = sha256_file(dataset_path)
    output = prepare_fresh_output_directory(
        output_directory
        if output_directory is not None
        else resolved["output"]["directory"]
    )

    pool_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    split_hashes: dict[str, str] = {}
    feature_metadata_hashes: dict[str, str] = {}

    for seed in seeds:
        for fraction in fractions:
            set_global_seed(seed)
            unit = build_low_data_evaluation_unit(
                saved,
                seed=seed,
                train_fraction=fraction,
                dataset_path=dataset_path,
            )
            partition_rows.append(unit.partition_record())
            audit = saved.audit_record(seed=seed, train_fraction=fraction)
            split_hashes[f"seed={seed}|fraction={fraction:g}"] = str(
                audit["source_id_split_hash"]
            )

            labeled = unit.labeled_train
            X_train, y_train, feature_names, feature_metadata = (
                build_feature_matrix_with_metadata(labeled, feature_config)
            )
            X_train = np.asarray(X_train, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32)
            contract = feature_contract_record(feature_metadata, feature_names)
            feature_metadata_hashes[f"seed={seed}|fraction={fraction:g}"] = str(
                contract["feature_metadata_hash"]
            )
            X_valid, y_valid, valid_names, valid_metadata = (
                build_feature_matrix_with_metadata(unit.validation, feature_config)
            )
            assert_feature_compatibility(
                X_train,
                list(feature_names),
                np.asarray(X_valid, dtype=np.float32),
                list(valid_names),
                real_metadata=feature_metadata,
                synthetic_metadata=valid_metadata,
            )
            X_valid = np.asarray(X_valid, dtype=np.float32)
            y_valid = np.asarray(y_valid, dtype=np.float32)

            base_record = {
                "evaluation_unit": unit.evaluation_unit,
                "seed": seed,
                "train_fraction": fraction,
                "n_labeled_train": int(len(labeled)),
                "n_hidden_outer_train": int(len(unit.hidden_outer_train)),
                "n_validation": int(len(unit.validation)),
            }

            for model_name, model_kwargs in model_specs:
                model = train_model(
                    get_model(model_name, seed=seed, **model_kwargs), X_train, y_train
                )
                metric_rows.extend(
                    _metric_rows(
                        y_valid,
                        predict_model(model, X_valid),
                        metrics_requested,
                        {
                            **base_record,
                            "result_family": "real_only_low_data_control",
                            "candidate_scope_mode": "not_applicable",
                            "transfer_kind": "none",
                            "model": model_name,
                            "n_synthetic": 0,
                        },
                    )
                )

            for mode in modes:
                scope = unit.candidate_scope(mode)
                _assert_hidden_identities_absent(scope, unit)
                for transfer_kind in _TRANSFER_KINDS:
                    generated = _generate(
                        transfer_kind,
                        resolved,
                        labeled,
                        X_train,
                        y_train,
                        feature_config=feature_config,
                        feature_names=list(feature_names),
                        feature_metadata=feature_metadata,
                        scope=scope,
                        seed=seed,
                    )
                    candidate_df = generated["candidate_df"]
                    policy_id = f"{transfer_kind}|{mode}|seed={seed}|fraction={fraction:g}"
                    if not candidate_df.empty:
                        audit_frames.append(
                            build_candidate_audit_frame(
                                candidate_df.assign(
                                    candidate_scope_mode_declared=mode,
                                    result_family=_FAMILY_BY_MODE[mode],
                                ),
                                transfer_kind=transfer_kind,
                                seed=seed,
                                train_fraction=fraction,
                                policy_id=policy_id,
                            )
                        )
                    pool_rows.append(
                        {
                            **base_record,
                            "result_family": _FAMILY_BY_MODE[mode],
                            "candidate_scope_mode": mode,
                            "transfer_kind": transfer_kind,
                            **_pool_statistics(candidate_df, scope),
                        }
                    )

                    # --- freeze before any hidden outcome becomes legible ---
                    bundle = FrozenCandidateBundle.freeze(
                        candidate_df,
                        evaluation_unit=unit.evaluation_unit,
                        train_fraction=fraction,
                        candidate_scope_mode=mode,
                    )
                    X_synthetic = np.asarray(generated["X_synthetic"], dtype=np.float32)
                    y_synthetic = np.asarray(generated["synthetic_y"], dtype=np.float32)
                    for model_name, model_kwargs in model_specs:
                        X_fit = (
                            np.vstack([X_train, X_synthetic])
                            if len(y_synthetic)
                            else X_train
                        )
                        y_fit = (
                            np.concatenate([y_train, y_synthetic])
                            if len(y_synthetic)
                            else y_train
                        )
                        model = train_model(
                            get_model(model_name, seed=seed, **model_kwargs), X_fit, y_fit
                        )
                        metric_rows.extend(
                            _metric_rows(
                                y_valid,
                                predict_model(model, X_valid),
                                metrics_requested,
                                {
                                    **base_record,
                                    "result_family": _FAMILY_BY_MODE[mode],
                                    "candidate_scope_mode": mode,
                                    "transfer_kind": transfer_kind,
                                    "model": model_name,
                                    "n_synthetic": int(len(y_synthetic)),
                                },
                            )
                        )

                    # --- retrospective oracle: the only hidden-yield read ---
                    if len(unit.hidden_outer_train):
                        oracle_records.append(
                            {
                                "seed": seed,
                                "transfer_kind": transfer_kind,
                                "result_family": WITHHELD_CELL_ORACLE_FAMILY,
                                **withheld_cell_oracle_diagnostic(
                                    bundle,
                                    hidden_outer_train=unit.hidden_outer_train_oracle(),
                                    labeled_train_identity_keys=(
                                        unit.labeled_train_identity_keys
                                    ),
                                    high_yield_threshold=high_yield_threshold,
                                    top_k_fractions=top_k_fractions,
                                ),
                            }
                        )

    paths = _write_outputs(
        output,
        config=resolved,
        config_path=config_path,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        pool_rows=pool_rows,
        metric_rows=metric_rows,
        oracle_records=oracle_records,
        partition_rows=partition_rows,
        audit_frames=audit_frames,
        split_hashes=split_hashes,
        feature_metadata_hashes=feature_metadata_hashes,
        modes=modes,
    )
    return paths


def candidate_scope_comparison(pool_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate-pool size between the two eligibility semantics.

    The ``pool_size_ratio`` is the headroom the low-data protocol recovers: how
    many times more eligible candidates the observed-only rule admits than the
    globally-unmeasured rule does at the same evaluation unit.
    """
    keys = ["evaluation_unit", "seed", "train_fraction", "transfer_kind"]
    wide = pool_frame.pivot_table(
        index=keys,
        columns="candidate_scope_mode",
        values=["accepted_candidate_count", "unique_accepted_identity_count"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{mode}" for metric, mode in wide.columns]
    wide = wide.reset_index()
    observed = f"accepted_candidate_count__{OBSERVED_ONLY_LOW_DATA}"
    prospective = f"accepted_candidate_count__{GLOBALLY_UNMEASURED_PROSPECTIVE}"
    if observed in wide and prospective in wide:
        wide["pool_size_delta"] = wide[observed] - wide[prospective]
        wide["pool_size_ratio"] = np.where(
            wide[prospective] > 0,
            wide[observed] / wide[prospective].replace(0, np.nan),
            np.nan,
        )
    return wide


def _resolved_modes(config: Mapping[str, Any]) -> tuple[str, ...]:
    requested = config.get("candidate_scope", {})
    if not isinstance(requested, Mapping):
        raise CandidateScopeViolation("candidate_scope must be a mapping.")
    modes = tuple(str(value) for value in requested.get("modes", GENERATION_SCOPE_MODES))
    unknown = sorted(set(modes) - set(GENERATION_SCOPE_MODES))
    if not modes or unknown:
        raise CandidateScopeViolation(
            f"candidate_scope.modes must be a nonempty subset of "
            f"{sorted(GENERATION_SCOPE_MODES)}; unknown={unknown}."
        )
    if len(set(modes)) != len(modes):
        raise CandidateScopeViolation("candidate_scope.modes must not repeat a mode.")
    return modes


def _resolved_models(config: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = config.get("models", ["xgboost"])
    if not isinstance(raw, list) or not raw:
        raise ValueError("models must be a nonempty list.")
    resolved: list[tuple[str, dict[str, Any]]] = []
    for item in raw:
        if isinstance(item, str):
            resolved.append((item, {}))
        elif isinstance(item, Mapping) and "name" in item:
            # ``params`` is the nested estimator-keyword block used by every
            # config in this repository.  It must be flattened into the keyword
            # arguments handed to ``get_model``; forwarding it as a single
            # ``params=`` keyword is silently ignored by XGBoost (which warns
            # "Parameters: { "params" } are not used") and fits the defaults.
            unknown = set(item) - {"name", "params"}
            if unknown:
                raise ValueError(
                    f"Unsupported model specification keys {sorted(unknown)!r} "
                    f"for model {item['name']!r}; use a nested 'params' mapping."
                )
            params = item.get("params", {})
            if not isinstance(params, Mapping):
                raise ValueError(f"Model params for {item['name']!r} must be a mapping.")
            resolved.append((str(item["name"]), dict(params)))
        else:
            raise ValueError(f"Unsupported model specification: {item!r}")
    return resolved


def _validate_metrics(metrics: Sequence[str]) -> None:
    unknown = sorted(set(metrics) - set(_METRICS))
    if not metrics or unknown:
        raise ValueError(f"Unsupported metrics requested: {unknown}.")


def _generate(
    transfer_kind: str,
    config: Mapping[str, Any],
    labeled: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_config: dict[str, Any],
    feature_names: list[str],
    feature_metadata: Any,
    scope: CandidateScopePolicy,
    seed: int,
) -> dict[str, Any]:
    transfer = dict(config.get("transfer", {}))
    if transfer_kind == "anonymous":
        policy = ConditionTransferConfig(
            **{**dict(transfer["anonymous"]), "random_state": seed}
        )
        return generate_condition_transfer_examples(
            labeled,
            X_train,
            y_train,
            policy,
            feature_config=feature_config,
            real_feature_names=feature_names,
            real_feature_metadata=feature_metadata,
            candidate_scope=scope,
        )
    policy = RoleAwareConditionTransferConfig(
        **{**dict(transfer["typed"]), "random_state": seed}
    )
    clear_role_aware_teacher_cache()
    try:
        return generate_role_aware_condition_transfer_examples(
            labeled,
            X_train,
            y_train,
            policy,
            feature_config=feature_config,
            real_feature_names=feature_names,
            real_feature_metadata=feature_metadata,
            candidate_scope=scope,
        )
    finally:
        clear_role_aware_teacher_cache()


def _assert_hidden_identities_absent(
    scope: CandidateScopePolicy,
    unit: Any,
) -> None:
    """Prove that hidden outer-training chemistry never reaches generation.

    Under the observed-only protocol this is the whole point: a hidden
    outer-training identity must be *generatable*, so it must not appear in the
    rejection set.  Under the prospective protocol hidden identities are
    legitimately ineligible, so the check does not apply.
    """
    if scope.mode != OBSERVED_ONLY_LOW_DATA:
        return
    scope.assert_excludes(
        unit.hidden_outer_train_identity_keys,
        description="Hidden outer-training identities",
    )


def _pool_statistics(
    candidate_df: pd.DataFrame,
    scope: CandidateScopePolicy,
) -> dict[str, Any]:
    """Kept as the historical name; the definition lives in ``pool_accounting``."""
    return pool_statistics(candidate_df, scope)


def _metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Sequence[str],
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {**context, "split": "valid", "metric": name, "value": float(_METRICS[name](y_true, y_pred))}
        for name in metrics
    ]


def _write_outputs(
    output: Path,
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    dataset_path: Path,
    dataset_hash: str,
    pool_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    oracle_records: list[dict[str, Any]],
    partition_rows: list[dict[str, Any]],
    audit_frames: list[pd.DataFrame],
    split_hashes: Mapping[str, str],
    feature_metadata_hashes: Mapping[str, str],
    modes: Sequence[str],
) -> dict[str, Path]:
    pool_frame = pd.DataFrame(pool_rows)
    metric_frame = pd.DataFrame(metric_rows)
    partition_frame = pd.DataFrame(partition_rows)
    oracle_frame = oracle_records_to_frame(oracle_records)
    strata_frame = oracle_strata_to_frame(oracle_records)
    comparison = (
        candidate_scope_comparison(pool_frame)
        if len(modes) > 1 and not pool_frame.empty
        else pd.DataFrame()
    )
    audit_frame = combine_candidate_audit_frames(audit_frames)

    paths = {
        "directory": output,
        "candidate_pool_statistics": output / "candidate_pool_statistics.csv",
        "candidate_scope_comparison": output / "candidate_scope_comparison.csv",
        "validation_metrics": output / "validation_metrics.csv",
        "withheld_cell_oracle": output / "withheld_cell_oracle.csv",
        "withheld_cell_oracle_by_change_type": (
            output / "withheld_cell_oracle_by_change_type.csv"
        ),
        "partition_audit": output / "partition_audit.csv",
        "candidate_audit": output / "candidate_audit.csv",
        "leakage_contracts": output / "leakage_contracts.json",
        "manifest": output / "manifest.json",
    }
    pool_frame.to_csv(paths["candidate_pool_statistics"], index=False)
    comparison.to_csv(paths["candidate_scope_comparison"], index=False)
    metric_frame.to_csv(paths["validation_metrics"], index=False)
    oracle_frame.to_csv(paths["withheld_cell_oracle"], index=False)
    strata_frame.to_csv(paths["withheld_cell_oracle_by_change_type"], index=False)
    partition_frame.to_csv(paths["partition_audit"], index=False)
    audit_frame.to_csv(paths["candidate_audit"], index=False)

    write_json(
        paths["leakage_contracts"],
        {
            "schema_version": CANDIDATE_SCOPE_REANALYSIS_SCHEMA_VERSION,
            "outer_test_materialized": False,
            "outer_test_labels_accessed": False,
            "outer_test_predictions_generated": False,
            "evaluation_registry_claimed": False,
            "selection_data_roles": ["labeled_train", "valid"],
            "hidden_outer_train_read_stage": "after_candidate_and_pseudo_label_freeze",
            "hidden_outer_train_influences": [],
            "candidate_scope_modes": list(modes),
            "result_families": sorted(
                {str(value) for value in pool_frame.get("result_family", pd.Series(dtype=str))}
                | ({WITHHELD_CELL_ORACLE_FAMILY} if len(oracle_records) else set())
            ),
            "frozen_bundle_hashes": sorted(
                {str(record["frozen_hash"]) for record in oracle_records}
            ),
            "candidate_pool_statistics_sha256": stable_hash(
                json.loads(pool_frame.to_json(orient="records"))
            ),
        },
    )
    manifest = build_run_manifest(
        config=config,
        config_path=config_path,
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        output_directory=output,
        split_hashes=split_hashes,
        feature_metadata_hash=dict(feature_metadata_hashes),
    )
    manifest["schema_version"] = CANDIDATE_SCOPE_REANALYSIS_SCHEMA_VERSION
    manifest["candidate_scope_modes"] = list(modes)
    # Key the hashes by file name, not by the logical artifact name: a verifier
    # must be able to resolve each entry to exactly one file on disk.
    manifest["output_hashes"] = {
        path.name: sha256_file(path)
        for name, path in sorted(paths.items())
        if name != "directory" and path.exists() and path.suffix in {".csv", ".json"}
    }
    write_json(paths["manifest"], manifest)
    return paths


__all__ = [
    "CANDIDATE_SCOPE_REANALYSIS_SCHEMA_VERSION",
    "GLOBALLY_UNMEASURED_FAMILY",
    "OBSERVED_ONLY_FAMILY",
    "WITHHELD_CELL_ORACLE_FAMILY",
    "candidate_scope_comparison",
    "run_candidate_scope_reanalysis",
]
