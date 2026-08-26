"""Tests for validation-only joint condition-transfer supervised-AE search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import bh_augmentation.ae_policy_search as search_module
from bh_augmentation.ae_policy_search import (
    AEPolicySearchContext,
    JointAEFitResult,
    _prepare_transfer_generations,
    _resolve_ae_search_config,
    _resolved_joint_ae_policies,
    run_ae_policy_search_command,
)
from bh_augmentation.augmentation.candidate_scope import observed_only_scope
from bh_augmentation.evaluation.ae_policy_protocol import AEInnerSplit
from bh_augmentation.features.compatibility import coordinate_feature_contract
from bh_augmentation.policy_search import LabeledPartition
from corrected_test_utils import write_corrected_config


def test_joint_policy_budget_is_complete_and_traversal_independent() -> None:
    raw = _search_config()
    raw["latent_dims"] = [8, 4]
    raw["synthetic_supervised_weights"] = [1.0, 0.5]
    raw["synthetic_reconstruction_weights"] = [1.0, 0.1]
    raw["downstream_models"] = [
        {"name": "ridge", "params": {"alpha": 1.0}},
        {"name": "ridge", "params": {"alpha": 0.1}},
    ]
    config = {"metrics": ["rmse"], "ae_policy_search": raw}
    resolved = _resolve_ae_search_config(config)
    policies = _resolved_joint_ae_policies(
        resolved,
        seed=0,
        fraction=0.5,
        input_dim=16,
        inner_split_hash="inner-hash",
    )

    reversed_raw = {
        **raw,
        "transfer_policies": list(reversed(raw["transfer_policies"])),
        "latent_dims": list(reversed(raw["latent_dims"])),
        "synthetic_supervised_weights": list(
            reversed(raw["synthetic_supervised_weights"])
        ),
        "synthetic_reconstruction_weights": list(
            reversed(raw["synthetic_reconstruction_weights"])
        ),
        "downstream_models": list(reversed(raw["downstream_models"])),
    }
    reversed_policies = _resolved_joint_ae_policies(
        _resolve_ae_search_config(
            {"metrics": ["rmse"], "ae_policy_search": reversed_raw}
        ),
        seed=0,
        fraction=0.5,
        input_dim=16,
        inner_split_hash="inner-hash",
    )

    assert len(policies) == 20
    assert [policy.policy_hash for policy in policies] == [
        policy.policy_hash for policy in reversed_policies
    ]
    assert all(
        {
            "transfer_method",
            "transfer_config",
            "latent_dim",
            "synthetic_supervised_weight",
            "synthetic_reconstruction_weight",
            "downstream_model",
            "downstream_params",
            "ae_settings",
            "inner_split_hash",
        }
        <= set(policy.config)
        for policy in policies
    )


def test_synthetic_parent_validation_rejects_internal_validation_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _dummy_context()
    raw = _search_config()
    resolved = _resolve_ae_search_config(
        {"metrics": ["rmse"], "ae_policy_search": raw}
    )
    anonymous = next(
        policy
        for policy in _resolved_joint_ae_policies(
            resolved,
            seed=0,
            fraction=0.5,
            input_dim=4,
            inner_split_hash=context.inner_split.split_hash,
        )
        if policy.config["transfer_method"] == "anonymous"
    )

    def contaminated_generation(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        return {
            "candidate_df": pd.DataFrame(
                {
                    "source_row_id": ["fit-1"],
                    "donor_row_id": ["internal-1"],
                }
            ),
            "synthetic_y": np.array([50.0]),
            "metadata": {},
        }

    monkeypatch.setattr(
        search_module,
        "generate_condition_transfer_examples",
        contaminated_generation,
    )
    with pytest.raises(ValueError, match="Synthetic parent leakage"):
        _prepare_transfer_generations([anonymous], context)


def test_search_redacts_test_and_excludes_infeasible_typed_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    production_materialize = search_module._materialize_search_frames
    redaction_observed = False
    generation_ids: set[str] = set()

    def inspecting_materialize(saved: Any, **kwargs: Any) -> Any:
        nonlocal redaction_observed
        rows = saved.low_data_assignments.loc[
            saved.low_data_assignments["seed"].eq(kwargs["seed"])
            & saved.low_data_assignments["train_fraction"].eq(
                kwargs["train_fraction"]
            )
        ]
        test_ids = set(
            rows.loc[rows["outer_split"].eq("test"), "source_row_id"].astype(str)
        )
        hidden = saved.canonical.loc[
            saved.canonical["source_row_id"].isin(test_ids)
        ]
        redaction_observed = bool(hidden["yield"].isna().all())
        return production_materialize(saved, **kwargs)

    def zero_generation(
        frame: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        config: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        del X, y, config, kwargs
        generation_ids.update(frame["source_row_id"].astype(str))
        return {
            "candidate_df": pd.DataFrame(
                columns=["source_row_id", "donor_row_id"]
            ),
            "synthetic_y": np.empty(0),
            "metadata": {"accepted_count": 0},
        }

    def fake_fit(
        policy: Any,
        context: AEPolicySearchContext,
        generation: object,
        *,
        metric_names: list[str],
    ) -> JointAEFitResult:
        assert policy.config["transfer_method"] == "real_only"
        assert generation is None
        assert set(context.inner_split.ae_train_source_ids).isdisjoint(
            context.inner_split.internal_validation_source_ids
        )
        assert set(context.ae_fit.source_row_ids).isdisjoint(
            context.policy_validation.source_row_ids
        )
        assert not set(context.ae_fit.source_row_ids) & context.forbidden_source_ids
        return _fake_result(policy, metric_names, value=5.0)

    monkeypatch.setattr(
        search_module,
        "_materialize_search_frames",
        inspecting_materialize,
    )
    monkeypatch.setattr(
        search_module,
        "generate_condition_transfer_examples",
        zero_generation,
    )
    monkeypatch.setattr(search_module, "_fit_joint_ae_policy", fake_fit)

    paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_joint_ae_search",
    )

    manifest = json.loads(paths["search_manifest"].read_text())
    metrics = pd.read_csv(paths["search_metrics"])
    exclusions = pd.read_csv(paths["search_exclusions"])
    assert redaction_observed
    assert generation_ids
    assert len(generation_ids) == manifest["payload"]["inner_split"]["n_ae_train"]
    assert metrics["transfer_method"].eq("real_only").all()
    assert not exclusions.empty
    assert exclusions["rejection_reason"].eq("zero_accepted_synthetic").all()
    assert manifest["payload"]["outer_test_labels_accessed"] is False
    assert manifest["payload"]["outer_test_predictions_generated"] is False
    assert manifest["payload"]["outer_test_metric_evaluations"] == 0
    assert manifest["payload"]["test_evaluated"] is False
    assert set(paths["directory"].iterdir()) == {
        paths["frozen_policy"],
        paths["search_metrics"],
        paths["search_manifest"],
        paths["search_exclusions"],
        paths["ae_training_audit"],
        paths["synthetic_transfer_audit"],
        paths["synthetic_candidate_audit"],
    }


def test_search_selects_one_global_joint_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["ae_policy_search"]["latent_dims"] = [4, 8]
    config["ae_policy_search"]["synthetic_reconstruction_weights"] = [0.1, 1.0]
    config["ae_policy_search"]["downstream_models"] = [
        {"name": "ridge", "params": {"alpha": 0.1}},
        {"name": "ridge", "params": {"alpha": 1.0}},
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def feasible_generation(
        frame: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        config: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        del X, y, config, kwargs
        parent_ids = frame["source_row_id"].astype(str).tolist()
        return {
            "candidate_df": pd.DataFrame(
                {
                    "source_row_id": [parent_ids[0]],
                    "donor_row_id": [parent_ids[1]],
                }
            ),
            "synthetic_y": np.array([50.0]),
            "metadata": {"accepted_count": 1},
        }

    def joint_score(
        policy: Any,
        context: AEPolicySearchContext,
        generation: object,
        *,
        metric_names: list[str],
    ) -> JointAEFitResult:
        del context, generation
        winner = (
            policy.config["transfer_method"] == "anonymous"
            and policy.config["latent_dim"] == 8
            and policy.config["synthetic_reconstruction_weight"] == 0.1
            and policy.config["downstream_params"]["alpha"] == 1.0
        )
        return _fake_result(policy, metric_names, value=1.0 if winner else 9.0)

    monkeypatch.setattr(
        search_module,
        "generate_condition_transfer_examples",
        feasible_generation,
    )
    monkeypatch.setattr(search_module, "_fit_joint_ae_policy", joint_score)
    paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_joint_tuple_search",
    )

    frozen = json.loads(paths["frozen_policy"].read_text())
    metrics = pd.read_csv(paths["search_metrics"])
    selected = frozen["resolved_policy"]
    assert selected["transfer_method"] == "anonymous"
    assert selected["latent_dim"] == 8
    assert selected["synthetic_reconstruction_weight"] == 0.1
    assert selected["downstream_params"]["alpha"] == 1.0
    assert metrics.loc[metrics["selected_policy"], "policy_id"].nunique() == 1
    assert frozen["training_protocol"]["refit_epoch_contract"] == (
        "predefined_resolved_epoch_count"
    )
    assert frozen["training_protocol"]["refit_epochs"] == 2


def test_real_only_joint_ae_search_runs_production_fit_path(tmp_path: Path) -> None:
    config_path = _protocol_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["ae_policy_search"]["transfer_policies"] = [{"method": "real_only"}]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_real_only_joint_ae_search",
    )

    metrics = pd.read_csv(paths["search_metrics"])
    training_audit = pd.read_csv(paths["ae_training_audit"])
    manifest = json.loads(paths["search_manifest"].read_text())
    assert metrics["split"].eq("valid").all()
    assert metrics["transfer_method"].eq("real_only").all()
    assert metrics["n_synthetic_train"].eq(0).all()
    assert np.isfinite(metrics["value"]).all()
    overlap_columns = [
        column for column in training_audit if column.endswith("_overlap_count")
    ]
    assert overlap_columns
    assert training_audit[overlap_columns].eq(0).all().all()
    assert training_audit["inner_split_hash"].eq(
        manifest["payload"]["inner_split"]["split_hash"]
    ).all()
    for name, expected_hash in manifest["payload"]["artifact_hashes"].items():
        assert _sha256(paths[name]) == expected_hash


def test_accepted_synthetic_fit_uses_only_declared_training_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _sentinel_context()
    resolved = _resolve_ae_search_config(
        {"metrics": ["rmse"], "ae_policy_search": _search_config()}
    )
    policy = next(
        item
        for item in _resolved_joint_ae_policies(
            resolved,
            seed=0,
            fraction=0.5,
            input_dim=4,
            inner_split_hash=context.inner_split.split_hash,
        )
        if item.config["transfer_method"] == "anonymous"
    )
    synthetic_X = np.full((1, 4), 7.0, dtype=np.float32)
    synthetic_y = np.array([70.0], dtype=np.float32)
    saved_test_X = np.full((1, 4), 31.0, dtype=np.float32)
    saved_test_y = np.array([310.0], dtype=np.float32)
    generation = {
        "candidate_df": pd.DataFrame(
            {
                "source_row_id": ["fit-1"],
                "donor_row_id": ["fit-2"],
            }
        ),
        "X_synthetic": synthetic_X,
        "synthetic_y": synthetic_y,
        "feature_names": list(context.ae_fit.feature_names),
        "feature_metadata": context.ae_fit.feature_metadata,
    }
    production_ae_fit = search_module.fit_supervised_autoencoder
    production_downstream_fit = search_module.train_model
    observed_downstream_y: list[np.ndarray] = []

    def checking_ae_fit(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        config: object,
        **kwargs: object,
    ) -> object:
        assert np.array_equal(
            X_train,
            np.vstack([context.ae_fit.X, synthetic_X]),
        )
        assert np.array_equal(
            y_train,
            np.concatenate([context.ae_fit.y, synthetic_y]),
        )
        assert np.array_equal(X_valid, context.internal_validation.X)
        assert np.array_equal(y_valid, context.internal_validation.y)
        assert not np.any(X_train == 11.0)
        assert not np.any(X_train == 21.0)
        assert not np.isin(X_train, saved_test_X).any()
        assert not np.isin(
            y_train,
            [*context.internal_validation.y, *context.policy_validation.y, *saved_test_y],
        ).any()
        return production_ae_fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            config,
            **kwargs,
        )

    def checking_downstream_fit(
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> object:
        observed_downstream_y.append(np.asarray(y_train).copy())
        assert np.array_equal(
            y_train,
            np.concatenate([context.ae_fit.y, synthetic_y]),
        )
        assert not np.isin(
            y_train,
            [*context.internal_validation.y, *context.policy_validation.y, *saved_test_y],
        ).any()
        return production_downstream_fit(
            model,
            X_train,
            y_train,
            sample_weight=sample_weight,
        )

    monkeypatch.setattr(
        search_module,
        "fit_supervised_autoencoder",
        checking_ae_fit,
    )
    monkeypatch.setattr(search_module, "train_model", checking_downstream_fit)
    result = search_module._fit_joint_ae_policy(
        policy,
        context,
        generation,
        metric_names=["rmse"],
    )

    assert len(observed_downstream_y) == 1
    assert result.metrics["split"].eq("valid").all()
    assert np.isfinite(result.metrics["value"]).all()
    overlap_fields = {
        name: value
        for name, value in result.audit.items()
        if name.endswith("_overlap_count")
    }
    assert overlap_fields
    assert set(overlap_fields.values()) == {0}


def _protocol_config(tmp_path: Path) -> Path:
    base_path = write_corrected_config(
        tmp_path,
        kind="anonymous",
        output_name="corrected_unused_legacy_ae_output",
    )
    config = yaml.safe_load(base_path.read_text())
    transfer = config.pop("condition_transfer")
    config.pop("models")
    config["metrics"] = ["rmse", "mae"]
    config["ae_policy_search"] = _search_config(transfer)
    config["output"] = {
        "ae_search_directory": str(tmp_path / "corrected_default_ae_search")
    }
    path = tmp_path / "joint_ae_protocol.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _search_config(
    transfer_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if transfer_source is None:
        transfer_config = {
            "donor_strategy": "random",
            "synthetic_multiplier": 0.5,
            "n_neighbors": 3,
            "label_strategy": "teacher_ensemble",
            "teacher_models": ["ridge"],
            "max_teacher_std": None,
            "min_similarity": None,
            "high_yield_threshold": 70.0,
            "clip_y_min": 0.0,
            "clip_y_max": 100.0,
            "candidates_per_real": 3,
            "random_state": 0,
            "donor_similarity_n_bits": 64,
            "donor_similarity_radius": 2,
            "donor_similarity_backend": "rdkit",
            "role_change_requirement": "all",
            "fallback_policy": "reject",
            "max_candidates_per_source": 3,
        }
    else:
        transfer_config = {
            "donor_strategy": transfer_source["donor_strategies"][0],
            "synthetic_multiplier": transfer_source["synthetic_multipliers"][0],
            "n_neighbors": transfer_source["n_neighbors"][0],
            "label_strategy": transfer_source["label_strategies"][0],
            "teacher_models": transfer_source["teacher_models"],
            "max_teacher_std": transfer_source["max_teacher_stds"][0],
            "min_similarity": transfer_source["min_similarities"][0],
            "high_yield_threshold": transfer_source["high_yield_threshold"],
            "clip_y_min": transfer_source["clip_y_min"],
            "clip_y_max": transfer_source["clip_y_max"],
            "candidates_per_real": transfer_source["candidates_per_real"],
            "random_state": 0,
            "donor_similarity_n_bits": transfer_source[
                "donor_similarity_n_bits"
            ],
            "donor_similarity_radius": transfer_source[
                "donor_similarity_radius"
            ],
            "donor_similarity_backend": transfer_source[
                "donor_similarity_backend"
            ],
            "role_change_requirement": transfer_source[
                "role_change_requirement"
            ],
            "fallback_policy": transfer_source["fallback_policy"],
        }
    return {
        "selection_metric": "rmse",
        "lower_is_better": True,
        "inner_split_seed_offset": 20_000,
        "transfer_policies": [
            {"method": "real_only"},
            {"method": "anonymous", "transfer_config": transfer_config},
        ],
        "latent_dims": [4],
        "synthetic_supervised_weights": [0.5],
        "synthetic_reconstruction_weights": [0.1],
        "downstream_models": [
            {"name": "ridge", "params": {"alpha": 1.0}}
        ],
        "ae_settings": {
            "hidden_dims": [8],
            "dropout": 0.0,
            "reconstruction_weight": 1.0,
            "yield_weight": 1.0,
            "latent_l2_weight": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 4,
            "max_epochs": 2,
            "patience": 1,
            "device": "cpu",
            "internal_valid_size": 0.2,
            "minimum_inner_rows": 4,
            "minimum_inner_groups": 2,
            "refit_epochs": 2,
        },
    }


def _fake_result(
    policy: Any,
    metric_names: list[str],
    *,
    value: float,
) -> JointAEFitResult:
    metrics = pd.DataFrame(
        [
            {
                "split": "valid",
                "metric": metric,
                "value": value,
                "transfer_method": policy.config["transfer_method"],
                "latent_dim": policy.config["latent_dim"],
                "synthetic_supervised_weight": policy.config[
                    "synthetic_supervised_weight"
                ],
                "synthetic_reconstruction_weight": policy.config[
                    "synthetic_reconstruction_weight"
                ],
                "downstream_model": policy.config["downstream_model"],
            }
            for metric in metric_names
        ]
    )
    return JointAEFitResult(
        metrics=metrics,
        audit={"policy_hash": policy.policy_hash, "ae_best_epoch": 1},
    )


def _dummy_context() -> AEPolicySearchContext:
    eligible = pd.DataFrame(
        {
            "source_row_id": ["fit-1", "fit-2", "internal-1", "internal-2"],
            "canonical_reaction_key": ["g1", "g2", "g3", "g4"],
        }
    )
    fit = eligible.iloc[:2].copy()
    internal = eligible.iloc[2:].copy()
    inner = AEInnerSplit.from_partitions(
        eligible,
        fit,
        internal,
        seed=0,
        valid_size=0.5,
        forbidden_outer_valid_source_ids=["policy-valid-1"],
        forbidden_outer_test_source_ids=["test-1"],
        minimum_rows=4,
        minimum_groups=2,
    )
    return AEPolicySearchContext(
        ae_fit=_dummy_partition(fit),
        internal_validation=_dummy_partition(internal),
        policy_validation=_dummy_partition(
            pd.DataFrame(
                {
                    "source_row_id": ["policy-valid-1"],
                    "canonical_reaction_key": ["g5"],
                }
            )
        ),
        inner_split=inner,
        forbidden_source_ids=frozenset({"policy-valid-1", "test-1"}),
        outer_validation_source_ids=frozenset({"policy-valid-1"}),
        outer_test_source_ids=frozenset({"test-1"}),
    )


def _dummy_partition(frame: pd.DataFrame) -> LabeledPartition:
    feature_names, metadata = coordinate_feature_contract("dummy-ae-space", 4)
    complete = frame.assign(yield_value=np.arange(len(frame), dtype=float))
    return LabeledPartition(
        X=np.zeros((len(frame), 4), dtype=np.float32),
        y=complete["yield_value"].to_numpy(dtype=np.float32),
        source_row_ids=tuple(frame["source_row_id"].astype(str)),
        frame=complete,
        feature_config={"kind": "dummy"},
        feature_names=tuple(feature_names),
        feature_metadata=metadata,
        candidate_scope=observed_only_scope(labeled_train_identity_keys=()),
    )


def _sentinel_context() -> AEPolicySearchContext:
    eligible = pd.DataFrame(
        {
            "source_row_id": ["fit-1", "fit-2", "internal-1", "internal-2"],
            "canonical_reaction_key": ["g1", "g2", "g3", "g4"],
        }
    )
    fit = eligible.iloc[:2].copy()
    internal = eligible.iloc[2:].copy()
    inner = AEInnerSplit.from_partitions(
        eligible,
        fit,
        internal,
        seed=0,
        valid_size=0.5,
        forbidden_outer_valid_source_ids=["policy-valid-1"],
        forbidden_outer_test_source_ids=["test-1"],
        minimum_rows=4,
        minimum_groups=2,
    )
    feature_names, metadata = coordinate_feature_contract(
        "sentinel-ae-space",
        4,
    )

    def partition(
        frame: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
    ) -> LabeledPartition:
        return LabeledPartition(
            X=X,
            y=y,
            source_row_ids=tuple(frame["source_row_id"].astype(str)),
            frame=frame.copy(),
            feature_config={"kind": "sentinel"},
            feature_names=tuple(feature_names),
            feature_metadata=metadata,
            candidate_scope=observed_only_scope(labeled_train_identity_keys=()),
        )

    policy_frame = pd.DataFrame(
        {
            "source_row_id": ["policy-valid-1"],
            "canonical_reaction_key": ["g5"],
        }
    )
    return AEPolicySearchContext(
        ae_fit=partition(
            fit,
            np.array([[1.0] * 4, [2.0] * 4], dtype=np.float32),
            np.array([10.0, 20.0], dtype=np.float32),
        ),
        internal_validation=partition(
            internal,
            np.array([[11.0] * 4, [12.0] * 4], dtype=np.float32),
            np.array([110.0, 120.0], dtype=np.float32),
        ),
        policy_validation=partition(
            policy_frame,
            np.array([[21.0] * 4], dtype=np.float32),
            np.array([210.0], dtype=np.float32),
        ),
        inner_split=inner,
        forbidden_source_ids=frozenset({"policy-valid-1", "test-1"}),
        outer_validation_source_ids=frozenset({"policy-valid-1"}),
        outer_test_source_ids=frozenset({"test-1"}),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
