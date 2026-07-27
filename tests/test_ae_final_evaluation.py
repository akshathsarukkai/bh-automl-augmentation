"""Tests for frozen joint supervised-AE final evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import bh_augmentation.ae_final_evaluation as final_module
import bh_augmentation.ae_policy_search as search_module
from bh_augmentation.ae_final_evaluation import run_ae_final_evaluation_command
from bh_augmentation.ae_policy_search import (
    JointAEFitResult,
    run_ae_policy_search_command,
)
from bh_augmentation.utils.corrected_runs import stable_hash
from corrected_test_utils import write_corrected_config


def test_frozen_real_only_ae_refits_before_one_outer_test_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path, transfer_policies=[{"method": "real_only"}])
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_ae_search",
    )
    production_fit = final_module.fit_supervised_autoencoder
    production_predict = final_module.predict_model
    refit_calls = 0
    prediction_batches = 0

    def checking_fit(
        X_train: object,
        y_train: object,
        X_valid: object,
        y_valid: object,
        config: object,
        **kwargs: object,
    ) -> object:
        nonlocal refit_calls
        refit_calls += 1
        assert X_valid is None
        assert y_valid is None
        assert config.max_epochs == 2
        return production_fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            config,
            **kwargs,
        )

    def counting_predict(model: object, X: object) -> object:
        nonlocal prediction_batches
        prediction_batches += 1
        return production_predict(model, X)

    monkeypatch.setattr(final_module, "fit_supervised_autoencoder", checking_fit)
    monkeypatch.setattr(final_module, "predict_model", counting_predict)
    final_paths = run_ae_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=tmp_path / "corrected_ae_final",
    )

    assert refit_calls == 1
    assert prediction_batches == 1
    assert {path.name for path in final_paths["directory"].iterdir()} == {
        "evaluation_claim.json",
        "final_evaluation_manifest.json",
        "final_test_metrics.csv",
    }
    metrics = pd.read_csv(final_paths["final_test_metrics"])
    manifest = json.loads(final_paths["final_evaluation_manifest"].read_text())
    claim = json.loads(final_paths["evaluation_claim"].read_text())
    assert metrics["split"].eq("test").all()
    assert metrics["test_prediction_batch"].eq(1).all()
    assert metrics["n_synthetic_refit"].eq(0).all()
    assert manifest["payload"]["outer_test_prediction_batches"] == 1
    assert manifest["payload"]["per_unit_test_evaluation_counts"] == {
        manifest["payload"]["evaluation_unit"]: 1
    }
    assert not manifest["payload"]["leakage_audit"]["leakage_detected"]
    assert all(
        value == 0
        for key, value in manifest["payload"]["leakage_audit"].items()
        if key.endswith("_count")
    )
    assert claim["status"] == "complete"
    assert claim["outer_test_prediction_batches"] == 1
    assert claim["final_test_metrics_sha256"] == hashlib.sha256(
        final_paths["final_test_metrics"].read_bytes()
    ).hexdigest()


def test_final_rejects_rehashed_runtime_candidate_tampering_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path, transfer_policies=[{"method": "real_only"}])
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_candidate_search",
    )
    manifest = json.loads(search_paths["search_manifest"].read_text())
    manifest["payload"]["candidate_policy_hashes"][0] = "forged-policy-hash"
    _rebind_search_and_frozen(search_paths, manifest)
    monkeypatch.setattr(
        final_module,
        "fit_supervised_autoencoder",
        lambda *args, **kwargs: pytest.fail("tampered candidate reached refit"),
    )
    output = tmp_path / "corrected_candidate_final"

    with pytest.raises(ValueError, match="candidate policies differ"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )
    assert not output.exists()


def test_final_replays_validation_winner_after_consistent_rehashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(
        tmp_path,
        transfer_policies=[{"method": "real_only"}],
        downstream_models=[
            {"name": "ridge", "params": {"alpha": 0.1}},
            {"name": "ridge", "params": {"alpha": 10.0}},
        ],
    )
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_winner_search",
    )
    metrics = pd.read_csv(search_paths["search_metrics"])
    selected_id = metrics.loc[metrics["selected_policy"], "policy_id"].iloc[0]
    loser_id = metrics.loc[~metrics["policy_id"].eq(selected_id), "policy_id"].iloc[0]
    metrics.loc[
        metrics["policy_id"].eq(loser_id) & metrics["metric"].eq("rmse"),
        "value",
    ] = -1.0
    metrics.to_csv(search_paths["search_metrics"], index=False)
    manifest = json.loads(search_paths["search_manifest"].read_text())
    manifest["payload"]["artifact_hashes"]["search_metrics"] = hashlib.sha256(
        search_paths["search_metrics"].read_bytes()
    ).hexdigest()
    _rebind_search_and_frozen(search_paths, manifest)
    monkeypatch.setattr(
        final_module,
        "fit_supervised_autoencoder",
        lambda *args, **kwargs: pytest.fail("forged winner reached refit"),
    )

    with pytest.raises(ValueError, match="deterministic validation winner"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_winner_final",
        )


def test_final_rejects_synthetic_parent_contamination_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _anonymous_transfer()
    config_path = _protocol_config(
        tmp_path,
        transfer_policies=[
            {"method": "real_only"},
            {"method": "anonymous", "transfer_config": transfer},
        ],
    )

    def feasible_search_generation(
        frame: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        config: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        del X, y, config, kwargs
        ids = frame["source_row_id"].astype(str).tolist()
        return {
            "candidate_df": pd.DataFrame(
                {"source_row_id": [ids[0]], "donor_row_id": [ids[1]]}
            ),
            "synthetic_y": np.array([50.0], dtype=np.float32),
            "metadata": {"accepted_count": 1},
        }

    def fake_search_fit(
        policy: Any,
        context: object,
        generation: object,
        *,
        metric_names: list[str],
    ) -> JointAEFitResult:
        del context, generation
        value = 1.0 if policy.config["transfer_method"] == "anonymous" else 9.0
        return JointAEFitResult(
            metrics=pd.DataFrame(
                [
                    {
                        "split": "valid",
                        "metric": metric,
                        "value": value,
                        "transfer_method": policy.config["transfer_method"],
                    }
                    for metric in metric_names
                ]
            ),
            audit={"policy_hash": policy.policy_hash, "ae_best_epoch": 1},
        )

    monkeypatch.setattr(
        search_module,
        "generate_condition_transfer_examples",
        feasible_search_generation,
    )
    monkeypatch.setattr(search_module, "_fit_joint_ae_policy", fake_search_fit)
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_parent_search",
    )

    def contaminated_generation(
        frame: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        config: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        del X, y, config, kwargs
        allowed = frame["source_row_id"].astype(str).iloc[0]
        return {
            "candidate_df": pd.DataFrame(
                {
                    "source_row_id": [allowed],
                    "donor_row_id": ["forbidden-test-source"],
                }
            ),
            "synthetic_y": np.array([50.0], dtype=np.float32),
            "metadata": {"accepted_count": 1},
        }

    monkeypatch.setattr(
        final_module,
        "generate_condition_transfer_examples",
        contaminated_generation,
    )
    monkeypatch.setattr(
        final_module,
        "predict_model",
        lambda *args, **kwargs: pytest.fail("contamination triggered prediction"),
    )
    output = tmp_path / "corrected_parent_final"
    with pytest.raises(ValueError, match="Synthetic parent leakage"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )
    assert not output.exists()


def test_failed_refit_retains_atomic_claim_and_never_materializes_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path, transfer_policies=[{"method": "real_only"}])
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_failed_refit_search",
    )
    output = tmp_path / "corrected_failed_refit_final"
    built_source_sets: list[set[str]] = []
    production_build = final_module._build_partition

    def recording_build(frame: pd.DataFrame, *args: object, **kwargs: object) -> object:
        built_source_sets.append(set(frame["source_row_id"].astype(str)))
        return production_build(frame, *args, **kwargs)

    monkeypatch.setattr(final_module, "_build_partition", recording_build)
    monkeypatch.setattr(
        final_module,
        "fit_supervised_autoencoder",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("refit failed")),
    )
    with pytest.raises(RuntimeError, match="refit failed"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )

    claim = json.loads((output / "evaluation_claim.json").read_text())
    assert claim["status"] == "refit_failed"
    assert claim["refit_error_type"] == "RuntimeError"
    assert claim["refit_error"] == "refit failed"
    assert claim["outer_test_attempts"] == 0
    assert claim["outer_test_labels_accessed"] is False
    assert claim["outer_test_prediction_batches"] == 0
    global_claim = json.loads(
        next(
            (
                search_paths["directory"] / ".outer_test_evaluation_claims"
            ).glob("*/evaluation_claim.json")
        ).read_text()
    )
    assert global_claim["status"] == "refit_failed"
    assert global_claim["outer_test_attempts"] == 0
    assert len(built_source_sets) == 4
    assert not (output / "final_test_metrics.csv").exists()
    assert not (output / "final_evaluation_manifest.json").exists()
    with pytest.raises(ValueError, match="already claimed"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )


def test_rehashed_unsupported_refit_protocol_fails_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path, transfer_policies=[{"method": "real_only"}])
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_protocol_search",
    )
    frozen = json.loads(search_paths["frozen_policy"].read_text())
    frozen["training_protocol"]["refit_validation_rows"] = "outer_test"
    frozen["frozen_policy_hash"] = stable_hash(
        {key: value for key, value in frozen.items() if key != "frozen_policy_hash"}
    )
    search_paths["frozen_policy"].write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    manifest = json.loads(search_paths["search_manifest"].read_text())
    manifest["frozen_policy_hash"] = frozen["frozen_policy_hash"]
    search_paths["search_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(
        final_module,
        "fit_supervised_autoencoder",
        lambda *args, **kwargs: pytest.fail("unsupported protocol reached refit"),
    )
    output = tmp_path / "corrected_protocol_final"
    with pytest.raises(ValueError, match="unsupported training protocol"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )
    assert not output.exists()


def test_same_frozen_unit_cannot_evaluate_into_a_second_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path, transfer_policies=[{"method": "real_only"}])
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_once_search",
    )
    run_ae_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=tmp_path / "corrected_once_first_final",
    )
    monkeypatch.setattr(
        final_module,
        "fit_supervised_autoencoder",
        lambda *args, **kwargs: pytest.fail("duplicate global unit reached refit"),
    )
    monkeypatch.setattr(
        final_module,
        "predict_model",
        lambda *args, **kwargs: pytest.fail("duplicate global unit reached test"),
    )
    second_output = tmp_path / "corrected_once_second_final"
    with pytest.raises(ValueError, match="already claimed"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=second_output,
        )
    assert not second_output.exists()
    claim_root = search_paths["directory"] / ".outer_test_evaluation_claims"
    assert len(list(claim_root.glob("*/evaluation_claim.json"))) == 1


def test_crash_after_prediction_conservatively_records_test_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path, transfer_policies=[{"method": "real_only"}])
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_crash_search",
    )
    production_metrics = final_module._metric_rows

    def crash_after_prediction(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("metric calculation crashed")

    monkeypatch.setattr(final_module, "_metric_rows", crash_after_prediction)
    output = tmp_path / "corrected_crash_final"
    with pytest.raises(RuntimeError, match="metric calculation crashed"):
        run_ae_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )
    monkeypatch.setattr(final_module, "_metric_rows", production_metrics)

    output_claim = json.loads((output / "evaluation_claim.json").read_text())
    global_claim_path = next(
        (
            search_paths["directory"] / ".outer_test_evaluation_claims"
        ).glob("*/evaluation_claim.json")
    )
    global_claim = json.loads(global_claim_path.read_text())
    for claim in (output_claim, global_claim):
        assert claim["status"] == "outer_test_prediction_complete"
        assert claim["outer_test_attempts"] == 1
        assert claim["outer_test_labels_accessed"] is True
        assert claim["outer_test_prediction_batches"] == 1
    assert not (output / "final_test_metrics.csv").exists()
    assert not (output / "final_evaluation_manifest.json").exists()


def test_successful_typed_refit_regenerates_and_manifests_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(
        tmp_path,
        transfer_policies=[
            {"method": "real_only"},
            {"method": "anonymous", "transfer_config": _anonymous_transfer()},
        ],
    )
    generation_source_pools: list[tuple[str, ...]] = []

    def deterministic_generation(
        frame: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        config: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        del y, config
        ids = tuple(frame["source_row_id"].astype(str))
        generation_source_pools.append(ids)
        label = 50.0
        candidate = pd.DataFrame(
            {
                "source_row_id": [ids[0]],
                "donor_row_id": [ids[1]],
                "canonical_reaction_hash": [f"canonical-{stable_hash(list(ids))[:16]}"],
                "feature_hash": [f"feature-{stable_hash(list(ids))[:16]}"],
                "synthetic_label": [label],
                "accepted": [True],
                "kept": [True],
            }
        )
        return {
            "candidate_df": candidate,
            "synthetic_df": pd.DataFrame(),
            "synthetic_y": np.array([label]),
            "X_synthetic": np.asarray(X[:1], dtype=np.float32),
            "metadata": {"accepted_count": 1, "source_pool_size": len(ids)},
            "feature_names": list(kwargs["real_feature_names"]),
            "feature_metadata": kwargs["real_feature_metadata"],
        }

    def select_typed_policy(
        policy: Any,
        context: object,
        generation: object,
        *,
        metric_names: list[str],
    ) -> JointAEFitResult:
        del context, generation
        value = 1.0 if policy.config["transfer_method"] == "anonymous" else 9.0
        return JointAEFitResult(
            metrics=pd.DataFrame(
                [
                    {
                        "split": "valid",
                        "metric": metric,
                        "value": value,
                        "transfer_method": policy.config["transfer_method"],
                    }
                    for metric in metric_names
                ]
            ),
            audit={"policy_hash": policy.policy_hash, "ae_best_epoch": 1},
        )

    monkeypatch.setattr(
        search_module,
        "generate_condition_transfer_examples",
        deterministic_generation,
    )
    monkeypatch.setattr(search_module, "_fit_joint_ae_policy", select_typed_policy)
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_typed_search",
    )
    monkeypatch.setattr(
        final_module,
        "generate_condition_transfer_examples",
        deterministic_generation,
    )
    prediction_batches = 0
    production_predict = final_module.predict_model

    def counting_predict(model: object, X: object) -> object:
        nonlocal prediction_batches
        prediction_batches += 1
        return production_predict(model, X)

    monkeypatch.setattr(final_module, "predict_model", counting_predict)
    final_paths = run_ae_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=tmp_path / "corrected_typed_final",
    )

    manifest = json.loads(final_paths["final_evaluation_manifest"].read_text())
    claim = json.loads(final_paths["evaluation_claim"].read_text())
    identity = manifest["payload"]["synthetic_identity"]
    assert len(generation_source_pools) == 3
    assert generation_source_pools[0] == generation_source_pools[1]
    assert set(generation_source_pools[0]) < set(generation_source_pools[2])
    assert manifest["payload"]["selected_transfer_method"] == "anonymous"
    assert manifest["payload"]["n_synthetic_refit"] == 1
    assert identity["used_synthetic_count"] == 1
    assert len(identity["accepted_canonical_reaction_hashes"]) == 1
    assert len(identity["accepted_feature_hashes"]) == 1
    assert identity["synthetic_label_hash"] == stable_hash([50.0])
    assert identity["candidate_audit_sha256"]
    assert identity["accepted_audit_sha256"]
    assert identity["generation_metadata_hash"] == stable_hash(
        identity["generation_metadata"]
    )
    assert claim["synthetic_identity"] == identity
    assert not manifest["payload"]["leakage_audit"]["leakage_detected"]
    assert prediction_batches == 1


def test_empty_typed_candidate_audit_schema_replays_semantically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(
        tmp_path,
        transfer_policies=[
            {"method": "real_only"},
            {"method": "anonymous", "transfer_config": _anonymous_transfer()},
        ],
    )

    def empty_generation(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        return {
            "candidate_df": pd.DataFrame(
                columns=["source_row_id", "donor_row_id"]
            ),
            "synthetic_y": np.empty(0, dtype=np.float32),
            "metadata": {"accepted_count": 0},
        }

    monkeypatch.setattr(
        search_module,
        "generate_condition_transfer_examples",
        empty_generation,
    )
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_empty_typed_search",
    )
    monkeypatch.setattr(
        final_module,
        "generate_condition_transfer_examples",
        empty_generation,
    )
    final_paths = run_ae_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=tmp_path / "corrected_empty_typed_final",
    )

    search_candidate_audit = pd.read_csv(
        search_paths["synthetic_candidate_audit"]
    )
    final_manifest = json.loads(
        final_paths["final_evaluation_manifest"].read_text()
    )
    assert search_candidate_audit.empty
    assert set(search_candidate_audit) == {
        "transfer_key",
        "source_row_id",
        "donor_row_id",
    }
    assert final_manifest["payload"]["selected_transfer_method"] == "real_only"


def _protocol_config(
    tmp_path: Path,
    *,
    transfer_policies: list[dict[str, Any]],
    downstream_models: list[dict[str, Any]] | None = None,
) -> Path:
    base_path = write_corrected_config(
        tmp_path,
        kind="anonymous",
        output_name="corrected_unused_combined_ae",
    )
    config = yaml.safe_load(base_path.read_text())
    config.pop("condition_transfer")
    config.pop("models")
    config["metrics"] = ["rmse", "mae"]
    config["ae_policy_search"] = {
        "selection_metric": "rmse",
        "lower_is_better": True,
        "inner_split_seed_offset": 20_000,
        "transfer_policies": transfer_policies,
        "latent_dims": [4],
        "synthetic_supervised_weights": [0.5],
        "synthetic_reconstruction_weights": [0.1],
        "downstream_models": downstream_models
        or [{"name": "ridge", "params": {"alpha": 1.0}}],
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
    config["output"] = {
        "ae_search_directory": str(tmp_path / "corrected_default_ae_search"),
        "ae_final_directory": str(tmp_path / "corrected_default_ae_final"),
    }
    path = tmp_path / "joint_ae_final_protocol.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _anonymous_transfer() -> dict[str, Any]:
    return {
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


def _rebind_search_and_frozen(
    search_paths: dict[str, Path],
    manifest: dict[str, Any],
) -> None:
    manifest["search_manifest_hash"] = stable_hash(manifest["payload"])
    frozen = json.loads(search_paths["frozen_policy"].read_text())
    frozen["search_manifest_hash"] = manifest["search_manifest_hash"]
    frozen["frozen_policy_hash"] = stable_hash(
        {key: value for key, value in frozen.items() if key != "frozen_policy_hash"}
    )
    manifest["frozen_policy_hash"] = frozen["frozen_policy_hash"]
    search_paths["frozen_policy"].write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    search_paths["search_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
