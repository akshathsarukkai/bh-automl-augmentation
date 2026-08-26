"""Integration tests for separated policy search and final evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import bh_augmentation.final_evaluation as final_module
from bh_augmentation.augmentation.candidate_scope import GLOBALLY_UNMEASURED_PROSPECTIVE
from bh_augmentation.final_evaluation import run_final_evaluation_command
from bh_augmentation.policy_search import run_policy_search_command
from bh_augmentation.utils.corrected_runs import stable_hash
from corrected_test_utils import write_corrected_config


def _protocol_config(tmp_path: Path) -> Path:
    fixture_path = write_corrected_config(
        tmp_path,
        kind="anonymous",
        output_name="corrected_unused_combined_output",
    )
    config = yaml.safe_load(fixture_path.read_text())
    config["metrics"] = ["rmse", "mae"]
    config["policy_search"] = {
        "selection_metric": "rmse",
        "lower_is_better": True,
        "model_policies": [
            {"policy_id": "ridge-alpha-01", "model": "ridge", "params": {"alpha": 0.1}},
            {"policy_id": "ridge-alpha-1", "model": "ridge", "params": {"alpha": 1.0}},
        ],
    }
    config["output"] = {
        "search_directory": str(tmp_path / "corrected_default_search"),
        "final_directory": str(tmp_path / "corrected_default_final"),
    }
    path = tmp_path / "policy_protocol.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_search_and_final_commands_keep_outer_test_separated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_run",
    )

    assert {path.name for path in search_paths["directory"].iterdir()} == {
        "frozen_policy.json",
        "search_metrics.csv",
        "search_manifest.json",
    }
    search_metrics = pd.read_csv(search_paths["search_metrics"])
    search_manifest = json.loads(search_paths["search_manifest"].read_text())
    assert search_metrics["split"].eq("valid").all()
    assert search_manifest["payload"]["selection_data_roles"] == ["train", "valid"]
    assert search_manifest["payload"]["outer_test_labels_accessed"] is False
    assert search_manifest["payload"]["outer_test_predictions_generated"] is False
    assert search_manifest["payload"]["outer_test_metric_evaluations"] == 0
    assert search_manifest["payload"]["test_evaluated"] is False

    prediction_batches = 0
    production_predict = final_module.predict_model

    def counting_predict(model: object, X: object) -> object:
        nonlocal prediction_batches
        prediction_batches += 1
        return production_predict(model, X)

    monkeypatch.setattr(final_module, "predict_model", counting_predict)
    final_paths = run_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=tmp_path / "corrected_final_run",
    )

    assert prediction_batches == 1
    assert {path.name for path in final_paths["directory"].iterdir()} == {
        "final_test_metrics.csv",
        "final_evaluation_manifest.json",
        "evaluation_claim.json",
    }
    final_metrics = pd.read_csv(final_paths["final_test_metrics"])
    final_manifest = json.loads(final_paths["final_evaluation_manifest"].read_text())
    assert final_metrics["split"].eq("test").all()
    assert final_metrics["method"].eq("real_only").all()
    assert final_metrics["test_prediction_batch"].eq(1).all()
    assert final_manifest["payload"]["test_evaluated"] is True
    assert final_manifest["payload"]["outer_test_prediction_batches"] == 1
    claim = json.loads(final_paths["evaluation_claim"].read_text())
    assert claim["schema_version"] == "bh-final-evaluation-claim-v1"
    assert claim["status"] == "complete"
    assert claim["outer_test_prediction_batches"] == 1
    unit = final_manifest["payload"]["evaluation_unit"]
    assert final_manifest["payload"]["per_unit_test_evaluation_counts"] == {unit: 1}


def test_final_refuses_unfrozen_policy_before_test_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_unfrozen",
    )
    frozen = json.loads(search_paths["frozen_policy"].read_text())
    frozen["status"] = "draft"
    frozen["frozen_policy_hash"] = stable_hash(
        {key: value for key, value in frozen.items() if key != "frozen_policy_hash"}
    )
    search_paths["frozen_policy"].write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    build_calls = 0
    production_build = final_module._build_partition

    def counting_build(*args: object, **kwargs: object) -> object:
        nonlocal build_calls
        build_calls += 1
        return production_build(*args, **kwargs)

    monkeypatch.setattr(final_module, "_build_partition", counting_build)
    with pytest.raises(ValueError, match="unfrozen"):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_final_unfrozen",
        )
    assert build_calls == 1


def test_final_refuses_binding_and_search_artifact_mismatches_before_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_mismatch",
    )
    config = yaml.safe_load(config_path.read_text())
    config["policy_search"]["model_policies"][0]["params"]["alpha"] = 2.0
    mismatched_config = tmp_path / "mismatched_protocol.yaml"
    mismatched_config.write_text(yaml.safe_dump(config, sort_keys=False))

    fit_calls = 0

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("Refit occurred before binding verification.")

    monkeypatch.setattr(final_module, "_fit_resolved_model", forbidden_fit)
    with pytest.raises(ValueError, match="binding mismatch"):
        run_final_evaluation_command(
            mismatched_config,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_final_mismatch",
        )
    assert fit_calls == 0

    search_paths["search_metrics"].write_text(
        search_paths["search_metrics"].read_text() + "\n"
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_final_tampered_metrics",
        )
    assert fit_calls == 0


def test_final_recomputes_winner_and_rejects_consistently_rehashed_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_replay",
    )
    metrics = pd.read_csv(search_paths["search_metrics"])
    selected_id = metrics.loc[metrics["selected_policy"], "policy_id"].iloc[0]
    loser_id = metrics.loc[~metrics["policy_id"].eq(selected_id), "policy_id"].iloc[0]
    metrics.loc[
        metrics["policy_id"].eq(loser_id) & metrics["metric"].eq("rmse"), "value"
    ] = -1.0
    metrics.to_csv(search_paths["search_metrics"], index=False)

    manifest = json.loads(search_paths["search_manifest"].read_text())
    manifest["payload"]["search_metrics_sha256"] = hashlib.sha256(
        search_paths["search_metrics"].read_bytes()
    ).hexdigest()
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
    monkeypatch.setattr(
        final_module,
        "_fit_resolved_model",
        lambda *args, **kwargs: pytest.fail("tampered winner reached refit"),
    )

    with pytest.raises(ValueError, match="deterministic validation winner"):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_final_replay",
        )


def test_final_rejects_rehashed_unsupported_training_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_protocol",
    )
    frozen = json.loads(search_paths["frozen_policy"].read_text())
    frozen["training_protocol"]["validation_rows"] = "silently_refit"
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
        "_fit_resolved_model",
        lambda *args, **kwargs: pytest.fail("unsupported protocol reached refit"),
    )

    with pytest.raises(ValueError, match="unsupported training protocol"):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_final_protocol",
        )


def test_failed_refit_retains_claim_and_prevents_test_access_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_refit_failure",
    )
    output = tmp_path / "corrected_final_refit_failure"
    build_calls = 0
    production_build = final_module._build_partition

    def counting_build(*args: object, **kwargs: object) -> object:
        nonlocal build_calls
        build_calls += 1
        return production_build(*args, **kwargs)

    monkeypatch.setattr(final_module, "_build_partition", counting_build)
    monkeypatch.setattr(
        final_module,
        "_fit_resolved_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("refit failed")),
    )
    with pytest.raises(RuntimeError, match="refit failed"):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )
    assert build_calls == 1
    claim = json.loads((output / "evaluation_claim.json").read_text())
    assert claim["status"] == "claimed"
    assert claim["outer_test_prediction_batches"] == 0

    monkeypatch.setattr(
        final_module,
        "predict_model",
        lambda *args, **kwargs: pytest.fail("failed claim triggered test prediction"),
    )
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=output,
        )
    assert build_calls == 2


def test_search_redacts_outer_test_outcomes_before_command_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bh_augmentation.policy_search as search_module

    config_path = _protocol_config(tmp_path)
    production_materialize = search_module._materialize_search_frames
    redaction_observed = False

    def inspecting_materialize(saved: object, **kwargs: object) -> object:
        nonlocal redaction_observed
        rows = saved.low_data_assignments.loc[
            saved.low_data_assignments["seed"].eq(kwargs["seed"])
            & saved.low_data_assignments["train_fraction"].eq(kwargs["train_fraction"])
        ]
        test_ids = set(rows.loc[rows["outer_split"].eq("test"), "source_row_id"])
        hidden = saved.canonical.loc[saved.canonical["source_row_id"].isin(test_ids)]
        redaction_observed = bool(hidden["yield"].isna().all())
        return production_materialize(saved, **kwargs)

    monkeypatch.setattr(search_module, "_materialize_search_frames", inspecting_materialize)
    run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_redaction",
    )
    assert redaction_observed


@pytest.mark.parametrize("method", ["anonymous", "role_aware"])
def test_typed_transfer_search_adapters_remain_validation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    import bh_augmentation.policy_search as search_module

    config_path = _protocol_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    if method == "anonymous":
        transfer_config = {
            "donor_strategy": "random",
            "synthetic_multiplier": 0.1,
            "n_neighbors": 3,
            "label_strategy": "source_label",
            "teacher_models": ["ridge"],
            "max_teacher_std": None,
            "min_similarity": None,
            "high_yield_threshold": 70.0,
            "clip_y_min": 0.0,
            "clip_y_max": 100.0,
            "candidates_per_real": 2,
            "random_state": 0,
            "donor_similarity_backend": "rdkit",
            "role_change_requirement": "all",
            "fallback_policy": "reject",
        }
    else:
        transfer_config = {
            "role_transfer_mode": "ligand_base_solvent_or_additive",
            "donor_strategy": "random",
            "label_strategy": "source_label",
            "synthetic_multiplier": 0.1,
            "max_candidates_per_source": 2,
            "teacher_models": ["ridge"],
            "max_teacher_std": None,
            "min_similarity": 0.0,
            "random_state": 0,
            "donor_similarity_backend": "rdkit",
            "role_change_requirement": "all",
            "fallback_policy": "reject",
        }
    config["policy_search"]["model_policies"] = [
        {
            "policy_id": f"{method}-ridge",
            "method": method,
            "model": "ridge",
            "params": {"alpha": 1.0},
            "transfer_config": transfer_config,
        }
    ]
    typed_config = tmp_path / f"{method}_protocol.yaml"
    typed_config.write_text(yaml.safe_dump(config, sort_keys=False))
    production_final_predict = final_module.predict_model
    monkeypatch.setattr(
        final_module,
        "predict_model",
        lambda *args, **kwargs: pytest.fail("search called final-module prediction"),
    )
    paths = run_policy_search_command(
        typed_config,
        output_directory=tmp_path / f"corrected_search_{method}",
    )
    metrics = pd.read_csv(paths["search_metrics"])
    manifest = json.loads(paths["search_manifest"].read_text())
    assert metrics["split"].eq("valid").all()
    assert manifest["payload"]["test_evaluated"] is False
    assert manifest["payload"]["supported_methods"] == [
        "real_only",
        "anonymous",
        "role_aware",
    ]

    generator_name = (
        "generate_condition_transfer_examples"
        if method == "anonymous"
        else "generate_role_aware_condition_transfer_examples"
    )
    production_generator = getattr(search_module, generator_name)
    observed_scopes: list[object] = []

    def recording_generator(*args: object, **kwargs: object) -> object:
        observed_scopes.append(kwargs["candidate_scope"])
        return production_generator(*args, **kwargs)

    prediction_batches = 0

    def counting_predict(model: object, X: object) -> object:
        nonlocal prediction_batches
        prediction_batches += 1
        return production_final_predict(model, X)

    monkeypatch.setattr(search_module, generator_name, recording_generator)
    monkeypatch.setattr(final_module, "predict_model", counting_predict)
    final_paths = run_final_evaluation_command(
        typed_config,
        frozen_policy_path=paths["frozen_policy"],
        search_manifest_path=paths["search_manifest"],
        output_directory=tmp_path / f"corrected_final_{method}",
    )
    final_metrics = pd.read_csv(final_paths["final_test_metrics"])
    final_manifest = json.loads(final_paths["final_evaluation_manifest"].read_text())
    assert prediction_batches == 1
    # Absent an explicit candidate_scope block the historical prospective rule
    # applies: eligibility is decided against the complete measured universe,
    # which is strictly larger than the refit partition.
    scope = observed_scopes[-1]
    assert scope.mode == GLOBALLY_UNMEASURED_PROSPECTIVE
    assert len(scope.rejection_identity_keys()) > int(final_metrics["n_refit"].iloc[0])
    assert final_metrics["method"].eq(method).all()
    assert final_manifest["payload"]["selected_method"] == method
    assert final_manifest["payload"]["outer_test_prediction_batches"] == 1


def test_output_paths_are_not_scientific_and_final_output_is_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _protocol_config(tmp_path)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search_output_hash",
    )
    config = yaml.safe_load(config_path.read_text())
    config["output"] = {
        "search_directory": "results/a_different_search_path",
        "final_directory": "results/a_different_final_path",
    }
    changed_output_config = tmp_path / "changed_output_protocol.yaml"
    changed_output_config.write_text(yaml.safe_dump(config, sort_keys=False))
    final_output = tmp_path / "corrected_final_output_hash"
    run_final_evaluation_command(
        changed_output_config,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=final_output,
    )

    prediction_calls = 0

    def forbidden_predict(*args: object, **kwargs: object) -> object:
        nonlocal prediction_calls
        prediction_calls += 1
        raise AssertionError("Reused final output triggered another test prediction.")

    monkeypatch.setattr(final_module, "predict_model", forbidden_predict)
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        run_final_evaluation_command(
            changed_output_config,
            frozen_policy_path=search_paths["frozen_policy"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=final_output,
        )
    assert prediction_calls == 0
