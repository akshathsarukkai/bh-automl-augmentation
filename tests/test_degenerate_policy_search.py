"""Degenerate candidate pools under the frozen-policy protocol.

Preregistration section 6 of ``PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md``
expects the globally-unmeasured control arm to accept zero synthetic rows on
most units and requires those units to be *reported with their counts*, not
scored as augmentation and not treated as a crash.  These tests pin that
behaviour: without the flag a degenerate pool still refuses to be credited as
augmentation; with the flag it is recorded, hash-sealed, and the unit's outer
test is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import bh_augmentation.policy_search as search_module
from bh_augmentation.augmentation.pool_accounting import pool_statistics
from bh_augmentation.candidate_scope_reanalysis import _pool_statistics, _resolved_models
from bh_augmentation.final_evaluation import run_final_evaluation_command
from bh_augmentation.policy_search import (
    DEGENERATE_POOL_MESSAGE,
    DegenerateAugmentationPool,
    load_degenerate_unit,
    run_policy_search_command,
)
from corrected_test_utils import write_corrected_config

pytest.importorskip("rdkit")

ANONYMOUS_TRANSFER = {
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


def _typed_config(tmp_path: Path, *, policies: list[dict]) -> Path:
    fixture_path = write_corrected_config(
        tmp_path,
        kind="anonymous",
        output_name="corrected_unused_combined_output",
    )
    config = yaml.safe_load(fixture_path.read_text())
    config["metrics"] = ["rmse", "mae"]
    config["candidate_scope"] = {"mode": "observed_only_low_data"}
    config["policy_search"] = {
        "selection_metric": "rmse",
        "lower_is_better": True,
        "model_policies": policies,
    }
    config["output"] = {
        "search_directory": str(tmp_path / "corrected_default_search"),
        "final_directory": str(tmp_path / "corrected_default_final"),
    }
    path = tmp_path / "typed_protocol.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _anonymous_policy(policy_id: str, *, random_state: int = 0) -> dict:
    return {
        "policy_id": policy_id,
        "method": "anonymous",
        "model": "ridge",
        "params": {"alpha": 1.0},
        "transfer_config": {**ANONYMOUS_TRANSFER, "random_state": random_state},
    }


def _emptying_generator(monkeypatch: pytest.MonkeyPatch, *, degenerate_states: set[int]):
    """Wrap the production generator so selected policies accept nothing.

    The wrapped result keeps the real feature contract, so the degeneracy is
    decided by the zero-row guard and not by a compatibility failure upstream.
    """
    production = search_module.generate_condition_transfer_examples
    calls: list[int] = []

    def generator(*args, **kwargs):
        result = production(*args, **kwargs)
        transfer = args[3]
        calls.append(int(transfer.random_state))
        if int(transfer.random_state) in degenerate_states:
            candidate_df = result["candidate_df"].copy()
            candidate_df["accepted"] = False
            if "kept" in candidate_df:
                candidate_df["kept"] = False
            candidate_df["rejection_reason"] = "observed_in_labeled_train"
            result = {
                **result,
                "candidate_df": candidate_df,
                "X_synthetic": np.empty((0, result["X_synthetic"].shape[1]), dtype=np.float32),
                "synthetic_y": np.empty(0, dtype=float),
            }
        return result

    monkeypatch.setattr(search_module, "generate_condition_transfer_examples", generator)
    return calls


def test_zero_accepted_pool_refuses_to_be_credited_as_augmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _typed_config(tmp_path, policies=[_anonymous_policy("anon-0")])
    _emptying_generator(monkeypatch, degenerate_states={0})

    with pytest.raises(DegenerateAugmentationPool) as excinfo:
        run_policy_search_command(config_path, output_directory=tmp_path / "corrected_search")

    assert str(excinfo.value) == DEGENERATE_POOL_MESSAGE
    assert isinstance(excinfo.value, ValueError)
    stats = excinfo.value.pool_statistics
    assert stats["accepted_candidate_count"] == 0
    assert stats["generated_candidate_count"] > 0
    assert stats["rejected_observed_in_labeled_train"] == stats["generated_candidate_count"]
    assert stats["candidate_scope_mode"] == "observed_only_low_data"
    assert not (tmp_path / "corrected_search").exists()


def test_record_degenerate_writes_sealed_record_and_no_frozen_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _typed_config(
        tmp_path,
        policies=[_anonymous_policy("anon-0"), _anonymous_policy("anon-1")],
    )
    _emptying_generator(monkeypatch, degenerate_states={0})
    monkeypatch.setattr(
        search_module,
        "run_policy_search",
        lambda *args, **kwargs: pytest.fail("a degenerate unit must not be searched"),
    )

    paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_degenerate",
        record_degenerate=True,
    )

    assert set(paths) == {"directory", "degenerate_unit"}
    assert {p.name for p in paths["directory"].iterdir()} == {"degenerate_unit.json"}
    record = load_degenerate_unit(paths["degenerate_unit"])
    payload = record["payload"]
    assert payload["family"] is None
    assert payload["candidate_scope_mode"] == "observed_only_low_data"
    assert payload["reason"] == DEGENERATE_POOL_MESSAGE
    assert [entry["policy_id"] for entry in payload["per_policy"]] == ["anon-0", "anon-1"]
    for entry in payload["per_policy"]:
        assert entry["method"] == "anonymous"
        assert entry["pool_statistics"]["accepted_candidate_count"] == 0
        assert entry["pool_statistics"]["generated_candidate_count"] > 0
    assert payload["outer_test_labels_accessed"] is False
    assert payload["frozen_policy_written"] is False
    assert payload["registry_claimed"] is False
    assert len(payload["candidate_policy_hashes"]) == 2
    assert payload["scientific_binding"]["config_hash"]

    # A degenerate unit has no frozen policy, so the outer test is unreachable.
    with pytest.raises((FileNotFoundError, ValueError)):
        run_final_evaluation_command(
            config_path,
            frozen_policy_path=paths["directory"] / "frozen_policy.json",
            search_manifest_path=paths["directory"] / "search_manifest.json",
            output_directory=tmp_path / "corrected_final_never",
        )
    assert not (tmp_path / "corrected_final_never" / "final_test_metrics.csv").exists()


def test_tampered_degenerate_record_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _typed_config(tmp_path, policies=[_anonymous_policy("anon-0")])
    _emptying_generator(monkeypatch, degenerate_states={0})
    paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_degenerate",
        record_degenerate=True,
    )
    record = json.loads(paths["degenerate_unit"].read_text())

    record["payload"]["per_policy"][0]["pool_statistics"]["accepted_candidate_count"] = 5
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_degenerate_unit(tampered)

    record = json.loads(paths["degenerate_unit"].read_text())
    record["payload"]["registry_claimed"] = True
    record["degenerate_unit_hash"] = search_module.stable_hash(record["payload"])
    tampered.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="forbidden activity"):
        load_degenerate_unit(tampered)


def test_mixed_degenerate_and_healthy_policies_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _typed_config(
        tmp_path,
        policies=[
            _anonymous_policy("anon-0", random_state=0),
            _anonymous_policy("anon-1", random_state=1),
        ],
    )
    _emptying_generator(monkeypatch, degenerate_states={1})

    with pytest.raises(DegenerateAugmentationPool):
        run_policy_search_command(
            config_path,
            output_directory=tmp_path / "corrected_mixed",
            record_degenerate=True,
        )
    assert not (tmp_path / "corrected_mixed").exists()


def test_record_degenerate_on_healthy_pool_adds_accounting_and_final_still_runs(
    tmp_path: Path,
) -> None:
    config_path = _typed_config(
        tmp_path,
        policies=[
            {"policy_id": "real-ridge", "model": "ridge", "params": {"alpha": 1.0}},
            _anonymous_policy("anon-0"),
        ],
    )

    paths = run_policy_search_command(
        config_path,
        output_directory=tmp_path / "corrected_search",
        record_degenerate=True,
    )

    assert {p.name for p in paths["directory"].iterdir()} == {
        "frozen_policy.json",
        "search_metrics.csv",
        "search_manifest.json",
        "pool_accounting.json",
    }
    accounting = json.loads(paths["pool_accounting"].read_text())
    payload = accounting["payload"]
    assert accounting["pool_accounting_hash"] == search_module.stable_hash(payload)
    manifest = json.loads(paths["search_manifest"].read_text())
    assert payload["search_manifest_hash"] == manifest["search_manifest_hash"]
    assert payload["frozen_policy_hash"] == manifest["frozen_policy_hash"]
    by_id = {entry["policy_id"]: entry for entry in payload["per_policy"]}
    assert by_id["real-ridge"]["method"] == "real_only"
    assert by_id["real-ridge"]["pool_statistics"]["generated_candidate_count"] == 0
    assert by_id["anon-0"]["pool_statistics"]["accepted_candidate_count"] > 0
    assert by_id["anon-0"]["pool_statistics"]["candidate_scope_mode"] == "observed_only_low_data"

    # The sidecar does not disturb the frozen-policy protocol downstream.
    final_paths = run_final_evaluation_command(
        config_path,
        frozen_policy_path=paths["frozen_policy"],
        search_manifest_path=paths["search_manifest"],
        output_directory=tmp_path / "corrected_final",
    )
    final_metrics = pd.read_csv(final_paths["final_test_metrics"])
    assert final_metrics["split"].eq("test").all()


def test_pool_statistics_is_the_reanalysis_definition() -> None:
    from bh_augmentation.augmentation.candidate_scope import observed_only_scope

    scope = observed_only_scope(labeled_train_identity_keys=["a", "b"])
    frame = pd.DataFrame(
        {
            "canonical_reaction_key": ["x", "x", "y", "z"],
            "accepted": [True, True, False, False],
            "kept": [True, False, False, False],
            "rejection_reason": ["", "", "observed_in_labeled_train", "weird"],
        }
    )
    stats = pool_statistics(frame, scope)
    assert stats == _pool_statistics(frame, scope)
    assert stats["generated_candidate_count"] == 4
    assert stats["accepted_candidate_count"] == 2
    assert stats["unique_accepted_identity_count"] == 1
    assert stats["selected_candidate_count"] == 1
    assert stats["rejected_observed_in_labeled_train"] == 1
    assert stats["rejected_other"] == 1
    assert stats["candidate_scope_mode"] == "observed_only_low_data"
    json.dumps(stats)


def test_resolved_models_flatten_nested_params() -> None:
    resolved = _resolved_models(
        {"models": [{"name": "xgboost", "params": {"n_estimators": 300, "max_depth": 4}}]}
    )
    assert resolved == [("xgboost", {"n_estimators": 300, "max_depth": 4})]
    assert _resolved_models({"models": ["ridge"]}) == [("ridge", {})]
    with pytest.raises(ValueError, match="Unsupported model specification keys"):
        _resolved_models({"models": [{"name": "xgboost", "n_estimators": 300}]})
    with pytest.raises(ValueError, match="must be a mapping"):
        _resolved_models({"models": [{"name": "xgboost", "params": 3}]})
