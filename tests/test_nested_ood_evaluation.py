from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import bh_augmentation.nested_ood_final as final_module
import bh_augmentation.nested_ood_search as search_module
from bh_augmentation.data.saved_canonical_splits import load_saved_canonical_splits
from bh_augmentation.evaluation.evaluation_registry import (
    EvaluationIdentity,
    EvaluationRegistry,
)
from bh_augmentation.evaluation.nested_ood import build_nested_group_ood_contract
from bh_augmentation.nested_ood_common import resolve_contract, scientific_projection
from bh_augmentation.nested_ood_final import run_nested_ood_final
from bh_augmentation.nested_ood_search import run_nested_ood_search
from bh_augmentation.utils.config import load_config
from bh_augmentation.utils.corrected_runs import stable_hash
from corrected_test_utils import write_corrected_config


@pytest.fixture(autouse=True)
def _isolated_evaluation_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        final_module,
        "EVALUATION_REGISTRY_DIRECTORY",
        tmp_path / "evaluation_registry",
    )


def _nested_config(tmp_path: Path) -> Path:
    base_path = write_corrected_config(
        tmp_path,
        kind="anonymous",
        output_name="unused_corrected_nested_output",
    )
    config = yaml.safe_load(base_path.read_text())
    config["splits"] = {
        "method": "nested_group_ood",
        "canonical_dependency_directory": config["splits"]["directory"],
    }
    config["features"]["kind"] = "bh_role_separated"
    config["metrics"] = ["rmse", "mae"]
    config["nested_ood"] = {
        "targets": ["product_key"],
        "selection_metric": "rmse",
        "lower_is_better": True,
        "selection_weighting": "group_weighted",
        "policies": [
            {
                "policy_id": "ridge-01",
                "method": "real_only",
                "model": "ridge",
                "params": {"alpha": 0.1},
            },
            {
                "policy_id": "ridge-1",
                "method": "real_only",
                "model": "ridge",
                "params": {"alpha": 1.0},
            },
        ],
    }
    config["output"] = {
        "nested_ood_search_directory": str(tmp_path / "corrected_nested_search"),
        "nested_ood_final_directory": str(tmp_path / "corrected_nested_final"),
    }
    config.pop("condition_transfer")
    path = tmp_path / "nested_ood.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_nested_ood_search_and_final_are_separate_group_disjoint_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _nested_config(tmp_path)
    inner_prediction_batches = 0
    production_search_predict = search_module.predict_model

    def count_inner_predict(model: object, X: object) -> object:
        nonlocal inner_prediction_batches
        inner_prediction_batches += 1
        return production_search_predict(model, X)

    monkeypatch.setattr(search_module, "predict_model", count_inner_predict)
    search_paths = run_nested_ood_search(config_path)
    inner = pd.read_csv(search_paths["inner_metrics"])
    aggregates = pd.read_csv(search_paths["aggregate_metrics"])
    definitions = pd.read_csv(search_paths["fold_definitions"])
    frozen = json.loads(search_paths["frozen_policies"].read_text())
    search_manifest = json.loads(search_paths["search_manifest"].read_text())

    assert not inner["outer_test_labels_accessed"].any()
    assert not inner["outer_test_predictions_generated"].any()
    assert search_manifest["outer_test_metric_evaluations"] == 0
    assert search_manifest["target_plans"]["product_key"]["expected_outer_groups"]
    assert definitions["all_overlaps_zero"].all()
    assert inner_prediction_batches == len(definitions) * 2
    assert set(aggregates["weighting"]) == {"group_weighted", "sample_weighted"}
    assert len(frozen["policies"]) == inner["outer_group"].nunique()
    assert frozen["status"] == "frozen"
    for selection in frozen["policies"]:
        candidate = aggregates.loc[
            aggregates["evaluation_unit"].eq(selection["evaluation_unit"])
            & aggregates["weighting"].eq(selection["selection_weighting"])
            & aggregates["metric"].eq(selection["selection_metric"])
        ].sort_values(["value", "policy_hash", "policy_id"], kind="mergesort")
        assert candidate.iloc[0]["policy_id"] == selection["policy"]["policy_id"]
        assert candidate.iloc[0]["value"] == pytest.approx(
            selection["selection_value"]
        )

    prediction_batches = 0
    production_predict = final_module.predict_model

    def count_predict(model: object, X: object) -> object:
        nonlocal prediction_batches
        prediction_batches += 1
        return production_predict(model, X)

    monkeypatch.setattr(final_module, "predict_model", count_predict)
    final_paths = run_nested_ood_final(
        config_path,
        frozen_policies_path=search_paths["frozen_policies"],
        search_manifest_path=search_paths["search_manifest"],
    )
    outer = pd.read_csv(final_paths["outer_metrics"])
    final_manifest = json.loads(final_paths["final_manifest"].read_text())
    summary = pd.read_csv(final_paths["outer_summary"])

    assert prediction_batches == len(frozen["policies"])
    assert outer["outer_test_prediction_batch"].eq(1).all()
    assert outer["outer_test_metric_evaluation_count"].eq(1).all()
    assert final_manifest["status"] == "complete"
    assert final_manifest["expected_evaluation_units"] == final_manifest[
        "completed_evaluation_units"
    ]
    assert set(final_manifest["per_unit_test_prediction_batches"].values()) == {1}
    assert set(summary["weighting"]) == {"group_weighted", "sample_weighted"}
    plan_sizes = {
        fold["outer_group"]: fold
        for fold in search_manifest["target_plans"]["product_key"][
            "fold_definitions"
        ]
    }
    for row in outer.to_dict(orient="records"):
        definition = plan_sizes[row["outer_group"]]
        assert row["n_refit"] == definition["outer_train_size"]
        assert row["n_test"] == definition["outer_test_size"]
    claims = list(final_module.EVALUATION_REGISTRY_DIRECTORY.rglob("*.json"))
    assert len(claims) == prediction_batches
    assert all(
        json.loads(path.read_text())["status"] == "metrics_complete"
        for path in claims
    )

    copied_search = tmp_path / "copied_corrected_nested_search"
    shutil.copytree(search_paths["directory"], copied_search)
    reused_paths = run_nested_ood_final(
        config_path,
        frozen_policies_path=copied_search / "frozen_policies.json",
        search_manifest_path=copied_search / "search_manifest.json",
        output_directory=tmp_path / "corrected_nested_second_final_root",
    )
    assert prediction_batches == len(frozen["policies"])
    pd.testing.assert_frame_equal(
        outer,
        pd.read_csv(reused_paths["outer_metrics"]),
        check_exact=True,
    )
    reused_manifest = json.loads(reused_paths["final_manifest"].read_text())
    assert set(
        reused_manifest[
            "per_unit_test_prediction_batches_this_invocation"
        ].values()
    ) == {0}


def test_nested_ood_final_rejects_tampered_inner_fold_evidence_before_output(
    tmp_path: Path,
) -> None:
    config_path = _nested_config(tmp_path)
    search_paths = run_nested_ood_search(config_path)
    inner = pd.read_csv(search_paths["inner_metrics"])
    inner.loc[0, "inner_assignment_hash"] = "tampered"
    inner.to_csv(search_paths["inner_metrics"], index=False)
    final_output = tmp_path / "corrected_nested_tampered_final"

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_nested_ood_final(
            config_path,
            frozen_policies_path=search_paths["frozen_policies"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=final_output,
        )

    assert not final_output.exists()


def test_invalid_final_output_does_not_consume_outer_evaluation_claims(
    tmp_path: Path,
) -> None:
    config_path = _nested_config(tmp_path)
    search_paths = run_nested_ood_search(config_path)
    occupied = tmp_path / "corrected_nested_occupied_final"
    occupied.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_nested_ood_final(
            config_path,
            frozen_policies_path=search_paths["frozen_policies"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=occupied,
        )

    assert not final_module.EVALUATION_REGISTRY_DIRECTORY.exists()


def test_nested_ood_search_refuses_invalid_historical_output_path(
    tmp_path: Path,
) -> None:
    config_path = _nested_config(tmp_path)
    invalid = tmp_path / "results" / "stress" / "logo_reactant" / "nested"

    with pytest.raises(ValueError, match="historical_logo_reactant"):
        run_nested_ood_search(config_path, output_directory=invalid)

    assert not invalid.exists()


def test_outer_search_helper_never_receives_poisoned_outer_test_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _nested_config(tmp_path)
    config = load_config(config_path)
    resolved = resolve_contract(config)
    saved = load_saved_canonical_splits(
        resolved["dataset_path"],
        resolved["split_directory"],
    )
    group_column = "canonical_product_key"
    identity = search_module._identity_frame(saved.canonical, group_column)
    outer_group = sorted(identity[group_column].unique())[0]
    contract = build_nested_group_ood_contract(
        identity,
        group_column=group_column,
        outer_group=outer_group,
    )
    poisoned = saved.canonical.copy()
    poisoned.loc[
        poisoned["source_row_id"].isin(contract.outer_test_source_ids),
        "yield",
    ] = np.nan
    outer_train = search_module._rows_by_id(
        poisoned,
        contract.outer_train_source_ids,
    )
    seen_ids: set[str] = set()
    production_builder = search_module.build_partition

    def spy_partition(frame: pd.DataFrame, feature_config: dict) -> tuple:
        assert frame["yield"].notna().all()
        seen_ids.update(frame["source_row_id"].astype(str))
        return production_builder(frame, feature_config)

    monkeypatch.setattr(search_module, "build_partition", spy_partition)

    result = search_module._search_outer(
        outer_train=outer_train,
        contract=contract,
        target="product_key",
        outer_index=0,
        resolved=resolved,
        dataset_hash=saved.dataset_hash,
        split_dependency_hash=saved.aggregate_split_hash,
        config_hash=stable_hash(scientific_projection(config)),
        commit="test-commit",
    )

    assert result["frozen"]["outer_group"] == outer_group
    assert seen_ids
    assert seen_ids.isdisjoint(contract.outer_test_source_ids)


def test_search_manifest_loader_rejects_every_forbidden_test_access_claim(
    tmp_path: Path,
) -> None:
    base = {
        "schema_version": "bh-nested-ood-search-v1",
        "status": "complete",
        "selection_data_roles": ["inner_train", "inner_validation"],
        "outer_test_labels_accessed": False,
        "outer_test_predictions_generated": False,
        "outer_test_metric_evaluations": 0,
        "test_evaluated": False,
        "output_hashes": {},
        "fold_manifest_hashes": {},
    }
    mutations = [
        ("outer_test_labels_accessed", True),
        ("outer_test_predictions_generated", True),
        ("outer_test_metric_evaluations", 1),
        ("test_evaluated", True),
        ("selection_data_roles", ["train", "test"]),
    ]
    for index, (key, value) in enumerate(mutations):
        document = copy.deepcopy(base)
        document[key] = value
        document["search_manifest_hash"] = stable_hash(document)
        path = tmp_path / f"forbidden_search_{index}.json"
        path.write_text(json.dumps(document))

        with pytest.raises(ValueError, match="invalid or incomplete"):
            final_module._load_search_manifest(path)


def test_interrupted_units_are_never_repeated_while_remaining_units_proceed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _nested_config(tmp_path)
    search_paths = run_nested_ood_search(config_path)
    frozen = json.loads(search_paths["frozen_policies"].read_text())
    search_manifest = json.loads(search_paths["search_manifest"].read_text())
    blocked = frozen["policies"][0]
    identity = EvaluationIdentity(
        dataset_hash=search_manifest["dataset_hash"],
        split_or_search_manifest_hash=search_manifest["search_manifest_hash"],
        frozen_policy_hash=blocked["frozen_policy_hash"],
        evaluation_unit=blocked["evaluation_unit"],
    )
    registry = EvaluationRegistry(final_module.EVALUATION_REGISTRY_DIRECTORY)
    registry.reserve(identity)
    prediction_batches = 0
    production_predict = final_module.predict_model

    def count_predict(model: object, X: object) -> object:
        nonlocal prediction_batches
        prediction_batches += 1
        return production_predict(model, X)

    monkeypatch.setattr(final_module, "predict_model", count_predict)

    with pytest.raises(RuntimeError, match="incomplete"):
        run_nested_ood_final(
            config_path,
            frozen_policies_path=search_paths["frozen_policies"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_nested_interrupted_first",
        )

    expected_completed = len(frozen["policies"]) - 1
    assert prediction_batches == expected_completed
    assert (
        tmp_path
        / "corrected_nested_interrupted_first"
        / "incomplete_manifest.json"
    ).exists()
    records = registry.records()
    assert sum(record.status == "reserved" for record in records) == 1
    assert sum(record.status == "metrics_complete" for record in records) == (
        expected_completed
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        run_nested_ood_final(
            config_path,
            frozen_policies_path=search_paths["frozen_policies"],
            search_manifest_path=search_paths["search_manifest"],
            output_directory=tmp_path / "corrected_nested_interrupted_second",
        )

    assert prediction_batches == expected_completed
