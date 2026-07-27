from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bh_augmentation.data.canonicalize_roles import CANONICALIZATION_VERSION
from bh_augmentation.evaluation.representation_splits import EvaluationSplitUnit
from bh_augmentation.representation_benchmark import (
    REPRESENTATION_IDS,
    _assert_paired_metrics,
    _assert_prediction_replay,
    _resolve_contract,
    _summarize,
)


def _config() -> dict[str, object]:
    return {
        "dataset": {"path": "canonical.csv"},
        "splits": {
            "canonical_directory": "splits",
            "random_seeds": [0],
            "random_fractions": [0.01],
            "logo_targets": ["product_key", "reactant_key"],
            "logo_artifact_directory": "logo",
            "chemical_ood_directory": "ood",
            "chemical_ood_families": ["maximum_similarity_bounded"],
        },
        "representations": list(REPRESENTATION_IDS),
        "features": {
            "n_bits": 64,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "model": {"name": "xgboost", "params": {"n_estimators": 5}},
        "metrics": ["rmse", "mae"],
        "base_seed": 0,
        "output": {"directory": "results/corrected_benchmark"},
    }


def test_benchmark_contract_requires_all_eight_methods_in_frozen_order() -> None:
    resolved = _resolve_contract(_config())
    assert resolved["chemical_ood_families"] == (
        "maximum_similarity_bounded",
    )
    broken = _config()
    broken["representations"] = list(REPRESENTATION_IDS[:-1])
    with pytest.raises(ValueError, match="exact eight"):
        _resolve_contract(broken)


def test_paired_gate_rejects_representation_specific_split_or_model_hash() -> None:
    rows = []
    for benchmark_id in REPRESENTATION_IDS:
        rows.append(
            {
                "evaluation_unit": "unit",
                "benchmark_id": benchmark_id,
                "metric": "rmse",
                "exact_split_hash": "split",
                "model_protocol_hash": "model",
                "model_seed": 7,
                "test_evaluation_count": 1,
                "test_used_for_selection": False,
            }
        )
    metrics = pd.DataFrame(rows)
    _assert_paired_metrics(metrics)
    metrics.loc[0, "exact_split_hash"] = "different"
    with pytest.raises(ValueError, match="Unpaired"):
        _assert_paired_metrics(metrics)


def test_paired_gate_rejects_test_selection_and_repeated_test_metrics() -> None:
    rows = []
    for benchmark_id in REPRESENTATION_IDS:
        rows.append(
            {
                "evaluation_unit": "unit",
                "benchmark_id": benchmark_id,
                "metric": "rmse",
                "exact_split_hash": "split",
                "model_protocol_hash": "model",
                "model_seed": 7,
                "test_evaluation_count": 1,
                "test_used_for_selection": False,
            }
        )
    metrics = pd.DataFrame(rows)
    contaminated = metrics.copy()
    contaminated.loc[0, "test_used_for_selection"] = True
    with pytest.raises(ValueError, match="Unpaired"):
        _assert_paired_metrics(contaminated)
    repeated = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Repeated"):
        _assert_paired_metrics(repeated)


def test_production_contract_uses_full_canonical_width_seeds_and_fractions() -> None:
    config = yaml.safe_load(
        Path(
            "configs/corrected_representation_benchmark_phase10.yaml"
        ).read_text()
    )
    assert config["features"]["n_bits"] == 2048
    assert config["splits"]["random_seeds"] == [0, 1, 2, 3, 4]
    assert config["splits"]["random_fractions"] == [0.01, 0.05, 0.10, 0.20, 1.0]
    assert tuple(config["representations"]) == REPRESENTATION_IDS


def test_random_summary_never_pools_training_fractions() -> None:
    metrics = pd.DataFrame(
        {
            "evidence_class": ["nested_random", "nested_random"],
            "split_target": ["reaction", "reaction"],
            "train_fraction": [0.01, 1.0],
            "benchmark_id": ["product_free", "product_free"],
            "metric": ["rmse", "rmse"],
            "value": [30.0, 10.0],
        }
    )
    summary = _summarize(metrics)
    assert len(summary) == 2
    assert set(summary["train_fraction"]) == {0.01, 1.0}


def test_prediction_replay_recomputes_metrics_from_canonical_outcome(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "canonical.csv"
    pd.DataFrame(
        {"source_row_id": ["test"], "yield": [10.0]}
    ).to_csv(dataset, index=False)
    unit = EvaluationSplitUnit(
        evaluation_unit="unit",
        split_family="chemical_ood",
        split_method="holdout",
        target="bounded",
        fold_index=0,
        heldout_group="group",
        seed=None,
        train_fraction=None,
        train_source_ids=("train",),
        validation_source_ids=(),
        test_source_ids=("test",),
        excluded_source_ids=(),
        dataset_hash="a" * 64,
        exact_split_hash="b" * 64,
        aggregate_assignment_hash="c" * 64,
        canonical_split_dependency_hash="d" * 64,
        upstream_split_hash=None,
        canonicalization_version=CANONICALIZATION_VERSION,
        split_schema_version="fixture-v1",
    )
    predictions = pd.DataFrame(
        [
            {
                "evaluation_unit": "unit",
                "benchmark_id": benchmark_id,
                "source_row_id": "test",
                "prediction": 8.0,
            }
            for benchmark_id in REPRESENTATION_IDS
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "evaluation_unit": "unit",
                "benchmark_id": benchmark_id,
                "metric": "rmse",
                "value": 2.0,
            }
            for benchmark_id in REPRESENTATION_IDS
        ]
    )
    _assert_prediction_replay(
        predictions,
        metrics,
        (unit,),
        str(dataset),
        ["rmse"],
        {"prediction_row_count": 8},
    )
    metrics.loc[0, "value"] = 999.0
    with pytest.raises(ValueError, match="metric prediction replay"):
        _assert_prediction_replay(
            predictions,
            metrics,
            (unit,),
            str(dataset),
            ["rmse"],
            {"prediction_row_count": 8},
        )
