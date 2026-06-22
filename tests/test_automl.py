"""Tests for compact optional Optuna AutoML search."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bh_augmentation.automl.search import run_automl_search
from bh_augmentation.data.split_data import random_split
from bh_augmentation.run_automl import run_automl


class FakeTrial:
    """Small deterministic Optuna trial stand-in."""

    def __init__(self, number: int) -> None:
        self.number = number

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        if name == "model_type":
            return choices[self.number % len(choices)]
        if name == "augmentation_type":
            return choices[self.number % len(choices)]
        return choices[0]

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        return low

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return low


class FakeStudy:
    """Small deterministic Optuna study stand-in."""

    def __init__(self) -> None:
        self.trials: list[FakeTrial] = []

    def optimize(self, objective: Any, n_trials: int) -> None:
        for number in range(n_trials):
            trial = FakeTrial(number)
            self.trials.append(trial)
            objective(trial)


def install_fake_optuna(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal fake Optuna module for tests."""
    fake_optuna = types.ModuleType("optuna")

    def create_study(**kwargs: Any) -> FakeStudy:
        return FakeStudy()

    fake_optuna.create_study = create_study
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)


def _fixture_df(n_rows: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(n_rows)],
            "reaction_smiles": [
                f"CC{'C' * (index % 3)}Br.N.O.C>>CC{'C' * (index % 3)}N"
                for index in range(n_rows)
            ],
            "product_key": [f"product_{index % 3}" for index in range(n_rows)],
            "reactant_key": [f"reactant_{index % 4}" for index in range(n_rows)],
            "yield": [float((index * 9) % 100) for index in range(n_rows)],
        }
    )


def test_run_automl_search_saves_outputs_and_keeps_test_unaugmented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AutoML should save trials, best config, and final test metrics."""
    install_fake_optuna(monkeypatch)
    df = _fixture_df()
    splits = random_split(df, train_size=0.6, valid_size=0.2, test_size=0.2, seed=7)
    output_config = {
        "trial_table_path": tmp_path / "trials.csv",
        "best_config_path": tmp_path / "best_config.json",
        "final_test_metrics_path": tmp_path / "final_test_metrics.csv",
    }

    paths = run_automl_search(
        splits=splits,
        base_feature_config={"kind": "reaction_role_concat_delta", "n_bits": 8},
        automl_config={
            "n_trials": 3,
            "primary_metric": "validation_top_k_hit_rate",
            "top_k": 2,
            "high_yield_threshold": 70,
            "candidate_models": ["ridge"],
            "feature_sets": ["reaction_role_concat_delta"],
            "augmentation_types": ["none"],
            "augmentation_ratios": [1, 2],
        },
        output_config=output_config,
        seed=7,
    )

    assert paths["trials"] == output_config["trial_table_path"]
    assert paths["best_config"] == output_config["best_config_path"]
    assert paths["final_test_metrics"] == output_config["final_test_metrics_path"]
    trials = pd.read_csv(output_config["trial_table_path"])
    assert len(trials) == 3
    assert set(trials["augmentation_type"]) == {"none"}

    best_config = json.loads(output_config["best_config_path"].read_text(encoding="utf-8"))
    assert "model_type" in best_config
    assert "validation_score" in best_config

    final_metrics = pd.read_csv(output_config["final_test_metrics_path"])
    assert set(final_metrics["split"]) == {"test"}
    assert (final_metrics["eval_augmented_rows"] == 0).all()


def test_run_automl_cli_works_with_fake_optuna(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run_automl CLI helper should work on a tiny local CSV."""
    install_fake_optuna(monkeypatch)
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "automl.yaml"
    trial_path = tmp_path / "trials.csv"
    best_path = tmp_path / "best_config.json"
    metrics_path = tmp_path / "final_test_metrics.csv"
    _fixture_df().to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 11
dataset:
  name: synthetic_bh
  path: {data_path}
splits:
  method: random
  train_size: 0.6
  valid_size: 0.2
  test_size: 0.2
features:
  kind: reaction_role_concat_delta
  n_bits: 8
automl:
  n_trials: 2
  primary_metric: validation_top_k_hit_rate
  top_k: 2
  high_yield_threshold: 70
  candidate_models:
    - ridge
  feature_sets:
    - reaction_role_concat_delta
  augmentation_types:
    - none
  augmentation_ratios:
    - 1
output:
  trial_table_path: {trial_path}
  best_config_path: {best_path}
  final_test_metrics_path: {metrics_path}
""",
        encoding="utf-8",
    )

    outputs = run_automl(config_path)

    assert outputs["trials"].exists()
    assert outputs["best_config"].exists()
    assert outputs["final_test_metrics"].exists()


def test_run_automl_search_requires_optuna(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Optuna should raise installation instructions."""
    monkeypatch.setitem(sys.modules, "optuna", None)
    df = _fixture_df()
    splits = random_split(df, train_size=0.6, valid_size=0.2, test_size=0.2, seed=7)

    with pytest.raises(ImportError, match="python -m pip install optuna"):
        run_automl_search(
            splits=splits,
            base_feature_config={"kind": "reaction_role_concat_delta", "n_bits": 8},
            automl_config={"n_trials": 1},
            output_config={},
            seed=7,
        )
