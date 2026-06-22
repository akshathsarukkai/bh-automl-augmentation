"""Tests for simulated reaction recommendation evaluation."""

from pathlib import Path

import pandas as pd

from bh_augmentation.evaluation.recommendation import simulate_single_round_recommendation
from bh_augmentation.run_recommendation import run_recommendation


def _recommendation_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [f"rxn_{index:03d}" for index in range(10)],
            "reaction_smiles": [
                f"CC{'C' * (index % 5)}Br.N.O.C>>CC{'C' * (index % 5)}N"
                for index in range(10)
            ],
            "product_key": [f"product_{index % 5}" for index in range(10)],
            "reactant_key": [f"reactant_{index % 4}" for index in range(10)],
            "yield": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        }
    )


def test_simulate_single_round_recommendation_is_deterministic() -> None:
    """A fixed seed should produce identical recommendation outputs."""
    df = _recommendation_fixture()
    kwargs = {
        "dataframe": df,
        "model_config": "ridge",
        "feature_config": {"kind": "reaction_morgan_sum", "n_bits": 8},
        "seed_size": 3,
        "k": 2,
        "high_yield_threshold": 70.0,
        "seed": 11,
    }

    first = simulate_single_round_recommendation(**kwargs)
    second = simulate_single_round_recommendation(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["strategy"]) == {"random", "model"}
    assert first["selected_reaction_ids"].map(len).tolist() == [2, 2]
    assert first["predicted_yields"].map(len).tolist() == [2, 2]
    assert first["true_yields"].map(len).tolist() == [2, 2]
    assert {"top_k_hit_rate", "regret", "experiments_to_first_hit"}.issubset(first.columns)


def test_simulate_single_round_recommendation_supports_candidate_pool() -> None:
    """Candidate pool IDs should constrain eligible recommendations."""
    df = _recommendation_fixture()

    results = simulate_single_round_recommendation(
        dataframe=df,
        model_config="ridge",
        feature_config={"kind": "reaction_morgan_sum", "n_bits": 8},
        seed_size=2,
        candidate_pool=["rxn_007", "rxn_008", "rxn_009"],
        k=2,
        high_yield_threshold=70.0,
        seed=2,
    )

    allowed = {"rxn_007", "rxn_008", "rxn_009"}
    for selected_ids in results["selected_reaction_ids"]:
        assert set(selected_ids).issubset(allowed)


def test_run_recommendation_saves_topk_metrics(tmp_path: Path) -> None:
    """The CLI runner should save recommendation metrics on fixture data."""
    data_path = tmp_path / "clean_bh.csv"
    config_path = tmp_path / "recommendation.yaml"
    metrics_path = tmp_path / "results" / "topk_metrics.csv"
    _recommendation_fixture().to_csv(data_path, index=False)
    config_path.write_text(
        f"""
seed: 5
dataset:
  name: synthetic_bh
  path: {data_path}
features:
  kind: reaction_morgan_sum
  n_bits: 8
models:
  - ridge
recommendation:
  seed_size: 3
  k: 2
  high_yield_threshold: 70
output:
  recommendation_metrics_path: {metrics_path}
""",
        encoding="utf-8",
    )

    output_path = run_recommendation(config_path)

    assert output_path == metrics_path
    metrics = pd.read_csv(metrics_path)
    assert set(metrics["strategy"]) == {"random", "model"}
    assert metrics_path.exists()
    assert set(["selected_reaction_ids", "predicted_yields", "true_yields"]).issubset(
        metrics.columns
    )
