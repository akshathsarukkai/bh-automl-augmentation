"""End-to-end smoke test for the MVP research pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.order_permutation import permute_reaction_components
from bh_augmentation.data.clean_data import clean_buchwald_hartwig
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.split_data import random_split
from bh_augmentation.evaluation.metrics import mae, rmse
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model
from bh_augmentation.reporting.make_report import save_metrics_csv

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_bh.csv"


def test_end_to_end_fixture_pipeline_guardrail(tmp_path: Path) -> None:
    """Run load-clean-split-featurize-train-augment-evaluate on fixture data."""
    raw = load_reaction_csv(FIXTURE_PATH)
    cleaned = clean_buchwald_hartwig(raw)
    dataset = _expand_cleaned_fixture(cleaned, repeats=4)

    splits = random_split(
        dataset,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
        seed=123,
    )
    feature_config = {
        "smiles_columns": [],
        "categorical_columns": ["solvent"],
    }

    baseline_metrics = _train_and_evaluate(
        train_df=splits["train"],
        valid_df=splits["valid"],
        test_df=splits["test"],
        feature_config=feature_config,
        strategy="baseline",
    )

    augmented_train = permute_reaction_components(
        splits["train"],
        component_columns=["aryl_halide_smiles", "amine_smiles"],
        n_permutations=1,
        seed=123,
        include_original=True,
    )
    assert augmented_train["is_augmented"].sum() > 0
    assert "is_augmented" not in splits["valid"].columns
    assert "is_augmented" not in splits["test"].columns

    augmented_metrics = _train_and_evaluate(
        train_df=augmented_train,
        valid_df=splits["valid"],
        test_df=splits["test"],
        feature_config=feature_config,
        strategy="order_permutation",
    )

    metrics_path = tmp_path / "metrics" / "end_to_end_metrics.csv"
    save_metrics_csv(baseline_metrics + augmented_metrics, metrics_path)

    metrics = pd.read_csv(metrics_path)
    assert metrics_path.exists()
    assert set(metrics["strategy"]) == {"baseline", "order_permutation"}
    assert set(metrics["split"]) == {"valid", "test"}
    assert set(metrics["metric"]) == {"rmse", "mae"}
    assert np.isfinite(metrics["value"]).all()


def _expand_cleaned_fixture(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Create enough deterministic fixture rows for train/valid/test splitting."""
    frames = []
    for repeat in range(repeats):
        copy = df.copy()
        copy["reaction_id"] = copy["reaction_id"].astype(str) + f"_copy_{repeat}"
        copy["yield"] = (copy["yield"] + repeat).clip(0, 100)
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


def _train_and_evaluate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_config: dict[str, list[str]],
    strategy: str,
) -> list[dict[str, object]]:
    """Train a Ridge model and return validation/test metrics."""
    combined = pd.concat(
        [
            train_df.assign(__split="train"),
            valid_df.assign(__split="valid"),
            test_df.assign(__split="test"),
        ],
        ignore_index=True,
    )
    X, y, _ = build_feature_matrix(combined, feature_config)
    train_mask = combined["__split"].to_numpy() == "train"
    model = train_model(get_model("ridge"), X[train_mask], y[train_mask])

    records: list[dict[str, object]] = []
    for split_name in ["valid", "test"]:
        split_mask = combined["__split"].to_numpy() == split_name
        y_true = y[split_mask]
        y_pred = predict_model(model, X[split_mask])
        records.extend(
            [
                {
                    "strategy": strategy,
                    "split": split_name,
                    "metric": "rmse",
                    "value": rmse(y_true, y_pred),
                },
                {
                    "strategy": strategy,
                    "split": split_name,
                    "metric": "mae",
                    "value": mae(y_true, y_pred),
                },
            ]
        )
    return records
