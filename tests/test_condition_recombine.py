"""Tests for reaction-aware condition recombination augmentation."""

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation.condition_recombine as recombine_module
from bh_augmentation.augmentation.condition_recombine import (
    build_reaction_smiles,
    generate_condition_recombined_candidates,
    parse_reaction_smiles,
    pseudo_label_candidates,
)


def test_parse_and_build_reaction_smiles() -> None:
    parts = parse_reaction_smiles("A.B.C.D>>P")

    assert parts == {
        "valid": True,
        "left_tokens": ["A", "B", "C", "D"],
        "substrate_tokens": ["A", "B"],
        "condition_tokens": ["C", "D"],
        "product": "P",
    }
    assert build_reaction_smiles(["A", "B"], ["C", "D"], "P") == "A.B.C.D>>P"


@pytest.mark.parametrize("value", ["", "A.B.C", "A>>P", "A.B>>"])
def test_parse_reaction_smiles_rejects_invalid_values(value: str) -> None:
    assert parse_reaction_smiles(value)["valid"] is False


def test_generate_candidates_changes_reactions_without_original_duplicates() -> None:
    train = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2", "r3"],
            "reaction_smiles": [
                "CCBr.N.O.Cl>>CCN",
                "CCC.N.Br.C>>CCCN",
                "CCCl.CN.P.O>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )

    candidates = generate_condition_recombined_candidates(
        train,
        synthetic_multiplier=1.0,
        max_synthetic_rows=3,
        random_state=7,
    )

    assert not candidates.empty
    assert candidates["reaction_id"].str.startswith("synthetic_").all()
    assert candidates["is_synthetic"].all()
    assert candidates["pseudo_label"].all()
    assert set(candidates["reaction_smiles"]).isdisjoint(train["reaction_smiles"])
    source_reactions = train.set_index("reaction_id")["reaction_smiles"]
    assert all(
        row.reaction_smiles != source_reactions[row.source_reaction_id]
        for row in candidates.itertuples()
    )


def test_pseudo_label_candidates_clips_teacher_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2"],
            "reaction_smiles": ["CCBr.N.O>>CCN", "CCCl.N.C>>CCN"],
            "yield": [20.0, 80.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "reaction_id": ["synthetic_1", "synthetic_2"],
            "reaction_smiles": ["CCBr.N.C>>CCN", "CCCl.N.O>>CCN"],
            "yield": [np.nan, np.nan],
            "pseudo_label": [True, True],
        }
    )

    def fake_features(
        df: pd.DataFrame, config: dict[str, object]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return (
            np.arange(len(df) * 2, dtype=float).reshape(len(df), 2),
            pd.to_numeric(df["yield"], errors="coerce").fillna(0).to_numpy(),
            ["f0", "f1"],
        )

    monkeypatch.setattr(recombine_module, "build_feature_matrix", fake_features)
    monkeypatch.setattr(recombine_module, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(recombine_module, "train_model", lambda model, X, y: model)
    monkeypatch.setattr(
        recombine_module,
        "predict_model",
        lambda model, X: np.array([-5.0, 120.0]),
    )

    labeled = pseudo_label_candidates(
        train,
        candidates,
        {"kind": "reaction_role_concat"},
        teacher_model_name="random_forest",
    )

    assert labeled["yield"].tolist() == [0.0, 100.0]
    assert labeled["pseudo_label"].all()
    assert set(labeled["pseudo_label_model"]) == {"random_forest"}
