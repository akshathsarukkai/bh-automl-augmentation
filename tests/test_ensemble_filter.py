"""Tests for teacher-ensemble uncertainty-filtered augmentation."""

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation.ensemble_filter as ensemble_module
from bh_augmentation.augmentation.condition_recombine import (
    generate_condition_recombined_candidates,
)
from bh_augmentation.augmentation.ensemble_filter import (
    add_teacher_ensemble_predictions,
    assign_ensemble_pseudo_labels,
    condition_recombine_ensemble_filter,
    filter_ensemble_candidates,
)
from bh_augmentation.augmentation.synthetic_identity import canonicalize_synthetic_roles
from bh_augmentation.data.reaction_roles import (
    ensure_reaction_role_columns,
    reaction_roles_from_row,
)


def _train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": ["r1", "r2", "r3"],
            "source_row_id": ["row-1", "row-2", "row-3"],
            "reaction_smiles": [
                "CCBr.N.[Pd].P(C)(C)C.N(C)(C)C.CCO>>CCN",
                "CCC.N.[Pd].P(CC)(CC)CC.N1CCCCC1.CCCO>>CCCN",
                "CCCl.CN.[Pd].P(C)(C)C.N1CCCCC1.CCOC>>CCNC",
            ],
            "yield": [30.0, 50.0, 70.0],
        }
    )


def test_recombined_candidates_preserve_source_roles_and_use_donor_conditions() -> None:
    train = _train_data()
    candidates = generate_condition_recombined_candidates(
        train, synthetic_multiplier=2.0, max_synthetic_rows=6, random_state=4
    )
    by_id = ensure_reaction_role_columns(train).set_index("reaction_id")
    real_reactions = set(train["reaction_smiles"])

    assert candidates["reaction_smiles"].is_unique
    assert set(candidates["reaction_smiles"]).isdisjoint(real_reactions)
    for row in candidates.itertuples():
        source = canonicalize_synthetic_roles(
            reaction_roles_from_row(by_id.loc[row.source_reaction_id])
        ).roles
        donor = canonicalize_synthetic_roles(
            reaction_roles_from_row(by_id.loc[row.donor_reaction_id])
        ).roles
        synthetic = reaction_roles_from_row(row._asdict())
        assert source is not None
        assert donor is not None
        assert synthetic.reactant_1 == source.reactant_1
        assert synthetic.reactant_2 == source.reactant_2
        assert synthetic.product == source.product
        assert synthetic.catalyst == donor.catalyst
        assert synthetic.ligand == donor.ligand
        assert synthetic.base == donor.base
        assert synthetic.solvent_or_additive == donor.solvent_or_additive


def test_teacher_ensemble_computes_statistics_and_clipped_mean_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _train_data().iloc[:2].copy()
    candidates = pd.DataFrame(
        {
            "reaction_smiles": ["CCBr.N.Br.C>>CCN", "CCC.N.O.Cl>>CCCN"],
            "yield": [np.nan, np.nan],
        }
    )

    def fake_features(
        df: pd.DataFrame, config: dict[str, object]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return (
            np.ones((len(df), 2)),
            pd.to_numeric(df["yield"], errors="coerce").fillna(0).to_numpy(),
            ["f0", "f1"],
        )

    monkeypatch.setattr(ensemble_module, "build_feature_matrix", fake_features)
    monkeypatch.setattr(
        ensemble_module,
        "get_model",
        lambda name, seed: {"name": name, "seed": seed},
    )
    monkeypatch.setattr(ensemble_module, "train_model", lambda model, X, y: model)
    monkeypatch.setattr(
        ensemble_module,
        "predict_model",
        lambda model, X: np.full(len(X), {-1: -20.0, 0: 40.0, 1: 120.0}[model["seed"]]),
    )

    scored = add_teacher_ensemble_predictions(
        train,
        candidates,
        {"kind": "reaction_morgan_sum"},
        [
            {"model": "ridge", "random_state": -1},
            {"model": "random_forest", "random_state": 0},
            {"model": "random_forest", "random_state": 1},
        ],
    )
    labeled = assign_ensemble_pseudo_labels(
        scored,
        {"mode": "ensemble_mean", "clip_min": 0.0, "clip_max": 40.0},
    )

    assert np.allclose(scored["teacher_mean_prediction"], 140.0 / 3.0)
    assert np.allclose(scored["teacher_prediction_range"], 140.0)
    assert (scored["teacher_std_prediction"] > 0).all()
    assert labeled["yield"].tolist() == [40.0, 40.0]


def test_uncertainty_and_similarity_filters_reject_independently() -> None:
    candidates = pd.DataFrame(
        {
            "reaction_id": ["accepted", "low_sim", "high_std", "high_range"],
            "nearest_train_similarity": [0.9, 0.7, 0.9, 0.9],
            "teacher_std_prediction": [5.0, 5.0, 13.0, 5.0],
            "teacher_prediction_range": [20.0, 20.0, 20.0, 40.0],
        }
    )
    config = {
        "min_neighbor_similarity": 0.8,
        "uncertainty_filter": {
            "enabled": True,
            "max_teacher_std": 12.0,
            "max_prediction_range": 35.0,
        },
    }

    accepted = filter_ensemble_candidates(candidates, config)

    assert accepted["reaction_id"].tolist() == ["accepted"]
    assert accepted["accepted_by_similarity_filter"].all()
    assert accepted["accepted_by_uncertainty_filter"].all()


def test_ensemble_filter_handles_zero_accepted_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ensemble_module,
        "add_teacher_ensemble_predictions",
        lambda train, candidates, feature_config, teachers, cache=None: candidates.assign(
            teacher_mean_prediction=50.0,
            teacher_std_prediction=99.0,
            teacher_min_prediction=0.0,
            teacher_max_prediction=100.0,
            teacher_prediction_range=100.0,
        ),
    )
    augmented, metadata = condition_recombine_ensemble_filter(
        _train_data(),
        {
            "kind": "bh_role_separated",
            "n_bits": 8,
            "fingerprint_backend": "rdkit",
        },
        {
            "synthetic_multiplier": 1.0,
            "min_neighbor_similarity": 0.0,
            "teachers": [{"model": "ridge", "random_state": 0}],
            "uncertainty_filter": {
                "enabled": True,
                "max_teacher_std": 1.0,
                "max_prediction_range": 1.0,
            },
        },
        random_state=3,
    )

    assert len(augmented) == len(_train_data())
    assert metadata["n_candidates_generated"] > 0
    assert metadata["n_candidates_accepted"] == 0
    assert metadata["acceptance_rate"] == 0.0
    assert metadata["rejection_reason_counts"][
        "teacher_uncertainty_above_threshold"
    ] > 0
    audit = augmented.attrs["synthetic_candidate_audit"]
    assert set(audit.loc[audit["accepted"], "rejection_reason"].dropna()) == set()
    assert not audit["accepted"].any()
