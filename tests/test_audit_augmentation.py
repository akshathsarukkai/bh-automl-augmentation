"""Tests for augmentation data and feature auditing."""

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.audit_augmentation as audit_module
from bh_augmentation.audit_augmentation import audit_training_data


def _fake_feature_builder(
    df: pd.DataFrame,
    feature_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vectors = {
        "CCO.N>>CCN": [1.0, 0.0],
        "CCC.N>>CCCN": [0.0, 1.0],
        "CCCl.N>>CCN": [1.0, 1.0],
    }
    X = np.array([vectors[value] for value in df["reaction_smiles"]], dtype=np.float32)
    y = df["yield"].to_numpy(dtype=np.float32)
    return X, y, ["feature_0", "feature_1"]


def _frames(changed_reaction: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = pd.DataFrame(
        {
            "reaction_id": ["r1", "r2"],
            "reaction_smiles": ["CCO.N>>CCN", "CCC.N>>CCCN"],
            "source_note": ["measured", "measured"],
            "yield": [50.0, 70.0],
        }
    )
    synthetic = original.iloc[[0]].copy()
    synthetic["reaction_id"] = "r1_aug_1"
    synthetic["source_reaction_id"] = "r1"
    synthetic["is_augmented"] = True
    synthetic["pseudo_label"] = True
    synthetic["nearest_train_similarity"] = 0.6
    synthetic["source_note"] = "synthetic"
    if changed_reaction:
        synthetic["reaction_smiles"] = "CCCl.N>>CCN"

    originals = original.copy()
    originals["source_reaction_id"] = originals["reaction_id"]
    originals["is_augmented"] = False
    augmented = pd.concat([originals, synthetic], ignore_index=True)
    valid = pd.DataFrame({"reaction_id": ["v1"]})
    test = pd.DataFrame({"reaction_id": ["t1"]})
    return original, augmented, valid, test


def test_audit_detects_metadata_only_duplicated_feature_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata-only changes should fail when model features remain duplicated."""
    monkeypatch.setattr(audit_module, "build_feature_matrix", _fake_feature_builder)
    original, augmented, valid, test = _frames(changed_reaction=False)

    audit = audit_training_data(
        original,
        augmented,
        valid,
        test,
        {"kind": "reaction_role_concat_delta"},
    )

    assert audit["status"] == "FAIL"
    assert audit["synthetic_feature_rows_duplicate_original"] == 1
    assert audit["synthetic_feature_rows_new"] == 0
    assert audit["all_synthetic_features_identical_to_source_rows"] is True
    assert "source_note" in audit["columns_changed_by_augmentation"]
    assert any("features are duplicated" in warning for warning in audit["warnings"])


def test_audit_detects_changed_feature_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed reaction representation should produce a new model input."""
    monkeypatch.setattr(audit_module, "build_feature_matrix", _fake_feature_builder)
    original, augmented, valid, test = _frames(changed_reaction=True)

    audit = audit_training_data(
        original,
        augmented,
        valid,
        test,
        {"kind": "reaction_role_concat_delta"},
    )

    assert audit["status"] == "PASS"
    assert audit["synthetic_feature_rows_new"] == 1
    assert audit["percent_synthetic_features_new"] == 100.0
    assert audit["percent_synthetic_changed_reaction_smiles"] == 100.0
    assert audit["synthetic_labels_pseudo_labeled"] == 1
    assert audit["nearest_train_similarity"]["median"] == 0.6


def test_audit_verifies_validation_and_test_are_not_augmented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation IDs should remain absent from augmented training data."""
    monkeypatch.setattr(audit_module, "build_feature_matrix", _fake_feature_builder)
    original, augmented, valid, test = _frames(changed_reaction=True)

    audit = audit_training_data(
        original,
        augmented,
        valid,
        test,
        {"kind": "reaction_role_concat_delta"},
    )

    assert audit["validation_test_rows_modified"] is False
    assert audit["eval_ids_in_augmented_training"] == []
