"""Matched-run guarantees for corrected anonymous and role-aware transfer."""

from pathlib import Path

import pandas as pd
import pytest
from scripts.compare_corrected_condition_transfer import (
    assert_matched_real_only_baselines,
)

from bh_augmentation.run_condition_transfer import run_condition_transfer
from bh_augmentation.run_role_aware_condition_transfer import (
    run_role_aware_condition_transfer,
)
from corrected_test_utils import write_corrected_config


@pytest.fixture(scope="module")
def corrected_transfer_outputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Path], dict[str, Path]]:
    pytest.importorskip("xgboost")
    root = tmp_path_factory.mktemp("corrected_matching")
    anonymous_config = write_corrected_config(
        root, kind="anonymous", output_name="corrected_anonymous_test"
    )
    anonymous = run_condition_transfer(anonymous_config)
    role_config = write_corrected_config(
        root, kind="role_aware", output_name="corrected_role_aware_test"
    )
    role = run_role_aware_condition_transfer(role_config)
    return anonymous, role


def test_tiny_corrected_anonymous_config_completes(
    corrected_transfer_outputs: tuple[dict[str, Path], dict[str, Path]],
) -> None:
    anonymous, _ = corrected_transfer_outputs
    assert anonymous["policy_metrics"].is_file()
    selected = pd.read_csv(anonymous["selected_policy_metrics"])
    assert set(selected["split"]) == {"valid", "test"}
    assert (selected["n_synthetic_train"] > 0).all()


def test_tiny_corrected_role_aware_config_completes(
    corrected_transfer_outputs: tuple[dict[str, Path], dict[str, Path]],
) -> None:
    _, role = corrected_transfer_outputs
    assert role["policy_metrics"].is_file()
    selected = pd.read_csv(role["selected_policy_metrics"])
    audit = pd.read_csv(role["audit"])
    assert set(selected["split"]) == {"valid", "test"}
    assert (selected["n_synthetic_train"] > 0).all()
    accepted = audit.loc[audit["n_synthetic_train"] > 0]
    assert accepted["changed_any_transferred_role_fraction"].eq(1.0).all()


def test_corrected_baselines_and_hashes_match_exactly(
    corrected_transfer_outputs: tuple[dict[str, Path], dict[str, Path]],
) -> None:
    anonymous, role = corrected_transfer_outputs
    anonymous_metrics = pd.read_csv(anonymous["policy_metrics"])
    role_metrics = pd.read_csv(role["policy_metrics"])

    matched = assert_matched_real_only_baselines(anonymous_metrics, role_metrics)

    assert not matched.empty
    assert matched["anonymous_split_hash"].equals(matched["role_aware_split_hash"])
    assert matched["anonymous_feature_metadata_hash"].equals(
        matched["role_aware_feature_metadata_hash"]
    )
    assert matched["anonymous_dataset_hash"].equals(matched["role_aware_dataset_hash"])


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("value", "metric mismatch"),
        ("split_hash", "split_hash mismatch"),
        ("feature_metadata_hash", "feature metadata mismatch"),
        ("dataset_hash", "dataset_hash mismatch"),
    ],
)
def test_comparison_fails_on_baseline_or_provenance_mismatch(
    column: str,
    message: str,
) -> None:
    anonymous = _baseline_frame()
    role = _baseline_frame()
    role.loc[0, column] = 9.0 if column == "value" else "different"

    with pytest.raises(ValueError, match=message):
        assert_matched_real_only_baselines(anonymous, role)


def _baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": 0,
                "train_fraction": 0.1,
                "model": "xgboost",
                "split": "test",
                "metric": "rmse",
                "representation": "bh_role_separated_real_only",
                "value": 1.0,
                "feature_metadata_hash": "feature",
                "split_hash": "split",
                "dataset_hash": "dataset",
            }
        ]
    )
