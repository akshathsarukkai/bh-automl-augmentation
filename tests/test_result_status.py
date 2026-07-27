"""Tests for invalid historical-result isolation."""

from pathlib import Path

import pandas as pd
import pytest

from bh_augmentation.results.status import (
    InvalidResultError,
    assert_result_directory_allowed,
    assert_result_table_allowed,
    read_result_csv,
)


def test_old_role_aware_v2_directory_is_rejected() -> None:
    with pytest.raises(InvalidResultError, match=r"RESULT_STATUS\.md"):
        assert_result_directory_allowed(
            Path("results/role_aware_condition_transfer_v2_xgboost")
        )


def test_role_aware_rows_are_rejected() -> None:
    table = pd.DataFrame({"representation": ["role_aware_condition_transfer"]})
    with pytest.raises(InvalidResultError, match="invalid role-aware result rows"):
        assert_result_table_allowed(table)


def test_historical_override_is_explicit() -> None:
    assert_result_directory_allowed(
        "results/role_aware_condition_transfer_v2_xgboost",
        allow_invalid=True,
    )


def test_result_csv_with_invalid_rows_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {"representation": ["role_aware_condition_transfer"], "value": [1.0]}
    ).to_csv(path, index=False)

    with pytest.raises(InvalidResultError, match=r"RESULT_STATUS\.md"):
        read_result_csv(path)


@pytest.mark.parametrize(
    "path",
    [
        Path("results/stress/logo_product/fold_metrics.csv"),
        Path("results/stress/logo_reactant/fold_metrics.csv"),
        Path("results/baseline/stress_logo_product_metrics.csv"),
        Path("results/baseline/stress_logo_reactant_metrics.csv"),
    ],
)
def test_historical_logo_result_csv_is_rejected_by_default(path: Path) -> None:
    with pytest.raises(InvalidResultError, match=r"historical_logo_.*RESULT_STATUS\.md"):
        read_result_csv(path)


@pytest.mark.parametrize(
    "path",
    [
        Path("results/stress/logo_product/fold_metrics.csv"),
        Path("results/stress/logo_reactant/fold_metrics.csv"),
        Path("results/baseline/stress_logo_product_metrics.csv"),
        Path("results/baseline/stress_logo_reactant_metrics.csv"),
    ],
)
def test_historical_logo_result_csv_allows_explicit_inspection(path: Path) -> None:
    table = read_result_csv(path, allow_invalid=True)

    assert not table.empty


@pytest.mark.parametrize(
    "path",
    [
        Path("results/corrected_logo_product_phase7/fold_metrics.csv"),
        Path("results/corrected_logo_reactant_phase7/fold_metrics.csv"),
        Path("results/stress/logo_productivity/fold_metrics.csv"),
    ],
)
def test_fresh_or_nonmatching_logo_paths_remain_allowed(path: Path) -> None:
    assert_result_directory_allowed(path)
