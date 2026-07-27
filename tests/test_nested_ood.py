"""Tests for deterministic nested group-aware OOD assignments."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pandas as pd
import pytest

from bh_augmentation.evaluation.nested_ood import (
    NestedGroupOODContract,
    aggregate_inner_ood_metrics,
    build_nested_group_ood_contract,
    build_nested_ood_all_outer_plan,
    validate_nested_group_ood_assignments,
    validate_nested_ood_all_outer_plan,
)
from bh_augmentation.utils.corrected_runs import stable_hash


def _canonical_frame() -> pd.DataFrame:
    group_values = ["group-a", "group-b", "group-b", "group-c", "group-c", "group-c", "group-d"]
    return pd.DataFrame(
        {
            "source_row_id": [
                f"row-{index:02d}" for index in range(len(group_values))
            ],
            "canonical_reaction_key": [
                f"reaction-{index:02d}" for index in range(len(group_values))
            ],
            "ood_group": group_values,
            "yield": [float(index * 10) for index in range(len(group_values))],
        }
    )


def test_outer_group_is_absent_from_complete_group_disjoint_inner_logo() -> None:
    frame = _canonical_frame()
    contract = build_nested_group_ood_contract(
        frame,
        group_column="ood_group",
        outer_group="group-d",
    )

    assert contract.expected_inner_groups == (
        "group-a",
        "group-b",
        "group-c",
    )
    assert contract.inner_fold_count == len(contract.expected_inner_groups)
    assert set(contract.inner_assignments["group_value"]) == set(
        contract.expected_inner_groups
    )
    assert not set(contract.outer_test_source_ids) & set(
        contract.inner_assignments["source_row_id"]
    )
    observed_valid_groups = []
    for fold in contract.inner_folds:
        observed_valid_groups.append(fold.validation_group)
        assert fold.source_id_overlap_count == 0
        assert fold.group_overlap_count == 0
        assert fold.canonical_reaction_key_overlap_count == 0
        assert set(fold.train_source_ids).isdisjoint(
            fold.validation_source_ids
        )
    assert observed_valid_groups == list(contract.expected_inner_groups)
    assert contract.inner_overlap_audit["all_overlaps_zero"].all()


def test_assignments_and_hashes_are_row_order_invariant() -> None:
    frame = _canonical_frame()
    first = build_nested_group_ood_contract(
        frame,
        group_column="ood_group",
        outer_group="group-c",
    )
    shuffled_frame = frame.sample(frac=1.0, random_state=13).reset_index(
        drop=True
    )
    second = validate_nested_group_ood_assignments(
        shuffled_frame,
        first.outer_assignments.sample(
            frac=1.0,
            random_state=7,
        ).reset_index(drop=True),
        first.inner_assignments.sample(
            frac=1.0,
            random_state=8,
        ).reset_index(drop=True),
        group_column="ood_group",
        outer_group="group-c",
    )

    assert first.outer_assignment_hash == second.outer_assignment_hash
    assert (
        first.per_inner_fold_assignment_hashes
        == second.per_inner_fold_assignment_hashes
    )
    assert first.aggregate_assignment_hash == second.aggregate_assignment_hash
    pd.testing.assert_frame_equal(
        first.outer_assignments,
        second.outer_assignments,
    )
    pd.testing.assert_frame_equal(
        first.inner_assignments,
        second.inner_assignments,
    )


def test_outer_group_contamination_of_inner_search_hard_fails() -> None:
    frame = _canonical_frame()
    contract = build_nested_group_ood_contract(
        frame,
        group_column="ood_group",
        outer_group="group-d",
    )
    inner = contract.inner_assignments
    outer_row = frame.loc[frame["ood_group"].eq("group-d")].iloc[0]
    contaminated_row = inner.iloc[0].copy()
    contaminated_row["source_row_id"] = outer_row["source_row_id"]
    contaminated_row["canonical_reaction_key"] = outer_row[
        "canonical_reaction_key"
    ]
    contaminated_row["group_value"] = outer_row["ood_group"]
    inner = pd.concat(
        [inner, contaminated_row.to_frame().T],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="entered inner OOD search"):
        validate_nested_group_ood_assignments(
            frame,
            contract.outer_assignments,
            inner,
            group_column="ood_group",
            outer_group="group-d",
        )


def test_duplicate_or_omitted_inner_validation_group_hard_fails() -> None:
    frame = _canonical_frame()
    contract = build_nested_group_ood_contract(
        frame,
        group_column="ood_group",
        outer_group="group-d",
    )
    inner = contract.inner_assignments
    omitted = inner.loc[~inner["inner_fold_index"].eq(2)]
    with pytest.raises(ValueError, match="Expected inner group count"):
        validate_nested_group_ood_assignments(
            frame,
            contract.outer_assignments,
            omitted,
            group_column="ood_group",
            outer_group="group-d",
        )

    duplicated = inner.copy()
    duplicated.loc[
        duplicated["inner_fold_index"].eq(1),
        "inner_validation_group",
    ] = "group-a"
    with pytest.raises(ValueError, match="exactly once"):
        validate_nested_group_ood_assignments(
            frame,
            contract.outer_assignments,
            duplicated,
            group_column="ood_group",
            outer_group="group-d",
        )


def test_outer_and_inner_assignments_must_hold_out_declared_groups_exactly() -> None:
    frame = _canonical_frame()
    contract = build_nested_group_ood_contract(
        frame,
        group_column="ood_group",
        outer_group="group-d",
    )
    outer = contract.outer_assignments
    outer.loc[outer["source_row_id"].eq("row-00"), "outer_split"] = "test"
    with pytest.raises(ValueError, match="do not hold out exactly outer_group"):
        validate_nested_group_ood_assignments(
            frame,
            outer,
            contract.inner_assignments,
            group_column="ood_group",
            outer_group="group-d",
        )

    inner = contract.inner_assignments
    row = inner["inner_fold_index"].eq(0) & inner["source_row_id"].eq(
        "row-01"
    )
    inner.loc[row, "inner_split"] = "valid"
    with pytest.raises(ValueError, match="do not hold out exactly"):
        validate_nested_group_ood_assignments(
            frame,
            contract.outer_assignments,
            inner,
            group_column="ood_group",
            outer_group="group-d",
        )


def test_canonical_reaction_keys_cannot_cross_ood_groups() -> None:
    frame = _canonical_frame()
    frame.loc[frame["source_row_id"].eq("row-01"), "canonical_reaction_key"] = (
        "reaction-00"
    )

    with pytest.raises(ValueError, match="maps to multiple"):
        build_nested_group_ood_contract(
            frame,
            group_column="ood_group",
            outer_group="group-d",
        )


def _inner_metrics(
    *,
    policy_id: str = "policy-a",
) -> tuple[object, pd.DataFrame]:
    contract = build_nested_group_ood_contract(
        _canonical_frame(),
        group_column="ood_group",
        outer_group="group-d",
    )
    policy_hash = stable_hash(policy_id)
    values = [0.0, 10.0, 20.0]
    rows = [
        {
            "policy_id": policy_id,
            "policy_hash": policy_hash,
            "metric": "rmse",
            "inner_fold_index": fold.fold_index,
            "inner_validation_group": fold.validation_group,
            "outer_group": contract.outer_group,
            "outer_assignment_hash": contract.outer_assignment_hash,
            "inner_assignment_hash": fold.assignment_hash,
            "aggregate_assignment_hash": (
                contract.aggregate_assignment_hash
            ),
            "n_validation_samples": fold.validation_size,
            "value": values[fold.fold_index],
        }
        for fold in contract.inner_folds
    ]
    return contract, pd.DataFrame(rows)


def _expected_keys(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[["policy_id", "policy_hash", "metric"]].drop_duplicates()


def test_group_and_sample_weighted_inner_aggregation_are_explicit() -> None:
    contract, metrics = _inner_metrics()

    group_weighted = aggregate_inner_ood_metrics(
        contract,
        metrics,
        expected_group_keys=_expected_keys(metrics),
        weighting="group_weighted",
    )
    sample_weighted = aggregate_inner_ood_metrics(
        contract,
        metrics,
        expected_group_keys=_expected_keys(metrics),
        weighting="sample_weighted",
    )

    assert group_weighted.iloc[0]["value"] == pytest.approx(10.0)
    assert sample_weighted.iloc[0]["value"] == pytest.approx(80.0 / 6.0)
    assert group_weighted.iloc[0]["n_inner_folds"] == 3
    assert sample_weighted.iloc[0]["n_inner_validation_samples"] == 6
    assert (
        sample_weighted.iloc[0]["aggregate_assignment_hash"]
        == contract.aggregate_assignment_hash
    )


def test_inner_aggregation_requires_every_fold_exactly_once() -> None:
    contract, metrics = _inner_metrics()
    incomplete = metrics.iloc[:-1]
    with pytest.raises(ValueError, match="exactly one row"):
        aggregate_inner_ood_metrics(
            contract,
            incomplete,
            expected_group_keys=_expected_keys(metrics),
        )

    duplicated = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row"):
        aggregate_inner_ood_metrics(
            contract,
            duplicated,
            expected_group_keys=_expected_keys(metrics),
        )


def test_inner_aggregation_validates_group_size_group_name_and_values() -> None:
    contract, metrics = _inner_metrics()
    wrong_size = metrics.copy()
    wrong_size.loc[0, "n_validation_samples"] = 999
    with pytest.raises(ValueError, match="sample count"):
        aggregate_inner_ood_metrics(
            contract,
            wrong_size,
            expected_group_keys=_expected_keys(metrics),
        )

    wrong_group = metrics.copy()
    wrong_group.loc[0, "inner_validation_group"] = "wrong-group"
    with pytest.raises(ValueError, match="validation group"):
        aggregate_inner_ood_metrics(
            contract,
            wrong_group,
            expected_group_keys=_expected_keys(metrics),
        )

    nonfinite = metrics.copy()
    nonfinite.loc[0, "value"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        aggregate_inner_ood_metrics(
            contract,
            nonfinite,
            expected_group_keys=_expected_keys(metrics),
        )


def test_multiple_policy_aggregates_are_deterministically_ordered() -> None:
    contract, metrics_a = _inner_metrics(policy_id="policy-z")
    _, metrics_b = _inner_metrics(policy_id="policy-a")
    combined = pd.concat([metrics_a, metrics_b], ignore_index=True)

    result = aggregate_inner_ood_metrics(
        contract,
        combined.sample(frac=1.0, random_state=12),
        expected_group_keys=_expected_keys(combined),
    )

    assert result["policy_id"].tolist() == ["policy-a", "policy-z"]


def test_metric_rows_are_bound_to_active_outer_inner_and_aggregate_hashes() -> None:
    contract, metrics = _inner_metrics()
    provenance_columns = [
        "outer_group",
        "outer_assignment_hash",
        "inner_assignment_hash",
        "aggregate_assignment_hash",
    ]
    assert not metrics[provenance_columns].isna().any().any()

    for column in provenance_columns:
        stale = metrics.copy()
        stale.loc[0, column] = f"stale-{column}"
        with pytest.raises(ValueError, match="assignment provenance"):
            aggregate_inner_ood_metrics(
                contract,
                stale,
                expected_group_keys=_expected_keys(metrics),
            )


def test_stale_contract_metrics_cannot_be_laundered_through_another_outer_fold() -> None:
    stale_contract, stale_metrics = _inner_metrics()
    active_contract = build_nested_group_ood_contract(
        _canonical_frame(),
        group_column="ood_group",
        outer_group="group-c",
    )
    assert stale_contract.outer_group != active_contract.outer_group

    with pytest.raises(ValueError, match="assignment provenance"):
        aggregate_inner_ood_metrics(
            active_contract,
            stale_metrics,
            expected_group_keys=_expected_keys(stale_metrics),
        )


def test_declared_search_grid_rejects_wholly_missing_or_unexpected_candidate() -> None:
    contract, metrics = _inner_metrics()
    expected = _expected_keys(metrics)
    absent = pd.DataFrame(
        [
            {
                "policy_id": "policy-absent",
                "policy_hash": stable_hash("policy-absent"),
                "metric": "rmse",
            }
        ]
    )
    with pytest.raises(ValueError, match="declared search grid"):
        aggregate_inner_ood_metrics(
            contract,
            metrics,
            expected_group_keys=pd.concat(
                [expected, absent],
                ignore_index=True,
            ),
        )

    unexpected = metrics.copy()
    unexpected.loc[:, "policy_id"] = "policy-unexpected"
    unexpected.loc[:, "policy_hash"] = stable_hash("policy-unexpected")
    with pytest.raises(ValueError, match="declared search grid"):
        aggregate_inner_ood_metrics(
            contract,
            pd.concat([metrics, unexpected], ignore_index=True),
            expected_group_keys=expected,
        )


def test_compact_all_outer_plan_covers_sorted_groups_from_streamed_contracts() -> None:
    frame = _canonical_frame()
    groups = sorted(frame["ood_group"].unique())
    yielded_groups: list[str] = []

    def contracts() -> Iterator[NestedGroupOODContract]:
        for outer_group in reversed(groups):
            yielded_groups.append(outer_group)
            yield build_nested_group_ood_contract(
                frame,
                group_column="ood_group",
                outer_group=outer_group,
            )

    plan = build_nested_ood_all_outer_plan(
        frame,
        contracts(),
        group_column="ood_group",
    )

    assert yielded_groups == list(reversed(groups))
    assert plan.expected_outer_groups == tuple(groups)
    assert [fold.outer_group for fold in plan.fold_definitions] == groups
    assert [fold.outer_fold_index for fold in plan.fold_definitions] == list(
        range(len(groups))
    )
    assert set(plan.per_outer_aggregate_assignment_hashes) == set(groups)
    assert plan.outer_fold_count == len(groups)
    assert not hasattr(plan, "outer_assignments")
    assert not hasattr(plan, "inner_assignments")
    validate_nested_ood_all_outer_plan(frame, plan)


def test_all_outer_plan_rejects_missing_duplicate_and_stale_definitions() -> None:
    frame = _canonical_frame()
    groups = sorted(frame["ood_group"].unique())
    contracts = [
        build_nested_group_ood_contract(
            frame,
            group_column="ood_group",
            outer_group=group,
        )
        for group in groups
    ]
    with pytest.raises(ValueError, match="every outer group exactly once"):
        build_nested_ood_all_outer_plan(
            frame,
            contracts[:-1],
            group_column="ood_group",
        )
    with pytest.raises(ValueError, match="Duplicate nested OOD outer group"):
        build_nested_ood_all_outer_plan(
            frame,
            [*contracts, contracts[0]],
            group_column="ood_group",
        )

    plan = build_nested_ood_all_outer_plan(
        frame,
        contracts,
        group_column="ood_group",
    )
    stale_fold = replace(
        plan.fold_definitions[0],
        aggregate_assignment_hash="stale-hash",
    )
    stale_plan = replace(
        plan,
        fold_definitions=(stale_fold, *plan.fold_definitions[1:]),
    )
    with pytest.raises(ValueError, match="plan hash mismatch"):
        validate_nested_ood_all_outer_plan(frame, stale_plan)


def test_requires_three_groups_and_present_outer_group() -> None:
    two_groups = _canonical_frame().loc[
        _canonical_frame()["ood_group"].isin(["group-a", "group-b"])
    ]
    with pytest.raises(ValueError, match="at least three"):
        build_nested_group_ood_contract(
            two_groups,
            group_column="ood_group",
            outer_group="group-a",
        )

    with pytest.raises(ValueError, match="is absent"):
        build_nested_group_ood_contract(
            _canonical_frame(),
            group_column="ood_group",
            outer_group="not-present",
        )


def test_assignment_frames_are_defensive_copies() -> None:
    contract = build_nested_group_ood_contract(
        _canonical_frame(),
        group_column="ood_group",
        outer_group="group-d",
    )
    outer = contract.outer_assignments
    inner = contract.inner_assignments
    outer.loc[:, "outer_split"] = "test"
    inner.loc[:, "inner_split"] = "valid"

    assert not contract.outer_assignments["outer_split"].eq("test").all()
    assert not contract.inner_assignments["inner_split"].eq("valid").all()
    assert contract.audit_record["all_inner_overlaps_zero"]
