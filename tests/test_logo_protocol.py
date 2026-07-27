"""Tests for strict canonical leave-one-group-out assignments."""

from __future__ import annotations

import pandas as pd
import pytest

from bh_augmentation.evaluation.logo_protocol import (
    LOGO_ASSIGNMENT_COLUMNS,
    LOGO_SPLIT_METHOD,
    build_canonical_logo_contract,
    resolve_logo_group_column,
    validate_canonical_logo_assignments,
)


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_row_id": "row-00",
                "canonical_reaction_key": "reaction-a",
                "canonical_product_key": "product-a",
                "canonical_substrate_key": "substrate-a",
                "yield": 10.0,
            },
            {
                "source_row_id": "row-01",
                "canonical_reaction_key": "reaction-a",
                "canonical_product_key": "product-a",
                "canonical_substrate_key": "substrate-a",
                "yield": 11.0,
            },
            {
                "source_row_id": "row-02",
                "canonical_reaction_key": "reaction-b",
                "canonical_product_key": "product-b",
                "canonical_substrate_key": "substrate-b",
                "yield": 20.0,
            },
            {
                "source_row_id": "row-03",
                "canonical_reaction_key": "reaction-c",
                "canonical_product_key": "product-b",
                "canonical_substrate_key": "substrate-c",
                "yield": 30.0,
            },
            {
                "source_row_id": "row-04",
                "canonical_reaction_key": "reaction-d",
                "canonical_product_key": "product-c",
                "canonical_substrate_key": "substrate-b",
                "yield": 40.0,
            },
            {
                "source_row_id": "row-05",
                "canonical_reaction_key": "reaction-e",
                "canonical_product_key": "product-c",
                "canonical_substrate_key": "substrate-c",
                "yield": 50.0,
            },
        ]
    )


@pytest.mark.parametrize(
    ("target", "group_column"),
    [
        ("product_key", "canonical_product_key"),
        ("reactant_key", "canonical_substrate_key"),
    ],
)
def test_builds_complete_deterministic_logo_contract_for_canonical_target(
    target: str,
    group_column: str,
) -> None:
    frame = _canonical_frame()

    contract = build_canonical_logo_contract(frame, target=target)

    assert resolve_logo_group_column(target) == group_column
    assert contract.group_column == group_column
    assert contract.split_method == LOGO_SPLIT_METHOD
    assert contract.expected_group_count == frame[group_column].nunique()
    assert contract.fold_count == contract.expected_group_count
    assert [fold.fold_group for fold in contract.folds] == sorted(
        frame[group_column].unique()
    )
    assert set(contract.per_fold_assignment_hashes) == set(
        frame[group_column]
    )
    assert contract.aggregate_assignment_hash
    assert contract.overlap_audit["all_overlaps_zero"].all()
    for fold in contract.folds:
        expected_test = set(
            frame.loc[
                frame[group_column].eq(fold.fold_group),
                "source_row_id",
            ]
        )
        assert set(fold.test_source_ids) == expected_test
        assert set(fold.train_source_ids) == set(frame["source_row_id"]) - expected_test
        assert fold.group_size == len(expected_test)


def test_assignment_and_hashes_are_invariant_to_all_input_row_order() -> None:
    frame = _canonical_frame()
    shuffled_frame = frame.sample(frac=1.0, random_state=8).reset_index(
        drop=True
    )
    first = build_canonical_logo_contract(frame, target="product_key")
    shuffled_assignments = first.assignments.sample(
        frac=1.0,
        random_state=19,
    ).reset_index(drop=True)

    second = validate_canonical_logo_assignments(
        shuffled_frame,
        shuffled_assignments,
        target="product_key",
    )

    assert first.aggregate_assignment_hash == second.aggregate_assignment_hash
    assert (
        first.per_fold_assignment_hashes
        == second.per_fold_assignment_hashes
    )
    pd.testing.assert_frame_equal(first.assignments, second.assignments)


def test_contract_exposes_defensive_assignment_and_audit_copies() -> None:
    contract = build_canonical_logo_contract(
        _canonical_frame(),
        target="product_key",
    )
    assignments = contract.assignments
    audit = contract.overlap_audit
    assignments.loc[:, "outer_split"] = "test"
    audit.loc[:, "all_overlaps_zero"] = False

    assert not contract.assignments["outer_split"].eq("test").all()
    assert contract.overlap_audit["all_overlaps_zero"].all()
    assert contract.audit_record["observed_fold_count"] == contract.fold_count


def test_random_split_mislabeled_as_logo_is_rejected() -> None:
    contract = build_canonical_logo_contract(
        _canonical_frame(),
        target="product_key",
    )
    assignments = contract.assignments
    fold_zero = assignments["fold_index"].eq(0)
    test_index = assignments.index[
        fold_zero & assignments["outer_split"].eq("test")
    ][0]
    train_index = assignments.index[
        fold_zero & assignments["outer_split"].eq("train")
    ][0]
    assignments.loc[test_index, "outer_split"] = "train"
    assignments.loc[train_index, "outer_split"] = "test"

    with pytest.raises(ValueError, match="do not hold out exactly"):
        validate_canonical_logo_assignments(
            _canonical_frame(),
            assignments,
            target="product_key",
        )


@pytest.mark.parametrize("mode", ["missing_column", "empty_value"])
def test_missing_fold_group_is_rejected(mode: str) -> None:
    assignments = build_canonical_logo_contract(
        _canonical_frame(),
        target="product_key",
    ).assignments
    if mode == "missing_column":
        assignments = assignments.drop(columns="fold_group")
        message = "schema mismatch"
    else:
        assignments.loc[assignments["fold_index"].eq(0), "fold_group"] = ""
        message = "empty values"

    with pytest.raises(ValueError, match=message):
        validate_canonical_logo_assignments(
            _canonical_frame(),
            assignments,
            target="product_key",
        )


def test_duplicate_fold_group_and_omitted_fold_are_rejected() -> None:
    assignments = build_canonical_logo_contract(
        _canonical_frame(),
        target="product_key",
    ).assignments
    groups = sorted(assignments["fold_group"].unique())
    duplicated = assignments.copy()
    duplicated.loc[
        duplicated["fold_group"].eq(groups[1]),
        "fold_group",
    ] = groups[0]
    with pytest.raises(ValueError, match="fold_group values must be unique"):
        validate_canonical_logo_assignments(
            _canonical_frame(),
            duplicated,
            target="product_key",
        )

    omitted = assignments.loc[~assignments["fold_group"].eq(groups[-1])]
    with pytest.raises(ValueError, match="expected group count"):
        validate_canonical_logo_assignments(
            _canonical_frame(),
            omitted,
            target="product_key",
        )


def test_canonical_reaction_key_crossing_declared_groups_is_rejected() -> None:
    contaminated = _canonical_frame()
    contaminated.loc[
        contaminated["source_row_id"].eq("row-02"),
        "canonical_reaction_key",
    ] = "reaction-a"

    with pytest.raises(ValueError, match="maps to multiple declared"):
        build_canonical_logo_contract(
            contaminated,
            target="product_key",
        )


def test_assignment_reaction_key_tampering_is_rejected() -> None:
    assignments = build_canonical_logo_contract(
        _canonical_frame(),
        target="reactant_key",
    ).assignments
    assignments.loc[0, "canonical_reaction_key"] = "tampered-reaction"

    with pytest.raises(ValueError, match="reaction keys do not match"):
        validate_canonical_logo_assignments(
            _canonical_frame(),
            assignments,
            target="reactant_key",
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.drop(columns="source_row_id"),
            "missing required columns",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "must be unique",
        ),
        (
            lambda frame: frame.assign(canonical_product_key="one-group"),
            "at least two groups",
        ),
    ],
)
def test_invalid_canonical_inputs_are_rejected(
    mutator: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_canonical_logo_contract(
            mutator(_canonical_frame()),  # type: ignore[operator]
            target="product_key",
        )


def test_assignment_schema_is_strict_and_stable() -> None:
    assignments = build_canonical_logo_contract(
        _canonical_frame(),
        target="product_key",
    ).assignments

    assert tuple(assignments.columns) == LOGO_ASSIGNMENT_COLUMNS
    assignments["unreviewed"] = "value"
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_canonical_logo_assignments(
            _canonical_frame(),
            assignments,
            target="product_key",
        )


def test_unsupported_logical_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported canonical LOGO target"):
        build_canonical_logo_contract(
            _canonical_frame(),
            target="raw_product",
        )
