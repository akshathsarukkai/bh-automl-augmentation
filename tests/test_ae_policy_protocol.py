"""Tests for the supervised-AE inner measured-data boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from bh_augmentation.evaluation.ae_policy_protocol import (
    AEInnerSplit,
    make_ae_inner_split,
)


def _measured_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_row_id": "row-00",
                "canonical_reaction_key": "group-a",
                "yield": 10.0,
            },
            {
                "source_row_id": "row-01",
                "canonical_reaction_key": "group-a",
                "yield": 11.0,
            },
            {
                "source_row_id": "row-02",
                "canonical_reaction_key": "group-b",
                "yield": 20.0,
            },
            {
                "source_row_id": "row-03",
                "canonical_reaction_key": "group-c",
                "yield": 30.0,
            },
            {
                "source_row_id": "row-04",
                "canonical_reaction_key": "group-d",
                "yield": 40.0,
            },
            {
                "source_row_id": "row-05",
                "canonical_reaction_key": "group-e",
                "yield": 50.0,
            },
        ]
    )


def test_grouped_inner_split_is_deterministic_and_keeps_duplicate_groups_intact() -> None:
    frame = _measured_frame()

    first = make_ae_inner_split(frame, valid_size=0.34, seed=17)
    second = make_ae_inner_split(frame, valid_size=0.34, seed=17)

    assert first.ae_train_source_ids == second.ae_train_source_ids
    assert (
        first.internal_validation_source_ids
        == second.internal_validation_source_ids
    )
    assert first.source_id_hashes == second.source_id_hashes
    assert first.split_hash == second.split_hash
    assignments = {}
    for partition_name, partition in (
        ("train", first.ae_train),
        ("valid", first.internal_validation),
    ):
        for group in partition["canonical_reaction_key"]:
            assignments.setdefault(group, set()).add(partition_name)
    assert all(len(partitions) == 1 for partitions in assignments.values())
    assert (
        ("row-00" in first.ae_train_source_ids)
        == ("row-01" in first.ae_train_source_ids)
    )


def test_membership_and_hashes_are_invariant_to_input_row_order() -> None:
    frame = _measured_frame()
    shuffled = frame.sample(frac=1.0, random_state=91).reset_index(drop=True)

    ordered_split = make_ae_inner_split(frame, valid_size=0.3, seed=8)
    shuffled_split = make_ae_inner_split(
        shuffled,
        valid_size=0.3,
        seed=8,
    )

    assert ordered_split.ae_train_source_ids == shuffled_split.ae_train_source_ids
    assert (
        ordered_split.internal_validation_source_ids
        == shuffled_split.internal_validation_source_ids
    )
    assert ordered_split.split_hash == shuffled_split.split_hash
    pd.testing.assert_frame_equal(
        ordered_split.eligible_outer_train,
        shuffled_split.eligible_outer_train,
    )


def test_one_inner_split_provides_shared_defensive_partition_membership() -> None:
    split = make_ae_inner_split(
        _measured_frame(),
        valid_size=0.3,
        seed=2,
    )
    candidate_memberships = [
        (
            split.ae_train_source_ids,
            split.internal_validation_source_ids,
        )
        for _ in ("latent-16", "latent-32", "latent-64")
    ]

    assert len(set(candidate_memberships)) == 1
    mutated = split.ae_train
    mutated.loc[:, "yield"] = -999.0
    assert not split.ae_train["yield"].eq(-999.0).any()
    assert split.audit_record["source_id_hashes"] == split.source_id_hashes


@pytest.mark.parametrize(
    ("valid_ids", "test_ids", "message"),
    [
        (["row-02"], [], "Forbidden outer rows"),
        ([], ["row-04"], "Forbidden outer rows"),
        (["external-overlap"], ["external-overlap"], "validation and test"),
    ],
)
def test_forbidden_outer_partition_contamination_is_rejected(
    valid_ids: list[str],
    test_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_ae_inner_split(
            _measured_frame(),
            valid_size=0.3,
            seed=3,
            forbidden_outer_valid_source_ids=valid_ids,
            forbidden_outer_test_source_ids=test_ids,
        )


def test_explicit_partitions_reject_overlap_and_inexact_coverage() -> None:
    frame = _measured_frame()
    train = frame.iloc[:4]
    valid_with_overlap = frame.iloc[3:]

    with pytest.raises(ValueError, match="source IDs overlap"):
        AEInnerSplit.from_partitions(
            frame,
            train,
            valid_with_overlap,
            seed=1,
            valid_size=0.3,
        )

    with pytest.raises(ValueError, match="do not exactly cover"):
        AEInnerSplit.from_partitions(
            frame,
            frame.iloc[:3],
            frame.iloc[4:],
            seed=1,
            valid_size=0.3,
        )


def test_explicit_partitions_reject_partial_canonical_group_and_changed_rows() -> None:
    frame = _measured_frame()
    split_group_train = frame.loc[
        frame["source_row_id"].isin(["row-00", "row-02", "row-03"])
    ]
    split_group_valid = frame.loc[
        ~frame["source_row_id"].isin(["row-00", "row-02", "row-03"])
    ]
    with pytest.raises(ValueError, match="Canonical reaction groups cross"):
        AEInnerSplit.from_partitions(
            frame,
            split_group_train,
            split_group_valid,
            seed=1,
            valid_size=0.3,
        )

    valid = frame.iloc[-2:].copy()
    valid.loc[valid["source_row_id"].eq("row-05"), "yield"] = 999.0
    with pytest.raises(ValueError, match="do not exactly match"):
        AEInnerSplit.from_partitions(
            frame,
            frame.iloc[:-2],
            valid,
            seed=1,
            valid_size=0.3,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"valid_size": 0.0, "seed": 1}, "valid_size"),
        ({"valid_size": 1.0, "seed": 1}, "valid_size"),
        ({"valid_size": float("nan"), "seed": 1}, "valid_size"),
        ({"valid_size": 0.2, "seed": True}, "seed"),
    ],
)
def test_invalid_split_parameters_are_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_ae_inner_split(_measured_frame(), **kwargs)


def test_insufficient_measured_row_or_group_support_is_rejected() -> None:
    with pytest.raises(ValueError, match="measured-row support"):
        make_ae_inner_split(
            _measured_frame().iloc[:3],
            valid_size=0.3,
            seed=1,
        )

    one_group = _measured_frame().assign(
        canonical_reaction_key="one-canonical-group"
    )
    with pytest.raises(ValueError, match="canonical-group support"):
        make_ae_inner_split(one_group, valid_size=0.3, seed=1)
