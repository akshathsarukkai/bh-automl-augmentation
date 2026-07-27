"""Contracts for deterministic train-only simple augmentation controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import bh_augmentation.augmentation.simple_controls as simple_controls
from bh_augmentation.augmentation.simple_controls import (
    SIMPLE_CONTROL_IDS,
    SIMPLE_CONTROL_NAMES,
    SIMPLE_CONTROL_SPECS,
    build_simple_augmentation_control,
)

X_INPUT = np.asarray(
    [
        [3.0, 5.0],  # row-d
        [0.0, 0.0],  # row-a
        [0.0, 4.0],  # row-c
        [2.0, 0.0],  # row-b
    ],
    dtype=np.float32,
)
Y_INPUT = np.asarray([90.0, 10.0, 70.0, 30.0], dtype=np.float32)
SOURCE_IDS = ("row-d", "row-a", "row-c", "row-b")
CANONICAL_X = X_INPUT[[1, 3, 2, 0]]
CANONICAL_Y = Y_INPUT[[1, 3, 2, 0]]
CANONICAL_IDS = ("row-a", "row-b", "row-c", "row-d")


def _build(control_id: str, **overrides: object):
    kwargs: dict[str, object] = {
        "added_count": 3,
        "random_state": 17,
        "n_yield_strata": 2,
        "mixup_alpha": 0.25,
    }
    kwargs.update(overrides)
    return build_simple_augmentation_control(
        control_id,
        X_INPUT,
        Y_INPUT,
        SOURCE_IDS,
        **kwargs,
    )


def test_control_registry_is_frozen_in_required_scientific_order() -> None:
    assert SIMPLE_CONTROL_IDS == (
        "real_only",
        "exact_duplication",
        "random_oversampling",
        "yield_stratified_oversampling",
        "sample_reweighting",
        "nearest_neighbor_pseudo_labeling",
        "self_training",
        "feature_mixup",
    )
    assert SIMPLE_CONTROL_NAMES == tuple(name for _, name in SIMPLE_CONTROL_SPECS)


@pytest.mark.parametrize("control_id", SIMPLE_CONTROL_IDS)
def test_every_control_is_deterministic_and_input_permutation_invariant(
    control_id: str,
) -> None:
    first = _build(control_id)
    second = _build(control_id)
    permutation = np.asarray([2, 0, 3, 1])
    permuted = build_simple_augmentation_control(
        control_id,
        X_INPUT[permutation],
        Y_INPUT[permutation],
        tuple(SOURCE_IDS[index] for index in permutation),
        added_count=3,
        random_state=17,
        n_yield_strata=2,
        mixup_alpha=0.25,
    )

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.X, permuted.X)
    np.testing.assert_array_equal(first.y, permuted.y)
    if first.sample_weight is None:
        assert second.sample_weight is None
        assert permuted.sample_weight is None
    else:
        np.testing.assert_array_equal(first.sample_weight, second.sample_weight)
        np.testing.assert_array_equal(first.sample_weight, permuted.sample_weight)
    pd.testing.assert_frame_equal(first.row_audit, second.row_audit)
    pd.testing.assert_frame_equal(first.row_audit, permuted.row_audit)
    assert tuple(first.X[:4].tolist()) == tuple(CANONICAL_X.tolist())
    assert first.requested_added_count == 3


def test_real_only_and_uniform_exact_cyclic_duplication_have_exact_formulas() -> None:
    real = _build("real_only")
    duplicated = _build("exact_duplication", added_count=6)

    np.testing.assert_array_equal(real.X, CANONICAL_X)
    np.testing.assert_array_equal(real.y, CANONICAL_Y)
    assert real.effective_added_count == 0
    assert real.total_sample_weight == 4.0

    expected_positions = np.asarray([0, 1, 2, 3, 0, 1])
    np.testing.assert_array_equal(duplicated.X[4:], CANONICAL_X[expected_positions])
    np.testing.assert_array_equal(duplicated.y[4:], CANONICAL_Y[expected_positions])
    assert duplicated.effective_added_count == 6
    assert duplicated.total_sample_weight == 10.0
    added = duplicated.row_audit.loc[duplicated.row_audit["is_added"]]
    assert tuple(added["source_row_id"]) == (
        "row-a",
        "row-b",
        "row-c",
        "row-d",
        "row-a",
        "row-b",
    )
    assert added["feature_duplicate"].all()


def test_random_and_yield_stratified_oversampling_use_exact_seeded_budgets() -> None:
    random = _build("random_oversampling", added_count=11)
    stratified = _build("yield_stratified_oversampling", added_count=11)

    assert random.effective_added_count == 11
    assert stratified.effective_added_count == 11
    random_added = random.row_audit.loc[random.row_audit["is_added"]]
    assert set(random_added["source_row_id"]) <= set(CANONICAL_IDS)
    stratified_added = stratified.row_audit.loc[stratified.row_audit["is_added"]]
    counts = stratified_added["yield_stratum"].value_counts()
    assert counts.max() - counts.min() <= 1
    source_yields = dict(zip(CANONICAL_IDS, CANONICAL_Y, strict=True))
    assert tuple(stratified.y[4:]) == tuple(
        source_yields[source] for source in stratified_added["source_row_id"]
    )


def test_yield_stratified_reweighting_matches_budget_without_adding_rows() -> None:
    result = _build("sample_reweighting", added_count=7)

    assert result.effective_added_count == 0
    assert result.sample_weight is not None
    assert result.total_sample_weight == pytest.approx(11.0)
    assert result.sample_weight.sum(dtype=np.float64) == pytest.approx(11.0)
    audit = result.row_audit.set_index("output_row_index")
    strata = audit["yield_stratum"].to_numpy(dtype=int)
    extras = result.sample_weight.astype(float) - 1.0
    extra_by_stratum = [
        float(extras[strata == stratum].sum())
        for stratum in sorted(set(strata))
    ]
    assert extra_by_stratum == pytest.approx([3.5, 3.5])
    assert (result.sample_weight >= 0).all()


def test_feature_controls_share_train_parent_convex_pool_and_never_claim_chemistry(
) -> None:
    nearest = _build("nearest_neighbor_pseudo_labeling")
    self_training = _build("self_training")
    mixup = _build("feature_mixup")

    np.testing.assert_array_equal(nearest.X, self_training.X)
    np.testing.assert_array_equal(nearest.X, mixup.X)
    for result in (nearest, self_training, mixup):
        assert result.effective_added_count == 3
        assert result.chemical_identity_applicable is False
        assert result.nonchemical_semantics is not None
        assert result.row_audit["canonical_reaction_key"].isna().all()
        assert result.row_audit["canonical_reaction_hash"].isna().all()
        referenced = set()
        for column in ("source_row_id", "donor_row_id", "label_source_row_id"):
            referenced |= set(result.row_audit[column].dropna())
        assert referenced <= set(CANONICAL_IDS)


def test_mixup_features_and_labels_match_audited_parent_formulas() -> None:
    alpha = 0.25
    result = _build("feature_mixup", mixup_alpha=alpha)
    source_position = {source: index for index, source in enumerate(CANONICAL_IDS)}
    accepted = result.row_audit.loc[
        result.row_audit["is_added"] & result.row_audit["accepted"]
    ].sort_values("output_row_index")

    for output_position, row in zip(
        range(4, len(result.X)),
        accepted.itertuples(),
        strict=True,
    ):
        left = source_position[row.source_row_id]
        right = source_position[row.donor_row_id]
        np.testing.assert_array_equal(
            result.X[output_position],
            (
                alpha * CANONICAL_X[left]
                + (1.0 - alpha) * CANONICAL_X[right]
            ).astype(np.float32),
        )
        assert result.y[output_position] == pytest.approx(
            alpha * CANONICAL_Y[left] + (1.0 - alpha) * CANONICAL_Y[right]
        )
        assert row.alpha == alpha


def test_nearest_neighbor_labels_use_real_train_only_and_lexical_ties() -> None:
    result = _build(
        "nearest_neighbor_pseudo_labeling",
        added_count=1,
        mixup_alpha=0.5,
        random_state=4,
    )
    added = result.row_audit.loc[
        result.row_audit["is_added"] & result.row_audit["accepted"]
    ].iloc[0]
    candidate = result.X[-1]
    distances = np.sum((CANONICAL_X - candidate) ** 2, axis=1)
    nearest = int(np.argmin(distances))

    assert added["label_source_row_id"] == CANONICAL_IDS[nearest]
    assert result.y[-1] == CANONICAL_Y[nearest]
    assert added["label_strategy"] == "nearest_real_train_yield_euclidean"


def test_self_training_teacher_fits_current_real_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_fit_rows: list[int] = []
    real_train_model = simple_controls.train_model

    def capture_train(model, X, y, sample_weight=None):
        observed_fit_rows.append(len(X))
        return real_train_model(model, X, y, sample_weight)

    monkeypatch.setattr(simple_controls, "train_model", capture_train)
    result = _build(
        "self_training",
        teacher_model_config={"name": "ridge", "params": {"alpha": 1.0}},
    )
    expected = Ridge(alpha=1.0).fit(CANONICAL_X, CANONICAL_Y).predict(result.X[4:])

    assert observed_fit_rows == [4]
    np.testing.assert_allclose(result.y[4:], expected, rtol=1e-6)
    added = result.row_audit.loc[result.row_audit["is_added"]]
    assert added["label_source_row_id"].isna().all()
    assert set(added["label_strategy"]) == {"real_train_only_teacher:ridge"}


def test_candidate_feature_deduplication_underfill_is_explicitly_audited() -> None:
    result = build_simple_augmentation_control(
        "feature_mixup",
        np.asarray([[0.0], [2.0]], dtype=np.float32),
        np.asarray([10.0, 30.0], dtype=np.float32),
        ("left", "right"),
        added_count=2,
        mixup_alpha=0.5,
    )

    assert result.requested_added_count == 2
    assert result.effective_added_count == 1
    assert "unique_candidate_pool_exhausted" in set(
        result.row_audit["rejection_reason"].dropna()
    )


@pytest.mark.parametrize(
    ("X", "y", "source_ids", "message"),
    [
        (
            np.asarray([[np.nan, 0.0]], dtype=np.float32),
            np.asarray([1.0]),
            ("row",),
            "finite",
        ),
        (
            np.asarray([[0.0], [1.0]], dtype=np.float32),
            np.asarray([1.0]),
            ("one", "two"),
            "same number",
        ),
        (
            np.asarray([[0.0], [1.0]], dtype=np.float32),
            np.asarray([1.0, 2.0]),
            ("duplicate", "duplicate"),
            "unique",
        ),
    ],
)
def test_invalid_or_ambiguous_training_inputs_are_rejected(
    X: np.ndarray,
    y: np.ndarray,
    source_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_simple_augmentation_control(
            "real_only",
            X,
            y,
            source_ids,
            added_count=0,
        )


def test_invalid_budget_alpha_teacher_and_control_are_rejected() -> None:
    with pytest.raises(ValueError, match="added_count"):
        _build("real_only", added_count=-1)
    with pytest.raises(ValueError, match="mixup_alpha"):
        _build("feature_mixup", mixup_alpha=1.0)
    with pytest.raises(ValueError, match="teacher_model_config"):
        _build("self_training", teacher_model_config={"name": "ridge"})
    with pytest.raises(ValueError, match="Unsupported"):
        _build("not-a-control")
