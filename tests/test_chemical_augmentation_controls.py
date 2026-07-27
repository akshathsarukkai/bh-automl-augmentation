"""Contracts for the frozen Phase 11 chemical augmentation controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

import bh_augmentation.augmentation.chemical_controls as controls
from bh_augmentation.features.compatibility import FeatureBlock, FeatureMetadata


def _inputs() -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "source_row_id": ["row-b", "row-a"],
            "yield": [20.0, 80.0],
        }
    )
    metadata = FeatureMetadata(
        representation_kind="bh_role_separated",
        n_bits=1,
        radius=2,
        fingerprint_backend="rdkit",
        role_ordering=(),
        block_slices=(FeatureBlock("features", 0, 2),),
        total_width=2,
    )
    return {
        "train_frame": frame,
        "X_train": np.array([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "y_train": np.array([20.0, 80.0], dtype=np.float32),
        "feature_config": {
            "kind": "bh_role_separated",
            "n_bits": 1,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        },
        "feature_names": ["feature-0", "feature-1"],
        "feature_metadata": metadata,
        "measured_identity_keys": {"measured-key"},
        "nominal_added_budget": 2,
        "seed": 17,
    }


def _candidate_rows(
    *,
    filtered: bool = False,
    outside_parent: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_reaction_key": ["candidate-a", "candidate-b"],
            "canonical_reaction_hash": ["hash-a", "hash-b"],
            "source_row_id": ["row-a", "row-b"],
            "donor_row_id": ["outside" if outside_parent else "row-b", "row-a"],
            "accepted": [True, not filtered],
            "kept": [True, not filtered],
            "teacher_std": [0.1, 2.0],
            "rejection_reason": [None, "teacher_uncertainty_above_threshold" if filtered else None],
        }
    )


def _generator_result(*, filtered: bool = False, underfill: bool = False) -> dict[str, Any]:
    candidates = _candidate_rows(filtered=filtered or underfill)
    kept = candidates["kept"].to_numpy(dtype=bool)
    count = int(kept.sum())
    return {
        "candidate_df": candidates,
        "synthetic_df": pd.DataFrame({"yield": np.arange(count, dtype=float)}),
        "synthetic_y": np.arange(count, dtype=np.float32) + 50.0,
        "X_synthetic": np.ones((count, 2), dtype=np.float32),
        "feature_names": ["feature-0", "feature-1"],
        "feature_metadata": _inputs()["feature_metadata"],
        "metadata": {
            "n_candidates_generated": len(candidates),
            "n_candidates_accepted": int(candidates["accepted"].sum()),
            "n_synthetic_train": count,
        },
    }


def test_specs_are_ordered_frozen_and_have_exact_semantics() -> None:
    assert tuple(spec.control_number for spec in controls.CHEMICAL_CONTROL_SPECS) == (
        9,
        10,
        11,
        12,
        13,
    )
    assert controls.CHEMICAL_CONTROL_IDS == (
        "anonymous_condition_transfer",
        "random_typed_transfer",
        "strict_context_matched_typed_transfer",
        "typed_transfer_without_uncertainty_filtering",
        "typed_transfer_with_uncertainty_filtering",
    )
    anonymous, random_typed, strict, unfiltered, filtered = (
        controls.CHEMICAL_CONTROL_SPECS
    )
    assert anonymous.requested_roles == ("ligand", "base", "solvent_or_additive")
    assert anonymous.label_strategy == "teacher_ensemble"
    assert random_typed.role_transfer_mode == strict.role_transfer_mode == "ligand_base"
    assert random_typed.donor_strategy == "random"
    assert strict.donor_strategy == "same_substrate_different_role"
    assert strict.context_definition == "exact_reactant_key"
    assert unfiltered.label_strategy == "teacher_ensemble"
    assert filtered.label_strategy == "uncertainty_filtered_teacher"
    assert all(spec.role_change_requirement == "all" for spec in controls.CHEMICAL_CONTROL_SPECS)
    assert all(spec.fallback_policy == "reject" for spec in controls.CHEMICAL_CONTROL_SPECS)
    assert all(
        spec.donor_similarity_backend == "rdkit"
        for spec in controls.CHEMICAL_CONTROL_SPECS
    )
    with pytest.raises(FrozenInstanceError):
        strict.donor_strategy = "random"  # type: ignore[misc]


@pytest.mark.parametrize("control_id", controls.CHEMICAL_CONTROL_IDS)
def test_builder_maps_each_control_to_exact_generator_config(
    monkeypatch: pytest.MonkeyPatch,
    control_id: str,
) -> None:
    observed: list[object] = []

    def anonymous_generator(*args: object, **kwargs: object) -> dict[str, Any]:
        observed.append(args[3])
        return _generator_result()

    def typed_generator(*args: object, **kwargs: object) -> dict[str, Any]:
        config = args[3]
        observed.append(config)
        filtered = config.label_strategy == "uncertainty_filtered_teacher"
        return _generator_result(filtered=filtered)

    monkeypatch.setattr(controls, "generate_condition_transfer_examples", anonymous_generator)
    monkeypatch.setattr(
        controls,
        "generate_role_aware_condition_transfer_examples",
        typed_generator,
    )
    kwargs = _inputs()
    if control_id == "typed_transfer_with_uncertainty_filtering":
        kwargs["max_teacher_std"] = 0.5
    result = controls.build_chemical_augmentation_control(control_id, **kwargs)

    spec = next(
        item for item in controls.CHEMICAL_CONTROL_SPECS if item.control_id == control_id
    )
    final_config = observed[-1]
    assert final_config.donor_strategy == spec.donor_strategy
    assert final_config.label_strategy == spec.label_strategy
    assert final_config.role_change_requirement == "all"
    assert final_config.fallback_policy == "reject"
    assert final_config.donor_similarity_backend == "rdkit"
    assert final_config.max_candidates_per_source == 3
    if spec.generator_family == "anonymous":
        assert tuple(final_config.requested_roles) == spec.requested_roles
    else:
        assert final_config.role_transfer_mode == "ligand_base"
    assert result.control_id == control_id
    assert result.requested_added_count == 2
    assert result.effective_added_count <= result.requested_added_count
    assert result.generator_metadata["used_validation_or_test_parents"] is False


def test_explicit_teacher_models_are_frozen_into_generator_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []

    def typed_generator(*args: object, **kwargs: object) -> dict[str, Any]:
        observed.append(args[3])
        return _generator_result()

    monkeypatch.setattr(
        controls,
        "generate_role_aware_condition_transfer_examples",
        typed_generator,
    )
    result = controls.build_chemical_augmentation_control(
        "typed_transfer_without_uncertainty_filtering",
        teacher_models=("ridge",),
        **_inputs(),
    )

    assert observed[-1].teacher_models == ["ridge"]
    assert result.resolved_config["teacher_models"] == ["ridge"]


def test_budget_underfill_is_recorded_and_never_backfilled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controls,
        "generate_role_aware_condition_transfer_examples",
        lambda *args, **kwargs: _generator_result(underfill=True),
    )
    result = controls.build_chemical_augmentation_control(
        "strict_context_matched_typed_transfer",
        **_inputs(),
    )

    assert result.requested_added_count == 2
    assert result.effective_added_count == 1
    assert result.budget_underfill_count == 1
    assert result.budget_underfill_reason is not None
    assert result.generator_metadata["budget_backfill_performed"] is False
    assert result.X.shape == (3, 2)
    assert result.y.shape == (3,)
    assert result.total_sample_weight == 3.0
    assert int(result.row_audit["is_added"].sum()) == 1


def test_row_audit_preserves_ranked_output_order_and_float32_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generator_result()
    generated["candidate_df"]["candidate_rank"] = [2, 1]
    generated["candidate_df"]["synthetic_label"] = [
        50.000000317,
        51.000000719,
    ]
    generated["synthetic_y"] = np.asarray(
        [51.000000719, 50.000000317], dtype=float
    )
    generated["X_synthetic"] = np.asarray(
        [[9.0, 1.0], [8.0, 2.0]], dtype=np.float32
    )
    monkeypatch.setattr(
        controls,
        "generate_role_aware_condition_transfer_examples",
        lambda *args, **kwargs: generated,
    )

    result = controls.build_chemical_augmentation_control(
        "strict_context_matched_typed_transfer",
        **_inputs(),
    )

    added = result.row_audit.loc[result.row_audit["is_added"].astype(bool)]
    added = added.sort_values("output_row_index", kind="mergesort")
    assert added["candidate_rank"].tolist() == [1, 2]
    assert added["synthetic_label"].to_numpy(float).tolist() == pytest.approx(
        result.y[-2:].astype(float).tolist(), abs=0.0
    )


def test_filtered_and_unfiltered_controls_hash_the_same_pool_and_assert_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def typed_generator(*args: object, **kwargs: object) -> dict[str, Any]:
        filtered = args[3].label_strategy == "uncertainty_filtered_teacher"
        return _generator_result(filtered=filtered)

    monkeypatch.setattr(
        controls,
        "generate_role_aware_condition_transfer_examples",
        typed_generator,
    )
    unfiltered = controls.build_chemical_augmentation_control(
        "typed_transfer_without_uncertainty_filtering",
        **_inputs(),
    )
    filtered_kwargs = _inputs()
    filtered_kwargs["max_teacher_std"] = 0.5
    filtered = controls.build_chemical_augmentation_control(
        "typed_transfer_with_uncertainty_filtering",
        **filtered_kwargs,
    )

    assert filtered.prefilter_pool_hash == unfiltered.prefilter_pool_hash
    assert filtered.uncertainty_accepted_keys_subset is True
    assert filtered.effective_added_count < unfiltered.effective_added_count
    assert (
        filtered.uncertainty_semantics
        == "raw_teacher_std_proxy_phase12_pending"
    )
    assert filtered.generator_metadata["uncertainty_semantics"].endswith(
        "phase12_pending"
    )


def test_filtered_pair_reuses_frozen_global_measured_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_keys: list[tuple[str, ...]] = []

    def typed_generator(*args: object, **kwargs: object) -> dict[str, Any]:
        observed_keys.append(tuple(kwargs["measured_identity_keys"]))
        filtered = args[3].label_strategy == "uncertainty_filtered_teacher"
        return _generator_result(filtered=filtered)

    monkeypatch.setattr(
        controls,
        "generate_role_aware_condition_transfer_examples",
        typed_generator,
    )
    kwargs = _inputs()
    kwargs["measured_identity_keys"] = iter(("measured-a", "measured-b"))
    kwargs["max_teacher_std"] = 0.5
    controls.build_chemical_augmentation_control(
        "typed_transfer_with_uncertainty_filtering",
        **kwargs,
    )

    assert observed_keys == [
        ("measured-a", "measured-b"),
        ("measured-a", "measured-b"),
    ]


def test_nontraining_parent_id_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _generator_result()
    result["candidate_df"] = _candidate_rows(outside_parent=True)
    monkeypatch.setattr(
        controls,
        "generate_condition_transfer_examples",
        lambda *args, **kwargs: result,
    )

    with pytest.raises(ValueError, match="non-training donor_row_id"):
        controls.build_chemical_augmentation_control(
            "anonymous_condition_transfer",
            **_inputs(),
        )


def test_missing_or_nonstring_training_parent_id_is_rejected() -> None:
    for invalid in (None, 7):
        kwargs = _inputs()
        kwargs["train_frame"] = kwargs["train_frame"].copy()
        kwargs["train_frame"].loc[0, "source_row_id"] = invalid
        with pytest.raises(ValueError, match="source_row_id"):
            controls.build_chemical_augmentation_control(
                "anonymous_condition_transfer",
                **kwargs,
            )


def test_filtered_threshold_is_required_and_explicitly_uncalibrated() -> None:
    with pytest.raises(ValueError, match="requires max_teacher_std"):
        controls.build_chemical_augmentation_control(
            "typed_transfer_with_uncertainty_filtering",
            **_inputs(),
        )
    with pytest.raises(ValueError, match="applies only"):
        controls.build_chemical_augmentation_control(
            "typed_transfer_without_uncertainty_filtering",
            max_teacher_std=1.0,
            **_inputs(),
        )
