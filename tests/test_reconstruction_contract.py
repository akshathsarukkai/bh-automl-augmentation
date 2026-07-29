"""Tests for reconstruction-specific feature-domain contracts."""

from __future__ import annotations

import numpy as np
import pytest

from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES
from bh_augmentation.features.compatibility import FeatureBlock, FeatureMetadata
from bh_augmentation.features.reconstruction_contract import (
    CANONICAL_COUNT_AWARE_EXCLUSION_REASON,
    VALUE_DOMAINS,
    ReconstructionFeatureContract,
    build_canonical_role_separated_reconstruction_contract,
    build_reconstruction_feature_contract,
    verify_observed_feature_domains,
)


def _canonical_metadata(n_bits: int = 4) -> FeatureMetadata:
    return FeatureMetadata(
        representation_kind="bh_role_separated",
        n_bits=n_bits,
        radius=2,
        fingerprint_backend="rdkit",
        role_ordering=CANONICAL_ROLE_NAMES,
        block_slices=tuple(
            FeatureBlock(role, index * n_bits, (index + 1) * n_bits)
            for index, role in enumerate(CANONICAL_ROLE_NAMES)
        ),
        total_width=7 * n_bits,
    )


def test_canonical_contract_is_binary_stable_and_preserves_legacy_metadata() -> None:
    metadata = _canonical_metadata()
    before = metadata.to_dict()
    observed = np.asarray(
        [[(row + column) % 2 for column in range(metadata.total_width)] for row in range(3)],
        dtype=np.float32,
    )

    first, audit = build_canonical_role_separated_reconstruction_contract(
        metadata,
        observed,
        objective="binary_cross_entropy",
    )
    second, _ = build_canonical_role_separated_reconstruction_contract(
        metadata,
        observed,
        objective="binary_cross_entropy",
    )

    assert metadata.to_dict() == before
    assert first.contract_hash == second.contract_hash
    assert first.feature_metadata == before
    assert first.selected_decoder_mode == "bernoulli_logits"
    assert first.count_aware_applicable is False
    assert (
        first.count_aware_exclusion_reason
        == CANONICAL_COUNT_AWARE_EXCLUSION_REASON
    )
    assert tuple(block.name for block in first.blocks) == CANONICAL_ROLE_NAMES
    assert all(block.value_domain == "binary_bit" for block in first.blocks)
    assert all(
        block.eligible_objectives
        == ("binary_cross_entropy", "mse", "positive_bit_weighted_mse")
        for block in first.blocks
    )
    assert audit.block_domains_verified == CANONICAL_ROLE_NAMES
    assert audit.contract_hash == first.contract_hash


@pytest.mark.parametrize(
    "objective",
    ("binary_cross_entropy", "mse", "positive_bit_weighted_mse"),
)
def test_canonical_contract_allows_only_applicable_objectives(objective: str) -> None:
    metadata = _canonical_metadata()
    observed = np.zeros((2, metadata.total_width), dtype=np.float32)

    contract, _ = build_canonical_role_separated_reconstruction_contract(
        metadata,
        observed,
        objective=objective,
    )

    assert contract.selected_objective == objective


def test_canonical_count_aware_mode_has_explicit_exclusion_reason() -> None:
    metadata = _canonical_metadata()
    observed = np.zeros((1, metadata.total_width), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="not applicable.*binary bits.*not.*counts",
    ):
        build_canonical_role_separated_reconstruction_contract(
            metadata,
            observed,
            objective="count_aware_mse",
        )


def test_canonical_observations_must_be_exactly_binary() -> None:
    metadata = _canonical_metadata()
    observed = np.zeros((2, metadata.total_width), dtype=np.float32)
    observed[0, 5] = 0.999999

    with pytest.raises(ValueError, match=r"exactly binary \(0 or 1\)"):
        build_canonical_role_separated_reconstruction_contract(
            metadata,
            observed,
            objective="mse",
        )


@pytest.mark.parametrize(
    ("domain", "objective", "decoder"),
    [
        ("binary_bit", "binary_cross_entropy", "bernoulli_logits"),
        ("one_hot", "positive_bit_weighted_mse", "identity"),
        ("nonnegative_integer_count", "count_aware_mse", "nonnegative_softplus"),
        ("signed_count_or_delta", "mse", "identity"),
        ("continuous", "mse", "identity"),
    ],
)
def test_generic_constructor_supports_every_declared_domain(
    domain: str,
    objective: str,
    decoder: str,
) -> None:
    metadata = FeatureMetadata(
        representation_kind="generic",
        n_bits=0,
        radius=0,
        fingerprint_backend="not_applicable",
        role_ordering=(),
        block_slices=(FeatureBlock("features", 0, 3),),
        total_width=3,
    )

    contract = build_reconstruction_feature_contract(
        metadata,
        {"features": domain},
        objective=objective,
    )

    assert domain in VALUE_DOMAINS
    assert contract.blocks[0].value_domain == domain
    assert contract.selected_decoder_mode == decoder
    assert len(contract.contract_hash) == 64


def test_generic_observed_domain_validation_is_strict() -> None:
    metadata = FeatureMetadata(
        "generic",
        0,
        0,
        "not_applicable",
        (),
        (
            FeatureBlock("one-hot", 0, 3),
            FeatureBlock("counts", 3, 5),
        ),
        5,
    )
    contract = build_reconstruction_feature_contract(
        metadata,
        {"one-hot": "one_hot", "counts": "nonnegative_integer_count"},
        objective="mse",
    )
    valid = np.asarray([[0, 1, 0, 2, 0], [1, 0, 0, 0, 3]], dtype=np.float32)

    audit = verify_observed_feature_domains(valid, contract)

    assert audit.row_count == 2
    invalid_one_hot = valid.copy()
    invalid_one_hot[0, 0] = 1
    with pytest.raises(ValueError, match="one-hot"):
        verify_observed_feature_domains(invalid_one_hot, contract)
    invalid_count = valid.copy()
    invalid_count[0, 3] = 0.5
    with pytest.raises(ValueError, match="nonnegative integers"):
        verify_observed_feature_domains(invalid_count, contract)


def test_missing_unknown_domain_or_incompatible_objective_fails() -> None:
    metadata = FeatureMetadata(
        "generic",
        0,
        0,
        "not_applicable",
        (),
        (FeatureBlock("a", 0, 2), FeatureBlock("b", 2, 4)),
        4,
    )
    with pytest.raises(ValueError, match="declare exactly"):
        build_reconstruction_feature_contract(
            metadata,
            {"a": "binary_bit"},
            objective="mse",
        )
    with pytest.raises(ValueError, match="Unsupported value domain"):
        build_reconstruction_feature_contract(
            metadata,
            {"a": "unknown", "b": "continuous"},
            objective="mse",
        )
    with pytest.raises(ValueError, match="incompatible"):
        build_reconstruction_feature_contract(
            metadata,
            {"a": "continuous", "b": "continuous"},
            objective="binary_cross_entropy",
        )


def test_objective_decoder_pair_cannot_be_mislabeled() -> None:
    metadata = FeatureMetadata(
        "generic",
        0,
        0,
        "not_applicable",
        (),
        (FeatureBlock("binary", 0, 2),),
        2,
    )

    with pytest.raises(ValueError, match="requires decoder mode"):
        build_reconstruction_feature_contract(
            metadata,
            {"binary": "binary_bit"},
            objective="binary_cross_entropy",
            decoder_mode="identity",
        )


def test_contract_hash_tampering_is_detected_before_observation_validation() -> None:
    metadata = _canonical_metadata()
    observed = np.zeros((1, metadata.total_width), dtype=np.float32)
    contract, _ = build_canonical_role_separated_reconstruction_contract(
        metadata,
        observed,
        objective="mse",
    )
    tampered = ReconstructionFeatureContract(
        schema_version=contract.schema_version,
        feature_metadata=contract.feature_metadata,
        feature_metadata_hash=contract.feature_metadata_hash,
        blocks=contract.blocks,
        selected_objective="binary_cross_entropy",
        selected_decoder_mode=contract.selected_decoder_mode,
        count_aware_applicable=contract.count_aware_applicable,
        count_aware_exclusion_reason=contract.count_aware_exclusion_reason,
        contract_hash=contract.contract_hash,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_observed_feature_domains(observed, tampered)


def test_canonical_role_order_and_slices_are_not_inferred_or_reordered() -> None:
    metadata = _canonical_metadata()
    reordered = FeatureMetadata(
        metadata.representation_kind,
        metadata.n_bits,
        metadata.radius,
        metadata.fingerprint_backend,
        tuple(reversed(metadata.role_ordering)),
        metadata.block_slices,
        metadata.total_width,
    )
    observed = np.zeros((1, metadata.total_width), dtype=np.float32)

    with pytest.raises(ValueError, match="role order mismatch"):
        build_canonical_role_separated_reconstruction_contract(
            reordered,
            observed,
            objective="mse",
        )
