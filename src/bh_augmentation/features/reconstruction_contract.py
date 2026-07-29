"""Strict feature-domain contracts for reconstruction objectives.

This module wraps :class:`FeatureMetadata` without modifying its schema or its
existing serialization.  Reconstruction semantics are maintained as a
separate, hash-addressed contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES
from bh_augmentation.features.compatibility import FeatureMetadata
from bh_augmentation.utils.corrected_runs import stable_hash

RECONSTRUCTION_CONTRACT_SCHEMA_VERSION = "bh-reconstruction-feature-contract-v1"
VALUE_DOMAINS = (
    "binary_bit",
    "one_hot",
    "nonnegative_integer_count",
    "signed_count_or_delta",
    "continuous",
)
RECONSTRUCTION_OBJECTIVES = (
    "mse",
    "positive_bit_weighted_mse",
    "binary_cross_entropy",
    "count_aware_mse",
)
DECODER_MODES = (
    "identity",
    "bernoulli_logits",
    "nonnegative_softplus",
)
CANONICAL_COUNT_AWARE_EXCLUSION_REASON = (
    "count-aware reconstruction is not applicable to canonical seven-role "
    "RDKit Morgan bit fingerprints; their values are binary bits, not "
    "molecular-feature counts"
)

_DOMAIN_OBJECTIVES = {
    "binary_bit": (
        "binary_cross_entropy",
        "mse",
        "positive_bit_weighted_mse",
    ),
    "one_hot": (
        "binary_cross_entropy",
        "mse",
        "positive_bit_weighted_mse",
    ),
    "nonnegative_integer_count": ("count_aware_mse", "mse"),
    "signed_count_or_delta": ("mse",),
    "continuous": ("mse",),
}
_OBJECTIVE_DECODER = {
    "mse": "identity",
    "positive_bit_weighted_mse": "identity",
    "binary_cross_entropy": "bernoulli_logits",
    "count_aware_mse": "nonnegative_softplus",
}


@dataclass(frozen=True, slots=True)
class ReconstructionBlockContract:
    """Value domain and permitted reconstruction semantics for one block."""

    name: str
    start: int
    stop: int
    value_domain: str
    eligible_objectives: tuple[str, ...]
    eligible_decoder_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconstructionFeatureContract:
    """An immutable reconstruction contract bound to feature metadata."""

    schema_version: str
    feature_metadata: Mapping[str, Any]
    feature_metadata_hash: str
    blocks: tuple[ReconstructionBlockContract, ...]
    selected_objective: str
    selected_decoder_mode: str
    count_aware_applicable: bool
    count_aware_exclusion_reason: str | None
    contract_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible contract values."""
        return {
            "schema_version": self.schema_version,
            "feature_metadata": dict(self.feature_metadata),
            "feature_metadata_hash": self.feature_metadata_hash,
            "blocks": [asdict(block) for block in self.blocks],
            "selected_objective": self.selected_objective,
            "selected_decoder_mode": self.selected_decoder_mode,
            "count_aware_applicable": self.count_aware_applicable,
            "count_aware_exclusion_reason": self.count_aware_exclusion_reason,
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True, slots=True)
class ObservedDomainAudit:
    """Proof that an observed matrix obeys every declared block domain."""

    row_count: int
    feature_width: int
    observed_dtype: str
    observed_matrix_hash: str
    contract_hash: str
    block_domains_verified: tuple[str, ...]


def build_reconstruction_feature_contract(
    metadata: FeatureMetadata,
    block_domains: Mapping[str, str],
    *,
    objective: str,
    decoder_mode: str | None = None,
    count_aware_exclusion_reason: str | None = None,
) -> ReconstructionFeatureContract:
    """Strictly bind declared block domains to unchanged feature metadata."""
    _validate_feature_metadata(metadata)
    if objective not in RECONSTRUCTION_OBJECTIVES:
        raise ValueError(f"Unsupported reconstruction objective {objective!r}.")
    selected_decoder = decoder_mode or _OBJECTIVE_DECODER[objective]
    if selected_decoder not in DECODER_MODES:
        raise ValueError(f"Unsupported decoder mode {selected_decoder!r}.")
    expected_decoder = _OBJECTIVE_DECODER[objective]
    if selected_decoder != expected_decoder:
        raise ValueError(
            f"Objective {objective!r} requires decoder mode {expected_decoder!r}; "
            f"got {selected_decoder!r}."
        )
    declared = dict(block_domains)
    block_names = tuple(block.name for block in metadata.block_slices)
    if set(declared) != set(block_names) or len(declared) != len(block_names):
        raise ValueError(
            "block_domains must declare exactly every FeatureMetadata block; "
            f"missing={sorted(set(block_names)-set(declared))}, "
            f"unknown={sorted(set(declared)-set(block_names))}."
        )
    blocks = []
    for feature_block in metadata.block_slices:
        domain = declared[feature_block.name]
        if domain not in VALUE_DOMAINS:
            raise ValueError(
                f"Unsupported value domain {domain!r} for block "
                f"{feature_block.name!r}."
            )
        eligible_objectives = _DOMAIN_OBJECTIVES[domain]
        eligible_decoders = tuple(
            dict.fromkeys(_OBJECTIVE_DECODER[item] for item in eligible_objectives)
        )
        if objective not in eligible_objectives:
            reason = (
                f" {count_aware_exclusion_reason}"
                if objective == "count_aware_mse" and count_aware_exclusion_reason
                else ""
            )
            raise ValueError(
                f"Objective {objective!r} is incompatible with block "
                f"{feature_block.name!r} domain {domain!r}.{reason}"
            )
        if selected_decoder not in eligible_decoders:
            raise ValueError(
                f"Decoder {selected_decoder!r} is incompatible with block "
                f"{feature_block.name!r} domain {domain!r}."
            )
        blocks.append(
            ReconstructionBlockContract(
                name=feature_block.name,
                start=feature_block.start,
                stop=feature_block.stop,
                value_domain=domain,
                eligible_objectives=eligible_objectives,
                eligible_decoder_modes=eligible_decoders,
            )
        )
    count_aware_applicable = all(
        "count_aware_mse" in block.eligible_objectives for block in blocks
    )
    exclusion_reason = (
        None if count_aware_applicable else count_aware_exclusion_reason
    )
    metadata_record = metadata.to_dict()
    payload = {
        "schema_version": RECONSTRUCTION_CONTRACT_SCHEMA_VERSION,
        "feature_metadata": metadata_record,
        "feature_metadata_hash": stable_hash(metadata_record),
        "blocks": [asdict(block) for block in blocks],
        "selected_objective": objective,
        "selected_decoder_mode": selected_decoder,
        "count_aware_applicable": count_aware_applicable,
        "count_aware_exclusion_reason": exclusion_reason,
    }
    return ReconstructionFeatureContract(
        schema_version=RECONSTRUCTION_CONTRACT_SCHEMA_VERSION,
        feature_metadata=metadata_record,
        feature_metadata_hash=payload["feature_metadata_hash"],
        blocks=tuple(blocks),
        selected_objective=objective,
        selected_decoder_mode=selected_decoder,
        count_aware_applicable=count_aware_applicable,
        count_aware_exclusion_reason=exclusion_reason,
        contract_hash=stable_hash(payload),
    )


def build_canonical_role_separated_reconstruction_contract(
    metadata: FeatureMetadata,
    observed_features: np.ndarray,
    *,
    objective: str,
) -> tuple[ReconstructionFeatureContract, ObservedDomainAudit]:
    """Build and verify the canonical seven-role RDKit Morgan-bit contract."""
    _validate_canonical_role_metadata(metadata)
    if objective == "count_aware_mse":
        raise ValueError(CANONICAL_COUNT_AWARE_EXCLUSION_REASON)
    domains = {role: "binary_bit" for role in CANONICAL_ROLE_NAMES}
    contract = build_reconstruction_feature_contract(
        metadata,
        domains,
        objective=objective,
        count_aware_exclusion_reason=CANONICAL_COUNT_AWARE_EXCLUSION_REASON,
    )
    audit = verify_observed_feature_domains(observed_features, contract)
    return contract, audit


def verify_observed_feature_domains(
    observed_features: np.ndarray,
    contract: ReconstructionFeatureContract,
) -> ObservedDomainAudit:
    """Hard-fail unless every observed value obeys its declared block domain."""
    _validate_contract_hash(contract)
    observed = np.asarray(observed_features)
    total_width = int(contract.feature_metadata["total_width"])
    if observed.ndim != 2 or observed.shape[1] != total_width:
        raise ValueError(
            f"Observed feature matrix must be 2D with width {total_width}; "
            f"got {observed.shape}."
        )
    if not np.issubdtype(observed.dtype, np.number) or not np.isfinite(observed).all():
        raise ValueError("Observed feature matrix must be finite and numeric.")
    verified = []
    for block in contract.blocks:
        values = observed[:, block.start : block.stop]
        _verify_block_values(values, block)
        verified.append(block.name)
    return ObservedDomainAudit(
        row_count=len(observed),
        feature_width=observed.shape[1],
        observed_dtype=str(observed.dtype),
        observed_matrix_hash=_array_hash(observed),
        contract_hash=contract.contract_hash,
        block_domains_verified=tuple(verified),
    )


def _validate_feature_metadata(metadata: FeatureMetadata) -> None:
    if not isinstance(metadata, FeatureMetadata):
        raise TypeError("metadata must be FeatureMetadata.")
    if metadata.total_width <= 0 or not metadata.block_slices:
        raise ValueError("FeatureMetadata requires positive width and blocks.")
    expected_start = 0
    names = set()
    for block in metadata.block_slices:
        if (
            not block.name
            or block.name in names
            or block.start != expected_start
            or block.stop <= block.start
        ):
            raise ValueError(
                "FeatureMetadata blocks must uniquely, contiguously cover width."
            )
        names.add(block.name)
        expected_start = block.stop
    if expected_start != metadata.total_width:
        raise ValueError("FeatureMetadata blocks do not exactly cover total_width.")


def _validate_canonical_role_metadata(metadata: FeatureMetadata) -> None:
    _validate_feature_metadata(metadata)
    if metadata.representation_kind != "bh_role_separated":
        raise ValueError("Canonical reconstruction requires bh_role_separated.")
    if metadata.fingerprint_backend != "rdkit":
        raise ValueError("Canonical reconstruction requires the RDKit backend.")
    if metadata.role_ordering != CANONICAL_ROLE_NAMES:
        raise ValueError("Canonical reconstruction role order mismatch.")
    if metadata.n_bits <= 0 or metadata.total_width != 7 * metadata.n_bits:
        raise ValueError("Canonical reconstruction Morgan block width mismatch.")
    expected_slices = tuple(
        (role, index * metadata.n_bits, (index + 1) * metadata.n_bits)
        for index, role in enumerate(CANONICAL_ROLE_NAMES)
    )
    observed_slices = tuple(
        (block.name, block.start, block.stop) for block in metadata.block_slices
    )
    if observed_slices != expected_slices:
        raise ValueError("Canonical reconstruction role slices mismatch.")


def _verify_block_values(
    values: np.ndarray,
    block: ReconstructionBlockContract,
) -> None:
    domain = block.value_domain
    if domain == "binary_bit":
        valid = np.logical_or(values == 0, values == 1).all()
        reason = "values must be exactly binary (0 or 1)"
    elif domain == "one_hot":
        valid = (
            np.logical_or(values == 0, values == 1).all()
            and np.equal(values.sum(axis=1), 1).all()
        )
        reason = "rows must be exactly one-hot"
    elif domain == "nonnegative_integer_count":
        valid = (values >= 0).all() and np.equal(values, np.floor(values)).all()
        reason = "values must be nonnegative integers"
    elif domain == "signed_count_or_delta":
        valid = np.equal(values, np.floor(values)).all()
        reason = "values must be signed integers"
    else:
        valid = True
        reason = "values must be finite continuous values"
    if not valid:
        raise ValueError(
            f"Observed block {block.name!r} violates domain {domain!r}: {reason}."
        )


def _validate_contract_hash(contract: ReconstructionFeatureContract) -> None:
    if not isinstance(contract, ReconstructionFeatureContract):
        raise TypeError("contract must be ReconstructionFeatureContract.")
    values = contract.to_dict()
    claimed = values.pop("contract_hash")
    if (
        contract.schema_version != RECONSTRUCTION_CONTRACT_SCHEMA_VERSION
        or claimed != stable_hash(values)
        or contract.feature_metadata_hash
        != stable_hash(contract.feature_metadata)
    ):
        raise ValueError("Reconstruction feature contract hash mismatch.")


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()
