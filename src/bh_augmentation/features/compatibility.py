"""Semantic compatibility checks for measured and synthetic feature matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureBlock:
    """One named half-open feature slice."""

    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class FeatureMetadata:
    """Scientific semantics required to compare or stack feature matrices."""

    representation_kind: str
    n_bits: int
    radius: int
    fingerprint_backend: str
    role_ordering: tuple[str, ...]
    block_slices: tuple[FeatureBlock, ...]
    total_width: int

    def to_dict(self) -> dict[str, Any]:
        """Return stable plain metadata for audits and equality assertions."""
        return asdict(self)


def assert_feature_compatibility(
    X_real: np.ndarray,
    real_feature_names: Sequence[str],
    X_synthetic: np.ndarray,
    synthetic_feature_names: Sequence[str],
    *,
    real_metadata: FeatureMetadata | Mapping[str, Any] | None = None,
    synthetic_metadata: FeatureMetadata | Mapping[str, Any] | None = None,
) -> None:
    """Reject feature stacks whose widths or scientific semantics differ."""
    real = np.asarray(X_real)
    synthetic = np.asarray(X_synthetic)
    if real.ndim != 2 or synthetic.ndim != 2:
        raise ValueError(
            f"Feature compatibility requires 2D matrices; got {real.shape} and {synthetic.shape}."
        )
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError(
            "Feature width mismatch before stacking: "
            f"real={real.shape[1]}, synthetic={synthetic.shape[1]}."
        )

    real_names = list(real_feature_names)
    synthetic_names = list(synthetic_feature_names)
    if real_names != synthetic_names:
        mismatch = _first_sequence_mismatch(real_names, synthetic_names)
        raise ValueError(f"Feature name mismatch before stacking: {mismatch}.")
    if len(real_names) != real.shape[1]:
        raise ValueError(
            f"Real feature-name count {len(real_names)} does not match matrix width {real.shape[1]}."
        )

    if real_metadata is None or synthetic_metadata is None:
        raise ValueError(
            "Feature metadata is required before stacking measured and synthetic matrices."
        )
    real_values = _metadata_mapping(real_metadata)
    synthetic_values = _metadata_mapping(synthetic_metadata)
    for field in [
        "representation_kind",
        "n_bits",
        "radius",
        "fingerprint_backend",
        "role_ordering",
        "block_slices",
        "total_width",
    ]:
        if real_values.get(field) != synthetic_values.get(field):
            raise ValueError(
                f"Feature metadata mismatch for '{field}': "
                f"real={real_values.get(field)!r}, synthetic={synthetic_values.get(field)!r}."
            )


def block_slice(metadata: FeatureMetadata, block_name: str) -> slice:
    """Return one named feature slice or raise a clear error."""
    for block in metadata.block_slices:
        if block.name == block_name:
            return slice(block.start, block.stop)
    available = ", ".join(block.name for block in metadata.block_slices)
    raise ValueError(f"Unknown feature block '{block_name}'. Available blocks: {available}.")


def coordinate_feature_contract(
    representation_kind: str,
    width: int,
) -> tuple[list[str], FeatureMetadata]:
    """Describe a non-molecular coordinate space such as an AE latent vector."""
    if width < 1:
        raise ValueError("Coordinate feature width must be at least 1.")
    names = [f"coordinate_{index}" for index in range(width)]
    metadata = FeatureMetadata(
        representation_kind=representation_kind,
        n_bits=0,
        radius=0,
        fingerprint_backend="not_applicable",
        role_ordering=(),
        block_slices=(FeatureBlock("coordinates", 0, width),),
        total_width=width,
    )
    return names, metadata


def _metadata_mapping(value: FeatureMetadata | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, FeatureMetadata):
        return value.to_dict()
    return dict(value)


def _first_sequence_mismatch(left: list[str], right: list[str]) -> str:
    for index, (left_name, right_name) in enumerate(zip(left, right, strict=False)):
        if left_name != right_name:
            return f"index {index}: real={left_name!r}, synthetic={right_name!r}"
    return f"different lengths: real={len(left)}, synthetic={len(right)}"
