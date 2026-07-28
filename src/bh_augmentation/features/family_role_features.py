"""Family-agnostic role-separated fingerprint features.

This is the representation counterpart of
:class:`bh_augmentation.data.reaction_family.ReactionFamilyAdapter`. The
Buchwald-Hartwig ``bh_role_separated`` feature kind in
:mod:`bh_augmentation.features.featurize` hard-codes the seven BH role names; the
*shape* of the representation (one fingerprint block per typed role, concatenated
in canonical role order, with strict rejection of missing or unparseable roles)
is not BH-specific at all.

This module reuses :func:`bh_augmentation.features.featurize.morgan_fingerprint`
and :class:`bh_augmentation.features.compatibility.FeatureMetadata` unchanged and
only substitutes the adapter's role vocabulary. The existing BH feature path is
not modified.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from bh_augmentation.data.reaction_family import ReactionFamilyAdapter, is_missing_molecule
from bh_augmentation.features.compatibility import FeatureBlock, FeatureMetadata
from bh_augmentation.features.featurize import morgan_fingerprint

FAMILY_ROLE_SEPARATED_SUFFIX = "role_separated"


def family_role_separated_kind(adapter: ReactionFamilyAdapter) -> str:
    """Return the representation-kind string for one family's role blocks."""
    return f"{adapter.family_id}_{FAMILY_ROLE_SEPARATED_SUFFIX}"


def family_role_separated_features(
    df: pd.DataFrame,
    adapter: ReactionFamilyAdapter,
    *,
    n_bits: int = 256,
    radius: int = 2,
    fingerprint_backend: str = "rdkit",
    roles: Sequence[str] | None = None,
    prefer_canonical_columns: bool = True,
) -> tuple[np.ndarray, list[str], FeatureMetadata]:
    """Build one Morgan fingerprint block per typed role, in canonical role order.

    Canonicalized role columns are preferred when present so that measured and
    synthetic rows are featurized from the identical canonical representation.
    Missing or unparseable roles are rejected rather than encoded as zero
    fingerprints.
    """
    if int(n_bits) < 1:
        raise ValueError("n_bits must be at least 1.")
    if int(radius) < 0:
        raise ValueError("radius must be non-negative.")
    included = tuple(roles) if roles is not None else adapter.role_names
    unknown = sorted(set(included) - set(adapter.role_names))
    if unknown:
        raise ValueError(f"Unknown roles for family {adapter.family_id!r}: {unknown}.")

    columns = {
        role: _resolve_role_column(df, adapter, role, prefer_canonical_columns)
        for role in included
    }
    blocks: list[np.ndarray] = []
    for role in included:
        values = df[columns[role]].tolist()
        vectors = []
        for position, value in enumerate(values):
            if is_missing_molecule(value):
                raise ValueError(
                    f"Row {position} has a missing required {adapter.family_id} role "
                    f"{role!r}; role-separated features refuse zero fingerprints."
                )
            vectors.append(
                morgan_fingerprint(
                    str(value).strip(),
                    radius=int(radius),
                    n_bits=int(n_bits),
                    warn_invalid=False,
                    backend=fingerprint_backend,
                )
            )
        block = (
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.empty((0, int(n_bits)), dtype=np.float32)
        )
        if block.size and not block.any(axis=1).all():
            offending = int(np.argmin(block.any(axis=1)))
            raise ValueError(
                f"Role {role!r} produced an all-zero fingerprint at row {offending}; "
                "this indicates an unparseable molecule, not a chemical identity."
            )
        blocks.append(block)

    features = (
        np.hstack(blocks).astype(np.float32)
        if blocks
        else np.empty((len(df), 0), dtype=np.float32)
    )
    names = [
        f"{role}__morgan_{bit}" for role in included for bit in range(int(n_bits))
    ]
    metadata = FeatureMetadata(
        representation_kind=family_role_separated_kind(adapter),
        n_bits=int(n_bits),
        radius=int(radius),
        fingerprint_backend=str(fingerprint_backend),
        role_ordering=tuple(included),
        block_slices=tuple(
            FeatureBlock(
                name=role,
                start=index * int(n_bits),
                stop=(index + 1) * int(n_bits),
            )
            for index, role in enumerate(included)
        ),
        total_width=int(features.shape[1]),
    )
    return features, names, metadata


def _resolve_role_column(
    df: pd.DataFrame,
    adapter: ReactionFamilyAdapter,
    role: str,
    prefer_canonical_columns: bool,
) -> str:
    canonical = f"canonical_{role}_smiles"
    if prefer_canonical_columns and canonical in df.columns:
        return canonical
    source = adapter.role_to_column[role]
    if source in df.columns:
        return source
    raise ValueError(
        f"Frame is missing both {canonical!r} and {source!r} for role {role!r}."
    )
