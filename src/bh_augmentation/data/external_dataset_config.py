"""Validation for external reaction-family validation configs.

An external dataset may only enter this pipeline with a complete provenance and
licensing record. Config validation therefore *rejects* a configuration that
omits any provenance field (source name, source URL, license, citation, retrieved
date, raw-file SHA-256), in addition to re-checking the scientific contracts the
Buchwald-Hartwig configs already enforce (RDKit canonicalization, isomeric SMILES,
grouped splits on ``canonical_reaction_key``, no hash-fingerprint fallback).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bh_augmentation.data.reaction_family import (
    DatasetProvenance,
    ReactionFamilyAdapter,
    adapter_for_family,
    available_reaction_families,
)
from bh_augmentation.features.family_role_features import family_role_separated_kind

EXTERNAL_FAMILY_CONFIG_SCHEMA_VERSION = "external-family-validation-config-v1"

REQUIRED_TOP_LEVEL_SECTIONS = (
    "reaction_family",
    "canonicalization",
    "dataset",
    "splits",
    "features",
    "selection",
    "output",
)


def validate_external_family_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one external-family config and return it as a plain dict."""
    if not isinstance(config, Mapping):
        raise ValueError("External family config must be a mapping.")
    resolved = dict(config)

    missing_sections = [name for name in REQUIRED_TOP_LEVEL_SECTIONS if name not in resolved]
    if missing_sections:
        raise ValueError(
            f"External family config is missing sections: {sorted(missing_sections)}."
        )
    if not resolved.get("scientific_run", False):
        raise ValueError("External family validation requires scientific_run: true.")
    if resolved.get("allow_hash_fingerprint_fallback", True):
        raise ValueError("allow_hash_fingerprint_fallback must be false.")

    family = _section(resolved, "reaction_family")
    family_id = str(family.get("family_id", "")).strip()
    if family_id not in available_reaction_families():
        raise ValueError(
            f"Unknown reaction family {family_id!r}; registered families: "
            + ", ".join(available_reaction_families())
        )
    if "provenance" not in family:
        raise ValueError(
            "External datasets require reaction_family.provenance with fields: "
            "source_name, source_url, license, citation, retrieved_date, raw_file_sha256."
        )
    DatasetProvenance.from_mapping(family["provenance"])

    canonicalization = _section(resolved, "canonicalization")
    if canonicalization.get("backend") != "rdkit":
        raise ValueError("External family validation requires canonicalization.backend: rdkit.")
    if canonicalization.get("isomeric_smiles") is not True:
        raise ValueError("External family validation requires isomeric_smiles: true.")
    if not canonicalization.get("preserve_role_order", False):
        raise ValueError("canonicalization.preserve_role_order must be true.")

    dataset = _section(resolved, "dataset")
    if not str(dataset.get("path", "")).strip():
        raise ValueError("dataset.path is required.")

    splits = _section(resolved, "splits")
    if splits.get("group_column") != "canonical_reaction_key":
        raise ValueError("splits.group_column must be canonical_reaction_key.")
    if not splits.get("seeds"):
        raise ValueError("splits.seeds must contain at least one seed.")
    sizes = {name: float(splits.get(name, 0.0)) for name in ("train_size", "valid_size", "test_size")}
    for name, value in sizes.items():
        if not 0 < value < 1:
            raise ValueError(f"splits.{name} must be strictly between 0 and 1.")
    if abs(sum(sizes.values()) - 1.0) > 1e-9:
        raise ValueError("splits train/valid/test proportions must sum to 1.0.")

    features = _section(resolved, "features")
    if features.get("fingerprint_backend") != "rdkit":
        raise ValueError("features.fingerprint_backend must be rdkit.")
    expected_kind = f"{family_id}_role_separated"
    if "kind" in features:
        raise ValueError(
            "External family configs declare features.representation_kind, not "
            "features.kind. features.kind is reserved for the Buchwald-Hartwig "
            "representation vocabulary."
        )
    if str(features.get("representation_kind", "")) != expected_kind:
        raise ValueError(
            f"features.representation_kind must be {expected_kind!r} for family "
            f"{family_id!r}."
        )
    if int(features.get("n_bits", 0)) < 1:
        raise ValueError("features.n_bits must be at least 1.")

    selection = _section(resolved, "selection")
    if not str(selection.get("metric", "")).strip():
        raise ValueError("selection.metric is required.")
    if not isinstance(selection.get("lower_is_better", True), bool):
        raise ValueError("selection.lower_is_better must be boolean.")

    output = _section(resolved, "output")
    if not str(output.get("directory", "")).strip():
        raise ValueError("output.directory is required.")

    transfer = resolved.get("condition_transfer", {})
    if transfer and not isinstance(transfer, Mapping):
        raise ValueError("condition_transfer must be a mapping.")
    if transfer.get("enabled", False):
        multipliers = list(transfer.get("synthetic_multipliers", []))
        if not multipliers:
            raise ValueError(
                "condition_transfer.synthetic_multipliers must list at least one value."
            )
        if any(float(value) < 0 for value in multipliers):
            raise ValueError("condition_transfer.synthetic_multipliers must be non-negative.")
    return resolved


def build_adapter_from_config(config: Mapping[str, Any]) -> ReactionFamilyAdapter:
    """Build the family adapter declared by an already-validated config."""
    family = _section(dict(config), "reaction_family")
    provenance = DatasetProvenance.from_mapping(family["provenance"])
    kwargs: dict[str, Any] = {}
    if family.get("role_columns"):
        kwargs["role_columns"] = tuple(str(value) for value in family["role_columns"])
    if family.get("transferable_roles"):
        kwargs["transferable_roles"] = tuple(
            str(value) for value in family["transferable_roles"]
        )
    adapter = adapter_for_family(str(family["family_id"]), provenance, **kwargs)
    expected_kind = family_role_separated_kind(adapter)
    declared_kind = str(_section(dict(config), "features").get("representation_kind", ""))
    if declared_kind != expected_kind:
        raise ValueError(
            f"features.representation_kind {declared_kind!r} does not match adapter "
            f"kind {expected_kind!r}."
        )
    return adapter


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"External family config section {name!r} must be a mapping.")
    return dict(value)
