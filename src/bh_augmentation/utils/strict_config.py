"""Shared strict configuration validation for scientific runners.

Several runners historically hand-rolled the same exact-key resolution helpers.
This module consolidates that logic so every scientific runner rejects unknown
fields, malformed hashes, unsupported feature/reconstruction combinations,
invalid weights, invalid dimensions, and output-directory reuse with the same
explicit messages.

All errors derive from :class:`ValueError` so existing callers and tests that
expect ``ValueError`` keep working.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from bh_augmentation.features.featurize import (
    DEPRECATED_FEATURE_ALIASES,
    REACTION_FEATURE_KINDS,
    REMOVED_FEATURE_KINDS,
)
from bh_augmentation.results.status import assert_result_directory_allowed

STRICT_CONFIG_SCHEMA_VERSION = "bh-strict-config-v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

#: Reconstruction vocabulary, declared locally so configuration validation does
#: not depend on any single model implementation module being importable.
#: :func:`assert_reconstruction_vocabulary_matches_contract` proves this table
#: stays identical to the matrix-level contract whenever that module is present.
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
DECODER_MODES = ("identity", "bernoulli_logits", "nonnegative_softplus")
DOMAIN_OBJECTIVES: dict[str, tuple[str, ...]] = {
    "binary_bit": ("binary_cross_entropy", "mse", "positive_bit_weighted_mse"),
    "one_hot": ("binary_cross_entropy", "mse", "positive_bit_weighted_mse"),
    "nonnegative_integer_count": ("count_aware_mse", "mse"),
    "signed_count_or_delta": ("mse",),
    "continuous": ("mse",),
}
OBJECTIVE_DECODER: dict[str, str] = {
    "mse": "identity",
    "positive_bit_weighted_mse": "identity",
    "binary_cross_entropy": "bernoulli_logits",
    "count_aware_mse": "nonnegative_softplus",
}

#: Value domain implied by each supported canonical feature kind.  Morgan bit
#: fingerprints are binary; ``*_delta`` representations contain signed
#: differences between fingerprint blocks and are therefore not binary.
FEATURE_KIND_VALUE_DOMAINS: dict[str, str] = {
    kind: ("signed_count_or_delta" if kind.endswith("_delta") else "binary_bit")
    for kind in sorted(REACTION_FEATURE_KINDS)
}


class StrictConfigError(ValueError):
    """Raised when a scientific configuration violates the strict contract."""


class OutputDirectoryReuseError(StrictConfigError, FileExistsError):
    """Raised when a runner would write into an existing output directory.

    It is both a :class:`ValueError` (strict-config rejection) and a
    :class:`FileExistsError`, matching the pre-existing repository convention
    for refusing to overwrite immutable scientific outputs.
    """


def load_strict_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a YAML path or mapping and require a top-level mapping."""
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    if not path.is_file():
        raise StrictConfigError(f"Config file does not exist: {path}")
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise StrictConfigError(f"Invalid YAML config at {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise StrictConfigError(f"Config file must contain a YAML mapping: {path}")
    return dict(loaded)


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping section and reject scalars and sequences."""
    if not isinstance(value, Mapping):
        raise StrictConfigError(f"{name} must be a mapping.")
    return value


def require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    name: str,
) -> Mapping[str, Any]:
    """Reject unknown, misspelled, and missing configuration fields."""
    expected_keys = set(expected)
    observed = require_mapping(value, name)
    observed_keys = set(observed)
    if observed_keys == expected_keys:
        return observed
    unknown = sorted(observed_keys - expected_keys)
    missing = sorted(expected_keys - observed_keys)
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(missing or expected_keys), n=1)
        if close:
            hints.append(f"{key!r} -> {close[0]!r}")
    message = (
        f"{name} keys mismatch: expected={sorted(expected_keys)}, "
        f"observed={sorted(observed_keys)}, unknown={unknown}, missing={missing}."
    )
    if hints:
        message += f" Did you mean: {', '.join(hints)}?"
    raise StrictConfigError(message)


def require_choice(value: Any, choices: Iterable[Any], name: str) -> Any:
    """Require membership in a declared choice set."""
    allowed = list(choices)
    if value not in allowed:
        raise StrictConfigError(
            f"{name} must be one of {sorted(map(str, allowed))}; got {value!r}."
        )
    return value


def require_text(value: Any, name: str) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise StrictConfigError(f"{name} must be a non-empty string.")
    return value


def require_sha256(value: Any, name: str) -> str:
    """Require a well-formed lowercase SHA-256 hexadecimal digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise StrictConfigError(
            f"{name} must be a lowercase 64-character SHA-256 digest; got {value!r}."
        )
    return value


def require_hash_matches(observed: Any, expected: Any, name: str) -> str:
    """Require two independently derived hashes to agree exactly."""
    observed_hash = require_sha256(observed, f"{name} (observed)")
    expected_hash = require_sha256(expected, f"{name} (declared)")
    if observed_hash != expected_hash:
        raise StrictConfigError(
            f"{name} is inconsistent: declared={expected_hash}, recomputed={observed_hash}."
        )
    return observed_hash


def require_int(value: Any, name: str) -> int:
    """Require a true integer and reject booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrictConfigError(f"{name} must be an integer; got {value!r}.")
    return int(value)


def require_positive_int(value: Any, name: str) -> int:
    """Require a strictly positive integer."""
    result = require_int(value, name)
    if result < 1:
        raise StrictConfigError(f"{name} must be positive; got {result}.")
    return result


def require_finite_float(value: Any, name: str) -> float:
    """Require a finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictConfigError(f"{name} must be a finite number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise StrictConfigError(f"{name} must be finite; got {value!r}.")
    return result


def require_positive_float(value: Any, name: str) -> float:
    """Require a finite, strictly positive float."""
    result = require_finite_float(value, name)
    if result <= 0.0:
        raise StrictConfigError(f"{name} must be positive; got {result}.")
    return result


def require_nonnegative_float(value: Any, name: str) -> float:
    """Require a finite, non-negative float."""
    result = require_finite_float(value, name)
    if result < 0.0:
        raise StrictConfigError(f"{name} must be non-negative; got {result}.")
    return result


def require_open_unit(value: Any, name: str) -> float:
    """Require a fraction strictly inside (0, 1)."""
    result = require_finite_float(value, name)
    if not 0.0 < result < 1.0:
        raise StrictConfigError(f"{name} must lie in (0, 1); got {result}.")
    return result


def require_weight(
    value: Any,
    name: str,
    *,
    choices: Sequence[float] | None = None,
    allow_zero: bool = True,
) -> float:
    """Require a finite, non-negative loss/objective weight.

    ``choices`` declares the only permitted weight values for a frozen policy
    grid; any other value is rejected instead of silently accepted.
    """
    result = require_finite_float(value, name)
    if result < 0.0:
        raise StrictConfigError(f"{name} must be non-negative; got {result}.")
    if not allow_zero and result == 0.0:
        raise StrictConfigError(f"{name} must be strictly positive; got {result}.")
    if choices is not None:
        allowed = [float(item) for item in choices]
        if not any(math.isclose(result, item, rel_tol=0.0, abs_tol=1e-12) for item in allowed):
            raise StrictConfigError(
                f"{name} must be one of the declared weights {allowed}; got {result}."
            )
    return result


def require_unique_ints(value: Any, name: str) -> tuple[int, ...]:
    """Require a non-empty list of unique integers."""
    if not isinstance(value, list) or not value:
        raise StrictConfigError(f"{name} must be a non-empty list.")
    result = tuple(require_int(item, name) for item in value)
    if len(result) != len(set(result)):
        raise StrictConfigError(f"{name} must contain unique values; got {list(result)}.")
    return result


def require_unique_fractions(value: Any, name: str) -> tuple[float, ...]:
    """Require a non-empty list of unique fractions in (0, 1]."""
    if not isinstance(value, list) or not value:
        raise StrictConfigError(f"{name} must be a non-empty list.")
    result = tuple(require_finite_float(item, name) for item in value)
    if len(result) != len(set(result)):
        raise StrictConfigError(f"{name} must contain unique values; got {list(result)}.")
    if any(not 0.0 < item <= 1.0 for item in result):
        raise StrictConfigError(f"{name} must contain values in (0, 1]; got {list(result)}.")
    return result


def require_dimensions(
    *,
    input_width: Any,
    latent_width: Any,
    name: str = "dimensions",
) -> tuple[int, int]:
    """Require positive integer dimensions with ``latent <= input``."""
    resolved_input = require_positive_int(input_width, f"{name}.input_width")
    resolved_latent = require_positive_int(latent_width, f"{name}.latent_width")
    if resolved_latent > resolved_input:
        raise StrictConfigError(
            f"{name}.latent_width must not exceed {name}.input_width; "
            f"latent={resolved_latent}, input={resolved_input}."
        )
    return resolved_input, resolved_latent


def require_feature_kind(value: Any, name: str = "features.kind") -> str:
    """Require an explicit, supported, non-deprecated canonical feature kind."""
    kind = require_text(value, name).strip()
    if kind in DEPRECATED_FEATURE_ALIASES:
        raise StrictConfigError(
            f"{name} may not use legacy alias {kind!r}; "
            f"use {DEPRECATED_FEATURE_ALIASES[kind]!r}."
        )
    if kind in REMOVED_FEATURE_KINDS:
        raise StrictConfigError(f"{name} uses removed feature kind {kind!r}.")
    if kind not in REACTION_FEATURE_KINDS:
        close = difflib.get_close_matches(kind, sorted(REACTION_FEATURE_KINDS), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise StrictConfigError(f"{name} uses unsupported feature kind {kind!r}.{hint}")
    return kind


def require_reconstruction_objective(
    value: Any,
    name: str = "reconstruction.objective",
) -> str:
    """Require a supported reconstruction objective."""
    objective = require_text(value, name)
    if objective not in RECONSTRUCTION_OBJECTIVES:
        raise StrictConfigError(
            f"{name} must be one of {sorted(RECONSTRUCTION_OBJECTIVES)}; got {objective!r}."
        )
    return objective


def require_feature_reconstruction_combination(
    feature_kind: Any,
    objective: Any,
    *,
    decoder_mode: Any | None = None,
    name: str = "reconstruction",
) -> dict[str, str]:
    """Reject unsupported feature-kind / reconstruction-objective combinations.

    The eligibility table is the same one enforced at matrix level by
    :mod:`bh_augmentation.features.reconstruction_contract`; this performs the
    check at configuration-resolution time, before any data is loaded.
    """
    kind = require_feature_kind(feature_kind)
    resolved_objective = require_reconstruction_objective(objective, f"{name}.objective")
    domain = FEATURE_KIND_VALUE_DOMAINS[kind]
    if domain not in VALUE_DOMAINS:  # pragma: no cover - defensive
        raise StrictConfigError(f"Unsupported value domain {domain!r} for feature kind {kind!r}.")
    eligible = DOMAIN_OBJECTIVES[domain]
    if resolved_objective not in eligible:
        raise StrictConfigError(
            f"{name}.objective {resolved_objective!r} is unsupported for features.kind "
            f"{kind!r} (value domain {domain!r}); eligible objectives are {sorted(eligible)}."
        )
    expected_decoder = OBJECTIVE_DECODER[resolved_objective]
    if decoder_mode is not None:
        selected = require_text(decoder_mode, f"{name}.decoder_mode")
        if selected not in DECODER_MODES:
            raise StrictConfigError(
                f"{name}.decoder_mode must be one of {sorted(DECODER_MODES)}; got {selected!r}."
            )
        if selected != expected_decoder:
            raise StrictConfigError(
                f"{name}.objective {resolved_objective!r} requires decoder mode "
                f"{expected_decoder!r}; got {selected!r}."
            )
    return {
        "feature_kind": kind,
        "value_domain": domain,
        "objective": resolved_objective,
        "decoder_mode": expected_decoder,
    }


def require_fresh_output_directory(
    path: str | Path,
    *,
    name: str = "output.directory",
    allow_invalid: bool = False,
) -> Path:
    """Reject output-directory reuse and known-invalid result families.

    The directory is not created; runners keep control of when they materialize
    outputs.  Reuse is refused whether the target exists as a directory, an
    empty directory, or a file.
    """
    directory = Path(path)
    if not str(directory).strip():
        raise StrictConfigError(f"{name} must be a non-empty path.")
    assert_result_directory_allowed(directory, allow_invalid=allow_invalid)
    if directory.exists():
        raise OutputDirectoryReuseError(
            f"Refusing to reuse or overwrite an existing {name}: {directory}. "
            "Scientific outputs are immutable; choose a fresh path."
        )
    return directory


def assert_reconstruction_vocabulary_matches_contract() -> None:
    """Prove the local reconstruction tables match the matrix-level contract.

    Configuration validation deliberately keeps its own copy of the
    reconstruction vocabulary so it never depends on a model implementation
    module being importable.  When
    :mod:`bh_augmentation.features.reconstruction_contract` is present, this
    guard fails loudly if the two definitions ever drift apart.
    """
    try:
        from bh_augmentation.features import reconstruction_contract
    except ImportError as exc:  # pragma: no cover - only in partial checkouts
        raise StrictConfigError(
            "The reconstruction feature contract module is not available in this "
            "checkout; reconstruction vocabulary cannot be cross-checked."
        ) from exc
    mismatches = []
    if tuple(reconstruction_contract.VALUE_DOMAINS) != VALUE_DOMAINS:
        mismatches.append("VALUE_DOMAINS")
    if tuple(reconstruction_contract.RECONSTRUCTION_OBJECTIVES) != RECONSTRUCTION_OBJECTIVES:
        mismatches.append("RECONSTRUCTION_OBJECTIVES")
    if tuple(reconstruction_contract.DECODER_MODES) != DECODER_MODES:
        mismatches.append("DECODER_MODES")
    if dict(reconstruction_contract._DOMAIN_OBJECTIVES) != DOMAIN_OBJECTIVES:
        mismatches.append("DOMAIN_OBJECTIVES")
    if dict(reconstruction_contract._OBJECTIVE_DECODER) != OBJECTIVE_DECODER:
        mismatches.append("OBJECTIVE_DECODER")
    if mismatches:
        raise StrictConfigError(
            "Strict-config reconstruction vocabulary has drifted from "
            f"bh_augmentation.features.reconstruction_contract: {sorted(mismatches)}."
        )
