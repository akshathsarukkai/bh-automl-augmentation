"""Tests for deterministic, identity-preserving RDKit canonicalization."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest
from tests.canonical_test_utils import measured_role_frame

from bh_augmentation.data.canonicalize_roles import (
    canonicalize_reaction_roles_dataframe,
    canonicalize_smiles,
)


def test_equivalent_smiles_and_whitespace_canonicalize_identically() -> None:
    first = canonicalize_smiles(" C(C)O ")
    second = canonicalize_smiles("CCO")
    assert first.canonical_smiles == second.canonical_smiles == "CCO"
    assert first.raw_smiles == " C(C)O "


def test_stereoisomers_remain_distinct() -> None:
    first = canonicalize_smiles("F[C@H](Cl)Br")
    second = canonicalize_smiles("F[C@@H](Cl)Br")
    assert first.parse_valid and second.parse_valid
    assert first.canonical_smiles != second.canonical_smiles


def test_charged_and_neutral_forms_remain_distinct() -> None:
    charged = canonicalize_smiles("[NH4+]")
    neutral = canonicalize_smiles("N")
    assert charged.canonical_smiles != neutral.canonical_smiles
    assert "+" in str(charged.canonical_smiles)


def test_invalid_smiles_is_flagged_without_empty_identity() -> None:
    result = canonicalize_smiles("C1(")
    assert not result.parse_valid
    assert result.canonical_smiles is None
    assert result.error_type == "rdkit_parse_failed"
    assert result.raw_smiles == "C1("


def test_canonicalization_is_deterministic() -> None:
    results = [canonicalize_smiles("[13CH3][C@H](O)[NH3+]") for _ in range(3)]
    assert len({result.canonical_smiles for result in results}) == 1
    assert "[13CH3]" in str(results[0].canonical_smiles)


def test_rdkit_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="RDKit is required"):
        canonicalize_smiles("CCO")


def test_audit_mode_preserves_and_flags_unknown_required_role() -> None:
    frame = measured_role_frame().iloc[:1].copy()
    frame["recovered_base_smiles"] = "UNKNOWN"
    result = canonicalize_reaction_roles_dataframe(frame, source_file_hash="f" * 64)
    assert len(result) == 1
    assert not bool(result.loc[0, "base_parse_valid"])
    assert result.loc[0, "raw_base_smiles"] == "UNKNOWN"
    assert pd.isna(result.loc[0, "canonical_reaction_key"])
