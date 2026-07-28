"""Config validation for external reaction-family datasets.

A Suzuki config that omits any provenance or licensing field must be rejected,
and the shipped production config must be structurally valid while pointing at a
dataset that does not exist.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from bh_augmentation.data.external_dataset_config import (
    build_adapter_from_config,
    validate_external_family_config,
)
from bh_augmentation.data.reaction_family import PROVENANCE_FIELDS
from bh_augmentation.run_external_family_validation import run_external_family_validation
from suzuki_fixture import suzuki_fixture_config, write_suzuki_fixture_csv

PRODUCTION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "suzuki_external_validation.yaml"
)


def _fixture_config(tmp_path: Path) -> dict:
    dataset_path = write_suzuki_fixture_csv(tmp_path)
    return suzuki_fixture_config(dataset_path, tmp_path / "out", n_bits=32)


# ----------------------------------------------------------------------
# Provenance is mandatory
# ----------------------------------------------------------------------
@pytest.mark.parametrize("field", PROVENANCE_FIELDS)
def test_config_missing_any_provenance_field_is_rejected(tmp_path, field: str) -> None:
    config = _fixture_config(tmp_path)
    config["reaction_family"]["provenance"].pop(field)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_external_family_config(config)


def test_config_with_blank_license_is_rejected(tmp_path) -> None:
    config = _fixture_config(tmp_path)
    config["reaction_family"]["provenance"]["license"] = "   "
    with pytest.raises(ValueError, match="license"):
        validate_external_family_config(config)


def test_config_without_a_provenance_block_is_rejected(tmp_path) -> None:
    config = _fixture_config(tmp_path)
    config["reaction_family"].pop("provenance")
    with pytest.raises(ValueError, match="require reaction_family.provenance"):
        validate_external_family_config(config)


def test_config_with_an_unknown_provenance_field_is_rejected(tmp_path) -> None:
    config = _fixture_config(tmp_path)
    config["reaction_family"]["provenance"]["maybe_ok_to_use"] = "yes"
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_external_family_config(config)


# ----------------------------------------------------------------------
# Scientific contracts
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update({"scientific_run": False}), "scientific_run"),
        (
            lambda c: c.update({"allow_hash_fingerprint_fallback": True}),
            "allow_hash_fingerprint_fallback",
        ),
        (lambda c: c["canonicalization"].update({"backend": "hash"}), "backend: rdkit"),
        (
            lambda c: c["canonicalization"].update({"isomeric_smiles": False}),
            "isomeric_smiles",
        ),
        (
            lambda c: c["canonicalization"].update({"preserve_role_order": False}),
            "preserve_role_order",
        ),
        (lambda c: c["splits"].update({"group_column": "reaction_id"}), "group_column"),
        (lambda c: c["splits"].update({"test_size": 0.5}), "sum to 1.0"),
        (
            lambda c: c["features"].update({"fingerprint_backend": "hash"}),
            "fingerprint_backend",
        ),
        (
            lambda c: c["features"].update(
                {"representation_kind": "bh_role_separated"}
            ),
            "features.representation_kind",
        ),
        (lambda c: c["features"].update({"kind": "bh_role_separated"}), "reserved for the"),
        (lambda c: c["reaction_family"].update({"family_id": "heck"}), "Unknown reaction family"),
        (
            lambda c: c["condition_transfer"].update({"synthetic_multipliers": []}),
            "synthetic_multipliers",
        ),
    ],
)
def test_broken_scientific_contracts_are_rejected(tmp_path, mutate, message: str) -> None:
    config = _fixture_config(tmp_path)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_external_family_config(config)


def test_valid_fixture_config_builds_the_suzuki_adapter(tmp_path) -> None:
    config = _fixture_config(tmp_path)
    assert validate_external_family_config(config) == config
    adapter = build_adapter_from_config(config)
    assert adapter.family_id == "suzuki_miyaura"
    assert adapter.transferable_roles == ("ligand", "base", "solvent_or_additive")
    assert adapter.provenance.raw_file_sha256 == (
        config["reaction_family"]["provenance"]["raw_file_sha256"]
    )


# ----------------------------------------------------------------------
# The shipped production config
# ----------------------------------------------------------------------
def test_production_config_is_structurally_valid_but_has_no_dataset() -> None:
    config = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    validate_external_family_config(config)
    adapter = build_adapter_from_config(config)
    assert adapter.family_id == "suzuki_miyaura"
    assert not Path(config["dataset"]["path"]).exists()


def test_production_config_refuses_to_run_without_the_dataset(tmp_path) -> None:
    config = copy.deepcopy(yaml.safe_load(PRODUCTION_CONFIG.read_text()))
    config["output"]["directory"] = str(tmp_path / "never_created")
    with pytest.raises(FileNotFoundError, match="EXTERNAL_DATASETS.md"):
        run_external_family_validation(config, config_path=PRODUCTION_CONFIG)
    assert not (tmp_path / "never_created").exists()


def test_documentation_records_the_blocked_status() -> None:
    doc = Path(__file__).resolve().parents[1] / "docs" / "EXTERNAL_DATASETS.md"
    text = doc.read_text()
    assert "implementation passed" in text
    assert "external empirical validation blocked" in text
    assert "no licensed Suzuki data present locally" in text
