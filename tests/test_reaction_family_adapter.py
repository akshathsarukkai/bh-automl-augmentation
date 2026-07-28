"""Tests for the reaction-family adapter seam.

The Buchwald-Hartwig adapter must be definitionally identical to the existing BH
schema, and no adapter may be constructed without complete dataset provenance.
"""

from __future__ import annotations

import pytest

from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    PARTIAL_KEY_SCHEMA_VERSION,
    REACTION_KEY_SCHEMA_VERSION,
    build_canonical_reaction_identity,
    canonicalize_reaction_roles_dataframe,
)
from bh_augmentation.data.reaction_family import (
    BUCHWALD_HARTWIG_ADAPTER,
    PROVENANCE_FIELDS,
    DatasetProvenance,
    ReactionFamilyAdapter,
    adapter_for_family,
    available_reaction_families,
    buchwald_hartwig_adapter,
    is_missing_molecule,
    no_family_eligibility_rule,
    suzuki_miyaura_adapter,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    CANONICAL_ROLE_NAMES,
    ROLE_TO_COLUMN,
    ReactionRoles,
    _missing_required_molecule,
)
from corrected_test_utils import corrected_dataset


def _provenance(**changes: str) -> DatasetProvenance:
    values = {
        "source_name": "fixture source",
        "source_url": "local://fixture",
        "license": "not applicable: fixture",
        "citation": "no citation: fixture",
        "retrieved_date": "not-applicable",
        "raw_file_sha256": "a" * 64,
    }
    values.update(changes)
    return DatasetProvenance(**values)


# ----------------------------------------------------------------------
# Buchwald-Hartwig adapter is the existing schema
# ----------------------------------------------------------------------
def test_bh_adapter_roles_columns_and_assembly_order_match_existing_constants() -> None:
    adapter = BUCHWALD_HARTWIG_ADAPTER
    assert adapter.role_names == CANONICAL_ROLE_NAMES
    assert adapter.role_columns == CANONICAL_ROLE_COLUMNS
    assert adapter.role_to_column == ROLE_TO_COLUMN
    assert adapter.canonicalization_version == CANONICALIZATION_VERSION
    assert adapter.reaction_key_schema_version == REACTION_KEY_SCHEMA_VERSION
    assert adapter.partial_key_schema_version == PARTIAL_KEY_SCHEMA_VERSION

    roles = ReactionRoles(
        reactant_1="R1",
        reactant_2="R2",
        catalyst="CAT",
        ligand="LIG",
        base="BASE",
        solvent_or_additive="SOL",
        product="PROD",
    )
    values = {role: getattr(roles, role) for role in CANONICAL_ROLE_NAMES}
    assert adapter.reaction_smiles(values) == roles.reaction_smiles()
    assert adapter.reaction_smiles_role_order == (
        "reactant_1",
        "reactant_2",
        "catalyst",
        "ligand",
        "base",
        "solvent_or_additive",
        "product",
    )


def test_bh_adapter_reproduces_existing_canonical_reaction_identities() -> None:
    adapter = BUCHWALD_HARTWIG_ADAPTER
    source = corrected_dataset(n_rows=12)
    existing = canonicalize_reaction_roles_dataframe(
        source,
        source_file_hash="b" * 64,
        source_row_positions=range(len(source)),
    )
    produced = adapter.canonicalize_dataframe(
        source,
        source_file_hash="b" * 64,
        source_row_positions=range(len(source)),
    )
    assert produced["source_row_id"].tolist() == existing["source_row_id"].tolist()
    assert (
        produced["canonical_reaction_key"].tolist()
        == existing["canonical_reaction_key"].tolist()
    )
    assert (
        produced["canonical_reaction_hash"].tolist()
        == existing["canonical_reaction_hash"].tolist()
    )
    assert (
        produced["canonical_substrate_key"].tolist()
        == existing["canonical_substrate_key"].tolist()
    )
    assert (
        produced["canonical_condition_key"].tolist()
        == existing["canonical_condition_key"].tolist()
    )
    assert (
        produced["canonical_product_key"].tolist()
        == existing["canonical_product_key"].tolist()
    )

    row = existing.to_dict(orient="records")[0]
    identity = build_canonical_reaction_identity(row)
    canonical_roles = {
        role: row[f"canonical_{role}_smiles"] for role in CANONICAL_ROLE_NAMES
    }
    assert adapter.reaction_key(canonical_roles) == identity["key"]


def test_missing_molecule_sentinels_agree_with_the_bh_implementation() -> None:
    for value in ["", " ", "UNKNOWN", "unknown", "NaN", "None", None, "CCO", "  CCO "]:
        assert is_missing_molecule(value) == _missing_required_molecule(value)


# ----------------------------------------------------------------------
# Provenance is required
# ----------------------------------------------------------------------
@pytest.mark.parametrize("field", PROVENANCE_FIELDS)
def test_provenance_rejects_any_missing_field(field: str) -> None:
    values = _provenance().to_dict()
    values.pop("provenance_schema_version")
    values.pop(field)
    with pytest.raises(ValueError, match="schema mismatch"):
        DatasetProvenance.from_mapping(values)
    with pytest.raises(ValueError, match=f"{field}"):
        DatasetProvenance(**{**values, field: "  "})


def test_provenance_requires_a_real_sha256_digest() -> None:
    with pytest.raises(ValueError, match="raw_file_sha256"):
        _provenance(raw_file_sha256="not-a-digest")
    with pytest.raises(ValueError, match="raw_file_sha256"):
        _provenance(raw_file_sha256="A" * 64)


def test_adapter_cannot_be_built_without_provenance() -> None:
    with pytest.raises(TypeError):
        suzuki_miyaura_adapter()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="require a DatasetProvenance"):
        buchwald_hartwig_adapter(provenance={"license": "unknown"})  # type: ignore[arg-type]


def test_bh_provenance_matches_the_committed_raw_export_when_present() -> None:
    import hashlib
    from pathlib import Path

    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "buchwald_hartwig_tdc.csv"
    if not raw.is_file():
        pytest.skip("Raw Buchwald-Hartwig export is not present in this checkout.")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert BUCHWALD_HARTWIG_ADAPTER.provenance.raw_file_sha256 == digest


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------
def _adapter(**changes) -> ReactionFamilyAdapter:
    values = {
        "family_id": "toy",
        "family_name": "Toy family",
        "role_names": ("a", "b", "c", "product"),
        "role_columns": ("a_smiles", "b_smiles", "c_smiles", "product_smiles"),
        "substrate_roles": ("a",),
        "condition_roles": ("b", "c"),
        "product_role": "product",
        "canonicalization_version": "toy-v1",
        "reaction_key_schema_version": "toy-key-v1",
        "partial_key_schema_version": "toy-partial-v1",
        "assembly_order": ("a", "b", "c"),
        "provenance": _provenance(),
        "eligibility_rule": no_family_eligibility_rule,
    }
    values.update(changes)
    return ReactionFamilyAdapter(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"substrate_roles": ("a", "b")}, "both a substrate and a condition"),
        ({"condition_roles": ("b",)}, "substrate, condition, or product"),
        ({"assembly_order": ("a", "b")}, "permutation of every non-product role"),
        ({"product_role": "a"}, "must not also be a substrate"),
        ({"role_columns": ("a_smiles", "a_smiles", "c_smiles", "p")}, "must be unique"),
        ({"transferable_roles": ("a",)}, "subset of condition_roles"),
        ({"role_names": ("a", "a", "c", "product")}, "must be unique"),
    ],
)
def test_adapter_schema_validation(changes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter(**changes)


def test_transferable_roles_default_to_all_condition_roles() -> None:
    assert _adapter().transferable_roles == ("b", "c")


def test_registry_exposes_both_families_and_rejects_unknown_ones() -> None:
    assert available_reaction_families() == ("buchwald_hartwig", "suzuki_miyaura")
    adapter = adapter_for_family("suzuki_miyaura", _provenance())
    assert adapter.family_id == "suzuki_miyaura"
    with pytest.raises(ValueError, match="Unknown reaction family"):
        adapter_for_family("heck", _provenance())


# ----------------------------------------------------------------------
# Suzuki schema is chemically honest
# ----------------------------------------------------------------------
def test_suzuki_roles_do_not_borrow_buchwald_hartwig_terminology() -> None:
    adapter = suzuki_miyaura_adapter(_provenance())
    assert adapter.role_names == (
        "organohalide",
        "organoboron",
        "catalyst",
        "ligand",
        "base",
        "solvent_or_additive",
        "product",
    )
    forbidden = {"reactant_1", "reactant_2", "amine", "aryl_halide"}
    assert not forbidden & set(adapter.role_names)
    assert not forbidden & set(adapter.role_columns)


def test_suzuki_boron_is_a_substrate_and_only_conditions_are_transferable() -> None:
    adapter = suzuki_miyaura_adapter(_provenance())
    assert adapter.substrate_roles == ("organohalide", "organoboron")
    assert "organoboron" not in adapter.condition_roles
    assert "organoboron" not in adapter.transferable_roles
    assert adapter.transferable_roles == ("ligand", "base", "solvent_or_additive")
    assert set(adapter.transferable_roles) <= set(adapter.condition_roles)
    assert adapter.product_role == "product"


def test_suzuki_versions_are_family_scoped_and_distinct_from_buchwald_hartwig() -> None:
    adapter = suzuki_miyaura_adapter(_provenance())
    assert adapter.canonicalization_version == "suzuki-rdkit-isomeric-v1"
    assert adapter.reaction_key_schema_version == "suzuki-seven-role-v1"
    assert adapter.canonicalization_version != CANONICALIZATION_VERSION
    assert adapter.reaction_key_schema_version != REACTION_KEY_SCHEMA_VERSION


def test_adapter_can_be_rebound_to_new_provenance_without_changing_the_schema() -> None:
    adapter = suzuki_miyaura_adapter(_provenance())
    rebound = adapter.with_provenance(_provenance(source_name="other"))
    assert rebound.schema_record() == adapter.schema_record()
    assert rebound.provenance != adapter.provenance
