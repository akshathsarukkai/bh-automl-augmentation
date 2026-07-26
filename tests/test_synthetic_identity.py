"""Regression tests for canonical synthetic-candidate identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    assert_accepted_identity_invariants,
    audit_candidate_identities,
    canonical_candidate_record,
    canonicalize_synthetic_roles,
    configured_feature_hash,
    measured_canonical_keys,
)
from bh_augmentation.data.reaction_roles import (
    ReactionRoles,
    reaction_roles_to_record,
)


def test_feature_hash_is_exact_and_deterministic() -> None:
    vector = np.array([0.0, 1.0, 2.5], dtype=np.float64)

    assert configured_feature_hash(vector) == configured_feature_hash(
        vector.astype(np.float32)
    )
    assert configured_feature_hash(vector) != configured_feature_hash(
        np.array([0.0, 1.0, 2.5001])
    )
    assert configured_feature_hash(np.array([0.0, np.nan])) is None


def test_canonicalization_preserves_isotope_charge_and_stereochemistry() -> None:
    roles = _roles(
        reactant_1="[13CH3][C@H](F)Cl",
        reactant_2="[NH3+]",
    )

    identity = canonicalize_synthetic_roles(roles)

    assert identity.chemical_parse_valid
    assert identity.roles is not None
    assert "13C" in identity.roles.reactant_1
    assert "@" in identity.roles.reactant_1
    assert "+" in identity.roles.reactant_2


def test_audit_distinguishes_chemical_duplicates_from_feature_collisions() -> None:
    source = _roles()
    donor = _roles(ligand="P(CC)(CC)CC", base="N1CCCCC1")
    measured_other = _roles(ligand="P(C)(C)C", base="N1CCCCC1")
    accepted = _roles(ligand="P(CC)(CC)CC", base="N(C)(C)C")
    feature_collision = _roles(
        ligand="P(C)(C)C",
        base="N(C)(C)C",
        solvent_or_additive="CCCO",
    )
    invalid = _roles(ligand="not_a_smiles")
    source_rows = [_row(source, "source"), _row(donor, "donor")]
    measured = pd.DataFrame(
        [_row(source, "source"), _row(donor, "donor"), _row(measured_other, "measured")]
    )
    candidates = pd.DataFrame(
        [
            _candidate(source),
            _candidate(measured_other),
            _candidate(accepted),
            _candidate(accepted),
            _candidate(feature_collision),
            _candidate(invalid),
        ]
    )
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )

    audited = audit_candidate_identities(
        candidates,
        features,
        measured_keys=measured_canonical_keys(measured),
        source_rows=source_rows,
    )

    assert set(REQUIRED_SYNTHETIC_AUDIT_FIELDS) <= set(audited.columns)
    assert audited["rejection_reason"].tolist() == [
        "source_identical",
        "already_measured",
        None,
        "duplicate_synthetic",
        "feature_duplicate_synthetic",
        "chemical_parse_invalid",
    ]
    assert not bool(audited.loc[4, "duplicate_synthetic"])
    assert bool(audited.loc[4, "feature_duplicate_synthetic"])
    audited["accepted"] = audited["rejection_reason"].isna()
    assert_accepted_identity_invariants(audited)


def test_candidate_audit_regeneration_has_identical_hashes() -> None:
    source = _roles()
    donor = _roles(ligand="P(CC)(CC)CC", base="N1CCCCC1")
    candidate = _roles(ligand="P(CC)(CC)CC", base="N(C)(C)C")
    source_rows = [_row(source, "source"), _row(donor, "donor")]
    frame = pd.DataFrame([_candidate(candidate)])
    features = np.array([[0.0, 1.0, 3.0]], dtype=np.float32)

    first = audit_candidate_identities(
        frame,
        features,
        measured_keys=measured_canonical_keys(pd.DataFrame(source_rows)),
        source_rows=source_rows,
    )
    second = audit_candidate_identities(
        frame,
        features.copy(),
        measured_keys=measured_canonical_keys(pd.DataFrame(source_rows)),
        source_rows=source_rows,
    )

    columns = ["canonical_reaction_key", "canonical_reaction_hash", "feature_hash"]
    assert first[columns].to_dict(orient="records") == second[columns].to_dict(
        orient="records"
    )


def _roles(**changes: str) -> ReactionRoles:
    values = {
        "reactant_1": "CCBr",
        "reactant_2": "N",
        "catalyst": "[Pd]",
        "ligand": "P(C)(C)C",
        "base": "N(C)(C)C",
        "solvent_or_additive": "CCO",
        "product": "CCN",
    }
    values.update(changes)
    return ReactionRoles(**values)


def _row(roles: ReactionRoles, row_id: str) -> dict[str, object]:
    return {
        "source_row_id": row_id,
        **reaction_roles_to_record(roles),
        "yield": 50.0,
    }


def _candidate(roles: ReactionRoles) -> dict[str, object]:
    return {
        "candidate_id": 0,
        "source_position": 0,
        "donor_position": 1,
        "source_index": 0,
        "donor_index": 1,
        **canonical_candidate_record(roles),
    }
