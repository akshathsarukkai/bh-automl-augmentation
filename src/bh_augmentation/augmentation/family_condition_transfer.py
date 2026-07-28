"""Family-agnostic typed condition transfer.

Typed condition transfer copies the *condition* roles of a donor reaction onto a
source reaction while leaving every substrate role and the product untouched. The
scientific contract is identical to the Buchwald-Hartwig role-aware transfer in
:mod:`bh_augmentation.augmentation.role_aware_condition_transfer`; only the role
vocabulary comes from a :class:`~bh_augmentation.data.reaction_family.ReactionFamilyAdapter`.

Contracts enforced here:

* transfer draws sources and donors from the training partition only;
* only roles in ``adapter.transferable_roles`` may change;
* the canonical substrate identity and canonical product identity of an accepted
  candidate are byte-identical to the source reaction's;
* a candidate whose canonical reaction key already exists in the measured data is
  rejected as ``already_measured`` -- chemical identity is decided by canonical
  reaction keys, never by fingerprint or feature-vector equality;
* candidates that duplicate an earlier accepted candidate are rejected;
* family-specific eligibility failures are recorded with their reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.reaction_family import ReactionFamilyAdapter

FAMILY_CONDITION_TRANSFER_SCHEMA_VERSION = "family-condition-transfer-v1"

CANDIDATE_AUDIT_COLUMNS = (
    "source_row_id",
    "donor_row_id",
    "source_position",
    "donor_position",
    "canonical_reaction_key",
    "canonical_reaction_hash",
    "canonical_substrate_key",
    "canonical_product_key",
    "source_substrate_key",
    "source_product_key",
    "changed_roles",
    "transferable_roles",
    "chemical_parse_valid",
    "family_eligibility_reason",
    "already_measured",
    "duplicate_synthetic",
    "rejection_reason",
    "accepted",
)


def build_condition_transfer_candidates(
    training: pd.DataFrame,
    adapter: ReactionFamilyAdapter,
    *,
    seed: int = 0,
    max_candidates_per_source: int = 2,
    transferable_roles: Sequence[str] | None = None,
    measured_canonical_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Generate typed condition-transfer candidates from training rows only.

    ``training`` must be an already-canonicalized frame (as produced by
    :meth:`ReactionFamilyAdapter.canonicalize_dataframe`) restricted to the
    training partition. If it carries an ``outer_split`` column, every value must
    be ``"train"``; this is a hard guard against validation or test rows entering
    augmentation.
    """
    roles = tuple(transferable_roles or adapter.transferable_roles)
    unknown = sorted(set(roles) - set(adapter.transferable_roles))
    if not roles or unknown:
        raise ValueError(
            "transferable_roles must be a non-empty subset of "
            f"{list(adapter.transferable_roles)}; unknown={unknown}."
        )
    if "outer_split" in training.columns:
        offending = sorted(set(training["outer_split"].astype(str)) - {"train"})
        if offending:
            raise ValueError(
                "Condition transfer may only read training rows; observed outer_split "
                f"values {offending}."
            )
    if int(max_candidates_per_source) < 1:
        raise ValueError("max_candidates_per_source must be at least 1.")

    rows = training.to_dict(orient="records")
    if len(rows) < 2:
        raise ValueError("Condition transfer requires at least two training reactions.")
    measured_keys = (
        {str(value) for value in measured_canonical_keys}
        if measured_canonical_keys is not None
        else adapter.measured_canonical_keys(training)
    )

    rng = np.random.default_rng(int(seed))
    generated_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    for source_position in range(len(rows)):
        donor_order = [
            position
            for position in rng.permutation(len(rows)).tolist()
            if position != source_position
        ]
        for donor_position in donor_order[: int(max_candidates_per_source)]:
            records.append(
                _candidate_record(
                    adapter,
                    rows=rows,
                    source_position=source_position,
                    donor_position=donor_position,
                    roles=roles,
                    measured_keys=measured_keys,
                    generated_keys=generated_keys,
                )
            )
    frame = pd.DataFrame(records)
    frame["transfer_schema_version"] = FAMILY_CONDITION_TRANSFER_SCHEMA_VERSION
    frame["reaction_family_id"] = adapter.family_id
    return frame


def assert_condition_transfer_invariants(
    candidates: pd.DataFrame,
    adapter: ReactionFamilyAdapter,
) -> None:
    """Hard-fail if any accepted candidate violates the typed-transfer contract."""
    missing = sorted(set(CANDIDATE_AUDIT_COLUMNS) - set(candidates.columns))
    if missing:
        raise ValueError(f"Condition-transfer audit is missing columns: {missing}.")
    accepted = candidates.loc[candidates["accepted"].astype(bool)]
    if accepted.empty:
        return
    if not accepted["chemical_parse_valid"].astype(bool).all():
        raise AssertionError("An accepted candidate has invalid chemistry.")
    if accepted["family_eligibility_reason"].notna().any():
        raise AssertionError("An accepted candidate failed the family eligibility rule.")
    if not accepted["canonical_substrate_key"].eq(accepted["source_substrate_key"]).all():
        raise AssertionError("An accepted candidate changed its substrate identity.")
    if not accepted["canonical_product_key"].eq(accepted["source_product_key"]).all():
        raise AssertionError("An accepted candidate changed its product identity.")
    if accepted["already_measured"].astype(bool).any():
        raise AssertionError("An accepted candidate duplicates measured chemistry.")
    if accepted["canonical_reaction_key"].duplicated().any():
        raise AssertionError("An accepted canonical reaction key appears more than once.")
    allowed = set(adapter.transferable_roles)
    for value in accepted["changed_roles"].fillna("").astype(str):
        changed = {role for role in value.split("|") if role}
        if not changed:
            raise AssertionError("An accepted candidate changed no condition role.")
        outside = sorted(changed - allowed)
        if outside:
            raise AssertionError(
                f"An accepted candidate changed non-transferable roles: {outside}."
            )


def _candidate_record(
    adapter: ReactionFamilyAdapter,
    *,
    rows: Sequence[dict[str, Any]],
    source_position: int,
    donor_position: int,
    roles: Sequence[str],
    measured_keys: set[str],
    generated_keys: set[str],
) -> dict[str, Any]:
    source_row = rows[source_position]
    donor_row = rows[donor_position]
    source_values = adapter.roles_from_row(source_row)
    donor_values = adapter.roles_from_row(donor_row)
    candidate_values = dict(source_values)
    for role in roles:
        candidate_values[role] = donor_values[role]

    source_identity = adapter.canonicalize_roles(source_values)
    identity = adapter.canonicalize_roles(candidate_values)
    canonical = identity.canonical_roles or {}
    source_canonical = source_identity.canonical_roles or {}
    changed = (
        [
            role
            for role in adapter.role_names
            if canonical.get(role) != source_canonical.get(role)
        ]
        if identity.parse_valid and source_identity.parse_valid
        else []
    )
    substrate_key = adapter.substrate_key(canonical) if canonical else None
    product_key = adapter.product_key(canonical) if canonical else None
    source_substrate_key = (
        adapter.substrate_key(source_canonical) if source_canonical else None
    )
    source_product_key = adapter.product_key(source_canonical) if source_canonical else None

    key = identity.reaction_key
    already_measured = bool(key is not None and key in measured_keys)
    duplicate_synthetic = bool(key is not None and key in generated_keys)
    rejection = _rejection_reason(
        identity=identity,
        changed=changed,
        allowed_roles=set(roles),
        substrate_key=substrate_key,
        product_key=product_key,
        source_substrate_key=source_substrate_key,
        source_product_key=source_product_key,
        already_measured=already_measured,
        duplicate_synthetic=duplicate_synthetic,
    )
    if rejection is None and key is not None:
        generated_keys.add(key)

    record: dict[str, Any] = {
        "source_row_id": source_row.get("source_row_id", str(source_position)),
        "donor_row_id": donor_row.get("source_row_id", str(donor_position)),
        "source_position": int(source_position),
        "donor_position": int(donor_position),
        "canonical_reaction_key": key,
        "canonical_reaction_hash": identity.reaction_hash,
        "canonical_substrate_key": substrate_key,
        "canonical_product_key": product_key,
        "source_substrate_key": source_substrate_key,
        "source_product_key": source_product_key,
        "changed_roles": "|".join(changed),
        "transferable_roles": "|".join(roles),
        "chemical_parse_valid": bool(identity.parse_valid),
        "family_eligibility_reason": identity.eligibility_reason
        if identity.eligibility_reason != "chemical_parse_invalid"
        else None,
        "already_measured": already_measured,
        "duplicate_synthetic": duplicate_synthetic,
        "rejection_reason": rejection,
        "accepted": rejection is None,
        "canonical_reaction_smiles": (
            adapter.reaction_smiles(canonical) if canonical else None
        ),
    }
    for role in adapter.role_names:
        record[adapter.role_to_column[role]] = candidate_values[role]
        record[f"canonical_{role}_smiles"] = canonical.get(role)
    return record


def _rejection_reason(
    *,
    identity: Any,
    changed: Sequence[str],
    allowed_roles: set[str],
    substrate_key: str | None,
    product_key: str | None,
    source_substrate_key: str | None,
    source_product_key: str | None,
    already_measured: bool,
    duplicate_synthetic: bool,
) -> str | None:
    if not identity.parse_valid:
        return "chemical_parse_invalid"
    if identity.eligibility_reason is not None:
        return f"family_ineligible:{identity.eligibility_reason}"
    if substrate_key != source_substrate_key:
        return "substrate_identity_changed"
    if product_key != source_product_key:
        return "product_identity_changed"
    outside = sorted(set(changed) - allowed_roles)
    if outside:
        return "non_transferable_role_changed"
    if not changed:
        return "no_condition_role_changed"
    if already_measured:
        return "already_measured"
    if duplicate_synthetic:
        return "duplicate_synthetic"
    return None
