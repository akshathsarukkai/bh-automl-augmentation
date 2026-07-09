"""Dataset-specific condition role recovery for processed Buchwald-Hartwig CSVs.

The parser in this module assumes the processed Buchwald-Hartwig reaction
strings used by this repository encode six left-side dot-separated tokens:

    reactant_1.reactant_2.catalyst_or_precatalyst.ligand.base.solvent_or_additive>>product

These positional roles are inferred from the current processed dataset format.
They should not be silently generalized to other reaction datasets or raw TDC
exports. Temperature is not encoded in these reaction strings and is therefore
not inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

UNKNOWN = "UNKNOWN"
NOT_RECOVERABLE = "NOT_RECOVERABLE"

PARSED_COLUMNS = [
    "parsed_reactant_1_smiles",
    "parsed_reactant_2_smiles",
    "parsed_catalyst_smiles",
    "parsed_ligand_smiles",
    "parsed_base_smiles",
    "parsed_solvent_or_additive_smiles",
    "parsed_product_smiles",
    "condition_block_smiles",
    "condition_parse_status",
    "condition_parse_error",
]

RECOVERED_COLUMNS = [
    "recovered_reactant_1_smiles",
    "recovered_reactant_2_smiles",
    "recovered_product_smiles",
    "recovered_catalyst_smiles",
    "recovered_ligand_smiles",
    "recovered_base_smiles",
    "recovered_solvent_or_additive_smiles",
    "recovered_temperature",
]

ROLE_VALIDATION_COLUMNS = [
    "catalyst_role_confidence",
    "ligand_role_confidence",
    "base_role_confidence",
    "solvent_or_additive_role_confidence",
    "role_validation_status",
    "role_validation_notes",
    "original_condition_token_2",
    "original_condition_token_3",
    "original_condition_token_4",
    "original_condition_token_5",
    "repaired_condition_order_smiles",
]

TRANSITION_METAL_MARKERS = [
    "Pd",
    "[Pd",
    "palladium",
    "Pt",
    "[Pt",
    "Ni",
    "[Ni",
    "Cu",
    "[Cu",
    "Rh",
    "[Rh",
    "Ir",
    "[Ir",
    "Ru",
    "[Ru",
    "Fe",
    "[Fe",
    "Co",
    "[Co",
]
KNOWN_BASE_STRINGS = {
    "k3po4",
    "k2co3",
    "cs2co3",
    "naotbu",
    "kotbu",
    "lihtmds",
    "lihmds",
    "dbu",
    "dbn",
    "dipea",
    "btmg",
    "tmg",
    "tbd",
    "mtbd",
    "net3",
    "naoh",
    "koh",
    "p1tbu",
    "p2et",
    "p4tbu",
    "t-buona",
    "t-buok",
    "tert-butoxide",
}
KNOWN_LIGAND_STRINGS = {
    "xphos",
    "sphos",
    "ruphos",
    "johnphos",
    "brettphos",
    "binap",
    "dppf",
    "phosphine",
}
KNOWN_SOLVENT_STRINGS = {
    "toluene",
    "dioxane",
    "thf",
    "dmf",
    "dma",
    "dmso",
    "mecn",
    "acetonitrile",
    "water",
    "h2o",
    "ethanol",
    "methanol",
    "propanol",
    "butanol",
}


def classify_bh_component(smiles: str) -> set[str]:
    """Classify a BH component with conservative deterministic heuristics.

    The returned roles are possible assignments, not a proof of identity. The
    heuristics are intentionally transparent and conservative so ambiguous
    condition blocks can be inspected before any downstream model training.
    """
    if not _is_known(smiles):
        return {"unknown"}

    text = _normalize_value(smiles)
    lower = text.lower()
    roles: set[str] = set()

    if _looks_like_transition_metal_component(text):
        return {"catalyst"}

    if _looks_like_base(text):
        roles.add("base")
    elif _looks_like_ligand(text):
        roles.add("ligand")

    if not roles and _looks_like_electrophile(text):
        roles.add("electrophile")
    if not roles and _looks_like_nucleophile(text):
        roles.add("nucleophile")
    if not roles and (
        any(name in lower for name in KNOWN_SOLVENT_STRINGS) or _looks_like_neutral_solvent_or_additive(text)
    ):
        roles.add("solvent_or_additive")

    return roles or {"unknown"}


def validate_condition_roles(parsed: dict) -> dict[str, object]:
    """Validate condition-token role assignments after positional parsing."""
    return repair_condition_roles(parsed, strict=False)


def is_organic_superbase_like(smiles: str) -> bool:
    """Return true for conservative guanidine/amidine/phosphazene base motifs."""
    if not _is_known(smiles):
        return False

    text = _normalize_value(smiles)
    lower = text.lower()
    compact = _compact_text(text)
    if compact in KNOWN_BASE_STRINGS or any(base in lower for base in KNOWN_BASE_STRINGS):
        return True

    n_count = text.count("N") + text.count("n")
    has_phosphazene_motif = "n=p" in lower or "p=n" in lower
    if has_phosphazene_motif:
        return True

    has_imine_carbon = "C(=N" in text or "c(=N" in text or "N=C" in text or "n=C" in text
    has_multiple_tertiary_amines = text.count("N(") >= 2
    return has_imine_carbon and n_count >= 3 and ("C" in text or "c" in text or has_multiple_tertiary_amines)


def repair_condition_roles(parsed: dict, strict: bool = False) -> dict[str, object]:
    """Repair condition-role assignments when each condition role is unique.

    If a unique catalyst, ligand, base, and solvent/additive are identifiable
    among condition tokens, recovered fields are reordered accordingly. If any
    role is missing or ambiguous, positional assignments are retained and the
    validation status records the uncertainty.
    """
    tokens = [
        _normalize_value(parsed.get("parsed_catalyst_smiles")),
        _normalize_value(parsed.get("parsed_ligand_smiles")),
        _normalize_value(parsed.get("parsed_base_smiles")),
        _normalize_value(parsed.get("parsed_solvent_or_additive_smiles")),
    ]
    result = _empty_role_validation_fields(tokens)
    if parsed.get("condition_parse_status") != "ok":
        result["role_validation_status"] = "invalid_or_unresolved"
        result["role_validation_notes"] = "condition roles not validated because positional parse failed"
        if strict:
            raise ValueError(str(result["role_validation_notes"]))
        return result

    classifications = [classify_bh_component(token) for token in tokens]
    role_to_indices = {
        role: [index for index, roles in enumerate(classifications) if role in roles]
        for role in ["catalyst", "ligand", "base", "solvent_or_additive"]
    }
    notes: list[str] = []
    for index, (token, roles) in enumerate(zip(tokens, classifications, strict=True), start=2):
        notes.append(f"token_{index}={token}: {','.join(sorted(roles))}")

    fallback_notes: list[str] = []
    solvent_fallback_applied = False
    if (
        len(role_to_indices["catalyst"]) == 1
        and len(role_to_indices["ligand"]) == 1
        and len(role_to_indices["base"]) == 1
        and len(role_to_indices["solvent_or_additive"]) == 0
    ):
        assigned_indices = {
            role_to_indices["catalyst"][0],
            role_to_indices["ligand"][0],
            role_to_indices["base"][0],
        }
        remaining_indices = [index for index in range(len(tokens)) if index not in assigned_indices]
        if len(remaining_indices) == 1:
            remaining_index = remaining_indices[0]
            remaining_token = tokens[remaining_index]
            remaining_roles = classifications[remaining_index]
            if _can_assign_remaining_condition_token_to_solvent_or_additive(remaining_token, remaining_roles):
                role_to_indices["solvent_or_additive"] = [remaining_index]
                solvent_fallback_applied = True
                fallback_notes.append(
                    "remaining_condition_token_assigned_to_solvent_or_additive="
                    f"token_{remaining_index + 2}:{remaining_token}"
                )

    ambiguous_roles = [
        f"{role}:{[tokens[index] for index in indices]}"
        for role, indices in role_to_indices.items()
        if len(indices) != 1
    ]
    if ambiguous_roles:
        result["role_validation_status"] = "ambiguous"
        result["role_validation_notes"] = "; ".join([*notes, f"ambiguous_or_missing={ambiguous_roles}"])
        _set_confidences(result, role_to_indices)
        if strict:
            raise ValueError(str(result["role_validation_notes"]))
        return result

    repaired = {
        "repaired_catalyst_smiles": tokens[role_to_indices["catalyst"][0]],
        "repaired_ligand_smiles": tokens[role_to_indices["ligand"][0]],
        "repaired_base_smiles": tokens[role_to_indices["base"][0]],
        "repaired_solvent_or_additive_smiles": tokens[role_to_indices["solvent_or_additive"][0]],
    }
    result.update(repaired)
    _set_confidences(result, role_to_indices)
    if solvent_fallback_applied:
        result["solvent_or_additive_role_confidence"] = "medium"
    repaired_order = [
        repaired["repaired_catalyst_smiles"],
        repaired["repaired_ligand_smiles"],
        repaired["repaired_base_smiles"],
        repaired["repaired_solvent_or_additive_smiles"],
    ]
    result["repaired_condition_order_smiles"] = ".".join(repaired_order)
    positional_order = tokens
    if repaired_order == positional_order:
        result["role_validation_status"] = "valid"
        result["role_validation_notes"] = "; ".join([*notes, *fallback_notes])
    else:
        result["role_validation_status"] = "repaired"
        result["role_validation_notes"] = "; ".join(
            [
                *notes,
                *fallback_notes,
                f"positional_order={'.'.join(positional_order)}",
                f"repaired_order={result['repaired_condition_order_smiles']}",
            ]
        )
    return result


def parse_bh_reaction_smiles(reaction_smiles: str) -> dict[str, object] | None:
    """Parse positional BH roles from a processed reaction SMILES string.

    This is a deterministic, dataset-specific parser for processed
    Buchwald-Hartwig CSVs where the left side has exactly six dot-separated
    tokens before ``>>``. It recovers component identities only; it does not
    infer missing experimental metadata such as temperature.
    """
    parsed = _empty_parsed_fields()
    if not isinstance(reaction_smiles, str) or not reaction_smiles.strip():
        parsed["condition_parse_status"] = "malformed"
        parsed["condition_parse_error"] = "missing reaction_smiles"
        return parsed

    text = reaction_smiles.strip()
    if text.count(">>") != 1:
        parsed["condition_parse_status"] = "malformed"
        parsed["condition_parse_error"] = "expected exactly one >> separator"
        return parsed

    left, product = (part.strip() for part in text.split(">>", 1))
    left_tokens = [token.strip() for token in left.split(".") if token.strip()]
    if not product:
        parsed["condition_parse_status"] = "malformed"
        parsed["condition_parse_error"] = "missing product"
        return parsed
    if len(left_tokens) != 6:
        parsed["condition_parse_status"] = "unexpected_left_token_count"
        parsed["condition_parse_error"] = f"expected 6 left tokens, got {len(left_tokens)}"
        parsed["parsed_product_smiles"] = product
        return parsed

    parsed.update(
        {
            "parsed_reactant_1_smiles": left_tokens[0],
            "parsed_reactant_2_smiles": left_tokens[1],
            "parsed_catalyst_smiles": left_tokens[2],
            "parsed_ligand_smiles": left_tokens[3],
            "parsed_base_smiles": left_tokens[4],
            "parsed_solvent_or_additive_smiles": left_tokens[5],
            "parsed_product_smiles": product,
            "condition_block_smiles": ".".join(left_tokens[2:]),
            "condition_parse_status": "ok",
            "condition_parse_error": "",
        }
    )
    return parsed


def recover_condition_fields(row: Mapping[str, Any], strict: bool = False) -> dict[str, object]:
    """Recover parsed and preferred condition/component fields for one row.

    Existing non-empty, non-``UNKNOWN`` columns are preserved. Missing or
    ``UNKNOWN`` component columns fall back to positional fields parsed from
    ``reaction_smiles`` where possible. Temperature is preserved only when an
    explicit non-``UNKNOWN`` value is present; otherwise it is marked
    ``NOT_RECOVERABLE``.
    """
    parsed = parse_bh_reaction_smiles(row.get("reaction_smiles", ""))
    if parsed is None:
        parsed = _empty_parsed_fields()
    role_repair = repair_condition_roles(parsed, strict=strict)

    status = str(parsed["condition_parse_status"])
    errors = [str(parsed["condition_parse_error"])] if parsed["condition_parse_error"] else []

    if status == "ok":
        mismatches = _mismatch_statuses(row, parsed)
        if mismatches:
            if strict:
                raise ValueError("; ".join(mismatches))
            status = "_and_".join(mismatches)
            errors.extend(mismatches)

    recovered = {
        "recovered_reactant_1_smiles": _prefer_existing(
            row.get("reactant_1_smiles"),
            parsed.get("parsed_reactant_1_smiles"),
        ),
        "recovered_reactant_2_smiles": _prefer_existing(
            row.get("reactant_2_smiles"),
            parsed.get("parsed_reactant_2_smiles"),
        ),
        "recovered_product_smiles": _prefer_existing(
            row.get("product_smiles"),
            parsed.get("parsed_product_smiles"),
        ),
        "recovered_catalyst_smiles": _prefer_existing(
            row.get("catalyst_smiles"),
            role_repair.get("repaired_catalyst_smiles", parsed.get("parsed_catalyst_smiles")),
        ),
        "recovered_ligand_smiles": _prefer_existing(
            row.get("ligand_smiles"),
            role_repair.get("repaired_ligand_smiles", parsed.get("parsed_ligand_smiles")),
        ),
        "recovered_base_smiles": _prefer_existing(
            row.get("base_smiles"),
            role_repair.get("repaired_base_smiles", parsed.get("parsed_base_smiles")),
        ),
        "recovered_solvent_or_additive_smiles": _prefer_existing(
            row.get("solvent"),
            role_repair.get(
                "repaired_solvent_or_additive_smiles",
                parsed.get("parsed_solvent_or_additive_smiles"),
            ),
        ),
        "recovered_temperature": (
            _normalize_value(row.get("temperature"))
            if _is_known(row.get("temperature"))
            else NOT_RECOVERABLE
        ),
    }

    result = {**parsed, **role_repair, **recovered}
    result["condition_parse_status"] = status
    result["condition_parse_error"] = "; ".join(errors)
    return result


def augment_bh_dataframe(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Return a copy of ``df`` with parsed and recovered BH condition fields."""
    rows = [recover_condition_fields(row, strict=strict) for row in df.to_dict("records")]
    recovered = pd.DataFrame(rows, index=df.index)
    result = df.copy()
    for column in [*PARSED_COLUMNS, *RECOVERED_COLUMNS, *ROLE_VALIDATION_COLUMNS]:
        result[column] = recovered[column] if column in recovered else UNKNOWN
    return result


def summarize_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize parse statuses and recovered condition-field cardinalities."""
    augmented = df if set(PARSED_COLUMNS + RECOVERED_COLUMNS).issubset(df.columns) else augment_bh_dataframe(df)
    rows: list[dict[str, object]] = []
    for status, count in augmented["condition_parse_status"].value_counts(dropna=False).items():
        rows.append({"section": "parse_status", "field": str(status), "value": int(count)})
    if "role_validation_status" in augmented:
        for status, count in augmented["role_validation_status"].value_counts(dropna=False).items():
            rows.append({"section": "role_validation_status", "field": str(status), "value": int(count)})
    for column in [
        "recovered_catalyst_smiles",
        "recovered_ligand_smiles",
        "recovered_base_smiles",
        "recovered_solvent_or_additive_smiles",
        "recovered_temperature",
    ]:
        series = augmented[column].map(_normalize_value)
        rows.append(
            {
                "section": "unique_recovered_values",
                "field": column,
                "value": int(series.nunique(dropna=False)),
            }
        )
        rows.append(
            {
                "section": "known_recovered_rows",
                "field": column,
                "value": int(series.map(_is_known).sum()),
            }
        )
    return pd.DataFrame(rows)


def _empty_parsed_fields() -> dict[str, object]:
    return {
        "parsed_reactant_1_smiles": UNKNOWN,
        "parsed_reactant_2_smiles": UNKNOWN,
        "parsed_catalyst_smiles": UNKNOWN,
        "parsed_ligand_smiles": UNKNOWN,
        "parsed_base_smiles": UNKNOWN,
        "parsed_solvent_or_additive_smiles": UNKNOWN,
        "parsed_product_smiles": UNKNOWN,
        "condition_block_smiles": UNKNOWN,
        "condition_parse_status": "malformed",
        "condition_parse_error": "",
    }


def _empty_role_validation_fields(tokens: list[str] | None = None) -> dict[str, object]:
    condition_tokens = tokens or [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN]
    return {
        "catalyst_role_confidence": "unknown",
        "ligand_role_confidence": "unknown",
        "base_role_confidence": "unknown",
        "solvent_or_additive_role_confidence": "unknown",
        "role_validation_status": "unknown",
        "role_validation_notes": "",
        "original_condition_token_2": condition_tokens[0],
        "original_condition_token_3": condition_tokens[1],
        "original_condition_token_4": condition_tokens[2],
        "original_condition_token_5": condition_tokens[3],
        "repaired_condition_order_smiles": ".".join(condition_tokens),
        "repaired_catalyst_smiles": condition_tokens[0],
        "repaired_ligand_smiles": condition_tokens[1],
        "repaired_base_smiles": condition_tokens[2],
        "repaired_solvent_or_additive_smiles": condition_tokens[3],
    }


def _set_confidences(result: dict[str, object], role_to_indices: Mapping[str, list[int]]) -> None:
    for role, confidence_column in [
        ("catalyst", "catalyst_role_confidence"),
        ("ligand", "ligand_role_confidence"),
        ("base", "base_role_confidence"),
        ("solvent_or_additive", "solvent_or_additive_role_confidence"),
    ]:
        if len(role_to_indices.get(role, [])) == 1:
            result[confidence_column] = "medium" if role == "solvent_or_additive" else "high"
        elif len(role_to_indices.get(role, [])) > 1:
            result[confidence_column] = "low"
        else:
            result[confidence_column] = "unknown"


def _looks_like_transition_metal_component(text: str) -> bool:
    lower = text.lower()
    for marker in TRANSITION_METAL_MARKERS:
        if marker.islower() and marker in lower:
            return True
        if marker in text:
            return True
    return False


def _looks_like_base(text: str) -> bool:
    compact = _compact_text(text)
    lower = text.lower()
    if compact in KNOWN_BASE_STRINGS or any(base in lower for base in KNOWN_BASE_STRINGS):
        return True
    if is_organic_superbase_like(text):
        return True
    if any(counterion in text for counterion in ["[Na+]", "[K+]", "[Cs+]", "[Li+]", "[Rb+]"]):
        if "[O-]" in text or "C(=O)" in text or "P(=O)" in text or "OP(" in text:
            return True
    if "[O-]" in text and ("C(=O)" in text or "P(=O)" in text or "P([O-])" in text):
        return True
    if "carbonate" in lower or "phosphate" in lower or "acetate" in lower or "hydroxide" in lower:
        return True
    return False


def _looks_like_ligand(text: str) -> bool:
    lower = text.lower()
    if any(name in lower for name in KNOWN_LIGAND_STRINGS):
        return True
    if "[P" in text or "P(" in text or "P1" in text or "P2" in text or "P3" in text:
        return True
    return False


def _looks_like_electrophile(text: str) -> bool:
    if text.startswith("[Cl") or text.startswith("[Br") or text.startswith("[I"):
        return False
    has_halogen = any(halogen in text for halogen in ["Br", "Cl", "I"])
    has_organic_framework = "c" in text or "C" in text
    return has_halogen and has_organic_framework and "+" not in text and "[O-]" not in text


def _looks_like_nucleophile(text: str) -> bool:
    compact = _compact_text(text)
    if compact in KNOWN_BASE_STRINGS or is_organic_superbase_like(text):
        return False
    return "N" in text and ("C" in text or "c" in text) and "+" not in text and "[O-]" not in text


def _can_assign_remaining_condition_token_to_solvent_or_additive(token: str, roles: set[str]) -> bool:
    if {"catalyst", "ligand", "base"} & roles:
        return False
    return _looks_like_neutral_solvent_or_additive(token)


def _looks_like_neutral_solvent_or_additive(text: str) -> bool:
    if "+" in text or "[-" in text or "-]" in text:
        return False
    if _looks_like_transition_metal_component(text) or _looks_like_base(text) or _looks_like_ligand(text):
        return False
    return len(text) <= 60 and any(atom in text for atom in ["C", "c", "O", "N", "n"])


def _compact_text(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )


def _mismatch_statuses(row: Mapping[str, Any], parsed: Mapping[str, object]) -> list[str]:
    mismatches: list[str] = []
    if _is_known(row.get("product_smiles")) and _normalize_value(row.get("product_smiles")) != parsed.get(
        "parsed_product_smiles"
    ):
        mismatches.append("product_mismatch")
    reactant_1_mismatch = _is_known(row.get("reactant_1_smiles")) and _normalize_value(
        row.get("reactant_1_smiles")
    ) != parsed.get("parsed_reactant_1_smiles")
    reactant_2_mismatch = _is_known(row.get("reactant_2_smiles")) and _normalize_value(
        row.get("reactant_2_smiles")
    ) != parsed.get("parsed_reactant_2_smiles")
    if reactant_1_mismatch or reactant_2_mismatch:
        mismatches.append("reactant_mismatch")
    return mismatches


def _prefer_existing(existing: object, fallback: object) -> str:
    if _is_known(existing):
        return _normalize_value(existing)
    if _is_known(fallback):
        return _normalize_value(fallback)
    return UNKNOWN


def _is_known(value: object) -> bool:
    normalized = _normalize_value(value)
    return normalized not in {"", UNKNOWN, NOT_RECOVERABLE, "NAN", "NONE", "<NA>"}


def _normalize_value(value: object) -> str:
    if pd.isna(value):
        return UNKNOWN
    return str(value).strip()
