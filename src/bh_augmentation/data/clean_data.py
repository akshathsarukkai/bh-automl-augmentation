"""Cleaning helpers for Buchwald-Hartwig reaction-yield data."""

from __future__ import annotations

import ast
import re
import warnings

import pandas as pd

ACTIVE_REQUIRED_SCHEMA = [
    "reaction_id",
    "reaction_smiles",
    "yield",
]

PARSED_HELPER_COLUMNS = [
    "product_smiles",
    "reactant_1_smiles",
    "reactant_2_smiles",
    "catalyst_smiles",
    "solvent",
    "temperature",
]

# Kept as the stable output ordering used by existing processed CSVs.
NORMALIZED_SCHEMA = [
    "reaction_id",
    *PARSED_HELPER_COLUMNS,
    "reaction_smiles",
    "yield",
]

OPTIONAL_CATEGORICAL_COLUMNS = [
    "product_smiles",
    "reactant_1_smiles",
    "reactant_2_smiles",
    "catalyst_smiles",
    "solvent",
]

COLUMN_ALIASES = {
    "id": "reaction_id",
    "rxn_id": "reaction_id",
    "reactionid": "reaction_id",
    "yield_percent": "yield",
    "yield_percentage": "yield",
    "yield_pct": "yield",
    "yield_": "yield",
}

# Input compatibility only. These columns are not part of the active schema.
LEGACY_COLUMN_ALIASES = {
    "aryl_halide": "aryl_halide_smiles",
    "amine": "amine_smiles",
    "ligand": "ligand_smiles",
    "base": "base_smiles",
    "additive": "additive_smiles",
}

TDC_COLUMN_ALIASES = {
    "reaction_id": "reaction_id",
    "drug_id": "reaction_id",
    "id": "reaction_id",
    "reaction": "reaction_smiles",
    "drug": "reaction_smiles",
    "x": "reaction_smiles",
    "yield": "yield",
    "y": "yield",
    "label": "yield",
}

TDC_OPTIONAL_COLUMNS = [
    "product_smiles",
    "reactant_1_smiles",
    "reactant_2_smiles",
    "catalyst_smiles",
    "solvent",
    "temperature",
]

COMPONENT_COLUMNS = [
    "product_smiles",
    "reactant_1_smiles",
    "reactant_2_smiles",
    "catalyst_smiles",
]

DEPRECATED_COMPONENT_COLUMNS = [
    "aryl_halide_smiles",
    "amine_smiles",
    "ligand_smiles",
    "base_smiles",
    "additive_smiles",
]


def clean_buchwald_hartwig(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean Buchwald-Hartwig reaction-yield records.

    Column names are standardized to lowercase snake case. Rows with missing,
    non-numeric, or out-of-range yields are dropped. The accepted yield range is
    `0 <= yield <= 100`; values outside that range are treated as invalid
    measurements rather than clipped.

    Optional categorical fields already present are filled with `"UNKNOWN"`. The
    returned DataFrame includes `attrs["original_row_count"]` and
    `attrs["dropped_invalid_yield_count"]` metadata.
    """
    cleaned = normalize_tdc_buchwald_hartwig(df) if _looks_like_tdc_yields(df) else df.copy()
    tdc_adapter_attrs = cleaned.attrs.get("tdc_adapter")
    original_row_count = len(cleaned)
    cleaned.columns = [_normalize_column_name(column) for column in cleaned.columns]
    cleaned = cleaned.rename(columns={**COLUMN_ALIASES, **LEGACY_COLUMN_ALIASES})

    if "reaction_id" not in cleaned.columns:
        cleaned.insert(0, "reaction_id", [_make_reaction_id(i) for i in range(original_row_count)])

    for column in ACTIVE_REQUIRED_SCHEMA:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA

    cleaned["yield"] = pd.to_numeric(cleaned["yield"], errors="coerce")
    valid_yield = cleaned["yield"].between(0, 100, inclusive="both")
    reaction_values = cleaned["reaction_smiles"].fillna("").astype(str).str.strip()
    valid_reaction = reaction_values.ne("") & reaction_values.ne("UNKNOWN")
    cleaned = cleaned.loc[valid_yield & valid_reaction].copy()

    for column in OPTIONAL_CATEGORICAL_COLUMNS:
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].fillna("UNKNOWN")
        cleaned[column] = cleaned[column].replace("", "UNKNOWN")

    cleaned = _drop_fully_unknown_deprecated_columns(cleaned)
    normalized_columns = [column for column in NORMALIZED_SCHEMA if column in cleaned.columns]
    extra_columns = [column for column in cleaned.columns if column not in normalized_columns]
    cleaned = cleaned[[*normalized_columns, *extra_columns]].reset_index(drop=True)
    if tdc_adapter_attrs is None:
        _warn_if_most_components_unknown(cleaned)
    cleaned.attrs["original_row_count"] = original_row_count
    cleaned.attrs["dropped_invalid_yield_count"] = int((~valid_yield).sum())
    cleaned.attrs["dropped_missing_reaction_count"] = int(
        (valid_yield & ~valid_reaction).sum()
    )
    if tdc_adapter_attrs is not None:
        cleaned.attrs["tdc_adapter"] = tdc_adapter_attrs
    return cleaned


def normalize_tdc_buchwald_hartwig(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common TDC Yields columns to the project reaction schema.

    TDC may return columns such as `Reaction_ID`, `Reaction`, and `Y`, or older
    aliases like `Drug_ID`, `Drug`, and `Y`. This adapter maps those fields to
    `reaction_id`, `reaction_smiles`, and `yield`, creates parsed helper columns
    when available, and drops rows only when `reaction_smiles` is
    missing/empty, `yield` is missing/non-numeric, or `yield` is outside 0-100.
    """
    original_row_count = len(df)
    raw_columns = list(df.columns)
    raw_to_normalized = {_normalize_column_name(column): column for column in raw_columns}
    normalized_raw = df.rename(columns={column: _normalize_column_name(column) for column in raw_columns})

    id_source = _first_present(normalized_raw.columns, ["reaction_id", "drug_id", "id"])
    reaction_source = _first_present(
        normalized_raw.columns,
        ["reaction_smiles", "reaction", "drug", "x"],
    )
    yield_source = _first_present(normalized_raw.columns, ["yield", "y", "label"])

    normalized = pd.DataFrame(index=normalized_raw.index)
    if id_source is not None:
        normalized["reaction_id"] = normalized_raw[id_source].astype(str)
    else:
        normalized["reaction_id"] = [_make_reaction_id(i) for i in range(len(normalized_raw))]

    reaction_records = (
        normalized_raw[reaction_source].map(_parse_tdc_reaction_record)
        if reaction_source is not None
        else pd.Series([{} for _ in range(len(normalized_raw))], index=normalized_raw.index)
    )
    parsed_fields = reaction_records.map(_tdc_reaction_fields)
    field_frame = pd.DataFrame(parsed_fields.tolist(), index=normalized_raw.index)
    for column in TDC_OPTIONAL_COLUMNS:
        normalized[column] = field_frame[column] if column in field_frame else "UNKNOWN"
    normalized["reaction_smiles"] = (
        field_frame["reaction_smiles"] if "reaction_smiles" in field_frame else pd.NA
    )

    if yield_source is not None:
        normalized["yield"] = pd.to_numeric(normalized_raw[yield_source], errors="coerce")
    else:
        normalized["yield"] = pd.NA

    consumed_sources = {id_source, reaction_source, yield_source}
    for column in normalized_raw.columns:
        if column not in consumed_sources and column not in normalized.columns:
            normalized[column] = normalized_raw[column]

    numeric_yield = normalized["yield"].dropna()
    yield_was_fraction_scaled = bool(not numeric_yield.empty and numeric_yield.max() <= 1.0)
    if yield_was_fraction_scaled:
        normalized["yield"] = normalized["yield"] * 100.0

    extra_columns = [column for column in normalized.columns if column not in NORMALIZED_SCHEMA]
    normalized = normalized[[*NORMALIZED_SCHEMA, *extra_columns]]
    diagnostics = {
        "raw_row_count": original_row_count,
        "raw_columns": [str(column) for column in raw_columns],
        "detected_id_column": _raw_column_for_normalized(raw_to_normalized, id_source),
        "detected_reaction_column": _raw_column_for_normalized(raw_to_normalized, reaction_source),
        "detected_yield_column": _raw_column_for_normalized(raw_to_normalized, yield_source),
        "rows_before": original_row_count,
        "rows_after_mapping": len(normalized),
        "rows_after_dataframe_construction": len(normalized),
        "yield_was_fraction_scaled": yield_was_fraction_scaled,
        "rows_after_yield_conversion": len(normalized),
        "parsed_reaction_record_count": int(reaction_records.map(bool).sum()),
    }

    reaction_smiles = normalized["reaction_smiles"]
    valid_reaction_smiles = reaction_smiles.notna() & reaction_smiles.astype(str).str.strip().ne("")
    normalized = normalized.loc[valid_reaction_smiles].copy()
    diagnostics["rows_after_reaction_smiles_filter"] = len(normalized)

    normalized["yield"] = pd.to_numeric(normalized["yield"], errors="coerce")
    normalized = normalized.loc[normalized["yield"].notna()].copy()
    diagnostics["rows_after_numeric_yield_filter"] = len(normalized)

    normalized = normalized.loc[normalized["yield"].between(0, 100, inclusive="both")].copy()
    diagnostics["rows_after_yield_range_filter"] = len(normalized)

    normalized = normalized[[*NORMALIZED_SCHEMA, *extra_columns]].reset_index(drop=True)
    normalized = _drop_fully_unknown_deprecated_columns(normalized)
    unknown_counts = count_unknown_components(normalized)
    normalized.attrs["tdc_adapter"] = {
        "mapped_reaction_smiles_column": _raw_column_for_normalized(raw_to_normalized, reaction_source),
        "mapped_yield_column": _raw_column_for_normalized(raw_to_normalized, yield_source),
        **diagnostics,
        "rows_after": len(normalized),
        "final_saved_row_count": len(normalized),
        "unknown_component_counts": unknown_counts,
        "non_unknown_component_counts": {
            column: int(len(normalized) - count)
            for column, count in unknown_counts.items()
        },
    }
    _warn_if_most_components_unknown(normalized)
    return normalized


def count_unknown_components(df: pd.DataFrame) -> dict[str, int]:
    """Count UNKNOWN values in expected component columns."""
    counts: dict[str, int] = {}
    for column in COMPONENT_COLUMNS:
        if column not in df.columns:
            counts[column] = len(df)
            continue
        values = df[column].fillna("UNKNOWN").astype(str).str.strip()
        counts[column] = int(values.eq("UNKNOWN").sum())
    return counts


def _normalize_column_name(column: object) -> str:
    """Convert a raw column name to lowercase snake case."""
    normalized = str(column).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _make_reaction_id(index: int) -> str:
    """Create a stable synthetic reaction identifier for a row index."""
    return f"rxn_{index + 1:06d}"


def _looks_like_tdc_yields(df: pd.DataFrame) -> bool:
    columns = {_normalize_column_name(column) for column in df.columns}
    has_raw_tdc_x = bool(columns & {"reaction", "drug", "x"})
    has_tdc_y = bool(columns & {"yield", "y", "label"})
    if has_raw_tdc_x and has_tdc_y:
        return True
    if "reaction_smiles" in columns and has_tdc_y:
        reaction_column = next(
            column for column in df.columns if _normalize_column_name(column) == "reaction_smiles"
        )
        values = df[reaction_column].dropna().astype(str).str.strip()
        return bool(values.head(20).str.startswith("{").any())
    return False


def _first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _raw_column_for_normalized(raw_to_normalized: dict[str, object], normalized_name: str | None) -> str:
    if normalized_name is None:
        return ""
    return str(raw_to_normalized.get(normalized_name, normalized_name))


def _reaction_value_to_string(value: object) -> object:
    if value is None:
        return pd.NA
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return pd.NA
    return str(value)


def _parse_tdc_reaction_record(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    text = _reaction_value_to_string(value)
    if not isinstance(text, str) or not text.strip():
        return {}
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {"reaction_smiles": stripped}


def _tdc_reaction_fields(record: dict[str, object]) -> dict[str, object]:
    fields = {column: "UNKNOWN" for column in TDC_OPTIONAL_COLUMNS}
    if not record:
        return {**fields, "reaction_smiles": pd.NA}

    product_tokens = _molecule_tokens(_first_record_value(record, ["product", "products"]))
    reactant_tokens = _molecule_tokens(_first_record_value(record, ["reactant", "reactants"]))
    catalyst_tokens = _molecule_tokens(_first_record_value(record, ["catalyst", "catalysts"]))
    ligand_tokens = _molecule_tokens(_first_record_value(record, ["ligand", "ligands"]))
    base_tokens = _molecule_tokens(_first_record_value(record, ["base", "bases"]))
    additive_tokens = _molecule_tokens(_first_record_value(record, ["additive", "additives"]))
    solvent_tokens = _molecule_tokens(_first_record_value(record, ["solvent", "solvents"]))
    reaction_tokens = _molecule_tokens(record.get("reaction_smiles"))

    if product_tokens:
        fields["product_smiles"] = product_tokens[0]
    if len(reactant_tokens) >= 1:
        fields["reactant_1_smiles"] = reactant_tokens[0]
    if len(reactant_tokens) >= 2:
        fields["reactant_2_smiles"] = reactant_tokens[1]
    if catalyst_tokens:
        fields["catalyst_smiles"] = ".".join(catalyst_tokens)
    # Preserve explicitly supplied legacy roles, but never require or infer them.
    if ligand_tokens:
        fields["ligand_smiles"] = ".".join(ligand_tokens)
    if base_tokens:
        fields["base_smiles"] = ".".join(base_tokens)
    if additive_tokens:
        fields["additive_smiles"] = ".".join(additive_tokens)
    if solvent_tokens:
        fields["solvent"] = ".".join(solvent_tokens)

    explicit_reaction = record.get("reaction_smiles")
    if isinstance(explicit_reaction, str) and explicit_reaction.strip() and ">" in explicit_reaction:
        reaction_smiles = explicit_reaction.strip()
    elif product_tokens or reactant_tokens or catalyst_tokens or ligand_tokens or base_tokens or additive_tokens:
        agents = catalyst_tokens + ligand_tokens + base_tokens + additive_tokens
        reaction_smiles = f"{'.'.join(reactant_tokens)}>{'.'.join(agents)}>{'.'.join(product_tokens)}"
    elif reaction_tokens:
        reaction_smiles = ".".join(reaction_tokens)
    else:
        reaction_smiles = pd.NA

    return {**fields, "reaction_smiles": reaction_smiles}


def _first_record_value(record: dict[str, object], keys: list[str]) -> object:
    normalized = {_normalize_column_name(key): value for key, value in record.items()}
    for key in keys:
        value = normalized.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _molecule_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_molecule_tokens(item))
        return tokens
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(".") if token.strip()]


def _warn_if_most_components_unknown(df: pd.DataFrame) -> None:
    if df.empty:
        return
    present_columns = [column for column in COMPONENT_COLUMNS if column in df.columns]
    if not present_columns:
        return
    counts = count_unknown_components(df)
    total_values = len(df) * len(present_columns)
    unknown_values = sum(counts[column] for column in present_columns)
    if total_values and unknown_values / total_values > 0.5:
        warnings.warn(
            "More than 50% of molecular component fields are UNKNOWN after normalization. "
            "Use reaction_smiles or generic reactant features when exact roles are absent.",
            stacklevel=2,
        )


def _drop_fully_unknown_deprecated_columns(df: pd.DataFrame) -> pd.DataFrame:
    removable: list[str] = []
    for column in DEPRECATED_COMPONENT_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].fillna("UNKNOWN").astype(str).str.strip()
        if values.isin(["", "UNKNOWN"]).all():
            removable.append(column)
    return df.drop(columns=removable)
