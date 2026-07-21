"""Featurization helpers for molecular components and reaction conditions."""

import ast
import hashlib
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_COLUMNS,
    CANONICAL_ROLE_NAMES,
    reaction_roles_from_row,
)
from bh_augmentation.features.compatibility import FeatureBlock, FeatureMetadata

REACTION_FEATURE_KINDS = {
    "reaction_morgan_sum",
    "reaction_section_concat",
    "reaction_section_concat_delta",
    "bh_role_separated",
    "bh_role_separated_delta",
}
DEPRECATED_FEATURE_ALIASES = {
    "reaction_smiles": "reaction_morgan_sum",
    "reaction_plus_components": "reaction_section_concat_delta",
    "reaction_combined_redundant": "reaction_section_concat_delta",
    "reaction_role_concat": "reaction_section_concat",
    "reaction_role_concat_delta": "reaction_section_concat_delta",
    "role_separated_conditions": "bh_role_separated",
    "role_separated_conditions_delta": "bh_role_separated_delta",
}
REMOVED_FEATURE_KINDS = {"fp_concat", "fp_plus_conditions", "categorical_conditions"}

DICT_REACTANT_KEYS = ["reactant", "reactants"]
DICT_AGENT_KEYS = [
    "catalyst",
    "catalysts",
    "ligand",
    "ligands",
    "base",
    "bases",
    "solvent",
    "solvents",
    "reagent",
    "reagents",
    "additive",
    "additives",
]
DICT_PRODUCT_KEYS = ["product", "products"]
_WARNED_HASH_FINGERPRINT_FALLBACK = False

ROLE_SEPARATED_CONDITION_COLUMNS = list(CANONICAL_ROLE_COLUMNS)


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    warn_invalid: bool = True,
    backend: str = "auto",
) -> np.ndarray:
    """Return a Morgan fingerprint bit vector for a SMILES string.

    Invalid or missing SMILES values return an all-zero fingerprint and emit a
    warning. RDKit is imported lazily so the package can be used for non-RDKit
    workflows until this function is called.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        if warn_invalid:
            warnings.warn("Invalid SMILES encountered; returning zero fingerprint.", stacklevel=2)
        return np.zeros(n_bits, dtype=np.float32)

    resolved_backend = _resolve_fingerprint_backend(backend)
    if resolved_backend == "hash":
        if backend == "auto":
            _warn_hash_fingerprint_fallback(warn_invalid)
        return _hash_smiles_fingerprint(smiles, radius=radius, n_bits=n_bits)

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "The RDKit fingerprint backend was requested but RDKit is not installed."
        ) from exc

    with _quiet_rdkit_errors(enabled=not warn_invalid):
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        if warn_invalid:
            warnings.warn(
                f"Invalid SMILES encountered; returning zero fingerprint: {smiles}",
                stacklevel=2,
            )
        return np.zeros(n_bits, dtype=np.float32)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius,
        nBits=n_bits,
    )
    array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array.astype(np.float32)


def _warn_hash_fingerprint_fallback(warn_invalid: bool) -> None:
    global _WARNED_HASH_FINGERPRINT_FALLBACK
    if warn_invalid and not _WARNED_HASH_FINGERPRINT_FALLBACK:
        warnings.warn(
            "RDKit is not installed; using deterministic hash fingerprints as a "
            "lightweight fallback. Install RDKit for true Morgan fingerprints.",
            UserWarning,
            stacklevel=3,
        )
        _WARNED_HASH_FINGERPRINT_FALLBACK = True


def _hash_smiles_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    """Return a deterministic non-chemical fallback fingerprint."""
    if _looks_invalid_smiles_token(smiles):
        return np.zeros(n_bits, dtype=np.float32)

    text = smiles.strip()
    tokens = {text}
    max_ngram = max(1, min(len(text), radius + 2))
    for width in range(1, max_ngram + 1):
        tokens.update(text[index : index + width] for index in range(len(text) - width + 1))

    fingerprint = np.zeros(n_bits, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        fingerprint[int.from_bytes(digest, "little") % n_bits] = 1.0
    return fingerprint


def _looks_invalid_smiles_token(smiles: str) -> bool:
    stripped = smiles.strip()
    if not stripped or stripped.upper() in {"UNKNOWN", "INVALID", "NAN", "NONE"}:
        return True
    return any(marker in stripped for marker in ["?", ">", "{", "}"])


def reaction_smiles_fingerprint(
    reaction_smiles: object,
    radius: int = 2,
    n_bits: int = 2048,
    mode: str = "sum",
    backend: str = "auto",
) -> np.ndarray:
    """Fingerprint a reaction string by splitting it into molecule SMILES."""
    parts = extract_reaction_parts(reaction_smiles)
    all_tokens = parts["reactants"] + parts["agents"] + parts["products"]
    summed = _sum_morgan_fingerprints(
        all_tokens,
        radius=radius,
        n_bits=n_bits,
        backend=backend,
    )
    if mode == "sum":
        return summed
    if mode in {"or", "binary_or"}:
        return (summed > 0).astype(np.float32)
    raise ValueError("reaction_smiles fingerprint mode must be 'sum' or 'or'.")


def extract_reaction_parts(value: str | dict[object, object]) -> dict[str, list[str]]:
    """Extract reactant, agent, and product molecule SMILES from a reaction value."""
    empty = {"reactants": [], "agents": [], "products": []}
    if isinstance(value, dict):
        return _extract_dict_reaction_parts(value)
    if not isinstance(value, str):
        return empty

    text = value.strip()
    if not text:
        return empty
    if text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return empty
        if isinstance(parsed, dict):
            return _extract_dict_reaction_parts(parsed)
        return empty

    if ">" not in text:
        return {"reactants": _split_molecule_section(text), "agents": [], "products": []}

    sections = text.split(">")
    if len(sections) != 3:
        return empty
    return {
        "reactants": _split_molecule_section(sections[0]),
        "agents": _split_molecule_section(sections[1]),
        "products": _split_molecule_section(sections[2]),
    }


def extract_smiles_tokens(value: object) -> list[str]:
    """Extract molecule SMILES tokens from molecule, reaction, or dict-like strings."""
    parts = extract_reaction_parts(value)
    return parts["reactants"] + parts["agents"] + parts["products"]


def _extract_dict_reaction_parts(
    value: dict[object, object],
) -> dict[str, list[str]]:
    normalized = {str(key).strip().lower(): raw for key, raw in value.items()}
    return {
        "reactants": _tokens_for_keys(normalized, DICT_REACTANT_KEYS),
        "agents": _tokens_for_keys(normalized, DICT_AGENT_KEYS),
        "products": _tokens_for_keys(normalized, DICT_PRODUCT_KEYS),
    }


def _tokens_for_keys(value: dict[str, object], keys: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, str):
            tokens.extend(_split_molecule_section(raw))
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    tokens.extend(_split_molecule_section(item))
    return tokens


def split_reaction_to_molecule_smiles(reaction_smiles: str) -> list[str]:
    """Split reaction SMILES into flat molecule SMILES without inferring roles."""
    return extract_smiles_tokens(reaction_smiles)


def _split_molecule_section(section: str) -> list[str]:
    return [part.strip() for part in section.split(".") if part.strip()]


def component_fingerprint_features(
    df: pd.DataFrame,
    smiles_columns: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    backend: str = "auto",
) -> np.ndarray:
    """Build legacy concatenated fingerprints for explicit component columns."""
    if not smiles_columns:
        return np.empty((len(df), 0), dtype=np.float32)

    column_features: list[np.ndarray] = []
    for column in smiles_columns:
        if column not in df.columns:
            raise ValueError(f"Missing SMILES column for featurization: {column}")

        if column == "reaction_smiles":
            fingerprints = [
                reaction_smiles_fingerprint(
                    smiles,
                    radius=radius,
                    n_bits=n_bits,
                    backend=backend,
                )
                for smiles in df[column].fillna("")
            ]
        else:
            fingerprints = [
                morgan_fingerprint(
                    smiles,
                    radius=radius,
                    n_bits=n_bits,
                    warn_invalid=False,
                    backend=backend,
                )
                for smiles in df[column].fillna("")
            ]
        column_features.append(np.vstack(fingerprints).astype(np.float32))

    return np.hstack(column_features).astype(np.float32)


def one_hot_condition_features(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode categorical condition columns deterministically.

    Categories are sorted lexicographically within each column. Missing values
    are encoded as `"UNKNOWN"`.
    """
    if not categorical_columns:
        return np.empty((len(df), 0), dtype=np.float32), []

    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in categorical_columns:
        if column not in df.columns:
            raise ValueError(f"Missing categorical column for featurization: {column}")

        values = df[column].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN")
        categories = sorted(values.unique().tolist())
        encoded = np.zeros((len(df), len(categories)), dtype=np.float32)
        category_to_index = {category: index for index, category in enumerate(categories)}
        for row_index, value in enumerate(values):
            encoded[row_index, category_to_index[value]] = 1.0

        arrays.append(encoded)
        names.extend([f"{column}__{category}" for category in categories])

    return np.hstack(arrays).astype(np.float32), names


def build_feature_matrix(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a feature matrix, target vector, and feature names.

    The target column is `yield`. Supported feature configuration keys are:
    `smiles_columns`, `categorical_columns`, `radius`, and `n_bits`.
    """
    if "yield" not in df.columns:
        raise ValueError("Target column is required for feature matrix construction: yield")

    resolved = normalize_feature_config(feature_config)
    categorical_columns = list(resolved.get("categorical_columns", []))
    radius = int(resolved["radius"])
    n_bits = int(resolved["n_bits"])
    kind = str(resolved["kind"])
    backend = str(resolved["fingerprint_backend"])

    if kind in REACTION_FEATURE_KINDS:
        fingerprint_features, fingerprint_names = _reaction_feature_matrix(
            df,
            kind=kind,
            radius=radius,
            n_bits=n_bits,
            backend=backend,
        )
    else:
        smiles_columns = _resolve_smiles_columns(df, feature_config)
        fingerprint_features = component_fingerprint_features(
            df,
            smiles_columns=smiles_columns,
            radius=radius,
            n_bits=n_bits,
            backend=backend,
        )
        fingerprint_names = [
            f"{column}__morgan_{bit_index}"
            for column in smiles_columns
            for bit_index in range(_fingerprint_width(column, n_bits))
        ]

    condition_features, condition_names = one_hot_condition_features(
        df,
        categorical_columns=categorical_columns,
    )

    X = np.hstack([fingerprint_features, condition_features]).astype(np.float32)
    y = pd.to_numeric(df["yield"], errors="coerce").to_numpy(dtype=np.float32)
    feature_names = fingerprint_names + condition_names
    return X, y, feature_names


def build_feature_matrix_with_metadata(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], FeatureMetadata]:
    """Build features plus immutable scientific representation metadata."""
    resolved = normalize_feature_config(feature_config)
    X, y, names = build_feature_matrix(df, resolved)
    metadata = feature_metadata(
        resolved,
        feature_names=names,
        width=X.shape[1],
    )
    return X, y, names, metadata


def normalize_feature_config(feature_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize representation settings without inferring them from matrix width."""
    resolved = dict(feature_config)
    default_kind = "" if resolved.get("smiles_columns") else "reaction_section_concat"
    resolved["kind"] = canonical_feature_kind(resolved.get("kind", default_kind))
    resolved["n_bits"] = int(resolved.get("n_bits", 2048))
    resolved["radius"] = int(resolved.get("radius", 2))
    if resolved["n_bits"] < 1:
        raise ValueError("features.n_bits must be at least 1.")
    if resolved["radius"] < 0:
        raise ValueError("features.radius must be non-negative.")
    resolved["fingerprint_backend"] = _resolve_fingerprint_backend(
        str(resolved.get("fingerprint_backend", "auto"))
    )
    resolved.setdefault("categorical_columns", [])
    return resolved


def feature_metadata(
    feature_config: dict[str, Any],
    *,
    feature_names: Sequence[str],
    width: int,
) -> FeatureMetadata:
    """Return block semantics for one already-built feature matrix."""
    resolved = normalize_feature_config(feature_config)
    kind = str(resolved["kind"])
    n_bits = int(resolved["n_bits"])
    block_names = (
        _feature_block_names(kind)
        if kind in REACTION_FEATURE_KINDS
        else [f"smiles:{column}" for column in resolved.get("smiles_columns", [])]
    )
    blocks: tuple[FeatureBlock, ...] = tuple(
        FeatureBlock(name=name, start=index * n_bits, stop=(index + 1) * n_bits)
        for index, name in enumerate(block_names)
    )
    fingerprint_width = len(block_names) * n_bits
    cursor = fingerprint_width
    for column in resolved.get("categorical_columns", []):
        prefix = f"{column}__"
        block_width = sum(name.startswith(prefix) for name in feature_names[fingerprint_width:])
        if block_width:
            blocks = (*blocks, FeatureBlock(f"categorical:{column}", cursor, cursor + block_width))
            cursor += block_width
    if cursor != width:
        raise ValueError(
            "Feature metadata could not account for the complete matrix width: "
            f"accounted={cursor}, actual={width}."
        )
    if len(feature_names) != width:
        raise ValueError(
            f"Feature-name count {len(feature_names)} does not match matrix width {width}."
        )
    role_ordering = CANONICAL_ROLE_NAMES if kind.startswith("bh_role_separated") else ()
    return FeatureMetadata(
        representation_kind=kind,
        n_bits=n_bits,
        radius=int(resolved["radius"]),
        fingerprint_backend=str(resolved["fingerprint_backend"]),
        role_ordering=tuple(role_ordering),
        block_slices=blocks,
        total_width=int(width),
    )


def canonical_feature_kind(kind: object) -> str:
    """Return the canonical feature kind, warning for supported legacy aliases."""
    normalized = str(kind or "").strip()
    if normalized in DEPRECATED_FEATURE_ALIASES:
        replacement = DEPRECATED_FEATURE_ALIASES[normalized]
        warnings.warn(
            f"Feature kind '{normalized}' is deprecated; use '{replacement}'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return replacement
    if normalized in REMOVED_FEATURE_KINDS:
        supported = ", ".join(sorted(REACTION_FEATURE_KINDS))
        raise ValueError(
            f"Feature kind '{normalized}' has been removed. "
            f"Use one of: {supported}."
        )
    return normalized


def _reaction_feature_matrix(
    df: pd.DataFrame,
    kind: str,
    radius: int,
    n_bits: int,
    backend: str,
) -> tuple[np.ndarray, list[str]]:
    if kind in {"bh_role_separated", "bh_role_separated_delta"}:
        return _role_separated_condition_feature_matrix(
            df,
            kind=kind,
            radius=radius,
            n_bits=n_bits,
            backend=backend,
        )

    if "reaction_smiles" not in df.columns:
        raise ValueError(f"Feature kind '{kind}' requires reaction_smiles.")

    rows = [
        _reaction_feature_vector(
            value,
            kind=kind,
            radius=radius,
            n_bits=n_bits,
            backend=backend,
        )
        for value in df["reaction_smiles"].fillna("")
    ]
    width = _reaction_feature_width(kind, n_bits)
    features = np.vstack(rows).astype(np.float32) if rows else np.empty((0, width), dtype=np.float32)
    return features, _reaction_feature_names(kind, n_bits)


def _reaction_feature_vector(
    value: object,
    kind: str,
    radius: int,
    n_bits: int,
    backend: str,
) -> np.ndarray:
    parts = extract_reaction_parts(value)
    reactants = _sum_morgan_fingerprints(parts["reactants"], radius, n_bits, backend)
    agents = _sum_morgan_fingerprints(parts["agents"], radius, n_bits, backend)
    products = _sum_morgan_fingerprints(parts["products"], radius, n_bits, backend)
    reaction_sum = reactants + agents + products
    delta = products - reactants

    if kind == "reaction_morgan_sum":
        return reaction_sum.astype(np.float32)
    if kind == "reaction_section_concat":
        return np.concatenate([reactants, agents, products]).astype(np.float32)
    if kind == "reaction_section_concat_delta":
        return np.concatenate([reactants, agents, products, delta]).astype(np.float32)
    raise ValueError(f"Unknown reaction feature kind: {kind}")


def _role_separated_condition_feature_matrix(
    df: pd.DataFrame,
    kind: str,
    radius: int,
    n_bits: int,
    backend: str,
) -> tuple[np.ndarray, list[str]]:
    _validate_role_separated_condition_columns(df)
    rows = [
        _role_separated_condition_feature_vector(
            row,
            kind=kind,
            radius=radius,
            n_bits=n_bits,
            backend=backend,
        )
        for _, row in df.iterrows()
    ]
    width = _reaction_feature_width(kind, n_bits)
    features = np.vstack(rows).astype(np.float32) if rows else np.empty((0, width), dtype=np.float32)
    return features, _reaction_feature_names(kind, n_bits)


def _role_separated_condition_feature_vector(
    row: pd.Series,
    kind: str,
    radius: int,
    n_bits: int,
    backend: str,
) -> np.ndarray:
    roles = reaction_roles_from_row(row)
    fingerprints = {
        role: morgan_fingerprint(
            getattr(roles, role),
            radius=radius,
            n_bits=n_bits,
            warn_invalid=False,
            backend=backend,
        )
        for role in CANONICAL_ROLE_NAMES
    }
    role_blocks = [fingerprints[role] for role in CANONICAL_ROLE_NAMES]
    if kind == "bh_role_separated":
        return np.concatenate(role_blocks).astype(np.float32)
    if kind == "bh_role_separated_delta":
        product = fingerprints["product"]
        reactant_1 = fingerprints["reactant_1"]
        reactant_2 = fingerprints["reactant_2"]
        reactant_pair = reactant_1 + reactant_2
        deltas = [
            product - reactant_1,
            product - reactant_2,
            product - reactant_pair,
        ]
        return np.concatenate([*role_blocks, *deltas]).astype(np.float32)
    raise ValueError(f"Unknown role-separated condition feature kind: {kind}")


def _validate_role_separated_condition_columns(df: pd.DataFrame) -> None:
    missing = [column for column in ROLE_SEPARATED_CONDITION_COLUMNS if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Feature kind 'bh_role_separated' requires recovered condition "
            f"columns, but these are missing: {missing_text}. Generate "
            "data/processed/bh_clean_stress_with_conditions.csv first with "
            "bh_condition_reader.py."
        )


def _sum_morgan_fingerprints(
    smiles_tokens: Sequence[str],
    radius: int,
    n_bits: int,
    backend: str = "auto",
) -> np.ndarray:
    summed = np.zeros(n_bits, dtype=np.float32)
    for smiles in smiles_tokens:
        summed += morgan_fingerprint(
            smiles,
            radius=radius,
            n_bits=n_bits,
            warn_invalid=False,
            backend=backend,
        )
    return summed


def _reaction_feature_width(kind: str, n_bits: int) -> int:
    multipliers = {
        "reaction_morgan_sum": 1,
        "reaction_section_concat": 3,
        "reaction_section_concat_delta": 4,
        "bh_role_separated": 7,
        "bh_role_separated_delta": 10,
    }
    return multipliers[kind] * n_bits


def _reaction_feature_names(kind: str, n_bits: int) -> list[str]:
    sections = {
        "reaction_morgan_sum": ["reaction_sum"],
        "reaction_section_concat": ["reactant_section", "agent_section", "product_section"],
        "reaction_section_concat_delta": [
            "reactants",
            "agents",
            "products",
            "delta_product_minus_reactant",
        ],
        "bh_role_separated": list(CANONICAL_ROLE_NAMES),
        "bh_role_separated_delta": [
            *CANONICAL_ROLE_NAMES,
            "delta_product_minus_reactant_1",
            "delta_product_minus_reactant_2",
            "delta_product_minus_reactant_pair",
        ],
    }
    return [f"{section}__morgan_{bit}" for section in sections[kind] for bit in range(n_bits)]


def _feature_block_names(kind: str) -> list[str]:
    names = _reaction_feature_names(kind, 1)
    return [name.removesuffix("__morgan_0") for name in names]


def _resolve_fingerprint_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in {"auto", "rdkit", "hash"}:
        raise ValueError("fingerprint_backend must be one of: auto, rdkit, hash.")
    if normalized != "auto":
        return normalized
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return "hash"
    return "rdkit"


def _fingerprint_width(column: str, n_bits: int) -> int:
    return n_bits


def _resolve_smiles_columns(df: pd.DataFrame, feature_config: dict[str, Any]) -> list[str]:
    configured = list(feature_config.get("smiles_columns", []))
    if configured:
        return [column for column in configured if _usable_smiles_column(df, column)]

    return []


def _usable_smiles_column(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    if column == "reaction_smiles":
        return True
    values = df[column].fillna("UNKNOWN").astype(str).str.strip()
    return bool(values.ne("UNKNOWN").any())


@contextmanager
def _quiet_rdkit_errors(enabled: bool):
    if not enabled:
        yield
        return

    try:
        from rdkit import RDLogger
    except ImportError:
        yield
        return

    RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.error")
