"""Role-aware Buchwald-Hartwig condition-transfer augmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from bh_augmentation.data.bh_condition_reader import (
    parse_bh_reaction_smiles,
    recover_condition_fields,
)
from bh_augmentation.features.featurize import morgan_fingerprint
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model

ROLE_COLUMNS = {
    "reactant_1": "recovered_reactant_1_smiles",
    "reactant_2": "recovered_reactant_2_smiles",
    "catalyst": "recovered_catalyst_smiles",
    "ligand": "recovered_ligand_smiles",
    "base": "recovered_base_smiles",
    "solvent_or_additive": "recovered_solvent_or_additive_smiles",
    "product": "recovered_product_smiles",
}
CONDITION_ROLE_COLUMNS = [
    ROLE_COLUMNS["catalyst"],
    ROLE_COLUMNS["ligand"],
    ROLE_COLUMNS["base"],
    ROLE_COLUMNS["solvent_or_additive"],
]
ROLE_TRANSFER_MODES = {
    "catalyst_only": ["catalyst"],
    "ligand_only": ["ligand"],
    "base_only": ["base"],
    "solvent_or_additive_only": ["solvent_or_additive"],
    "catalyst_ligand": ["catalyst", "ligand"],
    "ligand_base": ["ligand", "base"],
    "base_solvent_or_additive": ["base", "solvent_or_additive"],
    "catalyst_ligand_base": ["catalyst", "ligand", "base"],
    "full_condition_block": ["catalyst", "ligand", "base", "solvent_or_additive"],
}
DONOR_STRATEGIES = {"random", "nearest_substrate", "nearest_condition", "high_yield_nearest", "diverse_role_value"}
LABEL_STRATEGIES = {
    "teacher_ensemble",
    "uncertainty_filtered_teacher",
    "source_label",
    "average_source_donor_label",
}
_TEACHER_MODEL_CACHE: dict[tuple[object, ...], list[Any]] = {}
_SIMILARITY_MATRIX_CACHE: dict[tuple[object, ...], np.ndarray] = {}


@dataclass
class RoleAwareConditionTransferConfig:
    """Configuration for role-aware condition-transfer augmentation."""

    role_transfer_mode: str
    donor_strategy: str
    label_strategy: str
    synthetic_multiplier: float
    max_candidates_per_source: int
    teacher_models: list[str]
    max_teacher_std: float | None
    min_similarity: float | None
    clip_y_min: float = 0.0
    clip_y_max: float = 100.0
    random_state: int = 0
    max_resample_attempts: int = 10


def build_role_transferred_reaction_smiles(
    source_row: pd.Series | dict[str, Any],
    donor_row: pd.Series | dict[str, Any],
    role_transfer_mode: str,
) -> dict[str, Any]:
    """Build a synthetic reaction by replacing selected condition roles."""
    if role_transfer_mode not in ROLE_TRANSFER_MODES:
        raise ValueError(f"Unknown role_transfer_mode: {role_transfer_mode}")

    source = _row_mapping(source_row)
    donor = _row_mapping(donor_row)
    _validate_recovered_columns(source)
    _validate_recovered_columns(donor)
    transferred_roles = set(ROLE_TRANSFER_MODES[role_transfer_mode])

    synthetic = {
        "synthetic_reactant_1_smiles": str(source[ROLE_COLUMNS["reactant_1"]]),
        "synthetic_reactant_2_smiles": str(source[ROLE_COLUMNS["reactant_2"]]),
        "synthetic_catalyst_smiles": str(
            donor[ROLE_COLUMNS["catalyst"]]
            if "catalyst" in transferred_roles
            else source[ROLE_COLUMNS["catalyst"]]
        ),
        "synthetic_ligand_smiles": str(
            donor[ROLE_COLUMNS["ligand"]]
            if "ligand" in transferred_roles
            else source[ROLE_COLUMNS["ligand"]]
        ),
        "synthetic_base_smiles": str(
            donor[ROLE_COLUMNS["base"]]
            if "base" in transferred_roles
            else source[ROLE_COLUMNS["base"]]
        ),
        "synthetic_solvent_or_additive_smiles": str(
            donor[ROLE_COLUMNS["solvent_or_additive"]]
            if "solvent_or_additive" in transferred_roles
            else source[ROLE_COLUMNS["solvent_or_additive"]]
        ),
        "synthetic_product_smiles": str(source[ROLE_COLUMNS["product"]]),
    }
    synthetic["synthetic_reaction_smiles"] = (
        f"{synthetic['synthetic_reactant_1_smiles']}."
        f"{synthetic['synthetic_reactant_2_smiles']}."
        f"{synthetic['synthetic_catalyst_smiles']}."
        f"{synthetic['synthetic_ligand_smiles']}."
        f"{synthetic['synthetic_base_smiles']}."
        f"{synthetic['synthetic_solvent_or_additive_smiles']}>>"
        f"{synthetic['synthetic_product_smiles']}"
    )
    synthetic["changed_catalyst"] = synthetic["synthetic_catalyst_smiles"] != str(source[ROLE_COLUMNS["catalyst"]])
    synthetic["changed_ligand"] = synthetic["synthetic_ligand_smiles"] != str(source[ROLE_COLUMNS["ligand"]])
    synthetic["changed_base"] = synthetic["synthetic_base_smiles"] != str(source[ROLE_COLUMNS["base"]])
    synthetic["changed_solvent_or_additive"] = synthetic["synthetic_solvent_or_additive_smiles"] != str(
        source[ROLE_COLUMNS["solvent_or_additive"]]
    )
    return synthetic


def generate_role_aware_condition_transfer_examples(
    df_train: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> dict[str, Any]:
    """Generate role-aware synthetic reactions using only training rows."""
    _validate_config(config)
    _validate_training_frame(df_train)
    train = df_train.reset_index(drop=False).rename(columns={"index": "_source_dataframe_index"})
    X_train_array = np.asarray(X_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(train) != len(X_train_array) or len(train) != len(y_train_array):
        raise ValueError("df_train, X_train, and y_train must contain the same number of rows.")

    n_real_train = len(train)
    target_count = int(math.ceil(max(0.0, config.synthetic_multiplier) * n_real_train))
    if target_count == 0 or n_real_train < 2:
        candidate_df = _empty_candidate_df()
        metadata = _metadata_from_candidates(candidate_df, train, config)
        return _package_result(_empty_synthetic_df(), np.empty(0), metadata, candidate_df, np.empty((0, X_train_array.shape[1])))

    rng = np.random.default_rng(config.random_state)
    substrate_similarity = _role_similarity_matrix(train, ["reactant_1", "reactant_2", "product"])
    condition_similarity = _role_similarity_matrix(train, ["catalyst", "ligand", "base", "solvent_or_additive"])
    role_values = {
        role: train[ROLE_COLUMNS[role]].astype(str).to_numpy()
        for role in ["catalyst", "ligand", "base", "solvent_or_additive"]
    }
    train_records = train.to_dict("records")
    source_dataframe_indices = train["_source_dataframe_index"].to_numpy()
    np.fill_diagonal(substrate_similarity, -np.inf)
    np.fill_diagonal(condition_similarity, -np.inf)

    existing_reactions = set(train["reaction_smiles"].astype(str))
    rows: list[dict[str, Any]] = []
    n_identical_skipped = 0
    n_invalid_role_parse_skipped = 0
    max_candidates = max(1, int(config.max_candidates_per_source)) * n_real_train
    attempts = max(max_candidates, target_count * max(5, int(config.max_resample_attempts)))
    for candidate_id in range(attempts):
        if len(rows) >= max_candidates:
            break
        source_position = int(rng.integers(0, n_real_train))
        donor_position, donor_similarity = _select_donor_position(
            train=train,
            source_position=source_position,
            y_train=y_train_array,
            substrate_similarity=substrate_similarity,
            condition_similarity=condition_similarity,
            role_values=role_values,
            config=config,
            rng=rng,
        )
        if donor_position is None:
            continue

        synthetic = build_role_transferred_reaction_smiles(
            train_records[source_position],
            train_records[donor_position],
            config.role_transfer_mode,
        )
        if synthetic["synthetic_reaction_smiles"] == str(train_records[source_position]["reaction_smiles"]):
            n_identical_skipped += 1
            continue
        if synthetic["synthetic_reaction_smiles"] in existing_reactions:
            n_identical_skipped += 1
            continue
        validation = recover_condition_fields({"reaction_smiles": synthetic["synthetic_reaction_smiles"]})
        if validation.get("condition_parse_status") != "ok":
            n_invalid_role_parse_skipped += 1
            continue

        rows.append(
            {
                "candidate_id": candidate_id,
                "reaction_smiles": synthetic["synthetic_reaction_smiles"],
                "source_position": source_position,
                "donor_position": donor_position,
                "source_index": source_dataframe_indices[source_position],
                "donor_index": source_dataframe_indices[donor_position],
                "source_yield": float(y_train_array[source_position]),
                "donor_yield": float(y_train_array[donor_position]),
                "donor_similarity": float(donor_similarity),
                "teacher_mean": np.nan,
                "teacher_std": np.nan,
                "synthetic_label": np.nan,
                "accepted": False,
                "kept": False,
                **synthetic,
            }
        )

    candidate_df = pd.DataFrame(rows) if rows else _empty_candidate_df()
    if candidate_df.empty:
        metadata = _metadata_from_candidates(
            candidate_df,
            train,
            config,
            n_identical_skipped=n_identical_skipped,
            n_invalid_role_parse_skipped=n_invalid_role_parse_skipped,
        )
        return _package_result(_empty_synthetic_df(), np.empty(0), metadata, candidate_df, np.empty((0, X_train_array.shape[1])))

    X_synthetic = _featurize_synthetic(candidate_df, X_train_array)
    teacher_mean, teacher_std, teachers_used = _teacher_predictions(X_train_array, y_train_array, X_synthetic, config)
    candidate_df["teacher_mean"] = teacher_mean
    candidate_df["teacher_std"] = teacher_std
    candidate_df["teacher_models_used"] = ",".join(teachers_used)
    candidate_df["synthetic_label"] = np.clip(
        _assign_labels(candidate_df, config),
        float(config.clip_y_min),
        float(config.clip_y_max),
    )
    accepted = _acceptance_mask(candidate_df, X_synthetic, config)
    candidate_df["accepted"] = accepted
    kept_indices = np.flatnonzero(accepted)[:target_count]
    candidate_df.loc[kept_indices, "kept"] = True

    synthetic_df = candidate_df.loc[candidate_df["kept"], _synthetic_columns()].copy()
    synthetic_df["yield"] = candidate_df.loc[candidate_df["kept"], "synthetic_label"].to_numpy(dtype=float)
    synthetic_y = synthetic_df["yield"].to_numpy(dtype=float)
    metadata = _metadata_from_candidates(
        candidate_df,
        train,
        config,
        n_identical_skipped=n_identical_skipped,
        n_invalid_role_parse_skipped=n_invalid_role_parse_skipped,
    )
    metadata["teacher_models_used"] = ",".join(teachers_used)
    return _package_result(
        synthetic_df.reset_index(drop=True),
        synthetic_y,
        metadata,
        candidate_df,
        X_synthetic[candidate_df["kept"].to_numpy(dtype=bool)].astype(np.float32),
    )


def _select_donor_position(
    train: pd.DataFrame,
    source_position: int,
    y_train: np.ndarray,
    substrate_similarity: np.ndarray,
    condition_similarity: np.ndarray,
    role_values: dict[str, np.ndarray],
    config: RoleAwareConditionTransferConfig,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    candidates = [position for position in range(len(train)) if position != source_position]
    if config.donor_strategy == "diverse_role_value":
        roles = ROLE_TRANSFER_MODES[config.role_transfer_mode]
        candidates = [
            position
            for position in candidates
            if any(role_values[role][position] != role_values[role][source_position] for role in roles)
        ]
    if not candidates:
        return None, float("nan")

    if config.donor_strategy == "random":
        donor_position = int(rng.choice(candidates))
        return donor_position, float(substrate_similarity[source_position, donor_position])

    similarity = condition_similarity if config.donor_strategy == "nearest_condition" else substrate_similarity
    candidate_array = np.asarray(candidates, dtype=int)
    scores = similarity[source_position, candidate_array]
    if config.min_similarity is not None:
        keep = scores >= float(config.min_similarity)
        candidate_array = candidate_array[keep]
        scores = scores[keep]
    if len(candidate_array) == 0:
        return None, float("nan")

    if config.donor_strategy == "high_yield_nearest":
        order = np.lexsort((-scores, -y_train[candidate_array]))
        donor_position = int(candidate_array[order[0]])
        return donor_position, float(similarity[source_position, donor_position])

    best_index = int(np.argmax(scores))
    donor_position = int(candidate_array[best_index])
    return donor_position, float(scores[best_index])


def _role_similarity_matrix(train: pd.DataFrame, roles: list[str]) -> np.ndarray:
    key = (
        tuple(roles),
        tuple(
            tuple(train[ROLE_COLUMNS[role]].astype(str).tolist())
            for role in roles
        ),
    )
    if key in _SIMILARITY_MATRIX_CACHE:
        return _SIMILARITY_MATRIX_CACHE[key].copy()

    vectors = []
    for _, row in train.iterrows():
        fingerprint = np.zeros(256, dtype=np.float32)
        for role in roles:
            fingerprint += _cached_morgan_fingerprint(str(row[ROLE_COLUMNS[role]]), 2, 256)
        vectors.append(fingerprint)
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    similarity = normalized @ normalized.T
    _SIMILARITY_MATRIX_CACHE[key] = similarity
    return similarity.copy()


def _teacher_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synthetic: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(X_synthetic) == 0:
        return np.empty(0), np.empty(0), []
    predictions: list[np.ndarray] = []
    teachers = _fit_teacher_models(X_train, y_train, config)
    for fitted in teachers:
        predictions.append(np.asarray(predict_model(fitted, X_synthetic), dtype=float))
    if not predictions:
        raise ValueError("At least one teacher model is required for teacher label strategies.")
    stacked = np.vstack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0), list(config.teacher_models)


def _fit_teacher_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> list[Any]:
    key = (
        id(X_train),
        id(y_train),
        tuple(X_train.shape),
        tuple(y_train.shape),
        float(np.sum(y_train)),
        tuple(config.teacher_models),
        int(config.random_state),
    )
    if key not in _TEACHER_MODEL_CACHE:
        fitted_models = []
        for model_name in config.teacher_models:
            model = get_model(model_name, seed=config.random_state)
            fitted_models.append(train_model(model, X_train, y_train))
        _TEACHER_MODEL_CACHE[key] = fitted_models
    return _TEACHER_MODEL_CACHE[key]


def _assign_labels(candidate_df: pd.DataFrame, config: RoleAwareConditionTransferConfig) -> np.ndarray:
    if config.label_strategy in {"teacher_ensemble", "uncertainty_filtered_teacher"}:
        return candidate_df["teacher_mean"].to_numpy(dtype=float)
    if config.label_strategy == "source_label":
        return candidate_df["source_yield"].to_numpy(dtype=float)
    if config.label_strategy == "average_source_donor_label":
        return 0.5 * (
            candidate_df["source_yield"].to_numpy(dtype=float)
            + candidate_df["donor_yield"].to_numpy(dtype=float)
        )
    raise ValueError(f"Unknown label_strategy: {config.label_strategy}")


def _acceptance_mask(
    candidate_df: pd.DataFrame,
    X_synthetic: np.ndarray,
    config: RoleAwareConditionTransferConfig,
) -> np.ndarray:
    finite_features = np.isfinite(X_synthetic).all(axis=1)
    finite_labels = np.isfinite(candidate_df["synthetic_label"].to_numpy(dtype=float))
    accepted = finite_features & finite_labels
    if config.label_strategy == "uncertainty_filtered_teacher" and config.max_teacher_std is not None:
        accepted &= candidate_df["teacher_std"].to_numpy(dtype=float) <= float(config.max_teacher_std)
    return accepted


def _featurize_synthetic(candidate_df: pd.DataFrame, X_train: np.ndarray) -> np.ndarray:
    n_bits = _infer_n_bits(X_train)
    vectors = [
        _reaction_role_concat_vector(str(value), n_bits=n_bits)
        for value in candidate_df["reaction_smiles"].astype(str)
    ]
    return np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, X_train.shape[1]), dtype=np.float32)


def _reaction_role_concat_vector(reaction_smiles: str, n_bits: int) -> np.ndarray:
    parsed = parse_bh_reaction_smiles(reaction_smiles)
    if parsed is None or parsed.get("condition_parse_status") != "ok":
        return np.zeros(3 * n_bits, dtype=np.float32)
    reactants = (
        _cached_morgan_fingerprint(str(parsed["parsed_reactant_1_smiles"]), 2, n_bits)
        + _cached_morgan_fingerprint(str(parsed["parsed_reactant_2_smiles"]), 2, n_bits)
    )
    agents = (
        _cached_morgan_fingerprint(str(parsed["parsed_catalyst_smiles"]), 2, n_bits)
        + _cached_morgan_fingerprint(str(parsed["parsed_ligand_smiles"]), 2, n_bits)
        + _cached_morgan_fingerprint(str(parsed["parsed_base_smiles"]), 2, n_bits)
        + _cached_morgan_fingerprint(str(parsed["parsed_solvent_or_additive_smiles"]), 2, n_bits)
    )
    products = _cached_morgan_fingerprint(str(parsed["parsed_product_smiles"]), 2, n_bits)
    return np.concatenate([reactants, agents, products]).astype(np.float32)


@lru_cache(maxsize=8192)
def _cached_morgan_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    return morgan_fingerprint(smiles, radius=radius, n_bits=n_bits, warn_invalid=False)


def _infer_n_bits(X_train: np.ndarray) -> int:
    if X_train.shape[1] % 3 != 0:
        raise ValueError("Role-aware condition transfer currently expects reaction_role_concat features.")
    return int(X_train.shape[1] // 3)


def _metadata_from_candidates(
    candidate_df: pd.DataFrame,
    train: pd.DataFrame,
    config: RoleAwareConditionTransferConfig,
    n_identical_skipped: int = 0,
    n_invalid_role_parse_skipped: int = 0,
) -> dict[str, Any]:
    kept = candidate_df.loc[candidate_df.get("kept", False).eq(True)] if not candidate_df.empty else candidate_df
    accepted_count = int(candidate_df.get("accepted", pd.Series(dtype=bool)).sum()) if not candidate_df.empty else 0
    teacher_std = kept["teacher_std"].to_numpy(dtype=float) if not kept.empty else np.array([])
    synthetic_y = kept["synthetic_label"].to_numpy(dtype=float) if not kept.empty else np.array([])
    uncertainty_rejected = 0
    if (
        not candidate_df.empty
        and config.label_strategy == "uncertainty_filtered_teacher"
        and config.max_teacher_std is not None
    ):
        uncertainty_rejected = int((candidate_df["teacher_std"].to_numpy(dtype=float) > float(config.max_teacher_std)).sum())
    return {
        "role_transfer_mode": config.role_transfer_mode,
        "donor_strategy": config.donor_strategy,
        "label_strategy": config.label_strategy,
        "synthetic_multiplier": float(config.synthetic_multiplier),
        "min_similarity": np.nan if config.min_similarity is None else float(config.min_similarity),
        "max_teacher_std": np.nan if config.max_teacher_std is None else float(config.max_teacher_std),
        "n_candidates_generated": int(len(candidate_df)),
        "n_candidates_accepted": accepted_count,
        "n_synthetic_train": int(len(kept)),
        "filter_acceptance_rate": float(accepted_count / len(candidate_df)) if len(candidate_df) else 0.0,
        "mean_donor_similarity": _safe_mean(candidate_df.get("donor_similarity", pd.Series(dtype=float))),
        "mean_teacher_std": _safe_mean(teacher_std),
        "mean_synthetic_yield": _safe_mean(synthetic_y),
        "std_synthetic_yield": _safe_std(synthetic_y),
        "n_identical_skipped": int(n_identical_skipped),
        "n_invalid_role_parse_skipped": int(n_invalid_role_parse_skipped),
        "n_teacher_uncertainty_rejected": int(uncertainty_rejected),
        "unique_source_catalysts": int(train[ROLE_COLUMNS["catalyst"]].nunique(dropna=False)),
        "unique_source_ligands": int(train[ROLE_COLUMNS["ligand"]].nunique(dropna=False)),
        "unique_source_bases": int(train[ROLE_COLUMNS["base"]].nunique(dropna=False)),
        "unique_source_solvents": int(train[ROLE_COLUMNS["solvent_or_additive"]].nunique(dropna=False)),
        "unique_synthetic_catalysts": _safe_nunique(kept, "synthetic_catalyst_smiles"),
        "unique_synthetic_ligands": _safe_nunique(kept, "synthetic_ligand_smiles"),
        "unique_synthetic_bases": _safe_nunique(kept, "synthetic_base_smiles"),
        "unique_synthetic_solvents": _safe_nunique(kept, "synthetic_solvent_or_additive_smiles"),
        "changed_catalyst_fraction": _safe_bool_mean(kept, "changed_catalyst"),
        "changed_ligand_fraction": _safe_bool_mean(kept, "changed_ligand"),
        "changed_base_fraction": _safe_bool_mean(kept, "changed_base"),
        "changed_solvent_fraction": _safe_bool_mean(kept, "changed_solvent_or_additive"),
    }


def _package_result(
    synthetic_df: pd.DataFrame,
    synthetic_y: np.ndarray,
    metadata: dict[str, Any],
    candidate_df: pd.DataFrame,
    X_synthetic: np.ndarray,
) -> dict[str, Any]:
    return {
        "synthetic_df": synthetic_df,
        "synthetic_y": synthetic_y,
        "metadata": metadata,
        "candidate_df": candidate_df,
        "X_synthetic": X_synthetic,
    }


def _synthetic_columns() -> list[str]:
    return [
        "reaction_smiles",
        "source_index",
        "donor_index",
        "source_yield",
        "donor_yield",
        "donor_similarity",
        "teacher_mean",
        "teacher_std",
        "synthetic_label",
        "synthetic_reactant_1_smiles",
        "synthetic_reactant_2_smiles",
        "synthetic_catalyst_smiles",
        "synthetic_ligand_smiles",
        "synthetic_base_smiles",
        "synthetic_solvent_or_additive_smiles",
        "synthetic_product_smiles",
        "changed_catalyst",
        "changed_ligand",
        "changed_base",
        "changed_solvent_or_additive",
    ]


def _empty_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_synthetic_columns(), "candidate_id", "source_position", "donor_position", "accepted", "kept"])


def _empty_synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_synthetic_columns(), "yield"])


def _validate_config(config: RoleAwareConditionTransferConfig) -> None:
    if config.role_transfer_mode not in ROLE_TRANSFER_MODES:
        raise ValueError(f"Unknown role_transfer_mode: {config.role_transfer_mode}")
    if config.donor_strategy not in DONOR_STRATEGIES:
        raise ValueError(f"Unknown donor_strategy: {config.donor_strategy}")
    if config.label_strategy not in LABEL_STRATEGIES:
        raise ValueError(f"Unknown label_strategy: {config.label_strategy}")
    if not config.teacher_models and config.label_strategy in {"teacher_ensemble", "uncertainty_filtered_teacher"}:
        raise ValueError("teacher_models must be non-empty for teacher label strategies.")


def _validate_training_frame(df: pd.DataFrame) -> None:
    missing = [column for column in [*ROLE_COLUMNS.values(), "reaction_smiles"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing recovered role columns for role-aware condition transfer: {missing}")


def _validate_recovered_columns(row: dict[str, Any]) -> None:
    missing = [column for column in ROLE_COLUMNS.values() if column not in row]
    if missing:
        raise ValueError(f"Missing recovered role columns for role-aware transfer: {missing}")


def _row_mapping(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return row.to_dict() if isinstance(row, pd.Series) else dict(row)


def _safe_mean(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def _safe_std(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=0)) if len(array) else float("nan")


def _safe_nunique(df: pd.DataFrame, column: str) -> int:
    return int(df[column].nunique(dropna=False)) if column in df and not df.empty else 0


def _safe_bool_mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].astype(bool).mean()) if column in df and not df.empty else float("nan")
