"""Reaction-aware condition recombination with teacher pseudo-labeling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    REQUIRED_SYNTHETIC_RANKING_FIELDS,
    REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
    apply_filter_rejection,
    assert_accepted_identity_invariants,
    audit_candidate_identities,
    canonical_candidate_record,
    measured_canonical_keys,
)
from bh_augmentation.data.reaction_roles import (
    ReactionRoles,
    ensure_reaction_role_columns,
    reaction_roles_from_row,
)
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model

CANDIDATE_AUDIT_ATTR = "synthetic_candidate_audit"


def parse_reaction_smiles(rxn: str) -> dict[str, Any]:
    """Parse an MVP reaction string into substrate, condition, and product tokens."""
    invalid = {
        "valid": False,
        "left_tokens": [],
        "substrate_tokens": [],
        "condition_tokens": [],
        "product": "",
    }
    if not isinstance(rxn, str) or not rxn.strip() or rxn.count(">>") != 1:
        return invalid
    left, product = (part.strip() for part in rxn.split(">>", 1))
    left_tokens = [token.strip() for token in left.split(".") if token.strip()]
    if len(left_tokens) < 2 or not product:
        return invalid
    return {
        "valid": True,
        "left_tokens": left_tokens,
        "substrate_tokens": left_tokens[:2],
        "condition_tokens": left_tokens[2:],
        "product": product,
    }


def build_reaction_smiles(
    substrate_tokens: list[str],
    condition_tokens: list[str],
    product: str,
) -> str:
    """Build `substrates.conditions>>product` from parsed reaction sections."""
    left = ".".join([*substrate_tokens, *condition_tokens])
    return f"{left}>>{product}"


def build_condition_recombined_roles(
    source_row: pd.Series | dict[str, Any],
    donor_row: pd.Series | dict[str, Any],
) -> ReactionRoles:
    """Preserve source reactants/product and transfer all four donor conditions."""
    source = reaction_roles_from_row(source_row)
    donor = reaction_roles_from_row(donor_row)
    return ReactionRoles(
        reactant_1=source.reactant_1,
        reactant_2=source.reactant_2,
        catalyst=donor.catalyst,
        ligand=donor.ligand,
        base=donor.base,
        solvent_or_additive=donor.solvent_or_additive,
        product=source.product,
    )


def generate_condition_recombined_candidates(
    train_df: pd.DataFrame,
    synthetic_multiplier: float = 1.0,
    max_synthetic_rows: int | None = 3000,
    random_state: int = 42,
    min_condition_tokens: int = 1,
    *,
    feature_config: dict[str, Any] | None = None,
    measured_identity_keys: Iterable[str] = (),
    max_candidates_per_source: int | None = None,
) -> pd.DataFrame:
    """Generate canonical candidates by transferring all typed donor conditions.

    Supplying ``feature_config`` enables the complete scientific identity gate.
    The returned rows are accepted candidates only; all attempted candidates,
    including rejected ones, remain available in ``DataFrame.attrs`` under
    :data:`CANDIDATE_AUDIT_ATTR`.
    """
    if synthetic_multiplier < 0:
        raise ValueError("synthetic_multiplier must be non-negative.")
    if max_synthetic_rows is not None and max_synthetic_rows < 0:
        raise ValueError("max_synthetic_rows must be non-negative or None.")
    if min_condition_tokens < 0:
        raise ValueError("min_condition_tokens must be non-negative.")
    if max_candidates_per_source is not None and max_candidates_per_source <= 0:
        raise ValueError("max_candidates_per_source must be greater than zero.")
    if "reaction_smiles" not in train_df.columns:
        raise ValueError("condition recombination requires reaction_smiles.")

    role_train = ensure_reaction_role_columns(train_df, parse_if_missing=True)
    role_train = role_train.reset_index(drop=False).rename(
        columns={"index": "_source_dataframe_index"}
    )
    if "source_row_id" not in role_train.columns:
        role_train["source_row_id"] = [
            _stable_row_id(role_train.iloc[position])
            for position in range(len(role_train))
        ]
    role_train["_stable_row_key"] = [
        _stable_training_row_key(role_train.iloc[position])
        for position in range(len(role_train))
    ]
    role_train = role_train.sort_values(
        "_stable_row_key",
        kind="mergesort",
    ).drop(columns="_stable_row_key").reset_index(drop=True)

    target = int(synthetic_multiplier * len(train_df))
    if max_synthetic_rows is not None:
        target = min(target, max_synthetic_rows)
    if target <= 0 or train_df.empty:
        return _empty_candidates(role_train)

    parsed = [parse_reaction_smiles(value) for value in role_train["reaction_smiles"]]
    source_positions = [index for index, item in enumerate(parsed) if item["valid"]]
    donor_positions = [
        index
        for index, item in enumerate(parsed)
        if item["valid"] and len(item["condition_tokens"]) >= min_condition_tokens
    ]
    if not source_positions or not donor_positions:
        return _empty_candidates(role_train)

    rng = np.random.default_rng(random_state)
    measured_keys = measured_canonical_keys(
        role_train,
        additional_keys=measured_identity_keys,
    )
    provisional_generated_keys: set[str] = set()
    provisional_by_source: dict[str, set[str]] = {}
    rows: list[pd.Series] = []
    max_attempts = max(100, target * 30)
    configured_source_cap = (
        int(max_candidates_per_source)
        if max_candidates_per_source is not None
        else max(1, int(np.ceil(target / len(source_positions))))
    )

    for attempt in range(max_attempts):
        if feature_config is None and len(rows) >= target:
            break
        if (
            feature_config is not None
            and len(provisional_generated_keys) >= max(1, target * 4)
            and all(
                len(provisional_by_source.get(str(role_train.iloc[position]["source_row_id"]), set()))
                >= configured_source_cap
                for position in source_positions
            )
        ):
            break
        source_position = int(source_positions[attempt % len(source_positions)])
        donor_position = int(rng.choice(donor_positions))
        if source_position == donor_position and len(donor_positions) > 1:
            continue
        synthetic_roles = build_condition_recombined_roles(
            role_train.iloc[source_position],
            role_train.iloc[donor_position],
        )
        synthetic_record = canonical_candidate_record(synthetic_roles)
        key = synthetic_record["canonical_reaction_key"]
        if feature_config is None and (
            key is None
            or key in measured_keys
            or key in provisional_generated_keys
        ):
            continue

        source_row = role_train.iloc[source_position].copy()
        source_id = _reaction_id(role_train.iloc[source_position], source_position)
        donor_id = _reaction_id(role_train.iloc[donor_position], donor_position)
        source_row["candidate_id"] = int(attempt)
        source_row["source_position"] = int(source_position)
        source_row["donor_position"] = int(donor_position)
        source_row["source_index"] = role_train.iloc[source_position][
            "_source_dataframe_index"
        ]
        source_row["donor_index"] = role_train.iloc[donor_position][
            "_source_dataframe_index"
        ]
        source_row["reaction_id"] = f"synthetic_{attempt + 1:06d}"
        source_row["yield"] = np.nan
        source_row["is_synthetic"] = True
        source_row["is_augmented"] = True
        source_row["augmentation_method"] = "condition_recombine_pseudolabel"
        source_row["augmentation_type"] = "condition_recombine_pseudolabel"
        source_row["source_reaction_id"] = source_id
        source_row["donor_reaction_id"] = donor_id
        source_row["pseudo_label"] = True
        for column, value in synthetic_record.items():
            source_row[column] = value
        rows.append(source_row)
        if (
            isinstance(key, str)
            and key
            and key not in measured_keys
        ):
            provisional_generated_keys.add(key)
            provisional_by_source.setdefault(str(source_id), set()).add(key)

    if not rows:
        return _empty_candidates(role_train)

    candidates = pd.DataFrame(rows).reset_index(drop=True)
    if feature_config is None:
        return candidates

    features = _features_only(candidates, feature_config)
    audited = audit_candidate_identities(
        candidates,
        features,
        measured_keys=measured_keys,
        source_rows=role_train.to_dict(orient="records"),
    )
    _attach_recombination_ranking_audit(
        audited,
        features,
        role_train,
        feature_config,
    )
    ranked_positions = _rank_accepted_candidate_indices(audited)
    audited["candidate_rank"] = pd.Series(pd.NA, index=audited.index, dtype="Int64")
    source_counts: dict[str, int] = {}
    per_source_rejected = np.zeros(len(audited), dtype=bool)
    for rank, index in enumerate(ranked_positions, start=1):
        audited.at[index, "candidate_rank"] = rank
        source_id = str(audited.at[index, "source_row_id"])
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        audited.at[index, "source_candidate_ordinal"] = source_counts[source_id]
        if source_counts[source_id] > configured_source_cap:
            per_source_rejected[index] = True
    audited = apply_filter_rejection(
        audited,
        per_source_rejected,
        "per_source_cap",
    )
    ranked_positions = _rank_accepted_candidate_indices(audited)
    over_budget = audited["rejection_reason"].isna().to_numpy()
    over_budget[ranked_positions[:target]] = False
    audited = apply_filter_rejection(
        audited,
        over_budget,
        "candidate_budget_exceeded",
    )
    audited["max_candidates_per_source"] = configured_source_cap
    audited["accepted"] = audited["rejection_reason"].isna()
    assert_accepted_identity_invariants(audited)
    accepted = audited.loc[_rank_accepted_candidate_indices(audited)].reset_index(
        drop=True
    )
    return _with_candidate_audit(accepted, audited)


def filter_candidates_by_nearest_neighbor_similarity(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict[str, Any],
    min_similarity: float = 0.3,
) -> pd.DataFrame:
    """Keep candidates whose binary fingerprint is sufficiently similar to real training data."""
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1.")
    if candidate_df.empty:
        result = candidate_df.copy()
        result["nearest_train_similarity"] = pd.Series(dtype=float)
        if CANDIDATE_AUDIT_ATTR in candidate_df.attrs:
            result.attrs[CANDIDATE_AUDIT_ATTR] = candidate_df.attrs[
                CANDIDATE_AUDIT_ATTR
            ].copy()
        return result

    train_x = _features_only(train_df, feature_config)
    candidate_x = _features_only(candidate_df, feature_config)
    train_binary = sparse.csr_matrix(train_x != 0, dtype=np.float32)
    candidate_binary = sparse.csr_matrix(candidate_x != 0, dtype=np.float32)
    train_counts = np.asarray(train_binary.sum(axis=1)).reshape(-1)

    similarities: list[float] = []
    for row_index in range(candidate_binary.shape[0]):
        candidate = candidate_binary.getrow(row_index)
        intersections = (train_binary @ candidate.T).toarray().reshape(-1)
        candidate_count = float(candidate.sum())
        unions = train_counts + candidate_count - intersections
        scores = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        similarities.append(float(scores.max()) if len(scores) else 0.0)

    result = candidate_df.copy()
    result["nearest_train_similarity"] = similarities
    similarity_rejected = result["nearest_train_similarity"].lt(min_similarity)
    if "rejection_reason" not in result.columns:
        return result.loc[~similarity_rejected].reset_index(drop=True)

    result = apply_filter_rejection(
        result,
        similarity_rejected.to_numpy(dtype=bool),
        "nearest_train_similarity_below_threshold",
    )
    result["accepted"] = result["rejection_reason"].isna()
    assert_accepted_identity_invariants(result)
    audit = _updated_candidate_audit(result)
    accepted = result.loc[result["accepted"]].reset_index(drop=True)
    return _with_candidate_audit(accepted, audit)


def pseudo_label_candidates(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict[str, Any],
    teacher_model_name: str = "random_forest",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a teacher on real rows only and predict clipped synthetic yields."""
    if candidate_df.empty:
        result = candidate_df.copy()
        result["pseudo_label_model"] = pd.Series(dtype=object)
        if CANDIDATE_AUDIT_ATTR in candidate_df.attrs:
            result.attrs[CANDIDATE_AUDIT_ATTR] = candidate_df.attrs[
                CANDIDATE_AUDIT_ATTR
            ].copy()
        return result

    train_x, train_y, _ = build_feature_matrix(train_df, feature_config)
    teacher = train_model(
        get_model(teacher_model_name, seed=random_state),
        train_x,
        train_y,
    )
    candidate_x = _features_only(candidate_df, feature_config)
    predictions = np.clip(predict_model(teacher, candidate_x), 0.0, 100.0)

    result = candidate_df.copy()
    result["yield"] = predictions
    result["pseudo_label_model"] = teacher_model_name
    result["pseudo_label"] = True
    if CANDIDATE_AUDIT_ATTR in candidate_df.attrs:
        result.attrs[CANDIDATE_AUDIT_ATTR] = candidate_df.attrs[
            CANDIDATE_AUDIT_ATTR
        ].copy()
    return result


def condition_recombine_pseudolabel(
    train_df: pd.DataFrame,
    feature_config: dict[str, Any],
    synthetic_multiplier: float = 1.0,
    max_synthetic_rows: int | None = 3000,
    teacher_model: str = "random_forest",
    min_neighbor_similarity: float = 0.3,
    random_state: int = 42,
    *,
    measured_identity_keys: Iterable[str] = (),
    max_candidates_per_source: int | None = None,
) -> pd.DataFrame:
    """Return real training rows plus filtered, teacher-labeled recombined reactions."""
    role_train = ensure_reaction_role_columns(train_df, parse_if_missing=True)
    real = role_train.copy()
    real["is_synthetic"] = False
    real["is_augmented"] = False
    real["augmentation_method"] = "none"
    real["augmentation_type"] = "none"
    real["pseudo_label"] = False
    if "source_reaction_id" not in real.columns:
        real["source_reaction_id"] = real.get(
            "reaction_id", pd.Series(real.index, index=real.index)
        ).astype(str)

    candidates = generate_condition_recombined_candidates(
        role_train,
        synthetic_multiplier=synthetic_multiplier,
        max_synthetic_rows=max_synthetic_rows,
        random_state=random_state,
        feature_config=feature_config,
        measured_identity_keys=measured_identity_keys,
        max_candidates_per_source=max_candidates_per_source,
    )
    candidates = filter_candidates_by_nearest_neighbor_similarity(
        role_train,
        candidates,
        feature_config,
        min_similarity=min_neighbor_similarity,
    )
    candidates = pseudo_label_candidates(
        role_train,
        candidates,
        feature_config,
        teacher_model_name=teacher_model,
        random_state=random_state,
    )
    audit = candidates.attrs.get(CANDIDATE_AUDIT_ATTR, pd.DataFrame()).copy()
    combined = pd.concat([real, candidates], ignore_index=True, sort=False)
    combined.attrs[CANDIDATE_AUDIT_ATTR] = audit
    combined.attrs["augmentation_metadata"] = {
        "n_candidates_attempted": int(len(audit)),
        "n_candidates_accepted": int(len(candidates)),
        "rejection_reason_counts": {
            str(reason): int(count)
            for reason, count in audit["rejection_reason"]
            .fillna("accepted")
            .value_counts()
            .sort_index()
            .items()
        }
        if "rejection_reason" in audit
        else {},
    }
    return combined


def assign_synthetic_sample_weights(
    train_df: pd.DataFrame,
    weighting_config: dict[str, Any] | None,
) -> pd.DataFrame:
    """Assign real and similarity-based synthetic sample weights."""
    result = train_df.copy()
    config = weighting_config or {}
    real_weight = float(config.get("real_weight", 1.0))
    minimum = float(config.get("min_synthetic_weight", 0.25))
    maximum = float(config.get("max_synthetic_weight", 0.85))
    if minimum < 0 or maximum < minimum:
        raise ValueError("Synthetic weight bounds must satisfy 0 <= min <= max.")

    synthetic = (
        result.get("is_synthetic", pd.Series(False, index=result.index))
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    result["sample_weight"] = real_weight
    if not bool(config.get("enabled", False)) or not synthetic.any():
        return result
    if str(config.get("synthetic_weight_mode", "nearest_similarity")) != "nearest_similarity":
        raise ValueError("synthetic_weight_mode must be 'nearest_similarity'.")

    similarities = pd.to_numeric(
        result.loc[synthetic, "nearest_train_similarity"], errors="coerce"
    ).fillna(minimum)
    result.loc[synthetic, "sample_weight"] = similarities.clip(minimum, maximum)
    return result


def _features_only(df: pd.DataFrame, feature_config: dict[str, Any]) -> np.ndarray:
    data = df.copy()
    if "yield" not in data.columns:
        data["yield"] = 0.0
    else:
        data["yield"] = pd.to_numeric(data["yield"], errors="coerce").fillna(0.0)
    X, _, _ = build_feature_matrix(data, feature_config)
    return X


def _reaction_id(row: pd.Series, position: int) -> str:
    value = row.get("reaction_id")
    return str(value) if pd.notna(value) else str(position)


def _stable_row_id(row: pd.Series) -> str:
    for column in ("source_row_id", "reaction_id", "canonical_reaction_key"):
        value = row.get(column)
        if pd.notna(value) and str(value):
            return str(value)
    identity = canonical_candidate_record(reaction_roles_from_row(row))
    key = identity.get("canonical_reaction_key")
    if isinstance(key, str) and key:
        return f"canonical:{key}"
    return f"reaction:{str(row.get('reaction_smiles', ''))}"


def _stable_training_row_key(row: pd.Series) -> str:
    return "\0".join(
        [
            _stable_row_id(row),
            str(row.get("canonical_reaction_key", "")),
            str(row.get("reaction_smiles", "")),
            f"{float(pd.to_numeric(row.get('yield'), errors='coerce')):.17g}",
        ]
    )


def _attach_recombination_ranking_audit(
    candidates: pd.DataFrame,
    features: np.ndarray,
    role_train: pd.DataFrame,
    feature_config: dict[str, Any],
) -> None:
    parsed_train = [
        parse_reaction_smiles(value) for value in role_train["reaction_smiles"]
    ]
    substrate_scores: list[float] = []
    product_scores: list[float] = []
    condition_scores: list[float] = []
    for row in candidates.itertuples():
        source = parsed_train[int(row.source_position)]
        donor = parsed_train[int(row.donor_position)]
        substrate_scores.append(
            _token_jaccard(source["substrate_tokens"], donor["substrate_tokens"])
        )
        product_scores.append(_token_jaccard([source["product"]], [donor["product"]]))
        condition_scores.append(
            _token_jaccard(source["condition_tokens"], donor["condition_tokens"])
        )
    candidates["substrate_similarity"] = substrate_scores
    candidates["product_similarity"] = product_scores
    candidates["condition_similarity"] = condition_scores
    candidates["nontransferred_role_similarity"] = (
        candidates["substrate_similarity"] + candidates["product_similarity"]
    ) / 2.0
    candidates["overall_similarity"] = (
        candidates["substrate_similarity"]
        + candidates["product_similarity"]
        + candidates["condition_similarity"]
    ) / 3.0
    candidates["relevant_context_similarity"] = candidates[
        "nontransferred_role_similarity"
    ]

    train_features = _features_only(role_train, feature_config)
    nearest_similarity = _nearest_binary_tanimoto(features, train_features)
    candidates["nearest_train_similarity"] = nearest_similarity
    candidates["nearest_training_support_distance"] = 1.0 - nearest_similarity
    candidates["out_of_support_distance"] = candidates[
        "nearest_training_support_distance"
    ]
    candidates["support_distance_metric"] = "binary_tanimoto_distance"
    candidates["support_distance_backend"] = (
        "configured_feature_matrix_nonzero_bits"
    )
    candidates["diversity_contribution"] = _static_binary_tanimoto_diversity(features)

    source_yield = np.asarray(
        [
            pd.to_numeric(
                role_train.iloc[int(position)].get("yield"),
                errors="coerce",
            )
            for position in candidates["source_position"]
        ],
        dtype=float,
    )
    donor_yield = np.asarray(
        [
            pd.to_numeric(
                role_train.iloc[int(position)].get("yield"),
                errors="coerce",
            )
            for position in candidates["donor_position"]
        ],
        dtype=float,
    )
    train_yield = pd.to_numeric(role_train["yield"], errors="coerce").to_numpy(dtype=float)
    scale = max(float(np.nanstd(train_yield)), 1.0)
    proxy = np.abs(source_yield - donor_yield) / scale
    proxy = np.where(np.isfinite(proxy), proxy, candidates["out_of_support_distance"])
    candidates["calibrated_uncertainty"] = np.nan
    candidates["uncertainty_rank_value"] = proxy
    candidates["uncertainty_rank_basis"] = (
        "source_donor_label_gap_normalized_uncalibrated_proxy_phase12_pending"
    )


def _token_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(value) for value in left if str(value)}
    right_set = {str(value) for value in right if str(value)}
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def _nearest_binary_tanimoto(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
) -> np.ndarray:
    candidates = np.asarray(candidate_features) != 0
    reference = np.asarray(reference_features) != 0
    result = np.zeros(len(candidates), dtype=float)
    reference_counts = reference.sum(axis=1)
    for start in range(0, len(candidates), 128):
        stop = min(start + 128, len(candidates))
        intersections = candidates[start:stop].astype(np.float32) @ reference.T.astype(np.float32)
        unions = (
            candidates[start:stop].sum(axis=1, keepdims=True)
            + reference_counts[None, :]
            - intersections
        )
        scores = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        result[start:stop] = np.max(scores, axis=1)
    return result


def _static_binary_tanimoto_diversity(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features) != 0
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    counts = values.sum(axis=1)
    nearest = np.full(len(values), -np.inf, dtype=float)
    reference = values.T.astype(np.float32)
    for start in range(0, len(values), 128):
        stop = min(start + 128, len(values))
        intersections = values[start:stop].astype(np.float32) @ reference
        unions = counts[start:stop, None] + counts[None, :] - intersections
        scores = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest[start:stop] = np.max(scores, axis=1)
    return 1.0 - nearest


def _rank_accepted_candidate_indices(candidates: pd.DataFrame) -> list[int]:
    accepted = candidates.loc[candidates["rejection_reason"].isna()].copy()
    if accepted.empty:
        return []
    accepted["_stable_key"] = accepted["canonical_reaction_key"].fillna("~").astype(str)
    accepted["_stable_source"] = accepted["source_row_id"].fillna("").astype(str)
    accepted["_stable_donor"] = accepted["donor_row_id"].fillna("").astype(str)
    ranked = accepted.sort_values(
        [
            "uncertainty_rank_value",
            "relevant_context_similarity",
            "diversity_contribution",
            "out_of_support_distance",
            "_stable_key",
            "_stable_source",
            "_stable_donor",
        ],
        ascending=[True, False, False, True, True, True, True],
        kind="mergesort",
    )
    return [int(index) for index in ranked.index]


def _with_candidate_audit(result: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    result.attrs[CANDIDATE_AUDIT_ATTR] = audit.reset_index(drop=True).copy()
    return result


def _updated_candidate_audit(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Merge later filter fields into the complete generation audit."""
    prior = candidate_df.attrs.get(CANDIDATE_AUDIT_ATTR)
    if not isinstance(prior, pd.DataFrame) or prior.empty:
        return candidate_df.reset_index(drop=True).copy()
    if "candidate_id" not in prior.columns or "candidate_id" not in candidate_df.columns:
        return prior.copy()

    updated = prior.set_index("candidate_id", drop=False).copy()
    current = candidate_df.set_index("candidate_id", drop=False)
    shared_ids = updated.index.intersection(current.index)
    for column in current.columns:
        if column not in updated.columns:
            updated[column] = pd.Series(
                index=updated.index,
                dtype=current[column].dtype,
            )
        updated.loc[shared_ids, column] = current.loc[shared_ids, column]
    return updated.reset_index(drop=True)


def _empty_candidates(train_df: pd.DataFrame) -> pd.DataFrame:
    result = train_df.iloc[0:0].copy()
    for column, dtype in {
        "is_synthetic": bool,
        "is_augmented": bool,
        "augmentation_method": object,
        "source_reaction_id": object,
        "donor_reaction_id": object,
        "pseudo_label": bool,
        "candidate_id": int,
        "source_position": int,
        "donor_position": int,
        "source_index": object,
        "donor_index": object,
        "accepted": bool,
        "substrate_similarity": float,
        "product_similarity": float,
        "nontransferred_role_similarity": float,
        "condition_similarity": float,
        "overall_similarity": float,
        "nearest_training_support_distance": float,
        "calibrated_uncertainty": float,
        "uncertainty_rank_value": float,
        "uncertainty_rank_basis": object,
        "support_distance_metric": object,
        "support_distance_backend": object,
    }.items():
        result[column] = pd.Series(dtype=dtype)
    for column in REQUIRED_SYNTHETIC_AUDIT_FIELDS:
        if column not in result.columns:
            result[column] = pd.Series(dtype=object)
    for column in (
        *REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
        *REQUIRED_SYNTHETIC_RANKING_FIELDS,
    ):
        if column not in result.columns:
            result[column] = pd.Series(dtype=object)
    result.attrs[CANDIDATE_AUDIT_ATTR] = result.copy()
    return result
