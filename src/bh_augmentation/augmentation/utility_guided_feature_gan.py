"""Utility-guided feature-space generator augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from bh_augmentation.augmentation.candidate_scope import (
    REPRESENTATION_AUGMENTATION,
)
from bh_augmentation.augmentation.synthetic_identity import (
    REQUIRED_SYNTHETIC_AUDIT_FIELDS,
    REQUIRED_SYNTHETIC_RANKING_FIELDS,
    REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
    configured_feature_hash,
)
from bh_augmentation.evaluation.metrics import rmse
from bh_augmentation.features.compatibility import (
    assert_feature_compatibility,
    coordinate_feature_contract,
)
from bh_augmentation.features.featurize import build_feature_matrix
from bh_augmentation.models.baselines import get_model
from bh_augmentation.models.predict import predict_model
from bh_augmentation.models.train import train_model


@dataclass
class FeatureTransform:
    """Train-only feature transform with optional inverse reconstruction."""

    transformer: TruncatedSVD | None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.transformer is None:
            return X.astype(np.float32)
        return self.transformer.transform(X).astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.transformer is None:
            return Z.astype(np.float32)
        return self.transformer.inverse_transform(Z).astype(np.float32)


def run_utility_guided_feature_gan_augmentation(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_config: dict[str, Any],
    model_name: str,
    config: dict[str, Any],
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate useful synthetic feature rows from real training rows only."""
    _configured_source_cap(config)
    train_df = _stable_training_frame(train_df)
    rng = np.random.default_rng(random_state)
    representation = config.get("representation", {})
    train_student_in_latent_space = bool(
        representation.get("train_student_in_latent_space", False)
    )
    X_real, y_real, _ = build_feature_matrix(train_df, feature_config)
    X_real = X_real.astype(np.float32)
    y_real = y_real.astype(np.float32)
    if len(train_df) < 4:
        return _empty_result(X_real, y_real, config, train_df, reason="too_few_rows")

    transform = fit_feature_transform(X_real, representation, random_state)
    X_student_real = transform.transform(X_real) if train_student_in_latent_space else X_real

    gen_idx, reward_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=float(config.get("utility_guidance", {}).get("inner_valid_size", 0.2)),
        random_state=random_state,
    )
    X_gen_original, y_gen = X_real[gen_idx], y_real[gen_idx]
    X_gen_student = X_student_real[gen_idx]
    X_reward_student, y_reward = X_student_real[reward_idx], y_real[reward_idx]
    Z_gen = transform.transform(X_gen_original)
    cond_gen, conditioning = build_condition_matrix(
        train_df.iloc[gen_idx].reset_index(drop=True),
        y_gen,
        config.get("conditioning", {}),
    )

    generator = train_wgan_gp_generator(
        Z_gen,
        cond_gen,
        config.get("model", {}),
        random_state,
    )
    teachers = _fit_teacher_ensemble(
        X_gen_student,
        y_gen,
        config.get("pseudo_label", {}).get("teachers", []),
    )
    baseline_rmse = _reward_score(X_gen_student, y_gen, X_reward_student, y_reward)

    utility = config.get("utility_guidance", {})
    n_rounds = min(int(utility.get("n_rounds", 5)), int(config.get("max_effective_rounds", 2)))
    candidates_per_round = int(utility.get("candidates_per_round", 1000))
    batch_sizes = [int(size) for size in utility.get("batch_sizes", [16, 32, 64])][:2]
    elite_fraction = float(utility.get("elite_fraction", 0.25))
    target_synthetic = int(
        min(
            int(config.get("max_synthetic_rows", 1000)),
            max(1, float(config.get("synthetic_multiplier", 1.0)) * len(train_df)),
        )
    )

    bin_probabilities = np.ones(conditioning["n_yield_bins"], dtype=float)
    bin_probabilities /= bin_probabilities.sum()
    best_batch: pd.DataFrame | None = None
    best_reward = float("-inf")
    generated_total = 0
    accepted_total = 0
    reward_records: list[dict[str, Any]] = []
    candidate_audits: list[pd.DataFrame] = []
    generated_feature_hashes: set[str] = set()
    generator_source_ids = _row_ids(train_df.iloc[gen_idx])

    for round_index in range(n_rounds):
        generated = generate_feature_candidates(
            generator,
            transform,
            conditioning,
            n_candidates=candidates_per_round,
            model_config=config.get("model", {}),
            bin_probabilities=bin_probabilities,
            rng=rng,
            return_latent=train_student_in_latent_space,
        )
        scored, round_audit = score_and_filter_feature_candidates(
            generated,
            X_gen_student,
            teachers,
            config,
            source_row_ids=generator_source_ids,
            seen_feature_hashes=generated_feature_hashes,
            return_audit=True,
        )
        round_audit["generation_round"] = round_index
        round_audit["candidate_id"] = np.arange(
            generated_total,
            generated_total + len(round_audit),
            dtype=int,
        )
        candidate_audits.append(round_audit)
        generated_total += len(generated)
        accepted_total += len(scored)
        if scored.empty:
            continue

        batch_rewards = []
        for batch_size in batch_sizes:
            if len(scored) == 0:
                continue
            sample_size = min(batch_size, len(scored), target_synthetic)
            sampled = scored.sample(n=sample_size, random_state=random_state + round_index + batch_size)
            reward = compute_batch_reward(
                X_gen_student,
                y_gen,
                X_reward_student,
                y_reward,
                sampled,
                baseline_rmse,
            )
            batch_rewards.append((reward, sampled.copy()))
            reward_records.append(
                {"round": round_index, "batch_size": sample_size, "reward": reward}
            )
            if reward > best_reward:
                best_reward = reward
                best_batch = sampled.copy()

        if batch_rewards:
            elites = select_elite_batches(batch_rewards, elite_fraction)
            bin_probabilities = update_yield_bin_policy(elites, conditioning["n_yield_bins"])

    if best_batch is None or best_batch.empty:
        metadata = _metadata(
            train_df,
            len(gen_idx),
            len(reward_idx),
            generated_total,
            0,
            config,
            baseline_rmse,
            0.0,
            best_reward,
            transform,
            pd.DataFrame(),
            original_n_features=int(X_real.shape[1]),
            student_n_features=int(X_student_real.shape[1]),
        )
        return _package_result(
            X_student_real,
            y_real,
            None,
            pd.DataFrame(),
            metadata,
            transform,
            train_student_in_latent_space,
            candidate_audit=_concat_audits(candidate_audits),
        ), metadata

    final = best_batch.head(target_synthetic).reset_index(drop=True)
    X_synthetic = np.vstack(final["feature_vector"].to_numpy()).astype(np.float32)
    y_synthetic = final["yield"].to_numpy(dtype=np.float32)
    feature_names, feature_metadata = coordinate_feature_contract(
        "utility_guided_gan_feature_space",
        X_student_real.shape[1],
    )
    assert_feature_compatibility(
        X_student_real,
        feature_names,
        X_synthetic,
        feature_names,
        real_metadata=feature_metadata,
        synthetic_metadata=feature_metadata,
    )
    X_aug = np.vstack([X_student_real, X_synthetic]).astype(np.float32)
    y_aug = np.concatenate([y_real, y_synthetic]).astype(np.float32)
    sample_weight = np.ones(len(y_aug), dtype=np.float32)
    metadata = _metadata(
        train_df,
        len(gen_idx),
        len(reward_idx),
        generated_total,
        accepted_total,
        config,
        baseline_rmse,
        float(len(final) / generated_total) if generated_total else 0.0,
        best_reward,
        transform,
        final,
        original_n_features=int(X_real.shape[1]),
        student_n_features=int(X_aug.shape[1]),
    )
    return _package_result(
        X_aug,
        y_aug,
        sample_weight,
        final,
        metadata,
        transform,
        train_student_in_latent_space,
        candidate_audit=_concat_audits(candidate_audits),
    ), metadata


def fit_feature_transform(
    X_train: np.ndarray,
    representation_config: dict[str, Any],
    random_state: int,
) -> FeatureTransform:
    """Fit optional TruncatedSVD on training features only."""
    if not representation_config.get("use_dim_reduction", False):
        return FeatureTransform(None)
    if str(representation_config.get("method", "truncated_svd")) != "truncated_svd":
        raise ValueError("Only truncated_svd representation is supported.")
    max_components = max(1, min(X_train.shape[0] - 1, X_train.shape[1] - 1))
    n_components = min(int(representation_config.get("n_components", 128)), max_components)
    if n_components < 2:
        return FeatureTransform(None)
    transformer = TruncatedSVD(n_components=n_components, random_state=random_state)
    transformer.fit(X_train)
    return FeatureTransform(transformer)


def build_condition_matrix(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    conditioning_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build yield-bin and optional key conditioning from training rows only."""
    n_bins = int(conditioning_config.get("n_yield_bins", 5))
    clipped = np.clip(np.asarray(y_train, dtype=float), 0.0, 100.0)
    bins = np.minimum((clipped / 100.0 * n_bins).astype(int), n_bins - 1)
    parts = [np.eye(n_bins, dtype=np.float32)[bins]]
    if conditioning_config.get("include_yield_value", True):
        parts.append((clipped / 100.0).reshape(-1, 1).astype(np.float32))
    categories: dict[str, list[str]] = {}
    for key, enabled in [
        ("product_key", conditioning_config.get("include_product_key", False)),
        ("reactant_key", conditioning_config.get("include_reactant_key", False)),
    ]:
        if enabled and key in train_df.columns:
            values = train_df[key].fillna("UNKNOWN").astype(str)
            cats = sorted(values.unique())
            categories[key] = cats
            lookup = {value: index for index, value in enumerate(cats)}
            encoded = np.zeros((len(train_df), len(cats)), dtype=np.float32)
            for row, value in enumerate(values):
                encoded[row, lookup[value]] = 1.0
            parts.append(encoded)
    return (
        np.hstack(parts).astype(np.float32),
        {
            "n_yield_bins": n_bins,
            "include_yield_value": bool(
                conditioning_config.get("include_yield_value", True)
            ),
            "categories": categories,
        },
    )


def train_wgan_gp_generator(
    latent_real: np.ndarray,
    conditions: np.ndarray,
    model_config: dict[str, Any],
    random_state: int,
) -> Any:
    """Train a small WGAN-GP-style generator on training latent features."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for utility_guided_feature_gan.") from exc

    torch.manual_seed(random_state)
    latent_dim = latent_real.shape[1]
    cond_dim = conditions.shape[1]
    noise_dim = int(model_config.get("noise_dim", 64))
    hidden_dim = int(model_config.get("hidden_dim", 256))
    batch_size = max(1, min(int(model_config.get("batch_size", 32)), len(latent_real)))
    n_epochs = min(int(model_config.get("n_epochs", 200)), int(model_config.get("max_effective_epochs", 5)))
    critic_steps = min(int(model_config.get("critic_steps", 5)), 2)
    lr = float(model_config.get("learning_rate", 0.0002))
    beta1 = float(model_config.get("beta1", 0.5))
    beta2 = float(model_config.get("beta2", 0.9))
    gp_weight = float(model_config.get("gradient_penalty", 10.0))

    generator = _Generator(noise_dim, cond_dim, hidden_dim, latent_dim)
    critic = _Critic(latent_dim, cond_dim, hidden_dim)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    real = torch.tensor(latent_real, dtype=torch.float32)
    cond = torch.tensor(conditions, dtype=torch.float32)

    for _ in range(max(1, n_epochs)):
        indices = torch.randperm(len(real))
        for start in range(0, len(real), batch_size):
            batch_idx = indices[start : start + batch_size]
            real_batch = real[batch_idx]
            cond_batch = cond[batch_idx]
            for _ in range(critic_steps):
                z = torch.randn(len(real_batch), noise_dim)
                fake = generator(z, cond_batch).detach()
                loss_c = critic(fake, cond_batch).mean() - critic(real_batch, cond_batch).mean()
                loss_c = loss_c + gp_weight * _gradient_penalty(critic, real_batch, fake, cond_batch)
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
            z = torch.randn(len(real_batch), noise_dim)
            fake = generator(z, cond_batch)
            loss_g = -critic(fake, cond_batch).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
    generator.eval()
    generator.noise_dim = noise_dim
    return generator


def generate_feature_candidates(
    generator: Any,
    transform: FeatureTransform,
    conditioning: dict[str, Any],
    n_candidates: int,
    model_config: dict[str, Any],
    bin_probabilities: np.ndarray,
    rng: np.random.Generator,
    return_latent: bool = False,
) -> pd.DataFrame:
    """Sample generated feature candidates in latent or reconstructed feature space."""
    import torch

    n_bins = int(conditioning["n_yield_bins"])
    bins = rng.choice(np.arange(n_bins), size=n_candidates, p=bin_probabilities)
    cond_parts = [np.eye(n_bins, dtype=np.float32)[bins]]
    if bool(conditioning.get("include_yield_value", True)):
        cond_parts.append(((bins + 0.5) / n_bins).reshape(-1, 1).astype(np.float32))
    for categories in conditioning.get("categories", {}).values():
        if categories:
            sampled = rng.integers(0, len(categories), size=n_candidates)
            cond_parts.append(np.eye(len(categories), dtype=np.float32)[sampled])
    cond = np.hstack(cond_parts).astype(np.float32)
    noise = torch.randn(n_candidates, int(getattr(generator, "noise_dim", model_config.get("noise_dim", 64))))
    with torch.no_grad():
        latent = generator(noise, torch.tensor(cond, dtype=torch.float32)).numpy()
    X_generated = latent.astype(np.float32) if return_latent else transform.inverse_transform(latent).astype(np.float32)
    return pd.DataFrame(
        {
            "feature_vector": list(X_generated),
            "yield_bin": bins.astype(int),
        }
    )


def score_and_filter_feature_candidates(
    candidates: pd.DataFrame,
    X_real: np.ndarray,
    teachers: list[Any],
    config: dict[str, Any],
    *,
    source_row_ids: Sequence[str] | None = None,
    seen_feature_hashes: set[str] | None = None,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Pseudo-label, score, filter, and identity-audit generated coordinates.

    Generated feature vectors are development controls, not molecular records.
    Their exact coordinate hashes are deduplicated separately while chemical
    identity fields remain explicitly unavailable.
    """
    cap = _configured_source_cap(config)
    if candidates.empty:
        empty = _empty_feature_candidate_audit(candidates)
        return (empty, empty.copy()) if return_audit else empty
    X_candidates = np.vstack(candidates["feature_vector"].to_numpy()).astype(np.float32)
    teacher_predictions = np.vstack([predict_model(model, X_candidates) for model in teachers])
    scored = candidates.copy()
    scored["teacher_mean_prediction"] = teacher_predictions.mean(axis=0)
    scored["teacher_std_prediction"] = teacher_predictions.std(axis=0, ddof=0)
    scored["teacher_prediction_range"] = teacher_predictions.max(axis=0) - teacher_predictions.min(axis=0)
    pseudo = config.get("pseudo_label", {})
    scored["yield"] = scored["teacher_mean_prediction"].clip(
        float(pseudo.get("clip_min", 0.0)), float(pseudo.get("clip_max", 100.0))
    )
    parent_ids = _normalize_source_ids(source_row_ids, len(X_real))
    nearest_similarity, nearest_index = _nearest_cosine_similarity(
        X_candidates,
        X_real,
        stable_reference_keys=parent_ids,
    )
    scored["nearest_train_similarity"] = nearest_similarity
    scored["nearest_real_index"] = nearest_index
    scored["source_row_id"] = [parent_ids[index] for index in nearest_index]
    scored["donor_row_id"] = None
    scored["canonical_reaction_key"] = None
    scored["canonical_reaction_hash"] = None
    scored["feature_hash"] = [
        configured_feature_hash(vector) for vector in X_candidates
    ]
    scored["source_identical"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    scored["already_measured"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    # Semantic rule: representation_augmentation. A generated feature coordinate carries no
    # seven-role reaction identity, so the chemical-identity gate is
    # inapplicable rather than passed. Chemical duplicate questions stay NULL
    # (not False): they are unanswerable here, which is a different statement
    # from being answered "no". See docs/CANDIDATE_SCOPE.md.
    scored["candidate_scope_mode"] = REPRESENTATION_AUGMENTATION
    scored["duplicate_synthetic"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    scored["chemical_parse_valid"] = False
    scored["identity_classification"] = "feature_space_nonchemical"
    scored["chemical_identity_available"] = False
    scored["scientific_candidate_eligible"] = False
    scored["development_control_only"] = True
    scored["source_id_semantics"] = "nearest_training_support_not_generator_parent"
    scored["donor_id_semantics"] = "not_applicable"
    scored["chemical_identity_limitation"] = (
        "generated_coordinate_has_no_seven_role_reaction_identity"
    )
    scored["substrate_similarity"] = np.nan
    scored["product_similarity"] = np.nan
    scored["nontransferred_role_similarity"] = np.nan
    scored["condition_similarity"] = np.nan
    scored["overall_similarity"] = nearest_similarity
    scored["relevant_context_similarity"] = nearest_similarity
    scored["nearest_training_support_distance"] = 1.0 - nearest_similarity
    scored["out_of_support_distance"] = scored[
        "nearest_training_support_distance"
    ]
    scored["support_distance_metric"] = "cosine_distance"
    scored["support_distance_backend"] = "numpy_generated_feature_coordinates"
    scored["diversity_contribution"] = _static_cosine_diversity(X_candidates)
    scored["calibrated_uncertainty"] = np.nan
    scored["uncertainty_rank_value"] = scored["teacher_std_prediction"]
    scored["uncertainty_rank_basis"] = (
        "teacher_ensemble_std_uncalibrated_proxy_phase12_pending"
    )

    scored["feature_duplicate_synthetic"] = False
    scored["rejection_reason"] = None
    invalid_features = scored["feature_hash"].isna()
    scored.loc[invalid_features, "rejection_reason"] = "invalid_feature_vector"
    scored.loc[
        scored["feature_duplicate_synthetic"] & scored["rejection_reason"].isna(),
        "rejection_reason",
    ] = "feature_duplicate_synthetic"

    filters = config.get("filters", {})
    keep = scored["rejection_reason"].isna()
    if filters.get("enabled", True):
        threshold_keep = scored["nearest_train_similarity"].ge(
            float(filters.get("min_nearest_similarity", 0.6))
        )
        if filters.get("remove_near_duplicates", True):
            threshold_keep &= scored["nearest_train_similarity"].le(
                float(filters.get("max_nearest_similarity", 0.995))
            )
        threshold_keep &= scored["teacher_std_prediction"].le(
            float(filters.get("max_teacher_std", 12.0))
        )
        threshold_keep &= scored["teacher_prediction_range"].le(
            float(filters.get("max_prediction_range", 35.0))
        )
        scored.loc[
            ~threshold_keep & scored["rejection_reason"].isna(),
            "rejection_reason",
        ] = "candidate_filter_rejected"
        keep &= threshold_keep

    # Only candidates surviving their individual filters participate in the
    # accepted-generated feature set. A rejected vector cannot shadow a later,
    # otherwise valid candidate with the same coordinates.
    ranked_indices = _rank_feature_candidate_indices(scored.loc[keep])
    scored["candidate_rank"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
    for rank, index in enumerate(ranked_indices, start=1):
        scored.at[index, "candidate_rank"] = rank

    known_hashes = set(seen_feature_hashes or ())
    for index in ranked_indices:
        feature_hash = scored.at[index, "feature_hash"]
        if feature_hash in known_hashes:
            scored.at[index, "feature_duplicate_synthetic"] = True
            scored.at[index, "rejection_reason"] = "feature_duplicate_synthetic"
            keep.at[index] = False
        else:
            known_hashes.add(feature_hash)

    scored["accepted"] = keep
    source_counts: dict[int, int] = {}
    for index in _rank_feature_candidate_indices(scored.loc[keep]):
        nearest_real_index = int(scored.at[index, "nearest_real_index"])
        source_counts[nearest_real_index] = (
            source_counts.get(nearest_real_index, 0) + 1
        )
        scored.at[index, "source_candidate_ordinal"] = source_counts[
            nearest_real_index
        ]
        if source_counts[nearest_real_index] > cap:
            scored.at[index, "accepted"] = False
            scored.at[index, "rejection_reason"] = "per_source_cap"
            keep.at[index] = False
    filtered = scored.loc[_rank_feature_candidate_indices(scored.loc[keep])].copy()
    if seen_feature_hashes is not None:
        seen_feature_hashes.update(
            str(value)
            for value in scored.loc[scored["accepted"], "feature_hash"]
            if isinstance(value, str)
        )
    filtered = filtered.reset_index(drop=True)
    audit = scored.reset_index(drop=True)
    return (filtered, audit) if return_audit else filtered


def compute_batch_reward(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    synthetic_batch: pd.DataFrame,
    baseline_rmse: float,
) -> float:
    """Return baseline RMSE minus augmented RMSE for one synthetic batch."""
    if synthetic_batch.empty:
        return 0.0
    X_synth = np.vstack(synthetic_batch["feature_vector"].to_numpy()).astype(np.float32)
    y_synth = synthetic_batch["yield"].to_numpy(dtype=np.float32)
    feature_names, feature_metadata = coordinate_feature_contract(
        "utility_guided_gan_feature_space",
        X_train.shape[1],
    )
    assert_feature_compatibility(
        X_train,
        feature_names,
        X_synth,
        feature_names,
        real_metadata=feature_metadata,
        synthetic_metadata=feature_metadata,
    )
    X_aug = np.vstack([X_train, X_synth])
    y_aug = np.concatenate([y_train, y_synth])
    model = train_model(get_model("ridge"), X_aug, y_aug)
    pred = predict_model(model, X_valid)
    return float(baseline_rmse - rmse(y_valid, pred))


def select_elite_batches(
    batch_rewards: list[tuple[float, pd.DataFrame]],
    elite_fraction: float,
) -> list[pd.DataFrame]:
    """Select top-reward synthetic batches for CEM policy updates."""
    if not batch_rewards:
        return []
    n_elite = max(1, int(np.ceil(len(batch_rewards) * elite_fraction)))
    ranked = sorted(batch_rewards, key=lambda item: item[0], reverse=True)
    return [batch for _, batch in ranked[:n_elite]]


def update_yield_bin_policy(elite_batches: list[pd.DataFrame], n_bins: int) -> np.ndarray:
    """Update categorical yield-bin probabilities from elite candidates."""
    counts = np.ones(n_bins, dtype=float)
    for batch in elite_batches:
        for value in batch.get("yield_bin", pd.Series(dtype=int)).astype(int):
            if 0 <= value < n_bins:
                counts[value] += 1.0
    return counts / counts.sum()


def _fit_teacher_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    teachers: list[dict[str, Any]],
) -> list[Any]:
    if not teachers:
        teachers = [{"model": "ridge", "random_state": 0}]
    fitted = []
    for item in teachers:
        model = get_model(str(item.get("model", "ridge")), seed=int(item.get("random_state", 0)))
        fitted.append(train_model(model, X_train, y_train))
    return fitted


def _reward_score(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> float:
    model = train_model(get_model("ridge"), X_train, y_train)
    return rmse(y_valid, predict_model(model, X_valid))


def _nearest_cosine_similarity(
    X: np.ndarray,
    reference: np.ndarray,
    stable_reference_keys: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-8)
    if stable_reference_keys is None:
        stable_order = np.arange(len(reference), dtype=int)
    else:
        stable_order = np.asarray(
            sorted(
                range(len(reference)),
                key=lambda index: str(stable_reference_keys[index]),
            ),
            dtype=int,
        )
    scores = X_norm @ ref_norm[stable_order].T
    nearest_stable = scores.argmax(axis=1)
    nearest = stable_order[nearest_stable]
    return scores[np.arange(len(X)), nearest_stable], nearest


def _static_cosine_diversity(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    nearest = np.full(len(values), -np.inf, dtype=float)
    for start in range(0, len(values), 128):
        stop = min(start + 128, len(values))
        scores = np.clip(normalized[start:stop] @ normalized.T, -1.0, 1.0)
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest[start:stop] = np.max(scores, axis=1)
    return 1.0 - nearest


def _rank_feature_candidate_indices(candidates: pd.DataFrame) -> list[int]:
    if candidates.empty:
        return []
    ranked = candidates.assign(
        _stable_feature_key=candidates["feature_hash"].fillna("~").astype(str),
        _stable_source_key=candidates["source_row_id"].fillna("").astype(str),
    ).sort_values(
        [
            "uncertainty_rank_value",
            "relevant_context_similarity",
            "diversity_contribution",
            "out_of_support_distance",
            "_stable_feature_key",
            "_stable_source_key",
        ],
        ascending=[True, False, False, True, True, True],
        kind="mergesort",
    )
    return [int(index) for index in ranked.index]


class _Generator:  # torch.nn.Module without importing torch at module import time
    def __new__(cls, noise_dim: int, cond_dim: int, hidden_dim: int, latent_dim: int) -> Any:
        import torch
        from torch import nn

        class Generator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(noise_dim + cond_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, latent_dim),
                )

            def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
                return self.net(torch.cat([z, c], dim=1))

        return Generator()


class _Critic:
    def __new__(cls, latent_dim: int, cond_dim: int, hidden_dim: int) -> Any:
        import torch
        from torch import nn

        class Critic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim + cond_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
                return self.net(torch.cat([x, c], dim=1)).reshape(-1)

        return Critic()


def _gradient_penalty(critic: Any, real: Any, fake: Any, cond: Any) -> Any:
    import torch

    alpha = torch.rand(len(real), 1)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated, cond)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def _metadata(
    train_df: pd.DataFrame,
    n_generator_train: int,
    n_reward_valid: int,
    n_generated: int,
    n_accepted: int,
    config: dict[str, Any],
    baseline_rmse: float,
    acceptance_rate: float,
    best_reward: float,
    transform: FeatureTransform,
    synthetic: pd.DataFrame,
    original_n_features: int,
    student_n_features: int,
) -> dict[str, Any]:
    model_config = config.get("model", {})
    representation = config.get("representation", {})
    latent_dim = int(transform.transformer.n_components) if transform.transformer is not None else int(original_n_features)
    return {
        "augmentation_method": "utility_guided_feature_gan",
        "n_real_train": len(train_df),
        "n_generator_train": int(n_generator_train),
        "n_reward_valid": int(n_reward_valid),
        "n_synthetic_train": int(len(synthetic)),
        "n_candidates_generated": int(n_generated),
        "n_candidates_accepted": int(n_accepted),
        "filter_acceptance_rate": float(acceptance_rate),
        "utility_rounds": int(min(config.get("utility_guidance", {}).get("n_rounds", 5), config.get("max_effective_rounds", 2))),
        "best_inner_reward": float(best_reward) if np.isfinite(best_reward) else 0.0,
        "reward_baseline_rmse": float(baseline_rmse),
        "noise_dim": int(model_config.get("noise_dim", 64)),
        "hidden_dim": int(model_config.get("hidden_dim", 256)),
        "svd_components": int(representation.get("n_components", 0)) if transform.transformer is not None else 0,
        "train_student_in_latent_space": bool(
            representation.get("train_student_in_latent_space", False)
        ),
        "latent_dim": latent_dim,
        "original_n_features": int(original_n_features),
        "student_n_features": int(student_n_features),
        "synthetic_multiplier": float(config.get("synthetic_multiplier", 1.0)),
        "identity_classification": "feature_space_nonchemical",
        "chemical_identity_available": False,
        "scientific_candidate_eligible": False,
        "development_control_only": True,
        "mean_teacher_std": _mean_or_nan(synthetic.get("teacher_std_prediction")),
        "mean_prediction_range": _mean_or_nan(synthetic.get("teacher_prediction_range")),
        "mean_nearest_similarity": _mean_or_nan(synthetic.get("nearest_train_similarity")),
        "mean_synthetic_yield": _mean_or_nan(synthetic.get("yield")),
        "std_synthetic_yield": _std_or_nan(synthetic.get("yield")),
    }


def _package_result(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    sample_weight: np.ndarray | None,
    synthetic_metadata: pd.DataFrame,
    metadata: dict[str, Any],
    feature_transform: FeatureTransform,
    train_student_in_latent_space: bool,
    candidate_audit: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "X_train_augmented": X_aug.astype(np.float32),
        "y_train_augmented": y_aug.astype(np.float32),
        "sample_weight": sample_weight,
        "synthetic_metadata": synthetic_metadata.reset_index(drop=True),
        "candidate_audit": (
            candidate_audit.reset_index(drop=True)
            if candidate_audit is not None
            else _empty_feature_candidate_audit()
        ),
        "metadata": metadata,
        "feature_transform": feature_transform,
        "train_student_in_latent_space": train_student_in_latent_space,
    }


def _empty_result(
    X_real: np.ndarray,
    y_real: np.ndarray,
    config: dict[str, Any],
    train_df: pd.DataFrame,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    representation = config.get("representation", {})
    transform = FeatureTransform(None)
    train_student_in_latent_space = bool(
        representation.get("train_student_in_latent_space", False)
    )
    metadata = {
        "augmentation_method": "utility_guided_feature_gan",
        "n_real_train": len(train_df),
        "n_generator_train": len(train_df),
        "n_reward_valid": 0,
        "n_synthetic_train": 0,
        "n_candidates_generated": 0,
        "n_candidates_accepted": 0,
        "filter_acceptance_rate": 0.0,
        "utility_rounds": 0,
        "best_inner_reward": 0.0,
        "train_student_in_latent_space": train_student_in_latent_space,
        "latent_dim": int(X_real.shape[1]),
        "original_n_features": int(X_real.shape[1]),
        "student_n_features": int(X_real.shape[1]),
        "identity_classification": "feature_space_nonchemical",
        "chemical_identity_available": False,
        "scientific_candidate_eligible": False,
        "development_control_only": True,
        "empty_reason": reason,
    }
    return _package_result(
        X_real,
        y_real,
        None,
        pd.DataFrame(),
        metadata,
        transform,
        train_student_in_latent_space,
        candidate_audit=_empty_feature_candidate_audit(),
    ), metadata


def _mean_or_nan(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").mean())


def _row_ids(frame: pd.DataFrame) -> list[str]:
    for column in ("source_row_id", "reaction_id"):
        if column in frame:
            return [str(value) for value in frame[column]]
    return [f"train_position:{index}" for index in frame.index]


def _stable_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    stable_keys: list[str] = []
    for position, (_, row) in enumerate(result.iterrows()):
        row_id = ""
        for column in ("source_row_id", "reaction_id", "canonical_reaction_key"):
            value = row.get(column)
            if pd.notna(value) and str(value):
                row_id = str(value)
                break
        if not row_id:
            row_id = str(row.get("reaction_smiles", ""))
        stable_keys.append(
            "\0".join(
                [
                    row_id,
                    str(row.get("reaction_smiles", "")),
                    f"{float(pd.to_numeric(row.get('yield'), errors='coerce')):.17g}",
                    str(position) if not row_id else "",
                ]
            )
        )
    result["_utility_gan_stable_row_key"] = stable_keys
    return result.sort_values(
        "_utility_gan_stable_row_key",
        kind="mergesort",
    ).drop(columns="_utility_gan_stable_row_key").reset_index(drop=True)


def _configured_source_cap(config: dict[str, Any]) -> int:
    configured = config.get("max_candidates_per_source")
    if configured is None:
        configured = config.get("diversity", {}).get(
            "max_synthetic_per_nearest_real",
            5,
        )
    cap = int(configured)
    if cap <= 0:
        raise ValueError("max_candidates_per_source must be greater than zero.")
    return cap


def _normalize_source_ids(
    source_row_ids: Sequence[str] | None,
    n_rows: int,
) -> list[str]:
    if source_row_ids is None:
        return [f"train_position:{index}" for index in range(n_rows)]
    if len(source_row_ids) != n_rows:
        raise ValueError("source_row_ids must contain one identifier per real row.")
    normalized = [str(value) for value in source_row_ids]
    if any(not value for value in normalized):
        raise ValueError("source_row_ids must not contain empty identifiers.")
    return normalized


def _concat_audits(audits: list[pd.DataFrame]) -> pd.DataFrame:
    if not audits:
        return _empty_feature_candidate_audit()
    return pd.concat(audits, ignore_index=True)


def _empty_feature_candidate_audit(
    base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = base.iloc[0:0].copy() if isinstance(base, pd.DataFrame) else pd.DataFrame()
    generator_fields = (
        "source_row_id",
        "donor_row_id",
        "feature_hash",
        "feature_duplicate_synthetic",
        "accepted",
        "source_candidate_ordinal",
    )
    for column in (
        *generator_fields,
        *REQUIRED_SYNTHETIC_AUDIT_FIELDS,
        *REQUIRED_SYNTHETIC_SUPPORT_FIELDS,
        *REQUIRED_SYNTHETIC_RANKING_FIELDS,
    ):
        if column not in result:
            result[column] = pd.Series(dtype=object)
    return result


def _std_or_nan(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").std(ddof=0))
