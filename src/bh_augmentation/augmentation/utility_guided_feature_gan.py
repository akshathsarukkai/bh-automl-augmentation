"""Utility-guided feature-space generator augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

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
        scored = score_and_filter_feature_candidates(
            generated,
            X_gen_student,
            teachers,
            config,
        )
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
) -> pd.DataFrame:
    """Pseudo-label, score, and filter generated feature vectors."""
    if candidates.empty:
        return candidates.copy()
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
    nearest_similarity, nearest_index = _nearest_cosine_similarity(X_candidates, X_real)
    scored["nearest_train_similarity"] = nearest_similarity
    scored["nearest_real_index"] = nearest_index
    filters = config.get("filters", {})
    if not filters.get("enabled", True):
        return scored.reset_index(drop=True)
    keep = scored["nearest_train_similarity"].ge(float(filters.get("min_nearest_similarity", 0.6)))
    if filters.get("remove_near_duplicates", True):
        keep &= scored["nearest_train_similarity"].le(float(filters.get("max_nearest_similarity", 0.995)))
    keep &= scored["teacher_std_prediction"].le(float(filters.get("max_teacher_std", 12.0)))
    keep &= scored["teacher_prediction_range"].le(float(filters.get("max_prediction_range", 35.0)))
    filtered = scored.loc[keep].copy()
    cap = int(config.get("diversity", {}).get("max_synthetic_per_nearest_real", 5))
    if cap > 0 and not filtered.empty:
        filtered = filtered.groupby("nearest_real_index", group_keys=False).head(cap)
    return filtered.reset_index(drop=True)


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


def _nearest_cosine_similarity(X: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-8)
    scores = X_norm @ ref_norm.T
    nearest = scores.argmax(axis=1)
    return scores[np.arange(len(X)), nearest], nearest


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
) -> dict[str, Any]:
    return {
        "X_train_augmented": X_aug.astype(np.float32),
        "y_train_augmented": y_aug.astype(np.float32),
        "sample_weight": sample_weight,
        "synthetic_metadata": synthetic_metadata.reset_index(drop=True),
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
    ), metadata


def _mean_or_nan(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").mean())


def _std_or_nan(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").std(ddof=0))
