"""Supervised autoencoder representation learning for reaction features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SupervisedAEConfig:
    """Configuration for supervised autoencoder training."""

    input_dim: int
    latent_dim: int
    hidden_dims: list[int]
    dropout: float = 0.0
    reconstruction_weight: float = 1.0
    yield_weight: float = 1.0
    latent_l2_weight: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    random_state: int = 42
    device: str = "auto"


class SupervisedAutoencoder(nn.Module):
    """MLP autoencoder with a yield-prediction head on the latent code."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = _build_mlp(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = _build_mlp(latent_dim, list(reversed(hidden_dims)), input_dim, dropout)
        self.yield_head = nn.Linear(latent_dim, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return latent code, reconstruction, and scaled yield prediction."""
        z = self.encoder(X)
        x_reconstructed = self.decoder(z)
        y_pred_scaled = self.yield_head(z).squeeze(-1)
        return z, x_reconstructed, y_pred_scaled


def fit_supervised_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    config: SupervisedAEConfig,
) -> dict[str, Any]:
    """Fit a supervised autoencoder and return model artifacts.

    Label scaling statistics are fit on `y_train` only. Validation data, when
    provided, is used only for early stopping and history reporting.
    """
    _set_torch_seed(config.random_state)
    device = _resolve_device(config.device)
    X_train_array = _as_float32_matrix(X_train)
    y_train_array = _as_float32_vector(y_train)
    if X_train_array.shape[1] != config.input_dim:
        raise ValueError(
            f"config.input_dim={config.input_dim} does not match X_train width "
            f"{X_train_array.shape[1]}."
        )
    if len(X_train_array) != len(y_train_array):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if len(X_train_array) == 0:
        raise ValueError("Supervised autoencoder requires at least one training row.")

    has_valid = X_valid is not None and y_valid is not None and len(X_valid) > 0
    X_valid_array = _as_float32_matrix(X_valid) if has_valid else None
    y_valid_array = _as_float32_vector(y_valid) if has_valid else None
    if X_valid_array is not None and X_valid_array.shape[1] != config.input_dim:
        raise ValueError("X_valid width must match config.input_dim.")

    y_mean = float(y_train_array.mean())
    y_std = float(y_train_array.std())
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    y_train_scaled = ((y_train_array - y_mean) / y_std).astype(np.float32)
    y_valid_scaled = (
        ((y_valid_array - y_mean) / y_std).astype(np.float32)
        if y_valid_array is not None
        else None
    )

    model = SupervisedAutoencoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=list(config.hidden_dims),
        dropout=float(config.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_array),
        torch.from_numpy(y_train_scaled),
    )
    generator = torch.Generator()
    generator.manual_seed(int(config.random_state))
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(int(config.batch_size), len(train_dataset))),
        shuffle=True,
        generator=generator,
    )

    valid_tensors = (
        torch.from_numpy(X_valid_array).to(device),
        torch.from_numpy(y_valid_scaled).to(device),
    ) if X_valid_array is not None and y_valid_scaled is not None else None

    history: list[dict[str, float | int]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max(1, int(config.max_epochs)) + 1):
        model.train()
        train_totals = _empty_loss_totals()
        n_seen = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, x_reconstructed, y_pred_scaled = model(batch_X)
            losses = _compute_losses(batch_X, batch_y, z, x_reconstructed, y_pred_scaled, config)
            losses["total_loss"].backward()
            optimizer.step()

            batch_size = int(batch_X.shape[0])
            n_seen += batch_size
            _accumulate_losses(train_totals, losses, batch_size)

        train_metrics = _average_loss_totals(train_totals, n_seen)
        if valid_tensors is not None:
            valid_metrics = _evaluate_losses(model, valid_tensors[0], valid_tensors[1], config)
            monitor_loss = valid_metrics["total_loss"]
        else:
            valid_metrics = _nan_loss_metrics()
            monitor_loss = train_metrics["total_loss"]

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "train_reconstruction_loss": train_metrics["reconstruction_loss"],
                "train_yield_loss": train_metrics["yield_loss"],
                "internal_valid_total_loss": valid_metrics["total_loss"],
                "internal_valid_reconstruction_loss": valid_metrics["reconstruction_loss"],
                "internal_valid_yield_loss": valid_metrics["yield_loss"],
            }
        )

        if monitor_loss < best_loss - 1e-8:
            best_loss = float(monitor_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if valid_tensors is not None and epochs_without_improvement >= int(config.patience):
            break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": pd.DataFrame(history),
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "config": asdict(config),
        "device": str(device),
    }


def encode_with_supervised_autoencoder(
    artifacts: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    """Encode feature rows into the learned latent space."""
    model = _artifact_model(artifacts)
    device = torch.device(str(artifacts.get("device", next(model.parameters()).device)))
    X_array = _as_float32_matrix(X)
    with torch.no_grad():
        tensor = torch.from_numpy(X_array).to(device)
        z, _, _ = model(tensor)
    return z.detach().cpu().numpy().astype(np.float32)


def predict_yield_with_supervised_autoencoder(
    artifacts: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    """Predict yields in original units with the supervised AE yield head."""
    model = _artifact_model(artifacts)
    device = torch.device(str(artifacts.get("device", next(model.parameters()).device)))
    y_mean = float(artifacts["y_mean"])
    y_std = float(artifacts["y_std"])
    X_array = _as_float32_matrix(X)
    with torch.no_grad():
        tensor = torch.from_numpy(X_array).to(device)
        _, _, y_pred_scaled = model(tensor)
    return (y_pred_scaled.detach().cpu().numpy().astype(float) * y_std + y_mean).reshape(-1)


def predict_yield_from_latent(
    artifacts: dict[str, Any],
    z: np.ndarray,
) -> np.ndarray:
    """Predict yields in original units from latent vectors with the AE yield head."""
    model = _artifact_model(artifacts)
    device = torch.device(str(artifacts.get("device", next(model.parameters()).device)))
    y_mean = float(artifacts["y_mean"])
    y_std = float(artifacts["y_std"])
    z_array = _as_float32_matrix(z)
    with torch.no_grad():
        tensor = torch.from_numpy(z_array).to(device)
        y_pred_scaled = model.yield_head(tensor).squeeze(-1)
    return (y_pred_scaled.detach().cpu().numpy().astype(float) * y_std + y_mean).reshape(-1)


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        previous_dim = int(hidden_dim)
    layers.append(nn.Linear(previous_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _compute_losses(
    X: torch.Tensor,
    y_scaled: torch.Tensor,
    z: torch.Tensor,
    x_reconstructed: torch.Tensor,
    y_pred_scaled: torch.Tensor,
    config: SupervisedAEConfig,
) -> dict[str, torch.Tensor]:
    reconstruction_loss = nn.functional.mse_loss(x_reconstructed, X)
    yield_loss = nn.functional.mse_loss(y_pred_scaled, y_scaled)
    latent_l2 = z.pow(2).mean()
    total_loss = (
        float(config.reconstruction_weight) * reconstruction_loss
        + float(config.yield_weight) * yield_loss
        + float(config.latent_l2_weight) * latent_l2
    )
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "yield_loss": yield_loss,
        "latent_l2": latent_l2,
    }


def _evaluate_losses(
    model: SupervisedAutoencoder,
    X: torch.Tensor,
    y_scaled: torch.Tensor,
    config: SupervisedAEConfig,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        z, x_reconstructed, y_pred_scaled = model(X)
        losses = _compute_losses(X, y_scaled, z, x_reconstructed, y_pred_scaled, config)
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def _empty_loss_totals() -> dict[str, float]:
    return {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "yield_loss": 0.0,
        "latent_l2": 0.0,
    }


def _accumulate_losses(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for key in totals:
        totals[key] += float(losses[key].detach().cpu()) * batch_size


def _average_loss_totals(totals: dict[str, float], n_seen: int) -> dict[str, float]:
    denominator = max(1, int(n_seen))
    return {key: value / denominator for key, value in totals.items()}


def _nan_loss_metrics() -> dict[str, float]:
    return {
        "total_loss": float("nan"),
        "reconstruction_loss": float("nan"),
        "yield_loss": float("nan"),
        "latent_l2": float("nan"),
    }


def _artifact_model(artifacts: dict[str, Any]) -> SupervisedAutoencoder:
    model = artifacts.get("model")
    if not isinstance(model, SupervisedAutoencoder):
        raise ValueError("artifacts must contain a trained SupervisedAutoencoder under 'model'.")
    model.eval()
    return model


def _as_float32_matrix(X: np.ndarray | Any) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix; got shape {array.shape}.")
    return np.ascontiguousarray(array)


def _as_float32_vector(y: np.ndarray | Any) -> np.ndarray:
    array = np.asarray(y, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(array)


def _resolve_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for supervised AE training but is not available.")
    return resolved


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
