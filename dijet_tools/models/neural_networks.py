import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class PhysicsInformedNet(nn.Module):
    """
    Physics-informed neural network for predicting cos(θ*) with QCD constraints.
    """

    def __init__(self, input_dim: int, hidden_layers: List[int] = [256, 128, 64, 32]):
        """
        Initialize physics-informed neural network.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
        """
        super(PhysicsInformedNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers

        layers = []

        layers.append(nn.BatchNorm1d(input_dim))

        prev_dim = input_dim
        for i, units in enumerate(hidden_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, units),
                    nn.ReLU(),
                    nn.BatchNorm1d(units),
                    nn.Dropout(
                        0.3 * (len(hidden_layers) - i) / len(hidden_layers)
                    ),  # Progressive dropout
                ]
            )
            prev_dim = units

        # Output layer with sigmoid activation (cos θ* ∈ [0,1])
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss function incorporating QCD constraints.
    """

    def __init__(self, physics_weight: float = 0.1):
        super(PhysicsLoss, self).__init__()
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)

        bounds_penalty = torch.mean(
            torch.clamp(-y_pred, min=0.0)
        ) + torch.mean(
            torch.clamp(y_pred - 1.0, min=0.0)
        )

        # QCD forward scattering preference
        # Encourage higher cos θ* values (forward scattering dominance)
        forward_preference = torch.mean(torch.clamp(0.2 - y_pred, min=0.0))

        # Distribution shape constraint
        # Encourage realistic variance in predictions
        pred_variance = torch.var(y_pred)
        variance_penalty = torch.clamp(
            0.01 - pred_variance, min=0.0
        )  # Minimum variance

        # Combine losses
        total_loss = (
            mse_loss
            + self.physics_weight * bounds_penalty
            + self.physics_weight * 0.1 * forward_preference
            + self.physics_weight * 0.05 * variance_penalty
        )

        return total_loss


class PhysicsInformedNN:
    """
    Physics-informed neural network for predicting cos(θ*) with QCD constraints.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [256, 128, 64, 32],
        device: str = "auto",
    ):
        """
        Initialize physics-informed neural network.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.history = None

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

    def build_model(self, physics_loss_weight: float = 0.1) -> PhysicsInformedNet:
        """
        Build physics-informed neural network architecture.

        Args:
            physics_loss_weight: Weight for physics constraint terms in loss
        """
        self.model = PhysicsInformedNet(self.input_dim, self.hidden_layers)
        self.model.to(self.device)

        self.criterion = PhysicsLoss(physics_loss_weight)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Built physics-informed NN with {total_params:,} parameters")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        physics_loss_weight: float = 0.1,
        learning_rate: float = 0.001,
    ) -> Dict[str, float]:
        """
        Train the physics-informed neural network.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum training epochs
            batch_size: Training batch size
            physics_loss_weight: Weight for physics constraints
            learning_rate: Learning rate for optimizer
        """
        logger.info("Training physics-informed neural network...")

        if self.model is None:
            self.build_model(physics_loss_weight)

        # Create data loaders
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6, verbose=True
        )

        # Training loop with early stopping
        train_losses = []
        val_losses = []
        val_mse_scores = []
        val_mae_scores = []

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_true = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = self.model(batch_X)
                    loss = self.criterion(predictions, batch_y)
                    val_loss += loss.item()

                    all_val_preds.append(predictions.cpu().numpy())
                    all_val_true.append(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            val_preds = np.concatenate(all_val_preds).flatten()
            val_true = np.concatenate(all_val_true).flatten()

            val_mse = mean_squared_error(val_true, val_preds)
            val_mae = mean_absolute_error(val_true, val_preds)

            val_mse_scores.append(val_mse)
            val_mae_scores.append(val_mae)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_physics_nn.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(torch.load("best_physics_nn.pth"))
                break

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}"
                )

        self.history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_mse": val_mse_scores,
            "val_mae": val_mae_scores,
        }

        y_pred_val = self.predict(X_val)
        metrics = {
            "val_mse": mean_squared_error(y_val, y_pred_val),
            "val_mae": mean_absolute_error(y_val, y_pred_val),
            "val_r2": r2_score(y_val, y_pred_val),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
            "epochs_trained": len(train_losses),
            "best_val_loss": best_val_loss,
        }

        logger.info(f"Training complete. Validation R²: {metrics['val_r2']:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions.flatten()

    def get_layer_outputs(
        self, X: np.ndarray, layer_indices: List[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get intermediate layer outputs for interpretability.

        Args:
            X: Input features
            layer_indices: Specific layer indices to extract (None for key layers)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if layer_indices is None:
            layer_indices = [
                i * 4 + 3 for i in range(len(self.hidden_layers))
            ]

        self.model.eval()
        layer_outputs = {}

        def hook_fn(name):
            def hook(module, input, output):
                layer_outputs[name] = output.detach().cpu().numpy()

            return hook

        hooks = []
        for i, idx in enumerate(layer_indices):
            if idx < len(self.model.network):
                hook = self.model.network[idx].register_forward_hook(
                    hook_fn(f"layer_{idx}")
                )
                hooks.append(hook)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _ = self.model(X_tensor)

        for hook in hooks:
            hook.remove()

        return layer_outputs

    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_layers": self.hidden_layers,
                "history": self.history,
            },
            filepath,
        )

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, physics_loss_weight: float = 0.1):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.input_dim = checkpoint["input_dim"]
        self.hidden_layers = checkpoint["hidden_layers"]
        self.history = checkpoint.get("history", None)

        self.build_model(physics_loss_weight)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from {filepath}")


def train_physics_nn_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: List[int] = [256, 128, 64, 32],
    physics_loss_weight: float = 0.1,
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 0.001,
) -> Tuple[PhysicsInformedNN, Dict[str, float]]:
    """
    Complete pipeline for training physics-informed neural network.

    Returns:
        Trained model and training metrics
    """
    logger.info("Starting physics-informed NN training pipeline...")

    model = PhysicsInformedNN(input_dim=X_train.shape[1], hidden_layers=hidden_layers)

    metrics = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        physics_loss_weight=physics_loss_weight,
        learning_rate=learning_rate,
    )

    return model, metrics


def evaluate_physics_constraints(
    model: PhysicsInformedNN, X_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate how well the model satisfies physics constraints.

    Args:
        model: Trained physics-informed NN
        X_test: Test features

    Returns:
        Dictionary of constraint satisfaction metrics
    """
    predictions = model.predict(X_test)

    bounds_violations = np.sum((predictions < 0) | (predictions > 1)) / len(predictions)

    mean_cos_theta = np.mean(predictions)

    pred_std = np.std(predictions)
    pred_variance = np.var(predictions)

    return {
        "bounds_violation_rate": bounds_violations,
        "mean_cos_theta_star": mean_cos_theta,
        "prediction_std": pred_std,
        "prediction_variance": pred_variance,
        "forward_scattering_fraction": np.mean(predictions > 0.5),
    }
