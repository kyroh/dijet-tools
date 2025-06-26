import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class AutoencoderNet(nn.Module):
    """
    Autoencoder neural network for anomaly detection.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(AutoencoderNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector:
    """
    Multi-method anomaly detection for new physics searches in dijet events.
    """

    def __init__(
        self,
        method: str = "autoencoder",
        contamination: float = 0.05,
        device: str = "auto",
    ):
        """
        Initialize anomaly detector.

        Args:
            method: Detection method ('autoencoder', 'isolation_forest', 'elliptic_envelope')
            contamination: Expected fraction of anomalies
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.method = method
        self.contamination = contamination
        self.detector = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

    def build_autoencoder(
        self, input_dim: int, encoding_dim: int = 8
    ) -> AutoencoderNet:
        """
        Build autoencoder for anomaly detection.

        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoded representation
        """
        autoencoder = AutoencoderNet(input_dim, encoding_dim)
        autoencoder.to(self.device)
        return autoencoder

    def train_autoencoder(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
    ) -> Dict[str, float]:
        """Train autoencoder with early stopping."""

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.detector.parameters(), lr=learning_rate)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self.detector.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                reconstructed = self.detector(batch_X)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            if X_val is not None:
                self.detector.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        reconstructed = self.detector(batch_X)
                        loss = criterion(reconstructed, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.detector.state_dict(), "best_autoencoder.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    self.detector.load_state_dict(torch.load("best_autoencoder.pth"))
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                    )

        self.detector.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            train_reconstructions = self.detector(X_train_tensor).cpu().numpy()

        reconstruction_errors = np.mean(
            np.square(X_train - train_reconstructions), axis=1
        )
        self.threshold = np.percentile(
            reconstruction_errors, (1 - self.contamination) * 100
        )

        return {
            "threshold": self.threshold,
            "final_loss": train_losses[-1],
            "training_epochs": len(train_losses),
            "best_val_loss": best_val_loss if X_val is not None else None,
        }

    def train(
        self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train anomaly detector on presumed normal (QCD) events.

        Args:
            X_train: Training features (normal QCD events)
            X_val: Validation features (optional)
        """
        logger.info(f"Training {self.method} anomaly detector...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        if self.method == "autoencoder":
            self.detector = self.build_autoencoder(X_train.shape[1])
            metrics = self.train_autoencoder(X_train_scaled, X_val_scaled)

        elif self.method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.contamination, random_state=42, n_estimators=100
            )
            self.detector.fit(X_train_scaled)

            scores = self.detector.decision_function(X_train_scaled)
            self.threshold = np.percentile(scores, self.contamination * 100)

            metrics = {
                "threshold": self.threshold,
                "n_estimators": self.detector.n_estimators,
            }

        elif self.method == "elliptic_envelope":
            self.detector = EllipticEnvelope(
                contamination=self.contamination, random_state=42
            )
            self.detector.fit(X_train_scaled)

            metrics = {"contamination": self.contamination}

        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")

        self.is_fitted = True
        logger.info(f"Anomaly detector training complete. Method: {self.method}")
        return metrics

    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data.

        Args:
            X: Features to test for anomalies

        Returns:
            anomaly_scores: Continuous anomaly scores
            is_anomaly: Boolean array indicating anomalies
        """
        if not self.is_fitted:
            raise ValueError("Detector not trained yet!")

        X_scaled = self.scaler.transform(X)

        if self.method == "autoencoder":
            self.detector.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                reconstructions = self.detector(X_tensor).cpu().numpy()

            anomaly_scores = np.mean(np.square(X_scaled - reconstructions), axis=1)
            is_anomaly = anomaly_scores > self.threshold

        elif self.method == "isolation_forest":
            anomaly_scores = -self.detector.decision_function(
                X_scaled
            )
            is_anomaly = self.detector.predict(X_scaled) == -1

        elif self.method == "elliptic_envelope":
            anomaly_scores = self.detector.mahalanobis(X_scaled)
            is_anomaly = self.detector.predict(X_scaled) == -1

        logger.info(
            f"Detected {np.sum(is_anomaly)} anomalies out of {len(X)} events ({np.mean(is_anomaly):.1%})"
        )
        return anomaly_scores, is_anomaly

    def get_reconstruction_quality(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get detailed reconstruction quality metrics (for autoencoder method).

        Args:
            X: Input features

        Returns:
            Dictionary with reconstruction metrics
        """
        if self.method != "autoencoder":
            raise ValueError(
                "Reconstruction quality only available for autoencoder method"
            )

        if not self.is_fitted:
            raise ValueError("Detector not trained yet!")

        X_scaled = self.scaler.transform(X)

        self.detector.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructions = self.detector(X_tensor).cpu().numpy()

        total_error = np.mean(np.square(X_scaled - reconstructions), axis=1)
        feature_errors = np.square(X_scaled - reconstructions)
        max_feature_error = np.max(feature_errors, axis=1)
        mean_abs_error = np.mean(np.abs(X_scaled - reconstructions), axis=1)

        return {
            "total_reconstruction_error": total_error,
            "feature_wise_errors": feature_errors,
            "max_feature_error": max_feature_error,
            "mean_absolute_error": mean_abs_error,
            "reconstructed_features": self.scaler.inverse_transform(reconstructions),
        }


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection methods for robust detection.
    """

    def __init__(
        self,
        methods: List[str] = ["autoencoder", "isolation_forest"],
        contamination: float = 0.05,
        device: str = "auto",
    ):
        """
        Initialize ensemble detector.

        Args:
            methods: List of detection methods to use
            contamination: Expected fraction of anomalies
            device: Device to use for PyTorch models
        """
        self.methods = methods
        self.contamination = contamination
        self.device = device
        self.detectors = {}
        self.is_fitted = False

    def train(
        self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """Train all detectors in the ensemble."""
        logger.info(f"Training ensemble with methods: {self.methods}")

        results = {}
        for method in self.methods:
            detector = AnomalyDetector(
                method=method, contamination=self.contamination, device=self.device
            )
            metrics = detector.train(X_train, X_val)
            self.detectors[method] = detector
            results[method] = metrics

        self.is_fitted = True
        return results

    def detect_anomalies(
        self, X: np.ndarray, voting: str = "majority"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using ensemble voting.

        Args:
            X: Features to test
            voting: Voting strategy ('majority', 'unanimous', 'any')
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not trained yet!")

        all_scores = []
        all_predictions = []

        for method, detector in self.detectors.items():
            scores, predictions = detector.detect_anomalies(X)
            all_scores.append(scores)
            all_predictions.append(predictions)

        ensemble_scores = np.mean(all_scores, axis=0)

        predictions_array = np.array(all_predictions)

        if voting == "majority":
            ensemble_predictions = (
                np.sum(predictions_array, axis=0) > len(self.methods) / 2
            )
        elif voting == "unanimous":
            ensemble_predictions = np.all(predictions_array, axis=0)
        elif voting == "any":
            ensemble_predictions = np.any(predictions_array, axis=0)
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

        return ensemble_scores, ensemble_predictions
