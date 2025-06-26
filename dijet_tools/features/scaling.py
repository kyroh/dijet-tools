import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Flexible feature scaling with multiple methods.
    """

    def __init__(self, method: str = "robust"):
        """
        Initialize scaler.

        Args:
            method: Scaling method ('standard', 'robust', 'minmax')
        """
        self.method = method
        self.scaler = self._create_scaler()
        self.is_fitted = False

    def _create_scaler(self):
        """Create appropriate scaler based on method."""
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "robust":
            return RobustScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit scaler and transform features."""
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        logger.info(f"Fitted {self.method} scaler on {X.shape[0]} samples")
        return X_scaled

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet. Call fit_transform first.")
        return self.scaler.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet.")
        return self.scaler.inverse_transform(X_scaled)
