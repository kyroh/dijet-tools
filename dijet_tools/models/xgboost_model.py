import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost model for predicting cos(θ*) from dijet observables."""

    def __init__(self, params: Optional[Dict] = None):
        self.default_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 50,
        }
        self.params = params if params else self.default_params
        self.model = None
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train XGBoost model with validation."""

        self.feature_names = feature_names

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.params["early_stopping_rounds"],
            verbose_eval=False,
        )

        y_pred_val = self.predict(X_val)
        metrics = {
            "val_mse": mean_squared_error(y_val, y_pred_val),
            "val_mae": mean_absolute_error(y_val, y_pred_val),
            "val_r2": r2_score(y_val, y_pred_val),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
        }

        logger.info(
            f"XGBoost training complete. Validation R²: {metrics['val_r2']:.4f}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importance_dict = self.model.get_score(importance_type="weight")

        if self.feature_names:
            importance_df = pd.DataFrame(
                [
                    {"feature": name, "importance": importance_dict.get(name, 0)}
                    for name in self.feature_names
                ]
            )
        else:
            importance_df = pd.DataFrame(
                [
                    {"feature": f"f{i}", "importance": importance_dict.get(f"f{i}", 0)}
                    for i in range(len(importance_dict))
                ]
            )

        return importance_df.sort_values("importance", ascending=False)
