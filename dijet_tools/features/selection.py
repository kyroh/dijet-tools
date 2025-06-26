import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, f_regression,
                                       mutual_info_regression)

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Select most relevant features for cos(θ*) prediction.
    """

    def __init__(self, method: str = "mutual_info", k: int = 15):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('mutual_info', 'f_regression', 'tree_importance')
            k: Number of features to select
        """
        self.method = method
        self.k = k
        self.selected_features = None
        self.feature_scores = None

    def select_features(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select k best features based on specified method.

        Args:
            X: Feature matrix
            y: Target variable (cos θ*)

        Returns:
            Selected features DataFrame and feature names list
        """
        logger.info(f"Selecting {self.k} features using {self.method} method...")

        if self.method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=self.k)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            self.feature_scores = selector.scores_

        elif self.method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=self.k)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            self.feature_scores = selector.scores_

        elif self.method == "tree_importance":
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_

            selected_indices = np.argsort(importances)[-self.k :]
            X_selected = X.iloc[:, selected_indices]
            self.feature_scores = importances

        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        self.selected_features = X.columns[selected_indices].tolist()

        X_selected_df = pd.DataFrame(
            X_selected, columns=self.selected_features, index=X.index
        )

        logger.info(f"Selected features: {self.selected_features}")
        return X_selected_df, self.selected_features

    def get_feature_ranking(self, feature_names: List[str]) -> pd.DataFrame:
        """Get ranking of all features by importance/score."""
        if self.feature_scores is None:
            raise ValueError("Feature selection not performed yet.")

        ranking_df = pd.DataFrame(
            {"feature": feature_names, "score": self.feature_scores}
        ).sort_values("score", ascending=False)

        return ranking_df
