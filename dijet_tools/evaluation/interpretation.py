import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Tools for interpreting ML model predictions in physics context.
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize model interpreter.

        Args:
            model: Trained ML model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

    def analyze_prediction_sensitivity(
        self, X: np.ndarray, feature_perturbation: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how sensitive predictions are to feature perturbations.

        Args:
            X: Input features
            feature_perturbation: Relative perturbation size

        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info("Analyzing prediction sensitivity...")

        baseline_predictions = self.model.predict(X)
        sensitivities = {}

        for i, feature_name in enumerate(self.feature_names):
            X_perturbed = X.copy()
            perturbation = X[:, i] * feature_perturbation
            X_perturbed[:, i] += perturbation

            perturbed_predictions = self.model.predict(X_perturbed)

            sensitivity = (
                np.abs(perturbed_predictions - baseline_predictions)
                / feature_perturbation
            )
            sensitivities[feature_name] = sensitivity

        return {
            "baseline_predictions": baseline_predictions,
            "feature_sensitivities": sensitivities,
            "mean_sensitivities": {
                name: np.mean(sens) for name, sens in sensitivities.items()
            },
        }

    def explain_extreme_predictions(
        self, X: np.ndarray, y_pred: np.ndarray, n_examples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze characteristics of extreme predictions (very high/low cos θ*).

        Args:
            X: Input features
            y_pred: Predicted cos θ* values
            n_examples: Number of extreme examples to analyze

        Returns:
            Analysis of extreme predictions
        """
        high_indices = np.argsort(y_pred)[-n_examples:]
        low_indices = np.argsort(y_pred)[:n_examples]

        high_features = X[high_indices]
        low_features = X[low_indices]

        analysis = {
            "high_cos_theta_examples": {
                "predictions": y_pred[high_indices],
                "feature_means": np.mean(high_features, axis=0),
                "feature_stds": np.std(high_features, axis=0),
            },
            "low_cos_theta_examples": {
                "predictions": y_pred[low_indices],
                "feature_means": np.mean(low_features, axis=0),
                "feature_stds": np.std(low_features, axis=0),
            },
        }

        overall_means = np.mean(X, axis=0)
        overall_stds = np.std(X, axis=0)

        analysis["feature_differences"] = {
            "high_vs_overall": analysis["high_cos_theta_examples"]["feature_means"]
            - overall_means,
            "low_vs_overall": analysis["low_cos_theta_examples"]["feature_means"]
            - overall_means,
        }

        return analysis

    def physics_feature_correlation_analysis(
        self, X: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze correlations between features and predictions from physics perspective.

        Args:
            X: Input features
            y_pred: Predicted cos θ* values

        Returns:
            DataFrame with correlation analysis
        """
        correlations = []

        for i, feature_name in enumerate(self.feature_names):
            feature_values = X[:, i]
            corr_coef = np.corrcoef(feature_values, y_pred)[0, 1]

            physics_interpretation = self._interpret_feature_correlation(
                feature_name, corr_coef
            )

            correlations.append(
                {
                    "feature": feature_name,
                    "correlation_with_cos_theta": corr_coef,
                    "abs_correlation": abs(corr_coef),
                    "physics_interpretation": physics_interpretation,
                }
            )

        corr_df = pd.DataFrame(correlations)
        return corr_df.sort_values("abs_correlation", ascending=False)

    def _interpret_feature_correlation(
        self, feature_name: str, correlation: float
    ) -> str:
        """
        Provide physics interpretation of feature correlations.

        Args:
            feature_name: Name of the feature
            correlation: Correlation coefficient with cos θ*

        Returns:
            Physics interpretation string
        """
        abs_corr = abs(correlation)
        sign = "positive" if correlation > 0 else "negative"

        interpretations = {
            "delta_y": f"Rapidity separation shows {sign} correlation with cos θ*. "
            f"{'Higher Δy typically means more central scattering (lower cos θ*)' if correlation < 0 else 'Unexpected positive correlation with cos θ*'}",
            "chi": f"Chi variable (e^|Δy|) shows {sign} correlation. "
            f"{'Expected: higher χ means lower cos θ* (more central scattering)' if correlation < 0 else 'Unexpected: higher χ correlates with higher cos θ*'}",
            "mjj": f"Dijet mass shows {sign} correlation. "
            f"Physics: {'Higher masses may favor more forward scattering' if correlation > 0 else 'Higher masses correlate with more central scattering'}",
            "pt_balance": f"pT balance shows {sign} correlation. "
            f"{'Balanced events tend toward forward scattering' if correlation > 0 else 'Balanced events tend toward central scattering'}",
            "eta_centrality": f"Eta centrality shows {sign} correlation. "
            f"{'More central events (low |η|) favor forward scattering' if correlation < 0 else 'Forward events have higher average |η|'}",
        }

        base_interpretation = interpretations.get(
            feature_name, f"{feature_name} shows {sign} correlation with cos θ*"
        )

        if abs_corr > 0.5:
            strength = "Strong"
        elif abs_corr > 0.3:
            strength = "Moderate"
        elif abs_corr > 0.1:
            strength = "Weak"
        else:
            strength = "Very weak"

        return f"{strength} {base_interpretation}"
