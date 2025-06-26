import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class PhysicsMetrics:
    """Physics metrics for analysis validation."""

    @staticmethod
    def evaluate_physics_consistency(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Physics validation of cos(θ*) predictions.
        Args:
            y_true: True cos(θ*) values
            y_pred: Predicted cos(θ*) values
        Returns:
            Dict of validation metrics
        """
        metrics = {
            "r2_score": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        }


        forward_thresholds = [0.7, 0.8, 0.9]
        for threshold in forward_thresholds:
            forward_true = np.sum(y_true > threshold) / len(y_true)
            forward_pred = np.sum(y_pred > threshold) / len(y_pred)
            metrics[f"forward_fraction_true_{int(threshold * 100)}"] = forward_true
            metrics[f"forward_fraction_pred_{int(threshold * 100)}"] = forward_pred
            metrics[f"forward_difference_{int(threshold * 100)}"] = abs(
                forward_true - forward_pred
            )

        ks_statistic, ks_pvalue = stats.ks_2samp(y_true, y_pred)
        metrics.update(
            {
                "kolmogorov_smirnov_statistic": ks_statistic,
                "kolmogorov_smirnov_pvalue": ks_pvalue,
                "wasserstein_distance": stats.wasserstein_distance(y_true, y_pred),
            }
        )

        metrics.update(
            {
                "mean_true": np.mean(y_true),
                "mean_pred": np.mean(y_pred),
                "mean_difference": abs(np.mean(y_true) - np.mean(y_pred)),
                "std_true": np.std(y_true),
                "std_pred": np.std(y_pred),
                "std_ratio": np.std(y_pred) / np.std(y_true),
                "skewness_true": stats.skew(y_true),
                "skewness_pred": stats.skew(y_pred),
                "kurtosis_true": stats.kurtosis(y_true),
                "kurtosis_pred": stats.kurtosis(y_pred),
            }
        )

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for q in quantiles:
            q_true = np.quantile(y_true, q)
            q_pred = np.quantile(y_pred, q)
            metrics[f"quantile_{int(q * 100)}_true"] = q_true
            metrics[f"quantile_{int(q * 100)}_pred"] = q_pred
            metrics[f"quantile_{int(q * 100)}_difference"] = abs(q_true - q_pred)

        metrics.update(
            {
                "fraction_in_bounds_true": np.sum((y_true >= 0) & (y_true <= 1))
                / len(y_true),
                "fraction_in_bounds_pred": np.sum((y_pred >= 0) & (y_pred <= 1))
                / len(y_pred),
                "out_of_bounds_penalty": np.sum(
                    np.maximum(0, -y_pred) + np.maximum(0, y_pred - 1)
                ),
            }
        )

        pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)

        metrics.update(
            {
                "pearson_correlation": pearson_corr,
                "pearson_pvalue": pearson_p,
                "spearman_correlation": spearman_corr,
                "spearman_pvalue": spearman_p,
            }
        )

        qcd_score = (
            metrics["forward_difference_80"] * 2.0
            + metrics["mean_difference"] * 1.0
            + metrics["kolmogorov_smirnov_statistic"] * 0.5
        )
        metrics["qcd_consistency_score"] = qcd_score

        logger.info(
            f"Physics validation complete. R² = {metrics['r2_score']:.4f}, QCD score = {qcd_score:.4f}"
        )
        return metrics

    @staticmethod
    def binned_performance_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: np.ndarray = None,
        n_bins: int = 10,
    ) -> Dict[str, List]:
        """
        Analyze model performance in cos(θ*) bins.
        Args:
            y_true, y_pred: True and predicted values
            bins: Custom bin edges (None for uniform bins)
            n_bins: Number of bins if bins not provided
        Returns:
            Dict of binned metrics
        """
        if bins is None:
            bins = np.linspace(0, 1, n_bins + 1)

        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_metrics = {
            "bin_centers": bin_centers.tolist(),
            "bin_edges": bins.tolist(),
            "bin_r2": [],
            "bin_rmse": [],
            "bin_mae": [],
            "bin_bias": [],
            "bin_count": [],
            "bin_std_ratio": [],
        }

        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            n_events = np.sum(mask)

            if n_events > 10:
                y_true_bin = y_true[mask]
                y_pred_bin = y_pred[mask]

                bin_r2 = r2_score(y_true_bin, y_pred_bin)
                bin_rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
                bin_mae = mean_absolute_error(y_true_bin, y_pred_bin)
                bin_bias = np.mean(y_pred_bin - y_true_bin)
                bin_std_ratio = (
                    np.std(y_pred_bin) / np.std(y_true_bin)
                    if np.std(y_true_bin) > 0
                    else np.nan
                )
            else:
                bin_r2 = bin_rmse = bin_mae = bin_bias = bin_std_ratio = np.nan

            bin_metrics["bin_r2"].append(bin_r2)
            bin_metrics["bin_rmse"].append(bin_rmse)
            bin_metrics["bin_mae"].append(bin_mae)
            bin_metrics["bin_bias"].append(bin_bias)
            bin_metrics["bin_count"].append(n_events)
            bin_metrics["bin_std_ratio"].append(bin_std_ratio)

        return bin_metrics

    @staticmethod
    def calculate_feature_physics_correlations(
        features_df: pd.DataFrame, target: np.ndarray
    ) -> pd.DataFrame:
        """Correlations between features and target."""
        correlations = []

        for feature in features_df.columns:
            if features_df[feature].dtype in ["float64", "float32", "int64", "int32"]:
                corr_coef = np.corrcoef(features_df[feature], target)[0, 1]

                mutual_info = 0.0

                correlations.append(
                    {
                        "feature": feature,
                        "pearson_correlation": corr_coef,
                        "abs_correlation": abs(corr_coef),
                        "mutual_information": mutual_info,
                    }
                )

        corr_df = pd.DataFrame(correlations)
        return corr_df.sort_values("abs_correlation", ascending=False)
