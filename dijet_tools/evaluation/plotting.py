import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsPlotter:
    """
    Create comprehensive plots for model evaluation and physics validation.
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter with styling.

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")

    def plot_predictions_vs_truth(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        output_dir: str = "plots",
    ) -> str:
        """
        Create prediction vs truth scatter plot with residual analysis.

        Args:
            y_true, y_pred: True and predicted values
            model_name: Name of the model for title
            output_dir: Directory to save plot

        Returns:
            Path to saved plot
        """
        Path(output_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].scatter(y_true, y_pred, alpha=0.6, s=2, color="steelblue")
        axes[0].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Prediction")

        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        axes[0].set_xlabel("True cos θ*", fontsize=12)
        axes[0].set_ylabel("Predicted cos θ*", fontsize=12)
        axes[0].set_title(
            f"{model_name}: Predictions vs Truth\nR² = {r2:.4f}, RMSE = {rmse:.4f}",
            fontsize=14,
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        residuals = y_pred - y_true
        axes[1].scatter(y_true, residuals, alpha=0.6, s=2, color="orange")
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1].axhline(
            y=np.mean(residuals),
            color="green",
            linestyle=":",
            label=f"Mean = {np.mean(residuals):.4f}",
        )

        axes[1].set_xlabel("True cos θ*", fontsize=12)
        axes[1].set_ylabel("Residuals (Pred - True)", fontsize=12)
        axes[1].set_title("Residual Analysis", fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = (
            Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_predictions.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved prediction plot: {output_path}")
        return str(output_path)

    def plot_distribution_comparison(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        output_dir: str = "plots",
    ) -> str:
        """
        Compare predicted distributions with truth across multiple models.

        Args:
            y_true: True values
            predictions: Dictionary of model_name -> predictions
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].hist(
            y_true,
            bins=50,
            alpha=0.7,
            density=True,
            label="Truth",
            color="black",
            histtype="step",
            linewidth=2,
        )

        colors = ["red", "blue", "green", "orange", "purple"]
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            axes[0, 0].hist(
                y_pred,
                bins=50,
                alpha=0.6,
                density=True,
                label=model_name,
                color=color,
                histtype="step",
                linewidth=2,
            )

        axes[0, 0].set_xlabel("cos θ*")
        axes[0, 0].set_ylabel("Normalized Events")
        axes[0, 0].set_title("Distribution Comparison (Linear Scale)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(
            y_true,
            bins=50,
            alpha=0.7,
            density=True,
            label="Truth",
            color="black",
            histtype="step",
            linewidth=2,
        )
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            axes[0, 1].hist(
                y_pred,
                bins=50,
                alpha=0.6,
                density=True,
                label=model_name,
                color=color,
                histtype="step",
                linewidth=2,
            )

        axes[0, 1].set_xlabel("cos θ*")
        axes[0, 1].set_ylabel("Normalized Events")
        axes[0, 1].set_title("Distribution Comparison (Log Scale)")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(
            y_true,
            bins=100,
            alpha=0.7,
            density=True,
            cumulative=True,
            label="Truth",
            color="black",
            histtype="step",
            linewidth=2,
        )
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            axes[1, 0].hist(
                y_pred,
                bins=100,
                alpha=0.6,
                density=True,
                cumulative=True,
                label=model_name,
                color=color,
                histtype="step",
                linewidth=2,
            )

        axes[1, 0].set_xlabel("cos θ*")
        axes[1, 0].set_ylabel("Cumulative Probability")
        axes[1, 0].set_title("Cumulative Distribution Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if predictions:
            first_model = list(predictions.keys())[0]
            y_pred_first = predictions[first_model]

            quantiles = np.linspace(0.01, 0.99, 99)
            true_quantiles = np.quantile(y_true, quantiles)
            pred_quantiles = np.quantile(y_pred_first, quantiles)

            axes[1, 1].scatter(true_quantiles, pred_quantiles, alpha=0.7, s=20)
            axes[1, 1].plot(
                [0, 1], [0, 1], "r--", linewidth=2, label="Perfect Agreement"
            )
            axes[1, 1].set_xlabel("Truth Quantiles")
            axes[1, 1].set_ylabel(f"{first_model} Quantiles")
            axes[1, 1].set_title(f"Q-Q Plot: {first_model} vs Truth")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_dir) / "distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved distribution comparison: {output_path}")
        return str(output_path)

    def plot_physics_validation(
        self, physics_metrics: Dict[str, float], output_dir: str = "plots"
    ) -> str:
        """
        Create comprehensive physics validation plots.

        Args:
            physics_metrics: Dictionary of physics validation metrics
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        thresholds = [70, 80, 90]
        true_fractions = [
            physics_metrics[f"forward_fraction_true_{t}"] for t in thresholds
        ]
        pred_fractions = [
            physics_metrics[f"forward_fraction_pred_{t}"] for t in thresholds
        ]

        x = np.arange(len(thresholds))
        width = 0.35

        axes[0, 0].bar(x - width / 2, true_fractions, width, label="Truth", alpha=0.7)
        axes[0, 0].bar(
            x + width / 2, pred_fractions, width, label="Predicted", alpha=0.7
        )
        axes[0, 0].set_xlabel("cos θ* Threshold (%)")
        axes[0, 0].set_ylabel("Fraction of Events")
        axes[0, 0].set_title("Forward Scattering Fractions")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f">{t}%" for t in thresholds])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        quantiles = [10, 25, 50, 75, 90, 95, 99]
        true_quantiles = [physics_metrics[f"quantile_{q}_true"] for q in quantiles]
        pred_quantiles = [physics_metrics[f"quantile_{q}_pred"] for q in quantiles]

        axes[0, 1].plot(
            quantiles, true_quantiles, "o-", label="Truth", linewidth=2, markersize=6
        )
        axes[0, 1].plot(
            quantiles,
            pred_quantiles,
            "s-",
            label="Predicted",
            linewidth=2,
            markersize=6,
        )
        axes[0, 1].set_xlabel("Quantile (%)")
        axes[0, 1].set_ylabel("cos θ* Value")
        axes[0, 1].set_title("Quantile Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        moments = ["Mean", "Std", "Skewness", "Kurtosis"]
        true_moments = [
            physics_metrics["mean_true"],
            physics_metrics["std_true"],
            physics_metrics["skewness_true"],
            physics_metrics["kurtosis_true"],
        ]
        pred_moments = [
            physics_metrics["mean_pred"],
            physics_metrics["std_pred"],
            physics_metrics["skewness_pred"],
            physics_metrics["kurtosis_pred"],
        ]

        x = np.arange(len(moments))
        axes[1, 0].bar(x - width / 2, true_moments, width, label="Truth", alpha=0.7)
        axes[1, 0].bar(x + width / 2, pred_moments, width, label="Predicted", alpha=0.7)
        axes[1, 0].set_xlabel("Statistical Moment")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_title("Distribution Moments Comparison")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(moments)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        consistency_metrics = [
            "Forward Difference (80%)",
            "Mean Difference",
            "KS Statistic",
            "QCD Score",
        ]
        consistency_values = [
            physics_metrics["forward_difference_80"],
            physics_metrics["mean_difference"],
            physics_metrics["kolmogorov_smirnov_statistic"],
            physics_metrics["qcd_consistency_score"],
        ]

        colors = [
            "green" if v < 0.05 else "orange" if v < 0.1 else "red"
            for v in consistency_values
        ]
        bars = axes[1, 1].bar(
            consistency_metrics, consistency_values, color=colors, alpha=0.7
        )
        axes[1, 1].set_ylabel("Difference/Score")
        axes[1, 1].set_title(
            "QCD Consistency Metrics\n(Green: Good, Orange: OK, Red: Poor)"
        )
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        for bar, val in zip(bars, consistency_values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        output_path = Path(output_dir) / "physics_validation.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved physics validation plot: {output_path}")
        return str(output_path)

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, top_n: int = 15, output_dir: str = "plots"
    ) -> str:
        """
        Create feature importance visualization.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            output_dir: Directory to save plot
        """
        Path(output_dir).mkdir(exist_ok=True)

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        y_pos = range(len(top_features))

        bars = plt.barh(
            y_pos, top_features["importance"], color="lightgreen", alpha=0.8
        )
        plt.yticks(y_pos, top_features["feature"])
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title(f"Top {top_n} Feature Importance", fontsize=14)
        plt.grid(True, alpha=0.3, axis="x")

        for i, (bar, importance) in enumerate(zip(bars, top_features["importance"])):
            plt.text(
                bar.get_width() + max(top_features["importance"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.0f}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        output_path = Path(output_dir) / "feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature importance plot: {output_path}")
        return str(output_path)
