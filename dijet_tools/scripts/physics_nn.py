#!/usr/bin/env python3
"""
Command-line script for training physics-informed neural networks.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dijet_tools.features.engineering import FeatureEngineer
from dijet_tools.models.neural_networks import train_physics_nn_pipeline
from dijet_tools.utils.logging import setup_logging


def main():
    """Main entry point for dijet-physics-nn command."""
    parser = argparse.ArgumentParser(
        description="Train physics-informed neural networks for dijet analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        "-d",
        required=True,
        help="Path to processed data file or directory",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="models",
        help="Output directory for trained models",
    )

    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32],
        help="Hidden layer sizes",
    )

    parser.add_argument(
        "--physics-weight",
        type=float,
        default=0.1,
        help="Weight for physics constraints in loss function",
    )

    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )

    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_path = Path(args.data_path)
        if data_path.is_dir():
            data_files = list(data_path.glob("*.parquet"))
            if not data_files:
                logger.error(f"No parquet files found in {data_path}")
                sys.exit(1)

            logger.info(f"Loading {len(data_files)} data files")
            data_chunks = []
            for file_path in data_files:
                chunk = pd.read_parquet(file_path)
                data_chunks.append(chunk)

            data = pd.concat(data_chunks, ignore_index=True)
        else:
            data = pd.read_parquet(data_path)

        logger.info(f"Loaded data with {len(data)} events")

        feature_engineer = FeatureEngineer()
        features_data = feature_engineer.create_features(data)

        feature_cols = [
            col
            for col in features_data.columns
            if col not in ["event_index", "file_index", "chunk_start"]
        ]
        X = features_data[feature_cols].to_numpy()
        y = (
            features_data["cos_theta_star"].to_numpy()
            if "cos_theta_star" in features_data.columns
            else np.zeros(len(X))
        )

        logger.info(f"Prepared {X.shape[1]} features for training")

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.random_state
        )

        X_train_final = np.asarray(X_train_final)
        y_train_final = np.asarray(y_train_final)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        logger.info("Training physics-informed neural network...")

        model, history = train_physics_nn_pipeline(
            X_train_final,
            y_train_final,
            X_val,
            y_val,
            hidden_layers=args.hidden_layers,
            physics_loss_weight=args.physics_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        test_predictions = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score

        test_r2 = r2_score(y_test, test_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)

        logger.info(f"Test set performance:")
        logger.info(f"  R² Score: {test_r2:.4f}")
        logger.info(f"  MSE: {test_mse:.6f}")

        model_path = output_dir / "physics_nn_model.pt"
        model.save_model(str(model_path))
        logger.info(f"Physics-informed NN saved to: {model_path}")

        history_path = output_dir / "physics_nn_history.json"
        import json

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

        from dijet_tools.models.neural_networks import \
            evaluate_physics_constraints

        constraint_metrics = evaluate_physics_constraints(model, X_test)

        logger.info("Physics constraint evaluation:")
        for metric, value in constraint_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("Physics-informed neural network training complete!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
