#!/usr/bin/env python3
"""
Command-line script for training dijet analysis models.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dijet_tools.features.engineering import FeatureEngineer
from dijet_tools.models.neural_networks import PhysicsInformedNN
from dijet_tools.models.xgboost_model import XGBoostPredictor
from dijet_tools.utils.config import ConfigManager
from dijet_tools.utils.io import save_model
from dijet_tools.utils.logging import setup_logging


def main():
    """Main entry point for dijet-train command."""
    parser = argparse.ArgumentParser(
        description="Train machine learning models for dijet analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c", default="configs/default.yaml", help="Configuration file path"
    )

    parser.add_argument(
        "--data-path",
        "-d",
        required=True,
        help="Path to processed data directory or file",
    )

    parser.add_argument(
        "--model-type",
        "-m",
        choices=["xgboost", "neural_network", "both"],
        default="xgboost",
        help="Type of model to train",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="models",
        help="Output directory for trained models",
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
        config = ConfigManager.load_config(args.config)

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

        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} events")

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

        if args.model_type in ["xgboost", "both"]:
            logger.info("Training XGBoost model...")
            xgb_model = XGBoostPredictor()
            xgb_metrics = xgb_model.train(
                X_train_final, y_train_final, X_val, y_val, feature_names=feature_cols
            )

            xgb_predictions = xgb_model.predict(X_test)
            from sklearn.metrics import r2_score

            xgb_score = r2_score(y_test, xgb_predictions)
            logger.info(f"XGBoost test score: {xgb_score:.4f}")

            xgb_path = output_dir / "xgboost_model.pkl"
            save_model(xgb_model, str(xgb_path))
            logger.info(f"XGBoost model saved to: {xgb_path}")

        if args.model_type in ["neural_network", "both"]:
            logger.info("Training neural network model...")
            nn_model = PhysicsInformedNN(input_dim=X.shape[1])
            nn_model.build_model()

            history = nn_model.train(X_train_final, y_train_final, X_val, y_val)

            nn_predictions = nn_model.predict(X_test)
            from sklearn.metrics import r2_score

            nn_score = r2_score(y_test, nn_predictions)
            logger.info(f"Neural network test score: {nn_score:.4f}")

            nn_path = output_dir / "neural_network_model.pt"
            nn_model.save_model(str(nn_path))
            logger.info(f"Neural network model saved to: {nn_path}")

        logger.info("Model training complete!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
