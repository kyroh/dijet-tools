#!/usr/bin/env python3
"""
Command-line script for running inference with trained models.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from dijet_tools.features.engineering import FeatureEngineer
from dijet_tools.utils.io import load_model
from dijet_tools.utils.logging import setup_logging


def main():
    """Main entry point for dijet-predict command."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained dijet analysis models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path", "-m", required=True, help="Path to trained model file"
    )

    parser.add_argument(
        "--data-path",
        "-d",
        required=True,
        help="Path to processed data file or directory",
    )

    parser.add_argument(
        "--output-path", "-o", required=True, help="Output path for predictions"
    )

    parser.add_argument(
        "--model-type",
        choices=["xgboost", "neural_network", "auto"],
        default="auto",
        help="Type of model (auto-detect if not specified)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading model from: {args.model_path}")
        model = load_model(args.model_path)

        if args.model_type == "auto":
            if hasattr(model, "predict") and hasattr(model, "feature_names"):
                args.model_type = "xgboost"
            elif hasattr(model, "predict"):
                args.model_type = "neural_network"
            else:
                raise ValueError("Could not auto-detect model type")

        logger.info(f"Using model type: {args.model_type}")

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

        logger.info(f"Prepared {X.shape[1]} features for inference")

        logger.info("Running inference...")
        predictions = model.predict(X)

        results = pd.DataFrame(
            {
                "event_index": range(len(predictions)),
                "predicted_cos_theta_star": predictions.flatten(),
            }
        )

        if args.verbose:
            for i, col in enumerate(feature_cols):
                results[f"feature_{col}"] = X[:, i]

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".parquet":
            results.to_parquet(output_path, index=False)
        else:
            results.to_csv(output_path, index=False)

        logger.info(f"Predictions saved to: {output_path}")
        logger.info(f"Prediction statistics:")
        logger.info(f"  Mean: {predictions.mean():.4f}")
        logger.info(f"  Std:  {predictions.std():.4f}")
        logger.info(f"  Min:  {predictions.min():.4f}")
        logger.info(f"  Max:  {predictions.max():.4f}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
