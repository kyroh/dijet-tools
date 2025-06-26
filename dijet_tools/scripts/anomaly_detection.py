#!/usr/bin/env python3
"""
Command-line script for anomaly detection in dijet events.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dijet_tools.features.engineering import FeatureEngineer
from dijet_tools.models.anomaly_detection import AnomalyDetector
from dijet_tools.utils.logging import setup_logging


def main():
    """Main entry point for dijet-anomaly command."""
    parser = argparse.ArgumentParser(
        description="Detect anomalies in dijet events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        "-d",
        required=True,
        help="Path to processed data file or directory",
    )

    parser.add_argument(
        "--output-dir", "-o", required=True, help="Output directory for anomaly results"
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.95,
        help="Anomaly detection threshold (percentile)",
    )

    parser.add_argument(
        "--method",
        choices=["isolation_forest", "autoencoder", "both"],
        default="isolation_forest",
        help="Anomaly detection method",
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected fraction of anomalies (for isolation forest)",
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

        logger.info(f"Prepared {X.shape[1]} features for anomaly detection")

        if args.method in ["isolation_forest", "both"]:
            logger.info("Running Isolation Forest anomaly detection...")
            iso_detector = AnomalyDetector(
                method="isolation_forest", contamination=args.contamination
            )

            train_size = min(10000, len(X) // 2)
            X_train = X[:train_size]
            iso_detector.train(X_train)

            iso_scores, iso_anomalies = iso_detector.detect_anomalies(X)

            logger.info(
                f"Isolation Forest found {iso_anomalies.sum()} anomalies "
                f"({iso_anomalies.sum() / len(iso_anomalies) * 100:.2f}%)"
            )

            iso_results = pd.DataFrame(
                {
                    "event_index": range(len(iso_scores)),
                    "anomaly_score": iso_scores,
                    "is_anomaly": iso_anomalies,
                }
            )
            iso_results.to_parquet(
                output_dir / "isolation_forest_anomalies.parquet", index=False
            )

        if args.method in ["autoencoder", "both"]:
            logger.info("Running Autoencoder anomaly detection...")
            ae_detector = AnomalyDetector(
                method="autoencoder", contamination=args.contamination
            )

            train_size = min(10000, len(X) // 2)
            X_train = X[:train_size]
            ae_detector.train(X_train)

            ae_scores, ae_anomalies = ae_detector.detect_anomalies(X)

            logger.info(
                f"Autoencoder found {ae_anomalies.sum()} anomalies "
                f"({ae_anomalies.sum() / len(ae_anomalies) * 100:.2f}%)"
            )

            ae_results = pd.DataFrame(
                {
                    "event_index": range(len(ae_scores)),
                    "reconstruction_error": ae_scores,
                    "is_anomaly": ae_anomalies,
                }
            )
            ae_results.to_parquet(
                output_dir / "autoencoder_anomalies.parquet", index=False
            )

        if args.method == "both":
            combined_results = pd.DataFrame(
                {
                    "event_index": range(len(X)),
                    "iso_forest_score": iso_scores,
                    "iso_forest_anomaly": iso_anomalies,
                    "autoencoder_error": ae_scores,
                    "autoencoder_anomaly": ae_anomalies,
                    "both_anomaly": np.logical_and(iso_anomalies, ae_anomalies),
                }
            )
            combined_results.to_parquet(
                output_dir / "combined_anomalies.parquet", index=False
            )

            logger.info(
                f"Combined method found {combined_results['both_anomaly'].sum()} anomalies "
                f"({combined_results['both_anomaly'].sum() / len(combined_results) * 100:.2f}%)"
            )

        logger.info(f"Anomaly detection complete! Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
