import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.loaders import ATLASDataLoader
from ..data.selectors import ATLASEventSelector
from ..evaluation.metrics import PhysicsMetrics
from ..physics.kinematics import DijetKinematics
from ..utils.io import load_model, save_results

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline for making predictions on new ATLAS data.
    """

    def __init__(self, model_dir: str, config: Optional[Dict] = None):
        """
        Initialize inference pipeline.

        Args:
            model_dir: Directory containing trained models and preprocessing objects
            config: Optional configuration dictionary
        """
        self.model_dir = Path(model_dir)
        self.config = config or {}

        self.models = {}
        self.scaler = None
        self.feature_engineer = None
        self.feature_names = None

        self._load_trained_components()

    def _load_trained_components(self):
        """Load all trained models and preprocessing components."""
        logger.info(f"Loading trained components from {self.model_dir}")

        model_files = {
            "xgboost": self.model_dir / "xgboost_model.joblib",
            "neural_network": self.model_dir / "neural_network_model",
            "anomaly_detector": self.model_dir / "anomaly_detector.joblib",
        }

        for model_name, model_path in model_files.items():
            if model_path.exists():
                try:
                    self.models[model_name] = load_model(str(model_path))
                    logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")

        scaler_path = self.model_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            import joblib

            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")

        engineer_path = self.model_dir / "feature_engineer.joblib"
        if engineer_path.exists():
            import joblib

            self.feature_engineer = joblib.load(engineer_path)
            logger.info("Loaded feature engineer")

        features_path = self.model_dir / "feature_names.json"
        if features_path.exists():
            import json

            with open(features_path, "r") as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")

    def predict_from_files(
        self,
        file_paths: List[str],
        max_events: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make predictions on ATLAS data files.

        Args:
            file_paths: List of ATLAS ROOT files
            max_events: Maximum events to process (None for all)
            output_path: Path to save results (None to skip saving)

        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        logger.info(f"Starting inference on {len(file_paths)} files")

        data_loader = ATLASDataLoader(file_paths)
        event_selector = ATLASEventSelector()
        kinematics_calc = DijetKinematics()

        all_events = []
        total_processed = 0

        for chunk in data_loader.stream_events(chunk_size=25000, max_events=max_events):
            selected_events = event_selector.apply_jet_quality_cuts(chunk)

            if not selected_events.empty:
                physics_data = kinematics_calc.calculate_all_observables(
                    selected_events
                )
                all_events.append(physics_data)
                total_processed += len(physics_data)

                logger.info(f"Processed {total_processed:,} events so far...")

        if not all_events:
            raise ValueError("No events passed selection criteria")

        combined_data = pd.concat(all_events, ignore_index=True)
        logger.info(f"Total events for inference: {len(combined_data):,}")

        predictions = self._make_predictions(combined_data)

        results = {
            "predictions": predictions,
            "input_data": combined_data,
            "metadata": {
                "n_files": len(file_paths),
                "n_events": len(combined_data),
                "processing_time": time.time() - start_time,
                "models_used": list(self.models.keys()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        if output_path:
            save_results(results, output_path)
            logger.info(f"Results saved to {output_path}")

        logger.info(f"Inference complete in {time.time() - start_time:.1f}s")
        return results

    def _make_predictions(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using all available models.

        Args:
            data: Processed physics data

        Returns:
            Dictionary of model predictions
        """
        logger.info("Making predictions with trained models...")

        if self.feature_engineer:
            features_df = self.feature_engineer.create_features(data)
        else:
            basic_features = [
                "mjj",
                "delta_y",
                "chi",
                "pt_balance",
                "delta_phi",
                "leading_jet_pt",
                "subleading_jet_pt",
            ]
            features_df = data[basic_features]

        if self.scaler:
            X_scaled = self.scaler.transform(features_df)
        else:
            X_scaled = features_df.values

        predictions = {}

        for model_name, model in self.models.items():
            try:
                if model_name == "anomaly_detector":
                    scores, is_anomaly = model.detect_anomalies(X_scaled)
                    predictions[f"{model_name}_scores"] = scores
                    predictions[f"{model_name}_predictions"] = is_anomaly
                else:
                    pred = model.predict(X_scaled)
                    predictions[model_name] = pred

                logger.info(f"Generated predictions using {model_name}")

            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")

        return predictions

    def validate_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        true_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Validate predictions against physics expectations and truth (if available).

        Args:
            predictions: Dictionary of model predictions
            true_values: True cos θ* values (if available)

        Returns:
            Validation results
        """
        logger.info("Validating predictions...")

        validation_results = {}

        for model_name, pred_values in predictions.items():
            if model_name.endswith("_scores") or model_name.endswith("_predictions"):
                continue

            physics_metrics = PhysicsMetrics()

            if true_values is not None:
                consistency = physics_metrics.evaluate_physics_consistency(
                    true_values, pred_values
                )
            else:
                consistency = {
                    "forward_scattering_fraction": np.sum(pred_values > 0.8)
                    / len(pred_values),
                    "mean_cos_theta": np.mean(pred_values),
                    "std_cos_theta": np.std(pred_values),
                    "fraction_in_bounds": np.sum(
                        (pred_values >= 0) & (pred_values <= 1)
                    )
                    / len(pred_values),
                }

            validation_results[model_name] = consistency

        return validation_results

    def batch_inference(
        self, file_list_path: str, output_dir: str, batch_size: int = 10
    ) -> List[str]:
        """
        Run inference on multiple batches of files.

        Args:
            file_list_path: Path to text file containing list of ATLAS files
            output_dir: Directory to save batch results
            batch_size: Number of files per batch

        Returns:
            List of output file paths
        """
        with open(file_list_path, "r") as f:
            all_files = [line.strip() for line in f if line.strip()]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i : i + batch_size]
            batch_name = f"batch_{i//batch_size + 1:03d}"

            logger.info(f"Processing {batch_name}: {len(batch_files)} files")

            try:
                output_path = output_dir / f"{batch_name}_results.pkl"
                results = self.predict_from_files(
                    batch_files, output_path=str(output_path)
                )
                output_paths.append(str(output_path))

            except Exception as e:
                logger.error(f"Batch {batch_name} failed: {e}")

        logger.info(f"Batch inference complete. Processed {len(output_paths)} batches")
        return output_paths
