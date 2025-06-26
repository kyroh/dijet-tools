import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from ..data.loaders import ATLASDataLoader
from ..data.selectors import ATLASEventSelector
from ..evaluation.metrics import PhysicsMetrics
from ..features.engineering import FeatureEngineer
from ..models.xgboost_model import XGBoostPredictor
from ..physics.kinematics import DijetKinematics

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for ATLAS dijet analysis."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.data_loader = None
        self.event_selector = ATLASEventSelector()
        self.kinematics_calc = DijetKinematics()
        self.feature_engineer = FeatureEngineer()
        self.model = XGBoostPredictor()
        self.scaler = RobustScaler()

    def run_full_pipeline(
        self,
        file_paths: List[str],
        max_events: Optional[int] = None,
        test_size: float = 0.2,
    ) -> Dict:
        """Run complete training pipeline."""

        logger.info("Starting ATLAS Dijet Training Pipeline")

        logger.info("Loading ATLAS data...")
        self.data_loader = ATLASDataLoader(file_paths)

        all_selected_events = []
        for chunk in self.data_loader.stream_events(
            chunk_size=25000, max_events=max_events
        ):
            selected_chunk = self.event_selector.apply_jet_quality_cuts(chunk)
            if not selected_chunk.empty:
                all_selected_events.append(selected_chunk)

        if not all_selected_events:
            raise ValueError("No events passed selection!")

        selected_events = pd.concat(all_selected_events, ignore_index=True)
        logger.info(f"Selected {len(selected_events)} events")

        logger.info("Calculating physics observables...")
        physics_data = self.kinematics_calc.calculate_all_observables(selected_events)

        logger.info("Engineering features...")
        features_df = self.feature_engineer.create_features(physics_data)

        X = features_df.drop(["cos_theta_star"], axis=1, errors="ignore")
        y = physics_data["cos_theta_star"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Training XGBoost model...")
        training_metrics = self.model.train(
            X_train_scaled, y_train, X_val_scaled, y_val, feature_names=list(X.columns)
        )

        logger.info("Evaluating model...")
        y_pred_test = self.model.predict(X_test_scaled)

        physics_metrics = PhysicsMetrics()
        evaluation_results = physics_metrics.evaluate_physics_consistency(
            y_test, y_pred_test
        )

        results = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": list(X.columns),
            "training_metrics": training_metrics,
            "physics_evaluation": evaluation_results,
            "test_predictions": y_pred_test,
            "test_truth": y_test,
            "feature_importance": self.model.get_feature_importance(),
        }

        logger.info("Training pipeline complete!")
        logger.info(f"Test R²: {evaluation_results['r2_score']:.4f}")

        return results


if __name__ == "__main__":
    atlas_files = ["/path/to/atlas/data/file1.root", "/path/to/atlas/data/file2.root"]

    pipeline = TrainingPipeline()
    results = pipeline.run_full_pipeline(atlas_files, max_events=100000)

    logger.info(
        f"Training complete! R² = {results['physics_evaluation']['r2_score']:.4f}"
    )
