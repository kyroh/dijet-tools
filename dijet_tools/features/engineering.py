import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create physics-informed features for ML training.
    """

    def __init__(self):
        self.feature_names = []
        self.feature_descriptions = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from basic dijet observables.
        """
        logger.info("Creating engineered features...")

        features_df = df.copy()

        basic_features = [
            "mjj",
            "delta_y",
            "chi",
            "pt_balance",
            "delta_phi",
            "leading_jet_pt",
            "subleading_jet_pt",
            "leading_jet_eta",
            "subleading_jet_eta",
        ]

        if "mjj" in features_df.columns:
            features_df["log_mjj"] = np.log(features_df["mjj"])
            self.feature_descriptions["log_mjj"] = (
                "Log of dijet mass (handles dynamic range)"
            )

        if "chi" in features_df.columns:
            features_df["log_chi"] = np.log(features_df["chi"])
            self.feature_descriptions["log_chi"] = "Log of chi variable"

        if "pt_balance" in features_df.columns:
            features_df["sqrt_pt_balance"] = np.sqrt(features_df["pt_balance"])
            features_df["pt_balance_squared"] = features_df["pt_balance"] ** 2
            self.feature_descriptions["sqrt_pt_balance"] = "Square root of pT balance"

        if all(
            col in features_df.columns
            for col in ["leading_jet_eta", "subleading_jet_eta"]
        ):
            features_df["eta_centrality"] = (
                features_df["leading_jet_eta"] + features_df["subleading_jet_eta"]
            ) / 2
            features_df["eta_asymmetry"] = (
                np.abs(
                    features_df["leading_jet_eta"] - features_df["subleading_jet_eta"]
                )
                / 2
            )
            self.feature_descriptions["eta_centrality"] = "Average pseudorapidity"
            self.feature_descriptions["eta_asymmetry"] = "Pseudorapidity asymmetry"

        if all(
            col in features_df.columns
            for col in ["leading_jet_pt", "subleading_jet_pt"]
        ):
            features_df["geometric_mean_pt"] = np.sqrt(
                features_df["leading_jet_pt"] * features_df["subleading_jet_pt"]
            )
            features_df["harmonic_mean_pt"] = 2 / (
                1 / features_df["leading_jet_pt"] + 1 / features_df["subleading_jet_pt"]
            )
            features_df["pt_ratio"] = (
                features_df["leading_jet_pt"] / features_df["subleading_jet_pt"]
            )

            self.feature_descriptions["geometric_mean_pt"] = (
                "Geometric mean of jet pT values"
            )
            self.feature_descriptions["harmonic_mean_pt"] = (
                "Harmonic mean of jet pT values"
            )

        if all(
            col in features_df.columns
            for col in ["mjj", "leading_jet_pt", "subleading_jet_pt"]
        ):
            features_df["mjj_over_pt_sum"] = features_df["mjj"] / (
                features_df["leading_jet_pt"] + features_df["subleading_jet_pt"]
            )
            self.feature_descriptions["mjj_over_pt_sum"] = (
                "Dijet mass normalized by total pT"
            )

        if all(col in features_df.columns for col in ["delta_y", "delta_phi"]):
            features_df["angular_product"] = (
                features_df["delta_y"] * features_df["delta_phi"]
            )
            features_df["angular_ratio"] = features_df["delta_y"] / (
                features_df["delta_phi"] + 1e-6
            )

            self.feature_descriptions["angular_product"] = (
                "Product of angular separations"
            )
            self.feature_descriptions["angular_ratio"] = (
                "Ratio of rapidity to azimuthal separation"
            )

        if "delta_y" in features_df.columns:
            features_df["delta_y_squared"] = features_df["delta_y"] ** 2
            features_df["delta_y_cubed"] = features_df["delta_y"] ** 3

        if "eta_centrality" in features_df.columns and "delta_y" in features_df.columns:
            features_df["boost_invariant"] = features_df["delta_y"] / (
                1 + np.abs(features_df["eta_centrality"])
            )

        self.feature_names = [
            col
            for col in features_df.columns
            if col not in ["event_index", "file_index", "chunk_start"]
        ]

        logger.info(f"Created {len(self.feature_names)} features")
        return features_df

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all engineered features."""
        return self.feature_descriptions
