import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QCDObservables:
    """
    Calculate QCD observables and theoretical predictions for dijet events.
    """

    @staticmethod
    def calculate_boost_invariants(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate boost-invariant variables for QCD analysis."""
        result = df.copy()

        if all(
            col in df.columns
            for col in ["leading_jet_pt", "subleading_jet_pt", "delta_y"]
        ):
            result["invariant_ht"] = df["leading_jet_pt"] + df["subleading_jet_pt"]
            result["boost_weight"] = np.exp(-0.5 * df["delta_y"] ** 2)

        return result

    @staticmethod
    def qcd_cross_section_weight(
        mjj: Union[float, np.ndarray], process_type: str = "inclusive"
    ) -> Union[float, np.ndarray]:
        """
        Calculate QCD cross-section weights based on dijet mass.

        Args:
            mjj: Dijet invariant mass in GeV
            process_type: QCD process type ('inclusive', 'qq', 'qg', 'gg')
        """
        if process_type == "inclusive":
            return 1.0 / (mjj**4.5)
        elif process_type == "qq":
            return 0.4 / (mjj**4.3)
        elif process_type == "qg":
            return 0.35 / (mjj**4.4)
        elif process_type == "gg":
            return 0.25 / (mjj**4.6)
        else:
            return 1.0 / (mjj**4.5)

    @staticmethod
    def parton_level_correction(
        cos_theta_star: Union[float, np.ndarray], mjj: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Apply parton-level corrections for hadronization and detector effects.
        """
        energy_correction = 1.0 + 0.1 * np.exp(
            -mjj / 1000.0
        )
        angular_correction = 1.0 + 0.05 * cos_theta_star

        return cos_theta_star * energy_correction * angular_correction
