import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DijetKinematics:
    """
    Dijet kinematics and QCD observables from jet four-vectors.
    """

    @staticmethod
    def calculate_mass(
        pt1: Union[float, np.ndarray],
        eta1: Union[float, np.ndarray],
        phi1: Union[float, np.ndarray],
        m1: Union[float, np.ndarray],
        pt2: Union[float, np.ndarray],
        eta2: Union[float, np.ndarray],
        phi2: Union[float, np.ndarray],
        m2: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Dijet invariant mass from four-vectors."""
        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)
        E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)
        E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

        E_total = E1 + E2
        px_total = px1 + px2
        py_total = py1 + py2
        pz_total = pz1 + pz2

        mjj_squared = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
        return np.sqrt(np.maximum(mjj_squared, 0))

    @staticmethod
    def calculate_rapidity_separation(
        eta1: Union[float, np.ndarray], eta2: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Rapidity separation |y1 - y2|."""
        return np.abs(eta1 - eta2)

    @staticmethod
    def calculate_azimuthal_separation(
        phi1: Union[float, np.ndarray], phi2: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Azimuthal separation, 2π periodicity."""
        delta_phi = phi1 - phi2
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
        return np.abs(delta_phi)

    @staticmethod
    def calculate_chi_variable(
        eta1: Union[float, np.ndarray], eta2: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """χ = exp(|Δy|), related to scattering angle."""
        delta_y = DijetKinematics.calculate_rapidity_separation(eta1, eta2)
        return np.exp(delta_y)

    @staticmethod
    def calculate_cos_theta_star(
        eta1: Union[float, np.ndarray], eta2: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """|cos(θ*)| from rapidity separation."""
        chi = DijetKinematics.calculate_chi_variable(eta1, eta2)
        return (chi - 1) / (chi + 1)

    def calculate_all_observables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all dijet observables for a DataFrame.

        Expected columns: AnalysisJetsAuxDyn.pt, AnalysisJetsAuxDyn.eta, AnalysisJetsAuxDyn.phi, AnalysisJetsAuxDyn.m
        """
        result = df.copy()

        result["mjj"] = self.calculate_mass(
            np.array(df["AnalysisJetsAuxDyn.pt"]),
            np.array(df["AnalysisJetsAuxDyn.eta"]),
            np.array(df["AnalysisJetsAuxDyn.phi"]),
            np.array(df["AnalysisJetsAuxDyn.m"]),
            np.array(df["AnalysisJetsAuxDyn.pt"]),
            np.array(df["AnalysisJetsAuxDyn.eta"]),
            np.array(df["AnalysisJetsAuxDyn.phi"]),
            np.array(df["AnalysisJetsAuxDyn.m"]),
        )

        result["delta_y"] = self.calculate_rapidity_separation(
            np.array(df["AnalysisJetsAuxDyn.eta"]),
            np.array(df["AnalysisJetsAuxDyn.eta"]),
        )

        result["delta_phi"] = self.calculate_azimuthal_separation(
            np.array(df["AnalysisJetsAuxDyn.phi"]),
            np.array(df["AnalysisJetsAuxDyn.phi"]),
        )

        result["chi"] = self.calculate_chi_variable(
            np.array(df["AnalysisJetsAuxDyn.eta"]),
            np.array(df["AnalysisJetsAuxDyn.eta"]),
        )

        result["cos_theta_star"] = self.calculate_cos_theta_star(
            np.array(df["AnalysisJetsAuxDyn.eta"]),
            np.array(df["AnalysisJetsAuxDyn.eta"]),
        )

        result["pt_balance"] = df["AnalysisJetsAuxDyn.pt"] / df["AnalysisJetsAuxDyn.pt"]
        result["dijet_pt"] = df["AnalysisJetsAuxDyn.pt"] + df["AnalysisJetsAuxDyn.pt"]
        result["average_pt"] = (
            df["AnalysisJetsAuxDyn.pt"] + df["AnalysisJetsAuxDyn.pt"]
        ) / 2
        result["eta_centrality"] = (
            df["AnalysisJetsAuxDyn.eta"] + df["AnalysisJetsAuxDyn.eta"]
        ) / 2

        logger.info(f"Calculated observables for {len(result)} events")
        return result

    def calculate_processed_observables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all dijet observables for processed DataFrame.

        Expected columns: leading_jet_pt, leading_jet_eta, leading_jet_phi, leading_jet_m,
                         subleading_jet_pt, subleading_jet_eta, subleading_jet_phi, subleading_jet_m
        """
        result = df.copy()

        result["mjj"] = self.calculate_mass(
            np.array(df["leading_jet_pt"]),
            np.array(df["leading_jet_eta"]),
            np.array(df["leading_jet_phi"]),
            np.array(df["leading_jet_m"]),
            np.array(df["subleading_jet_pt"]),
            np.array(df["subleading_jet_eta"]),
            np.array(df["subleading_jet_phi"]),
            np.array(df["subleading_jet_m"]),
        )

        result["delta_y"] = self.calculate_rapidity_separation(
            np.array(df["leading_jet_eta"]), np.array(df["subleading_jet_eta"])
        )

        result["delta_phi"] = self.calculate_azimuthal_separation(
            np.array(df["leading_jet_phi"]), np.array(df["subleading_jet_phi"])
        )

        result["chi"] = self.calculate_chi_variable(
            np.array(df["leading_jet_eta"]), np.array(df["subleading_jet_eta"])
        )

        result["cos_theta_star"] = self.calculate_cos_theta_star(
            np.array(df["leading_jet_eta"]), np.array(df["subleading_jet_eta"])
        )

        result["pt_balance"] = df["subleading_jet_pt"] / df["leading_jet_pt"]
        result["dijet_pt"] = df["leading_jet_pt"] + df["subleading_jet_pt"]
        result["average_pt"] = (df["leading_jet_pt"] + df["subleading_jet_pt"]) / 2
        result["eta_centrality"] = (
            df["leading_jet_eta"] + df["subleading_jet_eta"]
        ) / 2

        logger.info(f"Calculated observables for {len(result)} events")
        return result
