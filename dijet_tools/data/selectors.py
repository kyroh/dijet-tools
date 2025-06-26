import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..physics.kinematics import DijetKinematics

logger = logging.getLogger(__name__)


class ATLASEventSelector:
    """Apply ATLAS-specific event selection for dijet analysis."""

    def __init__(self):
        self.default_selection = {
            "min_jets": 2,
            "pt_threshold_gev": 50.0,
            "eta_threshold": 4.5,
            "jvt_threshold": 0.5,
            "pt_balance_min": 0.3,
            "mjj_min_gev": 200.0,
        }

    def apply_jet_quality_cuts(
        self, df: pd.DataFrame, selection: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Apply jet quality and kinematic cuts."""
        if selection is None:
            selection = self.default_selection

        selected_events = []
        n_input = len(df)

        for idx, event in df.iterrows():
            # Extract jet arrays (convert from MeV to GeV)
            pts = np.array(event["AnalysisJetsAuxDyn.pt"]) / 1000.0
            etas = np.array(event["AnalysisJetsAuxDyn.eta"])
            phis = np.array(event["AnalysisJetsAuxDyn.phi"])
            masses = np.array(event["AnalysisJetsAuxDyn.m"]) / 1000.0
            jvts = np.array(event["AnalysisJetsAuxDyn.DFCommonJets_fJvt"])

            # Apply quality cuts
            quality_mask = (
                (pts > selection["pt_threshold_gev"])
                & (np.abs(etas) < selection["eta_threshold"])
                & (
                    (np.abs(etas) > 2.5) | (jvts > selection["jvt_threshold"])
                )  # JVT only for central jets
            )

            if np.sum(quality_mask) >= selection["min_jets"]:
                # Sort by pT
                good_indices = np.where(quality_mask)[0]
                pt_sorted_indices = good_indices[np.argsort(pts[good_indices])[::-1]]

                # Build event record
                event_record = {
                    "event_index": idx,
                    "n_jets": np.sum(quality_mask),
                    "leading_jet_pt": pts[pt_sorted_indices[0]],
                    "leading_jet_eta": etas[pt_sorted_indices[0]],
                    "leading_jet_phi": phis[pt_sorted_indices[0]],
                    "leading_jet_m": masses[pt_sorted_indices[0]],
                    "subleading_jet_pt": pts[pt_sorted_indices[1]],
                    "subleading_jet_eta": etas[pt_sorted_indices[1]],
                    "subleading_jet_phi": phis[pt_sorted_indices[1]],
                    "subleading_jet_m": masses[pt_sorted_indices[1]],
                }

                # Additional selection criteria
                pt_balance = (
                    event_record["subleading_jet_pt"] / event_record["leading_jet_pt"]
                )
                if pt_balance >= selection["pt_balance_min"]:
                    selected_events.append(event_record)

        result_df = pd.DataFrame(selected_events)

        if not result_df.empty:
            # Calculate mjj and apply mass cut
            kinematics = DijetKinematics()
            result_df["mjj"] = kinematics.calculate_mass(
                np.array(result_df["leading_jet_pt"]),
                np.array(result_df["leading_jet_eta"]),
                np.array(result_df["leading_jet_phi"]),
                np.array(result_df["leading_jet_m"]),
                np.array(result_df["subleading_jet_pt"]),
                np.array(result_df["subleading_jet_eta"]),
                np.array(result_df["subleading_jet_phi"]),
                np.array(result_df["subleading_jet_m"]),
            )

            # Apply mass cut
            mass_cut = result_df["mjj"] >= selection["mjj_min_gev"]
            result_df = result_df[mass_cut].reset_index(drop=True)
        else:
            # Ensure we have a proper DataFrame even when empty
            result_df = pd.DataFrame()

        efficiency = len(result_df) / n_input if n_input > 0 else 0
        logger.info(
            f"Event selection: {len(result_df)}/{n_input} events passed ({efficiency:.1%})"
        )

        return result_df
