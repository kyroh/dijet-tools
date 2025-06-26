import logging
from typing import Dict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class QCDValidator:
    """
    Validate ML predictions against QCD theoretical expectations.
    """

    def __init__(self):
        self.qcd_expectations = {
            "forward_scattering_fraction": 0.6,
            "mean_cos_theta_star": 0.75,
            "chi_distribution_slope": -1.0,
        }

    def validate_angular_distribution(
        self, cos_theta_pred: np.ndarray, cos_theta_true: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Validate angular distribution against QCD expectations.
        """
        results = {}

        forward_fraction = np.sum(cos_theta_pred > 0.8) / len(cos_theta_pred)
        results["forward_scattering_fraction"] = forward_fraction
        results["forward_scattering_deviation"] = abs(
            forward_fraction - self.qcd_expectations["forward_scattering_fraction"]
        )

        mean_cos_theta = np.mean(cos_theta_pred)
        results["mean_cos_theta_star"] = mean_cos_theta
        results["mean_deviation"] = abs(
            mean_cos_theta - self.qcd_expectations["mean_cos_theta_star"]
        )

        if cos_theta_true is not None:
            ks_stat, ks_pvalue = stats.ks_2samp(cos_theta_true, cos_theta_pred)
            results["ks_statistic"] = ks_stat
            results["ks_pvalue"] = ks_pvalue

        consistency_score = (
            results["forward_scattering_deviation"] * 2.0
            + results["mean_deviation"] * 1.0
        )
        results["qcd_consistency_score"] = consistency_score

        return results

    def validate_mass_spectrum(
        self, mjj: np.ndarray, weights: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Validate dijet mass spectrum against QCD expectations.
        """
        if weights is None:
            weights = np.ones(len(mjj))

        high_mass_mask = mjj > 500
        if np.sum(high_mass_mask) > 100:
            log_mjj = np.log(mjj[high_mass_mask])
            log_weights = np.log(weights[high_mass_mask] + 1e-10)

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_mjj, log_weights
            )

            return {
                "power_law_slope": slope,
                "power_law_r_squared": r_value**2,
                "expected_slope_range": (-5.0, -4.0),
                "slope_consistent": -5.0 < slope < -4.0,
            }
        else:
            return {"insufficient_high_mass_events": True}
