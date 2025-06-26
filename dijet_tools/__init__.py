__version__ = "1.0.0"
__author__ = "Aaron W. Tarajos"
__email__ = "awtarajos@berkeley.edu"
__description__ = (
    "Machine Learning Analysis of Dijet Angular Scattering in ATLAS Run 2 Data"
)

from .data.loaders import ATLASDataLoader
from .data.processors import LargeScaleATLASProcessor, ProcessingConfig
from .data.selectors import ATLASEventSelector
from .evaluation.metrics import PhysicsMetrics
from .features.engineering import FeatureEngineer
from .models.xgboost_model import XGBoostPredictor
from .physics.kinematics import DijetKinematics
from .physics.observables import QCDObservables
from .pipeline.training import TrainingPipeline

__all__ = [
    "ATLASDataLoader",
    "LargeScaleATLASProcessor",
    "ProcessingConfig",
    "ATLASEventSelector",
    "DijetKinematics",
    "QCDObservables",
    "XGBoostPredictor",
    "FeatureEngineer",
    "PhysicsMetrics",
    "TrainingPipeline",
]
