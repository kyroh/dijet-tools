import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing config."""

    chunk_size: int = 50000
    max_memory_gb: float = 8.0
    cache_dir: str = "cache"
    output_format: str = "parquet"  # parquet, hdf5, zarr
    compression: str = "snappy"
    progress_interval: int = 10000


@dataclass
class PhysicsConfig:
    """Physics analysis configuration."""

    min_jets: int = 2
    pt_threshold_gev: float = 50.0
    eta_threshold: float = 4.5
    jvt_threshold: float = 0.5
    mjj_min_gev: float = 200.0
    pt_balance_min: float = 0.3
    max_delta_y: float = 10.0


@dataclass
class ModelConfig:
    """Machine learning model configuration."""

    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    feature_scaling: str = "robust"  # robust, standard, minmax

    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 50,
        }
    )

    nn_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_layers": [256, 128, 64, 32],
            "dropout_rate": 0.2,
            "batch_size": 256,
            "epochs": 200,
            "learning_rate": 0.001,
            "physics_loss_weight": 0.1,
        }
    )


@dataclass
class ComputeConfig:
    """Computing infrastructure configuration."""

    n_workers: int = 4
    scheduler_address: Optional[str] = None
    memory_limit_per_worker: str = "4GB"
    use_distributed: bool = False
    gpu_enabled: bool = False


@dataclass
class AnalysisConfig:
    """Full analysis config."""

    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    input_files: List[str] = field(default_factory=list)
    output_dir: str = "output"
    log_level: str = "INFO"
    experiment_name: str = "atlas_dijet_analysis"


class ConfigManager:
    """Manage config loading, saving, and validation."""

    @staticmethod
    def load_config(config_path: str) -> AnalysisConfig:
        """
        Load config from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            AnalysisConfig object
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        data_config = DataConfig(**config_dict.get("data", {}))
        physics_config = PhysicsConfig(**config_dict.get("physics", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        compute_config = ComputeConfig(**config_dict.get("compute", {}))

        top_level = {
            k: v
            for k, v in config_dict.items()
            if k not in ["data", "physics", "model", "compute"]
        }

        analysis_config = AnalysisConfig(
            data=data_config,
            physics=physics_config,
            model=model_config,
            compute=compute_config,
            **top_level,
        )

        logger.info(f"Loaded configuration from {config_path}")
        return analysis_config

    @staticmethod
    def save_config(config: AnalysisConfig, output_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: AnalysisConfig object to save
            output_path: Path where to save the YAML file
        """
        config_dict = asdict(config)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, indent=2, sort_keys=False
            )

        logger.info(f"Saved configuration to {output_path}")

    @staticmethod
    def create_default_config() -> AnalysisConfig:
        """Create configuration with default values."""
        return AnalysisConfig()

    @staticmethod
    def validate_config(config: AnalysisConfig) -> List[str]:
        """
        Validate config and return list of issues.

        Args:
            config: Config to validate

        Returns:
            List of error messages (empty if valid)
        """
        issues = []

        if config.data.chunk_size <= 0:
            issues.append("Data chunk_size must be positive")

        if config.data.max_memory_gb <= 0:
            issues.append("Data max_memory_gb must be positive")

        if config.physics.pt_threshold_gev <= 0:
            issues.append("Physics pt_threshold_gev must be positive")

        if config.physics.eta_threshold <= 0:
            issues.append("Physics eta_threshold must be positive")

        if not 0 < config.physics.jvt_threshold < 1:
            issues.append("Physics jvt_threshold must be between 0 and 1")

        if not 0 < config.model.test_size < 1:
            issues.append("Model test_size must be between 0 and 1")

        if not 0 < config.model.validation_size < 1:
            issues.append("Model validation_size must be between 0 and 1")

        if config.model.test_size + config.model.validation_size >= 1:
            issues.append("Sum of test_size and validation_size must be less than 1")

        if config.compute.n_workers <= 0:
            issues.append("Compute n_workers must be positive")

        if config.input_files:
            missing_files = [f for f in config.input_files if not Path(f).exists()]
            if missing_files:
                issues.append(f"Missing input files: {missing_files}")

        return issues
