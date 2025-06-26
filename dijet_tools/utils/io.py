import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_results(
    results: Dict[str, Any], output_path: str, format: str = "pickle"
) -> str:
    """
    Save analysis results to file.

    Args:
        results: Dictionary containing analysis results
        output_path: Path where to save results
        format: File format ('pickle', 'joblib', 'json')

    Returns:
        Path to saved file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "pickle":
        with open(output_file, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif format == "joblib":
        joblib.dump(results, output_file, compress=3)

    elif format == "json":
        json_results = _prepare_for_json(results)
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)

    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Results saved to {output_file} (format: {format})")
    return str(output_file)


def load_results(
    file_path: Union[str, Path], format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load analysis results from file.

    Args:
        file_path: Path to results file
        format: File format (auto-detected if None)

    Returns:
        Dictionary containing loaded results
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    if format is None:
        suffix = file_path.suffix.lower()
        if suffix == ".pkl":
            format = "pickle"
        elif suffix == ".joblib":
            format = "joblib"
        elif suffix == ".json":
            format = "json"
        else:
            format = "pickle"

    if format == "pickle":
        with open(file_path, "rb") as f:
            results = pickle.load(f)

    elif format == "joblib":
        results = joblib.load(file_path)

    elif format == "json":
        with open(file_path, "r") as f:
            results = json.load(f)

    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Results loaded from {file_path} (format: {format})")
    return results


def _prepare_for_json(obj: Any) -> Any:
    """
    Recursively prepare object for JSON serialization.

    Args:
        obj: Object to prepare

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_prepare_for_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif hasattr(obj, "__dict__"):
        return {key: _prepare_for_json(value) for key, value in obj.__dict__.items()}
    else:
        return obj


def save_model(
    model: Any, model_path: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save trained model with optional metadata.

    Args:
        model: Trained model object
        model_path: Path where to save model
        metadata: Optional metadata dictionary

    Returns:
        Path to saved model
    """
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save"):
        model.save(model_file)
    else:
        joblib.dump(model, model_file)

    if metadata:
        metadata_file = model_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(_prepare_for_json(metadata), f, indent=2)
        logger.info(f"Model metadata saved to {metadata_file}")

    logger.info(f"Model saved to {model_file}")
    return str(model_file)


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load trained model from file.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model object
    """
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    try:
        import torch

        model = torch.load(model_file, map_location="cpu")
        logger.info(f"Loaded PyTorch model from {model_file}")
    except (ImportError, RuntimeError, ValueError):
        model = joblib.load(model_file)
        logger.info(f"Loaded model from {model_file}")

    return model
