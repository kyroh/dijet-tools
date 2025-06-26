import logging
import sys
import time
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None, log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up comprehensive logging for the analysis.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Specific log file name (None for auto-generated)
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"atlas_dijet_{timestamp}.log"

    log_file_path = log_path / log_file

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("atlas_dijet")

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info(f"Logging initialized. Log file: {log_file_path}")
    logger.info(f"Log level: {log_level}")

    return logger
