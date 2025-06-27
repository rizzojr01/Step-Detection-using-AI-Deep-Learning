"""Configuration settings for the step detection project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model parameters
DEFAULT_MODEL_CONFIG = {
    "input_shape": (6,),
    "num_classes": 3,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "validation_split": 0.2,
}

# Step detection parameters
STEP_DETECTION_CONFIG = {
    "start_threshold": 0.3,
    "end_threshold": 0.3,
    "window_size": 50,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
}

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
