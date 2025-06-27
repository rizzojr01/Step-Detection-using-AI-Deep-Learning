"""
Step Detection Package
A comprehensive solution for real-time step detection using deep learning.
"""

__version__ = "1.0.0"
__author__ = "Step Detection Team"
__email__ = "contact@stepdetection.com"

# Core functionality
from .core.detector import SimpleStepCounter, StepDetector

# Model utilities
from .models.model_utils import (
    create_cnn_model,
    evaluate_model,
    save_model_and_metadata,
    train_model,
)

# Data processing utilities
from .utils.data_processor import load_step_data, prepare_data_for_training

__all__ = [
    "StepDetector",
    "SimpleStepCounter",
    "load_step_data",
    "prepare_data_for_training",
    "create_cnn_model",
    "train_model",
    "evaluate_model",
    "save_model_and_metadata",
]
