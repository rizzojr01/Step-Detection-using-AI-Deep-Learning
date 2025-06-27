"""
Tests for the step detection package.
"""

import numpy as np
import pytest

from src.step_detection.models.model_utils import create_cnn_model
from src.step_detection.utils.data_processor import prepare_data_for_training


def test_model_creation():
    """Test that the CNN model can be created."""
    model = create_cnn_model()
    assert model is not None
    assert model.input_shape == (None, 6)
    assert model.output_shape == (None, 3)


def test_data_preparation():
    """Test data preparation with mock data."""
    import pandas as pd

    # Create mock data
    mock_data = pd.DataFrame(
        {
            "accel_x": np.random.randn(100),
            "accel_y": np.random.randn(100),
            "accel_z": np.random.randn(100),
            "gyro_x": np.random.randn(100),
            "gyro_y": np.random.randn(100),
            "gyro_z": np.random.randn(100),
            "Label": ["No Label"] * 80 + ["start"] * 10 + ["end"] * 10,
        }
    )

    # Test data preparation
    train_X, val_X, train_y, val_y = prepare_data_for_training(mock_data)

    assert train_X.shape[1] == 6  # 6 features
    assert train_y.shape[1] == 3  # 3 classes
    assert val_X.shape[1] == 6
    assert val_y.shape[1] == 3
    assert len(train_X) > len(val_X)  # Train set should be larger


def test_package_imports():
    """Test that all main package components can be imported."""
    from src.step_detection import (
        SimpleStepCounter,
        StepDetector,
        app,
        create_cnn_model,
        evaluate_model,
        load_step_data,
        prepare_data_for_training,
        save_model_and_metadata,
        train_model,
    )

    # Test that imports work
    assert callable(load_step_data)
    assert callable(prepare_data_for_training)
    assert callable(create_cnn_model)
    assert callable(train_model)
    assert callable(evaluate_model)
    assert callable(save_model_and_metadata)
    assert StepDetector is not None
    assert SimpleStepCounter is not None
    assert app is not None


if __name__ == "__main__":
    pytest.main([__file__])
