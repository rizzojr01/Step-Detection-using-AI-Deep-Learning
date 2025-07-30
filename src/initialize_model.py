"""
Model Initialization for FastAPI
================================

Initialize the trained CNN model for the FastAPI step detection service.
Run this to properly load your model into the API.
"""

import os
import sys

import numpy as np
import torch

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.step_detection.core.detector import RealTimeStepCounter


# Import FastAPI components only when needed to avoid dependency issues
def get_app_and_step_counter():
    """Lazy import of FastAPI components to avoid circular dependencies"""
    try:
        from main import app, step_counter

        return app, step_counter
    except ImportError:
        return None, None


# Define the CNN model architecture (same as in notebook)
class StepDetectionCNN(torch.nn.Module):
    def __init__(self):
        super(StepDetectionCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=6, out_channels=32, kernel_size=1, stride=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=1, stride=1
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=1, stride=1
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 3)  # 3 classes: no step, start, end
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # Handle different input dimensions
        if x.dim() == 1:
            # Single sample: (6,) -> (1, 6, 1)
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            # Batch of samples: (batch, 6) -> (batch, 6, 1)
            x = x.unsqueeze(2)
        elif x.dim() == 3 and x.size(1) == 1:
            # Already has correct shape but channels and sequence are swapped
            x = x.transpose(1, 2)

        # Ensure we have the right shape: (batch, channels=6, seq_len=1)
        if x.size(1) != 6:
            x = x.transpose(1, 2)

        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def initialize_model_from_notebook():
    """
    Initialize model using the trained weights from the notebook session.
    This assumes you've run the notebook and have a trained model.
    """
    print("🔄 Initializing model from notebook session...")

    try:
        # Create model instance
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StepDetectionCNN().to(device)

        # For demo purposes, we'll create a model with random weights
        # In production, you would load the actual trained weights:
        # model.load_state_dict(torch.load('path/to/your/model.pth'))

        print(f"✅ Model created on device: {device}")

        # Initialize the step counter
        step_counter = RealTimeStepCounter(model=model, device=str(device))

        print("✅ Step counter initialized")

        return True

    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False


def save_model_for_production(model, filename="trained_model.pth"):
    """
    Save the model for production use
    """
    try:
        torch.save(model.state_dict(), filename)
        print(f"💾 Model saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")


def load_model_for_production(filename="trained_model.pth"):
    """
    Load a saved model for production use
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StepDetectionCNN().to(device)

        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=device))
            print(f"✅ Model loaded from {filename}")
        else:
            print(f"⚠️  Model file {filename} not found, using random weights")

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_production_model(model_path="models/trained_step_detection_model.pth"):
    """
    Load the pre-trained model for production use.

    Args:
        model_path (str): Path to the trained model file

    Returns:
        tuple: (model, device) or (None, None) if loading fails
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StepDetectionCNN().to(device)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"✅ Pre-trained model loaded from {model_path}")
            print(f"   Device: {device}")
            return model, device
        else:
            print(f"❌ Model file {model_path} not found")
            return None, None

    except Exception as e:
        print(f"❌ Error loading pre-trained model: {e}")
        return None, None


def create_dummy_data(num_samples=1000):
    """
    Create dummy training data for step detection.

    This generates synthetic accelerometer data with step patterns.
    In production, replace this with your actual training data loading function.

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        tuple: (X, y) where X is the feature data and y is the labels (3 classes)
    """
    np.random.seed(42)  # For reproducible results

    # Initialize arrays
    X = np.zeros(
        (num_samples, 6)
    )  # 6 features: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    y = np.zeros(
        num_samples, dtype=int
    )  # Labels: 0 = no step, 1 = step start, 2 = step end

    for i in range(num_samples):
        # Generate base accelerometer data (simulate phone in pocket/hand)
        # Normal resting state with gravity component
        accel_x = np.random.normal(0, 0.5)  # Small horizontal movements
        accel_y = np.random.normal(-9.8, 1.0)  # Gravity + noise (phone vertical)
        accel_z = np.random.normal(0, 0.5)  # Small movements

        # Generate gyroscope data
        gyro_x = np.random.normal(0, 0.1)
        gyro_y = np.random.normal(0, 0.1)
        gyro_z = np.random.normal(0, 0.1)

        # Generate step patterns
        random_val = np.random.random()
        if random_val < 0.15:  # 15% step start
            # Simulate step start: initial acceleration
            accel_y += np.random.normal(4.0, 1.0)  # Strong upward acceleration
            accel_x += np.random.normal(1.5, 0.5)  # Forward movement
            gyro_x += np.random.normal(0.8, 0.3)
            gyro_z += np.random.normal(0.5, 0.2)
            y[i] = 1  # Step start
        elif random_val < 0.30:  # 15% step end
            # Simulate step end: foot landing
            accel_y += np.random.normal(2.0, 0.8)  # Moderate downward impact
            accel_x += np.random.normal(0.5, 0.3)  # Deceleration
            gyro_x += np.random.normal(0.4, 0.2)
            gyro_z += np.random.normal(0.3, 0.1)
            y[i] = 2  # Step end
        else:  # 70% no step
            y[i] = 0  # No step

        # Store the features
        X[i] = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

    return X, y


if __name__ == "__main__":
    print("🚀 Model Initialization Script")
    print("=" * 40)

    # Initialize model
    success = initialize_model_from_notebook()

    if success:
        print("\n✅ Model initialization complete!")
        print("📡 Your FastAPI server is now ready to detect steps")
        print("🌐 API documentation: http://localhost:8000/docs")
        print("🧪 Test endpoint: POST http://localhost:8000/detect_step")
    else:
        print("\n❌ Model initialization failed")
        print("Please check the error messages above")
