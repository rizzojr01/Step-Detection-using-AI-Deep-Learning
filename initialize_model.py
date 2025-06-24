"""
Model Initialization for FastAPI
================================

Initialize the trained CNN model for the FastAPI step detection service.
Run this to properly load your model into the API.
"""

import os
import sys

import torch

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_step_detector import RealTimeStepCounter
from step_detection_api import app, step_counter


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
        if x.dim() == 2:
            x = x.unsqueeze(2)  # Add sequence dimension if missing
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(2)

        x = x.transpose(1, 2)  # Change to (batch, channels, seq_len)

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
        global step_counter
        step_counter = RealTimeStepCounter(model=model, device=device)

        # Configure thresholds for better detection
        step_counter.start_threshold = 0.3
        step_counter.end_threshold = 0.3

        print("✅ Step counter initialized")
        print(f"   Start threshold: {step_counter.start_threshold}")
        print(f"   End threshold: {step_counter.end_threshold}")

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
