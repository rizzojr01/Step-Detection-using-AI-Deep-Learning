"""
Real-time Step Detection Module
Provides classes for real-time step detection using trained TensorFlow models.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


class StepDetector:
    """Real-time step detector using TensorFlow CNN model."""

    def __init__(
        self,
        model_path: str,
        start_threshold: float = 0.03,
        end_threshold: float = 0.03,
    ):
        """
        Initialize the step detector.

        Args:
            model_path: Path to the saved TensorFlow model
            start_threshold: Threshold for detecting step starts
            end_threshold: Threshold for detecting step ends
        """
        self.model = tf.keras.models.load_model(model_path)
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.reset()

    def reset(self):
        """Reset detector state."""
        self.step_count = 0
        self.current_step = None
        self.session_data = []

    def process_reading(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
    ) -> Dict:
        """
        Process a single sensor reading and detect steps.

        Args:
            accel_x, accel_y, accel_z: Accelerometer readings
            gyro_x, gyro_y, gyro_z: Gyroscope readings

        Returns:
            Dictionary with detection results
        """
        # Prepare input for model
        input_data = np.array(
            [[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]], dtype=np.float32
        )

        # Get predictions
        predictions = self.model.predict(input_data, verbose=0)
        start_prob = predictions[0][1]
        end_prob = predictions[0][2]

        # Detect step events
        step_start = start_prob > self.start_threshold
        step_end = end_prob > self.end_threshold

        result = {
            "timestamp": datetime.now().isoformat(),
            "sensor_data": {
                "accel_x": accel_x,
                "accel_y": accel_y,
                "accel_z": accel_z,
                "gyro_x": gyro_x,
                "gyro_y": gyro_y,
                "gyro_z": gyro_z,
            },
            "predictions": {
                "start_prob": float(start_prob),
                "end_prob": float(end_prob),
            },
            "step_start": step_start,
            "step_end": step_end,
            "step_count": self.step_count,
        }

        # Update step tracking
        if step_start and self.current_step is None:
            self.current_step = {
                "start_time": result["timestamp"],
                "start_data": result["sensor_data"].copy(),
            }

        if step_end and self.current_step is not None:
            self.current_step["end_time"] = result["timestamp"]
            self.current_step["end_data"] = result["sensor_data"].copy()
            self.step_count += 1
            result["completed_step"] = self.current_step.copy()
            self.current_step = None

        # Store session data
        self.session_data.append(result)

        return result

    def get_step_count(self) -> int:
        """Get current step count."""
        return self.step_count

    def get_session_summary(self) -> Dict:
        """Get summary of the current session."""
        return {
            "total_readings": len(self.session_data),
            "total_steps": self.step_count,
            "current_step_in_progress": self.current_step is not None,
            "thresholds": {
                "start_threshold": self.start_threshold,
                "end_threshold": self.end_threshold,
            },
        }

    def save_session(self, filename: str):
        """Save session data to file."""
        session_summary = {
            "session_info": self.get_session_summary(),
            "data": self.session_data,
        }
        with open(filename, "w") as f:
            json.dump(session_summary, f, indent=2)


class SimpleStepCounter:
    """Simple step counter for basic step counting."""

    def __init__(self, model_path: str, threshold: float = 0.03):
        """
        Initialize the step counter.

        Args:
            model_path: Path to the saved TensorFlow model
            threshold: Threshold for detecting steps
        """
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset step count."""
        self.step_count = 0
        self.last_detection = None

    def process_reading(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
    ) -> bool:
        """
        Process sensor reading and count steps.

        Returns:
            True if step detected, False otherwise
        """
        # Prepare input for model
        input_data = np.array(
            [[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]], dtype=np.float32
        )

        # Get predictions
        predictions = self.model.predict(input_data, verbose=0)
        start_prob = predictions[0][1]

        # Count step if threshold exceeded
        if start_prob > self.threshold:
            self.step_count += 1
            self.last_detection = {
                "timestamp": datetime.now().isoformat(),
                "step_number": self.step_count,
                "confidence": float(start_prob),
            }
            return True

        return False

    def get_count(self) -> int:
        """Get current step count."""
        return self.step_count


def load_model_info(metadata_path: str) -> Dict:
    """Load model metadata."""
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


if __name__ == "__main__":
    # Example usage
    print("Step Detection Module")
    print("====================")

    # This would require a trained model
    # detector = StepDetector("models/step_detection_model.keras")
    # counter = SimpleStepCounter("models/step_detection_model.keras")

    print("Classes defined successfully!")
    print("Use StepDetector for comprehensive step tracking")
    print("Use SimpleStepCounter for basic step counting")
