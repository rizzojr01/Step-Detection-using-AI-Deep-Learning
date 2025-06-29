"""
Real-time Step Detection Module
Provides classes for real-time step detection using trained TensorFlow models.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Try to import keras correctly
try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        load_model = None

# Import configuration
from ..utils.config import get_config


class StepDetector:
    """Real-time step detector using TensorFlow CNN model with enhanced sensitivity control."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: Optional[float] = None,
        magnitude_threshold: Optional[float] = None,
        start_threshold: Optional[float] = None,  # Kept for backward compatibility
        end_threshold: Optional[float] = None,  # Kept for backward compatibility
        metadata_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the step detector with enhanced sensitivity controls.

        Args:
            model_path: Path to the saved TensorFlow model
            confidence_threshold: Minimum confidence for step detection (overrides config)
            magnitude_threshold: Minimum movement magnitude required (overrides config)
            start_threshold: Legacy threshold for detecting step starts
            end_threshold: Legacy threshold for detecting step ends
            metadata_path: Optional path to model metadata JSON for optimal parameters
            config_path: Optional path to config.yaml file
        """
        if load_model is None:
            raise ImportError("Could not import keras load_model function")
        self.model = load_model(model_path)

        # Load configuration
        self.config = get_config(config_path)

        # Initialize thresholds from config, then override with parameters
        self.confidence_threshold = (
            confidence_threshold or self.config.get_confidence_threshold()
        )
        self.magnitude_threshold = (
            magnitude_threshold or self.config.get_magnitude_threshold()
        )
        self.start_threshold = start_threshold or self.config.get_start_threshold()
        self.end_threshold = end_threshold or self.config.get_end_threshold()

        # Load optimal thresholds from metadata if available (overrides config)
        if metadata_path:
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    optimization = metadata.get("optimization", {})
                    if "confidence_threshold" in optimization:
                        self.confidence_threshold = optimization["confidence_threshold"]
                    if "magnitude_threshold" in optimization:
                        self.magnitude_threshold = optimization["magnitude_threshold"]
                    print(f"✅ Loaded optimized thresholds from metadata:")
                    print(f"   Confidence: {self.confidence_threshold}")
                    print(f"   Magnitude: {self.magnitude_threshold}")
            except (FileNotFoundError, json.JSONDecodeError):
                print("⚠️  Could not load metadata, using config/parameter values")

        # Get filter settings from config
        self.enable_magnitude_filter = self.config.is_magnitude_filter_enabled()
        self.enable_confidence_filter = self.config.is_confidence_filter_enabled()

        print(f"🔧 StepDetector initialized with:")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   Magnitude threshold: {self.magnitude_threshold}")
        print(
            f"   Magnitude filter: {'enabled' if self.enable_magnitude_filter else 'disabled'}"
        )
        print(
            f"   Confidence filter: {'enabled' if self.enable_confidence_filter else 'disabled'}"
        )

        self.reset()

    def _calculate_movement_magnitude(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
    ) -> float:
        """Calculate total movement magnitude from sensor data."""
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        return accel_mag + gyro_mag

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
        predicted_class = np.argmax(predictions[0])
        max_confidence = np.max(predictions[0])

        # Legacy probabilities for backward compatibility
        start_prob = predictions[0][1]
        end_prob = predictions[0][2]

        # Calculate movement magnitude to filter out small shakes
        movement_magnitude = self._calculate_movement_magnitude(
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        )

        # Enhanced step detection with configurable sensitivity controls
        step_detected = False
        filtered_class = 0  # Default to "No Step"

        # Apply filters based on configuration
        passes_confidence_filter = True
        passes_magnitude_filter = True

        if self.enable_confidence_filter:
            passes_confidence_filter = max_confidence >= self.confidence_threshold

        if self.enable_magnitude_filter:
            passes_magnitude_filter = movement_magnitude >= self.magnitude_threshold

        # Final detection decision
        if passes_confidence_filter and passes_magnitude_filter and predicted_class > 0:
            step_detected = True
            filtered_class = predicted_class

        # Legacy step detection for backward compatibility
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
                "max_confidence": float(max_confidence),
                "predicted_class": int(predicted_class),
                "filtered_class": int(filtered_class),
            },
            "sensitivity_control": {
                "movement_magnitude": float(movement_magnitude),
                "confidence_threshold": self.confidence_threshold,
                "magnitude_threshold": self.magnitude_threshold,
                "passed_filters": step_detected,
            },
            "step_start": step_start,  # Legacy
            "step_end": step_end,  # Legacy
            "step_detected": step_detected,  # Enhanced detection
            "step_count": self.step_count,
        }

        # Update step tracking using enhanced detection
        # Use filtered detection for more accurate step counting
        if (
            step_detected and filtered_class == 1 and self.current_step is None
        ):  # Step start
            self.current_step = {
                "start_time": result["timestamp"],
                "start_data": result["sensor_data"].copy(),
                "start_confidence": float(max_confidence),
                "start_magnitude": float(movement_magnitude),
            }

        if (
            step_detected and filtered_class == 2 and self.current_step is not None
        ):  # Step end
            self.current_step["end_time"] = result["timestamp"]
            self.current_step["end_data"] = result["sensor_data"].copy()
            self.current_step["end_confidence"] = float(max_confidence)
            self.current_step["end_magnitude"] = float(movement_magnitude)
            self.step_count += 1

            # Store completed step in session data
            self.session_data.append(self.current_step.copy())
            self.current_step = None

        # Legacy step tracking (kept for backward compatibility)
        elif step_start and self.current_step is None:
            self.current_step = {
                "start_time": result["timestamp"],
                "start_data": result["sensor_data"].copy(),
            }

        elif step_end and self.current_step is not None:
            self.current_step["end_time"] = result["timestamp"]
            self.current_step["end_data"] = result["sensor_data"].copy()
            # Only increment legacy counter if not using enhanced detection
            if not step_detected:
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
    """Simple step counter for basic step counting with enhanced sensitivity."""

    def __init__(
        self,
        model_path: str,
        threshold: Optional[float] = None,
        magnitude_threshold: Optional[float] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the step counter with configuration support.

        Args:
            model_path: Path to the saved TensorFlow model
            threshold: Confidence threshold for detecting steps (overrides config)
            magnitude_threshold: Minimum movement magnitude (overrides config)
            config_path: Optional path to config.yaml file
        """
        if load_model is None:
            raise ImportError("Could not import keras load_model function")
        self.model = load_model(model_path)

        # Load configuration
        self.config = get_config(config_path)

        # Set thresholds from config or parameters
        self.threshold = threshold or self.config.get_confidence_threshold()
        self.magnitude_threshold = (
            magnitude_threshold or self.config.get_magnitude_threshold()
        )

        # Get filter settings from config
        self.enable_magnitude_filter = self.config.is_magnitude_filter_enabled()

        print(f"🔧 SimpleStepCounter initialized with:")
        print(f"   Confidence threshold: {self.threshold}")
        print(f"   Magnitude threshold: {self.magnitude_threshold}")
        print(
            f"   Magnitude filter: {'enabled' if self.enable_magnitude_filter else 'disabled'}"
        )

        self.step_count = 0
        self.last_detection = None

    def reset(self):
        """Reset step count."""
        self.step_count = 0
        self.last_detection = None

    def count(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
    ) -> bool:
        """
        Process sensor reading and count steps with enhanced sensitivity control.

        Returns:
            True if step detected, False otherwise
        """
        # Calculate movement magnitude to filter small movements
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        movement_magnitude = accel_mag + gyro_mag

        # Skip detection if movement is too small (phone shake filter)
        if movement_magnitude < self.magnitude_threshold:
            return False

        # Prepare input for model
        input_data = np.array(
            [[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]], dtype=np.float32
        )

        # Get predictions
        predictions = self.model.predict(input_data, verbose=0)
        max_confidence = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])

        # Count step if both confidence and magnitude thresholds are met
        if (
            max_confidence > self.threshold and predicted_class > 0
        ):  # Only count actual step predictions (not "no label")
            self.step_count += 1
            self.last_detection = {
                "timestamp": datetime.now().isoformat(),
                "step_number": self.step_count,
                "confidence": float(max_confidence),
                "predicted_class": int(predicted_class),
                "movement_magnitude": float(movement_magnitude),
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
