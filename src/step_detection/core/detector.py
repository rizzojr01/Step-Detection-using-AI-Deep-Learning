import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        load_model = None

from ..utils.config import get_config


class StepDetector:
    """
    Enhanced step detector with sensitivity controls and time constraints
    to prevent unrealistic step detection rates.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: Optional[float] = None,
        magnitude_threshold: Optional[float] = None,
        metadata_path: Optional[str] = None,
        config_path: Optional[str] = None,
        # Time constraint parameters
        min_step_interval: float = 0.3,  # Minimum 300ms between steps (max ~3.3 steps/sec)
        max_step_rate: float = 4.0,  # Maximum 4 steps per second
        step_rate_window: float = 1.0,  # Time window for rate calculation (1 second)
        min_step_duration: float = 0.1,  # Minimum step duration (100ms)
        max_step_duration: float = 2.0,  # Maximum step duration (2 seconds)
        enable_time_constraints: bool = True,  # Enable/disable time constraints
    ):
        """
        Initialize the step detector with enhanced sensitivity controls and time constraints.

        Args:
            model_path: Path to the saved TensorFlow model
            confidence_threshold: Minimum confidence for step detection (overrides config)
            magnitude_threshold: Minimum movement magnitude required (overrides config)
            metadata_path: Optional path to model metadata JSON for optimal parameters
            config_path: Optional path to config.yaml file
            min_step_interval: Minimum time between consecutive steps (seconds)
            max_step_rate: Maximum steps per second allowed
            step_rate_window: Time window for calculating step rate (seconds)
            min_step_duration: Minimum duration for a complete step (seconds)
            max_step_duration: Maximum duration for a complete step (seconds)
            enable_time_constraints: Whether to enable time constraint validation
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

        # Time constraint parameters
        self.enable_time_constraints = enable_time_constraints
        self.min_step_interval = min_step_interval
        self.max_step_rate = max_step_rate
        self.step_rate_window = step_rate_window
        self.min_step_duration = min_step_duration
        self.max_step_duration = max_step_duration

        # Time tracking variables
        self.last_step_time = 0.0
        self.current_step_start_time = None
        self.step_timestamps = deque()
        self.consecutive_rejections = 0

        print(f"🔧 StepDetector initialized with:")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   Magnitude threshold: {self.magnitude_threshold}")
        print(
            f"   Magnitude filter: {'enabled' if self.enable_magnitude_filter else 'disabled'}"
        )
        print(
            f"   Confidence filter: {'enabled' if self.enable_confidence_filter else 'disabled'}"
        )

        if self.enable_time_constraints:
            print(f"🕒 Time constraints enabled:")
            print(f"   Min step interval: {min_step_interval}s")
            print(f"   Max step rate: {max_step_rate} steps/sec")
            print(
                f"   Step duration range: {min_step_duration}s - {max_step_duration}s"
            )
        else:
            print("⏰ Time constraints disabled")

        self.reset()

    def reset(self):
        """Reset the detector state."""
        self.total_readings = 0
        self.step_count = 0
        self.in_step = False
        self.step_start_count = 0
        self.step_end_count = 0

        # Reset time tracking
        self.last_step_time = 0.0
        self.current_step_start_time = None
        self.step_timestamps.clear()
        self.consecutive_rejections = 0

    def _get_current_time(self) -> float:
        """Get current timestamp."""
        return time.time()

    def _is_step_rate_valid(self, current_time: float) -> bool:
        """
        Check if current step rate is within realistic limits.

        Args:
            current_time: Current timestamp

        Returns:
            True if step rate is valid, False otherwise
        """
        if not self.enable_time_constraints:
            return True

        # Remove old timestamps outside the window
        cutoff_time = current_time - self.step_rate_window
        while self.step_timestamps and self.step_timestamps[0] < cutoff_time:
            self.step_timestamps.popleft()

        # Check if adding another step would exceed max rate
        # We add 1 because we're checking if we can add a new step
        steps_in_window = len(self.step_timestamps)
        return (steps_in_window + 1) <= self.max_step_rate

    def _is_step_interval_valid(self, current_time: float) -> bool:
        """
        Check if enough time has passed since the last step.

        Args:
            current_time: Current timestamp

        Returns:
            True if interval is valid, False otherwise
        """
        if not self.enable_time_constraints or self.last_step_time == 0:
            return True

        time_since_last_step = current_time - self.last_step_time
        return time_since_last_step >= self.min_step_interval

    def _is_step_duration_valid(self, start_time: float, end_time: float) -> bool:
        """
        Check if step duration is within realistic limits.

        Args:
            start_time: Step start timestamp
            end_time: Step end timestamp

        Returns:
            True if duration is valid, False otherwise
        """
        if not self.enable_time_constraints:
            return True

        duration = end_time - start_time
        return self.min_step_duration <= duration <= self.max_step_duration

    def _register_step(self, current_time: float) -> None:
        """
        Register a valid step with timestamp tracking.

        Args:
            current_time: Current timestamp
        """
        # Always track step timestamps for consistency
        self.step_timestamps.append(current_time)
        self.last_step_time = current_time

        if self.enable_time_constraints:
            self.consecutive_rejections = 0

    def process_reading(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        timestamp: Optional[float] = None,
    ) -> Dict:
        """
        Process a single sensor reading with time constraints.

        Args:
            accel_x, accel_y, accel_z: Accelerometer readings (m/s²)
            gyro_x, gyro_y, gyro_z: Gyroscope readings (rad/s)
            timestamp: Optional timestamp (uses current time if None)

        Returns:
            Dictionary containing detection results and time constraint info
        """
        current_time = timestamp if timestamp is not None else self._get_current_time()

        # Increment reading counter
        self.total_readings += 1

        # Prepare input for model
        input_data = np.array(
            [[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]], dtype=np.float32
        )

        # Get model predictions
        predictions = self.model.predict(input_data, verbose=0)[0]

        # Calculate movement magnitude for filtering
        movement_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Extract prediction probabilities
        no_step_prob = float(predictions[0])
        start_prob = float(predictions[1])
        end_prob = float(predictions[2])
        max_confidence = max(start_prob, end_prob)

        # Initialize result structure
        result = {
            "step_detected": False,
            "step_start_detected": False,
            "step_end_detected": False,
            "completed_step": False,
            "step_count": self.step_count,
            "timestamp": current_time,
            "predictions": {
                "no_step_prob": no_step_prob,
                "start_prob": start_prob,
                "end_prob": end_prob,
                "max_confidence": max_confidence,
            },
            "sensitivity_control": {
                "movement_magnitude": movement_magnitude,
                "magnitude_filter_passed": True,
                "confidence_filter_passed": True,
                "passed_filters": True,
                "confidence_threshold": self.confidence_threshold,
                "magnitude_threshold": self.magnitude_threshold,
                "enable_magnitude_filter": self.enable_magnitude_filter,
                "enable_confidence_filter": self.enable_confidence_filter,
            },
            "time_constraints": {
                "enabled": self.enable_time_constraints,
                "current_time": current_time,
                "step_rate_valid": True,
                "step_interval_valid": True,
                "step_duration_valid": True,
                "rejection_reason": None,
                "current_step_rate": 0.0,
                "time_since_last_step": (
                    current_time - self.last_step_time
                    if self.last_step_time > 0
                    else float("inf")
                ),
            },
        }

        # Calculate current step rate for time constraints
        if self.enable_time_constraints:
            cutoff_time = current_time - self.step_rate_window
            recent_steps = [t for t in self.step_timestamps if t >= cutoff_time]
            result["time_constraints"]["current_step_rate"] = (
                len(recent_steps) / self.step_rate_window
            )

        # Apply magnitude filter
        magnitude_passed = True
        if (
            self.enable_magnitude_filter
            and movement_magnitude < self.magnitude_threshold
        ):
            result["sensitivity_control"]["magnitude_filter_passed"] = False
            magnitude_passed = False

        # Apply confidence filter
        confidence_passed = True
        if self.enable_confidence_filter and max_confidence < self.confidence_threshold:
            result["sensitivity_control"]["confidence_filter_passed"] = False
            confidence_passed = False

        # Update the passed_filters status (DEBUG: filters working)
        result["sensitivity_control"]["passed_filters"] = (
            magnitude_passed and confidence_passed
        )

        # If filters failed, return early
        if not (magnitude_passed and confidence_passed):
            return result

        # Update filter status for successful cases
        result["sensitivity_control"]["magnitude_filter_passed"] = magnitude_passed
        result["sensitivity_control"]["confidence_filter_passed"] = confidence_passed

        # Decide between start and end detection when both exceed threshold
        both_detected = (
            start_prob > self.confidence_threshold
            and end_prob > self.confidence_threshold
        )

        # Detect step start
        if start_prob > self.confidence_threshold and (
            not both_detected or start_prob >= end_prob
        ):
            # Check time constraints for step start
            if self.enable_time_constraints:
                rate_valid = self._is_step_rate_valid(current_time)
                interval_valid = self._is_step_interval_valid(current_time)
                result["time_constraints"]["step_rate_valid"] = rate_valid
                result["time_constraints"]["step_interval_valid"] = interval_valid

                if not rate_valid:
                    result["time_constraints"][
                        "rejection_reason"
                    ] = "step_start_rate_exceeded"
                    self.consecutive_rejections += 1
                    return result

                if not interval_valid:
                    result["time_constraints"][
                        "rejection_reason"
                    ] = "step_start_interval_too_short"
                    self.consecutive_rejections += 1
                    return result

            # Valid step start detected
            if not self.in_step:
                self.in_step = True
                self.current_step_start_time = current_time
                self.step_start_count += 1
                result["step_start_detected"] = True
                result["step_detected"] = True

        # Detect step end (only if not handling start, or if end has higher confidence)
        elif end_prob > self.confidence_threshold:
            if self.in_step and self.current_step_start_time is not None:
                # Check step duration constraints
                duration_valid = True
                if self.enable_time_constraints:
                    duration_valid = self._is_step_duration_valid(
                        self.current_step_start_time, current_time
                    )
                    result["time_constraints"]["step_duration_valid"] = duration_valid

                    if not duration_valid:
                        step_duration = current_time - self.current_step_start_time
                        if step_duration < self.min_step_duration:
                            result["time_constraints"][
                                "rejection_reason"
                            ] = "step_duration_too_short"
                        else:
                            result["time_constraints"][
                                "rejection_reason"
                            ] = "step_duration_too_long"
                        self.consecutive_rejections += 1
                        # Reset step state but don't count as completed step
                        self.in_step = False
                        self.current_step_start_time = None
                        return result

                # Check time constraints for step completion
                if self.enable_time_constraints:
                    rate_valid = self._is_step_rate_valid(current_time)
                    interval_valid = self._is_step_interval_valid(current_time)
                    result["time_constraints"]["step_rate_valid"] = rate_valid
                    result["time_constraints"]["step_interval_valid"] = interval_valid

                    if not rate_valid:
                        result["time_constraints"][
                            "rejection_reason"
                        ] = "step_rate_exceeded"
                        self.consecutive_rejections += 1
                        # Reset step state but don't count as completed step
                        self.in_step = False
                        self.current_step_start_time = None
                        return result

                    if not interval_valid:
                        result["time_constraints"][
                            "rejection_reason"
                        ] = "step_interval_too_short"
                        self.consecutive_rejections += 1
                        # Reset step state but don't count as completed step
                        self.in_step = False
                        self.current_step_start_time = None
                        return result

                # Valid step end detected - complete the step
                self.in_step = False
                self.step_end_count += 1
                self.step_count += 1

                # Register the step for time tracking
                self._register_step(current_time)

                result["step_end_detected"] = True
                result["step_detected"] = True
                result["completed_step"] = True
                result["step_count"] = self.step_count

                # Add step timing information
                if self.current_step_start_time is not None:
                    step_duration = current_time - self.current_step_start_time
                    result["step_timing"] = {
                        "start_time": self.current_step_start_time,
                        "end_time": current_time,
                        "duration": step_duration,
                    }

                self.current_step_start_time = None

        return result

    def get_step_count(self) -> int:
        """Get the current step count."""
        return self.step_count

    def get_session_summary(self) -> Dict:
        """Get a summary of the current detection session."""
        current_time = self._get_current_time()

        # Calculate session duration
        session_duration = current_time - (
            self.step_timestamps[0] if self.step_timestamps else current_time
        )

        # Calculate average step rate
        avg_step_rate = (
            self.step_count / session_duration if session_duration > 0 else 0.0
        )

        # Calculate recent step rate
        recent_cutoff = current_time - self.step_rate_window
        recent_steps = [t for t in self.step_timestamps if t >= recent_cutoff]
        recent_step_rate = len(recent_steps) / self.step_rate_window

        return {
            "total_readings": self.total_readings,
            "total_steps": self.step_count,
            "step_starts": self.step_start_count,
            "step_ends": self.step_end_count,
            "in_step": self.in_step,
            "session_duration": session_duration,
            "average_step_rate": avg_step_rate,
            "recent_step_rate": recent_step_rate,
            "consecutive_rejections": self.consecutive_rejections,
            "time_constraints": {
                "enabled": self.enable_time_constraints,
                "min_step_interval": self.min_step_interval,
                "max_step_rate": self.max_step_rate,
                "step_rate_window": self.step_rate_window,
                "min_step_duration": self.min_step_duration,
                "max_step_duration": self.max_step_duration,
            },
            "thresholds": {
                "confidence_threshold": self.confidence_threshold,
                "magnitude_threshold": self.magnitude_threshold,
            },
        }

    def save_session(self, filename: str):
        """Save session data to a JSON file."""
        session_data = {
            "session_summary": self.get_session_summary(),
            "step_timestamps": list(self.step_timestamps),
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "magnitude_threshold": self.magnitude_threshold,
                "enable_time_constraints": self.enable_time_constraints,
                "min_step_interval": self.min_step_interval,
                "max_step_rate": self.max_step_rate,
                "step_rate_window": self.step_rate_window,
                "min_step_duration": self.min_step_duration,
                "max_step_duration": self.max_step_duration,
            },
        }

        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2)

        print(f"Session data saved to {filename}")


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
        Process a single sensor reading for simple step counting.

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
        max_confidence = np.max(predictions[0])

        # Calculate movement magnitude
        movement_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Simple step detection
        step_detected = False
        if max_confidence > self.threshold:
            if (
                not self.enable_magnitude_filter
                or movement_magnitude >= self.magnitude_threshold
            ):
                step_detected = True
                self.step_count += 1

        return {
            "step_detected": step_detected,
            "step_count": self.step_count,
            "confidence": float(max_confidence),
            "movement_magnitude": float(movement_magnitude),
            "threshold": self.threshold,
            "magnitude_threshold": self.magnitude_threshold,
        }

    def get_step_count(self) -> int:
        """Get current step count."""
        return self.step_count

    def reset(self):
        """Reset step count."""
        self.step_count = 0


def load_model_info(model_path: str) -> Dict:
    """
    Load model information and metadata.

    Args:
        model_path: Path to the model file

    Returns:
        Dictionary with model information
    """
    try:
        if load_model is None:
            return {"error": "TensorFlow/Keras not available"}

        model = load_model(model_path)

        # Try to load metadata
        metadata_path = model_path.replace(".keras", "_metadata.json").replace(
            ".h5", "_metadata.json"
        )
        metadata = {}
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            pass

        return {
            "model_loaded": True,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "metadata": metadata,
        }
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}


if __name__ == "__main__":
    print("Step Detection Module")
    print("====================")

    # Example usage
    try:
        model_path = "models/step_detection_model.keras"
        detector = StepDetector(model_path)

        print(f"\n🧪 Testing with sample data...")

        # Test with sample sensor readings
        test_readings = [
            (0.1, 0.2, 9.8, 0.01, 0.02, 0.01),  # Small movement
            (2.5, -1.2, 9.8, 0.3, 0.1, -0.2),  # Step start
            (1.5, -0.8, 9.9, 0.2, 0.1, -0.1),  # Step end
        ]

        for i, reading in enumerate(test_readings, 1):
            result = detector.process_reading(*reading)
            print(
                f"Reading {i}: Step detected: {result['step_detected']}, Count: {result['step_count']}"
            )

        print(f"\n📊 Session Summary:")
        summary = detector.get_session_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model file exists and TensorFlow is installed")
