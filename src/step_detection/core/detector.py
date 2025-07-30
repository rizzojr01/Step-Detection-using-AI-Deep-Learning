"""
Real-Time Step Detection System
===============================

A production-ready system for real-time step detection using the trained CNN model.
This can be integrated with mobile apps, wearable devices, or IoT sensors.
"""

import json
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch

from ..utils.config_db import get_config_value


class RealTimeStepCounter:
    """
    Production-ready real-time step detection system
    """

    def __init__(self, model_path=None, model=None, device="cpu"):

        self.device = torch.device(device)

        # Load model
        if model is not None:
            self.model = model
        elif model_path:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            raise ValueError("Either model or model_path must be provided")

        self.model.eval()

        # Configuration from SQLite database
        self.window_size = int(get_config_value("window_size", 10))
        self.start_threshold = float(get_config_value("start_threshold", 0.3))
        self.end_threshold = float(get_config_value("end_threshold", 0.3))

        # AI model confidence thresholds
        self.ai_high_confidence_threshold = float(
            get_config_value("ai_high_confidence_threshold", 0.75)
        )
        self.ai_sensor_disagree_threshold = float(
            get_config_value("ai_sensor_disagree_threshold", 0.80)
        )

        # Data buffers
        self.sensor_buffer = deque(maxlen=self.window_size)
        self.prediction_buffer = deque(maxlen=self.window_size)

        # Step tracking
        self.total_steps = 0
        self.in_step = False
        self.last_step_time = None
        self.step_timestamps = []

        # Enhanced detection tracking
        self.recent_magnitudes = deque(
            maxlen=20
        )  # Track recent motion for adaptive thresholds
        self.baseline_motion = 0.0  # Baseline motion level
        self.peak_buffer = deque(maxlen=10)  # Buffer for peak detection
        self.motion_variance_buffer = deque(
            maxlen=15
        )  # Buffer for motion variance calculation
        self.is_stationary = False  # Track if user appears to be stationary
        self.stationary_count = 0  # Count of consecutive low-variance readings

        # Performance metrics
        buffer_size = int(get_config_value("processing_buffer_size", 100))
        self.processing_times = deque(maxlen=buffer_size)
        self.start_time = time.time()

        print("✅ Real-Time Step Counter initialized")
        print(
            f"   🎯 AI Thresholds: start={self.start_threshold}, end={self.end_threshold}"
        )
        print(
            f"   🧠 AI Confidence: high={self.ai_high_confidence_threshold}, disagree={self.ai_sensor_disagree_threshold}"
        )
        print(f"   🪟 Window size: {self.window_size} readings")
        print(f"   💻 Device: {self.device}")
        print("   " + "=" * 50)

    def add_sensor_reading(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):

        start_time = time.time()

        # Prepare sensor data
        sensor_data = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        self.sensor_buffer.append(sensor_data)

        # Convert to tensor and predict
        sensor_tensor = torch.tensor([sensor_data], dtype=torch.float32).to(
            self.device
        )  # Shape: [1, 6]

        with torch.no_grad():
            outputs = self.model(sensor_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        self.prediction_buffer.append(probabilities)

        # Detect steps
        step_detected = self._detect_step(probabilities)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self.processing_times.append(processing_time)

        # Calculate movement magnitude
        current_reading = (
            self.sensor_buffer[-1]
            if len(self.sensor_buffer) > 0
            else [0, 0, 0, 0, 0, 0]
        )
        movement_magnitude = (
            current_reading[0] ** 2 + current_reading[1] ** 2 + current_reading[2] ** 2
        ) ** 0.5

        # Extract probabilities
        no_step_prob, start_prob, end_prob = probabilities

        # Determine step start and end states
        step_start = not self.in_step and start_prob > self.start_threshold
        step_end = self.in_step and end_prob > self.end_threshold

        # Get predicted class (0: no_step, 1: start, 2: end)
        predicted_class = np.argmax(probabilities)

        # Max confidence
        max_confidence = max(probabilities)

        response = {
            "step_start": bool(step_start),
            "step_end": bool(step_end),
            "step_detected": bool(step_detected),
            "start_probability": float(start_prob),
            "end_probability": float(end_prob),
            "no_step_probability": float(no_step_prob),
            "max_confidence": float(max_confidence),
            "predicted_class": int(predicted_class),
            "step_count": int(self.total_steps),
            "movement_magnitude": float(movement_magnitude),
            "detector_has_current_step": self.in_step,
            "timestamp": str(datetime.now().isoformat()),
            "status": "success",
        }

        return response

    def reset(self):
        """
        Reset the step counter and buffers
        """
        previous_steps = self.total_steps
        self.sensor_buffer.clear()
        self.prediction_buffer.clear()
        self.recent_magnitudes.clear()
        self.peak_buffer.clear()
        self.motion_variance_buffer.clear()
        self.total_steps = 0
        self.in_step = False
        self.last_step_time = None
        self.step_timestamps.clear()
        self.baseline_motion = 0.0
        self.is_stationary = False
        self.stationary_count = 0

        print(f"🔄 STEP COUNTER RESET (was {previous_steps} steps)")
        print("   📊 All buffers cleared, counters reset to 0")

        return {
            "status": "success",
            "message": "Step counter has been reset",
            "total_steps": 0,
            "timestamp": str(datetime.now().isoformat()),
        }

    def _detect_step(self, probabilities):
        """
        Detect step based on probabilities with state tracking

        Since we're using a demo model (not properly trained), we'll use a
        simplified detection logic based on sensor data patterns.
        """
        no_step_prob, start_prob, end_prob = probabilities
        step_detected = False

        # ------------------------------------------------------------------
        # Helper: decide if we should print anything at all
        # ------------------------------------------------------------------
        def _should_log():
            # logging completely off?
            if not get_config_value("enable_step_detection_logs", "True") == "True":
                return False

            # When log_only_on_steps is True, only log actual steps
            if get_config_value("log_only_on_steps", "False") == "True":
                return step_detected

            # default: log everything that the old code logged
            return True

        # ------------------------------------------------------------------
        # Sensor / motion processing (unchanged)
        # ------------------------------------------------------------------
        if len(self.sensor_buffer) > 0:
            current_reading = self.sensor_buffer[-1]
            accel_magnitude = (
                current_reading[0] ** 2
                + current_reading[1] ** 2
                + current_reading[2] ** 2
            ) ** 0.5
            gyro_magnitude = (
                current_reading[3] ** 2
                + current_reading[4] ** 2
                + current_reading[5] ** 2
            ) ** 0.5

            # Track recent motion for adaptive thresholds
            total_magnitude = accel_magnitude + gyro_magnitude
            self.recent_magnitudes.append(total_magnitude)
            self.peak_buffer.append(accel_magnitude)
            self.motion_variance_buffer.append(accel_magnitude)

            # Calculate motion variance to detect walking vs stationary
            motion_variance = 0.0
            if len(self.motion_variance_buffer) >= 10:
                values = list(self.motion_variance_buffer)
                mean_val = sum(values) / len(values)
                motion_variance = sum((x - mean_val) ** 2 for x in values) / len(values)

            # Update stillness detection
            stillness_threshold = float(get_config_value("stillness_threshold", 1.5))
            if motion_variance < stillness_threshold:
                self.stationary_count += 1
                if self.stationary_count >= 10:
                    self.is_stationary = True
            else:
                self.stationary_count = 0
                self.is_stationary = False

            # Update baseline motion (rolling average)
            if len(self.recent_magnitudes) > 5:
                self.baseline_motion = sum(list(self.recent_magnitudes)[-10:]) / min(
                    10, len(self.recent_magnitudes)
                )

            # Get thresholds (potentially adaptive)
            motion_threshold = float(get_config_value("motion_threshold", 8.0))
            gyro_threshold = float(get_config_value("gyro_threshold", 0.5))

            if get_config_value("enable_adaptive_thresholds", "True") == "True":
                sensitivity = float(get_config_value("motion_sensitivity", 0.7))
                motion_threshold = max(motion_threshold * sensitivity, 10.5)
                gyro_threshold = max(gyro_threshold * sensitivity, 0.3)

            time_since_last_step = time.time() - getattr(self, "_last_step_time", 0)
            min_step_interval = float(get_config_value("min_step_interval", 0.4))

            # ------------------------------------------------------------------
            # Detection methods (unchanged)
            # ------------------------------------------------------------------
            accel_exceeds = accel_magnitude > motion_threshold
            gyro_exceeds = gyro_magnitude > gyro_threshold
            time_ok = time_since_last_step > min_step_interval

            motion_variance_ok = True
            if get_config_value("enable_motion_variance_filter", "True") == "True":
                min_variance = float(get_config_value("min_motion_variance", 1.0))
                strong_motion = (
                    accel_magnitude > motion_threshold * 1.3
                    or gyro_magnitude > gyro_threshold * 1.5
                )
                motion_variance_ok = (motion_variance >= min_variance) or strong_motion

            not_stationary = True
            if get_config_value("enable_stillness_detection", "True") == "True":
                gravity_threshold = 0.5
                is_gravity_reading = (
                    abs(accel_magnitude - 9.8) < gravity_threshold
                    and gyro_magnitude < 0.2
                )
                clearly_stationary = (
                    self.is_stationary or is_gravity_reading
                ) and motion_variance < 0.5
                not_stationary = not clearly_stationary

            basic_detection = (
                (accel_exceeds and gyro_exceeds)
                and time_ok
                and not_stationary
                and motion_variance_ok
            )

            strong_signal_detection = False
            if get_config_value("enable_hybrid_detection", "True") == "True":
                very_strong_accel = accel_magnitude > motion_threshold * 1.3
                significant_gyro = gyro_magnitude > gyro_threshold * 0.5
                strong_signal_detection = (
                    very_strong_accel
                    and significant_gyro
                    and time_ok
                    and not_stationary
                )

            peak_detection = False
            if (
                get_config_value("enable_peak_detection", "True") == "True"
                and len(self.peak_buffer) >= 3
            ):
                current_val = self.peak_buffer[-1]
                prev_val = self.peak_buffer[-2] if len(self.peak_buffer) > 1 else 0
                prev_prev_val = self.peak_buffer[-3] if len(self.peak_buffer) > 2 else 0
                is_peak = (
                    current_val > prev_val
                    and prev_val > prev_prev_val
                    and current_val > motion_threshold * 0.6
                )
                peak_detection = is_peak and time_ok and not_stationary

            motion_change_detection = False
            if len(self.recent_magnitudes) >= 3:
                recent_avg = sum(list(self.recent_magnitudes)[-3:]) / 3
                older_avg = (
                    sum(list(self.recent_magnitudes)[-6:-3]) / 3
                    if len(self.recent_magnitudes) >= 6
                    else recent_avg
                )
                motion_spike = recent_avg > older_avg * 1.3 and recent_avg > 5.0
                motion_change_detection = motion_spike and time_ok and not_stationary

            variance_detection = False
            if (
                motion_variance_ok
                and (accel_exceeds and gyro_exceeds)
                and time_ok
                and not_stationary
            ):
                variance_detection = True

            gyro_only_detection = False
            if (
                gyro_magnitude > gyro_threshold * 2.0
                and time_ok
                and not_stationary
                and motion_variance_ok
            ):
                gyro_only_detection = True

            step_detected_by_motion = (
                basic_detection
                or strong_signal_detection
                or peak_detection
                or motion_change_detection
                or variance_detection
                or gyro_only_detection
            )

            # ------------------------------------------------------------------
            # AI vs sensor arbitration (unchanged)
            # ------------------------------------------------------------------
            ai_predicts_step = (
                start_prob > self.start_threshold or end_prob > self.end_threshold
            )
            ai_step_confidence = max(start_prob, end_prob) if ai_predicts_step else 0.0
            predicted_class = ["no_step", "start", "end"][
                np.argmax([no_step_prob, start_prob, end_prob])
            ]
            max_confidence = max(no_step_prob, start_prob, end_prob)

            sensors_agree_with_ai = step_detected_by_motion
            detection_source = "None"

            if ai_predicts_step:
                if sensors_agree_with_ai:
                    if ai_step_confidence >= self.start_threshold:
                        step_detected = True
                        detection_source = "AI Model (Sensors Agree)"
                else:
                    if ai_step_confidence >= self.ai_sensor_disagree_threshold:
                        step_detected = True
                        detection_source = "AI Model (High Confidence)"
                    else:
                        step_detected = False
                        detection_source = "AI Rejected (Low Confidence)"

            if not step_detected and not ai_predicts_step and step_detected_by_motion:
                very_strong_motion = (
                    accel_magnitude > motion_threshold * 1.5
                    and gyro_magnitude > gyro_threshold * 1.5
                )
                if very_strong_motion:
                    step_detected = True
                    detection_source = "Sensor Fallback (Strong Motion)"

            # ------------------------------------------------------------------
            # Final step bookkeeping (unchanged)
            # ------------------------------------------------------------------
            if step_detected:
                self.total_steps += 1
                self.last_step_time = time.time()
                self._last_step_time = time.time()
                self.step_timestamps.append(datetime.now().isoformat())
                self.in_step = False

                if _should_log():
                    trigger_methods = []
                    if "AI Model" in detection_source:
                        if "Sensors Agree" in detection_source:
                            trigger_methods.append(
                                f"AI-Model({predicted_class}={ai_step_confidence:.1%})"
                            )
                            trigger_methods.append("Sensors-Agree")
                        elif "High Confidence" in detection_source:
                            trigger_methods.append(
                                f"AI-High-Confidence({predicted_class}={ai_step_confidence:.1%})"
                            )
                            trigger_methods.append("Sensors-Disagree-Override")
                    elif detection_source == "Sensor Fallback (Strong Motion)":
                        trigger_methods.append("AI-No-Step")
                        if basic_detection:
                            if accel_exceeds:
                                trigger_methods.append(
                                    f"Strong-Accel({accel_magnitude:.1f}>{motion_threshold*1.5:.1f})"
                                )
                            if gyro_exceeds:
                                trigger_methods.append(
                                    f"Strong-Gyro({gyro_magnitude:.1f}>{gyro_threshold*1.5:.1f})"
                                )
                        else:
                            if strong_signal_detection:
                                trigger_methods.append(
                                    f"Strong-Signal({max(accel_magnitude, gyro_magnitude):.1f})"
                                )
                            if peak_detection:
                                trigger_methods.append(
                                    f"Peak-Detection({accel_magnitude:.1f})"
                                )
                            if motion_change_detection:
                                trigger_methods.append(
                                    f"Motion-Spike({sum(list(self.recent_magnitudes)[-3:])/3:.1f})"
                                )
                            if variance_detection:
                                trigger_methods.append(
                                    f"Variance-Pattern({motion_variance:.2f})"
                                )
                            if gyro_only_detection:
                                trigger_methods.append(
                                    f"Gyro-Only({gyro_magnitude:.1f})"
                                )

                    trigger_str = (
                        " + ".join(trigger_methods) if trigger_methods else "Unknown"
                    )

                    print(f"🦶 STEP #{self.total_steps} DETECTED!")
                    print(f"   🔥 Trigger: {trigger_str}")
                    print(f"   🎯 Source: {detection_source}")
                    print(
                        f"   📊 Motion: Accel={accel_magnitude:.1f} ({'✅' if accel_exceeds else '❌'}>{motion_threshold:.1f}), Gyro={gyro_magnitude:.1f} ({'✅' if gyro_exceeds else '❌'}>{gyro_threshold:.1f})"
                    )
                    print(
                        f"   📈 Variance: {motion_variance:.2f} ({'✅' if motion_variance_ok else '❌'} walking pattern)"
                    )
                    print(
                        f"   🎯 Status: {'🚶‍♂️ Moving' if not self.is_stationary else '🧍‍♂️ Stationary'}"
                    )
                    print(
                        f"   🧠 Model: {predicted_class} ({max_confidence:.1%} confidence)"
                    )
                    print(
                        f"   📊 Probs: no_step={no_step_prob:.1%}, start={start_prob:.1%}, end={end_prob:.1%}"
                    )
                    print(
                        f"   ⚙️  Thresholds: start={self.start_threshold}, end={self.end_threshold}, motion={motion_threshold:.1f}, gyro={gyro_threshold:.1f}"
                    )
                    print(
                        f"   ⏱️  Time since last: {time_since_last_step:.1f}s (min: {min_step_interval}s)"
                    )
                    print("   " + "=" * 50)

            # ------------------------------------------------------------------
            # Occasional “no-step” explanation (only if we are not suppressing)
            # ------------------------------------------------------------------
            else:
                if _should_log() and len(self.sensor_buffer) % 50 == 0:
                    reasons = []
                    if (
                        ai_predicts_step
                        and detection_source == "AI Rejected (Low Confidence)"
                    ):
                        reasons.append(
                            f"AI predicted step ({predicted_class}={ai_step_confidence:.1%}) but confidence too low (need {self.ai_sensor_disagree_threshold:.1%} when sensors disagree)"
                        )
                    elif not ai_predicts_step and not step_detected_by_motion:
                        reasons.append(
                            f"Neither AI nor sensors detected step: AI confidence low ({max(start_prob, end_prob):.1%} < {self.start_threshold:.1%})"
                        )
                    elif not accel_exceeds and not gyro_exceeds:
                        reasons.append(
                            f"Motion below threshold: Accel={accel_magnitude:.1f}<={motion_threshold}, Gyro={gyro_magnitude:.1f}<={gyro_threshold}"
                        )
                    elif not time_ok:
                        reasons.append(
                            f"Too soon: {time_since_last_step:.1f}s < {min_step_interval}s"
                        )
                    elif not motion_variance_ok:
                        reasons.append(
                            f"Low variance: {motion_variance:.2f} < {get_config_value('min_motion_variance', 2.0)} (not walking pattern)"
                        )
                    elif self.is_stationary:
                        reasons.append(
                            f"Stationary: variance={motion_variance:.2f}, count={self.stationary_count}"
                        )

                    print(f"⏸️  NO STEP: {', '.join(reasons)}")
                    print(
                        f"   📊 Accel={accel_magnitude:.1f}, Gyro={gyro_magnitude:.1f}, Variance={motion_variance:.2f}"
                    )
                    print(
                        f"   🧠 AI: {predicted_class} ({max(start_prob, end_prob):.1%} confidence), Sensors: {'✅' if step_detected_by_motion else '❌'}"
                    )
                    print(
                        f"   🎯 Status: {'🧍‍♂️ Stationary' if self.is_stationary else '🚶‍♂️ Moving'}"
                    )
                    print("   " + "-" * 30)

        return step_detected


if __name__ == "__main__":
    print("🚀 Real-Time Step Detection System")
    print("This module provides production-ready step counting capabilities")
    print("Import this module to use in your applications")
