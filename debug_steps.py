#!/usr/bin/env python3
"""
Debug Step Detection
====================

Debug the step detection by sending a sequence of sensor readings that should trigger step detection.
"""

import json
import time

import requests

API_BASE_URL = "http://localhost:8000"


def test_step_sequence():
    """Test a realistic step sequence"""
    print("🔍 Debugging Step Detection Sequence")
    print("=" * 50)

    # First, reset the counter
    print("1. Resetting counter...")
    reset_response = requests.post(f"{API_BASE_URL}/reset")
    print(f"   Reset: {reset_response.json()}")

    print("\n2. Sending step sequence...")

    # Sequence of sensor readings that simulate a step
    step_sequence = [
        # Normal standing/walking baseline
        {
            "accel_x": 0.1,
            "accel_y": 0.2,
            "accel_z": 9.8,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
        },
        # Step START - higher motion
        {
            "accel_x": 3.0,
            "accel_y": 1.5,
            "accel_z": 12.0,
            "gyro_x": 0.8,
            "gyro_y": 0.6,
            "gyro_z": 0.4,
        },
        {
            "accel_x": 4.0,
            "accel_y": 2.0,
            "accel_z": 13.0,
            "gyro_x": 1.0,
            "gyro_y": 0.8,
            "gyro_z": 0.6,
        },
        # Peak motion
        {
            "accel_x": 6.0,
            "accel_y": 3.0,
            "accel_z": 15.0,
            "gyro_x": 1.5,
            "gyro_y": 1.2,
            "gyro_z": 0.8,
        },
        # Step END - motion settling
        {
            "accel_x": 2.0,
            "accel_y": 1.0,
            "accel_z": 11.0,
            "gyro_x": 0.5,
            "gyro_y": 0.3,
            "gyro_z": 0.2,
        },
        {
            "accel_x": 1.0,
            "accel_y": 0.5,
            "accel_z": 10.0,
            "gyro_x": 0.2,
            "gyro_y": 0.1,
            "gyro_z": 0.1,
        },
        # Back to baseline
        {
            "accel_x": 0.1,
            "accel_y": 0.2,
            "accel_z": 9.8,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
        },
    ]

    total_steps_detected = 0

    for i, reading in enumerate(step_sequence):
        response = requests.post(
            f"{API_BASE_URL}/detect_step",
            json=reading,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()["result"]
            probs = result["probabilities"]

            print(
                f"Reading {i+1:2d}: "
                f"no_step={probs[0]:.3f}, start={probs[1]:.3f}, end={probs[2]:.3f} | "
                f"in_step={result['in_step']}, detected={result['step_detected']}, "
                f"total={result['total_steps']}"
            )

            if result["step_detected"]:
                total_steps_detected += 1
                print(f"           🎉 STEP {total_steps_detected} DETECTED!")

        time.sleep(0.1)  # Small delay

    print(f"\n📊 Final Result: {total_steps_detected} steps detected")

    # Get final statistics
    stats_response = requests.get(f"{API_BASE_URL}/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()["statistics"]
        print(f"📈 Total steps in session: {stats['total_steps']}")
        print(f"⏱️  Session duration: {stats['elapsed_time_seconds']:.1f}s")


def test_with_real_data():
    """Test with patterns that should definitely be steps"""
    print("\n🏃‍♂️ Testing with Exaggerated Step Patterns")
    print("=" * 50)

    # Reset first
    requests.post(f"{API_BASE_URL}/reset")

    # Very obvious step pattern
    obvious_steps = [
        # Baseline
        {
            "accel_x": 0,
            "accel_y": 0,
            "accel_z": 9.8,
            "gyro_x": 0,
            "gyro_y": 0,
            "gyro_z": 0,
        },
        # VERY obvious start
        {
            "accel_x": 10,
            "accel_y": 5,
            "accel_z": 20,
            "gyro_x": 2,
            "gyro_y": 2,
            "gyro_z": 1,
        },
        # Peak
        {
            "accel_x": 15,
            "accel_y": 8,
            "accel_z": 25,
            "gyro_x": 3,
            "gyro_y": 3,
            "gyro_z": 2,
        },
        # VERY obvious end
        {
            "accel_x": 1,
            "accel_y": 1,
            "accel_z": 10,
            "gyro_x": 0.1,
            "gyro_y": 0.1,
            "gyro_z": 0,
        },
        # Back to baseline
        {
            "accel_x": 0,
            "accel_y": 0,
            "accel_z": 9.8,
            "gyro_x": 0,
            "gyro_y": 0,
            "gyro_z": 0,
        },
    ]

    for i, reading in enumerate(obvious_steps):
        response = requests.post(f"{API_BASE_URL}/detect_step", json=reading)
        result = response.json()["result"]
        probs = result["probabilities"]

        print(
            f"Reading {i+1}: "
            f"start={probs[1]:.3f}, end={probs[2]:.3f} | "
            f"in_step={result['in_step']}, detected={result['step_detected']}"
        )


if __name__ == "__main__":
    # Test if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print(
                "❌ API not running. Start with: uvicorn step_detection_api:app --reload"
            )
            exit(1)
    except:
        print(
            "❌ Cannot connect to API. Start with: uvicorn step_detection_api:app --reload"
        )
        exit(1)

    test_step_sequence()
    test_with_real_data()
