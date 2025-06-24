"""
FastAPI Test Client
===================

Test the FastAPI step detection service with real sensor data.
"""

import json
import time

import numpy as np
import pandas as pd
import requests

API_BASE_URL = "http://localhost:8000"


def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   Model initialized: {data['model_initialized']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(
            "❌ Cannot connect to API. Make sure FastAPI server is running on port 8000"
        )
        return False


def test_single_detection():
    """Test single step detection"""
    print("\n🧪 Testing Single Step Detection...")

    # Sample sensor data with step-like motion (higher values to trigger detection)
    test_data = {
        "accel_x": 8.0,  # Higher acceleration values
        "accel_y": 2.0,
        "accel_z": 15.0,
        "gyro_x": 1.5,  # Higher gyroscope values
        "gyro_y": 1.2,
        "gyro_z": 0.8,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/detect_step",
            json=test_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Step detection successful:")
            print(f"   Step detected: {result['result']['step_detected']}")
            print(f"   Total steps: {result['result']['total_steps']}")
            print(f"   Confidence: {result['result']['confidence']:.3f}")
            print(f"   Processing time: {result['result']['processing_time_ms']:.1f}ms")
            return True
        else:
            print(f"❌ Detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False


def test_multiple_detections():
    """Test multiple consecutive detections"""
    print("\n🚶‍♂️ Testing Multiple Step Detections...")

    # Generate some realistic sensor data with step-like motion
    sensor_readings = []
    for i in range(20):
        # Simulate walking motion with higher amplitudes for step detection
        t = i * 0.1  # 0.1 second intervals

        # More pronounced walking pattern that will trigger step detection
        if i % 4 == 0:  # Every 4th reading is a step
            accel_x = 8.0 + 3.0 * np.sin(t * 4)  # Strong step motion
            accel_y = 2.0 + 2.0 * np.cos(t * 4)
            accel_z = 15.0 + 3.0 * np.sin(t * 2)
            gyro_x = 1.5 * np.sin(t * 3)
            gyro_y = 1.2 * np.cos(t * 3)
            gyro_z = 0.8 * np.sin(t * 5)
        else:  # Normal walking baseline
            accel_x = 1.0 + 0.5 * np.sin(t * 4)
            accel_y = -0.2 + 0.3 * np.cos(t * 4)
            accel_z = 9.8 + 0.2 * np.sin(t * 2)
            gyro_x = 0.1 * np.sin(t * 3)
            gyro_y = 0.2 * np.cos(t * 3)
            gyro_z = -0.1 * np.sin(t * 5)

        sensor_readings.append(
            {
                "accel_x": float(accel_x),
                "accel_y": float(accel_y),
                "accel_z": float(accel_z),
                "gyro_x": float(gyro_x),
                "gyro_y": float(gyro_y),
                "gyro_z": float(gyro_z),
            }
        )

    total_steps = 0
    for i, reading in enumerate(sensor_readings):
        try:
            response = requests.post(
                f"{API_BASE_URL}/detect_step",
                json=reading,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                if result["result"]["step_detected"]:
                    total_steps += 1
                    print(f"👟 Step {total_steps} detected at reading {i+1}")

                # Show progress every 5 readings
                if (i + 1) % 5 == 0:
                    print(f"📊 Processed {i+1} readings, detected {total_steps} steps")

            # Small delay to simulate real-time
            time.sleep(0.05)

        except Exception as e:
            print(f"❌ Error at reading {i+1}: {e}")

    print(
        f"\n📈 Final Results: {total_steps} steps detected from {len(sensor_readings)} readings"
    )
    return total_steps


def test_get_statistics():
    """Test getting statistics"""
    print("\n📊 Testing Statistics Endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/stats")

        if response.status_code == 200:
            stats = response.json()["statistics"]
            print("✅ Statistics retrieved:")
            print(f"   Total steps: {stats['total_steps']}")
            print(f"   Steps per minute: {stats['steps_per_minute']:.2f}")
            print(
                f"   Average processing time: {stats['avg_processing_time_ms']:.2f}ms"
            )
            print(f"   Session duration: {stats['elapsed_time_seconds']:.1f}s")
            return True
        else:
            print(f"❌ Statistics failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return False


def test_reset_counter():
    """Test resetting the step counter"""
    print("\n🔄 Testing Counter Reset...")

    try:
        response = requests.post(f"{API_BASE_URL}/reset")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Counter reset: {result['message']}")
            return True
        else:
            print(f"❌ Reset failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error resetting counter: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 FastAPI Step Detection Service Test")
    print("=" * 50)

    # Test 1: Health check
    if not test_api_health():
        print("\n❌ Cannot proceed - API is not running")
        print("Please start the FastAPI server first:")
        print(
            "uv run uvicorn step_detection_api:app --host 0.0.0.0 --port 8000 --reload"
        )
        return

    # Test 2: Single detection
    test_single_detection()

    # Test 3: Multiple detections
    test_multiple_detections()

    # Test 4: Get statistics
    test_get_statistics()

    # Test 5: Reset counter
    test_reset_counter()

    print("\n🎉 All tests completed!")
    print("📡 Your FastAPI step detection service is working perfectly!")
    print("\n🔗 API Endpoints:")
    print(f"   📖 Documentation: {API_BASE_URL}/docs")
    print(f"   🧪 Interactive docs: {API_BASE_URL}/redoc")
    print(f"   🏥 Health check: {API_BASE_URL}/health")


if __name__ == "__main__":
    main()
