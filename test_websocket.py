#!/usr/bin/env python3
"""
Simple WebSocket test client for the step detection API.
"""

import asyncio
import json

try:
    import websockets
except ImportError:
    print("❌ websockets not available. Install with: uv add websockets")
    exit(1)


async def test_websocket():
    """Test the WebSocket endpoint."""
    uri = "ws://localhost:8000/ws/realtime"

    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")

            # Send multiple test readings to simulate a walking pattern
            test_readings = [
                {
                    "accel_x": 0.1,
                    "accel_y": 0.2,
                    "accel_z": 9.8,
                    "gyro_x": 0.01,
                    "gyro_y": 0.02,
                    "gyro_z": 0.01,
                },  # Normal
                {
                    "accel_x": 2.5,
                    "accel_y": 1.8,
                    "accel_z": 8.2,
                    "gyro_x": 0.3,
                    "gyro_y": 0.8,
                    "gyro_z": 0.4,
                },  # Step start
                {
                    "accel_x": 0.3,
                    "accel_y": -2.1,
                    "accel_z": 11.1,
                    "gyro_x": 0.5,
                    "gyro_y": -0.2,
                    "gyro_z": -0.3,
                },  # Step impact
                {
                    "accel_x": 1.8,
                    "accel_y": 0.4,
                    "accel_z": 9.2,
                    "gyro_x": -0.1,
                    "gyro_y": 0.3,
                    "gyro_z": 0.2,
                },  # Step transition
                {
                    "accel_x": 0.7,
                    "accel_y": -1.2,
                    "accel_z": 10.3,
                    "gyro_x": 0.3,
                    "gyro_y": -0.4,
                    "gyro_z": 0.1,
                },  # Step end
                {
                    "accel_x": 0.1,
                    "accel_y": 0.2,
                    "accel_z": 9.8,
                    "gyro_x": 0.01,
                    "gyro_y": 0.02,
                    "gyro_z": 0.01,
                },  # Normal again
            ]

            for i, test_data in enumerate(test_readings):
                print(f"\n📤 Reading {i+1}: {test_data}")
                await websocket.send(json.dumps(test_data))

                # Receive response
                response = await websocket.recv()
                result = json.loads(response)

                print(f"📥 Response {i+1}: {result}")

                if result.get("status") == "success":
                    start_prob = result.get("start_probability", 0)
                    end_prob = result.get("end_probability", 0)
                    step_count = result.get("step_count", 0)

                    print(
                        f"   Step detection: start={result.get('step_start')}, end={result.get('step_end')}"
                    )
                    print(
                        f"   Probabilities: start={start_prob:.4f}, end={end_prob:.4f}"
                    )
                    print(f"   Step count: {step_count}")

                    # Highlight significant detections
                    if start_prob > 0.1 or end_prob > 0.1:
                        print(f"   🎯 SIGNIFICANT DETECTION!")
                    if result.get("step_start") or result.get("step_end"):
                        print(f"   🚶 STEP EVENT DETECTED!")
                else:
                    print(f"❌ Error: {result.get('error')}")

                # Small delay between readings
                await asyncio.sleep(0.1)

            print(f"\n✅ WebSocket test completed!")

    except ConnectionRefusedError:
        print("❌ Connection refused. Is the API server running?")
        print("   Start with: python main.py -> option 4")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
