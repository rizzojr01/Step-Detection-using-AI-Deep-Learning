#!/usr/bin/env python3
"""
IP Address WebSocket Test
========================
Test WebSocket connection using the specific IP address your Flutter app is trying to connect to.
"""

import asyncio
import json

import websockets


async def test_websocket_ip():
    # Test both localhost and Docker container URLs
    uris = [
        "ws://localhost:8000/ws/realtime",  # Direct access
        "ws://192.168.18.179:8000/ws/realtime",  # Your Flutter app URL
        "ws://host.docker.internal:8000/ws/realtime",  # Docker host access
    ]

    for uri in uris:
        try:
            print(f"🔌 Connecting to WebSocket at {uri}...")

            # Use a 10-second timeout to match typical mobile app timeouts
            websocket = await asyncio.wait_for(websockets.connect(uri), timeout=10)

            print("✅ Connected successfully!")

            # Test data - realistic sensor readings
            test_data = {
                "accel_x": 1.2,
                "accel_y": -0.5,
                "accel_z": 9.8,
                "gyro_x": 0.1,
                "gyro_y": 0.2,
                "gyro_z": -0.1,
            }

            print("📤 Sending sensor data...")
            await websocket.send(json.dumps(test_data))

            print("📥 Waiting for response...")
            response = await websocket.recv()
            print("✅ Response received:")

            parsed_response = json.loads(response)
            print(f"   Raw response: {response}")
            print(f"   Parsed: {parsed_response}")
            print(f"   Step detected: {parsed_response.get('step_detected', 'N/A')}")
            print(f"   Total steps: {parsed_response.get('total_steps', 'N/A')}")
            print(f"   Confidence: {parsed_response.get('confidence', 'N/A')}")

            await websocket.close()
            print(f"✅ {uri} - SUCCESS!\n")
            break  # Exit on first successful connection

        except Exception as e:
            print(f"❌ {uri} - FAILED: {e}\n")
            continue

    print("❌ All connection attempts failed!")


if __name__ == "__main__":
    asyncio.run(test_websocket_ip())
