#!/usr/bin/env python3
"""
Simple WebSocket Test
====================
Test WebSocket connection with basic functionality.
"""

import asyncio
import json

import websockets


async def test_websocket():
    uri = "ws://localhost:8000/ws/realtime"

    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")

            # Send a test sensor reading
            test_data = {
                "accel_x": 8.0,
                "accel_y": 2.0,
                "accel_z": 15.0,
                "gyro_x": 1.5,
                "gyro_y": 1.2,
                "gyro_z": 0.8,
            }

            print("📤 Sending sensor data...")
            await websocket.send(json.dumps(test_data))

            print("📥 Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            result = json.loads(response)

            print("✅ Response received:")
            print(f"   Raw response: {response}")
            print(f"   Parsed: {result}")

            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   Step detected: {result.get('step_detected', 'N/A')}")
                print(f"   Total steps: {result.get('total_steps', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")

    except Exception as e:
        print(f"❌ WebSocket error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
