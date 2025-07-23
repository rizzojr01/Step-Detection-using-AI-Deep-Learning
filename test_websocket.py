#!/usr/bin/env python3
import asyncio
import json

import websockets


async def test_websocket():
    uri = "ws://0.0.0.0:8000/ws/realtime"

    # Test data with high prediction values like your logs
    test_data = {
        "accel_x": -0.215,
        "accel_y": 7.829,
        "accel_z": 5.161,
        "gyro_x": 0.181,
        "gyro_y": 0.612,
        "gyro_z": 0.498,
    }

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to websocket")
            await websocket.send(json.dumps(test_data))
            response = await websocket.recv()
            print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
