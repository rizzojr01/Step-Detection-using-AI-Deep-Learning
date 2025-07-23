#!/usr/bin/env python3
import asyncio
import json
import time

import websockets


async def test_complete_step():
    uri = "ws://0.0.0.0:8000/ws/realtime"

    # Step start data (high start probability)
    step_start_data = {
        "accel_x": -0.215,
        "accel_y": 7.829,
        "accel_z": 5.161,
        "gyro_x": 0.181,
        "gyro_y": 0.612,
        "gyro_z": 0.498,
    }

    # Step end data (high end probability)
    step_end_data = {
        "accel_x": -1.658,
        "accel_y": 7.789,
        "accel_z": 5.472,
        "gyro_x": 0.293,
        "gyro_y": 1.173,
        "gyro_z": 0.341,
    }

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to websocket")

            # Send step start
            print("Sending step start data...")
            await websocket.send(json.dumps(step_start_data))
            response1 = await websocket.recv()
            print("Step start response:", response1)

            # Wait a bit to satisfy time constraints
            await asyncio.sleep(0.5)

            # Send step end
            print("Sending step end data...")
            await websocket.send(json.dumps(step_end_data))
            response2 = await websocket.recv()
            print("Step end response:", response2)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_complete_step())
