#!/usr/bin/env python3
"""
Test script to simulate Flutter app workflow:
1. Check step count
2. Generate some steps via websocket
3. Check step count again
4. Reset counter
5. Verify reset worked
"""

import asyncio
import json
import time
import requests
import websockets

BASE_URL = "http://0.0.0.0:8000"
WS_URL = "ws://0.0.0.0:8000/ws/realtime"

# Sample step data (start then end)
STEP_DATA = [
    # Step start
    {
        "accel_x": -0.215,
        "accel_y": 7.829,
        "accel_z": 5.161,
        "gyro_x": 0.181,
        "gyro_y": 0.612,
        "gyro_z": 0.498,
    },
    # Step end (after a short delay)
    {
        "accel_x": -1.658,
        "accel_y": 7.789,
        "accel_z": 5.472,
        "gyro_x": 0.293,
        "gyro_y": 1.173,
        "gyro_z": 0.341,
    },
]


def get_step_count():
    """Get current step count from REST API."""
    try:
        response = requests.get(f"{BASE_URL}/step_count", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["step_count"]
    except Exception as e:
        print(f"❌ Error getting step count: {e}")
        return None


def reset_counter():
    """Reset step counter via REST API."""
    try:
        response = requests.post(f"{BASE_URL}/reset_count", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Reset response: {data}")
        return True
    except Exception as e:
        print(f"❌ Error resetting counter: {e}")
        return False


async def send_step_data():
    """Send step data via websocket to simulate flutter app."""
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("🔗 Connected to websocket")

            for i, data in enumerate(STEP_DATA):
                print(f"📤 Sending sensor data {i+1}/2...")
                await websocket.send(json.dumps(data))

                # Wait for response
                response = await websocket.recv()
                result = json.loads(response)

                print(
                    f"📥 Response: step_detected={result.get('step_detected')}, "
                    f"step_count={result.get('step_count')}, "
                    f"confidence={result.get('max_confidence', 0):.3f}"
                )

                # Small delay between start and end
                if i == 0:
                    await asyncio.sleep(0.5)

            print("✅ Websocket communication completed")

    except Exception as e:
        print(f"❌ Websocket error: {e}")


async def main():
    """Main test workflow."""
    print("🧪 Testing Flutter App Workflow")
    print("=" * 50)

    # Step 1: Check initial step count
    print("\n1️⃣ Checking initial step count...")
    initial_count = get_step_count()
    print(f"   Initial step count: {initial_count}")

    # Step 2: Generate steps via websocket
    print("\n2️⃣ Generating steps via websocket...")
    await send_step_data()

    # Step 3: Check step count after websocket
    print("\n3️⃣ Checking step count after websocket...")
    await asyncio.sleep(1)  # Give time for processing
    after_ws_count = get_step_count()
    print(f"   Step count after websocket: {after_ws_count}")

    if after_ws_count and initial_count is not None:
        steps_added = after_ws_count - initial_count
        print(f"   Steps added: {steps_added}")

    # Step 4: Reset counter (like Flutter app does)
    print("\n4️⃣ Resetting counter...")
    reset_success = reset_counter()

    # Step 5: Verify reset worked
    print("\n5️⃣ Verifying reset...")
    if reset_success:
        await asyncio.sleep(0.5)  # Give time for reset
        final_count = get_step_count()
        print(f"   Step count after reset: {final_count}")

        if final_count == 0:
            print("✅ Reset successful - counter is back to 0")
        else:
            print(f"❌ Reset failed - counter is {final_count}, expected 0")

    print("\n🏁 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
