#!/usr/bin/env python3
"""
Test websocket step detection after reset to ensure proper synchronization.
This test specifically addresses the issue where HTTP reset might not affect websocket state.
"""

import asyncio
import json
import time
import requests
import websockets

BASE_URL = "http://0.0.0.0:8000"
WS_URL = "ws://0.0.0.0:8000/ws/realtime"

# Sample step data (start then end for a complete step)
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


def get_step_count_http():
    """Get current step count via HTTP API."""
    try:
        response = requests.get(f"{BASE_URL}/step_count", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["step_count"]
    except Exception as e:
        print(f"❌ Error getting step count via HTTP: {e}")
        return None


def reset_counter_http():
    """Reset step counter via HTTP API."""
    try:
        response = requests.post(f"{BASE_URL}/reset_count", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ HTTP Reset response: {data}")
        return True
    except Exception as e:
        print(f"❌ Error resetting counter via HTTP: {e}")
        return False


async def get_step_count_websocket():
    """Get current step count via websocket by sending a minimal sensor reading."""
    try:
        async with websockets.connect(WS_URL) as websocket:
            # Send minimal sensor data to get current state
            minimal_data = {
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 9.8,  # Just gravity
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
            }

            await websocket.send(json.dumps(minimal_data))
            response = await websocket.recv()
            result = json.loads(response)

            return result.get("step_count", None)

    except Exception as e:
        print(f"❌ Error getting step count via websocket: {e}")
        return None


async def send_step_via_websocket():
    """Send step data via websocket and return the final step count."""
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("🔗 Connected to websocket for step generation")

            final_count = 0
            for i, data in enumerate(STEP_DATA):
                print(f"📤 Sending sensor data {i+1}/2...")
                await websocket.send(json.dumps(data))

                # Wait for response
                response = await websocket.recv()
                result = json.loads(response)

                step_detected = result.get("step_detected", False)
                step_count = result.get("step_count", 0)
                confidence = result.get("max_confidence", 0)

                print(
                    f"📥 Response: step_detected={step_detected}, "
                    f"step_count={step_count}, "
                    f"confidence={confidence:.3f}"
                )

                final_count = step_count

                # Small delay between start and end
                if i == 0:
                    await asyncio.sleep(0.5)

            print(f"✅ Websocket step generation completed, final count: {final_count}")
            return final_count

    except Exception as e:
        print(f"❌ Websocket step generation error: {e}")
        return None


async def main():
    """Main test to verify websocket reset functionality."""
    print("🧪 Testing Websocket Reset Synchronization")
    print("=" * 60)

    # Step 1: Check initial state via both HTTP and WebSocket
    print("\n1️⃣ Checking initial state...")
    http_count = get_step_count_http()
    ws_count = await get_step_count_websocket()
    print(f"   HTTP step count: {http_count}")
    print(f"   WebSocket step count: {ws_count}")

    if http_count != ws_count:
        print("⚠️  HTTP and WebSocket counts don't match initially!")

    # Step 2: Generate some steps via websocket
    print("\n2️⃣ Generating steps via websocket...")
    final_ws_count = await send_step_via_websocket()

    # Step 3: Verify counts match after websocket activity
    print("\n3️⃣ Verifying counts after websocket activity...")
    await asyncio.sleep(1)  # Give time for any async processing

    http_count_after_ws = get_step_count_http()
    ws_count_after_ws = await get_step_count_websocket()

    print(f"   HTTP step count after WS: {http_count_after_ws}")
    print(f"   WebSocket step count after WS: {ws_count_after_ws}")
    print(f"   Expected count: {final_ws_count}")

    if http_count_after_ws != ws_count_after_ws:
        print("❌ HTTP and WebSocket counts don't match after websocket activity!")
    elif http_count_after_ws == final_ws_count:
        print("✅ All counts match correctly")
    else:
        print(
            f"⚠️  Counts don't match expected value: got {http_count_after_ws}, expected {final_ws_count}"
        )

    # Step 4: Reset via HTTP
    print("\n4️⃣ Resetting via HTTP...")
    reset_success = reset_counter_http()

    if not reset_success:
        print("❌ HTTP reset failed, aborting test")
        return

    # Step 5: CRITICAL TEST - Check websocket count after HTTP reset
    print("\n5️⃣ CRITICAL TEST: Checking websocket count after HTTP reset...")
    await asyncio.sleep(1)  # Give time for reset to propagate

    http_count_after_reset = get_step_count_http()
    ws_count_after_reset = await get_step_count_websocket()

    print(f"   HTTP step count after reset: {http_count_after_reset}")
    print(f"   WebSocket step count after reset: {ws_count_after_reset}")

    # This is the critical test - both should be 0
    if http_count_after_reset == 0 and ws_count_after_reset == 0:
        print("✅ SUCCESS: Both HTTP and WebSocket show 0 after reset!")
    elif http_count_after_reset == 0 and ws_count_after_reset != 0:
        print(
            f"❌ PROBLEM: HTTP reset to 0 but WebSocket still shows {ws_count_after_reset}"
        )
        print("   This indicates websocket is not properly synchronized with reset!")
    elif http_count_after_reset != 0:
        print(f"❌ PROBLEM: HTTP reset failed, still shows {http_count_after_reset}")
    else:
        print("❌ Unexpected state after reset")

    # Step 6: Generate new step via websocket after reset
    print("\n6️⃣ Testing websocket after reset...")
    new_ws_count = await send_step_via_websocket()

    print(f"\n7️⃣ Final verification...")
    final_http_count = get_step_count_http()
    final_ws_count = await get_step_count_websocket()

    print(f"   Final HTTP count: {final_http_count}")
    print(f"   Final WebSocket count: {final_ws_count}")
    print(f"   Expected count: {new_ws_count}")

    if final_http_count == final_ws_count == new_ws_count:
        print("✅ SUCCESS: Reset and new step detection working correctly!")
    else:
        print("❌ PROBLEM: Counts still don't match after reset and new steps")

    print("\n🏁 Websocket reset test completed!")

    # Summary
    print("\n📊 SUMMARY:")
    print(
        f"   Reset cleared websocket state: {'✅' if ws_count_after_reset == 0 else '❌'}"
    )
    print(
        f"   HTTP/WS synchronization: {'✅' if final_http_count == final_ws_count else '❌'}"
    )
    print(
        f"   New steps after reset: {'✅' if new_ws_count and new_ws_count > 0 else '❌'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
