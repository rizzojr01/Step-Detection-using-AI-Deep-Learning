#!/usr/bin/env python3
"""
WebSocket Real-Time Step Detection Client
==========================================

Test and demonstrate the real-time WebSocket step detection functionality.
This simulates a mobile app or IoT device sending continuous sensor data.
"""

import asyncio
import json
import time

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

WS_URL = "ws://localhost:8000/ws/realtime"


class RealTimeStepClient:
    """Real-time step detection WebSocket client"""

    def __init__(self):
        self.websocket = None
        self.total_steps = 0
        self.is_running = False

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(WS_URL)
            print("✅ Connected to WebSocket server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            print("📡 Disconnected from WebSocket server")

    async def send_sensor_data(self, sensor_data):
        """Send sensor data and receive result"""
        try:
            await self.websocket.send(json.dumps(sensor_data))
            response = await self.websocket.recv()
            return json.loads(response)
        except ConnectionClosed:
            print("❌ WebSocket connection closed")
            return None
        except Exception as e:
            print(f"❌ Error sending data: {e}")
            return None

    async def simulate_walking(self, duration=10, steps_per_second=2.0):
        """Simulate walking motion for testing"""
        print(f"\n🚶‍♂️ Simulating {duration}s of walking motion...")
        print("=" * 50)

        start_time = time.time()
        reading_count = 0
        step_detections = []

        while time.time() - start_time < duration:
            # Generate realistic walking sensor data
            t = time.time() - start_time

            # Simulate step pattern every 1/steps_per_second seconds
            step_phase = (t * steps_per_second) % 1.0

            if step_phase < 0.3:  # Step impact phase
                accel_magnitude = 12 + 5 * np.sin(step_phase * 10)
                gyro_magnitude = 1.5 + 0.8 * np.cos(step_phase * 8)
            else:  # Normal walking phase
                accel_magnitude = 9.8 + 2 * np.sin(t * 3)
                gyro_magnitude = 0.3 + 0.2 * np.cos(t * 4)

            # Add some randomness
            accel_x = accel_magnitude * (0.6 + 0.4 * np.random.random())
            accel_y = accel_magnitude * (0.2 + 0.3 * np.random.random())
            accel_z = accel_magnitude * (0.8 + 0.4 * np.random.random())
            gyro_x = gyro_magnitude * (0.5 + 0.5 * np.random.random())
            gyro_y = gyro_magnitude * (0.3 + 0.7 * np.random.random())
            gyro_z = gyro_magnitude * (0.2 + 0.8 * np.random.random())

            sensor_data = {
                "accel_x": float(accel_x),
                "accel_y": float(accel_y),
                "accel_z": float(accel_z),
                "gyro_x": float(gyro_x),
                "gyro_y": float(gyro_y),
                "gyro_z": float(gyro_z),
            }

            # Send data and get result
            result = await self.send_sensor_data(sensor_data)

            if result:
                reading_count += 1

                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    break

                # Check for step detection
                if result.get("step_detected", False):
                    self.total_steps += 1
                    step_detections.append(
                        {
                            "time": t,
                            "reading": reading_count,
                            "confidence": result.get("confidence", 0),
                        }
                    )
                    print(
                        f"👟 Step {self.total_steps} detected! "
                        f"(Reading #{reading_count}, t={t:.1f}s, "
                        f"confidence={result.get('confidence', 0):.3f})"
                    )

                # Show progress every 2 seconds
                if reading_count % 20 == 0:
                    total_steps_server = result.get("total_steps", 0)
                    processing_time = result.get("processing_time_ms", 0)
                    print(
                        f"📊 Progress: {reading_count} readings, "
                        f"{total_steps_server} steps, "
                        f"{processing_time:.1f}ms processing"
                    )

            # Control reading rate (10 Hz)
            await asyncio.sleep(0.1)

        return step_detections

    async def simulate_stationary(self, duration=5):
        """Simulate stationary motion (should not detect steps)"""
        print(f"\n🧍‍♂️ Simulating {duration}s of stationary motion...")
        print("=" * 50)

        start_time = time.time()
        reading_count = 0

        while time.time() - start_time < duration:
            # Generate stationary sensor data (minimal motion)
            t = time.time() - start_time

            # Small random variations around gravity
            accel_x = 0.1 + 0.05 * np.sin(t * 2)
            accel_y = 0.2 + 0.03 * np.cos(t * 3)
            accel_z = 9.8 + 0.1 * np.sin(t * 1.5)
            gyro_x = 0.02 * np.sin(t * 2.5)
            gyro_y = 0.01 * np.cos(t * 3.2)
            gyro_z = 0.015 * np.sin(t * 1.8)

            sensor_data = {
                "accel_x": float(accel_x),
                "accel_y": float(accel_y),
                "accel_z": float(accel_z),
                "gyro_x": float(gyro_x),
                "gyro_y": float(gyro_y),
                "gyro_z": float(gyro_z),
            }

            result = await self.send_sensor_data(sensor_data)

            if result and "error" not in result:
                reading_count += 1

                if result.get("step_detected", False):
                    print(
                        f"⚠️  Unexpected step detected while stationary! "
                        f"(Reading #{reading_count})"
                    )

                # Show progress every 2 seconds
                if reading_count % 20 == 0:
                    total_steps = result.get("total_steps", 0)
                    print(
                        f"📊 Stationary readings: {reading_count}, "
                        f"total steps: {total_steps}"
                    )

            await asyncio.sleep(0.1)

        print(f"✅ Stationary test complete: {reading_count} readings processed")


async def main():
    """Main test function"""
    print("🚀 WebSocket Real-Time Step Detection Test")
    print("=" * 60)

    client = RealTimeStepClient()

    # Connect to WebSocket
    if not await client.connect():
        print("❌ Cannot connect to WebSocket server")
        print("💡 Make sure FastAPI server is running:")
        print("   uvicorn step_detection_api:app --host 0.0.0.0 --port 8000 --reload")
        return

    try:
        # Test 1: Stationary motion (should not detect steps)
        await client.simulate_stationary(duration=3)

        # Test 2: Walking simulation (should detect steps)
        step_detections = await client.simulate_walking(
            duration=8, steps_per_second=1.5
        )

        print(f"\n📈 Final Results:")
        print(f"   🦶 Total steps detected: {client.total_steps}")
        print(f"   📊 Step detection events: {len(step_detections)}")

        if step_detections:
            print(f"   ⏱️  First step at: {step_detections[0]['time']:.1f}s")
            print(f"   ⏱️  Last step at: {step_detections[-1]['time']:.1f}s")
            avg_confidence = np.mean([s["confidence"] for s in step_detections])
            print(f"   🎯 Average confidence: {avg_confidence:.3f}")

        print(f"\n🎉 Real-time WebSocket test completed successfully!")
        print(f"🔗 WebSocket URL: {WS_URL}")

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import websockets
    except ImportError:
        print("❌ websockets package not found")
        print("📦 Install with: uv add websockets")
        exit(1)

    # Run the test
    asyncio.run(main())
