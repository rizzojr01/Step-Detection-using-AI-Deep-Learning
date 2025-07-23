#!/usr/bin/env python3
"""
Advanced CLI for dual-server step detection comparison using Rich library.

This CLI provides a professional terminal interface for comparing step detection
performance between a local server and Modal cloud deployment. It features
real-time latency monitoring, step counting, and beautiful UI components.
"""

import asyncio
import json
import math
import random
import sys
import time
import termios
import select
import websockets
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich import box

console = Console()

# Configuration
LOCAL_WS_URL = "ws://localhost:8000/ws/realtime"
MODAL_WS_URL = "wss://nyu-vision--step-detection-app-fastapi-app.modal.run/ws/realtime"
SENSOR_FREQUENCY = 50  # Hz
STEP_FREQUENCY = 2.0  # Hz (2 steps per second when walking)


class SensorDataGenerator:
    """Generate realistic walking sensor data patterns."""
    
    def __init__(self):
        self.time_offset = 0.0
        self.step_phase = 0.0  # Phase of the step cycle (0-2π)
        self.base_accel_z = 9.8  # Gravity
        self.noise_level = 0.1

        # Walking pattern parameters
        self.step_amplitude = 2.0  # Acceleration amplitude during steps
        self.gyro_amplitude = 0.3  # Gyroscope amplitude during steps
        
    def generate_walking_data(self):
        """Generate realistic sensor data for walking."""
        # Update step phase based on step frequency
        dt = 1.0 / SENSOR_FREQUENCY
        self.step_phase += 2 * math.pi * STEP_FREQUENCY * dt
        if self.step_phase > 2 * math.pi:
            self.step_phase -= 2 * math.pi

        # Generate step pattern using sine waves with different phases
        step_pattern = math.sin(self.step_phase)
        heel_strike = math.sin(self.step_phase + math.pi / 4)
        push_off = math.sin(self.step_phase - math.pi / 4)

        # Add noise for realism
        noise_x = random.uniform(-self.noise_level, self.noise_level)
        noise_y = random.uniform(-self.noise_level, self.noise_level)
        noise_z = random.uniform(-self.noise_level, self.noise_level)

        # Generate accelerometer data (walking creates distinctive patterns)
        accel_x = self.step_amplitude * heel_strike + noise_x
        accel_y = self.step_amplitude * step_pattern * 0.6 + noise_y
        accel_z = self.base_accel_z + self.step_amplitude * push_off * 0.4 + noise_z

        # Generate gyroscope data (body rotation during walking)
        gyro_x = (
            self.gyro_amplitude * math.sin(self.step_phase + math.pi / 6)
            + noise_x * 0.1
        )
        gyro_y = self.gyro_amplitude * step_pattern * 0.8 + noise_y * 0.1
        gyro_z = self.gyro_amplitude * math.cos(self.step_phase) * 0.3 + noise_z * 0.1
        
        return {
            "timestamp": time.time() * 1000,
            "accel_x": round(accel_x, 6),
            "accel_y": round(accel_y, 6), 
            "accel_z": round(accel_z, 6),
            "gyro_x": round(gyro_x, 6),
            "gyro_y": round(gyro_y, 6),
            "gyro_z": round(gyro_z, 6)
        }
   
   
   
   
   
    
    def generate_idle_data(self):
        """Generate sensor data for standing still."""
        # Small random movements (breathing, micro-movements)
        noise_factor = 0.05

        return {
            "timestamp": time.time() * 1000,
            "accel_x": random.uniform(-noise_factor, noise_factor),
            "accel_y": random.uniform(-noise_factor, noise_factor),
            "accel_z": self.base_accel_z + random.uniform(-noise_factor, noise_factor),
            "gyro_x": random.uniform(-noise_factor, noise_factor),
            "gyro_y": random.uniform(-noise_factor, noise_factor),
            "gyro_z": random.uniform(-noise_factor, noise_factor)
        }


class KeyboardHandler:
    """Handle keyboard input in non-blocking mode."""
    
    def __init__(self):
        self.old_settings = None
        
    def setup_terminal(self):
        """Setup terminal for raw keyboard input."""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            # Set terminal to cbreak mode (raw input)
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
    
    def restore_terminal(self):
        """Restore terminal to original settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def kbhit(self):
        """Check if a key has been pressed."""
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    
    def getch(self):
        """Get a single character from stdin."""
        return sys.stdin.read(1)


class StepSimulatorCLI:
    """Main CLI application for dual server step detection comparison."""
    
    def __init__(self):
        self.sensor_generator = SensorDataGenerator()
        self.keyboard_handler = KeyboardHandler()
        
        # Connection states
        self.local_websocket = None
        self.modal_websocket = None
        
        # Application state
        self.running = True
        self.walking = False
        self.start_time = time.time()
        
        # Step counting
        self.local_step_count = 0
        self.modal_step_count = 0
        self.total_local_steps_detected = 0
        self.total_modal_steps_detected = 0
        
        # Response tracking
        self.local_last_step_detected = False
        self.modal_last_step_detected = False
        self.local_last_confidence = 0
        self.modal_last_confidence = 0
        
        # Latency tracking
        self.local_send_time = 0
        self.modal_send_time = 0
        self.local_latency_history = []
        self.modal_latency_history = []
        
        # Reset tracking to prevent overwriting reset values
        self.reset_sent_time = 0
        self.reset_ignore_duration = 2.0  # Ignore server responses for 2 seconds after reset
        self.max_latency_samples = 50
        self.local_latency_samples = []
        self.modal_latency_samples = []
        
        # Data transmission
        self.data_sent_count = 0

    def create_header_panel(self):
        """Create the header panel with title and controls."""
        header_content = Text.assemble(
            ("🏃 Dual Server Step Detection Comparison", "bold cyan"),
            "\n\n",
            ("Controls: ", "bold"),
            ("[W] ", "green"), ("Hold to simulate walking & send data  ", ""),
            ("[R] ", "yellow"), ("Reset step count  ", ""),
            ("[Q] ", "red"), ("Quit", ""),
            "\n\n",
            ("Local:  ", "bold"), (LOCAL_WS_URL, "dim"),
            "\n",
            ("Modal:  ", "bold"), (MODAL_WS_URL[:65] + "...", "dim")
        )
        
        return Panel(
            header_content,
            title="Step Detection CLI",
            border_style="blue",
            box=box.DOUBLE
        )

    def create_status_panel(self):
        """Create the status panel showing current activity."""
        if self.walking:
            status_text = Text("🚶 Walking", style="bold green")
            activity_desc = "Sending walking sensor data at 50Hz"
        else:
            status_text = Text("🧍 Idle", style="bold yellow")
            activity_desc = "No data transmission (press & hold W to walk)"
        
        session_duration = self.get_session_duration()
        
        status_content = Text.assemble(
            ("Status: ", "bold"), status_text, "\n",
            ("Activity: ", "bold"), (activity_desc, ""),
            "\n",
            ("Session Duration: ", "bold"), (session_duration, "cyan"),
            "\n",
            ("Data Packets Sent: ", "bold"), (str(self.data_sent_count), "magenta")
        )
        
        return Panel(
            status_content,
            title="Current Status",
            border_style="cyan"
        )

    def create_server_panel(self, server_name, url, connected, step_count, 
                          last_detected, confidence, latency):
        """Create a panel for server information."""
        # Connection status
        if connected:
            conn_status = Text("🟢 Connected", style="bold green")
        else:
            conn_status = Text("🔴 Disconnected", style="bold red")
        
        # Step detection indicator
        if last_detected:
            detection_status = Text("👟 Step Detected!", style="bold green blink")
            conf_text = f"{confidence:.2%}"
        else:
            detection_status = Text("⚪ No Step", style="dim")
            conf_text = "N/A"
        
        # Latency display
        if latency is not None:
            latency_text = f"{latency:.1f}ms"
            if latency < 50:
                latency_style = "green"
            elif latency < 100:
                latency_style = "yellow"
            else:
                latency_style = "red"
        else:
            latency_text = "N/A"
            latency_style = "dim"
        
        content = Text.assemble(
            ("URL: ", "bold"), (url[:40] + "..." if len(url) > 40 else url, "dim"), "\n",
            ("Status: ", "bold"), conn_status, "\n",
            ("Step Count: ", "bold"), (str(step_count), "cyan"), "\n",
            ("Detection: ", "bold"), detection_status, "\n",
            ("Confidence: ", "bold"), (conf_text, "yellow"), "\n",
            ("Latency: ", "bold"), (latency_text, latency_style)
        )
        
        return Panel(
            content,
            title=f"{server_name} Server",
            border_style="blue" if connected else "red"
        )

    def create_comparison_panel(self):
        """Create a comparison panel with side-by-side metrics."""
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="bold")
        table.add_column("Local Server", justify="center")
        table.add_column("Modal Server", justify="center")
        table.add_column("Difference", justify="center")
        
        # Step count comparison
        step_diff = self.modal_step_count - self.local_step_count
        step_diff_style = "green" if step_diff == 0 else "yellow"
        table.add_row(
            "Step Count",
            str(self.local_step_count),
            str(self.modal_step_count),
            f"{step_diff:+d}",
            style=step_diff_style if step_diff == 0 else ""
        )
        
        # Total detections
        det_diff = self.total_modal_steps_detected - self.total_local_steps_detected
        table.add_row(
            "Total Detections",
            str(self.total_local_steps_detected),
            str(self.total_modal_steps_detected),
            f"{det_diff:+d}"
        )
        
        # Average latency
        local_avg = self.get_local_average_latency()
        modal_avg = self.get_modal_average_latency()
        if local_avg is not None and modal_avg is not None:
            latency_diff = modal_avg - local_avg
            latency_diff_text = f"{latency_diff:+.1f}ms"
        else:
            latency_diff_text = "N/A"
        
        table.add_row(
            "Avg Latency",
            f"{local_avg:.1f}ms" if local_avg else "N/A",
            f"{modal_avg:.1f}ms" if modal_avg else "N/A",
            latency_diff_text
        )
        
        return Panel(
            table,
            title="Server Comparison",
            border_style="magenta"
        )

    def render_ui(self):
        """Render the complete UI layout."""
        layout = Layout()
        
        # Create main sections
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main", ratio=1),
            Layout(name="comparison", size=8)
        )
        
        # Split main area into status and servers
        layout["main"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="servers", ratio=2)
        )
        
        # Split servers into local and modal
        layout["servers"].split_row(
            Layout(name="local", ratio=1),
            Layout(name="modal", ratio=1)
        )
        
        # Populate panels
        layout["header"].update(self.create_header_panel())
        layout["status"].update(self.create_status_panel())
        
        local_latency = self.get_local_current_latency()
        modal_latency = self.get_modal_current_latency()
        
        layout["local"].update(
            self.create_server_panel(
                "Local", LOCAL_WS_URL, self.local_websocket is not None,
                self.local_step_count, self.local_last_step_detected,
                self.local_last_confidence, local_latency
            )
        )
        
        layout["modal"].update(
            self.create_server_panel(
                "Modal", MODAL_WS_URL, self.modal_websocket is not None,
                self.modal_step_count, self.modal_last_step_detected,
                self.modal_last_confidence, modal_latency
            )
        )
        
        layout["comparison"].update(self.create_comparison_panel())
        
        return layout

    def get_session_duration(self):
        """Get formatted session duration."""
        duration = time.time() - self.start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_local_average_latency(self):
        """Calculate local server average latency."""
        if not self.local_latency_samples:
            return None
        return sum(self.local_latency_samples) / len(self.local_latency_samples)

    def get_modal_average_latency(self):
        """Calculate modal server average latency."""
        if not self.modal_latency_samples:
            return None
        return sum(self.modal_latency_samples) / len(self.modal_latency_samples)

    def get_local_current_latency(self):
        """Get the most recent local latency."""
        return self.local_latency_samples[-1] if self.local_latency_samples else None

    def get_modal_current_latency(self):
        """Get the most recent modal latency."""
        return self.modal_latency_samples[-1] if self.modal_latency_samples else None

    def calculate_local_latency(self):
        """Calculate and store local server latency."""
        if self.local_send_time:
            latency = (time.time() - self.local_send_time) * 1000
            self.local_latency_samples.append(latency)
            if len(self.local_latency_samples) > self.max_latency_samples:
                self.local_latency_samples.pop(0)

    def calculate_modal_latency(self):
        """Calculate and store modal server latency."""
        if self.modal_send_time:
            latency = (time.time() - self.modal_send_time) * 1000
            self.modal_latency_samples.append(latency)
            if len(self.modal_latency_samples) > self.max_latency_samples:
                self.modal_latency_samples.pop(0)

    async def connect_local_websocket(self):
        """Connect to local WebSocket server."""
        try:
            self.local_websocket = await websockets.connect(LOCAL_WS_URL)
            return True
        except Exception:
            self.local_websocket = None
            return False

    async def connect_modal_websocket(self):
        """Connect to Modal WebSocket server."""
        try:
            self.modal_websocket = await websockets.connect(MODAL_WS_URL)
            return True
        except Exception:
            self.modal_websocket = None
            return False

    async def connect_websockets(self):
        """Connect to both WebSocket servers."""
        local_task = asyncio.create_task(self.connect_local_websocket())
        modal_task = asyncio.create_task(self.connect_modal_websocket())
        
        await asyncio.gather(local_task, modal_task, return_exceptions=True)
        
        return self.local_websocket is not None or self.modal_websocket is not None

    async def send_sensor_data(self, sensor_data):
        """Send sensor data to both servers."""
        message = json.dumps(sensor_data)
        
        # Send to local server
        if self.local_websocket:
            try:
                self.local_send_time = time.time()
                await self.local_websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                self.local_websocket = None

        # Send to modal server
        if self.modal_websocket:
            try:
                self.modal_send_time = time.time()
                await self.modal_websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                self.modal_websocket = None

        self.data_sent_count += 1

    async def reset_step_count(self):
        """Reset step counters on both servers."""
        reset_message = json.dumps({"action": "reset"})
        
        # Record when reset was sent to ignore server responses for a brief period
        self.reset_sent_time = time.time()
        
        if self.local_websocket:
            try:
                await self.local_websocket.send(reset_message)
            except websockets.exceptions.ConnectionClosed:
                pass
                
        if self.modal_websocket:
            try:
                await self.modal_websocket.send(reset_message)
            except websockets.exceptions.ConnectionClosed:
                pass
        
        # Reset local counters immediately
        self.local_step_count = 0
        self.modal_step_count = 0
        self.total_local_steps_detected = 0
        self.total_modal_steps_detected = 0

    def should_ignore_step_count_update(self):
        """Check if we should ignore step count updates from server (after recent reset)."""
        if self.reset_sent_time == 0:
            return False
        
        time_since_reset = time.time() - self.reset_sent_time
        return time_since_reset < self.reset_ignore_duration

    def handle_keyboard_input(self):
        """Handle keyboard input and return action."""
        if self.keyboard_handler.kbhit():
            key = self.keyboard_handler.getch().lower()
            
            if key == 'w':
                self.walking = True
                return "walking"
            elif key == 'r':
                return "reset"
            elif key == 'q':
                return "quit"
        else:
            # Released walking key - stop sending data
            self.walking = False

        return None

    async def run_simulation_loop(self):
        """Main simulation loop using Rich Live display."""
        console.clear()
        
        local_response = None
        modal_response = None
        loop_start_time = time.time()

        # Use Rich Live for real-time updates
        with Live(self.render_ui(), console=console, refresh_per_second=10, screen=True) as live:
            while self.running:
                try:
                    # Handle keyboard input
                    keyboard_action = self.handle_keyboard_input()

                    if keyboard_action == "reset":
                        await self.reset_step_count()
                    elif keyboard_action == "quit":
                        self.running = False
                        break

                    # Only generate and send sensor data when walking
                    if self.walking:
                        # Generate sensor data
                        sensor_data = self.sensor_generator.generate_walking_data()
                        await self.send_sensor_data(sensor_data)

                        # Reset step detection flags
                        self.local_last_step_detected = False
                        self.modal_last_step_detected = False

                        # Listen for WebSocket responses from both servers
                        if self.local_websocket:
                            try:
                                response = await asyncio.wait_for(
                                    self.local_websocket.recv(), timeout=0.001
                                )
                                local_response = json.loads(response)
                                
                                # Calculate local latency
                                self.calculate_local_latency()

                                # Update local step count from response (but not immediately after reset)
                                if "step_count" in local_response and not self.should_ignore_step_count_update():
                                    self.local_step_count = local_response["step_count"]
                                
                                # Check for step detection
                                if local_response.get("step_detected", False):
                                    self.local_last_step_detected = True
                                    self.local_last_confidence = local_response.get("max_confidence", 0)
                                    self.total_local_steps_detected += 1

                            except asyncio.TimeoutError:
                                pass  # No response yet, continue
                            except websockets.exceptions.ConnectionClosed:
                                await self.connect_local_websocket()

                        if self.modal_websocket:
                            try:
                                response = await asyncio.wait_for(
                                    self.modal_websocket.recv(), timeout=0.001
                                )
                                modal_response = json.loads(response)
                                
                                # Calculate modal latency
                                self.calculate_modal_latency()

                                # Update modal step count from response (but not immediately after reset)
                                if "step_count" in modal_response and not self.should_ignore_step_count_update():
                                    self.modal_step_count = modal_response["step_count"]
                                
                                # Check for step detection
                                if modal_response.get("step_detected", False):
                                    self.modal_last_step_detected = True
                                    self.modal_last_confidence = modal_response.get("max_confidence", 0)
                                    self.total_modal_steps_detected += 1

                            except asyncio.TimeoutError:
                                pass  # No response yet, continue
                            except websockets.exceptions.ConnectionClosed:
                                await self.connect_modal_websocket()

                    # Update the live display
                    live.update(self.render_ui())

                    # Maintain sensor frequency only when walking, otherwise use slower refresh
                    if self.walking:
                        elapsed = time.time() - loop_start_time
                        sleep_time = (1.0 / SENSOR_FREQUENCY) - elapsed
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
                    else:
                        # When idle, just refresh UI at lower frequency to save CPU
                        await asyncio.sleep(0.1)

                    loop_start_time = time.time()

                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    console.print(f"❌ Error in simulation loop: {e}", style="red")
                    await asyncio.sleep(0.1)

    async def run(self):
        """Run the CLI application."""
        try:
            # Setup terminal for keyboard input
            self.keyboard_handler.setup_terminal()

            # Connect to WebSockets
            if not await self.connect_websockets():
                console.print("❌ Failed to connect to any WebSocket server", style="red")
                return

            # Start simulation loop
            await self.run_simulation_loop()

        except KeyboardInterrupt:
            console.print("\n👋 Interrupted by user", style="cyan")
        except Exception as e:
            console.print(f"\n❌ Unexpected error: {e}", style="red")
        finally:
            # Cleanup
            if self.local_websocket:
                await self.local_websocket.close()
            if self.modal_websocket:
                await self.modal_websocket.close()
            self.keyboard_handler.restore_terminal()
            
            # Show session summary
            console.print("\n✅ Session Summary:", style="bold green")
            console.print(f"   Duration: {self.get_session_duration()}")
            console.print(f"   Data Packets Sent: [bold]{self.data_sent_count}[/bold]")
            console.print(f"   Local Steps: [bold]{self.local_step_count}[/bold] | Modal Steps: [bold]{self.modal_step_count}[/bold]")
            console.print(f"   Total Local Steps Detected: [bold]{self.total_local_steps_detected}[/bold]")
            console.print(f"   Total Modal Steps Detected: [bold]{self.total_modal_steps_detected}[/bold]")
            
            if self.local_latency_samples:
                local_avg_latency = self.get_local_average_latency()
                console.print(f"   Local Average Latency: [bold]{local_avg_latency:.1f}ms[/bold]")
            if self.modal_latency_samples:
                modal_avg_latency = self.get_modal_average_latency()
                console.print(f"   Modal Average Latency: [bold]{modal_avg_latency:.1f}ms[/bold]")
            
            console.print("   Thanks for using Dual Server Step Detection Comparison!", style="cyan")
            console.print("   Simulator closed", style="dim")


def main():
    """Main entry point."""
    console.print("🏃 Starting Dual Server Step Detection Comparison...", style="bold magenta")

    # Check if we're in a proper terminal
    if not sys.stdin.isatty():
        console.print("❌ This program requires a terminal with keyboard input support.", style="red")
        return

    # Run the async application
    simulator = StepSimulatorCLI()

    try:
        asyncio.run(simulator.run())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="cyan")


if __name__ == "__main__":
    main()
