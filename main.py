from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.initialize_model import load_production_model
from src.step_detection.core.detector import RealTimeStepCounter
from src.step_detection.utils.config import get_config

# FastAPI app instance
config = get_config()
app = FastAPI(
    title=config.get("api.title", "Step Detection API"),
    description="Real-time step detection using deep learning models",
    version=config.get("api.version", "2.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global step counter instance
step_counter = None


import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Starting Step Detection API...")
    initialize_model()
    yield
    print("🔌 Shutting down Step Detection API...")


app = FastAPI(
    title=config.get("api.title", "Step Detection API"),
    description="Real-time step detection using deep learning models",
    version=config.get("api.version", "2.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Global step counter instance
step_counter = None


def initialize_model():
    """Initialize the step detection model using pre-trained weights"""
    global step_counter

    try:
        # Load the pre-trained model
        model, device = load_production_model()

        if model is None:
            print(
                "❌ Failed to load pre-trained model, falling back to untrained model"
            )
            # Fallback to creating a new model with random weights
            from initialize_model import StepDetectionCNN

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = StepDetectionCNN().to(device)
            model.eval()
            print(f"⚠️  Using untrained model on {device}")

        # Initialize step counter with the loaded model
        step_counter = RealTimeStepCounter(model=model, device=str(device))

        print(f"✅ Step counter initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time step detection streaming

    Send sensor data as JSON and receive real-time step detection results
    """
    await websocket.accept()

    if step_counter is None:
        await websocket.send_json({"error": "Step detection model not initialized"})
        await websocket.close()
        return

    try:
        while True:
            try:
                # Receive sensor data
                data = await websocket.receive_json()

                # Check for reset action - HIGHEST PRIORITY
                if isinstance(data, dict) and data.get("action") == "reset":
                    step_counter.reset()  # Use the proper reset method
                    await websocket.send_json(
                        {
                            "type": "reset_response",
                            "status": "success",
                            "message": "Step counter has been reset",
                            "total_steps": 0,
                            "timestamp": str(time.time()),
                        }
                    )
                    continue

                # Validate required fields
                required_fields = [
                    "accel_x",
                    "accel_y",
                    "accel_z",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                ]
                if not all(field in data for field in required_fields):
                    try:
                        await websocket.send_json(
                            {"error": "Missing sensor data fields"}
                        )
                    except:
                        break  # Client disconnected
                    continue

                # Process sensor reading
                result = step_counter.add_sensor_reading(
                    accel_x=float(data["accel_x"]),
                    accel_y=float(data["accel_y"]),
                    accel_z=float(data["accel_z"]),
                    gyro_x=float(data["gyro_x"]),
                    gyro_y=float(data["gyro_y"]),
                    gyro_z=float(data["gyro_z"]),
                )

                # Convert to the format expected by frontend (matching Dart class structure)
                # Convert to the format expected by frontend (matching StepDetectionResult model)
                json_safe_result = {
                    "type": "step_detection",
                    "step_detected": bool(result["step_detected"]),
                    "prediction": {
                        "confidence": float(result["max_confidence"]),
                        "probabilities": [
                            float(result["start_probability"]),
                            float(result["end_probability"]),
                        ],
                        "start_probability": float(result["start_probability"]),
                        "end_probability": float(result["end_probability"]),
                        "no_step_probability": float(result["no_step_probability"]),
                        "predicted_class": int(result["predicted_class"]),
                        "movement_magnitude": float(result["movement_magnitude"]),
                        "step_start": bool(result["step_start"]),
                        "step_end": bool(result["step_end"]),
                        "detector_has_current_step": bool(
                            result["detector_has_current_step"]
                        ),
                    },
                    "total_steps": int(result["step_count"]),
                    "total_predictions": 1,  # Each call is one prediction
                    "buffer_size": (
                        len(step_counter.sensor_buffer)
                        if hasattr(step_counter, "sensor_buffer")
                        else 0
                    ),
                    "timestamp": str(time.time()),  # Unix timestamp as string
                    "message": (
                        "Step detection processed successfully"
                        if result["status"] == "success"
                        else None
                    ),
                    "error": result.get("error"),
                }

                # Send result back
                try:
                    await websocket.send_json(json_safe_result)
                except:
                    break  # Client disconnected

            except Exception as e:
                # Handle individual operation errors (e.g., invalid JSON, processing errors)
                try:
                    await websocket.send_json({"error": f"Processing error: {e}"})
                except:
                    break  # Client disconnected

    except Exception as e:
        # Only try to send error message if WebSocket is still open
        try:
            if websocket.client_state.value == 1:  # CONNECTED state
                await websocket.send_json({"error": f"WebSocket error: {e}"})
        except:
            pass  # WebSocket already closed, ignore
        finally:
            try:
                await websocket.close()
            except:
                pass  # WebSocket already closed, ignore


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    reload = config.get("api.reload", True)

    print("🚀 Starting Step Detection API with FastAPI")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
