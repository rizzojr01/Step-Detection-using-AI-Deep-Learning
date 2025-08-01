import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from src.initialize_model import load_production_model
from src.step_detection.core.detector import RealTimeStepCounter
from src.step_detection.utils.config_db import (
    get_all_config,
    get_config_value,
    set_config_value,
)

# FastAPI app instance

# Global step counter instance
# Global step counter instance

step_counter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Step Detection API...")
    initialize_model()
    yield
    print("🔌 Shutting down Step Detection API...")


app = FastAPI(
    title=get_config_value("api.title", "Step Detection API"),
    description="Real-time step detection using deep learning models",
    version=get_config_value("api.version", "2.0.0"),
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

    # Timeout settings - 5 minutes of inactivity
    timeout_seconds = 5 * 60  # 5 minutes
    last_activity = time.time()

    try:
        while True:
            try:
                # Check for timeout (5 minutes of inactivity)
                current_time = time.time()
                if current_time - last_activity > timeout_seconds:
                    await websocket.send_json(
                        {
                            "type": "timeout_response",
                            "status": "timeout",
                            "message": "Connection closed due to 5 minutes of inactivity",
                            "timestamp": str(current_time),
                        }
                    )
                    await websocket.close()
                    return

                # Receive sensor data with timeout
                import asyncio

                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(), timeout=60.0
                    )  # 1 minute timeout for each receive
                    last_activity = time.time()  # Update activity timestamp
                except asyncio.TimeoutError:
                    # Continue loop to check for overall timeout
                    continue

                # Check for stop action - HIGHEST PRIORITY
                if isinstance(data, dict) and data.get("action") == "stop":
                    await websocket.send_json(
                        {
                            "type": "stop_response",
                            "status": "success",
                            "message": "Step detection stopped. WebSocket connection will be closed.",
                            "timestamp": str(time.time()),
                        }
                    )
                    await websocket.close()
                    return

                # Check for reset action - SECOND PRIORITY
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


# --- Config API endpoints ---

from typing import Optional


class DetectionConfig(BaseModel):
    window_size: Optional[int] = None
    start_threshold: Optional[float] = None
    end_threshold: Optional[float] = None
    min_step_interval: Optional[float] = None
    motion_threshold: Optional[float] = None
    gyro_threshold: Optional[float] = None
    min_motion_variance: Optional[float] = None
    stillness_threshold: Optional[float] = None
    ai_high_confidence_threshold: Optional[float] = None
    ai_sensor_disagree_threshold: Optional[float] = None


class ConfigUpdateRequest(BaseModel):
    value: str


# Only allow these keys to be managed via API
ALLOWED_CONFIG_KEYS = [
    "window_size",
    "start_threshold",
    "end_threshold",
    "min_step_interval",
    "motion_threshold",
    "gyro_threshold",
    "min_motion_variance",
    "stillness_threshold",
    "ai_high_confidence_threshold",
    "ai_sensor_disagree_threshold",
]


@app.get("/config", response_model=DetectionConfig)
def get_config_api():
    """Get all important detection config values as a single object"""
    all_items = get_all_config()
    config_dict = {}
    for k in ALLOWED_CONFIG_KEYS:
        v = all_items.get(k)
        if v is not None:
            # Try to cast to correct type
            if k == "window_size":
                try:
                    config_dict[k] = int(float(v))
                except ValueError:
                    try:
                        config_dict[k] = float(v)
                    except ValueError:
                        config_dict[k] = None
            elif k == "min_step_interval":
                try:
                    config_dict[k] = float(v)
                except ValueError:
                    config_dict[k] = None
            else:
                try:
                    config_dict[k] = float(v)
                except ValueError:
                    config_dict[k] = None
        else:
            config_dict[k] = None
    return DetectionConfig(**config_dict)


@app.get("/config/{key}", response_model=DetectionConfig)
def get_config_key_api(key: str):
    if key not in ALLOWED_CONFIG_KEYS:
        raise HTTPException(status_code=403, detail="Config key not allowed")
    val = get_config_value(key)
    if val is None:
        raise HTTPException(status_code=404, detail="Config key not found")
    # Return config with only this key set
    from typing import Any, Dict, cast

    config_dict = cast(Dict[str, Any], {k: None for k in ALLOWED_CONFIG_KEYS})
    # Assign correct type for each key
    if key == "window_size":
        try:
            config_dict[key] = int(float(val))
        except ValueError:
            try:
                config_dict[key] = float(val)
            except ValueError:
                config_dict[key] = None
    elif key == "min_step_interval":
        try:
            config_dict[key] = float(val)
        except ValueError:
            config_dict[key] = None
    else:
        try:
            config_dict[key] = float(val)
        except ValueError:
            config_dict[key] = None
    return DetectionConfig(**config_dict)


@app.post("/config/{key}", response_model=DetectionConfig)
def set_config_key_api(key: str, req: ConfigUpdateRequest):
    if key not in ALLOWED_CONFIG_KEYS:
        raise HTTPException(status_code=403, detail="Config key not allowed")
    set_config_value(key, req.value)
    # Return updated config
    return get_config_api()


if __name__ == "__main__":
    import uvicorn

    host = get_config_value("api.host", "0.0.0.0")
    port = int(get_config_value("api.port", 8000))
    reload = get_config_value("api.reload", "True") == "True"

    print("🚀 Starting Step Detection API with FastAPI")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
