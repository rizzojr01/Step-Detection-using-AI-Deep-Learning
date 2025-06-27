"""
Step Detection API with FastAPI
===============================

A high-performance web API for real-time step detection that can be used by mobile apps
or other applications. Built with FastAPI for automatic API documentation and type safety.

Usage:
    uvicorn step_detection_api:app --host 0.0.0.0 --port 8000 --reload

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Endpoints:
    POST /detect_step - Process sensor reading
    GET /stats - Get current statistics
    POST /reset - Reset step counter
    GET /health - Health check
"""

from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from initialize_model import load_production_model
from realtime_step_detector import RealTimeStepCounter

# FastAPI app instance
app = FastAPI(
    title="Step Detection API",
    description="Real-time step detection using deep learning models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global step counter instance
step_counter = None


"""
Step Detection API with FastAPI
===============================

A high-performance web API for real-time step detection that can be used by mobile apps
or other applications. Built with FastAPI for automatic API documentation and type safety.

Usage:
    uvicorn step_detection_api:app --host 0.0.0.0 --port 8000 --reload

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Endpoints:
    POST /detect_step - Process sensor reading
    GET /stats - Get current statistics  
    POST /reset - Reset step counter
    GET /health - Health check
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from realtime_step_detector import RealTimeStepCounter

# FastAPI app instance
app = FastAPI(
    title="Step Detection API",
    description="Real-time step detection using deep learning models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global step counter instance
step_counter = None


# Pydantic models for request/response validation
class SensorReading(BaseModel):
    """Sensor reading input model"""

    accel_x: float = Field(..., description="Accelerometer X-axis reading")
    accel_y: float = Field(..., description="Accelerometer Y-axis reading")
    accel_z: float = Field(..., description="Accelerometer Z-axis reading")
    gyro_x: float = Field(..., description="Gyroscope X-axis reading")
    gyro_y: float = Field(..., description="Gyroscope Y-axis reading")
    gyro_z: float = Field(..., description="Gyroscope Z-axis reading")

    class Config:
        json_schema_extra = {
            "example": {
                "accel_x": 1.2,
                "accel_y": -0.5,
                "accel_z": 9.8,
                "gyro_x": 0.1,
                "gyro_y": 0.2,
                "gyro_z": -0.1,
            }
        }


class StepDetectionResult(BaseModel):
    """Step detection result model"""

    step_detected: bool = Field(..., description="Whether a step was detected")
    total_steps: int = Field(..., description="Total steps counted")
    probabilities: List[float] = Field(
        ..., description="Model prediction probabilities [no_step, start, end]"
    )
    confidence: float = Field(..., description="Maximum confidence score")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    in_step: bool = Field(..., description="Whether currently in a step")


class Statistics(BaseModel):
    """Statistics response model"""

    total_steps: int
    elapsed_time_seconds: float
    steps_per_minute: float
    steps_per_hour: float
    avg_processing_time_ms: float
    max_processing_time_ms: float
    buffer_utilization: float
    last_step_time: Optional[float]
    recent_step_timestamps: List[str]


class APIResponse(BaseModel):
    """Generic API response model"""

    status: str = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    result: Optional[StepDetectionResult] = Field(
        None, description="Step detection result"
    )
    statistics: Optional[Statistics] = Field(None, description="Current statistics")


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
        step_counter.start_threshold = 0.3
        step_counter.end_threshold = 0.3

        print(f"✅ Step counter initialized successfully")
        print(f"   Start threshold: {step_counter.start_threshold}")
        print(f"   End threshold: {step_counter.end_threshold}")
        return True

    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    print("🚀 Starting Step Detection API...")
    initialize_model()


@app.post("/detect_step", response_model=APIResponse)
async def detect_step(sensor_data: SensorReading):
    """
    Process a sensor reading and detect steps

    - **sensor_data**: 6D sensor reading (accelerometer + gyroscope)
    - **returns**: Step detection result with probabilities and metrics
    """
    if step_counter is None:
        raise HTTPException(
            status_code=500, detail="Step detection model not initialized"
        )

    try:
        # Process sensor reading
        result = step_counter.add_sensor_reading(
            accel_x=sensor_data.accel_x,
            accel_y=sensor_data.accel_y,
            accel_z=sensor_data.accel_z,
            gyro_x=sensor_data.gyro_x,
            gyro_y=sensor_data.gyro_y,
            gyro_z=sensor_data.gyro_z,
        )

        # Convert to response model
        step_result = StepDetectionResult(**result)

        return APIResponse(status="success", result=step_result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid sensor values: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.get("/stats", response_model=APIResponse)
async def get_statistics():
    """
    Get current step detection statistics

    - **returns**: Comprehensive statistics including step count, rate, and performance metrics
    """
    if step_counter is None:
        raise HTTPException(
            status_code=500, detail="Step detection model not initialized"
        )

    try:
        stats_dict = step_counter.get_statistics()
        stats = Statistics(**stats_dict)

        return APIResponse(status="success", statistics=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {e}")


@app.post("/reset", response_model=APIResponse)
async def reset_counter():
    """
    Reset the step counter

    - **returns**: Confirmation of reset operation
    """
    if step_counter is None:
        raise HTTPException(
            status_code=500, detail="Step detection model not initialized"
        )

    try:
        step_counter.reset()

        return APIResponse(status="success", message="Step counter reset successfully")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting counter: {e}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint

    - **returns**: API health status and model initialization state
    """
    return {
        "status": "healthy",
        "model_initialized": step_counter is not None,
        "api_version": "2.0.0",
        "framework": "FastAPI",
    }


@app.get("/")
async def root():
    """
    API information endpoint

    - **returns**: API documentation and usage information
    """
    return {
        "name": "Step Detection API",
        "version": "2.0.0",
        "framework": "FastAPI",
        "documentation": {"swagger_ui": "/docs", "redoc": "/redoc"},
        "endpoints": {
            "POST /detect_step": "Process sensor reading",
            "GET /stats": "Get current statistics",
            "POST /reset": "Reset step counter",
            "GET /health": "Health check",
        },
        "usage_example": {
            "curl": 'curl -X POST \'http://localhost:8000/detect_step\' -H \'Content-Type: application/json\' -d \'{"accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8, "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1}\''
        },
    }


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

                # Convert numpy types to native Python types for JSON serialization
                json_safe_result = {
                    "step_detected": bool(result["step_detected"]),
                    "total_steps": int(result["total_steps"]),
                    "probabilities": [float(p) for p in result["probabilities"]],
                    "confidence": float(result["confidence"]),
                    "processing_time_ms": float(result["processing_time_ms"]),
                    "in_step": bool(result["in_step"]),
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

    print("🚀 Starting Step Detection API with FastAPI")
    print("📖 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
