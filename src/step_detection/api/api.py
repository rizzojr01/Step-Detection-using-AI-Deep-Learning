"""
FastAPI Step Detection API
REST API for real-time step detection using TensorFlow models.
"""

import json
import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.detector import SimpleStepCounter, StepDetector, load_model_info
from ..utils.config import get_config


class SensorReading(BaseModel):
    """Sensor reading data model."""

    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class StepDetectionResponse(BaseModel):
    """Step detection response model."""

    step_start: bool
    step_end: bool
    start_probability: float
    end_probability: float
    step_count: int
    timestamp: str


class StepCountResponse(BaseModel):
    """Step count response model."""

    step_count: int
    last_detection: Optional[Dict] = None


# Initialize FastAPI app
app = FastAPI(
    title="Step Detection API",
    description="Real-time step detection using TensorFlow CNN",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for detectors
detector: Optional[StepDetector] = None
counter: Optional[SimpleStepCounter] = None
model_info: Dict = {}
config = None  # Configuration object
is_resetting = False  # Flag to block processing during reset
_last_step_count = 0  # Track step count changes
_raw_model_step_count = 0  # Track raw model detections


@app.on_event("startup")
async def startup_event():
    """Initialize detectors on startup."""
    global detector, counter, model_info, config

    # Load configuration
    config = get_config()

    model_path = "models/step_detection_model.keras"
    metadata_path = "models/model_metadata.json"

    if os.path.exists(model_path):
        try:
            detector = StepDetector(model_path)
            counter = SimpleStepCounter(model_path)
            model_info = load_model_info(metadata_path)
            print(f"✅ Models loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    else:
        print(f"⚠️ Model not found: {model_path}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Step Detection API",
        "version": "1.0.0",
        "status": "active" if detector is not None else "model_not_loaded",
        "endpoints": {
            "detect_step": "POST /detect_step - Detect steps from sensor data",
            "step_count": "GET /step_count - Get current step count",
            "reset_count": "POST /reset_count - Reset step count",
            "session_summary": "GET /session_summary - Get session summary",
            "model_info": "GET /model_info - Get model information",
            "websocket": "WS /ws/realtime - Real-time step detection via WebSocket",
        },
    }


@app.post("/detect_step", response_model=StepDetectionResponse)
async def detect_step(reading: SensorReading):
    """Detect steps from sensor reading."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = detector.process_reading(
            reading.accel_x,
            reading.accel_y,
            reading.accel_z,
            reading.gyro_x,
            reading.gyro_y,
            reading.gyro_z,
        )

        return StepDetectionResponse(
            step_start=result["step_start_detected"],
            step_end=result["step_end_detected"],
            start_probability=result["predictions"]["start_prob"],
            end_probability=result["predictions"]["end_prob"],
            step_count=result["step_count"],
            timestamp=str(result["timestamp"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.get("/step_count", response_model=StepCountResponse)
async def get_step_count():
    """Get current step count."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StepCountResponse(step_count=detector.get_step_count(), last_detection=None)


@app.post("/reset_count")
async def reset_step_count():
    """Reset step count."""
    global is_resetting
    print("🔄 EMERGENCY RESET REQUEST RECEIVED VIA HTTP - STOPPING ALL PROCESSING")

    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Set reset flag to block all processing
    is_resetting = True

    # Reset the primary detector (used by websocket)
    print("🔄 Resetting detector state...")
    detector.reset()

    # Also reset the counter if it exists (for consistency)
    if counter is not None:
        print("🔄 Resetting counter state...")
        counter.reset()

    # Reset tracking variables
    global _raw_model_step_count, _last_step_count
    _raw_model_step_count = 0
    _last_step_count = 0

    # Clear reset flag
    is_resetting = False
    print("✅ COMPLETE RESET FINISHED VIA HTTP - Ready to start fresh from zero")

    return {
        "message": "Complete reset successful",
        "step_count": 0,
        "status": "reset_complete",
    }


@app.get("/session_summary")
async def get_session_summary():
    """Get session summary."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return detector.get_session_summary()


@app.post("/save_session")
async def save_session(filename: str = "session.json"):
    """Save current session data."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        detector.save_session(filename)
        return {"message": f"Session saved to {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")


@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    if not model_info:
        return {"message": "Model info not available"}

    return {
        "model_info": model_info,
        "api_status": "active" if detector is not None else "model_not_loaded",
        "thresholds": (
            {
                "confidence_threshold": (
                    detector.confidence_threshold if detector else None
                ),
                "magnitude_threshold": (
                    detector.magnitude_threshold if detector else None
                ),
            }
            if detector
            else None
        ),
    }


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time step detection."""
    global _raw_model_step_count, _last_step_count, is_resetting

    await websocket.accept()

    if detector is None:
        await websocket.send_text(
            json.dumps({"error": "Model not loaded", "status": "error"})
        )
        await websocket.close(code=1003)
        return

    try:
        while True:
            # Receive sensor data from client
            data = await websocket.receive_text()

            try:
                sensor_data = json.loads(data)

                # Check if this is a reset command - IMMEDIATE PRIORITY
                if sensor_data.get("action") == "reset":
                    print(
                        "🔄 EMERGENCY RESET COMMAND RECEIVED - STOPPING ALL PROCESSING"
                    )

                    # Set reset flag to block all other processing immediately
                    is_resetting = True

                    # Reset the detector completely
                    print("🔄 Resetting detector state...")
                    detector.reset()

                    # Also reset the counter if it exists
                    if counter is not None:
                        print("🔄 Resetting counter state...")
                        counter.reset()

                    # Reset tracking variables
                    _raw_model_step_count = 0
                    _last_step_count = 0

                    # Clear reset flag
                    is_resetting = False
                    print("✅ COMPLETE RESET FINISHED - Ready to start fresh from zero")

                    # Send immediate confirmation response with zero values
                    await websocket.send_text(
                        json.dumps(
                            {
                                "step_start": False,
                                "step_end": False,
                                "step_detected": False,
                                "start_probability": 0.0,
                                "end_probability": 0.0,
                                "no_step_probability": 0.0,
                                "max_confidence": 0.0,
                                "predicted_class": 0,
                                "step_count": 0,
                                "movement_magnitude": 0.0,
                                "detector_has_current_step": False,
                                "timestamp": "",
                                "status": "reset_complete",
                            }
                        )
                    )
                    continue

                # Check if we're currently resetting - block all sensor processing
                if is_resetting:
                    print("⏸️ Blocking sensor processing - reset in progress")
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
                if not all(field in sensor_data for field in required_fields):
                    await websocket.send_text(
                        json.dumps(
                            {
                                "error": "Missing required sensor data fields",
                                "required": required_fields,
                                "status": "error",
                            }
                        )
                    )
                    continue

                # Process the reading
                result = detector.process_reading(
                    sensor_data["accel_x"],
                    sensor_data["accel_y"],
                    sensor_data["accel_z"],
                    sensor_data["gyro_x"],
                    sensor_data["gyro_y"],
                    sensor_data["gyro_z"],
                )

                # Extract raw model predictions for detailed logging
                predictions = result["predictions"]
                no_step_prob = 1.0 - predictions["start_prob"] - predictions["end_prob"]

                # Get all the detection states
                step_start = result["step_start_detected"]
                step_end = result["step_end_detected"]
                step_detected = result.get("step_detected", False)
                step_count = result["step_count"]

                # Get sensitivity control info if available
                sensitivity_info = result.get("sensitivity_control", {})
                movement_magnitude = sensitivity_info.get("movement_magnitude", 0.0)

                # Enhanced logging with raw model output
                # Log raw model step detection (before time filtering)
                confidence_threshold = (
                    detector.confidence_threshold
                    if hasattr(detector, "confidence_threshold")
                    else 0.5
                )
                raw_step_start = predictions["start_prob"] > confidence_threshold
                raw_step_end = predictions["end_prob"] > confidence_threshold

                # Track raw model detections
                if raw_step_start or raw_step_end:
                    _raw_model_step_count += 1

                # Only log when there's actual step activity (if enabled in config)
                should_log = (config and config.is_step_detection_logs_enabled()) and (
                    not (config and config.should_log_only_on_activity())
                    or raw_step_start
                    or raw_step_end
                    or step_start
                    or step_end
                    or step_detected
                )

                if should_log:
                    print("=" * 60)
                    print(
                        f"💾 PROBABILITIES: Start={predictions['start_prob']:.3f}, End={predictions['end_prob']:.3f}"
                    )

                    # Show confidence threshold if enabled in config
                    if config and config.should_show_confidence_threshold():
                        print(f"🎯 CONFIDENCE THRESHOLD: {confidence_threshold:.3f}")

                    # Show raw model tracking if enabled in config
                    if config and config.is_raw_model_tracking_enabled():
                        print(
                            f"🤖 RAW MODEL: Start={raw_step_start}, End={raw_step_end}, RAW COUNT={_raw_model_step_count}"
                        )

                    print(
                        f"📊 FILTERED: Start={step_start}, End={step_end}, FILTERED COUNT={step_count}"
                    )

                    # Compare before vs after filtering (only if raw model tracking is enabled)
                    if config and config.is_raw_model_tracking_enabled():
                        if (raw_step_start or raw_step_end) and not (
                            step_start or step_end
                        ):
                            pass
                            # Check what type of filter blocked it
                            # if not (config and config.is_time_filter_enabled()):
                            #     print("⚠️  NON-TIME FILTER BLOCKED MODEL DETECTION!")
                            # else:
                            #     print("⚠️  TIME FILTER BLOCKED MODEL DETECTION!")
                        elif (raw_step_start or raw_step_end) and (
                            step_start or step_end
                        ):
                            pass
                            # print("✅ FILTERS PASSED MODEL DETECTION")

                    if step_count > _last_step_count:
                        print(f"🚀 STEP COUNT INCREASED TO: {step_count}")
                        _last_step_count = step_count
                    print("-" * 60)

                # Get detector state for response
                detector_in_step = getattr(detector, "in_step", False)
                response = {
                    "step_start": bool(step_start),
                    "step_end": bool(step_end),
                    "step_detected": bool(step_detected),
                    "start_probability": float(predictions["start_prob"]),
                    "end_probability": float(predictions["end_prob"]),
                    "no_step_probability": float(no_step_prob),
                    "max_confidence": float(
                        predictions.get(
                            "max_confidence",
                            max(predictions["start_prob"], predictions["end_prob"]),
                        )
                    ),
                    "predicted_class": int(predictions.get("predicted_class", 0)),
                    "step_count": int(step_count),
                    "movement_magnitude": float(movement_magnitude),
                    "detector_has_current_step": detector_in_step,
                    "timestamp": str(result["timestamp"]),
                    "status": "success",
                }

                await websocket.send_text(json.dumps(response))

            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"error": "Invalid JSON format", "status": "error"})
                )
            except Exception as e:
                print(f"❌ Processing error: {str(e)}")
                await websocket.send_text(
                    json.dumps(
                        {"error": f"Processing error: {str(e)}", "status": "error"}
                    )
                )

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "api_version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Step Detection API...")
    print("Documentation available at: http://localhost:8000/docs")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
