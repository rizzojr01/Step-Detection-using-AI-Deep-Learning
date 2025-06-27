# API Documentation

## Overview

The Step Detection API provides both REST and WebSocket endpoints for real-time step detection using machine learning models.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production deployment, consider adding API keys or OAuth.

## REST Endpoints

### GET /

Get API information and available endpoints.

**Response:**
```json
{
  "message": "Step Detection API",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "detect_step": "POST /detect_step - Detect steps from sensor data",
    "step_count": "GET /step_count - Get current step count",
    "reset_count": "POST /reset_count - Reset step count",
    "session_summary": "GET /session_summary - Get session summary",
    "model_info": "GET /model_info - Get model information",
    "websocket": "WS /ws/realtime - Real-time step detection via WebSocket"
  }
}
```

### POST /detect_step

Detect steps from a single sensor reading.

**Request Body:**
```json
{
  "accel_x": 1.2,
  "accel_y": -0.5,
  "accel_z": 9.8,
  "gyro_x": 0.1,
  "gyro_y": 0.2,
  "gyro_z": -0.1
}
```

**Response:**
```json
{
  "step_start": false,
  "step_end": false,
  "start_probability": 0.0024,
  "end_probability": 0.0020,
  "step_count": 0,
  "timestamp": "2025-06-27T14:20:19.586914"
}
```

**Status Codes:**
- `200`: Success
- `503`: Model not loaded
- `500`: Processing error

### GET /step_count

Get the current step count from the simple counter.

**Response:**
```json
{
  "step_count": 42,
  "last_detection": {
    "timestamp": "2025-06-27T14:20:19.586914",
    "confidence": 0.85
  }
}
```

### POST /reset_count

Reset both the step detector and simple counter.

**Response:**
```json
{
  "message": "Step count reset",
  "step_count": 0
}
```

### GET /session_summary

Get a summary of the current detection session.

**Response:**
```json
{
  "total_readings": 150,
  "total_steps": 12,
  "current_step_in_progress": false,
  "thresholds": {
    "start_threshold": 0.15,
    "end_threshold": 0.15
  }
}
```

### GET /model_info

Get information about the loaded model.

**Response:**
```json
{
  "model_info": {
    "model_type": "CNN",
    "framework": "TensorFlow/Keras",
    "input_shape": [6],
    "output_classes": 3,
    "validation_accuracy": 0.9542
  },
  "api_status": "active",
  "thresholds": {
    "start_threshold": 0.15,
    "end_threshold": 0.15
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "1.0.0"
}
```

## WebSocket Endpoint

### WS /ws/realtime

Real-time step detection via WebSocket connection.

**Connection URL:**
```
ws://localhost:8000/ws/realtime
```

**Send Message Format:**
```json
{
  "accel_x": 1.2,
  "accel_y": -0.5,
  "accel_z": 9.8,
  "gyro_x": 0.1,
  "gyro_y": 0.2,
  "gyro_z": -0.1
}
```

**Receive Message Format:**
```json
{
  "step_start": false,
  "step_end": false,
  "start_probability": 0.0024,
  "end_probability": 0.0020,
  "step_count": 0,
  "timestamp": "2025-06-27T14:20:19.586914",
  "status": "success"
}
```

**Error Message Format:**
```json
{
  "error": "Missing required sensor data fields",
  "required": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
  "status": "error"
}
```

## Data Types

### SensorReading

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| accel_x | float | X-axis acceleration (m/s²) | -50 to 50 |
| accel_y | float | Y-axis acceleration (m/s²) | -50 to 50 |
| accel_z | float | Z-axis acceleration (m/s²) | -50 to 50 |
| gyro_x | float | X-axis angular velocity (rad/s) | -10 to 10 |
| gyro_y | float | Y-axis angular velocity (rad/s) | -10 to 10 |
| gyro_z | float | Z-axis angular velocity (rad/s) | -10 to 10 |

### StepDetectionResponse

| Field | Type | Description |
|-------|------|-------------|
| step_start | boolean | Whether a step start was detected |
| step_end | boolean | Whether a step end was detected |
| start_probability | float | Probability of step start (0-1) |
| end_probability | float | Probability of step end (0-1) |
| step_count | integer | Total steps detected in session |
| timestamp | string | ISO 8601 timestamp |

## Rate Limits

No rate limits are currently enforced. For production:
- Consider implementing rate limiting
- Monitor resource usage
- Set appropriate timeouts

## Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 503 | Model not loaded | Train and load a model |
| 500 | Processing error | Check input data format |
| 400 | Bad request | Validate request body |
| 403 | Forbidden (WebSocket) | Check endpoint URL |

### Error Response Format

```json
{
  "detail": "Error description",
  "status_code": 500
}
```

## Client Examples

### Python (requests)

```python
import requests

# REST API
response = requests.post(
    "http://localhost:8000/detect_step",
    json={
        "accel_x": 1.2,
        "accel_y": -0.5,
        "accel_z": 9.8,
        "gyro_x": 0.1,
        "gyro_y": 0.2,
        "gyro_z": -0.1
    }
)
print(response.json())
```

### Python (websockets)

```python
import asyncio
import json
import websockets

async def test_websocket():
    uri = "ws://localhost:8000/ws/realtime"
    async with websockets.connect(uri) as websocket:
        # Send data
        await websocket.send(json.dumps({
            "accel_x": 1.2,
            "accel_y": -0.5,
            "accel_z": 9.8,
            "gyro_x": 0.1,
            "gyro_y": 0.2,
            "gyro_z": -0.1
        }))
        
        # Receive response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(test_websocket())
```

### JavaScript (fetch)

```javascript
// REST API
fetch('http://localhost:8000/detect_step', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        accel_x: 1.2,
        accel_y: -0.5,
        accel_z: 9.8,
        gyro_x: 0.1,
        gyro_y: 0.2,
        gyro_z: -0.1
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

### JavaScript (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = function() {
    // Send data
    ws.send(JSON.stringify({
        accel_x: 1.2,
        accel_y: -0.5,
        accel_z: 9.8,
        gyro_x: 0.1,
        gyro_y: 0.2,
        gyro_z: -0.1
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## Performance Considerations

### REST API
- **Latency**: ~5-20ms per request
- **Throughput**: ~100-500 requests/second
- **Memory**: ~50MB base + model size

### WebSocket API
- **Latency**: ~1-5ms per message
- **Throughput**: ~1000+ messages/second
- **Memory**: Same as REST + connection overhead

## Interactive Documentation

When the API server is running, visit:
```
http://localhost:8000/docs
```

This provides an interactive Swagger UI for testing endpoints.
