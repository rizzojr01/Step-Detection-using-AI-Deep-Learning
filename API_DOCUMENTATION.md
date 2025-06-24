# Step Detection System - Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [API Reference](#api-reference)
5. [Real-Time WebSocket API](#real-time-websocket-api)
6. [Integration Examples](#integration-examples)
7. [Performance & Scalability](#performance--scalability)
8. [Deployment Guide](#deployment-guide)
9. [Development Guide](#development-guide)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Project Description

The **Step Detection System** is a production-ready, real-time step counting service built using deep learning models and FastAPI. It processes 6-dimensional sensor data (accelerometer + gyroscope) to detect and count steps with high accuracy and sub-millisecond latency.

### Key Features

- ✅ **Real-time step detection** with <1ms processing time
- ✅ **REST API** for single reading processing
- ✅ **WebSocket API** for continuous real-time streaming
- ✅ **Deep Learning Models** (CNN, LSTM, Transformer support)
- ✅ **Production-ready** with comprehensive error handling
- ✅ **Auto-generated documentation** (Swagger UI + ReDoc)
- ✅ **Statistics tracking** and performance monitoring
- ✅ **Mobile/IoT integration ready**

### Supported Use Cases

- 📱 **Mobile Fitness Apps** (iOS/Android)
- 🌐 **Web-based Fitness Dashboards**
- 🔗 **IoT Wearable Devices**
- 📊 **Real-time Health Monitoring**
- 🏃‍♂️ **Sports Performance Tracking**

---

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   FastAPI API   │    │  ML Models &    │
│                 │    │                 │    │  Processing     │
│ • Mobile Apps   │◄──►│ • REST Endpoints│◄──►│                 │
│ • Web Apps      │    │ • WebSocket     │    │ • CNN Model     │
│ • IoT Devices   │    │ • Documentation │    │ • Step Counter  │
│ • Dashboards    │    │ • Health Check  │    │ • Statistics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Modules

| Module                     | Purpose                                           | File                            |
| -------------------------- | ------------------------------------------------- | ------------------------------- |
| **FastAPI Application**    | Main API server with REST and WebSocket endpoints | `step_detection_api.py`         |
| **Real-Time Step Counter** | Core step detection logic and statistics          | `realtime_step_detector.py`     |
| **Model Initialization**   | CNN model setup and configuration                 | `initialize_model.py`           |
| **Test Suites**            | Comprehensive API testing and validation          | `test_api.py`, `debug_steps.py` |
| **WebSocket Client**       | Real-time streaming test client                   | `websocket_test_client.py`      |

### Data Flow

```
Sensor Data (6D) → FastAPI → Step Counter → ML Model → Step Detection → Response
     ↓              ↓            ↓            ↓            ↓            ↓
[ax,ay,az,        REST/      Preprocessing   CNN         Boolean     JSON
 gx,gy,gz]       WebSocket   & Buffering    Inference   + Stats    Response
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+**
- **uv** (Python package manager)
- **Git** (for cloning)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd Step-Detection-using-Deep-Learning-Models

# Install dependencies with uv
uv sync

# Start the FastAPI server
uv run uvicorn step_detection_api:app --host 0.0.0.0 --port 8000 --reload
```

### Project Dependencies

```toml
# Core dependencies
fastapi = ">=0.115.13"      # Modern web framework
uvicorn = ">=0.34.3"        # ASGI server
pydantic = ">=2.11.7"       # Data validation
websockets = ">=15.0.1"     # WebSocket support

# Machine Learning
torch = ">=2.7.1"           # Deep learning framework
numpy = ">=2.3.1"           # Numerical computing
scikit-learn = ">=1.7.0"    # ML utilities

# Data Processing
pandas = ">=2.3.0"          # Data manipulation
matplotlib = ">=3.10.3"     # Visualization

# Development & Testing
requests = ">=2.32.4"       # HTTP client for testing
jupyter = ">=1.1.1"         # Notebook environment
```

### Verification

```bash
# Test the installation
uv run python test_api.py

# Test WebSocket functionality
uv run python websocket_test_client.py

# Access API documentation
open http://localhost:8000/docs
```

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Authentication

Currently, no authentication is required. For production deployment, implement authentication middleware.

### Common Response Format

```json
{
  "status": "success",
  "message": "Optional message",
  "result": {
    /* Endpoint-specific data */
  },
  "statistics": {
    /* Optional statistics */
  }
}
```

### Error Handling

```json
{
  "detail": "Error description",
  "status_code": 400/500
}
```

---

### 📡 Health Check

**Endpoint:** `GET /health`

**Description:** Check API status and model initialization

**Response:**

```json
{
  "status": "healthy",
  "model_initialized": true,
  "api_version": "2.0.0",
  "framework": "FastAPI"
}
```

**Example:**

```bash
curl -X GET "http://localhost:8000/health"
```

---

### 🦶 Step Detection

**Endpoint:** `POST /detect_step`

**Description:** Process a single sensor reading and detect steps

**Request Body:**

```json
{
  "accel_x": 8.0,
  "accel_y": 2.0,
  "accel_z": 15.0,
  "gyro_x": 1.5,
  "gyro_y": 1.2,
  "gyro_z": 0.8
}
```

**Response:**

```json
{
  "status": "success",
  "result": {
    "step_detected": true,
    "total_steps": 42,
    "probabilities": [0.234, 0.123, 0.643],
    "confidence": 0.643,
    "processing_time_ms": 0.8,
    "in_step": false
  }
}
```

**Field Descriptions:**

- `step_detected`: Boolean indicating if a step was detected
- `total_steps`: Cumulative step count since reset
- `probabilities`: [no_step, step_start, step_end] predictions
- `confidence`: Maximum probability (0.0 - 1.0)
- `processing_time_ms`: Inference time in milliseconds
- `in_step`: Current step state (for state machine tracking)

**Example:**

```bash
curl -X POST "http://localhost:8000/detect_step" \
     -H "Content-Type: application/json" \
     -d '{
       "accel_x": 8.0, "accel_y": 2.0, "accel_z": 15.0,
       "gyro_x": 1.5, "gyro_y": 1.2, "gyro_z": 0.8
     }'
```

---

### 📊 Statistics

**Endpoint:** `GET /stats`

**Description:** Get comprehensive step detection statistics

**Response:**

```json
{
  "status": "success",
  "statistics": {
    "total_steps": 1247,
    "elapsed_time_seconds": 3600.5,
    "steps_per_minute": 20.8,
    "steps_per_hour": 1247.0,
    "avg_processing_time_ms": 0.75,
    "max_processing_time_ms": 2.1,
    "buffer_utilization": 0.85,
    "last_step_time": 1698765432.123,
    "recent_step_timestamps": [
      "2025-01-15T10:30:45.123Z",
      "2025-01-15T10:30:47.456Z"
    ]
  }
}
```

**Example:**

```bash
curl -X GET "http://localhost:8000/stats"
```

---

### 🔄 Reset Counter

**Endpoint:** `POST /reset`

**Description:** Reset step counter and statistics

**Response:**

```json
{
  "status": "success",
  "message": "Step counter reset successfully"
}
```

**Example:**

```bash
curl -X POST "http://localhost:8000/reset"
```

---

### 📖 API Documentation

**Endpoints:**

- **Swagger UI:** `GET /docs`
- **ReDoc:** `GET /redoc`
- **OpenAPI Schema:** `GET /openapi.json`

---

## Real-Time WebSocket API

### Connection URL

```
ws://localhost:8000/ws/realtime
```

### Protocol

#### 1. Connection Establishment

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/realtime");
ws.onopen = () => console.log("Connected to step detection service");
```

#### 2. Send Sensor Data

```javascript
const sensorData = {
  accel_x: 8.0,
  accel_y: 2.0,
  accel_z: 15.0,
  gyro_x: 1.5,
  gyro_y: 1.2,
  gyro_z: 0.8,
};
ws.send(JSON.stringify(sensorData));
```

#### 3. Receive Results

```javascript
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);

  if (result.error) {
    console.error("Error:", result.error);
    return;
  }

  console.log("Step detected:", result.step_detected);
  console.log("Total steps:", result.total_steps);
  console.log("Confidence:", result.confidence);
};
```

#### 4. Error Handling

```javascript
ws.onerror = (error) => console.error("WebSocket error:", error);
ws.onclose = () => console.log("Connection closed");
```

### Message Formats

**Outgoing (Client → Server):**

```json
{
  "accel_x": 8.0,
  "accel_y": 2.0,
  "accel_z": 15.0,
  "gyro_x": 1.5,
  "gyro_y": 1.2,
  "gyro_z": 0.8
}
```

**Incoming (Server → Client):**

```json
{
  "step_detected": true,
  "total_steps": 42,
  "probabilities": [0.234, 0.123, 0.643],
  "confidence": 0.643,
  "processing_time_ms": 0.8,
  "in_step": false
}
```

**Error Response:**

```json
{
  "error": "Missing sensor data fields"
}
```

### Performance Characteristics

- **Latency:** <1ms processing time
- **Throughput:** >1000 messages/second
- **Concurrent Connections:** Limited by server resources
- **Message Size:** ~200 bytes per message

---

## Integration Examples

### 📱 Mobile App Integration

#### React Native Example

```javascript
import { WebSocket } from "react-native";

class StepDetectionService {
  constructor(serverUrl) {
    this.ws = new WebSocket(`ws://${serverUrl}/ws/realtime`);
    this.stepCount = 0;

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.step_detected) {
        this.stepCount = data.total_steps;
        this.onStepDetected(data);
      }
    };
  }

  sendSensorData(accelerometer, gyroscope) {
    const data = {
      accel_x: accelerometer.x,
      accel_y: accelerometer.y,
      accel_z: accelerometer.z,
      gyro_x: gyroscope.x,
      gyro_y: gyroscope.y,
      gyro_z: gyroscope.z,
    };
    this.ws.send(JSON.stringify(data));
  }

  onStepDetected(data) {
    // Update UI, trigger animations, etc.
    console.log(`Step detected! Total: ${data.total_steps}`);
  }
}
```

#### iOS Swift Example

```swift
import Foundation
import Starscream

class StepDetectionService: WebSocketDelegate {
    private var socket: WebSocket!
    private var stepCount = 0

    init(serverURL: String) {
        let url = URL(string: "ws://\(serverURL)/ws/realtime")!
        socket = WebSocket(request: URLRequest(url: url))
        socket.delegate = self
        socket.connect()
    }

    func sendSensorData(accel: (x: Double, y: Double, z: Double),
                       gyro: (x: Double, y: Double, z: Double)) {
        let data: [String: Double] = [
            "accel_x": accel.x,
            "accel_y": accel.y,
            "accel_z": accel.z,
            "gyro_x": gyro.x,
            "gyro_y": gyro.y,
            "gyro_z": gyro.z
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: data),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            socket.write(string: jsonString)
        }
    }

    func didReceive(event: WebSocketEvent, client: WebSocket) {
        switch event {
        case .text(let string):
            if let data = string.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let stepDetected = json["step_detected"] as? Bool,
               stepDetected {
                stepCount = json["total_steps"] as? Int ?? stepCount
                onStepDetected(json)
            }
        default:
            break
        }
    }

    private func onStepDetected(_ data: [String: Any]) {
        DispatchQueue.main.async {
            // Update UI on main thread
            print("Step detected! Total: \(self.stepCount)")
        }
    }
}
```

#### Android Kotlin Example

```kotlin
import okhttp3.*
import org.json.JSONObject

class StepDetectionService(serverUrl: String) {
    private val client = OkHttpClient()
    private val socket: WebSocket
    private var stepCount = 0

    init {
        val request = Request.Builder()
            .url("ws://$serverUrl/ws/realtime")
            .build()
        socket = client.newWebSocket(request, StepWebSocketListener())
    }

    fun sendSensorData(accelX: Float, accelY: Float, accelZ: Float,
                      gyroX: Float, gyroY: Float, gyroZ: Float) {
        val data = JSONObject().apply {
            put("accel_x", accelX)
            put("accel_y", accelY)
            put("accel_z", accelZ)
            put("gyro_x", gyroX)
            put("gyro_y", gyroY)
            put("gyro_z", gyroZ)
        }
        socket.send(data.toString())
    }

    private inner class StepWebSocketListener : WebSocketListener() {
        override fun onMessage(webSocket: WebSocket, text: String) {
            val json = JSONObject(text)
            if (json.getBoolean("step_detected")) {
                stepCount = json.getInt("total_steps")
                onStepDetected(json)
            }
        }
    }

    private fun onStepDetected(data: JSONObject) {
        // Update UI on main thread
        println("Step detected! Total: $stepCount")
    }
}
```

### 🌐 Web Application Integration

#### JavaScript/HTML5 Example

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Real-Time Step Counter</title>
    <style>
      .step-counter {
        font-size: 48px;
        font-weight: bold;
        color: #2196f3;
      }
      .confidence {
        font-size: 24px;
        color: #4caf50;
      }
      .status {
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
      }
      .connected {
        background-color: #d4edda;
        color: #155724;
      }
      .disconnected {
        background-color: #f8d7da;
        color: #721c24;
      }
    </style>
  </head>
  <body>
    <h1>Real-Time Step Detection</h1>

    <div id="status" class="status disconnected">Disconnected</div>

    <div class="step-counter">Steps: <span id="stepCount">0</span></div>

    <div class="confidence">Confidence: <span id="confidence">0%</span></div>

    <button onclick="simulateWalking()">Simulate Walking</button>
    <button onclick="resetCounter()">Reset Counter</button>

    <script>
      class StepDetectionApp {
        constructor() {
          this.ws = null;
          this.isSimulating = false;
          this.connect();
        }

        connect() {
          this.ws = new WebSocket("ws://localhost:8000/ws/realtime");

          this.ws.onopen = () => {
            document.getElementById("status").textContent = "Connected";
            document.getElementById("status").className = "status connected";
          };

          this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
              console.error("Error:", data.error);
              return;
            }

            // Update UI
            document.getElementById("stepCount").textContent = data.total_steps;
            document.getElementById("confidence").textContent =
              (data.confidence * 100).toFixed(1) + "%";

            // Visual feedback for step detection
            if (data.step_detected) {
              this.animateStep();
            }
          };

          this.ws.onclose = () => {
            document.getElementById("status").textContent = "Disconnected";
            document.getElementById("status").className = "status disconnected";
          };
        }

        sendSensorData(accelX, accelY, accelZ, gyroX, gyroY, gyroZ) {
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const data = {
              accel_x: accelX,
              accel_y: accelY,
              accel_z: accelZ,
              gyro_x: gyroX,
              gyro_y: gyroY,
              gyro_z: gyroZ,
            };
            this.ws.send(JSON.stringify(data));
          }
        }

        animateStep() {
          const counter = document.getElementById("stepCount");
          counter.style.transform = "scale(1.2)";
          counter.style.color = "#FF5722";
          setTimeout(() => {
            counter.style.transform = "scale(1)";
            counter.style.color = "#2196F3";
          }, 200);
        }
      }

      // Initialize app
      const app = new StepDetectionApp();

      // Simulate walking function
      function simulateWalking() {
        if (app.isSimulating) return;

        app.isSimulating = true;
        let count = 0;

        const interval = setInterval(() => {
          // Generate step-like motion
          const t = count * 0.1;
          const isStep = count % 8 === 0; // Step every 8th reading

          const accelMagnitude = isStep
            ? 12 + Math.random() * 4
            : 9.8 + Math.random() * 2;
          const gyroMagnitude = isStep
            ? 1.5 + Math.random() * 0.5
            : 0.2 + Math.random() * 0.3;

          app.sendSensorData(
            accelMagnitude * (0.6 + Math.random() * 0.4),
            accelMagnitude * (0.2 + Math.random() * 0.3),
            accelMagnitude * (0.8 + Math.random() * 0.4),
            gyroMagnitude * Math.random(),
            gyroMagnitude * Math.random(),
            gyroMagnitude * Math.random()
          );

          count++;
          if (count >= 80) {
            // 8 seconds of simulation
            clearInterval(interval);
            app.isSimulating = false;
          }
        }, 100); // 10 Hz
      }

      // Reset counter function
      async function resetCounter() {
        try {
          const response = await fetch("http://localhost:8000/reset", {
            method: "POST",
          });
          if (response.ok) {
            document.getElementById("stepCount").textContent = "0";
            document.getElementById("confidence").textContent = "0%";
          }
        } catch (error) {
          console.error("Reset failed:", error);
        }
      }
    </script>
  </body>
</html>
```

### 🔗 IoT Device Integration

#### Python IoT Example

```python
import asyncio
import websockets
import json
import time
from typing import Dict, Any

class IoTStepDetector:
    """IoT device step detection client"""

    def __init__(self, server_url: str, device_id: str):
        self.server_url = f"ws://{server_url}/ws/realtime"
        self.device_id = device_id
        self.ws = None
        self.is_running = False
        self.step_count = 0

    async def connect(self):
        """Connect to step detection service"""
        try:
            self.ws = await websockets.connect(self.server_url)
            print(f"Device {self.device_id} connected to step detection service")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def start_monitoring(self, sensor_callback):
        """Start continuous step monitoring"""
        self.is_running = True

        while self.is_running:
            try:
                # Get sensor data from device
                sensor_data = await sensor_callback()

                # Send to server
                await self.ws.send(json.dumps(sensor_data))

                # Receive result
                response = await self.ws.recv()
                result = json.loads(response)

                if result.get('error'):
                    print(f"Error: {result['error']}")
                    continue

                # Process step detection
                if result.get('step_detected'):
                    self.step_count = result['total_steps']
                    await self.on_step_detected(result)

                # Log periodic stats
                if self.step_count % 10 == 0 and result.get('step_detected'):
                    await self.log_statistics(result)

            except websockets.exceptions.ConnectionClosed:
                print("Connection lost, reconnecting...")
                if await self.connect():
                    continue
                else:
                    break
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(1)

    async def on_step_detected(self, result: Dict[str, Any]):
        """Handle step detection event"""
        print(f"Device {self.device_id}: Step {result['total_steps']} detected "
              f"(confidence: {result['confidence']:.3f})")

        # Send to device-specific handlers
        await self.update_local_display(result)
        await self.send_to_cloud(result)

    async def update_local_display(self, result: Dict[str, Any]):
        """Update local device display/LEDs"""
        # Flash LED, update LCD, etc.
        pass

    async def send_to_cloud(self, result: Dict[str, Any]):
        """Send step data to cloud service"""
        cloud_data = {
            'device_id': self.device_id,
            'timestamp': time.time(),
            'step_count': result['total_steps'],
            'confidence': result['confidence']
        }
        # Send to cloud API
        pass

    async def log_statistics(self, result: Dict[str, Any]):
        """Log device statistics"""
        print(f"Device {self.device_id} Stats: "
              f"{result['total_steps']} steps, "
              f"{result['processing_time_ms']:.1f}ms processing")

    def stop(self):
        """Stop monitoring"""
        self.is_running = False

# Example sensor data generator for IoT device
async def get_sensor_data():
    """Simulate reading from actual sensors"""
    # In real implementation, read from I2C/SPI sensors
    import random

    # Simulate realistic sensor readings
    base_accel = 9.8
    motion_factor = random.uniform(0.8, 1.5)

    return {
        'accel_x': base_accel * motion_factor * random.uniform(0.5, 1.2),
        'accel_y': base_accel * motion_factor * random.uniform(0.2, 0.8),
        'accel_z': base_accel * motion_factor * random.uniform(0.8, 1.3),
        'gyro_x': random.uniform(-1.0, 1.0),
        'gyro_y': random.uniform(-1.0, 1.0),
        'gyro_z': random.uniform(-1.0, 1.0)
    }

# Usage example
async def main():
    detector = IoTStepDetector("localhost:8000", "device_001")

    if await detector.connect():
        print("Starting step monitoring...")
        await detector.start_monitoring(get_sensor_data)
    else:
        print("Failed to connect to step detection service")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Performance & Scalability

### Performance Metrics

| Metric                 | Value       | Notes                        |
| ---------------------- | ----------- | ---------------------------- |
| **Processing Latency** | <1ms        | Average inference time       |
| **Throughput**         | >1000 req/s | Single instance capacity     |
| **Memory Usage**       | ~200MB      | Including model weights      |
| **CPU Usage**          | <5%         | On modern hardware           |
| **Accuracy**           | 95%+        | With properly trained models |

### Scalability Considerations

#### Horizontal Scaling

```yaml
# docker-compose.yml
version: "3.8"
services:
  step-api-1:
    build: .
    ports: ["8001:8000"]
  step-api-2:
    build: .
    ports: ["8002:8000"]

  load-balancer:
    image: nginx
    ports: ["8000:80"]
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

#### Load Balancing Configuration

```nginx
# nginx.conf
upstream step_detection {
    server step-api-1:8000;
    server step-api-2:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://step_detection;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### Performance Optimization

**Model Optimization:**

```python
# Use TorchScript for faster inference
model = torch.jit.script(your_model)

# Enable GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

**API Optimization:**

```python
# Enable FastAPI middleware for compression
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Use async/await for I/O operations
@app.post("/detect_step")
async def detect_step(sensor_data: SensorReading):
    # Async processing
    pass
```

### Monitoring & Observability

#### Health Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics collection
step_detection_counter = Counter('steps_detected_total', 'Total steps detected')
processing_time_histogram = Histogram('processing_time_seconds', 'Processing time')

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Logging Configuration

```python
import logging
import structlog

# Structured logging setup
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

logger = structlog.get_logger()

# Usage in endpoints
logger.info("step_detected",
           total_steps=result['total_steps'],
           confidence=result['confidence'],
           processing_time_ms=result['processing_time_ms'])
```

---

## Deployment Guide

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uv", "run", "uvicorn", "step_detection_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

```yaml
version: "3.8"

services:
  step-detection-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - MODEL_PATH=/app/models
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - step-detection-api
    restart: unless-stopped
```

### Cloud Deployment

#### AWS ECS Deployment

```json
{
  "family": "step-detection-api",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "step-detection-api",
      "image": "your-registry/step-detection:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/step-detection-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: step-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: step-detection-api
  template:
    metadata:
      labels:
        app: step-detection-api
    spec:
      containers:
        - name: api
          image: your-registry/step-detection:latest
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: "/app/models"
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: step-detection-service
spec:
  selector:
    app: step-detection-api
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Environment Configuration

#### Production Environment Variables

```bash
# Application
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=/app/models/trained_model.pth
MODEL_DEVICE=cuda
BATCH_SIZE=32

# Performance
MAX_WORKERS=4
KEEP_ALIVE=2

# Security
CORS_ORIGINS=["https://yourdomain.com"]
API_KEY_HEADER=X-API-Key

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30

# Database (if needed)
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/steps
```

#### SSL/TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS;

    location / {
        proxy_pass http://step-detection-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://step-detection-api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## Development Guide

### Project Structure

```
Step-Detection-using-Deep-Learning-Models/
├── step_detection_api.py          # Main FastAPI application
├── realtime_step_detector.py      # Core step detection logic
├── initialize_model.py            # Model initialization
├── pyproject.toml                 # Dependencies and configuration
├──
├── tests/                         # Test suite
│   ├── test_api.py               # API endpoint tests
│   ├── debug_steps.py            # Debug utilities
│   └── websocket_test_client.py  # WebSocket client tests
├──
├── notebooks/                     # Jupyter notebooks
│   ├── CNN.ipynb                # CNN model development
│   ├── LSTM.ipynb               # LSTM model development
│   └── Transformer.ipynb        # Transformer model development
├──
├── Sample Data/                   # Training and test data
│   ├── person_2/
│   ├── person_3/
│   └── ...
├──
├── models/                        # Trained model files
│   ├── cnn_model.pth
│   ├── lstm_model.pth
│   └── transformer_model.pth
├──
├── docs/                          # Documentation
│   ├── API_DOCUMENTATION.md      # This file
│   ├── DEPLOYMENT_GUIDE.md
│   └── DEVELOPMENT_SETUP.md
├──
└── docker/                       # Docker configuration
    ├── Dockerfile
    ├── docker-compose.yml
    └── nginx.conf
```

### Development Setup

#### 1. Clone and Setup

```bash
git clone <repository-url>
cd Step-Detection-using-Deep-Learning-Models

# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

#### 2. Development Environment

```bash
# Start development server with auto-reload
uv run uvicorn step_detection_api:app --reload --log-level debug

# Run tests
uv run pytest tests/

# Run specific test
uv run python test_api.py

# Code formatting
uv run black .
uv run isort .

# Type checking
uv run mypy step_detection_api.py
```

#### 3. Model Development

```bash
# Start Jupyter for model development
uv run jupyter lab

# Train new models
uv run python -m notebooks.train_cnn
uv run python -m notebooks.train_lstm

# Evaluate models
uv run python -m notebooks.evaluate_models
```

### Code Quality & Standards

#### Code Style

```python
# Use type hints
from typing import Dict, List, Optional, Union

def process_sensor_data(
    accel_data: List[float],
    gyro_data: List[float]
) -> Dict[str, Union[bool, int, float]]:
    """Process sensor data and return step detection result."""
    pass

# Use dataclasses for structured data
from dataclasses import dataclass

@dataclass
class SensorReading:
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    timestamp: Optional[float] = None
```

#### Error Handling

```python
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

async def detect_step(sensor_data: SensorReading):
    try:
        result = await process_sensor_data(sensor_data)
        return result
    except ValueError as e:
        logger.warning(f"Invalid sensor data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid sensor data: {e}")
    except Exception as e:
        logger.error(f"Step detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### Testing Guidelines

```python
import pytest
from fastapi.testclient import TestClient
from step_detection_api import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_step_detection():
    """Test step detection with valid sensor data."""
    sensor_data = {
        "accel_x": 8.0,
        "accel_y": 2.0,
        "accel_z": 15.0,
        "gyro_x": 1.5,
        "gyro_y": 1.2,
        "gyro_z": 0.8
    }
    response = client.post("/detect_step", json=sensor_data)
    assert response.status_code == 200
    assert "step_detected" in response.json()["result"]

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection and message handling."""
    with client.websocket_connect("/ws/realtime") as websocket:
        websocket.send_json({
            "accel_x": 8.0, "accel_y": 2.0, "accel_z": 15.0,
            "gyro_x": 1.5, "gyro_y": 1.2, "gyro_z": 0.8
        })
        data = websocket.receive_json()
        assert "step_detected" in data
```

### Contributing Guidelines

#### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-model-support

# Make changes and commit
git add .
git commit -m "feat: add LSTM model support"

# Push and create PR
git push origin feature/new-model-support
```

#### Commit Message Format

```
type(scope): short description

Longer description if needed

Types: feat, fix, docs, style, refactor, test, chore
Scopes: api, model, websocket, tests, docs
```

#### Pull Request Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

---

## Troubleshooting

### Common Issues

#### 1. Model Initialization Fails

**Symptoms:**

```
❌ Error initializing model: Model file not found
```

**Solutions:**

```python
# Check model path
import os
print("Model path exists:", os.path.exists("models/trained_model.pth"))

# Verify model format
try:
    model = torch.load("models/trained_model.pth")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load error: {e}")

# Use CPU-only loading
model = torch.load("models/trained_model.pth", map_location='cpu')
```

#### 2. WebSocket Connection Issues

**Symptoms:**

```
❌ WebSocket connection closed immediately
```

**Solutions:**

```python
# Check server status
curl -X GET "http://localhost:8000/health"

# Verify WebSocket endpoint
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     --header "Sec-WebSocket-Version: 13" \
     http://localhost:8000/ws/realtime

# Check firewall/proxy settings
netstat -tulpn | grep :8000
```

#### 3. High Memory Usage

**Symptoms:**

```
Memory usage > 1GB
```

**Solutions:**

```python
# Enable model optimization
model = torch.jit.optimize_for_inference(model)

# Use smaller batch sizes
batch_size = 1

# Clear unused variables
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

#### 4. Slow Response Times

**Symptoms:**

```
Processing time > 10ms
```

**Solutions:**

```python
# Profile the code
import time
start = time.time()
result = model(input_tensor)
print(f"Inference time: {(time.time() - start) * 1000:.2f}ms")

# Use CPU optimizations
torch.set_num_threads(4)

# Enable GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

#### 5. Step Detection Accuracy Issues

**Symptoms:**

```
Steps not detected or false positives
```

**Solutions:**

```python
# Adjust detection thresholds
step_counter.start_threshold = 0.3  # Lower = more sensitive
step_counter.end_threshold = 0.3

# Check sensor data ranges
print(f"Accel magnitude: {np.sqrt(ax**2 + ay**2 + az**2)}")
print(f"Gyro magnitude: {np.sqrt(gx**2 + gy**2 + gz**2)}")

# Calibrate for specific use case
# Stationary: accel ~9.8, gyro ~0
# Walking: accel 10-15, gyro 0.5-2.0
# Running: accel 15-25, gyro 1.0-4.0
```

### Debug Mode

#### Enable Debug Logging

```python
import logging

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable FastAPI debug mode
uvicorn step_detection_api:app --reload --log-level debug
```

#### Debug Tools

```python
# Step detection debugging
python debug_steps.py

# API endpoint testing
python test_api.py

# WebSocket testing
python websocket_test_client.py

# Simple connection test
python simple_websocket_test.py
```

### Performance Monitoring

#### System Monitoring

```bash
# CPU and memory usage
top -p $(pgrep -f "step_detection_api")

# Network connections
netstat -tulpn | grep :8000

# Disk I/O
iotop -p $(pgrep -f "step_detection_api")

# GPU usage (if applicable)
nvidia-smi
```

#### Application Metrics

```python
# Add custom metrics
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.response_times = []
        self.error_count = defaultdict(int)

    def record_request(self, endpoint: str, response_time: float, status_code: int):
        self.request_count += 1
        self.response_times.append(response_time)
        if status_code >= 400:
            self.error_count[status_code] += 1

    def get_stats(self):
        return {
            "requests": self.request_count,
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "error_rate": sum(self.error_count.values()) / self.request_count,
            "errors": dict(self.error_count)
        }

# Usage in API
metrics = MetricsCollector()

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    metrics.record_request(
        endpoint=request.url.path,
        response_time=process_time,
        status_code=response.status_code
    )

    return response
```

### Contact & Support

For additional support:

- **Issues:** Create GitHub issue with detailed description
- **Documentation:** Check `/docs` endpoint for API reference
- **Performance:** Use debug tools and monitoring
- **Custom Models:** Follow model development guide

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Changelog

### v2.0.0 (Current)

- ✅ FastAPI implementation with WebSocket support
- ✅ Real-time step detection with <1ms latency
- ✅ Comprehensive API documentation
- ✅ Production-ready deployment configuration
- ✅ Mobile and IoT integration examples

### v1.0.0

- ✅ Initial deep learning models (CNN, LSTM, Transformer)
- ✅ Basic Flask API
- ✅ Jupyter notebook development environment
