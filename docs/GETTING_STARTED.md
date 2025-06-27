# 🚀 Getting Started Guide

This guide will help you get up and running with the Step Detection system in under 10 minutes.

## 📋 Prerequisites

- 🐍 Python 3.11 or higher
- 💾 At least 2GB available disk space
- 🧠 4GB RAM minimum (8GB recommended)
- 📡 Internet connection for initial setup

## ⚡ Quick Installation

### Option 1: One-Command Setup (Recommended)

```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/your-repo/setup.sh | bash
```

### Option 2: Manual Setup

```bash
# Clone repository
git clone <your-repository-url>
cd Step-Detection-using-AI-Deep-Learning

# Setup environment (choose one)
# Using UV (fastest)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && uv shell

# Using pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🎯 First Steps

### 1. 📊 Verify Installation

```bash
# Test package import
python -c "from src.step_detection import StepDetector; print('✅ Installation successful!')"

# Run health check
python main.py --health-check
```

### 2. 🤖 Download Pre-trained Model

```bash
# Download model (if not included)
python scripts/download_model.py

# Verify model
python -c "
from src.step_detection.core.detector import StepDetector
detector = StepDetector('models/step_detection_model.keras')
print('✅ Model loaded successfully!')
"
```

### 3. 🧪 Test Real-time Detection

```bash
# Quick test with sample data
python tests/test_quick_detection.py

# Interactive CLI test
python main.py
# Choose option 2: "Test real-time detection"
```

## 🎮 Interactive Tutorial

### Tutorial 1: Basic Step Detection

```python
# tutorial_1_basic.py
from src.step_detection.core.detector import StepDetector

# Initialize detector
detector = StepDetector("models/step_detection_model.keras")

# Simulate walking data (3 steps)
walking_data = [
    # Step 1: Start
    {"accel_x": 1.2, "accel_y": -0.3, "accel_z": 9.8, 
     "gyro_x": 0.1, "gyro_y": 0.0, "gyro_z": -0.1},
    
    # Step 1: End  
    {"accel_x": -0.8, "accel_y": 0.2, "accel_z": 9.9,
     "gyro_x": -0.1, "gyro_y": 0.1, "gyro_z": 0.0},
    
    # Step 2: Start
    {"accel_x": 1.1, "accel_y": -0.4, "accel_z": 9.7,
     "gyro_x": 0.2, "gyro_y": -0.1, "gyro_z": -0.2},
]

print("🚶‍♂️ Simulating walking...")
for i, data in enumerate(walking_data):
    result = detector.process_reading(**data)
    print(f"Reading {i+1}: Steps = {result['step_count']}, "
          f"Detected = {result['step_detected']}")

print(f"\n✅ Final step count: {detector.step_count}")
```

### Tutorial 2: API Usage

```python
# tutorial_2_api.py
import requests
import json

# Start API server first: python main.py -> option 3

# Test step detection endpoint
url = "http://localhost:8000/detect_step"
data = {
    "accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8,
    "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1
}

response = requests.post(url, json=data)
result = response.json()
print(f"🌐 API Response: {json.dumps(result, indent=2)}")

# Get step count
count_response = requests.get("http://localhost:8000/step_count")
print(f"📊 Step Count: {count_response.json()}")
```

### Tutorial 3: WebSocket Real-time

```python
# tutorial_3_websocket.py
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/realtime"
    
    async with websockets.connect(uri) as websocket:
        print("🔌 Connected to WebSocket")
        
        # Send sensor data
        data = {
            "accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8,
            "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1
        }
        
        await websocket.send(json.dumps(data))
        
        # Receive response
        response = await websocket.recv()
        result = json.loads(response)
        print(f"⚡ Real-time result: {result}")

# Run: python tutorial_3_websocket.py
asyncio.run(test_websocket())
```

## 🛠️ Common Setup Issues

### Issue: Import Error

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Model Not Found

**Problem**: `FileNotFoundError: models/step_detection_model.keras`

**Solution**:
```bash
# Download model
python scripts/download_model.py

# Or train a new model
python main.py  # Choose option 1
```

### Issue: TensorFlow Installation

**Problem**: TensorFlow compatibility issues

**Solution**:
```bash
# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal

# For CUDA support
pip install tensorflow[and-cuda]

# CPU-only version
pip install tensorflow-cpu
```

### Issue: Port Already in Use

**Problem**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
uvicorn src.step_detection.api.api:app --port 8001
```

## 🎯 Next Steps

Now that you have the system running:

1. 📚 **Read the Documentation**:
   - [API Reference](API.md) - Detailed API docs
   - [Training Guide](TRAINING.md) - Train custom models
   - [Architecture](ARCHITECTURE.md) - System design

2. 🧪 **Explore Examples**:
   - Check `notebooks/` for interactive examples
   - Run `tests/` to see testing patterns
   - Look at `scripts/` for utility tools

3. 🚀 **Build Your Application**:
   - Integrate with your mobile app
   - Create a web dashboard
   - Set up monitoring and alerts

4. 🤝 **Join the Community**:
   - Star the repository ⭐
   - Report issues 🐛
   - Contribute improvements 💡

## 📞 Support

If you encounter any issues:

1. 📖 Check this guide and other documentation
2. 🔍 Search existing [GitHub Issues](../../issues)
3. 💬 Join [GitHub Discussions](../../discussions)
4. 📧 Contact support: [your-email@domain.com]

---

**✅ You're all set! Happy step detecting! 🚶‍♂️🚶‍♀️**
