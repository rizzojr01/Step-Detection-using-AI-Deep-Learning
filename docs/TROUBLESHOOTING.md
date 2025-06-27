# 🔍 Troubleshooting Guide

Complete troubleshooting guide for the Step Detection system.

## 🎯 Common Issues & Solutions

This guide covers solutions for the most common issues you might encounter.

## 📦 Installation Issues

### Issue: Package Installation Fails

**Symptoms:**
```bash
ERROR: Could not find a version that satisfies the requirement tensorflow>=2.19.0
```

**Solutions:**

1. **Update Python and pip:**
```bash
# Update pip
python -m pip install --upgrade pip

# Check Python version (requires 3.11+)
python --version
```

2. **Platform-specific TensorFlow:**
```bash
# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal

# For systems with CUDA support
pip install tensorflow[and-cuda]

# For CPU-only
pip install tensorflow-cpu
```

3. **Use conda for complex dependencies:**
```bash
conda create -n step-detection python=3.11
conda activate step-detection
conda install tensorflow
pip install -r requirements.txt
```

### Issue: Virtual Environment Problems

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'src'
```

**Solutions:**

1. **Install in development mode:**
```bash
pip install -e .
```

2. **Add to Python path:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# On Windows: set PYTHONPATH=%PYTHONPATH%;%cd%
```

3. **Check virtual environment:**
```bash
# Verify you're in the right environment
which python
pip list
```

## 🤖 Model Issues

### Issue: Model File Not Found

**Symptoms:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'models/step_detection_model.keras'
```

**Solutions:**

1. **Check model directory:**
```bash
ls -la models/
```

2. **Download pre-trained model:**
```bash
python scripts/download_model.py
```

3. **Train a new model:**
```bash
python main.py
# Choose option 1: "Train a new model"
```

4. **Use absolute path:**
```python
import os
model_path = os.path.abspath("models/step_detection_model.keras")
detector = StepDetector(model_path)
```

### Issue: Model Loading Errors

**Symptoms:**
```python
ValueError: Unable to load model. Model format not recognized.
```

**Solutions:**

1. **Check TensorFlow version:**
```python
import tensorflow as tf
print(tf.__version__)
```

2. **Convert model format:**
```python
# Convert SavedModel to Keras
model = tf.saved_model.load("models/saved_model")
model.save("models/step_detection_model.keras")
```

3. **Rebuild model:**
```python
from src.step_detection.models.model_utils import create_cnn_model
model = create_cnn_model()
model.save("models/step_detection_model.keras")
```

### Issue: Model Performance Degradation

**Symptoms:**
- Low accuracy in step detection
- Many false positives/negatives

**Diagnostic Steps:**

1. **Check model metrics:**
```python
from src.step_detection.core.detector import StepDetector
detector = StepDetector("models/step_detection_model.keras")

# Test with known data
test_result = detector.process_reading(1.2, -0.5, 9.8, 0.1, 0.2, -0.1)
print(f"Test result: {test_result}")
```

2. **Optimize thresholds:**
```bash
python main.py
# Choose option 4: "Optimize thresholds"
```

3. **Retrain with more data:**
```python
# Add more training data to data/raw/
python main.py  # Option 1: Train model
```

## 🌐 API Issues

### Issue: API Server Won't Start

**Symptoms:**
```bash
OSError: [Errno 48] Address already in use
```

**Solutions:**

1. **Check port usage:**
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

2. **Use different port:**
```bash
uvicorn src.step_detection.api.api:app --port 8001
```

3. **Find available port:**
```python
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

port = find_free_port()
print(f"Use port: {port}")
```

### Issue: API Endpoints Not Responding

**Symptoms:**
- 404 errors on valid endpoints
- Connection refused errors

**Diagnostic Steps:**

1. **Check server status:**
```bash
curl http://localhost:8000/health
```

2. **Verify API documentation:**
```bash
# Open in browser
open http://localhost:8000/docs
```

3. **Check server logs:**
```bash
# Run with verbose logging
uvicorn src.step_detection.api.api:app --log-level debug
```

4. **Test with simple request:**
```python
import requests

response = requests.get("http://localhost:8000/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

### Issue: WebSocket Connection Fails

**Symptoms:**
```javascript
WebSocket connection to 'ws://localhost:8000/ws/realtime' failed
```

**Solutions:**

1. **Check WebSocket endpoint:**
```python
import asyncio
import websockets

async def test_websocket():
    try:
        async with websockets.connect('ws://localhost:8000/ws/realtime') as ws:
            print("✅ WebSocket connected")
            await ws.send('{"test": true}')
            response = await ws.recv()
            print(f"Response: {response}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

asyncio.run(test_websocket())
```

2. **Verify server supports WebSockets:**
```bash
pip install websockets
python test_websocket.py
```

3. **Check firewall/proxy settings:**
```bash
# Test direct connection
telnet localhost 8000
```

## 📊 Data Issues

### Issue: Data Loading Fails

**Symptoms:**
```python
FileNotFoundError: No data files found in data/raw/
```

**Solutions:**

1. **Check data directory structure:**
```bash
ls -la data/raw/
```

2. **Download sample data:**
```bash
python scripts/download_sample_data.py
```

3. **Verify data format:**
```python
import pandas as pd

# Check CSV format
df = pd.read_csv("data/raw/sample_data.csv")
print(df.head())
print(f"Columns: {df.columns.tolist()}")
```

4. **Create sample data:**
```python
import numpy as np
import pandas as pd

# Generate sample sensor data
n_samples = 1000
data = {
    'accel_x': np.random.randn(n_samples),
    'accel_y': np.random.randn(n_samples),
    'accel_z': np.random.randn(n_samples) + 9.8,
    'gyro_x': np.random.randn(n_samples) * 0.1,
    'gyro_y': np.random.randn(n_samples) * 0.1,
    'gyro_z': np.random.randn(n_samples) * 0.1,
    'label': np.random.choice([0, 1, 2], n_samples)
}

df = pd.DataFrame(data)
df.to_csv("data/raw/sample_data.csv", index=False)
print("✅ Sample data created")
```

### Issue: Data Format Errors

**Symptoms:**
```python
ValueError: Input data has wrong shape
```

**Solutions:**

1. **Check data dimensions:**
```python
import numpy as np

# Verify input shape
sensor_data = [1.2, -0.5, 9.8, 0.1, 0.2, -0.1]
print(f"Input shape: {np.array(sensor_data).shape}")
# Should be (6,)
```

2. **Fix data preprocessing:**
```python
def preprocess_sensor_data(data):
    """Ensure data is in correct format."""
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if data.shape[1] != 6:
        raise ValueError(f"Expected 6 features, got {data.shape[1]}")
    
    return data.astype(np.float32)
```

## ⚡ Performance Issues

### Issue: Slow Inference Speed

**Symptoms:**
- Long response times
- High CPU usage

**Solutions:**

1. **Use TensorFlow Lite:**
```python
# Convert to TFLite for faster inference
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)
```

2. **Enable model optimization:**
```python
# Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable XLA compilation
@tf.function(jit_compile=True)
def optimized_predict(model, input_data):
    return model(input_data)
```

3. **Batch processing:**
```python
# Process multiple readings at once
def batch_predict(model, sensor_readings):
    batch_data = np.array(sensor_readings)
    return model.predict(batch_data, verbose=0)
```

### Issue: High Memory Usage

**Symptoms:**
```bash
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Enable memory growth (GPU):**
```python
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

2. **Use data generators:**
```python
def data_generator(file_paths, batch_size=32):
    """Generate data in batches to save memory."""
    while True:
        for file_path in file_paths:
            # Load and yield small batches
            yield load_batch(file_path, batch_size)
```

3. **Clear model cache:**
```python
import gc
import tensorflow as tf

# Clear TensorFlow session
tf.keras.backend.clear_session()

# Force garbage collection
gc.collect()
```

## 🐳 Docker Issues

### Issue: Docker Build Fails

**Symptoms:**
```bash
ERROR: Package 'tensorflow' not found
```

**Solutions:**

1. **Check Dockerfile:**
```dockerfile
# Use specific Python version
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

2. **Build with no cache:**
```bash
docker build --no-cache -t step-detection .
```

3. **Check available space:**
```bash
docker system df
docker system prune  # Clean up if needed
```

### Issue: Container Won't Start

**Symptoms:**
```bash
docker: Error response from daemon: container exited immediately
```

**Solutions:**

1. **Check container logs:**
```bash
docker logs <container_id>
```

2. **Run in interactive mode:**
```bash
docker run -it step-detection /bin/bash
```

3. **Check file permissions:**
```bash
# In Dockerfile
COPY --chmod=755 main.py .
```

## 🧪 Testing Issues

### Issue: Tests Fail

**Symptoms:**
```bash
FAILED tests/test_detector.py::test_step_detection
```

**Solutions:**

1. **Run with verbose output:**
```bash
pytest tests/ -v -s
```

2. **Check test dependencies:**
```bash
pip install pytest pytest-cov
```

3. **Isolate failing test:**
```bash
pytest tests/test_detector.py::test_step_detection -v
```

4. **Update test data:**
```python
# Ensure test data is valid
@pytest.fixture
def valid_sensor_data():
    return {
        'accel_x': 1.2, 'accel_y': -0.5, 'accel_z': 9.8,
        'gyro_x': 0.1, 'gyro_y': 0.2, 'gyro_z': -0.1
    }
```

## 🔧 Environment Issues

### Issue: Import Errors

**Symptoms:**
```python
ImportError: cannot import name 'StepDetector' from 'src.step_detection'
```

**Solutions:**

1. **Check package structure:**
```bash
find src/ -name "*.py" | head -10
```

2. **Verify imports:**
```python
# In src/step_detection/__init__.py
from .core.detector import StepDetector
from .models.model_utils import create_cnn_model

__all__ = ['StepDetector', 'create_cnn_model']
```

3. **Install package properly:**
```bash
pip install -e .
```

### Issue: Version Conflicts

**Symptoms:**
```bash
ERROR: pip's dependency resolver does not currently handle version conflicts
```

**Solutions:**

1. **Create fresh environment:**
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

2. **Use conda for complex dependencies:**
```bash
conda env create -f environment.yml
```

3. **Pin compatible versions:**
```txt
# requirements.txt
tensorflow>=2.15.0,<2.20.0
fastapi>=0.100.0,<0.120.0
pandas>=1.5.0,<2.0.0
```

## 📱 Deployment Issues

### Issue: Production Server Crashes

**Symptoms:**
- Server stops responding
- High error rates

**Diagnostic Steps:**

1. **Check server logs:**
```bash
tail -f logs/step_detection.log
```

2. **Monitor system resources:**
```bash
htop
df -h  # Check disk space
```

3. **Test with minimal load:**
```bash
# Single worker for debugging
uvicorn src.step_detection.api.api:app --workers 1
```

**Solutions:**

1. **Add proper error handling:**
```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

2. **Configure worker timeout:**
```bash
uvicorn src.step_detection.api.api:app \
  --workers 4 \
  --timeout-keep-alive 5 \
  --timeout-graceful-shutdown 30
```

3. **Add health checks:**
```python
@app.get("/health")
async def health_check():
    try:
        # Test model loading
        detector = StepDetector("models/step_detection_model.keras")
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 🆘 Getting Help

### Diagnostic Information Script

```python
# diagnostic_info.py
import sys
import platform
import tensorflow as tf
import pandas as pd
import os

def collect_diagnostic_info():
    """Collect system diagnostic information."""
    info = {
        "System": {
            "Platform": platform.platform(),
            "Python Version": sys.version,
            "Architecture": platform.architecture()[0]
        },
        "Packages": {
            "TensorFlow": tf.__version__,
            "Pandas": pd.__version__,
        },
        "Environment": {
            "Current Directory": os.getcwd(),
            "Python Path": sys.path[:3],  # First 3 entries
            "Environment Variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith('STEP_DETECTION_')
            }
        },
        "Files": {
            "Models Directory": os.path.exists("models/"),
            "Data Directory": os.path.exists("data/"),
            "Config Directory": os.path.exists("config/")
        }
    }
    
    return info

if __name__ == "__main__":
    import json
    info = collect_diagnostic_info()
    print(json.dumps(info, indent=2))
```

### Support Checklist

Before asking for help, please:

1. ✅ **Run diagnostic script:**
```bash
python diagnostic_info.py > diagnostic_output.txt
```

2. ✅ **Check logs:**
```bash
tail -50 logs/step_detection.log
```

3. ✅ **Try minimal reproduction:**
```python
# Create a minimal example that demonstrates the issue
from src.step_detection.core.detector import StepDetector

try:
    detector = StepDetector("models/step_detection_model.keras")
    result = detector.process_reading(1, 0, 9.8, 0, 0, 0)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")
```

4. ✅ **Gather system info:**
- Operating system and version
- Python version
- Virtual environment details
- Error messages (full stack trace)

### Contact Support

- 📚 **Documentation**: Check all docs in `docs/` folder
- 🐛 **Bug Reports**: [GitHub Issues](../../issues)
- 💬 **Discussions**: [GitHub Discussions](../../discussions)  
- 📧 **Email**: [your-support-email@domain.com]

### Community Resources

- 🌟 **GitHub Repository**: [Link to repository]
- 📖 **Wiki**: [Link to wiki if available]
- 💬 **Discord/Slack**: [Link to community chat]
- 📺 **Video Tutorials**: [Link to video resources]

---

**🔧 Troubleshooting Complete! Most issues should now be resolved! 🎯**

If you're still experiencing problems, don't hesitate to reach out for support with the diagnostic information you've gathered.
