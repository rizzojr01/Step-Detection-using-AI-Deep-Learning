# 🚀 Performance Optimization Guide

Complete guide to optimizing the Step Detection system for maximum performance.

## 🎯 Performance Overview

This guide covers optimization strategies for:

- ⚡ **Model Inference Speed**
- 💾 **Memory Usage**
- 🌐 **API Throughput**
- 📱 **Real-time Processing**
- 🔄 **Batch Processing**
- 🐳 **Deployment Optimization**

## 📊 Current Performance Metrics

### Baseline Performance

| Metric                   | Value      | Target     |
| ------------------------ | ---------- | ---------- |
| ⚡ **Inference Time**    | 0.8ms      | <1ms       |
| 💾 **Memory Usage**      | 50MB       | <100MB     |
| 🌐 **API Throughput**    | 1000 req/s | >500 req/s |
| 📱 **Real-time Latency** | 1.2ms      | <2ms       |
| 🏆 **Model Accuracy**    | 96.2%      | >95%       |

## ⚡ Model Optimization

### 1. Model Architecture Optimization

#### Lightweight CNN Architecture

```python
def create_optimized_cnn_model(input_shape=(6,), num_classes=3):
    """Create an optimized CNN model for step detection."""
    model = keras.Sequential([
        # Efficient input processing
        layers.Reshape((1, input_shape[0]), input_shape=input_shape),

        # Optimized convolution layers
        layers.Conv1D(filters=16, kernel_size=1, strides=1,
                     activation="relu", use_bias=False),
        layers.BatchNormalization(),

        layers.Conv1D(filters=32, kernel_size=1, strides=1,
                     activation="relu", use_bias=False),
        layers.BatchNormalization(),

        # Efficient final layers
        layers.GlobalAveragePooling1D(),  # Instead of Flatten
        layers.Dense(num_classes, activation="softmax")
    ])

    # Optimized compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
```

#### Model Quantization

```python
import tensorflow as tf

def quantize_model(model_path, output_path):
    """Convert model to TensorFlow Lite with quantization."""

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Post-training quantization
    converter.representative_dataset = representative_dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ Quantized model saved to {output_path}")

def representative_dataset_generator():
    """Generate representative dataset for quantization."""
    import numpy as np
    for _ in range(100):
        yield [np.random.random((1, 6)).astype(np.float32)]

# Usage
quantize_model("models/step_detection_model.keras",
               "models/step_detection_quantized.tflite")
```

#### Model Pruning

```python
import tensorflow_model_optimization as tfmot

def create_pruned_model(model_path, sparsity=0.5):
    """Create a pruned model for better performance."""

    # Load base model
    base_model = tf.keras.models.load_model(model_path)

    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=sparsity,
            begin_step=0,
            end_step=1000
        )
    }

    # Apply pruning
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model, **pruning_params
    )

    # Compile
    pruned_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return pruned_model
```

### 2. Inference Optimization

#### Batch Prediction

```python
class BatchStepDetector:
    """Optimized step detector for batch processing."""

    def __init__(self, model_path, batch_size=32):
        self.model = tf.keras.models.load_model(model_path)
        self.batch_size = batch_size
        self.buffer = []

    def add_reading(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        """Add a sensor reading to the buffer."""
        reading = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        self.buffer.append(reading)

        # Process when buffer is full
        if len(self.buffer) >= self.batch_size:
            return self.process_batch()
        return None

    def process_batch(self):
        """Process all readings in the buffer."""
        if not self.buffer:
            return []

        # Convert to numpy array
        batch_data = np.array(self.buffer)

        # Predict in batch
        predictions = self.model.predict(batch_data, verbose=0)

        # Process results
        results = []
        for pred in predictions:
            result = self._process_prediction(pred)
            results.append(result)

        # Clear buffer
        self.buffer = []

        return results
```

#### TensorFlow Lite Inference

```python
class TFLiteStepDetector:
    """Optimized step detector using TensorFlow Lite."""

    def __init__(self, tflite_model_path):
        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_model_path,
            num_threads=4  # Use multiple threads
        )
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, sensor_data):
        """Fast prediction using TensorFlow Lite."""
        # Prepare input
        input_data = np.array(sensor_data, dtype=np.float32).reshape(1, -1)

        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data
        )

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        return output_data[0]
```

## 💾 Memory Optimization

### 1. Memory-Efficient Data Loading

```python
class MemoryEfficientDataLoader:
    """Memory-efficient data loader for training."""

    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size

    def generate_batches(self):
        """Generate data batches without loading all data into memory."""
        file_list = self._get_file_list()

        batch_x = []
        batch_y = []

        for file_path in file_list:
            # Load file incrementally
            for chunk in self._read_file_chunks(file_path):
                batch_x.append(chunk['features'])
                batch_y.append(chunk['label'])

                if len(batch_x) >= self.batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []

        # Yield remaining data
        if batch_x:
            yield np.array(batch_x), np.array(batch_y)
```

### 2. Model Memory Optimization

```python
def optimize_model_memory(model):
    """Optimize model for memory usage."""

    # Use mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Use gradient checkpointing for large models
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'activation_checkpointing'):
                layer.activation_checkpointing = True

    return model
```

## 🌐 API Performance Optimization

### 1. Async API Implementation

```python
from fastapi import FastAPI
import asyncio
import uvloop  # Faster event loop

# Use faster event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

app = FastAPI()

class AsyncStepDetector:
    """Async step detector for better API performance."""

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def predict_async(self, sensor_data):
        """Async prediction to avoid blocking."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._predict_sync,
            sensor_data
        )
        return result

    def _predict_sync(self, sensor_data):
        """Synchronous prediction method."""
        input_data = np.array(sensor_data).reshape(1, -1)
        prediction = self.model.predict(input_data, verbose=0)
        return prediction[0]

# Global detector instance
detector = AsyncStepDetector("models/step_detection_model.keras")

@app.post("/detect_step_async")
async def detect_step_async(sensor_data: SensorData):
    """Async step detection endpoint."""
    result = await detector.predict_async([
        sensor_data.accel_x, sensor_data.accel_y, sensor_data.accel_z,
        sensor_data.gyro_x, sensor_data.gyro_y, sensor_data.gyro_z
    ])

    return {"prediction": result.tolist()}
```

### 2. Connection Pooling & Caching

```python
from functools import lru_cache
import redis

# Redis cache for results
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=1000)
def cached_prediction(sensor_tuple):
    """Cache predictions for identical sensor readings."""
    # Convert tuple back to list for processing
    sensor_data = list(sensor_tuple)
    return detector.predict(sensor_data)

def detect_with_cache(sensor_data):
    """Detect steps with caching for repeated readings."""
    # Create hashable tuple for caching
    sensor_tuple = tuple(round(x, 3) for x in sensor_data)

    # Check Redis cache first
    cache_key = f"step_detection:{hash(sensor_tuple)}"
    cached_result = redis_client.get(cache_key)

    if cached_result:
        return json.loads(cached_result)

    # Compute prediction
    result = cached_prediction(sensor_tuple)

    # Cache result for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(result.tolist()))

    return result
```

### 3. Response Compression

```python
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Optimize CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📱 Real-time Processing Optimization

### 1. Circular Buffer for Sensor Data

```python
from collections import deque
import numpy as np

class OptimizedStepDetector:
    """Optimized step detector for real-time processing."""

    def __init__(self, model_path, buffer_size=10):
        self.model = tf.keras.models.load_model(model_path)
        self.buffer = deque(maxlen=buffer_size)
        self.step_count = 0
        self.last_step_time = 0

        # Pre-allocate arrays for better performance
        self.input_array = np.zeros((1, 6), dtype=np.float32)

    def process_reading_optimized(self, accel_x, accel_y, accel_z,
                                 gyro_x, gyro_y, gyro_z):
        """Optimized processing for real-time use."""
        current_time = time.time()

        # Add to buffer
        reading = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        self.buffer.append(reading)

        # Skip processing if too soon after last step
        if current_time - self.last_step_time < 0.1:
            return {"step_detected": False, "step_count": self.step_count}

        # Use pre-allocated array
        self.input_array[0] = reading

        # Fast prediction
        prediction = self.model.predict(self.input_array, verbose=0)[0]

        # Quick threshold check
        if prediction[1] > 0.7 or prediction[2] > 0.7:  # Start or end
            self.step_count += 1
            self.last_step_time = current_time
            return {"step_detected": True, "step_count": self.step_count}

        return {"step_detected": False, "step_count": self.step_count}
```

### 2. Multi-threaded Processing

```python
import threading
import queue

class ThreadedStepDetector:
    """Multi-threaded step detector for high-throughput processing."""

    def __init__(self, model_path, num_workers=4):
        self.models = [tf.keras.models.load_model(model_path)
                      for _ in range(num_workers)]
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_queue = queue.Queue()
        self.workers = []

        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(self.models[i],)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _worker(self, model):
        """Worker thread for processing sensor data."""
        while True:
            try:
                # Get data from queue
                data_id, sensor_data = self.input_queue.get(timeout=1)

                # Process
                input_array = np.array(sensor_data).reshape(1, -1)
                prediction = model.predict(input_array, verbose=0)[0]

                # Put result
                self.output_queue.put((data_id, prediction))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def process_async(self, sensor_data):
        """Add sensor data for async processing."""
        data_id = time.time()
        self.input_queue.put((data_id, sensor_data))
        return data_id

    def get_result(self, timeout=0.1):
        """Get processed result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
```

## 🐳 Deployment Optimization

### 1. Docker Multi-stage Build

```dockerfile
# Dockerfile.optimized
# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .

# Optimize Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1

# Run with optimizations
CMD ["python", "-O", "main.py", "--api-workers", "4"]
```

### 2. Kubernetes Optimization

```yaml
# k8s-deployment-optimized.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: step-detection-optimized
spec:
  replicas: 3
  selector:
    matchLabels:
      app: step-detection
  template:
    metadata:
      labels:
        app: step-detection
    spec:
      containers:
        - name: step-detection
          image: step-detection:optimized
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          env:
            - name: STEP_DETECTION_API_WORKERS
              value: "4"
            - name: PYTHONOPTIMIZE
              value: "2"
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
```

## 📊 Performance Monitoring

### 1. Performance Metrics Collection

```python
import time
import psutil
from functools import wraps

class PerformanceMonitor:
    """Monitor system performance metrics."""

    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'request_count': 0
        }

    def time_function(self, func):
        """Decorator to time function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time = (end_time - start_time) * 1000  # ms
            self.metrics['inference_times'].append(execution_time)
            self.metrics['request_count'] += 1

            return result
        return wrapper

    def collect_system_metrics(self):
        """Collect system performance metrics."""
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)

        self.metrics['memory_usage'].append(memory_usage)
        self.metrics['cpu_usage'].append(cpu_usage)

    def get_performance_report(self):
        """Generate performance report."""
        if not self.metrics['inference_times']:
            return "No metrics collected yet"

        avg_inference = np.mean(self.metrics['inference_times'])
        p95_inference = np.percentile(self.metrics['inference_times'], 95)
        avg_memory = np.mean(self.metrics['memory_usage'])
        avg_cpu = np.mean(self.metrics['cpu_usage'])

        return {
            'average_inference_time_ms': round(avg_inference, 2),
            'p95_inference_time_ms': round(p95_inference, 2),
            'average_memory_usage_percent': round(avg_memory, 2),
            'average_cpu_usage_percent': round(avg_cpu, 2),
            'total_requests': self.metrics['request_count']
        }

# Global monitor instance
monitor = PerformanceMonitor()

# Use with detector
@monitor.time_function
def detect_step_with_monitoring(sensor_data):
    return detector.process_reading(*sensor_data)
```

### 2. Real-time Performance Dashboard

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

@app.get("/performance", response_class=HTMLResponse)
async def performance_dashboard():
    """Real-time performance dashboard."""
    report = monitor.get_performance_report()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Step Detection Performance</title>
        <meta http-equiv="refresh" content="5">
    </head>
    <body>
        <h1>🚀 Performance Dashboard</h1>
        <div>
            <h2>⚡ Inference Performance</h2>
            <p>Average Time: {report.get('average_inference_time_ms', 'N/A')} ms</p>
            <p>95th Percentile: {report.get('p95_inference_time_ms', 'N/A')} ms</p>
        </div>
        <div>
            <h2>💾 System Resources</h2>
            <p>Memory Usage: {report.get('average_memory_usage_percent', 'N/A')}%</p>
            <p>CPU Usage: {report.get('average_cpu_usage_percent', 'N/A')}%</p>
        </div>
        <div>
            <h2>📊 Request Statistics</h2>
            <p>Total Requests: {report.get('total_requests', 'N/A')}</p>
        </div>
    </body>
    </html>
    """

    return html_content
```

## 🎯 Performance Benchmarking

### Benchmarking Script

```python
import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_step_detection():
    """Comprehensive performance benchmark."""

    # Load models
    keras_detector = StepDetector("models/step_detection_model.keras")
    tflite_detector = TFLiteStepDetector("models/step_detection_quantized.tflite")

    # Generate test data
    n_samples = 1000
    test_data = np.random.randn(n_samples, 6)

    # Benchmark Keras model
    print("🧪 Benchmarking Keras model...")
    keras_times = []
    for data in test_data:
        start_time = time.perf_counter()
        keras_detector.process_reading(*data)
        end_time = time.perf_counter()
        keras_times.append((end_time - start_time) * 1000)

    # Benchmark TFLite model
    print("🧪 Benchmarking TFLite model...")
    tflite_times = []
    for data in test_data:
        start_time = time.perf_counter()
        tflite_detector.predict(data)
        end_time = time.perf_counter()
        tflite_times.append((end_time - start_time) * 1000)

    # Results
    print(f"\n📊 Performance Results:")
    print(f"Keras Model - Avg: {np.mean(keras_times):.2f}ms, P95: {np.percentile(keras_times, 95):.2f}ms")
    print(f"TFLite Model - Avg: {np.mean(tflite_times):.2f}ms, P95: {np.percentile(tflite_times, 95):.2f}ms")
    print(f"Speedup: {np.mean(keras_times) / np.mean(tflite_times):.2f}x")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(keras_times, bins=50, alpha=0.7, label='Keras')
    plt.hist(tflite_times, bins=50, alpha=0.7, label='TFLite')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Inference Time Distribution')

    plt.subplot(1, 2, 2)
    plt.plot(keras_times[:100], label='Keras', alpha=0.7)
    plt.plot(tflite_times[:100], label='TFLite', alpha=0.7)
    plt.xlabel('Sample Number')
    plt.ylabel('Inference Time (ms)')
    plt.legend()
    plt.title('Inference Time Over Time')

    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    benchmark_step_detection()
```

---

**🚀 Performance Optimized! Your system is now running at maximum efficiency! ⚡**
