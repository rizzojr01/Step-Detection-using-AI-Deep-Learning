# Architecture Documentation

## Overview

The Step Detection system is designed with a modular, production-ready architecture that separates concerns and enables scalability, maintainability, and testability.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Web Frontend  │    │  Mobile Apps    │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │              ┌───────▼───────┐              │
          │              │  Load Balancer │              │
          │              │    (Nginx)     │              │
          │              └───────┬───────┘              │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      FastAPI Server     │
                    │   (REST + WebSocket)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Step Detection       │
                    │       Package           │
                    │                         │
                    │  ┌─────────────────┐    │
                    │  │   Core Models   │    │
                    │  │                 │    │
                    │  │ ┌─────────────┐ │    │
                    │  │ │ TensorFlow  │ │    │
                    │  │ │ CNN Model   │ │    │
                    │  │ └─────────────┘ │    │
                    │  └─────────────────┘    │
                    └─────────────────────────┘
```

### Component Architecture

```
src/step_detection/
├── __init__.py              # Package exports and initialization
├── core/                    # Core business logic
│   ├── __init__.py
│   └── detector.py          # StepDetector, SimpleStepCounter
├── models/                  # ML model utilities
│   ├── __init__.py
│   └── model_utils.py       # Model creation, training, evaluation
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── data_processor.py    # Data loading, preprocessing
└── api/                     # API layer
    ├── __init__.py
    └── api.py               # FastAPI application
```

## Design Principles

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:

- **Core**: Business logic for step detection
- **Models**: Machine learning operations
- **Utils**: Data processing and utilities
- **API**: External interface layer

### 2. Dependency Inversion

High-level modules don't depend on low-level modules. Both depend on abstractions.

```python
# Abstract interface
class StepDetectorInterface:
    def process_reading(self, *args) -> Dict: ...
    def get_step_count(self) -> int: ...
    def reset(self) -> None: ...

# Concrete implementation
class StepDetector(StepDetectorInterface):
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
    
    def process_reading(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        # Implementation details
```

### 3. Single Responsibility Principle

Each class has one reason to change:

- `StepDetector`: Real-time step detection with detailed results
- `SimpleStepCounter`: Simple step counting
- `DataProcessor`: Data loading and preprocessing
- `ModelUtils`: Model creation and training

### 4. Open/Closed Principle

Software entities are open for extension but closed for modification:

```python
# Base detector that can be extended
class BaseStepDetector:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def load_model(self, path: str):
        """Override in subclasses for different model types"""
        return tf.keras.models.load_model(path)
    
    def preprocess(self, data):
        """Override for custom preprocessing"""
        return data

# Extended for different use cases
class EnhancedStepDetector(BaseStepDetector):
    def preprocess(self, data):
        # Custom preprocessing
        return self.apply_filters(data)
```

## Data Flow

### 1. Training Data Flow

```
Raw Sensor Data → Data Processor → Preprocessed Data → Model Training → Trained Model
     ↓                 ↓               ↓                    ↓               ↓
CSV Files     Feature Extraction   Train/Val Split   CNN Training    .keras File
              Label Encoding       Normalization     Optimization    Metadata
              Validation          Augmentation      Evaluation      Thresholds
```

### 2. Inference Data Flow

```
Sensor Reading → API Endpoint → Detector → Model → Post-processing → Response
     ↓              ↓             ↓         ↓           ↓              ↓
6D Vector     Validation     Preprocessing  CNN      Threshold     JSON Result
(ax,ay,az,     Format Check   Normalization  Inference  Application   Step Events
 gx,gy,gz)     Type Check     Reshaping     Softmax    Step Logic    Probabilities
```

### 3. Real-time Detection Flow

```
Client → WebSocket → API Handler → Step Detector → Model Inference → Response → Client
  ↓         ↓           ↓              ↓               ↓               ↓         ↓
JSON     Connection   Validation   Process Reading   Predictions   JSON Reply  Update UI
Data     Management   Error Check   State Update    Thresholding  Broadcast   Show Steps
```

## Module Details

### Core Module (`src/step_detection/core/`)

**Purpose**: Contains the core business logic for step detection.

**Classes**:

1. **StepDetector**
   - Responsibilities: Real-time step detection, session management
   - Dependencies: TensorFlow model, numpy
   - Key Methods:
     - `process_reading()`: Process single sensor reading
     - `get_session_summary()`: Get session statistics
     - `reset()`: Reset detection state

2. **SimpleStepCounter**
   - Responsibilities: Simple step counting without detailed events
   - Dependencies: TensorFlow model
   - Key Methods:
     - `process_reading()`: Process reading and return boolean
     - `get_count()`: Get total step count
     - `reset()`: Reset counter

**Design Patterns**:
- **State Pattern**: Managing step detection state (start, in-progress, end)
- **Strategy Pattern**: Different detection strategies (threshold-based, ML-based)

### Models Module (`src/step_detection/models/`)

**Purpose**: Machine learning model operations and utilities.

**Functions**:

1. **create_cnn_model()**
   - Creates and compiles CNN architecture
   - Configurable input shape and classes
   - Returns compiled Keras model

2. **train_model()**
   - Handles model training with callbacks
   - Early stopping and learning rate scheduling
   - Returns training history

3. **evaluate_model()**
   - Comprehensive model evaluation
   - Classification metrics and confusion matrix
   - Returns evaluation results

**Design Patterns**:
- **Factory Pattern**: Model creation
- **Builder Pattern**: Complex model configuration

### Utils Module (`src/step_detection/utils/`)

**Purpose**: Data processing and utility functions.

**Functions**:

1. **load_step_data()**
   - Loads and combines CSV files
   - Data validation and cleaning
   - Returns pandas DataFrame

2. **prepare_data_for_training()**
   - Feature extraction and label encoding
   - Train/validation split with stratification
   - Data type optimization

**Design Patterns**:
- **Pipeline Pattern**: Data processing pipeline
- **Adapter Pattern**: Different data format handling

### API Module (`src/step_detection/api/`)

**Purpose**: External interface layer providing REST and WebSocket APIs.

**Components**:

1. **FastAPI Application**
   - RESTful endpoints for step detection
   - WebSocket for real-time communication
   - Request/response models with Pydantic

2. **Middleware**
   - CORS handling
   - Request logging
   - Error handling

**Design Patterns**:
- **Facade Pattern**: Simplified interface to complex subsystem
- **Observer Pattern**: WebSocket event handling

## Communication Patterns

### 1. Request-Response (REST)

```python
# Synchronous communication
@app.post("/detect_step")
async def detect_step(reading: SensorReading):
    result = detector.process_reading(**reading.dict())
    return StepDetectionResponse(**result)
```

### 2. Real-time Communication (WebSocket)

```python
# Asynchronous, bidirectional communication
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        result = process_sensor_data(data)
        await websocket.send_text(json.dumps(result))
```

### 3. Event-Driven Processing

```python
# Step detection events
class StepEvent:
    def __init__(self, event_type: str, timestamp: str, data: Dict):
        self.event_type = event_type  # "start", "end", "completed"
        self.timestamp = timestamp
        self.data = data

# Event handlers
def on_step_start(event: StepEvent):
    logger.info(f"Step started at {event.timestamp}")

def on_step_completed(event: StepEvent):
    logger.info(f"Step completed: {event.data}")
```

## Error Handling Strategy

### 1. Layered Error Handling

```python
# API Layer - HTTP errors
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

# Business Logic Layer - Domain errors
class StepDetectionError(Exception):
    pass

class ModelNotLoadedError(StepDetectionError):
    pass

# Infrastructure Layer - System errors
class ModelLoadError(Exception):
    pass
```

### 2. Graceful Degradation

```python
def process_reading_with_fallback(self, *args):
    try:
        return self.ml_detector.process_reading(*args)
    except ModelNotLoadedError:
        # Fallback to simple rule-based detection
        return self.rule_based_detector.process_reading(*args)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return self.create_error_response()
```

## Scalability Considerations

### 1. Horizontal Scaling

```yaml
# Multiple API instances
services:
  api-1:
    image: step-detection:latest
    ports: ["8001:8000"]
  
  api-2:
    image: step-detection:latest
    ports: ["8002:8000"]
  
  load-balancer:
    image: nginx:alpine
    ports: ["8000:80"]
    depends_on: [api-1, api-2]
```

### 2. Model Caching

```python
# Singleton pattern for model loading
class ModelManager:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_path: str):
        if self._model is None:
            self._model = tf.keras.models.load_model(model_path)
        return self._model
```

### 3. Asynchronous Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncStepDetector:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_reading_async(self, *args):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.detector.process_reading, 
            *args
        )
        return result
```

## Security Architecture

### 1. Authentication Layer

```python
# JWT-based authentication
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

jwt_authentication = JWTAuthentication(
    secret=SECRET_KEY,
    lifetime_seconds=3600,
    tokenUrl="/auth/login"
)

# Protected endpoints
@app.get("/protected")
async def protected_endpoint(user: User = Depends(current_user)):
    return {"message": f"Hello {user.email}"}
```

### 2. Input Validation

```python
# Pydantic models for validation
class SensorReading(BaseModel):
    accel_x: confloat(ge=-50, le=50)
    accel_y: confloat(ge=-50, le=50)
    accel_z: confloat(ge=-50, le=50)
    gyro_x: confloat(ge=-10, le=10)
    gyro_y: confloat(ge=-10, le=10)
    gyro_z: confloat(ge=-10, le=10)
```

### 3. Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/detect_step")
@limiter.limit("60/minute")
async def detect_step(request: Request, reading: SensorReading):
    # Protected endpoint
```

## Testing Architecture

### 1. Unit Tests

```python
# Test individual components
class TestStepDetector:
    def test_process_reading_valid_input(self):
        detector = StepDetector("test_model.keras")
        result = detector.process_reading(1.0, 0.5, 9.8, 0.1, 0.2, 0.0)
        assert "step_start" in result
        assert "step_end" in result
```

### 2. Integration Tests

```python
# Test component interactions
class TestAPIIntegration:
    def test_step_detection_endpoint(self):
        response = client.post("/detect_step", json={
            "accel_x": 1.0, "accel_y": 0.5, "accel_z": 9.8,
            "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": 0.0
        })
        assert response.status_code == 200
```

### 3. End-to-End Tests

```python
# Test complete workflows
class TestE2E:
    def test_training_to_inference_pipeline(self):
        # Train model
        model = train_model(training_data)
        
        # Save and load model
        save_model(model, "test_model.keras")
        detector = StepDetector("test_model.keras")
        
        # Test inference
        result = detector.process_reading(*test_data)
        assert result is not None
```

## Monitoring and Observability

### 1. Metrics Collection

```python
from prometheus_client import Counter, Histogram

# Business metrics
step_detection_requests = Counter('step_detection_requests_total')
step_detection_duration = Histogram('step_detection_duration_seconds')

# System metrics
model_inference_time = Histogram('model_inference_time_seconds')
websocket_connections = Counter('websocket_connections_total')
```

### 2. Structured Logging

```python
import structlog

logger = structlog.get_logger()

def process_reading(self, *args):
    logger.info(
        "processing_sensor_reading",
        accel_magnitude=calculate_magnitude(args[:3]),
        gyro_magnitude=calculate_magnitude(args[3:]),
        session_id=self.session_id
    )
```

### 3. Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "memory_usage": get_memory_usage(),
        "uptime": get_uptime()
    }
```

## Future Enhancements

### 1. Microservices Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   API       │  │   Model     │  │   Data      │
│  Gateway    │  │  Service    │  │  Service    │
└─────────────┘  └─────────────┘  └─────────────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
              ┌─────────────┐
              │  Message    │
              │   Queue     │
              └─────────────┘
```

### 2. Event Sourcing

```python
# Event store for step detection history
class StepEvent:
    def __init__(self, event_type: str, data: Dict, timestamp: datetime):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp

class EventStore:
    def append(self, event: StepEvent):
        # Store event
        
    def get_events(self, session_id: str) -> List[StepEvent]:
        # Retrieve events
```

### 3. CQRS (Command Query Responsibility Segregation)

```python
# Separate read and write models
class StepDetectionCommand:
    def execute(self, sensor_data: SensorReading):
        # Write operation
        
class StepDetectionQuery:
    def get_session_summary(self, session_id: str):
        # Read operation
```

This architecture provides a solid foundation for building scalable, maintainable, and production-ready step detection systems.
