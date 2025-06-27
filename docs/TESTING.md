# 🧪 Testing Guide

Comprehensive testing guide for the Step Detection system.

## 🎯 Testing Overview

Our testing strategy covers:

- ✅ **Unit Tests**: Individual component testing
- 🔄 **Integration Tests**: Component interaction testing  
- ⚡ **Performance Tests**: Speed and accuracy benchmarks
- 🌐 **API Tests**: REST and WebSocket endpoint testing
- 📱 **Real-time Tests**: Live detection validation

## 🚀 Quick Test Run

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_package.py -v          # Package tests
pytest tests/test_detector.py -v         # Detection tests
pytest tests/test_api.py -v              # API tests
```

## 📊 Test Categories

### 1. 📦 Package Tests (`test_package.py`)

Tests basic package functionality and imports:

```python
def test_package_imports():
    """Test that all main components can be imported."""
    
def test_model_loading():
    """Test model loading from different paths."""
    
def test_data_loading():
    """Test data loading and preprocessing."""
```

**Run**: `pytest tests/test_package.py -v`

### 2. 🔍 Detector Tests (`test_detector.py`)

Tests core detection algorithms:

```python
def test_step_detector_initialization():
    """Test StepDetector initialization."""
    
def test_single_step_detection():
    """Test detection of individual steps."""
    
def test_multiple_step_sequence():
    """Test detection of step sequences."""
    
def test_threshold_optimization():
    """Test threshold optimization logic."""
```

**Run**: `pytest tests/test_detector.py -v`

### 3. 🌐 API Tests (`test_api.py`)

Tests REST and WebSocket endpoints:

```python
def test_health_endpoint():
    """Test API health check."""
    
def test_step_detection_endpoint():
    """Test step detection POST endpoint."""
    
def test_websocket_connection():
    """Test WebSocket real-time detection."""
```

**Run**: `pytest tests/test_api.py -v`

### 4. ⚡ Performance Tests (`test_performance.py`)

Benchmarks system performance:

```python
def test_inference_speed():
    """Test model inference speed."""
    
def test_memory_usage():
    """Test memory consumption."""
    
def test_concurrent_requests():
    """Test API under load."""
```

**Run**: `pytest tests/test_performance.py -v`

## 🛠️ Test Configuration

### Test Settings (`tests/conftest.py`)

```python
import pytest
from src.step_detection.core.detector import StepDetector

@pytest.fixture
def sample_detector():
    """Provide a StepDetector instance for testing."""
    return StepDetector("models/step_detection_model.keras")

@pytest.fixture
def sample_sensor_data():
    """Provide sample sensor data for testing."""
    return {
        "accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8,
        "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1
    }
```

### Test Data (`tests/data/`)

```
tests/data/
├── sample_walking.csv      # Sample walking data
├── sample_running.csv      # Sample running data  
├── sample_stationary.csv   # Sample stationary data
└── edge_cases.csv          # Edge case test data
```

## 🎯 Running Specific Tests

### Test a Single Function

```bash
# Test specific function
pytest tests/test_detector.py::test_step_detection -v

# Test with debug output
pytest tests/test_detector.py::test_step_detection -v -s
```

### Test with Markers

```bash
# Run only fast tests
pytest -m "not slow" -v

# Run only integration tests
pytest -m integration -v

# Run only unit tests
pytest -m unit -v
```

### Test Configuration in `pytest.ini`

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    api: marks tests as API tests
```

## 📊 Coverage Reports

### Generate Coverage Report

```bash
# HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Terminal coverage report
pytest tests/ --cov=src --cov-report=term-missing

# XML coverage report (for CI)
pytest tests/ --cov=src --cov-report=xml
```

### View Coverage

```bash
# Open HTML report
open htmlcov/index.html

# View terminal report
pytest tests/ --cov=src --cov-report=term
```

### Coverage Configuration (`.coveragerc`)

```ini
[run]
source = src
omit = 
    */tests/*
    */venv/*
    */.venv/*
    */site-packages/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 🧪 Manual Testing

### 1. Real-time Detection Test

```bash
# Start interactive test
python tests/test_real_time_detection.py

# Or use main CLI
python main.py
# Choose option 2: "Test real-time detection"
```

### 2. API Server Test

```bash
# Terminal 1: Start server
python main.py
# Choose option 3: "Start API server"

# Terminal 2: Test endpoints
python tests/test_api_manual.py

# Terminal 3: Test WebSocket
python test_websocket.py
```

### 3. Model Training Test

```bash
# Test model training pipeline
python main.py
# Choose option 1: "Train a new model"

# Test with custom data
python tests/test_training_pipeline.py
```

## 🚀 Performance Testing

### Benchmark Script

```python
# benchmark.py
import time
import numpy as np
from src.step_detection.core.detector import StepDetector

def benchmark_inference_speed():
    detector = StepDetector("models/step_detection_model.keras")
    
    # Generate test data
    n_samples = 1000
    test_data = np.random.randn(n_samples, 6)
    
    # Benchmark
    start_time = time.time()
    for data in test_data:
        detector.process_reading(*data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_samples * 1000  # ms
    print(f"⚡ Average inference time: {avg_time:.2f} ms")
    print(f"🚀 Throughput: {1000/avg_time:.0f} samples/second")

if __name__ == "__main__":
    benchmark_inference_speed()
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler tests/test_memory_usage.py

# Line-by-line profiling
@profile
def test_memory_intensive_function():
    # Your test code here
    pass
```

### Load Testing

```bash
# Install locust for load testing
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🐛 Debugging Tests

### Debug Failed Tests

```bash
# Run with debugging
pytest tests/test_detector.py::test_failing_function --pdb

# Add print statements
pytest tests/test_detector.py -v -s

# Capture logging
pytest tests/test_detector.py --log-cli-level=DEBUG
```

### Test Isolation

```bash
# Run test in isolation
pytest tests/test_detector.py::test_specific -v --forked

# Clear cache between runs
pytest --cache-clear tests/
```

## 🔄 Continuous Integration

### GitHub Actions (`.github/workflows/test.yml`)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        args: [tests/, --tb=short]
        always_run: true
        pass_filenames: false
```

## 📈 Test Metrics

### Key Metrics to Track

- ✅ **Test Coverage**: >90% line coverage
- ⚡ **Test Speed**: <30 seconds total runtime
- 🎯 **Test Success Rate**: 100% pass rate
- 🔄 **Test Stability**: No flaky tests

### Monitoring Test Health

```bash
# Check test health
pytest tests/ --tb=short --durations=10

# Generate test report
pytest tests/ --html=report.html --self-contained-html
```

## 🛠️ Test Development Guidelines

### Writing Good Tests

1. **Test Structure**: Follow AAA pattern (Arrange, Act, Assert)
2. **Test Names**: Use descriptive names that explain what is being tested
3. **Test Data**: Use fixtures for reusable test data
4. **Assertions**: Use specific assertions with clear error messages
5. **Independence**: Tests should not depend on each other

### Example Test

```python
def test_step_detector_processes_valid_sensor_data(sample_detector, sample_sensor_data):
    """Test that StepDetector correctly processes valid sensor data."""
    # Arrange
    detector = sample_detector
    sensor_data = sample_sensor_data
    
    # Act
    result = detector.process_reading(**sensor_data)
    
    # Assert
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "step_detected" in result, "Result should contain step_detected"
    assert "step_count" in result, "Result should contain step_count"
    assert isinstance(result["step_detected"], bool), "step_detected should be boolean"
    assert isinstance(result["step_count"], int), "step_count should be integer"
    assert result["step_count"] >= 0, "step_count should be non-negative"
```

---

**Happy Testing! 🧪✅**
