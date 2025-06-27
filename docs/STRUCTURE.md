# Step Detection Project - Organized Structure

## 📁 Project Structure

```
Step-Detection-using-AI-Deep-Learning/
├── src/step_detection/           # Main package
│   ├── __init__.py              # Package initialization
│   ├── core/                    # Core detection functionality
│   │   ├── __init__.py
│   │   └── detector.py          # StepDetector, SimpleStepCounter classes
│   ├── models/                  # Model utilities
│   │   ├── __init__.py
│   │   └── model_utils.py       # Model creation, training, evaluation
│   ├── utils/                   # Data processing utilities
│   │   ├── __init__.py
│   │   └── data_processor.py    # Data loading and preprocessing
│   └── api/                     # FastAPI server
│       ├── __init__.py
│       └── api.py               # REST API endpoints
├── notebooks/                   # Jupyter notebooks
│   ├── CNN_TensorFlow_Clean.ipynb  # Clean training notebook
│   └── CNN_TensorFlow.ipynb        # Original notebook
├── scripts/                     # Executable scripts
│   ├── run_demo.py             # Demo script
│   ├── run_api.py              # API server script
│   └── main.py                 # Main CLI script
├── data/                       # Data directory
│   ├── raw/                    # Raw sensor data
│   └── processed/              # Processed data
├── models/                     # Saved models
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── config/                     # Configuration files
│   └── settings.py            # Project settings
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### 1. Installation

```bash
# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### 2. Training a Model

```bash
# Run the clean notebook
jupyter notebook notebooks/CNN_TensorFlow_Clean.ipynb
```

### 3. Running the Demo

```bash
python scripts/run_demo.py
```

### 4. Starting the API Server

```bash
python scripts/run_api.py
```

## 📚 Package Usage

### Import the Package

```python
# Import specific functions
from src.step_detection.utils.data_processor import load_step_data
from src.step_detection.models.model_utils import create_cnn_model
from src.step_detection.core.detector import StepDetector

# Or import the main package
import sys
import os
sys.path.append('src')
from step_detection import *
```

### Data Processing

```python
# Load step detection data
data = load_step_data("data/raw")

# Prepare for training
train_features, val_features, train_labels, val_labels = prepare_data_for_training(data)
```

### Model Training

```python
# Create and train model
model = create_cnn_model()
history = train_model(model, train_features, train_labels, val_features, val_labels)

# Evaluate model
accuracy = evaluate_model(model, val_features, val_labels)
```

### Real-time Detection

```python
# Initialize detector
detector = StepDetector("models/step_detection_model.keras")

# Process sensor readings
step_detected = detector.process_reading(
    accel_x, accel_y, accel_z,  # Accelerometer
    gyro_x, gyro_y, gyro_z      # Gyroscope
)

# Get step count
total_steps = detector.get_step_count()
```

## 🌐 API Usage

Start the server:

```bash
python scripts/run_api.py
```

Available endpoints:

- `POST /detect_step` - Detect steps from sensor data
- `GET /step_count` - Get current step count
- `POST /reset_count` - Reset step count
- `GET /session_summary` - Get session summary
- `GET /model_info` - Get model information
- `GET /health` - Health check

## 🧪 Testing

```bash
python tests/test_package.py
```

## 📊 Key Features

- **Modular Design**: Organized into logical packages
- **Clean Notebook**: Focus on model training only
- **Production Ready**: FastAPI server for deployment
- **Easy Installation**: Standard Python package structure
- **Comprehensive API**: Full REST API for integration
- **Real-time Processing**: Optimized for live sensor data

## 🔧 Configuration

Configuration settings are in `config/settings.py`:

- Model parameters
- Data paths
- API settings
- Step detection thresholds

## 📝 Development

The project follows Python packaging best practices:

- Source code in `src/` directory
- Separate notebooks, scripts, tests
- Proper package initialization
- Entry points for command-line tools
- Development dependencies in setup.py

## 🚀 Deployment

1. **Local Development**: Use `pip install -e .`
2. **Production**: Use `pip install .`
3. **Docker**: Build with provided Dockerfile
4. **API Server**: Deploy with uvicorn/gunicorn

This organized structure makes the codebase maintainable, testable, and production-ready!
