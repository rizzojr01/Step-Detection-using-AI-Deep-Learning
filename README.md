# Step Detection using AI Deep Learning

A comprehensive solution for real-time step detection using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## 🏗️ Project Structure

```
Step-Detection-using-AI-Deep-Learning/
├── src/                          # Source code
│   ├── step_detection/           # Main package
│   │   ├── core/                 # Core detection logic
│   │   ├── models/               # Model utilities
│   │   ├── utils/                # Data processing utilities
│   │   └── api/                  # FastAPI server
│   ├── initialize_model.py       # Model initialization
│   └── step_detection_api.py     # Legacy API (for reference)
├── notebooks/                    # Jupyter notebooks
│   ├── CNN_TensorFlow_Clean.ipynb  # Clean training notebook
│   └── CNN_TensorFlow.ipynb        # Original notebook
├── data/                         # Data directories
│   ├── raw/                      # Raw sensor data
│   └── processed/                # Processed data outputs
├── models/                       # Trained models
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
├── config/                       # Configuration files
├── logs/                         # Log files
├── docker/                       # Docker configuration
├── main.py                       # Main CLI interface
├── setup.py                      # Package setup
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Step-Detection-using-AI-Deep-Learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Package Installation

```bash
# Install the package in development mode
pip install -e .
```

### 3. Usage

#### Using the CLI Interface

```bash
python main.py
```

This will present you with options to:

1. Train a new model
2. Test real-time detection
3. Start the API server

#### Using as a Python Package

```python
from src.step_detection import (
    load_step_data,
    prepare_data_for_training,
    create_cnn_model,
    train_model,
    StepDetector
)

# Load and prepare data
data = load_step_data("data/raw")
train_X, val_X, train_y, val_y = prepare_data_for_training(data)

# Create and train model
model = create_cnn_model()
history = train_model(model, train_X, train_y, val_X, val_y)

# Use for real-time detection
detector = StepDetector("models/step_detection_model.keras")
result = detector.process_reading(1.2, -0.5, 9.8, 0.1, 0.2, -0.1)
```

#### Using the Jupyter Notebook

```bash
jupyter notebook notebooks/CNN_TensorFlow_Clean.ipynb
```

#### Starting the API Server

```bash
python main.py
# Choose option 3, or directly:
uvicorn src.step_detection.api.api:app --reload
```

API documentation will be available at: http://localhost:8000/docs

## 📊 Model Performance

- **Framework**: TensorFlow/Keras
- **Architecture**: 1D CNN optimized for sensor data
- **Input**: 6D sensor data (3-axis accelerometer + 3-axis gyroscope)
- **Output**: 3 classes (No Label, Start, End)
- **Validation Accuracy**: ~96%+

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## 📚 API Endpoints

- `POST /detect_step` - Detect steps from sensor data
- `GET /step_count` - Get current step count
- `POST /reset_count` - Reset step count
- `GET /session_summary` - Get session summary
- `GET /model_info` - Get model information
- `GET /health` - Health check

## 🏃‍♂️ Real-time Detection

The package provides two main classes for real-time detection:

1. **StepDetector**: Full detection with start/end events
2. **SimpleStepCounter**: Simple step counting

## 📱 Deployment

### Docker

```bash
docker build -f docker/Dockerfile.prod -t step-detection .
docker run -p 8000:8000 step-detection
```

### Production

The models are saved in multiple formats for different deployment scenarios:

- `.keras` format for TensorFlow applications
- TensorFlow Lite for mobile deployment
- SavedModel format for TensorFlow Serving

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and formatting checks
6. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- FastAPI team for the web framework
- Contributors and testers
