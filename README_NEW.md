# 🚶‍♂️ Step Detection using AI & Deep Learning

A complete, production-ready system for real-time step detection using TensorFlow CNN models and sensor data (accelerometer + gyroscope). Features both REST API and WebSocket support for real-time applications.

## 🌟 Features

- **Real-time Step Detection**: Process sensor data in real-time with optimized thresholds
- **CNN Model**: TensorFlow-based Convolutional Neural Network for accurate step classification
- **Dual API Support**: Both REST endpoints and WebSocket for different use cases
- **Modular Architecture**: Clean, maintainable code structure with separate modules
- **Production Ready**: Complete with logging, error handling, and deployment configs
- **Interactive CLI**: Easy-to-use command-line interface for training and testing
- **Comprehensive Testing**: Unit tests and integration tests included

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Or pip with Python virtual environment

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd Step-Detection-using-AI-Deep-Learning
```

2. **Install dependencies**:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Train the model** (if not already trained):

```bash
python main.py
# Choose option 1: Train new model
```

4. **Test the system**:

```bash
python main.py
# Choose option 2: Test real-time detection
```

5. **Start the API server**:

```bash
python main.py
# Choose option 4: Start API server
```

## 📁 Project Structure

```
Step-Detection-using-AI-Deep-Learning/
├── src/step_detection/           # Main package
│   ├── __init__.py              # Package initialization
│   ├── core/                    # Core detection logic
│   │   ├── __init__.py
│   │   └── detector.py          # StepDetector and SimpleStepCounter classes
│   ├── models/                  # Model utilities
│   │   ├── __init__.py
│   │   └── model_utils.py       # CNN model creation and training
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   └── data_processor.py    # Data loading and preprocessing
│   └── api/                     # FastAPI application
│       ├── __init__.py
│       └── api.py               # REST and WebSocket endpoints
├── notebooks/                   # Jupyter notebooks
│   ├── CNN_TensorFlow_Clean.ipynb  # Clean training notebook
│   └── CNN_TensorFlow.ipynb        # Original research notebook
├── data/                        # Data directory
│   ├── raw/                     # Raw sensor data
│   └── processed/               # Processed datasets
├── models/                      # Trained models
│   ├── step_detection_model.keras  # Main model file
│   └── model_metadata.json         # Model metadata
├── tests/                       # Test files
├── docs/                        # Documentation
├── config/                      # Configuration files
├── scripts/                     # Utility scripts
├── main.py                      # Main CLI interface
├── pyproject.toml              # Project configuration (uv)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Usage

### Command Line Interface

The main script provides an interactive menu:

```bash
python main.py
```

**Options:**

1. **Train new model** - Train a new CNN model from scratch
2. **Test real-time detection** - Test step detection with sample data
3. **Optimize detection thresholds** - Find optimal detection thresholds
4. **Start API server** - Launch the FastAPI server
5. **Exit** - Quit the application

### API Usage

#### REST API

Start the server:

```bash
python main.py  # Choose option 4
```

**Endpoints:**

- `GET /` - API information and available endpoints
- `POST /detect_step` - Detect steps from sensor data
- `GET /step_count` - Get current step count
- `POST /reset_count` - Reset step counter
- `GET /session_summary` - Get detection session summary
- `GET /model_info` - Get model information
- `GET /health` - Health check

**Example Request:**

```bash
curl -X POST "http://localhost:8000/detect_step" \
     -H "Content-Type: application/json" \
     -d '{
       "accel_x": 1.2,
       "accel_y": -0.5,
       "accel_z": 9.8,
       "gyro_x": 0.1,
       "gyro_y": 0.2,
       "gyro_z": -0.1
     }'
```

#### WebSocket API

Connect to: `ws://localhost:8000/ws/realtime`

**Send data:**

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

**Receive response:**

```json
{
  "step_start": false,
  "step_end": false,
  "start_probability": 0.0024,
  "end_probability": 0.002,
  "step_count": 0,
  "timestamp": "2025-06-27T14:20:19.586914",
  "status": "success"
}
```

### Python API

```python
from src.step_detection import StepDetector, SimpleStepCounter

# Initialize detectors
detector = StepDetector("models/step_detection_model.keras")
counter = SimpleStepCounter("models/step_detection_model.keras")

# Process sensor reading
result = detector.process_reading(1.2, -0.5, 9.8, 0.1, 0.2, -0.1)
step_detected = counter.process_reading(1.2, -0.5, 9.8, 0.1, 0.2, -0.1)

print(f"Step detection result: {result}")
print(f"Total steps: {counter.get_count()}")
```

## 📊 Model Details

### Architecture

- **Type**: 1D Convolutional Neural Network (CNN)
- **Input**: 6 features (accelerometer x,y,z + gyroscope x,y,z)
- **Output**: 3 classes (No Label, Step Start, Step End)
- **Framework**: TensorFlow/Keras

### Model Layers

```
Input(6,) → Reshape(1,6) → Conv1D(32) → MaxPooling1D → Conv1D(64) → Flatten → Dense(3, softmax)
```

### Performance

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Typical Accuracy**: >95% on validation set

## 🔧 Configuration

### Threshold Optimization

The system includes automatic threshold optimization:

```bash
python main.py  # Choose option 3
```

This analyzes validation data to find optimal detection thresholds that balance sensitivity and accuracy.

### Environment Variables

Create a `.env` file for custom configuration:

```env
MODEL_PATH=models/step_detection_model.keras
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/
```

### Test Individual Components

```bash
# Test the detector
python tests/test_step_detection.py

# Test the API
python test_websocket.py

# Test real-time detection
python test_real_time_detection.py
```

## 📚 Data Format

### Input Data

The system expects sensor data with 6 features:

- `accel_x`, `accel_y`, `accel_z`: Accelerometer readings (m/s²)
- `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope readings (rad/s)

### Training Data Format

CSV files with columns:

```
accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,Label
1.2,-0.5,9.8,0.1,0.2,-0.1,No Label
2.5,1.8,8.2,-0.3,0.8,0.4,start
0.3,-2.1,11.1,0.5,-0.2,-0.3,end
```

**Labels:**

- `No Label`: Normal movement (not a step)
- `start`: Beginning of a step
- `end`: End of a step

## 🚀 Deployment

### Docker Deployment

1. **Build the image**:

```bash
docker build -t step-detection .
```

2. **Run the container**:

```bash
docker run -p 8000:8000 step-detection
```

### Production Deployment

For production deployment, see:

- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Documentation](docs/API.md)
- [Model Training Guide](docs/TRAINING.md)

## 🔍 Troubleshooting

### Common Issues

1. **Model not found**: Train a new model using option 1 in the CLI
2. **WebSocket connection refused**: Ensure the API server is running
3. **Low step detection accuracy**: Run threshold optimization (option 3)
4. **Import errors**: Check that all dependencies are installed

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv add --dev pytest black flake8 mypy

# Run linting
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- FastAPI team for the excellent API framework
- Contributors and testers

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Made with ❤️ for step detection and fitness tracking applications**
