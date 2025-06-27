# Step Detection using TensorFlow - Clean Modular Implementation

A clean, modular implementation of step detection using Convolutional Neural Networks (CNN) with TensorFlow.

## 📁 Project Structure

```
├── CNN_TensorFlow_Clean.ipynb    # Clean notebook - MODEL TRAINING ONLY
├── main.py                       # Main script with menu interface
├── data_utils.py                 # Data loading and processing utilities
├── model_utils.py                # Model creation, training, and evaluation
├── step_detector.py              # Real-time step detection classes
├── api.py                        # FastAPI REST API server
├── requirements.txt              # Project dependencies
├── models/                       # Saved models and metadata
│   ├── step_detection_model.keras
│   └── model_metadata.json
└── sample_data/                  # Training data folder
```

## 🎯 Features

### Clean Separation of Concerns

- **Notebook**: Only model training and evaluation
- **Python modules**: All utility functions and classes
- **API**: Production-ready REST API
- **Main script**: Interactive interface

### Core Components

1. **Data Processing** (`data_utils.py`)

   - Load sensor data from CSV files
   - Process step annotations
   - Prepare data for training

2. **Model Utilities** (`model_utils.py`)

   - Create CNN architecture
   - Train with callbacks and optimization
   - Evaluate model performance
   - Save/load models with metadata

3. **Real-time Detection** (`step_detector.py`)

   - `StepDetector`: Comprehensive step tracking
   - `SimpleStepCounter`: Basic step counting
   - Session management and data export

4. **REST API** (`api.py`)
   - FastAPI implementation
   - Real-time step detection endpoints
   - Model information and health checks

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (Notebook)

Open `CNN_TensorFlow_Clean.ipynb` and run all cells to train the model.

### 3. Interactive Interface

```bash
python main.py
```

### 4. Start API Server

```bash
python api.py
```

API documentation: http://localhost:8000/docs

## 📊 Notebook Usage

The clean notebook (`CNN_TensorFlow_Clean.ipynb`) focuses only on:

- Data loading
- Model creation and training
- Performance evaluation
- Model saving

No verbose descriptions, just clean code.

## 🔧 API Endpoints

- `POST /detect_step` - Detect steps from sensor data
- `GET /step_count` - Get current step count
- `POST /reset_count` - Reset step count
- `GET /session_summary` - Get session summary
- `GET /model_info` - Get model information
- `GET /health` - Health check

## 💻 Example Usage

### Real-time Detection

```python
from step_detector import StepDetector

detector = StepDetector("models/step_detection_model.keras")
result = detector.process_reading(1.2, -0.5, 9.8, 0.1, 0.2, -0.1)
print(f"Step detected: {result['step_start']}")
```

### Simple Counting

```python
from step_detector import SimpleStepCounter

counter = SimpleStepCounter("models/step_detection_model.keras")
step_detected = counter.process_reading(1.2, -0.5, 9.8, 0.1, 0.2, -0.1)
print(f"Total steps: {counter.get_count()}")
```

### API Client

```python
import requests

response = requests.post("http://localhost:8000/detect_step", json={
    "accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8,
    "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1
})
print(response.json())
```

## 📈 Model Format

Uses native Keras format (`.keras`) to avoid TensorFlow deprecation warnings:

- Fast loading and saving
- Future-proof
- No legacy warnings

## 🛠️ Development

### Adding New Features

1. Data processing → `data_utils.py`
2. Model architectures → `model_utils.py`
3. Detection algorithms → `step_detector.py`
4. API endpoints → `api.py`

### Testing

```bash
python main.py  # Interactive testing
python step_detector.py  # Test detection classes
python model_utils.py  # Test model utilities
```

## 📝 Notes

- Model automatically saves in native Keras format (no warnings)
- All utility functions are properly documented
- Error handling included throughout
- Production-ready API with proper validation
- Modular design allows easy extension and maintenance

## 🎯 Benefits of This Structure

1. **Clean notebook** - Focus on model training only
2. **Modular code** - Reusable components
3. **Production ready** - Proper API and error handling
4. **Easy maintenance** - Clear separation of concerns
5. **No warnings** - Uses modern TensorFlow practices
