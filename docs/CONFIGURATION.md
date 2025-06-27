# 🔧 Configuration Guide

Complete guide for configuring the Step Detection system.

## 📋 Configuration Overview

The system supports multiple configuration methods:

- 📁 **Configuration Files**: JSON/YAML config files
- 🌍 **Environment Variables**: Runtime configuration
- 🎮 **CLI Arguments**: Command-line options
- 🐍 **Python Code**: Programmatic configuration

## 🏗️ Configuration Structure

### Default Configuration (`config/default.json`)

```json
{
  "model": {
    "path": "models/step_detection_model.keras",
    "input_shape": [6],
    "num_classes": 3,
    "confidence_threshold": 0.5
  },
  "detection": {
    "start_threshold": 0.7,
    "end_threshold": 0.6,
    "min_step_interval": 0.3,
    "max_step_duration": 2.0,
    "smoothing_window": 3
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,
    "workers": 1,
    "cors_origins": ["http://localhost:3000"]
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/step_detection.log",
    "max_size": "10MB",
    "backup_count": 5
  },
  "data": {
    "raw_data_path": "data/raw",
    "processed_data_path": "data/processed",
    "batch_size": 32,
    "validation_split": 0.2
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "save_best_only": true
  }
}
```

### Environment-Specific Configs

```
config/
├── default.json           # Base configuration
├── development.json       # Development overrides
├── production.json        # Production overrides
├── testing.json          # Testing overrides
└── docker.json           # Docker-specific config
```

## ⚙️ Configuration Parameters

### 🤖 Model Configuration

| Parameter                    | Type   | Default                             | Description                      |
| ---------------------------- | ------ | ----------------------------------- | -------------------------------- |
| `model.path`                 | string | `models/step_detection_model.keras` | Path to trained model            |
| `model.input_shape`          | array  | `[6]`                               | Input tensor shape               |
| `model.num_classes`          | int    | `3`                                 | Number of output classes         |
| `model.confidence_threshold` | float  | `0.5`                               | Minimum confidence for detection |

```json
{
  "model": {
    "path": "models/step_detection_model.keras",
    "input_shape": [6],
    "num_classes": 3,
    "confidence_threshold": 0.5,
    "preprocessing": {
      "normalize": true,
      "scale_factor": 1.0,
      "clip_values": [-10.0, 10.0]
    }
  }
}
```

### 🚶‍♂️ Step Detection Configuration

| Parameter                     | Type  | Default | Description                          |
| ----------------------------- | ----- | ------- | ------------------------------------ |
| `detection.start_threshold`   | float | `0.7`   | Threshold for step start detection   |
| `detection.end_threshold`     | float | `0.6`   | Threshold for step end detection     |
| `detection.min_step_interval` | float | `0.3`   | Minimum time between steps (seconds) |
| `detection.max_step_duration` | float | `2.0`   | Maximum step duration (seconds)      |
| `detection.smoothing_window`  | int   | `3`     | Number of readings for smoothing     |

```json
{
  "detection": {
    "start_threshold": 0.7,
    "end_threshold": 0.6,
    "min_step_interval": 0.3,
    "max_step_duration": 2.0,
    "smoothing_window": 3,
    "debounce": {
      "enabled": true,
      "window": 0.1
    },
    "filtering": {
      "low_pass_cutoff": 10.0,
      "high_pass_cutoff": 0.1
    }
  }
}
```

### 🌐 API Configuration

| Parameter          | Type   | Default   | Description                |
| ------------------ | ------ | --------- | -------------------------- |
| `api.host`         | string | `0.0.0.0` | Server host address        |
| `api.port`         | int    | `8000`    | Server port                |
| `api.reload`       | bool   | `false`   | Auto-reload on changes     |
| `api.workers`      | int    | `1`       | Number of worker processes |
| `api.cors_origins` | array  | `[]`      | Allowed CORS origins       |

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,
    "workers": 1,
    "cors_origins": ["http://localhost:3000"],
    "middleware": {
      "compression": true,
      "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 60
      }
    },
    "websocket": {
      "max_connections": 100,
      "heartbeat_interval": 30
    }
  }
}
```

### 📊 Logging Configuration

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/step_detection.log",
    "max_size": "10MB",
    "backup_count": 5,
    "console": {
      "enabled": true,
      "level": "INFO"
    },
    "file_handler": {
      "enabled": true,
      "level": "DEBUG"
    },
    "structured": {
      "enabled": false,
      "format": "json"
    }
  }
}
```

## 🌍 Environment Variables

### Core Environment Variables

```bash
# Model configuration
STEP_DETECTION_MODEL_PATH=models/step_detection_model.keras
STEP_DETECTION_CONFIDENCE_THRESHOLD=0.5

# Detection thresholds
STEP_DETECTION_START_THRESHOLD=0.7
STEP_DETECTION_END_THRESHOLD=0.6
STEP_DETECTION_MIN_INTERVAL=0.3

# API configuration
STEP_DETECTION_API_HOST=0.0.0.0
STEP_DETECTION_API_PORT=8000
STEP_DETECTION_API_WORKERS=1

# Logging
STEP_DETECTION_LOG_LEVEL=INFO
STEP_DETECTION_LOG_FILE=logs/step_detection.log

# Environment
STEP_DETECTION_ENV=production
STEP_DETECTION_DEBUG=false
```

### Environment Configuration Files

#### `.env` (Development)

```bash
STEP_DETECTION_ENV=development
STEP_DETECTION_DEBUG=true
STEP_DETECTION_LOG_LEVEL=DEBUG
STEP_DETECTION_API_RELOAD=true
STEP_DETECTION_API_WORKERS=1
```

#### `.env.production` (Production)

```bash
STEP_DETECTION_ENV=production
STEP_DETECTION_DEBUG=false
STEP_DETECTION_LOG_LEVEL=INFO
STEP_DETECTION_API_RELOAD=false
STEP_DETECTION_API_WORKERS=4
STEP_DETECTION_API_HOST=0.0.0.0
STEP_DETECTION_API_PORT=8000
```

#### `.env.docker` (Docker)

```bash
STEP_DETECTION_ENV=docker
STEP_DETECTION_MODEL_PATH=/app/models/step_detection_model.keras
STEP_DETECTION_LOG_FILE=/app/logs/step_detection.log
STEP_DETECTION_DATA_PATH=/app/data
```

## 🎮 CLI Configuration

### Command Line Arguments

```bash
# Start with custom configuration
python main.py --config config/production.json

# Override specific settings
python main.py \
  --model-path models/custom_model.keras \
  --api-port 8080 \
  --log-level DEBUG \
  --start-threshold 0.8

# Environment-specific startup
python main.py --env production
python main.py --env development
python main.py --env testing
```

### CLI Help

```bash
python main.py --help
```

Output:

```
🚶‍♂️ Step Detection System

Options:
  --config PATH              Configuration file path
  --env TEXT                Environment (development/production/testing)
  --model-path PATH         Path to model file
  --api-host TEXT           API server host
  --api-port INTEGER        API server port
  --log-level TEXT          Logging level (DEBUG/INFO/WARNING/ERROR)
  --start-threshold FLOAT   Step start detection threshold
  --end-threshold FLOAT     Step end detection threshold
  --help                    Show this message and exit
```

## 🐍 Programmatic Configuration

### Using Configuration Classes

```python
from src.step_detection.config import (
    ModelConfig,
    DetectionConfig,
    APIConfig,
    LoggingConfig,
    StepDetectionConfig
)

# Create configuration
config = StepDetectionConfig(
    model=ModelConfig(
        path="models/step_detection_model.keras",
        confidence_threshold=0.6
    ),
    detection=DetectionConfig(
        start_threshold=0.8,
        end_threshold=0.7
    ),
    api=APIConfig(
        host="localhost",
        port=8080
    ),
    logging=LoggingConfig(
        level="DEBUG",
        file="logs/debug.log"
    )
)

# Use with detector
from src.step_detection.core.detector import StepDetector
detector = StepDetector(config=config)
```

### Runtime Configuration Updates

```python
# Update thresholds during runtime
detector.update_thresholds(
    start_threshold=0.75,
    end_threshold=0.65
)

# Update API settings
api_server.update_config(
    cors_origins=["https://myapp.com"],
    rate_limit=120
)
```

## 🏭 Production Configuration

### Production Settings (`config/production.json`)

```json
{
  "model": {
    "path": "/app/models/step_detection_model.keras",
    "confidence_threshold": 0.6
  },
  "detection": {
    "start_threshold": 0.75,
    "end_threshold": 0.65,
    "min_step_interval": 0.25
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "reload": false,
    "cors_origins": ["https://yourdomain.com"],
    "middleware": {
      "compression": true,
      "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 100
      }
    }
  },
  "logging": {
    "level": "INFO",
    "file": "/app/logs/step_detection.log",
    "max_size": "50MB",
    "backup_count": 10,
    "structured": {
      "enabled": true,
      "format": "json"
    }
  },
  "monitoring": {
    "metrics_enabled": true,
    "health_check_interval": 30,
    "performance_tracking": true
  },
  "security": {
    "api_key_required": true,
    "rate_limiting": true,
    "cors_strict": true
  }
}
```

### Docker Configuration

```dockerfile
# Dockerfile configuration
ENV STEP_DETECTION_ENV=production
ENV STEP_DETECTION_MODEL_PATH=/app/models/step_detection_model.keras
ENV STEP_DETECTION_LOG_LEVEL=INFO
ENV STEP_DETECTION_API_WORKERS=4
```

### Kubernetes Configuration

```yaml
# k8s-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: step-detection-config
data:
  STEP_DETECTION_ENV: "production"
  STEP_DETECTION_API_HOST: "0.0.0.0"
  STEP_DETECTION_API_PORT: "8000"
  STEP_DETECTION_LOG_LEVEL: "INFO"
  config.json: |
    {
      "model": {
        "path": "/app/models/step_detection_model.keras"
      },
      "api": {
        "workers": 4
      }
    }
```

## 🔧 Advanced Configuration

### Custom Preprocessors

```python
# config/custom_processors.py
class CustomPreprocessor:
    def __init__(self, config):
        self.scale_factor = config.get('scale_factor', 1.0)

    def process(self, data):
        return data * self.scale_factor

# Register in configuration
{
  "preprocessing": {
    "custom_processor": {
      "class": "config.custom_processors.CustomPreprocessor",
      "params": {
        "scale_factor": 1.5
      }
    }
  }
}
```

### Feature Flags

```json
{
  "features": {
    "real_time_optimization": true,
    "advanced_filtering": false,
    "experimental_detection": false,
    "performance_monitoring": true,
    "debug_mode": false
  }
}
```

### A/B Testing Configuration

```json
{
  "ab_testing": {
    "enabled": true,
    "experiments": {
      "threshold_optimization": {
        "enabled": true,
        "variants": {
          "control": { "start_threshold": 0.7 },
          "treatment": { "start_threshold": 0.75 }
        },
        "split_ratio": 0.5
      }
    }
  }
}
```

## 🔒 Security Configuration

### API Security

```json
{
  "security": {
    "api_key": {
      "enabled": true,
      "header_name": "X-API-Key",
      "required_for": ["POST", "PUT", "DELETE"]
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst": 10
    },
    "cors": {
      "enabled": true,
      "origins": ["https://yourdomain.com"],
      "methods": ["GET", "POST"],
      "headers": ["Content-Type", "Authorization"]
    },
    "https": {
      "redirect": true,
      "hsts": true
    }
  }
}
```

## 📊 Monitoring Configuration

### Metrics and Monitoring

```json
{
  "monitoring": {
    "metrics": {
      "enabled": true,
      "endpoint": "/metrics",
      "include_system_metrics": true
    },
    "health_checks": {
      "enabled": true,
      "endpoint": "/health",
      "checks": ["model_loaded", "api_responsive", "disk_space"]
    },
    "alerting": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/...",
      "thresholds": {
        "error_rate": 0.05,
        "response_time": 1000
      }
    }
  }
}
```

## 🔄 Configuration Validation

### Schema Validation

```python
# config/schema.py
from pydantic import BaseModel, validator

class ModelConfig(BaseModel):
    path: str
    confidence_threshold: float

    @validator('confidence_threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v

class DetectionConfig(BaseModel):
    start_threshold: float
    end_threshold: float
    min_step_interval: float

    @validator('start_threshold', 'end_threshold')
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Thresholds must be between 0 and 1')
        return v
```

### Configuration Testing

```python
# tests/test_config.py
def test_config_validation():
    """Test configuration validation."""

def test_config_loading():
    """Test loading from different sources."""

def test_environment_override():
    """Test environment variable overrides."""
```

---

**🔧 Configuration Complete! Your system is now properly configured! ⚙️**
