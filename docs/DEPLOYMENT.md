# Deployment Guide

## Overview

This guide covers deploying the Step Detection API in various environments, from development to production.

## Deployment Options

### 1. Local Development

**Quick Start:**

```bash
python main.py  # Choose option 4
```

**Manual Start:**

```bash
uvicorn src.step_detection.api.api:app --host 0.0.0.0 --port 8000 --reload
```

**Configuration:**

- Host: `0.0.0.0`
- Port: `8000`
- Reload: `True` (development only)

### 2. Production Server

**Basic Production:**

```bash
uvicorn src.step_detection.api.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**With Gunicorn:**

```bash
gunicorn src.step_detection.api.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.step_detection.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and Run

```bash
# Build image
docker build -t step-detection .

# Run container
docker run -d \
    --name step-detection-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    step-detection
```

#### Docker Compose

```yaml
version: "3.8"

services:
  step-detection-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=models/step_detection_model.keras
      - LOG_LEVEL=INFO
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - step-detection-api
    restart: unless-stopped
```

### 4. Cloud Deployment

#### AWS EC2

1. **Launch EC2 Instance**

   - Amazon Linux 2 or Ubuntu 20.04
   - t3.medium or larger (2 vCPU, 4GB RAM minimum)
   - Security group allowing port 8000

2. **Setup Script**

```bash
#!/bin/bash
# install-step-detection.sh

# Update system
sudo yum update -y  # or apt update for Ubuntu

# Install Docker
sudo yum install -y docker  # or apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone <your-repo-url>
cd Step-Detection-using-AI-Deep-Learning

# Deploy with Docker Compose
docker-compose up -d
```

3. **Security Configuration**

```bash
# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable

# Setup SSL (Let's Encrypt)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

#### Google Cloud Platform

1. **Cloud Run Deployment**

```bash
# Build and push to Container Registry
docker tag step-detection gcr.io/your-project/step-detection
docker push gcr.io/your-project/step-detection

# Deploy to Cloud Run
gcloud run deploy step-detection \
    --image gcr.io/your-project/step-detection \
    --platform managed \
    --region us-central1 \
    --port 8000 \
    --memory 2Gi \
    --cpu 1 \
    --allow-unauthenticated
```

2. **App Engine Deployment**

```yaml
# app.yaml
runtime: python39

env_variables:
  MODEL_PATH: models/step_detection_model.keras

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name step-detection-rg --location eastus

# Deploy container
az container create \
    --resource-group step-detection-rg \
    --name step-detection-api \
    --image step-detection:latest \
    --ports 8000 \
    --memory 2 \
    --cpu 1 \
    --dns-name-label step-detection-api
```

## Production Configuration

### Environment Variables

```bash
# .env file
MODEL_PATH=models/step_detection_model.keras
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=100
TIMEOUT=60

# Security
API_KEY_REQUIRED=true
CORS_ORIGINS=["https://your-frontend.com"]
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Logging Configuration

```python
# config/logging.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/step_detection.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'INFO',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file'],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Nginx Configuration

```nginx
# nginx.conf
upstream step_detection {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://step_detection;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;
        proxy_pass http://step_detection;
    }
}
```

## Monitoring and Observability

### Health Checks

```python
# Enhanced health check
@app.get("/health")
async def health_check():
    checks = {
        "api": "healthy",
        "model_loaded": detector is not None,
        "memory_usage": get_memory_usage(),
        "disk_space": get_disk_space(),
        "timestamp": datetime.utcnow().isoformat()
    }

    # Determine overall status
    status = "healthy" if all([
        checks["model_loaded"],
        checks["memory_usage"] < 90,  # Less than 90% memory usage
        checks["disk_space"] > 1024   # More than 1GB disk space
    ]) else "unhealthy"

    checks["status"] = status

    return checks
```

### Metrics Collection

```python
# Add to your FastAPI app
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram

step_detection_requests = Counter(
    'step_detection_requests_total',
    'Total step detection requests',
    ['method', 'endpoint']
)

step_detection_duration = Histogram(
    'step_detection_request_duration_seconds',
    'Step detection request duration'
)
```

### Logging Best Practices

```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        "request_processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
        user_agent=request.headers.get("user-agent"),
        remote_addr=request.client.host
    )

    return response
```

## Security Considerations

### API Security

```python
# API Key authentication
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# Apply to endpoints
@app.post("/detect_step")
async def detect_step(
    reading: SensorReading,
    api_key: str = Depends(verify_api_key)
):
    # ... endpoint logic
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/detect_step")
@limiter.limit("60/minute")
async def detect_step(request: Request, reading: SensorReading):
    # ... endpoint logic
```

### Input Validation

```python
from pydantic import validator

class SensorReading(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

    @validator('accel_x', 'accel_y', 'accel_z')
    def validate_acceleration(cls, v):
        if not -50 <= v <= 50:
            raise ValueError('Acceleration values must be between -50 and 50')
        return v

    @validator('gyro_x', 'gyro_y', 'gyro_z')
    def validate_gyroscope(cls, v):
        if not -10 <= v <= 10:
            raise ValueError('Gyroscope values must be between -10 and 10')
        return v
```

## Performance Optimization

### Model Optimization

```python
# TensorFlow Lite conversion for edge deployment
import tensorflow as tf

def convert_to_tflite(model_path, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

# Quantization for smaller model size
def quantized_conversion(model_path, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

### Caching

```python
from functools import lru_cache
import redis

# Redis cache for model predictions
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_prediction(sensor_data, prediction):
    key = hash(tuple(sensor_data))
    redis_client.setex(key, 300, json.dumps(prediction))  # 5 min TTL

def get_cached_prediction(sensor_data):
    key = hash(tuple(sensor_data))
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None
```

### Load Balancing

```yaml
# docker-compose.yml for multiple instances
version: "3.8"

services:
  step-detection-api-1:
    build: .
    ports:
      - "8001:8000"
    # ... other config

  step-detection-api-2:
    build: .
    ports:
      - "8002:8000"
    # ... other config

  load-balancer:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - step-detection-api-1
      - step-detection-api-2
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**

   - Monitor TensorFlow memory allocation
   - Use model quantization
   - Implement connection pooling

2. **Slow Response Times**

   - Profile model inference time
   - Add request caching
   - Optimize data preprocessing

3. **WebSocket Connection Issues**

   - Check firewall settings
   - Verify WebSocket support in proxy
   - Monitor connection limits

4. **Model Loading Failures**
   - Verify model file permissions
   - Check TensorFlow version compatibility
   - Validate model file integrity

### Debugging Tools

```python
# Add debug middleware
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    if os.getenv("DEBUG"):
        body = await request.body()
        print(f"Request: {request.method} {request.url}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Body: {body}")

    response = await call_next(request)
    return response
```

### Performance Monitoring

```bash
# System monitoring
htop
iostat -x 1
netstat -tulpn

# Application monitoring
curl http://localhost:8000/health
curl http://localhost:8000/metrics  # If Prometheus enabled

# Docker monitoring
docker stats
docker logs step-detection-api
```

## Backup and Recovery

### Model Backup

```bash
#!/bin/bash
# backup-models.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/models"

mkdir -p $BACKUP_DIR

# Backup current model
cp models/step_detection_model.keras $BACKUP_DIR/model_$DATE.keras
cp models/model_metadata.json $BACKUP_DIR/metadata_$DATE.json

# Keep only last 10 backups
ls -t $BACKUP_DIR/model_*.keras | tail -n +11 | xargs rm -f
ls -t $BACKUP_DIR/metadata_*.json | tail -n +11 | xargs rm -f

echo "Model backup completed: $DATE"
```

### Database Backup (if applicable)

```bash
#!/bin/bash
# backup-data.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/data"

mkdir -p $BACKUP_DIR

# Backup training data
tar -czf $BACKUP_DIR/training_data_$DATE.tar.gz data/raw/
tar -czf $BACKUP_DIR/processed_data_$DATE.tar.gz data/processed/

echo "Data backup completed: $DATE"
```

### Automated Backups

```yaml
# Add to docker-compose.yml
services:
  backup:
    image: alpine:latest
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./backups:/backups
      - ./scripts:/scripts
    command: >
      sh -c "
        while true; do
          sleep 86400  # 24 hours
          /scripts/backup-models.sh
          /scripts/backup-data.sh
        done
      "
    restart: unless-stopped
```
