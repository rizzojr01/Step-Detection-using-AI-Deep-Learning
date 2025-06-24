# Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Step Detection API using Docker.

## 🚀 Quick Start

### Option 1: Simple Docker Run

```bash
# Build and run in one command
./build-and-run.sh
```

### Option 2: Docker Compose (Recommended)

```bash
# Development
docker-compose up -d

# Production with Nginx
docker-compose -f docker-compose.prod.yml up -d
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -t step-detection-api:latest .

# Run the container
docker run -d \
    --name step-detection-api \
    -p 8000:8000 \
    --restart unless-stopped \
    step-detection-api:latest
```

## 📁 Docker Files Overview

| File                      | Purpose                              |
| ------------------------- | ------------------------------------ |
| `Dockerfile`              | Development Docker image             |
| `Dockerfile.prod`         | Production multi-stage build         |
| `docker-compose.yml`      | Development compose setup            |
| `docker-compose.prod.yml` | Production with Nginx                |
| `.dockerignore`           | Files to exclude from Docker context |
| `nginx.conf`              | Nginx reverse proxy configuration    |
| `build-and-run.sh`        | Quick build and deploy script        |

## 🔧 Configuration Options

### Environment Variables

```bash
# Set Python environment
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# Custom port (default: 8000)
PORT=8000

# Model configuration
MODEL_THRESHOLD=0.3
```

### Volume Mounts

```bash
# Mount sample data for updates without rebuild
-v ./Sample\ Data:/app/Sample\ Data:ro

# Mount logs directory
-v ./logs:/app/logs
```

## 🌐 Network Access

### Access URLs

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **WebSocket**: ws://localhost:8000/ws/realtime

### Mobile Device Access

Replace `localhost` with your machine's IP address:

- **API**: http://192.168.18.179:8000/docs
- **WebSocket**: ws://192.168.18.179:8000/ws/realtime

## 🛠️ Management Commands

### Container Management

```bash
# View running containers
docker ps

# View logs
docker logs -f step-detection-api

# Stop container
docker stop step-detection-api

# Remove container
docker rm step-detection-api

# Restart container
docker restart step-detection-api
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi step-detection-api:latest

# Build without cache
docker build --no-cache -t step-detection-api:latest .
```

### Docker Compose Management

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Scale services (if needed)
docker-compose up -d --scale step-detection-api=3
```

## 🏥 Health Monitoring

### Health Checks

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' step-detection-api

# Manual health check
curl http://localhost:8000/health
```

### Monitoring Commands

```bash
# Resource usage
docker stats step-detection-api

# Container info
docker inspect step-detection-api

# Process list
docker exec step-detection-api ps aux
```

## 🔒 Security Considerations

### Production Security

1. **Non-root user**: Container runs as `app` user
2. **Read-only mounts**: Sample data mounted read-only
3. **Nginx proxy**: Rate limiting and security headers
4. **Health checks**: Automatic restart on failure

### Network Security

```bash
# Create custom network
docker network create --driver bridge step-detection-net

# Run with custom network
docker run -d \
    --name step-detection-api \
    --network step-detection-net \
    -p 8000:8000 \
    step-detection-api:latest
```

## 🚀 Production Deployment

### With Nginx Reverse Proxy

```bash
# Deploy with production settings
docker-compose -f docker-compose.prod.yml up -d

# Access via Nginx (port 80)
curl http://localhost/health
```

### Cloud Deployment (AWS/GCP/Azure)

```bash
# Build for multi-platform
docker buildx build --platform linux/amd64,linux/arm64 -t step-detection-api:latest .

# Tag for registry
docker tag step-detection-api:latest your-registry/step-detection-api:v1.0.0

# Push to registry
docker push your-registry/step-detection-api:v1.0.0
```

## 🧪 Testing Docker Deployment

### Test WebSocket Connection

```bash
# Run the test script
uv run python test_ip_websocket.py

# Test specific endpoint
curl -X POST http://localhost:8000/detect_step \
  -H "Content-Type: application/json" \
  -d '{"accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8, "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1}'
```

### Performance Testing

```bash
# Stress test with multiple connections
docker run --rm -it \
  --network container:step-detection-api \
  appropriate/curl \
  -X POST http://localhost:8000/detect_step \
  -H "Content-Type: application/json" \
  -d '{"accel_x": 1.2, "accel_y": -0.5, "accel_z": 9.8, "gyro_x": 0.1, "gyro_y": 0.2, "gyro_z": -0.1}'
```

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs step-detection-api

# Check if port is in use
lsof -i :8000

# Remove and recreate
docker rm -f step-detection-api
./build-and-run.sh
```

#### Can't Connect from Mobile Device

```bash
# Check if container is listening on all interfaces
docker exec step-detection-api netstat -tlnp

# Test from another container
docker run --rm curlimages/curl:latest \
  curl -f http://host.docker.internal:8000/health
```

#### Performance Issues

```bash
# Check resource usage
docker stats step-detection-api

# Increase memory limit
docker run -d \
    --name step-detection-api \
    --memory=2g \
    --cpus=2 \
    -p 8000:8000 \
    step-detection-api:latest
```

## 📊 Performance Optimization

### Resource Limits

```yaml
# In docker-compose.yml
services:
  step-detection-api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2"
        reservations:
          memory: 1G
          cpus: "1"
```

### Multi-stage Builds

Use `Dockerfile.prod` for smaller production images:

```bash
docker build -f Dockerfile.prod -t step-detection-api:prod .
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t step-detection-api:latest .
      - name: Run tests
        run: docker run --rm step-detection-api:latest python -m pytest
      - name: Deploy
        run: docker-compose up -d
```

This comprehensive Docker setup provides:

- ✅ Development and production configurations
- ✅ Nginx reverse proxy with security features
- ✅ Health checks and monitoring
- ✅ Easy deployment scripts
- ✅ Multi-platform support
- ✅ Security best practices
- ✅ Comprehensive documentation
