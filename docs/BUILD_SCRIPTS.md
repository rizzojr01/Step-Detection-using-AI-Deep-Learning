# Build and Deployment Scripts Guide

This document explains all the build and deployment scripts available in the Step Detection AI project.

## 📁 Scripts Overview

### Core Scripts

| Script           | Purpose                                     | Environment |
| ---------------- | ------------------------------------------- | ----------- |
| `quick-setup.sh` | 🚀 **Start here!** Interactive setup wizard | Both        |
| `build-all.sh`   | Unified build script for dev/prod           | Both        |
| `build-dev.sh`   | Development environment manager             | Development |
| `build.sh`       | Production build and push                   | Production  |
| `deploy.sh`      | Production deployment manager               | Production  |

---

## 🚀 Quick Start

**New to the project?** Start here:

```bash
./scripts/quick-setup.sh
```

This interactive script will:

- Check your system dependencies
- Guide you through setup options
- Get you running in minutes

---

## 📋 Detailed Script Reference

### 1. `quick-setup.sh` - Interactive Setup Wizard

**Purpose**: Get new users up and running quickly with an interactive setup process.

```bash
./scripts/quick-setup.sh
```

**Features**:

- ✅ Dependency checking (Docker, Python)
- 🎯 Multiple setup options (Docker dev/prod, local dev)
- 📚 Command reference guide
- 🎉 Post-setup guidance

**Best for**: First-time setup, new team members

---

### 2. `build-all.sh` - Unified Build Script

**Purpose**: Single script to handle both development and production builds.

```bash
# Development build
./scripts/build-all.sh dev

# Production build
./scripts/build-all.sh prod

# Production build with training
./scripts/build-all.sh prod --train

# Production build and push
./scripts/build-all.sh prod --push -u your_dockerhub_user
```

**Options**:

- `dev` - Build development image with hot-reload
- `prod` - Build production image (default)
- `--train` - Force model training before build
- `--push` - Push to Docker Hub (production only)
- `-u, --user` - Docker Hub username
- `-i, --image` - Custom image name
- `-t, --tag` - Custom tag

**Features**:

- 🔄 Automatic model training (if needed)
- 🐳 Different Dockerfiles for dev/prod
- 🔐 Docker Hub authentication
- ✅ Comprehensive error checking

---

### 3. `build-dev.sh` - Development Environment Manager

**Purpose**: Manage development environment with hot-reload and debugging features.

```bash
# Start development environment
./scripts/build-dev.sh up

# View logs
./scripts/build-dev.sh logs

# Access container shell
./scripts/build-dev.sh shell

# Run tests
./scripts/build-dev.sh test

# Stop environment
./scripts/build-dev.sh down
```

**Commands**:

- `build` - Build development image
- `up` - Start dev environment (auto-builds if needed)
- `down` - Stop dev environment
- `logs` - Show real-time logs
- `shell` - Access container shell
- `test` - Run tests in container

**Features**:

- 🔥 Hot-reload enabled
- 📝 Real-time logging
- 🧪 Integrated testing
- 🐚 Shell access for debugging

---

### 4. `build.sh` - Production Build & Push

**Purpose**: Build production-ready images and push to Docker Hub.

```bash
# Build and push with default settings
./scripts/build.sh

# Custom user and tag
./scripts/build.sh -u myuser -t v1.0

# Build only (no push)
./scripts/build.sh --no-push
```

**Options**:

- `-u, --user` - Docker Hub username
- `-i, --image` - Image name
- `-t, --tag` - Image tag
- `--no-push` - Build only, don't push

**Features**:

- 🧠 Automatic model training
- 🐳 Production Dockerfile
- 🔐 Docker Hub authentication
- ✅ Build validation

---

### 5. `deploy.sh` - Production Deployment Manager

**Purpose**: Deploy and manage production instances.

```bash
# Deploy locally with docker-compose
./scripts/deploy.sh local

# Deploy to remote server
./scripts/deploy.sh deploy -u your_dockerhub_user

# Build image
./scripts/deploy.sh build

# Push to Docker Hub
./scripts/deploy.sh push

# View logs
./scripts/deploy.sh logs

# Stop deployment
./scripts/deploy.sh stop
```

**Commands**:

- `build` - Build production image
- `local` - Deploy locally with docker-compose
- `push` - Push image to Docker Hub
- `deploy` - Deploy to remote server
- `stop` - Stop running container
- `logs` - Show container logs

**Options**:

- `-u, --user` - Docker Hub username
- `-i, --image` - Image name
- `-t, --tag` - Image tag
- `-p, --port` - Host port

---

## 🔄 Common Workflows

### Development Workflow

```bash
# 1. First time setup
./scripts/quick-setup.sh

# 2. Start development
./scripts/build-dev.sh up

# 3. Make code changes (hot-reload active)

# 4. Run tests
./scripts/build-dev.sh test

# 5. View logs
./scripts/build-dev.sh logs

# 6. Stop when done
./scripts/build-dev.sh down
```

### Production Deployment Workflow

```bash
# 1. Build and test locally
./scripts/build-all.sh prod --train

# 2. Test locally
./scripts/deploy.sh local

# 3. Push to Docker Hub
./scripts/build-all.sh prod --push -u your_user

# 4. Deploy to server
./scripts/deploy.sh deploy -u your_user

# 5. Monitor
./scripts/deploy.sh logs
```

### Quick Testing Workflow

```bash
# Start development environment
./scripts/build-dev.sh up

# Test API
curl http://localhost:8000/health

# Test WebSocket
python test_websocket.py

# Stop
./scripts/build-dev.sh down
```

---

## 🐳 Docker Configuration

### Development (`docker/Dockerfile.dev`)

- Based on Python 3.12
- Includes dev dependencies
- Hot-reload with volume mounts
- Debug-friendly

### Production (`docker/Dockerfile.prod`)

- Multi-stage optimized build
- Minimal runtime image
- Security hardening
- Non-root user

### Docker Compose Files

- `docker/docker-compose.dev.yml` - Development stack
- `docker/docker-compose.prod.yml` - Production stack with Nginx, Redis, monitoring

---

## 🔧 Environment Variables

### Build Scripts

- `DOCKER_TOKEN` - Docker Hub authentication token
- `DOCKER_USER` - Default Docker Hub username

### Runtime

- `STEP_DETECTION_ENV` - Environment (development/production)
- `STEP_DETECTION_API_WORKERS` - Number of API workers
- `STEP_DETECTION_LOG_LEVEL` - Logging level

---

## 🚨 Troubleshooting

### Common Issues

**Docker not running**:

```bash
# Start Docker Desktop or daemon
docker info
```

**Permission denied**:

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

**Model not found**:

```bash
# Force model training
./scripts/build-all.sh prod --train
```

**Port conflicts**:

```bash
# Use custom port
./scripts/deploy.sh local -p 9000
```

### Getting Help

1. Check `docs/TROUBLESHOOTING.md`
2. Run health check: `curl http://localhost:8000/health`
3. View logs: `./scripts/build-dev.sh logs`
4. Access shell: `./scripts/build-dev.sh shell`

---

## 📚 Related Documentation

- [README.md](../README.md) - Project overview
- [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) - Deployment guide
- [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md) - Development guide
- [QUICK_START.md](../QUICK_START.md) - Quick reference

---

## 🎯 Summary

- **New users**: Start with `./scripts/quick-setup.sh`
- **Development**: Use `./scripts/build-dev.sh up`
- **Production**: Use `./scripts/build-all.sh prod` → `./scripts/deploy.sh local`
- **Deployment**: Use `./scripts/deploy.sh deploy`

All scripts include help: `./scripts/script-name.sh --help`
