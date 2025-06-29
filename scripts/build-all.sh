#!/bin/bash
set -e

# ════════════════════════════════════════════════
# ║      STEP DETECTION: UNIFIED BUILD SCRIPT     ║
# ║      (Supports both dev and prod builds)      ║
# ════════════════════════════════════════════════

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default variables
DOCKER_USER="bibektimilsina000"
IMAGE_NAME="step-detection-ai"
TAG="latest"
PUSH=false
MODE="prod"  # prod or dev

# Usage
usage() {
  echo -e "${BOLD}Usage:${NC} $0 [MODE] [OPTIONS]"
  echo -e "${BOLD}Modes:${NC}"
  echo "  prod         Build production image (default)"
  echo "  dev          Build development image"
  echo -e "${BOLD}Options:${NC}"
  echo -e "  -u, --user USER      Docker Hub username (prod only)"
  echo -e "  -i, --image NAME     Image name"
  echo -e "  -t, --tag TAG        Image tag (default: latest)"
  echo -e "  --push               Push to Docker Hub (prod only)"
  echo -e "  --train              Force model training before build"
  echo -e "  -h, --help           Show this help"
  echo -e ""
  echo -e "${BOLD}Examples:${NC}"
  echo -e "  $0 prod --push -u myuser     # Build and push production image"
  echo -e "  $0 dev                       # Build development image"
  echo -e "  $0 prod --train              # Train model and build production"
  exit 1
}

# Utility functions
print_info() {
  echo -e "${CYAN}ℹ️  $1${NC}"
}

print_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
  echo -e "${RED}❌ $1${NC}"
}

print_section() {
  echo -e "\n${BLUE}═════════════════════════════════════${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}═════════════════════════════════════${NC}"
}

# Check for mode in first argument
if [[ $# -gt 0 && ("$1" == "prod" || "$1" == "dev") ]]; then
    MODE="$1"
    shift
fi

# Parse options
FORCE_TRAIN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--user)
      DOCKER_USER="$2"; shift 2;;
    -i|--image)
      IMAGE_NAME="$2"; shift 2;;
    -t|--tag)
      TAG="$2"; shift 2;;
    --push)
      PUSH=true; shift;;
    --train)
      FORCE_TRAIN=true; shift;;
    -h|--help)
      usage;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"; usage;;
  esac
done

# Set up variables based on mode
if [[ "$MODE" == "dev" ]]; then
  IMAGE_NAME="step-detection-dev"
  DOCKERFILE="docker/Dockerfile.dev"
  PUSH=false  # Never push dev images
else
  DOCKERFILE="docker/Dockerfile.prod"
fi

FULL_TAG="$DOCKER_USER/$IMAGE_NAME:$TAG"
if [[ "$MODE" == "dev" ]]; then
  FULL_TAG="$IMAGE_NAME:$TAG"  # No user prefix for dev
fi

print_section "STEP DETECTION: ${MODE^^} BUILD"
print_info "Mode: $MODE"
print_info "Image: $FULL_TAG"
print_info "Dockerfile: $DOCKERFILE"
if [[ "$MODE" == "prod" && "$PUSH" == true ]]; then
  print_info "Will push to Docker Hub: Yes"
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
  print_error "Dockerfile not found at $DOCKERFILE!"
  print_info "Expected modular project structure with docker/ directory"
  exit 1
fi

# Check if pyproject.toml exists (indicates proper project setup)
if [[ ! -f "pyproject.toml" ]]; then
  print_error "pyproject.toml not found! This script requires the new modular structure."
  exit 1
fi

# Model training check (only for production builds)
if [[ "$MODE" == "prod" ]]; then
  print_section "MODEL PREPARATION"
  
  MODEL_PATH="models/step_detection_model.keras"
  if [[ "$FORCE_TRAIN" == true ]] || [[ ! -f "$MODEL_PATH" ]]; then
    if [[ "$FORCE_TRAIN" == true ]]; then
      print_info "Force training requested..."
    else
      print_warning "Pre-trained model not found at $MODEL_PATH"
    fi
    
    print_info "Training step detection model using main.py..."
    mkdir -p models
    
    python main.py train
    
    if [[ $? -ne 0 ]]; then
      print_error "Model training failed!"
      exit 1
    fi
    
    print_success "Model training completed successfully!"
  else
    print_success "Pre-trained model found: $MODEL_PATH"
  fi
fi

# Build the image
print_section "DOCKER BUILD"
print_info "Building Docker image with $DOCKERFILE..."

docker build -f "$DOCKERFILE" -t "$FULL_TAG" .

if [[ $? -ne 0 ]]; then
  print_error "Docker build failed!"
  exit 1
fi

print_success "Image built successfully: $FULL_TAG"

# Push to Docker Hub (only for production)
if [[ "$MODE" == "prod" && "$PUSH" == true ]]; then
  print_section "DOCKER PUSH"
  
  # Validate Docker Hub username
  if [[ "$DOCKER_USER" == "your_dockerhub_username" ]]; then
    print_error "Please set your Docker Hub username!"
    print_info "Use: $0 $MODE -u your_actual_username --push"
    exit 1
  fi
  
  print_info "Pushing to Docker Hub..."
  
  # Check if Docker token is provided
  if [[ -z "$DOCKER_TOKEN" ]]; then
    print_warning "DOCKER_TOKEN environment variable not set"
    print_info "Please set it with: export DOCKER_TOKEN=your_token"
    print_info "Or login manually with: docker login"
    
    # Try to use existing docker login
    if ! docker info | grep -q "Username"; then
      print_error "Not logged in to Docker Hub. Please login first:"
      print_info "   docker login"
      exit 1
    else
      print_success "Using existing Docker login"
    fi
  else
    # Login using environment variable
    echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin
    if [ $? -ne 0 ]; then
      print_error "Docker login failed!"
      exit 1
    fi
    print_success "Docker login successful"
  fi
  
  docker push "$FULL_TAG"
  if [[ $? -ne 0 ]]; then
    print_error "Docker push failed!"
    exit 1
  fi
  
  print_success "Image pushed successfully: $FULL_TAG"
fi

# Final summary
print_section "BUILD COMPLETE"

if [[ "$MODE" == "dev" ]]; then
  print_success "Development image built: $FULL_TAG"
  print_info "Next steps:"
  print_info "  • Start dev environment: ./scripts/build-dev.sh up"
  print_info "  • Or use docker-compose: docker-compose -f docker/docker-compose.dev.yml up"
  print_info "  • Access API at: http://localhost:8000"
  print_info "  • Hot-reload enabled for development"
else
  print_success "Production image built: $FULL_TAG"
  if [[ "$PUSH" == true ]]; then
    print_info "Image pushed to Docker Hub and ready for deployment!"
    print_info "Deploy with: ./scripts/deploy.sh deploy -u $DOCKER_USER"
  else
    print_info "Next steps:"
    print_info "  • Push to hub: $0 prod --push -u $DOCKER_USER"
    print_info "  • Deploy locally: ./scripts/deploy.sh local"
    print_info "  • Deploy remote: ./scripts/deploy.sh deploy -u $DOCKER_USER"
  fi
fi

print_info "View images: docker images | grep step-detection"
