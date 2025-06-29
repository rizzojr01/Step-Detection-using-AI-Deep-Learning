#!/bin/bash
set -e

# ════════════════════════════════════════════════
# ║      STEP DETECTION: BUILD & PUSH SCRIPT      ║
# ║      (Updated for new modular structure)      ║
# ║      Uses docker/Dockerfile.prod              ║
# ════════════════════════════════════════════════

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configurable variables
DOCKER_USER="bibektimilsina000"   # Your Docker Hub username
IMAGE_NAME="step-detection-ai"
TAG="latest"
PUSH=true

# Usage
usage() {
  echo -e "${BOLD}Usage:${NC} $0 [-u dockerhub_user] [-i image_name] [-t tag] [--no-push]"
  echo -e "${BOLD}Options:${NC}"
  echo -e "  -u, --user USER      Docker Hub username"
  echo -e "  -i, --image NAME     Image name (default: step-detection-ai)"
  echo -e "  -t, --tag TAG        Image tag (default: latest)"
  echo -e "  --no-push            Build only, don't push to Docker Hub"
  echo -e "  -h, --help           Show this help"
  echo -e ""
  echo -e "${BOLD}Examples:${NC}"
  echo -e "  $0 -u myusername"
  echo -e "  $0 -u myusername -t v1.0"
  echo -e "  $0 --no-push         # Build locally only"
  exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--user)
      DOCKER_USER="$2"; shift 2;;
    -i|--image)
      IMAGE_NAME="$2"; shift 2;;
    -t|--tag)
      TAG="$2"; shift 2;;
    --no-push)
      PUSH=false; shift;;
    -h|--help)
      usage;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"; usage;;
  esac
done

# Validate Docker Hub username
if [[ "$DOCKER_USER" == "your_dockerhub_username" ]]; then
  echo -e "${RED}❌ Please set your Docker Hub username!${NC}"
  echo -e "Edit the script or use: $0 -u your_actual_username"
  exit 1
fi

FULL_TAG="$DOCKER_USER/$IMAGE_NAME:$TAG"

printf "${BLUE}═══════════════════════════════════════${NC}\n"
printf "${BLUE}  BUILDING: $FULL_TAG${NC}\n"
printf "${BLUE}═══════════════════════════════════════${NC}\n"

# Check if production Dockerfile exists
DOCKERFILE_PATH="docker/Dockerfile.prod"
if [[ ! -f "$DOCKERFILE_PATH" ]]; then
  echo -e "${RED}❌ Production Dockerfile not found at $DOCKERFILE_PATH!${NC}"
  echo -e "Expected modular project structure with docker/ directory"
  exit 1
fi

# Check if pyproject.toml exists (indicates proper project setup)
if [[ ! -f "pyproject.toml" ]]; then
  echo -e "${RED}❌ pyproject.toml not found! This script requires the new modular structure.${NC}"
  exit 1
fi

# Check if pre-trained model exists, if not, train it
printf "${CYAN}🧠 Checking for pre-trained model...${NC}\n"
if [[ ! -f "models/step_detection_model.keras" ]]; then
  printf "${YELLOW}⚠️  Pre-trained model not found. Training model first...${NC}\n"
  
  # Create models directory if it doesn't exist
  mkdir -p models
  
  # Train the model using the new main.py CLI
  printf "${CYAN}🏃‍♂️ Training step detection model...${NC}\n"
  python main.py train
  
  if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Model training failed!${NC}"
    exit 1
  fi
  
  printf "${GREEN}✅ Model training completed successfully!${NC}\n"
else
  printf "${GREEN}✅ Pre-trained model found: models/step_detection_model.keras${NC}\n"
fi

# 1. Build the image using production Dockerfile
printf "${CYAN}🔨 Building Docker image with production Dockerfile...${NC}\n"
docker build -f "$DOCKERFILE_PATH" -t "$FULL_TAG" .
printf "${GREEN}✅ Image built successfully: $FULL_TAG${NC}\n"

# 2. Push to Docker Hub (if enabled)
if [[ "$PUSH" == true ]]; then
  printf "${CYAN}🚀 Pushing to Docker Hub...${NC}\n"
  
  # Auto-login to Docker Hub using environment variable
  printf "${CYAN}🔑 Logging in to Docker Hub...${NC}\n"
  
  # Check if Docker token is provided
  if [[ -z "$DOCKER_TOKEN" ]]; then
    echo -e "${YELLOW}⚠️  DOCKER_TOKEN environment variable not set${NC}"
    echo -e "Please set it with: export DOCKER_TOKEN=your_token"
    echo -e "Or login manually with: docker login"
    
    # Try to use existing docker login
    if ! docker info | grep -q "Username"; then
      echo -e "${RED}❌ Not logged in to Docker Hub. Please login first:${NC}"
      echo -e "   docker login"
      exit 1
    else
      echo -e "${GREEN}✅ Using existing Docker login${NC}"
    fi
  else
    # Login using environment variable
    echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin
    if [ $? -ne 0 ]; then
      echo -e "${RED}❌ Docker login failed!${NC}"
      exit 1
    fi
    printf "${GREEN}✅ Docker login successful${NC}\n"
  fi
  
  docker push "$FULL_TAG"
  printf "${GREEN}✅ Image pushed successfully: $FULL_TAG${NC}\n"
  
  echo -e "\n${BOLD}${GREEN}🎉 BUILD & PUSH COMPLETE! 🎉${NC}"
  echo -e "${BOLD}Image:${NC} $FULL_TAG"
  echo -e "${BOLD}Ready for deployment!${NC}"
  echo -e "\nTo deploy on server, run:"
  echo -e "  ./deploy.sh -u $DOCKER_USER"
else
  echo -e "\n${BOLD}${GREEN}🎉 BUILD COMPLETE! 🎉${NC}"
  echo -e "${BOLD}Image:${NC} $FULL_TAG"
  echo -e "${BOLD}Status:${NC} Built locally only"
  echo -e "\nTo push later, run:"
  echo -e "  docker push $FULL_TAG"
fi

echo -e "\n${BOLD}${CYAN}Docker Images:${NC}"
docker images | grep "$DOCKER_USER/$IMAGE_NAME" || echo "No images found"
