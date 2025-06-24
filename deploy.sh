#!/bin/bash
set -e

# ════════════════════════════════════════════════
# ║      STEP DETECTION: SERVER DEPLOY SCRIPT     ║
# ║      (Run this on your server to deploy)      ║
# ════════════════════════════════════════════════

# Color definitions
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configurable variables
DOCKER_USER="bibektimilsina000"   # Your Docker Hub username
IMAGE_NAME="step-detection-ai"
TAG="latest"
CONTAINER_NAME="step-detection-api"
PORT=8080

# Usage
usage() {
  echo -e "${BOLD}Usage:${NC} $0 [-u dockerhub_user] [-i image_name] [-t tag] [-p port]"
  echo -e "${BOLD}Example:${NC} $0 -u myusername -p 8080"
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
    -p|--port)
      PORT="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option: $1"; usage;;
  esac
done

FULL_TAG="$DOCKER_USER/$IMAGE_NAME:$TAG"

printf "${CYAN}═══════════════════════════════════════${NC}\n"
printf "${CYAN}  DEPLOYING: $FULL_TAG${NC}\n"
printf "${CYAN}═══════════════════════════════════════${NC}\n"

# 1. Pull latest image
printf "${CYAN}Pulling latest image from Docker Hub...${NC}\n"
docker pull "$FULL_TAG"
printf "${GREEN}Image pulled: $FULL_TAG${NC}\n"

# 2. Stop and remove old container
printf "${CYAN}Stopping old container...${NC}\n"
docker stop "$CONTAINER_NAME" 2>/dev/null || printf "${YELLOW}No running container found${NC}\n"
docker rm "$CONTAINER_NAME" 2>/dev/null || printf "${YELLOW}No container to remove${NC}\n"

# 3. Run new container
printf "${CYAN}Starting new container...${NC}\n"
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p "$PORT:8080" \
  "$FULL_TAG"

printf "${GREEN}Deployment complete!${NC}\n"
printf "${GREEN}Container Status:${NC}\n"
docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n${BOLD}${GREEN}✨ App running at: http://localhost:$PORT ✨${NC}\n"
