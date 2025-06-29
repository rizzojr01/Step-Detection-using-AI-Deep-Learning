#!/bin/bash
set -e

# ════════════════════════════════════════════════
# ║      STEP DETECTION: DEPLOYMENT SCRIPT        ║
# ║      (Updated for new modular structure)      ║
# ════════════════════════════════════════════════

# Color definitions
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configurable variables
DOCKER_USER="bibektimilsina000"   # Your Docker Hub username
IMAGE_NAME="step-detection-ai"
TAG="latest"
CONTAINER_NAME="step-detection-api"
API_PORT="8000"
PORT=8080

# Print functions
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Usage
usage() {
  echo -e "${BOLD}Step Detection API Deployment Script${NC}"
  echo ""
  echo -e "${BOLD}Usage:${NC} $0 [COMMAND] [OPTIONS]"
  echo ""
  echo -e "${BOLD}Commands:${NC}"
  echo "  local        Deploy locally using Docker Compose"
  echo "  build        Build the Docker image"
  echo "  push         Push image to Docker Hub" 
  echo "  deploy       Deploy to remote server (pull and run)"
  echo "  stop         Stop the running container"
  echo "  logs         Show container logs"
  echo ""
  echo -e "${BOLD}Options:${NC}"
  echo "  -u, --user     Docker Hub username (default: $DOCKER_USER)"
  echo "  -i, --image    Image name (default: $IMAGE_NAME)"
  echo "  -t, --tag      Image tag (default: $TAG)"
  echo "  -p, --port     Host port (default: $API_PORT)"
  echo "  -h, --help     Show this help"
  echo ""
  echo -e "${BOLD}Examples:${NC}"
  echo "  $0 build                    # Build image locally"
  echo "  $0 local                    # Deploy with docker-compose"
  echo "  $0 deploy -p 8080           # Deploy on port 8080"
  echo "  $0 push -u myuser           # Push to Docker Hub"
  exit 1
}

# Parse command
COMMAND=""
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    COMMAND="$1"
    shift
fi

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--user)
      DOCKER_USER="$2"; shift 2;;
    -i|--image)
      IMAGE_NAME="$2"; shift 2;;
    -t|--tag)
      TAG="$2"; shift 2;;
    -p|--port)
      API_PORT="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option: $1"; usage;;
  esac
done

FULL_TAG="$DOCKER_USER/$IMAGE_NAME:$TAG"

printf "${CYAN}═══════════════════════════════════════${NC}\n"
printf "${CYAN}  STEP DETECTION API DEPLOYMENT${NC}\n"
printf "${CYAN}═══════════════════════════════════════${NC}\n"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

case "$COMMAND" in
    "build")
        print_info "Building Docker image..."
        docker build -f docker/Dockerfile.prod -t "$IMAGE_NAME:$TAG" .
        docker tag "$IMAGE_NAME:$TAG" "$FULL_TAG"
        print_status "Image built: $FULL_TAG"
        ;;
        
    "local")
        print_info "Deploying locally with Docker Compose..."
        docker-compose -f docker/docker-compose.prod.yml down
        docker-compose -f docker/docker-compose.prod.yml up -d
        print_status "Local deployment complete!"
        print_info "API available at: http://localhost:$API_PORT"
        print_info "Documentation at: http://localhost:$API_PORT/docs"
        ;;
        
    "push")
        print_info "Pushing image to Docker Hub..."
        docker push "$FULL_TAG"
        print_status "Image pushed: $FULL_TAG"
        ;;
        
    "deploy")
        print_info "Deploying to remote server..."
        
        # Pull latest image
        print_info "Pulling latest image from Docker Hub..."
        docker pull "$FULL_TAG"
        
        # Stop and remove old container
        print_info "Stopping old container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || print_warning "No running container found"
        docker rm "$CONTAINER_NAME" 2>/dev/null || print_warning "No container to remove"
        
        # Run new container with new structure
        print_info "Starting new container..."
        docker run -d \
          --name "$CONTAINER_NAME" \
          --restart unless-stopped \
          -p "$API_PORT:8000" \
          -e STEP_DETECTION_ENV=production \
          -e STEP_DETECTION_API_WORKERS=4 \
          -v "$(pwd)/logs:/app/logs" \
          -v "$(pwd)/models:/app/models:ro" \
          "$FULL_TAG"
        
        print_status "Deployment complete!"
        print_info "Container Status:"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        print_info "API available at: http://localhost:$API_PORT"
        print_info "Health check: http://localhost:$API_PORT/health"
        ;;
        
    "stop")
        print_info "Stopping container..."
        docker stop "$CONTAINER_NAME"
        print_status "Container stopped"
        ;;
        
    "logs")
        print_info "Showing container logs..."
        docker logs -f "$CONTAINER_NAME"
        ;;
        
    *)
        print_error "Unknown command: $COMMAND"
        usage
        ;;
esac

print_info "For logs, use: ./scripts/deploy.sh logs"
print_info "For health check: curl http://localhost:$API_PORT/health"
