#!/bin/bash
set -e

# ════════════════════════════════════════════════
# ║      STEP DETECTION: DEV BUILD SCRIPT         ║
# ║      (Build and run development environment)  ║
# ║      Uses docker/Dockerfile.dev               ║
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
IMAGE_NAME="step-detection-dev"
TAG="latest"
COMPOSE_FILE="docker/docker-compose.dev.yml"

# Usage
usage() {
  echo -e "${BOLD}Usage:${NC} $0 [COMMAND] [OPTIONS]"
  echo -e "${BOLD}Commands:${NC}"
  echo "  build        Build development Docker image"
  echo "  up           Start development environment (with hot-reload)"
  echo "  down         Stop development environment"
  echo "  logs         Show development logs"
  echo "  shell        Access development container shell"
  echo "  test         Run tests in development environment"
  echo -e "${BOLD}Options:${NC}"
  echo -e "  -i, --image NAME     Image name (default: step-detection-dev)"
  echo -e "  -t, --tag TAG        Image tag (default: latest)"
  echo -e "  -h, --help           Show this help"
  echo -e ""
  echo -e "${BOLD}Examples:${NC}"
  echo -e "  $0 build             # Build dev image"
  echo -e "  $0 up               # Start development environment"
  echo -e "  $0 test             # Run tests"
  echo -e "  $0 shell            # Access container shell"
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

# Parse command
COMMAND=""
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    COMMAND="$1"
    shift
fi

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--image)
      IMAGE_NAME="$2"; shift 2;;
    -t|--tag)
      TAG="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option: $1"; usage;;
  esac
done

FULL_TAG="$IMAGE_NAME:$TAG"

printf "${BLUE}═══════════════════════════════════════${NC}\n"
printf "${BLUE}  STEP DETECTION: DEVELOPMENT BUILD${NC}\n"
printf "${BLUE}═══════════════════════════════════════${NC}\n"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check required files
if [[ ! -f "docker/Dockerfile.dev" ]]; then
  print_error "Development Dockerfile not found at docker/Dockerfile.dev!"
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  print_error "Development docker-compose file not found at $COMPOSE_FILE!"
  exit 1
fi

case "$COMMAND" in
    "build")
        print_info "Building development Docker image..."
        docker build -f docker/Dockerfile.dev -t "$FULL_TAG" .
        print_success "Development image built: $FULL_TAG"
        ;;
        
    "up")
        print_info "Starting development environment..."
        # Build if image doesn't exist
        if ! docker image inspect "$FULL_TAG" >/dev/null 2>&1; then
            print_info "Image not found, building first..."
            docker build -f docker/Dockerfile.dev -t "$FULL_TAG" .
        fi
        
        docker-compose -f "$COMPOSE_FILE" up -d
        print_success "Development environment started!"
        print_info "API available at: http://localhost:8000"
        print_info "Documentation at: http://localhost:8000/docs"
        print_info "Hot-reload is enabled - code changes will automatically restart the server"
        print_info "Use '$0 logs' to see real-time logs"
        ;;
        
    "down")
        print_info "Stopping development environment..."
        docker-compose -f "$COMPOSE_FILE" down
        print_success "Development environment stopped"
        ;;
        
    "logs")
        print_info "Showing development logs (Ctrl+C to exit)..."
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
        
    "shell")
        print_info "Accessing development container shell..."
        CONTAINER_NAME=$(docker-compose -f "$COMPOSE_FILE" ps -q app 2>/dev/null)
        if [[ -z "$CONTAINER_NAME" ]]; then
            print_error "Development container not running. Use '$0 up' first."
            exit 1
        fi
        docker exec -it "$CONTAINER_NAME" /bin/bash
        ;;
        
    "test")
        print_info "Running tests in development environment..."
        CONTAINER_NAME=$(docker-compose -f "$COMPOSE_FILE" ps -q app 2>/dev/null)
        if [[ -z "$CONTAINER_NAME" ]]; then
            print_error "Development container not running. Use '$0 up' first."
            exit 1
        fi
        docker exec -it "$CONTAINER_NAME" python -m pytest tests/ -v
        ;;
        
    *)
        if [[ -z "$COMMAND" ]]; then
            print_info "No command specified. Starting development environment..."
            $0 up
        else
            print_error "Unknown command: $COMMAND"
            usage
        fi
        ;;
esac

print_info "Development commands:"
print_info "  Build: $0 build"
print_info "  Start: $0 up"
print_info "  Stop:  $0 down"
print_info "  Logs:  $0 logs"
print_info "  Shell: $0 shell"
print_info "  Test:  $0 test"
