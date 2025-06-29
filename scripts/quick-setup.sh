#!/bin/bash
set -e

# ════════════════════════════════════════════════
# ║      STEP DETECTION: QUICK SETUP SCRIPT       ║
# ║      (Get up and running in minutes)          ║
# ════════════════════════════════════════════════

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

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

print_header() {
  echo -e "${BOLD}${CYAN}"
  echo "┌─────────────────────────────────────────────────────────────┐"
  echo "│                   STEP DETECTION AI                         │"
  echo "│                 🚀 QUICK SETUP SCRIPT 🚀                   │"
  echo "│                                                             │"
  echo "│  This script will help you get the Step Detection AI       │"
  echo "│  project up and running quickly!                           │"
  echo "└─────────────────────────────────────────────────────────────┘"
  echo -e "${NC}"
}

check_dependencies() {
  print_section "CHECKING DEPENDENCIES"
  
  # Check Docker
  if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      print_success "Docker is installed and running"
    else
      print_error "Docker is installed but not running"
      print_info "Please start Docker and run this script again"
      exit 1
    fi
  else
    print_error "Docker is not installed"
    print_info "Please install Docker from: https://www.docker.com/get-started"
    exit 1
  fi
  
  # Check Docker Compose
  if command -v docker-compose >/dev/null 2>&1; then
    print_success "Docker Compose is installed"
  else
    print_warning "Docker Compose not found, checking for 'docker compose'..."
    if docker compose version >/dev/null 2>&1; then
      print_success "Docker Compose (v2) is available"
    else
      print_error "Docker Compose is not available"
      print_info "Please install Docker Compose"
      exit 1
    fi
  fi
  
  # Check Python (for local development)
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION is installed"
  else
    print_warning "Python3 not found (needed for local development)"
  fi
}

show_options() {
  print_section "SETUP OPTIONS"
  echo -e "${BOLD}Choose your setup preference:${NC}"
  echo ""
  echo "1) 🐳 Docker Development (Recommended for quick start)"
  echo "   - Build and run in Docker with hot-reload"
  echo "   - No local Python setup required"
  echo "   - Great for testing and development"
  echo ""
  echo "2) 🐳 Docker Production (For deployment testing)"
  echo "   - Build production-ready Docker image"
  echo "   - Optimized for performance"
  echo "   - Includes model training"
  echo ""
  echo "3) 🐍 Local Development (Advanced users)"
  echo "   - Install dependencies locally"
  echo "   - Full control over environment"
  echo "   - Best for active development"
  echo ""
  echo "4) 📚 Just show me the commands"
  echo "   - Display setup commands without running"
  echo ""
  read -p "Enter your choice (1-4): " CHOICE
}

setup_docker_dev() {
  print_section "DOCKER DEVELOPMENT SETUP"
  
  print_info "Building development Docker image..."
  ./scripts/build-all.sh dev
  
  print_info "Starting development environment..."
  ./scripts/build-dev.sh up
  
  print_success "Development environment is running!"
  print_info "🌐 API: http://localhost:8000"
  print_info "📖 Docs: http://localhost:8000/docs"
  print_info "🔍 Health: http://localhost:8000/health"
  
  echo ""
  print_info "Useful commands:"
  print_info "  View logs: ./scripts/build-dev.sh logs"
  print_info "  Stop: ./scripts/build-dev.sh down"
  print_info "  Shell access: ./scripts/build-dev.sh shell"
  print_info "  Run tests: ./scripts/build-dev.sh test"
}

setup_docker_prod() {
  print_section "DOCKER PRODUCTION SETUP"
  
  print_info "This will build a production-ready image with model training..."
  read -p "Continue? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    return
  fi
  
  print_info "Building production Docker image (this may take a while)..."
  ./scripts/build-all.sh prod --train
  
  print_info "Starting production environment..."
  ./scripts/deploy.sh local
  
  print_success "Production environment is running!"
  print_info "🌐 API: http://localhost:8080"
  print_info "📖 Docs: http://localhost:8080/docs"
  print_info "🔍 Health: http://localhost:8080/health"
}

setup_local_dev() {
  print_section "LOCAL DEVELOPMENT SETUP"
  
  if ! command -v python3 >/dev/null 2>&1; then
    print_error "Python3 is required for local development"
    return
  fi
  
  print_info "Setting up local Python environment..."
  
  # Create virtual environment
  print_info "Creating virtual environment..."
  python3 -m venv venv
  
  # Activate virtual environment
  print_info "Activating virtual environment..."
  source venv/bin/activate
  
  # Install dependencies
  print_info "Installing dependencies..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[dev]"
  else
    pip install -e ".[dev]"
  fi
  
  # Train model if needed
  if [[ ! -f "models/step_detection_model.keras" ]]; then
    print_info "Training model..."
    python main.py train
  fi
  
  print_success "Local development environment ready!"
  print_info "📝 Activate with: source venv/bin/activate"
  print_info "🚀 Start API: python main.py api"
  print_info "🧪 Run tests: python main.py test"
  print_info "🎯 Train model: python main.py train"
}

show_commands() {
  print_section "SETUP COMMANDS REFERENCE"
  
  echo -e "${BOLD}Docker Development:${NC}"
  echo "  ./scripts/build-all.sh dev        # Build dev image"
  echo "  ./scripts/build-dev.sh up         # Start dev environment"
  echo "  ./scripts/build-dev.sh logs       # View logs"
  echo "  ./scripts/build-dev.sh down       # Stop environment"
  echo ""
  
  echo -e "${BOLD}Docker Production:${NC}"
  echo "  ./scripts/build-all.sh prod --train  # Build prod image"
  echo "  ./scripts/deploy.sh local            # Deploy locally"
  echo "  ./scripts/deploy.sh logs             # View logs"
  echo "  ./scripts/deploy.sh stop             # Stop deployment"
  echo ""
  
  echo -e "${BOLD}Local Development:${NC}"
  echo "  python3 -m venv venv              # Create virtual env"
  echo "  source venv/bin/activate          # Activate env"
  echo "  pip install -e '.[dev]'           # Install dependencies"
  echo "  python main.py train              # Train model"
  echo "  python main.py api                # Start API"
  echo ""
  
  echo -e "${BOLD}Testing:${NC}"
  echo "  python main.py test               # Run all tests"
  echo "  curl http://localhost:8000/health # Health check"
  echo "  python test_websocket.py          # Test WebSocket"
}

# Main execution
print_header

check_dependencies

show_options

case $CHOICE in
  1)
    setup_docker_dev
    ;;
  2)
    setup_docker_prod
    ;;
  3)
    setup_local_dev
    ;;
  4)
    show_commands
    ;;
  *)
    print_error "Invalid choice. Please run the script again."
    exit 1
    ;;
esac

print_section "NEXT STEPS"
print_info "🎉 Setup complete! Here's what you can do next:"
print_info ""
print_info "📱 Test the API:"
print_info "  • Open http://localhost:8000/docs (or :8080 for prod)"
print_info "  • Try the /predict endpoint with accelerometer data"
print_info "  • Use the WebSocket endpoint for real-time detection"
print_info ""
print_info "🧪 Explore the code:"
print_info "  • Main CLI: main.py"
print_info "  • API code: src/step_detection/api/"
print_info "  • Core logic: src/step_detection/core/"
print_info "  • Notebooks: notebooks/"
print_info ""
print_info "📚 Documentation:"
print_info "  • README.md - Project overview"
print_info "  • docs/ - Detailed documentation"
print_info "  • QUICK_START.md - Quick reference"
print_info ""
print_info "🤝 Need help? Check the troubleshooting guide in docs/TROUBLESHOOTING.md"
