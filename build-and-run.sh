#!/bin/bash
# Build and run the Step Detection API with Docker

set -e

echo "🚀 Building Step Detection API Docker image..."

# Build the Docker image
docker build -t step-detection-api:latest .

echo "✅ Docker image built successfully!"

# Check if container is already running
if [ "$(docker ps -q -f name=step-detection-api)" ]; then
    echo "🔄 Stopping existing container..."
    docker stop step-detection-api
    docker rm step-detection-api
fi

echo "🚀 Starting Step Detection API container..."

# Run the container
docker run -d \
    --name step-detection-api \
    -p 8000:8000 \
    --restart unless-stopped \
    step-detection-api:latest

echo "✅ Container started successfully!"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 Container logs: docker logs -f step-detection-api"
echo "🛑 Stop container: docker stop step-detection-api"

# Wait a moment and check if container is running
sleep 3
if [ "$(docker ps -q -f name=step-detection-api)" ]; then
    echo "✅ Container is running healthy!"
    docker logs --tail 10 step-detection-api
else
    echo "❌ Container failed to start. Check logs:"
    docker logs step-detection-api
    exit 1
fi
