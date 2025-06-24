#!/bin/bash
# Build and run the Step Detection API with Docker

set -e

echo "🚀 Building Step Detection API Docker image..."

# Check if pre-trained model exists, if not, train it
echo "🧠 Checking for pre-trained model..."
if [ ! -f "models/trained_step_detection_model.pth" ]; then
    echo "⚠️  Pre-trained model not found. Training model first..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Train the model
    echo "🏃‍♂️ Training step detection model..."
    python train_and_save_model.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Model training failed!"
        exit 1
    fi
    
    echo "✅ Model training completed successfully!"
else
    echo "✅ Pre-trained model found: models/trained_step_detection_model.pth"
fi

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
