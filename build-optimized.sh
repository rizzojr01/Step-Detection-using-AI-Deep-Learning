#!/bin/bash

# Build Script with Pre-trained Model
# ===================================
# This script ensures the model is trained before building the Docker image
# for optimized deployment.

set -e  # Exit on any error

echo "🏗️ Building Step Detection API with Pre-trained Model"
echo "=" * 60

# Check if we're in the right directory
if [ ! -f "train_and_save_model.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Step 1: Train the model if it doesn't exist
MODEL_FILE="models/trained_step_detection_model.pth"
if [ ! -f "$MODEL_FILE" ]; then
    echo "🚂 Training model (this may take a few minutes)..."
    python train_and_save_model.py
    
    if [ ! -f "$MODEL_FILE" ]; then
        echo "❌ Error: Model training failed - $MODEL_FILE not found"
        exit 1
    fi
else
    echo "✅ Pre-trained model found: $MODEL_FILE"
fi

# Step 2: Build Docker image
echo "🐳 Building Docker image..."
if [ "$1" = "prod" ]; then
    echo "📦 Building production image..."
    docker build -f Dockerfile.prod -t step-detection-api:prod .
    echo "✅ Production image built successfully!"
    echo "🚀 To run: docker run -p 8000:8000 step-detection-api:prod"
else
    echo "🛠️ Building development image..."
    docker build -f Dockerfile -t step-detection-api:latest .
    echo "✅ Development image built successfully!"
    echo "🚀 To run: docker run -p 8000:8000 step-detection-api:latest"
fi

echo ""
echo "✅ Build completed successfully!"
echo "📊 Model info:"
ls -lh "$MODEL_FILE"
echo ""
echo "🌐 API will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
