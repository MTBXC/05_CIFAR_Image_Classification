#!/bin/bash

# Docker build script for CIFAR-10 API
set -e

echo "Building Docker image for CIFAR-10 API..."

# Build the image
docker build -t cifar-api:latest .

echo "Docker image built successfully!"
echo "To run the container:"
echo "  docker run -p 8000:8000 -v \$(pwd)/data:/app/data -v \$(pwd)/models:/app/models cifar-api:latest"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up"



