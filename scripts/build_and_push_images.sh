#!/bin/bash

# Build and push Docker images to AWS ECR
# This script builds the API, MLflow, and Training images and pushes them to ECR

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Image names
API_IMAGE="cifar-cnn-api"
MLFLOW_IMAGE="cifar-cnn-mlflow"
TRAINING_IMAGE="cifar-cnn-training"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building and pushing Docker images to ECR${NC}"
echo -e "${YELLOW}AWS Account ID: ${AWS_ACCOUNT_ID}${NC}"
echo -e "${YELLOW}AWS Region: ${AWS_REGION}${NC}"
echo -e "${YELLOW}ECR Registry: ${ECR_REGISTRY}${NC}"
echo ""

# Function to build and push image
build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local context=$3
    
    echo -e "${GREEN}Building ${image_name}...${NC}"
    
    # Build image
    docker build -f "$dockerfile" -t "$image_name" "$context"
    
    # Tag for ECR
    docker tag "$image_name" "${ECR_REGISTRY}/${image_name}:latest"
    docker tag "$image_name" "${ECR_REGISTRY}/${image_name}:$(date +%Y%m%d-%H%M%S)"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    
    # Push to ECR
    echo -e "${GREEN}Pushing ${image_name} to ECR...${NC}"
    docker push "${ECR_REGISTRY}/${image_name}:latest"
    docker push "${ECR_REGISTRY}/${image_name}:$(date +%Y%m%d-%H%M%S)"
    
    echo -e "${GREEN}Successfully pushed ${image_name}${NC}"
    echo ""
}

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured or no valid credentials${NC}"
    echo "Please run 'aws configure' to set up your credentials"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Create ECR repositories if they don't exist
echo -e "${GREEN}Creating ECR repositories...${NC}"
for repo in "$API_IMAGE" "$MLFLOW_IMAGE" "$TRAINING_IMAGE"; do
    if ! aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" > /dev/null 2>&1; then
        echo "Creating repository: $repo"
        aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION"
    else
        echo "Repository $repo already exists"
    fi
done
echo ""

# Build and push API image
build_and_push "$API_IMAGE" "Dockerfile" "."

# Build and push MLflow image
build_and_push "$MLFLOW_IMAGE" "Dockerfile.mlflow" "."

# Build and push Training image
build_and_push "$TRAINING_IMAGE" "Dockerfile.training" "."

echo -e "${GREEN}All images built and pushed successfully!${NC}"
echo ""
echo -e "${YELLOW}Image URIs:${NC}"
echo "API: ${ECR_REGISTRY}/${API_IMAGE}:latest"
echo "MLflow: ${ECR_REGISTRY}/${MLFLOW_IMAGE}:latest"
echo "Training: ${ECR_REGISTRY}/${TRAINING_IMAGE}:latest"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update ECS task definitions with the new image URIs"
echo "2. Deploy the infrastructure using Terraform"
echo "3. Update ECS services to use the new images"

