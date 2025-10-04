#!/bin/bash

# AWS setup script for CIFAR-10 API deployment
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="cifar-api"
ECS_CLUSTER="cifar-cluster"
ECS_SERVICE="cifar-service"
TASK_DEFINITION="cifar-task"

echo "Setting up AWS infrastructure for CIFAR-10 API..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

echo "1. Creating ECR repository..."
aws ecr create-repository \
    --repository-name $ECR_REPOSITORY \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    || echo "Repository already exists"

echo "2. Getting ECR login token..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com

echo "3. Building and tagging image..."
docker build -t $ECR_REPOSITORY:latest .
docker tag $ECR_REPOSITORY:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

echo "4. Pushing image to ECR..."
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

echo "5. Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $ECS_CLUSTER \
    --region $AWS_REGION \
    || echo "Cluster already exists"

echo "AWS setup completed!"
echo "Next steps:"
echo "1. Create ECS task definition (see infra/aws/task-definition.json)"
echo "2. Create ECS service"
echo "3. Set up Application Load Balancer if needed"



