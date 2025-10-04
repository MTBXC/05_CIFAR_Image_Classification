#!/bin/bash

# Deploy AWS infrastructure using Terraform
# This script initializes Terraform, plans, and applies the infrastructure

set -e

# Configuration
TERRAFORM_DIR="infra/aws/terraform"
AWS_REGION=${AWS_REGION:-us-east-1}
ENVIRONMENT=${ENVIRONMENT:-dev}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying AWS infrastructure for CIFAR-10 CNN project${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}AWS Region: ${AWS_REGION}${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured or no valid credentials${NC}"
    echo "Please run 'aws configure' to set up your credentials"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: Terraform is not installed${NC}"
    echo "Please install Terraform from https://www.terraform.io/downloads.html"
    exit 1
fi

# Navigate to Terraform directory
cd "$TERRAFORM_DIR"

# Initialize Terraform
echo -e "${BLUE}Initializing Terraform...${NC}"
terraform init

# Create terraform.tfvars file if it doesn't exist
if [ ! -f "terraform.tfvars" ]; then
    echo -e "${YELLOW}Creating terraform.tfvars file...${NC}"
    cat > terraform.tfvars << EOF
aws_region = "${AWS_REGION}"
environment = "${ENVIRONMENT}"
db_password = "mlflow_password_123!"
EOF
    echo -e "${GREEN}Created terraform.tfvars with default values${NC}"
    echo -e "${YELLOW}Please review and update terraform.tfvars if needed${NC}"
fi

# Plan the deployment
echo -e "${BLUE}Planning Terraform deployment...${NC}"
terraform plan -var-file="terraform.tfvars" -out="terraform.tfplan"

# Ask for confirmation
echo ""
echo -e "${YELLOW}Review the plan above. Do you want to proceed with the deployment? (y/N)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deployment cancelled${NC}"
    exit 0
fi

# Apply the deployment
echo -e "${BLUE}Applying Terraform deployment...${NC}"
terraform apply "terraform.tfplan"

# Get outputs
echo -e "${GREEN}Infrastructure deployed successfully!${NC}"
echo ""
echo -e "${YELLOW}Infrastructure outputs:${NC}"
terraform output

# Save outputs to file for other scripts
echo ""
echo -e "${BLUE}Saving outputs to terraform-outputs.json...${NC}"
terraform output -json > ../../terraform-outputs.json

echo ""
echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Build and push Docker images: ./scripts/build_and_push_images.sh"
echo "2. Update ECS services with new images"
echo "3. Upload CIFAR-10 data to S3: ./scripts/upload_data.sh"
echo "4. Test the deployment"

