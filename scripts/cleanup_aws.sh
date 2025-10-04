#!/bin/bash

# Cleanup AWS resources for CIFAR-10 CNN project
# This script removes all AWS resources created by the deployment

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ENVIRONMENT=${ENVIRONMENT:-dev}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}⚠️  WARNING: This will delete ALL AWS resources for the CIFAR-10 CNN project!${NC}"
echo -e "${YELLOW}This action cannot be undone.${NC}"
echo ""
echo -e "${YELLOW}Resources that will be deleted:${NC}"
echo "  - ECS Cluster and Services"
echo "  - Application Load Balancer"
echo "  - RDS PostgreSQL Database"
echo "  - S3 Buckets (with all data)"
echo "  - ECR Repositories"
echo "  - VPC and Networking"
echo "  - IAM Roles and Policies"
echo "  - CloudWatch Log Groups"
echo ""

# Ask for confirmation
echo -e "${YELLOW}Are you sure you want to proceed? Type 'DELETE' to confirm:${NC}"
read -r confirmation

if [ "$confirmation" != "DELETE" ]; then
    echo -e "${GREEN}Cleanup cancelled.${NC}"
    exit 0
fi

echo -e "${BLUE}Starting cleanup...${NC}"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured or no valid credentials${NC}"
    exit 1
fi

# Check if Terraform is available
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: Terraform is not installed${NC}"
    exit 1
fi

# Navigate to Terraform directory
cd infra/aws/terraform

# Check if terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo -e "${YELLOW}No Terraform state found. Nothing to clean up.${NC}"
    exit 0
fi

# Show what will be destroyed
echo -e "${BLUE}Planning destruction...${NC}"
terraform plan -destroy -var-file="terraform.tfvars" -out="destroy.tfplan"

# Ask for final confirmation
echo ""
echo -e "${RED}Final confirmation: Do you want to destroy all resources? (y/N)${NC}"
read -r final_confirmation

if [[ ! "$final_confirmation" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Destruction cancelled.${NC}"
    exit 0
fi

# Destroy infrastructure
echo -e "${BLUE}Destroying infrastructure...${NC}"
terraform destroy -auto-approve -var-file="terraform.tfvars"

# Clean up local files
echo -e "${BLUE}Cleaning up local files...${NC}"
rm -f terraform.tfplan
rm -f destroy.tfplan
rm -f terraform.tfvars

# Clean up parent directory
cd ../..
rm -f terraform-outputs.json

echo -e "${GREEN}✅ Cleanup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Note: Some resources may take a few minutes to be fully deleted.${NC}"
echo -e "${YELLOW}You can check the AWS Console to verify all resources are removed.${NC}"

