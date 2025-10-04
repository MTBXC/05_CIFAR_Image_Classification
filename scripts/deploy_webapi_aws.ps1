# PowerShell script to deploy WebAPI to AWS ECS
# This script handles the complete deployment pipeline

param(
    [string]$Action = "plan",  # plan, build, deploy, destroy
    [switch]$AutoApprove = $false,
    [string]$ImageTag = "latest"
)

Write-Host "🚀 WebAPI AWS Deployment Script" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

# Configuration
$TerraformDir = "infra\aws\terraform-webapi"
$DockerFile = "Dockerfile.webapi"

# Check prerequisites
function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Cyan
    
    # Check Terraform
    try {
        $terraformVersion = terraform version
        Write-Host "✅ Terraform: $($terraformVersion[0])" -ForegroundColor Green
    } catch {
        Write-Host "❌ Terraform not found. Please install Terraform." -ForegroundColor Red
        exit 1
    }
    
    # Check AWS CLI
    try {
        $awsIdentity = aws sts get-caller-identity 2>$null | ConvertFrom-Json
        Write-Host "✅ AWS CLI configured for account: $($awsIdentity.Account)" -ForegroundColor Green
    } catch {
        Write-Host "❌ AWS CLI not configured. Please run 'aws configure'." -ForegroundColor Red
        exit 1
    }
    
    # Check Docker
    try {
        $dockerVersion = docker version --format '{{.Client.Version}}' 2>$null
        Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ Docker not found. Please install Docker." -ForegroundColor Red
        exit 1
    }
    
    # Check required files
    if (-not (Test-Path $DockerFile)) {
        Write-Host "❌ Dockerfile not found: $DockerFile" -ForegroundColor Red
        exit 1
    }
    
    if (-not (Test-Path $TerraformDir)) {
        Write-Host "❌ Terraform directory not found: $TerraformDir" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ All prerequisites satisfied!" -ForegroundColor Green
}

function Initialize-Terraform {
    Write-Host "`n🔧 Initializing Terraform..." -ForegroundColor Cyan
    
    Set-Location $TerraformDir
    
    # Copy tfvars if needed
    if (-not (Test-Path "terraform.tfvars")) {
        if (Test-Path "terraform.tfvars.example") {
            Copy-Item "terraform.tfvars.example" "terraform.tfvars"
            Write-Host "📝 Created terraform.tfvars from example" -ForegroundColor Yellow
        }
    }
    
    terraform init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Terraform init failed" -ForegroundColor Red
        exit 1
    }
    
    terraform validate
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Terraform validation failed" -ForegroundColor Red
        exit 1
    }
    
    Set-Location ..\..\..
}

function Build-And-Push-Image {
    Write-Host "`n🏗️ Building and pushing Docker image..." -ForegroundColor Cyan
    
    # Get ECR repository URL from Terraform output
    Set-Location $TerraformDir
    
    try {
        $terraformOutput = terraform output -json | ConvertFrom-Json
        $ecrUrl = $terraformOutput.ecr_repository_url.value
        Write-Host "📦 ECR Repository: $ecrUrl" -ForegroundColor Yellow
    } catch {
        Write-Host "❌ Could not get ECR repository URL from Terraform output" -ForegroundColor Red
        Write-Host "   Make sure to run terraform plan/apply first" -ForegroundColor Yellow
        exit 1
    }
    
    Set-Location ..\..\..
    
    # Get AWS region and account ID
    $awsRegion = aws configure get region
    $accountId = (aws sts get-caller-identity --query Account --output text)
    
    # Login to ECR
    Write-Host "🔐 Logging in to ECR..." -ForegroundColor Cyan
    aws ecr get-login-password --region $awsRegion | docker login --username AWS --password-stdin "$accountId.dkr.ecr.$awsRegion.amazonaws.com"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ ECR login failed" -ForegroundColor Red
        exit 1
    }
    
    # Build image
    $imageUri = "$ecrUrl`:$ImageTag"
    Write-Host "🏗️ Building Docker image: $imageUri" -ForegroundColor Cyan
    
    docker build -f $DockerFile -t $imageUri .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker build failed" -ForegroundColor Red
        exit 1
    }
    
    # Push image
    Write-Host "📤 Pushing image to ECR..." -ForegroundColor Cyan
    docker push $imageUri
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker push failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Image built and pushed successfully!" -ForegroundColor Green
    Write-Host "   Image URI: $imageUri" -ForegroundColor Yellow
    
    return $imageUri
}

function Deploy-Infrastructure {
    param([string]$ImageUri)
    
    Write-Host "`n🚀 Deploying infrastructure..." -ForegroundColor Cyan
    
    Set-Location $TerraformDir
    
    # Update terraform.tfvars with image URI
    if ($ImageUri) {
        $tfvarsContent = Get-Content "terraform.tfvars" -Raw
        $tfvarsContent = $tfvarsContent -replace 'docker_image_uri = ".*"', "docker_image_uri = `"$ImageUri`""
        Set-Content "terraform.tfvars" $tfvarsContent
        Write-Host "📝 Updated terraform.tfvars with image URI" -ForegroundColor Yellow
    }
    
    if ($AutoApprove) {
        terraform apply -auto-approve
    } else {
        terraform apply
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Infrastructure deployed successfully!" -ForegroundColor Green
        
        # Show deployment info
        Write-Host "`n📊 Deployment Information:" -ForegroundColor Cyan
        $output = terraform output -json | ConvertFrom-Json
        $deploymentInfo = $output.deployment_info.value
        
        Write-Host "🌐 API URL: $($deploymentInfo.api_url)" -ForegroundColor Green
        Write-Host "🏥 Health Check: $($deploymentInfo.health_check_url)" -ForegroundColor Green
        Write-Host "📦 ECR Repository: $($deploymentInfo.ecr_repository)" -ForegroundColor Yellow
        Write-Host "🎯 ECS Service: $($deploymentInfo.ecs_service)" -ForegroundColor Yellow
        
        Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
        Write-Host "1. Wait ~2-3 minutes for ECS service to start" -ForegroundColor White
        Write-Host "2. Test health check: curl $($deploymentInfo.health_check_url)" -ForegroundColor White
        Write-Host "3. Test API: curl $($deploymentInfo.api_url)/api/model-info" -ForegroundColor White
        
    } else {
        Write-Host "❌ Infrastructure deployment failed" -ForegroundColor Red
        exit 1
    }
    
    Set-Location ..\..\..
}

function Remove-Infrastructure {
    Write-Host "`n⚠️ DESTROYING infrastructure..." -ForegroundColor Red
    
    Set-Location $TerraformDir
    
    Write-Host "This will delete all AWS resources created by this deployment." -ForegroundColor Yellow
    $confirmation = Read-Host "Are you sure? Type 'yes' to confirm"
    
    if ($confirmation -eq "yes") {
        if ($AutoApprove) {
            terraform destroy -auto-approve
        } else {
            terraform destroy
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Infrastructure destroyed successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ Infrastructure destruction failed" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Destruction cancelled" -ForegroundColor Yellow
    }
    
    Set-Location ..\..\..
}

# Main execution
Test-Prerequisites

switch ($Action.ToLower()) {
    "plan" {
        Initialize-Terraform
        Set-Location $TerraformDir
        terraform plan -out=tfplan
        Set-Location ..\..\..
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✅ Terraform plan completed!" -ForegroundColor Green
            Write-Host "Next steps:" -ForegroundColor Yellow
            Write-Host "1. Review the plan above" -ForegroundColor White
            Write-Host "2. Run: .\scripts\deploy_webapi_aws.ps1 -Action build" -ForegroundColor White
            Write-Host "3. Then: .\scripts\deploy_webapi_aws.ps1 -Action deploy" -ForegroundColor White
        }
    }
    
    "build" {
        Initialize-Terraform
        $imageUri = Build-And-Push-Image
        Write-Host "`n✅ Build completed!" -ForegroundColor Green
        Write-Host "Image URI: $imageUri" -ForegroundColor Yellow
        Write-Host "Next: .\scripts\deploy_webapi_aws.ps1 -Action deploy" -ForegroundColor White
    }
    
    "deploy" {
        Initialize-Terraform
        $imageUri = Build-And-Push-Image
        Deploy-Infrastructure -ImageUri $imageUri
    }
    
    "destroy" {
        Initialize-Terraform
        Remove-Infrastructure
    }
    
    default {
        Write-Host "❌ Invalid action: $Action" -ForegroundColor Red
        Write-Host "Valid actions: plan, build, deploy, destroy" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "`n✅ Script completed!" -ForegroundColor Green











