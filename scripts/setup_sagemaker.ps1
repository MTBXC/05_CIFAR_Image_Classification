# SageMaker Setup Script for Windows PowerShell
# This script sets up the SageMaker environment and creates a test training job

param(
    [switch]$SkipTerraform,
    [switch]$SkipDataUpload,
    [switch]$SkipTestJob
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

Write-Host "🚀 Setting up SageMaker Environment for CIFAR-10 Training" -ForegroundColor $Green
Write-Host "==========================================================" -ForegroundColor $Green

# Check if we're in the right directory
if (-not (Test-Path "infra/aws/terraform-sagemaker/main.tf")) {
    Write-Error "Please run this script from the project root directory"
    exit 1
}

# Check if AWS CLI is configured
try {
    $null = aws sts get-caller-identity 2>$null
    Write-Success "AWS CLI is configured"
} catch {
    Write-Error "AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
}

# Check if Terraform is installed
try {
    $null = terraform version 2>$null
    Write-Success "Terraform is installed"
} catch {
    Write-Error "Terraform is not installed. Please install Terraform first."
    exit 1
}

# Get MLFlow server IP from minimal deployment
Write-Status "Getting MLFlow server information..."
Set-Location "infra/aws/terraform-minimal"

if (-not (Test-Path "terraform.tfstate")) {
    Write-Error "Terraform state not found. Please deploy the minimal MLFlow infrastructure first."
    exit 1
}

$MLFLOW_URI = terraform output -raw mlflow_tracking_uri
if (-not $MLFLOW_URI) {
    Write-Error "Could not get MLFlow server URI from Terraform output"
    exit 1
}

$MLFLOW_IP = $MLFLOW_URI -replace "http://", "" -replace ":5000", ""
Write-Success "MLFlow server IP: $MLFLOW_IP"

# Go back to project root
Set-Location "../../.."

if (-not $SkipTerraform) {
    # Deploy SageMaker infrastructure
    Write-Status "Deploying SageMaker infrastructure..."
    Set-Location "infra/aws/terraform-sagemaker"

    # Check if terraform.tfvars exists
    if (-not (Test-Path "terraform.tfvars")) {
        Write-Warning "terraform.tfvars not found. Creating from example..."
        Copy-Item "terraform.tfvars.example" "terraform.tfvars"
        
        # Update MLFlow tracking URI
        (Get-Content "terraform.tfvars") -replace "http://YOUR_MLFLOW_SERVER_IP:5000", $MLFLOW_URI | Set-Content "terraform.tfvars"
        
        Write-Success "Created terraform.tfvars with MLFlow server IP"
    }

    # Initialize Terraform
    Write-Status "Initializing Terraform..."
    terraform init

    # Plan deployment
    Write-Status "Planning SageMaker infrastructure deployment..."
    terraform plan -out=tfplan

    # Ask for confirmation
    Write-Host ""
    $confirmation = Read-Host "Do you want to deploy the SageMaker infrastructure? (y/N)"
    if ($confirmation -notmatch "^[Yy]$") {
        Write-Warning "Deployment cancelled by user"
        exit 0
    }

    # Apply deployment
    Write-Status "Deploying SageMaker infrastructure..."
    terraform apply tfplan

    # Get outputs
    $DATA_BUCKET = terraform output -raw sagemaker_data_bucket_name
    $MODEL_BUCKET = terraform output -raw sagemaker_models_bucket_name
    $NOTEBOOK_URL = terraform output -raw sagemaker_notebook_url

    Write-Success "SageMaker infrastructure deployed successfully!"
    Write-Status "Data bucket: $DATA_BUCKET"
    Write-Status "Model bucket: $MODEL_BUCKET"
    Write-Status "Notebook URL: $NOTEBOOK_URL"

    # Go back to project root
    Set-Location "../../.."
} else {
    Write-Status "Skipping Terraform deployment..."
    Set-Location "infra/aws/terraform-sagemaker"
    $DATA_BUCKET = terraform output -raw sagemaker_data_bucket_name
    $MODEL_BUCKET = terraform output -raw sagemaker_models_bucket_name
    $NOTEBOOK_URL = terraform output -raw sagemaker_notebook_url
    Set-Location "../../.."
}

if (-not $SkipDataUpload) {
    # Upload CIFAR-10 data to S3
    Write-Status "Uploading CIFAR-10 data to S3..."
    if (Test-Path "data/raw/cifar-10-python.tar.gz") {
        aws s3 cp "data/raw/cifar-10-python.tar.gz" "s3://$DATA_BUCKET/data/"
        Write-Success "CIFAR-10 data uploaded to S3"
    } else {
        Write-Warning "CIFAR-10 data not found at data/raw/cifar-10-python.tar.gz"
        Write-Status "Please download and upload the data manually:"
        Write-Status "aws s3 cp data/raw/cifar-10-python.tar.gz s3://$DATA_BUCKET/data/"
    }
} else {
    Write-Status "Skipping data upload..."
}

if (-not $SkipTestJob) {
    # Create a test training job
    Write-Status "Creating test training job..."
    Set-Location "infra/aws/sagemaker"

    # Create a test job
    $JOB_NAME = "cifar10-test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Write-Status "Creating training job: $JOB_NAME"

    python sagemaker-training.py create `
        --job-name "$JOB_NAME" `
        --model-type base_cnn `
        --epochs 5 `
        --batch-size 32 `
        --learning-rate 0.001 `
        --instance-type ml.m5.large `
        --mlflow-tracking-uri "$MLFLOW_URI" `
        --data-bucket "$DATA_BUCKET" `
        --model-bucket "$MODEL_BUCKET" `
        --wait

    Write-Success "Test training job completed!"

    # Go back to project root
    Set-Location "../../.."
} else {
    Write-Status "Skipping test job creation..."
}

# Print summary
Write-Host ""
Write-Host "🎉 SageMaker Setup Complete!" -ForegroundColor $Green
Write-Host "============================" -ForegroundColor $Green
Write-Host ""
Write-Host "📊 MLFlow Server: $MLFLOW_URI" -ForegroundColor $Blue
Write-Host "📓 SageMaker Notebook: $NOTEBOOK_URL" -ForegroundColor $Blue
Write-Host "🗄️  Data Bucket: $DATA_BUCKET" -ForegroundColor $Blue
Write-Host "🤖 Model Bucket: $MODEL_BUCKET" -ForegroundColor $Blue
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor $Yellow
Write-Host "1. Access the SageMaker Notebook Instance to develop new models"
Write-Host "2. Use the training scripts to create more training jobs"
Write-Host "3. Monitor experiments in the MLFlow UI"
Write-Host "4. Deploy trained models using SageMaker endpoints"
Write-Host ""
Write-Host "🔧 Useful Commands:" -ForegroundColor $Yellow
Write-Host "  # List training jobs"
Write-Host "  python infra/aws/sagemaker/sagemaker-training.py list"
Write-Host ""
Write-Host "  # Monitor a specific job"
Write-Host "  python infra/aws/sagemaker/sagemaker-training.py monitor --job-name <job-name>"
Write-Host ""
Write-Host "  # Create a new training job"
Write-Host "  python infra/aws/sagemaker/sagemaker-training.py create --job-name <job-name> --model-type resnet --epochs 50"
Write-Host ""












