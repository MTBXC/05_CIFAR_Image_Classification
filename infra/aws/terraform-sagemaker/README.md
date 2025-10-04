# SageMaker Infrastructure for CIFAR-10 Training

This Terraform configuration creates a complete SageMaker environment for training CIFAR-10 models with MLFlow integration.

## Overview

This infrastructure provides:
- **SageMaker Notebook Instance** for interactive development and experimentation
- **S3 Buckets** for training data and model artifacts
- **IAM Roles** with appropriate permissions for SageMaker operations
- **VPC Configuration** with public and private subnets
- **MLFlow Integration** for experiment tracking and model management

## Prerequisites

1. **Existing MLFlow Infrastructure**: This assumes you have already deployed the MLFlow server using the `terraform-minimal` configuration.

2. **AWS CLI Configured**: Make sure your AWS credentials are configured.

3. **Terraform Installed**: Version 1.0 or higher.

## Quick Start

### 1. Get MLFlow Server IP

First, get the IP address of your MLFlow server from the minimal deployment:

```bash
cd ../terraform-minimal
terraform output mlflow_tracking_uri
```

### 2. Configure Variables

Copy the example variables file and update it:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and update the `mlflow_tracking_uri` with your actual MLFlow server IP:

```hcl
mlflow_tracking_uri = "http://YOUR_MLFLOW_SERVER_IP:5000"
```

### 3. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Plan the deployment
terraform plan

# Apply the configuration
terraform apply
```

### 4. Access SageMaker Notebook

After deployment, you can access the SageMaker Notebook Instance through the AWS Console or use the output URL:

```bash
terraform output sagemaker_notebook_url
```

## Infrastructure Components

### VPC and Networking
- **VPC**: Isolated network environment for SageMaker resources
- **Public Subnets**: For NAT Gateway and internet access
- **Private Subnets**: For SageMaker training jobs and notebook instances
- **NAT Gateway**: For outbound internet access from private subnets
- **VPC Endpoints**: For efficient S3 access without internet routing

### SageMaker Resources
- **Notebook Instance**: Interactive development environment
- **Execution Role**: IAM role for SageMaker training jobs
- **Notebook Role**: IAM role for SageMaker notebook operations

### S3 Storage
- **Data Bucket**: For training data and datasets
- **Models Bucket**: For trained model artifacts

### Security
- **Security Groups**: Network access control
- **IAM Policies**: Fine-grained permissions for AWS services

## MLFlow Integration

The infrastructure is pre-configured to connect to your existing MLFlow server:

1. **Environment Variables**: MLFlow tracking URI is set in the notebook environment
2. **Network Access**: Security groups allow communication with MLFlow server
3. **Authentication**: Uses IAM roles for secure AWS service access

## Usage

### 1. Upload Training Data

Upload your CIFAR-10 dataset to the data bucket:

```bash
# Get bucket name
DATA_BUCKET=$(terraform output -raw sagemaker_data_bucket_name)

# Upload CIFAR-10 data
aws s3 cp data/raw/cifar-10-python.tar.gz s3://$DATA_BUCKET/data/
```

### 2. Create Training Job

Use the SageMaker Notebook Instance to create and run training jobs:

```python
import sagemaker
from sagemaker.tensorflow import TensorFlow

# Create training job
estimator = TensorFlow(
    entry_point='train.py',
    source_dir='./src',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.11',
    py_version='py39',
    hyperparameters={
        'epochs': 50,
        'batch-size': 32,
        'learning-rate': 0.001
    }
)

# Start training
estimator.fit({'training': f's3://{DATA_BUCKET}/data/'})
```

### 3. Monitor Training

Training logs and metrics will be automatically sent to your MLFlow server. You can monitor progress through:

- **MLFlow UI**: `http://YOUR_MLFLOW_SERVER_IP:5000`
- **SageMaker Console**: Training job logs and metrics
- **CloudWatch**: Detailed logs and monitoring

## Cost Optimization

### Instance Types
- **Notebook**: `ml.t3.medium` (suitable for development)
- **Training**: `ml.m5.large` (good balance of performance and cost)

### Storage
- **S3**: Pay-per-use storage for data and models
- **EBS**: 20GB for notebook instance (adjustable)

### Networking
- **NAT Gateway**: Consider using NAT Instance for cost savings in development
- **VPC Endpoints**: Reduce NAT Gateway usage for S3 access

## Security Considerations

1. **Network Isolation**: Resources are deployed in private subnets
2. **IAM Roles**: Least privilege access for SageMaker operations
3. **S3 Encryption**: Buckets are configured with encryption at rest
4. **Security Groups**: Restrictive network access rules

## Troubleshooting

### Common Issues

1. **MLFlow Connection Failed**
   - Verify MLFlow server is running and accessible
   - Check security group rules allow port 5000
   - Confirm MLFlow tracking URI is correct

2. **SageMaker Training Job Fails**
   - Check IAM role permissions
   - Verify S3 bucket access
   - Review CloudWatch logs for detailed error messages

3. **Notebook Instance Won't Start**
   - Check IAM role permissions
   - Verify subnet and security group configuration
   - Review lifecycle configuration logs

### Useful Commands

```bash
# Check MLFlow server status
curl http://YOUR_MLFLOW_SERVER_IP:5000/health

# List SageMaker training jobs
aws sagemaker list-training-jobs

# Check notebook instance status
aws sagemaker describe-notebook-instance --notebook-instance-name cifar-sagemaker-notebook
```

## Cleanup

To destroy the infrastructure:

```bash
terraform destroy
```

**Note**: This will delete all resources including S3 buckets. Make sure to backup any important data before running destroy.

## Next Steps

1. **Model Development**: Use the notebook instance to develop and test new model architectures
2. **Hyperparameter Tuning**: Implement SageMaker Hyperparameter Tuning for automated optimization
3. **Model Deployment**: Deploy trained models using SageMaker Endpoints
4. **Monitoring**: Set up CloudWatch alarms for training job monitoring
5. **CI/CD**: Integrate with GitHub Actions or AWS CodePipeline for automated training workflows












