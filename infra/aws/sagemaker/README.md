# SageMaker Training for CIFAR-10 Models

This directory contains scripts and configurations for training CIFAR-10 models using Amazon SageMaker with MLFlow integration.

## Overview

The SageMaker training setup provides:
- **Multiple Model Architectures**: Base CNN, ResNet, and EfficientNet
- **MLFlow Integration**: Automatic experiment tracking and model logging
- **Scalable Training**: Support for distributed training on multiple instances
- **S3 Integration**: Automatic data loading and model artifact storage
- **Comprehensive Logging**: Detailed training metrics and model evaluation

## Files

- `train.py` - Main training script for SageMaker containers
- `sagemaker-training.py` - Training job launcher and manager
- `requirements.txt` - Python dependencies for training
- `README.md` - This documentation

## Quick Start

### 1. Prerequisites

- AWS CLI configured with appropriate permissions
- SageMaker execution role with necessary permissions
- MLFlow server running and accessible
- CIFAR-10 dataset uploaded to S3

### 2. Upload Training Data

```bash
# Upload CIFAR-10 dataset to S3
aws s3 cp data/raw/cifar-10-python.tar.gz s3://your-data-bucket/data/
```

### 3. Create Training Job

```bash
# Basic training job
python sagemaker-training.py create \
    --job-name "cifar10-base-cnn-$(date +%Y%m%d-%H%M%S)" \
    --model-type base_cnn \
    --epochs 50 \
    --mlflow-tracking-uri "http://your-mlflow-server:5000" \
    --data-bucket "your-data-bucket" \
    --model-bucket "your-model-bucket" \
    --wait

# Advanced training with ResNet
python sagemaker-training.py create \
    --job-name "cifar10-resnet-$(date +%Y%m%d-%H%M%S)" \
    --model-type resnet \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --instance-type ml.m5.xlarge \
    --mlflow-tracking-uri "http://your-mlflow-server:5000" \
    --data-bucket "your-data-bucket" \
    --model-bucket "your-model-bucket" \
    --wait
```

### 4. Monitor Training

```bash
# Monitor a specific job
python sagemaker-training.py monitor --job-name "your-job-name"

# List recent training jobs
python sagemaker-training.py list --limit 10

# Get training logs
python sagemaker-training.py logs --job-name "your-job-name"
```

## Model Architectures

### Base CNN
- Simple convolutional neural network
- Good for baseline comparisons
- Fast training and inference

### ResNet
- Residual neural network architecture
- Better gradient flow for deeper networks
- Improved accuracy on complex datasets

### EfficientNet
- State-of-the-art efficiency and accuracy
- Uses compound scaling
- Best performance with transfer learning

## Training Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model-type` | `base_cnn` | Model architecture to use |
| `epochs` | `50` | Number of training epochs |
| `batch-size` | `32` | Training batch size |
| `learning-rate` | `0.001` | Learning rate for optimizer |

### Instance Types

| Instance Type | vCPUs | Memory | GPU | Use Case |
|---------------|-------|--------|-----|----------|
| `ml.m5.large` | 2 | 8 GB | No | Development, small models |
| `ml.m5.xlarge` | 4 | 16 GB | No | Medium models, faster training |
| `ml.m5.2xlarge` | 8 | 32 GB | No | Large models, distributed training |
| `ml.p3.2xlarge` | 8 | 61 GB | 1x V100 | GPU training, deep networks |

## MLFlow Integration

### Automatic Tracking

The training script automatically logs:
- **Parameters**: Model type, hyperparameters, training configuration
- **Metrics**: Training/validation loss and accuracy, final test metrics
- **Artifacts**: Trained model, training history plots, evaluation reports
- **Tags**: SageMaker job information, model metadata

### Experiment Organization

Experiments are organized by:
- **Experiment Name**: `cifar10_{model_type}_sagemaker`
- **Run Name**: `sagemaker_{model_type}_{epochs}epochs`
- **Tags**: Include SageMaker job metadata

### Accessing Results

1. **MLFlow UI**: Visit your MLFlow server to view experiments
2. **S3 Artifacts**: Models and logs are stored in S3 buckets
3. **CloudWatch Logs**: Detailed training logs in CloudWatch

## Advanced Usage

### Custom Model Architectures

To add a new model architecture:

1. Add the model creation function to `train.py`:
```python
def create_custom_model(input_shape, num_classes):
    # Your custom model implementation
    pass
```

2. Update the model type choices in argument parser
3. Add the new type to the `create_model()` function

### Distributed Training

For distributed training across multiple instances:

```bash
python sagemaker-training.py create \
    --job-name "cifar10-distributed" \
    --instance-count 2 \
    --instance-type ml.m5.xlarge \
    --model-type resnet \
    --epochs 100
```

### Hyperparameter Tuning

Use SageMaker Hyperparameter Tuning for automated optimization:

```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

hyperparameter_ranges = {
    'learning-rate': ContinuousParameter(0.0001, 0.01),
    'batch-size': IntegerParameter(16, 128),
    'epochs': IntegerParameter(20, 100)
}

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='test_accuracy',
    objective_type='Maximize',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=2
)
```

## Troubleshooting

### Common Issues

1. **MLFlow Connection Failed**
   - Verify MLFlow server is accessible from SageMaker
   - Check security group rules allow port 5000
   - Confirm MLFlow tracking URI is correct

2. **Training Job Fails to Start**
   - Check IAM role permissions
   - Verify S3 bucket access
   - Review instance type availability in region

3. **Out of Memory Errors**
   - Reduce batch size
   - Use larger instance type
   - Implement gradient accumulation

4. **Slow Training**
   - Use GPU instances for deep models
   - Increase batch size if memory allows
   - Use distributed training for large datasets

### Debugging

1. **Check Training Logs**:
```bash
python sagemaker-training.py logs --job-name "your-job-name"
```

2. **Monitor Job Status**:
```bash
python sagemaker-training.py monitor --job-name "your-job-name"
```

3. **Review CloudWatch Metrics**:
   - Visit AWS CloudWatch console
   - Check SageMaker training job metrics
   - Review custom metrics logged by MLFlow

## Cost Optimization

### Instance Selection
- Use CPU instances for development and small models
- Use GPU instances only for deep networks or large datasets
- Consider spot instances for cost savings (with checkpointing)

### Training Efficiency
- Use appropriate batch sizes for your instance type
- Implement early stopping to avoid unnecessary training
- Use learning rate scheduling for faster convergence

### Storage Optimization
- Clean up old model artifacts from S3
- Use S3 lifecycle policies for automatic cleanup
- Compress model artifacts when possible

## Best Practices

1. **Experiment Tracking**
   - Use descriptive run names
   - Tag experiments with meaningful metadata
   - Regularly review and compare experiments

2. **Model Management**
   - Version your models in MLFlow
   - Store model artifacts in organized S3 structure
   - Document model performance and use cases

3. **Training Efficiency**
   - Start with smaller models for experimentation
   - Use transfer learning when appropriate
   - Implement proper data augmentation

4. **Monitoring**
   - Set up CloudWatch alarms for training jobs
   - Monitor MLFlow server health
   - Track training costs and resource usage

## Integration with Terraform

This SageMaker setup integrates with the Terraform infrastructure:

1. **Automatic Configuration**: Use Terraform outputs for bucket names and MLFlow URI
2. **IAM Integration**: SageMaker roles are created by Terraform
3. **Network Configuration**: VPC and security groups are managed by Terraform

Example integration:
```bash
# Get configuration from Terraform
DATA_BUCKET=$(terraform -chdir=../terraform-sagemaker output -raw sagemaker_data_bucket_name)
MODEL_BUCKET=$(terraform -chdir=../terraform-sagemaker output -raw sagemaker_models_bucket_name)
MLFLOW_URI=$(terraform -chdir=../terraform-minimal output -raw mlflow_tracking_uri)

# Create training job
python sagemaker-training.py create \
    --job-name "cifar10-integrated-$(date +%Y%m%d-%H%M%S)" \
    --data-bucket "$DATA_BUCKET" \
    --model-bucket "$MODEL_BUCKET" \
    --mlflow-tracking-uri "$MLFLOW_URI" \
    --wait
```












