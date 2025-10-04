# CIFAR-10 MLOps Pipeline

End-to-end machine learning pipeline for CIFAR-10 image classification with experiment tracking and API deployment.

## Architecture

The pipeline consists of three main components:

1. **Training** - Kaggle Notebooks with GPU acceleration
2. **Tracking** - MLflow server for experiment management
3. **Deployment** - FastAPI service for model inference

```
Kaggle Notebook (GPU) → MLflow Server (EC2) → FastAPI API (ECS)
      ↓                        ↓                      ↓
   Training              Experiments              Predictions
   Metrics              Model Registry           Web Interface
   Artifacts            S3 Storage               REST API
```

## Project Structure

```
├── api/
│   └── app/
│       ├── main.py              # FastAPI endpoints
│       └── model_manager.py     # Model loading from MLflow
│
├── src/
│   ├── data/                    # CIFAR-10 data loaders
│   ├── models/                  # CNN architecture
│   ├── training/                # Training pipeline
│   └── monitoring/              # MLflow integration
│
├── infra/aws/
│   ├── terraform-minimal/       # MLflow server on EC2
│   └── terraform-webapi/        # FastAPI on ECS
│
├── scripts/                     # Deployment utilities
├── web/public/                  # Web UI
│
├── kaggle_mlflow_training.py   # Main training script
├── train_base_cnn.py           # Local training option
├── run_web_app.py              # Launch API locally
│
├── Dockerfile.mlflow           # MLflow server image
├── Dockerfile.webapi           # FastAPI image
├── Dockerfile.training         # Training image
└── docker-compose.yml          # Local development
```

## Quick Start

### Training

**On Kaggle:**
1. Upload `kaggle_mlflow_training.py` to a new Kaggle notebook
2. Enable GPU accelerator
3. Set MLflow tracking URI environment variable
4. Run the notebook

**Locally:**
```bash
pip install -r requirements.txt
python train_base_cnn.py
```

### MLflow Server

**Deploy on AWS:**
```bash
cd infra/aws/terraform-minimal
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your AWS settings
terraform init
terraform apply
```

**Run locally:**
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### API Deployment

**Deploy on AWS:**
```bash
cd infra/aws/terraform-webapi
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init
terraform apply
```

**Run locally:**
```bash
python run_web_app.py
# Access at http://localhost:8000
```

**With Docker:**
```bash
docker-compose up
```

## Configuration

### Training Config

Edit `kaggle_mlflow_training.py`:
```python
config = TrainingConfig(
    epochs=30,
    batch_size=64,
    learning_rate=0.001,
    dropout_rate=0.3,
    use_augmentation=True
)
```

### MLflow Config

Set environment variables:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### API Config

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MODEL_NAME=base_cnn
export MODEL_VERSION=latest
```

## Model

Simple CNN architecture:
- Input: 32×32×3 RGB images
- 3 convolutional blocks with max pooling
- Dense layer with dropout
- Output: 10 classes (softmax)
- Parameters: ~122K
- Training time: 5-10 minutes on Kaggle GPU
- Test accuracy: 70-75%

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check |
| `/api/images` | GET | Random test images |
| `/api/predict` | POST | Image classification |
| `/api/model-info` | GET | Model metadata |
| `/docs` | GET | API documentation |

## Infrastructure

### AWS Resources

**MLflow Server (terraform-minimal):**
- EC2 instance with MLflow
- S3 bucket for artifacts
- Security groups
- IAM roles

**FastAPI Service (terraform-webapi):**
- ECS cluster with Fargate
- Application Load Balancer
- ECR for Docker images
- Auto-scaling

### Deployment

```bash
# Deploy MLflow
cd infra/aws/terraform-minimal
terraform apply

# Deploy API
cd ../terraform-webapi
terraform apply
```

### Cleanup

```bash
terraform destroy
```

## Development

### Local Setup

```bash
git clone <repo-url>
cd cifar-image-classification
pip install -r requirements.txt
```

### Docker Development

```bash
docker-compose up --build
```

Services:
- FastAPI: http://localhost:8000
- MLflow: http://localhost:5000

## Security

- Never commit credentials (blocked by `.gitignore`)
- Use IAM roles instead of access keys
- Store secrets in AWS Secrets Manager
- Enable VPC security groups

## Documentation

- [AWS Deployment Guide](README_AWS_Deployment.md)
- [Minimal Setup Guide](README_AWS_Minimal_Deployment.md)
- [Kaggle Setup](KAGGLE_AWS_SETUP.md)
- [Quick Start](QUICK_START_AWS.md)

## Tech Stack

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- MLflow
- Docker
- Terraform
- AWS (EC2, ECS, S3, ECR)

## License

MIT License
