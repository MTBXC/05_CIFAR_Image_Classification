# 🎯 CIFAR-10 Image Classification: MLOps Pipeline

A production-ready MLOps pipeline for CIFAR-10 image classification, featuring **Kaggle Notebook training**, **MLflow experiment tracking**, and **FastAPI deployment** with AWS infrastructure.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

## 🌟 Overview

This project demonstrates a complete MLOps workflow for training, tracking, and deploying a deep learning image classification model:

1. **Training**: Train models on Kaggle Notebooks with GPU acceleration
2. **Tracking**: Track experiments, metrics, and models with MLflow
3. **Deployment**: Deploy models as REST API using FastAPI
4. **Infrastructure**: Automated AWS deployment with Terraform

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CIFAR-10 MLOps Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Kaggle Notebook │      │   MLflow Server  │      │   FastAPI App    │
│                  │      │                  │      │                  │
│  • GPU Training  │─────▶│  • Tracking      │─────▶│  • REST API      │
│  • Data Aug.     │      │  • Metrics       │      │  • Predictions   │
│  • Automation    │      │  • Model Registry│      │  • Web UI        │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                         │                         │
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Infrastructure                            │
│  • EC2 (MLflow)  • ECS (FastAPI)  • S3 (Artifacts)             │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🚂 Training Pipeline
- **Kaggle Integration**: Leverage free GPU resources for training
- **Automated Workflow**: One-click training with `kaggle_mlflow_training.py`
- **Data Augmentation**: Built-in augmentation for improved generalization
- **Flexible Architecture**: Easy to extend with new model architectures

### 📊 Experiment Tracking
- **MLflow Integration**: Complete experiment lifecycle management
- **Metrics Logging**: Track accuracy, loss, F1-scores, and custom metrics
- **Model Registry**: Version control for trained models
- **Artifact Storage**: Save models, plots, and training curves
- **S3 Backend**: Scalable artifact storage on AWS

### 🌐 API Deployment
- **FastAPI Backend**: High-performance async REST API
- **Interactive UI**: Web interface for real-time predictions
- **Model Management**: Hot-reload models from MLflow registry
- **Docker Support**: Containerized deployment ready for production
- **AWS ECS Deployment**: Scalable container orchestration

### ☁️ Infrastructure as Code
- **Terraform Modules**: Automated AWS resource provisioning
- **Multi-Environment**: Separate configs for minimal, sagemaker, and production
- **Cost Optimized**: Smart resource allocation to minimize costs
- **Security**: IAM roles, security groups, and credential management

## 📁 Project Structure

```
├── api/                          # FastAPI application
│   └── app/
│       ├── main.py              # API endpoints and web UI
│       └── model_manager.py     # Model loading and management
│
├── src/                          # Core ML modules
│   ├── data/                    # Data loading and preprocessing
│   │   ├── load_cifar10.py     # CIFAR-10 dataset loader
│   │   └── augmentations.py    # Data augmentation pipeline
│   ├── models/                  # Model architectures
│   │   └── base_cnn.py         # Lightweight CNN model
│   ├── training/                # Training pipeline
│   │   └── trainer.py          # Training loop with callbacks
│   └── monitoring/              # MLflow integration
│       └── mlflow_monitor.py   # Experiment tracking utilities
│
├── infra/                        # Infrastructure as Code
│   └── aws/
│       ├── terraform-minimal/   # Basic EC2 + MLflow setup
│       ├── terraform-sagemaker/ # SageMaker training setup
│       └── terraform-webapi/    # ECS + FastAPI deployment
│
├── scripts/                      # Utility scripts
│   ├── deploy_infrastructure.sh # Deploy AWS resources
│   ├── start_training.py       # Start Kaggle training job
│   └── test_deployment.py      # Test deployed API
│
├── docker/                       # Docker configurations
│   └── nginx.conf               # Reverse proxy config
│
├── web/                          # Frontend files
│   └── public/
│       └── index.html           # Interactive prediction UI
│
├── kaggle_mlflow_training.py    # Main training script for Kaggle
├── train_base_cnn.py            # Local training script
├── run_web_app.py               # Launch API server
│
├── Dockerfile.training          # Docker image for training
├── Dockerfile.mlflow            # Docker image for MLflow server
├── Dockerfile.webapi            # Docker image for FastAPI
├── docker-compose.yml           # Multi-container orchestration
│
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Kaggle account (for training)
- AWS account (for deployment, optional)
- Docker (for containerized deployment, optional)

### 1️⃣ Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cifar-image-classification.git
cd cifar-image-classification

# Install dependencies
pip install -r requirements.txt

# Download CIFAR-10 dataset (auto-downloaded on first run)
python -c "from src.data.load_cifar10 import load_cifar10_data; load_cifar10_data()"
```

### 2️⃣ Training on Kaggle

```bash
# Upload kaggle_mlflow_training.py to Kaggle Notebook
# Configure MLflow tracking URI in notebook
# Run the notebook with GPU accelerator

# Or use automation script
python scripts/start_training.py --kaggle-notebook YOUR_NOTEBOOK
```

**Kaggle Notebook Setup:**
1. Create new notebook on Kaggle
2. Enable GPU accelerator
3. Upload `kaggle_mlflow_training.py`
4. Set MLflow tracking URI (your MLflow server)
5. Run the notebook

### 3️⃣ MLflow Server

**Option A: Local MLflow**
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access at http://localhost:5000
```

**Option B: AWS MLflow Server**
```bash
# Deploy MLflow on EC2
cd infra/aws/terraform-minimal
terraform init
terraform apply

# MLflow will be accessible at EC2 public IP
```

### 4️⃣ API Deployment

**Option A: Local Development**
```bash
# Start FastAPI server
python run_web_app.py

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Option B: Docker**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Services:
# - FastAPI: http://localhost:8000
# - MLflow: http://localhost:5000
```

**Option C: AWS ECS**
```bash
# Deploy to AWS ECS
cd infra/aws/terraform-webapi
terraform init
terraform apply

# API will be accessible via Load Balancer DNS
```

## 🔧 Configuration

### Training Configuration

Edit `kaggle_mlflow_training.py`:
```python
config = TrainingConfig(
    epochs=30,              # Number of training epochs
    batch_size=64,          # Batch size for training
    learning_rate=0.001,    # Initial learning rate
    dropout_rate=0.3,       # Dropout for regularization
    use_augmentation=True   # Enable data augmentation
)
```

### MLflow Configuration

Set environment variables:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

### API Configuration

Edit `api/app/main.py` or set environment variables:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MODEL_NAME=base_cnn
export MODEL_VERSION=latest
```

## 📊 Model Architecture

**Base CNN Model:**
- **Input**: 32×32×3 RGB images
- **Architecture**: 
  - Conv2D(32, 3×3) + ReLU + MaxPool
  - Conv2D(64, 3×3) + ReLU + MaxPool
  - Conv2D(64, 3×3) + ReLU + MaxPool
  - Flatten + Dense(64) + Dropout + Dense(10)
- **Parameters**: ~122K trainable parameters
- **Training Time**: ~5-10 minutes on Kaggle GPU
- **Accuracy**: 70-75% on CIFAR-10 test set

## 🧪 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI for predictions |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/images` | Get random test images |
| `POST` | `/api/predict` | Classify an image |
| `GET` | `/api/model-info` | Get model metadata |
| `GET` | `/docs` | Interactive API documentation |

### Example Usage

```bash
# Health check
curl http://localhost:8000/api/health

# Get random images
curl http://localhost:8000/api/images

# Make prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image_index": 42}'
```

## 🏗️ Infrastructure Deployment

### AWS Components

**Terraform Modules:**

1. **terraform-minimal**: Basic MLflow server setup
   - EC2 instance with MLflow
   - S3 bucket for artifacts
   - Security groups and IAM roles

2. **terraform-sagemaker**: SageMaker training setup
   - SageMaker notebook instance
   - IAM roles for training
   - S3 integration

3. **terraform-webapi**: Production API deployment
   - ECS Cluster with Fargate
   - Application Load Balancer
   - ECR for Docker images
   - Auto-scaling policies

### Deployment Steps

```bash
# 1. Configure AWS credentials
aws configure

# 2. Deploy MLflow server
cd infra/aws/terraform-minimal
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init
terraform apply

# 3. Deploy FastAPI
cd ../terraform-webapi
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init
terraform apply
```

## 🔒 Security Best Practices

- ✅ Never commit credentials (protected by `.gitignore`)
- ✅ Use IAM roles instead of access keys
- ✅ Enable VPC security groups
- ✅ Use HTTPS for production APIs
- ✅ Rotate credentials regularly
- ✅ Use AWS Secrets Manager for sensitive data

## 📈 Monitoring & Logging

### MLflow Tracking
- Training/validation metrics
- Per-class F1-scores
- Confusion matrices
- Learning curves
- Model artifacts

### API Monitoring
- Request/response logging
- Error tracking
- Performance metrics
- Health checks

## 🧰 Development

### Local Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black src/ api/

# Lint code
flake8 src/ api/
```

### Docker Development

```bash
# Build Docker images
docker-compose build

# Run services
docker-compose up

# View logs
docker-compose logs -f
```

## 📚 Documentation

- [AWS Deployment Guide](README_AWS_Deployment.md)
- [Minimal Setup Guide](README_AWS_Minimal_Deployment.md)
- [Kaggle Setup Guide](KAGGLE_AWS_SETUP.md)
- [Quick Start Guide](QUICK_START_AWS.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **TensorFlow/Keras**: Google Brain Team
- **MLflow**: Databricks
- **FastAPI**: Sebastián Ramírez
- **Kaggle**: Free GPU resources for training

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/cifar-image-classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cifar-image-classification/discussions)

---

**Built with ❤️ for the ML community**

*Showcase your MLOps skills with this production-ready pipeline!*
