#!/bin/bash

# SageMaker Notebook Instance Lifecycle Configuration - On Create
# This script runs when the notebook instance is created

set -e

echo "Starting SageMaker Notebook Instance setup..."

# Update system packages
apt-get update -y

# Install Python packages
pip3 install --upgrade pip

# Install MLFlow and dependencies
pip3 install mlflow==2.8.1
pip3 install boto3==1.34.0
pip3 install sagemaker==2.200.0
pip3 install tensorflow==2.15.0
pip3 install numpy==1.24.3
pip3 install matplotlib==3.7.2
pip3 install seaborn==0.12.2
pip3 install scikit-learn==1.3.0
pip3 install pandas==2.0.3

# Install additional useful packages
pip3 install jupyterlab==4.0.9
pip3 install ipywidgets==8.1.1
pip3 install plotly==5.17.0

# Set MLFlow tracking URI
echo "export MLFLOW_TRACKING_URI=${mlflow_tracking_uri}" >> /home/ec2-user/.bashrc

# Create project directory
mkdir -p /home/ec2-user/cifar-sagemaker
cd /home/ec2-user/cifar-sagemaker

# Clone or create project structure
mkdir -p src/{data,models,training,monitoring,utils}
mkdir -p notebooks
mkdir -p scripts

# Create a simple test script
cat > /home/ec2-user/cifar-sagemaker/test_mlflow_connection.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify MLFlow connection from SageMaker Notebook
"""

import os
import mlflow
import requests
from urllib.parse import urlparse

def test_mlflow_connection():
    """Test connection to MLFlow server"""
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    
    print(f"Testing MLFlow connection to: {tracking_uri}")
    
    try:
        # Parse the URI
        parsed = urlparse(tracking_uri)
        host = parsed.hostname
        port = parsed.port or 5000
        
        # Test HTTP connection
        response = requests.get(f"http://{host}:{port}/health", timeout=10)
        if response.status_code == 200:
            print("✅ MLFlow server is accessible!")
        else:
            print(f"⚠️  MLFlow server responded with status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to MLFlow server: {e}")
        return False
    
    try:
        # Test MLFlow client
        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        print(f"✅ MLFlow client connected! Found {len(experiments)} experiments.")
        return True
        
    except Exception as e:
        print(f"❌ MLFlow client error: {e}")
        return False

if __name__ == "__main__":
    test_mlflow_connection()
EOF

chmod +x /home/ec2-user/cifar-sagemaker/test_mlflow_connection.py

# Create a sample training notebook
cat > /home/ec2-user/cifar-sagemaker/notebooks/sagemaker_training_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SageMaker CIFAR-10 Training Demo\n",
    "\n",
    "This notebook demonstrates how to train CIFAR-10 models using SageMaker with MLFlow integration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test MLFlow connection\n",
    "import sys\n",
    "sys.path.append('/home/ec2-user/cifar-sagemaker')\n",
    "\n",
    "from test_mlflow_connection import test_mlflow_connection\n",
    "test_mlflow_connection()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\n",
    "import boto3\n",
    "import sagemaker\n",
    "import mlflow\n",
    "import os\n",
    "from sagemaker.tensorflow import TensorFlow\n",
    "\n",
    "# Get SageMaker session\n",
    "sagemaker_session = sagemaker.Session()\n",
    "role = sagemaker.get_execution_role()\n",
    "\n",
    "print(f\"SageMaker Role: {role}\")\n",
    "print(f\"MLFlow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Set proper ownership
chown -R ec2-user:ec2-user /home/ec2-user/cifar-sagemaker

echo "SageMaker Notebook Instance setup completed successfully!"












