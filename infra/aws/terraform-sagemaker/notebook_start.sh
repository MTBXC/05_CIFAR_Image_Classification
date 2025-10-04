#!/bin/bash

# SageMaker Notebook Instance Lifecycle Configuration - On Start
# This script runs every time the notebook instance starts

set -e

echo "Starting SageMaker Notebook Instance..."

# Set MLFlow tracking URI
export MLFLOW_TRACKING_URI=${mlflow_tracking_uri}

# Start JupyterLab
cd /home/ec2-user/cifar-sagemaker

# Create a simple status check
cat > /home/ec2-user/cifar-sagemaker/status_check.py << 'EOF'
#!/usr/bin/env python3
"""
Status check for SageMaker Notebook Instance
"""

import os
import mlflow
import boto3
import sagemaker

def check_status():
    """Check the status of various components"""
    print("🔍 SageMaker Notebook Status Check")
    print("=" * 50)
    
    # Check MLFlow connection
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    print(f"MLFlow Tracking URI: {tracking_uri}")
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        print(f"✅ MLFlow: Connected ({len(experiments)} experiments)")
    except Exception as e:
        print(f"❌ MLFlow: Connection failed - {e}")
    
    # Check SageMaker session
    try:
        session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        print(f"✅ SageMaker: Session active (Role: {role})")
    except Exception as e:
        print(f"❌ SageMaker: Session failed - {e}")
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS: Credentials valid (Account: {identity['Account']})")
    except Exception as e:
        print(f"❌ AWS: Credentials failed - {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    check_status()
EOF

chmod +x /home/ec2-user/cifar-sagemaker/status_check.py

echo "SageMaker Notebook Instance started successfully!"












