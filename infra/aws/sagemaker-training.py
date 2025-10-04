"""
AWS SageMaker training script for CIFAR-10 CNN models.
This script can be used to train different CNN architectures on SageMaker with MLflow integration.
"""

import os
import json
import boto3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class SageMakerTrainer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.ecr = boto3.client('ecr', region_name=region)
        
    def create_training_job(self, 
                          job_name: str,
                          model_type: str = 'base_cnn',
                          instance_type: str = 'ml.g4dn.xlarge',
                          instance_count: int = 1,
                          hyperparameters: Dict[str, str] = None,
                          bucket_name: str = None,
                          mlflow_tracking_uri: str = None,
                          ecr_repository_uri: str = None) -> Dict[str, Any]:
        """
        Create a SageMaker training job for CIFAR-10 model.
        
        Args:
            job_name: Unique name for the training job
            model_type: Type of model to train ('base_cnn', 'resnet', 'efficientnet')
            instance_type: SageMaker instance type
            instance_count: Number of instances
            hyperparameters: Training hyperparameters
            bucket_name: S3 bucket for storing data and models
            mlflow_tracking_uri: MLflow tracking server URI
            ecr_repository_uri: ECR repository URI for custom training image
            
        Returns:
            Dictionary with training job response
        """
        
        if hyperparameters is None:
            hyperparameters = {
                'epochs': '50',
                'batch_size': '32',
                'learning_rate': '0.001',
                'model_type': model_type
            }
        
        # Add MLflow tracking URI to hyperparameters
        if mlflow_tracking_uri:
            hyperparameters['mlflow_tracking_uri'] = mlflow_tracking_uri
        
        # Default bucket name if not provided
        if bucket_name is None:
            account_id = boto3.client('sts').get_caller_identity()['Account']
            bucket_name = f'cifar-training-{account_id}-{self.region}'
        
        # Create bucket if it doesn't exist
        self._ensure_bucket_exists(bucket_name)
        
        # Upload training data
        data_path = self._upload_data(bucket_name)
        
        # Use custom ECR image or default TensorFlow image
        training_image = ecr_repository_uri or self._get_training_image()
        
        # Create training job
        training_job_config = {
            'TrainingJobName': job_name,
            'RoleArn': self._get_sagemaker_role(),
            'AlgorithmSpecification': {
                'TrainingInputMode': 'File',
                'TrainingImage': training_image
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': instance_count,
                'VolumeSizeInGB': 30
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': data_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{bucket_name}/output/'
            },
            'HyperParameters': hyperparameters,
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 7200  # 2 hours
            },
            'Environment': {
                'AWS_DEFAULT_REGION': self.region
            }
        }
        
        try:
            response = self.sagemaker.create_training_job(**training_job_config)
            print(f"Training job '{job_name}' created successfully!")
            print(f"Job ARN: {response['TrainingJobArn']}")
            return response
        except Exception as e:
            print(f"Error creating training job: {e}")
            raise
    
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure S3 bucket exists, create if it doesn't."""
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except:
            print(f"Creating S3 bucket: {bucket_name}")
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
    
    def wait_for_training_job(self, job_name: str, max_wait_time: int = 3600) -> str:
        """
        Wait for training job to complete.
        
        Args:
            job_name: Name of the training job
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Final status of the training job
        """
        print(f"Waiting for training job '{job_name}' to complete...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            print(f"Training job status: {status}")
            
            if status in ['Completed', 'Failed', 'Stopped']:
                if status == 'Completed':
                    print("Training job completed successfully!")
                else:
                    print(f"Training job {status.lower()}!")
                    if 'FailureReason' in response:
                        print(f"Failure reason: {response['FailureReason']}")
                return status
            
            time.sleep(30)  # Wait 30 seconds before checking again
        
        print(f"Training job did not complete within {max_wait_time} seconds")
        return 'Timeout'
    
    def get_training_job_logs(self, job_name: str) -> List[str]:
        """
        Get CloudWatch logs for a training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            List of log entries
        """
        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            log_group_name = f"/aws/sagemaker/TrainingJobs/{job_name}"
            
            logs_client = boto3.client('logs', region_name=self.region)
            
            # Get log streams
            log_streams = logs_client.describe_log_streams(
                logGroupName=log_group_name,
                orderBy='LastEventTime',
                descending=True
            )
            
            logs = []
            for stream in log_streams['logStreams']:
                events = logs_client.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=stream['logStreamName']
                )
                
                for event in events['events']:
                    logs.append(f"[{stream['logStreamName']}] {event['message']}")
            
            return logs
        except Exception as e:
            print(f"Error getting logs: {e}")
            return []
    
    def list_training_jobs(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List recent training jobs.
        
        Args:
            max_results: Maximum number of jobs to return
            
        Returns:
            List of training job information
        """
        try:
            response = self.sagemaker.list_training_jobs(MaxResults=max_results)
            return response['TrainingJobSummaries']
        except Exception as e:
            print(f"Error listing training jobs: {e}")
            return []
    
    def _upload_data(self, bucket_name: str) -> str:
        """Upload CIFAR-10 data to S3."""
        data_key = 'data/cifar-10-python.tar.gz'
        
        # Check if data already exists
        try:
            self.s3.head_object(Bucket=bucket_name, Key=data_key)
            print(f"Data already exists at s3://{bucket_name}/{data_key}")
        except:
            print("Please upload CIFAR-10 data to S3 first:")
            print(f"aws s3 cp data/raw/cifar-10-python.tar.gz s3://{bucket_name}/{data_key}")
        
        return f's3://{bucket_name}/data/'
    
    def _get_sagemaker_role(self) -> str:
        """Get SageMaker execution role ARN."""
        # This should be replaced with your actual SageMaker role ARN
        account_id = boto3.client('sts').get_caller_identity()['Account']
        return f'arn:aws:iam::{account_id}:role/SageMakerExecutionRole'
    
    def _get_training_image(self) -> str:
        """Get SageMaker training image URI."""
        # Using TensorFlow 2.11 CPU image
        return '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.11.0-cpu-py39-ubuntu20.04-sagemaker'
    
    def _generate_training_script(self, model_type: str) -> str:
        """Generate training script for SageMaker."""
        return f'''
import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import tarfile

def load_cifar10_data():
    """Load CIFAR-10 data from S3."""
    # This will be implemented based on your data loading logic
    pass

def create_model(model_type='base_cnn'):
    """Create model based on type."""
    if model_type == 'base_cnn':
        # Your Base_CNN implementation
        pass
    elif model_type == 'resnet':
        # ResNet implementation
        pass
    elif model_type == 'efficientnet':
        # EfficientNet implementation
        pass
    
def train_model():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_type', type=str, default='base_cnn')
    
    args = parser.parse_args()
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    
    # Create model
    model = create_model(args.model_type)
    
    # Train model
    # ... training logic ...
    
    # Save model
    model.save('/opt/ml/model/model.h5')
    
    print("Training completed successfully!")

if __name__ == '__main__':
    train_model()
'''

def main():
    """Example usage of SageMakerTrainer."""
    trainer = SageMakerTrainer()
    
    # Train different model architectures
    models_to_train = [
        {'model_type': 'base_cnn', 'instance_type': 'ml.g4dn.xlarge'},
        {'model_type': 'resnet', 'instance_type': 'ml.g4dn.2xlarge'},
        {'model_type': 'efficientnet', 'instance_type': 'ml.g4dn.4xlarge'}
    ]
    
    for i, config in enumerate(models_to_train):
        job_name = f'cifar-{config["model_type"]}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        
        print(f"Starting training job for {config['model_type']}...")
        trainer.create_training_job(
            job_name=job_name,
            model_type=config['model_type'],
            instance_type=config['instance_type'],
            hyperparameters={
                'epochs': '100',
                'batch_size': '64',
                'learning_rate': '0.001',
                'model_type': config['model_type']
            }
        )

if __name__ == '__main__':
    main()



