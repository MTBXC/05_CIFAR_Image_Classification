#!/usr/bin/env python3
"""
Simple SageMaker training job launcher for CIFAR-10 CNN models.
This script creates and manages simple SageMaker training jobs with MLFlow integration.
"""

print("Script starting...")
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

def get_sagemaker_session():
    """Get SageMaker session and execution role."""
    session = sagemaker.Session()
    # Use the specific SageMaker execution role created by Terraform
    role = "arn:aws:iam::217522444105:role/cifar-sagemaker-execution-role"
    return session, role

def create_simple_training_job(
    job_name: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    instance_type: str = 'ml.m5.large',
    mlflow_tracking_uri: str = None,
    data_bucket: str = None,
    model_bucket: str = None
):
    """Create and start a simple SageMaker training job."""
    
    print(f"Creating Simple SageMaker training job: {job_name}")
    print("="*60)
    
    # Get SageMaker session
    session, role = get_sagemaker_session()
    
    # Prepare hyperparameters
    hyperparameters = {
        'epochs': epochs,
        'batch-size': batch_size,
        'learning-rate': learning_rate
    }
    
    # Prepare environment variables
    environment = {
        'MLFLOW_TRACKING_URI': mlflow_tracking_uri or 'http://localhost:5000'
    }
    
    # Create TensorFlow estimator
    estimator = TensorFlow(
        entry_point='train_simple.py',
        source_dir='.',
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='2.11',
        py_version='py39',
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=f's3://{model_bucket}/output/' if model_bucket else None,
        base_job_name=job_name,
        max_run=24*60*60,  # 24 hours max
        keep_alive_period_in_seconds=0
    )
    
    # Prepare input data
    if data_bucket:
        inputs = {
            'training': f's3://{data_bucket}/data/'
        }
    else:
        inputs = None
    
    print(f"Training job configuration:")
    print(f"  Job Name: {job_name}")
    print(f"  Instance Type: {instance_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  MLFlow URI: {mlflow_tracking_uri}")
    print(f"  Data Source: {inputs}")
    print("="*60)
    
    # Start training job
    try:
        estimator.fit(inputs, wait=False)
        print(f"✅ Training job '{job_name}' started successfully!")
        print(f"Job ARN: {estimator.latest_training_job.job_arn}")
        return estimator
    except Exception as e:
        print(f"❌ Failed to start training job: {e}")
        raise

def monitor_training_job(job_name: str, wait_for_completion: bool = True):
    """Monitor a SageMaker training job."""
    sagemaker_client = boto3.client('sagemaker')
    
    print(f"Monitoring training job: {job_name}")
    print("="*60)
    
    while True:
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            print(f"Status: {status}")
            
            if status in ['Completed', 'Failed', 'Stopped']:
                if status == 'Completed':
                    print("✅ Training job completed successfully!")
                    
                    # Print final metrics
                    if 'FinalMetricDataList' in response:
                        print("\nFinal Metrics:")
                        for metric in response['FinalMetricDataList']:
                            print(f"  {metric['MetricName']}: {metric['Value']}")
                    
                    # Print model artifacts
                    if 'ModelArtifacts' in response:
                        print(f"\nModel Artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
                
                elif status == 'Failed':
                    print("❌ Training job failed!")
                    if 'FailureReason' in response:
                        print(f"Failure Reason: {response['FailureReason']}")
                
                elif status == 'Stopped':
                    print("⏹️  Training job was stopped!")
                
                break
            
            elif status in ['InProgress', 'Starting']:
                print(f"⏳ Training in progress... (Status: {status})")
                
                # Print current metrics if available
                if 'SecondaryStatusTransitions' in response:
                    latest_transition = response['SecondaryStatusTransitions'][-1]
                    print(f"  Current Phase: {latest_transition['StatusMessage']}")
            
            if not wait_for_completion:
                break
                
            time.sleep(30)  # Wait 30 seconds before next check
            
        except Exception as e:
            print(f"Error monitoring job: {e}")
            break

def list_training_jobs(limit: int = 10):
    """List recent SageMaker training jobs."""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        response = sagemaker_client.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=limit
        )
        
        print(f"Recent Training Jobs (last {limit}):")
        print("="*80)
        print(f"{'Job Name':<40} {'Status':<15} {'Creation Time':<20}")
        print("-"*80)
        
        for job in response['TrainingJobSummaries']:
            creation_time = job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"{job['TrainingJobName']:<40} {job['TrainingJobStatus']:<15} {creation_time}")
        
    except Exception as e:
        print(f"Error listing training jobs: {e}")

def main():
    """Main entry point."""
    print("Starting SageMaker Simple Training Job Manager...")
    parser = argparse.ArgumentParser(description='Simple SageMaker CIFAR-10 Training Job Manager')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create training job command
    create_parser = subparsers.add_parser('create', help='Create a new simple training job')
    create_parser.add_argument('--job-name', type=str, required=True,
                              help='Name for the training job')
    create_parser.add_argument('--epochs', type=int, default=20,
                              help='Number of training epochs')
    create_parser.add_argument('--batch-size', type=int, default=32,
                              help='Training batch size')
    create_parser.add_argument('--learning-rate', type=float, default=0.001,
                              help='Learning rate')
    create_parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                              help='SageMaker instance type')
    create_parser.add_argument('--mlflow-tracking-uri', type=str,
                              help='MLFlow tracking URI')
    create_parser.add_argument('--data-bucket', type=str,
                              help='S3 bucket containing training data')
    create_parser.add_argument('--model-bucket', type=str,
                              help='S3 bucket for model artifacts')
    create_parser.add_argument('--wait', action='store_true',
                              help='Wait for training job to complete')
    
    # Monitor training job command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor a training job')
    monitor_parser.add_argument('--job-name', type=str, required=True,
                               help='Name of the training job to monitor')
    monitor_parser.add_argument('--wait', action='store_true', default=True,
                               help='Wait for completion')
    
    # List training jobs command
    list_parser = subparsers.add_parser('list', help='List recent training jobs')
    list_parser.add_argument('--limit', type=int, default=10,
                            help='Maximum number of jobs to list')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        estimator = create_simple_training_job(
            job_name=args.job_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            instance_type=args.instance_type,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            data_bucket=args.data_bucket,
            model_bucket=args.model_bucket
        )
        
        if args.wait:
            monitor_training_job(args.job_name, wait_for_completion=True)
    
    elif args.command == 'monitor':
        monitor_training_job(args.job_name, wait_for_completion=args.wait)
    
    elif args.command == 'list':
        list_training_jobs(limit=args.limit)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
