#!/usr/bin/env python3
"""
Script to start SageMaker training jobs for CIFAR-10 CNN models.
This script provides an easy way to start training jobs with different configurations.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add infra to path
sys.path.append(str(Path(__file__).parent.parent / "infra" / "aws"))

from sagemaker_training import SageMakerTrainer

def load_terraform_outputs():
    """Load Terraform outputs."""
    outputs_file = Path("terraform-outputs.json")
    if not outputs_file.exists():
        print("Error: terraform-outputs.json not found")
        print("Please run ./scripts/deploy_infrastructure.sh first")
        sys.exit(1)
    
    with open(outputs_file) as f:
        return json.load(f)

def start_training_job(model_type: str, 
                      epochs: int = 50,
                      batch_size: int = 32,
                      learning_rate: float = 0.001,
                      instance_type: str = "ml.g4dn.xlarge",
                      wait_for_completion: bool = False):
    """Start a SageMaker training job."""
    
    # Load Terraform outputs
    outputs = load_terraform_outputs()
    
    # Get configuration from Terraform outputs
    training_bucket = outputs['s3_training_data_bucket']['value']
    mlflow_url = outputs['mlflow_url']['value']
    sagemaker_role_arn = outputs['sagemaker_execution_role_arn']['value']
    
    # Create trainer
    trainer = SageMakerTrainer()
    
    # Generate job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"cifar-{model_type}-{timestamp}"
    
    # Hyperparameters
    hyperparameters = {
        'epochs': str(epochs),
        'batch_size': str(batch_size),
        'learning_rate': str(learning_rate),
        'model_type': model_type
    }
    
    print(f"Starting training job: {job_name}")
    print(f"Model type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Instance type: {instance_type}")
    print(f"Training bucket: {training_bucket}")
    print(f"MLflow URL: {mlflow_url}")
    print()
    
    try:
        # Create training job
        response = trainer.create_training_job(
            job_name=job_name,
            model_type=model_type,
            instance_type=instance_type,
            hyperparameters=hyperparameters,
            bucket_name=training_bucket,
            mlflow_tracking_uri=mlflow_url
        )
        
        print(f"Training job created successfully!")
        print(f"Job ARN: {response['TrainingJobArn']}")
        print(f"Job name: {job_name}")
        print()
        
        if wait_for_completion:
            print("Waiting for training job to complete...")
            status = trainer.wait_for_training_job(job_name, max_wait_time=7200)  # 2 hours
            
            if status == 'Completed':
                print("Training completed successfully!")
                
                # Get logs
                logs = trainer.get_training_job_logs(job_name)
                if logs:
                    print("\nLast 10 log entries:")
                    for log in logs[-10:]:
                        print(f"  {log}")
            else:
                print(f"Training job {status.lower()}!")
                
                # Get logs for debugging
                logs = trainer.get_training_job_logs(job_name)
                if logs:
                    print("\nLast 10 log entries:")
                    for log in logs[-10:]:
                        print(f"  {log}")
        else:
            print("Training job started. You can monitor it in:")
            print(f"1. SageMaker Console: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs")
            print(f"2. MLflow UI: {mlflow_url}")
            print()
            print("To check status, run:")
            print(f"aws sagemaker describe-training-job --training-job-name {job_name}")
    
    except Exception as e:
        print(f"Error starting training job: {e}")
        sys.exit(1)

def list_training_jobs():
    """List recent training jobs."""
    trainer = SageMakerTrainer()
    jobs = trainer.list_training_jobs(max_results=10)
    
    if not jobs:
        print("No training jobs found")
        return
    
    print("Recent training jobs:")
    print("-" * 80)
    print(f"{'Job Name':<30} {'Status':<15} {'Creation Time':<20}")
    print("-" * 80)
    
    for job in jobs:
        creation_time = job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{job['TrainingJobName']:<30} {job['TrainingJobStatus']:<15} {creation_time:<20}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Start SageMaker training jobs')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start training command
    start_parser = subparsers.add_parser('start', help='Start a training job')
    start_parser.add_argument('--model-type', type=str, default='base_cnn',
                            choices=['base_cnn', 'resnet', 'efficientnet'],
                            help='Type of model to train')
    start_parser.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs')
    start_parser.add_argument('--batch-size', type=int, default=32,
                            help='Training batch size')
    start_parser.add_argument('--learning-rate', type=float, default=0.001,
                            help='Learning rate')
    start_parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                            help='SageMaker instance type')
    start_parser.add_argument('--wait', action='store_true',
                            help='Wait for training job to complete')
    
    # List jobs command
    list_parser = subparsers.add_parser('list', help='List recent training jobs')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_training_job(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            instance_type=args.instance_type,
            wait_for_completion=args.wait
        )
    elif args.command == 'list':
        list_training_jobs()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

