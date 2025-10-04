"""
SageMaker training script for CIFAR-10 CNN models.
This script is designed to run in SageMaker containers with MLflow integration.
"""

import os
import sys
import json
import argparse
import tarfile
import pickle
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import boto3
from pathlib import Path

# Add src to path
sys.path.append('/opt/ml/code')

from src.models.base_cnn import Base_CNN
from src.data.load_cifar10 import load_cifar10_from_raw
from src.monitoring.mlflow_monitor import create_mlflow_monitor, CIFAR10_CLASS_NAMES

# SageMaker paths
SAGEMAKER_INPUT_DATA = '/opt/ml/input/data'
SAGEMAKER_MODEL_OUTPUT = '/opt/ml/model'
SAGEMAKER_OUTPUT = '/opt/ml/output'


def download_data_from_s3(bucket_name: str, key: str, local_path: str):
    """Download data from S3."""
    s3 = boto3.client('s3')
    print(f"Downloading {key} from s3://{bucket_name} to {local_path}")
    s3.download_file(bucket_name, key, local_path)


def upload_model_to_s3(model_path: str, bucket_name: str, key: str):
    """Upload model to S3."""
    s3 = boto3.client('s3')
    print(f"Uploading {model_path} to s3://{bucket_name}/{key}")
    s3.upload_file(model_path, bucket_name, key)


def load_cifar10_data():
    """Load CIFAR-10 data from SageMaker input."""
    print("Loading CIFAR-10 data...")
    
    # Check if data is already extracted
    data_dir = Path('/opt/ml/input/data/training')
    cifar_dir = data_dir / 'cifar-10-batches-py'
    
    if not cifar_dir.exists():
        # Extract data from tar file
        tar_path = data_dir / 'cifar-10-python.tar.gz'
        if tar_path.exists():
            print(f"Extracting {tar_path}")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_dir)
        else:
            raise FileNotFoundError(f"CIFAR-10 data not found at {tar_path}")
    
    # Load data using existing function
    train_data, test_data = load_cifar10_from_raw(data_dir)
    
    print(f"Data loaded: {train_data[0].shape[0]} training samples, {test_data[0].shape[0]} test samples")
    return train_data, test_data


def create_model(model_type: str = 'base_cnn', input_shape=(32, 32, 3), num_classes=10):
    """Create model based on type."""
    print(f"Creating {model_type} model...")
    
    if model_type == 'base_cnn':
        model = Base_CNN(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model created with {model.count_params():,} parameters")
    return model


def train_model(args):
    """Main training function."""
    print("="*60)
    print("SageMaker CIFAR-10 Training")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create model
    model = create_model(
        model_type=args.model_type,
        input_shape=x_train.shape[1:],
        num_classes=len(CIFAR10_CLASS_NAMES)
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup MLflow tracking
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Create MLflow monitor
    mlflow_monitor = create_mlflow_monitor(
        experiment_name=f"cifar10_{args.model_type}_sagemaker",
        run_name=f"sagemaker_{args.model_type}_{args.epochs}epochs"
    )
    
    # Start MLflow run
    run_id = mlflow_monitor.start_run()
    print(f"MLFlow Run ID: {run_id}")
    
    try:
        # Log parameters
        training_params = {
            'model_type': args.model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'total_parameters': model.count_params(),
            'input_shape': str(x_train.shape[1:]),
            'num_classes': len(CIFAR10_CLASS_NAMES),
            'training_samples': len(x_train),
            'test_samples': len(x_test)
        }
        mlflow_monitor.log_parameters(training_params)
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print("Starting training...")
        history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Get predictions for detailed evaluation
        y_pred_proba = model.predict(x_test, batch_size=args.batch_size, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Log results
        test_results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        }
        mlflow_monitor.log_metrics(test_results)
        mlflow_monitor.log_training_history(history)
        mlflow_monitor.log_evaluation_metrics(y_test, y_pred, CIFAR10_CLASS_NAMES)
        
        # Log model
        mlflow_monitor.log_model(model, f"{args.model_type}_sagemaker")
        
        # Save model
        model_path = os.path.join(SAGEMAKER_MODEL_OUTPUT, 'model.h5')
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save model summary
        summary_path = os.path.join(SAGEMAKER_MODEL_OUTPUT, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Save training results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'training_epochs': len(history.history['loss']),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_loss': float(min(history.history['val_loss']))
        }
        
        results_path = os.path.join(SAGEMAKER_OUTPUT, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Training Epochs: {len(history.history['loss'])}")
        print(f"MLFlow Run ID: {run_id}")
        print("="*60)
        
        # Upload model to S3 if bucket is specified
        if args.model_bucket:
            model_s3_key = f"models/{args.model_type}/model_{run_id}.h5"
            upload_model_to_s3(model_path, args.model_bucket, model_s3_key)
            print(f"Model uploaded to s3://{args.model_bucket}/{model_s3_key}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Get run info
        run_info = mlflow_monitor.get_run_info()
        if run_info:
            print(f"\nMLFlow Run Info:")
            print(f"  Status: {run_info.get('status', 'Unknown')}")
            print(f"  Artifact URI: {run_info.get('artifact_uri', 'Unknown')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train CIFAR-10 model on SageMaker')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='base_cnn',
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    # S3 parameters
    parser.add_argument('--model-bucket', type=str, default=None,
                       help='S3 bucket to upload trained model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()

