"""
Simple SageMaker training script for CIFAR-10 CNN models.
This script focuses on a minimal CNN architecture with MLflow integration.
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
import requests
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SageMaker paths
SAGEMAKER_INPUT_DATA = '/opt/ml/input/data'
SAGEMAKER_MODEL_OUTPUT = '/opt/ml/model'
SAGEMAKER_OUTPUT = '/opt/ml/output'

# CIFAR-10 class names
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def test_mlflow_connection(mlflow_uri: str) -> bool:
    """Test connection to MLFlow server."""
    try:
        # Test HTTP connection
        parsed = requests.utils.urlparse(mlflow_uri)
        host = parsed.hostname
        port = parsed.port or 5000
        
        response = requests.get(f"http://{host}:{port}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ MLFlow server is accessible!")
            return True
        else:
            logger.warning(f"⚠️  MLFlow server responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to MLFlow server: {e}")
        return False


def load_cifar10_data():
    """Load CIFAR-10 data from SageMaker input."""
    logger.info("Loading CIFAR-10 data...")
    
    # Check if data is already extracted
    data_dir = Path('/opt/ml/input/data/training')
    cifar_dir = data_dir / 'cifar-10-batches-py'
    
    if not cifar_dir.exists():
        # Extract data from tar file
        tar_path = data_dir / 'cifar-10-python.tar.gz'
        if tar_path.exists():
            logger.info(f"Extracting {tar_path}")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_dir)
        else:
            raise FileNotFoundError(f"CIFAR-10 data not found at {tar_path}")
    
    # Load CIFAR-10 data manually
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict_data = pickle.load(fo, encoding='bytes')
        return dict_data
    
    # Load training data
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = cifar_dir / f'data_batch_{i}'
        batch_data = unpickle(batch_file)
        train_data.append(batch_data[b'data'])
        train_labels.extend(batch_data[b'labels'])
    
    # Load test data
    test_batch = unpickle(cifar_dir / 'test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    # Reshape and normalize
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    logger.info(f"Data loaded: {train_data.shape[0]} training samples, {test_data.shape[0]} test samples")
    return (train_data, train_labels), (test_data, test_labels)


def create_simple_cnn(input_shape, num_classes):
    """Very simple CNN model for CIFAR-10 - minimal architecture."""
    model = tf.keras.Sequential([
        # First conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block  
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(args):
    """Main training function."""
    logger.info("="*60)
    logger.info("SageMaker CIFAR-10 Simple CNN Training")
    logger.info("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create simple model
    model = create_simple_cnn(
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
    logger.info(f"MLFlow Tracking URI: {mlflow_tracking_uri}")
    
    # Test MLFlow connection
    if not test_mlflow_connection(mlflow_tracking_uri):
        logger.warning("MLFlow connection failed, continuing without tracking...")
        mlflow_monitor = None
        run_id = None
    else:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Start MLflow run
        try:
            experiment_name = f"cifar10_simple_cnn_sagemaker"
            run_name = f"sagemaker_simple_cnn_{args.epochs}epochs"
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception:
                experiment_id = mlflow.create_experiment(experiment_name)
            
            # Start run
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                run_id = run.info.run_id
                logger.info(f"MLFlow Run ID: {run_id}")
                
                # Log parameters
                training_params = {
                    'model_type': 'simple_cnn',
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'total_parameters': model.count_params(),
                    'input_shape': str(x_train.shape[1:]),
                    'num_classes': len(CIFAR10_CLASS_NAMES),
                    'training_samples': len(x_train),
                    'test_samples': len(x_test),
                    'sagemaker_job_name': os.environ.get('SAGEMAKER_JOB_NAME', 'unknown'),
                    'training_start_time': datetime.now().isoformat()
                }
                mlflow.log_params(training_params)
                
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
                logger.info("Starting training...")
                history = model.fit(
                    x_train, y_train,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate model
                logger.info("Evaluating model...")
                test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
                
                # Log results
                test_results = {
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'training_epochs': len(history.history['loss']),
                    'best_val_accuracy': max(history.history['val_accuracy']),
                    'best_val_loss': min(history.history['val_loss']),
                    'final_train_accuracy': history.history['accuracy'][-1],
                    'final_train_loss': history.history['loss'][-1]
                }
                mlflow.log_metrics(test_results)
                
                # Log model
                mlflow.tensorflow.log_model(
                    model, 
                    "model",
                    registered_model_name="cifar10_simple_cnn"
                )
                
        except Exception as e:
            logger.error(f"Failed to create MLflow run: {e}")
            mlflow_monitor = None
            run_id = None
    
    # Save model
    model_path = os.path.join(SAGEMAKER_MODEL_OUTPUT, 'model.h5')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training results
    results = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_epochs': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'model_type': 'simple_cnn',
        'total_parameters': int(model.count_params()),
        'training_completed_at': datetime.now().isoformat(),
        'mlflow_run_id': run_id
    }
    
    results_path = os.path.join(SAGEMAKER_OUTPUT, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    logger.info(f"Training Epochs: {len(history.history['loss'])}")
    logger.info(f"MLFlow Run ID: {run_id}")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train simple CIFAR-10 CNN on SageMaker')
    
    # Model parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    # MLFlow parameters
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                       help='MLFlow tracking URI (overrides environment variable)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Override MLFlow tracking URI if provided
    if args.mlflow_tracking_uri:
        os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_tracking_uri
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()












