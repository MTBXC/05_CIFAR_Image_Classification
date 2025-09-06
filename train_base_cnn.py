"""
Main training script for Base_CNN on CIFAR-10 dataset.
Integrates training, monitoring, and evaluation in a complete pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.trainer import Trainer, TrainingConfig
from monitoring.mlflow_monitor import create_mlflow_monitor, CIFAR10_CLASS_NAMES


def main():
    """Main training pipeline with MLFlow monitoring."""
    
    print("="*60)
    print("CIFAR-10 Base_CNN Training Pipeline")
    print("="*60)
    
    # Create training configuration (CPU optimized for speed)
    config = TrainingConfig(
        epochs=10,                    # 10 epochs as requested
        batch_size=64,                # Larger batch for faster training
        learning_rate=0.001,
        dropout_rate=0.2,             # Reduced dropout
        use_augmentation=False,       # Disabled for speed
        early_stopping_patience=5,
        reduce_lr_patience=3
    )
    
    print(f"Training Configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Dropout Rate: {config.dropout_rate}")
    print(f"  Use Augmentation: {config.use_augmentation}")
    print()
    
    # Create MLFlow monitor
    mlflow_monitor = create_mlflow_monitor(
        experiment_name="cifar10_base_cnn_fast",
        run_name=f"simple_cnn_epochs_{config.epochs}_batch_{config.batch_size}"
    )
    
    # Start MLFlow run
    run_id = mlflow_monitor.start_run()
    print(f"MLFlow Run ID: {run_id}")
    print()
    
    try:
        # Create trainer
        trainer = Trainer(config)
        
        # Load and prepare data
        print("Loading CIFAR-10 data...")
        trainer.load_data()
        trainer.prepare_data()
        
        # Create model
        print("Creating Base_CNN model...")
        model = trainer.create_model()
        
        # Log model parameters to MLFlow
        model_params = {
            'model_name': 'Base_CNN',
            'total_parameters': model.count_params(),
            'input_shape': str(config.input_shape),
            'num_classes': config.num_classes,
            'dropout_rate': config.dropout_rate,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'use_augmentation': config.use_augmentation,
            'early_stopping_patience': config.early_stopping_patience,
            'reduce_lr_patience': config.reduce_lr_patience
        }
        mlflow_monitor.log_parameters(model_params)
        
        # Train model
        print("Starting training...")
        history = trainer.train()
        
        # Evaluate model
        print("Evaluating model...")
        test_results = trainer.evaluate()
        
        # Get predictions for detailed evaluation
        x_test, y_test = trainer.test_data
        y_pred_proba = model.predict(x_test, batch_size=config.batch_size, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Log complete experiment to MLFlow
        mlflow_monitor.log_complete_experiment(
            model=model,
            history=history,
            training_config=model_params,
            test_results=test_results,
            y_true=y_test,
            y_pred=y_pred,
            class_names=CIFAR10_CLASS_NAMES
        )
        
        # Save model
        model_path = trainer.save_model("base_cnn_cifar10_cpu")
        
        # Print final summary
        summary = trainer.get_training_summary()
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"Final Test Loss: {test_results['test_loss']:.4f}")
        print(f"Total Parameters: {summary['total_parameters']:,}")
        print(f"Training Epochs: {summary['training_epochs']}")
        print(f"Model saved to: {model_path}")
        print(f"MLFlow Run ID: {run_id}")
        print("="*60)
        
        # Print per-class accuracy
        print("\nPer-class Accuracy:")
        from sklearn.metrics import accuracy_score
        for i, class_name in enumerate(CIFAR10_CLASS_NAMES):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                print(f"  {class_name}: {class_acc:.4f}")
        
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


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    main()

