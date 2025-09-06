"""
Training module for CIFAR-10 classification with Base_CNN model.
Optimized for CPU training with small batch sizes.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.base_cnn import create_base_cnn_model
from data.load_cifar10 import load_cifar10_from_raw
from data.augmentations import build_augmentation_pipeline


@dataclass
class TrainingConfig:
    """Configuration for training the Base_CNN model."""
    
    # Data paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    
    # Model parameters
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    
    # Training parameters (CPU optimized)
    batch_size: int = 64  # Larger batch for faster training
    epochs: int = 10
    validation_split: float = 0.2
    
    # Augmentation
    use_augmentation: bool = False  # Disabled for speed
    
    # Callbacks
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    
    # Model saving
    model_save_dir: Path = Path("models")
    save_best_only: bool = True
    
    # MLFlow
    experiment_name: str = "cifar10_base_cnn"
    run_name: Optional[str] = None


class Trainer:
    """
    Trainer class for Base_CNN model on CIFAR-10 dataset.
    Optimized for CPU training with monitoring capabilities.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.history = None
        
        # Set up TensorFlow for CPU optimization
        self._setup_tensorflow()
        
        # Create directories
        self._create_directories()
    
    def _setup_tensorflow(self):
        """Configure TensorFlow for CPU training."""
        # Limit GPU memory growth if GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        # Set mixed precision for better CPU performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled for CPU optimization")
    
    def _create_directories(self):
        """Create necessary directories."""
        self.config.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load CIFAR-10 data using the existing data loading module.
        
        Returns:
            Tuple of (train_data, test_data) where each is (X, y)
        """
        print("Loading CIFAR-10 data...")
        
        # Load raw data
        train_data, test_data = load_cifar10_from_raw(self.config.raw_data_dir)
        
        # Normalize pixel values to [0, 1]
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        
        return train_data, test_data
    
    def prepare_data(self):
        """Prepare data for training with augmentation if enabled."""
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        
        # Create data generators
        if self.config.use_augmentation:
            print("Setting up data augmentation...")
            augmentation_pipeline = build_augmentation_pipeline()
            
            # Create training data generator with augmentation
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=lambda x: augmentation_pipeline(x, training=True),
                validation_split=self.config.validation_split
            )
            
            # Create validation data generator without augmentation
            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                validation_split=self.config.validation_split
            )
            
            # Generate training and validation data
            self.train_generator = train_datagen.flow(
                x_train, y_train,
                batch_size=self.config.batch_size,
                subset='training',
                shuffle=True
            )
            
            self.val_generator = val_datagen.flow(
                x_train, y_train,
                batch_size=self.config.batch_size,
                subset='validation',
                shuffle=False
            )
            
            # Calculate steps per epoch
            self.train_steps = len(self.train_generator)
            self.val_steps = len(self.val_generator)
            
            print(f"Training steps per epoch: {self.train_steps}")
            print(f"Validation steps per epoch: {self.val_steps}")
        
        else:
            # Simple train/validation split without augmentation
            from sklearn.model_selection import train_test_split
            
            x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
                x_train, y_train,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=y_train
            )
            
            self.train_data = (x_train_split, y_train_split)
            self.val_data = (x_val_split, y_val_split)
            
            print(f"Training samples: {len(x_train_split)}")
            print(f"Validation samples: {len(x_val_split)}")
    
    def create_model(self):
        """Create and compile the Base_CNN model."""
        print("Creating Base_CNN model...")
        
        self.model = create_base_cnn_model(
            input_shape=self.config.input_shape,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            learning_rate=self.config.learning_rate
        )
        
        print(f"Model created with {self.model.count_params():,} parameters")
        return self.model
    
    def setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(self) -> tf.keras.callbacks.History:
        """
        Train the Base_CNN model.
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        print("Starting training...")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        if hasattr(self, 'train_generator'):
            # Training with data generators (augmentation)
            self.history = self.model.fit(
                self.train_generator,
                steps_per_epoch=self.train_steps,
                validation_data=self.val_generator,
                validation_steps=self.val_steps,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Training with simple data split
            x_train, y_train = self.train_data
            x_val, y_val = self.val_data
            
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Evaluating model on test data...")
        x_test, y_test = self.test_data
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            x_test, y_test,
            batch_size=self.config.batch_size,
            verbose=1
        )
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def save_model(self, model_name: str = "base_cnn_cifar10"):
        """
        Save the trained model.
        
        Args:
            model_name: Name for the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path = self.config.model_save_dir / f"{model_name}.h5"
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        return model_path
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary including model info and results.
        
        Returns:
            Dictionary with training summary
        """
        if self.model is None or self.history is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get final metrics
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        # Get test results
        test_results = self.evaluate()
        
        summary = {
            'model_name': 'Base_CNN',
            'total_parameters': self.model.count_params(),
            'training_epochs': len(self.history.history['loss']),
            'final_train_accuracy': float(final_train_acc),
            'final_val_accuracy': float(final_val_acc),
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'test_accuracy': test_results['test_accuracy'],
            'test_loss': test_results['test_loss'],
            'config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'dropout_rate': self.config.dropout_rate,
                'use_augmentation': self.config.use_augmentation
            }
        }
        
        return summary


def main():
    """Main function to run training pipeline."""
    # Create training configuration
    config = TrainingConfig(
        epochs=10,
        batch_size=32,  # Small batch for CPU
        use_augmentation=True
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load and prepare data
    trainer.load_data()
    trainer.prepare_data()
    
    # Create and train model
    trainer.create_model()
    trainer.train()
    
    # Evaluate and save
    results = trainer.evaluate()
    model_path = trainer.save_model()
    
    # Print summary
    summary = trainer.get_training_summary()
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for key, value in summary.items():
        if key != 'config':
            print(f"{key}: {value}")
    
    print(f"\nModel saved to: {model_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
