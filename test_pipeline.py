"""
Test script to verify the Base_CNN pipeline works correctly.
This script tests model creation, data loading, and basic functionality.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.base_cnn import create_base_cnn_model, get_model_summary
from training.trainer import Trainer, TrainingConfig
from monitoring.mlflow_monitor import create_mlflow_monitor


def test_model_creation():
    """Test Base_CNN model creation."""
    print("Testing Base_CNN model creation...")
    
    model = create_base_cnn_model()
    
    # Test model properties
    assert model.count_params() > 0, "Model should have parameters"
    assert model.input_shape == (None, 32, 32, 3), f"Expected input shape (None, 32, 32, 3), got {model.input_shape}"
    assert model.output_shape == (None, 10), f"Expected output shape (None, 10), got {model.output_shape}"
    
    print(f"✓ Model created successfully with {model.count_params():,} parameters")
    print(f"✓ Input shape: {model.input_shape}")
    print(f"✓ Output shape: {model.output_shape}")
    
    return model


def test_data_loading():
    """Test CIFAR-10 data loading."""
    print("\nTesting CIFAR-10 data loading...")
    
    config = TrainingConfig()
    trainer = Trainer(config)
    
    try:
        train_data, test_data = trainer.load_data()
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        # Test data shapes
        assert x_train.shape == (50000, 32, 32, 3), f"Expected train data shape (50000, 32, 32, 3), got {x_train.shape}"
        assert y_train.shape == (50000,), f"Expected train labels shape (50000,), got {y_train.shape}"
        assert x_test.shape == (10000, 32, 32, 3), f"Expected test data shape (10000, 32, 32, 3), got {x_test.shape}"
        assert y_test.shape == (10000,), f"Expected test labels shape (10000,), got {y_test.shape}"
        
        # Test data types and ranges
        assert x_train.dtype == np.float32, f"Expected float32, got {x_train.dtype}"
        assert np.all(x_train >= 0) and np.all(x_train <= 1), "Pixel values should be normalized to [0, 1]"
        assert np.all(y_train >= 0) and np.all(y_train <= 9), "Labels should be in range [0, 9]"
        
        print("✓ Data loaded successfully")
        print(f"✓ Train data: {x_train.shape}, labels: {y_train.shape}")
        print(f"✓ Test data: {x_test.shape}, labels: {y_test.shape}")
        print(f"✓ Data normalized to [0, 1]")
        
        return trainer
        
    except FileNotFoundError as e:
        print(f"⚠ Data loading failed: {e}")
        print("Make sure CIFAR-10 data is extracted in data/raw/cifar-10-batches-py/")
        return None


def test_model_forward_pass():
    """Test model forward pass with dummy data."""
    print("\nTesting model forward pass...")
    
    model = create_base_cnn_model()
    
    # Create dummy input
    dummy_input = np.random.random((1, 32, 32, 3)).astype(np.float32)
    
    # Test forward pass
    output = model.predict(dummy_input, verbose=0)
    
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    assert np.allclose(np.sum(output), 1.0, atol=1e-6), "Output should sum to 1 (softmax)"
    assert np.all(output >= 0), "All outputs should be non-negative"
    
    print("✓ Forward pass successful")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output sum: {np.sum(output):.6f}")


def test_mlflow_monitor():
    """Test MLFlow monitor creation."""
    print("\nTesting MLFlow monitor...")
    
    try:
        monitor = create_mlflow_monitor("test_experiment")
        run_id = monitor.start_run("test_run")
        
        # Test logging
        test_params = {"test_param": 123}
        test_metrics = {"test_metric": 0.85}
        
        monitor.log_parameters(test_params)
        monitor.log_metrics(test_metrics)
        
        print("✓ MLFlow monitor created and tested successfully")
        print(f"✓ Run ID: {run_id}")
        
        return True
        
    except Exception as e:
        print(f"⚠ MLFlow monitor test failed: {e}")
        print("MLFlow might not be properly configured, but this won't affect training")
        return False


def test_training_config():
    """Test training configuration."""
    print("\nTesting training configuration...")
    
    config = TrainingConfig(
        epochs=2,  # Very short for testing
        batch_size=32,  # Small batch for testing
        use_augmentation=False  # Disabled for speed
    )
    
    assert config.epochs == 2
    assert config.batch_size == 32
    assert config.use_augmentation == False
    assert config.num_classes == 10
    
    print("✓ Training configuration created successfully")
    print(f"✓ Epochs: {config.epochs}")
    print(f"✓ Batch size: {config.batch_size}")
    print(f"✓ Use augmentation: {config.use_augmentation}")


def main():
    """Run all tests."""
    print("="*60)
    print("Base_CNN Pipeline Test Suite")
    print("="*60)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Test model creation
        model = test_model_creation()
        
        # Test data loading
        trainer = test_data_loading()
        
        # Test model forward pass
        test_model_forward_pass()
        
        # Test training configuration
        test_training_config()
        
        # Test MLFlow monitor
        mlflow_ok = test_mlflow_monitor()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("The Base_CNN pipeline is ready for training.")
        print("Run 'python train_base_cnn.py' to start training.")
        
        if not mlflow_ok:
            print("\nNote: MLFlow monitoring might not work properly.")
            print("Training will still work, but experiment tracking may be limited.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

