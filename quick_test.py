"""
Quick test of simplified Base_CNN architecture.
Checks parameter count and forward pass speed.
"""

import sys
from pathlib import Path
import time
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.base_cnn import create_base_cnn_model


def test_model_speed():
    """Speed test for the new model."""
    print("="*50)
    print("Simplified Base_CNN Architecture Test")
    print("="*50)
    
    # Create model
    print("Creating model...")
    model = create_base_cnn_model()
    
    # Model info
    param_count = model.count_params()
    print(f"Parameter count: {param_count:,}")
    print(f"Model size: ~{param_count * 4 / 1024:.1f} KB")
    
    # Test forward pass speed
    print("\nTesting forward pass speed...")
    
    # Create dummy batch (simulating CIFAR-10 batch)
    batch_size = 64
    dummy_input = np.random.random((batch_size, 32, 32, 3)).astype(np.float32)
    
    # Warm up
    _ = model.predict(dummy_input, verbose=0)
    
    # Time multiple forward passes
    num_tests = 10
    start_time = time.time()
    
    for i in range(num_tests):
        _ = model.predict(dummy_input, verbose=0)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_tests
    samples_per_sec = batch_size / avg_time
    
    print(f"Average time per batch ({batch_size} samples): {avg_time:.3f}s")
    print(f"Samples per second: {samples_per_sec:.0f}")
    
    # Estimate training time
    total_samples = 50000  # CIFAR-10 training samples
    batches_per_epoch = total_samples // batch_size
    time_per_epoch = batches_per_epoch * avg_time
    
    print(f"\nEstimated time per epoch:")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Time per epoch: {time_per_epoch:.1f}s ({time_per_epoch/60:.1f} min)")
    print(f"  Time for 10 epochs: {time_per_epoch*10/60:.1f} min")
    
    # Test output shape
    output = model.predict(dummy_input[:1], verbose=0)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output sum (should be ~1): {np.sum(output):.6f}")
    
    print("\n" + "="*50)
    print("Model ready for fast training! ✓")
    print("="*50)


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    tf.random.set_seed(42)
    
    test_model_speed()