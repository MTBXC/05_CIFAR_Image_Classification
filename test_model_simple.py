"""
Simple script for testing Base_CNN model on 20 random examples.
Text-only version without matplotlib visualization.
"""

import sys
from pathlib import Path
import numpy as np
import random

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.load_cifar10 import load_cifar10_from_raw

# CIFAR-10 class names
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_test_data():
    """Load and prepare test data."""
    print("Loading CIFAR-10 test data...")
    
    # Load data
    raw_data_dir = Path("data/raw")
    train_data, test_data = load_cifar10_from_raw(raw_data_dir)
    
    x_test, y_test = test_data
    
    # Normalize pixel values
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_test, y_test


def load_trained_model():
    """Load the trained model."""
    print("Loading trained model...")
    
    model_path = Path("models/base_cnn_cifar10_cpu.h5")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run training first: python train_base_cnn.py")
        return None
    
    try:
        import tensorflow as tf
        from models.base_cnn import Base_CNN
        
        # Load model with custom object scope
        with tf.keras.utils.custom_object_scope({'Base_CNN': Base_CNN}):
            model = tf.keras.models.load_model(model_path)
        
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_predictions(model, x_test, y_test, num_samples=20):
    """Test model predictions on random samples."""
    
    # Select random samples
    test_size = len(x_test)
    indices = random.sample(range(test_size), num_samples)
    indices = sorted(indices)
    
    print(f"\nTesting on {num_samples} random samples (indices: {indices})")
    print("="*80)
    
    # Get predictions
    x_selected = x_test[indices]
    predictions = model.predict(x_selected, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    correct_predictions = 0
    
    # Print results for each sample
    for i, idx in enumerate(indices):
        true_label = y_test[idx]
        predicted_label = predicted_classes[i]
        confidence = confidence_scores[i]
        is_correct = true_label == predicted_label
        
        if is_correct:
            correct_predictions += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{i+1:2d}. Index {idx:5d} | {status} | "
              f"True: {CIFAR10_CLASS_NAMES[true_label]:>10} | "
              f"Pred: {CIFAR10_CLASS_NAMES[predicted_label]:>10} | "
              f"Conf: {confidence:.3f}")
    
    # Calculate accuracy
    accuracy = correct_predictions / num_samples
    
    print("="*80)
    print(f"RESULTS: {correct_predictions}/{num_samples} correct ({accuracy:.1%})")
    print("="*80)
    
    # Show confusion for incorrect predictions
    incorrect_indices = [i for i, idx in enumerate(indices) if y_test[idx] != predicted_classes[i]]
    if incorrect_indices:
        print(f"\nIncorrect predictions ({len(incorrect_indices)}):")
        for i in incorrect_indices:
            idx = indices[i]
            true_label = y_test[idx]
            predicted_label = predicted_classes[i]
            confidence = confidence_scores[i]
            print(f"  Index {idx}: {CIFAR10_CLASS_NAMES[true_label]} → {CIFAR10_CLASS_NAMES[predicted_label]} (conf: {confidence:.3f})")
    
    # Show most confident predictions
    print(f"\nMost confident predictions:")
    top_confident = np.argsort(confidence_scores)[-3:][::-1]
    for i in top_confident:
        idx = indices[i]
        true_label = y_test[idx]
        predicted_label = predicted_classes[i]
        confidence = confidence_scores[i]
        status = "✓" if true_label == predicted_label else "✗"
        print(f"  {status} Index {idx}: {CIFAR10_CLASS_NAMES[predicted_label]} (conf: {confidence:.3f})")
    
    # Show least confident predictions
    print(f"\nLeast confident predictions:")
    least_confident = np.argsort(confidence_scores)[:3]
    for i in least_confident:
        idx = indices[i]
        true_label = y_test[idx]
        predicted_label = predicted_classes[i]
        confidence = confidence_scores[i]
        status = "✓" if true_label == predicted_label else "✗"
        print(f"  {status} Index {idx}: {CIFAR10_CLASS_NAMES[predicted_label]} (conf: {confidence:.3f})")
    
    return accuracy, correct_predictions


def main():
    """Main function."""
    
    print("="*60)
    print("Base_CNN Model Prediction Test (Simple Version)")
    print("="*60)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Load data and model
    x_test, y_test = load_test_data()
    model = load_trained_model()
    
    if model is None:
        return
    
    # Test predictions
    accuracy, correct = test_predictions(model, x_test, y_test, num_samples=20)
    
    print(f"\n" + "="*60)
    print("TEST COMPLETED!")
    print(f"Model accuracy on 20 random samples: {accuracy:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
