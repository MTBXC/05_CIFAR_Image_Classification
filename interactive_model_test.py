"""
Interactive script for testing Base_CNN model.
Allows selection of specific indices or random samples.
"""

import sys
from pathlib import Path
import numpy as np

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
    
    raw_data_dir = Path("data/raw")
    train_data, test_data = load_cifar10_from_raw(raw_data_dir)
    
    x_test, y_test = test_data
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Test data loaded: {x_test.shape[0]} samples")
    return x_test, y_test


def load_trained_model():
    """Load the trained model."""
    model_path = Path("models/base_cnn_cifar10_cpu.h5")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    try:
        import tensorflow as tf
        from models.base_cnn import Base_CNN
        
        # Load model with custom object scope
        with tf.keras.utils.custom_object_scope({'Base_CNN': Base_CNN}):
            model = tf.keras.models.load_model(model_path)
        
        print(f"Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_single_prediction(model, x_test, y_test, index):
    """Test prediction for a single sample."""
    
    if index < 0 or index >= len(x_test):
        print(f"Index {index} is out of range (0-{len(x_test)-1})")
        return
    
    # Get prediction
    sample = x_test[index:index+1]
    prediction = model.predict(sample, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    true_class = y_test[index]
    
    print(f"\nSample {index}:")
    print(f"  True class:  {CIFAR10_CLASS_NAMES[true_class]} (class {true_class})")
    print(f"  Predicted:   {CIFAR10_CLASS_NAMES[predicted_class]} (class {predicted_class})")
    print(f"  Confidence:  {confidence:.4f}")
    print(f"  Correct:     {'✓ YES' if true_class == predicted_class else '✗ NO'}")
    
    # Show all class probabilities
    print(f"\n  All class probabilities:")
    for i, prob in enumerate(prediction[0]):
        marker = "→" if i == predicted_class else " "
        print(f"    {marker} {CIFAR10_CLASS_NAMES[i]:>10}: {prob:.4f}")


def test_multiple_predictions(model, x_test, y_test, indices):
    """Test predictions for multiple samples."""
    
    print(f"\nTesting {len(indices)} samples...")
    print("="*70)
    
    # Get predictions
    x_selected = x_test[indices]
    predictions = model.predict(x_selected, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    correct = 0
    
    for i, idx in enumerate(indices):
        true_class = y_test[idx]
        predicted_class = predicted_classes[i]
        confidence = confidence_scores[i]
        is_correct = true_class == predicted_class
        
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{i+1:2d}. Index {idx:5d} | {status} | "
              f"True: {CIFAR10_CLASS_NAMES[true_class]:>10} | "
              f"Pred: {CIFAR10_CLASS_NAMES[predicted_class]:>10} | "
              f"Conf: {confidence:.3f}")
    
    accuracy = correct / len(indices)
    print("="*70)
    print(f"Results: {correct}/{len(indices)} correct ({accuracy:.1%})")
    
    return accuracy


def show_class_distribution(y_test, indices):
    """Show class distribution in selected samples."""
    
    print(f"\nClass distribution in selected samples:")
    class_counts = {}
    for idx in indices:
        class_name = CIFAR10_CLASS_NAMES[y_test[idx]]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:>10}: {count:2d} samples")


def main():
    """Main interactive function."""
    
    print("="*60)
    print("Interactive Base_CNN Model Tester")
    print("="*60)
    
    # Load data and model
    x_test, y_test = load_test_data()
    model = load_trained_model()
    
    if model is None:
        return
    
    print(f"\nTest set contains {len(x_test)} samples (indices 0-{len(x_test)-1})")
    
    while True:
        print(f"\n" + "="*50)
        print("Choose an option:")
        print("1. Test single sample by index")
        print("2. Test random samples")
        print("3. Test specific indices")
        print("4. Test all samples of a specific class")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            try:
                index = int(input(f"Enter sample index (0-{len(x_test)-1}): "))
                test_single_prediction(model, x_test, y_test, index)
            except ValueError:
                print("Invalid index. Please enter a number.")
        
        elif choice == "2":
            try:
                num_samples = int(input("How many random samples? (default 20): ") or "20")
                if num_samples <= 0 or num_samples > len(x_test):
                    print(f"Invalid number. Please enter 1-{len(x_test)}")
                    continue
                
                import random
                random.seed(42)  # For reproducibility
                indices = random.sample(range(len(x_test)), num_samples)
                indices.sort()
                
                show_class_distribution(y_test, indices)
                test_multiple_predictions(model, x_test, y_test, indices)
                
            except ValueError:
                print("Invalid number. Please enter a valid integer.")
        
        elif choice == "3":
            try:
                indices_input = input("Enter indices separated by commas (e.g., 0,5,10,15): ")
                indices = [int(x.strip()) for x in indices_input.split(",")]
                
                # Validate indices
                valid_indices = [i for i in indices if 0 <= i < len(x_test)]
                if len(valid_indices) != len(indices):
                    print(f"Some indices are out of range. Using valid ones: {valid_indices}")
                
                if valid_indices:
                    show_class_distribution(y_test, valid_indices)
                    test_multiple_predictions(model, x_test, y_test, valid_indices)
                else:
                    print("No valid indices provided.")
                    
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
        
        elif choice == "4":
            print("\nAvailable classes:")
            for i, class_name in enumerate(CIFAR10_CLASS_NAMES):
                print(f"  {i}: {class_name}")
            
            try:
                class_idx = int(input("Enter class index (0-9): "))
                if class_idx < 0 or class_idx > 9:
                    print("Invalid class index.")
                    continue
                
                # Find all samples of this class
                class_indices = [i for i, label in enumerate(y_test) if label == class_idx]
                print(f"\nFound {len(class_indices)} samples of class '{CIFAR10_CLASS_NAMES[class_idx]}'")
                
                if len(class_indices) > 50:
                    print("Too many samples. Testing first 50...")
                    class_indices = class_indices[:50]
                
                test_multiple_predictions(model, x_test, y_test, class_indices)
                
            except ValueError:
                print("Invalid class index.")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
