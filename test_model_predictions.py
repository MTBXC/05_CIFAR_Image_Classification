"""
Script for testing Base_CNN model on 20 random examples from CIFAR-10 test set.
Shows images, true labels, predictions and model confidence.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.load_cifar10 import load_cifar10_from_raw
from models.base_cnn import create_base_cnn_model

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


def get_predictions(model, x_test, indices):
    """Get predictions for specific test samples."""
    # Get the selected samples
    x_selected = x_test[indices]
    
    # Get predictions
    predictions = model.predict(x_selected, verbose=0)
    
    # Get predicted classes and confidence
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    return predicted_classes, confidence_scores, predictions


def visualize_predictions(x_test, y_test, indices, predicted_classes, confidence_scores, predictions):
    """Visualize predictions with images and results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Base_CNN Predictions on CIFAR-10 Test Set', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    correct_predictions = 0
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Get image and labels
        image = x_test[idx]
        true_label = y_test[idx]
        predicted_label = predicted_classes[i]
        confidence = confidence_scores[i]
        
        # Check if prediction is correct
        is_correct = true_label == predicted_label
        if is_correct:
            correct_predictions += 1
        
        # Display image
        ax.imshow(image)
        ax.axis('off')
        
        # Set title with results
        title_color = 'green' if is_correct else 'red'
        title = f"True: {CIFAR10_CLASS_NAMES[true_label]}\n"
        title += f"Pred: {CIFAR10_CLASS_NAMES[predicted_label]}\n"
        title += f"Conf: {confidence:.3f}"
        
        ax.set_title(title, color=title_color, fontsize=10, fontweight='bold')
        
        # Add border color based on correctness
        for spine in ax.spines.values():
            spine.set_edgecolor(title_color)
            spine.set_linewidth(3)
    
    # Calculate accuracy
    accuracy = correct_predictions / len(indices)
    
    # Add overall results
    fig.text(0.5, 0.02, f'Accuracy on 20 samples: {accuracy:.1%} ({correct_predictions}/20)', 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    
    # Save the plot
    output_path = "model_predictions_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    return accuracy, correct_predictions


def print_detailed_results(indices, y_test, predicted_classes, confidence_scores, predictions):
    """Print detailed results for each prediction."""
    
    print("\n" + "="*80)
    print("DETAILED PREDICTION RESULTS")
    print("="*80)
    
    correct_predictions = 0
    
    for i, idx in enumerate(indices):
        true_label = y_test[idx]
        predicted_label = predicted_classes[i]
        confidence = confidence_scores[i]
        is_correct = true_label == predicted_label
        
        if is_correct:
            correct_predictions += 1
        
        print(f"\nSample {i+1:2d} (Index {idx:5d}):")
        print(f"  True Label:     {CIFAR10_CLASS_NAMES[true_label]:>10} (class {true_label})")
        print(f"  Predicted:      {CIFAR10_CLASS_NAMES[predicted_label]:>10} (class {predicted_label})")
        print(f"  Confidence:     {confidence:.4f}")
        print(f"  Correct:        {'✓ YES' if is_correct else '✗ NO'}")
        
        # Show top 3 predictions
        top3_indices = np.argsort(predictions[i])[-3:][::-1]
        print(f"  Top 3 predictions:")
        for j, class_idx in enumerate(top3_indices):
            prob = predictions[i][class_idx]
            print(f"    {j+1}. {CIFAR10_CLASS_NAMES[class_idx]:>10}: {prob:.4f}")
    
    accuracy = correct_predictions / len(indices)
    print(f"\n" + "="*80)
    print(f"SUMMARY: {correct_predictions}/{len(indices)} correct predictions ({accuracy:.1%})")
    print("="*80)
    
    return accuracy


def analyze_prediction_patterns(y_test, predicted_classes, confidence_scores):
    """Analyze patterns in predictions."""
    
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Per-class accuracy
    print("\nPer-class accuracy on selected samples:")
    for class_idx, class_name in enumerate(CIFAR10_CLASS_NAMES):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_predictions = predicted_classes[class_mask]
            class_accuracy = np.mean(class_predictions == class_idx)
            print(f"  {class_name:>10}: {class_accuracy:.1%}")
    
    # Confidence analysis
    correct_mask = y_test == predicted_classes
    correct_confidence = confidence_scores[correct_mask]
    incorrect_confidence = confidence_scores[~correct_mask]
    
    print(f"\nConfidence Analysis:")
    print(f"  Average confidence (correct):   {np.mean(correct_confidence):.4f}")
    print(f"  Average confidence (incorrect): {np.mean(incorrect_confidence):.4f}")
    print(f"  Min confidence:                 {np.min(confidence_scores):.4f}")
    print(f"  Max confidence:                 {np.max(confidence_scores):.4f}")
    
    # Most confident incorrect predictions
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        incorrect_confidences = confidence_scores[incorrect_indices]
        most_confident_wrong = np.argmax(incorrect_confidences)
        wrong_idx = incorrect_indices[most_confident_wrong]
        
        print(f"\nMost confident incorrect prediction:")
        print(f"  True: {CIFAR10_CLASS_NAMES[y_test[wrong_idx]]}")
        print(f"  Pred: {CIFAR10_CLASS_NAMES[predicted_classes[wrong_idx]]}")
        print(f"  Conf: {confidence_scores[wrong_idx]:.4f}")


def main():
    """Main function to test model predictions."""
    
    print("="*60)
    print("Base_CNN Model Prediction Test")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load test data
    x_test, y_test = load_test_data()
    
    # Load trained model
    model = load_trained_model()
    if model is None:
        return
    
    # Select 20 random samples
    test_size = len(x_test)
    indices = random.sample(range(test_size), 20)
    indices = sorted(indices)  # Sort for better visualization
    
    print(f"\nSelected 20 random test samples (indices: {indices})")
    
    # Get predictions
    print("\nGetting predictions...")
    predicted_classes, confidence_scores, predictions = get_predictions(model, x_test, indices)
    
    # Print detailed results
    accuracy = print_detailed_results(indices, y_test, predicted_classes, confidence_scores, predictions)
    
    # Analyze patterns
    analyze_prediction_patterns(y_test[indices], predicted_classes, confidence_scores)
    
    # Create visualization
    print("\nCreating visualization...")
    try:
        visualize_predictions(x_test, y_test, indices, predicted_classes, confidence_scores, predictions)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Continuing without visualization...")
    
    print(f"\n" + "="*60)
    print("PREDICTION TEST COMPLETED!")
    print(f"Model accuracy on 20 random samples: {accuracy:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
