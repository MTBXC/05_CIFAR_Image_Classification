# Base_CNN for CIFAR-10 - Guide

This project contains a simple CNN architecture in TensorFlow for CIFAR-10 image classification, with full MLFlow integration for experiment monitoring.

## Project Structure

```
src/
├── models/
│   ├── __init__.py
│   └── base_cnn.py          # Base_CNN architecture
├── training/
│   ├── __init__.py
│   └── trainer.py           # Training module
├── monitoring/
│   ├── __init__.py
│   └── mlflow_monitor.py    # MLFlow integration
└── data/
    ├── load_cifar10.py      # CIFAR-10 data loading
    └── augmentations.py     # Data augmentations
```

## Base_CNN Architecture

The Base_CNN is a lightweight CNN optimized for CPU training:

### Architecture Details:
- **Input**: 32x32x3 RGB images
- **Conv Block 1**: Conv2D(16) → MaxPool2D(2,2)
- **Conv Block 2**: Conv2D(32) → MaxPool2D(2,2)
- **Dense Layer**: Dense(64) → Dropout(0.2) → Dense(10)
- **Total Parameters**: ~50,000 (lightweight)
- **Training Time**: ~2-5 minutes per epoch on CPU

### Key Features:
- **CPU Optimized**: Designed for fast training on CPU
- **Lightweight**: Minimal parameters for quick experimentation
- **Simple Architecture**: Easy to understand and modify
- **Good Performance**: ~60-70% accuracy on CIFAR-10

## Training Configuration

### Default Settings (CPU Optimized):
```python
TrainingConfig(
    epochs=10,                    # 10 epochs as requested
    batch_size=64,                # Larger batch for faster training
    learning_rate=0.001,          # Standard learning rate
    dropout_rate=0.2,             # Reduced dropout
    use_augmentation=False,       # Disabled for speed
    early_stopping_patience=5,    # Early stopping
    reduce_lr_patience=3          # Learning rate reduction
)
```

### Performance Characteristics:
- **Training Time**: ~2-5 minutes per epoch
- **Memory Usage**: Low (suitable for CPU)
- **Convergence**: Usually converges within 10 epochs
- **Accuracy**: 60-70% on CIFAR-10 test set

## Usage

### 1. Training the Model
```bash
python train_base_cnn.py
```

### 2. Testing the Model
```bash
# Simple text-based testing
python test_model_simple.py

# Interactive testing
python interactive_model_test.py

# Testing with visualizations
python test_model_predictions.py
```

### 3. MLFlow Monitoring
```bash
# Start MLFlow UI
py -m mlflow ui

# View experiments at: http://localhost:5000
```

## MLFlow Integration

The project includes comprehensive MLFlow tracking:

### Logged Metrics:
- Training/validation accuracy and loss
- Per-epoch metrics
- Test accuracy and loss
- Per-class precision, recall, F1-score
- Macro and weighted averages

### Logged Artifacts:
- Trained model files
- Confusion matrices
- Training curves
- Classification reports
- Model summaries

### Model Registry:
- Automatic model versioning
- Model metadata tracking
- Experiment comparison

## Model Performance

### Typical Results:
- **Test Accuracy**: 60-70%
- **Training Time**: 20-50 minutes (10 epochs)
- **Memory Usage**: <2GB RAM
- **Model Size**: ~1.7MB

### Per-Class Performance:
- **Best Classes**: airplane, automobile, ship, truck
- **Challenging Classes**: cat, dog, bird (similar features)
- **Overall**: Good performance for lightweight model

## Customization

### Modifying Architecture:
```python
# In src/models/base_cnn.py
class Base_CNN(tf.keras.Model):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        # Modify layers here
        self.conv1 = tf.keras.layers.Conv2D(32, ...)  # More filters
        self.dense1 = tf.keras.layers.Dense(128, ...)  # More neurons
```

### Training Parameters:
```python
# In train_base_cnn.py
config = TrainingConfig(
    epochs=20,                    # More epochs
    batch_size=32,                # Smaller batch
    learning_rate=0.0005,         # Lower learning rate
    use_augmentation=True,        # Enable augmentation
)
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**:
   - Reduce batch_size to 16 or 32
   - Disable data augmentation
   - Use smaller model architecture

2. **Slow Training**:
   - Increase batch_size to 64 or 128
   - Disable data augmentation
   - Use CPU-optimized settings

3. **Poor Accuracy**:
   - Increase epochs to 20-30
   - Enable data augmentation
   - Adjust learning rate
   - Add more layers/filters

### Performance Tips:
- Use `use_augmentation=False` for faster training
- Set `batch_size=64` for CPU optimization
- Monitor training with MLFlow UI
- Use early stopping to prevent overfitting

## File Descriptions

### Core Files:
- `src/models/base_cnn.py`: CNN architecture definition
- `src/training/trainer.py`: Training pipeline
- `src/monitoring/mlflow_monitor.py`: MLFlow integration
- `train_base_cnn.py`: Main training script

### Testing Files:
- `test_model_simple.py`: Simple text-based testing
- `interactive_model_test.py`: Interactive testing
- `test_model_predictions.py`: Testing with visualizations
- `test_pipeline.py`: Complete pipeline testing

### Configuration:
- `requirements.txt`: Python dependencies
- `README_Base_CNN.md`: This documentation

## Next Steps

1. **Experiment with Architecture**: Try different layer configurations
2. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
3. **Data Augmentation**: Enable and tune augmentation parameters
4. **Model Comparison**: Compare with other architectures
5. **Deployment**: Deploy model for production use

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- MLFlow
- NumPy
- Matplotlib (for visualizations)
- Scikit-learn (for metrics)

See `requirements.txt` for complete dependency list.

---

**Happy Training! 🚀**