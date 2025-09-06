# Model Testing Guide

This guide explains how to test the trained Base_CNN model using various testing scripts.

## Available Testing Scripts

### 1. Simple Text Testing (`test_model_simple.py`)
**Purpose**: Quick text-based testing without visualizations
**Best for**: Fast verification of model functionality

```bash
python test_model_simple.py
```

**Features**:
- Tests 20 random samples
- Text-only output
- Shows predictions and confidence scores
- Calculates accuracy on random samples

**Output Example**:
```
Sample  1: ✅ CORRECT
  True: airplane      | Predicted: airplane      | Confidence: 0.856
  Top 3: airplane(0.856), ship(0.089), truck(0.032)

Sample  2: ❌ WRONG
  True: cat           | Predicted: dog           | Confidence: 0.623
  Top 3: dog(0.623), cat(0.234), deer(0.089)
```

### 2. Interactive Testing (`interactive_model_test.py`)
**Purpose**: Interactive testing with user input
**Best for**: Detailed analysis of specific samples

```bash
python interactive_model_test.py
```

**Features**:
- Choose specific image indices
- Test random samples
- Detailed prediction analysis
- Interactive menu system

**Usage Options**:
1. **Test specific index**: Enter image index (0-9999)
2. **Test random samples**: Enter number of random samples
3. **Exit**: Quit the program

### 3. Visual Testing (`test_model_predictions.py`)
**Purpose**: Testing with matplotlib visualizations
**Best for**: Visual analysis and debugging

```bash
python test_model_predictions.py
```

**Features**:
- Matplotlib visualizations
- Image display with predictions
- Confusion matrix visualization
- Training curve plots
- Detailed performance analysis

### 4. Pipeline Testing (`test_pipeline.py`)
**Purpose**: Complete pipeline testing
**Best for**: End-to-end verification

```bash
python test_pipeline.py
```

**Features**:
- Full training pipeline test
- Model creation and training
- Evaluation and metrics
- MLFlow integration test

## Testing Workflow

### Step 1: Verify Model Exists
```bash
# Check if model file exists
ls models/base_cnn_cifar10_cpu.h5
```

### Step 2: Quick Test
```bash
# Run simple test
python test_model_simple.py
```

### Step 3: Interactive Analysis
```bash
# Run interactive test
python interactive_model_test.py
```

### Step 4: Visual Analysis (Optional)
```bash
# Run visual test (requires matplotlib)
python test_model_predictions.py
```

## Understanding Test Results

### Accuracy Metrics
- **Sample Accuracy**: Accuracy on tested samples
- **Confidence**: Model's confidence in predictions
- **Top-3 Accuracy**: Whether correct class is in top 3 predictions

### Performance Indicators
- **High Confidence + Correct**: Model is confident and right
- **High Confidence + Wrong**: Model is overconfident
- **Low Confidence + Correct**: Model is uncertain but right
- **Low Confidence + Wrong**: Model is uncertain and wrong

### Common Patterns
- **Airplane/Ship/Truck**: Usually well-classified
- **Cat/Dog/Bird**: Often confused (similar features)
- **Deer/Horse**: Sometimes confused
- **Frog**: Usually well-classified

## Troubleshooting

### Model Loading Issues
```python
# Error: Unknown layer 'Base_CNN'
# Solution: Use custom object scope
with tf.keras.utils.custom_object_scope({'Base_CNN': Base_CNN}):
    model = tf.keras.models.load_model(model_path)
```

### Data Loading Issues
```python
# Error: Data not found
# Solution: Ensure CIFAR-10 data is in data/raw/
# Download from: https://www.cs.toronto.edu/~kriz/cifar.html
```

### Performance Issues
- **Slow predictions**: Normal for CPU inference
- **Memory errors**: Reduce batch size
- **Import errors**: Check Python path and dependencies

## Custom Testing

### Testing Specific Classes
```python
# Test only specific classes
def test_specific_classes(model, x_test, y_test, target_classes):
    mask = np.isin(y_test, target_classes)
    x_subset = x_test[mask]
    y_subset = y_test[mask]
    
    predictions = model.predict(x_subset)
    accuracy = calculate_accuracy(y_subset, predictions)
    return accuracy

# Test only animals
animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
accuracy = test_specific_classes(model, x_test, y_test, animal_classes)
```

### Testing with Custom Images
```python
# Load and test custom image
def test_custom_image(model, image_path):
    from PIL import Image
    import numpy as np
    
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return predicted_class, confidence
```

## Performance Benchmarks

### Expected Performance:
- **Inference Time**: ~0.1-0.5 seconds per image (CPU)
- **Memory Usage**: ~100-200MB for model
- **Accuracy**: 60-70% on CIFAR-10 test set
- **Confidence**: Usually 0.3-0.9 for correct predictions

### Optimization Tips:
- Use GPU for faster inference
- Batch multiple images together
- Use model quantization for deployment
- Cache model loading for repeated tests

## Integration with MLFlow

### Logging Test Results
```python
import mlflow

# Log test results to MLFlow
with mlflow.start_run():
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("avg_confidence", avg_confidence)
    mlflow.log_param("test_samples", num_samples)
```

### Comparing Models
```python
# Compare different model versions
experiments = mlflow.search_experiments()
for exp in experiments:
    runs = mlflow.search_runs(exp.experiment_id)
    # Analyze and compare results
```

## Best Practices

### Testing Strategy:
1. **Start Simple**: Use `test_model_simple.py` first
2. **Interactive Analysis**: Use `interactive_model_test.py` for details
3. **Visual Verification**: Use `test_model_predictions.py` for debugging
4. **Document Results**: Keep track of performance metrics

### Quality Assurance:
- Test on diverse samples
- Verify edge cases
- Monitor confidence scores
- Compare with baseline models

### Continuous Testing:
- Automate testing in CI/CD
- Set up performance benchmarks
- Monitor model drift
- Regular validation on new data

---

**Happy Testing! 🧪**