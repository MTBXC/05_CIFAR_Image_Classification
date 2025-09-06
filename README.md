# CIFAR-10 Image Classification with Base_CNN

A complete machine learning project for CIFAR-10 image classification using a custom CNN architecture, featuring training, monitoring with MLFlow, and a web application for interactive predictions.

## 🚀 Features

- **Custom CNN Architecture**: Lightweight Base_CNN model optimized for CPU training
- **MLFlow Integration**: Complete experiment tracking, metrics logging, and model versioning
- **Web Application**: Interactive FastAPI backend with modern HTML/JavaScript frontend
- **Real-time Predictions**: Live image classification with confidence scores
- **Comprehensive Monitoring**: Training curves, confusion matrices, and detailed metrics

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # CNN model architecture
│   ├── training/          # Training pipeline
│   ├── monitoring/        # MLFlow logging utilities
│   └── utils/             # Utility functions
├── api/                   # FastAPI backend
│   └── app/              # Web application
├── web/                   # Frontend files
│   └── public/           # HTML, CSS, JavaScript
├── models/                # Trained model files (gitignored)
├── data/                  # Dataset files (gitignored)
├── mlflow/                # MLFlow experiments (gitignored)
└── scripts/               # Utility scripts
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CIFAR-Image-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download CIFAR-10 dataset**:
   ```bash
   # The dataset will be automatically downloaded when running training
   # Place cifar-10-python.tar.gz in data/raw/ directory
   ```

## 🎯 Quick Start

### 1. Train the Model
```bash
python train_base_cnn.py
```

### 2. Test the Model
```bash
python test_model_simple.py
```

### 3. Launch Web Application
```bash
python run_web_app.py
```
Then open: `http://localhost:8000`

### 4. View MLFlow Experiments
```bash
py -m mlflow ui
```
Then open: `http://localhost:5000`

## 🧠 Model Architecture

**Base_CNN** - Lightweight CNN optimized for CPU training:
- **Input**: 32x32x3 RGB images
- **Architecture**: 2 Conv blocks + 1 Dense layer
- **Parameters**: ~50,000 (lightweight)
- **Training Time**: ~2-5 minutes per epoch on CPU
- **Accuracy**: ~60-70% on CIFAR-10 test set

### Architecture Details:
```
Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Flatten → Dense(64) → Dropout → Dense(10)
```

## 📊 MLFlow Integration

The project includes comprehensive MLFlow tracking:

- **Metrics**: Training/validation accuracy, loss, F1-scores
- **Parameters**: Model hyperparameters and training config
- **Artifacts**: Model files, confusion matrices, training curves
- **Model Registry**: Versioned model storage

## 🌐 Web Application

### Backend (FastAPI)
- **Endpoints**: Image loading, prediction, health check
- **Features**: Real-time classification, confidence scoring
- **API Documentation**: Available at `/docs`

### Frontend (HTML/JavaScript)
- **Interactive Interface**: 20 random images from test set
- **Real-time Predictions**: Click to classify with instant results
- **Modern UI**: Responsive design with animations
- **Update Feature**: Refresh with new random images

## 📈 Usage Examples

### Training with Custom Parameters
```python
from src.training.trainer import TrainingConfig, train_model

config = TrainingConfig(
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    dropout_rate=0.3
)

model = train_model(config)
```

### Making Predictions
```python
from src.models.base_cnn import create_base_cnn_model
import tensorflow as tf

model = tf.keras.models.load_model('models/base_cnn_cifar10_cpu.h5')
prediction = model.predict(image_array)
```

## 🔧 Configuration

### Training Configuration
- **Epochs**: 10 (default)
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Dropout Rate**: 0.2
- **Data Augmentation**: Disabled (for speed)

### Model Configuration
- **Input Shape**: (32, 32, 3)
- **Classes**: 10 (CIFAR-10)
- **Activation**: ReLU + Softmax
- **Optimizer**: Adam

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- MLFlow
- NumPy
- PIL/Pillow
- Uvicorn

See `requirements.txt` for complete dependency list.

## 🚀 Deployment

### Local Development
```bash
# Start web app
python run_web_app.py

# Start MLFlow UI
py -m mlflow ui
```

### Production Deployment
- Use Docker for containerization
- Deploy FastAPI with Gunicorn/Uvicorn
- Use external MLFlow tracking server
- Configure environment variables

## 📝 API Documentation

### Endpoints

- `GET /` - Main web interface
- `GET /api/health` - Health check
- `GET /api/images` - Get 20 random images
- `POST /api/predict` - Classify selected image
- `GET /api/model-info` - Model information

### Example API Usage
```bash
# Get random images
curl http://localhost:8000/api/images

# Classify image
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image_index": 123}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- TensorFlow/Keras for deep learning framework
- MLFlow for experiment tracking
- FastAPI for web framework

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Classifying! 🎯**
