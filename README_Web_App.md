# Web Application Guide

This guide explains how to use the CIFAR-10 Base_CNN web application for interactive image classification.

## Overview

The web application provides an interactive interface for testing the trained Base_CNN model on CIFAR-10 images. Users can select images, make predictions, and view detailed results in a modern web interface.

## Quick Start

### 1. Start the Web Application
```bash
python run_web_app.py
```

### 2. Open in Browser
Navigate to: `http://localhost:8000`

### 3. Use the Interface
1. **View Images**: 20 random CIFAR-10 images are displayed
2. **Select Image**: Click on any image to select it
3. **Classify**: Click "🧠 Classify" button
4. **View Results**: See prediction, confidence, and all class probabilities
5. **Update Images**: Click "🔄 Update Images" for new random samples

## Features

### Image Display
- **Grid Layout**: 20 images in responsive grid
- **Image Information**: Shows image index and true class
- **Selection Feedback**: Visual highlighting of selected image
- **Hover Effects**: Interactive image cards with animations

### Classification Results
- **Prediction**: Predicted class with confidence score
- **Correctness**: Visual indicator (✅/❌) for prediction accuracy
- **All Predictions**: Complete list of all 10 classes with probabilities
- **Confidence Levels**: Color-coded confidence (high/medium/low)

## API Endpoints

### Health Check
```http
GET /api/health
```

### Get Random Images
```http
GET /api/images
```

### Make Prediction
```http
POST /api/predict
Content-Type: application/json
{
  "image_index": 123
}
```

### Model Information
```http
GET /api/model-info
```

## Technical Details

### Backend (FastAPI)
- **Framework**: FastAPI with Uvicorn server
- **Endpoints**: RESTful API for image loading and prediction
- **Model Integration**: TensorFlow/Keras model serving
- **Data Handling**: CIFAR-10 test set management

### Frontend (HTML/JavaScript)
- **Interface**: Modern responsive web design
- **Interactivity**: Real-time image selection and classification
- **Visualization**: Dynamic results display with animations
- **User Experience**: Intuitive controls and feedback

## Deployment

### Local Development
```bash
python run_web_app.py
```

### Production Deployment
```bash
gunicorn api.app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Troubleshooting

### Common Issues
1. **Model Not Loading**: Train the model first using `train_base_cnn.py`
2. **Data Not Loading**: Ensure CIFAR-10 data is in `data/raw/` directory
3. **Port Already in Use**: Change port in `run_web_app.py`
4. **CORS Issues**: Add CORS middleware to FastAPI app

---

**Happy Classifying! 🌐**
