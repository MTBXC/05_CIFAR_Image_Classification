"""
FastAPI backend for CIFAR-10 Base_CNN model classification.
Provides endpoints for image classification and random image selection.
"""

import sys
from pathlib import Path
import random
import base64
import io
from typing import List, Dict, Any
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.base_cnn import Base_CNN
from data.load_cifar10 import load_cifar10_from_raw

# CIFAR-10 class names
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Global variables
app = FastAPI(title="CIFAR-10 Base_CNN Classifier", version="1.0.0")
model = None
x_test = None
y_test = None
current_indices = []

# Pydantic models
class PredictionRequest(BaseModel):
    image_index: int

class PredictionItem(BaseModel):
    class_name: str
    probability: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: List[PredictionItem]
    true_class: str
    is_correct: bool

class RandomImagesResponse(BaseModel):
    images: List[Dict[str, Any]]
    indices: List[int]

class ImageData(BaseModel):
    index: int
    image_base64: str
    true_class: str
    true_class_name: str


def load_model_and_data():
    """Load the trained model and test data."""
    global model, x_test, y_test
    
    print("Loading model and data...")
    
    # Load model
    model_path = Path("models/base_cnn_cifar10_cpu.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with tf.keras.utils.custom_object_scope({'Base_CNN': Base_CNN}):
        model = tf.keras.models.load_model(model_path)
    
    print(f"Model loaded successfully with {model.count_params():,} parameters")
    
    # Load test data
    raw_data_dir = Path("data/raw")
    train_data, test_data = load_cifar10_from_raw(raw_data_dir)
    
    x_test, y_test = test_data
    x_test = x_test.astype('float32') / 255.0  # Normalize
    
    print(f"Test data loaded: {x_test.shape[0]} samples")
    
    return model, x_test, y_test


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64 string."""
    from PIL import Image
    
    # Convert from float [0,1] to uint8 [0,255]
    image_uint8 = (image_array * 255).astype(np.uint8)
    
    # Create PIL Image
    pil_image = Image.fromarray(image_uint8)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


def get_random_indices(num_images: int = 20) -> List[int]:
    """Get random indices from test set."""
    global x_test
    
    if x_test is None:
        raise HTTPException(status_code=500, detail="Test data not loaded")
    
    # Generate truly random indices (no fixed seed)
    indices = random.sample(range(len(x_test)), num_images)
    indices.sort()  # Sort for better display
    
    return indices


@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup."""
    try:
        load_model_and_data()
        global current_indices
        current_indices = get_random_indices(20)
        print("Application startup completed successfully!")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("web/public/index.html")


@app.get("/api/random-images", response_model=RandomImagesResponse)
async def get_random_images():
    """Get 20 random images from test set."""
    global current_indices, x_test, y_test
    
    if x_test is None or y_test is None:
        raise HTTPException(status_code=500, detail="Test data not loaded")
    
    # Generate new random indices
    current_indices = get_random_indices(20)
    
    images = []
    for idx in current_indices:
        image_data = {
            "index": idx,
            "image_base64": image_to_base64(x_test[idx]),
            "true_class": int(y_test[idx]),
            "true_class_name": CIFAR10_CLASS_NAMES[y_test[idx]]
        }
        images.append(image_data)
    
    return RandomImagesResponse(images=images, indices=current_indices)


@app.get("/api/images", response_model=RandomImagesResponse)
async def get_images():
    """Get 20 random images from test set (alias for /api/random-images)."""
    return await get_random_images()


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_image(request: PredictionRequest):
    """Predict the class of an image by its index."""
    global model, x_test, y_test, current_indices
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if x_test is None or y_test is None:
        raise HTTPException(status_code=500, detail="Test data not loaded")
    
    image_index = request.image_index
    
    # Validate index
    if image_index < 0 or image_index >= len(x_test):
        raise HTTPException(status_code=400, detail="Invalid image index")
    
    # Get prediction
    image = x_test[image_index:image_index+1]  # Add batch dimension
    predictions = model.predict(image, verbose=0)
    
    # Get results
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class = CIFAR10_CLASS_NAMES[predicted_class_idx]
    
    # Get true class
    true_class_idx = int(y_test[image_index])
    true_class = CIFAR10_CLASS_NAMES[true_class_idx]
    is_correct = predicted_class_idx == true_class_idx
    
    # Get all predictions
    all_predictions = []
    for i, prob in enumerate(predictions[0]):
        all_predictions.append(PredictionItem(
            class_name=CIFAR10_CLASS_NAMES[i],
            probability=float(prob)
        ))
    
    # Sort by probability (descending)
    all_predictions.sort(key=lambda x: x.probability, reverse=True)
    
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        all_predictions=all_predictions,
        true_class=true_class,
        is_correct=is_correct
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": x_test is not None and y_test is not None,
        "model_parameters": model.count_params() if model else 0
    }


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": "Base_CNN",
        "parameters": model.count_params(),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "classes": CIFAR10_CLASS_NAMES,
        "test_samples": len(x_test) if x_test is not None else 0
    }


# Mount static files
app.mount("/static", StaticFiles(directory="web/public"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
