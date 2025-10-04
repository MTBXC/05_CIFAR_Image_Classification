"""
FastAPI backend for CIFAR-10 Base_CNN model classification.
Provides endpoints for image classification and random image selection.
"""

import sys
from pathlib import Path
import random
import base64
import io
import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.base_cnn import Base_CNN
from data.load_cifar10 import load_cifar10_from_raw
from .model_manager import ModelSelector, ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Global variables
app = FastAPI(title="CIFAR-10 CIFAR Classifier", version="2.0.0")
model_manager = None
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

class ModelSwitchRequest(BaseModel):
    model_type: str  # "s3" or "local"
    model_path: str = None  # Optional for local models


def load_model_and_data():
    """Load the trained model and test data."""
    global model_manager, x_test, y_test
    
    logger.info("Loading model and data...")
    
    # Load model using ModelSelector
    try:
        model_manager = ModelSelector.get_best_available_model()
        model_info = model_manager.get_model_info()
        
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Source: {model_info.get('source', 'unknown')}")
        logger.info(f"  Parameters: {model_info.get('parameters', 0):,}")
        
        if model_info.get('source') == 's3':
            logger.info(f"  S3 URI: {model_info.get('s3_uri', 'unknown')}")
            logger.info(f"  File size: {model_info.get('file_size_mb', 0):.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Try to load test data (optional for demo images)
    x_test, y_test = None, None
    try:
        raw_data_dir = Path("data/raw")
        train_data, test_data = load_cifar10_from_raw(raw_data_dir)
        
        x_test, y_test = test_data
        x_test = x_test.astype('float32') / 255.0  # Normalize
        
        logger.info(f"Test data loaded: {x_test.shape[0]} samples")
    except Exception as e:
        logger.warning(f"Could not load test data (demo images disabled): {e}")
        logger.info("API will work for prediction, but demo images won't be available")
    
    return model_manager, x_test, y_test


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
        global current_indices, x_test
        if x_test is not None:
            current_indices = get_random_indices(20)
            print("Application startup completed successfully with demo images!")
        else:
            current_indices = []
            print("Application startup completed successfully (demo images disabled)!")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("web/public/index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint for load balancer."""
    global model_manager
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None,
        "demo_images_available": x_test is not None
    }


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
    global model_manager, x_test, y_test, current_indices
    
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if x_test is None or y_test is None:
        raise HTTPException(status_code=500, detail="Test data not loaded")
    
    image_index = request.image_index
    
    # Validate index
    if image_index < 0 or image_index >= len(x_test):
        raise HTTPException(status_code=400, detail="Invalid image index")
    
    # Get prediction
    image = x_test[image_index:image_index+1]  # Add batch dimension
    predictions = model_manager.predict(image)
    
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
    model_info = model_manager.get_model_info() if model_manager else {}
    
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None,
        "data_loaded": x_test is not None and y_test is not None,
        "model_parameters": model_info.get('parameters', 0),
        "model_source": model_info.get('source', 'none')
    }


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_info = model_manager.get_model_info()
    
    return {
        "model_name": model_info.get('model_name', 'CIFAR-10 CNN'),
        "source": model_info.get('source', 'unknown'),
        "parameters": model_info.get('parameters', 0),
        "input_shape": model_info.get('input_shape'),
        "output_shape": model_info.get('output_shape'),
        "classes": CIFAR10_CLASS_NAMES,
        "test_samples": len(x_test) if x_test is not None else 0,
        "s3_info": {
            "bucket": model_info.get('bucket'),
            "key": model_info.get('key'),
            "s3_uri": model_info.get('s3_uri'),
            "file_size_mb": model_info.get('file_size_mb')
        } if model_info.get('source') == 's3' else None
    }


@app.post("/api/load-production-model")
async def load_production_model():
    """Reload the production model from MLFlow S3."""
    global model_manager
    
    try:
        if model_manager:
            model_manager.cleanup()
        
        # Force reload production model from MLFlow
        logger.info("Reloading production model from MLFlow...")
        model_manager = ModelSelector.get_mlflow_production_model()
        
        model_info = model_manager.get_model_info()
        
        return {
            "status": "success",
            "message": "Production model reloaded successfully",
            "model_info": {
                "source": model_info.get('source'),
                "parameters": model_info.get('parameters'),
                "s3_uri": model_info.get('s3_uri')
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to reload production model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload production model: {str(e)}")


@app.post("/api/switch-model")
async def switch_model(request: ModelSwitchRequest):
    """Switch between S3 and local models."""
    global model_manager
    
    try:
        # Cleanup current model
        if model_manager:
            model_manager.cleanup()
        
        # Load new model
        if request.model_type == "s3":
            logger.info("Switching to S3 production model...")
            model_manager = ModelSelector.force_s3_model()
        elif request.model_type == "local":
            logger.info("Switching to local model...")
            model_path = request.model_path or "models/base_cnn_cifar10_cpu.h5"
            model_manager = ModelSelector.force_local_model(model_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 's3' or 'local'")
        
        model_info = model_manager.get_model_info()
        
        return {
            "status": "success",
            "message": f"Successfully switched to {request.model_type} model",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


@app.post("/api/deploy-model-from-s3")
async def deploy_model_from_s3(request: dict):
    """
    Professional model deployment endpoint
    Deploy model from specific S3 location with full tracking
    """
    global model_manager
    
    try:
        # Extract deployment parameters
        bucket = request.get('bucket')
        key = request.get('key') 
        model_name = request.get('model_name', 'Deployed Model')
        run_id = request.get('run_id')
        deployment_id = request.get('deployment_id', f"deploy_{int(time.time())}")
        
        if not bucket or not key:
            raise HTTPException(status_code=400, detail="bucket and key are required")
        
        logger.info(f"🚀 Professional deployment starting...")
        logger.info(f"   Deployment ID: {deployment_id}")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Source: s3://{bucket}/{key}")
        logger.info(f"   Run ID: {run_id}")
        
        # Pre-deployment validation
        deployment_start = datetime.now()
        
        # Cleanup current model
        if model_manager:
            previous_model_info = model_manager.get_model_info()
            logger.info(f"   Previous model: {previous_model_info.get('model_name', 'Unknown')}")
            model_manager.cleanup()
        else:
            previous_model_info = {}
        
        # Load new model from specific S3 location
        model_manager = ModelManager()
        
        # Use the load_model_by_path method for specific S3 deployment with custom name
        model_manager.load_model_by_path(bucket, key, model_name)
        
        # Get model info and enhance with deployment metadata
        model_info = model_manager.get_model_info()
        
        # Add deployment tracking information
        deployment_metadata = {
            'deployment_id': deployment_id,
            'deployed_at': deployment_start.isoformat(),
            'deployment_source': 'professional_deployment_system',
            'run_id': run_id,
            'model_name': model_name,
            's3_source': f's3://{bucket}/{key}',
            'previous_model': previous_model_info.get('model_name', 'None'),
            'deployment_method': 'api_deploy_from_s3'
        }
        
        # Update model info with deployment metadata
        model_info.update(deployment_metadata)
        
        deployment_end = datetime.now()
        deployment_time = (deployment_end - deployment_start).total_seconds()
        
        logger.info(f"✅ Deployment successful!")
        logger.info(f"   Time: {deployment_time:.2f}s")
        logger.info(f"   Model: {model_info.get('model_name', 'Unknown')}")
        logger.info(f"   Parameters: {model_info.get('parameters', 0):,}")
        
        return {
            "status": "success",
            "message": f"Successfully deployed model: {model_name}",
            "deployment_info": {
                "deployment_id": deployment_id,
                "deployment_time_seconds": deployment_time,
                "deployed_at": deployment_start.isoformat()
            },
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"❌ Professional deployment failed: {e}")
        
        # Try to restore previous model if deployment fails
        if 'previous_model_info' in locals() and previous_model_info:
            try:
                logger.warning("🔄 Attempting to restore previous model...")
                # This would need implementation based on previous model source
            except Exception as restore_error:
                logger.error(f"❌ Failed to restore previous model: {restore_error}")
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": f"Professional deployment failed: {str(e)}",
                "deployment_id": request.get('deployment_id', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
        )


# Mount static files
app.mount("/static", StaticFiles(directory="web/public"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
