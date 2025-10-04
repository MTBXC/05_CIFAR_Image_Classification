"""
Model Manager for downloading and loading models from S3.
Handles both local models and S3-based models for production use.
"""

import os
import tempfile
import boto3
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading from local files or S3."""
    
    def __init__(self, region_name: str = "eu-north-1"):
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.model = None
        self.model_info = {}
        self.temp_model_path = None
        
    def load_local_model(self, model_path: str) -> tf.keras.Model:
        """Load model from local file."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading local model from {model_path}")
        
        try:
            # Try loading with custom objects first (for Base_CNN)
            from models.base_cnn import Base_CNN
            with tf.keras.utils.custom_object_scope({'Base_CNN': Base_CNN}):
                self.model = tf.keras.models.load_model(model_path)
        except:
            # Fallback to standard loading
            self.model = tf.keras.models.load_model(model_path)
        
        # Determine model name based on path
        model_name = "Base CNN (Local)"
        if "base_cnn" in str(model_path).lower():
            model_name = "Base CNN v1.0"
        elif "advanced" in str(model_path).lower():
            model_name = "Advanced CNN v2.0"
        elif "production" in str(model_path).lower():
            model_name = "Production CNN"
            
        self.model_info = {
            'source': 'local',
            'path': str(model_path),
            'model_name': model_name,
            'parameters': self.model.count_params(),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape
        }
        
        logger.info(f"Local model loaded successfully with {self.model.count_params():,} parameters")
        return self.model
    
    def load_s3_model(self, bucket_name: str, model_key: str, custom_model_name: str = None) -> tf.keras.Model:
        """Load model from S3."""
        logger.info(f"Loading model from S3: s3://{bucket_name}/{model_key}")
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                self.temp_model_path = tmp_file.name
            
            # Download from S3
            logger.info("Downloading model from S3...")
            self.s3_client.download_file(bucket_name, model_key, self.temp_model_path)
            
            # Check file size
            file_size = os.path.getsize(self.temp_model_path)
            logger.info(f"Downloaded model size: {file_size / (1024*1024):.2f} MB")
            
            # Load model
            logger.info("Loading model into memory...")
            self.model = tf.keras.models.load_model(self.temp_model_path)
            
            # Use custom name if provided, otherwise determine from S3 key
            if custom_model_name:
                model_name = custom_model_name
                logger.info(f"Using custom model name: {model_name}")
            else:
                # Fallback to automatic naming
                model_name = "Production CNN Model"
                if "production" in model_key.lower():
                    if "v1" in model_key.lower():
                        model_name = "Production CNN v1.0"
                    elif "v2" in model_key.lower():
                        model_name = "Production CNN v2.0"
                    else:
                        model_name = "Production CNN (Latest)"
                elif "sagemaker" in bucket_name.lower():
                    model_name = "SageMaker CNN Model"
                elif "mlflow" in bucket_name.lower():
                    model_name = "MLFlow CNN Model"
                elif "base" in model_key.lower():
                    model_name = "Base CNN (S3)"
                elif "advanced" in model_key.lower():
                    model_name = "Advanced CNN (S3)"
                
            self.model_info = {
                'source': 's3',
                'bucket': bucket_name,
                'key': model_key,
                's3_uri': f"s3://{bucket_name}/{model_key}",
                'model_name': model_name,
                'parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'file_size_mb': file_size / (1024*1024)
            }
            
            logger.info(f"S3 model loaded successfully with {self.model.count_params():,} parameters")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading S3 model: {e}")
            self.cleanup()
            raise
    
    def load_production_model(self) -> tf.keras.Model:
        """Load the current production model from MLFlow S3."""
        # Try MLFlow bucket first (from our MLFlow deployment)
        mlflow_bucket = "mlflow-minimal-zqhynp3j"  # Our MLFlow S3 bucket
        
        # Look for production model in MLFlow artifacts
        production_paths = [
            "models/production/model.h5",  # MLFlow standard path
            "artifacts/model/model.h5",    # Alternative MLFlow path
            "production-models/cifar10_production_v1.h5"  # Fallback to SageMaker
        ]
        
        # Try different buckets and paths
        buckets_to_try = [
            mlflow_bucket,
            "cifar-sagemaker-models-itwm0v09"  # Fallback to SageMaker bucket
        ]
        
        for bucket in buckets_to_try:
            for model_path in production_paths:
                try:
                    logger.info(f"Trying to load model from s3://{bucket}/{model_path}")
                    return self.load_s3_model(bucket, model_path)
                except Exception as e:
                    logger.debug(f"Failed to load from s3://{bucket}/{model_path}: {e}")
                    continue
        
        # If all S3 attempts fail, raise error
        raise RuntimeError("Could not load production model from any S3 location")
    
    def load_model_by_path(self, bucket_name: str, model_key: str, custom_model_name: str = None) -> tf.keras.Model:
        """Load a specific model from S3 by bucket and key with optional custom name."""
        logger.info(f"Loading specific model from s3://{bucket_name}/{model_key}")
        if custom_model_name:
            logger.info(f"Using custom model name: {custom_model_name}")
        return self.load_s3_model(bucket_name, model_key, custom_model_name)
    
    def predict(self, image_array) -> Dict[str, Any]:
        """Make prediction using loaded model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_local_model() or load_s3_model() first.")
        
        # Ensure proper input format
        if len(image_array.shape) == 3:
            image_array = tf.expand_dims(image_array, 0)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return self.model_info.copy()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_model_path and os.path.exists(self.temp_model_path):
            try:
                os.unlink(self.temp_model_path)
                logger.info("Temporary model file cleaned up")
            except Exception as e:
                logger.warning(f"Could not clean up temporary file: {e}")
            finally:
                self.temp_model_path = None
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()


class ModelSelector:
    """Helper class to choose between local and S3 models."""
    
    @staticmethod
    def get_best_available_model() -> ModelManager:
        """
        Try to load the best available model:
        1. First try S3 production model
        2. Fall back to local model
        """
        manager = ModelManager()
        
        # Try S3 production model first
        try:
            logger.info("Attempting to load production model from S3...")
            manager.load_production_model()
            logger.info("✅ Successfully loaded production model from S3")
            return manager
            
        except Exception as e:
            logger.warning(f"Could not load S3 model: {e}")
            logger.info("Falling back to local model...")
            
            # Fall back to local model
            try:
                local_model_path = "models/base_cnn_cifar10_cpu.h5"
                manager.load_local_model(local_model_path)
                logger.info("✅ Successfully loaded local model")
                return manager
                
            except Exception as e2:
                logger.error(f"Could not load local model: {e2}")
                raise RuntimeError(f"Could not load any model. S3 error: {e}, Local error: {e2}")
    
    @staticmethod
    def force_s3_model() -> ModelManager:
        """Force loading S3 production model."""
        manager = ModelManager()
        manager.load_production_model()
        return manager
    
    @staticmethod
    def force_local_model(model_path: str = "models/base_cnn_cifar10_cpu.h5") -> ModelManager:
        """Force loading local model."""
        manager = ModelManager()
        manager.load_local_model(model_path)
        return manager
    
    @staticmethod
    def get_mlflow_production_model() -> ModelManager:
        """Force reload production model from MLFlow (bypass cache)."""
        manager = ModelManager()
        manager.load_production_model()
        return manager
    
    @staticmethod
    def load_specific_model(bucket_name: str, model_key: str) -> ModelManager:
        """Load a specific model from S3."""
        manager = ModelManager()
        manager.load_model_by_path(bucket_name, model_key)
        return manager



