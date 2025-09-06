"""
Simple test for loading Base_CNN model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_model_loading():
    """Test if model can be loaded."""
    print("Testing model loading...")
    
    try:
        import tensorflow as tf
        from models.base_cnn import Base_CNN
        
        model_path = Path("models/base_cnn_cifar10_cpu.h5")
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            return False
        
        print(f"✅ Model file exists: {model_path}")
        
        # Load model with custom object scope
        with tf.keras.utils.custom_object_scope({'Base_CNN': Base_CNN}):
            model = tf.keras.models.load_model(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Model parameters: {model.count_params():,}")
        
        # Test prediction
        import numpy as np
        dummy_input = np.random.random((1, 32, 32, 3)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Model prediction works! Output shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
    else:
        print("\n💥 Model loading test FAILED!")
