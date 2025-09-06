"""
Script to run the CIFAR-10 Base_CNN web application.
Starts FastAPI backend server with static file serving.
"""

import sys
from pathlib import Path
import uvicorn

def main():
    """Run the web application."""
    
    print("="*60)
    print("🚀 Starting CIFAR-10 Base_CNN Web Application")
    print("="*60)
    
    # Check if model exists
    model_path = Path("models/base_cnn_cifar10_cpu.h5")
    if not model_path.exists():
        print("❌ Model not found!")
        print(f"Expected model at: {model_path}")
        print("Please run training first: python train_base_cnn.py")
        return
    
    # Check if data exists
    data_path = Path("data/raw/cifar-10-batches-py")
    if not data_path.exists():
        print("❌ CIFAR-10 data not found!")
        print(f"Expected data at: {data_path}")
        print("Please extract CIFAR-10 data first.")
        return
    
    print("✅ Model and data found!")
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Run the FastAPI application
        uvicorn.run(
            "api.app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Auto-reload on code changes
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


if __name__ == "__main__":
    main()
