# 🚀 KAGGLE NOTEBOOK - MLFlow Training Integration
# Paste this into Kaggle Notebook for FREE GPU training

import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================
MLFLOW_SERVER_URI = "http://13.51.104.28:5000"  # Your MLFlow server
WEBAPI_URL = "http://cifar-webapi-alb-1528959105.eu-north-1.elb.amazonaws.com"

# Set MLFlow tracking
mlflow.set_tracking_uri(MLFLOW_SERVER_URI)

def setup_kaggle_environment():
    """Setup environment for Kaggle notebook"""
    print("🔧 Setting up Kaggle environment...")
    
    # Install additional packages if needed
    os.system("pip install mlflow boto3 -q")
    
    # Enable GPU if available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✅ GPU available: {physical_devices[0]}")
    else:
        print("⚠️ Using CPU - consider enabling GPU in Kaggle settings")

def load_and_preprocess_cifar10():
    """Load and preprocess CIFAR-10 dataset"""
    print("📊 Loading CIFAR-10 dataset...")
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"✅ Data loaded:")
    print(f"   Train: {x_train.shape}, Test: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_enhanced_cnn():
    """Create enhanced CNN model for better performance"""
    model = tf.keras.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_with_mlflow(experiment_name, model_description="Enhanced CNN from Kaggle"):
    """Train model and log to MLFlow"""
    
    # Create/get experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {experiment_name}")
    
    # Start MLFlow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_name = f"kaggle_enhanced_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("training_platform", "Kaggle")
        mlflow.set_tag("model_type", "Enhanced CNN")
        
        print(f"🚀 MLFlow Run started: {run.info.run_id}")
        
        # Load data
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()
        
        # Create model
        model = create_enhanced_cnn()
        print(f"🏗️ Model created - Parameters: {model.count_params():,}")
        
        # Log parameters
        mlflow.log_param("model_type", "Enhanced CNN")
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 20)
        mlflow.log_param("training_platform", "Kaggle")
        mlflow.log_param("total_parameters", model.count_params())
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
        
        # Train model
        print("🎯 Starting training...")
        history = model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=20,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("final_train_accuracy", max(history.history['accuracy']))
        mlflow.log_metric("final_val_accuracy", max(history.history['val_accuracy']))
        
        # Save model locally
        model_filename = f"enhanced_cnn_kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        model.save(model_filename)
        print(f"💾 Model saved: {model_filename}")
        
        # Log model as artifact
        mlflow.log_artifact(model_filename, artifact_path="model")
        
        # Create and log training plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_plots.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('training_plots.png', artifact_path="plots")
        plt.show()
        
        # Confusion matrix
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png', artifact_path="plots")
        plt.show()
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"🔗 MLFlow Run: {MLFLOW_SERVER_URI}/#/experiments/{experiment_id}/runs/{run.info.run_id}")
        
        return model, history, run.info.run_id, test_accuracy

def deploy_to_webapi(run_id, model_name, test_accuracy):
    """Deploy trained model to WebAPI"""
    print(f"\n🚀 Deploying model to WebAPI...")
    
    # Note: This would require the model to be accessible via S3
    # For now, just show the deployment concept
    print(f"   Model: {model_name}")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   Run ID: {run_id}")
    print(f"   WebAPI: {WEBAPI_URL}")
    
    print("💡 To deploy: Use the SageMaker deployment notebook with this run_id")

# ============================================
# MAIN TRAINING WORKFLOW
# ============================================

def main():
    """Main training workflow"""
    
    print("🚀 KAGGLE → MLFlow → WebAPI WORKFLOW")
    print("="*60)
    
    # Setup environment
    setup_kaggle_environment()
    
    # Train model
    experiment_name = "Kaggle Enhanced CNN"
    model, history, run_id, accuracy = train_with_mlflow(experiment_name)
    
    # Show deployment info
    deploy_to_webapi(run_id, experiment_name, accuracy)
    
    print(f"\n✅ WORKFLOW COMPLETED!")
    print(f"🌐 Check MLFlow: {MLFLOW_SERVER_URI}")
    print(f"🌐 Check WebAPI: {WEBAPI_URL}")

# Run the workflow
if __name__ == "__main__":
    main()








