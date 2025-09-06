"""
MLFlow monitoring module for CIFAR-10 classification experiments.
Provides comprehensive logging of metrics, models, and artifacts.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient


@dataclass
class MLFlowConfig:
    """Configuration for MLFlow tracking."""
    
    # MLFlow settings
    experiment_name: str = "cifar10_base_cnn"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None  # If None, uses local file store
    
    # Artifact paths
    model_artifact_path: str = "model"
    plots_artifact_path: str = "plots"
    metrics_artifact_path: str = "metrics"
    
    # Logging settings
    log_model: bool = True
    log_plots: bool = True
    log_metrics: bool = True
    log_params: bool = True
    
    # Plot settings
    plot_dpi: int = 300
    plot_format: str = "png"


class MLFlowMonitor:
    """
    MLFlow monitor for CIFAR-10 classification experiments.
    Handles logging of metrics, models, confusion matrices, and training plots.
    """
    
    def __init__(self, config: MLFlowConfig):
        self.config = config
        self.client = None
        self.run_id = None
        self.experiment_id = None
        
        # Setup MLFlow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLFlow tracking."""
        # Set tracking URI if provided
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Initialize client
        self.client = MlflowClient()
        
        # Create or get experiment
        try:
            experiment = self.client.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self.experiment_id = self.client.create_experiment(self.config.experiment_name)
                print(f"Created new experiment: {self.config.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                print(f"Using existing experiment: {self.config.experiment_name}")
        except Exception as e:
            print(f"Error setting up experiment: {e}")
            # Fallback to default experiment
            self.experiment_id = "0"
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLFlow run.
        
        Args:
            run_name: Name for the run
            
        Returns:
            Run ID
        """
        run_name = run_name or self.config.run_name or f"base_cnn_run_{np.random.randint(1000, 9999)}"
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
            self.run_id = run.info.run_id
            print(f"Started MLFlow run: {run_name} (ID: {self.run_id})")
        
        return self.run_id
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log training parameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.config.log_params:
            return
        
        try:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_params(params)
            print(f"Logged {len(params)} parameters")
        except Exception as e:
            print(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLFlow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        if not self.config.log_metrics:
            return
        
        try:
            with mlflow.start_run(run_id=self.run_id):
                for name, value in metrics.items():
                    mlflow.log_metric(name, value, step=step)
            print(f"Logged {len(metrics)} metrics")
        except Exception as e:
            print(f"Error logging metrics: {e}")
    
    def log_training_history(self, history: tf.keras.callbacks.History):
        """
        Log training history metrics.
        
        Args:
            history: Keras training history
        """
        if not self.config.log_metrics:
            return
        
        try:
            with mlflow.start_run(run_id=self.run_id):
                # Log final metrics
                final_metrics = {
                    'final_train_accuracy': history.history['accuracy'][-1],
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'best_val_accuracy': max(history.history['val_accuracy']),
                    'best_val_loss': min(history.history['val_loss']),
                    'total_epochs': len(history.history['loss'])
                }
                
                for name, value in final_metrics.items():
                    mlflow.log_metric(name, value)
                
                # Log metrics per epoch
                for epoch, (train_acc, train_loss, val_acc, val_loss) in enumerate(
                    zip(history.history['accuracy'], 
                        history.history['loss'],
                        history.history['val_accuracy'],
                        history.history['val_loss'])
                ):
                    mlflow.log_metrics({
                        'train_accuracy': train_acc,
                        'train_loss': train_loss,
                        'val_accuracy': val_acc,
                        'val_loss': val_loss
                    }, step=epoch)
            
            print("Logged training history metrics")
        except Exception as e:
            print(f"Error logging training history: {e}")
    
    def log_model(self, model: tf.keras.Model, model_name: str = "base_cnn"):
        """
        Log the trained model to MLFlow.
        
        Args:
            model: Trained Keras model
            model_name: Name for the logged model
        """
        if not self.config.log_model:
            return
        
        try:
            with mlflow.start_run(run_id=self.run_id):
                # Log model using MLFlow's TensorFlow integration
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path=self.config.model_artifact_path,
                    registered_model_name=model_name
                )
            print(f"Logged model: {model_name}")
        except Exception as e:
            print(f"Error logging model: {e}")
    
    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   class_names: List[str]) -> plt.Figure:
        """
        Create confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Matplotlib figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_training_plots(self, history: tf.keras.callbacks.History) -> List[plt.Figure]:
        """
        Create training plots (accuracy and loss curves).
        
        Args:
            history: Keras training history
            
        Returns:
            List of matplotlib figures
        """
        figures = []
        
        # Accuracy plot
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        figures.append(fig1)
        
        return figures
    
    def log_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str]):
        """
        Log evaluation metrics including confusion matrix and classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        if not self.config.log_plots:
            return
        
        try:
            with mlflow.start_run(run_id=self.run_id):
                # Create confusion matrix plot
                cm_fig = self.create_confusion_matrix_plot(y_true, y_pred, class_names)
                mlflow.log_figure(cm_fig, f"{self.config.plots_artifact_path}/confusion_matrix.png")
                plt.close(cm_fig)
                
                # Log classification report
                report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
                
                # Log per-class metrics
                for class_name in class_names:
                    if class_name in report:
                        mlflow.log_metrics({
                            f'precision_{class_name}': report[class_name]['precision'],
                            f'recall_{class_name}': report[class_name]['recall'],
                            f'f1_{class_name}': report[class_name]['f1-score']
                        })
                
                # Log overall metrics
                mlflow.log_metrics({
                    'macro_avg_precision': report['macro avg']['precision'],
                    'macro_avg_recall': report['macro avg']['recall'],
                    'macro_avg_f1': report['macro avg']['f1-score'],
                    'weighted_avg_precision': report['weighted avg']['precision'],
                    'weighted_avg_recall': report['weighted avg']['recall'],
                    'weighted_avg_f1': report['weighted avg']['f1-score']
                })
                
                # Save classification report as artifact
                report_path = Path("temp_classification_report.txt")
                with open(report_path, 'w') as f:
                    f.write(classification_report(y_true, y_pred, target_names=class_names))
                
                mlflow.log_artifact(report_path, self.config.metrics_artifact_path)
                report_path.unlink()  # Clean up temp file
            
            print("Logged evaluation metrics and plots")
        except Exception as e:
            print(f"Error logging evaluation metrics: {e}")
    
    def log_training_plots(self, history: tf.keras.callbacks.History):
        """
        Log training plots to MLFlow.
        
        Args:
            history: Keras training history
        """
        if not self.config.log_plots:
            return
        
        try:
            with mlflow.start_run(run_id=self.run_id):
                figures = self.create_training_plots(history)
                
                for i, fig in enumerate(figures):
                    mlflow.log_figure(fig, f"{self.config.plots_artifact_path}/training_curves_{i}.png")
                    plt.close(fig)
            
            print("Logged training plots")
        except Exception as e:
            print(f"Error logging training plots: {e}")
    
    def log_model_summary(self, model: tf.keras.Model):
        """
        Log model summary as artifact.
        
        Args:
            model: Keras model
        """
        try:
            with mlflow.start_run(run_id=self.run_id):
                # Get model summary
                import io
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                model.summary()
                sys.stdout = old_stdout
                
                summary_text = buffer.getvalue()
                
                # Save summary to file
                summary_path = Path("temp_model_summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(summary_text)
                
                mlflow.log_artifact(summary_path, self.config.metrics_artifact_path)
                summary_path.unlink()  # Clean up temp file
            
            print("Logged model summary")
        except Exception as e:
            print(f"Error logging model summary: {e}")
    
    def log_complete_experiment(self, 
                              model: tf.keras.Model,
                              history: tf.keras.callbacks.History,
                              training_config: Dict[str, Any],
                              test_results: Dict[str, float],
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: List[str]):
        """
        Log complete experiment data to MLFlow.
        
        Args:
            model: Trained model
            history: Training history
            training_config: Training configuration
            test_results: Test evaluation results
            y_true: True test labels
            y_pred: Predicted test labels
            class_names: List of class names
        """
        print("Logging complete experiment to MLFlow...")
        
        # Log parameters
        self.log_parameters(training_config)
        
        # Log test results
        self.log_metrics(test_results)
        
        # Log training history
        self.log_training_history(history)
        
        # Log model
        self.log_model(model)
        
        # Log plots and evaluation metrics
        self.log_training_plots(history)
        self.log_evaluation_metrics(y_true, y_pred, class_names)
        
        # Log model summary
        self.log_model_summary(model)
        
        print("Complete experiment logged to MLFlow!")
    
    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the current run.
        
        Returns:
            Dictionary with run information
        """
        if self.run_id is None:
            return {}
        
        try:
            run = self.client.get_run(self.run_id)
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri
            }
        except Exception as e:
            print(f"Error getting run info: {e}")
            return {}


# CIFAR-10 class names
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def create_mlflow_monitor(experiment_name: str = "cifar10_base_cnn", 
                         run_name: Optional[str] = None) -> MLFlowMonitor:
    """
    Create and configure MLFlow monitor.
    
    Args:
        experiment_name: Name of the MLFlow experiment
        run_name: Name for the run
        
    Returns:
        Configured MLFlowMonitor instance
    """
    config = MLFlowConfig(
        experiment_name=experiment_name,
        run_name=run_name
    )
    
    monitor = MLFlowMonitor(config)
    return monitor


if __name__ == "__main__":
    # Test MLFlow monitor
    monitor = create_mlflow_monitor()
    run_id = monitor.start_run("test_run")
    
    # Test logging
    test_params = {"batch_size": 32, "learning_rate": 0.001}
    test_metrics = {"accuracy": 0.85, "loss": 0.45}
    
    monitor.log_parameters(test_params)
    monitor.log_metrics(test_metrics)
    
    print(f"Test run completed: {run_id}")
    print("MLFlow monitor is working correctly!")

