"""
Monitoring module for CIFAR-10 image classification.
Provides MLFlow integration for experiment tracking and model logging.
"""

from .mlflow_monitor import MLFlowMonitor, MLFlowConfig

__all__ = ['MLFlowMonitor', 'MLFlowConfig']
