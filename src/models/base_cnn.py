"""
Base CNN model for CIFAR-10 classification.
Simple architecture optimized for CPU training.
"""

import tensorflow as tf
from typing import Optional, Dict, Any


class Base_CNN(tf.keras.Model):
    """
    Very simple CNN architecture for CIFAR-10 classification.
    Designed for fast CPU training with minimal complexity.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2, **kwargs):
        super(Base_CNN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Simplified convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(3, 3), 
            activation='relu',
            padding='same',
            name='conv1'
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            activation='relu',
            padding='same',
            name='conv2'
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')
        
        # Single dense layer
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    
    def call(self, inputs, training=None):
        # First conv block
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.pool2(x)
        
        # Dense layer
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.output_layer(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        })
        return config


def create_base_cnn_model(
    input_shape: tuple = (32, 32, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Create and compile a Base_CNN model for CIFAR-10 classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model ready for training
    """
    # Create model
    inputs = tf.keras.Input(shape=input_shape, name='input')
    model = Base_CNN(num_classes=num_classes, dropout_rate=dropout_rate)
    outputs = model(inputs)
    
    # Create functional model
    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Base_CNN')
    
    # Compile model
    functional_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return functional_model


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get model summary as string.
    
    Args:
        model: Keras model
    
    Returns:
        Model summary string
    """
    import io
    import sys
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    return buffer.getvalue()


if __name__ == "__main__":
    # Test model creation
    model = create_base_cnn_model()
    print("Base_CNN Model created successfully!")
    print(f"Model parameters: {model.count_params():,}")
    print("\nModel Summary:")
    print(get_model_summary(model))

