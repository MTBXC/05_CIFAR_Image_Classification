from __future__ import annotations

import tensorflow as tf


def build_augmentation_pipeline() -> tf.keras.Sequential:
    # On-the-fly augmentations suitable for CIFAR-10
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="cifar_augmentations")


