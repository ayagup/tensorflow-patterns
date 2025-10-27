"""
Residual Connection Pattern
Skip connections that help with gradient flow in deep networks.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def residual_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    """Create a residual block with skip connection."""
    # Main path
    fx = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation(activation)(fx)
    fx = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(fx)
    fx = layers.BatchNormalization()(fx)
    
    # Skip connection
    if stride != 1 or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
    
    # Add skip connection
    output = layers.Add()([x, fx])
    output = layers.Activation(activation)(output)
    
    return output

# Generate dummy image data
X_train = np.random.random((1000, 32, 32, 3))
y_train = np.random.randint(0, 10, (1000,))

# Build model with residual blocks
inputs = keras.Input(shape=(32, 32, 3))

x = layers.Conv2D(32, 3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

# Stack residual blocks
x = residual_block(x, 32)
x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)

# Classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs, name='residual_model')

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\nResidual connections help gradients flow through deep networks!")
