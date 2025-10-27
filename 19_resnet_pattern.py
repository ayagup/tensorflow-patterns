"""
ResNet Pattern
Residual Network with skip connections for training very deep networks.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("ResNet Pattern\n")

# Generate dummy image data
X_train = np.random.random((1000, 32, 32, 3)).astype(np.float32)
y_train = np.random.randint(0, 10, (1000,))

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    """
    Residual block with skip connection.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Conv kernel size
        stride: Stride for conv
        conv_shortcut: Use conv for shortcut if True, identity if False
    """
    shortcut = x
    
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut connection
    if conv_shortcut or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut to main path
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def bottleneck_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    """
    Bottleneck residual block (used in ResNet-50+).
    Uses 1x1 conv to reduce then expand dimensions.
    """
    shortcut = x
    
    # 1x1 conv to reduce dimensions
    x = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 3x3 conv
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 1x1 conv to expand dimensions
    x = layers.Conv2D(filters * 4, 1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut
    if conv_shortcut or stride != 1:
        shortcut = layers.Conv2D(filters * 4, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# Example 1: Simple ResNet-like model
print("Example 1: Simple ResNet")

def create_resnet_small(input_shape=(32, 32, 3), num_classes=10):
    """Create a small ResNet model."""
    inputs = keras.Input(shape=input_shape)
    
    # Initial conv layer
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet_small')
    return model

resnet_model = create_resnet_small()
resnet_model.summary()

resnet_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = resnet_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 2: ResNet with Bottleneck blocks
print("\nExample 2: ResNet with Bottleneck Blocks")

def create_resnet_bottleneck(input_shape=(32, 32, 3), num_classes=10):
    """Create ResNet with bottleneck blocks."""
    inputs = keras.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Bottleneck blocks
    x = bottleneck_block(x, 64, conv_shortcut=True)
    x = bottleneck_block(x, 64)
    
    x = bottleneck_block(x, 128, stride=2, conv_shortcut=True)
    x = bottleneck_block(x, 128)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet_bottleneck')
    return model

resnet_bottleneck = create_resnet_bottleneck()
print(f"Bottleneck ResNet created with {len(resnet_bottleneck.layers)} layers")

# Example 3: ResNet as a feature extractor
print("\nExample 3: ResNet as Feature Extractor")

def create_resnet_backbone(input_shape=(32, 32, 3)):
    """Create ResNet backbone for feature extraction."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    features = layers.GlobalAveragePooling2D()(x)
    
    model = keras.Model(inputs=inputs, outputs=features, name='resnet_backbone')
    return model

backbone = create_resnet_backbone()
print(f"Backbone output shape: {backbone.output_shape}")

# Example 4: Using pre-trained ResNet
print("\nExample 4: Pre-trained ResNet50")

# Load pre-trained ResNet50
base_model = keras.applications.ResNet50(
    include_top=False,
    weights=None,  # Set to 'imagenet' to use pre-trained weights
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

transfer_model = keras.Model(inputs=inputs, outputs=outputs, name='resnet50_transfer')
print(f"Transfer learning model created")

print("\nResNet Key Features:")
print("1. Skip connections allow training very deep networks (50-152+ layers)")
print("2. Solves vanishing gradient problem")
print("3. Identity shortcuts preserve information flow")
print("4. Bottleneck blocks reduce computational cost")
print("5. Enables gradient highway for easier optimization")
