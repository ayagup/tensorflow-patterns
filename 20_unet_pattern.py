"""
U-Net Pattern
U-shaped architecture for image segmentation with skip connections.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("U-Net Pattern\n")

# Generate dummy segmentation data
X_train = np.random.random((100, 128, 128, 3)).astype(np.float32)
y_train = np.random.randint(0, 2, (100, 128, 128, 1)).astype(np.float32)

def conv_block(inputs, num_filters):
    """Basic convolutional block for U-Net."""
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def encoder_block(inputs, num_filters):
    """Encoder block: conv_block + max pooling."""
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D(2)(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Decoder block: upsampling + concatenation + conv_block."""
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def create_unet(input_shape=(128, 128, 3), num_classes=1):
    """
    Create U-Net architecture.
    
    Architecture:
    - Contracting path (encoder): captures context
    - Expanding path (decoder): enables precise localization
    - Skip connections: combine low and high level features
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b = conv_block(p4, 1024)
    
    # Decoder (Expanding Path)
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Output layer
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    
    outputs = layers.Conv2D(num_classes, 1, activation=activation)(d4)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='unet')
    return model

# Example 1: Standard U-Net
print("Example 1: Standard U-Net")

unet_model = create_unet(input_shape=(128, 128, 3), num_classes=1)
unet_model.summary()

# Compile with appropriate loss for segmentation
unet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # For binary segmentation
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
)

# Train
history = unet_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# Example 2: U-Net with Attention Gates
print("\nExample 2: U-Net with Attention Gates")

def attention_gate(x, g, num_filters):
    """
    Attention gate to focus on relevant features.
    x: features from encoder
    g: gating signal from decoder
    """
    # Transform x and g to same number of filters
    x_transform = layers.Conv2D(num_filters, 1, padding='same')(x)
    x_transform = layers.BatchNormalization()(x_transform)
    
    g_transform = layers.Conv2D(num_filters, 1, padding='same')(g)
    g_transform = layers.BatchNormalization()(g_transform)
    
    # Add and apply activation
    combined = layers.Add()([x_transform, g_transform])
    combined = layers.Activation('relu')(combined)
    
    # Attention coefficients
    attention = layers.Conv2D(1, 1, padding='same')(combined)
    attention = layers.Activation('sigmoid')(attention)
    
    # Apply attention
    output = layers.Multiply()([x, attention])
    
    return output

def decoder_block_with_attention(inputs, skip_features, num_filters):
    """Decoder block with attention gate."""
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    
    # Apply attention gate
    skip_features = attention_gate(skip_features, x, num_filters)
    
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def create_attention_unet(input_shape=(128, 128, 3), num_classes=1):
    """Create Attention U-Net."""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b = conv_block(p4, 1024)
    
    # Decoder with attention
    d1 = decoder_block_with_attention(b, s4, 512)
    d2 = decoder_block_with_attention(d1, s3, 256)
    d3 = decoder_block_with_attention(d2, s2, 128)
    d4 = decoder_block_with_attention(d3, s1, 64)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(d4)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='attention_unet')
    return model

attention_unet = create_attention_unet()
print(f"Attention U-Net created with {len(attention_unet.layers)} layers")

# Example 3: Multi-class U-Net
print("\nExample 3: Multi-class U-Net")

# Dummy multi-class segmentation data
y_multiclass = np.random.randint(0, 5, (100, 128, 128, 1)).astype(np.int32)

multiclass_unet = create_unet(input_shape=(128, 128, 3), num_classes=5)

multiclass_unet.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=5)]
)

print("Multi-class U-Net for segmenting multiple object types")

# Example 4: U-Net with Residual Connections
print("\nExample 4: Residual U-Net")

def residual_conv_block(inputs, num_filters):
    """Convolutional block with residual connection."""
    shortcut = inputs
    
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Match dimensions if needed
    if shortcut.shape[-1] != num_filters:
        shortcut = layers.Conv2D(num_filters, 1, padding='same')(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

print("Residual U-Net combines skip connections from U-Net and ResNet")

# Example 5: Custom loss for U-Net
print("\nExample 5: Dice Loss for Segmentation")

def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient for segmentation."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation."""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_dice_bce_loss(y_true, y_pred):
    """Combined Dice and Binary Cross-Entropy loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

unet_dice = create_unet()
unet_dice.compile(
    optimizer='adam',
    loss=combined_dice_bce_loss,
    metrics=[dice_coefficient, 'accuracy']
)

print("Dice loss helps with class imbalance in segmentation")

# Example 6: U-Net prediction
print("\nExample 6: U-Net Prediction")

# Make prediction
sample_image = X_train[:1]
prediction = unet_model.predict(sample_image)

print(f"Input shape: {sample_image.shape}")
print(f"Output shape: {prediction.shape}")
print(f"Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")

print("\nU-Net Key Features:")
print("1. Symmetric U-shaped architecture")
print("2. Skip connections preserve spatial information")
print("3. Contracting path captures context")
print("4. Expanding path enables precise localization")
print("5. Widely used for medical image segmentation")
print("6. Works well with limited training data")
