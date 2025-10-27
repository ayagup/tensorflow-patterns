"""
Batch Normalization Pattern
Normalize layer inputs to improve training speed and stability.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 28, 28, 1)).astype(np.float32)
y_train = np.random.randint(0, 10, (1000,))

print("Batch Normalization Pattern\n")

# Example 1: Batch Normalization in CNN
print("Example 1: Batch Normalization in CNN")

model_with_bn = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),  # After conv, before activation
    layers.Activation('relu'),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2),
    
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
], name='model_with_bn')

model_with_bn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_with_bn.summary()

# Train
history_bn = model_with_bn.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 2: Batch Normalization placement options
print("\nExample 2: Different BN Placement")

# Option A: After activation
model_bn_after = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),  # After activation
    layers.Dense(10, activation='softmax')
])

# Option B: Before activation (recommended)
model_bn_before = keras.Sequential([
    layers.Dense(64),
    layers.BatchNormalization(),  # Before activation
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])

# Example 3: Batch Normalization with custom training
print("\nExample 3: Batch Normalization in Custom Training")

class CustomBNModel(keras.Model):
    def __init__(self):
        super(CustomBNModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(64, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128)
        self.bn3 = layers.BatchNormalization()
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)  # Pass training flag
        x = tf.nn.relu(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        
        return self.dense2(x)

custom_bn_model = CustomBNModel()
custom_bn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Example 4: Batch Normalization parameters
print("\nExample 4: Batch Normalization Parameters")

bn_layer = layers.BatchNormalization(
    axis=-1,  # Feature axis (usually -1 for channels_last)
    momentum=0.99,  # Momentum for moving average
    epsilon=0.001,  # Small constant for numerical stability
    center=True,  # Learn beta parameter
    scale=True,  # Learn gamma parameter
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    name='custom_bn'
)

print(f"BN Layer config: {bn_layer.get_config()}")

# Example 5: Comparison with and without BN
print("\nExample 5: Model Without Batch Normalization")

model_without_bn = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='model_without_bn')

model_without_bn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_no_bn = model_without_bn.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"\nWith BN - Final accuracy: {history_bn.history['accuracy'][-1]:.4f}")
print(f"Without BN - Final accuracy: {history_no_bn.history['accuracy'][-1]:.4f}")

# Example 6: Batch Normalization inference mode
print("\nExample 6: BN in Training vs Inference Mode")

x_sample = X_train[:1]
# Training mode (uses batch statistics)
output_train = model_with_bn(x_sample, training=True)
# Inference mode (uses moving averages)
output_inference = model_with_bn(x_sample, training=False)

print(f"Output training: {output_train[0][:3].numpy()}")
print(f"Output inference: {output_inference[0][:3].numpy()}")

print("\nBatch Normalization benefits:")
print("1. Faster training convergence")
print("2. Allows higher learning rates")
print("3. Reduces sensitivity to weight initialization")
print("4. Acts as a regularizer")
print("5. Reduces internal covariate shift")
