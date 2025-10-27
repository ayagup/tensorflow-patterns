"""
L2 Regularization Pattern
Add penalty on large weights to prevent overfitting.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

# Generate dummy data with potential for overfitting
X_train = np.random.random((500, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (500, 1)).astype(np.float32)

print("L2 Regularization Pattern\n")

# Example 1: L2 Regularization in Dense layers
print("Example 1: L2 Regularization in Dense Network")

l2_lambda = 0.01

model_with_l2 = keras.Sequential([
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(l2_lambda),
                 input_shape=(20,)),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(l2_lambda)),
    layers.Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(l2_lambda)),
    layers.Dense(1, activation='sigmoid',
                 kernel_regularizer=regularizers.l2(l2_lambda))
], name='model_with_l2')

model_with_l2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_with_l2.summary()

history_l2 = model_with_l2.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 2: L1 Regularization (Lasso)
print("\nExample 2: L1 Regularization")

model_with_l1 = keras.Sequential([
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l1(0.01),
                 input_shape=(20,)),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l1(0.01)),
    layers.Dense(1, activation='sigmoid')
])

model_with_l1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("L1 regularization encourages sparsity (many weights become zero)")

# Example 3: L1L2 Regularization (Elastic Net)
print("\nExample 3: L1L2 Regularization (Elastic Net)")

model_with_l1l2 = keras.Sequential([
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                 input_shape=(20,)),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Dense(1, activation='sigmoid')
])

model_with_l1l2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("L1L2 combines benefits of both L1 and L2 regularization")

# Example 4: Regularization on different components
print("\nExample 4: Regularizing Different Components")

model_comprehensive = keras.Sequential([
    layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01),  # Weight regularization
        bias_regularizer=regularizers.l2(0.01),    # Bias regularization
        activity_regularizer=regularizers.l2(0.01), # Output regularization
        input_shape=(20,)
    ),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1, activation='sigmoid')
])

print("Can regularize kernels, biases, and activations separately")

# Example 5: L2 in Convolutional layers
print("\nExample 5: L2 Regularization in CNN")

X_image = np.random.random((500, 28, 28, 1)).astype(np.float32)
y_image = np.random.randint(0, 10, (500,))

cnn_with_l2 = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')
])

cnn_with_l2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Example 6: Custom regularizer
print("\nExample 6: Custom Regularizer")

class CustomRegularizer(regularizers.Regularizer):
    def __init__(self, strength=0.01):
        self.strength = strength
    
    def __call__(self, x):
        # Custom penalty: sum of absolute values squared
        return self.strength * tf.reduce_sum(tf.square(tf.abs(x)))
    
    def get_config(self):
        return {'strength': self.strength}

custom_reg_model = keras.Sequential([
    layers.Dense(64, activation='relu',
                 kernel_regularizer=CustomRegularizer(0.01),
                 input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

print("Custom regularizers allow for specialized penalty functions")

# Example 7: Weight Decay (alternative to L2)
print("\nExample 7: Weight Decay in Optimizer")

# Weight decay in optimizer (different from L2 in loss)
model_weight_decay = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Use AdamW optimizer with weight decay
optimizer_with_decay = tf.keras.optimizers.experimental.AdamW(
    learning_rate=0.001,
    weight_decay=0.01
)

model_weight_decay.compile(
    optimizer=optimizer_with_decay,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Weight decay in optimizer is computationally equivalent but mathematically different from L2")

# Example 8: Comparison - No Regularization vs L2
print("\nExample 8: Comparison")

# Model without regularization
model_no_reg = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_no_reg.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_no_reg = model_no_reg.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Compare
print(f"\nWith L2 Regularization:")
print(f"  Train acc: {history_l2.history['accuracy'][-1]:.4f}")
print(f"  Val acc: {history_l2.history['val_accuracy'][-1]:.4f}")
print(f"  Overfitting gap: {history_l2.history['accuracy'][-1] - history_l2.history['val_accuracy'][-1]:.4f}")

print(f"\nWithout Regularization:")
print(f"  Train acc: {history_no_reg.history['accuracy'][-1]:.4f}")
print(f"  Val acc: {history_no_reg.history['val_accuracy'][-1]:.4f}")
print(f"  Overfitting gap: {history_no_reg.history['accuracy'][-1] - history_no_reg.history['val_accuracy'][-1]:.4f}")

# Example 9: Examining weight magnitudes
print("\nExample 9: Weight Magnitudes")

# Get weights from both models
weights_l2 = model_with_l2.layers[0].get_weights()[0]
weights_no_reg = model_no_reg.layers[0].get_weights()[0]

print(f"L2 regularized weights - Mean abs: {np.abs(weights_l2).mean():.6f}, Max abs: {np.abs(weights_l2).max():.6f}")
print(f"Non-regularized weights - Mean abs: {np.abs(weights_no_reg).mean():.6f}, Max abs: {np.abs(weights_no_reg).max():.6f}")

print("\nL2 Regularization Guidelines:")
print("1. Common lambda values: 0.001 to 0.1")
print("2. Penalizes large weights, encouraging smaller distributed weights")
print("3. Loss = Original Loss + λ * Σ(w²)")
print("4. Helps prevent overfitting")
print("5. Makes model more robust to noise")
print("6. Can be combined with dropout for better regularization")
