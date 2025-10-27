"""
Gradient Tape Pattern
Manual control over gradient computation for custom operations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.random((1000, 1)).astype(np.float32)

# Create model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Loss and optimizer
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

# Example 1: Basic gradient tape
print("Example 1: Basic Gradient Computation")
x_batch = X_train[:32]
y_batch = y_train[:32]

with tf.GradientTape() as tape:
    predictions = model(x_batch, training=True)
    loss = loss_fn(y_batch, predictions)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print(f"Loss: {loss.numpy():.4f}")

# Example 2: Persistent tape for multiple gradient computations
print("\nExample 2: Persistent Tape")
with tf.GradientTape(persistent=True) as tape:
    predictions = model(x_batch, training=True)
    loss = loss_fn(y_batch, predictions)
    
    # Custom regularization
    l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
    total_loss = loss + 0.01 * l2_reg

# Compute gradients with respect to loss
grad_loss = tape.gradient(loss, model.trainable_variables)
# Compute gradients with respect to total_loss
grad_total = tape.gradient(total_loss, model.trainable_variables)

del tape  # Delete persistent tape to free resources

print(f"Loss: {loss.numpy():.4f}, Total Loss: {total_loss.numpy():.4f}")

# Example 3: Watching non-trainable tensors
print("\nExample 3: Watching Non-Trainable Tensors")
x = tf.constant([[1.0, 2.0, 3.0]])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x ** 2)

dy_dx = tape.gradient(y, x)
print(f"Input: {x.numpy()}")
print(f"Gradient: {dy_dx.numpy()}")

# Example 4: Higher-order gradients
print("\nExample 4: Second-Order Gradients")
x = tf.Variable(3.0)
with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        y = x ** 3
    dy_dx = tape2.gradient(y, x)
d2y_dx2 = tape1.gradient(dy_dx, x)

print(f"y = x^3, x = {x.numpy()}")
print(f"dy/dx = {dy_dx.numpy()}")
print(f"d²y/dx² = {d2y_dx2.numpy()}")
