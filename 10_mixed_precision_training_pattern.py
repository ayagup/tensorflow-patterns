"""
Mixed Precision Training Pattern
Use float16 for faster training with automatic loss scaling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print(f'Compute dtype: {policy.compute_dtype}')
print(f'Variable dtype: {policy.variable_dtype}')

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 10, (1000,))

# Create model - layers will automatically use mixed precision
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    # Output layer uses float32 for numerical stability
    layers.Dense(10, activation='softmax', dtype='float32')
], name='mixed_precision_model')

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Build model
model.build((None, 20))

# Check layer dtypes
print("\nLayer dtypes:")
for layer in model.layers:
    print(f"{layer.name}: compute={layer.dtype}, "
          f"variables={layer.weights[0].dtype if layer.weights else 'N/A'}")

# Train
print("\nTraining with mixed precision...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Custom training loop with mixed precision
print("\n\nCustom Training Loop with Mixed Precision:")
optimizer = keras.optimizers.Adam()
# Wrap optimizer with loss scale optimizer for mixed precision
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        # Scale loss for mixed precision
        scaled_loss = optimizer.get_scaled_loss(loss)
    
    # Compute scaled gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # Unscale gradients
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Train for one epoch
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
for step, (x_batch, y_batch) in enumerate(dataset.take(5)):
    loss = train_step(x_batch, y_batch)
    if step % 2 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

print("\nMixed precision training can provide 2-3x speedup on modern GPUs!")
