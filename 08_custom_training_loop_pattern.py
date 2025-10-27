"""
Custom Training Loop Pattern
Full control over the training process with custom logic.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

# Create simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Define loss and optimizer
loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

# Custom training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss(loss)
    train_accuracy(y, predictions)
    
    return loss

# Training loop
EPOCHS = 5
BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

print("Starting custom training loop...")
for epoch in range(EPOCHS):
    # Reset metrics at the start of each epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # Training
    for step, (x_batch, y_batch) in enumerate(dataset):
        loss = train_step(x_batch, y_batch)
    
    print(f'Epoch {epoch + 1}/{EPOCHS}, '
          f'Loss: {train_loss.result():.4f}, '
          f'Accuracy: {train_accuracy.result():.4f}')

# Predict
predictions = model.predict(X_train[:5])
print(f"\nPredictions: {predictions.flatten()}")
