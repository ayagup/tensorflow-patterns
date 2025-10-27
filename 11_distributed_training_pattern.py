"""
Distributed Training Pattern
Train across multiple GPUs or machines using tf.distribute.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Create distribution strategy
# For single machine with multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"\nNumber of devices: {strategy.num_replicas_in_sync}")

# Generate dummy data
X_train = np.random.random((10000, 20))
y_train = np.random.randint(0, 10, (10000,))

# Global batch size (will be split across replicas)
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(GLOBAL_BATCH_SIZE)

# Distribute dataset
dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Create model inside strategy scope
with strategy.scope():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(20,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    # Define loss and optimizer inside strategy scope
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(
            per_example_loss,
            global_batch_size=GLOBAL_BATCH_SIZE
        )
    
    optimizer = keras.optimizers.Adam()
    
    # Metrics
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

# Define training step
@tf.function
def distributed_train_step(inputs):
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = compute_loss(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_accuracy.update_state(y, predictions)
        return loss
    
    per_replica_losses = strategy.run(train_step, args=inputs)
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM,
        per_replica_losses,
        axis=None
    )

# Training loop
EPOCHS = 3
print("\nTraining with distributed strategy...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    
    for batch, inputs in enumerate(dist_dataset):
        loss = distributed_train_step(inputs)
        total_loss += loss
        num_batches += 1
        
        if batch % 10 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch}, Loss: {loss:.4f}')
    
    train_loss = total_loss / num_batches
    print(f'Epoch {epoch + 1}, Average Loss: {train_loss:.4f}, '
          f'Accuracy: {train_accuracy.result():.4f}')
    train_accuracy.reset_states()

print("\nDistributed training scales to multiple GPUs/TPUs efficiently!")
