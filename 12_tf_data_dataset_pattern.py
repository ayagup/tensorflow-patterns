"""
tf.data.Dataset Pattern
Efficient input pipeline for loading and preprocessing data.
"""

import tensorflow as tf
import numpy as np

# Example 1: From NumPy arrays
print("Example 1: From NumPy Arrays")
X = np.random.random((1000, 10))
y = np.random.randint(0, 2, (1000,))

dataset = tf.data.Dataset.from_tensor_slices((X, y))
print(f"Dataset: {dataset}")
print(f"Element spec: {dataset.element_spec}")

# Example 2: Batching
print("\nExample 2: Batching")
batched_dataset = dataset.batch(32)
for batch_x, batch_y in batched_dataset.take(1):
    print(f"Batch X shape: {batch_x.shape}, Batch y shape: {batch_y.shape}")

# Example 3: Shuffling
print("\nExample 3: Shuffling")
shuffled_dataset = dataset.shuffle(buffer_size=1000).batch(32)

# Example 4: Mapping (preprocessing)
print("\nExample 4: Mapping")
def preprocess(x, y):
    x = tf.cast(x, tf.float32) * 2.0  # Scale
    y = tf.cast(y, tf.int32)
    return x, y

preprocessed_dataset = dataset.map(preprocess).batch(32)
for batch_x, batch_y in preprocessed_dataset.take(1):
    print(f"Preprocessed batch X (first 3): {batch_x[0, :3].numpy()}")

# Example 5: Repeat and Prefetch
print("\nExample 5: Repeat and Prefetch")
train_dataset = (dataset
                 .shuffle(1000)
                 .batch(32)
                 .repeat(2)  # Repeat for 2 epochs
                 .prefetch(tf.data.AUTOTUNE))

# Example 6: From generator
print("\nExample 6: From Generator")
def data_generator():
    for i in range(100):
        yield (np.random.random(10), np.random.randint(0, 2))

gen_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(10,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
)
print(f"Generator dataset: {gen_dataset}")

# Example 7: Parallel mapping
print("\nExample 7: Parallel Mapping")
parallel_dataset = dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(32)

# Example 8: Caching
print("\nExample 8: Caching")
cached_dataset = (dataset
                  .map(preprocess)
                  .cache()  # Cache in memory
                  .shuffle(1000)
                  .batch(32)
                  .prefetch(tf.data.AUTOTUNE))

# Example 9: Zip datasets
print("\nExample 9: Zip Datasets")
dataset1 = tf.data.Dataset.range(10)
dataset2 = tf.data.Dataset.range(10, 20)
zipped_dataset = tf.data.Dataset.zip((dataset1, dataset2))
for d1, d2 in zipped_dataset.take(3):
    print(f"({d1.numpy()}, {d2.numpy()})")

# Example 10: Complete pipeline
print("\nExample 10: Complete Training Pipeline")
def create_dataset(X, y, batch_size=32, shuffle=True, augment=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    if augment:
        dataset = dataset.map(
            lambda x, y: (x + tf.random.normal(x.shape, 0, 0.1), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = (dataset
               .batch(batch_size)
               .cache()
               .prefetch(tf.data.AUTOTUNE))
    
    return dataset

train_ds = create_dataset(X[:800], y[:800], batch_size=32)
val_ds = create_dataset(X[800:], y[800:], batch_size=32, shuffle=False, augment=False)

print(f"Train dataset: {train_ds}")
print(f"Val dataset: {val_ds}")

# Count batches
num_train_batches = sum(1 for _ in train_ds)
num_val_batches = sum(1 for _ in val_ds)
print(f"Training batches: {num_train_batches}, Validation batches: {num_val_batches}")
