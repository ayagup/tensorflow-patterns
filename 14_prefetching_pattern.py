"""
Prefetching Pattern
Overlap data preprocessing and model execution for better performance.
"""

import tensorflow as tf
import numpy as np
import time

# Generate dummy data
X_train = np.random.random((10000, 100))
y_train = np.random.randint(0, 10, (10000,))

print("Prefetching Performance Comparison\n")

# Simulate expensive preprocessing
def expensive_preprocess(x, y):
    # Simulate computation time
    x = tf.py_function(lambda x: (time.sleep(0.001), x)[1], [x], tf.float64)
    x = tf.cast(x, tf.float32)
    x = x * 2.0 + 1.0
    return x, y

# Example 1: Without Prefetching
print("Example 1: Without Prefetching")
dataset_no_prefetch = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(expensive_preprocess)
    .batch(32)
)

start = time.time()
for i, (x, y) in enumerate(dataset_no_prefetch.take(20)):
    # Simulate training step
    pass
time_no_prefetch = time.time() - start
print(f"Time without prefetch: {time_no_prefetch:.2f}s")

# Example 2: With Prefetching
print("\nExample 2: With Prefetching")
dataset_with_prefetch = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(expensive_preprocess)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)  # Let TensorFlow automatically tune
)

start = time.time()
for i, (x, y) in enumerate(dataset_with_prefetch.take(20)):
    # Simulate training step
    pass
time_with_prefetch = time.time() - start
print(f"Time with prefetch: {time_with_prefetch:.2f}s")
print(f"Speedup: {time_no_prefetch / time_with_prefetch:.2f}x")

# Example 3: Manual prefetch buffer
print("\nExample 3: Manual Prefetch Buffer Size")
dataset_manual_prefetch = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(expensive_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(buffer_size=3)  # Manually set buffer size
)

# Example 4: Complete optimized pipeline
print("\nExample 4: Complete Optimized Pipeline")
optimized_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(1000)
    .map(expensive_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .cache()  # Cache after preprocessing
    .prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
)

start = time.time()
for i, (x, y) in enumerate(optimized_dataset.take(20)):
    pass
time_optimized = time.time() - start
print(f"Time with full optimization: {time_optimized:.2f}s")

# Example 5: Prefetch with model training
print("\nExample 5: Prefetch with Model Training")
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(100,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Dataset with prefetching
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train[:8000], y_train[:8000]))
    .shuffle(8000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train[8000:], y_train[8000:]))
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

print("\nTraining with prefetched dataset...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    verbose=1
)

# Example 6: Prefetch benefits visualization
print("\nExample 6: Understanding Prefetch Benefits")
print("""
Without Prefetch:
[Prepare Data] -> [Train] -> [Prepare Data] -> [Train] -> ...
        T1          T2           T3          T4

With Prefetch:
[Prepare Data 1] -> [Train 1 + Prepare Data 2] -> [Train 2 + Prepare Data 3] -> ...
       T1              T2 (overlapped)              T3 (overlapped)

Prefetch allows data preparation and model training to happen in parallel!
""")

print("\nKey Prefetch Tips:")
print("1. Use tf.data.AUTOTUNE for automatic buffer size tuning")
print("2. Place prefetch() at the end of your pipeline")
print("3. Combine with parallel mapping for best performance")
print("4. Cache before prefetch if data fits in memory")
