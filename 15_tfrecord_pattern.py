"""
TFRecord Pattern
Efficient binary format for storing and loading large datasets.
"""

import tensorflow as tf
import numpy as np
import os

# Create output directory
output_dir = "tfrecords_data"
os.makedirs(output_dir, exist_ok=True)

print("TFRecord Pattern Examples\n")

# Example 1: Writing TFRecords
print("Example 1: Writing TFRecords")

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feature_array, label):
    """Create a tf.train.Example message."""
    feature = {
        'feature': _bytes_feature(tf.io.serialize_tensor(feature_array)),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Generate dummy data
num_samples = 1000
X_data = np.random.random((num_samples, 20)).astype(np.float32)
y_data = np.random.randint(0, 10, num_samples)

# Write to TFRecord file
tfrecord_filename = os.path.join(output_dir, 'data.tfrecord')
with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    for i in range(num_samples):
        example = serialize_example(X_data[i], y_data[i])
        writer.write(example)

print(f"Written {num_samples} examples to {tfrecord_filename}")

# Example 2: Reading TFRecords
print("\nExample 2: Reading TFRecords")

def _parse_function(example_proto):
    """Parse the input tf.train.Example proto."""
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize the feature tensor
    feature = tf.io.parse_tensor(parsed_features['feature'], out_type=tf.float32)
    feature.set_shape([20])  # Set the shape explicitly
    label = parsed_features['label']
    
    return feature, label

# Create dataset from TFRecord
raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
parsed_dataset = raw_dataset.map(_parse_function)

# Verify
for features, label in parsed_dataset.take(3):
    print(f"Features shape: {features.shape}, Label: {label.numpy()}")

# Example 3: Optimized TFRecord pipeline
print("\nExample 3: Optimized TFRecord Pipeline")

def create_tfrecord_dataset(filenames, batch_size=32, shuffle=True):
    """Create an optimized dataset from TFRecord files."""
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = (dataset
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))
    
    return dataset

train_dataset = create_tfrecord_dataset([tfrecord_filename], batch_size=32)
print(f"Created optimized dataset: {train_dataset}")

# Example 4: Multiple TFRecord files (sharding)
print("\nExample 4: Multiple TFRecord Files (Sharding)")

# Write to multiple shard files
num_shards = 3
samples_per_shard = num_samples // num_shards

for shard in range(num_shards):
    shard_filename = os.path.join(output_dir, f'data_shard_{shard}.tfrecord')
    start_idx = shard * samples_per_shard
    end_idx = start_idx + samples_per_shard
    
    with tf.io.TFRecordWriter(shard_filename) as writer:
        for i in range(start_idx, end_idx):
            example = serialize_example(X_data[i], y_data[i])
            writer.write(example)
    
    print(f"Written shard {shard} to {shard_filename}")

# Read from multiple shards
shard_filenames = [
    os.path.join(output_dir, f'data_shard_{i}.tfrecord')
    for i in range(num_shards)
]

multi_shard_dataset = create_tfrecord_dataset(shard_filenames, batch_size=32)

# Example 5: Train model with TFRecord dataset
print("\nExample 5: Train Model with TFRecord Dataset")
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=3,
    verbose=1
)

# Example 6: Image TFRecord
print("\nExample 6: Image TFRecord Pattern")

def serialize_image_example(image, label):
    """Create a tf.train.Example for image data."""
    # Encode image
    image_encoded = tf.io.encode_jpeg(image)
    
    feature = {
        'image': _bytes_feature(image_encoded),
        'label': _int64_feature(label),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_image_function(example_proto):
    """Parse image TFRecord."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    label = parsed_features['label']
    
    return image, label

print("TFRecord format is ideal for large datasets and distributed training!")

# Cleanup info
print(f"\nTFRecord files created in: {output_dir}/")
print("Benefits: Efficient storage, fast loading, supports streaming, works well with distributed training")
