"""
Data Augmentation Pattern
Enhance training data with random transformations to improve generalization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Generate dummy image data
X_train = np.random.random((100, 32, 32, 3)).astype(np.float32)
y_train = np.random.randint(0, 10, (100,))

print("Image Data Augmentation Patterns")

# Method 1: Using Keras Preprocessing Layers
print("\nMethod 1: Keras Preprocessing Layers")
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Apply augmentation
augmented_image = data_augmentation(X_train[:1])
print(f"Original shape: {X_train[0].shape}, Augmented shape: {augmented_image.shape}")

# Method 2: Augmentation as part of model
print("\nMethod 2: Augmentation in Model")
model = keras.Sequential([
    # Data augmentation layers (only active during training)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    # Model layers
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with augmentation
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Method 3: Custom augmentation function
print("\nMethod 3: Custom Augmentation Function")
@tf.function
def augment(image, label):
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Random hue
    image = tf.image.random_hue(image, max_delta=0.1)
    # Random flip
    image = tf.image.random_flip_left_right(image)
    # Random flip vertical
    image = tf.image.random_flip_up_down(image)
    # Clip values
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Create augmented dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
augmented_dataset = dataset.map(
    augment,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(16).prefetch(tf.data.AUTOTUNE)

# Method 4: tf.image augmentation
print("\nMethod 4: tf.image Transformations")
def advanced_augment(image):
    # Random crop and resize
    image = tf.image.random_crop(image, size=[28, 28, 3])
    image = tf.image.resize(image, [32, 32])
    
    # Random jpeg quality (compression)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=75, max_jpeg_quality=100)
    
    # Per image standardization
    image = tf.image.per_image_standardization(image)
    
    return image

# Method 5: MixUp augmentation
print("\nMethod 5: MixUp Augmentation")
def mixup(image1, label1, image2, label2, alpha=0.2):
    """Mix two images and their labels."""
    lambda_param = np.random.beta(alpha, alpha)
    mixed_image = lambda_param * image1 + (1 - lambda_param) * image2
    mixed_label = lambda_param * label1 + (1 - lambda_param) * label2
    return mixed_image, mixed_label

# Example usage
idx1, idx2 = 0, 1
label1 = tf.one_hot(y_train[idx1], 10)
label2 = tf.one_hot(y_train[idx2], 10)
mixed_img, mixed_label = mixup(X_train[idx1], label1, X_train[idx2], label2)
print(f"Mixed image shape: {mixed_img.shape}")

# Method 6: CutMix augmentation
print("\nMethod 6: CutMix Augmentation")
def cutmix(image1, label1, image2, label2, alpha=1.0):
    """Apply CutMix augmentation."""
    h, w = image1.shape[0], image1.shape[1]
    lambda_param = np.random.beta(alpha, alpha)
    
    # Random box
    cut_ratio = np.sqrt(1.0 - lambda_param)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Cut and mix
    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    
    # Mix labels
    lambda_param = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
    mixed_label = lambda_param * label1 + (1 - lambda_param) * label2
    
    return mixed_image, mixed_label

print("\nData augmentation helps prevent overfitting and improves generalization!")
