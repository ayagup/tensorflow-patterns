"""
Custom Loss Function Pattern
Create custom loss functions for specialized training objectives.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("Custom Loss Function Pattern\n")

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

# Example 1: Simple Custom Loss
print("Example 1: Simple Custom Loss Function")

def custom_mse(y_true, y_pred):
    """Custom mean squared error."""
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=custom_mse, metrics=['mae'])
print("Custom MSE loss function defined")

# Example 2: Weighted Binary Cross-Entropy
print("\nExample 2: Weighted Binary Cross-Entropy")

def weighted_binary_crossentropy(pos_weight=2.0):
    """Binary cross-entropy with class weighting."""
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate weighted loss
        bce = -y_true * tf.math.log(y_pred) * pos_weight - \
              (1 - y_true) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(bce)
    return loss

model_weighted = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

model_weighted.compile(
    optimizer='adam',
    loss=weighted_binary_crossentropy(pos_weight=2.0),
    metrics=['accuracy']
)

print("Weighted BCE emphasizes positive class")

# Example 3: Focal Loss
print("\nExample 3: Focal Loss for Imbalanced Classes")

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss focuses on hard examples.
    Reduces loss contribution from easy examples.
    """
    def loss(y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        
        focal = weight * cross_entropy
        return tf.reduce_mean(focal)
    return loss

model_focal = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

model_focal.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

print("Focal loss handles class imbalance")

# Example 4: Dice Loss for Segmentation
print("\nExample 4: Dice Loss for Segmentation")

def dice_loss(smooth=1.0):
    """Dice loss for segmentation tasks."""
    def loss(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )
        
        return 1 - dice_coef
    return loss

print("Dice loss maximizes overlap between prediction and ground truth")

# Example 5: Contrastive Loss
print("\nExample 5: Contrastive Loss for Similarity Learning")

def contrastive_loss(margin=1.0):
    """Contrastive loss for learning embeddings."""
    def loss(y_true, y_pred):
        # y_true: 1 for similar, 0 for dissimilar
        # y_pred: distance between embeddings
        
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        
        loss = y_true * square_pred + (1 - y_true) * margin_square
        return tf.reduce_mean(loss)
    return loss

print("Contrastive loss pulls similar pairs together, pushes dissimilar apart")

# Example 6: Triplet Loss
print("\nExample 6: Triplet Loss")

def triplet_loss(margin=1.0):
    """Triplet loss for metric learning."""
    def loss(y_true, y_pred):
        # y_pred contains [anchor, positive, negative] embeddings
        anchor, positive, negative = tf.split(y_pred, 3, axis=1)
        
        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Triplet loss
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.maximum(basic_loss, 0.0)
        
        return tf.reduce_mean(loss)
    return loss

print("Triplet loss: anchor should be closer to positive than negative")

# Example 7: Huber Loss
print("\nExample 7: Huber Loss (Smooth L1)")

def huber_loss(delta=1.0):
    """Huber loss, less sensitive to outliers than MSE."""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * tf.square(quadratic) + delta * linear
        return tf.reduce_mean(loss)
    return loss

model_huber = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1)
])

model_huber.compile(optimizer='adam', loss=huber_loss(delta=1.0))
print("Huber loss combines MSE and MAE benefits")

# Example 8: Custom Loss with Regularization
print("\nExample 8: Loss with Custom Regularization")

def loss_with_l2_regularization(l2_lambda=0.01):
    """Custom loss with L2 regularization on weights."""
    def loss(y_true, y_pred):
        # Base loss
        base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # L2 regularization (would need model reference)
        # This is simplified; in practice, get weights from model
        return tf.reduce_mean(base_loss)
    return loss

print("Can combine multiple loss components")

# Example 9: Multi-Task Loss
print("\nExample 9: Multi-Task Custom Loss")

def multi_task_loss(task1_weight=1.0, task2_weight=1.0):
    """Combine losses from multiple tasks."""
    def loss(y_true, y_pred):
        # Split predictions and targets
        task1_true, task2_true = tf.split(y_true, 2, axis=1)
        task1_pred, task2_pred = tf.split(y_pred, 2, axis=1)
        
        # Task 1 loss (classification)
        task1_loss = tf.keras.losses.binary_crossentropy(task1_true, task1_pred)
        
        # Task 2 loss (regression)
        task2_loss = tf.keras.losses.mse(task2_true, task2_pred)
        
        # Weighted combination
        combined_loss = task1_weight * task1_loss + task2_weight * task2_loss
        return tf.reduce_mean(combined_loss)
    return loss

print("Multi-task loss balances multiple objectives")

# Example 10: Custom Loss Class
print("\nExample 10: Loss as a Class")

class CustomLossClass(keras.losses.Loss):
    """Custom loss implemented as a class."""
    def __init__(self, alpha=0.5, beta=0.5, name='custom_loss'):
        super(CustomLossClass, self).__init__(name=name)
        self.alpha = alpha
        self.beta = beta
    
    def call(self, y_true, y_pred):
        # Component 1: MSE
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Component 2: MAE
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Weighted combination
        return self.alpha * mse + self.beta * mae
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta
        })
        return config

custom_loss_obj = CustomLossClass(alpha=0.7, beta=0.3)

model_class = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1)
])

model_class.compile(optimizer='adam', loss=custom_loss_obj)
print("Class-based loss is more flexible and configurable")

# Example 11: Training with Custom Loss
print("\nExample 11: Training with Custom Loss")

# Use focal loss for imbalanced data
model_train = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_train.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

history = model_train.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 12: Loss Combination
print("\nExample 12: Combining Multiple Loss Functions")

def combined_loss(y_true, y_pred):
    """Combine multiple loss functions."""
    # Binary cross-entropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Mean squared error
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Mean absolute error
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Weighted combination
    return 0.5 * bce + 0.3 * mse + 0.2 * mae

print("Combined losses can leverage multiple objectives")

print("\nCustom Loss Function Best Practices:")
print("""
1. Return scalar loss value per sample
2. Use tf operations for GPU acceleration
3. Clip predictions to avoid numerical instability
4. Add smooth parameter for division operations
5. Make loss configurable with parameters
6. Test loss function on simple data first
7. Use reduction='none' for sample weighting

Common Custom Losses:

- Focal Loss: Imbalanced classification
- Dice Loss: Segmentation tasks
- Triplet Loss: Metric learning
- Contrastive Loss: Similarity learning
- Huber Loss: Robust regression
- Weighted BCE: Class imbalance
- Custom combinations: Multi-objective

Implementation Tips:

1. Function-based: Simple, quick prototyping
2. Class-based: Configurable, reusable
3. Closure: When parameters needed
4. Always test gradient flow
5. Use @tf.function for performance
6. Consider numerical stability
7. Document expected input/output shapes
""")
