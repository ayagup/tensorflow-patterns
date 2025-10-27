"""
Custom Loss Functions Pattern

This module demonstrates how to create custom loss functions in TensorFlow
for various use cases beyond standard losses.

Patterns covered:
1. Basic Custom Loss Function
2. Weighted Loss
3. Focal Loss for Imbalanced Classification
4. Dice Loss for Segmentation
5. IoU Loss for Object Detection
6. Triplet Loss for Similarity Learning
7. Contrastive Loss for Siamese Networks
8. Perceptual Loss for Style Transfer
9. Combined/Composite Loss
10. Loss with Regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Basic Custom Loss Function
def custom_mse_loss(y_true, y_pred):
    """Custom Mean Squared Error loss."""
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


class CustomMSELoss(keras.losses.Loss):
    """Custom MSE loss as a class."""
    
    def __init__(self, name='custom_mse_loss'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference)


def example_custom_loss():
    """Example: Basic custom loss function."""
    # Create simple model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(1)
    ])
    
    # Compile with custom loss
    model.compile(optimizer='adam', loss=CustomMSELoss())
    
    # Train
    X = np.random.randn(100, 10)
    y = np.random.randn(100, 1)
    
    print("Custom Loss Function Example:")
    history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
    return model


# Pattern 2: Weighted Loss
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Weighted binary cross-entropy for imbalanced classes."""
    
    def __init__(self, pos_weight=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Weighted binary cross-entropy
        bce = -y_true * tf.math.log(y_pred) * self.pos_weight - \
              (1 - y_true) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(bce)


def example_weighted_loss():
    """Example: Weighted loss for imbalanced data."""
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Higher weight for positive class (minority)
    model.compile(
        optimizer='adam',
        loss=WeightedBinaryCrossentropy(pos_weight=3.0),
        metrics=['accuracy']
    )
    
    # Imbalanced data: 90% negative, 10% positive
    X = np.random.randn(1000, 10)
    y = np.random.choice([0, 1], size=(1000, 1), p=[0.9, 0.1])
    
    print("\nWeighted Loss Example:")
    print(f"Class distribution: 0s={np.sum(y==0)}, 1s={np.sum(y==1)}")
    history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model


# Pattern 3: Focal Loss
class FocalLoss(keras.losses.Loss):
    """Focal loss for addressing class imbalance in classification."""
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Focal weight: (1 - p_t)^gamma
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Focal loss
        focal_loss = self.alpha * focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


def example_focal_loss():
    """Example: Focal loss for hard examples."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss=FocalLoss(alpha=0.25, gamma=2.0),
        metrics=['accuracy']
    )
    
    # Data with hard examples
    X = np.random.randn(1000, 10)
    y = keras.utils.to_categorical(np.random.randint(0, 3, 1000))
    
    print("\nFocal Loss Example:")
    history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print("Focuses on hard-to-classify examples")
    
    return model


# Pattern 4: Dice Loss
class DiceLoss(keras.losses.Loss):
    """Dice loss for image segmentation."""
    
    def __init__(self, smooth=1.0, name='dice_loss'):
        super().__init__(name=name)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss (1 - dice)
        return 1.0 - dice


def example_dice_loss():
    """Example: Dice loss for segmentation."""
    # Simple segmentation model
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(64, 64, 1)),
        layers.Conv2D(1, 1, activation='sigmoid', padding='same')
    ])
    
    model.compile(optimizer='adam', loss=DiceLoss())
    
    # Dummy segmentation data
    X = np.random.rand(10, 64, 64, 1).astype(np.float32)
    y = np.random.randint(0, 2, (10, 64, 64, 1)).astype(np.float32)
    
    print("\nDice Loss Example:")
    history = model.fit(X, y, epochs=2, batch_size=2, verbose=0)
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print("Commonly used for medical image segmentation")
    
    return model


# Pattern 5: IoU Loss
class IoULoss(keras.losses.Loss):
    """Intersection over Union loss for bounding boxes."""
    
    def __init__(self, name='iou_loss'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        # y_true, y_pred: (batch, 4) with [x1, y1, x2, y2]
        
        # Intersection coordinates
        x1_inter = tf.maximum(y_true[:, 0], y_pred[:, 0])
        y1_inter = tf.maximum(y_true[:, 1], y_pred[:, 1])
        x2_inter = tf.minimum(y_true[:, 2], y_pred[:, 2])
        y2_inter = tf.minimum(y_true[:, 3], y_pred[:, 3])
        
        # Intersection area
        inter_width = tf.maximum(0.0, x2_inter - x1_inter)
        inter_height = tf.maximum(0.0, y2_inter - y1_inter)
        intersection = inter_width * inter_height
        
        # Union area
        area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
        area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
        union = area_true + area_pred - intersection
        
        # IoU
        iou = intersection / (union + 1e-7)
        
        # IoU loss
        return 1.0 - tf.reduce_mean(iou)


def example_iou_loss():
    """Example: IoU loss for bounding box regression."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(4)  # [x1, y1, x2, y2]
    ])
    
    model.compile(optimizer='adam', loss=IoULoss())
    
    # Dummy bounding box data
    X = np.random.randn(100, 10)
    y = np.random.rand(100, 4)  # Boxes in [0, 1]
    
    print("\nIoU Loss Example:")
    history = model.fit(X, y, epochs=3, batch_size=16, verbose=0)
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print("Better than MSE for bounding box regression")
    
    return model


# Pattern 6: Triplet Loss
class TripletLoss(keras.losses.Loss):
    """Triplet loss for similarity learning."""
    
    def __init__(self, margin=1.0, name='triplet_loss'):
        super().__init__(name=name)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        # y_pred contains [anchor, positive, negative] embeddings
        # Shape: (batch, 3, embedding_dim)
        
        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        negative = y_pred[:, 2, :]
        
        # Distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # Triplet loss
        loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)
        
        return tf.reduce_mean(loss)


def example_triplet_loss():
    """Example: Triplet loss for face recognition."""
    # Embedding network
    embedding_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(10,)),
        layers.Dense(64)  # Embedding dimension
    ])
    
    # Triplet model (processes anchor, positive, negative)
    anchor_input = keras.Input(shape=(10,))
    positive_input = keras.Input(shape=(10,))
    negative_input = keras.Input(shape=(10,))
    
    anchor_emb = embedding_model(anchor_input)
    positive_emb = embedding_model(positive_input)
    negative_emb = embedding_model(negative_input)
    
    # Stack embeddings
    embeddings = layers.Concatenate(axis=1)([
        tf.expand_dims(anchor_emb, 1),
        tf.expand_dims(positive_emb, 1),
        tf.expand_dims(negative_emb, 1)
    ])
    
    model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=embeddings
    )
    
    model.compile(optimizer='adam', loss=TripletLoss(margin=1.0))
    
    print("\nTriplet Loss Example:")
    print("Used for face recognition, person re-identification")
    print("Learns embeddings where similar items are close")
    
    return model


# Pattern 7: Contrastive Loss
class ContrastiveLoss(keras.losses.Loss):
    """Contrastive loss for Siamese networks."""
    
    def __init__(self, margin=1.0, name='contrastive_loss'):
        super().__init__(name=name)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        # y_true: 1 for similar pairs, 0 for dissimilar
        # y_pred: distance between embeddings
        
        # Loss for similar pairs
        similar_loss = y_true * tf.square(y_pred)
        
        # Loss for dissimilar pairs
        dissimilar_loss = (1 - y_true) * tf.square(
            tf.maximum(0.0, self.margin - y_pred)
        )
        
        return tf.reduce_mean(similar_loss + dissimilar_loss)


def example_contrastive_loss():
    """Example: Contrastive loss for similarity learning."""
    print("\nContrastive Loss Example:")
    print("Used in Siamese networks")
    print("Minimizes distance for similar pairs")
    print("Maximizes distance for dissimilar pairs")
    
    # Conceptual example
    loss_fn = ContrastiveLoss(margin=1.0)
    
    # Simulated embeddings
    similar_distance = tf.constant([0.2, 0.3])
    dissimilar_distance = tf.constant([0.8, 0.9])
    
    y_true_similar = tf.constant([1.0, 1.0])
    y_true_dissimilar = tf.constant([0.0, 0.0])
    
    loss_similar = loss_fn(y_true_similar, similar_distance)
    loss_dissimilar = loss_fn(y_true_dissimilar, dissimilar_distance)
    
    print(f"Loss for similar pairs: {loss_similar.numpy():.4f}")
    print(f"Loss for dissimilar pairs: {loss_dissimilar.numpy():.4f}")
    
    return loss_fn


# Pattern 8: Perceptual Loss
class PerceptualLoss(keras.losses.Loss):
    """Perceptual loss using pretrained VGG features."""
    
    def __init__(self, feature_layers=['block3_conv3'], name='perceptual_loss'):
        super().__init__(name=name)
        
        # Load VGG19 and extract intermediate layers
        vgg = keras.applications.VGG19(include_top=False, weights=None)
        outputs = [vgg.get_layer(name).output for name in feature_layers]
        self.feature_extractor = keras.Model(vgg.input, outputs)
        self.feature_extractor.trainable = False
    
    def call(self, y_true, y_pred):
        # Extract features
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        
        # MSE on features
        if isinstance(true_features, list):
            loss = sum([
                tf.reduce_mean(tf.square(t - p))
                for t, p in zip(true_features, pred_features)
            ])
        else:
            loss = tf.reduce_mean(tf.square(true_features - pred_features))
        
        return loss


def example_perceptual_loss():
    """Example: Perceptual loss for style transfer."""
    print("\nPerceptual Loss Example:")
    print("Uses features from pretrained network")
    print("Better captures perceptual similarity than pixel-wise MSE")
    print("Used in style transfer, super-resolution")
    
    # Note: Would require pretrained weights in practice
    loss_fn = PerceptualLoss(feature_layers=['block3_conv3'])
    
    return loss_fn


# Pattern 9: Combined Loss
class CombinedLoss(keras.losses.Loss):
    """Combination of multiple losses."""
    
    def __init__(self, losses, weights=None, name='combined_loss'):
        super().__init__(name=name)
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)
    
    def call(self, y_true, y_pred):
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(y_true, y_pred)
        
        return total_loss


def example_combined_loss():
    """Example: Combined loss for multi-objective optimization."""
    # Combine BCE and Dice loss for segmentation
    combined_loss = CombinedLoss(
        losses=[
            keras.losses.BinaryCrossentropy(),
            DiceLoss()
        ],
        weights=[0.5, 0.5]
    )
    
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(64, 64, 1)),
        layers.Conv2D(1, 1, activation='sigmoid', padding='same')
    ])
    
    model.compile(optimizer='adam', loss=combined_loss)
    
    print("\nCombined Loss Example:")
    print("Combines BCE (0.5) + Dice (0.5)")
    print("Leverages strengths of multiple losses")
    
    return model


# Pattern 10: Loss with Regularization
class LossWithRegularization(keras.losses.Loss):
    """Loss function with custom regularization."""
    
    def __init__(self, base_loss, l1_weight=0.01, l2_weight=0.01, name='loss_with_reg'):
        super().__init__(name=name)
        self.base_loss = base_loss
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def call(self, y_true, y_pred):
        # Base loss
        loss = self.base_loss(y_true, y_pred)
        
        # Add regularization (would need model weights in practice)
        # This is a simplified example
        return loss


def example_loss_with_regularization():
    """Example: Custom loss with regularization."""
    loss_fn = LossWithRegularization(
        base_loss=keras.losses.MeanSquaredError(),
        l1_weight=0.01,
        l2_weight=0.01
    )
    
    print("\nLoss with Regularization Example:")
    print("Combines task loss with regularization")
    print("L1: promotes sparsity")
    print("L2: prevents large weights")
    
    return loss_fn


if __name__ == "__main__":
    print("Custom Loss Functions Pattern\n" + "="*60)
    
    # Example 1: Basic Custom Loss
    print("\n1. Basic Custom Loss Function")
    model1 = example_custom_loss()
    
    # Example 2: Weighted Loss
    print("\n2. Weighted Loss for Imbalanced Data")
    model2 = example_weighted_loss()
    
    # Example 3: Focal Loss
    print("\n3. Focal Loss")
    model3 = example_focal_loss()
    
    # Example 4: Dice Loss
    print("\n4. Dice Loss for Segmentation")
    model4 = example_dice_loss()
    
    # Example 5: IoU Loss
    print("\n5. IoU Loss for Object Detection")
    model5 = example_iou_loss()
    
    # Example 6: Triplet Loss
    print("\n6. Triplet Loss for Similarity Learning")
    model6 = example_triplet_loss()
    
    # Example 7: Contrastive Loss
    print("\n7. Contrastive Loss")
    loss7 = example_contrastive_loss()
    
    # Example 8: Perceptual Loss
    print("\n8. Perceptual Loss")
    loss8 = example_perceptual_loss()
    
    # Example 9: Combined Loss
    print("\n9. Combined Loss")
    model9 = example_combined_loss()
    
    # Example 10: Loss with Regularization
    print("\n10. Loss with Regularization")
    loss10 = example_loss_with_regularization()
    
    print("\n" + "="*60)
    print("Custom Loss Best Practices:")
    print("1. Inherit from keras.losses.Loss for proper integration")
    print("2. Use tf operations for GPU acceleration")
    print("3. Add small epsilon to prevent division by zero")
    print("4. Clip predictions to avoid log(0)")
    print("5. Test loss function with known inputs")
    print("6. Consider numerical stability")
    print("7. Use reduction='none' for per-sample losses")
    print("8. Combine losses with appropriate weights")
    print("9. Monitor loss values during training")
    print("10. Document loss function purpose and parameters")
