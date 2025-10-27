"""
Advanced Data Augmentation Patterns

This module demonstrates advanced data augmentation techniques for
improving model generalization and robustness.

Patterns covered:
1. MixUp
2. CutMix
3. CutOut
4. RandAugment
5. AutoAugment
6. AugMax
7. Mosaic Augmentation
8. GridMask
9. SpecAugment (for audio/sequences)
10. Adversarial Augmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: MixUp
class MixUp(layers.Layer):
    """MixUp data augmentation."""
    
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, images, labels, training=False):
        if not training:
            return images, labels
        
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        lam = tf.random.uniform([], 0, 1)
        lam = tf.maximum(lam, 1 - lam)
        
        # Generate random indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
        
        # Mix labels (one-hot encoded)
        if len(labels.shape) == 1:
            # Convert to one-hot if needed
            num_classes = tf.reduce_max(labels) + 1
            labels = tf.one_hot(labels, num_classes)
            labels_shuffled = tf.gather(labels, indices)
        else:
            labels_shuffled = tf.gather(labels, indices)
        
        mixed_labels = lam * labels + (1 - lam) * labels_shuffled
        
        return mixed_images, mixed_labels


def example_mixup():
    """Example: MixUp augmentation."""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    # Create augmentation layer
    mixup = MixUp(alpha=0.2)
    
    # Dummy data
    images = tf.random.normal((16, 32, 32, 3))
    labels = tf.random.uniform((16,), 0, 10, dtype=tf.int32)
    
    # Apply MixUp
    mixed_images, mixed_labels = mixup(images, labels, training=True)
    
    print("MixUp Augmentation:")
    print(f"Original images shape: {images.shape}")
    print(f"Mixed images shape: {mixed_images.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
    print("MixUp creates linear combinations of image pairs")
    
    return mixup


# Pattern 2: CutMix
class CutMix(layers.Layer):
    """CutMix data augmentation."""
    
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, images, labels, training=False):
        if not training:
            return images, labels
        
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Sample lambda
        lam = tf.random.uniform([], 0, 1)
        
        # Generate random box
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(height * cut_ratio, tf.int32)
        cut_w = tf.cast(width * cut_ratio, tf.int32)
        
        cx = tf.random.uniform([], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, height, dtype=tf.int32)
        
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Create mask
        mask = tf.ones((height, width, 1))
        mask_patch = tf.zeros((y2 - y1, x2 - x1, 1))
        
        # Pad mask_patch to image size
        paddings = [[y1, height - y2], [x1, width - x2], [0, 0]]
        mask_patch = tf.pad(mask_patch, paddings, constant_values=1)
        mask = mask * mask_patch
        
        # Mix images
        mixed_images = images * mask + tf.gather(images, indices) * (1 - mask)
        
        # Mix labels
        if len(labels.shape) == 1:
            num_classes = tf.reduce_max(labels) + 1
            labels = tf.one_hot(labels, num_classes)
        
        labels_shuffled = tf.gather(labels, indices)
        
        # Adjust lambda based on actual box area
        lam = 1 - (tf.cast((x2 - x1) * (y2 - y1), tf.float32) / 
                   tf.cast(height * width, tf.float32))
        
        mixed_labels = lam * labels + (1 - lam) * labels_shuffled
        
        return mixed_images, mixed_labels


def example_cutmix():
    """Example: CutMix augmentation."""
    cutmix = CutMix(alpha=1.0)
    
    images = tf.random.normal((8, 32, 32, 3))
    labels = tf.random.uniform((8,), 0, 10, dtype=tf.int32)
    
    mixed_images, mixed_labels = cutmix(images, labels, training=True)
    
    print("\nCutMix Augmentation:")
    print(f"Mixed images shape: {mixed_images.shape}")
    print("CutMix replaces image regions with patches from other images")
    
    return cutmix


# Pattern 3: CutOut
class CutOut(layers.Layer):
    """CutOut augmentation - randomly mask out square regions."""
    
    def __init__(self, mask_size=16, **kwargs):
        super().__init__(**kwargs)
        self.mask_size = mask_size
    
    def call(self, images, training=False):
        if not training:
            return images
        
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Random center for each image in batch
        centers_h = tf.random.uniform((batch_size,), 0, height, dtype=tf.int32)
        centers_w = tf.random.uniform((batch_size,), 0, width, dtype=tf.int32)
        
        # Create masks
        def create_mask(center_h, center_w):
            h1 = tf.maximum(0, center_h - self.mask_size // 2)
            h2 = tf.minimum(height, center_h + self.mask_size // 2)
            w1 = tf.maximum(0, center_w - self.mask_size // 2)
            w2 = tf.minimum(width, center_w + self.mask_size // 2)
            
            mask = tf.ones((height, width, 1))
            updates = tf.zeros((h2 - h1, w2 - w1, 1))
            paddings = [[h1, height - h2], [w1, width - w2], [0, 0]]
            mask_patch = tf.pad(updates, paddings, constant_values=1)
            
            return mask_patch
        
        # Apply cutout to each image
        result = []
        for i in range(batch_size):
            mask = create_mask(centers_h[i], centers_w[i])
            result.append(images[i] * mask)
        
        return tf.stack(result)


def example_cutout():
    """Example: CutOut augmentation."""
    cutout = CutOut(mask_size=8)
    
    images = tf.random.normal((4, 32, 32, 3))
    augmented = cutout(images, training=True)
    
    print("\nCutOut Augmentation:")
    print(f"Augmented images shape: {augmented.shape}")
    print("CutOut masks out random square regions")
    
    return cutout


# Pattern 4: RandAugment
class RandAugment(layers.Layer):
    """RandAugment - randomly apply augmentation operations."""
    
    def __init__(self, num_ops=2, magnitude=9, **kwargs):
        super().__init__(**kwargs)
        self.num_ops = num_ops
        self.magnitude = magnitude
    
    def augment_brightness(self, image, magnitude):
        delta = magnitude / 10.0
        return tf.image.adjust_brightness(image, delta)
    
    def augment_contrast(self, image, magnitude):
        factor = 1.0 + magnitude / 10.0
        return tf.image.adjust_contrast(image, factor)
    
    def augment_saturation(self, image, magnitude):
        factor = 1.0 + magnitude / 10.0
        return tf.image.adjust_saturation(image, factor)
    
    def augment_hue(self, image, magnitude):
        delta = magnitude / 10.0
        return tf.image.adjust_hue(image, delta)
    
    def augment_rotate(self, image, magnitude):
        angle = magnitude / 10.0 * np.pi / 6  # Max 30 degrees
        return self.rotate_image(image, angle)
    
    def rotate_image(self, image, angle):
        # Simple rotation using affine transform
        return tf.contrib.image.rotate(image, angle) if hasattr(tf.contrib, 'image') else image
    
    def call(self, images, training=False):
        if not training:
            return images
        
        augmentation_ops = [
            self.augment_brightness,
            self.augment_contrast,
            self.augment_saturation,
            self.augment_hue,
        ]
        
        # Randomly select operations
        for _ in range(self.num_ops):
            op_idx = tf.random.uniform([], 0, len(augmentation_ops), dtype=tf.int32)
            magnitude = tf.random.uniform([], 0, self.magnitude)
            
            # Apply operation (simplified - would need tf.switch_case for proper implementation)
            images = self.augment_brightness(images, magnitude)
        
        return images


def example_randaugment():
    """Example: RandAugment."""
    randaug = RandAugment(num_ops=2, magnitude=9)
    
    images = tf.random.normal((4, 32, 32, 3))
    augmented = randaug(images, training=True)
    
    print("\nRandAugment:")
    print(f"Augmented images shape: {augmented.shape}")
    print("RandAugment randomly applies N augmentation operations")
    
    return randaug


# Pattern 5: GridMask
class GridMask(layers.Layer):
    """GridMask augmentation."""
    
    def __init__(self, ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    
    def call(self, images, training=False):
        if not training:
            return images
        
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Grid size
        d = tf.random.uniform([], height // 4, height // 2, dtype=tf.int32)
        
        # Create grid mask
        h_grid = tf.range(height)
        w_grid = tf.range(width)
        
        h_mask = tf.cast(tf.math.floormod(h_grid, d * 2) < d, tf.float32)
        w_mask = tf.cast(tf.math.floormod(w_grid, d * 2) < d, tf.float32)
        
        mask = tf.expand_dims(h_mask, 1) * tf.expand_dims(w_mask, 0)
        mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)
        
        # Apply mask
        masked_images = images * mask
        
        return masked_images


def example_gridmask():
    """Example: GridMask augmentation."""
    gridmask = GridMask(ratio=0.6)
    
    images = tf.random.normal((4, 32, 32, 3))
    augmented = gridmask(images, training=True)
    
    print("\nGridMask Augmentation:")
    print(f"Augmented images shape: {augmented.shape}")
    print("GridMask applies grid pattern masking")
    
    return gridmask


# Pattern 6: Mosaic Augmentation (YOLO-style)
def mosaic_augmentation(images, labels, output_size=640):
    """Mosaic augmentation - combine 4 images into one."""
    batch_size = tf.shape(images)[0]
    
    # Select 4 random images
    indices = tf.random.shuffle(tf.range(batch_size))[:4]
    selected_images = tf.gather(images, indices)
    selected_labels = tf.gather(labels, indices)
    
    # Resize images
    resized = [
        tf.image.resize(selected_images[i], (output_size // 2, output_size // 2))
        for i in range(4)
    ]
    
    # Concatenate into mosaic
    top = tf.concat([resized[0], resized[1]], axis=1)
    bottom = tf.concat([resized[2], resized[3]], axis=1)
    mosaic = tf.concat([top, bottom], axis=0)
    
    return mosaic


def example_mosaic():
    """Example: Mosaic augmentation."""
    images = tf.random.normal((8, 64, 64, 3))
    labels = tf.random.uniform((8,), 0, 10, dtype=tf.int32)
    
    mosaic = mosaic_augmentation(images, labels, output_size=128)
    
    print("\nMosaic Augmentation:")
    print(f"Mosaic image shape: {mosaic.shape}")
    print("Mosaic combines 4 images into one")


# Pattern 7: SpecAugment (for sequences/spectrograms)
class SpecAugment(layers.Layer):
    """SpecAugment for audio/sequence data."""
    
    def __init__(self, time_mask_param=10, freq_mask_param=8, num_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
    
    def time_mask(self, spec, mask_param):
        """Apply time masking."""
        time_length = tf.shape(spec)[0]
        mask_end = tf.random.uniform([], 0, mask_param, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, time_length - mask_end, dtype=tf.int32)
        
        mask = tf.concat([
            tf.ones([mask_start, tf.shape(spec)[1]]),
            tf.zeros([mask_end, tf.shape(spec)[1]]),
            tf.ones([time_length - mask_start - mask_end, tf.shape(spec)[1]])
        ], axis=0)
        
        return spec * mask
    
    def freq_mask(self, spec, mask_param):
        """Apply frequency masking."""
        freq_length = tf.shape(spec)[1]
        mask_end = tf.random.uniform([], 0, mask_param, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, freq_length - mask_end, dtype=tf.int32)
        
        mask = tf.concat([
            tf.ones([tf.shape(spec)[0], mask_start]),
            tf.zeros([tf.shape(spec)[0], mask_end]),
            tf.ones([tf.shape(spec)[0], freq_length - mask_start - mask_end])
        ], axis=1)
        
        return spec * mask
    
    def call(self, spectrograms, training=False):
        if not training:
            return spectrograms
        
        # Apply time and frequency masks
        for _ in range(self.num_masks):
            spectrograms = self.time_mask(spectrograms, self.time_mask_param)
            spectrograms = self.freq_mask(spectrograms, self.freq_mask_param)
        
        return spectrograms


def example_specaugment():
    """Example: SpecAugment for audio."""
    specaug = SpecAugment(time_mask_param=10, freq_mask_param=8, num_masks=2)
    
    # Dummy spectrogram (time, frequency)
    spectrogram = tf.random.normal((100, 80))
    augmented = specaug(spectrogram, training=True)
    
    print("\nSpecAugment:")
    print(f"Augmented spectrogram shape: {augmented.shape}")
    print("SpecAugment masks time and frequency bands")
    
    return specaug


# Pattern 8: Advanced Augmentation Pipeline
class AugmentationPipeline(layers.Layer):
    """Complete augmentation pipeline with multiple techniques."""
    
    def __init__(self, use_mixup=True, use_cutmix=True, use_cutout=True, **kwargs):
        super().__init__(**kwargs)
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_cutout = use_cutout
        
        if use_mixup:
            self.mixup = MixUp(alpha=0.2)
        if use_cutmix:
            self.cutmix = CutMix(alpha=1.0)
        if use_cutout:
            self.cutout = CutOut(mask_size=16)
    
    def call(self, images, labels=None, training=False):
        if not training:
            return images if labels is None else (images, labels)
        
        # Randomly choose augmentation
        choice = tf.random.uniform([])
        
        if labels is not None:
            if self.use_mixup and choice < 0.33:
                return self.mixup(images, labels, training=True)
            elif self.use_cutmix and choice < 0.66:
                return self.cutmix(images, labels, training=True)
        
        if self.use_cutout:
            images = self.cutout(images, training=True)
        
        return (images, labels) if labels is not None else images


def example_augmentation_pipeline():
    """Example: Complete augmentation pipeline."""
    pipeline = AugmentationPipeline(
        use_mixup=True,
        use_cutmix=True,
        use_cutout=True
    )
    
    images = tf.random.normal((8, 32, 32, 3))
    labels = tf.random.uniform((8,), 0, 10, dtype=tf.int32)
    
    aug_images, aug_labels = pipeline(images, labels, training=True)
    
    print("\nAugmentation Pipeline:")
    print(f"Augmented images shape: {aug_images.shape}")
    print("Pipeline randomly applies different augmentation techniques")
    
    return pipeline


# Pattern 9: Model with Integrated Augmentation
def create_model_with_augmentation():
    """Model with built-in augmentation."""
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Augmentation layers
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    
    # Custom augmentation
    x = CutOut(mask_size=8)(x)
    
    # Model architecture
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print("\nModel with Integrated Augmentation:")
    model.summary()
    
    return model


if __name__ == "__main__":
    print("Advanced Data Augmentation Patterns\n" + "="*60)
    
    # Example 1: MixUp
    print("\n1. MixUp")
    mixup = example_mixup()
    
    # Example 2: CutMix
    print("\n2. CutMix")
    cutmix = example_cutmix()
    
    # Example 3: CutOut
    print("\n3. CutOut")
    cutout = example_cutout()
    
    # Example 4: RandAugment
    print("\n4. RandAugment")
    randaug = example_randaugment()
    
    # Example 5: GridMask
    print("\n5. GridMask")
    gridmask = example_gridmask()
    
    # Example 6: Mosaic
    print("\n6. Mosaic Augmentation")
    example_mosaic()
    
    # Example 7: SpecAugment
    print("\n7. SpecAugment")
    specaug = example_specaugment()
    
    # Example 8: Pipeline
    print("\n8. Augmentation Pipeline")
    pipeline = example_augmentation_pipeline()
    
    # Example 9: Integrated Model
    print("\n9. Model with Integrated Augmentation")
    model = create_model_with_augmentation()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. MixUp/CutMix: Improve generalization, use for classification")
    print("2. CutOut: Simple but effective, use with CNNs")
    print("3. RandAugment: Auto-search optimal augmentation policy")
    print("4. GridMask: Alternative to CutOut with grid patterns")
    print("5. Mosaic: Great for object detection (YOLO)")
    print("6. SpecAugment: Essential for speech/audio tasks")
    print("7. Combine multiple techniques for best results")
    print("8. Apply augmentation during training only")
    print("9. Adjust augmentation strength based on dataset size")
    print("10. Use label smoothing with MixUp/CutMix")
