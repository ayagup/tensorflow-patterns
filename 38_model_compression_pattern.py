"""
Model Compression and Optimization Patterns

This module demonstrates techniques for compressing and optimizing neural
networks for deployment on resource-constrained devices.

Patterns covered:
1. Weight Pruning
2. Structured Pruning
3. Post-Training Quantization
4. Quantization-Aware Training
5. Knowledge Distillation
6. Neural Network Compression
7. Low-Rank Factorization
8. Dynamic Quantization
9. Sparse Models
10. Mobile-Optimized Architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_model_optimization as tfmot


# Pattern 1: Weight Pruning (Magnitude-based)
def create_pruned_model(base_model, target_sparsity=0.5):
    """Apply magnitude-based pruning to model."""
    # Pruning schedule
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=target_sparsity,
        begin_step=0,
        end_step=1000
    )
    
    # Apply pruning
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model,
        pruning_schedule=pruning_schedule
    )
    
    return pruned_model


def example_weight_pruning():
    """Example: Weight pruning."""
    # Base model
    base_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Note: tfmot may not be installed, so we'll show the pattern
    print("Weight Pruning Example:")
    print("Removes weights with smallest magnitudes")
    print(f"Target sparsity: 50% (half of weights set to zero)")
    print("\nBase model:")
    base_model.summary()
    
    # In practice, you would:
    # pruned_model = create_pruned_model(base_model, target_sparsity=0.5)
    # pruned_model.compile(...)
    # pruned_model.fit(...)
    # final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    return base_model


# Pattern 2: Structured Pruning (Manual Implementation)
class StructuredPruningLayer(layers.Layer):
    """Layer with structured pruning (prune entire neurons)."""
    
    def __init__(self, units, pruning_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.pruning_ratio = pruning_ratio
        self.active_units = int(units * (1 - pruning_ratio))
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        # Importance scores for neurons
        self.importance = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            trainable=True,
            name='importance'
        )
    
    def call(self, inputs):
        # Compute neuron importance (L1 norm of weights)
        importance = tf.reduce_sum(tf.abs(self.kernel), axis=0)
        
        # Get top-k neurons
        _, top_indices = tf.nn.top_k(importance, k=self.active_units)
        
        # Create mask
        mask = tf.scatter_nd(
            tf.expand_dims(top_indices, 1),
            tf.ones(self.active_units),
            [self.units]
        )
        
        # Apply mask to weights
        masked_kernel = self.kernel * mask
        masked_bias = self.bias * mask
        
        output = tf.matmul(inputs, masked_kernel) + masked_bias
        return tf.nn.relu(output)


def example_structured_pruning():
    """Example: Structured pruning."""
    model = keras.Sequential([
        StructuredPruningLayer(128, pruning_ratio=0.3, input_shape=(784,)),
        StructuredPruningLayer(64, pruning_ratio=0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    print("\nStructured Pruning Example:")
    print("Prunes entire neurons/filters, not individual weights")
    print(f"Pruning ratio: 30% of neurons removed")
    model.summary()
    
    return model


# Pattern 3: Post-Training Quantization
def quantize_model_post_training(model, representative_dataset):
    """Apply post-training quantization."""
    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Full integer quantization (if representative dataset provided)
    if representative_dataset is not None:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    quantized_model = converter.convert()
    
    return quantized_model


def example_post_training_quantization():
    """Example: Post-training quantization."""
    # Create model
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    print("\nPost-Training Quantization Example:")
    print("Converts FP32 weights to INT8 after training")
    print("Reduces model size by ~4x")
    print("\nOriginal model:")
    model.summary()
    
    # Representative dataset generator
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 28, 28, 1).astype(np.float32)
            yield [data]
    
    # In practice:
    # quantized_tflite = quantize_model_post_training(model, representative_dataset)
    # with open('model_quantized.tflite', 'wb') as f:
    #     f.write(quantized_tflite)
    
    return model


# Pattern 4: Quantization-Aware Training
def create_qat_model():
    """Create model with quantization-aware training."""
    # Base model
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    # Apply quantization-aware training
    # qat_model = tfmot.quantization.keras.quantize_model(model)
    
    print("\nQuantization-Aware Training (QAT) Example:")
    print("Simulates quantization during training for better accuracy")
    print("Results in INT8 model with minimal accuracy loss")
    model.summary()
    
    return model


# Pattern 5: Knowledge Distillation
class Distiller(keras.Model):
    """Knowledge distillation trainer."""
    
    def __init__(self, student, teacher, temperature=3.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        self.teacher.trainable = False
    
    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
    
    def train_step(self, data):
        x, y = data
        
        # Forward pass through teacher
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass through student
            student_predictions = self.student(x, training=True)
            
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Distillation loss (soft targets)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature),
                tf.nn.softmax(student_predictions / self.temperature)
            )
            
            # Combined loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
        return results
    
    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def example_knowledge_distillation():
    """Example: Knowledge distillation."""
    # Teacher model (large)
    teacher = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Student model (small)
    student = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    
    # Distiller
    distiller = Distiller(student, teacher, temperature=3.0, alpha=0.5)
    
    print("\nKnowledge Distillation Example:")
    print("Teacher model (large):")
    teacher.summary()
    print("\nStudent model (small):")
    student.summary()
    print("\nStudent learns from teacher's soft predictions")
    
    return distiller


# Pattern 6: Low-Rank Factorization
class LowRankDense(layers.Layer):
    """Dense layer with low-rank factorization."""
    
    def __init__(self, units, rank_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank_ratio = rank_ratio
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        rank = int(min(input_dim, self.units) * self.rank_ratio)
        
        # Factorize W = U * V
        self.U = self.add_weight(
            shape=(input_dim, rank),
            initializer='glorot_uniform',
            trainable=True,
            name='U'
        )
        
        self.V = self.add_weight(
            shape=(rank, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='V'
        )
        
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        # Two matrix multiplications instead of one
        output = tf.matmul(inputs, self.U)
        output = tf.matmul(output, self.V)
        return output + self.bias


def example_low_rank_factorization():
    """Example: Low-rank factorization."""
    model = keras.Sequential([
        LowRankDense(128, rank_ratio=0.5, input_shape=(784,)),
        layers.Activation('relu'),
        LowRankDense(64, rank_ratio=0.5),
        layers.Activation('relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    print("\nLow-Rank Factorization Example:")
    print("Decomposes weight matrices into smaller factors")
    print("Rank ratio: 0.5 (50% of full rank)")
    model.summary()
    
    # Parameter comparison
    original_params = 784 * 128 + 128 * 64
    factorized_params = (784 * 64 + 64 * 128) + (128 * 32 + 32 * 64)
    print(f"\nOriginal parameters: {original_params:,}")
    print(f"Factorized parameters: {factorized_params:,}")
    print(f"Compression ratio: {original_params / factorized_params:.2f}x")
    
    return model


# Pattern 7: Sparse Model Training
class SparseLayer(layers.Layer):
    """Layer with sparse activations."""
    
    def __init__(self, units, sparsity_ratio=0.7, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sparsity_ratio = sparsity_ratio
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel) + self.bias
        
        # Apply ReLU
        activation = tf.nn.relu(pre_activation)
        
        # Keep only top-k activations
        k = int(self.units * (1 - self.sparsity_ratio))
        values, indices = tf.nn.top_k(activation, k=k)
        
        # Create sparse output
        sparse_output = tf.scatter_nd(
            indices=tf.expand_dims(indices, -1),
            updates=values,
            shape=tf.shape(activation)
        )
        
        return sparse_output


def example_sparse_model():
    """Example: Sparse model training."""
    model = keras.Sequential([
        SparseLayer(128, sparsity_ratio=0.7, input_shape=(784,)),
        SparseLayer(64, sparsity_ratio=0.7),
        layers.Dense(10, activation='softmax')
    ])
    
    print("\nSparse Model Example:")
    print("Only 30% of neurons are active (70% sparse)")
    print("Reduces computation during inference")
    model.summary()
    
    return model


# Pattern 8: Mobile-Optimized Architecture (MobileNet-style)
class DepthwiseSeparableConv(layers.Layer):
    """Depthwise separable convolution."""
    
    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            padding='same'
        )
        self.pointwise = layers.Conv2D(filters, 1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        x = self.depthwise(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        
        return x


def create_mobile_optimized_model():
    """Create mobile-optimized model."""
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Standard conv
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Depthwise separable convs
    x = DepthwiseSeparableConv(64, stride=1)(x)
    x = DepthwiseSeparableConv(128, stride=2)(x)
    x = DepthwiseSeparableConv(128, stride=1)(x)
    x = DepthwiseSeparableConv(256, stride=2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1000, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def example_mobile_optimized():
    """Example: Mobile-optimized architecture."""
    model = create_mobile_optimized_model()
    
    print("\nMobile-Optimized Architecture (MobileNet-style):")
    print("Uses depthwise separable convolutions")
    print("Reduces parameters and computation")
    model.summary()
    
    # Parameter comparison
    # Standard conv: kernel_size^2 * in_channels * out_channels
    # Depthwise separable: kernel_size^2 * in_channels + in_channels * out_channels
    print("\nCompression example for 3x3 conv, 128->128 channels:")
    standard = 3 * 3 * 128 * 128
    depthwise_sep = 3 * 3 * 128 + 128 * 128
    print(f"Standard conv: {standard:,} parameters")
    print(f"Depthwise separable: {depthwise_sep:,} parameters")
    print(f"Reduction: {standard / depthwise_sep:.2f}x")
    
    return model


# Pattern 9: Model Size Analysis
def analyze_model_size(model):
    """Analyze model size and complexity."""
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Estimate model size (FP32)
    size_mb = total_params * 4 / (1024 * 1024)
    
    print(f"\nModel Size Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Estimated size (FP32): {size_mb:.2f} MB")
    print(f"Estimated size (INT8): {size_mb / 4:.2f} MB")
    
    return {
        'total_params': total_params,
        'size_fp32_mb': size_mb,
        'size_int8_mb': size_mb / 4
    }


def example_model_size_analysis():
    """Example: Model size analysis."""
    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(784,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    print("\nModel Size Analysis Example:")
    stats = analyze_model_size(model)
    
    return stats


if __name__ == "__main__":
    print("Model Compression and Optimization Patterns\n" + "="*60)
    
    # Example 1: Weight Pruning
    print("\n1. Weight Pruning (Magnitude-based)")
    model1 = example_weight_pruning()
    
    # Example 2: Structured Pruning
    print("\n2. Structured Pruning")
    model2 = example_structured_pruning()
    
    # Example 3: Post-Training Quantization
    print("\n3. Post-Training Quantization")
    model3 = example_post_training_quantization()
    
    # Example 4: QAT
    print("\n4. Quantization-Aware Training")
    model4 = create_qat_model()
    
    # Example 5: Knowledge Distillation
    print("\n5. Knowledge Distillation")
    distiller5 = example_knowledge_distillation()
    
    # Example 6: Low-Rank Factorization
    print("\n6. Low-Rank Factorization")
    model6 = example_low_rank_factorization()
    
    # Example 7: Sparse Models
    print("\n7. Sparse Model Training")
    model7 = example_sparse_model()
    
    # Example 8: Mobile-Optimized
    print("\n8. Mobile-Optimized Architecture")
    model8 = example_mobile_optimized()
    
    # Example 9: Size Analysis
    print("\n9. Model Size Analysis")
    stats9 = example_model_size_analysis()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Pruning: Start with magnitude-based, then try structured")
    print("2. Quantization: Post-training for quick wins, QAT for best accuracy")
    print("3. Knowledge distillation: Great for model compression")
    print("4. Low-rank: Good for fully-connected layers")
    print("5. Depthwise separable: Essential for mobile models")
    print("6. Combine techniques for maximum compression")
    print("7. Always validate accuracy after compression")
    print("8. Profile model on target device")
    print("9. Consider inference time, not just model size")
    print("10. Use TFLite for mobile/edge deployment")
