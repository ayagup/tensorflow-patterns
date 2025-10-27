# TensorFlow Patterns Collection

A comprehensive collection of TensorFlow/Keras design patterns and best practices. Each file demonstrates a specific pattern with working code examples.

## Overview

This repository contains 25+ complete, runnable examples covering essential TensorFlow patterns from basic model architectures to advanced training techniques.

## Pattern Categories

### Core Architecture Patterns (01-07)
- **01_sequential_model_pattern.py** - Simple linear stack of layers
- **02_functional_api_pattern.py** - Flexible model definition with complex topologies
- **03_model_subclassing_pattern.py** - Object-oriented custom model approach
- **04_custom_layer_pattern.py** - Create reusable custom layers
- **05_multi_input_model_pattern.py** - Handle multiple input sources
- **06_multi_output_model_pattern.py** - Predict multiple targets
- **07_residual_connection_pattern.py** - Skip connections for deep networks

### Training Patterns (08-11)
- **08_custom_training_loop_pattern.py** - Full control over training process
- **09_gradient_tape_pattern.py** - Manual gradient computation
- **10_mixed_precision_training_pattern.py** - FP16 for faster training
- **11_distributed_training_pattern.py** - Multi-GPU/TPU training

### Data Pipeline Patterns (12-15)
- **12_tf_data_dataset_pattern.py** - Efficient input pipelines
- **13_data_augmentation_pattern.py** - Random transformations for images
- **14_prefetching_pattern.py** - Overlap data loading and training
- **15_tfrecord_pattern.py** - Binary format for large datasets

### Regularization Patterns (16-18)
- **16_batch_normalization_pattern.py** - Normalize layer inputs
- **17_dropout_pattern.py** - Random unit dropout to prevent overfitting
- **18_l2_regularization_pattern.py** - Weight penalty regularization

### Computer Vision Patterns (19-21)
- **19_resnet_pattern.py** - Residual networks for image classification
- **20_unet_pattern.py** - U-shaped architecture for segmentation
- **21_gan_pattern.py** - Generative Adversarial Networks

### NLP Patterns (22-23)
- **22_lstm_pattern.py** - Long Short-Term Memory for sequences
- **23_transformer_attention_pattern.py** - Self-attention mechanisms

### Advanced Patterns (24-25)
- **24_transfer_learning_pattern.py** - Use pre-trained models
- **25_checkpoint_early_stopping_pattern.py** - Model saving and callbacks

## Requirements

```bash
pip install tensorflow numpy matplotlib
```

See `requirements.txt` for specific versions.

## Usage

Each file is self-contained and can be run independently:

```bash
python 01_sequential_model_pattern.py
python 02_functional_api_pattern.py
# ... etc
```

## Pattern Structure

Each pattern file includes:
- **Description** - What the pattern does
- **Multiple Examples** - Different use cases and variations
- **Working Code** - Complete, runnable implementations
- **Best Practices** - Tips and guidelines
- **Key Concepts** - Important takeaways

## Quick Start Examples

### Sequential Model
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Functional API
```python
inputs = keras.Input(shape=(28, 28, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

### Custom Training Loop
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

## Pattern Categories Summary

| Category | Patterns | Use Cases |
|----------|----------|-----------|
| Architecture | 7 | Model building, custom layers |
| Training | 4 | Custom training, distributed training |
| Data | 4 | Efficient data loading, augmentation |
| Regularization | 3 | Prevent overfitting |
| Computer Vision | 3 | Image classification, segmentation, generation |
| NLP | 2 | Text processing, sequence modeling |
| Advanced | 2 | Transfer learning, callbacks |

## Learning Path

### Beginner
1. Sequential Model (01)
2. Functional API (02)
3. Data Pipeline (12)
4. Dropout (17)
5. Callbacks (25)

### Intermediate
1. Model Subclassing (03)
2. Custom Layer (04)
3. Multi-Input/Output (05, 06)
4. Batch Normalization (16)
5. Transfer Learning (24)

### Advanced
1. Custom Training Loop (08)
2. Gradient Tape (09)
3. Mixed Precision (10)
4. Distributed Training (11)
5. ResNet (19)
6. Transformer (23)
7. GAN (21)

## Common Use Cases

### Image Classification
- Use: 01, 02, 16, 17, 19, 24

### Object Detection / Segmentation
- Use: 20, 07

### Text Classification
- Use: 22, 23

### Time Series Forecasting
- Use: 22

### Generative Models
- Use: 21

## Best Practices

1. **Data Pipeline**: Always use `tf.data` for efficient data loading
2. **Regularization**: Combine dropout, batch normalization, and L2
3. **Learning Rate**: Start high, use schedulers or reduce on plateau
4. **Callbacks**: Use early stopping and model checkpointing
5. **Transfer Learning**: For small datasets, use pre-trained models
6. **Mixed Precision**: Enable for faster training on modern GPUs
7. **Distributed Training**: Scale to multiple GPUs when needed

## Performance Tips

- Use `prefetch()` for overlapping data loading
- Enable mixed precision training (10)
- Use `@tf.function` for graph optimization
- Batch size: Start with 32, increase if memory allows
- Learning rate: 0.001 (Adam), 0.01 (SGD)

## Contributing

This is a learning resource. Each pattern is designed to be:
- **Self-contained**: Run independently
- **Well-documented**: Clear explanations
- **Practical**: Real-world applicable
- **Progressive**: From simple to complex

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## License

Educational use - feel free to learn and adapt!

## Pattern Index

Complete list of all 25 patterns with quick reference:

1. Sequential Model - Basic model building
2. Functional API - Complex architectures
3. Model Subclassing - OOP approach
4. Custom Layer - Reusable components
5. Multi-Input - Multiple data sources
6. Multi-Output - Multiple predictions
7. Residual Connection - Skip connections
8. Custom Training Loop - Training control
9. Gradient Tape - Manual gradients
10. Mixed Precision - FP16 training
11. Distributed Training - Multi-device
12. tf.data.Dataset - Data pipelines
13. Data Augmentation - Image transforms
14. Prefetching - Performance optimization
15. TFRecord - Binary data format
16. Batch Normalization - Layer normalization
17. Dropout - Overfitting prevention
18. L2 Regularization - Weight penalties
19. ResNet - Deep networks
20. U-Net - Image segmentation
21. GAN - Generative models
22. LSTM - Sequence modeling
23. Transformer - Attention mechanism
24. Transfer Learning - Pre-trained models
25. Callbacks - Training monitoring

---

**Note**: The import errors shown are expected if TensorFlow is not installed. Install with:
```bash
pip install tensorflow
```
