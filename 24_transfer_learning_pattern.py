"""
Transfer Learning Pattern
Use pre-trained models as feature extractors or for fine-tuning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("Transfer Learning Pattern\n")

# Generate dummy image data
X_train = np.random.random((500, 224, 224, 3)).astype(np.float32)
y_train = np.random.randint(0, 5, (500,))

# Example 1: Feature Extraction (Frozen Base)
print("Example 1: Feature Extraction with Frozen Base Model")

# Load pre-trained model without top layers
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None  # Set to 'imagenet' to use pre-trained weights
)

# Freeze base model
base_model.trainable = False

# Add custom classification head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # training=False for BatchNorm
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model_frozen = keras.Model(inputs, outputs, name='transfer_frozen')

model_frozen.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Base model layers: {len(base_model.layers)}, Trainable: {base_model.trainable}")
model_frozen.summary()

# Train
history_frozen = model_frozen.fit(
    X_train, y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Example 2: Fine-Tuning
print("\nExample 2: Fine-Tuning Pre-trained Model")

# Unfreeze base model
base_model.trainable = True

# Fine-tune from a specific layer
fine_tune_at = 100  # Freeze layers before this index

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model_frozen.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Fine-tuning from layer {fine_tune_at}")
print(f"Trainable layers: {sum([1 for layer in base_model.layers if layer.trainable])}")

# Continue training
history_finetune = model_frozen.fit(
    X_train, y_train,
    epochs=2,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Example 3: Different Pre-trained Models
print("\nExample 3: Using Different Pre-trained Architectures")

def create_transfer_model(base_model_fn, input_shape, num_classes):
    """Create transfer learning model with any base architecture."""
    base = base_model_fn(
        include_top=False,
        weights=None,
        input_shape=input_shape
    )
    base.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# ResNet50
model_resnet = create_transfer_model(
    keras.applications.ResNet50,
    (224, 224, 3),
    num_classes=5
)
print("Created ResNet50 transfer model")

# VGG16
model_vgg = create_transfer_model(
    keras.applications.VGG16,
    (224, 224, 3),
    num_classes=5
)
print("Created VGG16 transfer model")

# InceptionV3
model_inception = create_transfer_model(
    keras.applications.InceptionV3,
    (299, 299, 3),
    num_classes=5
)
print("Created InceptionV3 transfer model")

# EfficientNetB0
model_efficient = create_transfer_model(
    keras.applications.EfficientNetB0,
    (224, 224, 3),
    num_classes=5
)
print("Created EfficientNetB0 transfer model")

# Example 4: Progressive Fine-Tuning
print("\nExample 4: Progressive Fine-Tuning Strategy")

def progressive_unfreeze(model, base_model, num_stages=3):
    """Progressively unfreeze layers in stages."""
    total_layers = len(base_model.layers)
    layers_per_stage = total_layers // num_stages
    
    for stage in range(num_stages):
        print(f"\nStage {stage + 1}/{num_stages}")
        
        # Unfreeze next set of layers
        start_idx = total_layers - (stage + 1) * layers_per_stage
        for layer in base_model.layers[start_idx:]:
            layer.trainable = True
        
        # Compile with decreasing learning rate
        lr = 1e-4 / (10 ** stage)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Learning rate: {lr}")
        print(f"Trainable layers: {sum([1 for l in base_model.layers if l.trainable])}")

# Example 5: Feature Extraction for Multiple Tasks
print("\nExample 5: Multi-Task Transfer Learning")

base_model_multi = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)
base_model_multi.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
features = base_model_multi(inputs, training=False)
features = layers.GlobalAveragePooling2D()(features)

# Task 1: Classification
x1 = layers.Dense(128, activation='relu')(features)
x1 = layers.Dropout(0.5)(x1)
output1 = layers.Dense(5, activation='softmax', name='classification')(x1)

# Task 2: Regression
x2 = layers.Dense(64, activation='relu')(features)
x2 = layers.Dropout(0.3)(x2)
output2 = layers.Dense(1, name='regression')(x2)

model_multitask = keras.Model(
    inputs=inputs,
    outputs=[output1, output2],
    name='multi_task_transfer'
)

model_multitask.compile(
    optimizer='adam',
    loss={
        'classification': 'sparse_categorical_crossentropy',
        'regression': 'mse'
    },
    metrics={
        'classification': ['accuracy'],
        'regression': ['mae']
    }
)

print("Multi-task model using shared pre-trained features")

# Example 6: Domain Adaptation
print("\nExample 6: Domain Adaptation Pattern")

# Load base model trained on source domain
source_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

# Freeze early layers (general features)
for layer in source_model.layers[:50]:
    layer.trainable = False

# Fine-tune later layers (domain-specific features)
for layer in source_model.layers[50:]:
    layer.trainable = True

inputs = keras.Input(shape=(224, 224, 3))
x = source_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model_adapted = keras.Model(inputs, outputs, name='domain_adapted')

print("Domain adaptation: early layers frozen, late layers fine-tuned")

# Example 7: Best Practices
print("\nExample 7: Transfer Learning Best Practices")

print("""
Transfer Learning Best Practices:

1. Feature Extraction (Small Dataset):
   - Freeze entire base model
   - Train only custom head
   - Use moderate learning rate (1e-3)

2. Fine-Tuning (Medium Dataset):
   - Freeze early layers
   - Fine-tune top layers
   - Use small learning rate (1e-5)

3. Full Training (Large Dataset):
   - Unfreeze all layers
   - Use very small learning rate (1e-6)
   - Consider longer training

4. Learning Rate Strategy:
   - Feature extraction: 1e-3
   - Fine-tuning: 1e-4 to 1e-5
   - Lower LR for unfrozen layers

5. Data Augmentation:
   - Always use augmentation
   - Helps prevent overfitting
   - Essential for small datasets

6. Batch Normalization:
   - Keep training=False during feature extraction
   - Use pre-trained statistics
   - Switch to training=True when fine-tuning

7. Progressive Unfreezing:
   - Start with frozen base
   - Gradually unfreeze layers
   - Adjust LR accordingly
""")

print("\nTransfer Learning Advantages:")
print("- Reduces training time")
print("- Requires less data")
print("- Better generalization")
print("- State-of-the-art performance on many tasks")
