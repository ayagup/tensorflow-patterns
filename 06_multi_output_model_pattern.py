"""
Multi-Output Model Pattern
Predict multiple targets simultaneously from the same input.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 20))
# Output 1: Binary classification
y1_train = np.random.randint(0, 2, (1000, 1))
# Output 2: Multi-class classification
y2_train = np.random.randint(0, 5, (1000,))

# Shared input
inputs = keras.Input(shape=(20,), name='input')

# Shared layers
x = layers.Dense(64, activation='relu', name='shared_dense_1')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu', name='shared_dense_2')(x)
x = layers.Dropout(0.3)(x)

# Output 1: Binary classification
output1 = layers.Dense(32, activation='relu')(x)
output1 = layers.Dense(1, activation='sigmoid', name='binary_output')(output1)

# Output 2: Multi-class classification
output2 = layers.Dense(32, activation='relu')(x)
output2 = layers.Dense(5, activation='softmax', name='multiclass_output')(output2)

# Create model
model = keras.Model(
    inputs=inputs,
    outputs=[output1, output2],
    name='multi_output_model'
)

# Compile with different losses for each output
model.compile(
    optimizer='adam',
    loss={
        'binary_output': 'binary_crossentropy',
        'multiclass_output': 'sparse_categorical_crossentropy'
    },
    loss_weights={
        'binary_output': 1.0,
        'multiclass_output': 1.0
    },
    metrics={
        'binary_output': ['accuracy'],
        'multiclass_output': ['accuracy']
    }
)

# Summary
model.summary()

# Train with multiple outputs
history = model.fit(
    X_train,
    {'binary_output': y1_train, 'multiclass_output': y2_train},
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predict
predictions = model.predict(X_train[:5])
print(f"\nBinary predictions: {predictions[0].flatten()}")
print(f"Multiclass predictions: {predictions[1].argmax(axis=1)}")
