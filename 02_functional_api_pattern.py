"""
Functional API Pattern
Flexible way to define models with complex topologies, shared layers, and multiple inputs/outputs.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(0, 2, (1000, 1))

# Define inputs
inputs = keras.Input(shape=(20,), name='input_layer')

# Define layers
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dropout(0.5, name='dropout_1')(x)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
x = layers.Dropout(0.5, name='dropout_2')(x)
outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs, name='functional_model')

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predict
predictions = model.predict(X_train[:5])
print(f"\nPredictions: {predictions.flatten()}")
