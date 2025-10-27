"""
Sequential Model Pattern
Simple linear stack of layers for straightforward architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(0, 2, (1000, 1))

# Create Sequential Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
], name='sequential_model')

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
