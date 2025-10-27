"""
Multi-Input Model Pattern
Handle multiple input sources with different characteristics.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
# Input 1: Numerical features
X_numerical = np.random.random((1000, 10))
# Input 2: Categorical features (embedded)
X_categorical = np.random.randint(0, 50, (1000, 5))
# Target
y_train = np.random.randint(0, 2, (1000, 1))

# Input 1: Numerical features branch
input_numerical = keras.Input(shape=(10,), name='numerical_input')
x1 = layers.Dense(32, activation='relu')(input_numerical)
x1 = layers.Dropout(0.3)(x1)

# Input 2: Categorical features branch
input_categorical = keras.Input(shape=(5,), name='categorical_input')
x2 = layers.Embedding(input_dim=50, output_dim=16)(input_categorical)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(32, activation='relu')(x2)
x2 = layers.Dropout(0.3)(x2)

# Merge branches
merged = layers.concatenate([x1, x2])

# Output layers
x = layers.Dense(64, activation='relu')(merged)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# Create model
model = keras.Model(
    inputs=[input_numerical, input_categorical],
    outputs=outputs,
    name='multi_input_model'
)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train with multiple inputs
history = model.fit(
    [X_numerical, X_categorical],
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predict
predictions = model.predict([X_numerical[:5], X_categorical[:5]])
print(f"\nPredictions: {predictions.flatten()}")
