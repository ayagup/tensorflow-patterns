"""
Custom Layer Pattern
Create reusable custom layers with learnable parameters.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class CustomDenseLayer(layers.Layer):
    def __init__(self, units=32, activation=None):
        super(CustomDenseLayer, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create weights
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation)
        })
        return config

# Generate dummy data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(0, 2, (1000, 1))

# Create model using custom layer
model = keras.Sequential([
    CustomDenseLayer(64, activation='relu'),
    layers.Dropout(0.5),
    CustomDenseLayer(32, activation='relu'),
    layers.Dropout(0.5),
    CustomDenseLayer(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build and show summary
model.build((None, 20))
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
