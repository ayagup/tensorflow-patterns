"""
Model Subclassing Pattern
Object-oriented approach for maximum flexibility and custom behavior.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class CustomModel(keras.Model):
    def __init__(self, num_classes=1):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dropout1 = layers.Dropout(0.5, name='dropout_1')
        self.dense2 = layers.Dense(64, activation='relu', name='dense_2')
        self.dropout2 = layers.Dropout(0.5, name='dropout_2')
        self.classifier = layers.Dense(num_classes, activation='sigmoid', name='classifier')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)
    
    def get_config(self):
        return {"num_classes": self.classifier.units}

# Generate dummy data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(0, 2, (1000, 1))

# Create model
model = CustomModel(num_classes=1)

# Build the model by calling it once
_ = model(X_train[:1])

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
