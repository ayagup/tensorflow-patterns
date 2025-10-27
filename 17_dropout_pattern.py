"""
Dropout Pattern
Randomly drop units during training to prevent overfitting.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

print("Dropout Pattern Examples\n")

# Example 1: Basic Dropout
print("Example 1: Basic Dropout in Dense Network")

model_with_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.Dropout(0.5),  # Drop 50% of units
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Drop 30% of units
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),  # Drop 20% of units
    layers.Dense(1, activation='sigmoid')
], name='model_with_dropout')

model_with_dropout.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_with_dropout.summary()

history_dropout = model_with_dropout.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 2: Dropout in CNN
print("\nExample 2: Dropout in CNN")

X_image = np.random.random((1000, 28, 28, 1)).astype(np.float32)
y_image = np.random.randint(0, 10, (1000,))

cnn_model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),  # Spatial dropout for conv layers
    
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Higher dropout for dense layers
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Example 3: Spatial Dropout (for CNNs)
print("\nExample 3: Spatial Dropout")

spatial_dropout_model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.SpatialDropout2D(0.2),  # Drops entire feature maps
    layers.Conv2D(64, 3, activation='relu'),
    layers.SpatialDropout2D(0.2),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

print("SpatialDropout2D drops entire feature maps instead of individual values")

# Example 4: Dropout in RNN/LSTM
print("\nExample 4: Dropout in RNN")

X_sequence = np.random.random((1000, 10, 5)).astype(np.float32)
y_sequence = np.random.randint(0, 2, (1000, 1))

rnn_model = keras.Sequential([
    layers.LSTM(64, 
                dropout=0.2,  # Dropout for inputs
                recurrent_dropout=0.2,  # Dropout for recurrent connections
                return_sequences=True),
    layers.LSTM(32,
                dropout=0.2,
                recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("LSTM with dropout and recurrent_dropout parameters")

# Example 5: Custom Dropout usage
print("\nExample 5: Custom Dropout with training flag")

class CustomDropoutModel(keras.Model):
    def __init__(self):
        super(CustomDropoutModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(32, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)  # Only active during training
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

custom_model = CustomDropoutModel()
custom_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Example 6: Monte Carlo Dropout (for uncertainty estimation)
print("\nExample 6: Monte Carlo Dropout for Uncertainty")

# Train model
mc_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

mc_model.compile(optimizer='adam', loss='binary_crossentropy')
mc_model.fit(X_train, y_train, epochs=3, verbose=0)

# Make predictions with dropout enabled (Monte Carlo sampling)
def mc_dropout_predict(model, x, num_samples=100):
    predictions = []
    for _ in range(num_samples):
        # Enable dropout during inference
        pred = model(x, training=True)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    
    return mean, std

x_test = X_train[:5]
mean_pred, std_pred = mc_dropout_predict(mc_model, x_test, num_samples=50)

print(f"Predictions with uncertainty:")
for i in range(5):
    print(f"  Sample {i}: {mean_pred[i][0]:.3f} ± {std_pred[i][0]:.3f}")

# Example 7: Comparison with and without Dropout
print("\nExample 7: Overfitting Comparison")

# Model without dropout (prone to overfitting)
model_no_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_no_dropout.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_no_dropout = model_no_dropout.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Compare training vs validation performance
print(f"\nWith Dropout:")
print(f"  Train acc: {history_dropout.history['accuracy'][-1]:.4f}")
print(f"  Val acc: {history_dropout.history['val_accuracy'][-1]:.4f}")
print(f"  Gap: {history_dropout.history['accuracy'][-1] - history_dropout.history['val_accuracy'][-1]:.4f}")

print(f"\nWithout Dropout:")
print(f"  Train acc: {history_no_dropout.history['accuracy'][-1]:.4f}")
print(f"  Val acc: {history_no_dropout.history['val_accuracy'][-1]:.4f}")
print(f"  Gap: {history_no_dropout.history['accuracy'][-1] - history_no_dropout.history['val_accuracy'][-1]:.4f}")

print("\nDropout Guidelines:")
print("1. Common rates: 0.2-0.5 for hidden layers")
print("2. Higher rates (0.5) for dense layers")
print("3. Lower rates (0.2-0.3) for conv layers")
print("4. Always disabled during inference (training=False)")
print("5. Acts as ensemble of multiple networks")
