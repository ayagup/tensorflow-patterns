"""
LSTM (Long Short-Term Memory) Pattern
Recurrent neural network for sequence data with memory cells.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("LSTM Pattern\n")

# Generate dummy sequence data
X_train = np.random.random((1000, 10, 5)).astype(np.float32)  # (samples, timesteps, features)
y_train = np.random.randint(0, 2, (1000, 1))

# Example 1: Basic LSTM
print("Example 1: Basic LSTM for Sequence Classification")

model_simple = keras.Sequential([
    layers.LSTM(64, input_shape=(10, 5)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
], name='simple_lstm')

model_simple.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_simple.summary()

history = model_simple.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 2: Stacked LSTM
print("\nExample 2: Stacked LSTM")

model_stacked = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(10, 5)),
    layers.Dropout(0.3),
    layers.LSTM(32, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(16),
    layers.Dense(1, activation='sigmoid')
], name='stacked_lstm')

model_stacked.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Stacked LSTM with return_sequences=True for intermediate layers")

# Example 3: Bidirectional LSTM
print("\nExample 3: Bidirectional LSTM")

model_bidirectional = keras.Sequential([
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(10, 5)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
], name='bidirectional_lstm')

model_bidirectional.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_bidirectional.summary()

print("Bidirectional LSTM processes sequences in both directions")

# Example 4: LSTM for Sequence-to-Sequence
print("\nExample 4: LSTM for Sequence-to-Sequence")

# Dummy seq2seq data
encoder_input = np.random.random((1000, 10, 8)).astype(np.float32)
decoder_input = np.random.random((1000, 15, 6)).astype(np.float32)
decoder_output = np.random.random((1000, 15, 6)).astype(np.float32)

# Encoder
encoder_inputs = keras.Input(shape=(10, 8))
encoder_lstm = layers.LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(15, 6))
decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = layers.Dense(6, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model_seq2seq = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq')

model_seq2seq.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

print("Sequence-to-sequence LSTM for tasks like translation")

# Example 5: LSTM with Attention
print("\nExample 5: LSTM with Attention")

class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
    
    def call(self, x):
        # Calculate attention scores
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)
        
        # Apply attention
        context_vector = x * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

# Build model with attention
inputs = keras.Input(shape=(10, 5))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.Dropout(0.3)(x)
x = AttentionLayer()(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model_attention = keras.Model(inputs=inputs, outputs=outputs, name='lstm_attention')

model_attention.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("LSTM with attention mechanism focuses on important parts of sequence")

# Example 6: Many-to-Many LSTM
print("\nExample 6: Many-to-Many LSTM (Sequence Labeling)")

# Sequence labeling data (e.g., POS tagging)
X_seq_label = np.random.random((1000, 10, 5)).astype(np.float32)
y_seq_label = np.random.randint(0, 5, (1000, 10, 5)).astype(np.float32)

model_many2many = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(10, 5)),
    layers.Dropout(0.3),
    layers.LSTM(32, return_sequences=True),
    layers.TimeDistributed(layers.Dense(5, activation='softmax'))
], name='many_to_many_lstm')

model_many2many.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Many-to-many LSTM outputs a label for each timestep")

# Example 7: Stateful LSTM
print("\nExample 7: Stateful LSTM")

batch_size = 32
model_stateful = keras.Sequential([
    layers.LSTM(64, stateful=True, 
                batch_input_shape=(batch_size, 10, 5)),
    layers.Dense(1, activation='sigmoid')
], name='stateful_lstm')

model_stateful.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Stateful LSTM maintains state across batches")

# Example 8: LSTM with Dropout and Recurrent Dropout
print("\nExample 8: LSTM with Dropout")

model_dropout = keras.Sequential([
    layers.LSTM(
        64,
        dropout=0.2,  # Dropout for inputs
        recurrent_dropout=0.2,  # Dropout for recurrent connections
        return_sequences=True,
        input_shape=(10, 5)
    ),
    layers.LSTM(
        32,
        dropout=0.2,
        recurrent_dropout=0.2
    ),
    layers.Dense(1, activation='sigmoid')
], name='lstm_dropout')

model_dropout.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Dropout helps prevent overfitting in LSTM")

# Example 9: Time Series Forecasting with LSTM
print("\nExample 9: Time Series Forecasting")

# Create time series data
time_steps = 50
features = 1
X_ts = np.random.random((1000, time_steps, features)).astype(np.float32)
y_ts = np.random.random((1000, 1)).astype(np.float32)

model_forecast = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(time_steps, features)),
    layers.Dense(25, activation='relu'),
    layers.Dense(1)
], name='lstm_forecasting')

model_forecast.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history_ts = model_forecast.fit(
    X_ts, y_ts,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Make prediction
prediction = model_forecast.predict(X_ts[:1])
print(f"Forecast: {prediction[0][0]:.4f}")

print("\nLSTM Key Features:")
print("1. Handles long-term dependencies better than vanilla RNN")
print("2. Cell state acts as memory")
print("3. Gates (forget, input, output) control information flow")
print("4. Solves vanishing gradient problem")
print("5. Good for sequences: time series, text, audio, video")
print("6. Can be bidirectional or stacked for better performance")
