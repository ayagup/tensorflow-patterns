"""
Time Series Forecasting Patterns

This module demonstrates various patterns for time series analysis and
forecasting using TensorFlow.

Patterns covered:
1. Sliding Window Dataset
2. Univariate LSTM Forecasting
3. Multivariate Time Series
4. Multi-Step Forecasting
5. Encoder-Decoder for Sequences
6. Attention-based Time Series
7. CNN for Time Series
8. WaveNet-style Model
9. Temporal Convolutional Network (TCN)
10. Transformer for Time Series
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Sliding Window Dataset
def create_sliding_window_dataset(data, window_size, horizon=1, batch_size=32):
    """Create sliding window dataset for time series."""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    # Create windows
    dataset = dataset.window(window_size + horizon, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + horizon))
    
    # Split into features and labels
    dataset = dataset.map(lambda window: (window[:-horizon], window[-horizon:]))
    
    # Shuffle and batch
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def example_sliding_window():
    """Example: Create sliding window dataset."""
    # Generate synthetic time series
    time = np.arange(0, 100, 0.1)
    data = np.sin(time) + np.random.randn(len(time)) * 0.1
    data = data.astype(np.float32)
    
    window_size = 20
    horizon = 5
    
    dataset = create_sliding_window_dataset(data, window_size, horizon)
    
    print("Sliding Window Dataset:")
    for x, y in dataset.take(1):
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
    
    return dataset


# Pattern 2: Univariate LSTM Forecasting
def create_univariate_lstm(window_size, horizon=1):
    """LSTM model for univariate forecasting."""
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(horizon)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def example_univariate_lstm():
    """Example: Univariate LSTM forecasting."""
    model = create_univariate_lstm(window_size=20, horizon=5)
    
    print("\nUnivariate LSTM Model:")
    model.summary()
    
    return model


# Pattern 3: Multivariate Time Series
def create_multivariate_model(window_size, n_features, horizon=1):
    """LSTM model for multivariate time series."""
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(window_size, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_features * horizon)
    ])
    
    # Reshape output
    inputs = keras.Input(shape=(window_size, n_features))
    x = model(inputs)
    outputs = layers.Reshape((horizon, n_features))(x)
    
    final_model = keras.Model(inputs, outputs)
    
    final_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return final_model


def example_multivariate():
    """Example: Multivariate time series model."""
    model = create_multivariate_model(
        window_size=24,
        n_features=5,
        horizon=6
    )
    
    print("\nMultivariate Time Series Model:")
    model.summary()
    
    return model


# Pattern 4: Multi-Step Forecasting
class MultiStepModel(keras.Model):
    """Multi-step forecasting with different strategies."""
    
    def __init__(self, units=64, horizon=10):
        super().__init__()
        self.horizon = horizon
        
        self.lstm = layers.LSTM(units, return_sequences=True)
        self.dense_layers = [
            layers.Dense(32, activation='relu') for _ in range(horizon)
        ]
        self.output_layers = [
            layers.Dense(1) for _ in range(horizon)
        ]
    
    def call(self, inputs):
        x = self.lstm(inputs)
        
        # Get predictions for each time step
        predictions = []
        for i in range(self.horizon):
            step_output = self.dense_layers[i](x[:, -1, :])
            step_pred = self.output_layers[i](step_output)
            predictions.append(step_pred)
        
        return tf.concat(predictions, axis=-1)


def example_multi_step():
    """Example: Multi-step forecasting."""
    model = MultiStepModel(units=64, horizon=10)
    
    # Dummy data
    x = tf.random.normal((32, 20, 1))
    y = model(x)
    
    print("\nMulti-Step Forecasting Model:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Predicts multiple future time steps")
    
    return model


# Pattern 5: Encoder-Decoder for Sequences
def create_seq2seq_model(window_size, n_features, horizon):
    """Encoder-decoder model for sequence forecasting."""
    # Encoder
    encoder_inputs = keras.Input(shape=(window_size, n_features))
    encoder = layers.LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = keras.Input(shape=(horizon, n_features))
    decoder_lstm = layers.LSTM(128, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs,
        initial_state=encoder_states
    )
    
    decoder_dense = layers.Dense(n_features)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def example_seq2seq():
    """Example: Sequence-to-sequence model."""
    model = create_seq2seq_model(
        window_size=24,
        n_features=3,
        horizon=12
    )
    
    print("\nSequence-to-Sequence Model:")
    model.summary()
    
    return model


# Pattern 6: Attention-based Time Series
class AttentionLayer(layers.Layer):
    """Attention mechanism for time series."""
    
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # query: (batch, hidden_units)
        # values: (batch, time_steps, hidden_units)
        
        # Expand query to match time dimension
        query_with_time = tf.expand_dims(query, 1)  # (batch, 1, hidden_units)
        
        # Attention scores
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time) + self.W2(values)
        ))  # (batch, time_steps, 1)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context = attention_weights * values
        context = tf.reduce_sum(context, axis=1)
        
        return context, attention_weights


class AttentionTimeSeries(keras.Model):
    """Time series model with attention."""
    
    def __init__(self, units=64, output_dim=1):
        super().__init__()
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.attention = AttentionLayer(units)
        self.dense = layers.Dense(output_dim)
    
    def call(self, inputs):
        lstm_out, state_h, state_c = self.lstm(inputs)
        context, attention_weights = self.attention(state_h, lstm_out)
        output = self.dense(context)
        return output


def example_attention_timeseries():
    """Example: Attention-based time series."""
    model = AttentionTimeSeries(units=64, output_dim=1)
    
    # Dummy data
    x = tf.random.normal((32, 20, 5))
    y = model(x)
    
    print("\nAttention-based Time Series Model:")
    print(f"Output shape: {y.shape}")
    print("Uses attention to focus on important time steps")
    
    return model


# Pattern 7: CNN for Time Series
def create_cnn_timeseries(window_size, n_features, horizon=1):
    """CNN model for time series forecasting."""
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(window_size, n_features)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(horizon * n_features),
        layers.Reshape((horizon, n_features))
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def example_cnn_timeseries():
    """Example: CNN for time series."""
    model = create_cnn_timeseries(window_size=50, n_features=3, horizon=10)
    
    print("\nCNN Time Series Model:")
    model.summary()
    
    return model


# Pattern 8: WaveNet-style Model
class CausalConv1D(layers.Layer):
    """Causal convolution for time series."""
    
    def __init__(self, filters, kernel_size, dilation_rate):
        super().__init__()
        self.causal_conv = layers.Conv1D(
            filters,
            kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu'
        )
    
    def call(self, inputs):
        return self.causal_conv(inputs)


class WaveNetBlock(layers.Layer):
    """WaveNet-style residual block."""
    
    def __init__(self, filters, kernel_size, dilation_rate):
        super().__init__()
        self.causal_conv = CausalConv1D(filters, kernel_size, dilation_rate)
        self.skip_conv = layers.Conv1D(filters, 1)
        self.residual_conv = layers.Conv1D(filters, 1)
    
    def call(self, inputs):
        x = self.causal_conv(inputs)
        skip = self.skip_conv(x)
        residual = self.residual_conv(x)
        return residual + inputs, skip


def create_wavenet_model(window_size, n_features, num_blocks=3):
    """WaveNet-style model for time series."""
    inputs = keras.Input(shape=(window_size, n_features))
    
    x = layers.Conv1D(64, 1)(inputs)
    
    skip_connections = []
    for i in range(num_blocks):
        x, skip = WaveNetBlock(
            filters=64,
            kernel_size=2,
            dilation_rate=2**i
        )(x)
        skip_connections.append(skip)
    
    # Combine skip connections
    x = tf.add_n(skip_connections)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.Conv1D(1, 1)(x)
    
    # Take last time step
    outputs = x[:, -1, :]
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def example_wavenet():
    """Example: WaveNet-style model."""
    model = create_wavenet_model(window_size=32, n_features=3, num_blocks=4)
    
    print("\nWaveNet-style Model:")
    model.summary()
    
    return model


# Pattern 9: Temporal Convolutional Network (TCN)
class TCNBlock(layers.Layer):
    """TCN residual block."""
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout=0.2):
        super().__init__()
        self.conv1 = layers.Conv1D(
            filters, kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout)
        
        self.conv2 = layers.Conv1D(
            filters, kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout)
        
        self.downsample = layers.Conv1D(filters, 1) if filters != None else None
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        
        if self.downsample:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        
        return tf.nn.relu(x + residual)


def create_tcn_model(window_size, n_features, num_blocks=3):
    """Temporal Convolutional Network."""
    inputs = keras.Input(shape=(window_size, n_features))
    
    x = inputs
    for i in range(num_blocks):
        x = TCNBlock(
            filters=64,
            kernel_size=3,
            dilation_rate=2**i,
            dropout=0.2
        )(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def example_tcn():
    """Example: Temporal Convolutional Network."""
    model = create_tcn_model(window_size=50, n_features=5, num_blocks=4)
    
    print("\nTemporal Convolutional Network (TCN):")
    model.summary()
    
    return model


# Pattern 10: Transformer for Time Series
class TimeSeriesTransformer(keras.Model):
    """Transformer model for time series."""
    
    def __init__(self, d_model=64, num_heads=4, num_layers=2, output_dim=1):
        super().__init__()
        
        self.input_projection = layers.Dense(d_model)
        
        # Transformer encoder layers
        self.transformer_blocks = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
            for _ in range(num_layers)
        ]
        
        self.ffn_blocks = [
            keras.Sequential([
                layers.Dense(d_model * 4, activation='relu'),
                layers.Dense(d_model)
            ])
            for _ in range(num_layers)
        ]
        
        self.norm_blocks = [
            layers.LayerNormalization() for _ in range(num_layers * 2)
        ]
        
        self.output_layer = layers.Dense(output_dim)
    
    def call(self, inputs):
        x = self.input_projection(inputs)
        
        for i, (attn, ffn) in enumerate(zip(self.transformer_blocks, self.ffn_blocks)):
            # Multi-head attention
            attn_output = attn(x, x)
            x = self.norm_blocks[i * 2](x + attn_output)
            
            # Feed-forward
            ffn_output = ffn(x)
            x = self.norm_blocks[i * 2 + 1](x + ffn_output)
        
        # Global pooling
        x = tf.reduce_mean(x, axis=1)
        outputs = self.output_layer(x)
        
        return outputs


def example_transformer_timeseries():
    """Example: Transformer for time series."""
    model = TimeSeriesTransformer(
        d_model=64,
        num_heads=4,
        num_layers=2,
        output_dim=1
    )
    
    # Dummy data
    x = tf.random.normal((32, 20, 5))
    y = model(x)
    
    print("\nTransformer for Time Series:")
    print(f"Output shape: {y.shape}")
    print("Uses self-attention for temporal dependencies")
    
    return model


if __name__ == "__main__":
    print("Time Series Forecasting Patterns\n" + "="*60)
    
    # Example 1: Sliding Window
    print("\n1. Sliding Window Dataset")
    dataset1 = example_sliding_window()
    
    # Example 2: Univariate LSTM
    print("\n2. Univariate LSTM Forecasting")
    model2 = example_univariate_lstm()
    
    # Example 3: Multivariate
    print("\n3. Multivariate Time Series")
    model3 = example_multivariate()
    
    # Example 4: Multi-Step
    print("\n4. Multi-Step Forecasting")
    model4 = example_multi_step()
    
    # Example 5: Seq2Seq
    print("\n5. Sequence-to-Sequence Model")
    model5 = example_seq2seq()
    
    # Example 6: Attention
    print("\n6. Attention-based Time Series")
    model6 = example_attention_timeseries()
    
    # Example 7: CNN
    print("\n7. CNN for Time Series")
    model7 = example_cnn_timeseries()
    
    # Example 8: WaveNet
    print("\n8. WaveNet-style Model")
    model8 = example_wavenet()
    
    # Example 9: TCN
    print("\n9. Temporal Convolutional Network (TCN)")
    model9 = example_tcn()
    
    # Example 10: Transformer
    print("\n10. Transformer for Time Series")
    model10 = example_transformer_timeseries()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Normalize/standardize time series data")
    print("2. Use appropriate window size based on seasonality")
    print("3. LSTM: good for long-term dependencies")
    print("4. CNN: faster, good for local patterns")
    print("5. Attention: interpretable, focuses on important steps")
    print("6. Seq2Seq: great for multi-step forecasting")
    print("7. TCN: combines benefits of CNN and RNN")
    print("8. Transformer: state-of-the-art for many tasks")
    print("9. Use teacher forcing for training seq2seq")
    print("10. Monitor validation loss to prevent overfitting")
