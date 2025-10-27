"""
Transformer Attention Pattern
Self-attention mechanism for processing sequences in parallel.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("Transformer Attention Pattern\n")

# Example 1: Scaled Dot-Product Attention
print("Example 1: Scaled Dot-Product Attention")

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Calculate scaled dot-product attention.
    
    Args:
        query: Query tensor (batch_size, seq_len, d_k)
        key: Key tensor (batch_size, seq_len, d_k)
        value: Value tensor (batch_size, seq_len, d_v)
        mask: Optional mask tensor
    
    Returns:
        output: Attention output
        attention_weights: Attention weights
    """
    # Calculate attention scores
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # Scale
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    
    # Apply mask (if provided)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # Apply attention to values
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights

# Test scaled dot-product attention
batch_size, seq_len, d_k = 2, 10, 64
query = tf.random.normal((batch_size, seq_len, d_k))
key = tf.random.normal((batch_size, seq_len, d_k))
value = tf.random.normal((batch_size, seq_len, d_k))

output, weights = scaled_dot_product_attention(query, key, value)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

# Example 2: Multi-Head Attention Layer
print("\nExample 2: Multi-Head Attention")

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        
        return output, attention_weights

# Test multi-head attention
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

x = tf.random.normal((2, 10, d_model))
output, attention = mha(x, x, x)
print(f"Multi-head attention output shape: {output.shape}")

# Example 3: Positional Encoding
print("\nExample 3: Positional Encoding")

def positional_encoding(position, d_model):
    """
    Create positional encoding for transformer.
    
    Args:
        position: Maximum position
        d_model: Model dimension
    
    Returns:
        Positional encoding tensor
    """
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, d_model)
print(f"Positional encoding shape: {pos_encoding.shape}")

# Example 4: Transformer Encoder Layer
print("\nExample 4: Transformer Encoder Layer")

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2

# Test encoder layer
encoder_layer = TransformerEncoderLayer(d_model=128, num_heads=8, dff=512)
x = tf.random.normal((2, 10, 128))
encoded = encoder_layer(x, training=False)
print(f"Encoder layer output shape: {encoded.shape}")

# Example 5: Complete Transformer Encoder
print("\nExample 5: Complete Transformer Encoder")

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x

# Example 6: Transformer for Classification
print("\nExample 6: Transformer for Sequence Classification")

def create_transformer_classifier(vocab_size=10000, max_len=100, 
                                   d_model=128, num_heads=8, 
                                   num_layers=4, dff=512, num_classes=2):
    """Create a transformer-based classifier."""
    
    inputs = keras.Input(shape=(max_len,), dtype=tf.int32)
    
    # Transformer encoder
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        maximum_position_encoding=max_len
    )
    
    x = encoder(inputs, training=True)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_classifier')
    return model

transformer_model = create_transformer_classifier()
print(f"Transformer classifier created")

# Compile and test
transformer_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Dummy data
X_dummy = np.random.randint(0, 10000, (1000, 100))
y_dummy = np.random.randint(0, 2, (1000,))

history = transformer_model.fit(
    X_dummy, y_dummy,
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\nTransformer Key Features:")
print("1. Self-attention allows parallel processing of sequences")
print("2. Multi-head attention captures different relationships")
print("3. Positional encoding adds position information")
print("4. Residual connections and layer normalization")
print("5. No recurrence - faster training than RNNs")
print("6. Captures long-range dependencies effectively")
print("7. Foundation for BERT, GPT, and modern NLP models")
