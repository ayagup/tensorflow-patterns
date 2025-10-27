"""
Attention Mechanisms Patterns

This module demonstrates various attention mechanisms used in
deep learning, from basic attention to advanced variants.

Patterns covered:
1. Additive (Bahdanau) Attention
2. Multiplicative (Luong) Attention
3. Scaled Dot-Product Attention
4. Multi-Head Attention
5. Self-Attention
6. Cross-Attention
7. Local Attention
8. Hierarchical Attention
9. Attention Pooling
10. Channel Attention (Squeeze-and-Excitation)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Additive (Bahdanau) Attention
class AdditiveAttention(layers.Layer):
    """Bahdanau attention mechanism."""
    
    def __init__(self, units):
        super().__init__()
        
        self.W_query = layers.Dense(units, use_bias=False)
        self.W_key = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
    
    def call(self, query, key, value, mask=None):
        # query: (batch, query_len, d_query)
        # key, value: (batch, key_len, d_key)
        
        # Expand query for broadcasting
        query_expanded = tf.expand_dims(query, 2)  # (batch, query_len, 1, d_query)
        key_expanded = tf.expand_dims(key, 1)      # (batch, 1, key_len, d_key)
        
        # Compute attention scores
        score = self.V(tf.nn.tanh(
            self.W_query(query_expanded) + self.W_key(key_expanded)
        ))  # (batch, query_len, key_len, 1)
        
        score = tf.squeeze(score, -1)  # (batch, query_len, key_len)
        
        # Apply mask if provided
        if mask is not None:
            score += (mask * -1e9)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Context vector
        context = tf.matmul(attention_weights, value)
        
        return context, attention_weights


def example_additive_attention():
    """Example: Additive attention."""
    attention = AdditiveAttention(units=64)
    
    query = tf.random.normal((2, 5, 32))   # 5 queries
    key = tf.random.normal((2, 10, 32))    # 10 keys
    value = tf.random.normal((2, 10, 32))  # 10 values
    
    context, weights = attention(query, key, value)
    
    print("Additive (Bahdanau) Attention Example:")
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    return attention


# Pattern 2: Multiplicative (Luong) Attention
class MultiplicativeAttention(layers.Layer):
    """Luong attention mechanism."""
    
    def __init__(self, units):
        super().__init__()
        
        self.W = layers.Dense(units, use_bias=False)
    
    def call(self, query, key, value, mask=None):
        # query: (batch, query_len, d_query)
        # key, value: (batch, key_len, d_key)
        
        # Transform query
        query_transformed = self.W(query)  # (batch, query_len, units)
        
        # Compute attention scores
        score = tf.matmul(query_transformed, key, transpose_b=True)
        # (batch, query_len, key_len)
        
        # Apply mask if provided
        if mask is not None:
            score += (mask * -1e9)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Context vector
        context = tf.matmul(attention_weights, value)
        
        return context, attention_weights


def example_multiplicative_attention():
    """Example: Multiplicative attention."""
    attention = MultiplicativeAttention(units=32)
    
    query = tf.random.normal((2, 5, 32))
    key = tf.random.normal((2, 10, 32))
    value = tf.random.normal((2, 10, 32))
    
    context, weights = attention(query, key, value)
    
    print("\nMultiplicative (Luong) Attention Example:")
    print(f"Context shape: {context.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print("More computationally efficient than additive")
    
    return attention


# Pattern 3: Scaled Dot-Product Attention
class ScaledDotProductAttention(layers.Layer):
    """Scaled dot-product attention (used in Transformers)."""
    
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, query, key, value, mask=None, training=False):
        # query, key, value: (batch, seq_len, d_k)
        
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Weighted sum
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights


def example_scaled_dot_product_attention():
    """Example: Scaled dot-product attention."""
    attention = ScaledDotProductAttention()
    
    d_k = 64
    query = tf.random.normal((2, 10, d_k))
    key = tf.random.normal((2, 10, d_k))
    value = tf.random.normal((2, 10, d_k))
    
    output, weights = attention(query, key, value)
    
    print("\nScaled Dot-Product Attention Example:")
    print(f"Output shape: {output.shape}")
    print(f"Scaling factor: sqrt({d_k}) = {np.sqrt(d_k):.2f}")
    print("Used in Transformer architecture")
    
    return attention


# Pattern 4: Multi-Head Attention
class MultiHeadAttention(layers.Layer):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        self.W_o = layers.Dense(d_model)
        
        self.attention = ScaledDotProductAttention()
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth)."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, num_heads, seq_len, d_k)
    
    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask, training)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights


def example_multi_head_attention():
    """Example: Multi-head attention."""
    d_model = 128
    num_heads = 8
    
    attention = MultiHeadAttention(d_model, num_heads)
    
    x = tf.random.normal((2, 10, d_model))
    
    output, weights = attention(x, x, x)
    
    print("\nMulti-Head Attention Example:")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"d_k per head: {d_model // num_heads}")
    print(f"Output shape: {output.shape}")
    print("Allows model to attend to different aspects")
    
    return attention


# Pattern 5: Self-Attention
class SelfAttention(layers.Layer):
    """Self-attention layer."""
    
    def __init__(self, d_model):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads=8)
        self.norm = layers.LayerNormalization()
    
    def call(self, x, mask=None, training=False):
        # Self-attention: query, key, value are all the same
        attn_output, _ = self.mha(x, x, x, mask, training)
        
        # Residual connection and normalization
        output = self.norm(x + attn_output)
        
        return output


def example_self_attention():
    """Example: Self-attention."""
    d_model = 128
    self_attn = SelfAttention(d_model)
    
    x = tf.random.normal((2, 10, d_model))
    output = self_attn(x)
    
    print("\nSelf-Attention Example:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Each position attends to all positions in sequence")
    
    return self_attn


# Pattern 6: Cross-Attention
class CrossAttention(layers.Layer):
    """Cross-attention between two sequences."""
    
    def __init__(self, d_model):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads=8)
        self.norm = layers.LayerNormalization()
    
    def call(self, query_seq, key_value_seq, mask=None, training=False):
        # Query from one sequence, key/value from another
        attn_output, attn_weights = self.mha(
            query_seq, key_value_seq, key_value_seq, mask, training
        )
        
        # Residual connection and normalization
        output = self.norm(query_seq + attn_output)
        
        return output, attn_weights


def example_cross_attention():
    """Example: Cross-attention."""
    d_model = 128
    cross_attn = CrossAttention(d_model)
    
    # Two different sequences
    seq1 = tf.random.normal((2, 10, d_model))  # e.g., decoder
    seq2 = tf.random.normal((2, 15, d_model))  # e.g., encoder
    
    output, weights = cross_attn(seq1, seq2)
    
    print("\nCross-Attention Example:")
    print(f"Query sequence shape: {seq1.shape}")
    print(f"Key/Value sequence shape: {seq2.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print("Used in encoder-decoder architectures")
    
    return cross_attn


# Pattern 7: Local Attention
class LocalAttention(layers.Layer):
    """Local attention with limited window."""
    
    def __init__(self, d_model, window_size=5):
        super().__init__()
        
        self.d_model = d_model
        self.window_size = window_size
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
    
    def call(self, x):
        # x: (batch, seq_len, d_model)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        outputs = []
        
        for i in range(seq_len):
            # Define attention window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Local query and keys
            q_i = Q[:, i:i+1, :]
            k_local = K[:, start:end, :]
            v_local = V[:, start:end, :]
            
            # Attention scores
            scores = tf.matmul(q_i, k_local, transpose_b=True)
            scores = scores / tf.sqrt(tf.cast(self.d_model, tf.float32))
            
            # Attention weights
            weights = tf.nn.softmax(scores, axis=-1)
            
            # Context
            context = tf.matmul(weights, v_local)
            outputs.append(context)
        
        output = tf.concat(outputs, axis=1)
        
        return output


def example_local_attention():
    """Example: Local attention."""
    d_model = 64
    window_size = 5
    
    local_attn = LocalAttention(d_model, window_size)
    
    x = tf.random.normal((2, 20, d_model))
    output = local_attn(x)
    
    print("\nLocal Attention Example:")
    print(f"Sequence length: {x.shape[1]}")
    print(f"Window size: {window_size}")
    print(f"Output shape: {output.shape}")
    print("Reduces computation for long sequences")
    
    return local_attn


# Pattern 8: Hierarchical Attention
class HierarchicalAttention(keras.Model):
    """Hierarchical attention (word-level and sentence-level)."""
    
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        
        # Word-level attention
        self.word_attention = MultiHeadAttention(d_model, num_heads)
        
        # Sentence-level attention
        self.sentence_attention = MultiHeadAttention(d_model, num_heads)
        
        self.word_dense = layers.Dense(d_model)
        self.sentence_dense = layers.Dense(d_model)
    
    def call(self, words, num_sentences):
        # words: (batch, total_words, d_model)
        # num_sentences: number of sentences per batch
        
        # Word-level attention within each sentence
        word_contexts, _ = self.word_attention(words, words, words)
        
        # Aggregate words into sentences
        # In practice, would need sentence boundaries
        # Simplified: average pooling
        sentence_repr = tf.reduce_mean(word_contexts, axis=1, keepdims=True)
        
        # Sentence-level attention
        sentence_contexts, _ = self.sentence_attention(
            sentence_repr, sentence_repr, sentence_repr
        )
        
        # Final document representation
        doc_repr = self.sentence_dense(sentence_contexts)
        
        return doc_repr


def example_hierarchical_attention():
    """Example: Hierarchical attention."""
    d_model = 128
    
    hier_attn = HierarchicalAttention(d_model)
    
    # Document with words
    words = tf.random.normal((2, 100, d_model))
    
    doc_repr = hier_attn(words, num_sentences=10)
    
    print("\nHierarchical Attention Example:")
    print(f"Word embeddings shape: {words.shape}")
    print(f"Document representation shape: {doc_repr.shape}")
    print("Two-level attention: words → sentences → document")
    
    return hier_attn


# Pattern 9: Attention Pooling
class AttentionPooling(layers.Layer):
    """Attention-based pooling for sequence aggregation."""
    
    def __init__(self, d_model):
        super().__init__()
        
        self.attention_vector = self.add_weight(
            shape=(d_model, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_vector'
        )
    
    def call(self, x):
        # x: (batch, seq_len, d_model)
        
        # Compute attention scores
        scores = tf.matmul(x, self.attention_vector)  # (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # Weighted sum
        pooled = tf.reduce_sum(x * attention_weights, axis=1)  # (batch, d_model)
        
        return pooled, attention_weights


def example_attention_pooling():
    """Example: Attention pooling."""
    d_model = 64
    
    attn_pool = AttentionPooling(d_model)
    
    x = tf.random.normal((2, 10, d_model))
    pooled, weights = attn_pool(x)
    
    print("\nAttention Pooling Example:")
    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print("Learns which positions are important for pooling")
    
    return attn_pool


# Pattern 10: Channel Attention (Squeeze-and-Excitation)
class ChannelAttention(layers.Layer):
    """Channel attention (Squeeze-and-Excitation)."""
    
    def __init__(self, reduction_ratio=16):
        super().__init__()
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
    
    def call(self, x):
        # x: (batch, height, width, channels)
        
        # Squeeze: global average pooling
        squeeze = self.global_pool(x)  # (batch, channels)
        
        # Excitation: FC layers
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        
        # Scale: broadcast and multiply
        excitation = tf.reshape(excitation, (-1, 1, 1, tf.shape(x)[-1]))
        scaled = x * excitation
        
        return scaled


def example_channel_attention():
    """Example: Channel attention."""
    channel_attn = ChannelAttention(reduction_ratio=16)
    
    # Feature map
    x = tf.random.normal((2, 32, 32, 256))
    output = channel_attn(x)
    
    print("\nChannel Attention (Squeeze-and-Excitation) Example:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Recalibrates channel-wise feature responses")
    print("Boosts informative channels, suppresses less useful ones")
    
    return channel_attn


if __name__ == "__main__":
    print("Attention Mechanisms Patterns\n" + "="*60)
    
    # Example 1: Additive Attention
    print("\n1. Additive (Bahdanau) Attention")
    attn1 = example_additive_attention()
    
    # Example 2: Multiplicative Attention
    print("\n2. Multiplicative (Luong) Attention")
    attn2 = example_multiplicative_attention()
    
    # Example 3: Scaled Dot-Product Attention
    print("\n3. Scaled Dot-Product Attention")
    attn3 = example_scaled_dot_product_attention()
    
    # Example 4: Multi-Head Attention
    print("\n4. Multi-Head Attention")
    attn4 = example_multi_head_attention()
    
    # Example 5: Self-Attention
    print("\n5. Self-Attention")
    attn5 = example_self_attention()
    
    # Example 6: Cross-Attention
    print("\n6. Cross-Attention")
    attn6 = example_cross_attention()
    
    # Example 7: Local Attention
    print("\n7. Local Attention")
    attn7 = example_local_attention()
    
    # Example 8: Hierarchical Attention
    print("\n8. Hierarchical Attention")
    attn8 = example_hierarchical_attention()
    
    # Example 9: Attention Pooling
    print("\n9. Attention Pooling")
    attn9 = example_attention_pooling()
    
    # Example 10: Channel Attention
    print("\n10. Channel Attention (Squeeze-and-Excitation)")
    attn10 = example_channel_attention()
    
    print("\n" + "="*60)
    print("Attention Mechanisms Best Practices:")
    print("1. Use scaled dot-product for efficiency")
    print("2. Multi-head attention captures diverse patterns")
    print("3. Add residual connections with attention")
    print("4. Apply layer normalization after attention")
    print("5. Use masking for padding and future tokens")
    print("6. Local attention for very long sequences")
    print("7. Channel attention boosts CNN performance")
    print("8. Visualize attention weights for interpretability")
    print("9. Dropout on attention weights prevents overfitting")
    print("10. Choose attention type based on task requirements")
