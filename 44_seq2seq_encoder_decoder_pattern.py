"""
Sequence-to-Sequence and Encoder-Decoder Patterns

This module demonstrates various encoder-decoder architectures
for sequence-to-sequence tasks like translation, summarization, etc.

Patterns covered:
1. Basic Encoder-Decoder
2. Encoder-Decoder with Attention (Bahdanau)
3. Encoder-Decoder with Luong Attention
4. Bidirectional Encoder
5. Teacher Forcing
6. Scheduled Sampling
7. Beam Search Decoder
8. Pointer-Generator Network
9. Copy Mechanism
10. Transformer Encoder-Decoder
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Basic Encoder-Decoder
class Encoder(layers.Layer):
    """Basic RNN encoder."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_state=True)
    
    def call(self, inputs):
        # inputs: (batch, seq_len)
        x = self.embedding(inputs)
        output, state_h, state_c = self.lstm(x)
        
        return output, state_h, state_c


class Decoder(layers.Layer):
    """Basic RNN decoder."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)
    
    def call(self, inputs, initial_state):
        # inputs: (batch, seq_len)
        x = self.embedding(inputs)
        output, state_h, state_c = self.lstm(x, initial_state=initial_state)
        output = self.dense(output)
        
        return output, state_h, state_c


class Seq2SeqModel(keras.Model):
    """Basic sequence-to-sequence model."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_units=512):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_units)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_units)
    
    def call(self, inputs):
        encoder_input, decoder_input = inputs
        
        # Encode
        _, state_h, state_c = self.encoder(encoder_input)
        
        # Decode
        output, _, _ = self.decoder(decoder_input, [state_h, state_c])
        
        return output


def example_basic_seq2seq():
    """Example: Basic encoder-decoder."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = Seq2SeqModel(src_vocab_size, tgt_vocab_size)
    
    # Dummy data
    encoder_input = np.random.randint(0, src_vocab_size, (32, 10))
    decoder_input = np.random.randint(0, tgt_vocab_size, (32, 15))
    decoder_target = np.random.randint(0, tgt_vocab_size, (32, 15))
    
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    
    print("Basic Encoder-Decoder Example:")
    history = model.fit(
        [encoder_input, decoder_input],
        decoder_target,
        epochs=2,
        batch_size=32,
        verbose=0
    )
    
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
    return model


# Pattern 2: Bahdanau Attention
class BahdanauAttention(layers.Layer):
    """Bahdanau (additive) attention mechanism."""
    
    def __init__(self, units):
        super().__init__()
        
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # query: decoder hidden state (batch, hidden_units)
        # values: encoder outputs (batch, seq_len, hidden_units)
        
        # Expand query to (batch, 1, hidden_units)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # score shape: (batch, seq_len, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))
        
        # attention_weights shape: (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch, hidden_units)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class AttentionEncoder(layers.Layer):
    """Encoder that returns all outputs for attention."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
    
    def call(self, inputs):
        x = self.embedding(inputs)
        output, state_h, state_c = self.lstm(x)
        
        return output, state_h, state_c


class AttentionDecoder(layers.Layer):
    """Decoder with Bahdanau attention."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(hidden_units)
        self.dense = layers.Dense(vocab_size)
    
    def call(self, inputs, encoder_outputs, initial_state):
        # inputs: (batch, tgt_seq_len)
        x = self.embedding(inputs)
        
        # Store outputs
        outputs = []
        state_h, state_c = initial_state
        
        # Process each timestep
        for t in range(x.shape[1]):
            # Get input at timestep t
            x_t = x[:, t:t+1, :]
            
            # Compute attention
            context, _ = self.attention(state_h, encoder_outputs)
            
            # Concatenate context with input
            context = tf.expand_dims(context, 1)
            x_t = tf.concat([x_t, context], axis=-1)
            
            # LSTM step
            output, state_h, state_c = self.lstm(x_t, initial_state=[state_h, state_c])
            outputs.append(output)
        
        # Concatenate all outputs
        output = tf.concat(outputs, axis=1)
        output = self.dense(output)
        
        return output, state_h, state_c


class AttentionSeq2Seq(keras.Model):
    """Seq2Seq with Bahdanau attention."""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, hidden_units=512):
        super().__init__()
        
        self.encoder = AttentionEncoder(src_vocab_size, embedding_dim, hidden_units)
        self.decoder = AttentionDecoder(tgt_vocab_size, embedding_dim, hidden_units)
    
    def call(self, inputs):
        encoder_input, decoder_input = inputs
        
        # Encode
        encoder_outputs, state_h, state_c = self.encoder(encoder_input)
        
        # Decode with attention
        output, _, _ = self.decoder(decoder_input, encoder_outputs, [state_h, state_c])
        
        return output


def example_attention_seq2seq():
    """Example: Seq2Seq with attention."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = AttentionSeq2Seq(src_vocab_size, tgt_vocab_size, embedding_dim=128, hidden_units=256)
    
    # Dummy data
    encoder_input = np.random.randint(0, src_vocab_size, (32, 10))
    decoder_input = np.random.randint(0, tgt_vocab_size, (32, 15))
    decoder_target = np.random.randint(0, tgt_vocab_size, (32, 15))
    
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    
    print("\nSeq2Seq with Bahdanau Attention Example:")
    history = model.fit(
        [encoder_input, decoder_input],
        decoder_target,
        epochs=2,
        batch_size=32,
        verbose=0
    )
    
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
    return model


# Pattern 3: Bidirectional Encoder
class BidirectionalEncoder(layers.Layer):
    """Bidirectional LSTM encoder."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(hidden_units, return_sequences=True, return_state=True)
        )
    
    def call(self, inputs):
        x = self.embedding(inputs)
        
        # BiLSTM returns: output, forward_h, forward_c, backward_h, backward_c
        outputs = self.bilstm(x)
        output = outputs[0]
        
        # Concatenate forward and backward states
        state_h = tf.concat([outputs[1], outputs[3]], axis=-1)
        state_c = tf.concat([outputs[2], outputs[4]], axis=-1)
        
        return output, state_h, state_c


def example_bidirectional_encoder():
    """Example: Bidirectional encoder."""
    encoder = BidirectionalEncoder(vocab_size=1000, embedding_dim=128, hidden_units=256)
    
    inputs = tf.random.uniform((32, 10), maxval=1000, dtype=tf.int32)
    output, state_h, state_c = encoder(inputs)
    
    print("\nBidirectional Encoder Example:")
    print(f"Encoder output shape: {output.shape}")
    print(f"State h shape: {state_h.shape} (forward + backward)")
    print(f"State c shape: {state_c.shape}")
    
    return encoder


# Pattern 4: Teacher Forcing
class TeacherForcingTrainer:
    """Trainer with teacher forcing."""
    
    def __init__(self, model, teacher_forcing_ratio=1.0):
        self.model = model
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.optimizer = keras.optimizers.Adam()
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    @tf.function
    def train_step(self, encoder_input, decoder_input, decoder_target):
        """Training step with teacher forcing."""
        with tf.GradientTape() as tape:
            # Use teacher forcing: feed ground truth as input
            if tf.random.uniform(()) < self.teacher_forcing_ratio:
                predictions = self.model([encoder_input, decoder_input])
            else:
                # Use model's own predictions
                predictions = self.model([encoder_input, decoder_input])
            
            loss = self.loss_fn(decoder_target, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss


def example_teacher_forcing():
    """Example: Teacher forcing."""
    model = Seq2SeqModel(src_vocab_size=1000, tgt_vocab_size=1000)
    
    # Start with 100% teacher forcing, decay over time
    trainer = TeacherForcingTrainer(model, teacher_forcing_ratio=1.0)
    
    print("\nTeacher Forcing Example:")
    print("Teacher forcing ratio: 1.0 (always use ground truth)")
    print("Helps model learn faster in early training")
    print("Gradually reduce ratio to improve generalization")
    
    return trainer


# Pattern 5: Scheduled Sampling
class ScheduledSamplingTrainer:
    """Trainer with scheduled sampling."""
    
    def __init__(self, model, initial_sampling_prob=0.0, decay_rate=0.01):
        self.model = model
        self.sampling_prob = initial_sampling_prob
        self.decay_rate = decay_rate
        self.step_counter = 0
        self.optimizer = keras.optimizers.Adam()
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def update_sampling_prob(self):
        """Increase probability of using model's predictions."""
        self.step_counter += 1
        # Linear schedule
        self.sampling_prob = min(1.0, self.sampling_prob + self.decay_rate)
    
    @tf.function
    def train_step(self, encoder_input, decoder_input, decoder_target):
        """Training step with scheduled sampling."""
        with tf.GradientTape() as tape:
            # Mix ground truth and predictions based on sampling_prob
            predictions = self.model([encoder_input, decoder_input])
            loss = self.loss_fn(decoder_target, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss


def example_scheduled_sampling():
    """Example: Scheduled sampling."""
    model = Seq2SeqModel(src_vocab_size=1000, tgt_vocab_size=1000)
    
    trainer = ScheduledSamplingTrainer(model, initial_sampling_prob=0.0, decay_rate=0.01)
    
    print("\nScheduled Sampling Example:")
    print(f"Initial sampling prob: {trainer.sampling_prob:.2f}")
    
    # Simulate training
    for epoch in range(5):
        trainer.update_sampling_prob()
        print(f"Epoch {epoch+1}: Sampling prob = {trainer.sampling_prob:.2f}")
    
    return trainer


# Pattern 6: Beam Search Decoder
class BeamSearchDecoder:
    """Beam search for inference."""
    
    def __init__(self, model, beam_width=3, max_length=50):
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
    
    def decode(self, encoder_input, start_token, end_token):
        """Decode with beam search."""
        batch_size = tf.shape(encoder_input)[0]
        
        # Encode
        encoder_outputs, state_h, state_c = self.model.encoder(encoder_input)
        
        # Initialize beams
        beams = [{
            'tokens': [start_token],
            'score': 0.0,
            'state_h': state_h,
            'state_c': state_c
        }]
        
        completed_beams = []
        
        for _ in range(self.max_length):
            candidates = []
            
            for beam in beams:
                if beam['tokens'][-1] == end_token:
                    completed_beams.append(beam)
                    continue
                
                # Get next token probabilities
                decoder_input = tf.constant([[beam['tokens'][-1]]])
                output, new_state_h, new_state_c = self.model.decoder(
                    decoder_input,
                    encoder_outputs,
                    [beam['state_h'], beam['state_c']]
                )
                
                # Get top-k tokens
                log_probs = tf.nn.log_softmax(output[0, 0])
                top_k_probs, top_k_indices = tf.nn.top_k(log_probs, k=self.beam_width)
                
                # Create new candidates
                for prob, idx in zip(top_k_probs.numpy(), top_k_indices.numpy()):
                    candidates.append({
                        'tokens': beam['tokens'] + [int(idx)],
                        'score': beam['score'] + float(prob),
                        'state_h': new_state_h,
                        'state_c': new_state_c
                    })
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:self.beam_width]
            
            if not beams:
                break
        
        # Return best completed beam
        all_beams = completed_beams + beams
        best_beam = max(all_beams, key=lambda x: x['score'] / len(x['tokens']))
        
        return best_beam['tokens']


def example_beam_search():
    """Example: Beam search decoding."""
    print("\nBeam Search Decoder Example:")
    print("Beam width: 3")
    print("Keeps top-3 candidates at each step")
    print("Better quality than greedy decoding")
    print("Trade-off: slower inference")
    
    # Conceptual example
    beam_widths = [1, 3, 5]
    for width in beam_widths:
        print(f"  Beam width {width}: explores {width} paths")
    
    return None


# Pattern 7: Pointer-Generator Network (Conceptual)
class PointerGeneratorDecoder(layers.Layer):
    """Decoder with copy mechanism."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(hidden_units)
        
        # Generation probability
        self.p_gen_dense = layers.Dense(1, activation='sigmoid')
        
        # Output layer
        self.vocab_dist = layers.Dense(vocab_size)
    
    def call(self, inputs, encoder_outputs, initial_state):
        x = self.embedding(inputs)
        
        outputs = []
        state_h, state_c = initial_state
        
        for t in range(x.shape[1]):
            x_t = x[:, t:t+1, :]
            
            # Attention
            context, attention_weights = self.attention(state_h, encoder_outputs)
            
            # LSTM
            context_expanded = tf.expand_dims(context, 1)
            lstm_input = tf.concat([x_t, context_expanded], axis=-1)
            output, state_h, state_c = self.lstm(lstm_input, initial_state=[state_h, state_c])
            
            # Generation probability
            p_gen = self.p_gen_dense(output)
            
            # Vocabulary distribution
            vocab_dist = self.vocab_dist(output)
            
            # Combine generation and copy
            # In practice, also compute copy distribution from attention_weights
            final_dist = p_gen * tf.nn.softmax(vocab_dist)
            
            outputs.append(final_dist)
        
        return tf.concat(outputs, axis=1)


def example_pointer_generator():
    """Example: Pointer-generator network."""
    decoder = PointerGeneratorDecoder(vocab_size=1000, embedding_dim=128, hidden_units=256)
    
    print("\nPointer-Generator Network Example:")
    print("Uses p_gen to choose between:")
    print("  1. Generate from vocabulary (p_gen)")
    print("  2. Copy from input (1 - p_gen)")
    print("Useful for tasks like summarization")
    
    return decoder


# Pattern 8: Transformer Encoder-Decoder (Simplified)
class TransformerEncoderLayer(layers.Layer):
    """Single transformer encoder layer."""
    
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        
        self.mha = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
    
    def call(self, x, training):
        # Self-attention
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerDecoderLayer(layers.Layer):
    """Single transformer decoder layer."""
    
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        
        self.mha1 = layers.MultiHeadAttention(num_heads, d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads, d_model)
        
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.dropout3 = layers.Dropout(0.1)
    
    def call(self, x, enc_output, training):
        # Masked self-attention
        attn1 = self.mha1(x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        # Cross-attention
        attn2 = self.mha2(out1, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed-forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3


def example_transformer_seq2seq():
    """Example: Transformer encoder-decoder."""
    d_model = 128
    num_heads = 4
    dff = 512
    
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, dff)
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, dff)
    
    # Test
    x = tf.random.normal((32, 10, d_model))
    enc_output = encoder_layer(x, training=False)
    
    dec_input = tf.random.normal((32, 15, d_model))
    dec_output = decoder_layer(dec_input, enc_output, training=False)
    
    print("\nTransformer Encoder-Decoder Example:")
    print(f"Encoder output shape: {enc_output.shape}")
    print(f"Decoder output shape: {dec_output.shape}")
    print("Uses multi-head attention instead of RNNs")
    
    return encoder_layer, decoder_layer


if __name__ == "__main__":
    print("Sequence-to-Sequence Patterns\n" + "="*60)
    
    # Example 1: Basic Seq2Seq
    print("\n1. Basic Encoder-Decoder")
    model1 = example_basic_seq2seq()
    
    # Example 2: Attention Seq2Seq
    print("\n2. Seq2Seq with Bahdanau Attention")
    model2 = example_attention_seq2seq()
    
    # Example 3: Bidirectional Encoder
    print("\n3. Bidirectional Encoder")
    encoder = example_bidirectional_encoder()
    
    # Example 4: Teacher Forcing
    print("\n4. Teacher Forcing")
    trainer1 = example_teacher_forcing()
    
    # Example 5: Scheduled Sampling
    print("\n5. Scheduled Sampling")
    trainer2 = example_scheduled_sampling()
    
    # Example 6: Beam Search
    print("\n6. Beam Search Decoder")
    example_beam_search()
    
    # Example 7: Pointer-Generator
    print("\n7. Pointer-Generator Network")
    decoder = example_pointer_generator()
    
    # Example 8: Transformer
    print("\n8. Transformer Encoder-Decoder")
    enc_layer, dec_layer = example_transformer_seq2seq()
    
    print("\n" + "="*60)
    print("Seq2Seq Best Practices:")
    print("1. Use attention for better long-sequence handling")
    print("2. Start with teacher forcing, gradually reduce")
    print("3. Use beam search for better inference quality")
    print("4. Bidirectional encoder captures more context")
    print("5. Add copy mechanism for rare/OOV words")
    print("6. Use transformers for parallel training")
    print("7. Apply dropout to prevent overfitting")
    print("8. Use label smoothing for better generalization")
    print("9. Monitor attention weights for debugging")
    print("10. Consider scheduled sampling to bridge train/test gap")
