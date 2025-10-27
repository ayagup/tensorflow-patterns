"""
Autoencoder Pattern
Neural network that learns to compress and reconstruct data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("Autoencoder Pattern\n")

# Generate dummy data
X_train = np.random.random((1000, 28, 28, 1)).astype(np.float32)

# Example 1: Basic Autoencoder
print("Example 1: Basic Autoencoder")

# Encoder
encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_input)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
encoded = layers.Dense(32, activation='relu', name='encoding')(x)

# Decoder
x = layers.Dense(64, activation='relu')(encoded)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_output = layers.Reshape((28, 28, 1))(x)

# Full autoencoder
autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

autoencoder.summary()

# Train
history = autoencoder.fit(
    X_train, X_train,  # Input and target are the same
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Example 2: Convolutional Autoencoder
print("\nExample 2: Convolutional Autoencoder")

def build_conv_autoencoder(input_shape=(28, 28, 1)):
    """Build convolutional autoencoder."""
    # Encoder
    encoder_input = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(2, padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    model = keras.Model(encoder_input, decoded, name='conv_autoencoder')
    return model

conv_autoencoder = build_conv_autoencoder()
conv_autoencoder.compile(optimizer='adam', loss='mse')

print("Convolutional autoencoder for image reconstruction")

# Example 3: Denoising Autoencoder
print("\nExample 3: Denoising Autoencoder")

# Add noise to training data
noise_factor = 0.3
X_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_noisy = np.clip(X_noisy, 0., 1.)

# Train to denoise
denoising_autoencoder = build_conv_autoencoder()
denoising_autoencoder.compile(optimizer='adam', loss='mse')

history_denoise = denoising_autoencoder.fit(
    X_noisy, X_train,  # Input is noisy, target is clean
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("Denoising autoencoder learns to remove noise")

# Example 4: Variational Autoencoder (VAE)
print("\nExample 4: Variational Autoencoder (VAE)")

class Sampling(layers.Layer):
    """Sampling layer for VAE."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape=(28, 28, 1), latent_dim=2):
    """Build Variational Autoencoder."""
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(28 * 28, activation='sigmoid')(x)
    decoder_outputs = layers.Reshape((28, 28, 1))(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
    
    return encoder, decoder

vae_encoder, vae_decoder = build_vae(latent_dim=10)

# VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(vae_encoder, vae_decoder)
vae.compile(optimizer='adam')

print("VAE learns a probabilistic latent space")

# Example 5: Sparse Autoencoder
print("\nExample 5: Sparse Autoencoder")

def sparse_loss(z, sparsity_param=0.05):
    """Add sparsity constraint to activations."""
    p_hat = tf.reduce_mean(z, axis=0)
    kl_div = sparsity_param * tf.math.log(sparsity_param / p_hat) + \
             (1 - sparsity_param) * tf.math.log((1 - sparsity_param) / (1 - p_hat))
    return tf.reduce_sum(kl_div)

# Encoder with sparsity
encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_input)
x = layers.Dense(128, activation='relu')(x)
encoded = layers.Dense(64, activation='sigmoid')(x)  # Sigmoid for sparsity

# Decoder
x = layers.Dense(128, activation='relu')(encoded)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoded = layers.Reshape((28, 28, 1))(x)

sparse_autoencoder = keras.Model(encoder_input, decoded)

print("Sparse autoencoder encourages sparse representations")

# Example 6: Contractive Autoencoder
print("\nExample 6: Contractive Autoencoder")

class ContractiveAutoencoder(keras.Model):
    def __init__(self, encoding_dim=32):
        super(ContractiveAutoencoder, self).__init__()
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation='relu'),
            layers.Dense(encoding_dim, activation='relu')
        ])
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(28 * 28, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

contractive_ae = ContractiveAutoencoder()
print("Contractive autoencoder adds robustness through gradient penalty")

# Example 7: Using Encoder for Feature Extraction
print("\nExample 7: Using Encoder for Feature Extraction")

# Extract encoder
encoder = keras.Model(
    autoencoder.input,
    autoencoder.get_layer('encoding').output,
    name='encoder_only'
)

# Generate encodings
encodings = encoder.predict(X_train[:100])
print(f"Encoded shape: {encodings.shape}")

# Use encodings for clustering or classification
from tensorflow.keras import layers as downstream_layers

classification_model = keras.Sequential([
    encoder,
    downstream_layers.Dense(64, activation='relu'),
    downstream_layers.Dropout(0.5),
    downstream_layers.Dense(10, activation='softmax')
])

print("Encoder can be used as feature extractor for downstream tasks")

# Example 8: Reconstruction Quality
print("\nExample 8: Evaluating Reconstruction")

# Reconstruct samples
reconstructed = autoencoder.predict(X_train[:5])

# Calculate reconstruction error
mse = np.mean((X_train[:5] - reconstructed) ** 2, axis=(1, 2, 3))
print(f"Mean Squared Errors: {mse}")

print("\nAutoencoder Key Features:")
print("1. Unsupervised learning of compressed representations")
print("2. Useful for dimensionality reduction")
print("3. Can learn to denoise data")
print("4. VAE enables generation of new samples")
print("5. Encoder extracts meaningful features")
print("6. Applications: anomaly detection, compression, denoising")
print("7. Bottleneck forces learning of efficient encoding")
