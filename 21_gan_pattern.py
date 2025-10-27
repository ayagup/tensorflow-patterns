"""
GAN (Generative Adversarial Network) Pattern
Two networks (generator and discriminator) competing against each other.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("GAN Pattern\n")

# Hyperparameters
latent_dim = 100
img_shape = (28, 28, 1)

# Example 1: Basic GAN Architecture
print("Example 1: Basic GAN")

def build_generator(latent_dim):
    """Build generator network."""
    model = keras.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(momentum=0.8),
        
        layers.Dense(np.prod(img_shape), activation='tanh'),
        layers.Reshape(img_shape)
    ], name='generator')
    
    return model

def build_discriminator(img_shape):
    """Build discriminator network."""
    model = keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid')
    ], name='discriminator')
    
    return model

# Build generator and discriminator
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

generator.summary()
discriminator.summary()

# Example 2: Compile GAN components
print("\nExample 2: Compile GAN")

# Compile discriminator
discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build combined model (generator + discriminator)
# Freeze discriminator weights when training generator
discriminator.trainable = False

# GAN input (noise)
z = keras.Input(shape=(latent_dim,))
# Generate image from noise
img = generator(z)
# Discriminator determines validity
validity = discriminator(img)

# Combined model
combined = keras.Model(z, validity, name='gan')
combined.compile(
    optimizer=keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy'
)

print("Combined GAN model created")

# Example 3: GAN Training Loop
print("\nExample 3: GAN Training Loop")

def train_gan(generator, discriminator, combined, epochs, batch_size=32, sample_interval=50):
    """Train GAN."""
    
    # Generate dummy data (normally you'd load real images)
    X_train = np.random.random((1000, 28, 28, 1)).astype(np.float32)
    X_train = (X_train - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select random real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Train generator (wants discriminator to mistake fake as real)
        g_loss = combined.train_on_batch(noise, valid)
        
        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# Train for a few epochs
train_gan(generator, discriminator, combined, epochs=100, batch_size=32, sample_interval=20)

# Example 4: DCGAN (Deep Convolutional GAN)
print("\nExample 4: DCGAN")

def build_dcgan_generator(latent_dim):
    """Build DCGAN generator with transpose convolutions."""
    model = keras.Sequential([
        # Foundation for 7x7 image
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        layers.Reshape((7, 7, 256)),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        
        # Upsample to 14x14
        layers.Conv2DTranspose(128, 5, strides=2, padding='same'),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        
        # Upsample to 28x28
        layers.Conv2DTranspose(64, 5, strides=2, padding='same'),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        
        # Output layer
        layers.Conv2D(1, 5, padding='same', activation='tanh')
    ], name='dcgan_generator')
    
    return model

def build_dcgan_discriminator(img_shape):
    """Build DCGAN discriminator with strided convolutions."""
    model = keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ], name='dcgan_discriminator')
    
    return model

dcgan_generator = build_dcgan_generator(latent_dim)
dcgan_discriminator = build_dcgan_discriminator(img_shape)

print("DCGAN uses convolutional layers instead of dense layers")

# Example 5: Conditional GAN
print("\nExample 5: Conditional GAN")

num_classes = 10

def build_conditional_generator(latent_dim, num_classes):
    """Build conditional generator that takes class label as input."""
    # Noise input
    noise = keras.Input(shape=(latent_dim,))
    # Label input
    label = keras.Input(shape=(1,), dtype='int32')
    
    # Embed label
    label_embedding = layers.Embedding(num_classes, latent_dim)(label)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Concatenate noise and label
    model_input = layers.Concatenate()([noise, label_embedding])
    
    # Generator network
    x = layers.Dense(256)(model_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Dense(np.prod(img_shape), activation='tanh')(x)
    img = layers.Reshape(img_shape)(x)
    
    model = keras.Model([noise, label], img, name='conditional_generator')
    return model

conditional_gen = build_conditional_generator(latent_dim, num_classes)
print("Conditional GAN allows generating specific classes")

# Example 6: Wasserstein GAN loss
print("\nExample 6: Wasserstein GAN")

def wasserstein_loss(y_true, y_pred):
    """Wasserstein loss for WGAN."""
    return tf.reduce_mean(y_true * y_pred)

def build_wgan_critic(img_shape):
    """Build critic (not discriminator) for WGAN."""
    model = keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1)  # No sigmoid activation
    ], name='wgan_critic')
    
    return model

wgan_critic = build_wgan_critic(img_shape)
wgan_critic.compile(
    optimizer=keras.optimizers.RMSprop(0.00005),
    loss=wasserstein_loss
)

print("WGAN uses Wasserstein distance instead of JS divergence")

print("\nGAN Key Concepts:")
print("1. Generator creates fake samples from noise")
print("2. Discriminator distinguishes real from fake")
print("3. Minimax game: Generator tries to fool discriminator")
print("4. Training alternates between D and G updates")
print("5. Nash equilibrium when G generates realistic samples")
print("6. Challenges: mode collapse, training instability")
