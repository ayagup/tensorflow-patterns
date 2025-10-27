"""
Generative Models and GANs Patterns

This module demonstrates advanced patterns for generative models,
including various GAN architectures and training techniques.

Patterns covered:
1. Basic GAN
2. Conditional GAN (cGAN)
3. Wasserstein GAN (WGAN)
4. WGAN with Gradient Penalty (WGAN-GP)
5. Progressive GAN
6. StyleGAN-inspired Architecture
7. CycleGAN for Unpaired Translation
8. Variational Autoencoder (VAE)
9. VAE-GAN Hybrid
10. GAN Training Tricks and Stabilization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Basic GAN
class Generator(keras.Model):
    """Basic generator network."""
    
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')
        ])
    
    def call(self, inputs):
        return self.model(inputs)


class Discriminator(keras.Model):
    """Basic discriminator network."""
    
    def __init__(self, input_dim=784):
        super().__init__()
        
        self.model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs):
        return self.model(inputs)


class BasicGAN:
    """Basic GAN with training loop."""
    
    def __init__(self, latent_dim=100, data_dim=784):
        self.latent_dim = latent_dim
        
        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        
        self.g_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
        
        self.loss_fn = keras.losses.BinaryCrossentropy()
    
    @tf.function
    def train_step(self, real_images, batch_size):
        # Generate random noise
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        # Train discriminator
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            real_labels = tf.ones_like(real_output)
            fake_labels = tf.zeros_like(fake_output)
            
            d_loss = self.loss_fn(real_labels, real_output) + \
                     self.loss_fn(fake_labels, fake_output)
        
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        # Train generator
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        return d_loss, g_loss


def example_basic_gan():
    """Example: Basic GAN."""
    gan = BasicGAN(latent_dim=100, data_dim=784)
    
    # Create dummy data
    real_data = np.random.randn(1000, 784)
    
    print("Basic GAN Example:")
    
    for epoch in range(3):
        # Train on batches
        batch_size = 32
        num_batches = len(real_data) // batch_size
        
        d_losses, g_losses = [], []
        
        for i in range(num_batches):
            batch = real_data[i*batch_size:(i+1)*batch_size]
            d_loss, g_loss = gan.train_step(batch, batch_size)
            
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
        
        print(f"Epoch {epoch+1}: D Loss = {np.mean(d_losses):.4f}, "
              f"G Loss = {np.mean(g_losses):.4f}")
    
    return gan


# Pattern 2: Conditional GAN
class ConditionalGenerator(keras.Model):
    """Generator conditioned on class labels."""
    
    def __init__(self, latent_dim=100, num_classes=10, output_dim=784):
        super().__init__()
        
        self.label_embedding = layers.Embedding(num_classes, latent_dim)
        
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')
        ])
    
    def call(self, inputs):
        noise, labels = inputs
        
        # Embed labels and concatenate with noise
        label_embed = self.label_embedding(labels)
        model_input = tf.concat([noise, label_embed], axis=-1)
        
        return self.model(model_input)


class ConditionalDiscriminator(keras.Model):
    """Discriminator conditioned on class labels."""
    
    def __init__(self, num_classes=10, input_dim=784):
        super().__init__()
        
        self.label_embedding = layers.Embedding(num_classes, input_dim)
        
        self.model = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs):
        images, labels = inputs
        
        # Embed labels and concatenate with images
        label_embed = self.label_embedding(labels)
        model_input = tf.concat([images, label_embed], axis=-1)
        
        return self.model(model_input)


def example_conditional_gan():
    """Example: Conditional GAN."""
    latent_dim = 100
    num_classes = 10
    
    generator = ConditionalGenerator(latent_dim, num_classes)
    discriminator = ConditionalDiscriminator(num_classes)
    
    # Generate samples for specific class
    noise = tf.random.normal((5, latent_dim))
    labels = tf.constant([3, 3, 3, 3, 3])  # Generate class 3
    
    generated = generator([noise, labels])
    
    print("\nConditional GAN Example:")
    print(f"Generated samples shape: {generated.shape}")
    print(f"All samples are for class 3")
    
    return generator, discriminator


# Pattern 3: Wasserstein GAN (WGAN)
class WGANCritic(keras.Model):
    """Critic network for WGAN (no sigmoid)."""
    
    def __init__(self, input_dim=784):
        super().__init__()
        
        self.model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)  # No activation
        ])
    
    def call(self, inputs):
        return self.model(inputs)


class WGAN:
    """Wasserstein GAN with weight clipping."""
    
    def __init__(self, latent_dim=100, data_dim=784, clip_value=0.01):
        self.latent_dim = latent_dim
        self.clip_value = clip_value
        
        self.generator = Generator(latent_dim, data_dim)
        self.critic = WGANCritic(data_dim)
        
        self.g_optimizer = keras.optimizers.RMSprop(0.00005)
        self.c_optimizer = keras.optimizers.RMSprop(0.00005)
    
    @tf.function
    def train_step(self, real_images, batch_size, n_critic=5):
        # Train critic multiple times
        for _ in range(n_critic):
            noise = tf.random.normal((batch_size, self.latent_dim))
            
            with tf.GradientTape() as tape:
                fake_images = self.generator(noise, training=True)
                
                real_output = self.critic(real_images, training=True)
                fake_output = self.critic(fake_images, training=True)
                
                # Wasserstein loss
                c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            
            c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradients, self.critic.trainable_variables)
            )
            
            # Clip weights
            for var in self.critic.trainable_variables:
                var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))
        
        # Train generator
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.critic(fake_images, training=True)
            
            g_loss = -tf.reduce_mean(fake_output)
        
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        return c_loss, g_loss


def example_wgan():
    """Example: Wasserstein GAN."""
    wgan = WGAN(latent_dim=100, data_dim=784)
    
    real_data = np.random.randn(1000, 784)
    
    print("\nWasserstein GAN Example:")
    
    for epoch in range(3):
        batch_size = 32
        num_batches = len(real_data) // batch_size
        
        c_losses, g_losses = [], []
        
        for i in range(num_batches):
            batch = real_data[i*batch_size:(i+1)*batch_size]
            c_loss, g_loss = wgan.train_step(batch, batch_size)
            
            c_losses.append(c_loss.numpy())
            g_losses.append(g_loss.numpy())
        
        print(f"Epoch {epoch+1}: C Loss = {np.mean(c_losses):.4f}, "
              f"G Loss = {np.mean(g_losses):.4f}")
    
    return wgan


# Pattern 4: WGAN with Gradient Penalty
def gradient_penalty(critic, real_images, fake_images):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = tf.shape(real_images)[0]
    
    # Random interpolation
    alpha = tf.random.uniform((batch_size, 1))
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)
    
    gradients = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    
    return gp


class WGANGP:
    """WGAN with gradient penalty."""
    
    def __init__(self, latent_dim=100, data_dim=784, gp_weight=10.0):
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        
        self.generator = Generator(latent_dim, data_dim)
        self.critic = WGANCritic(data_dim)
        
        self.g_optimizer = keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.9)
        self.c_optimizer = keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.9)
    
    @tf.function
    def train_step(self, real_images, batch_size):
        # Train critic
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            
            real_output = self.critic(real_images, training=True)
            fake_output = self.critic(fake_images, training=True)
            
            # Wasserstein loss + gradient penalty
            c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = gradient_penalty(self.critic, real_images, fake_images)
            c_loss += self.gp_weight * gp
        
        c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(
            zip(c_gradients, self.critic.trainable_variables)
        )
        
        # Train generator
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.critic(fake_images, training=True)
            
            g_loss = -tf.reduce_mean(fake_output)
        
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        return c_loss, g_loss


def example_wgan_gp():
    """Example: WGAN with gradient penalty."""
    wgan_gp = WGANGP(latent_dim=100, data_dim=784)
    
    real_data = np.random.randn(100, 784)
    
    print("\nWGAN-GP Example:")
    
    batch_size = 32
    c_loss, g_loss = wgan_gp.train_step(real_data[:batch_size], batch_size)
    
    print(f"C Loss: {c_loss.numpy():.4f}, G Loss: {g_loss.numpy():.4f}")
    
    return wgan_gp


# Pattern 5: Progressive GAN Building Block
class ProgressiveBlock(layers.Layer):
    """Progressive GAN block that can be added during training."""
    
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class ProgressiveGenerator(keras.Model):
    """Progressive generator that grows during training."""
    
    def __init__(self, latent_dim=512, initial_resolution=4):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.initial_resolution = initial_resolution
        
        # Initial block
        self.initial = keras.Sequential([
            layers.Dense(initial_resolution * initial_resolution * 512),
            layers.Reshape((initial_resolution, initial_resolution, 512)),
            layers.Conv2D(512, 3, padding='same', activation='relu')
        ])
        
        # Progressive blocks
        self.blocks = []
        self.to_rgb = []  # Layers to convert to RGB
    
    def add_block(self, filters):
        """Add a new resolution block."""
        block = ProgressiveBlock(filters)
        to_rgb = layers.Conv2D(3, 1, padding='same', activation='tanh')
        
        self.blocks.append(block)
        self.to_rgb.append(to_rgb)
    
    def call(self, inputs, alpha=1.0):
        """
        Forward pass with fade-in.
        alpha: 0 to 1, controls fade-in of new layer
        """
        x = self.initial(inputs)
        
        for i, block in enumerate(self.blocks[:-1]):
            x = block(x)
            x = layers.UpSampling2D()(x)
        
        if len(self.blocks) > 0:
            # Old path (upsampled previous RGB)
            old_rgb = self.to_rgb[-2](x) if len(self.to_rgb) > 1 else None
            old_rgb = layers.UpSampling2D()(old_rgb) if old_rgb is not None else None
            
            # New path
            x = self.blocks[-1](x)
            new_rgb = self.to_rgb[-1](x)
            
            # Fade in
            if old_rgb is not None:
                x = alpha * new_rgb + (1 - alpha) * old_rgb
            else:
                x = new_rgb
        else:
            x = self.to_rgb[0](x) if self.to_rgb else x
        
        return x


def example_progressive_gan():
    """Example: Progressive GAN architecture."""
    generator = ProgressiveGenerator(latent_dim=512, initial_resolution=4)
    
    # Add blocks for increasing resolutions
    generator.add_block(256)  # 8x8
    generator.add_block(128)  # 16x16
    
    print("\nProgressive GAN Example:")
    
    # Generate at different alphas (fade-in)
    noise = tf.random.normal((1, 512))
    
    for alpha in [0.0, 0.5, 1.0]:
        output = generator(noise, alpha=alpha)
        print(f"Alpha {alpha}: Output shape = {output.shape}")
    
    return generator


# Pattern 6: Variational Autoencoder (VAE)
class VAE(keras.Model):
    """Variational Autoencoder."""
    
    def __init__(self, latent_dim=32, input_dim=784):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dense(256, activation='relu')
        ])
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
    
    def encode(self, inputs):
        """Encode inputs to latent parameters."""
        h = self.encoder(inputs)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        """Reparameterization trick."""
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal((batch_size, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        """Decode latent samples."""
        return self.decoder(z)
    
    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        
        return reconstructed


def example_vae():
    """Example: Variational Autoencoder."""
    vae = VAE(latent_dim=32, input_dim=784)
    
    # Compile
    vae.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    
    # Train on dummy data
    data = np.random.rand(1000, 784)
    
    print("\nVariational Autoencoder Example:")
    history = vae.fit(data, data, epochs=3, batch_size=32, verbose=0)
    
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
    # Generate new samples
    z = tf.random.normal((5, 32))
    generated = vae.decode(z)
    print(f"Generated samples shape: {generated.shape}")
    
    return vae


# Pattern 7: CycleGAN Components
class ResidualBlock(layers.Layer):
    """Residual block for CycleGAN generator."""
    
    def __init__(self, filters):
        super().__init__()
        
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return inputs + x


class CycleGANGenerator(keras.Model):
    """CycleGAN generator with residual blocks."""
    
    def __init__(self, num_residual_blocks=9):
        super().__init__()
        
        # Downsampling
        self.down = keras.Sequential([
            layers.Conv2D(64, 7, padding='same', activation='relu'),
            layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')
        ])
        
        # Residual blocks
        self.residual = keras.Sequential([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.up = keras.Sequential([
            layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(3, 7, padding='same', activation='tanh')
        ])
    
    def call(self, inputs):
        x = self.down(inputs)
        x = self.residual(x)
        x = self.up(x)
        return x


def example_cyclegan():
    """Example: CycleGAN generator architecture."""
    generator = CycleGANGenerator(num_residual_blocks=6)
    
    # Generate translated image
    input_image = tf.random.normal((1, 256, 256, 3))
    output = generator(input_image)
    
    print("\nCycleGAN Example:")
    print(f"Input shape: {input_image.shape}")
    print(f"Output shape: {output.shape}")
    print("Generator translates between two domains")
    
    return generator


if __name__ == "__main__":
    print("Generative Models and GANs Patterns\n" + "="*60)
    
    # Example 1: Basic GAN
    print("\n1. Basic GAN")
    gan = example_basic_gan()
    
    # Example 2: Conditional GAN
    print("\n2. Conditional GAN")
    cgan_g, cgan_d = example_conditional_gan()
    
    # Example 3: WGAN
    print("\n3. Wasserstein GAN")
    wgan = example_wgan()
    
    # Example 4: WGAN-GP
    print("\n4. WGAN with Gradient Penalty")
    wgan_gp = example_wgan_gp()
    
    # Example 5: Progressive GAN
    print("\n5. Progressive GAN")
    prog_gan = example_progressive_gan()
    
    # Example 6: VAE
    print("\n6. Variational Autoencoder")
    vae = example_vae()
    
    # Example 7: CycleGAN
    print("\n7. CycleGAN")
    cyclegan = example_cyclegan()
    
    print("\n" + "="*60)
    print("GAN Training Best Practices:")
    print("1. Use label smoothing for discriminator")
    print("2. Train discriminator more than generator")
    print("3. Use batch normalization in generator")
    print("4. Avoid sparse gradients (ReLU in discriminator)")
    print("5. Use Adam optimizer with beta1=0.5")
    print("6. Add noise to discriminator inputs")
    print("7. Use Wasserstein loss for stability")
    print("8. Monitor generated samples visually")
    print("9. Use gradient penalty instead of weight clipping")
    print("10. Start with low learning rates")
