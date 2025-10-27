"""
Learning Rate Scheduling Pattern
Adjust learning rate during training for better convergence.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt

print("Learning Rate Scheduling Pattern\n")

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1))

# Create model
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Example 1: Exponential Decay
print("Example 1: Exponential Decay Schedule")

initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

model1 = create_model()
model1.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Initial LR: {initial_learning_rate}")
print("LR will decay exponentially every 100 steps")

# Example 2: Step Decay with Callback
print("\nExample 2: Step Decay with LearningRateScheduler")

def step_decay(epoch):
    """Reduce learning rate by factor every few epochs."""
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 5
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

lr_scheduler = callbacks.LearningRateScheduler(step_decay, verbose=1)

model2 = create_model()
model2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history2 = model2.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[lr_scheduler],
    verbose=1
)

# Example 3: Reduce on Plateau
print("\nExample 3: Reduce Learning Rate on Plateau")

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce by half
    patience=3,  # Wait 3 epochs
    min_lr=1e-7,
    verbose=1,
    mode='min'
)

model3 = create_model()
model3.compile(
    optimizer=optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history3 = model3.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr],
    verbose=0
)

print("LR reduced when validation loss plateaus")

# Example 4: Cosine Decay
print("\nExample 4: Cosine Decay Schedule")

cosine_decay = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    alpha=0.0  # Minimum learning rate
)

model4 = create_model()
model4.compile(
    optimizer=optimizers.Adam(learning_rate=cosine_decay),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Cosine decay smoothly reduces learning rate")

# Example 5: Cosine Decay with Restarts
print("\nExample 5: Cosine Decay with Warm Restarts")

cosine_restart = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.01,
    first_decay_steps=100,
    t_mul=2.0,  # Double the period after each restart
    m_mul=0.9,  # Reduce max LR by 10% after each restart
    alpha=0.0
)

model5 = create_model()
model5.compile(
    optimizer=optimizers.Adam(learning_rate=cosine_restart),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Warm restarts help escape local minima")

# Example 6: Polynomial Decay
print("\nExample 6: Polynomial Decay Schedule")

polynomial_decay = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    end_learning_rate=0.0001,
    power=1.0  # Linear decay (use 2.0 for quadratic)
)

model6 = create_model()
model6.compile(
    optimizer=optimizers.Adam(learning_rate=polynomial_decay),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Polynomial decay provides smooth transition")

# Example 7: Piecewise Constant Decay
print("\nExample 7: Piecewise Constant Decay")

boundaries = [200, 400, 600]
values = [0.01, 0.005, 0.001, 0.0001]

piecewise_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values
)

model7 = create_model()
model7.compile(
    optimizer=optimizers.Adam(learning_rate=piecewise_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Piecewise: manually set LR at specific steps")

# Example 8: Cyclical Learning Rate
print("\nExample 8: Cyclical Learning Rate (Custom)")

class CyclicLR(callbacks.Callback):
    """Custom cyclical learning rate callback."""
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=200, mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0
        self.history = {}
    
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (0.99999 ** self.clr_iterations)
        
        return lr
    
    def on_train_begin(self, logs=None):
        keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
    
    def on_batch_end(self, batch, logs=None):
        self.clr_iterations += 1
        lr = self.clr()
        keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.setdefault('lr', []).append(lr)

cyclic_lr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=100, mode='triangular')

model8 = create_model()
model8.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Cyclical LR oscillates between bounds")

# Example 9: Warm-up Schedule
print("\nExample 9: Warm-up Learning Rate")

class WarmUpSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warm-up."""
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps):
        super(WarmUpSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        # Warm-up phase
        warmup_lr = (self.initial_learning_rate / self.warmup_steps) * step
        
        # Decay phase
        decay_lr = self.initial_learning_rate * (
            self.decay_steps / (self.decay_steps + step - self.warmup_steps)
        )
        
        # Choose based on step
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

warmup_schedule = WarmUpSchedule(
    initial_learning_rate=0.01,
    warmup_steps=100,
    decay_steps=1000
)

model9 = create_model()
model9.compile(
    optimizer=optimizers.Adam(learning_rate=warmup_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Warm-up gradually increases LR from 0")

# Example 10: Learning Rate Finder
print("\nExample 10: Learning Rate Finder")

class LRFinder(callbacks.Callback):
    """Find optimal learning rate by exponentially increasing it."""
    def __init__(self, min_lr=1e-7, max_lr=1.0, steps_per_epoch=None):
        super(LRFinder, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.lrs = []
        self.losses = []
    
    def on_train_begin(self, logs=None):
        self.step = 0
        keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.step / self.steps_per_epoch)
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
        self.lrs.append(lr)
        self.losses.append(logs['loss'])
        
        # Stop if loss explodes
        if logs['loss'] > self.losses[0] * 4:
            self.model.stop_training = True

# Use LR finder
lr_finder = LRFinder(min_lr=1e-5, max_lr=1.0, steps_per_epoch=len(X_train) // 32)

model10 = create_model()
model10.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("LR finder helps identify optimal learning rate range")

# Example 11: Visualize Learning Rate Schedules
print("\nExample 11: Visualizing LR Schedules")

steps = np.arange(0, 1000)

# Exponential decay
exp_lrs = [keras.optimizers.schedules.ExponentialDecay(0.01, 100, 0.96)(step).numpy() for step in steps]

# Cosine decay
cos_lrs = [keras.optimizers.schedules.CosineDecay(0.01, 1000)(step).numpy() for step in steps]

# Polynomial decay
poly_lrs = [keras.optimizers.schedules.PolynomialDecay(0.01, 1000, 0.0001)(step).numpy() for step in steps]

print("Different schedules produce different LR curves")

print("\nLearning Rate Scheduling Best Practices:")
print("""
1. Exponential Decay:
   - Good for most tasks
   - Decay rate: 0.9-0.99

2. Step Decay:
   - Simple and effective
   - Reduce by 0.5 or 0.1 every 5-10 epochs

3. Reduce on Plateau:
   - Adaptive to training
   - Factor: 0.5, Patience: 3-5 epochs

4. Cosine Decay:
   - Smooth reduction
   - Good for long training

5. Warm Restarts:
   - Helps escape local minima
   - Use for complex loss landscapes

6. Cyclical LR:
   - Faster convergence
   - Base: 0.001, Max: 0.006

7. Warm-up:
   - Essential for large batch sizes
   - 100-1000 steps typically

8. LR Finder:
   - Find optimal LR before training
   - Look for steepest descent

General Guidelines:
- Start with ReduceLROnPlateau for simplicity
- Use warm-up for transformers
- Cosine decay for vision models
- Cyclical LR for fast experimentation
- Always monitor loss curves
""")
