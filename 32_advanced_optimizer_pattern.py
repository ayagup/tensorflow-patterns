"""
Advanced Optimizer Patterns

This module demonstrates advanced optimization techniques and custom optimizer
patterns in TensorFlow.

Patterns covered:
1. Custom Optimizer Implementation
2. Lookahead Optimizer
3. RAdam (Rectified Adam)
4. AdaBelief Optimizer
5. Gradient Centralization
6. SAM (Sharpness Aware Minimization)
7. LAMB Optimizer
8. Learning Rate Warmup with Cosine Decay
9. Gradient Accumulation
10. Mixed Optimizer Strategy
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


# Pattern 1: Custom Optimizer Implementation
class CustomSGD(keras.optimizers.Optimizer):
    """Custom SGD optimizer with momentum."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9, name="CustomSGD", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", momentum)
    
    def _create_slots(self, var_list):
        # Create momentum variables
        for var in var_list:
            self.add_slot(var, "momentum")
    
    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("learning_rate")
        momentum = self._get_hyper("momentum")
        
        momentum_var = self.get_slot(var, "momentum")
        
        # Update momentum: m = momentum * m + grad
        momentum_var.assign(momentum * momentum_var + grad)
        
        # Update variable: var = var - lr * m
        var.assign_sub(lr * momentum_var)
    
    def _resource_apply_sparse(self, grad, var, indices):
        # Sparse update implementation
        return self._resource_apply_dense(grad, var)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
        })
        return config


def example_custom_optimizer():
    """Example: Using custom optimizer."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    optimizer = CustomSGD(learning_rate=0.01, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Custom Optimizer Model:")
    model.summary()
    return model


# Pattern 2: Lookahead Optimizer
class Lookahead(keras.optimizers.Optimizer):
    """Lookahead optimizer wrapper."""
    
    def __init__(self, optimizer, sync_period=5, slow_step_size=0.5, name="Lookahead", **kwargs):
        super().__init__(name, **kwargs)
        self.optimizer = optimizer
        self.sync_period = sync_period
        self.slow_step_size = slow_step_size
        self._counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    
    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)
        for var in var_list:
            self.add_slot(var, "slow")
            self.get_slot(var, "slow").assign(var)
    
    def _resource_apply_dense(self, grad, var):
        # Fast weights update
        self.optimizer._resource_apply_dense(grad, var)
        
        # Slow weights update
        self._counter.assign_add(1)
        
        def sync_slow_weights():
            slow_var = self.get_slot(var, "slow")
            # slow = slow + slow_step_size * (fast - slow)
            diff = var - slow_var
            slow_var.assign_add(self.slow_step_size * diff)
            var.assign(slow_var)
        
        tf.cond(
            tf.equal(tf.math.floormod(self._counter, self.sync_period), 0),
            sync_slow_weights,
            lambda: None
        )
    
    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_dense(grad, var)
    
    def get_config(self):
        config = {
            "optimizer": keras.optimizers.serialize(self.optimizer),
            "sync_period": self.sync_period,
            "slow_step_size": self.slow_step_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def example_lookahead_optimizer():
    """Example: Lookahead optimizer."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Wrap Adam with Lookahead
    base_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = Lookahead(base_optimizer, sync_period=5, slow_step_size=0.5)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nLookahead Optimizer Model:")
    model.summary()
    return model


# Pattern 3: RAdam (Rectified Adam)
class RAdam(keras.optimizers.Optimizer):
    """Rectified Adam optimizer."""
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                 epsilon=1e-7, name="RAdam", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")  # First moment
            self.add_slot(var, "v")  # Second moment
    
    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("learning_rate")
        beta_1 = self._get_hyper("beta_1")
        beta_2 = self._get_hyper("beta_2")
        
        local_step = tf.cast(self.iterations + 1, tf.float32)
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        # Update biased moments
        m.assign(beta_1 * m + (1 - beta_1) * grad)
        v.assign(beta_2 * v + (1 - beta_2) * tf.square(grad))
        
        # Bias correction
        m_hat = m / (1 - tf.pow(beta_1, local_step))
        
        # Compute rectification term
        rho_inf = 2.0 / (1.0 - beta_2) - 1.0
        rho_t = rho_inf - 2.0 * local_step * tf.pow(beta_2, local_step) / (1.0 - tf.pow(beta_2, local_step))
        
        # Adaptive learning rate
        def use_adaptive_lr():
            v_hat = v / (1 - tf.pow(beta_2, local_step))
            r_t = tf.sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
            return lr * r_t * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        
        def use_momentum():
            return lr * m_hat
        
        update = tf.cond(rho_t > 4.0, use_adaptive_lr, use_momentum)
        var.assign_sub(update)
    
    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_dense(grad, var)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
        })
        return config


def example_radam_optimizer():
    """Example: RAdam optimizer."""
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    optimizer = RAdam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nRAdam Optimizer Model:")
    model.summary()
    return model


# Pattern 4: Gradient Centralization
class GradientCentralization(keras.optimizers.Optimizer):
    """Optimizer wrapper that applies gradient centralization."""
    
    def __init__(self, optimizer, name="GC", **kwargs):
        super().__init__(name, **kwargs)
        self.optimizer = optimizer
    
    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)
    
    def _centralize_gradients(self, grad):
        """Apply gradient centralization."""
        if len(grad.shape) > 1:
            # For weights (not biases), centralize by subtracting mean
            return grad - tf.reduce_mean(grad, axis=list(range(len(grad.shape) - 1)), keepdims=True)
        return grad
    
    def _resource_apply_dense(self, grad, var):
        centralized_grad = self._centralize_gradients(grad)
        return self.optimizer._resource_apply_dense(centralized_grad, var)
    
    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_dense(grad, var)
    
    def get_config(self):
        config = {
            "optimizer": keras.optimizers.serialize(self.optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}


def example_gradient_centralization():
    """Example: Gradient centralization."""
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    base_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = GradientCentralization(base_optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nGradient Centralization Model:")
    model.summary()
    return model


# Pattern 5: Sharpness Aware Minimization (SAM)
class SAM(keras.optimizers.Optimizer):
    """Sharpness Aware Minimization optimizer."""
    
    def __init__(self, optimizer, rho=0.05, name="SAM", **kwargs):
        super().__init__(name, **kwargs)
        self.optimizer = optimizer
        self.rho = rho
    
    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)
    
    def minimize(self, loss, var_list, tape=None):
        """Custom minimize with SAM algorithm."""
        # First forward-backward pass
        with tf.GradientTape() as tape:
            loss_value = loss()
        
        gradients = tape.gradient(loss_value, var_list)
        
        # Compute epsilon (perturbation)
        grad_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in gradients]))
        epsilon = [(g * self.rho / (grad_norm + 1e-12)) for g in gradients]
        
        # Add perturbation
        for var, eps in zip(var_list, epsilon):
            var.assign_add(eps)
        
        # Second forward-backward pass
        with tf.GradientTape() as tape:
            loss_value = loss()
        
        gradients = tape.gradient(loss_value, var_list)
        
        # Remove perturbation and apply gradients
        for var, eps in zip(var_list, epsilon):
            var.assign_sub(eps)
        
        self.optimizer.apply_gradients(zip(gradients, var_list))
        
        return loss_value


def example_sam_optimizer():
    """Example: SAM optimizer with custom training."""
    model = keras.Sequential([
        keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    base_optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    optimizer = SAM(base_optimizer, rho=0.05)
    
    print("\nSAM Optimizer Model:")
    model.summary()
    
    # Custom training loop required for SAM
    @tf.function
    def train_step(x, y):
        def loss_fn():
            predictions = model(x, training=True)
            return keras.losses.sparse_categorical_crossentropy(y, predictions)
        
        loss = optimizer.minimize(loss_fn, model.trainable_variables)
        return tf.reduce_mean(loss)
    
    return model, train_step


# Pattern 6: LAMB Optimizer (Layer-wise Adaptive Moments)
class LAMB(keras.optimizers.Optimizer):
    """LAMB optimizer for large batch training."""
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, weight_decay=0.01, name="LAMB", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon
        self.weight_decay = weight_decay
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
    
    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("learning_rate")
        beta_1 = self._get_hyper("beta_1")
        beta_2 = self._get_hyper("beta_2")
        
        local_step = tf.cast(self.iterations + 1, tf.float32)
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        # Update moments
        m.assign(beta_1 * m + (1 - beta_1) * grad)
        v.assign(beta_2 * v + (1 - beta_2) * tf.square(grad))
        
        # Bias correction
        m_hat = m / (1 - tf.pow(beta_1, local_step))
        v_hat = v / (1 - tf.pow(beta_2, local_step))
        
        # Compute update
        update = m_hat / (tf.sqrt(v_hat) + self.epsilon) + self.weight_decay * var
        
        # Layer-wise adaptation
        w_norm = tf.norm(var, ord=2)
        g_norm = tf.norm(update, ord=2)
        
        ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), w_norm / g_norm, 1.0),
            1.0
        )
        
        var.assign_sub(lr * ratio * update)
    
    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_dense(grad, var)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        })
        return config


def example_lamb_optimizer():
    """Example: LAMB optimizer for large batch training."""
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    optimizer = LAMB(learning_rate=0.002, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nLAMB Optimizer Model:")
    model.summary()
    return model


# Pattern 7: Learning Rate Warmup with Cosine Decay
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay."""
    
    def __init__(self, initial_learning_rate, warmup_steps, total_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)
        
        # Cosine decay phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * progress))
        decay_lr = self.initial_learning_rate * (self.alpha + (1 - self.alpha) * cosine_decay)
        
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
        }


def example_warmup_cosine_decay():
    """Example: Warmup with cosine decay."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Learning rate schedule
    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=0.001,
        warmup_steps=1000,
        total_steps=10000,
        alpha=0.0
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nWarmup Cosine Decay Model:")
    model.summary()
    return model


# Pattern 8: Gradient Accumulation
class GradientAccumulation(keras.Model):
    """Model wrapper for gradient accumulation."""
    
    def __init__(self, model, accumulation_steps=4):
        super().__init__()
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulation = None
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        # Initialize gradient accumulation
        if self.gradient_accumulation is None:
            self.gradient_accumulation = [
                tf.Variable(tf.zeros_like(var), trainable=False)
                for var in self.trainable_variables
            ]
        
        # Accumulate gradients
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred) / self.accumulation_steps
        
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Add to accumulated gradients
        for i, grad in enumerate(gradients):
            self.gradient_accumulation[i].assign_add(grad)
        
        # Apply accumulated gradients every N steps
        def apply_accumulated_gradients():
            self.optimizer.apply_gradients(
                zip(self.gradient_accumulation, self.trainable_variables)
            )
            # Reset accumulation
            for acc_grad in self.gradient_accumulation:
                acc_grad.assign(tf.zeros_like(acc_grad))
        
        tf.cond(
            tf.equal(tf.math.floormod(self.optimizer.iterations, self.accumulation_steps), 0),
            apply_accumulated_gradients,
            lambda: None
        )
        
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def example_gradient_accumulation():
    """Example: Gradient accumulation for large effective batch sizes."""
    base_model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model = GradientAccumulation(base_model, accumulation_steps=4)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nGradient Accumulation Model:")
    base_model.summary()
    return model


# Pattern 9: Mixed Optimizer Strategy
def example_mixed_optimizer():
    """Example: Different optimizers for different layers."""
    # Input
    inputs = keras.Input(shape=(784,))
    
    # Feature extractor
    x = keras.layers.Dense(256, activation='relu', name='feature1')(inputs)
    x = keras.layers.Dense(128, activation='relu', name='feature2')(x)
    
    # Classifier head
    outputs = keras.layers.Dense(10, activation='softmax', name='classifier')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Custom training loop with different optimizers
    feature_optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    classifier_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(x, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        # Get feature and classifier variables
        feature_vars = [v for v in model.trainable_variables if 'feature' in v.name]
        classifier_vars = [v for v in model.trainable_variables if 'classifier' in v.name]
        
        # Compute gradients
        feature_grads = tape.gradient(loss, feature_vars)
        classifier_grads = tape.gradient(loss, classifier_vars)
        
        # Apply different optimizers
        feature_optimizer.apply_gradients(zip(feature_grads, feature_vars))
        classifier_optimizer.apply_gradients(zip(classifier_grads, classifier_vars))
        
        del tape
        return loss
    
    print("\nMixed Optimizer Strategy Model:")
    model.summary()
    return model, train_step


if __name__ == "__main__":
    print("Advanced Optimizer Patterns\n" + "="*60)
    
    # Example 1: Custom Optimizer
    print("\n1. Custom SGD Optimizer")
    model1 = example_custom_optimizer()
    
    # Example 2: Lookahead
    print("\n2. Lookahead Optimizer")
    model2 = example_lookahead_optimizer()
    
    # Example 3: RAdam
    print("\n3. RAdam (Rectified Adam)")
    model3 = example_radam_optimizer()
    
    # Example 4: Gradient Centralization
    print("\n4. Gradient Centralization")
    model4 = example_gradient_centralization()
    
    # Example 5: SAM
    print("\n5. Sharpness Aware Minimization (SAM)")
    model5, train_step5 = example_sam_optimizer()
    
    # Example 6: LAMB
    print("\n6. LAMB Optimizer")
    model6 = example_lamb_optimizer()
    
    # Example 7: Warmup Cosine Decay
    print("\n7. Warmup with Cosine Decay")
    model7 = example_warmup_cosine_decay()
    
    # Example 8: Gradient Accumulation
    print("\n8. Gradient Accumulation")
    model8 = example_gradient_accumulation()
    
    # Example 9: Mixed Optimizer
    print("\n9. Mixed Optimizer Strategy")
    model9, train_step9 = example_mixed_optimizer()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Use Adam for most cases, SGD with momentum for computer vision")
    print("2. Apply warmup for large batch training")
    print("3. Use gradient accumulation when memory is limited")
    print("4. Consider LAMB for very large batch sizes (>1024)")
    print("5. SAM improves generalization but requires 2x computation")
    print("6. Lookahead provides more stable training")
    print("7. RAdam removes need for warmup in many cases")
    print("8. Gradient centralization improves convergence")
    print("9. Use mixed optimizers for transfer learning scenarios")
    print("10. Monitor gradient norms to detect training issues")
