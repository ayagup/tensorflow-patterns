"""
Distributed Training and Scalability Patterns

This module demonstrates patterns for scaling TensorFlow training
across multiple GPUs and machines.

Patterns covered:
1. Data Parallelism with MirroredStrategy
2. MultiWorkerMirroredStrategy for Multi-Machine Training
3. TPUStrategy for TPU Training
4. ParameterServerStrategy
5. Custom Training Loop with Distribution
6. Mixed Precision Training
7. Gradient Accumulation for Large Batches
8. Model Parallelism for Large Models
9. Pipeline Parallelism
10. Efficient Data Loading with tf.data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Data Parallelism with MirroredStrategy
def example_mirrored_strategy():
    """Example: Train with MirroredStrategy across GPUs."""
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # Create model in strategy scope
    with strategy.scope():
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(10,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Create dataset
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Train - will automatically distribute across GPUs
    print("\nMirroredStrategy Example:")
    history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model, strategy


# Pattern 2: MultiWorkerMirroredStrategy
def example_multi_worker_strategy():
    """Example: Multi-worker distributed training setup."""
    # In practice, set these environment variables on each worker
    # os.environ['TF_CONFIG'] = json.dumps({
    #     'cluster': {
    #         'worker': ['host1:port', 'host2:port']
    #     },
    #     'task': {'type': 'worker', 'index': 0}
    # })
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    print("\nMultiWorkerMirroredStrategy Example:")
    print(f"Number of workers: {strategy.num_replicas_in_sync}")
    
    with strategy.scope():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(10,)),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    print("Model ready for multi-worker training")
    print("Set TF_CONFIG environment variable on each worker")
    
    return model, strategy


# Pattern 3: TPU Strategy
def example_tpu_strategy():
    """Example: TPU training setup."""
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        strategy = tf.distribute.TPUStrategy(resolver)
        
        print(f"\nTPU Strategy Example:")
        print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")
        
        with strategy.scope():
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(10,)),
                layers.Dense(2, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy'
            )
        
        print("Model ready for TPU training")
        
        return model, strategy
    
    except:
        print("\nTPU Strategy Example:")
        print("TPU not available in this environment")
        print("Use TPUStrategy when training on Google Cloud TPUs")
        return None, None


# Pattern 4: Custom Training Loop with Distribution
class DistributedTrainer:
    """Custom training loop with distribution strategy."""
    
    def __init__(self, model, strategy):
        self.model = model
        self.strategy = strategy
        self.optimizer = keras.optimizers.Adam()
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
    
    def compute_loss(self, labels, predictions):
        """Compute per-replica loss."""
        per_example_loss = self.loss_fn(labels, predictions)
        return tf.nn.compute_average_loss(
            per_example_loss,
            global_batch_size=self.strategy.num_replicas_in_sync * 32
        )
    
    @tf.function
    def train_step(self, inputs):
        """Distributed training step."""
        def step_fn(inputs):
            features, labels = inputs
            
            with tf.GradientTape() as tape:
                predictions = self.model(features, training=True)
                loss = self.compute_loss(labels, predictions)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            
            return loss
        
        # Run on all replicas
        per_replica_losses = self.strategy.run(step_fn, args=(inputs,))
        
        # Reduce across replicas
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None
        )
    
    def train(self, dataset, epochs):
        """Training loop."""
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
            
            print(f"Epoch {epoch+1}: Loss = {total_loss/num_batches:.4f}")


def example_custom_distributed_training():
    """Example: Custom distributed training loop."""
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(10,)),
            layers.Dense(2, activation='softmax')
        ])
    
    # Create distributed dataset
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(32)
    dataset = strategy.experimental_distribute_dataset(dataset)
    
    # Train
    print("\nCustom Distributed Training Loop Example:")
    trainer = DistributedTrainer(model, strategy)
    trainer.train(dataset, epochs=3)
    
    return trainer


# Pattern 5: Mixed Precision Training
def example_mixed_precision():
    """Example: Mixed precision training for faster training."""
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print("\nMixed Precision Training Example:")
    print(f"Compute dtype: {policy.compute_dtype}")
    print(f"Variable dtype: {policy.variable_dtype}")
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(10,)),
        layers.Dense(64, activation='relu'),
        # Output layer should be float32
        layers.Dense(2, activation='softmax', dtype='float32')
    ])
    
    # Use loss scaling to prevent underflow
    optimizer = keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    X = np.random.randn(1000, 10).astype(np.float32)
    y = np.random.randint(0, 2, 1000)
    
    history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Reset policy
    tf.keras.mixed_precision.set_global_policy('float32')
    
    return model


# Pattern 6: Gradient Accumulation
class GradientAccumulationModel(keras.Model):
    """Model with gradient accumulation for large effective batch sizes."""
    
    def __init__(self, base_model, accumulation_steps=4):
        super().__init__()
        self.base_model = base_model
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulation = []
    
    def train_step(self, data):
        x, y = data
        
        # Split batch
        batch_size = tf.shape(x)[0]
        mini_batch_size = batch_size // self.accumulation_steps
        
        total_loss = 0.0
        
        for i in range(self.accumulation_steps):
            start = i * mini_batch_size
            end = start + mini_batch_size
            
            x_mini = x[start:end]
            y_mini = y[start:end]
            
            with tf.GradientTape() as tape:
                y_pred = self.base_model(x_mini, training=True)
                loss = self.compiled_loss(y_mini, y_pred)
            
            # Accumulate gradients
            gradients = tape.gradient(loss, self.base_model.trainable_variables)
            
            if i == 0:
                self.gradient_accumulation = gradients
            else:
                self.gradient_accumulation = [
                    acc_grad + grad
                    for acc_grad, grad in zip(self.gradient_accumulation, gradients)
                ]
            
            total_loss += loss
        
        # Average accumulated gradients
        self.gradient_accumulation = [
            grad / self.accumulation_steps
            for grad in self.gradient_accumulation
        ]
        
        # Apply gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.base_model.trainable_variables)
        )
        
        # Update metrics
        self.compiled_metrics.update_state(y, self.base_model(x, training=False))
        
        return {m.name: m.result() for m in self.metrics}


def example_gradient_accumulation():
    """Example: Gradient accumulation for large batches."""
    base_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(2, activation='softmax')
    ])
    
    # Wrap with gradient accumulation
    model = GradientAccumulationModel(base_model, accumulation_steps=4)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    print("\nGradient Accumulation Example:")
    print("Accumulating gradients over 4 mini-batches")
    
    history = model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model


# Pattern 7: Efficient Data Loading
def create_efficient_dataset(X, y, batch_size=32):
    """Create optimized tf.data pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle with buffer
    dataset = dataset.shuffle(buffer_size=10000)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Prefetch for pipeline
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Cache if dataset fits in memory
    dataset = dataset.cache()
    
    return dataset


def example_efficient_data_loading():
    """Example: Efficient data loading with tf.data."""
    X = np.random.randn(10000, 10)
    y = np.random.randint(0, 2, 10000)
    
    print("\nEfficient Data Loading Example:")
    
    # Create optimized dataset
    dataset = create_efficient_dataset(X, y, batch_size=32)
    
    # Model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with optimized pipeline
    history = model.fit(dataset, epochs=3, verbose=0)
    
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print("Dataset uses: shuffle, batch, prefetch, cache")
    
    return dataset


# Pattern 8: Model Parallelism (Conceptual)
class ModelParallelNetwork(keras.Model):
    """Example of splitting model across devices."""
    
    def __init__(self):
        super().__init__()
        
        # Place different parts on different devices
        with tf.device('/GPU:0'):
            self.part1 = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(10,)),
                layers.Dense(256, activation='relu')
            ])
        
        with tf.device('/GPU:1'):
            self.part2 = keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dense(2, activation='softmax')
            ])
    
    def call(self, inputs):
        # Forward pass across devices
        x = self.part1(inputs)
        x = self.part2(x)
        return x


def example_model_parallelism():
    """Example: Model parallelism across devices."""
    print("\nModel Parallelism Example:")
    
    try:
        model = ModelParallelNetwork()
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy'
        )
        
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)
        
        print("Model split across GPU:0 and GPU:1")
        print("Part 1 (layers 1-2) on GPU:0")
        print("Part 2 (layers 3-4) on GPU:1")
    
    except:
        print("Multiple GPUs not available")
        print("Model parallelism requires multiple devices")
        print("Use tf.device() to place layers on specific devices")
    
    return None


# Pattern 9: Pipeline Parallelism (Conceptual)
class PipelineParallelModel(keras.Model):
    """Pipeline parallelism with micro-batching."""
    
    def __init__(self, num_stages=2, micro_batch_size=8):
        super().__init__()
        
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size
        
        # Stage 1
        self.stage1 = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(10,))
        ])
        
        # Stage 2
        self.stage2 = keras.Sequential([
            layers.Dense(2, activation='softmax')
        ])
    
    def call(self, inputs):
        """Process through pipeline stages."""
        # In practice, stages would be on different devices
        # and process micro-batches concurrently
        x = self.stage1(inputs)
        x = self.stage2(x)
        return x


def example_pipeline_parallelism():
    """Example: Pipeline parallelism concept."""
    model = PipelineParallelModel(num_stages=2, micro_batch_size=8)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    print("\nPipeline Parallelism Example:")
    print("Splits batch into micro-batches")
    print("Stage 1 processes micro-batch 1 while")
    print("Stage 2 processes micro-batch 0")
    
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    
    return model


# Pattern 10: Sharded Data Loading
def create_sharded_dataset(file_pattern, batch_size=32):
    """Create dataset from sharded files."""
    # List shard files
    files = tf.io.gfile.glob(file_pattern)
    
    # Create dataset from files
    dataset = tf.data.Dataset.from_tensor_slices(files)
    
    # Interleave reading from shards
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Parse, batch, prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def example_sharded_data():
    """Example: Loading from sharded data files."""
    print("\nSharded Data Loading Example:")
    print("Use tf.data.Dataset.interleave for sharded files")
    print("Pattern: 'data/train-*.tfrecord'")
    print("Benefits:")
    print("  - Parallel reading from multiple shards")
    print("  - Better I/O utilization")
    print("  - Scales to large datasets")
    
    # Conceptual code
    # dataset = create_sharded_dataset('data/train-*.tfrecord')
    
    return None


if __name__ == "__main__":
    print("Distributed Training and Scalability Patterns\n" + "="*60)
    
    # Example 1: MirroredStrategy
    print("\n1. MirroredStrategy (Data Parallelism)")
    try:
        model1, strategy1 = example_mirrored_strategy()
    except:
        print("GPU not available, skipping")
    
    # Example 2: MultiWorkerMirroredStrategy
    print("\n2. MultiWorkerMirroredStrategy")
    try:
        model2, strategy2 = example_multi_worker_strategy()
    except:
        print("Multi-worker setup not configured")
    
    # Example 3: TPU Strategy
    print("\n3. TPU Strategy")
    model3, strategy3 = example_tpu_strategy()
    
    # Example 4: Custom Distributed Training
    print("\n4. Custom Distributed Training Loop")
    try:
        trainer = example_custom_distributed_training()
    except:
        print("Skipping custom distributed training")
    
    # Example 5: Mixed Precision
    print("\n5. Mixed Precision Training")
    model5 = example_mixed_precision()
    
    # Example 6: Gradient Accumulation
    print("\n6. Gradient Accumulation")
    model6 = example_gradient_accumulation()
    
    # Example 7: Efficient Data Loading
    print("\n7. Efficient Data Loading")
    dataset = example_efficient_data_loading()
    
    # Example 8: Model Parallelism
    print("\n8. Model Parallelism")
    example_model_parallelism()
    
    # Example 9: Pipeline Parallelism
    print("\n9. Pipeline Parallelism")
    model9 = example_pipeline_parallelism()
    
    # Example 10: Sharded Data
    print("\n10. Sharded Data Loading")
    example_sharded_data()
    
    print("\n" + "="*60)
    print("Distributed Training Best Practices:")
    print("1. Use MirroredStrategy for single-machine multi-GPU")
    print("2. Use MultiWorkerMirroredStrategy for multi-machine")
    print("3. Enable mixed precision for faster training")
    print("4. Use gradient accumulation for large batches")
    print("5. Optimize data pipeline with prefetch and cache")
    print("6. Shard data files for parallel loading")
    print("7. Monitor GPU utilization and bottlenecks")
    print("8. Use model parallelism for very large models")
    print("9. Consider TPUs for maximum performance")
    print("10. Profile and optimize the slowest stage")
