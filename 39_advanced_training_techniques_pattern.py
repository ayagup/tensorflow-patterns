"""
Advanced Training Techniques Patterns

This module demonstrates advanced training strategies and techniques
for improving model performance and training efficiency.

Patterns covered:
1. Curriculum Learning
2. Self-Supervised Learning (SimCLR-style)
3. Contrastive Learning
4. Meta-Learning (MAML)
5. Few-Shot Learning
6. Active Learning
7. Semi-Supervised Learning
8. Adversarial Training
9. Progressive Training
10. Cyclical Training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Curriculum Learning
class CurriculumLearning:
    """Train model with progressively harder examples."""
    
    def __init__(self, model, easy_data, medium_data, hard_data):
        self.model = model
        self.datasets = {
            'easy': easy_data,
            'medium': medium_data,
            'hard': hard_data
        }
        self.current_stage = 'easy'
    
    def train_stage(self, stage, epochs):
        """Train on specific difficulty stage."""
        print(f"Training on {stage} data for {epochs} epochs")
        
        dataset = self.datasets[stage]
        history = self.model.fit(
            dataset,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def train_curriculum(self, stage_epochs):
        """Train through curriculum stages."""
        stages = ['easy', 'medium', 'hard']
        
        for stage in stages:
            print(f"\n{'='*50}")
            print(f"Stage: {stage.upper()}")
            print('='*50)
            
            self.current_stage = stage
            self.train_stage(stage, stage_epochs[stage])


def example_curriculum_learning():
    """Example: Curriculum learning."""
    # Create model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Simulate datasets of different difficulty
    easy_data = tf.data.Dataset.from_tensor_slices((
        np.random.randn(100, 20).astype(np.float32),
        np.random.randint(0, 10, 100)
    )).batch(32)
    
    medium_data = tf.data.Dataset.from_tensor_slices((
        np.random.randn(100, 20).astype(np.float32) * 1.5,
        np.random.randint(0, 10, 100)
    )).batch(32)
    
    hard_data = tf.data.Dataset.from_tensor_slices((
        np.random.randn(100, 20).astype(np.float32) * 2.0,
        np.random.randint(0, 10, 100)
    )).batch(32)
    
    curriculum = CurriculumLearning(model, easy_data, medium_data, hard_data)
    
    print("Curriculum Learning Example:")
    print("Train on easy examples first, then progressively harder ones")
    
    # curriculum.train_curriculum({'easy': 2, 'medium': 2, 'hard': 2})
    
    return curriculum


# Pattern 2: Self-Supervised Learning (SimCLR-style)
class ContrastiveProjectionHead(layers.Layer):
    """Projection head for contrastive learning."""
    
    def __init__(self, hidden_dim=128, output_dim=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(output_dim)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class SimCLRModel(keras.Model):
    """SimCLR-style self-supervised model."""
    
    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        self.projection_head = ContrastiveProjectionHead(output_dim=projection_dim)
    
    def call(self, inputs):
        features = self.encoder(inputs)
        projections = self.projection_head(features)
        return projections


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)."""
    batch_size = tf.shape(z_i)[0]
    
    # Normalize embeddings
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    
    # Compute similarity matrix
    representations = tf.concat([z_i, z_j], axis=0)
    similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
    similarity_matrix = similarity_matrix / temperature
    
    # Create labels (positive pairs)
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)
    
    # Mask out self-similarity
    mask = tf.eye(2 * batch_size)
    similarity_matrix = similarity_matrix - mask * 1e9
    
    # Compute loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=similarity_matrix
    )
    
    return tf.reduce_mean(loss)


def example_self_supervised_learning():
    """Example: Self-supervised learning (SimCLR-style)."""
    # Encoder network
    encoder = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu')
    ])
    
    # SimCLR model
    model = SimCLRModel(encoder, projection_dim=64)
    
    print("\nSelf-Supervised Learning (SimCLR) Example:")
    print("Learns representations from unlabeled data using contrastive loss")
    
    # Dummy augmented pairs
    x_i = tf.random.normal((16, 32, 32, 3))
    x_j = tf.random.normal((16, 32, 32, 3))
    
    z_i = model(x_i)
    z_j = model(x_j)
    
    loss = nt_xent_loss(z_i, z_j)
    print(f"Contrastive loss: {loss:.4f}")
    
    return model


# Pattern 3: Contrastive Learning
class ContrastiveLearningModel(keras.Model):
    """Model for contrastive learning with triplet loss."""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(embedding_dim)
        ])
    
    def call(self, inputs):
        return tf.nn.l2_normalize(self.encoder(inputs), axis=1)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss for contrastive learning."""
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


def example_contrastive_learning():
    """Example: Contrastive learning with triplet loss."""
    model = ContrastiveLearningModel(embedding_dim=128)
    
    # Dummy data (anchor, positive, negative)
    anchor = tf.random.normal((32, 28, 28, 1))
    positive = tf.random.normal((32, 28, 28, 1))
    negative = tf.random.normal((32, 28, 28, 1))
    
    # Get embeddings
    anchor_emb = model(anchor)
    positive_emb = model(positive)
    negative_emb = model(negative)
    
    # Compute loss
    loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
    
    print("\nContrastive Learning (Triplet Loss) Example:")
    print(f"Triplet loss: {loss:.4f}")
    print("Learns to separate similar and dissimilar examples")
    
    return model


# Pattern 4: Meta-Learning (MAML-inspired)
class MAMLModel(keras.Model):
    """Model-Agnostic Meta-Learning inspired approach."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu', input_shape=(input_dim,))
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(output_dim)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


class MAML:
    """MAML meta-learning algorithm."""
    
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_optimizer = keras.optimizers.Adam(outer_lr)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def inner_loop(self, support_x, support_y, steps=5):
        """Adapt model to task using support set."""
        # Clone model weights
        adapted_weights = [tf.identity(w) for w in self.model.trainable_variables]
        
        for _ in range(steps):
            with tf.GradientTape() as tape:
                # Watch the adapted weights
                for w in adapted_weights:
                    tape.watch(w)
                
                # Forward pass with adapted weights
                predictions = self.forward_with_weights(support_x, adapted_weights)
                loss = self.loss_fn(support_y, predictions)
            
            # Compute gradients and update adapted weights
            grads = tape.gradient(loss, adapted_weights)
            adapted_weights = [
                w - self.inner_lr * g if g is not None else w
                for w, g in zip(adapted_weights, grads)
            ]
        
        return adapted_weights
    
    def forward_with_weights(self, inputs, weights):
        """Forward pass with custom weights."""
        # Simplified forward pass
        x = tf.matmul(inputs, weights[0]) + weights[1]
        x = tf.nn.relu(x)
        x = tf.matmul(x, weights[2]) + weights[3]
        x = tf.nn.relu(x)
        x = tf.matmul(x, weights[4]) + weights[5]
        return x
    
    def meta_train_step(self, tasks):
        """Meta-training step across multiple tasks."""
        meta_loss = 0.0
        
        with tf.GradientTape() as outer_tape:
            for support_x, support_y, query_x, query_y in tasks:
                # Inner loop adaptation
                adapted_weights = self.inner_loop(support_x, support_y)
                
                # Evaluate on query set
                query_pred = self.forward_with_weights(query_x, adapted_weights)
                task_loss = self.loss_fn(query_y, query_pred)
                meta_loss += task_loss
            
            meta_loss /= len(tasks)
        
        # Meta-update
        meta_grads = outer_tape.gradient(meta_loss, self.model.trainable_variables)
        self.outer_optimizer.apply_gradients(
            zip(meta_grads, self.model.trainable_variables)
        )
        
        return meta_loss


def example_meta_learning():
    """Example: Meta-learning (MAML)."""
    model = MAMLModel(input_dim=20, output_dim=5)
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001)
    
    print("\nMeta-Learning (MAML) Example:")
    print("Learns to quickly adapt to new tasks with few examples")
    print("Inner loop: Task-specific adaptation")
    print("Outer loop: Meta-update across tasks")
    
    return maml


# Pattern 5: Few-Shot Learning
class PrototypicalNetwork(keras.Model):
    """Prototypical network for few-shot learning."""
    
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(embedding_dim)
        ])
    
    def call(self, inputs):
        return self.encoder(inputs)
    
    def compute_prototypes(self, support_embeddings, support_labels, n_classes):
        """Compute class prototypes from support set."""
        prototypes = []
        
        for c in range(n_classes):
            mask = tf.equal(support_labels, c)
            class_embeddings = tf.boolean_mask(support_embeddings, mask)
            prototype = tf.reduce_mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        return tf.stack(prototypes)
    
    def classify(self, query_embeddings, prototypes):
        """Classify query samples using prototypes."""
        # Euclidean distance to prototypes
        distances = tf.reduce_sum(
            tf.square(tf.expand_dims(query_embeddings, 1) - prototypes),
            axis=2
        )
        
        # Negative distance as logits
        logits = -distances
        return tf.nn.softmax(logits)


def example_few_shot_learning():
    """Example: Few-shot learning with prototypical networks."""
    model = PrototypicalNetwork(embedding_dim=64)
    
    print("\nFew-Shot Learning (Prototypical Network) Example:")
    print("Learns to classify from few examples per class")
    
    # 5-way 1-shot example
    n_classes = 5
    n_support = 1
    n_query = 5
    
    # Dummy data
    support_x = tf.random.normal((n_classes * n_support, 28, 28, 1))
    support_y = tf.repeat(tf.range(n_classes), n_support)
    query_x = tf.random.normal((n_query, 28, 28, 1))
    
    # Compute embeddings
    support_embeddings = model(support_x)
    query_embeddings = model(query_x)
    
    # Compute prototypes and classify
    prototypes = model.compute_prototypes(support_embeddings, support_y, n_classes)
    predictions = model.classify(query_embeddings, prototypes)
    
    print(f"Support set: {n_classes} classes, {n_support} examples each")
    print(f"Query predictions shape: {predictions.shape}")
    
    return model


# Pattern 6: Active Learning
class ActiveLearningStrategy:
    """Active learning with uncertainty sampling."""
    
    def __init__(self, model, unlabeled_data):
        self.model = model
        self.unlabeled_data = unlabeled_data
        self.labeled_indices = []
    
    def uncertainty_sampling(self, pool_size=100, batch_size=10):
        """Select most uncertain examples to label."""
        # Get predictions for unlabeled pool
        pool_data = self.unlabeled_data[:pool_size]
        predictions = self.model(pool_data, training=False)
        
        # Compute uncertainty (entropy)
        entropy = -tf.reduce_sum(
            predictions * tf.math.log(predictions + 1e-10),
            axis=1
        )
        
        # Select top uncertain examples
        _, top_indices = tf.nn.top_k(entropy, k=batch_size)
        
        return top_indices.numpy()
    
    def margin_sampling(self, pool_size=100, batch_size=10):
        """Select examples with smallest margin between top-2 predictions."""
        pool_data = self.unlabeled_data[:pool_size]
        predictions = self.model(pool_data, training=False)
        
        # Get top-2 predictions
        top2 = tf.nn.top_k(predictions, k=2)
        
        # Compute margin
        margin = top2.values[:, 0] - top2.values[:, 1]
        
        # Select smallest margins (most uncertain)
        _, bottom_indices = tf.nn.top_k(-margin, k=batch_size)
        
        return bottom_indices.numpy()


def example_active_learning():
    """Example: Active learning strategy."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dense(10, activation='softmax')
    ])
    
    # Unlabeled pool
    unlabeled_data = tf.random.normal((1000, 20))
    
    strategy = ActiveLearningStrategy(model, unlabeled_data)
    
    print("\nActive Learning Example:")
    print("Selects most informative examples for labeling")
    
    # Select uncertain examples
    uncertain_indices = strategy.uncertainty_sampling(pool_size=100, batch_size=10)
    print(f"Selected {len(uncertain_indices)} examples for labeling")
    print(f"Indices: {uncertain_indices}")
    
    return strategy


# Pattern 7: Semi-Supervised Learning
class PseudoLabelingModel(keras.Model):
    """Semi-supervised learning with pseudo-labeling."""
    
    def __init__(self, base_model, confidence_threshold=0.9):
        super().__init__()
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def generate_pseudo_labels(self, unlabeled_data):
        """Generate pseudo-labels for confident predictions."""
        predictions = self(unlabeled_data, training=False)
        
        # Get max probability and predicted class
        max_probs = tf.reduce_max(predictions, axis=1)
        pseudo_labels = tf.argmax(predictions, axis=1)
        
        # Filter by confidence
        confident_mask = max_probs >= self.confidence_threshold
        
        confident_data = tf.boolean_mask(unlabeled_data, confident_mask)
        confident_labels = tf.boolean_mask(pseudo_labels, confident_mask)
        
        return confident_data, confident_labels, tf.reduce_sum(tf.cast(confident_mask, tf.int32))


def example_semi_supervised_learning():
    """Example: Semi-supervised learning with pseudo-labeling."""
    base_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model = PseudoLabelingModel(base_model, confidence_threshold=0.9)
    
    print("\nSemi-Supervised Learning (Pseudo-Labeling) Example:")
    print("Uses confident predictions as pseudo-labels for unlabeled data")
    
    # Unlabeled data
    unlabeled_data = tf.random.normal((100, 784))
    
    pseudo_data, pseudo_labels, count = model.generate_pseudo_labels(unlabeled_data)
    
    print(f"Confident predictions: {count.numpy()} out of {unlabeled_data.shape[0]}")
    
    return model


# Pattern 8: Adversarial Training
class AdversarialTraining:
    """Adversarial training for robustness."""
    
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    def generate_adversarial_examples(self, images, labels):
        """Generate adversarial examples using FGSM."""
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images, training=False)
            loss = self.loss_fn(labels, predictions)
        
        gradients = tape.gradient(loss, images)
        signed_grad = tf.sign(gradients)
        adversarial_images = images + self.epsilon * signed_grad
        
        # Clip to valid range
        adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
        
        return adversarial_images
    
    def train_step(self, images, labels, optimizer):
        """Train with both clean and adversarial examples."""
        # Generate adversarial examples
        adv_images = self.generate_adversarial_examples(images, labels)
        
        # Combine clean and adversarial
        combined_images = tf.concat([images, adv_images], axis=0)
        combined_labels = tf.concat([labels, labels], axis=0)
        
        with tf.GradientTape() as tape:
            predictions = self.model(combined_images, training=True)
            loss = self.loss_fn(combined_labels, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss


def example_adversarial_training():
    """Example: Adversarial training."""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    trainer = AdversarialTraining(model, epsilon=0.1)
    
    print("\nAdversarial Training Example:")
    print("Trains on both clean and adversarial examples")
    print("Improves model robustness to adversarial attacks")
    
    return trainer


# Pattern 9: Progressive Training
class ProgressiveTraining:
    """Progressive training with increasing complexity."""
    
    def __init__(self, model):
        self.model = model
        self.current_resolution = 32
    
    def train_at_resolution(self, resolution, dataset, epochs):
        """Train at specific resolution."""
        print(f"Training at {resolution}x{resolution} resolution")
        
        # Resize dataset to current resolution
        def resize_fn(image, label):
            image = tf.image.resize(image, (resolution, resolution))
            return image, label
        
        resized_dataset = dataset.map(resize_fn)
        
        history = self.model.fit(resized_dataset, epochs=epochs, verbose=1)
        
        return history
    
    def progressive_train(self, dataset, resolutions, epochs_per_stage):
        """Train progressively through resolutions."""
        for resolution in resolutions:
            print(f"\n{'='*50}")
            print(f"Resolution: {resolution}x{resolution}")
            print('='*50)
            
            self.current_resolution = resolution
            self.train_at_resolution(resolution, dataset, epochs_per_stage)


def example_progressive_training():
    """Example: Progressive training."""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(None, None, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    trainer = ProgressiveTraining(model)
    
    print("\nProgressive Training Example:")
    print("Starts training with low resolution, gradually increases")
    print("Improves training stability and speed")
    
    return trainer


if __name__ == "__main__":
    print("Advanced Training Techniques Patterns\n" + "="*60)
    
    # Example 1: Curriculum Learning
    print("\n1. Curriculum Learning")
    curriculum = example_curriculum_learning()
    
    # Example 2: Self-Supervised Learning
    print("\n2. Self-Supervised Learning (SimCLR)")
    ssl_model = example_self_supervised_learning()
    
    # Example 3: Contrastive Learning
    print("\n3. Contrastive Learning (Triplet Loss)")
    contrastive_model = example_contrastive_learning()
    
    # Example 4: Meta-Learning
    print("\n4. Meta-Learning (MAML)")
    maml = example_meta_learning()
    
    # Example 5: Few-Shot Learning
    print("\n5. Few-Shot Learning (Prototypical Networks)")
    fsl_model = example_few_shot_learning()
    
    # Example 6: Active Learning
    print("\n6. Active Learning")
    active_strategy = example_active_learning()
    
    # Example 7: Semi-Supervised Learning
    print("\n7. Semi-Supervised Learning (Pseudo-Labeling)")
    ssl_model = example_semi_supervised_learning()
    
    # Example 8: Adversarial Training
    print("\n8. Adversarial Training")
    adv_trainer = example_adversarial_training()
    
    # Example 9: Progressive Training
    print("\n9. Progressive Training")
    prog_trainer = example_progressive_training()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Curriculum: Start easy, gradually increase difficulty")
    print("2. Self-supervised: Great for unlabeled data")
    print("3. Contrastive: Learn discriminative representations")
    print("4. Meta-learning: Quick adaptation to new tasks")
    print("5. Few-shot: Useful when data is scarce")
    print("6. Active learning: Efficient labeling strategy")
    print("7. Semi-supervised: Leverage unlabeled data")
    print("8. Adversarial training: Improve robustness")
    print("9. Progressive: Stable training for high-resolution")
    print("10. Combine techniques for best results")
