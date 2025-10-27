"""
Multi-Task and Multi-Modal Learning Patterns

This module demonstrates patterns for training models on multiple tasks
or learning from multiple types of data (images, text, audio, etc.).

Patterns covered:
1. Hard Parameter Sharing Multi-Task Learning
2. Soft Parameter Sharing Multi-Task Learning
3. Task-Specific Layers
4. Multi-Modal Fusion (Early, Late, Hybrid)
5. Cross-Modal Attention
6. Multi-Modal Contrastive Learning
7. Task Weighting and Balancing
8. Auxiliary Tasks
9. Multi-Head Multi-Task Architecture
10. Conditional Computation for Tasks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Hard Parameter Sharing Multi-Task Learning
class HardParameterSharingMTL(keras.Model):
    """Multi-task model with shared backbone and task-specific heads."""
    
    def __init__(self, shared_units=128, num_tasks=2, task_units=64):
        super().__init__()
        
        # Shared layers
        self.shared_layers = keras.Sequential([
            layers.Dense(shared_units, activation='relu'),
            layers.Dense(shared_units, activation='relu')
        ])
        
        # Task-specific heads
        self.task_heads = [
            keras.Sequential([
                layers.Dense(task_units, activation='relu'),
                layers.Dense(1)
            ])
            for _ in range(num_tasks)
        ]
    
    def call(self, inputs):
        # Shared representation
        shared_features = self.shared_layers(inputs)
        
        # Task-specific outputs
        outputs = [head(shared_features) for head in self.task_heads]
        
        return outputs


def example_hard_parameter_sharing():
    """Example: Hard parameter sharing for multi-task learning."""
    # Create dataset (two regression tasks)
    X = np.random.randn(1000, 10)
    y1 = (X[:, 0] + X[:, 1]).reshape(-1, 1)  # Task 1: sum of first two features
    y2 = (X[:, 0] * X[:, 1]).reshape(-1, 1)  # Task 2: product of first two features
    
    # Create model
    model = HardParameterSharingMTL(shared_units=64, num_tasks=2, task_units=32)
    
    # Compile with task-specific losses
    model.compile(
        optimizer='adam',
        loss=['mse', 'mse'],
        metrics=[['mae'], ['mae']]
    )
    
    # Train
    print("Hard Parameter Sharing Example:")
    history = model.fit(
        X, [y1, y2],
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    print(f"Final losses: {history.history['loss'][-1]:.4f}")
    print(f"Task 1 MAE: {history.history['sequential_mae'][-1]:.4f}")
    print(f"Task 2 MAE: {history.history['sequential_1_mae'][-1]:.4f}")
    
    return model


# Pattern 2: Soft Parameter Sharing Multi-Task Learning
class SoftParameterSharingMTL(keras.Model):
    """Multi-task model with task-specific networks and cross-stitch units."""
    
    def __init__(self, num_tasks=2, hidden_units=64):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Task-specific networks
        self.task_networks = [
            keras.Sequential([
                layers.Dense(hidden_units, activation='relu'),
                layers.Dense(hidden_units, activation='relu')
            ])
            for _ in range(num_tasks)
        ]
        
        # Cross-stitch units for information sharing
        self.cross_stitch = layers.Dense(
            num_tasks * hidden_units,
            use_bias=False
        )
        
        # Output heads
        self.output_heads = [
            layers.Dense(1)
            for _ in range(num_tasks)
        ]
    
    def call(self, inputs):
        # Get task-specific features
        task_features = [net(inputs) for net in self.task_networks]
        
        # Concatenate and apply cross-stitch
        concat_features = tf.concat(task_features, axis=-1)
        shared_features = self.cross_stitch(concat_features)
        
        # Split back to tasks
        task_dim = shared_features.shape[-1] // self.num_tasks
        split_features = tf.split(shared_features, self.num_tasks, axis=-1)
        
        # Generate outputs
        outputs = [head(feat) for head, feat in zip(self.output_heads, split_features)]
        
        return outputs


def example_soft_parameter_sharing():
    """Example: Soft parameter sharing for multi-task learning."""
    # Create dataset
    X = np.random.randn(1000, 10)
    y1 = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    y2 = (X[:, 2] + X[:, 3]).reshape(-1, 1)
    
    # Create model
    model = SoftParameterSharingMTL(num_tasks=2, hidden_units=32)
    
    # Compile
    model.compile(
        optimizer='adam',
        loss=['mse', 'mse']
    )
    
    # Train
    print("\nSoft Parameter Sharing Example:")
    history = model.fit(
        X, [y1, y2],
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
    return model


# Pattern 3: Multi-Modal Fusion
class MultiModalFusion(keras.Model):
    """Multi-modal learning with different fusion strategies."""
    
    def __init__(self, fusion_type='late', hidden_units=64, num_classes=10):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Modality 1 encoder (e.g., image)
        self.encoder1 = keras.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(hidden_units, activation='relu')
        ])
        
        # Modality 2 encoder (e.g., text)
        self.encoder2 = keras.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(hidden_units, activation='relu')
        ])
        
        # Fusion layer
        if fusion_type == 'early':
            self.fusion = keras.Sequential([
                layers.Dense(hidden_units, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
        elif fusion_type == 'late':
            self.classifier1 = layers.Dense(num_classes, activation='softmax')
            self.classifier2 = layers.Dense(num_classes, activation='softmax')
        elif fusion_type == 'hybrid':
            self.fusion = keras.Sequential([
                layers.Dense(hidden_units * 2, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
    
    def call(self, inputs):
        modal1, modal2 = inputs
        
        if self.fusion_type == 'early':
            # Concatenate at input level
            concat = tf.concat([modal1, modal2], axis=-1)
            features = self.fusion(concat)
            return features
        
        elif self.fusion_type == 'late':
            # Separate processing and combine outputs
            feat1 = self.encoder1(modal1)
            feat2 = self.encoder2(modal2)
            
            out1 = self.classifier1(feat1)
            out2 = self.classifier2(feat2)
            
            # Average predictions
            return (out1 + out2) / 2
        
        elif self.fusion_type == 'hybrid':
            # Process separately then fuse features
            feat1 = self.encoder1(modal1)
            feat2 = self.encoder2(modal2)
            
            concat = tf.concat([feat1, feat2], axis=-1)
            return self.fusion(concat)


def example_multimodal_fusion():
    """Example: Multi-modal fusion."""
    # Create multi-modal dataset
    modal1 = np.random.randn(1000, 20)  # e.g., image features
    modal2 = np.random.randn(1000, 15)  # e.g., text features
    y = keras.utils.to_categorical(np.random.randint(0, 10, 1000))
    
    print("\nMulti-Modal Fusion Example:")
    
    for fusion_type in ['early', 'late', 'hybrid']:
        model = MultiModalFusion(
            fusion_type=fusion_type,
            hidden_units=32,
            num_classes=10
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            [modal1, modal2], y,
            epochs=3,
            batch_size=32,
            verbose=0
        )
        
        print(f"{fusion_type.capitalize()} fusion - Accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model


# Pattern 4: Cross-Modal Attention
class CrossModalAttention(layers.Layer):
    """Cross-attention between two modalities."""
    
    def __init__(self, units):
        super().__init__()
        self.units = units
        
        self.query = layers.Dense(units)
        self.key = layers.Dense(units)
        self.value = layers.Dense(units)
    
    def call(self, inputs):
        modal1, modal2 = inputs
        
        # modal1 attends to modal2
        Q = self.query(modal1)
        K = self.key(modal2)
        V = self.value(modal2)
        
        # Attention scores
        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        
        # Attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Attended values
        attended = tf.matmul(attention_weights, V)
        
        return attended, attention_weights


class CrossModalAttentionModel(keras.Model):
    """Multi-modal model with cross-attention."""
    
    def __init__(self, hidden_units=64, num_classes=10):
        super().__init__()
        
        self.encoder1 = layers.Dense(hidden_units, activation='relu')
        self.encoder2 = layers.Dense(hidden_units, activation='relu')
        
        self.cross_attention = CrossModalAttention(hidden_units)
        
        self.classifier = keras.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs):
        modal1, modal2 = inputs
        
        # Encode modalities
        feat1 = self.encoder1(modal1)
        feat2 = self.encoder2(modal2)
        
        # Add batch and sequence dimensions for attention
        feat1_expanded = tf.expand_dims(feat1, 1)
        feat2_expanded = tf.expand_dims(feat2, 1)
        
        # Cross-attention: modal1 attends to modal2
        attended, _ = self.cross_attention([feat1_expanded, feat2_expanded])
        attended = tf.squeeze(attended, 1)
        
        # Combine
        combined = feat1 + attended
        
        # Classify
        output = self.classifier(combined)
        
        return output


def example_cross_modal_attention():
    """Example: Cross-modal attention."""
    # Create dataset
    modal1 = np.random.randn(1000, 20)
    modal2 = np.random.randn(1000, 15)
    y = keras.utils.to_categorical(np.random.randint(0, 10, 1000))
    
    # Create model
    model = CrossModalAttentionModel(hidden_units=32, num_classes=10)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nCross-Modal Attention Example:")
    history = model.fit(
        [modal1, modal2], y,
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model


# Pattern 5: Multi-Modal Contrastive Learning
class MultiModalContrastiveLearning(keras.Model):
    """Contrastive learning for multi-modal data (e.g., CLIP-style)."""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Modality encoders
        self.image_encoder = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(embedding_dim)
        ])
        
        self.text_encoder = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(embedding_dim)
        ])
        
        # Temperature parameter for contrastive loss
        self.temperature = tf.Variable(0.07, trainable=True)
    
    def call(self, inputs):
        images, texts = inputs
        
        # Get embeddings
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(texts)
        
        # Normalize
        image_embeds = tf.nn.l2_normalize(image_embeds, axis=-1)
        text_embeds = tf.nn.l2_normalize(text_embeds, axis=-1)
        
        return image_embeds, text_embeds
    
    def contrastive_loss(self, image_embeds, text_embeds):
        """Compute symmetric contrastive loss."""
        batch_size = tf.shape(image_embeds)[0]
        
        # Compute similarity matrix
        logits = tf.matmul(image_embeds, text_embeds, transpose_b=True) / self.temperature
        
        # Labels are diagonal (i.e., matching pairs)
        labels = tf.range(batch_size)
        
        # Loss for both directions
        loss_i2t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss_t2i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, tf.transpose(logits))
        
        return (tf.reduce_mean(loss_i2t) + tf.reduce_mean(loss_t2i)) / 2


def example_multimodal_contrastive():
    """Example: Multi-modal contrastive learning."""
    # Create paired data
    images = np.random.randn(100, 50)  # Image features
    texts = np.random.randn(100, 30)   # Text features
    
    # Create model
    model = MultiModalContrastiveLearning(embedding_dim=64)
    
    # Custom training loop
    optimizer = keras.optimizers.Adam(0.001)
    
    print("\nMulti-Modal Contrastive Learning Example:")
    
    for epoch in range(5):
        with tf.GradientTape() as tape:
            image_embeds, text_embeds = model([images, texts])
            loss = model.contrastive_loss(image_embeds, text_embeds)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
    
    return model


# Pattern 6: Task Weighting
class TaskWeightedMultiTask(keras.Model):
    """Multi-task learning with learned task weights."""
    
    def __init__(self, shared_units=128, num_tasks=2):
        super().__init__()
        
        # Shared layers
        self.shared = keras.Sequential([
            layers.Dense(shared_units, activation='relu'),
            layers.Dense(shared_units, activation='relu')
        ])
        
        # Task-specific heads
        self.task_heads = [layers.Dense(1) for _ in range(num_tasks)]
        
        # Learnable task weights (log variance)
        self.log_vars = [
            tf.Variable(0.0, trainable=True, name=f'log_var_{i}')
            for i in range(num_tasks)
        ]
    
    def call(self, inputs):
        shared_features = self.shared(inputs)
        outputs = [head(shared_features) for head in self.task_heads]
        return outputs
    
    def weighted_loss(self, y_true_list, y_pred_list):
        """Compute weighted multi-task loss."""
        total_loss = 0.0
        
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            # MSE loss
            task_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Weight by learned uncertainty
            precision = tf.exp(-self.log_vars[i])
            weighted_task_loss = precision * task_loss + self.log_vars[i]
            
            total_loss += weighted_task_loss
        
        return total_loss


def example_task_weighting():
    """Example: Task weighting in multi-task learning."""
    # Create dataset with different task difficulties
    X = np.random.randn(1000, 10)
    y1 = (X[:, 0] + X[:, 1]).reshape(-1, 1)  # Easy task
    y2 = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]).reshape(-1, 1)  # Harder task
    
    # Create model
    model = TaskWeightedMultiTask(shared_units=64, num_tasks=2)
    
    # Custom training
    optimizer = keras.optimizers.Adam(0.001)
    
    print("\nTask Weighting Example:")
    
    for epoch in range(5):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = model.weighted_loss([y1, y2], y_pred)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print task weights
        weights = [tf.exp(-lv).numpy() for lv in model.log_vars]
        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Weights: {weights}")
    
    return model


# Pattern 7: Auxiliary Tasks
class AuxiliaryTaskModel(keras.Model):
    """Main task with auxiliary tasks for better representation learning."""
    
    def __init__(self, hidden_units=128, num_classes=10):
        super().__init__()
        
        # Shared encoder
        self.encoder = keras.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(hidden_units, activation='relu')
        ])
        
        # Main task head
        self.main_head = layers.Dense(num_classes, activation='softmax')
        
        # Auxiliary task heads
        self.aux_rotation = layers.Dense(4, activation='softmax')  # Rotation prediction
        self.aux_reconstruction = layers.Dense(10)  # Input reconstruction
    
    def call(self, inputs, training=False):
        features = self.encoder(inputs)
        
        # Main task
        main_output = self.main_head(features)
        
        if training:
            # Auxiliary tasks during training
            aux_rot = self.aux_rotation(features)
            aux_recon = self.aux_reconstruction(features)
            
            return main_output, aux_rot, aux_recon
        
        return main_output


def example_auxiliary_tasks():
    """Example: Auxiliary tasks for representation learning."""
    # Create dataset
    X = np.random.randn(1000, 10)
    y_main = keras.utils.to_categorical(np.random.randint(0, 10, 1000))
    y_rotation = keras.utils.to_categorical(np.random.randint(0, 4, 1000))
    y_recon = X  # Reconstruction target
    
    # Create model
    model = AuxiliaryTaskModel(hidden_units=64, num_classes=10)
    
    # Compile with multiple losses
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'mse'],
        loss_weights=[1.0, 0.3, 0.3],  # Weight auxiliary tasks lower
        metrics=[['accuracy'], ['accuracy'], ['mae']]
    )
    
    print("\nAuxiliary Tasks Example:")
    history = model.fit(
        X, [y_main, y_rotation, y_recon],
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    print(f"Main task accuracy: {history.history['dense_accuracy'][-1]:.4f}")
    
    return model


if __name__ == "__main__":
    print("Multi-Task and Multi-Modal Learning Patterns\n" + "="*60)
    
    # Example 1: Hard Parameter Sharing
    print("\n1. Hard Parameter Sharing")
    model1 = example_hard_parameter_sharing()
    
    # Example 2: Soft Parameter Sharing
    print("\n2. Soft Parameter Sharing")
    model2 = example_soft_parameter_sharing()
    
    # Example 3: Multi-Modal Fusion
    print("\n3. Multi-Modal Fusion")
    model3 = example_multimodal_fusion()
    
    # Example 4: Cross-Modal Attention
    print("\n4. Cross-Modal Attention")
    model4 = example_cross_modal_attention()
    
    # Example 5: Multi-Modal Contrastive Learning
    print("\n5. Multi-Modal Contrastive Learning")
    model5 = example_multimodal_contrastive()
    
    # Example 6: Task Weighting
    print("\n6. Task Weighting")
    model6 = example_task_weighting()
    
    # Example 7: Auxiliary Tasks
    print("\n7. Auxiliary Tasks")
    model7 = example_auxiliary_tasks()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Share parameters when tasks are related")
    print("2. Use soft sharing for loosely related tasks")
    print("3. Experiment with different fusion strategies")
    print("4. Use cross-modal attention for complementary modalities")
    print("5. Balance task losses with learned weights")
    print("6. Add auxiliary tasks for better representations")
    print("7. Monitor individual task performance")
    print("8. Use gradient clipping for stable multi-task training")
    print("9. Normalize embeddings for contrastive learning")
    print("10. Consider task interference and negative transfer")
