"""
Model Interpretability and Explainability Patterns

This module demonstrates techniques for understanding and explaining
neural network predictions in TensorFlow.

Patterns covered:
1. Gradient-based Saliency Maps
2. Integrated Gradients
3. Grad-CAM (Gradient-weighted Class Activation Mapping)
4. Attention Visualization
5. Layer Activation Visualization
6. Feature Importance
7. LIME (Local Interpretable Model-agnostic Explanations)
8. SHAP-inspired Attribution
9. Adversarial Examples
10. Model Uncertainty Quantification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Gradient-based Saliency Maps
class SaliencyMap:
    """Compute saliency maps using gradients."""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, images, class_idx=None):
        """Compute saliency map for given images."""
        images = tf.convert_to_tensor(images)
        
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images)
            
            if class_idx is None:
                # Use predicted class
                class_idx = tf.argmax(predictions, axis=-1)
            
            # Get score for target class
            target_class = tf.reduce_max(predictions, axis=-1)
        
        # Compute gradients
        gradients = tape.gradient(target_class, images)
        
        # Take absolute value and max across color channels
        saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
        
        return saliency.numpy()


def example_saliency_map():
    """Example: Saliency map visualization."""
    # Simple CNN model
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    # Dummy image
    image = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    # Compute saliency
    saliency = SaliencyMap(model)
    saliency_map = saliency.compute(image)
    
    print("Saliency Map Example:")
    print(f"Saliency map shape: {saliency_map.shape}")
    print("Shows which pixels are most important for prediction")
    
    return model, saliency


# Pattern 2: Integrated Gradients
class IntegratedGradients:
    """Compute integrated gradients for attribution."""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, images, baseline=None, steps=50):
        """Compute integrated gradients."""
        if baseline is None:
            baseline = tf.zeros_like(images)
        
        # Generate interpolated images
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        
        interpolated_images = []
        for alpha in alphas:
            interpolated = baseline + alpha * (images - baseline)
            interpolated_images.append(interpolated)
        
        interpolated_images = tf.concat(interpolated_images, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            predictions = self.model(interpolated_images)
            target_class = tf.argmax(predictions, axis=-1)
            target_score = tf.reduce_max(predictions, axis=-1)
        
        gradients = tape.gradient(target_score, interpolated_images)
        
        # Average gradients
        avg_gradients = tf.reduce_mean(gradients, axis=0, keepdims=True)
        
        # Multiply by (image - baseline)
        integrated_grads = (images - baseline) * avg_gradients
        
        return integrated_grads.numpy()


def example_integrated_gradients():
    """Example: Integrated gradients."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dense(10, activation='softmax')
    ])
    
    image = np.random.randn(1, 20).astype(np.float32)
    
    ig = IntegratedGradients(model)
    attributions = ig.compute(image)
    
    print("\nIntegrated Gradients Example:")
    print(f"Attribution shape: {attributions.shape}")
    print("Shows feature importance with better axioms than simple gradients")
    
    return ig


# Pattern 3: Grad-CAM
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        
        # Create a model that outputs both predictions and conv features
        self.grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.output, model.get_layer(layer_name).output]
        )
    
    def compute(self, images, class_idx=None):
        """Compute Grad-CAM heatmap."""
        images = tf.convert_to_tensor(images)
        
        with tf.GradientTape() as tape:
            predictions, conv_outputs = self.grad_model(images)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get score for target class
            class_output = predictions[:, class_idx]
        
        # Compute gradients of class output w.r.t. feature map
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination of feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
        
        # ReLU to keep only positive influences
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize heatmap
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()


def example_gradcam():
    """Example: Grad-CAM for CNN visualization."""
    # Build a simple CNN
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', name='conv2')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, 'conv2')
    
    # Dummy image
    image = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    heatmap = gradcam.compute(image)
    
    print("\nGrad-CAM Example:")
    print(f"Heatmap shape: {heatmap.shape}")
    print("Shows which regions of image are important for classification")
    
    return gradcam


# Pattern 4: Attention Visualization
def visualize_attention_weights(model, inputs):
    """Visualize attention weights from attention layers."""
    # Extract attention layer
    attention_layer = None
    for layer in model.layers:
        if 'attention' in layer.name.lower():
            attention_layer = layer
            break
    
    if attention_layer is None:
        print("No attention layer found")
        return None
    
    # Create model that outputs attention weights
    attention_model = keras.Model(
        inputs=model.input,
        outputs=attention_layer.output
    )
    
    attention_weights = attention_model(inputs)
    
    return attention_weights


def example_attention_visualization():
    """Example: Attention weight visualization."""
    # Simple model with attention
    inputs = keras.Input(shape=(10, 64))
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=16,
        name='multi_head_attention'
    )(inputs, inputs)
    
    x = layers.GlobalAveragePooling1D()(attention_output)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print("\nAttention Visualization Example:")
    print("Extract and visualize attention weights")
    print("Shows which parts of input the model focuses on")
    
    return model


# Pattern 5: Layer Activation Visualization
class LayerActivationVisualizer:
    """Visualize intermediate layer activations."""
    
    def __init__(self, model, layer_names=None):
        self.model = model
        
        if layer_names is None:
            # Get all convolutional and dense layers
            layer_names = [
                layer.name for layer in model.layers
                if isinstance(layer, (layers.Conv2D, layers.Dense))
            ]
        
        self.layer_names = layer_names
        
        # Create models for each layer
        self.activation_models = {}
        for name in layer_names:
            self.activation_models[name] = keras.Model(
                inputs=model.input,
                outputs=model.get_layer(name).output
            )
    
    def visualize(self, image):
        """Get activations for all layers."""
        activations = {}
        for name, activation_model in self.activation_models.items():
            activations[name] = activation_model(image).numpy()
        
        return activations


def example_layer_activation():
    """Example: Layer activation visualization."""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D(name='pool1'),
        layers.Conv2D(64, 3, activation='relu', name='conv2'),
        layers.MaxPooling2D(name='pool2'),
        layers.Flatten(name='flatten'),
        layers.Dense(10, activation='softmax', name='output')
    ])
    
    visualizer = LayerActivationVisualizer(model)
    
    # Dummy image
    image = np.random.randn(1, 28, 28, 1).astype(np.float32)
    
    activations = visualizer.visualize(image)
    
    print("\nLayer Activation Visualization:")
    for name, activation in activations.items():
        print(f"{name}: {activation.shape}")
    
    return visualizer


# Pattern 6: Feature Importance (Permutation)
class PermutationImportance:
    """Compute feature importance using permutation."""
    
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
    
    def compute(self, x, y, n_repeats=10):
        """Compute permutation importance for each feature."""
        # Baseline score
        baseline_pred = self.model(x, training=False)
        baseline_loss = self.loss_fn(y, baseline_pred).numpy()
        
        n_features = x.shape[-1]
        importances = np.zeros((n_features, n_repeats))
        
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # Permute feature
                x_permuted = x.numpy().copy()
                np.random.shuffle(x_permuted[:, feature_idx])
                x_permuted = tf.convert_to_tensor(x_permuted)
                
                # Compute loss with permuted feature
                pred = self.model(x_permuted, training=False)
                loss = self.loss_fn(y, pred).numpy()
                
                # Importance is increase in loss
                importances[feature_idx, repeat] = loss - baseline_loss
        
        # Average over repeats
        mean_importance = np.mean(importances, axis=1)
        std_importance = np.std(importances, axis=1)
        
        return mean_importance, std_importance


def example_feature_importance():
    """Example: Permutation feature importance."""
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(10,)),
        layers.Dense(1)
    ])
    
    # Dummy data
    x = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    
    loss_fn = keras.losses.MeanSquaredError()
    
    pi = PermutationImportance(model, loss_fn)
    importance, std = pi.compute(x, y, n_repeats=5)
    
    print("\nPermutation Importance Example:")
    print(f"Feature importances shape: {importance.shape}")
    print("Higher values indicate more important features")
    
    return pi


# Pattern 7: Model Uncertainty Quantification
class MCDropout(keras.Model):
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, base_model, dropout_rate=0.5):
        super().__init__()
        self.base_model = base_model
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        # Always apply dropout for uncertainty estimation
        x = self.dropout(x, training=True)
        return x
    
    def predict_with_uncertainty(self, inputs, n_samples=100):
        """Predict with uncertainty estimation."""
        predictions = []
        
        for _ in range(n_samples):
            pred = self(inputs, training=False)
            predictions.append(pred)
        
        predictions = tf.stack(predictions)
        
        # Mean prediction
        mean = tf.reduce_mean(predictions, axis=0)
        
        # Uncertainty (variance)
        variance = tf.math.reduce_variance(predictions, axis=0)
        
        return mean.numpy(), variance.numpy()


def example_uncertainty_quantification():
    """Example: Uncertainty quantification with MC Dropout."""
    base_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    mc_model = MCDropout(base_model, dropout_rate=0.3)
    
    # Dummy input
    x = np.random.randn(5, 20).astype(np.float32)
    
    mean, variance = mc_model.predict_with_uncertainty(x, n_samples=50)
    
    print("\nUncertainty Quantification Example:")
    print(f"Mean predictions shape: {mean.shape}")
    print(f"Variance (uncertainty) shape: {variance.shape}")
    print("Higher variance indicates higher model uncertainty")
    
    return mc_model


# Pattern 8: Adversarial Examples (FGSM)
class AdversarialExamples:
    """Generate adversarial examples using FGSM."""
    
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
    
    def fgsm_attack(self, images, labels, epsilon=0.01):
        """Fast Gradient Sign Method attack."""
        images = tf.convert_to_tensor(images)
        labels = tf.convert_to_tensor(labels)
        
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images, training=False)
            loss = self.loss_fn(labels, predictions)
        
        # Get gradient of loss w.r.t. images
        gradients = tape.gradient(loss, images)
        
        # Create adversarial examples
        signed_grad = tf.sign(gradients)
        adversarial_images = images + epsilon * signed_grad
        
        # Clip to valid range
        adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
        
        return adversarial_images.numpy()
    
    def pgd_attack(self, images, labels, epsilon=0.01, alpha=0.001, num_steps=10):
        """Projected Gradient Descent attack."""
        adv_images = tf.identity(images)
        
        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(adv_images)
                predictions = self.model(adv_images, training=False)
                loss = self.loss_fn(labels, predictions)
            
            gradients = tape.gradient(loss, adv_images)
            adv_images = adv_images + alpha * tf.sign(gradients)
            
            # Project back to epsilon ball
            perturbation = adv_images - images
            perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
            adv_images = images + perturbation
            
            # Clip to valid range
            adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)
        
        return adv_images.numpy()


def example_adversarial_examples():
    """Example: Generate adversarial examples."""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    
    adv = AdversarialExamples(model, loss_fn)
    
    # Dummy data
    images = np.random.rand(5, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, 5)
    
    adversarial_images = adv.fgsm_attack(images, labels, epsilon=0.1)
    
    print("\nAdversarial Examples:")
    print(f"Original images shape: {images.shape}")
    print(f"Adversarial images shape: {adversarial_images.shape}")
    print("Adversarial examples can fool the model")
    
    return adv


# Pattern 9: Concept Activation Vectors (TCAV-inspired)
def compute_directional_derivative(model, layer_name, images, concept_vector):
    """Compute directional derivative along concept vector."""
    # Create model up to target layer
    layer_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    with tf.GradientTape() as tape:
        tape.watch(images)
        activations = layer_model(images)
        # Project onto concept vector
        projection = tf.reduce_sum(activations * concept_vector, axis=-1)
    
    gradients = tape.gradient(projection, images)
    
    return gradients


def example_concept_activation():
    """Example: Concept-based explanation."""
    print("\nConcept Activation Vectors (TCAV-inspired):")
    print("Measures model sensitivity to human-friendly concepts")
    print("Requires concept examples to compute concept vectors")
    print("Useful for understanding what concepts the model uses")


# Pattern 10: Influence Functions (Simplified)
def compute_influence_score(model, train_sample, test_sample, loss_fn):
    """Simplified influence score computation."""
    # This is a simplified version - full implementation requires Hessian
    
    with tf.GradientTape() as tape:
        pred = model(tf.expand_dims(train_sample, 0), training=False)
        loss = loss_fn(tf.constant([[0]]), pred)
    
    train_grad = tape.gradient(loss, model.trainable_variables)
    
    with tf.GradientTape() as tape:
        pred = model(tf.expand_dims(test_sample, 0), training=False)
        loss = loss_fn(tf.constant([[0]]), pred)
    
    test_grad = tape.gradient(loss, model.trainable_variables)
    
    # Dot product of gradients (simplified influence)
    influence = sum([
        tf.reduce_sum(tg * ttg)
        for tg, ttg in zip(train_grad, test_grad)
    ])
    
    return influence.numpy()


def example_influence_functions():
    """Example: Influence functions for training data importance."""
    print("\nInfluence Functions:")
    print("Identify which training samples most influenced a prediction")
    print("Useful for debugging and understanding model behavior")
    print("Can help identify mislabeled or outlier training examples")


if __name__ == "__main__":
    print("Model Interpretability Patterns\n" + "="*60)
    
    # Example 1: Saliency Maps
    print("\n1. Saliency Maps")
    model1, saliency = example_saliency_map()
    
    # Example 2: Integrated Gradients
    print("\n2. Integrated Gradients")
    ig = example_integrated_gradients()
    
    # Example 3: Grad-CAM
    print("\n3. Grad-CAM")
    gradcam = example_gradcam()
    
    # Example 4: Attention Visualization
    print("\n4. Attention Visualization")
    attention_model = example_attention_visualization()
    
    # Example 5: Layer Activations
    print("\n5. Layer Activation Visualization")
    visualizer = example_layer_activation()
    
    # Example 6: Feature Importance
    print("\n6. Permutation Feature Importance")
    pi = example_feature_importance()
    
    # Example 7: Uncertainty Quantification
    print("\n7. Uncertainty Quantification (MC Dropout)")
    mc_model = example_uncertainty_quantification()
    
    # Example 8: Adversarial Examples
    print("\n8. Adversarial Examples")
    adv = example_adversarial_examples()
    
    # Example 9: Concept Activation
    print("\n9. Concept Activation Vectors")
    example_concept_activation()
    
    # Example 10: Influence Functions
    print("\n10. Influence Functions")
    example_influence_functions()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Use multiple interpretation methods for robust understanding")
    print("2. Saliency maps: good for pixel-level attribution")
    print("3. Integrated Gradients: satisfies important axioms")
    print("4. Grad-CAM: best for CNNs, shows spatial importance")
    print("5. Attention weights: directly interpretable for transformers")
    print("6. MC Dropout: simple uncertainty quantification")
    print("7. Test adversarial robustness for safety-critical applications")
    print("8. Feature importance: good for tabular data")
    print("9. Visualize layer activations to debug representations")
    print("10. Combine multiple methods for comprehensive analysis")
