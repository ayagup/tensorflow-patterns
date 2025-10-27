"""
Ensemble Pattern
Combine multiple models to improve prediction accuracy and robustness.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("Ensemble Pattern\n")

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 3, (1000,))
X_test = np.random.random((200, 20)).astype(np.float32)
y_test = np.random.randint(0, 3, (200,))

# Example 1: Simple Averaging Ensemble
print("Example 1: Simple Averaging Ensemble")

def create_base_model(name):
    """Create a base model with different architecture."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ], name=name)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train multiple models
num_models = 3
models = []

for i in range(num_models):
    print(f"\nTraining model {i+1}/{num_models}")
    model = create_base_model(f'model_{i+1}')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0, validation_split=0.2)
    models.append(model)

# Make predictions with ensemble
predictions = np.array([model.predict(X_test, verbose=0) for model in models])
ensemble_pred = np.mean(predictions, axis=0)
ensemble_class = np.argmax(ensemble_pred, axis=1)

# Evaluate
individual_accs = [model.evaluate(X_test, y_test, verbose=0)[1] for model in models]
ensemble_acc = np.mean(ensemble_class == y_test)

print(f"\nIndividual accuracies: {[f'{acc:.3f}' for acc in individual_accs]}")
print(f"Ensemble accuracy: {ensemble_acc:.3f}")

# Example 2: Weighted Ensemble
print("\nExample 2: Weighted Ensemble")

# Assign weights based on validation performance
weights = np.array(individual_accs)
weights = weights / np.sum(weights)  # Normalize

weighted_pred = np.sum([w * pred for w, pred in zip(weights, predictions)], axis=0)
weighted_class = np.argmax(weighted_pred, axis=1)
weighted_acc = np.mean(weighted_class == y_test)

print(f"Weights: {weights}")
print(f"Weighted ensemble accuracy: {weighted_acc:.3f}")

# Example 3: Voting Ensemble
print("\nExample 3: Majority Voting Ensemble")

# Get class predictions from each model
class_predictions = np.array([np.argmax(pred, axis=1) for pred in predictions])

# Majority vote
from scipy import stats
voting_pred, _ = stats.mode(class_predictions, axis=0)
voting_acc = np.mean(voting_pred.flatten() == y_test)

print(f"Voting ensemble accuracy: {voting_acc:.3f}")

# Example 4: Stacking Ensemble
print("\nExample 4: Stacking Ensemble")

# Get predictions from base models
base_predictions_train = np.column_stack([
    model.predict(X_train, verbose=0) for model in models
])

base_predictions_test = np.column_stack([
    model.predict(X_test, verbose=0) for model in models
])

# Meta-learner
meta_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(num_models * 3,)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
], name='meta_learner')

meta_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train meta-learner
meta_model.fit(base_predictions_train, y_train, epochs=10, batch_size=32, verbose=0)

# Evaluate stacking
stacking_pred = meta_model.predict(base_predictions_test, verbose=0)
stacking_class = np.argmax(stacking_pred, axis=1)
stacking_acc = np.mean(stacking_class == y_test)

print(f"Stacking ensemble accuracy: {stacking_acc:.3f}")

# Example 5: Bagging with Bootstrap
print("\nExample 5: Bagging (Bootstrap Aggregating)")

def create_bootstrap_sample(X, y, sample_size=None):
    """Create bootstrap sample."""
    if sample_size is None:
        sample_size = len(X)
    
    indices = np.random.choice(len(X), size=sample_size, replace=True)
    return X[indices], y[indices]

# Train models on bootstrap samples
bagging_models = []
num_bags = 5

for i in range(num_bags):
    X_boot, y_boot = create_bootstrap_sample(X_train, y_train)
    model = create_base_model(f'bagging_model_{i+1}')
    model.fit(X_boot, y_boot, epochs=5, batch_size=32, verbose=0)
    bagging_models.append(model)

# Bagging prediction
bagging_predictions = np.array([model.predict(X_test, verbose=0) for model in bagging_models])
bagging_pred = np.mean(bagging_predictions, axis=0)
bagging_class = np.argmax(bagging_pred, axis=1)
bagging_acc = np.mean(bagging_class == y_test)

print(f"Bagging ensemble accuracy: {bagging_acc:.3f}")

# Example 6: Diverse Architectures Ensemble
print("\nExample 6: Ensemble with Diverse Architectures")

# Model 1: Deep and narrow
model_deep = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(20,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
], name='deep_narrow')

# Model 2: Shallow and wide
model_wide = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
], name='shallow_wide')

# Model 3: With batch normalization
model_bn = keras.Sequential([
    layers.Dense(64, input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(3, activation='softmax')
], name='with_bn')

diverse_models = [model_deep, model_wide, model_bn]

for model in diverse_models:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Ensemble prediction
diverse_predictions = np.array([model.predict(X_test, verbose=0) for model in diverse_models])
diverse_pred = np.mean(diverse_predictions, axis=0)
diverse_class = np.argmax(diverse_pred, axis=1)
diverse_acc = np.mean(diverse_class == y_test)

print(f"Diverse architectures ensemble accuracy: {diverse_acc:.3f}")

# Example 7: Snapshot Ensemble
print("\nExample 7: Snapshot Ensemble")

class SnapshotEnsemble(keras.callbacks.Callback):
    """Save model snapshots during training."""
    def __init__(self, epochs_per_cycle, num_cycles):
        super(SnapshotEnsemble, self).__init__()
        self.epochs_per_cycle = epochs_per_cycle
        self.num_cycles = num_cycles
        self.snapshots = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Save at end of each cycle
        if (epoch + 1) % self.epochs_per_cycle == 0:
            snapshot = keras.models.clone_model(self.model)
            snapshot.set_weights(self.model.get_weights())
            self.snapshots.append(snapshot)
            print(f"\nSnapshot saved at epoch {epoch + 1}")

snapshot_callback = SnapshotEnsemble(epochs_per_cycle=5, num_cycles=3)

model_snapshot = create_base_model('snapshot_model')
model_snapshot.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    verbose=0,
    callbacks=[snapshot_callback]
)

# Use snapshots as ensemble
if snapshot_callback.snapshots:
    snapshot_predictions = np.array([
        model.predict(X_test, verbose=0) for model in snapshot_callback.snapshots
    ])
    snapshot_pred = np.mean(snapshot_predictions, axis=0)
    snapshot_class = np.argmax(snapshot_pred, axis=1)
    snapshot_acc = np.mean(snapshot_class == y_test)
    print(f"Snapshot ensemble accuracy: {snapshot_acc:.3f}")

# Example 8: Ensemble Class
print("\nExample 8: Reusable Ensemble Class")

class EnsembleModel:
    """Ensemble of multiple models."""
    def __init__(self, models, method='average', weights=None):
        self.models = models
        self.method = method
        self.weights = weights
    
    def predict(self, X):
        predictions = np.array([model.predict(X, verbose=0) for model in self.models])
        
        if self.method == 'average':
            return np.mean(predictions, axis=0)
        elif self.method == 'weighted':
            if self.weights is None:
                raise ValueError("Weights required for weighted ensemble")
            return np.sum([w * pred for w, pred in zip(self.weights, predictions)], axis=0)
        elif self.method == 'voting':
            class_preds = np.array([np.argmax(pred, axis=1) for pred in predictions])
            voting_result, _ = stats.mode(class_preds, axis=0)
            return voting_result.flatten()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def evaluate(self, X, y):
        pred = self.predict(X)
        if self.method == 'voting':
            acc = np.mean(pred == y)
        else:
            class_pred = np.argmax(pred, axis=1)
            acc = np.mean(class_pred == y)
        return acc

# Create ensemble
ensemble = EnsembleModel(models, method='average')
accuracy = ensemble.evaluate(X_test, y_test)
print(f"Ensemble class accuracy: {accuracy:.3f}")

# Example 9: Uncertainty Estimation
print("\nExample 9: Uncertainty Estimation with Ensemble")

# Get prediction variance
test_sample = X_test[:1]
sample_predictions = np.array([model.predict(test_sample, verbose=0) for model in models])

mean_pred = np.mean(sample_predictions, axis=0)
std_pred = np.std(sample_predictions, axis=0)

print(f"Prediction: {mean_pred[0]}")
print(f"Uncertainty (std): {std_pred[0]}")
print("Higher uncertainty indicates disagreement among models")

print("\nEnsemble Best Practices:")
print("""
1. Averaging Ensemble:
   - Simple and effective
   - Works well with 3-10 models
   - Reduces variance

2. Weighted Ensemble:
   - Weight by validation performance
   - Better than simple average
   - Requires validation set

3. Voting Ensemble:
   - Good for classification
   - Robust to outliers
   - Odd number of models preferred

4. Stacking:
   - Most powerful but complex
   - Meta-learner learns optimal combination
   - Risk of overfitting

5. Bagging:
   - Reduces variance
   - Bootstrap sampling creates diversity
   - Good for unstable models

6. Diverse Architectures:
   - Different architectures capture different patterns
   - More diversity = better ensemble
   - Mix deep/shallow, wide/narrow

7. Snapshot Ensemble:
   - Single training run
   - Efficient use of computation
   - Use with cosine annealing

8. Uncertainty:
   - Ensemble disagreement indicates uncertainty
   - Useful for active learning
   - Helps identify out-of-distribution samples

General Tips:
- More diverse models = better ensemble
- 3-5 models often sufficient
- Diminishing returns beyond 10 models
- Computational cost increases linearly
- Test-time inference slower
""")
