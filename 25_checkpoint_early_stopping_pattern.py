"""
Model Checkpoint and Early Stopping Pattern
Save models during training and stop when performance plateaus.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import os

print("Model Checkpoint and Early Stopping Pattern\n")

# Generate dummy data
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1))

# Create model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Example 1: Model Checkpoint
print("Example 1: Model Checkpoint")

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Save best model only
checkpoint_best = callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# Save model at each epoch
checkpoint_all = callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_valloss_{val_loss:.2f}.h5'),
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

print("Model checkpoints configured")

# Example 2: Early Stopping
print("\nExample 2: Early Stopping")

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement
    restore_best_weights=True,  # Restore weights from best epoch
    verbose=1,
    mode='min'
)

print("Early stopping will halt training after 5 epochs without improvement")

# Example 3: TensorBoard Callback
print("\nExample 3: TensorBoard Logging")

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

tensorboard = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log weight histograms every epoch
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch='2,5'  # Profile batches 2-5
)

print(f"TensorBoard logs will be saved to: {log_dir}")

# Example 4: Learning Rate Reduction
print("\nExample 4: Reduce Learning Rate on Plateau")

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce LR by half
    patience=3,
    min_lr=1e-7,
    verbose=1,
    mode='min'
)

print("Learning rate will be reduced when validation loss plateaus")

# Example 5: CSV Logger
print("\nExample 5: CSV Logger")

csv_logger = callbacks.CSVLogger(
    'training_log.csv',
    separator=',',
    append=False
)

print("Training metrics will be logged to training_log.csv")

# Example 6: Custom Callback
print("\nExample 6: Custom Callback")

class CustomCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Training is starting...")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} completed")
        print(f"  Train loss: {logs['loss']:.4f}, Train acc: {logs['accuracy']:.4f}")
        print(f"  Val loss: {logs['val_loss']:.4f}, Val acc: {logs['val_accuracy']:.4f}")
        
        # Custom logic: stop if accuracy > 95%
        if logs['accuracy'] > 0.95:
            print("Reached 95% accuracy! Stopping training.")
            self.model.stop_training = True
    
    def on_train_end(self, logs=None):
        print("Training has ended")

custom_callback = CustomCallback()

# Example 7: Train with Multiple Callbacks
print("\nExample 7: Training with Multiple Callbacks")

callback_list = [
    checkpoint_best,
    early_stop,
    reduce_lr,
    csv_logger,
    tensorboard,
    custom_callback
]

history = model.fit(
    X_train, y_train,
    epochs=20,  # Will stop early if performance plateaus
    batch_size=32,
    validation_split=0.2,
    callbacks=callback_list,
    verbose=1
)

# Example 8: Load Best Model
print("\nExample 8: Loading Saved Model")

# Load the best saved model
if os.path.exists(os.path.join(checkpoint_dir, 'best_model.h5')):
    best_model = keras.models.load_model(os.path.join(checkpoint_dir, 'best_model.h5'))
    print("Best model loaded successfully")
    
    # Evaluate
    loss, acc = best_model.evaluate(X_train[:200], y_train[:200], verbose=0)
    print(f"Best model - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Example 9: Checkpoint with Weights Only
print("\nExample 9: Saving Weights Only")

checkpoint_weights = callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'weights_epoch_{epoch:02d}.ckpt'),
    save_weights_only=True,
    save_best_only=False,
    verbose=0
)

# To load weights later:
# model.load_weights(os.path.join(checkpoint_dir, 'weights_epoch_10.ckpt'))

# Example 10: Backup and Restore Callback
print("\nExample 10: Backup and Restore")

backup_restore = callbacks.BackupAndRestore(
    backup_dir=os.path.join(checkpoint_dir, 'backup'),
    save_freq='epoch',
    delete_checkpoint=False
)

print("Backup and restore enables fault tolerance during training")

# Example 11: Terminate on NaN
print("\nExample 11: Terminate on NaN")

terminate_nan = callbacks.TerminateOnNaN()

print("Training will stop if NaN loss is detected")

# Example 12: Lambda Callback
print("\nExample 12: Lambda Callback")

# Simple callback using lambda
print_lr = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Learning rate: {model.optimizer.learning_rate.numpy()}")
)

# Example 13: Save Final Model
print("\nExample 13: Saving Final Model")

# Save in different formats
model.save('final_model.h5')  # HDF5 format
model.save('final_model_savedmodel')  # SavedModel format
model.save_weights('final_weights.h5')  # Weights only

print("Model saved in multiple formats")

print("\nCallback Best Practices:")
print("""
1. ModelCheckpoint:
   - Save best model based on validation metrics
   - Use informative filenames with metrics
   - Save full model or weights based on need

2. EarlyStopping:
   - Set appropriate patience (5-10 epochs)
   - Always restore_best_weights=True
   - Monitor validation metrics, not training

3. ReduceLROnPlateau:
   - Factor: 0.5 or 0.2
   - Patience: 3-5 epochs
   - Set minimum learning rate

4. TensorBoard:
   - Essential for tracking experiments
   - Visualize training progress
   - Compare multiple runs

5. Custom Callbacks:
   - Implement specific monitoring logic
   - Add custom metrics or visualizations
   - Implement warm restarts or cyclic training

6. Callback Order Matters:
   - BackupAndRestore should be first
   - EarlyStopping should be last
   - Others in between
""")

print("\nCleanup complete. Check 'checkpoints/', 'logs/', and saved models.")
