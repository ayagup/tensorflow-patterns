"""
Model Deployment Pattern
Save, export, and serve TensorFlow models for production.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

print("Model Deployment Pattern\n")

# Create and train a sample model
X_train = np.random.random((1000, 20)).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1))

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='deployment_model')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

print("Model trained for deployment examples\n")

# Example 1: SavedModel Format (Recommended)
print("Example 1: SavedModel Format")

saved_model_path = 'saved_model_dir'
model.save(saved_model_path)
print(f"Model saved to: {saved_model_path}")

# Load SavedModel
loaded_model = keras.models.load_model(saved_model_path)
print("Model loaded successfully")

# Verify
test_input = X_train[:1]
original_pred = model.predict(test_input, verbose=0)
loaded_pred = loaded_model.predict(test_input, verbose=0)
print(f"Predictions match: {np.allclose(original_pred, loaded_pred)}")

# Example 2: HDF5 Format
print("\nExample 2: HDF5 Format")

h5_path = 'model.h5'
model.save(h5_path)
print(f"Model saved as: {h5_path}")

# Load HDF5
loaded_h5 = keras.models.load_model(h5_path)
print("HDF5 model loaded")

# Example 3: Weights Only
print("\nExample 3: Save/Load Weights Only")

weights_path = 'model_weights.h5'
model.save_weights(weights_path)
print(f"Weights saved to: {weights_path}")

# To load weights, need same architecture
new_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

new_model.load_weights(weights_path)
print("Weights loaded into new model")

# Example 4: TensorFlow Lite Conversion
print("\nExample 4: TensorFlow Lite for Mobile/Edge")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved: {tflite_path}")

# Use TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make prediction
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
tflite_pred = interpreter.get_tensor(output_details[0]['index'])
print(f"TFLite prediction: {tflite_pred[0][0]:.4f}")

# Example 5: TFLite with Quantization
print("\nExample 5: TFLite with Quantization (Smaller Size)")

# Quantization for smaller model size
converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for calibration
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1]]

converter_quant.representative_dataset = representative_dataset
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.int8
converter_quant.inference_output_type = tf.int8

tflite_quant_model = converter_quant.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print(f"Original size: {len(tflite_model)} bytes")
print(f"Quantized size: {len(tflite_quant_model)} bytes")
print(f"Size reduction: {(1 - len(tflite_quant_model)/len(tflite_model))*100:.1f}%")

# Example 6: TensorFlow.js Conversion
print("\nExample 6: TensorFlow.js for Web Deployment")

tfjs_path = 'tfjs_model'
# Requires tensorflowjs package: pip install tensorflowjs
# tfjs.converters.save_keras_model(model, tfjs_path)
print(f"TensorFlow.js model would be saved to: {tfjs_path}")
print("Install: pip install tensorflowjs")
print("Convert: tensorflowjs_converter --input_format=keras model.h5 tfjs_model/")

# Example 7: ONNX Export
print("\nExample 7: ONNX Format for Interoperability")

# Requires tf2onnx: pip install tf2onnx
print("ONNX allows deploying TF models on other frameworks")
print("Convert: python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx")

# Example 8: Model Serving with Preprocessing
print("\nExample 8: SavedModel with Preprocessing")

class DeploymentModel(keras.Model):
    """Model with built-in preprocessing."""
    def __init__(self, base_model):
        super(DeploymentModel, self).__init__()
        self.base_model = base_model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 20], dtype=tf.float32)])
    def serve(self, inputs):
        # Preprocessing
        x = inputs / 255.0  # Example normalization
        # Prediction
        return self.base_model(x, training=False)

deployment_model = DeploymentModel(model)
print("Model with preprocessing for consistent inference")

# Example 9: Model Signatures
print("\nExample 9: Custom Serving Signatures")

# Define custom serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 20], dtype=tf.float32)])
def serving_fn(inputs):
    outputs = model(inputs, training=False)
    return {'predictions': outputs}

# Save with signature
signatures = {
    'serving_default': serving_fn,
    'predict': serving_fn
}

tf.saved_model.save(
    model,
    'model_with_signatures',
    signatures=signatures
)
print("Model saved with custom signatures")

# Example 10: Model Versioning
print("\nExample 10: Model Versioning Strategy")

def save_versioned_model(model, base_path, version):
    """Save model with version number."""
    version_path = os.path.join(base_path, f'v{version}')
    model.save(version_path)
    print(f"Model v{version} saved to: {version_path}")
    return version_path

# Save different versions
v1_path = save_versioned_model(model, 'models', version=1)
# After improvements, save v2
v2_path = save_versioned_model(model, 'models', version=2)

print("Version control enables A/B testing and rollback")

# Example 11: Batch Prediction Pipeline
print("\nExample 11: Batch Prediction Pipeline")

def batch_predict(model_path, input_data, batch_size=32):
    """Efficient batch prediction."""
    loaded = keras.models.load_model(model_path)
    
    predictions = []
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i+batch_size]
        pred = loaded.predict(batch, verbose=0)
        predictions.append(pred)
    
    return np.vstack(predictions)

# Example batch prediction
batch_preds = batch_predict(saved_model_path, X_train[:100], batch_size=32)
print(f"Batch predictions shape: {batch_preds.shape}")

# Example 12: Model Metadata
print("\nExample 12: Save Model with Metadata")

metadata = {
    'model_name': 'deployment_model',
    'version': '1.0.0',
    'training_date': '2025-10-25',
    'accuracy': 0.85,
    'input_shape': [20],
    'output_classes': 2,
    'framework': 'TensorFlow 2.x'
}

# Save metadata
import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved for model documentation")

# Example 13: Docker Deployment Preparation
print("\nExample 13: Prepare for Docker Deployment")

dockerfile_content = """
FROM tensorflow/tensorflow:latest

WORKDIR /app

# Copy model and dependencies
COPY saved_model_dir /app/model
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy serving script
COPY serve.py /app/

EXPOSE 8501

CMD ["python", "serve.py"]
"""

print("Example Dockerfile for TensorFlow model:")
print(dockerfile_content)

# Example 14: REST API Serving Script
print("\nExample 14: Simple REST API (Flask)")

flask_serving = """
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('saved_model_dir')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    input_data = np.array(data)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""

print("Flask serving script example:")
print(flask_serving[:200] + "...")

print("\nDeployment Best Practices:")
print("""
1. SavedModel Format:
   - Recommended for production
   - Contains graph and weights
   - Platform independent

2. Model Optimization:
   - TFLite for mobile/edge
   - Quantization for smaller size
   - Pruning for faster inference

3. Versioning:
   - Semantic versioning (v1.0.0)
   - Track model lineage
   - Enable rollback

4. Serving:
   - TensorFlow Serving for scale
   - REST/gRPC APIs
   - Batch prediction for efficiency

5. Monitoring:
   - Log predictions and inputs
   - Track latency and throughput
   - Monitor model drift

6. Security:
   - Validate inputs
   - Rate limiting
   - Authentication

7. Preprocessing:
   - Include in model graph
   - Consistent across train/serve
   - Reduce client complexity

8. Testing:
   - Unit tests for inference
   - Load testing
   - A/B testing

9. Documentation:
   - Input/output specs
   - Performance metrics
   - API documentation

10. Formats:
    - SavedModel: General purpose
    - TFLite: Mobile/embedded
    - TensorFlow.js: Web browsers
    - ONNX: Cross-platform
""")
