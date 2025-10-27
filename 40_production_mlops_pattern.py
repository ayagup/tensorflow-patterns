"""
Production and MLOps Patterns

This module demonstrates patterns for deploying and managing ML models
in production environments.

Patterns covered:
1. Input Validation and Preprocessing
2. Model Versioning
3. A/B Testing Framework
4. Model Monitoring
5. Feature Store Pattern
6. Model Registry
7. Experiment Tracking
8. Online Learning / Continuous Training
9. Model Serving with TensorFlow Serving
10. Error Handling and Logging
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
from datetime import datetime
from collections import defaultdict


# Pattern 1: Input Validation and Preprocessing
class InputValidator:
    """Validate and preprocess input data."""
    
    def __init__(self, feature_spec):
        """
        feature_spec: dict with feature names and expected properties
        Example: {
            'feature1': {'type': 'float', 'min': 0, 'max': 1, 'required': True},
            'feature2': {'type': 'int', 'min': 0, 'max': 100, 'required': False}
        }
        """
        self.feature_spec = feature_spec
    
    def validate(self, data):
        """Validate input data against spec."""
        errors = []
        
        for feature_name, spec in self.feature_spec.items():
            # Check required fields
            if spec.get('required', False) and feature_name not in data:
                errors.append(f"Missing required feature: {feature_name}")
                continue
            
            if feature_name not in data:
                continue
            
            value = data[feature_name]
            
            # Type validation
            expected_type = spec.get('type')
            if expected_type == 'float':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    errors.append(f"{feature_name}: Expected float, got {type(value)}")
                    continue
            elif expected_type == 'int':
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    errors.append(f"{feature_name}: Expected int, got {type(value)}")
                    continue
            
            # Range validation
            if 'min' in spec and value < spec['min']:
                errors.append(f"{feature_name}: Value {value} below minimum {spec['min']}")
            
            if 'max' in spec and value > spec['max']:
                errors.append(f"{feature_name}: Value {value} above maximum {spec['max']}")
        
        return len(errors) == 0, errors
    
    def preprocess(self, data):
        """Preprocess validated data."""
        processed = {}
        
        for feature_name, spec in self.feature_spec.items():
            if feature_name in data:
                value = data[feature_name]
                
                # Type conversion
                if spec.get('type') == 'float':
                    processed[feature_name] = float(value)
                elif spec.get('type') == 'int':
                    processed[feature_name] = int(value)
                else:
                    processed[feature_name] = value
            else:
                # Use default if provided
                if 'default' in spec:
                    processed[feature_name] = spec['default']
        
        return processed


def example_input_validation():
    """Example: Input validation."""
    feature_spec = {
        'age': {'type': 'int', 'min': 0, 'max': 120, 'required': True},
        'income': {'type': 'float', 'min': 0, 'required': True},
        'score': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.5}
    }
    
    validator = InputValidator(feature_spec)
    
    # Valid input
    valid_data = {'age': 25, 'income': 50000.0, 'score': 0.8}
    is_valid, errors = validator.validate(valid_data)
    
    print("Input Validation Example:")
    print(f"Valid data: {valid_data}")
    print(f"Is valid: {is_valid}")
    
    # Invalid input
    invalid_data = {'age': 150, 'income': -1000}
    is_valid, errors = validator.validate(invalid_data)
    print(f"\nInvalid data: {invalid_data}")
    print(f"Is valid: {is_valid}")
    print(f"Errors: {errors}")
    
    return validator


# Pattern 2: Model Versioning
class ModelVersion:
    """Track model versions and metadata."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.versions = {}
        self.current_version = None
    
    def register_version(self, version, model, metadata=None):
        """Register a new model version."""
        self.versions[version] = {
            'model': model,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'metrics': {}
        }
        
        print(f"Registered {self.model_name} version {version}")
    
    def set_current_version(self, version):
        """Set the currently active version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        self.current_version = version
        print(f"Set current version to {version}")
    
    def get_model(self, version=None):
        """Get model by version (or current version)."""
        version = version or self.current_version
        
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        return self.versions[version]['model']
    
    def update_metrics(self, version, metrics):
        """Update metrics for a version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        self.versions[version]['metrics'].update(metrics)
    
    def list_versions(self):
        """List all registered versions."""
        return list(self.versions.keys())


def example_model_versioning():
    """Example: Model versioning."""
    versioning = ModelVersion('classifier')
    
    # Register versions
    model_v1 = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(2, activation='softmax')
    ])
    
    model_v2 = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    versioning.register_version('v1.0', model_v1, {'description': 'Initial model'})
    versioning.register_version('v2.0', model_v2, {'description': 'Larger model'})
    
    versioning.set_current_version('v2.0')
    
    print("\nModel Versioning Example:")
    print(f"Available versions: {versioning.list_versions()}")
    print(f"Current version: {versioning.current_version}")
    
    return versioning


# Pattern 3: A/B Testing Framework
class ABTestingFramework:
    """Framework for A/B testing models."""
    
    def __init__(self):
        self.models = {}
        self.traffic_split = {}
        self.metrics = defaultdict(lambda: {'predictions': 0, 'correct': 0})
    
    def register_variant(self, variant_name, model, traffic_percentage):
        """Register a model variant with traffic percentage."""
        self.models[variant_name] = model
        self.traffic_split[variant_name] = traffic_percentage
        
        print(f"Registered variant '{variant_name}' with {traffic_percentage}% traffic")
    
    def select_variant(self):
        """Select variant based on traffic split."""
        rand = np.random.random() * 100
        
        cumulative = 0
        for variant, percentage in self.traffic_split.items():
            cumulative += percentage
            if rand < cumulative:
                return variant
        
        # Default to first variant
        return list(self.models.keys())[0]
    
    def predict(self, inputs, variant=None):
        """Make prediction using selected or specified variant."""
        if variant is None:
            variant = self.select_variant()
        
        model = self.models[variant]
        prediction = model(inputs, training=False)
        
        return prediction, variant
    
    def record_result(self, variant, correct):
        """Record prediction result for metrics."""
        self.metrics[variant]['predictions'] += 1
        if correct:
            self.metrics[variant]['correct'] += 1
    
    def get_metrics(self):
        """Get accuracy metrics for all variants."""
        results = {}
        
        for variant, stats in self.metrics.items():
            if stats['predictions'] > 0:
                accuracy = stats['correct'] / stats['predictions']
                results[variant] = {
                    'accuracy': accuracy,
                    'predictions': stats['predictions']
                }
        
        return results


def example_ab_testing():
    """Example: A/B testing framework."""
    framework = ABTestingFramework()
    
    # Create two model variants
    model_a = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(10,)),
        layers.Dense(2, activation='softmax')
    ])
    
    model_b = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(2, activation='softmax')
    ])
    
    # Register variants with 50/50 split
    framework.register_variant('model_a', model_a, 50)
    framework.register_variant('model_b', model_b, 50)
    
    print("\nA/B Testing Framework Example:")
    print("Testing two model variants with 50/50 traffic split")
    
    # Simulate predictions
    for _ in range(10):
        inputs = tf.random.normal((1, 10))
        prediction, variant = framework.predict(inputs)
        framework.record_result(variant, correct=np.random.random() > 0.5)
    
    metrics = framework.get_metrics()
    print(f"\nMetrics: {metrics}")
    
    return framework


# Pattern 4: Model Monitoring
class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.predictions = []
        self.actuals = []
        self.latencies = []
        self.errors = []
    
    def log_prediction(self, input_data, prediction, actual=None, latency=None):
        """Log a prediction with optional ground truth and latency."""
        self.predictions.append({
            'timestamp': datetime.now().isoformat(),
            'input_shape': input_data.shape if hasattr(input_data, 'shape') else None,
            'prediction': prediction,
            'actual': actual,
            'latency': latency
        })
        
        if actual is not None:
            self.actuals.append(actual)
        
        if latency is not None:
            self.latencies.append(latency)
        
        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            if self.actuals:
                self.actuals.pop(0)
            if self.latencies:
                self.latencies.pop(0)
    
    def log_error(self, error_type, error_message):
        """Log an error."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message
        })
    
    def get_statistics(self):
        """Get monitoring statistics."""
        stats = {
            'total_predictions': len(self.predictions),
            'error_count': len(self.errors)
        }
        
        if self.latencies:
            stats['avg_latency_ms'] = np.mean(self.latencies)
            stats['p95_latency_ms'] = np.percentile(self.latencies, 95)
            stats['p99_latency_ms'] = np.percentile(self.latencies, 99)
        
        if self.actuals and self.predictions:
            # Calculate accuracy
            correct = sum([
                1 for p, a in zip(self.predictions[-len(self.actuals):], self.actuals)
                if np.argmax(p['prediction']) == a
            ])
            stats['accuracy'] = correct / len(self.actuals)
        
        return stats
    
    def check_drift(self, reference_mean, threshold=0.1):
        """Check for data drift in predictions."""
        if len(self.predictions) < 10:
            return False, 0.0
        
        recent_predictions = [p['prediction'] for p in self.predictions[-10:]]
        recent_mean = np.mean([np.argmax(p) for p in recent_predictions])
        
        drift = abs(recent_mean - reference_mean)
        
        return drift > threshold, drift


def example_model_monitoring():
    """Example: Model monitoring."""
    monitor = ModelMonitor(window_size=100)
    
    print("\nModel Monitoring Example:")
    
    # Simulate predictions
    for i in range(20):
        inputs = tf.random.normal((1, 10))
        prediction = tf.nn.softmax(tf.random.normal((1, 5)))
        actual = np.random.randint(0, 5)
        latency = np.random.uniform(10, 50)  # ms
        
        monitor.log_prediction(inputs, prediction.numpy(), actual, latency)
    
    # Simulate errors
    monitor.log_error('ValueError', 'Invalid input shape')
    monitor.log_error('TimeoutError', 'Prediction timeout')
    
    stats = monitor.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # Check for drift
    has_drift, drift_value = monitor.check_drift(reference_mean=2.5, threshold=0.5)
    print(f"\nData drift detected: {has_drift} (drift: {drift_value:.4f})")
    
    return monitor


# Pattern 5: Feature Store
class FeatureStore:
    """Simple feature store for ML features."""
    
    def __init__(self):
        self.features = {}
        self.feature_metadata = {}
    
    def register_feature(self, feature_name, feature_fn, metadata=None):
        """Register a feature with its computation function."""
        self.features[feature_name] = feature_fn
        self.feature_metadata[feature_name] = metadata or {}
        
        print(f"Registered feature: {feature_name}")
    
    def get_features(self, entity_id, feature_names):
        """Get features for an entity."""
        result = {}
        
        for feature_name in feature_names:
            if feature_name not in self.features:
                raise ValueError(f"Feature {feature_name} not found")
            
            # Compute feature
            feature_fn = self.features[feature_name]
            result[feature_name] = feature_fn(entity_id)
        
        return result
    
    def get_feature_vector(self, entity_id, feature_names):
        """Get features as a vector."""
        features = self.get_features(entity_id, feature_names)
        return np.array([features[name] for name in feature_names])


def example_feature_store():
    """Example: Feature store."""
    store = FeatureStore()
    
    # Register features
    store.register_feature(
        'user_age',
        lambda entity_id: np.random.randint(18, 80),
        {'type': 'int', 'description': 'User age'}
    )
    
    store.register_feature(
        'user_income',
        lambda entity_id: np.random.uniform(20000, 200000),
        {'type': 'float', 'description': 'Annual income'}
    )
    
    store.register_feature(
        'user_score',
        lambda entity_id: np.random.random(),
        {'type': 'float', 'description': 'User score'}
    )
    
    print("\nFeature Store Example:")
    
    # Get features for entity
    entity_id = "user_123"
    features = store.get_features(entity_id, ['user_age', 'user_income', 'user_score'])
    print(f"Features for {entity_id}: {features}")
    
    # Get as vector
    feature_vector = store.get_feature_vector(
        entity_id,
        ['user_age', 'user_income', 'user_score']
    )
    print(f"Feature vector: {feature_vector}")
    
    return store


# Pattern 6: Experiment Tracking
class ExperimentTracker:
    """Track ML experiments and their results."""
    
    def __init__(self):
        self.experiments = {}
        self.current_experiment = None
    
    def start_experiment(self, experiment_name, config=None):
        """Start a new experiment."""
        self.current_experiment = experiment_name
        self.experiments[experiment_name] = {
            'config': config or {},
            'metrics': {},
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        print(f"Started experiment: {experiment_name}")
    
    def log_metric(self, metric_name, value, step=None):
        """Log a metric value."""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        experiment = self.experiments[self.current_experiment]
        
        if metric_name not in experiment['metrics']:
            experiment['metrics'][metric_name] = []
        
        experiment['metrics'][metric_name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_params(self, params):
        """Log experiment parameters."""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        self.experiments[self.current_experiment]['config'].update(params)
    
    def end_experiment(self, status='completed'):
        """End the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment")
        
        self.experiments[self.current_experiment]['status'] = status
        self.experiments[self.current_experiment]['ended_at'] = datetime.now().isoformat()
        
        print(f"Ended experiment: {self.current_experiment} ({status})")
        self.current_experiment = None
    
    def get_best_experiment(self, metric_name, mode='max'):
        """Get best experiment based on metric."""
        best_exp = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for exp_name, exp_data in self.experiments.items():
            if metric_name in exp_data['metrics']:
                values = [m['value'] for m in exp_data['metrics'][metric_name]]
                final_value = values[-1] if values else None
                
                if final_value is not None:
                    if mode == 'max' and final_value > best_value:
                        best_value = final_value
                        best_exp = exp_name
                    elif mode == 'min' and final_value < best_value:
                        best_value = final_value
                        best_exp = exp_name
        
        return best_exp, best_value


def example_experiment_tracking():
    """Example: Experiment tracking."""
    tracker = ExperimentTracker()
    
    print("\nExperiment Tracking Example:")
    
    # Experiment 1
    tracker.start_experiment('exp_1', config={'learning_rate': 0.001, 'batch_size': 32})
    tracker.log_metric('train_loss', 0.5, step=0)
    tracker.log_metric('train_loss', 0.3, step=100)
    tracker.log_metric('val_accuracy', 0.85, step=100)
    tracker.end_experiment('completed')
    
    # Experiment 2
    tracker.start_experiment('exp_2', config={'learning_rate': 0.01, 'batch_size': 64})
    tracker.log_metric('train_loss', 0.4, step=0)
    tracker.log_metric('train_loss', 0.2, step=100)
    tracker.log_metric('val_accuracy', 0.90, step=100)
    tracker.end_experiment('completed')
    
    # Find best experiment
    best_exp, best_acc = tracker.get_best_experiment('val_accuracy', mode='max')
    print(f"Best experiment: {best_exp} (accuracy: {best_acc:.4f})")
    
    return tracker


# Pattern 7: Model Serving Wrapper
class ModelServingWrapper:
    """Wrapper for model serving with preprocessing and postprocessing."""
    
    def __init__(self, model, preprocessor=None, postprocessor=None):
        self.model = model
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)
        self.request_count = 0
    
    def predict(self, inputs):
        """Make prediction with pre/post processing."""
        self.request_count += 1
        
        # Preprocess
        processed_inputs = self.preprocessor(inputs)
        
        # Predict
        predictions = self.model(processed_inputs, training=False)
        
        # Postprocess
        result = self.postprocessor(predictions)
        
        return result
    
    def batch_predict(self, inputs_list):
        """Batch prediction for efficiency."""
        # Stack inputs
        batch_inputs = tf.stack([self.preprocessor(x) for x in inputs_list])
        
        # Predict
        predictions = self.model(batch_inputs, training=False)
        
        # Postprocess each result
        results = [self.postprocessor(predictions[i:i+1]) for i in range(len(inputs_list))]
        
        return results
    
    def health_check(self):
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'request_count': self.request_count
        }


def example_model_serving():
    """Example: Model serving wrapper."""
    # Create model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(3, activation='softmax')
    ])
    
    # Preprocessing and postprocessing functions
    def preprocess(x):
        return tf.cast(x, tf.float32) / 255.0
    
    def postprocess(predictions):
        class_idx = tf.argmax(predictions, axis=-1)
        confidence = tf.reduce_max(predictions, axis=-1)
        return {
            'class': int(class_idx[0]),
            'confidence': float(confidence[0])
        }
    
    # Create serving wrapper
    serving = ModelServingWrapper(model, preprocess, postprocess)
    
    print("\nModel Serving Wrapper Example:")
    
    # Make prediction
    inputs = tf.random.uniform((1, 10), 0, 255)
    result = serving.predict(inputs)
    print(f"Prediction result: {result}")
    
    # Health check
    health = serving.health_check()
    print(f"Health check: {health}")
    
    return serving


if __name__ == "__main__":
    print("Production and MLOps Patterns\n" + "="*60)
    
    # Example 1: Input Validation
    print("\n1. Input Validation")
    validator = example_input_validation()
    
    # Example 2: Model Versioning
    print("\n2. Model Versioning")
    versioning = example_model_versioning()
    
    # Example 3: A/B Testing
    print("\n3. A/B Testing Framework")
    ab_framework = example_ab_testing()
    
    # Example 4: Model Monitoring
    print("\n4. Model Monitoring")
    monitor = example_model_monitoring()
    
    # Example 5: Feature Store
    print("\n5. Feature Store")
    feature_store = example_feature_store()
    
    # Example 6: Experiment Tracking
    print("\n6. Experiment Tracking")
    tracker = example_experiment_tracking()
    
    # Example 7: Model Serving
    print("\n7. Model Serving Wrapper")
    serving = example_model_serving()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Always validate inputs before inference")
    print("2. Use model versioning for safe deployments")
    print("3. A/B test new models before full rollout")
    print("4. Monitor latency, accuracy, and errors")
    print("5. Use feature store for consistent features")
    print("6. Track experiments for reproducibility")
    print("7. Implement proper error handling")
    print("8. Log all predictions for analysis")
    print("9. Set up alerts for anomalies")
    print("10. Plan for model rollback strategy")
