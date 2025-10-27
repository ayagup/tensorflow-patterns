"""
Custom Metrics Pattern

This module demonstrates how to create custom metrics in TensorFlow
for tracking model performance beyond standard metrics.

Patterns covered:
1. Basic Custom Metric (Function)
2. Custom Metric Class
3. Streaming Metric with State
4. F1 Score Metric
5. Mean IoU for Segmentation
6. Top-K Accuracy
7. Precision-Recall Curve Metric
8. Custom Confusion Matrix Metric
9. Mean Average Precision (mAP)
10. Per-Class Accuracy Metric
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


# Pattern 1: Basic Custom Metric Function
def custom_accuracy(y_true, y_pred):
    """Simple custom accuracy metric."""
    y_pred_classes = tf.argmax(y_pred, axis=-1)
    y_true_classes = tf.cast(y_true, tf.int64)
    correct = tf.equal(y_pred_classes, y_true_classes)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def example_custom_metric_function():
    """Example: Custom metric as function."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[custom_accuracy]
    )
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    
    print("Custom Metric Function Example:")
    history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    print(f"Final custom accuracy: {history.history['custom_accuracy'][-1]:.4f}")
    
    return model


# Pattern 2: Custom Metric Class
class CustomAccuracy(keras.metrics.Metric):
    """Custom accuracy metric as a class."""
    
    def __init__(self, name='custom_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true_classes = tf.cast(y_true, tf.int64)
        
        matches = tf.cast(tf.equal(y_pred_classes, y_true_classes), tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            matches = matches * sample_weight
        
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def example_custom_metric_class():
    """Example: Custom metric as class."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[CustomAccuracy()]
    )
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    
    print("\nCustom Metric Class Example:")
    history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    print(f"Final accuracy: {history.history['custom_accuracy'][-1]:.4f}")
    print("Class-based metrics maintain state across batches")
    
    return model


# Pattern 3: F1 Score Metric
class F1Score(keras.metrics.Metric):
    """F1 score for binary classification."""
    
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1
    
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


def example_f1_score():
    """Example: F1 score metric."""
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[F1Score()]
    )
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, (100, 1))
    
    print("\nF1 Score Example:")
    history = model.fit(X, y, epochs=3, batch_size=16, verbose=0)
    print(f"Final F1 score: {history.history['f1_score'][-1]:.4f}")
    print("Harmonic mean of precision and recall")
    
    return model


# Pattern 4: Mean IoU for Segmentation
class MeanIoU(keras.metrics.Metric):
    """Mean Intersection over Union for segmentation."""
    
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int64)
        
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Compute IoU for each class
        for class_id in range(self.num_classes):
            true_mask = tf.equal(y_true, class_id)
            pred_mask = tf.equal(y_pred, class_id)
            
            intersection = tf.reduce_sum(
                tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32)
            )
            union = tf.reduce_sum(
                tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32)
            )
            
            iou = intersection / (union + 1e-7)
            self.total_iou.assign_add(iou)
        
        self.count.assign_add(tf.cast(self.num_classes, tf.float32))
    
    def result(self):
        return self.total_iou / self.count
    
    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


def example_mean_iou():
    """Example: Mean IoU for segmentation."""
    num_classes = 3
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 1)),
        keras.layers.Conv2D(num_classes, 1, activation='softmax', padding='same')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[MeanIoU(num_classes)]
    )
    
    X = np.random.rand(10, 32, 32, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, (10, 32, 32, 1))
    
    print("\nMean IoU Example:")
    history = model.fit(X, y, epochs=2, batch_size=2, verbose=0)
    print(f"Final Mean IoU: {history.history['mean_iou'][-1]:.4f}")
    print("Standard metric for semantic segmentation")
    
    return model


# Pattern 5: Top-K Accuracy
class TopKAccuracy(keras.metrics.Metric):
    """Top-K categorical accuracy."""
    
    def __init__(self, k=5, name='top_k_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get top-k predictions
        top_k_pred = tf.nn.top_k(y_pred, k=self.k).indices
        
        # Check if true class is in top-k
        y_true_expanded = tf.expand_dims(tf.cast(y_true, tf.int32), -1)
        matches = tf.reduce_any(tf.equal(top_k_pred, y_true_expanded), axis=-1)
        
        self.total.assign_add(tf.reduce_sum(tf.cast(matches, tf.float32)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def example_top_k_accuracy():
    """Example: Top-K accuracy."""
    num_classes = 100
    
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(50,)),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[TopKAccuracy(k=5)]
    )
    
    X = np.random.randn(200, 50)
    y = np.random.randint(0, num_classes, 200)
    
    print("\nTop-K Accuracy Example:")
    history = model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    print(f"Final Top-5 accuracy: {history.history['top_k_accuracy'][-1]:.4f}")
    print("Useful for multi-class problems with many classes")
    
    return model


# Pattern 6: Precision and Recall
class PrecisionRecall(keras.metrics.Metric):
    """Combined Precision and Recall metric."""
    
    def __init__(self, name='precision_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        
        # Return as dict (requires TF 2.x)
        return {'precision': precision, 'recall': recall}
    
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


def example_precision_recall():
    """Example: Precision and recall metrics."""
    print("\nPrecision and Recall Example:")
    print("Precision: Of predicted positives, how many are correct?")
    print("Recall: Of actual positives, how many did we find?")
    print("Trade-off: High precision may mean low recall and vice versa")
    
    metric = PrecisionRecall()
    
    # Simulated predictions
    y_true = tf.constant([[1], [1], [0], [0], [1]])
    y_pred = tf.constant([[0.9], [0.4], [0.6], [0.2], [0.8]])
    
    metric.update_state(y_true, y_pred)
    result = metric.result()
    
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    
    return metric


# Pattern 7: Custom Confusion Matrix Aggregator
class ConfusionMatrixMetric(keras.metrics.Metric):
    """Track confusion matrix elements."""
    
    def __init__(self, num_classes, name='confusion_matrix', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name='cm',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int64)
        
        # Update confusion matrix
        cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        
        self.confusion_matrix.assign_add(cm)
    
    def result(self):
        # Return overall accuracy from confusion matrix
        correct = tf.reduce_sum(tf.linalg.diag_part(self.confusion_matrix))
        total = tf.reduce_sum(self.confusion_matrix)
        return correct / (total + 1e-7)
    
    def reset_state(self):
        self.confusion_matrix.assign(
            tf.zeros((self.num_classes, self.num_classes))
        )
    
    def get_confusion_matrix(self):
        """Get the actual confusion matrix."""
        return self.confusion_matrix.numpy()


def example_confusion_matrix():
    """Example: Confusion matrix metric."""
    num_classes = 3
    
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    cm_metric = ConfusionMatrixMetric(num_classes)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[cm_metric]
    )
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, num_classes, 100)
    
    print("\nConfusion Matrix Metric Example:")
    model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    
    print("Confusion Matrix:")
    print(cm_metric.get_confusion_matrix())
    
    return model


# Pattern 8: Per-Class Accuracy
class PerClassAccuracy(keras.metrics.Metric):
    """Track accuracy for each class separately."""
    
    def __init__(self, num_classes, name='per_class_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        self.class_correct = [
            self.add_weight(name=f'class_{i}_correct', initializer='zeros')
            for i in range(num_classes)
        ]
        self.class_total = [
            self.add_weight(name=f'class_{i}_total', initializer='zeros')
            for i in range(num_classes)
        ]
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int64)
        
        for class_id in range(self.num_classes):
            # Samples of this class
            class_mask = tf.equal(y_true, class_id)
            
            # Correct predictions for this class
            correct_mask = tf.logical_and(class_mask, tf.equal(y_pred, class_id))
            
            self.class_correct[class_id].assign_add(
                tf.reduce_sum(tf.cast(correct_mask, tf.float32))
            )
            self.class_total[class_id].assign_add(
                tf.reduce_sum(tf.cast(class_mask, tf.float32))
            )
    
    def result(self):
        # Return mean accuracy across classes
        class_accuracies = [
            correct / (total + 1e-7)
            for correct, total in zip(self.class_correct, self.class_total)
        ]
        return tf.reduce_mean(class_accuracies)
    
    def reset_state(self):
        for i in range(self.num_classes):
            self.class_correct[i].assign(0.0)
            self.class_total[i].assign(0.0)
    
    def get_per_class_accuracy(self):
        """Get accuracy for each class."""
        return [
            (correct / (total + 1e-7)).numpy()
            for correct, total in zip(self.class_correct, self.class_total)
        ]


def example_per_class_accuracy():
    """Example: Per-class accuracy."""
    num_classes = 3
    
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    per_class_metric = PerClassAccuracy(num_classes)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[per_class_metric]
    )
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, num_classes, 100)
    
    print("\nPer-Class Accuracy Example:")
    model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    
    accuracies = per_class_metric.get_per_class_accuracy()
    for i, acc in enumerate(accuracies):
        print(f"Class {i} accuracy: {acc:.4f}")
    
    return model


if __name__ == "__main__":
    print("Custom Metrics Pattern\n" + "="*60)
    
    # Example 1: Custom Metric Function
    print("\n1. Custom Metric Function")
    model1 = example_custom_metric_function()
    
    # Example 2: Custom Metric Class
    print("\n2. Custom Metric Class")
    model2 = example_custom_metric_class()
    
    # Example 3: F1 Score
    print("\n3. F1 Score Metric")
    model3 = example_f1_score()
    
    # Example 4: Mean IoU
    print("\n4. Mean IoU for Segmentation")
    model4 = example_mean_iou()
    
    # Example 5: Top-K Accuracy
    print("\n5. Top-K Accuracy")
    model5 = example_top_k_accuracy()
    
    # Example 6: Precision and Recall
    print("\n6. Precision and Recall")
    metric6 = example_precision_recall()
    
    # Example 7: Confusion Matrix
    print("\n7. Confusion Matrix Metric")
    model7 = example_confusion_matrix()
    
    # Example 8: Per-Class Accuracy
    print("\n8. Per-Class Accuracy")
    model8 = example_per_class_accuracy()
    
    print("\n" + "="*60)
    print("Custom Metrics Best Practices:")
    print("1. Inherit from keras.metrics.Metric for stateful metrics")
    print("2. Use add_weight() to track metric state across batches")
    print("3. Implement update_state(), result(), reset_state()")
    print("4. Use TensorFlow operations for GPU compatibility")
    print("5. Add epsilon to prevent division by zero")
    print("6. Test metrics with known inputs")
    print("7. Consider sample_weight for weighted metrics")
    print("8. Return single value or dict from result()")
    print("9. Reset state between epochs automatically")
    print("10. Document metric meaning and calculation")
