"""
Object Detection Patterns

This module demonstrates patterns for object detection tasks,
including various architectures and techniques.

Patterns covered:
1. Region Proposal Network (RPN)
2. Region-based CNN (R-CNN) Components
3. Fast R-CNN
4. Faster R-CNN Components
5. YOLO-style Single Shot Detection
6. SSD (Single Shot MultiBox Detector)
7. Feature Pyramid Network (FPN)
8. Anchor Boxes and Matching
9. Non-Maximum Suppression (NMS)
10. Focal Loss for Imbalanced Detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Region Proposal Network (RPN)
class RPN(layers.Layer):
    """Region Proposal Network for generating object proposals."""
    
    def __init__(self, num_anchors=9):
        super().__init__()
        
        self.num_anchors = num_anchors
        
        # Shared convolutional layer
        self.conv = layers.Conv2D(512, 3, padding='same', activation='relu')
        
        # Classification layer (object vs background)
        self.cls_layer = layers.Conv2D(num_anchors * 2, 1)
        
        # Regression layer (bounding box coordinates)
        self.reg_layer = layers.Conv2D(num_anchors * 4, 1)
    
    def call(self, feature_map):
        # Shared features
        x = self.conv(feature_map)
        
        # Classification scores
        cls_scores = self.cls_layer(x)
        batch_size = tf.shape(cls_scores)[0]
        height = tf.shape(cls_scores)[1]
        width = tf.shape(cls_scores)[2]
        
        cls_scores = tf.reshape(cls_scores, (batch_size, height, width, self.num_anchors, 2))
        
        # Bounding box regression
        bbox_deltas = self.reg_layer(x)
        bbox_deltas = tf.reshape(bbox_deltas, (batch_size, height, width, self.num_anchors, 4))
        
        return cls_scores, bbox_deltas


def example_rpn():
    """Example: Region Proposal Network."""
    rpn = RPN(num_anchors=9)
    
    # Feature map from backbone (e.g., ResNet)
    feature_map = tf.random.normal((2, 32, 32, 512))
    
    cls_scores, bbox_deltas = rpn(feature_map)
    
    print("Region Proposal Network Example:")
    print(f"Feature map shape: {feature_map.shape}")
    print(f"Classification scores shape: {cls_scores.shape}")
    print(f"Bounding box deltas shape: {bbox_deltas.shape}")
    print(f"Total proposals: {32 * 32 * 9} per image")
    
    return rpn


# Pattern 2: ROI Pooling
class ROIPooling(layers.Layer):
    """Region of Interest pooling layer."""
    
    def __init__(self, pool_size=(7, 7)):
        super().__init__()
        self.pool_size = pool_size
    
    def call(self, feature_maps, rois):
        """
        feature_maps: (batch, height, width, channels)
        rois: (num_rois, 5) where each row is [batch_idx, x1, y1, x2, y2]
        """
        pooled_features = []
        
        for roi in rois:
            batch_idx = tf.cast(roi[0], tf.int32)
            x1, y1, x2, y2 = roi[1:]
            
            # Extract ROI from feature map
            feature_map = feature_maps[batch_idx]
            
            # Convert normalized coordinates to feature map coordinates
            h = tf.shape(feature_map)[0]
            w = tf.shape(feature_map)[1]
            
            x1 = tf.cast(x1 * tf.cast(w, tf.float32), tf.int32)
            y1 = tf.cast(y1 * tf.cast(h, tf.float32), tf.int32)
            x2 = tf.cast(x2 * tf.cast(w, tf.float32), tf.int32)
            y2 = tf.cast(y2 * tf.cast(h, tf.float32), tf.int32)
            
            # Crop and resize
            roi_feature = feature_map[y1:y2, x1:x2]
            roi_feature = tf.expand_dims(roi_feature, 0)
            pooled = tf.image.resize(roi_feature, self.pool_size)
            
            pooled_features.append(pooled)
        
        return tf.concat(pooled_features, axis=0)


def example_roi_pooling():
    """Example: ROI pooling."""
    roi_pool = ROIPooling(pool_size=(7, 7))
    
    # Feature map
    feature_maps = tf.random.normal((2, 32, 32, 512))
    
    # Sample ROIs (batch_idx, x1, y1, x2, y2) in normalized coordinates
    rois = tf.constant([
        [0, 0.1, 0.1, 0.5, 0.5],
        [0, 0.6, 0.6, 0.9, 0.9],
        [1, 0.2, 0.3, 0.7, 0.8]
    ])
    
    pooled = roi_pool(feature_maps, rois)
    
    print("\nROI Pooling Example:")
    print(f"Feature maps shape: {feature_maps.shape}")
    print(f"Number of ROIs: {rois.shape[0]}")
    print(f"Pooled features shape: {pooled.shape}")
    print("Each ROI pooled to 7x7 regardless of size")
    
    return roi_pool


# Pattern 3: Fast R-CNN Head
class FastRCNNHead(layers.Layer):
    """Fast R-CNN detection head."""
    
    def __init__(self, num_classes, roi_size=(7, 7)):
        super().__init__()
        
        self.num_classes = num_classes
        self.roi_pooling = ROIPooling(roi_size)
        
        # Fully connected layers
        self.fc1 = layers.Dense(4096, activation='relu')
        self.fc2 = layers.Dense(4096, activation='relu')
        
        # Classification and regression branches
        self.cls_score = layers.Dense(num_classes)
        self.bbox_pred = layers.Dense(num_classes * 4)
    
    def call(self, feature_maps, rois):
        # ROI pooling
        pooled = self.roi_pooling(feature_maps, rois)
        
        # Flatten
        x = layers.Flatten()(pooled)
        
        # FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Classification scores
        cls_scores = self.cls_score(x)
        
        # Bounding box regression
        bbox_deltas = self.bbox_pred(x)
        bbox_deltas = tf.reshape(bbox_deltas, (-1, self.num_classes, 4))
        
        return cls_scores, bbox_deltas


def example_fast_rcnn():
    """Example: Fast R-CNN head."""
    head = FastRCNNHead(num_classes=20)
    
    feature_maps = tf.random.normal((2, 32, 32, 512))
    rois = tf.constant([
        [0, 0.1, 0.1, 0.5, 0.5],
        [0, 0.6, 0.6, 0.9, 0.9]
    ])
    
    cls_scores, bbox_deltas = head(feature_maps, rois)
    
    print("\nFast R-CNN Head Example:")
    print(f"Classification scores shape: {cls_scores.shape}")
    print(f"Bounding box deltas shape: {bbox_deltas.shape}")
    print("Predicts class and refines bounding box for each ROI")
    
    return head


# Pattern 4: YOLO-style Detection Head
class YOLOHead(layers.Layer):
    """YOLO-style single-shot detection head."""
    
    def __init__(self, num_classes, num_anchors=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Output: (x, y, w, h, confidence, class_probs)
        self.output_channels = num_anchors * (5 + num_classes)
        
        self.conv = layers.Conv2D(self.output_channels, 1)
    
    def call(self, feature_map):
        output = self.conv(feature_map)
        
        batch_size = tf.shape(output)[0]
        grid_h = tf.shape(output)[1]
        grid_w = tf.shape(output)[2]
        
        # Reshape to (batch, grid_h, grid_w, num_anchors, 5 + num_classes)
        output = tf.reshape(
            output,
            (batch_size, grid_h, grid_w, self.num_anchors, 5 + self.num_classes)
        )
        
        # Split into components
        xy = tf.sigmoid(output[..., 0:2])
        wh = output[..., 2:4]
        confidence = tf.sigmoid(output[..., 4:5])
        class_probs = tf.sigmoid(output[..., 5:])
        
        return xy, wh, confidence, class_probs


def example_yolo_head():
    """Example: YOLO detection head."""
    head = YOLOHead(num_classes=20, num_anchors=3)
    
    # Feature map from backbone
    feature_map = tf.random.normal((2, 13, 13, 1024))
    
    xy, wh, confidence, class_probs = head(feature_map)
    
    print("\nYOLO Head Example:")
    print(f"Feature map shape: {feature_map.shape}")
    print(f"Box centers (xy) shape: {xy.shape}")
    print(f"Box sizes (wh) shape: {wh.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Class probabilities shape: {class_probs.shape}")
    print(f"Total predictions: {13 * 13 * 3} per image")
    
    return head


# Pattern 5: Feature Pyramid Network (FPN)
class FPN(layers.Layer):
    """Feature Pyramid Network for multi-scale features."""
    
    def __init__(self, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_c5 = layers.Conv2D(out_channels, 1)
        self.lateral_c4 = layers.Conv2D(out_channels, 1)
        self.lateral_c3 = layers.Conv2D(out_channels, 1)
        
        # Output convolutions
        self.output_p5 = layers.Conv2D(out_channels, 3, padding='same')
        self.output_p4 = layers.Conv2D(out_channels, 3, padding='same')
        self.output_p3 = layers.Conv2D(out_channels, 3, padding='same')
    
    def call(self, c3, c4, c5):
        """
        c3, c4, c5: Feature maps from backbone at different scales
        """
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        
        p4 = self.lateral_c4(c4)
        p4_upsampled = layers.UpSampling2D(size=(2, 2))(p5)
        p4 = p4 + p4_upsampled
        
        p3 = self.lateral_c3(c3)
        p3_upsampled = layers.UpSampling2D(size=(2, 2))(p4)
        p3 = p3 + p3_upsampled
        
        # Output convolutions
        p5 = self.output_p5(p5)
        p4 = self.output_p4(p4)
        p3 = self.output_p3(p3)
        
        return p3, p4, p5


def example_fpn():
    """Example: Feature Pyramid Network."""
    fpn = FPN(out_channels=256)
    
    # Multi-scale features from backbone
    c3 = tf.random.normal((2, 64, 64, 512))
    c4 = tf.random.normal((2, 32, 32, 1024))
    c5 = tf.random.normal((2, 16, 16, 2048))
    
    p3, p4, p5 = fpn(c3, c4, c5)
    
    print("\nFeature Pyramid Network Example:")
    print(f"Input C3 shape: {c3.shape}")
    print(f"Input C4 shape: {c4.shape}")
    print(f"Input C5 shape: {c5.shape}")
    print(f"Output P3 shape: {p3.shape}")
    print(f"Output P4 shape: {p4.shape}")
    print(f"Output P5 shape: {p5.shape}")
    print("Multi-scale features for detecting objects at different sizes")
    
    return fpn


# Pattern 6: Anchor Boxes
def generate_anchor_boxes(feature_map_size, scales, aspect_ratios):
    """Generate anchor boxes for a feature map."""
    grid_h, grid_w = feature_map_size
    
    anchors = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Center of grid cell
            cx = (j + 0.5) / grid_w
            cy = (i + 0.5) / grid_h
            
            for scale in scales:
                for ratio in aspect_ratios:
                    # Width and height
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    
                    anchors.append([cx, cy, w, h])
    
    return np.array(anchors)


def example_anchor_boxes():
    """Example: Generate anchor boxes."""
    feature_map_size = (13, 13)
    scales = [0.1, 0.2, 0.3]
    aspect_ratios = [0.5, 1.0, 2.0]
    
    anchors = generate_anchor_boxes(feature_map_size, scales, aspect_ratios)
    
    print("\nAnchor Boxes Example:")
    print(f"Feature map size: {feature_map_size}")
    print(f"Scales: {scales}")
    print(f"Aspect ratios: {aspect_ratios}")
    print(f"Total anchors: {anchors.shape[0]}")
    print(f"Anchors per location: {len(scales) * len(aspect_ratios)}")
    
    return anchors


# Pattern 7: IoU Calculation
def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    # boxes: (N, 4) in format [x1, y1, x2, y2]
    
    # Expand dimensions for broadcasting
    boxes1 = tf.expand_dims(boxes1, 1)  # (N, 1, 4)
    boxes2 = tf.expand_dims(boxes2, 0)  # (1, M, 4)
    
    # Intersection coordinates
    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    
    # Intersection area
    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    
    # Union area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / (union + 1e-7)
    
    return iou


def example_iou():
    """Example: IoU calculation."""
    # Sample boxes [x1, y1, x2, y2]
    boxes1 = tf.constant([
        [0.1, 0.1, 0.5, 0.5],
        [0.6, 0.6, 0.9, 0.9]
    ])
    
    boxes2 = tf.constant([
        [0.2, 0.2, 0.6, 0.6],
        [0.5, 0.5, 0.8, 0.8]
    ])
    
    iou = compute_iou(boxes1, boxes2)
    
    print("\nIoU Calculation Example:")
    print(f"Boxes 1:\n{boxes1.numpy()}")
    print(f"Boxes 2:\n{boxes2.numpy()}")
    print(f"IoU matrix:\n{iou.numpy()}")
    
    return iou


# Pattern 8: Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.5, max_output_size=100):
    """Non-Maximum Suppression."""
    selected_indices = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold
    )
    
    return selected_indices


def example_nms():
    """Example: Non-Maximum Suppression."""
    # Sample overlapping boxes
    boxes = tf.constant([
        [0.1, 0.1, 0.5, 0.5],
        [0.15, 0.15, 0.55, 0.55],  # Overlaps with first
        [0.6, 0.6, 0.9, 0.9],
        [0.65, 0.65, 0.95, 0.95]   # Overlaps with third
    ])
    
    scores = tf.constant([0.9, 0.7, 0.85, 0.6])
    
    selected = nms(boxes, scores, iou_threshold=0.5)
    
    print("\nNon-Maximum Suppression Example:")
    print(f"Input boxes: {boxes.shape[0]}")
    print(f"Scores: {scores.numpy()}")
    print(f"Selected indices: {selected.numpy()}")
    print(f"Kept boxes: {len(selected)}")
    print("Removes overlapping boxes, keeps highest scoring")
    
    return selected


# Pattern 9: Focal Loss
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance in object detection."""
    # y_true: (batch, num_boxes)
    # y_pred: (batch, num_boxes, num_classes)
    
    # Compute cross entropy
    ce = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    
    # Get prediction probabilities
    p = tf.nn.softmax(y_pred)
    p_t = tf.reduce_sum(y_true * p, axis=-1)
    
    # Focal weight
    focal_weight = alpha * tf.pow(1 - p_t, gamma)
    
    # Focal loss
    loss = focal_weight * ce
    
    return tf.reduce_mean(loss)


def example_focal_loss():
    """Example: Focal loss."""
    # Simulated predictions
    y_true = tf.constant([
        [1, 0, 0],  # Class 0
        [0, 1, 0],  # Class 1
        [0, 0, 1]   # Class 2
    ], dtype=tf.float32)
    
    y_pred = tf.constant([
        [2.0, 0.1, 0.1],  # Confident correct
        [0.1, 1.0, 0.1],  # Less confident correct
        [0.3, 0.3, 0.3]   # Uncertain
    ])
    
    loss = focal_loss(y_true, y_pred)
    
    print("\nFocal Loss Example:")
    print(f"Loss value: {loss.numpy():.4f}")
    print("Focuses on hard examples (uncertain predictions)")
    print("Down-weights easy examples (confident predictions)")
    
    return loss


if __name__ == "__main__":
    print("Object Detection Patterns\n" + "="*60)
    
    # Example 1: RPN
    print("\n1. Region Proposal Network")
    rpn = example_rpn()
    
    # Example 2: ROI Pooling
    print("\n2. ROI Pooling")
    roi_pool = example_roi_pooling()
    
    # Example 3: Fast R-CNN
    print("\n3. Fast R-CNN Head")
    fast_rcnn = example_fast_rcnn()
    
    # Example 4: YOLO Head
    print("\n4. YOLO Detection Head")
    yolo_head = example_yolo_head()
    
    # Example 5: FPN
    print("\n5. Feature Pyramid Network")
    fpn = example_fpn()
    
    # Example 6: Anchor Boxes
    print("\n6. Anchor Boxes")
    anchors = example_anchor_boxes()
    
    # Example 7: IoU
    print("\n7. IoU Calculation")
    iou = example_iou()
    
    # Example 8: NMS
    print("\n8. Non-Maximum Suppression")
    selected = example_nms()
    
    # Example 9: Focal Loss
    print("\n9. Focal Loss")
    loss = example_focal_loss()
    
    print("\n" + "="*60)
    print("Object Detection Best Practices:")
    print("1. Use FPN for multi-scale detection")
    print("2. Apply NMS to remove duplicate detections")
    print("3. Use focal loss for class imbalance")
    print("4. Anchor boxes at multiple scales and ratios")
    print("5. Data augmentation crucial for small datasets")
    print("6. Train with high-resolution images when possible")
    print("7. Use pretrained backbone (ResNet, EfficientNet)")
    print("8. Monitor mAP (mean Average Precision) metric")
    print("9. Balance speed vs accuracy based on use case")
    print("10. Consider IoU threshold for matching predictions")
