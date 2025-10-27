"""
Graph Neural Network (GNN) Patterns

This module demonstrates various GNN architectures and patterns for processing
graph-structured data in TensorFlow.

Patterns covered:
1. Graph Convolutional Network (GCN)
2. Graph Attention Network (GAT)
3. GraphSAGE
4. Message Passing Neural Network (MPNN)
5. Graph Pooling
6. Node Classification
7. Graph Classification
8. Link Prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Pattern 1: Graph Convolutional Network (GCN) Layer
class GCNLayer(layers.Layer):
    """Graph Convolutional Network layer."""
    
    def __init__(self, units, activation='relu', use_bias=True, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        # input_shape = [(batch, num_nodes, features), (batch, num_nodes, num_nodes)]
        feature_dim = input_shape[0][-1]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
    
    def call(self, inputs):
        features, adjacency = inputs
        
        # Normalize adjacency matrix (symmetric normalization)
        degree = tf.reduce_sum(adjacency, axis=-1)
        degree = tf.where(degree == 0, tf.ones_like(degree), degree)
        degree_inv_sqrt = tf.pow(degree, -0.5)
        degree_inv_sqrt = tf.expand_dims(degree_inv_sqrt, -1)
        
        # A_norm = D^(-1/2) * A * D^(-1/2)
        norm_adj = adjacency * degree_inv_sqrt * tf.transpose(degree_inv_sqrt, [0, 2, 1])
        
        # Aggregate: A_norm * X
        aggregated = tf.matmul(norm_adj, features)
        
        # Transform: aggregated * W
        output = tf.matmul(aggregated, self.kernel)
        
        if self.use_bias:
            output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output


def example_gcn_model():
    """Example: GCN for node classification."""
    num_nodes = 100
    num_features = 16
    num_classes = 7
    
    # Inputs
    node_features = keras.Input(shape=(num_nodes, num_features), name='features')
    adjacency_matrix = keras.Input(shape=(num_nodes, num_nodes), name='adjacency')
    
    # GCN layers
    x = GCNLayer(32, activation='relu')([node_features, adjacency_matrix])
    x = layers.Dropout(0.5)(x)
    x = GCNLayer(num_classes, activation='softmax')([x, adjacency_matrix])
    
    model = keras.Model(inputs=[node_features, adjacency_matrix], outputs=x)
    
    print("GCN Model:")
    model.summary()
    return model


# Pattern 2: Graph Attention Network (GAT) Layer
class GATLayer(layers.Layer):
    """Graph Attention Network layer with multi-head attention."""
    
    def __init__(self, units, num_heads=8, dropout_rate=0.1, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        
        # Weights for each attention head
        self.W = self.add_weight(
            shape=(self.num_heads, feature_dim, self.units),
            initializer='glorot_uniform',
            name='W'
        )
        
        # Attention weights
        self.a = self.add_weight(
            shape=(self.num_heads, 2 * self.units, 1),
            initializer='glorot_uniform',
            name='attention'
        )
    
    def call(self, inputs, training=False):
        features, adjacency = inputs
        batch_size = tf.shape(features)[0]
        num_nodes = tf.shape(features)[1]
        
        # Transform features for each head: (batch, num_heads, num_nodes, units)
        h = tf.einsum('bni,hiu->bhnu', features, self.W)
        
        # Compute attention scores
        # Concatenate h_i and h_j for all pairs
        h_i = tf.expand_dims(h, 3)  # (batch, heads, nodes, 1, units)
        h_j = tf.expand_dims(h, 2)  # (batch, heads, 1, nodes, units)
        h_i = tf.tile(h_i, [1, 1, 1, num_nodes, 1])
        h_j = tf.tile(h_j, [1, 1, num_nodes, 1, 1])
        
        # Concatenate and compute attention
        concat = tf.concat([h_i, h_j], axis=-1)  # (batch, heads, nodes, nodes, 2*units)
        e = tf.nn.leaky_relu(tf.einsum('bhnmu,huo->bhnm', concat, self.a))
        e = tf.squeeze(e, -1)  # (batch, heads, nodes, nodes)
        
        # Mask attention scores with adjacency matrix
        mask = tf.cast(tf.expand_dims(adjacency, 1), tf.bool)
        e = tf.where(mask, e, -1e9 * tf.ones_like(e))
        
        # Softmax attention weights
        alpha = tf.nn.softmax(e, axis=-1)
        
        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)
        
        # Aggregate features
        output = tf.einsum('bhnm,bhmu->bhnu', alpha, h)
        
        # Average over heads
        output = tf.reduce_mean(output, axis=1)
        
        return output


def example_gat_model():
    """Example: GAT for node classification."""
    num_nodes = 100
    num_features = 16
    num_classes = 7
    
    node_features = keras.Input(shape=(num_nodes, num_features))
    adjacency_matrix = keras.Input(shape=(num_nodes, num_nodes))
    
    x = GATLayer(32, num_heads=4)([node_features, adjacency_matrix])
    x = layers.Dropout(0.5)(x)
    x = GATLayer(num_classes, num_heads=1)([x, adjacency_matrix])
    x = layers.Softmax()(x)
    
    model = keras.Model(inputs=[node_features, adjacency_matrix], outputs=x)
    
    print("\nGAT Model:")
    model.summary()
    return model


# Pattern 3: GraphSAGE Layer
class GraphSAGELayer(layers.Layer):
    """GraphSAGE layer with neighborhood sampling."""
    
    def __init__(self, units, aggregator='mean', **kwargs):
        super(GraphSAGELayer, self).__init__(**kwargs)
        self.units = units
        self.aggregator = aggregator
    
    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        
        self.W_self = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='W_self'
        )
        
        self.W_neighbor = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='W_neighbor'
        )
    
    def call(self, inputs):
        features, adjacency = inputs
        
        # Aggregate neighbor features
        if self.aggregator == 'mean':
            # Average pooling
            degree = tf.reduce_sum(adjacency, axis=-1, keepdims=True)
            degree = tf.where(degree == 0, tf.ones_like(degree), degree)
            neighbor_features = tf.matmul(adjacency, features) / degree
        elif self.aggregator == 'max':
            # Max pooling
            expanded_features = tf.expand_dims(features, 1)
            expanded_adj = tf.expand_dims(adjacency, -1)
            masked_features = expanded_features * expanded_adj
            neighbor_features = tf.reduce_max(masked_features, axis=2)
        else:
            # Sum aggregation
            neighbor_features = tf.matmul(adjacency, features)
        
        # Transform self and neighbor features
        self_transformed = tf.matmul(features, self.W_self)
        neighbor_transformed = tf.matmul(neighbor_features, self.W_neighbor)
        
        # Concatenate and normalize
        output = self_transformed + neighbor_transformed
        output = tf.nn.l2_normalize(output, axis=-1)
        
        return tf.nn.relu(output)


def example_graphsage_model():
    """Example: GraphSAGE model."""
    num_nodes = 100
    num_features = 16
    
    node_features = keras.Input(shape=(num_nodes, num_features))
    adjacency_matrix = keras.Input(shape=(num_nodes, num_nodes))
    
    x = GraphSAGELayer(64, aggregator='mean')([node_features, adjacency_matrix])
    x = GraphSAGELayer(32, aggregator='mean')([x, adjacency_matrix])
    x = layers.Dense(7, activation='softmax')(x)
    
    model = keras.Model(inputs=[node_features, adjacency_matrix], outputs=x)
    
    print("\nGraphSAGE Model:")
    model.summary()
    return model


# Pattern 4: Message Passing Neural Network
class MPNNLayer(layers.Layer):
    """Message Passing Neural Network layer."""
    
    def __init__(self, units, **kwargs):
        super(MPNNLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        
        # Message function
        self.message_nn = keras.Sequential([
            layers.Dense(self.units, activation='relu'),
            layers.Dense(self.units)
        ])
        
        # Update function (GRU)
        self.gru = layers.GRUCell(self.units)
    
    def call(self, inputs):
        features, adjacency = inputs
        batch_size = tf.shape(features)[0]
        num_nodes = tf.shape(features)[1]
        
        # Compute messages
        # For each edge, compute message from source to target
        expanded_features = tf.expand_dims(features, 2)  # (batch, nodes, 1, features)
        expanded_features = tf.tile(expanded_features, [1, 1, num_nodes, 1])
        
        messages = self.message_nn(expanded_features)  # (batch, nodes, nodes, units)
        
        # Mask messages with adjacency matrix
        adjacency_expanded = tf.expand_dims(adjacency, -1)
        messages = messages * adjacency_expanded
        
        # Aggregate messages
        aggregated = tf.reduce_sum(messages, axis=2)  # (batch, nodes, units)
        
        # Update node states with GRU
        # Reshape for GRU
        features_flat = tf.reshape(features, [-1, tf.shape(features)[-1]])
        aggregated_flat = tf.reshape(aggregated, [-1, self.units])
        
        output_flat, _ = self.gru(aggregated_flat, [features_flat])
        output = tf.reshape(output_flat, [batch_size, num_nodes, self.units])
        
        return output


def example_mpnn_model():
    """Example: MPNN for graph classification."""
    num_nodes = 50
    num_features = 16
    
    node_features = keras.Input(shape=(num_nodes, num_features))
    adjacency_matrix = keras.Input(shape=(num_nodes, num_nodes))
    
    # Message passing layers
    x = MPNNLayer(64)([node_features, adjacency_matrix])
    x = MPNNLayer(64)([x, adjacency_matrix])
    
    # Global pooling for graph-level prediction
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=[node_features, adjacency_matrix], outputs=outputs)
    
    print("\nMPNN Model:")
    model.summary()
    return model


# Pattern 5: Graph Pooling
def example_graph_pooling():
    """Example: Different graph pooling strategies."""
    num_nodes = 100
    num_features = 32
    
    node_features = keras.Input(shape=(num_nodes, num_features))
    
    # Global average pooling
    avg_pool = layers.GlobalAveragePooling1D()(node_features)
    
    # Global max pooling
    max_pool = layers.GlobalMaxPooling1D()(node_features)
    
    # Attention-based pooling
    attention_scores = layers.Dense(1, activation='sigmoid')(node_features)
    attention_pool = tf.reduce_sum(node_features * attention_scores, axis=1)
    
    # Combine pooled representations
    combined = layers.Concatenate()([avg_pool, max_pool, attention_pool])
    outputs = layers.Dense(10, activation='softmax')(combined)
    
    model = keras.Model(inputs=node_features, outputs=outputs)
    
    print("\nGraph Pooling Model:")
    model.summary()
    return model


# Pattern 6: Complete GNN for Node Classification
def create_node_classification_model():
    """Complete GNN model for node classification."""
    num_nodes = 2708  # Cora dataset size
    num_features = 1433
    num_classes = 7
    
    node_features = keras.Input(shape=(num_nodes, num_features), name='features')
    adjacency = keras.Input(shape=(num_nodes, num_nodes), name='adjacency')
    
    # GCN layers
    x = GCNLayer(64, activation='relu')([node_features, adjacency])
    x = layers.Dropout(0.5)(x)
    x = GCNLayer(32, activation='relu')([x, adjacency])
    x = layers.Dropout(0.5)(x)
    x = GCNLayer(num_classes, activation='softmax')([x, adjacency])
    
    model = keras.Model(inputs=[node_features, adjacency], outputs=x)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nNode Classification Model:")
    model.summary()
    return model


# Pattern 7: Link Prediction Model
def create_link_prediction_model():
    """GNN model for link prediction."""
    num_nodes = 100
    num_features = 16
    
    node_features = keras.Input(shape=(num_nodes, num_features))
    adjacency = keras.Input(shape=(num_nodes, num_nodes))
    
    # Encode nodes with GNN
    x = GCNLayer(64, activation='relu')([node_features, adjacency])
    x = layers.Dropout(0.3)(x)
    node_embeddings = GCNLayer(32, activation='relu')([x, adjacency])
    
    # For link prediction, we need to score pairs of nodes
    # This is a simplified version - in practice, you'd sample node pairs
    
    # Compute pairwise scores using dot product
    # embeddings_i: (batch, nodes, 1, dim)
    # embeddings_j: (batch, 1, nodes, dim)
    emb_i = tf.expand_dims(node_embeddings, 2)
    emb_j = tf.expand_dims(node_embeddings, 1)
    
    # Dot product similarity
    scores = tf.reduce_sum(emb_i * emb_j, axis=-1)  # (batch, nodes, nodes)
    link_prob = tf.nn.sigmoid(scores)
    
    model = keras.Model(inputs=[node_features, adjacency], outputs=link_prob)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nLink Prediction Model:")
    model.summary()
    return model


# Pattern 8: Graph Classification Model
def create_graph_classification_model():
    """GNN for graph-level classification."""
    num_nodes = 50
    num_features = 16
    num_graphs = 32
    
    node_features = keras.Input(shape=(num_nodes, num_features))
    adjacency = keras.Input(shape=(num_nodes, num_nodes))
    
    # Node-level processing
    x = GCNLayer(64, activation='relu')([node_features, adjacency])
    x = layers.Dropout(0.3)(x)
    x = GCNLayer(32, activation='relu')([x, adjacency])
    
    # Graph-level pooling
    # Average pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    
    # Max pooling
    max_pool = layers.GlobalMaxPooling1D()(x)
    
    # Combine and classify
    combined = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=[node_features, adjacency], outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nGraph Classification Model:")
    model.summary()
    return model


if __name__ == "__main__":
    print("Graph Neural Network Patterns\n" + "="*50)
    
    # Example 1: GCN
    print("\n1. Graph Convolutional Network (GCN)")
    gcn_model = example_gcn_model()
    
    # Example 2: GAT
    print("\n2. Graph Attention Network (GAT)")
    gat_model = example_gat_model()
    
    # Example 3: GraphSAGE
    print("\n3. GraphSAGE")
    sage_model = example_graphsage_model()
    
    # Example 4: MPNN
    print("\n4. Message Passing Neural Network (MPNN)")
    mpnn_model = example_mpnn_model()
    
    # Example 5: Graph Pooling
    print("\n5. Graph Pooling Strategies")
    pooling_model = example_graph_pooling()
    
    # Example 6: Node Classification
    print("\n6. Node Classification")
    node_clf_model = create_node_classification_model()
    
    # Example 7: Link Prediction
    print("\n7. Link Prediction")
    link_pred_model = create_link_prediction_model()
    
    # Example 8: Graph Classification
    print("\n8. Graph Classification")
    graph_clf_model = create_graph_classification_model()
    
    print("\n" + "="*50)
    print("Best Practices:")
    print("1. Normalize adjacency matrices for stable training")
    print("2. Use dropout to prevent overfitting on graph data")
    print("3. Consider multi-head attention for better expressiveness")
    print("4. Use appropriate pooling for graph-level tasks")
    print("5. Sample neighborhoods for scalability (GraphSAGE)")
    print("6. Add self-loops to adjacency for better node features")
    print("7. Normalize node features before feeding to GNN")
    print("8. Use residual connections for deep GNNs")
