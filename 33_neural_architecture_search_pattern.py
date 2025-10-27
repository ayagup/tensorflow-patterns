"""
Neural Architecture Search (NAS) Patterns

This module demonstrates Neural Architecture Search techniques including
automated model design, hyperparameter optimization, and efficient search strategies.

Patterns covered:
1. Random Search
2. Grid Search
3. Bayesian Optimization
4. Evolutionary Algorithm
5. Differentiable Architecture Search (DARTS)
6. Efficient Neural Architecture Search (ENAS)
7. Network Morphism
8. Weight Sharing
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from itertools import product
import random


# Pattern 1: Random Search for Hyperparameters
class RandomSearch:
    """Random search for hyperparameter tuning."""
    
    def __init__(self, param_space, n_trials=20):
        self.param_space = param_space
        self.n_trials = n_trials
        self.results = []
    
    def sample_params(self):
        """Sample random hyperparameters."""
        params = {}
        for key, values in self.param_space.items():
            if isinstance(values, tuple) and len(values) == 2:
                # Continuous range
                params[key] = np.random.uniform(values[0], values[1])
            else:
                # Discrete choices
                params[key] = np.random.choice(values)
        return params
    
    def build_model(self, params):
        """Build model with given parameters."""
        model = keras.Sequential([
            layers.Dense(params['units_1'], activation='relu', input_shape=(784,)),
            layers.Dropout(params['dropout']),
            layers.Dense(params['units_2'], activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def search(self, x_train, y_train, x_val, y_val):
        """Perform random search."""
        for trial in range(self.n_trials):
            params = self.sample_params()
            print(f"\nTrial {trial + 1}/{self.n_trials}")
            print(f"Parameters: {params}")
            
            model = self.build_model(params)
            
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=5,
                batch_size=128,
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            self.results.append({
                'params': params,
                'val_accuracy': val_acc
            })
            print(f"Validation accuracy: {val_acc:.4f}")
        
        # Find best parameters
        best_result = max(self.results, key=lambda x: x['val_accuracy'])
        return best_result


def example_random_search():
    """Example: Random search for hyperparameters."""
    param_space = {
        'units_1': [64, 128, 256, 512],
        'units_2': [32, 64, 128],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': (1e-4, 1e-2)
    }
    
    searcher = RandomSearch(param_space, n_trials=10)
    
    # Generate dummy data
    x_train = np.random.randn(1000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000)
    x_val = np.random.randn(200, 784).astype(np.float32)
    y_val = np.random.randint(0, 10, 200)
    
    print("Random Search Example:")
    # best_result = searcher.search(x_train, y_train, x_val, y_val)
    # print(f"\nBest parameters: {best_result['params']}")
    # print(f"Best validation accuracy: {best_result['val_accuracy']:.4f}")


# Pattern 2: Grid Search
class GridSearch:
    """Grid search for hyperparameter tuning."""
    
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.results = []
    
    def get_param_combinations(self):
        """Generate all parameter combinations."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in product(*values)]
    
    def build_model(self, params):
        """Build model with given parameters."""
        model = keras.Sequential([
            layers.Conv2D(params['filters_1'], 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Conv2D(params['filters_2'], 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def search(self, x_train, y_train, x_val, y_val):
        """Perform grid search."""
        param_combinations = self.get_param_combinations()
        
        for i, params in enumerate(param_combinations):
            print(f"\nConfiguration {i + 1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            model = self.build_model(params)
            
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=3,
                batch_size=128,
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            self.results.append({
                'params': params,
                'val_accuracy': val_acc
            })
            print(f"Validation accuracy: {val_acc:.4f}")
        
        best_result = max(self.results, key=lambda x: x['val_accuracy'])
        return best_result


def example_grid_search():
    """Example: Grid search for CNN hyperparameters."""
    param_grid = {
        'filters_1': [16, 32],
        'filters_2': [32, 64],
        'learning_rate': [0.001, 0.0001]
    }
    
    searcher = GridSearch(param_grid)
    print("Grid Search Example:")
    print(f"Total configurations: {len(searcher.get_param_combinations())}")


# Pattern 3: Evolutionary Algorithm for Architecture Search
class ArchitectureGenome:
    """Represents a neural network architecture."""
    
    def __init__(self, genes=None):
        if genes is None:
            # Random initialization
            self.genes = {
                'num_layers': random.randint(2, 5),
                'layer_types': [random.choice(['dense', 'conv']) for _ in range(5)],
                'units': [random.choice([32, 64, 128, 256]) for _ in range(5)],
                'activations': [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(5)],
                'dropout_rate': random.uniform(0.1, 0.5)
            }
        else:
            self.genes = genes
        self.fitness = 0
    
    def build_model(self, input_shape=(28, 28, 1)):
        """Build model from genome."""
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        for i in range(self.genes['num_layers']):
            if self.genes['layer_types'][i] == 'conv':
                x = layers.Conv2D(
                    self.genes['units'][i], 3,
                    activation=self.genes['activations'][i],
                    padding='same'
                )(x)
                x = layers.MaxPooling2D()(x)
            else:
                if len(x.shape) > 2:
                    x = layers.Flatten()(x)
                x = layers.Dense(
                    self.genes['units'][i],
                    activation=self.genes['activations'][i]
                )(x)
            
            x = layers.Dropout(self.genes['dropout_rate'])(x)
        
        if len(x.shape) > 2:
            x = layers.GlobalAveragePooling2D()(x)
        
        outputs = layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def mutate(self, mutation_rate=0.1):
        """Mutate the genome."""
        new_genes = self.genes.copy()
        
        if random.random() < mutation_rate:
            new_genes['num_layers'] = max(2, min(5, self.genes['num_layers'] + random.choice([-1, 1])))
        
        if random.random() < mutation_rate:
            idx = random.randint(0, 4)
            new_genes['layer_types'][idx] = random.choice(['dense', 'conv'])
        
        if random.random() < mutation_rate:
            idx = random.randint(0, 4)
            new_genes['units'][idx] = random.choice([32, 64, 128, 256])
        
        return ArchitectureGenome(new_genes)
    
    def crossover(self, other):
        """Crossover with another genome."""
        child_genes = {}
        for key in self.genes.keys():
            if isinstance(self.genes[key], list):
                # List genes: randomly select from parents
                child_genes[key] = [
                    random.choice([self.genes[key][i], other.genes[key][i]])
                    for i in range(len(self.genes[key]))
                ]
            else:
                # Scalar genes: randomly select from parents
                child_genes[key] = random.choice([self.genes[key], other.genes[key]])
        
        return ArchitectureGenome(child_genes)


class EvolutionarySearch:
    """Evolutionary algorithm for architecture search."""
    
    def __init__(self, population_size=20, generations=10):
        self.population_size = population_size
        self.generations = generations
        self.population = [ArchitectureGenome() for _ in range(population_size)]
    
    def evaluate_fitness(self, genome, x_train, y_train, x_val, y_val):
        """Evaluate genome fitness."""
        try:
            model = genome.build_model()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=3,
                batch_size=128,
                verbose=0
            )
            
            genome.fitness = max(history.history['val_accuracy'])
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            genome.fitness = 0
        
        return genome.fitness
    
    def select_parents(self):
        """Tournament selection."""
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def evolve(self, x_train, y_train, x_val, y_val):
        """Run evolutionary algorithm."""
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate population
            for i, genome in enumerate(self.population):
                fitness = self.evaluate_fitness(genome, x_train, y_train, x_val, y_val)
                print(f"Individual {i + 1}: Fitness = {fitness:.4f}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = 2
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            new_population.extend(sorted_pop[:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                
                child = parent1.crossover(parent2)
                child = child.mutate(mutation_rate=0.2)
                
                new_population.append(child)
            
            self.population = new_population
            
            best_fitness = max(g.fitness for g in self.population)
            print(f"Best fitness: {best_fitness:.4f}")
        
        # Return best genome
        return max(self.population, key=lambda x: x.fitness)


def example_evolutionary_search():
    """Example: Evolutionary architecture search."""
    print("Evolutionary Architecture Search Example:")
    print("This would search through different architectures")
    print("Population-based optimization with mutation and crossover")
    
    # Small example
    searcher = EvolutionarySearch(population_size=5, generations=2)
    print(f"Population size: {searcher.population_size}")
    print(f"Generations: {searcher.generations}")


# Pattern 4: Weight Sharing (Simplified ENAS concept)
class SuperNet(keras.Model):
    """Super network for weight sharing."""
    
    def __init__(self):
        super().__init__()
        # Define a super network with multiple paths
        self.conv1_options = [
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu')
        ]
        
        self.conv2_options = [
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu')
        ]
        
        self.pool = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, architecture=None):
        """Forward pass with specified architecture."""
        if architecture is None:
            # Default architecture
            architecture = [0, 0]
        
        x = self.conv1_options[architecture[0]](inputs)
        x = self.pool(x)
        x = self.conv2_options[architecture[1]](x)
        x = self.pool(x)
        x = self.flatten(x)
        outputs = self.dense(x)
        
        return outputs


def example_weight_sharing():
    """Example: Weight sharing for efficient NAS."""
    supernet = SuperNet()
    
    # Different architectures share weights
    inputs = keras.Input(shape=(28, 28, 1))
    
    # Architecture 1: [0, 0]
    arch1 = supernet(inputs, architecture=[0, 0])
    
    # Architecture 2: [1, 1]
    arch2 = supernet(inputs, architecture=[1, 1])
    
    print("\nWeight Sharing Example:")
    print("SuperNet allows sampling different architectures")
    print("All sampled architectures share the same weights")
    print("This enables efficient architecture search")


# Pattern 5: Network Morphism
def widen_layer(model, layer_idx, new_width):
    """Widen a layer while preserving function."""
    # Get old layer
    old_layer = model.layers[layer_idx]
    old_weights = old_layer.get_weights()
    
    if len(old_weights) == 0:
        return model
    
    old_kernel, old_bias = old_weights[0], old_weights[1]
    old_width = old_kernel.shape[-1]
    
    # Create new wider kernel
    new_kernel = np.zeros(old_kernel.shape[:-1] + (new_width,))
    new_kernel[..., :old_width] = old_kernel
    
    # Random initialization for new neurons
    if new_width > old_width:
        new_kernel[..., old_width:] = np.random.randn(
            *old_kernel.shape[:-1], new_width - old_width
        ) * 0.01
    
    # Create new bias
    new_bias = np.zeros(new_width)
    new_bias[:old_width] = old_bias
    
    print(f"Widened layer {layer_idx} from {old_width} to {new_width} units")
    
    return new_kernel, new_bias


def example_network_morphism():
    """Example: Network morphism for architecture modification."""
    print("\nNetwork Morphism Example:")
    print("Morphism allows modifying architecture while preserving function")
    
    # Original model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dense(10, activation='softmax')
    ])
    
    print("\nOriginal model:")
    model.summary()
    
    # Widen first layer
    new_kernel, new_bias = widen_layer(model, 0, new_width=128)
    
    print("\nAfter morphism: first layer widened to 128 units")


# Pattern 6: One-Shot Architecture Search
class OneShot(keras.Model):
    """One-shot model for architecture search."""
    
    def __init__(self):
        super().__init__()
        # Mixed operations
        self.ops = [
            layers.Conv2D(32, 3, padding='same'),
            layers.Conv2D(32, 5, padding='same'),
            layers.DepthwiseConv2D(3, padding='same'),
            layers.Identity()
        ]
        
        # Architecture parameters
        self.alpha = tf.Variable(
            tf.ones((len(self.ops),)) / len(self.ops),
            trainable=True
        )
    
    def call(self, inputs, training=False):
        """Forward pass with mixed operations."""
        if training:
            # Training: mix all operations with softmax weights
            alpha_softmax = tf.nn.softmax(self.alpha)
            outputs = sum([
                alpha_softmax[i] * op(inputs)
                for i, op in enumerate(self.ops)
            ])
        else:
            # Inference: use best operation
            best_op_idx = tf.argmax(self.alpha)
            outputs = self.ops[best_op_idx](inputs)
        
        return outputs


def example_oneshot_search():
    """Example: One-shot architecture search."""
    print("\nOne-Shot Architecture Search:")
    
    oneshot = OneShot()
    
    # Build
    inputs = keras.Input(shape=(28, 28, 3))
    x = oneshot(inputs)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print("One-shot model combines multiple operations")
    print("Architecture weights learned jointly with model weights")
    model.summary()


if __name__ == "__main__":
    print("Neural Architecture Search Patterns\n" + "="*60)
    
    # Example 1: Random Search
    print("\n1. Random Search")
    example_random_search()
    
    # Example 2: Grid Search
    print("\n2. Grid Search")
    example_grid_search()
    
    # Example 3: Evolutionary Search
    print("\n3. Evolutionary Algorithm")
    example_evolutionary_search()
    
    # Example 4: Weight Sharing
    print("\n4. Weight Sharing (ENAS concept)")
    example_weight_sharing()
    
    # Example 5: Network Morphism
    print("\n5. Network Morphism")
    example_network_morphism()
    
    # Example 6: One-Shot Search
    print("\n6. One-Shot Architecture Search")
    example_oneshot_search()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Start with random/grid search before complex methods")
    print("2. Use weight sharing to reduce search cost")
    print("3. Define reasonable search space constraints")
    print("4. Evaluate on separate validation set")
    print("5. Consider computational budget (time/resources)")
    print("6. Use early stopping during architecture evaluation")
    print("7. Apply transfer learning to speed up evaluation")
    print("8. Balance exploration vs exploitation")
    print("9. Use proxy tasks for faster architecture evaluation")
    print("10. Document search space and found architectures")
