"""
Reinforcement Learning Patterns

This module demonstrates reinforcement learning algorithms and patterns
implemented in TensorFlow.

Patterns covered:
1. Deep Q-Network (DQN)
2. Double DQN
3. Dueling DQN
4. Policy Gradient (REINFORCE)
5. Actor-Critic (A2C)
6. Experience Replay Buffer
7. Target Network
8. Epsilon-Greedy Exploration
9. PPO (Proximal Policy Optimization)
10. Advantage Estimation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque
import random


# Pattern 1: Deep Q-Network (DQN)
class DQN(keras.Model):
    """Deep Q-Network for value-based RL."""
    
    def __init__(self, state_dim, action_dim, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(hidden_units, activation='relu')
        self.q_values = layers.Dense(action_dim)
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.q_values(x)


class DQNAgent:
    """DQN Agent with training logic."""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Q-network
        self.q_network = DQN(state_dim, action_dim)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.q_network(state)
        return tf.argmax(q_values[0]).numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=32):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Compute target Q-values
        next_q_values = self.q_network(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Train network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values_selected = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_values_selected))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.numpy()


def example_dqn():
    """Example: DQN agent."""
    state_dim = 4
    action_dim = 2
    
    agent = DQNAgent(state_dim, action_dim)
    
    print("Deep Q-Network (DQN) Example:")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Epsilon: {agent.epsilon:.4f}")
    
    # Simulate episode
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    return agent


# Pattern 2: Double DQN
class DoubleDQNAgent(DQNAgent):
    """Double DQN with target network."""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        super().__init__(state_dim, action_dim, learning_rate, gamma)
        
        # Target network
        self.target_network = DQN(state_dim, action_dim)
        self.update_target_network()
        
        self.update_frequency = 100
        self.step_count = 0
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def train_step(self, batch_size=32):
        """Train with Double DQN algorithm."""
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Double DQN: use Q-network to select actions, target network to evaluate
        next_q_values = self.q_network(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)
        
        target_next_q = self.target_network(next_states)
        target_next_q = tf.reduce_sum(
            target_next_q * tf.one_hot(best_actions, self.action_dim),
            axis=1
        )
        
        targets = rewards + self.gamma * target_next_q * (1 - dones)
        
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            q_values_selected = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_values_selected))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self.update_target_network()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.numpy()


def example_double_dqn():
    """Example: Double DQN."""
    agent = DoubleDQNAgent(state_dim=4, action_dim=2)
    
    print("\nDouble DQN Example:")
    print("Uses separate target network for stable learning")
    print(f"Update frequency: {agent.update_frequency}")
    
    return agent


# Pattern 3: Dueling DQN
class DuelingDQN(keras.Model):
    """Dueling DQN architecture."""
    
    def __init__(self, state_dim, action_dim, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        
        # Value stream
        self.value_stream = keras.Sequential([
            layers.Dense(hidden_units // 2, activation='relu'),
            layers.Dense(1)
        ])
        
        # Advantage stream
        self.advantage_stream = keras.Sequential([
            layers.Dense(hidden_units // 2, activation='relu'),
            layers.Dense(action_dim)
        ])
    
    def call(self, state):
        x = self.dense1(state)
        
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        
        return q_values


def example_dueling_dqn():
    """Example: Dueling DQN architecture."""
    model = DuelingDQN(state_dim=4, action_dim=2)
    
    state = tf.random.normal((1, 4))
    q_values = model(state)
    
    print("\nDueling DQN Example:")
    print(f"Q-values shape: {q_values.shape}")
    print("Separates value and advantage streams")
    
    return model


# Pattern 4: Policy Gradient (REINFORCE)
class PolicyNetwork(keras.Model):
    """Policy network for REINFORCE."""
    
    def __init__(self, state_dim, action_dim, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(hidden_units, activation='relu')
        self.action_probs = layers.Dense(action_dim, activation='softmax')
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.action_probs(x)


class REINFORCEAgent:
    """REINFORCE algorithm implementation."""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gamma = gamma
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
    
    def select_action(self, state):
        """Sample action from policy."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state)[0].numpy()
        action = np.random.choice(len(probs), p=probs)
        return action
    
    def store_transition(self, state, action, reward):
        """Store transition during episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train_episode(self):
        """Train after episode completion."""
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            action_probs = self.policy(states)
            action_masks = tf.one_hot(actions, action_probs.shape[1])
            selected_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            
            # Policy gradient loss
            loss = -tf.reduce_mean(tf.math.log(selected_probs + 1e-8) * returns)
        
        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        
        # Clear episode memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        
        return loss.numpy()


def example_reinforce():
    """Example: REINFORCE algorithm."""
    agent = REINFORCEAgent(state_dim=4, action_dim=2)
    
    print("\nREINFORCE (Policy Gradient) Example:")
    print("Learns policy directly without Q-values")
    
    # Simulate episode
    state = np.random.randn(4).astype(np.float32)
    action = agent.select_action(state)
    print(f"Sampled action: {action}")
    
    return agent


# Pattern 5: Actor-Critic (A2C)
class ActorCritic(keras.Model):
    """Actor-Critic network."""
    
    def __init__(self, state_dim, action_dim, hidden_units=64):
        super().__init__()
        self.shared = layers.Dense(hidden_units, activation='relu')
        
        # Actor (policy)
        self.actor = keras.Sequential([
            layers.Dense(hidden_units // 2, activation='relu'),
            layers.Dense(action_dim, activation='softmax')
        ])
        
        # Critic (value)
        self.critic = keras.Sequential([
            layers.Dense(hidden_units // 2, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, state):
        x = self.shared(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


class A2CAgent:
    """Advantage Actor-Critic agent."""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.ac_network = ActorCritic(state_dim, action_dim)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.gamma = gamma
    
    def select_action(self, state):
        """Select action from policy."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs, _ = self.ac_network(state)
        action = np.random.choice(action_probs.shape[1], p=action_probs[0].numpy())
        return action
    
    def train_step(self, state, action, reward, next_state, done):
        """Single-step A2C update."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs, value = self.ac_network(state)
            _, next_value = self.ac_network(next_state)
            
            # Compute advantage
            target = reward + self.gamma * next_value * (1 - done)
            advantage = target - value
            
            # Actor loss (policy gradient with advantage)
            action_mask = tf.one_hot([action], action_probs.shape[1])
            selected_prob = tf.reduce_sum(action_probs * action_mask, axis=1)
            actor_loss = -tf.math.log(selected_prob + 1e-8) * tf.stop_gradient(advantage)
            
            # Critic loss (value function)
            critic_loss = tf.square(advantage)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.ac_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.ac_network.trainable_variables))
        
        return loss.numpy()


def example_a2c():
    """Example: Actor-Critic (A2C)."""
    agent = A2CAgent(state_dim=4, action_dim=2)
    
    print("\nActor-Critic (A2C) Example:")
    print("Combines policy gradient (actor) with value function (critic)")
    
    # Simulate transition
    state = np.random.randn(4).astype(np.float32)
    action = agent.select_action(state)
    next_state = np.random.randn(4).astype(np.float32)
    reward = 1.0
    done = 0
    
    loss = agent.train_step(state, action, reward, next_state, done)
    print(f"Training loss: {loss:.4f}")
    
    return agent


# Pattern 6: Experience Replay Buffer
class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


def example_replay_buffer():
    """Example: Experience replay buffer."""
    buffer = ReplayBuffer(capacity=1000)
    
    # Add experiences
    for _ in range(100):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.randint(2)
        
        buffer.add(state, action, reward, next_state, done)
    
    # Sample batch
    batch = buffer.sample(32)
    
    print("\nExperience Replay Buffer Example:")
    print(f"Buffer size: {len(buffer)}")
    print(f"Batch shapes: states={batch[0].shape}, actions={batch[1].shape}")
    
    return buffer


# Pattern 7: Prioritized Experience Replay
class PrioritizedReplayBuffer:
    """Prioritized experience replay."""
    
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """Sample batch with priorities."""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states = np.array([x[0] for x in samples])
        actions = np.array([x[1] for x in samples])
        rewards = np.array([x[2] for x in samples])
        next_states = np.array([x[3] for x in samples])
        dones = np.array([x[4] for x in samples])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


def example_prioritized_replay():
    """Example: Prioritized experience replay."""
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
    
    # Add experiences
    for _ in range(100):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.randint(2)
        
        buffer.add(state, action, reward, next_state, done)
    
    print("\nPrioritized Experience Replay Example:")
    print(f"Buffer size: {len(buffer)}")
    print("Samples high-priority transitions more frequently")
    
    return buffer


# Pattern 8: Epsilon-Greedy Exploration
class EpsilonGreedy:
    """Epsilon-greedy exploration strategy."""
    
    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, q_values, explore=True):
        """Select action with epsilon-greedy."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)
    
    def decay(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def example_epsilon_greedy():
    """Example: Epsilon-greedy exploration."""
    explorer = EpsilonGreedy(epsilon_start=1.0, epsilon_end=0.01)
    
    q_values = np.array([0.5, 0.8, 0.3])
    
    print("\nEpsilon-Greedy Exploration:")
    print(f"Initial epsilon: {explorer.epsilon:.4f}")
    
    # Simulate action selection
    actions = []
    for _ in range(10):
        action = explorer.select_action(q_values)
        actions.append(action)
        explorer.decay()
    
    print(f"Final epsilon: {explorer.epsilon:.4f}")
    print(f"Actions selected: {actions}")
    
    return explorer


if __name__ == "__main__":
    print("Reinforcement Learning Patterns\n" + "="*60)
    
    # Example 1: DQN
    print("\n1. Deep Q-Network (DQN)")
    agent1 = example_dqn()
    
    # Example 2: Double DQN
    print("\n2. Double DQN")
    agent2 = example_double_dqn()
    
    # Example 3: Dueling DQN
    print("\n3. Dueling DQN")
    model3 = example_dueling_dqn()
    
    # Example 4: REINFORCE
    print("\n4. REINFORCE (Policy Gradient)")
    agent4 = example_reinforce()
    
    # Example 5: A2C
    print("\n5. Actor-Critic (A2C)")
    agent5 = example_a2c()
    
    # Example 6: Replay Buffer
    print("\n6. Experience Replay Buffer")
    buffer6 = example_replay_buffer()
    
    # Example 7: Prioritized Replay
    print("\n7. Prioritized Experience Replay")
    buffer7 = example_prioritized_replay()
    
    # Example 8: Epsilon-Greedy
    print("\n8. Epsilon-Greedy Exploration")
    explorer8 = example_epsilon_greedy()
    
    print("\n" + "="*60)
    print("Best Practices:")
    print("1. Use experience replay for sample efficiency")
    print("2. Target network improves stability (Double DQN)")
    print("3. Dueling architecture separates value and advantage")
    print("4. Policy gradient for continuous action spaces")
    print("5. Actor-Critic combines benefits of both approaches")
    print("6. Prioritized replay for important transitions")
    print("7. Epsilon-greedy balances exploration/exploitation")
    print("8. Normalize rewards for stable training")
    print("9. Use discount factor (gamma) close to 1.0")
    print("10. Monitor episode rewards and Q-value estimates")
