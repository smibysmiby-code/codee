"""
Deep Q-Network (DQN) Agent for Task Offloading
==============================================

This module implements a DQN agent that learns optimal offloading decisions.

WHAT IS DQN?
------------
DQN (Deep Q-Network) is a reinforcement learning algorithm that:
1. Uses a neural network to approximate the Q-function
2. Q(s, a) estimates the expected future reward for taking action 'a' in state 's'
3. The agent learns by interacting with the environment and updating Q-values

KEY COMPONENTS:
---------------
1. Neural Network: Maps state → Q-values for each action
2. Replay Buffer: Stores past experiences for stable learning
3. Target Network: A copy of the main network for stable Q-value targets
4. Epsilon-Greedy: Balances exploration vs exploitation

LEARNING PROCESS:
-----------------
1. Agent observes state 's'
2. Selects action 'a' (explore randomly OR exploit best known action)
3. Receives reward 'r' and next state 's''
4. Stores experience (s, a, r, s', done) in replay buffer
5. Samples batch from buffer and updates network using:

   Loss = (Q(s,a) - (r + γ * max Q(s',a')))²

   Where γ (gamma) is the discount factor for future rewards.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List
import pickle


class ReplayBuffer:
    """
    Experience Replay Buffer

    WHY USE REPLAY BUFFER?
    ----------------------
    1. Breaks correlation between consecutive experiences
    2. Allows reuse of past experiences multiple times
    3. Improves sample efficiency and stability

    Each experience is a tuple: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)

        # Unpack batch into separate arrays
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class NeuralNetwork:
    """
    Simple Neural Network implemented from scratch (no PyTorch/TensorFlow needed).

    Architecture: Input → Hidden1 → Hidden2 → Output
    Activation: ReLU for hidden layers, Linear for output

    This is a basic implementation for educational purposes.
    For larger problems, use PyTorch or TensorFlow.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 learning_rate: float = 0.001):
        """
        Initialize network with random weights.

        Args:
            input_dim: Size of input (state dimension)
            hidden_dims: List of hidden layer sizes, e.g., [64, 64]
            output_dim: Size of output (number of actions)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = []
        self.biases = []

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization for better gradient flow
            scale = np.sqrt(2.0 / dims[i])
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros((1, dims[i+1]))
            self.layers.append(W)
            self.biases.append(b)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Tuple of (output, list of intermediate activations for backprop)
        """
        activations = [x]

        for i, (W, b) in enumerate(zip(self.layers, self.biases)):
            x = np.dot(x, W) + b

            # Apply ReLU to all layers except the last
            if i < len(self.layers) - 1:
                x = self.relu(x)

            activations.append(x)

        return x, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get network output (Q-values) for given states."""
        output, _ = self.forward(x)
        return output

    def train_step(self, states: np.ndarray, actions: np.ndarray,
                   targets: np.ndarray) -> float:
        """
        Perform one training step (forward + backward pass).

        Args:
            states: Batch of states
            actions: Batch of actions taken
            targets: Target Q-values to learn

        Returns:
            Loss value
        """
        batch_size = states.shape[0]

        # Forward pass
        output, activations = self.forward(states)

        # Calculate loss (MSE) only for the taken actions
        # We only update Q-value for the action that was taken
        q_values = output.copy()
        loss = 0.0

        # Create target tensor (copy current Q-values, update only taken action)
        target_q = output.copy()
        for i in range(batch_size):
            target_q[i, actions[i]] = targets[i]
            loss += (output[i, actions[i]] - targets[i]) ** 2

        loss /= batch_size

        # Backward pass (gradient descent)
        # Output layer gradient
        dL_dout = 2 * (output - target_q) / batch_size

        # Backpropagate through layers
        gradients_W = []
        gradients_b = []
        delta = dL_dout

        for i in range(len(self.layers) - 1, -1, -1):
            # Gradient for weights
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            gradients_W.insert(0, dW)
            gradients_b.insert(0, db)

            if i > 0:
                # Propagate gradient to previous layer
                delta = np.dot(delta, self.layers[i].T)
                # Apply ReLU derivative
                delta = delta * self.relu_derivative(activations[i])

        # Update weights
        for i in range(len(self.layers)):
            self.layers[i] -= self.learning_rate * gradients_W[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

        return loss

    def copy_weights_from(self, other_network):
        """Copy weights from another network (for target network update)."""
        for i in range(len(self.layers)):
            self.layers[i] = other_network.layers[i].copy()
            self.biases[i] = other_network.biases[i].copy()


class DQNAgent:
    """
    Deep Q-Network Agent for Task Offloading

    The agent learns to make binary offloading decisions:
    - Action 0: Process task locally
    - Action 1: Offload task to UAV

    The goal is to minimize total delay and energy consumption.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (2 for binary offloading)
            hidden_dims: Hidden layer sizes for neural network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor (how much to value future rewards)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate of epsilon decay per episode
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network (episodes)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon for exploration (ε-greedy policy)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-Network (main network that gets updated)
        self.q_network = NeuralNetwork(
            state_dim, hidden_dims, action_dim, learning_rate
        )

        # Target Network (provides stable targets for learning)
        # This is a key DQN innovation - it's updated less frequently
        self.target_network = NeuralNetwork(
            state_dim, hidden_dims, action_dim, learning_rate
        )
        self.target_network.copy_weights_from(self.q_network)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training statistics
        self.training_step = 0
        self.episode_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using ε-greedy policy.

        EPSILON-GREEDY STRATEGY:
        - With probability ε: choose random action (EXPLORE)
        - With probability 1-ε: choose best action (EXPLOIT)

        During training, we explore more initially and gradually exploit more.
        During evaluation, we always choose the best action.

        Args:
            state: Current state observation
            training: If True, use ε-greedy; if False, always exploit

        Returns:
            Selected action (0 or 1)
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: choose action with highest Q-value
            state = state.reshape(1, -1)
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store an experience in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> float:
        """
        Train the agent using a batch of experiences from replay buffer.

        THE DQN UPDATE RULE:
        -------------------
        For each experience (s, a, r, s', done):

        if done:
            target = r
        else:
            target = r + γ * max_a' Q_target(s', a')

        Then minimize: (Q(s, a) - target)²

        Returns:
            Training loss
        """
        # Don't train if not enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Calculate target Q-values using target network
        next_q_values = self.target_network.predict(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        # Compute targets
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        # Train Q-network
        loss = self.q_network.train_step(states, actions, targets)

        self.training_step += 1

        return loss

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.copy_weights_from(self.q_network)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()

    def save(self, filepath: str):
        """Save agent to file."""
        data = {
            'q_network_layers': self.q_network.layers,
            'q_network_biases': self.q_network.biases,
            'target_network_layers': self.target_network.layers,
            'target_network_biases': self.target_network.biases,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_network.layers = data['q_network_layers']
        self.q_network.biases = data['q_network_biases']
        self.target_network.layers = data['target_network_layers']
        self.target_network.biases = data['target_network_biases']
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']


class AllLocalBaseline:
    """
    Baseline policy that always processes tasks locally.
    This is a simple baseline for comparison.
    """

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        return 0  # Always local


class AllOffloadBaseline:
    """
    Baseline policy that always offloads tasks to UAV.
    This is another simple baseline for comparison.
    """

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        return 1  # Always offload


class RandomBaseline:
    """
    Baseline policy that randomly chooses between local and offload.
    """

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        return random.randint(0, 1)


# Example usage
if __name__ == "__main__":
    # Create a simple test
    state_dim = 5
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim)

    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"Selected action: {action}")

    # Test storing and training
    for i in range(100):
        state = np.random.randn(state_dim)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = np.random.randn(state_dim)
        done = i == 99

        agent.store_experience(state, action, reward, next_state, done)

    # Train
    loss = agent.train()
    print(f"Training loss: {loss:.4f}")
