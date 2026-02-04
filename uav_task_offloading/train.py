"""
Training Script for DQN Task Offloading
=======================================

This script trains the DQN agent and compares it with baseline policies.

TRAINING PROCESS:
-----------------
1. Initialize environment and DQN agent
2. For each episode:
   a. Reset environment
   b. For each step:
      - Observe state
      - Select action (ε-greedy)
      - Execute action, get reward and next state
      - Store experience in replay buffer
      - Train on batch from buffer
   c. Decay exploration rate
3. Evaluate trained agent against baselines

BASELINES:
----------
1. All Local: Process every task locally
2. All Offload: Offload every task to UAV
3. Random: Randomly choose local or offload
"""

import numpy as np
from typing import Dict, List, Tuple
import time

from environment import UAVOffloadingEnv
from dqn_agent import DQNAgent, AllLocalBaseline, AllOffloadBaseline, RandomBaseline


def train_dqn(
    env: UAVOffloadingEnv,
    agent: DQNAgent,
    num_episodes: int = 500,
    print_every: int = 50
) -> Dict[str, List]:
    """
    Train the DQN agent.

    Args:
        env: The UAV offloading environment
        agent: The DQN agent to train
        num_episodes: Number of training episodes
        print_every: Print progress every N episodes

    Returns:
        Dictionary containing training history
    """
    history = {
        'episode_rewards': [],
        'episode_delays': [],
        'episode_energies': [],
        'losses': [],
        'epsilons': []
    }

    print("=" * 60)
    print("STARTING DQN TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print("=" * 60)

    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        episode_losses = []

        # Run one episode
        done = False
        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train()
            if loss > 0:
                episode_losses.append(loss)

            # Track metrics
            episode_reward += reward
            episode_delay += info['delay']
            episode_energy += info['energy']

            state = next_state

        # End of episode
        agent.decay_epsilon()

        # Store history
        history['episode_rewards'].append(episode_reward)
        history['episode_delays'].append(episode_delay)
        history['episode_energies'].append(episode_energy)
        history['losses'].append(np.mean(episode_losses) if episode_losses else 0)
        history['epsilons'].append(agent.epsilon)

        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(history['episode_rewards'][-print_every:])
            avg_delay = np.mean(history['episode_delays'][-print_every:])
            avg_energy = np.mean(history['episode_energies'][-print_every:])
            avg_loss = np.mean(history['losses'][-print_every:])

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Delay: {avg_delay*1000:.2f} ms")
            print(f"  Avg Energy: {avg_energy*1000:.4f} mJ")
            print(f"  Avg Loss: {avg_loss:.6f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print("-" * 40)

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")

    return history


def evaluate_policy(
    env: UAVOffloadingEnv,
    agent,
    num_episodes: int = 100,
    policy_name: str = "Policy"
) -> Dict[str, float]:
    """
    Evaluate a policy (DQN or baseline).

    Args:
        env: The environment
        agent: The policy to evaluate (must have select_action method)
        num_episodes: Number of evaluation episodes
        policy_name: Name of the policy for printing

    Returns:
        Dictionary of average metrics
    """
    total_rewards = []
    total_delays = []
    total_energies = []
    local_count = 0
    offload_count = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0

        done = False
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_delay += info['delay']
            episode_energy += info['energy']

            if action == 0:
                local_count += 1
            else:
                offload_count += 1

            state = next_state

        total_rewards.append(episode_reward)
        total_delays.append(episode_delay)
        total_energies.append(episode_energy)

    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_delay': np.mean(total_delays),
        'std_delay': np.std(total_delays),
        'avg_energy': np.mean(total_energies),
        'std_energy': np.std(total_energies),
        'local_ratio': local_count / (local_count + offload_count),
        'offload_ratio': offload_count / (local_count + offload_count)
    }

    return results


def compare_policies(env: UAVOffloadingEnv, dqn_agent: DQNAgent, num_eval_episodes: int = 100):
    """
    Compare DQN agent with baseline policies.

    Args:
        env: The environment
        dqn_agent: Trained DQN agent
        num_eval_episodes: Number of episodes for evaluation
    """
    print("\n" + "=" * 60)
    print("POLICY COMPARISON")
    print("=" * 60)

    # Create baselines
    baselines = {
        'DQN': dqn_agent,
        'All Local': AllLocalBaseline(),
        'All Offload': AllOffloadBaseline(),
        'Random': RandomBaseline()
    }

    results = {}

    for name, policy in baselines.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_policy(env, policy, num_eval_episodes, name)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Policy':<15} {'Avg Reward':<15} {'Avg Delay (ms)':<18} {'Avg Energy (mJ)':<18} {'Local %':<10}")
    print("-" * 80)

    for name, res in results.items():
        print(f"{name:<15} {res['avg_reward']:<15.2f} {res['avg_delay']*1000:<18.2f} "
              f"{res['avg_energy']*1000:<18.4f} {res['local_ratio']*100:<10.1f}")

    print("=" * 80)

    # Improvement analysis
    print("\n" + "=" * 60)
    print("DQN IMPROVEMENT OVER BASELINES")
    print("=" * 60)

    dqn_reward = results['DQN']['avg_reward']
    for name in ['All Local', 'All Offload', 'Random']:
        baseline_reward = results[name]['avg_reward']
        improvement = ((dqn_reward - baseline_reward) / abs(baseline_reward)) * 100
        print(f"vs {name}: {improvement:+.2f}% reward improvement")

    return results


def run_training_and_evaluation(
    num_devices: int = 10,
    num_uavs: int = 3,
    num_train_episodes: int = 500,
    num_eval_episodes: int = 100
):
    """
    Complete training and evaluation pipeline.

    Args:
        num_devices: Number of ground devices
        num_uavs: Number of UAVs
        num_train_episodes: Training episodes
        num_eval_episodes: Evaluation episodes

    Returns:
        Tuple of (trained_agent, training_history, comparison_results)
    """
    print("\n" + "=" * 60)
    print("UAV TASK OFFLOADING WITH DQN")
    print("=" * 60)
    print(f"Ground Devices: {num_devices}")
    print(f"UAVs: {num_uavs}")
    print("=" * 60)

    # Create environment
    env = UAVOffloadingEnv(
        num_devices=num_devices,
        num_uavs=num_uavs,
        episode_length=100  # 100 tasks per episode
    )

    # Create DQN agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dims=[64, 64],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )

    # Train
    history = train_dqn(env, agent, num_train_episodes)

    # Compare with baselines
    results = compare_policies(env, agent, num_eval_episodes)

    return agent, history, results


if __name__ == "__main__":
    # Run the complete pipeline
    agent, history, results = run_training_and_evaluation(
        num_devices=10,
        num_uavs=3,
        num_train_episodes=500,
        num_eval_episodes=100
    )
