#!/usr/bin/env python3
"""
=============================================================================
DQN for UAV Task Offloading - Complete Tutorial
=============================================================================

Author: Your Name
Purpose: Educational implementation of DQN for task offloading in edge-enabled UAV networks

=============================================================================
PROBLEM DESCRIPTION
=============================================================================

In edge-enabled UAV (Unmanned Aerial Vehicle) networks:
- Ground devices (IoT sensors, mobile phones) generate computational tasks
- Each device has LIMITED computing power and battery
- UAVs fly overhead and offer POWERFUL edge computing services

THE DECISION: For each task, should the device:
  (A) Process LOCALLY? (slow, uses device battery)
  (B) OFFLOAD to UAV? (transmission cost, but faster processing)

GOAL: Minimize total DELAY + ENERGY consumption across all tasks

=============================================================================
WHY REINFORCEMENT LEARNING?
=============================================================================

The optimal decision depends on many factors:
- Task size and complexity
- Device's remaining energy
- Distance to nearest UAV
- Channel quality (wireless signal strength)

Traditional approaches (like optimization or heuristics) require:
- Perfect knowledge of future tasks
- Solving complex optimization problems in real-time

RL ADVANTAGE: The agent LEARNS from experience and can make quick decisions
based on the current state, without needing to solve complex problems.

=============================================================================
DQN (Deep Q-Network) EXPLAINED
=============================================================================

Q-Learning Basics:
------------------
- Q(s, a) = Expected future reward for taking action 'a' in state 's'
- If we knew Q for all (state, action) pairs, we'd always pick the best action!

The Problem:
------------
- State space is continuous (infinite states possible)
- We can't store Q-values for every state in a table

DQN Solution:
-------------
- Use a Neural Network to APPROXIMATE Q-values
- Input: State (task size, device energy, distance, etc.)
- Output: Q-values for each action (local, offload)

Training Process:
-----------------
1. Agent interacts with environment, collects experiences
2. Stores experiences in Replay Buffer (for stable learning)
3. Samples batches and updates network using:

   Loss = (Q_predicted - Q_target)²

   Where Q_target = reward + γ * max(Q_next_state)

=============================================================================
PROJECT STRUCTURE
=============================================================================

uav_task_offloading/
│
├── environment.py  - Simulation environment (devices, UAVs, channel model)
├── dqn_agent.py    - DQN agent implementation (neural network, replay buffer)
├── train.py        - Training loop and baseline comparisons
├── visualize.py    - Plotting and visualization functions
├── main.py         - THIS FILE - Main entry point with full explanation
└── requirements.txt - Dependencies

=============================================================================
"""

import numpy as np
import time
import os

# Import our modules
from environment import UAVOffloadingEnv, Task, GroundDevice, UAV
from dqn_agent import DQNAgent, AllLocalBaseline, AllOffloadBaseline, RandomBaseline
from train import train_dqn, evaluate_policy, compare_policies
from visualize import create_all_plots


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def explain_environment():
    """Explain the simulation environment."""
    print_section("STEP 1: UNDERSTANDING THE ENVIRONMENT")

    explanation = """
    Our environment simulates a UAV-assisted edge computing network:

    GROUND DEVICES (IoT devices):
    ┌─────────────────────────────────────────────────────────────────┐
    │ • 10 devices scattered in a 500m x 500m area                    │
    │ • Each has limited CPU: 0.5-1.5 GHz (like a smartphone)         │
    │ • Limited battery: starts with 100 Joules                       │
    │ • Generates computational tasks randomly                        │
    └─────────────────────────────────────────────────────────────────┘

    UAVs (Flying Edge Servers):
    ┌─────────────────────────────────────────────────────────────────┐
    │ • 3 UAVs flying at 80-120m altitude                             │
    │ • Powerful CPUs: 8-12 GHz (much faster than ground devices!)    │
    │ • Wireless communication with ground devices                    │
    └─────────────────────────────────────────────────────────────────┘

    TASKS:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Data size: 100-500 KB (affects transmission time)             │
    │ • CPU cycles: 100-500 Mega cycles (affects processing time)     │
    │ • Each episode has 100 tasks to process                         │
    └─────────────────────────────────────────────────────────────────┘

    STATE (What the agent observes):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Task data size (normalized)                                  │
    │ 2. Task CPU cycles (normalized)                                 │
    │ 3. Device's remaining energy                                    │
    │ 4. Distance to nearest UAV                                      │
    │ 5. Channel quality (signal strength)                            │
    └─────────────────────────────────────────────────────────────────┘

    ACTIONS:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Action 0: Process task LOCALLY                                │
    │ • Action 1: OFFLOAD task to UAV                                 │
    └─────────────────────────────────────────────────────────────────┘

    REWARD:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Reward = -(delay_weight × delay + energy_weight × energy)       │
    │                                                                 │
    │ We want to MINIMIZE delay and energy, so reward is NEGATIVE.    │
    │ Higher reward (closer to 0) = better performance!               │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(explanation)


def explain_dqn():
    """Explain the DQN algorithm."""
    print_section("STEP 2: UNDERSTANDING DQN")

    explanation = """
    DQN (Deep Q-Network) is a reinforcement learning algorithm:

    COMPONENTS:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Q-NETWORK (Neural Network)                                   │
    │    • Input: 5-dimensional state                                 │
    │    • Hidden layers: [64, 64] neurons with ReLU activation       │
    │    • Output: 2 Q-values (one for local, one for offload)        │
    │                                                                 │
    │ 2. TARGET NETWORK (Stable learning targets)                     │
    │    • Copy of Q-network, updated less frequently                 │
    │    • Prevents oscillations during training                      │
    │                                                                 │
    │ 3. REPLAY BUFFER (Experience memory)                            │
    │    • Stores (state, action, reward, next_state) tuples          │
    │    • Randomly samples batches for training                      │
    │    • Breaks correlation between consecutive experiences         │
    └─────────────────────────────────────────────────────────────────┘

    TRAINING PROCESS:
    ┌─────────────────────────────────────────────────────────────────┐
    │ For each step:                                                  │
    │   1. Observe current state                                      │
    │   2. Select action using ε-greedy policy:                       │
    │      • With probability ε: random action (EXPLORE)              │
    │      • With probability 1-ε: best action (EXPLOIT)              │
    │   3. Execute action, receive reward and next state              │
    │   4. Store experience in replay buffer                          │
    │   5. Sample batch and update Q-network:                         │
    │                                                                 │
    │      target = reward + γ × max(Q_target(next_state))            │
    │      loss = (Q(state, action) - target)²                        │
    │                                                                 │
    │ After each episode:                                             │
    │   • Decay ε (explore less, exploit more)                        │
    │   • Periodically update target network                          │
    └─────────────────────────────────────────────────────────────────┘

    HYPERPARAMETERS:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Learning rate: 0.001 (how fast to update weights)             │
    │ • Gamma (γ): 0.99 (discount factor for future rewards)          │
    │ • Epsilon: 1.0 → 0.01 (exploration rate decay)                  │
    │ • Buffer size: 10,000 (experience memory capacity)              │
    │ • Batch size: 64 (samples per training step)                    │
    │ • Target update: every 10 episodes                              │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(explanation)


def explain_baselines():
    """Explain the baseline policies."""
    print_section("STEP 3: BASELINE POLICIES")

    explanation = """
    We compare DQN against simple baseline policies:

    1. ALL LOCAL BASELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Always processes tasks locally (never offloads)               │
    │ • Advantage: No transmission delay or energy                    │
    │ • Disadvantage: Slow processing, high device energy usage       │
    └─────────────────────────────────────────────────────────────────┘

    2. ALL OFFLOAD BASELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Always offloads to UAV (never processes locally)              │
    │ • Advantage: Fast UAV processing, saves device computation      │
    │ • Disadvantage: Transmission delay and energy, network overhead │
    └─────────────────────────────────────────────────────────────────┘

    3. RANDOM BASELINE:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Randomly chooses local or offload (50/50)                     │
    │ • No intelligence, just for reference                           │
    └─────────────────────────────────────────────────────────────────┘

    WHY DQN SHOULD BE BETTER:
    ┌─────────────────────────────────────────────────────────────────┐
    │ DQN learns WHEN to offload based on:                            │
    │ • Task characteristics (small tasks → local, large → offload)   │
    │ • Device state (low energy → offload to save battery)           │
    │ • Channel conditions (good signal → offload, bad → local)       │
    │                                                                 │
    │ This adaptive behavior should outperform fixed policies!        │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(explanation)


def demonstrate_environment():
    """Create and demonstrate the environment."""
    print_section("STEP 4: ENVIRONMENT DEMONSTRATION")

    # Create environment
    env = UAVOffloadingEnv(num_devices=10, num_uavs=3, episode_length=100)

    print("\nEnvironment created successfully!")
    print(f"  • Number of ground devices: {env.num_devices}")
    print(f"  • Number of UAVs: {env.num_uavs}")
    print(f"  • Area size: {env.area_size}m x {env.area_size}m")
    print(f"  • State dimension: {env.state_dim}")
    print(f"  • Action dimension: {env.action_dim}")

    # Show a sample state
    state = env.reset()
    print(f"\nSample initial state: {state}")
    print("  [0]: Task data size (normalized)")
    print("  [1]: Task CPU cycles (normalized)")
    print("  [2]: Device energy level (normalized)")
    print("  [3]: Distance to UAV (normalized)")
    print("  [4]: Channel quality (normalized)")

    # Demonstrate one step with each action
    print("\nDemonstrating one step with LOCAL processing (action=0):")
    state = env.reset()
    _, reward, _, info = env.step(0)
    print(f"  Delay: {info['delay']*1000:.2f} ms")
    print(f"  Energy: {info['energy']*1000:.4f} mJ")
    print(f"  Reward: {reward:.4f}")

    print("\nDemonstrating one step with OFFLOAD (action=1):")
    state = env.reset()
    _, reward, _, info = env.step(1)
    print(f"  Delay: {info['delay']*1000:.2f} ms")
    print(f"  Energy: {info['energy']*1000:.4f} mJ")
    print(f"  Reward: {reward:.4f}")

    return env


def run_complete_experiment(
    num_devices: int = 10,
    num_uavs: int = 3,
    num_train_episodes: int = 500,
    num_eval_episodes: int = 100,
    output_dir: str = "./results"
):
    """
    Run the complete training and evaluation experiment.

    Args:
        num_devices: Number of ground devices
        num_uavs: Number of UAVs
        num_train_episodes: Number of training episodes
        num_eval_episodes: Number of evaluation episodes
        output_dir: Directory to save results

    Returns:
        Tuple of (agent, history, results)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print_section("STEP 5: RUNNING THE EXPERIMENT")

    print(f"\nConfiguration:")
    print(f"  • Ground devices: {num_devices}")
    print(f"  • UAVs: {num_uavs}")
    print(f"  • Training episodes: {num_train_episodes}")
    print(f"  • Evaluation episodes: {num_eval_episodes}")
    print(f"  • Output directory: {output_dir}")

    # Create environment
    env = UAVOffloadingEnv(
        num_devices=num_devices,
        num_uavs=num_uavs,
        episode_length=100
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
    print_section("TRAINING DQN AGENT")
    start_time = time.time()
    history = train_dqn(env, agent, num_train_episodes, print_every=50)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Compare with baselines
    print_section("EVALUATING AND COMPARING POLICIES")
    results = compare_policies(env, agent, num_eval_episodes)

    # Create visualizations
    print_section("GENERATING VISUALIZATIONS")
    create_all_plots(env, history, results, output_dir)

    # Save the trained agent
    agent_path = f"{output_dir}/trained_agent.pkl"
    agent.save(agent_path)
    print(f"\nTrained agent saved to: {agent_path}")

    return agent, history, results, env


def print_final_summary(results):
    """Print a final summary of results."""
    print_section("FINAL SUMMARY")

    print("""
    EXPERIMENT COMPLETED SUCCESSFULLY!

    KEY FINDINGS:
    """)

    # Calculate improvements
    dqn_reward = results['DQN']['avg_reward']
    dqn_delay = results['DQN']['avg_delay']
    dqn_energy = results['DQN']['avg_energy']

    print(f"    DQN Performance:")
    print(f"    ├── Average Reward: {dqn_reward:.2f}")
    print(f"    ├── Average Delay: {dqn_delay*1000:.2f} ms")
    print(f"    ├── Average Energy: {dqn_energy*1000:.4f} mJ")
    print(f"    └── Offload Ratio: {results['DQN']['offload_ratio']*100:.1f}%")

    print("\n    Improvement over baselines:")
    for name in ['All Local', 'All Offload', 'Random']:
        baseline_reward = results[name]['avg_reward']
        improvement = ((dqn_reward - baseline_reward) / abs(baseline_reward)) * 100
        print(f"    • vs {name}: {improvement:+.1f}% reward improvement")

    print("""
    INTERPRETATION:
    ───────────────────────────────────────────────────────────────────
    The DQN agent learned to make intelligent offloading decisions by
    considering the task characteristics, device state, and channel
    conditions. Unlike fixed policies, DQN adapts its strategy based
    on the current situation, leading to better overall performance.

    The learned policy balances:
    • Processing small/simple tasks locally (low transmission overhead)
    • Offloading large/complex tasks to UAVs (faster processing)
    • Considering device energy levels to prevent battery drain

    OUTPUT FILES:
    ───────────────────────────────────────────────────────────────────
    Check the 'results' directory for:
    • training_history.png  - Training progress over episodes
    • policy_comparison.png - Bar chart comparing all policies
    • offload_ratio.png     - Local vs offload distribution
    • network_topology.png  - Device and UAV positions
    • reward_comparison.png - DQN learning curve
    • trained_agent.pkl     - Saved model for later use
    """)


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   DQN FOR UAV TASK OFFLOADING - EDUCATIONAL IMPLEMENTATION        ║
    ║                                                                   ║
    ║   This program demonstrates how Deep Q-Networks can be used       ║
    ║   to solve task offloading problems in UAV-assisted edge          ║
    ║   computing networks.                                             ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Step-by-step explanations
    explain_environment()
    input("\nPress Enter to continue to DQN explanation...")

    explain_dqn()
    input("\nPress Enter to continue to baseline explanation...")

    explain_baselines()
    input("\nPress Enter to demonstrate the environment...")

    demonstrate_environment()
    input("\nPress Enter to start the training experiment...")

    # Run the experiment
    agent, history, results, env = run_complete_experiment(
        num_devices=10,
        num_uavs=3,
        num_train_episodes=500,
        num_eval_episodes=100,
        output_dir="./results"
    )

    # Print final summary
    print_final_summary(results)

    print("\nExperiment completed! Check the 'results' folder for visualizations.")
    return agent, history, results


def run_quick_experiment():
    """Run a quick experiment without interactive prompts (for testing)."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║   QUICK EXPERIMENT MODE (No interactive prompts)                  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    agent, history, results, env = run_complete_experiment(
        num_devices=10,
        num_uavs=3,
        num_train_episodes=300,  # Shorter for quick test
        num_eval_episodes=50,
        output_dir="./results"
    )

    print_final_summary(results)
    return agent, history, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick mode without prompts
        run_quick_experiment()
    else:
        # Interactive mode with explanations
        main()
