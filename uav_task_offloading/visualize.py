"""
Visualization Module for DQN Task Offloading
============================================

This module provides functions to visualize:
1. Training progress (rewards, losses, epsilon)
2. Policy comparison (bar charts)
3. Network topology (device and UAV positions)

All plots are saved to files for easy viewing.
"""

import numpy as np
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt


def plot_training_history(history: Dict[str, List], save_path: str = "training_history.png"):
    """
    Plot training metrics over episodes.

    Creates a 2x2 subplot showing:
    - Episode rewards
    - Episode delays
    - Training losses
    - Epsilon decay

    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')

    episodes = range(1, len(history['episode_rewards']) + 1)

    # Smooth function for visualization
    def smooth(data, window=20):
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2)
            smoothed.append(np.mean(data[start:end]))
        return smoothed

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, history['episode_rewards'], alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smooth(history['episode_rewards']), color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards (Higher is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Episode Delays
    ax2 = axes[0, 1]
    delays_ms = [d * 1000 for d in history['episode_delays']]
    ax2.plot(episodes, delays_ms, alpha=0.3, color='red', label='Raw')
    ax2.plot(episodes, smooth(delays_ms), color='red', linewidth=2, label='Smoothed')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Delay (ms)')
    ax2.set_title('Episode Delays (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training Losses
    ax3 = axes[1, 0]
    ax3.plot(episodes, history['losses'], alpha=0.3, color='green', label='Raw')
    ax3.plot(episodes, smooth(history['losses']), color='green', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Epsilon Decay
    ax4 = axes[1, 1]
    ax4.plot(episodes, history['epsilons'], color='purple', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon (ε)')
    ax4.set_title('Exploration Rate (ε-greedy)')
    ax4.axhline(y=0.01, color='gray', linestyle='--', label='Min ε')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def plot_policy_comparison(results: Dict[str, Dict], save_path: str = "policy_comparison.png"):
    """
    Plot bar charts comparing different policies.

    Creates a 1x3 subplot showing:
    - Average Reward comparison
    - Average Delay comparison
    - Average Energy comparison

    Args:
        results: Dictionary of policy results
        save_path: Path to save the figure
    """
    policies = list(results.keys())
    x = np.arange(len(policies))
    width = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Policy Comparison', fontsize=16, fontweight='bold')

    # Colors for each policy
    colors = {
        'DQN': '#2ecc71',  # Green for DQN (best)
        'All Local': '#e74c3c',  # Red
        'All Offload': '#3498db',  # Blue
        'Random': '#9b59b6'  # Purple
    }
    bar_colors = [colors.get(p, '#95a5a6') for p in policies]

    # Plot 1: Average Reward
    ax1 = axes[0]
    rewards = [results[p]['avg_reward'] for p in policies]
    reward_stds = [results[p]['std_reward'] for p in policies]
    bars1 = ax1.bar(x, rewards, width, color=bar_colors, yerr=reward_stds, capsize=5)
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight best
    best_idx = np.argmax(rewards)
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(3)

    # Plot 2: Average Delay
    ax2 = axes[1]
    delays = [results[p]['avg_delay'] * 1000 for p in policies]  # Convert to ms
    delay_stds = [results[p]['std_delay'] * 1000 for p in policies]
    bars2 = ax2.bar(x, delays, width, color=bar_colors, yerr=delay_stds, capsize=5)
    ax2.set_xlabel('Policy')
    ax2.set_ylabel('Average Delay (ms)')
    ax2.set_title('Average Delay (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies)
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight best (lowest)
    best_idx = np.argmin(delays)
    bars2[best_idx].set_edgecolor('gold')
    bars2[best_idx].set_linewidth(3)

    # Plot 3: Average Energy
    ax3 = axes[2]
    energies = [results[p]['avg_energy'] * 1000 for p in policies]  # Convert to mJ
    energy_stds = [results[p]['std_energy'] * 1000 for p in policies]
    bars3 = ax3.bar(x, energies, width, color=bar_colors, yerr=energy_stds, capsize=5)
    ax3.set_xlabel('Policy')
    ax3.set_ylabel('Average Energy (mJ)')
    ax3.set_title('Average Energy (Lower is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(policies)
    ax3.grid(True, alpha=0.3, axis='y')

    # Highlight best (lowest)
    best_idx = np.argmin(energies)
    bars3[best_idx].set_edgecolor('gold')
    bars3[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Policy comparison plot saved to: {save_path}")


def plot_offload_ratio(results: Dict[str, Dict], save_path: str = "offload_ratio.png"):
    """
    Plot the offloading ratio for each policy.

    Shows what percentage of tasks are processed locally vs offloaded.

    Args:
        results: Dictionary of policy results
        save_path: Path to save the figure
    """
    policies = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Task Processing Distribution', fontsize=16, fontweight='bold')

    x = np.arange(len(policies))
    width = 0.6

    local_ratios = [results[p]['local_ratio'] * 100 for p in policies]
    offload_ratios = [results[p]['offload_ratio'] * 100 for p in policies]

    # Stacked bar chart
    bars1 = ax.bar(x, local_ratios, width, label='Local Processing', color='#3498db')
    bars2 = ax.bar(x, offload_ratios, width, bottom=local_ratios, label='Offloaded to UAV', color='#e74c3c')

    ax.set_xlabel('Policy')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Local vs Offload Decision Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        local_pct = local_ratios[i]
        offload_pct = offload_ratios[i]

        if local_pct > 5:
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2,
                   f'{local_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        if offload_pct > 5:
            ax.text(bar2.get_x() + bar2.get_width()/2, local_pct + offload_pct/2,
                   f'{offload_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Offload ratio plot saved to: {save_path}")


def plot_network_topology(env, save_path: str = "network_topology.png"):
    """
    Plot the network topology showing device and UAV positions.

    Args:
        env: The UAVOffloadingEnv environment
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Network Topology', fontsize=16, fontweight='bold')

    # Plot ground devices
    device_x = env.device_positions[:, 0]
    device_y = env.device_positions[:, 1]
    ax.scatter(device_x, device_y, c='blue', s=100, marker='s', label='Ground Devices', zorder=2)

    # Label devices
    for i, (x, y) in enumerate(zip(device_x, device_y)):
        ax.annotate(f'D{i}', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Plot UAVs
    uav_x = [pos[0] for pos in env.uav_positions]
    uav_y = [pos[1] for pos in env.uav_positions]
    ax.scatter(uav_x, uav_y, c='red', s=300, marker='^', label='UAVs', zorder=3)

    # Label UAVs with altitude
    for i, pos in enumerate(env.uav_positions):
        ax.annotate(f'UAV{i}\n({pos[2]:.0f}m)', (pos[0], pos[1]),
                   textcoords="offset points", xytext=(10, 10), fontsize=9, fontweight='bold')

    # Draw communication ranges (simplified as circles)
    for i, pos in enumerate(env.uav_positions):
        circle = plt.Circle((pos[0], pos[1]), 200, color='red', fill=False,
                           linestyle='--', alpha=0.3, label='Coverage' if i == 0 else '')
        ax.add_patch(circle)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Device and UAV Positions\n{env.num_devices} Devices, {env.num_uavs} UAVs')
    ax.legend()
    ax.set_xlim(-50, env.area_size + 50)
    ax.set_ylim(-50, env.area_size + 50)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Network topology plot saved to: {save_path}")


def plot_reward_comparison_over_time(
    dqn_rewards: List[float],
    save_path: str = "reward_comparison.png"
):
    """
    Plot reward improvement during training compared to baseline rewards.

    Args:
        dqn_rewards: List of DQN episode rewards during training
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('DQN Learning Curve vs Baseline Performance', fontsize=16, fontweight='bold')

    episodes = range(1, len(dqn_rewards) + 1)

    # Calculate smoothed DQN rewards
    window = 20
    smoothed_rewards = []
    for i in range(len(dqn_rewards)):
        start = max(0, i - window // 2)
        end = min(len(dqn_rewards), i + window // 2)
        smoothed_rewards.append(np.mean(dqn_rewards[start:end]))

    # Plot DQN learning curve
    ax.plot(episodes, dqn_rewards, alpha=0.2, color='green', label='DQN (raw)')
    ax.plot(episodes, smoothed_rewards, color='green', linewidth=2, label='DQN (smoothed)')

    # Reference lines for where typical baselines perform
    # These are approximate values
    ax.axhline(y=np.mean(dqn_rewards[-50:]), color='green', linestyle='--',
               alpha=0.5, label=f'DQN Final Avg: {np.mean(dqn_rewards[-50:]):.1f}')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('DQN Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reward comparison plot saved to: {save_path}")


def create_all_plots(env, history: Dict, results: Dict, output_dir: str = "."):
    """
    Create all visualization plots.

    Args:
        env: The environment (for topology plot)
        history: Training history dictionary
        results: Policy comparison results dictionary
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    plot_training_history(history, f"{output_dir}/training_history.png")
    plot_policy_comparison(results, f"{output_dir}/policy_comparison.png")
    plot_offload_ratio(results, f"{output_dir}/offload_ratio.png")
    plot_network_topology(env, f"{output_dir}/network_topology.png")
    plot_reward_comparison_over_time(history['episode_rewards'], f"{output_dir}/reward_comparison.png")

    print("=" * 60)
    print("All plots saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Test with dummy data
    dummy_history = {
        'episode_rewards': list(np.random.randn(100).cumsum()),
        'episode_delays': list(np.random.uniform(0.1, 0.5, 100)),
        'episode_energies': list(np.random.uniform(0.01, 0.1, 100)),
        'losses': list(np.random.uniform(0, 1, 100) * np.exp(-np.arange(100) / 50)),
        'epsilons': list(1.0 * 0.99 ** np.arange(100))
    }

    dummy_results = {
        'DQN': {'avg_reward': -5, 'std_reward': 1, 'avg_delay': 0.2, 'std_delay': 0.05,
                'avg_energy': 0.05, 'std_energy': 0.01, 'local_ratio': 0.4, 'offload_ratio': 0.6},
        'All Local': {'avg_reward': -10, 'std_reward': 2, 'avg_delay': 0.4, 'std_delay': 0.1,
                      'avg_energy': 0.1, 'std_energy': 0.02, 'local_ratio': 1.0, 'offload_ratio': 0.0},
        'All Offload': {'avg_reward': -8, 'std_reward': 1.5, 'avg_delay': 0.25, 'std_delay': 0.08,
                        'avg_energy': 0.03, 'std_energy': 0.01, 'local_ratio': 0.0, 'offload_ratio': 1.0},
        'Random': {'avg_reward': -9, 'std_reward': 2, 'avg_delay': 0.32, 'std_delay': 0.1,
                   'avg_energy': 0.06, 'std_energy': 0.02, 'local_ratio': 0.5, 'offload_ratio': 0.5}
    }

    plot_training_history(dummy_history, "test_training.png")
    plot_policy_comparison(dummy_results, "test_comparison.png")
    plot_offload_ratio(dummy_results, "test_offload.png")
    print("Test plots created successfully!")
