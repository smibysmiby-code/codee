"""
UAV Task Offloading with DQN
============================

A simple educational implementation of Deep Q-Network (DQN) for
task offloading in UAV-assisted edge computing networks.

Modules:
--------
- environment: Simulation environment for UAV task offloading
- dqn_agent: DQN agent implementation with neural network
- train: Training loop and evaluation functions
- visualize: Plotting and visualization utilities
- main: Main entry point with full explanations

Usage:
------
    # Interactive mode with explanations
    python main.py

    # Quick mode (no prompts)
    python main.py --quick

Author: Educational Implementation
"""

from .environment import UAVOffloadingEnv, Task, GroundDevice, UAV
from .dqn_agent import DQNAgent, AllLocalBaseline, AllOffloadBaseline, RandomBaseline
from .train import train_dqn, evaluate_policy, compare_policies

__version__ = "1.0.0"
__all__ = [
    'UAVOffloadingEnv',
    'Task',
    'GroundDevice',
    'UAV',
    'DQNAgent',
    'AllLocalBaseline',
    'AllOffloadBaseline',
    'RandomBaseline',
    'train_dqn',
    'evaluate_policy',
    'compare_policies'
]
