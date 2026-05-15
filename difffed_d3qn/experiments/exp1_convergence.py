"""Experiment 1: convergence comparison across all methods."""

from copy import deepcopy

from ..train import TrainConfig, run_method
from .utils import plot_with_ci


METHODS = [
    "random", "all-local", "all-uav", "dqn", "dueling-dqn",
    "d3qn", "fedavg-d3qn", "synther", "difffed-d3qn",
]


def run(num_episodes: int = 80, seeds=(0, 1, 2)):
    curves = {m: [] for m in METHODS}
    for s in seeds:
        for m in METHODS:
            cfg = TrainConfig(num_episodes=num_episodes, seed=s)
            hist, _ = run_method(m, cfg)
            curves[m].append(hist["reward"])
    plot_with_ci(curves, "Episode", "Episode Reward",
                 "Convergence Comparison", "convergence.pdf")


if __name__ == "__main__":
    run()
