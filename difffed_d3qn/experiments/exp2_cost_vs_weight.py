"""Experiment 2: system cost as a function of the delay weight omega1."""

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..train import TrainConfig, run_method
from .utils import out


METHODS = ["random", "all-local", "all-uav", "d3qn", "fedavg-d3qn", "synther", "difffed-d3qn"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run(num_episodes: int = 50, seeds=(0,)):
    results = defaultdict(dict)  # method -> {omega: cost}
    for w in WEIGHTS:
        for m in METHODS:
            cs = []
            for s in seeds:
                cfg = TrainConfig(num_episodes=num_episodes, seed=s,
                                  omega_delay=w, omega_energy=1.0 - w)
                hist, _ = run_method(m, cfg)
                cs.append(np.mean(hist["cost"][-10:]))
            results[m][w] = float(np.mean(cs))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(WEIGHTS))
    width = 0.11
    for i, m in enumerate(METHODS):
        y = [results[m][w] for w in WEIGHTS]
        ax.bar(x + (i - len(METHODS) / 2) * width, y, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.2f}" for w in WEIGHTS])
    ax.set_xlabel(r"$\omega_1$ (delay weight)")
    ax.set_ylabel("System Cost")
    ax.set_title("System Cost vs Delay Weight")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("cost_vs_weight.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run()
