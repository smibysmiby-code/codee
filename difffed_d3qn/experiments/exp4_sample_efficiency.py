"""Experiment 4: final reward vs real-environment interaction budget."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..train import TrainConfig, run_method
from .utils import out


METHODS = ["d3qn", "synther", "difffed-d3qn"]
BUDGETS = [1000, 5000, 10000, 25000, 50000, 100000]


def run(seeds=(0,)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        means, stds = [], []
        for budget in BUDGETS:
            num_users = 50
            steps_per_ep = 100
            interactions_per_ep = num_users * steps_per_ep
            episodes = max(2, budget // interactions_per_ep)
            vals = []
            for s in seeds:
                cfg = TrainConfig(num_episodes=episodes, seed=s)
                hist, _ = run_method(m, cfg)
                vals.append(np.mean(hist["reward"][-5:]))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(BUDGETS, means, yerr=stds, marker="o", label=m, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Real interactions")
    ax.set_ylabel("Final episode reward")
    ax.set_title("Sample Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out("sample_efficiency.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run()
