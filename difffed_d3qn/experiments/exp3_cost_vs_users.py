"""Experiment 3: system cost vs number of ground users."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..train import TrainConfig, run_method
from .utils import out


METHODS = ["all-local", "all-uav", "d3qn", "fedavg-d3qn", "synther", "difffed-d3qn"]
USERS = [10, 20, 30, 40, 50]


def run(num_episodes: int = 50, seeds=(0,)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        y = []
        for n in USERS:
            cs = []
            for s in seeds:
                cfg = TrainConfig(num_episodes=num_episodes, seed=s, num_users=n)
                hist, _ = run_method(m, cfg)
                cs.append(np.mean(hist["cost"][-10:]))
            y.append(np.mean(cs))
        ax.plot(USERS, y, marker="o", label=m)
    ax.set_xlabel("Number of users")
    ax.set_ylabel("System Cost")
    ax.set_title("Scalability: Cost vs Users")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("cost_vs_users.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run()
