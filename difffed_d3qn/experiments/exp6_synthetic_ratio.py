"""Experiment 6: synthetic-data ratio sweep (KEY EXPERIMENT)."""

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..d3qn_agent import D3QNAgent, D3QNConfig
from ..train import TrainConfig, make_env, train_with_diffusion
from .utils import out


RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run(num_episodes: int = 60, seeds=(0, 1, 2)):
    rew, delay, energy, cost = [], [], [], []
    rew_std = []
    for r in RATIOS:
        rs, ds, es, cs = [], [], [], []
        for s in seeds:
            np.random.seed(s)
            torch.manual_seed(s)
            cfg = TrainConfig(num_episodes=num_episodes, seed=s, synth_ratio=r)
            env = make_env(cfg)
            ag = D3QNAgent(D3QNConfig(device=cfg.device))
            hist = train_with_diffusion(ag, env, cfg, conditional=True,
                                         use_federated=True, num_clients=cfg.num_clients)
            rs.append(np.mean(hist["reward"][-10:]))
            ds.append(np.mean(hist["delay"][-10:]))
            es.append(np.mean(hist["energy"][-10:]))
            cs.append(np.mean(hist["cost"][-10:]))
        rew.append(np.mean(rs)); rew_std.append(np.std(rs))
        delay.append(np.mean(ds))
        energy.append(np.mean(es))
        cost.append(np.mean(cs))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    rew = np.array(rew); rew_std = np.array(rew_std)
    axes[0, 0].errorbar(RATIOS, rew, yerr=rew_std, marker="o")
    axes[0, 0].set_title("(a) Reward vs Synthetic Ratio")
    axes[0, 0].set_xlabel("Synthetic ratio"); axes[0, 0].set_ylabel("Reward")
    best = int(np.argmax(rew))
    axes[0, 0].annotate(f"optimal={RATIOS[best]:.1f}",
                        xy=(RATIOS[best], rew[best]),
                        xytext=(RATIOS[best], rew[best] + 0.5 * max(1e-3, rew.std())),
                        arrowprops=dict(arrowstyle="->", color="red"), color="red")

    axes[0, 1].plot(RATIOS, delay, marker="s", color="orange")
    axes[0, 1].set_title("(b) Delay vs Synthetic Ratio")
    axes[0, 1].set_xlabel("Synthetic ratio"); axes[0, 1].set_ylabel("Delay (s)")

    axes[1, 0].plot(RATIOS, energy, marker="^", color="green")
    axes[1, 0].set_title("(c) Energy vs Synthetic Ratio")
    axes[1, 0].set_xlabel("Synthetic ratio"); axes[1, 0].set_ylabel("Energy (J)")

    axes[1, 1].plot(RATIOS, cost, marker="d", color="red")
    axes[1, 1].set_title("(d) System Cost vs Synthetic Ratio")
    axes[1, 1].set_xlabel("Synthetic ratio"); axes[1, 1].set_ylabel("Cost")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("synthetic_ratio.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run()
