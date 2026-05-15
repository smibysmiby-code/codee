"""Experiment 8: effect of federated aggregation interval."""

from copy import deepcopy

import numpy as np
import torch

from ..d3qn_agent import D3QNAgent, D3QNConfig
from ..train import TrainConfig, make_env, train_with_diffusion
from .utils import plot_with_ci


INTERVALS = [1, 5, 10, 20]


def run(num_episodes: int = 60, seeds=(0, 1, 2)):
    curves = {f"agg={k}": [] for k in INTERVALS}
    for s in seeds:
        for k in INTERVALS:
            np.random.seed(s); torch.manual_seed(s)
            cfg = TrainConfig(num_episodes=num_episodes, seed=s, fed_every=k)
            env = make_env(cfg)
            ag = D3QNAgent(D3QNConfig(device=cfg.device))
            hist = train_with_diffusion(ag, env, cfg, conditional=True,
                                         use_federated=True, num_clients=cfg.num_clients,
                                         agg_every=k)
            curves[f"agg={k}"].append(hist["reward"])
    plot_with_ci(curves, "Episode", "Episode Reward",
                 "Aggregation-Interval Sweep", "agg_frequency.pdf")


if __name__ == "__main__":
    run()
