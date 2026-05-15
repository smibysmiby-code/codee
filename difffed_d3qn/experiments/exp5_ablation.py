"""Experiment 5: ablation study on the synthetic-data and federated components."""

from copy import deepcopy

import numpy as np
import torch

from ..d3qn_agent import D3QNAgent, D3QNConfig
from ..train import TrainConfig, make_env, train_simple, train_with_diffusion
from .utils import bar_chart


VARIANTS = [
    "D3QN only",
    "D3QN + Random",
    "D3QN + Uncond. Diffusion",
    "D3QN + Cond. Diffusion",
    "DiffFed-D3QN (FULL)",
]


def _run_one(variant: str, cfg: TrainConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_env(cfg)
    ag = D3QNAgent(D3QNConfig(device=cfg.device))
    if variant == "D3QN only":
        hist = train_simple(ag, env, cfg)
    elif variant == "D3QN + Random":
        c = deepcopy(cfg)
        hist = train_with_diffusion(ag, env, c, conditional=False, use_federated=False,
                                     random_gen=True)
    elif variant == "D3QN + Uncond. Diffusion":
        hist = train_with_diffusion(ag, env, cfg, conditional=False, use_federated=False)
    elif variant == "D3QN + Cond. Diffusion":
        hist = train_with_diffusion(ag, env, cfg, conditional=True, use_federated=False)
    else:
        hist = train_with_diffusion(ag, env, cfg, conditional=True,
                                     use_federated=True, num_clients=cfg.num_clients)
    return float(np.mean(hist["reward"][-10:]))


def run(num_episodes: int = 60, seeds=(0, 1, 2)):
    means, errs = {}, {}
    for v in VARIANTS:
        vals = []
        for s in seeds:
            cfg = TrainConfig(num_episodes=num_episodes, seed=s)
            vals.append(_run_one(v, cfg))
        means[v] = float(np.mean(vals))
        errs[v] = float(np.std(vals))
    bar_chart(means, "Final Episode Reward", "Ablation Study",
              "ablation.pdf", errs=errs)


if __name__ == "__main__":
    run()
