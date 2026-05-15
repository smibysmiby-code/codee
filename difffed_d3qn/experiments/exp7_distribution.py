"""Experiment 7: real vs synthetic experience distribution comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..d3qn_agent import D3QNAgent, D3QNConfig
from ..diffusion import ConditionalDiffusion, DiffusionConfig
from ..train import TrainConfig, make_env
from .utils import out


def _collect_real(env, agent, n_steps: int, contexts):
    s = env.reset()
    transitions = []
    while len(transitions) < n_steps:
        a = agent.act(s, greedy=False)
        ns, r, done, info = env.step(a)
        for i in range(env.num_users):
            transitions.append((s[i].copy(), int(a[i]), float(r), ns[i].copy()))
            contexts.append(env.network_context.copy())
            if len(transitions) >= n_steps:
                break
        s = ns
        if done:
            s = env.reset()
    return transitions


def run(seed: int = 0, n_real: int = 2000, n_synth: int = 2000):
    np.random.seed(seed); torch.manual_seed(seed)
    cfg = TrainConfig(num_episodes=1, seed=seed)
    env = make_env(cfg)
    agent = D3QNAgent(D3QNConfig(device=cfg.device))
    contexts = []
    real = _collect_real(env, agent, n_real, contexts)

    diffusion = ConditionalDiffusion(DiffusionConfig(device=cfg.device))
    diffusion.fit(real, contexts, epochs=80, batch_size=64)
    ctx_vec = env.network_context
    s_g, a_g, r_g, ns_g = diffusion.generate(n_synth, ctx_vec)
    s_g = s_g.cpu().numpy()

    real_s = np.array([t[0] for t in real])

    # 2D PCA visualization on states
    p = PCA(n_components=2)
    z_real = p.fit_transform(real_s)
    z_synth = p.transform(s_g)

    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(z_real[:, 0], z_real[:, 1], s=4, alpha=0.4, label="Real", c="#1f77b4")
    ax1.scatter(z_synth[:, 0], z_synth[:, 1], s=4, alpha=0.4, label="Synthetic", c="#d62728")
    ax1.set_title("PCA projection")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    labels = ["task_MB", "cycles_G", "uav_load", "user_GHz", "dist_m", "snr_dB"]
    for i in range(6):
        ax = fig.add_subplot(2, 4, i + 3 if i >= 2 else i + 2)
    # Re-layout cleanly: 2x3 grid + first PCA = simpler — redo
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes[0, 0].scatter(z_real[:, 0], z_real[:, 1], s=4, alpha=0.4, label="Real")
    axes[0, 0].scatter(z_synth[:, 0], z_synth[:, 1], s=4, alpha=0.4, label="Synth")
    axes[0, 0].legend(); axes[0, 0].set_title("PCA"); axes[0, 0].grid(True, alpha=0.3)
    for i, lab in enumerate(labels):
        ax = axes[(i + 1) // 4, (i + 1) % 4]
        ax.hist(real_s[:, i], bins=40, alpha=0.5, label="Real", density=True)
        ax.hist(s_g[:, i], bins=40, alpha=0.5, label="Synth", density=True)
        ax.set_title(lab); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[1, 3].axis("off")
    fig.tight_layout()
    fig.savefig(out("distribution.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run()
