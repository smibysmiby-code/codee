"""Conditional DDPM for generating synthetic D3QN experiences.

Generates tuples (state, action, reward, next_state) conditioned on a
network-context vector. Discrete action is treated as a one-hot embedded
into the continuous diffusion space and discretized at sampling time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time / condition embeddings
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class DenoisingMLP(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_embed = ConditionEmbedding(cond_dim, hidden_dim)
        in_dim = x_dim + time_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        ce = self.cond_embed(c)
        h = torch.cat([x, te, ce], dim=-1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 128
    lr: float = 1e-3
    device: str = "cpu"
    cond_dim: int = 4
    state_dim: int = 6
    action_dim: int = 2


def _make_schedule(T: int, beta_start: float, beta_end: float, device: torch.device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cum


class ConditionalDiffusion:
    """DDPM that learns p(experience | network_context).

    The experience vector layout is:
        [ state (state_dim) | action_onehot (action_dim) | reward (1) | next_state (state_dim) ]
    """

    def __init__(self, config: DiffusionConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.x_dim = config.state_dim * 2 + config.action_dim + 1
        self.model = DenoisingMLP(self.x_dim, config.cond_dim, config.hidden_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.betas, self.alphas, self.alphas_cum = _make_schedule(
            config.T, config.beta_start, config.beta_end, self.device
        )

        # Running stats for state/reward normalization (set at training time)
        self._mean = torch.zeros(self.x_dim, device=self.device)
        self._std = torch.ones(self.x_dim, device=self.device)

    # ------------------------------------------------------------------
    def _encode(self, s, a, r, ns) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        a_oh = F.one_hot(a, num_classes=self.cfg.action_dim).float() * 2.0 - 1.0
        return torch.cat([s, a_oh, r, ns], dim=-1)

    def _decode(self, x: torch.Tensor):
        sd = self.cfg.state_dim
        ad = self.cfg.action_dim
        s = x[:, :sd]
        a_logits = x[:, sd : sd + ad]
        r = x[:, sd + ad : sd + ad + 1].squeeze(-1)
        ns = x[:, sd + ad + 1 :]
        a = a_logits.argmax(dim=-1)
        return s, a, r, ns

    # ------------------------------------------------------------------
    def fit_normalization(self, experiences):
        """experiences: list of (s, a, r, ns) tuples in raw scale."""
        if not experiences:
            return
        s, a, r, ns = zip(*experiences)
        x = self._encode(np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns))
        self._mean = x.mean(dim=0)
        self._std = x.std(dim=0).clamp(min=1e-3)

    def _norm(self, x):
        return (x - self._mean) / self._std

    def _denorm(self, x):
        return x * self._std + self._mean

    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cum[t].unsqueeze(-1)
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def train_step(self, x0_raw: torch.Tensor, cond: torch.Tensor) -> float:
        x0 = self._norm(x0_raw)
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t, cond)
        loss = F.mse_loss(pred, noise)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()
        return float(loss.item())

    def fit(self, experiences, contexts, epochs: int = 100, batch_size: int = 64,
            verbose: bool = False) -> list:
        if len(experiences) < batch_size:
            return []
        self.fit_normalization(experiences)
        s, a, r, ns = zip(*experiences)
        x0 = self._encode(np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns))
        c = torch.as_tensor(np.stack(contexts), dtype=torch.float32, device=self.device)
        N = x0.shape[0]
        losses = []
        for ep in range(epochs):
            idx = torch.randperm(N, device=self.device)
            ep_losses = []
            for i in range(0, N, batch_size):
                j = idx[i:i + batch_size]
                ep_losses.append(self.train_step(x0[j], c[j]))
            losses.append(float(np.mean(ep_losses)))
            if verbose and (ep + 1) % 20 == 0:
                print(f"  diffusion epoch {ep+1}/{epochs}  loss={losses[-1]:.4f}")
        return losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, cond: torch.Tensor) -> torch.Tensor:
        """Reverse-process sampling. cond: (n, cond_dim)."""
        x = torch.randn(n, self.x_dim, device=self.device)
        for t in reversed(range(self.cfg.T)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, tt, cond)
            alpha = self.alphas[t]
            alpha_cum = self.alphas_cum[t]
            beta = self.betas[t]
            mean = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_cum).sqrt()) * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
        return self._denorm(x)

    def generate(self, n: int, context_vec: np.ndarray):
        c = torch.as_tensor(np.tile(context_vec, (n, 1)), dtype=torch.float32, device=self.device)
        x = self.sample(n, c)
        return self._decode(x)


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def filter_by_q(agent, s, a, r, ns, drop_frac: float = 0.10) -> Tuple[np.ndarray, ...]:
    """Discard the bottom drop_frac of generated experiences by anomalous Q-score.

    Score = |Q(s, a) - (r + gamma * max_a' Q(ns, a'))|. Drop largest residuals
    (most likely synthetic outliers).
    """
    if len(s) == 0:
        return s, a, r, ns
    q_s = agent.q_values(s)
    q_ns = agent.q_values(ns)
    chosen = q_s[np.arange(len(a)), a]
    target = r + agent.cfg.gamma * q_ns.max(axis=-1)
    score = np.abs(chosen - target)
    cutoff = np.quantile(score, 1.0 - drop_frac)
    keep = score <= cutoff
    return s[keep], a[keep], r[keep], ns[keep]
