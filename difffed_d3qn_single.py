"""DiffFed-D3QN: Diffusion-Augmented Federated Deep RL for UAV-MEC Offloading.

Single-file implementation containing:
  - UAV-MEC Gym environment
  - D3QN agent (Dueling + Double DQN + soft target update)
  - Conditional DDPM with Q-value quality filter
  - FedAvg aggregator
  - Baselines (Random, All-Local, All-UAV, DQN, Dueling-DQN, SynthER)
  - 8 experiments (convergence, cost-vs-weight, cost-vs-users,
                   sample efficiency, ablation, synthetic ratio,
                   distribution, aggregation frequency)
  - Top-level CLI driver

Install:
    pip install numpy torch matplotlib scikit-learn

Quick smoke test:
    python difffed_d3qn.py --quick

Run all experiments at full settings:
    python difffed_d3qn.py

Run specific experiments:
    python difffed_d3qn.py 1 6 7

Figures are written to ./results/ as PDFs.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. ENVIRONMENT
# ============================================================================

UAV_ALTITUDE_M = 100.0
COVERAGE_RADIUS_M = 50.0

BANDWIDTH_HZ = 1e6
NOISE_POWER_DBM = -100.0
TX_POWER_DBM = 23.0
PATH_LOSS_REF_DB = 30.0
PATH_LOSS_EXP = 2.5

UAV_CPU_HZ = 5e9
USER_CPU_LOW = 0.5e9
USER_CPU_HIGH = 1.5e9
KAPPA = 1e-27

TASK_SIZE_LOW_MB = 0.1
TASK_SIZE_HIGH_MB = 1.0
CPU_CYCLES_PER_BIT = 200.0


def _dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


@dataclass
class EnvConfig:
    num_users: int = 50
    omega_delay: float = 0.5
    omega_energy: float = 0.5
    max_steps: int = 100
    seed: Optional[int] = None


class UAVMECEnv:
    """UAV-MEC environment.

    Observation: shape (num_users, 6)
        [task_size_MB, cpu_cycles_G, uav_load, user_cpu_GHz, distance_m, snr_dB]
    Action: shape (num_users,), entries in {0=local, 1=offload}.
    """

    state_dim = 6

    def __init__(self, config: Optional[EnvConfig] = None, **overrides):
        self.config = config or EnvConfig()
        for k, v in overrides.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.rng = np.random.default_rng(self.config.seed)
        self._step = 0
        self._uav_load = 0.0
        self._positions = None
        self._user_cpu = None
        self._tasks = None
        self._reset_internal()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def _reset_internal(self):
        n = self.config.num_users
        angles = self.rng.uniform(0, 2 * math.pi, n)
        radii = COVERAGE_RADIUS_M * np.sqrt(self.rng.uniform(0, 1, n))
        self._positions = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        self._user_cpu = self.rng.uniform(USER_CPU_LOW, USER_CPU_HIGH, n)
        self._uav_load = 0.0
        self._step = 0
        self._sample_tasks()

    def _sample_tasks(self):
        n = self.config.num_users
        sizes_mb = self.rng.uniform(TASK_SIZE_LOW_MB, TASK_SIZE_HIGH_MB, n)
        size_bits = sizes_mb * 8 * 1e6
        cycles = size_bits * CPU_CYCLES_PER_BIT
        self._tasks = (sizes_mb, cycles)

    def _obs(self) -> np.ndarray:
        sizes_mb, cycles = self._tasks
        dist = np.sqrt((self._positions ** 2).sum(axis=1) + UAV_ALTITUDE_M ** 2)
        path_loss_db = PATH_LOSS_REF_DB + 10.0 * PATH_LOSS_EXP * np.log10(np.maximum(dist, 1.0))
        snr_db = TX_POWER_DBM - path_loss_db - NOISE_POWER_DBM
        load = np.full(self.config.num_users, self._uav_load)
        return np.stack(
            [sizes_mb, cycles / 1e9, load, self._user_cpu / 1e9, dist, snr_db],
            axis=1,
        ).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        return self._obs()

    def _compute_costs(self, actions):
        sizes_mb, cycles = self._tasks
        size_bits = sizes_mb * 8 * 1e6
        dist = np.sqrt((self._positions ** 2).sum(axis=1) + UAV_ALTITUDE_M ** 2)
        path_loss_db = PATH_LOSS_REF_DB + 10.0 * PATH_LOSS_EXP * np.log10(np.maximum(dist, 1.0))
        rx_w = _dbm_to_w(TX_POWER_DBM - path_loss_db)
        noise_w = _dbm_to_w(NOISE_POWER_DBM)
        rate = BANDWIDTH_HZ * np.log2(1.0 + rx_w / noise_w)

        local_delay = cycles / np.maximum(self._user_cpu, 1.0)
        local_energy = KAPPA * cycles * (self._user_cpu ** 2)

        comm_delay = size_bits / np.maximum(rate, 1.0)
        offload_mask = (actions == 1)
        n_off = max(int(offload_mask.sum()), 1)
        per_uav_cpu = UAV_CPU_HZ / n_off
        uav_delay = cycles / per_uav_cpu
        tx_energy = _dbm_to_w(TX_POWER_DBM) * comm_delay
        offload_delay = comm_delay + uav_delay
        offload_energy = tx_energy

        delay = np.where(offload_mask, offload_delay, local_delay)
        energy = np.where(offload_mask, offload_energy, local_energy)
        self._uav_load = float(offload_mask.mean())
        return delay, energy

    def step(self, actions):
        actions = np.asarray(actions).astype(np.int64).reshape(-1)
        delay, energy = self._compute_costs(actions)
        cfg = self.config
        norm_delay = delay / 5.0
        norm_energy = energy / 5.0
        cost = cfg.omega_delay * norm_delay + cfg.omega_energy * norm_energy
        reward = float(-cost.mean())
        self._step += 1
        self._sample_tasks()
        done = self._step >= cfg.max_steps
        info = {
            "delay": float(delay.mean()),
            "energy": float(energy.mean()),
            "system_cost": float(cost.mean()),
            "offload_ratio": float((actions == 1).mean()),
        }
        return self._obs(), reward, done, info

    @property
    def num_users(self) -> int:
        return self.config.num_users

    @property
    def network_context(self) -> np.ndarray:
        sizes_mb, _ = self._tasks
        return np.array(
            [
                self.config.num_users / 50.0,
                float(sizes_mb.mean()) / TASK_SIZE_HIGH_MB,
                self._uav_load,
                self._step / self.config.max_steps,
            ],
            dtype=np.float32,
        )


# ============================================================================
# 2. D3QN AGENT
# ============================================================================

@dataclass
class D3QNConfig:
    state_dim: int = 6
    action_dim: int = 2
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 3000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    tau: float = 0.005
    device: str = "cpu"


class DuelingQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((np.asarray(s, dtype=np.float32), int(a), float(r),
                         np.asarray(ns, dtype=np.float32), bool(d)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class D3QNAgent:
    def __init__(self, config: D3QNConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.online = DuelingQNet(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target = DuelingQNet(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.eps = config.eps_start

    def act(self, states, greedy=False):
        if (not greedy) and random.random() < self.eps:
            return np.random.randint(0, self.cfg.action_dim, size=states.shape[0])
        with torch.no_grad():
            s = torch.from_numpy(states.astype(np.float32)).to(self.device)
            return self.online(s).argmax(dim=-1).cpu().numpy()

    def decay_epsilon(self):
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def q_values(self, states):
        with torch.no_grad():
            s = torch.from_numpy(states.astype(np.float32)).to(self.device)
            return self.online(s).cpu().numpy()

    def soft_update(self):
        tau = self.cfg.tau
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * op.data)

    def update_from_batch(self, s, a, r, ns, d):
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).to(self.device)
        ns = torch.from_numpy(ns).to(self.device)
        d = torch.from_numpy(d).to(self.device)

        q = self.online(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_a = self.online(ns).argmax(dim=-1, keepdim=True)
            next_q = self.target(ns).gather(1, next_a).squeeze(-1)
            target = r + self.cfg.gamma * (1.0 - d) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        return self.update_from_batch(*self.buffer.sample(self.cfg.batch_size))

    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict()}

    def load_state_dict(self, sd):
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["target"])


# ============================================================================
# 3. CONDITIONAL DIFFUSION MODEL
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, c):
        return self.net(c)


class DenoisingMLP(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_embed = ConditionEmbedding(cond_dim, hidden_dim)
        in_dim = x_dim + time_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.time_embed(t), self.cond_embed(c)], dim=-1))


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


def _make_schedule(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


class ConditionalDiffusion:
    """DDPM that learns p(experience | network_context).

    Experience layout: [s | a_onehot | r | next_s].
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
        self._mean = torch.zeros(self.x_dim, device=self.device)
        self._std = torch.ones(self.x_dim, device=self.device)

    def _encode(self, s, a, r, ns):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        a_oh = F.one_hot(a, num_classes=self.cfg.action_dim).float() * 2.0 - 1.0
        return torch.cat([s, a_oh, r, ns], dim=-1)

    def _decode(self, x):
        sd = self.cfg.state_dim
        ad = self.cfg.action_dim
        s = x[:, :sd]
        a_logits = x[:, sd:sd + ad]
        r = x[:, sd + ad:sd + ad + 1].squeeze(-1)
        ns = x[:, sd + ad + 1:]
        return s, a_logits.argmax(dim=-1), r, ns

    def fit_normalization(self, experiences):
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

    def q_sample(self, x0, t, noise):
        ac = self.alphas_cum[t].unsqueeze(-1)
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def train_step(self, x0_raw, cond):
        x0 = self._norm(x0_raw)
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        pred = self.model(self.q_sample(x0, t, noise), t, cond)
        loss = F.mse_loss(pred, noise)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()
        return float(loss.item())

    def fit(self, experiences, contexts, epochs=100, batch_size=64, verbose=False):
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
                print(f"  diffusion epoch {ep + 1}/{epochs} loss={losses[-1]:.4f}")
        return losses

    @torch.no_grad()
    def sample(self, n, cond):
        x = torch.randn(n, self.x_dim, device=self.device)
        for t in reversed(range(self.cfg.T)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, tt, cond)
            alpha = self.alphas[t]
            alpha_cum = self.alphas_cum[t]
            beta = self.betas[t]
            mean = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_cum).sqrt()) * pred_noise)
            if t > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return self._denorm(x)

    def generate(self, n, context_vec):
        c = torch.as_tensor(np.tile(context_vec, (n, 1)), dtype=torch.float32, device=self.device)
        return self._decode(self.sample(n, c))


def filter_by_q(agent, s, a, r, ns, drop_frac=0.10):
    """Drop the worst `drop_frac` of generated experiences by |Q(s,a) - target|."""
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


# ============================================================================
# 4. FEDERATED AGGREGATION
# ============================================================================

def fedavg_weights(state_dicts):
    avg = deepcopy(state_dicts[0])
    for key in avg:
        for pkey in avg[key]:
            stacked = torch.stack([sd[key][pkey].float() for sd in state_dicts], dim=0)
            avg[key][pkey] = stacked.mean(dim=0)
    return avg


def aggregate(agents):
    agents = list(agents)
    sd = fedavg_weights([a.state_dict() for a in agents])
    for a in agents:
        a.load_state_dict(sd)
    return sd


# ============================================================================
# 5. BASELINES
# ============================================================================

class RandomPolicy:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def act(self, states, greedy=True):
        return np.random.randint(0, self.action_dim, size=states.shape[0])

    def update(self, *a, **kw): return 0.0
    def decay_epsilon(self): pass


class AllLocalPolicy:
    def act(self, states, greedy=True):
        return np.zeros(states.shape[0], dtype=np.int64)
    def update(self, *a, **kw): return 0.0
    def decay_epsilon(self): pass


class AllUAVPolicy:
    def act(self, states, greedy=True):
        return np.ones(states.shape[0], dtype=np.int64)
    def update(self, *a, **kw): return 0.0
    def decay_epsilon(self): pass


class _PlainQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(D3QNAgent):
    """Vanilla DQN: plain MLP, no Double Q."""

    def __init__(self, config: D3QNConfig):
        super().__init__(config)
        self.online = _PlainQNet(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target = _PlainQNet(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=config.lr)

    def update_from_batch(self, s, a, r, ns, d):
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).to(self.device)
        ns = torch.from_numpy(ns).to(self.device)
        d = torch.from_numpy(d).to(self.device)
        q = self.online(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=-1).values
            target = r + self.cfg.gamma * (1.0 - d) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())


class DuelingDQNAgent(D3QNAgent):
    """Dueling architecture, single-net bootstrap (no Double Q)."""

    def update_from_batch(self, s, a, r, ns, d):
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).to(self.device)
        ns = torch.from_numpy(ns).to(self.device)
        d = torch.from_numpy(d).to(self.device)
        q = self.online(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=-1).values
            target = r + self.cfg.gamma * (1.0 - d) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())


# ============================================================================
# 6. TRAINING LOOPS
# ============================================================================

@dataclass
class TrainConfig:
    num_episodes: int = 200
    max_steps: int = 100
    num_users: int = 50
    omega_delay: float = 0.5
    omega_energy: float = 0.5
    warmup_steps: int = 500
    diffusion_train_epochs: int = 60
    diffusion_regen_every: int = 20
    synth_ratio: float = 0.3
    num_clients: int = 5
    fed_every: int = 5
    seed: int = 0
    device: str = "cpu"
    quality_filter: bool = True


def _eval_episode(env, policy, max_steps):
    s = env.reset()
    total_r = 0.0
    delays, energies, costs = [], [], []
    for _ in range(max_steps):
        a = policy.act(s, greedy=True) if hasattr(policy, "act") else policy(s)
        s, r, done, info = env.step(a)
        total_r += r
        delays.append(info["delay"])
        energies.append(info["energy"])
        costs.append(info["system_cost"])
        if done:
            break
    return {
        "reward": total_r,
        "delay": float(np.mean(delays)),
        "energy": float(np.mean(energies)),
        "system_cost": float(np.mean(costs)),
    }


def _rollout_into_buffer(env, agent, max_steps, ctx_collector=None):
    s = env.reset()
    total_r = 0.0
    transitions = []
    for _ in range(max_steps):
        a = agent.act(s, greedy=False)
        ns, r, done, info = env.step(a)
        for i in range(env.num_users):
            agent.buffer.push(s[i], int(a[i]), float(r), ns[i], bool(done))
            transitions.append((s[i].copy(), int(a[i]), float(r), ns[i].copy()))
            if ctx_collector is not None:
                ctx_collector.append(env.network_context.copy())
        total_r += r
        s = ns
        if done:
            break
    return total_r, transitions


def train_simple(agent, env, cfg):
    history = {"reward": [], "delay": [], "energy": [], "cost": []}
    for _ in range(cfg.num_episodes):
        _rollout_into_buffer(env, agent, cfg.max_steps)
        for _ in range(cfg.max_steps):
            agent.update()
        agent.decay_epsilon()
        ev = _eval_episode(env, agent, cfg.max_steps)
        history["reward"].append(ev["reward"])
        history["delay"].append(ev["delay"])
        history["energy"].append(ev["energy"])
        history["cost"].append(ev["system_cost"])
    return history


def _make_diffusion(cfg, conditional=True):
    return ConditionalDiffusion(DiffusionConfig(
        T=100, hidden_dim=128, device=cfg.device,
        cond_dim=4 if conditional else 1, state_dim=6, action_dim=2,
    ))


def _augmented_batch(agent, synth_pool, ratio, batch_size):
    n_synth = int(round(batch_size * ratio)) if synth_pool is not None and len(synth_pool[0]) > 0 else 0
    n_real = batch_size - n_synth
    real = agent.buffer.sample(max(n_real, 1))
    rs, ra, rr, rns, rd = real
    if n_synth > 0:
        ss, sa, sr, sns = synth_pool
        idx = np.random.randint(0, len(ss), size=n_synth)
        rs = np.concatenate([rs[:n_real], ss[idx]], axis=0)
        ra = np.concatenate([ra[:n_real], sa[idx]], axis=0)
        rr = np.concatenate([rr[:n_real], sr[idx]], axis=0)
        rns = np.concatenate([rns[:n_real], sns[idx]], axis=0)
        rd = np.concatenate([rd[:n_real], np.zeros(n_synth, dtype=np.float32)], axis=0)
    return rs, ra, rr, rns, rd


def _clone_agent(agent):
    new = D3QNAgent(deepcopy(agent.cfg))
    new.load_state_dict(deepcopy(agent.state_dict()))
    new.eps = agent.eps
    return new


def train_with_diffusion(agent, env, cfg, conditional=True, use_federated=False,
                         num_clients=1, agg_every=None, random_gen=False):
    if use_federated and num_clients > 1:
        clients = [agent] + [_clone_agent(agent) for _ in range(num_clients - 1)]
        envs = [env] + [UAVMECEnv(EnvConfig(
            num_users=cfg.num_users, omega_delay=cfg.omega_delay,
            omega_energy=cfg.omega_energy, max_steps=cfg.max_steps,
            seed=cfg.seed + 17 * i)) for i in range(1, num_clients)]
    else:
        clients = [agent]
        envs = [env]

    diffusion = _make_diffusion(cfg, conditional=conditional)
    synth_pool = None
    contexts_seen, experiences_seen = [], []
    history = {"reward": [], "delay": [], "energy": [], "cost": []}
    agg_interval = agg_every if agg_every is not None else cfg.fed_every

    for ep in range(cfg.num_episodes):
        for ag, en in zip(clients, envs):
            _, transitions = _rollout_into_buffer(en, ag, cfg.max_steps,
                                                  ctx_collector=contexts_seen)
            experiences_seen.extend(transitions)

        if not random_gen and len(experiences_seen) >= cfg.warmup_steps and ep % cfg.diffusion_regen_every == 0:
            ctxs = contexts_seen if conditional else [np.zeros(1, dtype=np.float32) for _ in experiences_seen]
            diffusion.fit(experiences_seen, ctxs, epochs=cfg.diffusion_train_epochs, batch_size=64)
            n_gen = max(256, cfg.max_steps * 4)
            ctx_vec = envs[0].network_context if conditional else np.zeros(1, dtype=np.float32)
            s_g, a_g, r_g, ns_g = diffusion.generate(n_gen, ctx_vec)
            s_g, a_g, r_g, ns_g = (s_g.cpu().numpy(), a_g.cpu().numpy(),
                                   r_g.cpu().numpy(), ns_g.cpu().numpy())
            if cfg.quality_filter:
                s_g, a_g, r_g, ns_g = filter_by_q(clients[0], s_g, a_g, r_g, ns_g, drop_frac=0.1)
            synth_pool = (s_g, a_g, r_g, ns_g)
        elif random_gen and ep == 0 and len(experiences_seen) >= cfg.warmup_steps:
            s_arr = np.array([t[0] for t in experiences_seen])
            a_arr = np.array([t[1] for t in experiences_seen])
            r_arr = np.array([t[2] for t in experiences_seen])
            ns_arr = np.array([t[3] for t in experiences_seen])
            idx = np.random.randint(0, len(s_arr), 1024)
            synth_pool = (
                s_arr[idx] + np.random.randn(*s_arr[idx].shape) * 0.1,
                a_arr[idx],
                r_arr[idx] + np.random.randn(len(idx)) * 0.05,
                ns_arr[idx] + np.random.randn(*ns_arr[idx].shape) * 0.1,
            )

        for ag in clients:
            for _ in range(cfg.max_steps):
                if synth_pool is None or cfg.synth_ratio <= 0.0:
                    ag.update()
                else:
                    if len(ag.buffer) < ag.cfg.batch_size:
                        continue
                    batch = _augmented_batch(ag, synth_pool, cfg.synth_ratio, ag.cfg.batch_size)
                    ag.update_from_batch(*batch)
            ag.decay_epsilon()

        if use_federated and num_clients > 1 and (ep + 1) % agg_interval == 0:
            aggregate(clients)

        ev = _eval_episode(envs[0], clients[0], cfg.max_steps)
        history["reward"].append(ev["reward"])
        history["delay"].append(ev["delay"])
        history["energy"].append(ev["energy"])
        history["cost"].append(ev["system_cost"])
    return history


def make_env(cfg):
    return UAVMECEnv(EnvConfig(
        num_users=cfg.num_users, omega_delay=cfg.omega_delay,
        omega_energy=cfg.omega_energy, max_steps=cfg.max_steps, seed=cfg.seed))


def make_d3qn(cfg):
    return D3QNAgent(D3QNConfig(device=cfg.device))


def run_method(name, cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_env(cfg)
    name = name.lower()

    def _baseline(policy):
        hist = {"reward": [], "delay": [], "energy": [], "cost": []}
        for _ in range(cfg.num_episodes):
            ev = _eval_episode(env, policy, cfg.max_steps)
            hist["reward"].append(ev["reward"])
            hist["delay"].append(ev["delay"])
            hist["energy"].append(ev["energy"])
            hist["cost"].append(ev["system_cost"])
        return hist, policy

    if name == "random":      return _baseline(RandomPolicy())
    if name == "all-local":   return _baseline(AllLocalPolicy())
    if name == "all-uav":     return _baseline(AllUAVPolicy())
    if name == "dqn":         return train_simple(DQNAgent(D3QNConfig(device=cfg.device)), env, cfg), None
    if name == "dueling-dqn": return train_simple(DuelingDQNAgent(D3QNConfig(device=cfg.device)), env, cfg), None
    if name == "d3qn":        return train_simple(make_d3qn(cfg), env, cfg), None
    if name == "fedavg-d3qn":
        c = deepcopy(cfg)
        c.synth_ratio = 0.0
        ag = make_d3qn(cfg)
        return train_with_diffusion(ag, env, c, conditional=False,
                                    use_federated=True, num_clients=cfg.num_clients), ag
    if name == "synther":
        ag = make_d3qn(cfg)
        return train_with_diffusion(ag, env, cfg, conditional=False, use_federated=False), ag
    if name == "difffed-d3qn":
        ag = make_d3qn(cfg)
        return train_with_diffusion(ag, env, cfg, conditional=True,
                                    use_federated=True, num_clients=cfg.num_clients), ag
    raise ValueError(f"unknown method: {name}")


# ============================================================================
# 7. EXPERIMENT SCRIPTS
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _out(path):
    return os.path.join(RESULTS_DIR, path)


def _plot_with_ci(curves, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, runs in curves.items():
        arr = np.array(runs, dtype=np.float32)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out(path))
    plt.close(fig)


def _bar_chart(values, ylabel, title, path, errs=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(values.keys())
    y = [values[n] for n in names]
    e = [errs[n] for n in names] if errs else None
    ax.bar(names, y, yerr=e, capsize=4, color="#4c72b0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(_out(path))
    plt.close(fig)


# ---- Experiment 1: convergence comparison ----------------------------------

EXP1_METHODS = ["random", "all-local", "all-uav", "dqn", "dueling-dqn",
                "d3qn", "fedavg-d3qn", "synther", "difffed-d3qn"]


def exp1_convergence(num_episodes=80, seeds=(0, 1, 2)):
    curves = {m: [] for m in EXP1_METHODS}
    for s in seeds:
        for m in EXP1_METHODS:
            cfg = TrainConfig(num_episodes=num_episodes, seed=s)
            hist, _ = run_method(m, cfg)
            curves[m].append(hist["reward"])
    _plot_with_ci(curves, "Episode", "Episode Reward",
                  "Convergence Comparison", "convergence.pdf")


# ---- Experiment 2: cost vs delay weight -----------------------------------

EXP2_METHODS = ["random", "all-local", "all-uav", "d3qn", "fedavg-d3qn",
                "synther", "difffed-d3qn"]
EXP2_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def exp2_cost_vs_weight(num_episodes=50, seeds=(0,)):
    results = defaultdict(dict)
    for w in EXP2_WEIGHTS:
        for m in EXP2_METHODS:
            cs = []
            for s in seeds:
                cfg = TrainConfig(num_episodes=num_episodes, seed=s,
                                  omega_delay=w, omega_energy=1.0 - w)
                hist, _ = run_method(m, cfg)
                cs.append(np.mean(hist["cost"][-10:]))
            results[m][w] = float(np.mean(cs))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(EXP2_WEIGHTS))
    width = 0.11
    for i, m in enumerate(EXP2_METHODS):
        y = [results[m][w] for w in EXP2_WEIGHTS]
        ax.bar(x + (i - len(EXP2_METHODS) / 2) * width, y, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.2f}" for w in EXP2_WEIGHTS])
    ax.set_xlabel(r"$\omega_1$ (delay weight)")
    ax.set_ylabel("System Cost")
    ax.set_title("System Cost vs Delay Weight")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out("cost_vs_weight.pdf"))
    plt.close(fig)


# ---- Experiment 3: cost vs number of users --------------------------------

EXP3_METHODS = ["all-local", "all-uav", "d3qn", "fedavg-d3qn",
                "synther", "difffed-d3qn"]
EXP3_USERS = [10, 20, 30, 40, 50]


def exp3_cost_vs_users(num_episodes=50, seeds=(0,)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in EXP3_METHODS:
        y = []
        for n in EXP3_USERS:
            cs = []
            for s in seeds:
                cfg = TrainConfig(num_episodes=num_episodes, seed=s, num_users=n)
                hist, _ = run_method(m, cfg)
                cs.append(np.mean(hist["cost"][-10:]))
            y.append(np.mean(cs))
        ax.plot(EXP3_USERS, y, marker="o", label=m)
    ax.set_xlabel("Number of users")
    ax.set_ylabel("System Cost")
    ax.set_title("Scalability: Cost vs Users")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out("cost_vs_users.pdf"))
    plt.close(fig)


# ---- Experiment 4: sample efficiency --------------------------------------

EXP4_METHODS = ["d3qn", "synther", "difffed-d3qn"]
EXP4_BUDGETS = [1000, 5000, 10000, 25000, 50000, 100000]


def exp4_sample_efficiency(seeds=(0,)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in EXP4_METHODS:
        means, stds = [], []
        for budget in EXP4_BUDGETS:
            interactions_per_ep = 50 * 100
            episodes = max(2, budget // interactions_per_ep)
            vals = []
            for s in seeds:
                cfg = TrainConfig(num_episodes=episodes, seed=s)
                hist, _ = run_method(m, cfg)
                vals.append(np.mean(hist["reward"][-5:]))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(EXP4_BUDGETS, means, yerr=stds, marker="o", label=m, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Real interactions")
    ax.set_ylabel("Final episode reward")
    ax.set_title("Sample Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(_out("sample_efficiency.pdf"))
    plt.close(fig)


# ---- Experiment 5: ablation -----------------------------------------------

EXP5_VARIANTS = [
    "D3QN only",
    "D3QN + Random",
    "D3QN + Uncond. Diffusion",
    "D3QN + Cond. Diffusion",
    "DiffFed-D3QN (FULL)",
]


def _exp5_run_one(variant, cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_env(cfg)
    ag = D3QNAgent(D3QNConfig(device=cfg.device))
    if variant == "D3QN only":
        hist = train_simple(ag, env, cfg)
    elif variant == "D3QN + Random":
        hist = train_with_diffusion(ag, env, cfg, conditional=False, use_federated=False, random_gen=True)
    elif variant == "D3QN + Uncond. Diffusion":
        hist = train_with_diffusion(ag, env, cfg, conditional=False, use_federated=False)
    elif variant == "D3QN + Cond. Diffusion":
        hist = train_with_diffusion(ag, env, cfg, conditional=True, use_federated=False)
    else:
        hist = train_with_diffusion(ag, env, cfg, conditional=True,
                                    use_federated=True, num_clients=cfg.num_clients)
    return float(np.mean(hist["reward"][-10:]))


def exp5_ablation(num_episodes=60, seeds=(0, 1, 2)):
    means, errs = {}, {}
    for v in EXP5_VARIANTS:
        vals = [_exp5_run_one(v, TrainConfig(num_episodes=num_episodes, seed=s))
                for s in seeds]
        means[v] = float(np.mean(vals))
        errs[v] = float(np.std(vals))
    _bar_chart(means, "Final Episode Reward", "Ablation Study", "ablation.pdf", errs=errs)


# ---- Experiment 6: synthetic-ratio sweep ----------------------------------

EXP6_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def exp6_synthetic_ratio(num_episodes=60, seeds=(0, 1, 2)):
    rew, rew_std, delay, energy, cost = [], [], [], [], []
    for r in EXP6_RATIOS:
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
        delay.append(np.mean(ds)); energy.append(np.mean(es)); cost.append(np.mean(cs))

    rew = np.array(rew); rew_std = np.array(rew_std)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].errorbar(EXP6_RATIOS, rew, yerr=rew_std, marker="o")
    axes[0, 0].set_title("(a) Reward vs Synthetic Ratio")
    axes[0, 0].set_xlabel("Synthetic ratio"); axes[0, 0].set_ylabel("Reward")
    best = int(np.argmax(rew))
    axes[0, 0].annotate(f"optimal={EXP6_RATIOS[best]:.1f}",
                        xy=(EXP6_RATIOS[best], rew[best]),
                        xytext=(EXP6_RATIOS[best], rew[best] + 0.5 * max(1e-3, rew.std())),
                        arrowprops=dict(arrowstyle="->", color="red"), color="red")
    axes[0, 1].plot(EXP6_RATIOS, delay, marker="s", color="orange")
    axes[0, 1].set_title("(b) Delay vs Synthetic Ratio")
    axes[0, 1].set_xlabel("Synthetic ratio"); axes[0, 1].set_ylabel("Delay (s)")
    axes[1, 0].plot(EXP6_RATIOS, energy, marker="^", color="green")
    axes[1, 0].set_title("(c) Energy vs Synthetic Ratio")
    axes[1, 0].set_xlabel("Synthetic ratio"); axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 1].plot(EXP6_RATIOS, cost, marker="d", color="red")
    axes[1, 1].set_title("(d) System Cost vs Synthetic Ratio")
    axes[1, 1].set_xlabel("Synthetic ratio"); axes[1, 1].set_ylabel("Cost")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out("synthetic_ratio.pdf"))
    plt.close(fig)


# ---- Experiment 7: distribution comparison --------------------------------

def exp7_distribution(seed=0, n_real=2000, n_synth=2000):
    from sklearn.decomposition import PCA
    np.random.seed(seed); torch.manual_seed(seed)
    cfg = TrainConfig(num_episodes=1, seed=seed)
    env = make_env(cfg)
    agent = D3QNAgent(D3QNConfig(device=cfg.device))
    contexts = []

    s = env.reset()
    real = []
    while len(real) < n_real:
        a = agent.act(s, greedy=False)
        ns, r, done, info = env.step(a)
        for i in range(env.num_users):
            real.append((s[i].copy(), int(a[i]), float(r), ns[i].copy()))
            contexts.append(env.network_context.copy())
            if len(real) >= n_real:
                break
        s = ns
        if done:
            s = env.reset()

    diffusion = ConditionalDiffusion(DiffusionConfig(device=cfg.device))
    diffusion.fit(real, contexts, epochs=80, batch_size=64)
    ctx_vec = env.network_context
    s_g, a_g, r_g, ns_g = diffusion.generate(n_synth, ctx_vec)
    s_g = s_g.cpu().numpy()
    real_s = np.array([t[0] for t in real])

    p = PCA(n_components=2)
    z_real = p.fit_transform(real_s)
    z_synth = p.transform(s_g)

    labels = ["task_MB", "cycles_G", "uav_load", "user_GHz", "dist_m", "snr_dB"]
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
    fig.savefig(_out("distribution.pdf"))
    plt.close(fig)


# ---- Experiment 8: aggregation frequency ----------------------------------

EXP8_INTERVALS = [1, 5, 10, 20]


def exp8_agg_frequency(num_episodes=60, seeds=(0, 1, 2)):
    curves = {f"agg={k}": [] for k in EXP8_INTERVALS}
    for s in seeds:
        for k in EXP8_INTERVALS:
            np.random.seed(s); torch.manual_seed(s)
            cfg = TrainConfig(num_episodes=num_episodes, seed=s, fed_every=k)
            env = make_env(cfg)
            ag = D3QNAgent(D3QNConfig(device=cfg.device))
            hist = train_with_diffusion(ag, env, cfg, conditional=True,
                                        use_federated=True,
                                        num_clients=cfg.num_clients, agg_every=k)
            curves[f"agg={k}"].append(hist["reward"])
    _plot_with_ci(curves, "Episode", "Episode Reward",
                  "Aggregation-Interval Sweep", "agg_frequency.pdf")


# ============================================================================
# 8. CLI DRIVER
# ============================================================================

EXPS = {
    1: ("convergence",            exp1_convergence),
    2: ("cost vs weight",         exp2_cost_vs_weight),
    3: ("cost vs users",          exp3_cost_vs_users),
    4: ("sample efficiency",      exp4_sample_efficiency),
    5: ("ablation",               exp5_ablation),
    6: ("synthetic ratio",        exp6_synthetic_ratio),
    7: ("distribution",           exp7_distribution),
    8: ("aggregation frequency",  exp8_agg_frequency),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*", type=int, help="experiment ids to run")
    parser.add_argument("--quick", action="store_true",
                        help="reduce episodes/seeds for a smoke test")
    args = parser.parse_args()
    selected = args.ids or list(EXPS.keys())
    for i in selected:
        name, fn = EXPS[i]
        print(f"\n[exp{i}] {name} ...")
        t0 = time.time()
        if args.quick:
            try:
                if i == 7:
                    fn(seed=0, n_real=400, n_synth=400)
                else:
                    fn(num_episodes=15, seeds=(0,))
            except TypeError:
                fn()
        else:
            fn()
        print(f"[exp{i}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
