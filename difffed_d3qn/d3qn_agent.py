"""Dueling Double Deep Q-Network (D3QN) agent."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((np.asarray(s, dtype=np.float32), int(a), float(r),
                         np.asarray(ns, dtype=np.float32), bool(d)))

    def sample(self, batch_size: int):
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

    # ------------------------------------------------------------------
    def act(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        if (not greedy) and random.random() < self.eps:
            return np.random.randint(0, self.cfg.action_dim, size=states.shape[0])
        with torch.no_grad():
            s = torch.from_numpy(states.astype(np.float32)).to(self.device)
            q = self.online(s)
            return q.argmax(dim=-1).cpu().numpy()

    def decay_epsilon(self):
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    # ------------------------------------------------------------------
    def q_values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(states.astype(np.float32)).to(self.device)
            return self.online(s).cpu().numpy()

    def soft_update(self):
        tau = self.cfg.tau
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * op.data)

    # ------------------------------------------------------------------
    def update_from_batch(self, s, a, r, ns, d) -> float:
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

    def update(self) -> float:
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        batch = self.buffer.sample(self.cfg.batch_size)
        return self.update_from_batch(*batch)

    # ------------------------------------------------------------------
    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict()}

    def load_state_dict(self, sd):
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["target"])
