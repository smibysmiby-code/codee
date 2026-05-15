"""Non-D3QN baseline policies and minimal DQN / Dueling-DQN variants."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .d3qn_agent import D3QNAgent, D3QNConfig, ReplayBuffer


# ---------------------------------------------------------------------------
# Rule-based baselines
# ---------------------------------------------------------------------------

class RandomPolicy:
    def __init__(self, action_dim: int = 2):
        self.action_dim = action_dim

    def act(self, states, greedy: bool = True):
        return np.random.randint(0, self.action_dim, size=states.shape[0])

    def update(self, *args, **kwargs):
        return 0.0

    def decay_epsilon(self):
        pass


class AllLocalPolicy:
    def act(self, states, greedy: bool = True):
        return np.zeros(states.shape[0], dtype=np.int64)

    def update(self, *args, **kwargs):
        return 0.0

    def decay_epsilon(self):
        pass


class AllUAVPolicy:
    def act(self, states, greedy: bool = True):
        return np.ones(states.shape[0], dtype=np.int64)

    def update(self, *args, **kwargs):
        return 0.0

    def decay_epsilon(self):
        pass


# ---------------------------------------------------------------------------
# Vanilla DQN / Dueling DQN by sharing D3QN's training machinery
# ---------------------------------------------------------------------------

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
    """Vanilla DQN: plain MLP + single network bootstrap (no Double Q)."""

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
    """Dueling architecture but no Double Q (use target net for max as well)."""

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
