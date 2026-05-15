"""Federated averaging for D3QN agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List

import torch

from .d3qn_agent import D3QNAgent


def fedavg_weights(state_dicts: List[dict]) -> dict:
    """Element-wise mean of a list of nested state dicts ({'online':..., 'target':...})."""
    avg = deepcopy(state_dicts[0])
    for key in avg:
        for pkey in avg[key]:
            stacked = torch.stack([sd[key][pkey].float() for sd in state_dicts], dim=0)
            avg[key][pkey] = stacked.mean(dim=0)
    return avg


def aggregate(agents: Iterable[D3QNAgent]):
    agents = list(agents)
    sd = fedavg_weights([a.state_dict() for a in agents])
    for a in agents:
        a.load_state_dict(sd)
    return sd
