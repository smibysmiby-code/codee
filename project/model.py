"""
model.py
========

A small multi-layer perceptron (MLP) for binary task-offloading decisions.

Architecture (kept deliberately tiny so it trains fast on a CPU):

    Input  : 5 features  (task_size, cpu_cycles, deadline, channel, battery)
             (already min-max normalised to [0,1] by the caller)
    FC1    : 5  -> 32   (ReLU)
    FC2    : 32 -> 16   (ReLU)
    FC3    : 16 -> 2    (logits for "local" / "offload")

The same model class is used for the centralised baseline and for every
device's local copy in the federated training loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config


class OffloadingMLP(nn.Module):
    """Tiny 2-hidden-layer MLP with 2-class output."""

    def __init__(self, in_features: int = 5, num_classes: int = 2) -> None:
        super().__init__()
        # nn.Sequential is more than enough for a 3-layer net and keeps
        # the FedAvg parameter-aggregation code very small.
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass returning raw class logits (no softmax)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers shared between centralised and federated trainers
# ---------------------------------------------------------------------------
def normalise_features(df: pd.DataFrame) -> np.ndarray:
    """Min-max normalise the feature columns of `df` into [0, 1] using the
    *global* feature bounds defined in `config.FEATURE_BOUNDS`.

    Using global bounds (not per-CSV bounds) is important: every device's
    local model and the central model must agree on the input scale,
    otherwise FedAvg averaging is meaningless.
    """
    out = np.empty((len(df), len(config.FEATURE_COLUMNS)), dtype=np.float32)
    for i, col in enumerate(config.FEATURE_COLUMNS):
        lo, hi = config.FEATURE_BOUNDS[col]
        out[:, i] = (df[col].to_numpy(dtype=np.float32) - lo) / (hi - lo)
    # Numerical safety: clip in case a value falls outside the declared bounds.
    return np.clip(out, 0.0, 1.0)


def dataframe_to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a labelled DataFrame to (X, y) PyTorch tensors."""
    X = torch.from_numpy(normalise_features(df))
    # `.copy()` -> ensures the array is writable so PyTorch doesn't warn.
    y = torch.from_numpy(df[config.LABEL_COLUMN].to_numpy(dtype=np.int64).copy())
    return X, y


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Adam with the global learning rate from `config`."""
    return torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


def build_loss() -> nn.Module:
    """Cross-entropy is the standard choice for multi-class classification
    (and works fine for the binary case)."""
    return nn.CrossEntropyLoss()
