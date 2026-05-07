"""
federated_training.py
=====================

Phase 3: hand-rolled FedAvg training loop.

Each round, the UAV (the aggregator):
    1. broadcasts the current global model weights to every IoT device,
    2. each device performs `FED_LOCAL_EPOCHS` of SGD on its private CSV,
    3. devices send back their updated weights,
    4. the UAV averages the weights, weighted by each device's dataset
       size (this is plain FedAvg [McMahan et al. 2017]).

Round-by-round test accuracy (on the mixed-distribution test set) is
printed and returned, so `plot_results.py` can draw a convergence curve.

We deliberately do *not* use Flower / PySyft / FedML -- the whole point
is to see FedAvg working in ~80 lines of plain PyTorch.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from model import (
    OffloadingMLP,
    dataframe_to_tensors,
    make_optimizer,
    build_loss,
)
from centralized_training import evaluate


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_device_tensors() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Load each device's CSV into (X, y) tensors -- one per device."""
    out = []
    for dev in range(config.NUM_DEVICES):
        df = pd.read_csv(config.device_csv_path(dev))
        out.append(dataframe_to_tensors(df))
    return out


def _local_train(model: nn.Module,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 epochs: int,
                 batch_size: int,
                 lr: float) -> nn.Module:
    """Train `model` on (X, y) for `epochs` epochs of mini-batch SGD/Adam.

    Returns the same `model` object (parameters are updated in place).
    Each device gets a *fresh optimiser* every round, which is the
    standard FedAvg behaviour (we don't synchronise optimiser state).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    n = X.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X[idx])
            loss   = loss_fn(logits, y[idx])
            loss.backward()
            optimizer.step()
    return model


def _fedavg_aggregate(local_states: List[Dict[str, torch.Tensor]],
                      sample_counts: List[int]) -> Dict[str, torch.Tensor]:
    """Weighted average of `local_states` using `sample_counts` as weights.

    weight_k = n_k / sum(n_k)            -- each device contributes
    global_w = sum_k  weight_k * w_k     -- proportionally to its data
    """
    total = float(sum(sample_counts))
    weights = [n / total for n in sample_counts]

    # Initialise the aggregated state with zeros, with the same shape as
    # the first local state.  We iterate parameter by parameter.
    avg_state: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v) for k, v in local_states[0].items()
    }
    for w, state in zip(weights, local_states):
        for k, v in state.items():
            avg_state[k] += w * v
    return avg_state


def train_federated() -> Tuple[OffloadingMLP, List[float]]:
    """Run FedAvg.  Returns (final_global_model, accuracy_per_round)."""
    _seed_everything(config.RANDOM_SEED)

    # Per-device data
    device_data = _load_device_tensors()
    sample_counts = [X.size(0) for X, _ in device_data]
    print(f"[federated] Device dataset sizes: {sample_counts}")

    # Mixed-distribution test set, used to evaluate the global model
    # after every round (this is why the test set is "held out" from any
    # device's training data).
    test_df = pd.read_csv(config.test_csv_path())
    X_test, y_test = dataframe_to_tensors(test_df)

    # Initialise the global model from a single fixed seed so every run
    # of this script starts from the same point.
    global_model = OffloadingMLP()

    accuracy_history: List[float] = []

    print(f"[federated] Running {config.FED_ROUNDS} rounds, "
          f"{config.FED_LOCAL_EPOCHS} local epochs each.")
    for rnd in range(1, config.FED_ROUNDS + 1):
        local_states: List[Dict[str, torch.Tensor]] = []

        # ---- Step 1: each device trains a *copy* of the global model ----
        for dev_id, (X, y) in enumerate(device_data):
            local_model = copy.deepcopy(global_model)
            _local_train(
                local_model, X, y,
                epochs=config.FED_LOCAL_EPOCHS,
                batch_size=config.FED_BATCH_SIZE,
                lr=config.LEARNING_RATE,
            )
            local_states.append(local_model.state_dict())

        # ---- Step 2: UAV aggregates updates with dataset-size weighting ----
        new_state = _fedavg_aggregate(local_states, sample_counts)
        global_model.load_state_dict(new_state)

        # ---- Step 3: evaluate the fresh global model ----
        acc, _ = evaluate(global_model, X_test, y_test)
        accuracy_history.append(acc)
        print(f"  round {rnd:2d}/{config.FED_ROUNDS}: test_acc={acc:.4f}")

    # Persist the final global model
    model_path = os.path.join(config.MODEL_DIR, "federated.pt")
    torch.save(global_model.state_dict(), model_path)
    print(f"[federated] Saved global model -> {model_path}")

    return global_model, accuracy_history


if __name__ == "__main__":
    train_federated()
