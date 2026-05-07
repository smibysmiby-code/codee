"""
evaluate.py
===========

Compare 5 task-offloading policies on the mixed-distribution test set:

    a) Federated FedAvg model       (the contribution)
    b) Centralised model             (upper bound, requires data sharing)
    c) Always-local baseline         (heuristic)
    d) Always-offload baseline       (heuristic)
    e) Random 50/50 baseline         (heuristic)

For each policy we compute:

    * classification accuracy vs the optimal label   (where applicable)
    * mean cost J on the test tasks
    * mean delay
    * mean energy
    * deadline-violation rate         (fraction of tasks where the chosen
                                       action's delay exceeded the deadline)

Returns a `dict` of dicts so `plot_results.py` can build the comparison
charts and `main.py` can print a summary table.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import config
from model import OffloadingMLP, dataframe_to_tensors
from simulator import (
    Task,
    compute_local_cost,
    compute_offload_cost,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _model_predictions(model: OffloadingMLP, X: torch.Tensor) -> np.ndarray:
    """Argmax predictions from a trained MLP (numpy 1D int array)."""
    model.eval()
    with torch.no_grad():
        return model(X).argmax(dim=1).cpu().numpy()


def _physical_metrics(test_df: pd.DataFrame,
                      actions: np.ndarray) -> Dict[str, float]:
    """Compute mean cost / delay / energy / violation-rate for the chosen
    action sequence on the test tasks.

    `actions[i] == 0` -> task i is run locally,
    `actions[i] == 1` -> task i is offloaded to the UAV.
    """
    delays:     List[float] = []
    energies:   List[float] = []
    costs:      List[float] = []
    violations: List[int]   = []

    for (_, row), a in zip(test_df.iterrows(), actions):
        task = Task(
            task_size_MB = row["task_size_MB"],
            cpu_cycles   = row["cpu_cycles"],
            deadline_s   = row["deadline_s"],
            channel_gain = row["channel_gain"],
            iot_battery  = row["iot_battery"],
        )
        if a == 0:
            t, e, j = compute_local_cost(task)
        else:
            t, e, j = compute_offload_cost(task)
        delays.append(t)
        energies.append(e)
        costs.append(j)
        violations.append(int(t > task.deadline_s))

    return {
        "mean_cost":      float(np.mean(costs)),
        "mean_delay":     float(np.mean(delays)),
        "mean_energy":    float(np.mean(energies)),
        "violation_rate": float(np.mean(violations)),
    }


def _accuracy(actions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(actions == labels))


# ---------------------------------------------------------------------------
# Policy generators -- each returns an array of 0/1 actions per test task
# ---------------------------------------------------------------------------
def _federated_actions(test_df: pd.DataFrame) -> np.ndarray:
    model = OffloadingMLP()
    model.load_state_dict(torch.load(
        os.path.join(config.MODEL_DIR, "federated.pt"),
        weights_only=True,
    ))
    X, _ = dataframe_to_tensors(test_df)
    return _model_predictions(model, X)


def _centralized_actions(test_df: pd.DataFrame) -> np.ndarray:
    model = OffloadingMLP()
    model.load_state_dict(torch.load(
        os.path.join(config.MODEL_DIR, "centralized.pt"),
        weights_only=True,
    ))
    X, _ = dataframe_to_tensors(test_df)
    return _model_predictions(model, X)


def _always_local_actions(n: int)   -> np.ndarray: return np.zeros(n, dtype=np.int64)
def _always_offload_actions(n: int) -> np.ndarray: return np.ones(n,  dtype=np.int64)


def _random_actions(n: int) -> np.ndarray:
    rng = np.random.default_rng(config.RANDOM_SEED + 7)  # local stable RNG
    return rng.integers(low=0, high=2, size=n)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def evaluate_all_methods() -> Dict[str, Dict[str, float]]:
    """Evaluate the 5 policies and return a result dict keyed by method name."""
    _seed_everything(config.RANDOM_SEED)

    test_df = pd.read_csv(config.test_csv_path())
    n       = len(test_df)
    labels  = test_df[config.LABEL_COLUMN].to_numpy()

    print(f"[evaluate] Loaded {n} test tasks")

    methods = {
        "Federated":      _federated_actions(test_df),
        "Centralized":    _centralized_actions(test_df),
        "Always-local":   _always_local_actions(n),
        "Always-offload": _always_offload_actions(n),
        "Random 50/50":   _random_actions(n),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, actions in methods.items():
        metrics = _physical_metrics(test_df, actions)
        metrics["accuracy"] = _accuracy(actions, labels)
        results[name] = metrics

    # Pretty-print a table.
    print()
    print(f"{'Method':<16} {'Accuracy':>9} {'MeanCost':>10} "
          f"{'MeanDelay':>11} {'MeanEnergy':>12} {'ViolRate':>9}")
    print("-" * 72)
    for name, m in results.items():
        print(f"{name:<16} {m['accuracy']:>9.4f} {m['mean_cost']:>10.4f} "
              f"{m['mean_delay']:>11.4f} {m['mean_energy']:>12.4f} "
              f"{m['violation_rate']:>9.3f}")
    return results


if __name__ == "__main__":
    evaluate_all_methods()
