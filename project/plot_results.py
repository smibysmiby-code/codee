"""
plot_results.py
===============

Generate the four figures used in the paper:

    1. fl_convergence.png       FL accuracy vs. communication rounds
    2. method_accuracy.png      Bar chart -- accuracy per policy
    3. method_cost.png          Bar chart -- mean cost per policy
    4. label_distribution.png   Per-device label distribution (non-IID viz)

All figures are saved as PNG into `config.FIG_DIR`.

`make_all_figures()` is the orchestrator -- main.py calls it once with
results from previous phases.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless backend -- no $DISPLAY needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def plot_fl_convergence(accuracy_history: List[float]) -> str:
    """Line plot of FL test accuracy over communication rounds."""
    rounds = np.arange(1, len(accuracy_history) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, accuracy_history, marker="o", color="C0",
            label="FedAvg global model")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Federated learning convergence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(config.FIG_DIR, "fl_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_method_bar(results: Dict[str, Dict[str, float]],
                    metric: str,
                    ylabel: str,
                    title: str,
                    out_name: str) -> str:
    """Generic bar chart over the 5 methods, on a single metric."""
    names  = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["C0", "C1", "C2", "C3", "C4"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    # Annotate each bar with its value
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()

    path = os.path.join(config.FIG_DIR, out_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_label_distribution() -> str:
    """Stacked bar chart of (local, offload) counts per device.

    Reading this figure is the cleanest way to *see* the non-IID nature
    of the federated dataset.
    """
    locals_, offloads = [], []
    for dev in range(config.NUM_DEVICES):
        df = pd.read_csv(config.device_csv_path(dev))
        locals_.append(int((df[config.LABEL_COLUMN] == 0).sum()))
        offloads.append(int((df[config.LABEL_COLUMN] == 1).sum()))

    x = np.arange(config.NUM_DEVICES)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, locals_,   label="Local (0)",   color="C0")
    ax.bar(x, offloads,  label="Offload (1)", color="C1", bottom=locals_)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dev{i}" for i in x])
    ax.set_ylabel("Number of tasks")
    ax.set_title("Per-device label distribution (non-IID)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(config.FIG_DIR, "label_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all_figures(accuracy_history: List[float],
                     results: Dict[str, Dict[str, float]]) -> List[str]:
    """Render all four figures and return the list of saved paths."""
    paths = [
        plot_fl_convergence(accuracy_history),
        plot_method_bar(results, "accuracy",
                        "Accuracy", "Test accuracy per method",
                        "method_accuracy.png"),
        plot_method_bar(results, "mean_cost",
                        "Mean cost J", "Mean cost per method",
                        "method_cost.png"),
        plot_label_distribution(),
    ]
    for p in paths:
        print(f"[plot] saved {p}")
    return paths
