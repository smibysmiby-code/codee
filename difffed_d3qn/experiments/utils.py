"""Shared plotting / result-saving helpers."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def out(path: str) -> str:
    return os.path.join(RESULTS_DIR, path)


def smooth(x, k: int = 5):
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 1:
        return x
    k = min(k, len(x))
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def plot_with_ci(curves: Dict[str, List[List[float]]], xlabel: str, ylabel: str,
                 title: str, path: str):
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
    fig.savefig(out(path))
    plt.close(fig)


def bar_chart(values: Dict[str, float], ylabel: str, title: str, path: str,
              errs: Dict[str, float] = None):
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
    fig.savefig(out(path))
    plt.close(fig)
