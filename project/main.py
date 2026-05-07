"""
main.py
=======

End-to-end orchestrator for the federated-learning UAV-offloading
prototype.  Running this single script reproduces the entire experiment:

    1. (re)generate the per-device CSV files and the test set,
    2. train the centralised baseline,
    3. train the federated FedAvg model,
    4. evaluate all 5 policies on the mixed test set,
    5. render the four figures used in the paper,
    6. print a final summary table.

Usage:

    python main.py
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch

import config
import data_generator
import centralized_training
import federated_training
import evaluate as evaluate_module
import plot_results


def _seed_everything(seed: int) -> None:
    """Seed every RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_summary(results: dict, fed_history: list) -> None:
    _print_header("FINAL SUMMARY")
    print(f"{'Method':<16} {'Accuracy':>9} {'MeanCost':>10} "
          f"{'MeanDelay':>11} {'MeanEnergy':>12} {'ViolRate':>9}")
    print("-" * 72)
    for name, m in results.items():
        print(f"{name:<16} {m['accuracy']:>9.4f} {m['mean_cost']:>10.4f} "
              f"{m['mean_delay']:>11.4f} {m['mean_energy']:>12.4f} "
              f"{m['violation_rate']:>9.3f}")

    print()
    print(f"Federated round-by-round accuracy: "
          f"{[f'{a:.3f}' for a in fed_history]}")
    print(f"Federated final accuracy:    {fed_history[-1]:.4f}")
    print(f"Centralized accuracy:        {results['Centralized']['accuracy']:.4f}")
    print(f"Optimal-cost reference:      "
          f"see 'Federated' / 'Centralized' rows above.")


def main() -> None:
    t0 = time.time()
    _seed_everything(config.RANDOM_SEED)

    _print_header("PHASE 1: DATA GENERATION")
    data_generator.main()

    _print_header("PHASE 2: CENTRALIZED TRAINING")
    centralized_training.train_centralized()

    _print_header("PHASE 3: FEDERATED TRAINING (FedAvg)")
    _, fed_history = federated_training.train_federated()

    _print_header("PHASE 4: EVALUATION")
    results = evaluate_module.evaluate_all_methods()

    _print_header("PHASE 5: FIGURES")
    plot_results.make_all_figures(fed_history, results)

    _print_summary(results, fed_history)

    elapsed = time.time() - t0
    print(f"\n[main] Wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
