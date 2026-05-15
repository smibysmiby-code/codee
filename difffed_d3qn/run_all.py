"""Run every experiment sequentially.

Usage:
    python -m difffed_d3qn.run_all              # all
    python -m difffed_d3qn.run_all 1 6          # just experiments 1 and 6
    python -m difffed_d3qn.run_all --quick      # quick smoke run
"""

from __future__ import annotations

import argparse
import time

from .experiments import (
    exp1_convergence,
    exp2_cost_vs_weight,
    exp3_cost_vs_users,
    exp4_sample_efficiency,
    exp5_ablation,
    exp6_synthetic_ratio,
    exp7_distribution,
    exp8_agg_frequency,
)


EXPS = {
    1: ("convergence",          exp1_convergence.run),
    2: ("cost vs weight",       exp2_cost_vs_weight.run),
    3: ("cost vs users",        exp3_cost_vs_users.run),
    4: ("sample efficiency",    exp4_sample_efficiency.run),
    5: ("ablation",             exp5_ablation.run),
    6: ("synthetic ratio",      exp6_synthetic_ratio.run),
    7: ("distribution",         exp7_distribution.run),
    8: ("aggregation frequency", exp8_agg_frequency.run),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*", type=int, help="which experiments to run")
    parser.add_argument("--quick", action="store_true",
                        help="reduce episodes/seeds for a smoke test")
    args = parser.parse_args()

    selected = args.ids or list(EXPS.keys())
    for i in selected:
        name, fn = EXPS[i]
        print(f"\n[exp{i}] {name} ...")
        t0 = time.time()
        if args.quick:
            # Pass conservative defaults if the run() accepts them
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
