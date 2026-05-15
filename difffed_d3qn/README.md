# DiffFed-D3QN

Diffusion-Augmented Federated Deep Reinforcement Learning for UAV-MEC Task Offloading.

## Install

```bash
pip install -r difffed_d3qn/requirements.txt
```

## Quick smoke test (all 8 experiments, reduced settings)

```bash
python -m difffed_d3qn.run_all --quick
```

## Run one experiment

```bash
python -m difffed_d3qn.experiments.exp1_convergence
python -m difffed_d3qn.experiments.exp6_synthetic_ratio
# ...
```

## Run everything (full settings)

```bash
python -m difffed_d3qn.run_all
```

Outputs are written to `difffed_d3qn/results/` as `.pdf` figures:

| File | Experiment |
|------|------------|
| convergence.pdf       | 1 — Reward vs episodes (all methods) |
| cost_vs_weight.pdf    | 2 — Cost vs delay weight $\omega_1$ |
| cost_vs_users.pdf     | 3 — Scalability vs user count |
| sample_efficiency.pdf | 4 — Final reward vs real-interaction budget |
| ablation.pdf          | 5 — Ablation over synthetic + FL components |
| synthetic_ratio.pdf   | 6 — 2x2 sweep over synthetic ratio |
| distribution.pdf      | 7 — PCA + per-dim histograms (real vs synth) |
| agg_frequency.pdf     | 8 — Federated aggregation interval |

## Module layout

```
difffed_d3qn/
├── environment.py       UAV-MEC Gym env (6-dim state, 2 discrete actions)
├── d3qn_agent.py        Dueling Double DQN agent + replay buffer
├── diffusion.py         Conditional DDPM + Q-value quality filter
├── federated.py         FedAvg weight aggregation
├── baselines.py         Random / All-Local / All-UAV / DQN / Dueling-DQN
├── train.py             Training loops + method dispatcher
├── experiments/         Eight experiment scripts
└── run_all.py           Top-level driver
```
