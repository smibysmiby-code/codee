# Federated Learning for Task Offloading in UAV-assisted Edge Networks

A small, self-contained PyTorch prototype that learns the optimal binary
offloading decision (`local` vs. `offload-to-UAV`) for IoT tasks, using
**FedAvg** federated learning with no cloud and a single UAV as both
edge server and aggregator.

## System model

* 5 IoT devices on the ground (each with limited CPU/battery).
* 1 UAV hovering above, acting as **edge server** *and* **FL aggregator**.
* For every task, the IoT device must decide:
  - `0` = execute locally on the IoT device
  - `1` = offload to the UAV

Every task is described by 5 features:
`task_size_MB`, `cpu_cycles`, `deadline_s`, `channel_gain`, `iot_battery`.

The optimal label is the action with the lower cost
`J = mu * E + eta * T` among those that meet the deadline.

## Project layout

```
project/
├── config.py                 # constants (system + hyper-parameters)
├── simulator.py              # one-task sampling + cost / label logic
├── data_generator.py         # produce 5 device CSVs + a test CSV
├── model.py                  # MLP definition + normalisation helpers
├── centralized_training.py   # Phase 2 baseline
├── federated_training.py     # Phase 3: hand-rolled FedAvg
├── evaluate.py               # accuracy, cost, delay, violations -- 5 policies
├── plot_results.py           # 4 PNG figures
└── main.py                   # runs everything end-to-end
data/                         # generated CSVs
figures/                      # generated PNGs
models/                       # saved PyTorch state-dicts
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the experiment

```bash
cd project
python main.py
```

Total wall-clock time on a CPU is about 30-60 seconds.  The script:

1. **Generates data** -- 1000 labelled tasks per device + 500 mixed test tasks.
   Each device's task distribution is **non-IID** (cameras vs sensors vs
   smart meters etc.).  CSVs are written into `data/`.
2. **Trains the centralised baseline** -- pools all 5 device CSVs and
   trains a single MLP for 50 epochs.  This is the upper bound that
   would require devices to share raw data (which is precisely the
   privacy concern motivating FL).
3. **Trains the federated model with FedAvg** -- 20 communication
   rounds, 5 local epochs each, weighted aggregation by dataset size.
   No external FL library is used (it's about 80 lines of plain
   PyTorch).
4. **Evaluates 5 policies** on the held-out test set:
   - Federated FedAvg (the contribution),
   - Centralised model (upper bound),
   - Always-local heuristic,
   - Always-offload heuristic,
   - Random 50/50 heuristic.
5. **Plots 4 figures** into `figures/`.
6. **Prints a summary table.**

## Notes on system parameters

The brief's literal parameters (500 Mbps link with 100-300 MB tasks at
sub-second deadlines) make almost every action infeasible.  Two
constants in `config.py` are bumped to physically reasonable values so
that the dataset is non-trivial:

* `DATA_RATE`: 1.5 Gbps (was 500 Mbps) -- realistic UAV mmWave / Wi-Fi-6.
* `cpu_cycles` default range: `(5e8, 2e9)` (was `(1e9, 6e9)`) -- so
  that `T_local` overlaps with the deadlines.

Both edits are clearly commented in `config.py` and easy to revert.

## Reproducibility

Every script seeds `random`, NumPy and PyTorch with `RANDOM_SEED = 42`.
Two consecutive runs produce identical numbers.

## Per-device profiles (non-IID)

| Device | Role            | task_size_MB | deadline_s |
|--------|-----------------|--------------|------------|
| 0      | smart camera    | 200--300     | 0.3--0.8   |
| 1      | env sensor      | 50--120      | 1.0--3.0   |
| 2      | smart meter     | 10--50       | 2.0--5.0   |
| 3      | wearable        | 100--200     | 0.5--1.5   |
| 4      | second camera   | 180--280     | 0.4--1.0   |

Other features (`cpu_cycles`, `channel_gain`, `iot_battery`) follow the
default ranges for all devices.
