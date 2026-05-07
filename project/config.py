"""
config.py
=========

Centralised configuration for the federated learning prototype.

All constants (system parameters, hyper-parameters, file paths, and
non-IID distributions per IoT device) are kept in one place so that the
rest of the codebase stays clean and the experiment is easy to tweak.

Reference:
    "Federated Learning for Task Offloading in UAV-assisted Edge Networks"
    -- a research prototype, single UAV, no cloud.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# A single random seed is reused everywhere (numpy, torch, python `random`)
# so that two runs of main.py produce identical numbers.
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# System model parameters
# ---------------------------------------------------------------------------
# We have N IoT devices on the ground and 1 UAV hovering above.  The UAV is
# both an *edge server* (it can execute offloaded tasks) and a *federated
# aggregator* (it averages model updates from all IoT devices).
NUM_DEVICES: int = 5

# CPU frequencies (Hz)
IOT_CPU_FREQ: float = 1e9   # 1 GHz   -- weak ground device
UAV_CPU_FREQ: float = 4e9   # 4 GHz   -- powerful UAV edge server

# Wireless link parameters
TRANSMISSION_POWER: float = 0.5                 # Watts
# NOTE on data rate
# ----------------
# The original brief specified `data_rate = 500e6 / 8` B/s (i.e. 500 Mbps).
# Combined with the per-device task sizes (up to ~300 MB) and tight
# deadlines (down to 0.3 s), almost no offload action could meet the
# deadline (transmitting 300 MB at 500 Mbps already takes 4.8 s).  We
# bump the link to 1.5 Gbps -- a realistic UAV mmWave / Wi-Fi-6 figure
# -- which yields a near 50/50 global label split AND interesting
# non-IID behaviour: bandwidth-bound devices (cameras) prefer local
# execution, compute-bound devices (sensors / meters) prefer offload.
DATA_RATE: float = 1.5e9 / 8                    # bytes per second (1.5 Gbps)

# CPU energy coefficient (Joules per cycle^3) -- standard DVFS model
KAPPA: float = 1e-27

# Cost-function weights -- "J = mu * E + eta * T"
MU: float = 0.5    # weight on energy
ETA: float = 0.5   # weight on delay


# ---------------------------------------------------------------------------
# Default ranges for task features (used unless overridden per device)
# ---------------------------------------------------------------------------
# Each task is described by 5 floats.  These ranges are also used when the
# evaluation script generates a *mixed-distribution* test set.
# NOTE on cpu_cycles
# ------------------
# The brief specified `cpu_cycles in (1e9, 6e9)`, which makes T_local
# range from 1s to 6s on a 1 GHz IoT CPU.  Several device deadlines are
# tighter than 1s, so local execution would *always* be infeasible
# there.  We narrow the range to (5e8, 2e9) so that local execution is
# sometimes the better choice -- otherwise the dataset would be 100%
# offload labels and the FL classifier would be trivial.
DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
    "task_size_MB":  (100.0, 300.0),
    "cpu_cycles":    (5e8,   2e9),
    "deadline_s":    (0.5,   2.0),
    "channel_gain":  (0.2,   1.0),
    "iot_battery":   (0.3,   1.0),
}


# ---------------------------------------------------------------------------
# Per-device non-IID overrides
# ---------------------------------------------------------------------------
# Each IoT device represents a different real-world sensor / appliance.  We
# only override `task_size_MB` and `deadline_s` -- the other features keep
# the default ranges defined above.  This produces non-IID local datasets
# which is the whole point of using federated learning.
DEVICE_PROFILES: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {  # smart camera (large frames, tight deadlines)
        "task_size_MB": (200.0, 300.0),
        "deadline_s":   (0.3,   0.8),
    },
    1: {  # environmental sensor (small payload, relaxed deadlines)
        "task_size_MB": (50.0,  120.0),
        "deadline_s":   (1.0,   3.0),
    },
    2: {  # smart meter (very small payload, very relaxed deadlines)
        "task_size_MB": (10.0,  50.0),
        "deadline_s":   (2.0,   5.0),
    },
    3: {  # wearable (moderate payload and deadline)
        "task_size_MB": (100.0, 200.0),
        "deadline_s":   (0.5,   1.5),
    },
    4: {  # second camera (similar to device 0 but slightly different)
        "task_size_MB": (180.0, 280.0),
        "deadline_s":   (0.4,   1.0),
    },
}


# ---------------------------------------------------------------------------
# Dataset / training hyper-parameters
# ---------------------------------------------------------------------------
TASKS_PER_DEVICE: int = 1000        # how many labelled tasks per CSV
TEST_SET_SIZE: int = 500            # mixed-distribution evaluation set

# Centralised baseline
CENTRALIZED_EPOCHS: int = 50
CENTRALIZED_TEST_SPLIT: float = 0.2  # 80/20 train/test

# Federated training
FED_ROUNDS: int = 20                 # communication rounds
FED_LOCAL_EPOCHS: int = 5            # local epochs per round
FED_BATCH_SIZE: int = 32

# Optimiser
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Feature normalisation bounds (for min-max scaling of NN inputs)
# ---------------------------------------------------------------------------
# Inputs to the MLP are scaled into [0, 1] using these *global* bounds, so
# that all devices and the centralised model agree on the feature scale.
# Bounds are taken as the union of per-device ranges (i.e. min/max across
# the whole system).
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "task_size_MB": (10.0,  300.0),   # widest device-specific range
    "cpu_cycles":   (1e9,   6e9),
    "deadline_s":   (0.3,   5.0),     # widest deadline range
    "channel_gain": (0.2,   1.0),
    "iot_battery":  (0.3,   1.0),
}

# Feature column order used everywhere (CSV columns, NN input vector, ...)
FEATURE_COLUMNS = [
    "task_size_MB",
    "cpu_cycles",
    "deadline_s",
    "channel_gain",
    "iot_battery",
]
LABEL_COLUMN = "label"


# ---------------------------------------------------------------------------
# File-system layout
# ---------------------------------------------------------------------------
PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str    = os.path.join(PROJECT_DIR, "data")
FIG_DIR: str     = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR: str   = os.path.join(PROJECT_DIR, "models")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def device_csv_path(device_id: int) -> str:
    """Return the absolute path of the CSV file for device `device_id`."""
    return os.path.join(DATA_DIR, f"device_{device_id}.csv")


def test_csv_path() -> str:
    """Path of the mixed-distribution evaluation CSV."""
    return os.path.join(DATA_DIR, "test_set.csv")
