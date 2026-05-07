"""
simulator.py
============

Simulates a single IoT task in the UAV-assisted edge-computing system and
produces the optimal binary offloading label (0 = local, 1 = offload).

The math comes directly from the system model in the project brief.

We expose three top-level helpers:

    * `compute_local_cost(...)`   -- delay & energy if executed locally
    * `compute_offload_cost(...)` -- delay & energy if offloaded to UAV
    * `optimal_label(...)`        -- pick the lower-cost feasible option

A `Task` namedtuple keeps the 5 raw features grouped together to make the
rest of the code easier to read.
"""

from __future__ import annotations

import random
from typing import NamedTuple, Optional, Tuple, Dict

import numpy as np

from config import (
    IOT_CPU_FREQ,
    UAV_CPU_FREQ,
    TRANSMISSION_POWER,
    DATA_RATE,
    KAPPA,
    MU,
    ETA,
    DEFAULT_RANGES,
    DEVICE_PROFILES,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class Task(NamedTuple):
    """Raw features of a single computation task."""
    task_size_MB: float    # input data size in megabytes
    cpu_cycles: float      # CPU cycles required to process it
    deadline_s: float      # latency deadline in seconds
    channel_gain: float    # normalised wireless channel gain (0..1)
    iot_battery: float     # remaining battery fraction (0..1)


# ---------------------------------------------------------------------------
# Cost / latency / energy equations
# ---------------------------------------------------------------------------
def compute_local_cost(task: Task) -> Tuple[float, float, float]:
    """Compute (delay, energy, cost J) when the IoT device runs the task itself.

    * Delay  : T_local = cpu_cycles / iot_cpu_freq
    * Energy : E_local = kappa * cpu_cycles * (iot_cpu_freq ** 2)
    * Cost   : J = mu * E + eta * T
    """
    t_local = task.cpu_cycles / IOT_CPU_FREQ
    e_local = KAPPA * task.cpu_cycles * (IOT_CPU_FREQ ** 2)
    j_local = MU * e_local + ETA * t_local
    return t_local, e_local, j_local


def compute_offload_cost(task: Task) -> Tuple[float, float, float]:
    """Compute (delay, energy, cost J) when the task is offloaded to the UAV.

    * T_transmit = (size_MB * 1e6) / data_rate         # bytes / (B/s)
    * T_compute  = cpu_cycles / uav_cpu_freq
    * T_offload  = T_transmit + T_compute
    * E_offload  = transmission_power * T_transmit     # only the radio
                                                       # consumes energy
                                                       # on the IoT side
    * Cost       : J = mu * E + eta * T
    """
    t_tx       = (task.task_size_MB * 1e6) / DATA_RATE
    t_compute  = task.cpu_cycles / UAV_CPU_FREQ
    t_offload  = t_tx + t_compute
    e_offload  = TRANSMISSION_POWER * t_tx
    j_offload  = MU * e_offload + ETA * t_offload
    return t_offload, e_offload, j_offload


def optimal_label(task: Task) -> Optional[int]:
    """Return the optimal action (0 = local, 1 = offload) for `task`.

    Logic:
        * If both options meet the deadline -> pick the lower-cost option.
        * If only one meets the deadline    -> pick that one.
        * If neither meets the deadline     -> return None  (caller should
                                               discard the task and resample).
    """
    t_loc,  _, j_loc = compute_local_cost(task)
    t_off,  _, j_off = compute_offload_cost(task)

    local_ok   = t_loc <= task.deadline_s
    offload_ok = t_off <= task.deadline_s

    if local_ok and offload_ok:
        return 0 if j_loc <= j_off else 1
    if local_ok:
        return 0
    if offload_ok:
        return 1
    return None  # infeasible: caller resamples


# ---------------------------------------------------------------------------
# Random task generation
# ---------------------------------------------------------------------------
def _ranges_for_device(device_id: Optional[int]) -> Dict[str, Tuple[float, float]]:
    """Merge the per-device overrides on top of the default feature ranges."""
    ranges = dict(DEFAULT_RANGES)
    if device_id is not None and device_id in DEVICE_PROFILES:
        ranges.update(DEVICE_PROFILES[device_id])
    return ranges


def sample_task(rng: np.random.Generator,
                device_id: Optional[int] = None) -> Task:
    """Sample one random `Task`.  If `device_id` is None, default ranges are
    used (this is what the mixed-distribution test set does)."""
    r = _ranges_for_device(device_id)
    return Task(
        task_size_MB = float(rng.uniform(*r["task_size_MB"])),
        cpu_cycles   = float(rng.uniform(*r["cpu_cycles"])),
        deadline_s   = float(rng.uniform(*r["deadline_s"])),
        channel_gain = float(rng.uniform(*r["channel_gain"])),
        iot_battery  = float(rng.uniform(*r["iot_battery"])),
    )


def sample_labelled_task(rng: np.random.Generator,
                         device_id: Optional[int] = None,
                         max_attempts: int = 1000) -> Tuple[Task, int]:
    """Sample tasks until one is *feasible* (optimal label is 0 or 1) and
    return the (task, label) pair.  We resample (rather than try harder)
    because some random draws have neither option meeting the deadline."""
    for _ in range(max_attempts):
        task = sample_task(rng, device_id=device_id)
        label = optimal_label(task)
        if label is not None:
            return task, label
    raise RuntimeError(
        f"Could not sample a feasible task for device {device_id} "
        f"after {max_attempts} attempts -- check your config!"
    )


# ---------------------------------------------------------------------------
# Quick smoke-test (runs only when this file is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import config
    random.seed(config.RANDOM_SEED)
    rng = np.random.default_rng(config.RANDOM_SEED)
    for dev in range(config.NUM_DEVICES):
        t, y = sample_labelled_task(rng, device_id=dev)
        print(f"device={dev}  label={y}  task={t}")
