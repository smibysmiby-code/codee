"""UAV-MEC Environment for Task Offloading.

Single UAV hovering at fixed altitude with N ground users. Each user has a
6-dim state and a binary action (local vs offload). Reward is negative
weighted cost of (delay + energy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical / system constants
# ---------------------------------------------------------------------------

UAV_ALTITUDE_M = 100.0
COVERAGE_RADIUS_M = 50.0

# Bandwidth and channel parameters
BANDWIDTH_HZ = 1e6              # 1 MHz per user
NOISE_POWER_DBM = -100.0        # noise power
TX_POWER_DBM = 23.0             # transmit power (200 mW)
PATH_LOSS_REF_DB = 30.0         # path loss @ 1m reference
PATH_LOSS_EXP = 2.5

# Compute parameters
UAV_CPU_HZ = 5e9                # 5 GHz UAV CPU (shared)
USER_CPU_LOW = 0.5e9            # 0.5 GHz minimum
USER_CPU_HIGH = 1.5e9           # 1.5 GHz maximum
KAPPA = 1e-27                   # effective switched capacitance for CPU energy

# Task parameters
TASK_SIZE_LOW_MB = 0.1
TASK_SIZE_HIGH_MB = 1.0
CPU_CYCLES_PER_BIT = 200.0      # cycles required per data bit


def _dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


@dataclass
class EnvConfig:
    num_users: int = 50
    omega_delay: float = 0.5
    omega_energy: float = 0.5
    max_steps: int = 100
    seed: Optional[int] = None


class UAVMECEnv:
    """OpenAI-Gym compatible UAV-MEC environment.

    Observation: shape (num_users, 6)
        [task_size_MB, cpu_cycles_G, uav_load, user_cpu_GHz, distance_m, snr_dB]

    Action: shape (num_users,), entries in {0, 1}
        0 = local execution
        1 = offload to UAV
    """

    state_dim = 6

    def __init__(self, config: Optional[EnvConfig] = None, **overrides):
        self.config = config or EnvConfig()
        for k, v in overrides.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.rng = np.random.default_rng(self.config.seed)

        self._step = 0
        self._uav_load = 0.0
        self._positions = None
        self._user_cpu = None
        self._tasks = None  # (size_MB, cycles)
        self._reset_internal()

    # ------------------------------------------------------------------
    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def _reset_internal(self):
        n = self.config.num_users
        angles = self.rng.uniform(0, 2 * math.pi, n)
        radii = COVERAGE_RADIUS_M * np.sqrt(self.rng.uniform(0, 1, n))
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        self._positions = np.stack([xs, ys], axis=1)
        self._user_cpu = self.rng.uniform(USER_CPU_LOW, USER_CPU_HIGH, n)
        self._uav_load = 0.0
        self._step = 0
        self._sample_tasks()

    def _sample_tasks(self):
        n = self.config.num_users
        sizes_mb = self.rng.uniform(TASK_SIZE_LOW_MB, TASK_SIZE_HIGH_MB, n)
        size_bits = sizes_mb * 8 * 1e6
        cycles = size_bits * CPU_CYCLES_PER_BIT
        self._tasks = (sizes_mb, cycles)

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        sizes_mb, cycles = self._tasks
        dist = np.sqrt((self._positions ** 2).sum(axis=1) + UAV_ALTITUDE_M ** 2)
        path_loss_db = PATH_LOSS_REF_DB + 10.0 * PATH_LOSS_EXP * np.log10(np.maximum(dist, 1.0))
        snr_db = TX_POWER_DBM - path_loss_db - NOISE_POWER_DBM
        load = np.full(self.config.num_users, self._uav_load)
        obs = np.stack(
            [
                sizes_mb,
                cycles / 1e9,
                load,
                self._user_cpu / 1e9,
                dist,
                snr_db,
            ],
            axis=1,
        ).astype(np.float32)
        return obs

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        return self._obs()

    # ------------------------------------------------------------------
    def _compute_costs(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sizes_mb, cycles = self._tasks
        size_bits = sizes_mb * 8 * 1e6
        dist = np.sqrt((self._positions ** 2).sum(axis=1) + UAV_ALTITUDE_M ** 2)
        path_loss_db = PATH_LOSS_REF_DB + 10.0 * PATH_LOSS_EXP * np.log10(np.maximum(dist, 1.0))
        rx_dbm = TX_POWER_DBM - path_loss_db
        rx_w = _dbm_to_w(rx_dbm)
        noise_w = _dbm_to_w(NOISE_POWER_DBM)
        snr_lin = rx_w / noise_w
        rate = BANDWIDTH_HZ * np.log2(1.0 + snr_lin)  # bits/s

        # Local
        local_delay = cycles / np.maximum(self._user_cpu, 1.0)
        local_energy = KAPPA * cycles * (self._user_cpu ** 2)

        # Offload
        comm_delay = size_bits / np.maximum(rate, 1.0)
        offload_mask = (actions == 1)
        n_off = max(int(offload_mask.sum()), 1)
        per_uav_cpu = UAV_CPU_HZ / n_off
        uav_delay = cycles / per_uav_cpu
        tx_energy = _dbm_to_w(TX_POWER_DBM) * comm_delay
        offload_delay = comm_delay + uav_delay
        offload_energy = tx_energy  # user-side energy only

        delay = np.where(offload_mask, offload_delay, local_delay)
        energy = np.where(offload_mask, offload_energy, local_energy)

        # Update UAV load (fraction of capacity utilized this step)
        self._uav_load = float(offload_mask.mean())

        return delay, energy

    def step(self, actions) -> Tuple[np.ndarray, float, bool, dict]:
        actions = np.asarray(actions).astype(np.int64).reshape(-1)
        delay, energy = self._compute_costs(actions)
        cfg = self.config

        # Normalize per-user costs to roughly O(1) before weighting
        norm_delay = delay / 5.0
        norm_energy = energy / 5.0
        cost = cfg.omega_delay * norm_delay + cfg.omega_energy * norm_energy
        reward = float(-cost.mean())

        self._step += 1
        self._sample_tasks()
        done = self._step >= cfg.max_steps
        info = {
            "delay": float(delay.mean()),
            "energy": float(energy.mean()),
            "system_cost": float(cost.mean()),
            "offload_ratio": float((actions == 1).mean()),
        }
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    @property
    def num_users(self) -> int:
        return self.config.num_users

    @property
    def network_context(self) -> np.ndarray:
        sizes_mb, _ = self._tasks
        return np.array(
            [
                self.config.num_users / 50.0,
                float(sizes_mb.mean()) / TASK_SIZE_HIGH_MB,
                self._uav_load,
                self._step / self.config.max_steps,
            ],
            dtype=np.float32,
        )
