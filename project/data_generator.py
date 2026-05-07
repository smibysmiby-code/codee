"""
data_generator.py
=================

Generate per-device CSV datasets for the federated learning experiment.

For each of the 5 IoT devices we sample `TASKS_PER_DEVICE` feasible
labelled tasks (using the device-specific non-IID feature ranges from
`config.DEVICE_PROFILES`) and write a CSV file `data/device_<id>.csv`.

We also generate a *mixed-distribution* test set used by `evaluate.py`
to compare federated vs centralised vs heuristic baselines on
out-of-distribution-ish data.

Run directly to regenerate everything:

    python data_generator.py
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

import config
from simulator import sample_labelled_task


def _seed_everything(seed: int) -> None:
    """Seed Python `random` and NumPy for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _generate_dataframe(num_tasks: int,
                        rng: np.random.Generator,
                        device_id: Optional[int]) -> pd.DataFrame:
    """Sample `num_tasks` labelled tasks and return them as a DataFrame.

    `device_id=None` means "use the default (mixed) feature ranges".
    """
    rows = []
    for _ in range(num_tasks):
        task, label = sample_labelled_task(rng, device_id=device_id)
        rows.append({
            "task_size_MB": task.task_size_MB,
            "cpu_cycles":   task.cpu_cycles,
            "deadline_s":   task.deadline_s,
            "channel_gain": task.channel_gain,
            "iot_battery":  task.iot_battery,
            "label":        label,
        })
    return pd.DataFrame(rows)


def generate_device_datasets() -> None:
    """Generate one CSV per IoT device (the federated training data)."""
    _seed_everything(config.RANDOM_SEED)
    rng = np.random.default_rng(config.RANDOM_SEED)

    print(f"[data] Generating {config.TASKS_PER_DEVICE} tasks per device "
          f"({config.NUM_DEVICES} devices)...")

    overall_local, overall_offload = 0, 0
    for dev in range(config.NUM_DEVICES):
        df = _generate_dataframe(config.TASKS_PER_DEVICE, rng, device_id=dev)
        path = config.device_csv_path(dev)
        df.to_csv(path, index=False)

        n_local   = int((df[config.LABEL_COLUMN] == 0).sum())
        n_offload = int((df[config.LABEL_COLUMN] == 1).sum())
        overall_local   += n_local
        overall_offload += n_offload
        print(f"  device {dev}: {len(df):4d} tasks  "
              f"local={n_local:4d}  offload={n_offload:4d}  "
              f"-> {path}")

    total = overall_local + overall_offload
    print(f"[data] Overall label split: "
          f"local={overall_local} ({100*overall_local/total:.1f}%)  "
          f"offload={overall_offload} ({100*overall_offload/total:.1f}%)")


def generate_test_set() -> None:
    """Generate a single CSV with mixed-distribution evaluation tasks.

    Tasks are sampled with `device_id=None`, i.e. using the *default*
    (system-wide) feature ranges, so that the test set isn't biased
    toward any particular device's profile.
    """
    # Use a *different* seed offset so the test set doesn't collide with
    # the training data sampler's number stream.
    rng = np.random.default_rng(config.RANDOM_SEED + 1000)

    df = _generate_dataframe(config.TEST_SET_SIZE, rng, device_id=None)
    path = config.test_csv_path()
    df.to_csv(path, index=False)

    n_local   = int((df[config.LABEL_COLUMN] == 0).sum())
    n_offload = int((df[config.LABEL_COLUMN] == 1).sum())
    print(f"[data] Test set: {len(df)} tasks  "
          f"local={n_local} offload={n_offload}  -> {path}")


def main() -> None:
    generate_device_datasets()
    generate_test_set()


if __name__ == "__main__":
    main()
