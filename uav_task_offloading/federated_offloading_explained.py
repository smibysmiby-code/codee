"""
Federated Learning - Step by step, with explanations.

Scenario
--------
We have a small fleet of IoT devices and one drone that flies over them.
Whenever a device gets a task, it must decide:

    * compute it LOCALLY  (use my own CPU/battery), OR
    * OFFLOAD it to the drone (the drone has more compute, but we use radio).

Each device has logged a few past situations + the decision that worked best.
We want to TRAIN a small neural network that predicts the right decision
from features like cpu_load, battery, task_size, network_quality, distance.

The catch: we do NOT want to send the raw sensor logs to the drone
(privacy + bandwidth). So we use FEDERATED LEARNING:

    - each device trains on its OWN data,
    - only the model WEIGHTS travel to the drone,
    - the drone AVERAGES the weights into a new "global" model (FedAvg),
    - it sends the global model back to the devices,
    - repeat for several rounds.

We compare it against a centralized baseline (drone collects all raw data),
just to see that FL gets close in accuracy while keeping data local.
"""

import copy
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)


# =============================================================
# Pretty-print helpers (just for the educational output)
# =============================================================
def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)

def step(num, title):
    print(f"\n--- STEP {num}: {title} ---")

def explain(text):
    for line in text.strip().splitlines():
        print("    " + line.strip())


# =============================================================
# 1) Build a fake dataset for each IoT device
# =============================================================
# Features: [cpu_load, battery, task_size, network_quality, distance]
# Each device's "best decision" follows a noisy rule:
#   offload when task is big, network is good, battery is low,
#   cpu is busy, and the drone is not too far.
def make_device_data(device_id, n=120):
    rng = np.random.RandomState(device_id)
    X = rng.rand(n, 5).astype(np.float32)
    # Heterogeneous data: each device has slightly different behavior.
    bias = rng.uniform(-0.1, 0.1)
    score = (0.40 * X[:, 2]      # task size  -> push to offload
             + 0.30 * X[:, 3]    # network ok -> push to offload
             - 0.25 * X[:, 1]    # battery    -> if high, keep local
             + 0.30 * X[:, 0]    # cpu busy   -> push to offload
             - 0.15 * X[:, 4]    # distance   -> if far, keep local
             + bias)
    y = (score > 0.45).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


# =============================================================
# 2) The tiny neural network: "local (0) vs offload (1)"
# =============================================================
class OffloadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================
# 3) Helpers: train, evaluate, average weights
# =============================================================
def train_one_model(model, X, y, epochs, lr=0.01):
    """Plain supervised training on (X, y)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    last_loss = None
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return model, last_loss

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
    model.train()
    return (pred == y).float().mean().item()

def average_weights(state_dicts):
    """
    FedAvg: average each parameter tensor across the devices.
    Equivalent to: new_w = (w_device1 + w_device2 + ... + w_deviceN) / N
    """
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        for i in range(1, len(state_dicts)):
            avg[k] = avg[k] + state_dicts[i][k]
        avg[k] = avg[k] / len(state_dicts)
    return avg

def weight_distance(sd_a, sd_b):
    """L2 distance between two state_dicts. Used to *visualize* how much
    a local model drifted from the global one during local training."""
    total = 0.0
    for k in sd_a.keys():
        total += torch.sum((sd_a[k] - sd_b[k]) ** 2).item()
    return total ** 0.5


# =============================================================
# 4) Build the world (devices + held-out test set)
# =============================================================
NUM_DEVICES = 5
SAMPLES_PER_DEVICE = 120

banner("FEDERATED LEARNING FOR IoT TASK OFFLOADING")
explain(f"""
We will simulate {NUM_DEVICES} IoT devices and 1 drone (the server).
Each device has {SAMPLES_PER_DEVICE} private samples it will NEVER share.
The drone holds a 'global model' and only sees model weights.
""")

step(1, "Generate private datasets for each device")
device_data = [make_device_data(i, n=SAMPLES_PER_DEVICE) for i in range(NUM_DEVICES)]
X_test, y_test = make_device_data(999, n=400)  # held-out test set

for i, (Xd, yd) in enumerate(device_data):
    pos = int(yd.sum().item())
    print(f"    Device {i}: {Xd.shape[0]} samples  "
          f"| offload={pos}  local={Xd.shape[0]-pos}")
print(f"    Test set : {X_test.shape[0]} samples (drone never trains on this)")
explain("""
Notice each device has a DIFFERENT class balance. This is "non-IID" data,
which is the realistic setting where federated learning matters most.
""")


# =============================================================
# 5) Centralized baseline (drone collects raw data) - for reference
# =============================================================
step(2, "Centralized baseline (privacy-violating, just for comparison)")
explain("""
Pretend every device uploads its raw features+labels to the drone.
The drone trains ONE model on the merged dataset.
This is the "best case" for accuracy but the worst case for privacy.
""")
X_all = torch.cat([d[0] for d in device_data])
y_all = torch.cat([d[1] for d in device_data])
central_model, central_loss = train_one_model(OffloadNet(), X_all, y_all, epochs=200)
acc_central = accuracy(central_model, X_test, y_test)
raw_bytes = X_all.numel() * 4 + y_all.numel() * 8
print(f"    Combined dataset size : {X_all.shape[0]} samples")
print(f"    Final training loss   : {central_loss:.4f}")
print(f"    Test accuracy         : {acc_central*100:.2f}%")
print(f"    Bytes 'sent' to drone : {raw_bytes} (raw features + labels)")


# =============================================================
# 6) Federated learning: round-by-round, with explanation
# =============================================================
banner("FEDERATED TRAINING (FedAvg)")
explain("""
In each round we will do 4 things:
  (a) BROADCAST: drone sends the current global model to every device.
  (b) LOCAL TRAIN: each device trains a few epochs on its OWN data.
  (c) UPLOAD: each device sends ONLY its updated weights back to the drone.
  (d) AGGREGATE: drone averages the weights -> new global model.

The raw data NEVER leaves the device.
""")

global_model = OffloadNet()           # initial random global model on the drone
NUM_ROUNDS = 15
LOCAL_EPOCHS = 5

# Count communication so we can compare with the centralized case at the end.
n_params = sum(p.numel() for p in global_model.parameters())
print(f"    Model size          : {n_params} parameters")
print(f"    Bytes per model copy: {n_params * 4} (float32)")
print(f"    Rounds              : {NUM_ROUNDS}")
print(f"    Local epochs/round  : {LOCAL_EPOCHS}")

acc0 = accuracy(global_model, X_test, y_test)
print(f"\n    Global model accuracy BEFORE training: {acc0*100:.2f}%  "
      "(random init, should be near 50%)")

total_fl_bytes = 0

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n  >>> Round {rnd}/{NUM_ROUNDS}")

    # (a) BROADCAST the global model to every device.
    global_state = copy.deepcopy(global_model.state_dict())
    broadcast_bytes = n_params * 4 * NUM_DEVICES   # drone -> every device
    print(f"     (a) Drone broadcasts global model to {NUM_DEVICES} devices "
          f"({broadcast_bytes} B down).")

    # (b) LOCAL training on each device.
    local_states = []
    local_losses = []
    drift = []
    for dev_id in range(NUM_DEVICES):
        local = OffloadNet()
        local.load_state_dict(copy.deepcopy(global_state))   # start from global
        X_dev, y_dev = device_data[dev_id]
        _, last_loss = train_one_model(local, X_dev, y_dev, epochs=LOCAL_EPOCHS)
        local_states.append(local.state_dict())
        local_losses.append(last_loss)
        drift.append(weight_distance(local.state_dict(), global_state))
    losses_str = ", ".join(f"{l:.3f}" for l in local_losses)
    drift_str  = ", ".join(f"{d:.3f}" for d in drift)
    print(f"     (b) Each device trains locally for {LOCAL_EPOCHS} epochs.")
    print(f"         final losses per device : [{losses_str}]")
    print(f"         weight drift from global: [{drift_str}]")

    # (c) UPLOAD: each device sends its weights up. (We just simulate.)
    upload_bytes = n_params * 4 * NUM_DEVICES
    print(f"     (c) Devices upload weights to drone ({upload_bytes} B up). "
          "Raw data stays on device.")

    # (d) AGGREGATE on the drone (FedAvg).
    new_global_state = average_weights(local_states)
    global_model.load_state_dict(new_global_state)
    print("     (d) Drone averages the weights -> new global model.")

    # Evaluate the new global model on the held-out test set.
    acc = accuracy(global_model, X_test, y_test)
    print(f"         global model test accuracy: {acc*100:5.2f}%")

    total_fl_bytes += broadcast_bytes + upload_bytes

acc_fed = accuracy(global_model, X_test, y_test)


# =============================================================
# 7) Summary: accuracy + communication cost + privacy
# =============================================================
banner("SUMMARY")
print(f"  Centralized accuracy : {acc_central*100:6.2f}%   "
      f"(needed {raw_bytes} B of RAW data)")
print(f"  Federated  accuracy  : {acc_fed*100:6.2f}%   "
      f"(needed {total_fl_bytes} B of WEIGHTS over {NUM_ROUNDS} rounds)")
print(f"  Federated  vs central: {(acc_fed - acc_central)*100:+.2f} percentage points")

explain("""
Take-aways

  1. The drone never saw a single sensor reading. Privacy is preserved.
  2. Federated accuracy is usually a bit below centralized, because each
     device only sees its own slice of the data, and FedAvg has to reconcile
     models that drifted in different directions (see the 'weight drift'
     numbers - they shrink as the global model stabilizes).
  3. Communication cost is proportional to (model size) x (devices) x (rounds),
     not to the dataset size. For small models on big data, FL is very cheap;
     for huge models on small data, FL is expensive - that is why people
     research things like model compression, client selection, and FedProx.

Knobs you can tune to learn more
  - LOCAL_EPOCHS:  more local epochs = less communication, but more drift.
  - NUM_ROUNDS:    more rounds = more accuracy, but more bandwidth.
  - NUM_DEVICES:   try 2 vs 20 devices and watch the accuracy curve change.
  - The bias term in make_device_data controls how non-IID the data is;
    increase it to make federation harder.
""")
