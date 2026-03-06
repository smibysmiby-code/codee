#!/usr/bin/env python3
"""
================================================================================
DUELING DQN FOR UAV TASK OFFLOADING
================================================================================
Per-Device Binary Decision Architecture -- Scales to any number of devices.
Dueling Double DQN separates state-value and advantage streams for better
value estimation in binary task offloading for UAV-assisted edge computing.

To run: python 2_dueling_dqn.py
Dependencies: pip install numpy matplotlib torch
================================================================================
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# =============================================================
#  REPRODUCIBILITY
# =============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================
#  SYSTEM PARAMETERS
# =============================================================
SLOT_DURATION  = 0.1
TASK_PROB      = 0.7
N_DEVICES      = 30
N_SLOTS        = 300
N_EPISODES     = 2000
F_LOCAL        = 0.5e9
F_UAV          = 5e9
B              = 0.5e6
N0             = 1e-10
P_TX           = 0.1
TASK_SIZE_MIN  = 0.1e6
TASK_SIZE_MAX  = 0.5e6
CYCLES_MIN     = 1e6
CYCLES_MAX     = 80e6
FEATURES_PER_DEVICE = 6

UAV_POS = np.array([0, 0, 100])
DEVICE_POSITIONS = [
    np.array([50,   50,  0]), np.array([100,  30,  0]),
    np.array([150, 120,  0]), np.array([80,  150,  0]),
    np.array([120,  80,  0]), np.array([200,  60,  0]),
    np.array([60,  200,  0]), np.array([180, 170,  0]),
    np.array([30,  100,  0]), np.array([250, 100,  0]),
    np.array([40,  180,  0]), np.array([160,  40,  0]),
    np.array([220, 140,  0]), np.array([90,  220,  0]),
    np.array([270,  50,  0]), np.array([130, 190,  0]),
    np.array([190, 110,  0]), np.array([70,   70,  0]),
    np.array([240, 200,  0]), np.array([110, 130,  0]),
    np.array([300,  80,  0]), np.array([20,  250,  0]),
    np.array([280, 180,  0]), np.array([140, 260,  0]),
    np.array([320, 140,  0]), np.array([50,  280,  0]),
    np.array([260, 240,  0]), np.array([180, 280,  0]),
    np.array([330,  30,  0]), np.array([100, 300,  0]),
]

DROP_PENALTY = 0.15
DISTANCES    = [np.linalg.norm(UAV_POS - dp) for dp in DEVICE_POSITIONS]
SNRS         = [P_TX / (N0 * d ** 2) for d in DISTANCES]
RATES        = [B * np.log2(1 + snr) for snr in SNRS]

# =============================================================
#  ENVIRONMENT
# =============================================================
class UAVEnvironment:
    def __init__(self):
        self.state_size = N_DEVICES * FEATURES_PER_DEVICE + 1

    def reset(self):
        self.uav_queue_time = 0.0
        self.current_tasks  = self._generate_tasks()
        return self._get_state(self.current_tasks)

    def _generate_tasks(self):
        tasks = []
        for _ in range(N_DEVICES):
            if random.random() < TASK_PROB:
                D = random.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)
                C = random.uniform(CYCLES_MIN, CYCLES_MAX)
                tasks.append((D, C))
            else:
                tasks.append(None)
        return tasks

    def _get_state(self, tasks=None):
        state = []
        for i in range(N_DEVICES):
            if tasks and tasks[i] is not None:
                D, C = tasks[i]
                local_time = C / F_LOCAL
                state.append(D / TASK_SIZE_MAX)
                state.append(C / CYCLES_MAX)
                state.append(DISTANCES[i] / 400)
                state.append(1.0)
                state.append(min(np.log10(SNRS[i] + 1) / 10.0, 1.0))
                state.append(1.0 if local_time <= SLOT_DURATION else 0.0)
            else:
                state.append(0.0)
                state.append(0.0)
                state.append(DISTANCES[i] / 400)
                state.append(0.0)
                state.append(min(np.log10(SNRS[i] + 1) / 10.0, 1.0))
                state.append(1.0)
        state.append(min(self.uav_queue_time / SLOT_DURATION, 1.0))
        return np.array(state, dtype=np.float32)

    def step(self, decisions):
        tasks = self.current_tasks
        total_latency = 0.0
        info = []
        for i in range(N_DEVICES):
            if tasks[i] is None:
                info.append({"device": i, "action": "no task",
                             "latency": 0, "dropped": False})
                continue
            D, C = tasks[i]
            if decisions[i] == 0:
                latency = C / F_LOCAL
                dropped = latency > SLOT_DURATION
                if dropped:
                    latency = DROP_PENALTY
                info.append({"device": i, "action": "local",
                             "latency": latency, "dropped": dropped})
            else:
                t_upload = D / RATES[i]
                t_wait   = max(0.0, self.uav_queue_time)
                t_exec   = C / F_UAV
                latency  = t_upload + t_wait + t_exec
                self.uav_queue_time += t_exec
                dropped  = latency > SLOT_DURATION
                if dropped:
                    latency = DROP_PENALTY
                info.append({"device": i, "action": "offload",
                             "latency": latency, "dropped": dropped})
            total_latency += latency
        self.uav_queue_time = max(0.0, self.uav_queue_time - SLOT_DURATION)
        reward             = -total_latency
        self.current_tasks = self._generate_tasks()
        next_state         = self._get_state(self.current_tasks)
        return next_state, reward, total_latency, info

# =============================================================
#  DUELING PER-DEVICE DQN NETWORK
# =============================================================
class DuelingPerDeviceDQNNet(nn.Module):
    """
    Dueling architecture: separates Q(s,a) into V(s) + A(s,a).
    The trunk produces a shared representation. Then for each device:
      - A value stream estimates V(s) for that device context
      - An advantage stream estimates A(s,a) for local vs offload
      - Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    This helps the network learn which states are valuable regardless
    of the action, leading to better policy evaluation.
    """
    def __init__(self, state_size):
        super().__init__()
        self.n_devices = N_DEVICES

        # Shared trunk processes global state
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Value stream: V(s) -- single scalar per device
        self.value_stream = nn.Sequential(
            nn.Linear(256 + FEATURES_PER_DEVICE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream: A(s,a) -- 2 values per device (local, offload)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256 + FEATURES_PER_DEVICE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        batch = x.shape[0]
        features = self.trunk(x)

        # Extract per-device features
        device_feats = x[:, :self.n_devices * FEATURES_PER_DEVICE]
        device_feats = device_feats.view(batch, self.n_devices,
                                         FEATURES_PER_DEVICE)

        # Expand trunk features for each device
        features_exp = features.unsqueeze(1).expand(-1, self.n_devices, -1)
        combined = torch.cat([features_exp, device_feats], dim=2)
        flat = combined.reshape(batch * self.n_devices, -1)

        # Dueling: V(s) + A(s,a) - mean(A)
        value     = self.value_stream(flat)       # (batch*N, 1)
        advantage = self.advantage_stream(flat)    # (batch*N, 2)

        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q.view(batch, self.n_devices, 2)

# =============================================================
#  DUELING DQN AGENT
# =============================================================
class DuelingDQNAgent:
    def __init__(self, state_size):
        self.state_size    = state_size
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.99995
        self.tau           = 0.005
        self.lr            = 0.0003
        self.batch_size    = 128
        self.grad_clip     = 10.0
        self.memory        = deque(maxlen=50000)

        self.model        = DuelingPerDeviceDQNNet(state_size).to(DEVICE)
        self.target_model = DuelingPerDeviceDQNNet(state_size).to(DEVICE)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.lr,
                                       weight_decay=1e-5)
        self.loss_fn      = nn.SmoothL1Loss()
        self._hard_update_target()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {n_params:,}")

    def _hard_update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        for tp, op in zip(self.target_model.parameters(),
                          self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def act(self, state):
        if random.random() < self.epsilon:
            return [random.randint(0, 1) for _ in range(N_DEVICES)]
        with torch.no_grad():
            q = self.model(
                torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
        return q.squeeze(0).argmax(dim=1).cpu().tolist()

    def remember(self, s, a, r, s2):
        self.memory.append((s, a, r, s2))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch       = random.sample(self.memory, self.batch_size)
        s, a, r, s2 = zip(*batch)
        s  = torch.FloatTensor(np.array(s)).to(DEVICE)
        a  = torch.LongTensor(np.array(a)).to(DEVICE)
        r  = torch.FloatTensor(r).to(DEVICE)
        s2 = torch.FloatTensor(np.array(s2)).to(DEVICE)

        q_all     = self.model(s)
        current_q = q_all.gather(2, a.unsqueeze(2)).squeeze(2)
        current_q_total = current_q.sum(dim=1)

        with torch.no_grad():
            # Double DQN: online selects, target evaluates
            q_next_online  = self.model(s2)
            best_actions   = q_next_online.argmax(dim=2)
            q_next_target  = self.target_model(s2)
            next_q = q_next_target.gather(
                2, best_actions.unsqueeze(2)).squeeze(2)
            next_q_total = next_q.sum(dim=1)

        target_q = r + self.gamma * next_q_total
        loss = self.loss_fn(current_q_total, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self._soft_update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# =============================================================
#  BASELINES
# =============================================================
def run_baseline(name, action_fn):
    random.seed(SEED)
    np.random.seed(SEED)
    latencies, rewards = [], []
    for ep in range(N_EPISODES):
        env_b = UAVEnvironment()
        env_b.reset()
        ep_lat, ep_rew = [], 0.0
        for slot in range(N_SLOTS):
            _, reward, total_lat, info = env_b.step(action_fn())
            ep_rew += reward
            active = [x for x in info if x["action"] != "no task"]
            if active:
                ep_lat.append(total_lat / len(active))
        latencies.append(np.mean(ep_lat) if ep_lat else 0)
        rewards.append(ep_rew)
    print(f"  {name}: {np.mean(latencies):.4f}s")
    return latencies, rewards

# =============================================================
#  TRAINING
# =============================================================
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env   = UAVEnvironment()
    agent = DuelingDQNAgent(env.state_size)

    latencies, drops, rewards, losses = [], [], [], []

    print(f"\n{'=' * 70}")
    print(f"  DUELING DQN -- {N_DEVICES} Devices, 1 UAV, {N_EPISODES} Episodes")
    print(f"  Per-device binary decisions (VDN decomposition)")
    print(f"  Dueling streams: V(s) + A(s,a) - mean(A)")
    print(f"{'=' * 70}")
    print(f"{'Episode':>8} | {'Avg Latency':>12} | {'Drop Rate':>10} | "
          f"{'Avg Loss':>10} | {'Epsilon':>8}")
    print("-" * 70)

    for ep in range(N_EPISODES):
        state      = env.reset()
        ep_latency = []
        ep_drops   = []
        ep_reward  = 0.0
        ep_losses  = []

        for slot in range(N_SLOTS):
            decisions = agent.act(state)
            next_state, reward, total_lat, info = env.step(decisions)
            agent.remember(state, decisions, reward, next_state)
            loss_val = agent.learn()
            state      = next_state
            ep_reward += reward
            if loss_val is not None:
                ep_losses.append(loss_val)
            active = [x for x in info if x["action"] != "no task"]
            if active:
                ep_latency.append(total_lat / len(active))
                ep_drops.append(
                    sum(1 for x in active if x["dropped"]) / len(active))

        avg_lat  = np.mean(ep_latency) if ep_latency else 0
        avg_drop = np.mean(ep_drops)   if ep_drops   else 0
        avg_loss = np.mean(ep_losses)  if ep_losses  else 0
        latencies.append(avg_lat)
        drops.append(avg_drop)
        rewards.append(ep_reward)
        losses.append(avg_loss)

        if ep % 100 == 0:
            print(f"{ep:>8} | {avg_lat:>12.4f}s | {avg_drop:>9.1%} | "
                  f"{avg_loss:>10.4f} | {agent.epsilon:>8.3f}")

    print(f"\n  Dueling DQN complete!")
    print(f"   Last 100 ep avg latency: {np.mean(latencies[-100:]):.4f}s")
    return latencies, drops, rewards, losses

# =============================================================
#  MAIN
# =============================================================
if __name__ == "__main__":
    lat, drops, rew, losses = train()

    # Baselines
    print("\nRunning baselines...")
    local_lat, local_rew = run_baseline("All Local",
        lambda: [0] * N_DEVICES)
    offload_lat, offload_rew = run_baseline("All Offload",
        lambda: [1] * N_DEVICES)
    random_lat, random_rew = run_baseline("Random",
        lambda: [random.randint(0, 1) for _ in range(N_DEVICES)])

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  FINAL RESULTS -- Dueling DQN")
    print(f"{'=' * 50}")
    results = [
        ("Dueling DQN", np.mean(lat[-100:])),
        ("Random",      np.mean(random_lat[-100:])),
        ("All Offload", np.mean(offload_lat[-100:])),
        ("All Local",   np.mean(local_lat[-100:])),
    ]
    best = min(r[1] for r in results)
    for name, val in results:
        marker = " <- BEST" if val == best else f"  ({(val-best)/best*100:.1f}% worse)"
        print(f"  {name:<18}: {val:.4f}s{marker}")

    # Plot
    os.makedirs('outputs', exist_ok=True)
    window = 30
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    x_range = range(window - 1, N_EPISODES)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dueling DQN -- UAV Task Offloading\n"
                 f"{N_DEVICES} Devices, {N_EPISODES} Episodes",
                 fontsize=14, fontweight='bold')

    # Reward
    axes[0, 0].plot(rew, alpha=0.15, color='blue')
    axes[0, 0].plot(x_range, moving_avg(rew, window),
                    color='blue', linewidth=2.5, label='Dueling DQN')
    axes[0, 0].axhline(np.mean(local_rew), color='red', linestyle='--',
                       label='All Local')
    axes[0, 0].axhline(np.mean(offload_rew), color='green', linestyle='-.',
                       label='All Offload')
    axes[0, 0].axhline(np.mean(random_rew), color='orange', linestyle=':',
                       label='Random')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Latency
    axes[0, 1].plot(lat, alpha=0.15, color='blue')
    axes[0, 1].plot(x_range, moving_avg(lat, window),
                    color='blue', linewidth=2.5, label='Dueling DQN')
    axes[0, 1].axhline(np.mean(local_lat), color='red', linestyle='--',
                       label='All Local')
    axes[0, 1].axhline(np.mean(offload_lat), color='green', linestyle='-.',
                       label='All Offload')
    axes[0, 1].axhline(np.mean(random_lat), color='orange', linestyle=':',
                       label='Random')
    axes[0, 1].axhline(SLOT_DURATION, color='black', linestyle=':',
                       linewidth=1.5, label='SLA Limit')
    axes[0, 1].set_title('Average Latency per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Avg Latency (s)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Loss
    axes[1, 0].plot(losses, alpha=0.15, color='blue')
    axes[1, 0].plot(x_range, moving_avg(losses, window),
                    color='blue', linewidth=2.5, label='Dueling DQN')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Huber Loss')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bar chart
    methods    = ['Dueling\nDQN', 'Random', 'All\nOffload', 'All\nLocal']
    lat_vals   = [np.mean(lat[-100:]), np.mean(random_lat[-100:]),
                  np.mean(offload_lat[-100:]), np.mean(local_lat[-100:])]
    bar_colors = ['blue', 'darkorange', 'green', 'red']
    bars = axes[1, 1].bar(methods, lat_vals, color=bar_colors,
                          edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, lat_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.001,
                        f'{val:.4f}s', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
    axes[1, 1].set_title('Final Average Latency (Last 100 Episodes)')
    axes[1, 1].set_ylabel('Avg Latency (s)')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/2_dueling_dqn.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as outputs/2_dueling_dqn.png")
