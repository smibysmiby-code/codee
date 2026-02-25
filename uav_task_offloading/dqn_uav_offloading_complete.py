#!/usr/bin/env python3
"""
================================================================================
HEURISTIC-GUIDED DQN FOR UAV TASK OFFLOADING - COMPLETE IMPLEMENTATION
================================================================================

Compares Standard DQN vs Dueling DQN vs Heuristic-Guided Dueling DQN
for binary task offloading in UAV-assisted edge computing networks.

To run: python dqn_uav_offloading_complete.py

Dependencies: pip install numpy matplotlib torch

================================================================================
KEY FEATURES:
================================================================================
1. Agent sees current tasks BEFORE deciding
2. Double DQN prevents Q-value overestimation
3. Gradient clipping prevents training divergence
4. Soft target updates for smoother learning
5. Larger replay buffer for better sample diversity
6. Huber loss for robust training
7. Cost-based heuristic-guided exploration for faster early convergence
================================================================================
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# =============================================================
#  SYSTEM PARAMETERS
# =============================================================
SLOT_DURATION  = 0.1
TASK_PROB      = 0.7
N_DEVICES      = 5
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

UAV_POS = np.array([0, 0, 100])
DEVICE_POSITIONS = [
    np.array([50,  50,  0]),
    np.array([100, 30,  0]),
    np.array([150, 120, 0]),
    np.array([80,  150, 0]),
    np.array([120, 80,  0]),
]

DROP_PENALTY = 0.15
DISTANCES    = [np.linalg.norm(UAV_POS - dp) for dp in DEVICE_POSITIONS]

# Heuristic guidance settings
HEUR_EPSILON_THRESHOLD = 0.4   # only use heuristic when epsilon > 0.4 (early training)
HEUR_CALL_PROB         = 0.25  # 25% of exploration steps use heuristic (rest random)
HEUR_NOISE_PROB        = 0.10  # 10% chance to flip each device decision (diversity)


# =============================================================
#  COST-BASED HEURISTIC GUIDANCE FUNCTION
# =============================================================
def heuristic_suggest_action(state, tasks):
    """
    Cost-based heuristic: computes actual local vs offload latency for
    each device, accounts for cumulative UAV queue, and picks the
    lower-cost option.  Devices that MUST offload (local would drop)
    are processed first to get priority queue access.

    A small noise probability flips individual decisions to maintain
    exploration diversity and prevent excessive bias in the replay buffer.
    """
    queue_time = state[-1] * SLOT_DURATION  # de-normalise queue load

    # ---------- gather per-device cost info ----------
    device_costs = []
    for i in range(N_DEVICES):
        if tasks[i] is None:
            device_costs.append(None)
            continue
        D, C = tasks[i]

        local_time = C / F_LOCAL

        SNR      = P_TX / (N0 * DISTANCES[i] ** 2)
        R        = B * np.log2(1 + SNR)
        t_upload = D / R
        t_exec   = C / F_UAV
        # benefit of offloading (positive = offloading saves time)
        benefit  = local_time - (t_upload + t_exec)

        device_costs.append({
            'idx': i,
            'local_time': local_time,
            't_upload': t_upload,
            't_exec': t_exec,
            'benefit': benefit,
            'must_offload': local_time > SLOT_DURATION,
        })

    # ---------- sort: must-offload first, then by descending benefit ----------
    active = [d for d in device_costs if d is not None]
    active.sort(key=lambda d: (-int(d['must_offload']), -d['benefit']))

    decisions = [0] * N_DEVICES
    estimated_queue = queue_time

    for d in active:
        i = d['idx']
        local_time = d['local_time']
        local_dropped = local_time > SLOT_DURATION
        local_cost = DROP_PENALTY if local_dropped else local_time

        offload_time = d['t_upload'] + max(0.0, estimated_queue) + d['t_exec']
        offload_dropped = offload_time > SLOT_DURATION
        offload_cost = DROP_PENALTY if offload_dropped else offload_time

        if offload_cost < local_cost:
            decisions[i] = 1
            estimated_queue += d['t_exec']

    # ---------- inject noise for exploration diversity ----------
    for i in range(N_DEVICES):
        if random.random() < HEUR_NOISE_PROB:
            decisions[i] = 1 - decisions[i]

    action = sum(d << i for i, d in enumerate(decisions))
    return action


# =============================================================
#  ENVIRONMENT
# =============================================================
class UAVEnvironment:
    def __init__(self):
        self.state_size  = N_DEVICES * 4 + 1
        self.action_size = 2 ** N_DEVICES

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
                state.append(D / TASK_SIZE_MAX)
                state.append(C / CYCLES_MAX)
                state.append(DISTANCES[i] / 250)
                state.append(1.0)
            else:
                state.append(0.0)
                state.append(0.0)
                state.append(DISTANCES[i] / 250)
                state.append(0.0)
        state.append(min(self.uav_queue_time / SLOT_DURATION, 1.0))
        return np.array(state, dtype=np.float32)

    def step(self, action):
        tasks     = self.current_tasks
        decisions = [(action >> i) & 1 for i in range(N_DEVICES)]
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
                SNR      = P_TX / (N0 * DISTANCES[i] ** 2)
                R        = B * np.log2(1 + SNR)
                t_upload = D / R
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

        return next_state, reward, total_latency, info, tasks, decisions


# =============================================================
#  STANDARD DQN NETWORK
# =============================================================
class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================
#  DUELING DQN NETWORK
# =============================================================
class DuelingDQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


# =============================================================
#  GENERIC AGENT -- Standard or Dueling, with optional Heuristic
# =============================================================
class DQNAgent:
    def __init__(self, state_size, action_size, dueling=False, use_heuristic=False):
        self.state_size      = state_size
        self.action_size     = action_size
        self.use_heuristic   = use_heuristic
        self.heuristic_calls = 0
        self.memory        = deque(maxlen=20000)
        self.batch_size    = 64
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.99995
        self.lr            = 0.0005
        self.tau           = 0.005

        NetClass          = DuelingDQNNet if dueling else DQNNet
        self.model        = NetClass(state_size, action_size)
        self.target_model = NetClass(state_size, action_size)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn      = nn.SmoothL1Loss()
        self._hard_update_target()

    def _hard_update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        for tp, op in zip(self.target_model.parameters(),
                          self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def act(self, state, tasks=None):
        """
        Heuristic-guided epsilon-greedy:
          - epsilon > HEUR_EPSILON_THRESHOLD AND random < HEUR_CALL_PROB
            -> use cost-based heuristic for smart action
          - else if random < epsilon
            -> random action
          - else
            -> greedy from Q-network
        """
        if random.random() < self.epsilon:
            # Exploration phase
            if (self.use_heuristic
                    and tasks is not None
                    and self.epsilon > HEUR_EPSILON_THRESHOLD
                    and random.random() < HEUR_CALL_PROB):
                self.heuristic_calls += 1
                return heuristic_suggest_action(state, tasks)
            else:
                return random.randrange(self.action_size)
        else:
            # Exploitation: use Q-network
            with torch.no_grad():
                q = self.model(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax().item()

    def remember(self, s, a, r, s2):
        self.memory.append((s, a, r, s2))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch       = random.sample(self.memory, self.batch_size)
        s, a, r, s2 = zip(*batch)

        s  = torch.FloatTensor(np.array(s))
        a  = torch.LongTensor(a)
        r  = torch.FloatTensor(r)
        s2 = torch.FloatTensor(np.array(s2))

        current_q    = self.model(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            best_actions = self.model(s2).argmax(1)
            next_q       = self.target_model(s2).gather(
                               1, best_actions.unsqueeze(1)).squeeze()
        target_q = r + self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        self._soft_update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


# =============================================================
#  TRAINING FUNCTION
# =============================================================
def train_agent(state_size, action_size, dueling, use_heuristic, label):
    agent = DQNAgent(state_size, action_size,
                     dueling=dueling, use_heuristic=use_heuristic)
    env   = UAVEnvironment()

    latencies, drops, rewards, losses = [], [], [], []

    print(f"\n{'=' * 70}")
    print(f"  {label} -- 5 Devices, 1 UAV, {N_EPISODES} Episodes")
    if use_heuristic:
        print(f"  Heuristic guidance: ON  (epsilon>{HEUR_EPSILON_THRESHOLD}, "
              f"call_prob={HEUR_CALL_PROB})")
    print(f"{'=' * 70}")
    print(f"{'Episode':>8} | {'Avg Latency':>12} | {'Drop Rate':>10} | "
          f"{'Avg Loss':>10} | {'Epsilon':>8} | {'Heur calls':>10}")
    print("-" * 75)

    for ep in range(N_EPISODES):
        state      = env.reset()
        ep_latency = []
        ep_drops   = []
        ep_reward  = 0.0
        ep_losses  = []

        for slot in range(N_SLOTS):
            # Pass current tasks to agent for heuristic guidance
            action = agent.act(state, tasks=env.current_tasks)
            next_state, reward, total_lat, info, _, _ = env.step(action)

            agent.remember(state, action, reward, next_state)
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

        if ep % 50 == 0:
            print(f"{ep:>8} | {avg_lat:>12.4f}s | {avg_drop:>9.1%} | "
                  f"{avg_loss:>10.4f} | {agent.epsilon:>8.3f} | "
                  f"{agent.heuristic_calls:>10}")

    print(f"\n  {label} complete!")
    print(f"   Last 50 ep avg latency : {np.mean(latencies[-50:]):.4f}s")
    print(f"   Total heuristic calls  : {agent.heuristic_calls}")
    return latencies, drops, rewards, losses, env, agent


# =============================================================
#  BASELINE RUNNER
# =============================================================
def run_baseline(name, action_fn):
    latencies, rewards = [], []
    print(f"Running {name}...")
    for ep in range(N_EPISODES):
        env_b = UAVEnvironment()
        env_b.reset()
        ep_lat, ep_rew = [], 0.0
        for slot in range(N_SLOTS):
            _, reward, total_lat, info, _, _ = env_b.step(action_fn())
            ep_rew += reward
            active = [x for x in info if x["action"] != "no task"]
            if active:
                ep_lat.append(total_lat / len(active))
        latencies.append(np.mean(ep_lat) if ep_lat else 0)
        rewards.append(ep_rew)
    print(f"  {name}: {np.mean(latencies):.4f}s")
    return latencies, rewards


# =============================================================
#  TRAIN ALL AGENTS
# =============================================================
env_tmp     = UAVEnvironment()
state_size  = env_tmp.state_size
action_size = env_tmp.action_size

# 1. Standard DQN (no guidance, no Dueling)
std_lat,  std_drop,  std_rew,  std_loss,  _, std_agent  = train_agent(
    state_size, action_size,
    dueling=False, use_heuristic=False,
    label="Standard DQN")

# 2. Dueling DQN (no guidance)
duel_lat, duel_drop, duel_rew, duel_loss, _, duel_agent = train_agent(
    state_size, action_size,
    dueling=True, use_heuristic=False,
    label="Dueling DQN")

# 3. Heuristic-Guided Dueling DQN -- proposed method
heur_lat, heur_drop, heur_rew, heur_loss, env_heur, heur_agent = train_agent(
    state_size, action_size,
    dueling=True, use_heuristic=True,
    label="Heuristic-Guided Dueling DQN")


# =============================================================
#  BASELINES
# =============================================================
print("\nRunning baselines...")
local_lat,   local_rew   = run_baseline("All Local",
    lambda: 0)
offload_lat, offload_rew = run_baseline("All Offload",
    lambda: 2**N_DEVICES - 1)
random_lat,  random_rew  = run_baseline("Random",
    lambda: random.randrange(2**N_DEVICES))


# =============================================================
#  FINAL SUMMARY
# =============================================================
print(f"\n{'=' * 60}")
print(f"  FINAL COMPARISON")
print(f"{'=' * 60}")

results = [
    ("Heur Dueling DQN",  np.mean(heur_lat[-50:])),
    ("Dueling DQN",       np.mean(duel_lat[-50:])),
    ("Standard DQN",      np.mean(std_lat[-50:])),
    ("Random",            np.mean(random_lat[-50:])),
    ("All Offload",       np.mean(offload_lat[-50:])),
    ("All Local",         np.mean(local_lat[-50:])),
]
best = results[0][1]

for name, val in results:
    improvement = (1 - best/val)*100 if val != best else 0
    marker = " <- proposed" if name == "Heur Dueling DQN" else \
             f"  ({improvement:.1f}% worse)" if improvement > 0 else ""
    print(f"  {name:<20}: {val:.4f}s{marker}")


# =============================================================
#  PLOTS -- 2x3 grid
# =============================================================
os.makedirs('/mnt/user-data/outputs', exist_ok=True)

window = 30

def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')

x_range = range(window - 1, N_EPISODES)

fig, axes = plt.subplots(2, 3, figsize=(21, 12))
fig.suptitle(
    "Heuristic-Guided Dueling DQN for UAV Task Offloading -- 5 Devices, 1 UAV",
    fontsize=14, fontweight='bold')

all_lines = [
    (heur_rew,   heur_lat,   'blue',       '-',  'Heur Dueling DQN (proposed)'),
    (duel_rew,   duel_lat,   'deepskyblue','--', 'Dueling DQN'),
    (std_rew,    std_lat,    'purple',     ':',  'Standard DQN'),
    (random_rew, random_lat, 'darkorange', ':',  'Random'),
    (offload_rew,offload_lat,'green',      '-.', 'All Offload'),
    (local_rew,  local_lat,  'red',        '--', 'All Local'),
]

# ---- Plot 1: Reward ----
for rew, _, color, style, label in all_lines:
    axes[0, 0].plot(rew, alpha=0.1, color=color)
    axes[0, 0].plot(x_range, moving_avg(rew, window),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[0, 0].set_title('Episode Reward')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True)

# ---- Plot 2: Latency ----
for _, lat, color, style, label in all_lines:
    axes[0, 1].plot(lat, alpha=0.1, color=color)
    axes[0, 1].plot(x_range, moving_avg(lat, window),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[0, 1].axhline(SLOT_DURATION, color='black', linestyle=':',
                   linewidth=1.5, label='Slot limit')
axes[0, 1].set_title('Average Latency per Episode')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Avg Latency (s)')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True)

# ---- Plot 3: Convergence zoom (first 300 episodes) ----
zoom_lines = [
    (heur_lat[:300], 'blue',       '-',  'Heur Dueling DQN'),
    (duel_lat[:300], 'deepskyblue','--', 'Dueling DQN'),
    (std_lat[:300],  'purple',     ':',  'Standard DQN'),
]
for lat, color, style, label in zoom_lines:
    axes[0, 2].plot(lat, alpha=0.15, color=color)
    w = min(window, len(lat))
    axes[0, 2].plot(range(w-1, len(lat)),
                    moving_avg(lat, w),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[0, 2].set_title('Convergence Speed (First 300 Episodes)')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Avg Latency (s)')
axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(True)

# ---- Plot 4: Training Loss ----
for loss, color, style, label in [
    (heur_loss, 'blue',       '-',  'Heur Dueling DQN'),
    (duel_loss, 'deepskyblue','--', 'Dueling DQN'),
    (std_loss,  'purple',     ':',  'Standard DQN'),
]:
    axes[1, 0].plot(loss, alpha=0.15, color=color)
    axes[1, 0].plot(x_range, moving_avg(loss, window),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[1, 0].set_title('Training Loss')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Huber Loss')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True)

# ---- Plot 5: Final Latency Bar ----
methods    = ['Heur\nDueling', 'Dueling\nDQN', 'Standard\nDQN',
              'Random', 'All\nOffload', 'All\nLocal']
lat_vals   = [np.mean(heur_lat[-50:]),    np.mean(duel_lat[-50:]),
              np.mean(std_lat[-50:]),      np.mean(random_lat[-50:]),
              np.mean(offload_lat[-50:]),  np.mean(local_lat[-50:])]
bar_colors = ['blue', 'deepskyblue', 'purple', 'darkorange', 'green', 'red']

bars = axes[1, 1].bar(methods, lat_vals, color=bar_colors,
                      edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, lat_vals):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}s', ha='center', va='bottom', fontsize=8)
axes[1, 1].set_title('Final Average Latency')
axes[1, 1].set_ylabel('Avg Latency (s)')
axes[1, 1].grid(True, axis='y')

# ---- Plot 6: Per-device Latency (Heur-Dueling DQN) ----
state = env_heur.reset()
d_lat = [[] for _ in range(N_DEVICES)]
d_dec = [[] for _ in range(N_DEVICES)]
for slot in range(N_SLOTS):
    action = heur_agent.act(state)
    next_state, _, _, info, _, _ = env_heur.step(action)
    state = next_state
    for x in info:
        i = x["device"]
        if x["action"] != "no task":
            d_lat[i].append(x["latency"])
            d_dec[i].append(1 if x["action"] == "offload" else 0)

labels   = [f"D{i}\n({DISTANCES[i]:.0f}m)" for i in range(N_DEVICES)]
avgs     = [np.mean(d_lat[i]) if d_lat[i] else 0 for i in range(N_DEVICES)]
offrates = [np.mean(d_dec[i]) if d_dec[i] else 0 for i in range(N_DEVICES)]
dev_colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']

bars = axes[1, 2].bar(labels, avgs, color=dev_colors,
                      edgecolor='black', linewidth=0.8)
for bar, rate in zip(bars, offrates):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{rate:.0%}', ha='center', va='bottom', fontsize=9)
axes[1, 2].axhline(SLOT_DURATION, color='black', linestyle='--',
                   label='Slot limit')
axes[1, 2].set_title('Per-Device Latency: Heur Dueling DQN')
axes[1, 2].set_ylabel('Avg Latency (s)')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(True, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/result_heur_dqn.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as /mnt/user-data/outputs/result_heur_dqn.png")
