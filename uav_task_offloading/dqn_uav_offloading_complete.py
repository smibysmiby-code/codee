#!/usr/bin/env python3
"""
================================================================================
DQN FOR UAV TASK OFFLOADING - COMPLETE IMPLEMENTATION
================================================================================

Compares Standard DQN vs Dueling DQN for binary task offloading
in UAV-assisted edge computing networks.

To run: python dqn_uav_offloading_complete.py

Dependencies: pip install numpy matplotlib torch

================================================================================
KEY FIXES APPLIED:
================================================================================
1. CRITICAL: Agent now sees current tasks BEFORE deciding (was blind before)
2. Double DQN prevents Q-value overestimation
3. Gradient clipping prevents training divergence
4. Soft target updates for smoother learning
5. Larger replay buffer for better sample diversity
6. Huber loss for robust training
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
N_EPISODES     = 500

F_LOCAL        = 0.5e9      # device CPU: 0.5 GHz
F_UAV          = 5e9        # UAV CPU:   5 GHz
B              = 0.5e6      # bandwidth: 0.5 MHz
N0             = 1e-10
P_TX           = 0.1

TASK_SIZE_MIN  = 0.1e6      # small tasks -> local clearly faster
TASK_SIZE_MAX  = 0.5e6
CYCLES_MIN     = 1e6        # light tasks -> local = 0.002s
CYCLES_MAX     = 80e6       # heavy tasks -> local = 0.16s -> must offload

UAV_POS = np.array([0, 0, 100])
DEVICE_POSITIONS = [
    np.array([50,  50,  0]),
    np.array([100, 30,  0]),
    np.array([150, 120, 0]),
    np.array([80,  150, 0]),
    np.array([120, 80,  0]),
]

DROP_PENALTY = 0.15

DISTANCES = [np.linalg.norm(UAV_POS - dp) for dp in DEVICE_POSITIONS]


# =============================================================
#  ENVIRONMENT
# =============================================================
class UAVEnvironment:
    def __init__(self):
        self.state_size  = N_DEVICES * 4 + 1
        self.action_size = 2 ** N_DEVICES

    def reset(self):
        self.uav_queue_time = 0.0
        # FIX: Generate tasks FIRST so the agent can see them before deciding
        self.current_tasks = self._generate_tasks()
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
        # FIX: Use the tasks the agent already saw in its state
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

        # FIX: Generate NEXT step's tasks for the next state
        self.current_tasks = self._generate_tasks()
        next_state = self._get_state(self.current_tasks)

        reward = -total_latency
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
    """
    Splits into two streams:
      Value stream     -> V(s)    how good is this state
      Advantage stream -> A(s,a)  how much better is each action
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    def __init__(self, state_size, action_size):
        super().__init__()

        # Shared layers
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        features  = self.feature(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# =============================================================
#  GENERIC AGENT (with Double DQN + improvements)
# =============================================================
class DQNAgent:
    def __init__(self, state_size, action_size, dueling=False):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=20000)
        self.batch_size    = 64
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.99995
        self.lr            = 0.0005
        self.tau           = 0.005   # soft target update rate

        NetClass          = DuelingDQNNet if dueling else DQNNet
        self.model        = NetClass(state_size, action_size)
        self.target_model = NetClass(state_size, action_size)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn      = nn.SmoothL1Loss()  # Huber loss for robustness
        self._hard_update_target()

    def _hard_update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        """Polyak averaging: slowly blend online weights into target."""
        for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
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

        # Current Q-values
        current_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()

        # Double DQN: online network selects best action, target evaluates it
        with torch.no_grad():
            best_actions = self.model(s2).argmax(1)
            next_q = self.target_model(s2).gather(1, best_actions.unsqueeze(1)).squeeze()
        target_q = r + self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update every training step
        self._soft_update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


# =============================================================
#  TRAINING FUNCTION
# =============================================================
def train_agent(state_size, action_size, dueling, label):
    agent = DQNAgent(state_size, action_size, dueling=dueling)
    env   = UAVEnvironment()

    latencies, drops, rewards, losses = [], [], [], []

    print(f"\n{'=' * 65}")
    print(f"  {label} -- 5 Devices, 1 UAV, {N_EPISODES} Episodes")
    print(f"{'=' * 65}")
    print(f"{'Episode':>8} | {'Avg Latency':>12} | {'Drop Rate':>10} | "
          f"{'Avg Loss':>10} | {'Epsilon':>8}")
    print("-" * 65)

    for ep in range(N_EPISODES):
        state      = env.reset()
        ep_latency = []
        ep_drops   = []
        ep_reward  = 0.0
        ep_losses  = []

        for slot in range(N_SLOTS):
            action = agent.act(state)
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
                  f"{avg_loss:>10.4f} | {agent.epsilon:>8.3f}")

    print(f"\n  {label} complete!")
    print(f"   Last 50 ep avg latency: {np.mean(latencies[-50:]):.4f}s")
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
#  TRAIN BOTH AGENTS
# =============================================================
env_tmp = UAVEnvironment()
state_size  = env_tmp.state_size
action_size = env_tmp.action_size

std_latencies,  std_drops,  std_rewards,  std_losses,  env_std,  std_agent  = train_agent(
    state_size, action_size, dueling=False, label="Standard DQN")

duel_latencies, duel_drops, duel_rewards, duel_losses, env_duel, duel_agent = train_agent(
    state_size, action_size, dueling=True,  label="Dueling DQN")

# =============================================================
#  RUN BASELINES
# =============================================================
print("\nRunning baselines...")
local_lat,   local_rew   = run_baseline("All Local",   lambda: 0)
offload_lat, offload_rew = run_baseline("All Offload", lambda: 2**N_DEVICES - 1)
random_lat,  random_rew  = run_baseline("Random",      lambda: random.randrange(2**N_DEVICES))

# =============================================================
#  FINAL SUMMARY
# =============================================================
duel_avg    = np.mean(duel_latencies[-50:])
std_avg     = np.mean(std_latencies[-50:])
local_avg   = np.mean(local_lat[-50:])
offload_avg = np.mean(offload_lat[-50:])
random_avg  = np.mean(random_lat[-50:])

print(f"\n{'=' * 55}")
print(f"  FINAL COMPARISON")
print(f"{'=' * 55}")
print(f"  Dueling DQN  : {duel_avg:.4f}s  <- proposed")
print(f"  Standard DQN : {std_avg:.4f}s  ({(1-duel_avg/std_avg)*100:.1f}% better with Dueling)")
print(f"  Random       : {random_avg:.4f}s  ({(1-duel_avg/random_avg)*100:.1f}% better than Random)")
print(f"  All Offload  : {offload_avg:.4f}s  ({(1-duel_avg/offload_avg)*100:.1f}% better than All Offload)")
print(f"  All Local    : {local_avg:.4f}s  ({(1-duel_avg/local_avg)*100:.1f}% better than All Local)")


# =============================================================
#  PLOTS
# =============================================================
os.makedirs('/mnt/user-data/outputs', exist_ok=True)

window = 30

def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')

x_range = range(window - 1, N_EPISODES)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Dueling DQN vs Standard DQN vs Baselines -- 5 Devices, 1 UAV",
             fontsize=14, fontweight='bold')

lines = [
    (duel_rewards,   duel_latencies,   'blue',       '-',  'Dueling DQN'),
    (std_rewards,    std_latencies,    'deepskyblue','--', 'Standard DQN'),
    (random_rew,     random_lat,       'darkorange', ':',  'Random'),
    (offload_rew,    offload_lat,      'green',      '-.', 'All Offload'),
    (local_rew,      local_lat,        'red',        '--', 'All Local'),
]

# Plot 1: Reward
for rews, _, color, style, label in lines:
    axes[0, 0].plot(rews, alpha=0.1, color=color)
    axes[0, 0].plot(x_range, moving_avg(rews, window),
                    color=color, linewidth=2.5, linestyle=style, label=label)
axes[0, 0].set_title('Episode Reward')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True)

# Plot 2: Loss comparison
axes[0, 1].plot(duel_losses, alpha=0.2, color='blue')
axes[0, 1].plot(x_range, moving_avg(duel_losses, window),
                color='blue', linewidth=2.5, label='Dueling DQN')
axes[0, 1].plot(std_losses, alpha=0.2, color='deepskyblue')
axes[0, 1].plot(x_range, moving_avg(std_losses, window),
                color='deepskyblue', linewidth=2.5, linestyle='--', label='Standard DQN')
axes[0, 1].set_title('Training Loss: Dueling vs Standard DQN')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Huber Loss')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True)

# Plot 3: Latency
for _, lats, color, style, label in lines:
    axes[1, 0].plot(lats, alpha=0.1, color=color)
    axes[1, 0].plot(x_range, moving_avg(lats, window),
                    color=color, linewidth=2.5, linestyle=style, label=label)
axes[1, 0].axhline(SLOT_DURATION, color='black', linestyle=':',
                   linewidth=1.5, label=f'Slot limit ({SLOT_DURATION}s)')
axes[1, 0].set_title('Average Latency per Episode')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Avg Latency (s)')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True)

# Plot 4: Per-device Dueling DQN
state = env_duel.reset()
d_lat = [[] for _ in range(N_DEVICES)]
d_dec = [[] for _ in range(N_DEVICES)]
for slot in range(N_SLOTS):
    action = duel_agent.act(state)
    next_state, _, _, info, _, _ = env_duel.step(action)
    state = next_state
    for x in info:
        i = x["device"]
        if x["action"] != "no task":
            d_lat[i].append(x["latency"])
            d_dec[i].append(1 if x["action"] == "offload" else 0)

labels  = [f"D{i}\n({DISTANCES[i]:.0f}m)" for i in range(N_DEVICES)]
avgs    = [np.mean(d_lat[i]) if d_lat[i] else 0 for i in range(N_DEVICES)]
offrate = [np.mean(d_dec[i]) if d_dec[i] else 0 for i in range(N_DEVICES)]
colors_bar  = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

bars = axes[1, 1].bar(labels, avgs, color=colors_bar, edgecolor='black', linewidth=0.8)
for bar, rate in zip(bars, offrate):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f'{rate:.0%}',
                    ha='center', va='bottom', fontsize=9)
axes[1, 1].axhline(SLOT_DURATION, color='black', linestyle='--',
                   label=f'Slot limit ({SLOT_DURATION}s)')
axes[1, 1].set_title('Final Episode: Per-Device Latency (Dueling DQN)')
axes[1, 1].set_ylabel('Avg Latency (s)')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/result_dueling_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as /mnt/user-data/outputs/result_dueling_final.png")
