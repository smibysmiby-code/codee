#!/usr/bin/env python3
"""
================================================================================
PPO (PROXIMAL POLICY OPTIMIZATION) BASELINE FOR UAV TASK OFFLOADING
================================================================================
Per-Device Binary Decision Architecture -- Scales to any number of devices.
Actor-Critic with clipped surrogate objective, GAE advantages, and entropy
regularization for binary task offloading in UAV-assisted edge computing.

To run: python 4_ppo_baseline.py
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
from torch.distributions import Categorical
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

# PPO hyperparameters
PPO_CLIP       = 0.2
PPO_EPOCHS     = 4
PPO_MINI_BATCH = 64
GAMMA          = 0.95
GAE_LAMBDA     = 0.95
ENTROPY_COEFF  = 0.01
VALUE_COEFF    = 0.5
LR             = 3e-4
GRAD_CLIP      = 0.5

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
#  PPO ACTOR-CRITIC NETWORK (Per-Device Binary Decisions)
# =============================================================
class PPOPerDeviceNet(nn.Module):
    """
    Actor-Critic with per-device binary decision heads:
      - Shared trunk processes full state
      - Actor: per-device logits -> Categorical(local, offload)
      - Critic: scalar V(s) for the full state
    """
    def __init__(self, state_size):
        super().__init__()
        self.n_devices = N_DEVICES

        self.trunk = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(256 + FEATURES_PER_DEVICE, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch = x.shape[0]
        trunk_out = self.trunk(x)

        device_feats = x[:, :self.n_devices * FEATURES_PER_DEVICE]
        device_feats = device_feats.view(batch, self.n_devices,
                                         FEATURES_PER_DEVICE)

        trunk_exp = trunk_out.unsqueeze(1).expand(-1, self.n_devices, -1)
        combined  = torch.cat([trunk_exp, device_feats], dim=2)
        flat      = combined.reshape(batch * self.n_devices, -1)
        logits    = self.actor_head(flat).view(batch, self.n_devices, 2)

        value = self.critic_head(trunk_out).squeeze(-1)

        return logits, value

# =============================================================
#  ROLLOUT BUFFER
# =============================================================
class RolloutBuffer:
    def __init__(self):
        self.states     = []
        self.actions    = []
        self.log_probs  = []
        self.rewards    = []
        self.values     = []
        self.dones      = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def compute_gae(self, last_value):
        rewards    = np.array(self.rewards)
        values     = np.array(self.values + [last_value])
        dones      = np.array(self.dones)
        T          = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values)
        return advantages, returns

# =============================================================
#  PPO AGENT
# =============================================================
class PPOAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.model      = PPOPerDeviceNet(state_size).to(DEVICE)
        self.optimizer  = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.buffer     = RolloutBuffer()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {n_params:,}")

    def act(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits, value = self.model(state_t)

        dists   = Categorical(logits=logits.squeeze(0))
        actions = dists.sample()
        log_probs = dists.log_prob(actions).sum().item()

        return (actions.cpu().tolist(),
                log_probs,
                value.item())

    def update(self, last_value):
        advantages, returns = self.buffer.compute_gae(last_value)

        states    = torch.FloatTensor(np.array(self.buffer.states)).to(DEVICE)
        actions   = torch.LongTensor(np.array(self.buffer.actions)).to(DEVICE)
        old_lp    = torch.FloatTensor(np.array(self.buffer.log_probs)).to(DEVICE)
        advs      = torch.FloatTensor(advantages).to(DEVICE)
        rets      = torch.FloatTensor(returns).to(DEVICE)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T = len(self.buffer.states)
        total_loss_val = 0.0
        n_updates = 0

        for _ in range(PPO_EPOCHS):
            indices = np.arange(T)
            np.random.shuffle(indices)

            for start in range(0, T, PPO_MINI_BATCH):
                end = min(start + PPO_MINI_BATCH, T)
                idx = indices[start:end]

                b_states  = states[idx]
                b_actions = actions[idx]
                b_old_lp  = old_lp[idx]
                b_advs    = advs[idx]
                b_rets    = rets[idx]

                logits, values = self.model(b_states)
                dists = Categorical(logits=logits)
                new_lp = dists.log_prob(b_actions).sum(dim=1)
                entropy = dists.entropy().sum(dim=1).mean()

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP,
                                    1.0 + PPO_CLIP) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, b_rets)

                loss = (policy_loss
                        + VALUE_COEFF * value_loss
                        - ENTROPY_COEFF * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

                total_loss_val += loss.item()
                n_updates += 1

        self.buffer.clear()
        return total_loss_val / max(1, n_updates)

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
    agent = PPOAgent(env.state_size)

    latencies, drops, rewards, losses = [], [], [], []

    print(f"\n{'=' * 70}")
    print(f"  PPO BASELINE -- {N_DEVICES} Devices, 1 UAV, {N_EPISODES} Episodes")
    print(f"  Per-device binary decisions (Actor-Critic)")
    print(f"  Clip={PPO_CLIP}, K={PPO_EPOCHS} epochs, "
          f"GAE lambda={GAE_LAMBDA}")
    print(f"{'=' * 70}")
    print(f"{'Episode':>8} | {'Avg Latency':>12} | {'Drop Rate':>10} | "
          f"{'Avg Loss':>10}")
    print("-" * 55)

    for ep in range(N_EPISODES):
        state      = env.reset()
        ep_latency = []
        ep_drops   = []
        ep_reward  = 0.0

        for slot in range(N_SLOTS):
            actions, log_prob, value = agent.act(state)
            next_state, reward, total_lat, info = env.step(actions)

            done = (slot == N_SLOTS - 1)
            agent.buffer.store(state, actions, log_prob, reward, value, done)

            state      = next_state
            ep_reward += reward
            active = [x for x in info if x["action"] != "no task"]
            if active:
                ep_latency.append(total_lat / len(active))
                ep_drops.append(
                    sum(1 for x in active if x["dropped"]) / len(active))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            _, last_val = agent.model(state_t)
        loss_val = agent.update(last_val.item())

        avg_lat  = np.mean(ep_latency) if ep_latency else 0
        avg_drop = np.mean(ep_drops)   if ep_drops   else 0
        latencies.append(avg_lat)
        drops.append(avg_drop)
        rewards.append(ep_reward)
        losses.append(loss_val)

        if ep % 100 == 0:
            print(f"{ep:>8} | {avg_lat:>12.4f}s | {avg_drop:>9.1%} | "
                  f"{loss_val:>10.4f}")

    print(f"\n  PPO complete!")
    print(f"   Last 100 ep avg latency: {np.mean(latencies[-100:]):.4f}s")
    return latencies, drops, rewards, losses

# =============================================================
#  MAIN
# =============================================================
if __name__ == "__main__":
    lat, drops, rew, losses = train()

    print("\nRunning baselines...")
    local_lat, local_rew = run_baseline("All Local",
        lambda: [0] * N_DEVICES)
    offload_lat, offload_rew = run_baseline("All Offload",
        lambda: [1] * N_DEVICES)
    random_lat, random_rew = run_baseline("Random",
        lambda: [random.randint(0, 1) for _ in range(N_DEVICES)])

    print(f"\n{'=' * 50}")
    print(f"  FINAL RESULTS -- PPO Baseline")
    print(f"{'=' * 50}")
    results = [
        ("PPO",         np.mean(lat[-100:])),
        ("Random",      np.mean(random_lat[-100:])),
        ("All Offload", np.mean(offload_lat[-100:])),
        ("All Local",   np.mean(local_lat[-100:])),
    ]
    best = min(r[1] for r in results)
    for name, val in results:
        marker = " <- BEST" if val == best else f"  ({(val-best)/best*100:.1f}% worse)"
        print(f"  {name:<18}: {val:.4f}s{marker}")

    os.makedirs('outputs', exist_ok=True)
    window = 30
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    x_range = range(window - 1, N_EPISODES)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PPO Baseline -- UAV Task Offloading\n"
                 f"{N_DEVICES} Devices, {N_EPISODES} Episodes",
                 fontsize=14, fontweight='bold')

    axes[0, 0].plot(rew, alpha=0.15, color='teal')
    axes[0, 0].plot(x_range, moving_avg(rew, window),
                    color='teal', linewidth=2.5, label='PPO')
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

    axes[0, 1].plot(lat, alpha=0.15, color='teal')
    axes[0, 1].plot(x_range, moving_avg(lat, window),
                    color='teal', linewidth=2.5, label='PPO')
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

    axes[1, 0].plot(losses, alpha=0.15, color='teal')
    axes[1, 0].plot(x_range, moving_avg(losses, window),
                    color='teal', linewidth=2.5, label='PPO')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    methods    = ['PPO', 'Random', 'All\nOffload', 'All\nLocal']
    lat_vals   = [np.mean(lat[-100:]), np.mean(random_lat[-100:]),
                  np.mean(offload_lat[-100:]), np.mean(local_lat[-100:])]
    bar_colors = ['teal', 'darkorange', 'green', 'red']
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
    plt.savefig('outputs/4_ppo_baseline.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as outputs/4_ppo_baseline.png")
