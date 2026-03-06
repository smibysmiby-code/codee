#!/usr/bin/env python3
"""
================================================================================
ATTENTION-ENHANCED HEURISTIC-GUIDED DUELING DQN FOR UAV TASK OFFLOADING
================================================================================
Per-Device Binary Decision Architecture -- Scales to any number of devices.
Combines:
  1. Dueling Architecture: V(s) + A(s,a) decomposition for better value estimation
  2. Multi-Head Self-Attention: Transformer encoder models inter-device dependencies
  3. Heuristic Guidance: Cost-based heuristic provides warm-start and early exploration
  4. Double DQN: Online network selects actions, target network evaluates

To run: python 3_attention_heur_dueling_dqn.py
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

# Heuristic guidance parameters
HEUR_EPSILON_THRESHOLD = 0.6
HEUR_CALL_PROB         = 0.50
HEUR_NOISE_PROB        = 0.05
WARMUP_STEPS           = 5000

# =============================================================
#  COST-BASED HEURISTIC GUIDANCE FUNCTION
# =============================================================
def heuristic_suggest_action(state, tasks):
    """
    Cost-based heuristic: for each device, compare local execution cost
    vs offloading cost (upload + queue wait + UAV execution).
    Prioritizes devices that MUST offload (local would be dropped).
    Adds small noise to avoid overfitting to heuristic.
    """
    queue_time = state[-1] * SLOT_DURATION
    device_costs = []
    for i in range(N_DEVICES):
        if tasks[i] is None:
            device_costs.append(None)
            continue
        D, C = tasks[i]
        local_time = C / F_LOCAL
        t_upload = D / RATES[i]
        t_exec   = C / F_UAV
        benefit  = local_time - (t_upload + t_exec)
        device_costs.append({
            'idx': i,
            'local_time': local_time,
            't_upload': t_upload,
            't_exec': t_exec,
            'benefit': benefit,
            'must_offload': local_time > SLOT_DURATION,
        })
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
    for i in range(N_DEVICES):
        if random.random() < HEUR_NOISE_PROB:
            decisions[i] = 1 - decisions[i]
    return decisions

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
#  ATTENTION-ENHANCED DUELING PER-DEVICE DQN NETWORK
# =============================================================
class AttentionDuelingPerDeviceDQNNet(nn.Module):
    """
    Combines Transformer self-attention with Dueling architecture:
      1. Per-device features are projected into tokens
      2. Transformer encoder learns inter-device dependencies via attention
      3. Queue state is injected as global context
      4. Dueling head splits into V(s) and A(s,a) streams
      5. Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    def __init__(self, state_size,
                 d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_devices = N_DEVICES
        self.d_model   = d_model

        # Project per-device features into d_model tokens
        self.token_proj = nn.Sequential(
            nn.Linear(FEATURES_PER_DEVICE, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embeddings for each device
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_devices, d_model) * 0.02)

        # Transformer encoder with pre-norm and GELU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Project queue scalar into d_model space
        self.queue_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Dueling Value stream: V(s) per device
        self.value_stream = nn.Sequential(
            nn.Linear(d_model + d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Dueling Advantage stream: A(s,a) per device
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model + d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self._attn_weights = None

    def forward(self, x, return_attention=False):
        batch = x.shape[0]

        # Extract per-device features and queue state
        device_feats = x[:, :self.n_devices * FEATURES_PER_DEVICE]
        device_feats = device_feats.view(
            batch, self.n_devices, FEATURES_PER_DEVICE)
        queue_feat = x[:, -1:]

        # Project to tokens + positional embedding
        tokens = self.token_proj(device_feats) + self.pos_embed

        # Transformer self-attention
        if return_attention:
            attn_out, self._attn_weights = self._forward_with_attn(tokens)
        else:
            attn_out = self.transformer(tokens)

        # Queue context injection
        queue_emb = self.queue_proj(queue_feat)
        queue_exp = queue_emb.unsqueeze(1).expand(-1, self.n_devices, -1)

        # Concatenate attention output with queue context
        combined = torch.cat([attn_out, queue_exp], dim=2)
        flat = combined.reshape(batch * self.n_devices, -1)

        # Dueling: V(s) + A(s,a) - mean(A)
        value     = self.value_stream(flat)        # (batch*N, 1)
        advantage = self.advantage_stream(flat)     # (batch*N, 2)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q.view(batch, self.n_devices, 2)

    def _forward_with_attn(self, tokens):
        """Forward pass that also captures attention weights."""
        x = tokens
        weights_all = []
        for layer in self.transformer.layers:
            x_norm = layer.norm1(x)
            x2, w = layer.self_attn(x_norm, x_norm, x_norm,
                                    need_weights=True,
                                    average_attn_weights=True)
            weights_all.append(w.detach())
            x = x + layer.dropout1(x2)
            x = x + layer._ff_block(layer.norm2(x))
        if self.transformer.norm is not None:
            x = self.transformer.norm(x)
        avg_w = torch.stack(weights_all).mean(0)
        return x, avg_w

# =============================================================
#  AGENT
# =============================================================
class AttentionDuelingDQNAgent:
    def __init__(self, state_size):
        self.state_size      = state_size
        self.heuristic_calls = 0
        self.gamma           = 0.95
        self.epsilon         = 1.0
        self.epsilon_min     = 0.01
        self.epsilon_decay   = 0.999975
        self.tau             = 0.005
        self.lr              = 0.0002
        self.batch_size      = 256
        self.grad_clip       = 5.0
        self.memory          = deque(maxlen=100000)

        self.model        = AttentionDuelingPerDeviceDQNNet(state_size).to(DEVICE)
        self.target_model = AttentionDuelingPerDeviceDQNNet(state_size).to(DEVICE)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.lr,
                                       weight_decay=1e-5)
        self.loss_fn      = nn.SmoothL1Loss()

        # Cosine annealing LR with warmup
        total_steps    = N_EPISODES * N_SLOTS
        warmup_steps_lr = 100 * N_SLOTS
        def lr_lambda(step):
            if step < warmup_steps_lr:
                return step / max(1, warmup_steps_lr)
            progress = (step - warmup_steps_lr) / max(
                1, total_steps - warmup_steps_lr)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda)

        self._hard_update_target()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {n_params:,}")

    def _hard_update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        for tp, op in zip(self.target_model.parameters(),
                          self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def act(self, state, tasks=None):
        if random.random() < self.epsilon:
            # Heuristic-guided exploration when epsilon is high
            if (tasks is not None
                    and self.epsilon > HEUR_EPSILON_THRESHOLD
                    and random.random() < HEUR_CALL_PROB):
                self.heuristic_calls += 1
                return heuristic_suggest_action(state, tasks)
            else:
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
        self.scheduler.step()
        self._soft_update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# =============================================================
#  HEURISTIC WARMUP
# =============================================================
def warmup_with_heuristic(agent):
    """Pre-fill replay buffer with heuristic-guided transitions."""
    env_w = UAVEnvironment()
    state = env_w.reset()
    for _ in range(WARMUP_STEPS):
        decisions  = heuristic_suggest_action(state, env_w.current_tasks)
        next_state, reward, _, _ = env_w.step(decisions)
        agent.remember(state, decisions, reward, next_state)
        state = next_state
        if random.random() < 1.0 / N_SLOTS:
            state = env_w.reset()
    print(f"  Warmup complete: {WARMUP_STEPS} heuristic transitions stored")

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
    agent = AttentionDuelingDQNAgent(env.state_size)

    # Heuristic warmup
    warmup_with_heuristic(agent)

    latencies, drops, rewards, losses = [], [], [], []

    print(f"\n{'=' * 75}")
    print(f"  ATTENTION-ENHANCED HEURISTIC DUELING DQN -- {N_DEVICES} Devices, "
          f"1 UAV, {N_EPISODES} Episodes")
    print(f"  Per-device binary decisions (VDN decomposition)")
    print(f"  Attention (8-head, 3-layer) + Dueling V/A streams + "
          f"Heuristic guidance")
    print(f"{'=' * 75}")
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
            decisions = agent.act(state, tasks=env.current_tasks)
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
                  f"{avg_loss:>10.4f} | {agent.epsilon:>8.3f} | "
                  f"{agent.heuristic_calls:>10}")

    print(f"\n  Attention Heuristic Dueling DQN complete!")
    print(f"   Last 100 ep avg latency: {np.mean(latencies[-100:]):.4f}s")
    print(f"   Total heuristic calls  : {agent.heuristic_calls}")
    return latencies, drops, rewards, losses, env, agent

# =============================================================
#  MAIN
# =============================================================
if __name__ == "__main__":
    lat, drops, rew, losses, env_final, agent_final = train()

    # Baselines
    print("\nRunning baselines...")
    local_lat, local_rew = run_baseline("All Local",
        lambda: [0] * N_DEVICES)
    offload_lat, offload_rew = run_baseline("All Offload",
        lambda: [1] * N_DEVICES)
    random_lat, random_rew = run_baseline("Random",
        lambda: [random.randint(0, 1) for _ in range(N_DEVICES)])

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  FINAL RESULTS -- Attention Heuristic Dueling DQN")
    print(f"{'=' * 55}")
    results = [
        ("Attn Heur Dueling DQN", np.mean(lat[-100:])),
        ("Random",                np.mean(random_lat[-100:])),
        ("All Offload",           np.mean(offload_lat[-100:])),
        ("All Local",             np.mean(local_lat[-100:])),
    ]
    best = min(r[1] for r in results)
    for name, val in results:
        marker = " <- BEST" if val == best else f"  ({(val-best)/best*100:.1f}% worse)"
        if "Attn" in name:
            marker += "  [proposed]"
        print(f"  {name:<24}: {val:.4f}s{marker}")

    # Plots
    os.makedirs('outputs', exist_ok=True)
    window = 30
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    x_range = range(window - 1, N_EPISODES)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Attention-Enhanced Heuristic Dueling DQN -- UAV Task Offloading\n"
        f"{N_DEVICES} Devices, {N_EPISODES} Episodes, "
        f"Transformer + Dueling V/A + Heuristic Guidance",
        fontsize=14, fontweight='bold')

    # Reward
    axes[0, 0].plot(rew, alpha=0.15, color='crimson')
    axes[0, 0].plot(x_range, moving_avg(rew, window),
                    color='crimson', linewidth=2.5,
                    label='Attn Heur Dueling DQN')
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
    axes[0, 1].plot(lat, alpha=0.15, color='crimson')
    axes[0, 1].plot(x_range, moving_avg(lat, window),
                    color='crimson', linewidth=2.5,
                    label='Attn Heur Dueling DQN')
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

    # Convergence zoom
    zoom_ep = min(500, N_EPISODES)
    axes[0, 2].plot(lat[:zoom_ep], alpha=0.15, color='crimson')
    w = min(window, zoom_ep)
    axes[0, 2].plot(range(w-1, zoom_ep),
                    moving_avg(lat[:zoom_ep], w),
                    color='crimson', linewidth=2.5,
                    label='Attn Heur Dueling DQN')
    axes[0, 2].set_title(f'Convergence Speed (First {zoom_ep} Episodes)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Avg Latency (s)')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Loss
    axes[1, 0].plot(losses, alpha=0.15, color='crimson')
    axes[1, 0].plot(x_range, moving_avg(losses, window),
                    color='crimson', linewidth=2.5,
                    label='Attn Heur Dueling DQN')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Huber Loss')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bar chart
    methods    = ['Attn Heur\nDueling DQN', 'Random',
                  'All\nOffload', 'All\nLocal']
    lat_vals   = [np.mean(lat[-100:]), np.mean(random_lat[-100:]),
                  np.mean(offload_lat[-100:]), np.mean(local_lat[-100:])]
    bar_colors = ['crimson', 'darkorange', 'green', 'red']
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

    # Attention heatmap
    random.seed(SEED)
    np.random.seed(SEED)
    n_attn_samples = 100
    attn_weights_accum = None
    for _ in range(n_attn_samples):
        state_sample = env_final.reset()
        with torch.no_grad():
            state_t = torch.FloatTensor(state_sample).unsqueeze(0).to(DEVICE)
            _ = agent_final.model(state_t, return_attention=True)
            w = agent_final.model._attn_weights.squeeze(0).cpu().numpy()
            if attn_weights_accum is None:
                attn_weights_accum = w
            else:
                attn_weights_accum += w
    attn_weights_avg = attn_weights_accum / n_attn_samples

    dev_labels = [f"D{i}" for i in range(N_DEVICES)]
    im = axes[1, 2].imshow(attn_weights_avg, cmap='YlOrRd', aspect='equal',
                           vmin=0, vmax=attn_weights_avg.max())
    axes[1, 2].set_xticks(range(N_DEVICES))
    axes[1, 2].set_yticks(range(N_DEVICES))
    axes[1, 2].set_xticklabels(dev_labels, fontsize=5, rotation=45)
    axes[1, 2].set_yticklabels(dev_labels, fontsize=5)
    axes[1, 2].set_xlabel('Key Device')
    axes[1, 2].set_ylabel('Query Device')
    axes[1, 2].set_title(f'Attention Weights (avg over {n_attn_samples} states)')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('outputs/3_attention_heur_dueling_dqn.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as outputs/3_attention_heur_dueling_dqn.png")
