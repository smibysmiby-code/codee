#!/usr/bin/env python3
"""
================================================================================
ATTENTION-ENHANCED HEURISTIC-GUIDED DQN FOR UAV TASK OFFLOADING
================================================================================
Per-Device Binary Decision Architecture -- Scales to any number of devices.
Compares Standard DQN vs Heuristic-Guided DQN vs Attention-Enhanced Heuristic DQN
for binary task offloading in UAV-assisted edge computing networks.
To run: python dqn_uav_offloading_complete.py
Dependencies: pip install numpy matplotlib torch
================================================================================
KEY ARCHITECTURAL CHANGE (vs previous 2^N action space version):
================================================================================
Instead of outputting Q-values for all 2^N joint actions (infeasible at N=20),
each network outputs per-device binary Q-values: Q(s, local) and Q(s, offload)
for each device independently. The total Q-value is the sum of per-device
Q-values (Value Decomposition Network / VDN decomposition).

This enables:
  - Linear scaling with N (instead of exponential)
  - Per-device attention embeddings that naturally map to per-device decisions
  - 20+ devices where attention genuinely helps model inter-device dependencies

Standard DQN:   shared MLP trunk -> per-device Q-values (no cross-device modeling)
Attention DQN:  self-attention across device tokens -> per-device Q-values
                (explicit cross-device dependency modeling)
================================================================================
"""
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
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
N_DEVICES      = 20
N_SLOTS        = 300
N_EPISODES     = 1000
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
    # Original 10 devices
    np.array([50,   50,  0]),
    np.array([100,  30,  0]),
    np.array([150, 120,  0]),
    np.array([80,  150,  0]),
    np.array([120,  80,  0]),
    np.array([200,  60,  0]),
    np.array([60,  200,  0]),
    np.array([180, 170,  0]),
    np.array([30,  100,  0]),
    np.array([250, 100,  0]),
    # 10 additional devices (20-device scenario)
    np.array([40,  180,  0]),
    np.array([160,  40,  0]),
    np.array([220, 140,  0]),
    np.array([90,  220,  0]),
    np.array([270,  50,  0]),
    np.array([130, 190,  0]),
    np.array([190, 110,  0]),
    np.array([70,   70,  0]),
    np.array([240, 200,  0]),
    np.array([110, 130,  0]),
]
DROP_PENALTY = 0.15
DISTANCES    = [np.linalg.norm(UAV_POS - dp) for dp in DEVICE_POSITIONS]
# Heuristic guidance settings
HEUR_EPSILON_THRESHOLD = 0.6
HEUR_CALL_PROB         = 0.50
HEUR_NOISE_PROB        = 0.05
WARMUP_STEPS           = 3000
# =============================================================
#  COST-BASED HEURISTIC GUIDANCE FUNCTION
# =============================================================
def heuristic_suggest_action(state, tasks):
    """
    Cost-based heuristic: computes actual local vs offload latency for
    each device, accounts for cumulative UAV queue, and picks the
    lower-cost option.  Returns a list of N_DEVICES binary decisions.
    """
    queue_time = state[-1] * SLOT_DURATION
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
        self.state_size = N_DEVICES * 4 + 1
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
                state.append(DISTANCES[i] / 350)
                state.append(1.0)
            else:
                state.append(0.0)
                state.append(0.0)
                state.append(DISTANCES[i] / 350)
                state.append(0.0)
        state.append(min(self.uav_queue_time / SLOT_DURATION, 1.0))
        return np.array(state, dtype=np.float32)
    def step(self, decisions):
        """decisions: list of N_DEVICES binary values (0=local, 1=offload)."""
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
#  STANDARD PER-DEVICE DQN NETWORK
# =============================================================
class PerDeviceDQNNet(nn.Module):
    """
    Standard DQN with per-device binary decision heads.
    A shared MLP trunk processes the full state, then a shared decision
    head (conditioned on each device's own features) outputs Q(local)
    and Q(offload) for every device.

    Architecture:
        state (81-d) -> trunk: 512->512->256
        For each device i:
            [trunk_output || device_i_features] -> shared head: 260->128->2
        Output: (batch, N_DEVICES, 2) per-device Q-values

    Total Q(s, a) = sum of per-device Q-values (VDN decomposition)
    """
    FEATURES_PER_DEVICE = 4

    def __init__(self, state_size):
        super().__init__()
        self.n_devices = N_DEVICES
        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Shared decision head: trunk features + device-specific features -> 2
        self.head = nn.Sequential(
            nn.Linear(256 + self.FEATURES_PER_DEVICE, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """Returns (batch, N_DEVICES, 2) per-device Q-values."""
        batch = x.shape[0]
        features = self.trunk(x)                                    # (B, 256)

        # Extract per-device features from state
        device_feats = x[:, :self.n_devices * self.FEATURES_PER_DEVICE]
        device_feats = device_feats.view(batch, self.n_devices,
                                         self.FEATURES_PER_DEVICE)  # (B, N, 4)

        # Expand trunk features to match device dimension
        features_exp = features.unsqueeze(1).expand(
            -1, self.n_devices, -1)                                 # (B, N, 256)

        # Concatenate and apply shared head (vectorised, no loop)
        combined = torch.cat([features_exp, device_feats], dim=2)   # (B, N, 260)
        flat = combined.reshape(batch * self.n_devices, -1)         # (B*N, 260)
        q_flat = self.head(flat)                                    # (B*N, 2)
        return q_flat.view(batch, self.n_devices, 2)                # (B, N, 2)
# =============================================================
#  ATTENTION-ENHANCED PER-DEVICE DQN NETWORK  (proposed)
# =============================================================
class AttentionPerDeviceDQNNet(nn.Module):
    """
    Attention-Enhanced DQN with per-device binary decision heads.
    Self-attention across device tokens enables coordinated decisions:
    each device's embedding is enriched by information about ALL other
    devices' workloads, channel conditions, and task requirements.

    Architecture:
        state (81-d) -> split into 20 device tokens (4-d each) + queue (1-d)
                     -> Linear projection to d_model (64)
                     -> + learned positional encoding
                     -> 2-layer Multi-Head Self-Attention (4 heads)
                     -> each device gets attention-enriched embedding
        For each device i:
            [attn_embedding_i || queue_embedding] -> shared head: 128->128->2
        Output: (batch, N_DEVICES, 2) per-device Q-values

    Why attention helps at 20 devices:
        - 20 devices sharing one UAV queue = 380 pairwise interactions
        - MLP trunk compresses all into a flat vector, losing structure
        - Self-attention explicitly models which device pairs affect each other
        - Device 5 can "see" that device 3 is offloading heavy work and stay local
    """
    FEATURES_PER_DEVICE = 4

    def __init__(self, state_size,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_devices = N_DEVICES
        self.d_model   = d_model

        # Per-device token projection
        self.token_proj = nn.Linear(self.FEATURES_PER_DEVICE, d_model)

        # Learned positional encoding for device slots
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_devices, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Global queue feature projection
        self.queue_proj = nn.Linear(1, d_model)

        # Shared per-device decision head
        self.head = nn.Sequential(
            nn.Linear(d_model + d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        # For extracting attention weights during visualisation
        self._attn_weights = None

    def forward(self, x, return_attention=False):
        """Returns (batch, N_DEVICES, 2) per-device Q-values."""
        batch = x.shape[0]

        # Split state into per-device tokens and queue
        device_feats = x[:, :self.n_devices * self.FEATURES_PER_DEVICE]
        device_feats = device_feats.view(
            batch, self.n_devices, self.FEATURES_PER_DEVICE)
        queue_feat = x[:, -1:]                                    # (B, 1)

        # Project tokens + positional encoding
        tokens = self.token_proj(device_feats) + self.pos_embed   # (B, N, d)

        # Self-attention across devices
        if return_attention:
            attn_out, self._attn_weights = self._forward_with_attn(tokens)
        else:
            attn_out = self.transformer(tokens)                   # (B, N, d)

        # Queue embedding, expanded to all devices
        queue_emb = self.queue_proj(queue_feat)                   # (B, d)
        queue_exp = queue_emb.unsqueeze(1).expand(
            -1, self.n_devices, -1)                               # (B, N, d)

        # Concatenate attention embeddings + queue, apply shared head
        combined = torch.cat([attn_out, queue_exp], dim=2)        # (B, N, 2d)
        flat = combined.reshape(batch * self.n_devices, -1)       # (B*N, 2d)
        q_flat = self.head(flat)                                  # (B*N, 2)
        return q_flat.view(batch, self.n_devices, 2)              # (B, N, 2)

    def _forward_with_attn(self, tokens):
        """Run transformer manually to capture attention weights."""
        x = tokens
        weights_all = []
        for layer in self.transformer.layers:
            x2, w = layer.self_attn(x, x, x, need_weights=True,
                                    average_attn_weights=True)
            weights_all.append(w.detach())
            x = layer.norm1(x + layer.dropout1(x2))
            x = layer.norm2(x + layer._ff_block(x))
        avg_w = torch.stack(weights_all).mean(0)
        return x, avg_w
# =============================================================
#  AGENT -- Per-Device DQN with VDN Decomposition
# =============================================================
class DQNAgent:
    def __init__(self, state_size,
                 use_heuristic=False, use_attention=False):
        self.state_size    = state_size
        self.use_heuristic = use_heuristic
        self.use_attention = use_attention
        self.heuristic_calls = 0
        self.memory      = deque(maxlen=50000)
        self.batch_size  = 128
        self.gamma       = 0.95
        self.epsilon     = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.lr          = 0.0003
        self.tau         = 0.005
        if use_attention:
            self.model        = AttentionPerDeviceDQNNet(state_size).to(DEVICE)
            self.target_model = AttentionPerDeviceDQNNet(state_size).to(DEVICE)
        else:
            self.model        = PerDeviceDQNNet(state_size).to(DEVICE)
            self.target_model = PerDeviceDQNNet(state_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn   = nn.SmoothL1Loss()
        # LR warmup + cosine decay for attention model
        self.scheduler = None
        if use_attention:
            total_steps    = N_EPISODES * N_SLOTS
            warmup_steps_lr = 50 * N_SLOTS
            def lr_lambda(step):
                if step < warmup_steps_lr:
                    return step / max(1, warmup_steps_lr)
                progress = (step - warmup_steps_lr) / max(
                    1, total_steps - warmup_steps_lr)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda)
        self._hard_update_target()

    def _hard_update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self):
        for tp, op in zip(self.target_model.parameters(),
                          self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def act(self, state, tasks=None):
        """Returns list of N_DEVICES binary decisions."""
        if random.random() < self.epsilon:
            if (self.use_heuristic
                    and tasks is not None
                    and self.epsilon > HEUR_EPSILON_THRESHOLD
                    and random.random() < HEUR_CALL_PROB):
                self.heuristic_calls += 1
                return heuristic_suggest_action(state, tasks)
            else:
                return [random.randint(0, 1) for _ in range(N_DEVICES)]
        else:
            with torch.no_grad():
                q = self.model(
                    torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
                # q: (1, N_DEVICES, 2) -> argmax per device
            return q.squeeze(0).argmax(dim=1).cpu().tolist()

    def remember(self, s, a, r, s2):
        """a is a list of N_DEVICES binary decisions."""
        self.memory.append((s, a, r, s2))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch       = random.sample(self.memory, self.batch_size)
        s, a, r, s2 = zip(*batch)
        s  = torch.FloatTensor(np.array(s)).to(DEVICE)        # (B, state)
        a  = torch.LongTensor(np.array(a)).to(DEVICE)         # (B, N_DEVICES)
        r  = torch.FloatTensor(r).to(DEVICE)                  # (B,)
        s2 = torch.FloatTensor(np.array(s2)).to(DEVICE)       # (B, state)
        # Per-device Q-values for current state
        q_all     = self.model(s)                              # (B, N, 2)
        current_q = q_all.gather(
            2, a.unsqueeze(2)).squeeze(2)                      # (B, N)
        # VDN: sum per-device Q-values to get total Q(s, a)
        current_q_total = current_q.sum(dim=1)                 # (B,)
        with torch.no_grad():
            # Double DQN: online model selects, target model evaluates
            q_next_online  = self.model(s2)                    # (B, N, 2)
            best_actions   = q_next_online.argmax(dim=2)       # (B, N)
            q_next_target  = self.target_model(s2)             # (B, N, 2)
            next_q = q_next_target.gather(
                2, best_actions.unsqueeze(2)).squeeze(2)       # (B, N)
            next_q_total = next_q.sum(dim=1)                   # (B,)
        target_q = r + self.gamma * next_q_total
        loss = self.loss_fn(current_q_total, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._soft_update_target()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()
# =============================================================
#  HEURISTIC WARMUP
# =============================================================
def warmup_with_heuristic(agent, warmup_steps):
    env_w = UAVEnvironment()
    state = env_w.reset()
    for _ in range(warmup_steps):
        decisions  = heuristic_suggest_action(state, env_w.current_tasks)
        next_state, reward, _, _, _, _ = env_w.step(decisions)
        agent.remember(state, decisions, reward, next_state)
        state = next_state
        if random.random() < 1.0 / N_SLOTS:
            state = env_w.reset()
    print(f"  Warmup complete: {warmup_steps} heuristic transitions stored")
# =============================================================
#  TRAINING FUNCTION
# =============================================================
def train_agent(state_size, use_heuristic, label, use_attention=False):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    agent = DQNAgent(state_size,
                     use_heuristic=use_heuristic,
                     use_attention=use_attention)
    env = UAVEnvironment()
    if use_heuristic:
        steps = 5000 if use_attention else WARMUP_STEPS
        warmup_with_heuristic(agent, steps)
    # Count parameters
    n_params = sum(p.numel() for p in agent.model.parameters())
    latencies, drops, rewards, losses = [], [], [], []
    print(f"\n{'=' * 75}")
    print(f"  {label} -- {N_DEVICES} Devices, 1 UAV, {N_EPISODES} Episodes")
    print(f"  Per-device binary decisions (VDN decomposition)")
    print(f"  Model parameters: {n_params:,}")
    if use_heuristic:
        ws = 5000 if use_attention else WARMUP_STEPS
        print(f"  Heuristic guidance: ON  (eps>{HEUR_EPSILON_THRESHOLD}, "
              f"call_prob={HEUR_CALL_PROB}, warmup={ws})")
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
            next_state, reward, total_lat, info, _, _ = env.step(decisions)
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
    random.seed(SEED)
    np.random.seed(SEED)
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
env_tmp    = UAVEnvironment()
state_size = env_tmp.state_size
print(f"State size: {state_size} ({N_DEVICES} devices x 4 features + 1 queue)")
print(f"Per-device binary decisions (VDN) -- scales linearly with N")
print(f"Device distances: {[f'{d:.0f}m' for d in DISTANCES]}")
# 1. Standard DQN
std_lat,  std_drop,  std_rew,  std_loss,  _, std_agent  = train_agent(
    state_size,
    use_heuristic=False,
    label="Standard DQN")
# 2. Heuristic-Guided DQN
heur_lat, heur_drop, heur_rew, heur_loss, env_heur, heur_agent = train_agent(
    state_size,
    use_heuristic=True,
    label="Heuristic-Guided DQN")
# 3. Attention-Enhanced Heuristic DQN -- proposed method
attn_lat, attn_drop, attn_rew, attn_loss, env_attn, attn_agent = train_agent(
    state_size,
    use_heuristic=True, use_attention=True,
    label="Attention-Enhanced Heuristic DQN")
# =============================================================
#  BASELINES
# =============================================================
print("\nRunning baselines...")
local_lat,   local_rew   = run_baseline("All Local",
    lambda: [0] * N_DEVICES)
offload_lat, offload_rew = run_baseline("All Offload",
    lambda: [1] * N_DEVICES)
random_lat,  random_rew  = run_baseline("Random",
    lambda: [random.randint(0, 1) for _ in range(N_DEVICES)])
# =============================================================
#  CONVERGENCE SPEED COMPARISON
# =============================================================
window = 30
def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')
print(f"\n{'=' * 60}")
print(f"  CONVERGENCE SPEED")
print(f"{'=' * 60}")
# Use the best DQN agent's final performance as target
all_final = {
    "Attn Heur DQN": np.mean(attn_lat[-50:]),
    "Heur DQN":      np.mean(heur_lat[-50:]),
    "Standard DQN":  np.mean(std_lat[-50:]),
}
best_name = min(all_final, key=all_final.get)
target_lat = all_final[best_name] * 1.05
print(f"  Target: {target_lat:.4f}s (105% of {best_name}'s final)")
for name, lat in [("Attn Heur DQN", attn_lat),
                  ("Heur DQN", heur_lat),
                  ("Standard DQN", std_lat)]:
    smoothed = moving_avg(lat, window)
    converged = [i for i, l in enumerate(smoothed) if l < target_lat]
    if converged:
        print(f"  {name:<24}: reached target at episode {converged[0] + window}")
    else:
        print(f"  {name:<24}: did NOT reach target")
# =============================================================
#  FINAL SUMMARY
# =============================================================
print(f"\n{'=' * 60}")
print(f"  FINAL COMPARISON  ({N_DEVICES} devices, per-device VDN)")
print(f"{'=' * 60}")
results = [
    ("Attn Heur DQN",      np.mean(attn_lat[-50:])),
    ("Heur DQN",           np.mean(heur_lat[-50:])),
    ("Standard DQN",       np.mean(std_lat[-50:])),
    ("Random",             np.mean(random_lat[-50:])),
    ("All Offload",        np.mean(offload_lat[-50:])),
    ("All Local",          np.mean(local_lat[-50:])),
]
best = min(r[1] for r in results)
for name, val in results:
    if val == best:
        marker = " <- BEST"
    else:
        pct = (val - best) / best * 100
        marker = f"  ({pct:.1f}% worse)"
    if name == "Attn Heur DQN":
        marker += "  [proposed]"
    print(f"  {name:<22}: {val:.4f}s{marker}")
# =============================================================
#  PLOTS -- 2x3 grid
# =============================================================
os.makedirs('/mnt/user-data/outputs', exist_ok=True)
x_range = range(window - 1, N_EPISODES)
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(
    f"Attention-Enhanced Heuristic DQN for UAV Task Offloading "
    f"-- {N_DEVICES} Devices, 1 UAV, Per-Device VDN",
    fontsize=14, fontweight='bold')
all_lines = [
    (attn_rew,   attn_lat,   'crimson',    '-',  'Attention Heur DQN (proposed)'),
    (heur_rew,   heur_lat,   'blue',       '-.',  'Heuristic-Guided DQN'),
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
axes[0, 0].legend(fontsize=7)
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
axes[0, 1].legend(fontsize=7)
axes[0, 1].grid(True)
# ---- Plot 3: Convergence zoom (first 500 episodes) ----
zoom_ep = min(500, N_EPISODES)
zoom_lines = [
    (attn_lat[:zoom_ep], 'crimson',    '-',  'Attention Heur DQN'),
    (heur_lat[:zoom_ep], 'blue',       '-.',  'Heuristic-Guided DQN'),
    (std_lat[:zoom_ep],  'purple',     ':',  'Standard DQN'),
]
for lat, color, style, label in zoom_lines:
    axes[0, 2].plot(lat, alpha=0.15, color=color)
    w = min(window, len(lat))
    axes[0, 2].plot(range(w-1, len(lat)),
                    moving_avg(lat, w),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[0, 2].set_title(f'Convergence Speed (First {zoom_ep} Episodes)')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Avg Latency (s)')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True)
# ---- Plot 4: Training Loss ----
for loss, color, style, label in [
    (attn_loss, 'crimson',    '-',  'Attention Heur DQN'),
    (heur_loss, 'blue',       '-.',  'Heuristic-Guided DQN'),
    (std_loss,  'purple',     ':',  'Standard DQN'),
]:
    axes[1, 0].plot(loss, alpha=0.15, color=color)
    axes[1, 0].plot(x_range, moving_avg(loss, window),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[1, 0].set_title('Training Loss')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Huber Loss')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True)
# ---- Plot 5: Final Latency Bar ----
methods    = ['Attn Heur\nDQN', 'Heur\nDQN',
              'Standard\nDQN', 'Random', 'All\nOffload', 'All\nLocal']
lat_vals   = [np.mean(attn_lat[-50:]),    np.mean(heur_lat[-50:]),
              np.mean(std_lat[-50:]),
              np.mean(random_lat[-50:]),
              np.mean(offload_lat[-50:]),  np.mean(local_lat[-50:])]
bar_colors = ['crimson', 'blue', 'purple',
              'darkorange', 'green', 'red']
bars = axes[1, 1].bar(methods, lat_vals, color=bar_colors,
                      edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, lat_vals):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}s', ha='center', va='bottom', fontsize=7)
axes[1, 1].set_title('Final Average Latency (Last 50 Episodes)')
axes[1, 1].set_ylabel('Avg Latency (s)')
axes[1, 1].grid(True, axis='y')
# ---- Plot 6: Attention Heatmap (20x20) ----
random.seed(SEED)
np.random.seed(SEED)
state_sample = env_attn.reset()
with torch.no_grad():
    state_t = torch.FloatTensor(state_sample).unsqueeze(0).to(DEVICE)
    _ = attn_agent.model(state_t, return_attention=True)
    attn_weights = attn_agent.model._attn_weights.squeeze(0).cpu().numpy()
dev_labels = [f"D{i}" for i in range(N_DEVICES)]
im = axes[1, 2].imshow(attn_weights, cmap='YlOrRd', aspect='equal',
                       vmin=0, vmax=attn_weights.max())
axes[1, 2].set_xticks(range(N_DEVICES))
axes[1, 2].set_yticks(range(N_DEVICES))
axes[1, 2].set_xticklabels(dev_labels, fontsize=6, rotation=45)
axes[1, 2].set_yticklabels(dev_labels, fontsize=6)
axes[1, 2].set_xlabel('Key Device')
axes[1, 2].set_ylabel('Query Device')
axes[1, 2].set_title('Learned Device Attention Weights')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/result_attn_heur_dqn.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as /mnt/user-data/outputs/result_attn_heur_dqn.png")
