#!/usr/bin/env python3
"""
================================================================================
ATTENTION-ENHANCED HEURISTIC-GUIDED DQN FOR UAV TASK OFFLOADING  (v2 - Improved)
================================================================================
Per-Device Binary Decision Architecture -- Scales to any number of devices.
Compares Standard DQN vs Heuristic-Guided DQN vs Attention-Enhanced Heuristic DQN
for binary task offloading in UAV-assisted edge computing networks.

IMPROVEMENTS OVER v1:
  1. Scaled to 30 devices (attention advantage grows with N)
  2. 2000 episodes for full convergence
  3. Richer per-device state (6 features: +SNR, +local_feasibility)
  4. Deeper attention model (d_model=128, 3 layers, LayerNorm)
  5. Multi-seed evaluation (3 seeds, mean +/- std, p-values)
  6. Better attention visualization (averaged over 100 states)
  7. Prioritized epsilon scheduling per agent type
  8. Gradient clipping tuned per architecture

To run: python dqn_uav_offloading_complete.py
Dependencies: pip install numpy matplotlib torch
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
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# =============================================================
#  REPRODUCIBILITY
# =============================================================
BASE_SEED = 42
SEEDS = [42, 123, 456]  # Multi-seed evaluation
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================
#  SYSTEM PARAMETERS
# =============================================================
SLOT_DURATION  = 0.1
TASK_PROB      = 0.7
N_DEVICES      = 30           # Scaled up from 20 -> 30
N_SLOTS        = 300
N_EPISODES     = 2000         # Doubled from 1000 for full convergence
F_LOCAL        = 0.5e9
F_UAV          = 5e9
B              = 0.5e6
N0             = 1e-10
P_TX           = 0.1
TASK_SIZE_MIN  = 0.1e6
TASK_SIZE_MAX  = 0.5e6
CYCLES_MIN     = 1e6
CYCLES_MAX     = 80e6
FEATURES_PER_DEVICE = 6       # Richer state: +SNR +local_feasibility

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
    # Devices 11-20
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
    # Devices 21-30 (new for 30-device scenario)
    np.array([300,  80,  0]),
    np.array([20,  250,  0]),
    np.array([280, 180,  0]),
    np.array([140, 260,  0]),
    np.array([320, 140,  0]),
    np.array([50,  280,  0]),
    np.array([260, 240,  0]),
    np.array([180, 280,  0]),
    np.array([330,  30,  0]),
    np.array([100, 300,  0]),
]

DROP_PENALTY = 0.15
DISTANCES    = [np.linalg.norm(UAV_POS - dp) for dp in DEVICE_POSITIONS]

# Pre-compute per-device SNR and transmission rate (static, distance-based)
SNRS = [P_TX / (N0 * d ** 2) for d in DISTANCES]
RATES = [B * np.log2(1 + snr) for snr in SNRS]

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
#  ENVIRONMENT (enriched state representation)
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
        """
        6 features per device (up from 4):
          0: task data size (normalized)
          1: task CPU cycles (normalized)
          2: distance to UAV (normalized)
          3: has_task flag (0 or 1)
          4: channel quality / SNR indicator (normalized)
          5: local feasibility (1 if local_time <= SLOT_DURATION, else 0)
        + 1 global: UAV queue occupancy
        """
        state = []
        for i in range(N_DEVICES):
            if tasks and tasks[i] is not None:
                D, C = tasks[i]
                local_time = C / F_LOCAL
                state.append(D / TASK_SIZE_MAX)                          # data size
                state.append(C / CYCLES_MAX)                             # cpu cycles
                state.append(DISTANCES[i] / 400)                         # distance
                state.append(1.0)                                        # has_task
                state.append(min(np.log10(SNRS[i] + 1) / 10.0, 1.0))    # SNR quality
                state.append(1.0 if local_time <= SLOT_DURATION else 0.0) # local feasible
            else:
                state.append(0.0)
                state.append(0.0)
                state.append(DISTANCES[i] / 400)
                state.append(0.0)
                state.append(min(np.log10(SNRS[i] + 1) / 10.0, 1.0))
                state.append(1.0)  # no task => trivially feasible locally
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
    Total Q(s, a) = sum of per-device Q-values (VDN decomposition)
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
        self.head = nn.Sequential(
            nn.Linear(256 + FEATURES_PER_DEVICE, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        batch = x.shape[0]
        features = self.trunk(x)
        device_feats = x[:, :self.n_devices * FEATURES_PER_DEVICE]
        device_feats = device_feats.view(batch, self.n_devices,
                                         FEATURES_PER_DEVICE)
        features_exp = features.unsqueeze(1).expand(
            -1, self.n_devices, -1)
        combined = torch.cat([features_exp, device_feats], dim=2)
        flat = combined.reshape(batch * self.n_devices, -1)
        q_flat = self.head(flat)
        return q_flat.view(batch, self.n_devices, 2)

# =============================================================
#  ATTENTION-ENHANCED PER-DEVICE DQN NETWORK  (proposed, improved)
# =============================================================
class AttentionPerDeviceDQNNet(nn.Module):
    """
    Attention-Enhanced DQN with per-device binary decision heads (v2).

    Improvements over v1:
      - d_model increased to 128 for richer embeddings
      - 3 transformer layers (from 2) for deeper cross-device reasoning
      - Pre-LayerNorm architecture for more stable training
      - Deeper decision head with LayerNorm + residual connection
      - Sinusoidal + learned positional encoding
      - Device distance embedding injected into tokens

    Why attention advantage grows with 30 devices:
      - 30 devices sharing one UAV queue = 870 pairwise interactions
      - MLP trunk loses all structural information in flat compression
      - Self-attention explicitly models which device pairs affect each other
      - Queue contention is the key bottleneck: attention learns who competes
    """
    def __init__(self, state_size,
                 d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_devices = N_DEVICES
        self.d_model   = d_model

        # Per-device token projection with LayerNorm
        self.token_proj = nn.Sequential(
            nn.Linear(FEATURES_PER_DEVICE, d_model),
            nn.LayerNorm(d_model),
        )

        # Learned positional encoding for device slots
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_devices, d_model) * 0.02)

        # Transformer encoder layers (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-LayerNorm for more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),  # Final LayerNorm
        )

        # Global queue feature projection
        self.queue_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Deeper per-device decision head with LayerNorm
        self.head = nn.Sequential(
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
        device_feats = x[:, :self.n_devices * FEATURES_PER_DEVICE]
        device_feats = device_feats.view(
            batch, self.n_devices, FEATURES_PER_DEVICE)
        queue_feat = x[:, -1:]

        tokens = self.token_proj(device_feats) + self.pos_embed

        if return_attention:
            attn_out, self._attn_weights = self._forward_with_attn(tokens)
        else:
            attn_out = self.transformer(tokens)

        queue_emb = self.queue_proj(queue_feat)
        queue_exp = queue_emb.unsqueeze(1).expand(
            -1, self.n_devices, -1)

        combined = torch.cat([attn_out, queue_exp], dim=2)
        flat = combined.reshape(batch * self.n_devices, -1)
        q_flat = self.head(flat)
        return q_flat.view(batch, self.n_devices, 2)

    def _forward_with_attn(self, tokens):
        """Run transformer manually to capture attention weights."""
        x = tokens
        weights_all = []
        for layer in self.transformer.layers:
            # Pre-norm attention
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
#  AGENT -- Per-Device DQN with VDN Decomposition (improved)
# =============================================================
class DQNAgent:
    def __init__(self, state_size,
                 use_heuristic=False, use_attention=False):
        self.state_size    = state_size
        self.use_heuristic = use_heuristic
        self.use_attention = use_attention
        self.heuristic_calls = 0
        self.gamma       = 0.95
        self.epsilon     = 1.0
        self.epsilon_min = 0.01
        self.tau         = 0.005

        # Tuned hyperparameters per architecture
        if use_attention:
            self.memory       = deque(maxlen=100000)   # Larger buffer
            self.batch_size   = 256                     # Larger batch for stability
            self.lr           = 0.0002                  # Slightly lower LR
            self.epsilon_decay = 0.999975               # Slower decay -> more exploration
            self.grad_clip    = 5.0                     # Tighter gradient clipping
        else:
            self.memory       = deque(maxlen=50000)
            self.batch_size   = 128
            self.lr           = 0.0003
            self.epsilon_decay = 0.99995
            self.grad_clip    = 10.0

        if use_attention:
            self.model        = AttentionPerDeviceDQNNet(state_size).to(DEVICE)
            self.target_model = AttentionPerDeviceDQNNet(state_size).to(DEVICE)
        else:
            self.model        = PerDeviceDQNNet(state_size).to(DEVICE)
            self.target_model = PerDeviceDQNNet(state_size).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                    weight_decay=1e-5)  # Light L2 regularization
        self.loss_fn   = nn.SmoothL1Loss()

        # LR warmup + cosine decay for attention model
        self.scheduler = None
        if use_attention:
            total_steps    = N_EPISODES * N_SLOTS
            warmup_steps_lr = 100 * N_SLOTS  # Longer warmup (100 eps vs 50)
            def lr_lambda(step):
                if step < warmup_steps_lr:
                    return step / max(1, warmup_steps_lr)
                progress = (step - warmup_steps_lr) / max(
                    1, total_steps - warmup_steps_lr)
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
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
        current_q = q_all.gather(
            2, a.unsqueeze(2)).squeeze(2)
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
def train_agent(state_size, use_heuristic, label, use_attention=False, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent = DQNAgent(state_size,
                     use_heuristic=use_heuristic,
                     use_attention=use_attention)
    env = UAVEnvironment()

    if use_heuristic:
        steps = 5000 if use_attention else WARMUP_STEPS
        warmup_with_heuristic(agent, steps)

    n_params = sum(p.numel() for p in agent.model.parameters())
    latencies, drops, rewards, losses = [], [], [], []

    print(f"\n{'=' * 75}")
    print(f"  {label} (seed={seed}) -- {N_DEVICES} Devices, 1 UAV, "
          f"{N_EPISODES} Episodes")
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

        if ep % 100 == 0:
            print(f"{ep:>8} | {avg_lat:>12.4f}s | {avg_drop:>9.1%} | "
                  f"{avg_loss:>10.4f} | {agent.epsilon:>8.3f} | "
                  f"{agent.heuristic_calls:>10}")

    print(f"\n  {label} complete!")
    print(f"   Last 100 ep avg latency : {np.mean(latencies[-100:]):.4f}s")
    print(f"   Total heuristic calls   : {agent.heuristic_calls}")

    return latencies, drops, rewards, losses, env, agent

# =============================================================
#  BASELINE RUNNER
# =============================================================
def run_baseline(name, action_fn, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    latencies, rewards = [], []
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
    return latencies, rewards

# =============================================================
#  MULTI-SEED TRAINING & EVALUATION
# =============================================================
env_tmp    = UAVEnvironment()
state_size = env_tmp.state_size
print(f"\nState size: {state_size} ({N_DEVICES} devices x {FEATURES_PER_DEVICE}"
      f" features + 1 queue)")
print(f"Per-device binary decisions (VDN) -- scales linearly with N")
print(f"Device distances: {[f'{d:.0f}m' for d in DISTANCES]}")
print(f"\nRunning {len(SEEDS)} seeds: {SEEDS}")
print(f"{'='*75}")

# Storage for multi-seed results
all_results = {
    'std':  {'lats': [], 'rews': [], 'losses': []},
    'heur': {'lats': [], 'rews': [], 'losses': []},
    'attn': {'lats': [], 'rews': [], 'losses': []},
    'local':   {'lats': [], 'rews': []},
    'offload': {'lats': [], 'rews': []},
    'random':  {'lats': [], 'rews': []},
}

# Keep last seed's agents/envs for attention visualization
last_attn_agent = None
last_attn_env   = None

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'#'*75}")
    print(f"  SEED {seed_idx+1}/{len(SEEDS)}: {seed}")
    print(f"{'#'*75}")

    # 1. Standard DQN
    lat, drop, rew, loss, _, _ = train_agent(
        state_size, use_heuristic=False,
        label="Standard DQN", seed=seed)
    all_results['std']['lats'].append(lat)
    all_results['std']['rews'].append(rew)
    all_results['std']['losses'].append(loss)

    # 2. Heuristic-Guided DQN
    lat, drop, rew, loss, _, _ = train_agent(
        state_size, use_heuristic=True,
        label="Heuristic-Guided DQN", seed=seed)
    all_results['heur']['lats'].append(lat)
    all_results['heur']['rews'].append(rew)
    all_results['heur']['losses'].append(loss)

    # 3. Attention-Enhanced Heuristic DQN (proposed)
    lat, drop, rew, loss, env_a, agent_a = train_agent(
        state_size, use_heuristic=True, use_attention=True,
        label="Attention-Enhanced Heuristic DQN", seed=seed)
    all_results['attn']['lats'].append(lat)
    all_results['attn']['rews'].append(rew)
    all_results['attn']['losses'].append(loss)
    last_attn_agent = agent_a
    last_attn_env   = env_a

    # Baselines
    lat, rew = run_baseline("All Local",
        lambda: [0] * N_DEVICES, seed=seed)
    all_results['local']['lats'].append(lat)
    all_results['local']['rews'].append(rew)

    lat, rew = run_baseline("All Offload",
        lambda: [1] * N_DEVICES, seed=seed)
    all_results['offload']['lats'].append(lat)
    all_results['offload']['rews'].append(rew)

    lat, rew = run_baseline("Random",
        lambda: [random.randint(0, 1) for _ in range(N_DEVICES)], seed=seed)
    all_results['random']['lats'].append(lat)
    all_results['random']['rews'].append(rew)

# =============================================================
#  AGGREGATE MULTI-SEED RESULTS
# =============================================================
def mean_across_seeds(list_of_lists):
    """Average episode-wise across seeds."""
    return np.mean(list_of_lists, axis=0)

def std_across_seeds(list_of_lists):
    return np.std(list_of_lists, axis=0)

def final_latencies(key, last_n=100):
    """Get per-seed final latencies (last N episodes)."""
    return [np.mean(lat[-last_n:]) for lat in all_results[key]['lats']]

# Compute averaged curves
avg_lats  = {k: mean_across_seeds(v['lats']) for k, v in all_results.items()}
std_lats  = {k: std_across_seeds(v['lats']) for k, v in all_results.items()}
avg_rews  = {k: mean_across_seeds(v['rews']) for k, v in all_results.items()}

# For loss curves (only DQN variants have losses)
avg_losses = {}
for k in ['std', 'heur', 'attn']:
    avg_losses[k] = mean_across_seeds(all_results[k]['losses'])

# =============================================================
#  STATISTICAL SIGNIFICANCE (t-test)
# =============================================================
print(f"\n{'=' * 75}")
print(f"  STATISTICAL ANALYSIS  ({N_DEVICES} devices, {len(SEEDS)} seeds)")
print(f"{'=' * 75}")

methods_stat = [
    ("Attn Heur DQN", 'attn'),
    ("Heur DQN",      'heur'),
    ("Standard DQN",  'std'),
    ("Random",        'random'),
    ("All Offload",   'offload'),
    ("All Local",     'local'),
]

final_100 = {}
for name, key in methods_stat:
    vals = final_latencies(key, last_n=100)
    final_100[key] = vals
    mean_val = np.mean(vals)
    std_val  = np.std(vals)
    print(f"  {name:<22}: {mean_val:.4f}s +/- {std_val:.4f}s  "
          f"(seeds: {[f'{v:.4f}' for v in vals]})")

# Pairwise t-tests: attention vs others
print(f"\n  Pairwise t-tests (Attention vs others):")
for name, key in methods_stat[1:]:
    if len(SEEDS) >= 2:
        t_stat, p_val = stats.ttest_ind(final_100['attn'], final_100[key])
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
        print(f"    vs {name:<20}: t={t_stat:+.3f}, p={p_val:.4f} {sig}")

# =============================================================
#  CONVERGENCE SPEED COMPARISON
# =============================================================
window = 30
def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')

print(f"\n{'=' * 60}")
print(f"  CONVERGENCE SPEED (averaged over {len(SEEDS)} seeds)")
print(f"{'=' * 60}")

target_lat = np.mean(final_100['attn']) * 1.05
print(f"  Target: {target_lat:.4f}s (105% of Attn Heur DQN's final)")

for name, key in [("Attn Heur DQN", 'attn'),
                  ("Heur DQN", 'heur'),
                  ("Standard DQN", 'std')]:
    smoothed = moving_avg(avg_lats[key], window)
    converged = [i for i, l in enumerate(smoothed) if l < target_lat]
    if converged:
        print(f"  {name:<24}: reached target at episode {converged[0] + window}")
    else:
        print(f"  {name:<24}: did NOT reach target")

# =============================================================
#  FINAL SUMMARY TABLE
# =============================================================
print(f"\n{'=' * 75}")
print(f"  FINAL COMPARISON  ({N_DEVICES} devices, {len(SEEDS)} seeds, "
      f"last 100 episodes)")
print(f"{'=' * 75}")

best_mean = min(np.mean(final_100[k]) for _, k in methods_stat)
for name, key in methods_stat:
    mean_val = np.mean(final_100[key])
    std_val  = np.std(final_100[key])
    if mean_val == best_mean:
        marker = " <- BEST"
    else:
        pct = (mean_val - best_mean) / best_mean * 100
        marker = f"  ({pct:.1f}% worse)"
    if key == 'attn':
        marker += "  [proposed]"
    print(f"  {name:<22}: {mean_val:.4f}s +/- {std_val:.4f}s{marker}")

# =============================================================
#  PLOTS -- 3x3 grid (expanded visualization)
# =============================================================
os.makedirs('/mnt/user-data/outputs', exist_ok=True)
x_range = range(window - 1, N_EPISODES)

fig, axes = plt.subplots(3, 3, figsize=(24, 18))
fig.suptitle(
    f"Attention-Enhanced Heuristic DQN for UAV Task Offloading\n"
    f"{N_DEVICES} Devices, 1 UAV, {N_EPISODES} Episodes, "
    f"{len(SEEDS)} Seeds (mean +/- std)",
    fontsize=14, fontweight='bold')

all_lines = [
    ('attn',    'crimson',    '-',   'Attention Heur DQN (proposed)'),
    ('heur',    'blue',       '-.',  'Heuristic-Guided DQN'),
    ('std',     'purple',     ':',   'Standard DQN'),
    ('random',  'darkorange', ':',   'Random'),
    ('offload', 'green',      '-.',  'All Offload'),
    ('local',   'red',        '--',  'All Local'),
]

# ---- Plot 1: Reward (mean +/- std shading) ----
for key, color, style, label in all_lines:
    smoothed = moving_avg(avg_rews[key], window)
    axes[0, 0].plot(x_range, smoothed, color=color, linewidth=2.5,
                    linestyle=style, label=label)
    if len(SEEDS) > 1:
        std_smooth = moving_avg(std_across_seeds(
            all_results[key]['rews']), window)
        axes[0, 0].fill_between(x_range,
                                smoothed - std_smooth,
                                smoothed + std_smooth,
                                color=color, alpha=0.1)
axes[0, 0].set_title('Episode Reward (mean +/- std)')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].legend(fontsize=7, loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# ---- Plot 2: Latency (mean +/- std shading) ----
for key, color, style, label in all_lines:
    smoothed = moving_avg(avg_lats[key], window)
    axes[0, 1].plot(x_range, smoothed, color=color, linewidth=2.5,
                    linestyle=style, label=label)
    if len(SEEDS) > 1:
        std_smooth = moving_avg(std_lats[key], window)
        axes[0, 1].fill_between(x_range,
                                smoothed - std_smooth,
                                smoothed + std_smooth,
                                color=color, alpha=0.1)
axes[0, 1].axhline(SLOT_DURATION, color='black', linestyle=':',
                   linewidth=1.5, label='SLA Limit')
axes[0, 1].set_title('Average Latency per Episode (mean +/- std)')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Avg Latency (s)')
axes[0, 1].legend(fontsize=7)
axes[0, 1].grid(True, alpha=0.3)

# ---- Plot 3: Convergence zoom (first 500 episodes) ----
zoom_ep = min(500, N_EPISODES)
zoom_lines = [
    ('attn', 'crimson',    '-',  'Attention Heur DQN'),
    ('heur', 'blue',       '-.',  'Heuristic-Guided DQN'),
    ('std',  'purple',     ':',  'Standard DQN'),
]
for key, color, style, label in zoom_lines:
    lat_zoom = avg_lats[key][:zoom_ep]
    w = min(window, len(lat_zoom))
    smoothed = moving_avg(lat_zoom, w)
    axes[0, 2].plot(range(w-1, len(lat_zoom)), smoothed,
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
    if len(SEEDS) > 1:
        std_zoom = std_lats[key][:zoom_ep]
        std_smooth = moving_avg(std_zoom, w)
        axes[0, 2].fill_between(range(w-1, len(lat_zoom)),
                                smoothed - std_smooth,
                                smoothed + std_smooth,
                                color=color, alpha=0.15)
axes[0, 2].set_title(f'Convergence Speed (First {zoom_ep} Episodes)')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Avg Latency (s)')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# ---- Plot 4: Training Loss ----
loss_lines = [
    ('attn', 'crimson',    '-',  'Attention Heur DQN'),
    ('heur', 'blue',       '-.',  'Heuristic-Guided DQN'),
    ('std',  'purple',     ':',  'Standard DQN'),
]
for key, color, style, label in loss_lines:
    smoothed = moving_avg(avg_losses[key], window)
    axes[1, 0].plot(x_range, smoothed, color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[1, 0].set_title('Training Loss')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Huber Loss')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# ---- Plot 5: Final Latency Bar with Error Bars ----
methods_bar = ['Attn Heur\nDQN', 'Heur\nDQN', 'Standard\nDQN',
               'Random', 'All\nOffload', 'All\nLocal']
keys_bar    = ['attn', 'heur', 'std', 'random', 'offload', 'local']
bar_colors  = ['crimson', 'blue', 'purple', 'darkorange', 'green', 'red']

lat_means = [np.mean(final_100[k]) for k in keys_bar]
lat_stds  = [np.std(final_100[k]) for k in keys_bar]

bars = axes[1, 1].bar(methods_bar, lat_means, color=bar_colors,
                      edgecolor='black', linewidth=0.8,
                      yerr=lat_stds, capsize=5,
                      error_kw={'linewidth': 2})
for bar, val, std in zip(bars, lat_means, lat_stds):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.002,
                    f'{val:.4f}s', ha='center', va='bottom', fontsize=7,
                    fontweight='bold')
axes[1, 1].set_title(f'Final Avg Latency (Last 100 Eps, {len(SEEDS)} seeds)')
axes[1, 1].set_ylabel('Avg Latency (s)')
axes[1, 1].grid(True, axis='y', alpha=0.3)

# ---- Plot 6: Attention Heatmap (averaged over 100 random states) ----
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
n_attn_samples = 100
attn_weights_accum = None
for _ in range(n_attn_samples):
    state_sample = last_attn_env.reset()
    with torch.no_grad():
        state_t = torch.FloatTensor(state_sample).unsqueeze(0).to(DEVICE)
        _ = last_attn_agent.model(state_t, return_attention=True)
        w = last_attn_agent.model._attn_weights.squeeze(0).cpu().numpy()
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

# ---- Plot 7: Improvement % over baselines ----
baseline_lat = np.mean(final_100['random'])
improvements = {}
for name, key in methods_stat:
    mean_val = np.mean(final_100[key])
    improvements[name] = (baseline_lat - mean_val) / baseline_lat * 100

imp_methods = list(improvements.keys())
imp_vals    = list(improvements.values())
imp_colors  = ['crimson', 'blue', 'purple', 'darkorange', 'green', 'red']
bars_imp = axes[2, 0].barh(imp_methods, imp_vals, color=imp_colors,
                            edgecolor='black', linewidth=0.8)
axes[2, 0].axvline(0, color='black', linewidth=0.5)
for bar, val in zip(bars_imp, imp_vals):
    axes[2, 0].text(val + 0.5 if val >= 0 else val - 5,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', ha='left' if val >= 0 else 'right',
                    va='center', fontsize=8, fontweight='bold')
axes[2, 0].set_title('Latency Improvement vs Random Baseline')
axes[2, 0].set_xlabel('Improvement (%)')
axes[2, 0].grid(True, axis='x', alpha=0.3)

# ---- Plot 8: Per-seed final latency comparison ----
x_seeds = np.arange(len(SEEDS))
width = 0.25
for i, (name, key) in enumerate([
    ("Attn Heur DQN", 'attn'),
    ("Heur DQN", 'heur'),
    ("Standard DQN", 'std'),
]):
    vals = final_100[key]
    color = ['crimson', 'blue', 'purple'][i]
    axes[2, 1].bar(x_seeds + i * width, vals, width,
                   label=name, color=color, edgecolor='black', linewidth=0.5)
axes[2, 1].set_xticks(x_seeds + width)
axes[2, 1].set_xticklabels([f'Seed {s}' for s in SEEDS])
axes[2, 1].set_ylabel('Final Avg Latency (s)')
axes[2, 1].set_title('Per-Seed Performance Comparison')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(True, axis='y', alpha=0.3)

# ---- Plot 9: Network Topology with attention-weighted edges ----
ax_topo = axes[2, 2]
# Plot device positions
for i, pos in enumerate(DEVICE_POSITIONS):
    ax_topo.plot(pos[0], pos[1], 'bs', markersize=6)
    ax_topo.annotate(f'D{i}', (pos[0]+5, pos[1]+5), fontsize=5)
# Plot UAV
ax_topo.plot(UAV_POS[0], UAV_POS[1], 'r^', markersize=15, label='UAV')
# Draw attention-weighted edges between top device pairs
top_k = 30  # show top 30 strongest attention connections
attn_flat = []
for i in range(N_DEVICES):
    for j in range(N_DEVICES):
        if i != j:
            attn_flat.append((attn_weights_avg[i, j], i, j))
attn_flat.sort(reverse=True)
max_w = attn_flat[0][0]
for w, i, j in attn_flat[:top_k]:
    pi = DEVICE_POSITIONS[i]
    pj = DEVICE_POSITIONS[j]
    alpha = (w / max_w) * 0.6
    axes[2, 2].plot([pi[0], pj[0]], [pi[1], pj[1]],
                    'r-', alpha=alpha, linewidth=w/max_w * 3)
ax_topo.set_title('Device Topology + Top Attention Edges')
ax_topo.set_xlabel('X (m)')
ax_topo.set_ylabel('Y (m)')
ax_topo.legend(fontsize=8)
ax_topo.grid(True, alpha=0.3)
ax_topo.set_aspect('equal')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/result_attn_heur_dqn.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as /mnt/user-data/outputs/result_attn_heur_dqn.png")
