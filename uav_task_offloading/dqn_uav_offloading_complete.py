#!/usr/bin/env python3
"""
================================================================================
ATTENTION-ENHANCED HEURISTIC-GUIDED DQN FOR UAV TASK OFFLOADING
================================================================================
Compares Standard DQN vs Dueling DQN vs Heuristic-Guided Dueling DQN
vs Attention-Enhanced Heuristic Dueling DQN (proposed)
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
8. Heuristic warmup phase pre-fills replay buffer with quality experiences
9. 10 devices / 1024 actions -- large action space where guidance matters
10. Multi-Head Self-Attention: devices attend to each other's states,
    learning inter-device dependencies (e.g. shared UAV queue contention)
    for coordinated offloading decisions -- the proposed enhancement
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
# =============================================================
#  SYSTEM PARAMETERS
# =============================================================
SLOT_DURATION  = 0.1
TASK_PROB      = 0.7
N_DEVICES      = 10
N_SLOTS        = 300
N_EPISODES     = 300
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
]
DROP_PENALTY = 0.15
DISTANCES    = [np.linalg.norm(UAV_POS - dp) for dp in DEVICE_POSITIONS]
# Heuristic guidance settings
HEUR_EPSILON_THRESHOLD = 0.6   # use heuristic when epsilon > 0.6
HEUR_CALL_PROB         = 0.50  # 50% of exploration steps use heuristic
HEUR_NOISE_PROB        = 0.05  # 5% chance to flip each device decision
WARMUP_STEPS           = 2000  # pre-fill replay buffer with heuristic demos
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
                state.append(DISTANCES[i] / 350)
                state.append(1.0)
            else:
                state.append(0.0)
                state.append(0.0)
                state.append(DISTANCES[i] / 350)
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
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
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
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))
# =============================================================
#  ATTENTION-ENHANCED DUELING DQN NETWORK  (proposed)
# =============================================================
class AttentionDuelingDQNNet(nn.Module):
    """
    Dueling DQN whose feature extractor uses Multi-Head Self-Attention
    over per-device token embeddings.  Each device's 4-dim state vector
    (data_size, cpu_cycles, distance, has_task) is projected into an
    embedding, positional-encoded, then fed through a Transformer-style
    self-attention block.  This lets every device's representation be
    informed by all other devices' workloads and channel conditions --
    critical for coordinated offloading under a shared UAV queue.

    Architecture:
        state (41-d) -> split into 10 device tokens (4-d each) + 1 global (queue)
                     -> Linear projection to d_model
                     -> + learned positional encoding
                     -> Multi-Head Self-Attention (2 heads, 2 layers)
                     -> concat attentive features + global queue
                     -> Dueling value / advantage streams -> Q-values
    """
    FEATURES_PER_DEVICE = 4   # (data_size, cpu_cycles, distance, has_task)

    def __init__(self, state_size, action_size,
                 d_model=64, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_devices   = N_DEVICES
        self.d_model     = d_model
        self.action_size = action_size

        # --- Per-device token projection ---
        self.token_proj = nn.Linear(self.FEATURES_PER_DEVICE, d_model)

        # --- Learned positional encoding for device slots ---
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_devices, d_model) * 0.02)

        # --- Transformer encoder layers ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Global queue feature projection ---
        self.queue_proj = nn.Linear(1, d_model)

        # --- Merge: attention features + queue -> shared representation ---
        merge_dim = d_model * self.n_devices + d_model
        self.merge = nn.Sequential(
            nn.Linear(merge_dim, 512),
            nn.ReLU(),
        )

        # --- Dueling streams ---
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, action_size)
        )

        # For extracting attention weights during visualisation
        self._attn_weights = None

    def forward(self, x, return_attention=False):
        batch = x.shape[0]

        # Split state into per-device tokens and global queue feature
        device_feats = x[:, :self.n_devices * self.FEATURES_PER_DEVICE]
        device_feats = device_feats.view(batch, self.n_devices, self.FEATURES_PER_DEVICE)
        queue_feat   = x[:, -1:]                        # (batch, 1)

        # Project device tokens + add positional encoding
        tokens = self.token_proj(device_feats)           # (batch, n_dev, d_model)
        tokens = tokens + self.pos_embed

        # Self-attention across devices
        if return_attention:
            attn_out, self._attn_weights = self._forward_with_attn(tokens)
        else:
            attn_out = self.transformer(tokens)          # (batch, n_dev, d_model)

        # Flatten attentive device representations
        attn_flat = attn_out.reshape(batch, -1)          # (batch, n_dev * d_model)

        # Queue embedding
        queue_emb = self.queue_proj(queue_feat)          # (batch, d_model)

        # Merge
        merged = self.merge(torch.cat([attn_flat, queue_emb], dim=1))

        # Dueling
        v = self.value_stream(merged)
        a = self.advantage_stream(merged)
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q

    def _forward_with_attn(self, tokens):
        """Run transformer manually to capture attention weights."""
        x = tokens
        weights_all = []
        for layer in self.transformer.layers:
            # Self-attention with weight capture
            x2, w = layer.self_attn(x, x, x, need_weights=True,
                                    average_attn_weights=True)
            weights_all.append(w.detach())
            x = layer.norm1(x + layer.dropout1(x2))
            x = layer.norm2(x + layer._ff_block(x))
        # Average attention across layers
        avg_w = torch.stack(weights_all).mean(0)         # (batch, n_dev, n_dev)
        return x, avg_w

# =============================================================
#  GENERIC AGENT -- Standard or Dueling, with optional Heuristic
# =============================================================
class DQNAgent:
    def __init__(self, state_size, action_size, dueling=False,
                 use_heuristic=False, use_attention=False):
        self.state_size      = state_size
        self.action_size     = action_size
        self.use_heuristic   = use_heuristic
        self.use_attention   = use_attention
        self.heuristic_calls = 0
        self.memory        = deque(maxlen=50000)
        self.batch_size    = 128
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.99995
        self.lr            = 0.0003
        self.tau           = 0.005
        if use_attention:
            NetClass = AttentionDuelingDQNNet
        elif dueling:
            NetClass = DuelingDQNNet
        else:
            NetClass = DQNNet
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
        if random.random() < self.epsilon:
            if (self.use_heuristic
                    and tasks is not None
                    and self.epsilon > HEUR_EPSILON_THRESHOLD
                    and random.random() < HEUR_CALL_PROB):
                self.heuristic_calls += 1
                return heuristic_suggest_action(state, tasks)
            else:
                return random.randrange(self.action_size)
        else:
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
#  HEURISTIC WARMUP
# =============================================================
def warmup_with_heuristic(agent, warmup_steps):
    env_w = UAVEnvironment()
    state = env_w.reset()
    for _ in range(warmup_steps):
        action     = heuristic_suggest_action(state, env_w.current_tasks)
        next_state, reward, _, _, _, _ = env_w.step(action)
        agent.remember(state, action, reward, next_state)
        state = next_state
        if random.random() < 1.0 / N_SLOTS:
            state = env_w.reset()
    print(f"  Warmup complete: {warmup_steps} heuristic transitions stored")
# =============================================================
#  TRAINING FUNCTION
# =============================================================
def train_agent(state_size, action_size, dueling, use_heuristic, label,
                use_attention=False):
    # Reset seeds before each agent for fair comparison
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    agent = DQNAgent(state_size, action_size,
                     dueling=dueling, use_heuristic=use_heuristic,
                     use_attention=use_attention)
    env   = UAVEnvironment()
    if use_heuristic:
        warmup_with_heuristic(agent, WARMUP_STEPS)
    latencies, drops, rewards, losses = [], [], [], []
    print(f"\n{'=' * 70}")
    print(f"  {label} -- {N_DEVICES} Devices, 1 UAV, {N_EPISODES} Episodes")
    print(f"  Action space: {action_size} actions")
    if use_heuristic:
        print(f"  Heuristic guidance: ON  (eps>{HEUR_EPSILON_THRESHOLD}, "
              f"call_prob={HEUR_CALL_PROB}, warmup={WARMUP_STEPS})")
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
env_tmp     = UAVEnvironment()
state_size  = env_tmp.state_size
action_size = env_tmp.action_size
print(f"State size: {state_size}, Action space: {action_size} ({N_DEVICES} devices)")
print(f"Device distances: {[f'{d:.0f}m' for d in DISTANCES]}")
# 1. Standard DQN
std_lat,  std_drop,  std_rew,  std_loss,  _, std_agent  = train_agent(
    state_size, action_size,
    dueling=False, use_heuristic=False,
    label="Standard DQN")
# 2. Dueling DQN
duel_lat, duel_drop, duel_rew, duel_loss, _, duel_agent = train_agent(
    state_size, action_size,
    dueling=True, use_heuristic=False,
    label="Dueling DQN")
# 3. Heuristic-Guided Dueling DQN
heur_lat, heur_drop, heur_rew, heur_loss, env_heur, heur_agent = train_agent(
    state_size, action_size,
    dueling=True, use_heuristic=True,
    label="Heuristic-Guided Dueling DQN")
# 4. Attention-Enhanced Heuristic Dueling DQN -- proposed method
attn_lat, attn_drop, attn_rew, attn_loss, env_attn, attn_agent = train_agent(
    state_size, action_size,
    dueling=True, use_heuristic=True, use_attention=True,
    label="Attention-Enhanced Heur Dueling DQN")
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
#  CONVERGENCE SPEED COMPARISON
# =============================================================
window = 30
def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')
print(f"\n{'=' * 60}")
print(f"  CONVERGENCE SPEED")
print(f"{'=' * 60}")
target_lat = np.mean(attn_lat[-50:]) * 1.05
for name, lat in [("Attn Heur Dueling DQN", attn_lat),
                  ("Heur Dueling DQN", heur_lat),
                  ("Dueling DQN", duel_lat),
                  ("Standard DQN", std_lat)]:
    smoothed = moving_avg(lat, window)
    converged = [i for i, l in enumerate(smoothed) if l < target_lat]
    if converged:
        print(f"  {name:<24}: reached {target_lat:.4f}s at episode {converged[0] + window}")
    else:
        print(f"  {name:<24}: did NOT reach {target_lat:.4f}s")
# =============================================================
#  FINAL SUMMARY
# =============================================================
print(f"\n{'=' * 60}")
print(f"  FINAL COMPARISON  ({N_DEVICES} devices, {action_size} actions)")
print(f"{'=' * 60}")
results = [
    ("Attn Heur Dueling",  np.mean(attn_lat[-50:])),
    ("Heur Dueling DQN",   np.mean(heur_lat[-50:])),
    ("Dueling DQN",        np.mean(duel_lat[-50:])),
    ("Standard DQN",       np.mean(std_lat[-50:])),
    ("Random",             np.mean(random_lat[-50:])),
    ("All Offload",        np.mean(offload_lat[-50:])),
    ("All Local",          np.mean(local_lat[-50:])),
]
best = results[0][1]
for name, val in results:
    improvement = (1 - best/val)*100 if val != best else 0
    marker = " <- proposed (attention)" if name == "Attn Heur Dueling" else \
             f"  ({improvement:.1f}% worse)" if improvement > 0 else ""
    print(f"  {name:<22}: {val:.4f}s{marker}")
# =============================================================
#  PLOTS -- 2x3 grid
# =============================================================
os.makedirs('/mnt/user-data/outputs', exist_ok=True)
x_range = range(window - 1, N_EPISODES)
fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.suptitle(
    f"Attention-Enhanced Heuristic DQN for UAV Task Offloading -- {N_DEVICES} Devices, 1 UAV",
    fontsize=14, fontweight='bold')
all_lines = [
    (attn_rew,   attn_lat,   'crimson',    '-',  'Attn Heur Dueling DQN (proposed)'),
    (heur_rew,   heur_lat,   'blue',       '-.',  'Heur Dueling DQN'),
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
# ---- Plot 3: Convergence zoom (first 400 episodes) ----
zoom_ep = 400
zoom_lines = [
    (attn_lat[:zoom_ep], 'crimson',    '-',  'Attn Heur Dueling DQN'),
    (heur_lat[:zoom_ep], 'blue',       '-.',  'Heur Dueling DQN'),
    (duel_lat[:zoom_ep], 'deepskyblue','--', 'Dueling DQN'),
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
# ---- Plot 4: Drop Rate ----
drop_lines = [
    (attn_drop, 'crimson',    '-',  'Attn Heur Dueling DQN'),
    (heur_drop, 'blue',       '-.',  'Heur Dueling DQN'),
    (duel_drop, 'deepskyblue','--', 'Dueling DQN'),
    (std_drop,  'purple',     ':',  'Standard DQN'),
]
for drop, color, style, label in drop_lines:
    axes[0, 3].plot(drop, alpha=0.15, color=color)
    axes[0, 3].plot(x_range, moving_avg(drop, window),
                    color=color, linewidth=2.5,
                    linestyle=style, label=label)
axes[0, 3].set_title('Task Drop Rate')
axes[0, 3].set_xlabel('Episode')
axes[0, 3].set_ylabel('Drop Rate')
axes[0, 3].legend(fontsize=8)
axes[0, 3].grid(True)
# ---- Plot 5: Training Loss ----
for loss, color, style, label in [
    (attn_loss, 'crimson',    '-',  'Attn Heur Dueling DQN'),
    (heur_loss, 'blue',       '-.',  'Heur Dueling DQN'),
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
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True)
# ---- Plot 6: Final Latency Bar ----
methods    = ['Attn Heur\nDueling', 'Heur\nDueling', 'Dueling\nDQN',
              'Standard\nDQN', 'Random', 'All\nOffload', 'All\nLocal']
lat_vals   = [np.mean(attn_lat[-50:]),    np.mean(heur_lat[-50:]),
              np.mean(duel_lat[-50:]),     np.mean(std_lat[-50:]),
              np.mean(random_lat[-50:]),
              np.mean(offload_lat[-50:]),  np.mean(local_lat[-50:])]
bar_colors = ['crimson', 'blue', 'deepskyblue', 'purple',
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
# ---- Plot 7: Per-device Latency (Attn-Heur-Dueling DQN) ----
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
state = env_attn.reset()
d_lat = [[] for _ in range(N_DEVICES)]
d_dec = [[] for _ in range(N_DEVICES)]
for slot in range(N_SLOTS):
    action = attn_agent.act(state)
    next_state, _, _, info, _, _ = env_attn.step(action)
    state = next_state
    for x in info:
        i = x["device"]
        if x["action"] != "no task":
            d_lat[i].append(x["latency"])
            d_dec[i].append(1 if x["action"] == "offload" else 0)
labels   = [f"D{i}\n({DISTANCES[i]:.0f}m)" for i in range(N_DEVICES)]
avgs     = [np.mean(d_lat[i]) if d_lat[i] else 0 for i in range(N_DEVICES)]
offrates = [np.mean(d_dec[i]) if d_dec[i] else 0 for i in range(N_DEVICES)]
dev_colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6',
              '#1abc9c','#e67e22','#34495e','#27ae60','#c0392b']
bars = axes[1, 2].bar(labels, avgs, color=dev_colors,
                      edgecolor='black', linewidth=0.8)
for bar, rate in zip(bars, offrates):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{rate:.0%}', ha='center', va='bottom', fontsize=7)
axes[1, 2].axhline(SLOT_DURATION, color='black', linestyle='--',
                   label='Slot limit')
axes[1, 2].set_title('Per-Device Latency: Attn Heur Dueling DQN')
axes[1, 2].set_ylabel('Avg Latency (s)')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, axis='y')
# ---- Plot 8: Attention Heatmap ----
# Extract learned attention weights from the proposed model
random.seed(SEED)
np.random.seed(SEED)
state_sample = env_attn.reset()
with torch.no_grad():
    state_t = torch.FloatTensor(state_sample).unsqueeze(0)
    _ = attn_agent.model(state_t, return_attention=True)
    attn_weights = attn_agent.model._attn_weights.squeeze(0).numpy()
dev_labels = [f"D{i}" for i in range(N_DEVICES)]
im = axes[1, 3].imshow(attn_weights, cmap='YlOrRd', aspect='equal',
                       vmin=0, vmax=attn_weights.max())
axes[1, 3].set_xticks(range(N_DEVICES))
axes[1, 3].set_yticks(range(N_DEVICES))
axes[1, 3].set_xticklabels(dev_labels, fontsize=8)
axes[1, 3].set_yticklabels(dev_labels, fontsize=8)
axes[1, 3].set_xlabel('Key Device')
axes[1, 3].set_ylabel('Query Device')
axes[1, 3].set_title('Learned Device Attention Weights')
# Annotate cells
for ii in range(N_DEVICES):
    for jj in range(N_DEVICES):
        axes[1, 3].text(jj, ii, f'{attn_weights[ii, jj]:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='white' if attn_weights[ii, jj] > attn_weights.max()*0.6
                        else 'black')
plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/result_attn_heur_dqn.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as /mnt/user-data/outputs/result_attn_heur_dqn.png")
