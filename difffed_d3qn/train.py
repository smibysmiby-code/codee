"""Training loops for D3QN, FedAvg-D3QN, SynthER, and DiffFed-D3QN."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .baselines import AllLocalPolicy, AllUAVPolicy, DQNAgent, DuelingDQNAgent, RandomPolicy
from .d3qn_agent import D3QNAgent, D3QNConfig
from .diffusion import ConditionalDiffusion, DiffusionConfig, filter_by_q
from .environment import EnvConfig, UAVMECEnv
from .federated import aggregate


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    num_episodes: int = 200
    max_steps: int = 100
    num_users: int = 50
    omega_delay: float = 0.5
    omega_energy: float = 0.5
    warmup_steps: int = 500
    diffusion_train_epochs: int = 60
    diffusion_regen_every: int = 20      # episodes
    synth_ratio: float = 0.3
    num_clients: int = 5
    fed_every: int = 5                   # episodes
    seed: int = 0
    device: str = "cpu"
    quality_filter: bool = True


def _eval_episode(env: UAVMECEnv, policy, max_steps: int) -> Dict[str, float]:
    s = env.reset()
    total_r = 0.0
    delays, energies, costs = [], [], []
    for _ in range(max_steps):
        a = policy.act(s, greedy=True) if hasattr(policy, 'act') else policy(s)
        s, r, done, info = env.step(a)
        total_r += r
        delays.append(info['delay'])
        energies.append(info['energy'])
        costs.append(info['system_cost'])
        if done:
            break
    return {
        'reward': total_r,
        'delay': float(np.mean(delays)),
        'energy': float(np.mean(energies)),
        'system_cost': float(np.mean(costs)),
    }


# ---------------------------------------------------------------------------
# Episode rollout helpers (per-user transitions stored in the buffer)
# ---------------------------------------------------------------------------

def _rollout_into_buffer(env, agent, max_steps, ctx_collector=None) -> Tuple[float, List]:
    s = env.reset()
    total_r = 0.0
    transitions = []
    for _ in range(max_steps):
        a = agent.act(s, greedy=False)
        ns, r, done, info = env.step(a)
        for i in range(env.num_users):
            agent.buffer.push(s[i], int(a[i]), float(r), ns[i], bool(done))
            transitions.append((s[i].copy(), int(a[i]), float(r), ns[i].copy()))
            if ctx_collector is not None:
                ctx_collector.append(env.network_context.copy())
        total_r += r
        s = ns
        if done:
            break
    return total_r, transitions


# ---------------------------------------------------------------------------
# Simple training (no synth, no FL): used for baseline DQN variants and D3QN.
# ---------------------------------------------------------------------------

def train_simple(agent, env: UAVMECEnv, cfg: TrainConfig,
                 log_every: int = 1) -> Dict[str, List[float]]:
    history = {'reward': [], 'delay': [], 'energy': [], 'cost': []}
    for ep in range(cfg.num_episodes):
        ep_r, _ = _rollout_into_buffer(env, agent, cfg.max_steps)
        for _ in range(cfg.max_steps):
            agent.update()
        agent.decay_epsilon()
        ev = _eval_episode(env, agent, cfg.max_steps)
        for k, v in zip(['reward', 'delay', 'energy', 'cost'],
                        [ev['reward'], ev['delay'], ev['energy'], ev['system_cost']]):
            history[k].append(v)
    return history


# ---------------------------------------------------------------------------
# Synthetic-augmented training (SynthER-style: unconditional / no FL)
# ---------------------------------------------------------------------------

def _make_diffusion(cfg: TrainConfig, conditional: bool = True) -> ConditionalDiffusion:
    dcfg = DiffusionConfig(
        T=100, hidden_dim=128, device=cfg.device,
        cond_dim=4 if conditional else 1, state_dim=6, action_dim=2,
    )
    return ConditionalDiffusion(dcfg)


def _augmented_batch(agent: D3QNAgent, synth_pool, ratio: float, batch_size: int):
    n_synth = int(round(batch_size * ratio)) if synth_pool is not None and len(synth_pool[0]) > 0 else 0
    n_real = batch_size - n_synth
    real = agent.buffer.sample(max(n_real, 1))
    rs, ra, rr, rns, rd = real
    if n_synth > 0:
        ss, sa, sr, sns = synth_pool
        idx = np.random.randint(0, len(ss), size=n_synth)
        rs = np.concatenate([rs[:n_real], ss[idx]], axis=0)
        ra = np.concatenate([ra[:n_real], sa[idx]], axis=0)
        rr = np.concatenate([rr[:n_real], sr[idx]], axis=0)
        rns = np.concatenate([rns[:n_real], sns[idx]], axis=0)
        rd = np.concatenate([rd[:n_real], np.zeros(n_synth, dtype=np.float32)], axis=0)
    return rs, ra, rr, rns, rd


def train_with_diffusion(agent: D3QNAgent, env: UAVMECEnv, cfg: TrainConfig,
                         conditional: bool = True,
                         use_federated: bool = False,
                         num_clients: int = 1,
                         agg_every: Optional[int] = None,
                         random_gen: bool = False) -> Dict[str, List[float]]:
    """Trains a (set of) D3QN agent(s) with optional synthetic data and FL."""
    if use_federated and num_clients > 1:
        clients = [agent] + [_clone_agent(agent) for _ in range(num_clients - 1)]
        envs = [env] + [UAVMECEnv(EnvConfig(num_users=cfg.num_users,
                                            omega_delay=cfg.omega_delay,
                                            omega_energy=cfg.omega_energy,
                                            max_steps=cfg.max_steps,
                                            seed=cfg.seed + 17 * i))
                        for i in range(1, num_clients)]
    else:
        clients = [agent]
        envs = [env]

    diffusion = _make_diffusion(cfg, conditional=conditional)
    synth_pool = None
    contexts_seen: List[np.ndarray] = []
    experiences_seen: List = []

    history = {'reward': [], 'delay': [], 'energy': [], 'cost': []}
    agg_interval = agg_every if agg_every is not None else cfg.fed_every

    for ep in range(cfg.num_episodes):
        # Rollouts (each client into its own env / buffer)
        for ag, en in zip(clients, envs):
            ep_r, transitions = _rollout_into_buffer(en, ag, cfg.max_steps,
                                                    ctx_collector=contexts_seen)
            experiences_seen.extend(transitions)

        # Phase B: train diffusion once enough data exists
        if not random_gen and (len(experiences_seen) >= cfg.warmup_steps) and ep % cfg.diffusion_regen_every == 0:
            ctxs = contexts_seen if conditional else [np.zeros(1, dtype=np.float32) for _ in experiences_seen]
            diffusion.fit(experiences_seen, ctxs, epochs=cfg.diffusion_train_epochs, batch_size=64)
            # Generate a fresh pool
            n_gen = max(256, cfg.max_steps * 4)
            if conditional:
                ctx_vec = envs[0].network_context
            else:
                ctx_vec = np.zeros(1, dtype=np.float32)
            s_g, a_g, r_g, ns_g = diffusion.generate(n_gen, ctx_vec)
            s_g, a_g, r_g, ns_g = (s_g.cpu().numpy(), a_g.cpu().numpy(),
                                    r_g.cpu().numpy(), ns_g.cpu().numpy())
            if cfg.quality_filter:
                s_g, a_g, r_g, ns_g = filter_by_q(clients[0], s_g, a_g, r_g, ns_g, drop_frac=0.1)
            synth_pool = (s_g, a_g, r_g, ns_g)
        elif random_gen and ep == 0 and len(experiences_seen) >= cfg.warmup_steps:
            # Random generation: just bootstrap-sample from the buffer with
            # additive Gaussian noise (no learned generator).
            s_arr = np.array([t[0] for t in experiences_seen])
            a_arr = np.array([t[1] for t in experiences_seen])
            r_arr = np.array([t[2] for t in experiences_seen])
            ns_arr = np.array([t[3] for t in experiences_seen])
            idx = np.random.randint(0, len(s_arr), 1024)
            synth_pool = (
                s_arr[idx] + np.random.randn(*s_arr[idx].shape) * 0.1,
                a_arr[idx],
                r_arr[idx] + np.random.randn(len(idx)) * 0.05,
                ns_arr[idx] + np.random.randn(*ns_arr[idx].shape) * 0.1,
            )

        # Phase C: agent updates on mixed batches
        for ag in clients:
            for _ in range(cfg.max_steps):
                if synth_pool is None or cfg.synth_ratio <= 0.0:
                    ag.update()
                else:
                    if len(ag.buffer) < ag.cfg.batch_size:
                        continue
                    batch = _augmented_batch(ag, synth_pool, cfg.synth_ratio, ag.cfg.batch_size)
                    ag.update_from_batch(*batch)
            ag.decay_epsilon()

        # Phase D: FedAvg
        if use_federated and num_clients > 1 and (ep + 1) % agg_interval == 0:
            aggregate(clients)

        ev = _eval_episode(envs[0], clients[0], cfg.max_steps)
        history['reward'].append(ev['reward'])
        history['delay'].append(ev['delay'])
        history['energy'].append(ev['energy'])
        history['cost'].append(ev['system_cost'])

    return history


def _clone_agent(agent: D3QNAgent) -> D3QNAgent:
    new = D3QNAgent(deepcopy(agent.cfg))
    new.load_state_dict(deepcopy(agent.state_dict()))
    new.eps = agent.eps
    return new


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_env(cfg: TrainConfig) -> UAVMECEnv:
    return UAVMECEnv(EnvConfig(
        num_users=cfg.num_users,
        omega_delay=cfg.omega_delay,
        omega_energy=cfg.omega_energy,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
    ))


def make_d3qn(cfg: TrainConfig) -> D3QNAgent:
    return D3QNAgent(D3QNConfig(device=cfg.device))


def evaluate_policy(policy, cfg: TrainConfig, episodes: int = 5) -> Dict[str, float]:
    env = make_env(cfg)
    accum = {'reward': [], 'delay': [], 'energy': [], 'system_cost': []}
    for _ in range(episodes):
        ev = _eval_episode(env, policy, cfg.max_steps)
        for k in accum:
            accum[k].append(ev[k])
    return {k: float(np.mean(v)) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Method dispatcher: produces one (history, policy) pair per named method.
# ---------------------------------------------------------------------------

def run_method(name: str, cfg: TrainConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg)
    name = name.lower()
    if name == 'random':
        pol = RandomPolicy()
        hist = {'reward': [], 'delay': [], 'energy': [], 'cost': []}
        for _ in range(cfg.num_episodes):
            ev = _eval_episode(env, pol, cfg.max_steps)
            hist['reward'].append(ev['reward'])
            hist['delay'].append(ev['delay'])
            hist['energy'].append(ev['energy'])
            hist['cost'].append(ev['system_cost'])
        return hist, pol
    if name == 'all-local':
        pol = AllLocalPolicy()
        hist = {'reward': [], 'delay': [], 'energy': [], 'cost': []}
        for _ in range(cfg.num_episodes):
            ev = _eval_episode(env, pol, cfg.max_steps)
            hist['reward'].append(ev['reward'])
            hist['delay'].append(ev['delay'])
            hist['energy'].append(ev['energy'])
            hist['cost'].append(ev['system_cost'])
        return hist, pol
    if name == 'all-uav':
        pol = AllUAVPolicy()
        hist = {'reward': [], 'delay': [], 'energy': [], 'cost': []}
        for _ in range(cfg.num_episodes):
            ev = _eval_episode(env, pol, cfg.max_steps)
            hist['reward'].append(ev['reward'])
            hist['delay'].append(ev['delay'])
            hist['energy'].append(ev['energy'])
            hist['cost'].append(ev['system_cost'])
        return hist, pol
    if name == 'dqn':
        ag = DQNAgent(D3QNConfig(device=cfg.device))
        return train_simple(ag, env, cfg), ag
    if name == 'dueling-dqn':
        ag = DuelingDQNAgent(D3QNConfig(device=cfg.device))
        return train_simple(ag, env, cfg), ag
    if name == 'd3qn':
        ag = make_d3qn(cfg)
        return train_simple(ag, env, cfg), ag
    if name == 'fedavg-d3qn':
        ag = make_d3qn(cfg)
        new_cfg = deepcopy(cfg)
        new_cfg.synth_ratio = 0.0
        return train_with_diffusion(ag, env, new_cfg, conditional=False,
                                     use_federated=True, num_clients=cfg.num_clients), ag
    if name == 'synther':
        ag = make_d3qn(cfg)
        return train_with_diffusion(ag, env, cfg, conditional=False, use_federated=False), ag
    if name == 'difffed-d3qn':
        ag = make_d3qn(cfg)
        return train_with_diffusion(ag, env, cfg, conditional=True,
                                     use_federated=True,
                                     num_clients=cfg.num_clients), ag
    raise ValueError(f"unknown method: {name}")
