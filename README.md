# ONTS Scheduling with GNN + Double DQN (PyTorch Geometric)

This repository contains a research/prototype implementation of a Reinforcement Learning (RL) agent for an **ONTS-like task scheduling problem** (jobs × time steps) under **energy and job constraints**, using a **Graph Neural Network (GNN)** as a Q-network.

The agent is trained with a **DQN-style algorithm** enhanced with:
- **Target Network**
- **Double DQN targets**
- **Experience Replay** storing `(state, action, reward, next_state, done)` correctly

Comments in the code are in Portuguese.

---

## Problem overview (ONTS in this repo)

We schedule **J jobs** over **T time steps**. The schedule is a binary matrix:

- `x[j, t] ∈ {0,1}` indicates whether job `j` is active at time `t`.

### Action space
The agent chooses an action in `0..(J*T-1)` and it is mapped to:

- `job, time_step = divmod(action, T)`

Then the environment **toggles** the selected decision:

- `x[j, t] = 1 - x[j, t]`

---

## Environment

The environment is implemented in `ONTSEnv` and maintains:
- `x_state`: schedule matrix
- `phi_state`: auxiliary matrix tracking activations/transitions
- `SoC_t`: battery state of charge (SoC)

### Reward and termination
Reward is computed by:
1. `check_energy_constraints()`  
   - penalizes and **terminates** on energy violations or SoC overflow  
2. `check_job_constraints()`  
   - returns penalties for violations of job-specific constraints  
3. If no penalties occurred, adds a **positive reward** based on:
   - job priority and energy slack

Episode ends when:
- a terminal constraint is violated, or
- `steps_taken >= max_steps`

---

## Graph representation (PyTorch Geometric)

Each node corresponds to a pair `(job, time_step)`.

### Nodes
- total nodes: `J*T`

### Node features (enriched)
Each node uses 4 features:
1. `x[j,t]` (0/1)
2. job priority (normalized)
3. job energy consumption (normalized)
4. available energy at time `t` (normalized)

### Edges
Two types of edges are created:

1) **Temporal edges** (per job)  
Connect consecutive time steps within the same job:
- `(job,t) ↔ (job,t+1)`

2) **Optional time-coupling edges** (across jobs)  
Connect jobs at the same time step (captures energy competition):
- `(job,t) ↔ (job+1,t)`

Enable/disable with:
- `add_time_coupling_edges=True/False` in `ONTSEnv(...)`

---

## Model: GNN Q-network

The Q-network is a **GCN + pooling + MLP head**:

- `GCNConv(in_channels → 128 → 128 → 64)`
- `global_mean_pool`
- MLP: `64 → 32 → (J*T)` Q-values

The forward pass supports both:
- single `Data` graphs and
- batched `Batch` graphs

---

## Algorithm: Double DQN + Target Network

Training uses:
- ε-greedy exploration with exponential decay
- replay buffer sampling
- **Double DQN target**:
  - action selection via `policy_net`
  - value evaluation via `target_net`

TD target:
- `target = r + (1 - done) * gamma * Q_target(s', argmax_a Q_policy(s',a))`

Target network is synced every `target_update_every` optimization steps.

---

## Requirements

- Python 3.9+ recommended
- PyTorch
- PyTorch Geometric (`torch-geometric`)

Install (generic):
```bash
pip install torch
pip install torch-geometric
