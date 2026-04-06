# 🏎️ 3D Car Racing with ML Agents and Reinforcement Learning

An autonomous racing agent trained using **Proximal Policy Optimization (PPO)** in the **Donkey Car Simulator**. The agent fuses visual (CNN) and sensor (MLP) inputs to learn real-time steering and throttle control on complex 3D tracks.

> **Course Project — Group 43** | UC Irvine  
> [🔗 Repository](https://github.com/pysac17/aiProject/tree/main)

---

## 📌 Problem Statement

Autonomous racing presents several unique RL challenges:

- **High-Dimensional State Space** — raw pixel input from a forward-facing camera
- **Continuous Action Space** — fine-grained steering `[-1, 1]` and throttle `[0, 1]` control
- **Delayed Rewards** — consequences of decisions (e.g., taking a turn too fast) are not immediately apparent
- **Partial Observability** — the camera provides only a limited forward field of view

---

## 🧠 Architecture

A **sensor-fusion actor-critic** network with two separate input pathways that are merged before the policy and value heads.

### Visual Pathway (CNN)

Processes a `120×160×3` RGB image through 3 convolutional blocks:

```
Input (640×480×3)
  → Conv Block 1: 32 filters, 3×3, stride=2  →  320×240×32
  → Conv Block 2: 64 filters, 3×3, stride=2  →  160×120×64
  → Conv Block 3: 64 filters, 3×3, stride=1  →  80×60×64
  → Adaptive Pooling                          →  64×11×16 feature maps
  → Flatten
```

Effective receptive field of the final layer: ~18×18 pixels — captures lane boundaries and track edges.

### Sensor Pathway (MLP)

A 2-layer MLP takes a 4-dimensional sensor vector as input:

| Input Feature | Description |
|---|---|
| Speed | Current vehicle speed |
| Position X | Relative track position (x-axis) |
| Position Y | Relative track position (y-axis) |
| Track Angle | Angle relative to track direction (radians) |

Output: 64-dimensional embedding encoding speed, position, and track angle.

### Policy & Value Heads

After concatenating CNN features + sensor embedding:

- **Policy Head** — outputs mean (μ) and log std (log σ) of a Gaussian distribution over actions. Actions passed through `tanh` to enforce bounds.
- **Value Head** — outputs scalar state-value estimate V(s).

---

## ⚙️ PPO Algorithm

Uses **Proximal Policy Optimization** with a clipped objective:

```
L_CLIP(θ) = E[ min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]
```

Where `r_t(θ)` is the new/old policy probability ratio and `A_t` is the advantage estimate.

**Key techniques:**
- **GAE** (Generalized Advantage Estimation) with γ = 0.995, λ = 0.95
- **Entropy regularization** to encourage exploration
- **Gradient clipping** for training stability
- **Warm-start** phase using rule-based actions early in training
- **Action smoothing** to prevent jerky control

---

## 🏆 Reward Design

| Component | Formula |
|---|---|
| Progress | `r = Δd / Δt` (change in track progress over time) |
| Speed Alignment | `r = v · cos(θ)` (speed projected onto track direction) |
| Centerline Following | `r = 1 - (d / d_max)²` (distance from centerline) |
| Collision Penalty | `-50 to -400` |
| Off-track Penalty | `-80` |
| Steering Penalty | `-100 · |steer|` |

---

## 📊 Results

### Quantitative

| Metric | Ours | Baseline | Improvement |
|---|---|---|---|
| Success Rate | 92% | 78% | +14% |
| Avg. Speed | 3.2 m/s | 2.8 m/s | +14.3% |
| Collisions/Lap | 0.3 | 1.2 | −75% |

### Qualitative Behaviors Observed

- Smooth cornering with speed reduction before sharp turns
- Higher throttle on straights
- Gentle self-correction when drifting toward track edges
- Occasionally over-conservative on very tight corners (reflects uncertainty)

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | 2e-4 (Adam) |
| Batch Size | 64 |
| Episodes | 200 |
| Discount (γ) | 0.995 |
| Lambda (λ) for GAE | 0.95 |
| Clip Range (ε) | 0.2 |
| Entropy Coefficient | 0.08 |

---

## 🗂️ Project Structure

```
.
├── models/
│   └── network.py        # CNN + MLP sensor-fusion actor-critic
├── train.py              # PPO training loop
├── eval.py               # Evaluation and rollout scripts
├── rewards.py            # Reward shaping logic
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Donkey Car Simulator

Follow the setup guide at [Donkey Car Simulator Docs](https://docs.donkeycar.com/guide/simulator/).

### 3. Train the agent

```bash
python train.py
```

### 4. Evaluate

```bash
python eval.py --model checkpoints/best_model.pt
```

---

## 📚 References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Donkey Car Community. *Donkey Car Simulator Documentation.*
- Lillicrap et al. (2016). *Continuous Control with Deep Reinforcement Learning.* ICLR.
- Brockman et al. (2016). *OpenAI Gym.* arXiv:1606.01540.
