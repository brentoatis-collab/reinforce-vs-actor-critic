# Policy Gradient vs Actor-Critic in GridWorld

**Author:** Brent Oatis  
**Course:** CST 643 – Reinforcement Learning  
**Institution:** Concordia University St. Paul  

---

## 📌 Project Overview

This project implements and compares two reinforcement learning algorithms:

- **REINFORCE (Policy Gradient)**
- **Actor-Critic**

Both agents are trained in a custom **GridWorld environment** to navigate from a start position to a goal while avoiding obstacles.

The objective is to evaluate and compare:
- Learning stability  
- Convergence speed  
- Task success performance  

---

## 🧠 Key Concepts

- Policy Gradient Methods  
- Stochastic Policies  
- Advantage Estimation  
- Temporal Difference (TD) Learning  
- Variance Reduction using Value Functions  

---

## 🗺️ GridWorld Environment

- **Grid Size:** 5x5  
- **Start State:** (0, 0)  
- **Goal State:** (4, 4)  
- **Actions:** up, down, left, right  
- **Rewards:**
  - Step: -1  
  - Obstacle: -10 (episode terminates)  
  - Goal: +10 (episode completes)  

---

## ⚙️ Implementation Details

### 🔹 REINFORCE Agent
- Policy network outputs action probabilities via softmax  
- Uses log-probability trick for gradient computation  
- Computes full episode discounted returns  
- Loss: -log(π(a|s)) * G
- **Characteristic:** Fast convergence but high variance  

---

### 🔹 Actor-Critic Agent
- Actor: policy network  
- Critic: value function (V(s))  
- TD Target: r + γV(s’)
- Advantage: TD target - V(s)
- Combined loss:
- Actor loss + Critic loss  
- **Characteristic:** Lower variance, more stable learning  

---

## 📊 Performance Comparison

| Metric              | REINFORCE | Actor-Critic |
|--------------------|----------|-------------|
| Convergence Speed  | ~58 episodes | ~540 episodes |
| Stability          | Low (high variance) | High (smooth learning) |
| Final Reward       | ~-11.0 | ~-10.29 |
| Success Rate       | ~97.7% | ~54.6% |

### 🔍 Key Insight
Actor-Critic achieves more stable learning by reducing variance through the value function baseline.

---

## 📈 Visualization

The training comparison chart illustrates:

- REINFORCE: rapid convergence with noisy reward fluctuations  
- Actor-Critic: slower but smoother and more stable convergence  

Saved output: comparison_rewards.png
---

## ⚠️ Challenges Encountered

During implementation, several real-world RL challenges were addressed:

- Numerical instability leading to **NaN values**
- Probability collapse from softmax outputs
- Tensor detachment errors in TD target computation
- Mixing scalar and tensor operations causing instability
- Learning rate sensitivity impacting convergence

### ✅ Solutions Applied
- Clamping logits before softmax  
- Normalizing probability distributions  
- Proper use of `.detach()` in TD calculations  
- Converting values consistently to tensors  
- Learning rate tuning  

---

## 📁 Project Structure
CST643_Week2_RL_Compare/
│
├── env.py                  # GridWorld environment
├── reinforce_agent.py      # REINFORCE implementation
├── actor_critic_agent.py  # Actor-Critic implementation
├── train_compare.py        # Training + evaluation + visualization
├── claude_log.md           # Development log (vibe coding process)
├── comparison_rewards.png  # Output chart
└── README.md               # Project documentation
