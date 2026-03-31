# Week 2 Assignment – Claude Code Conversation Log

## Project
Policy-Gradient and Actor-Critic Agents for GridWorld Navigation

## Objective
Implement and compare two reinforcement learning agents in a custom GridWorld environment:
1. A Policy-Gradient agent using REINFORCE
2. An Actor-Critic agent using separate policy and value outputs

## Files Used
- env.py
- reinforce_agent.py
- actor_critic_agent.py
- train_compare.py

---

## Step 1: GridWorld Environment Setup

### Goal
Create a reusable GridWorld environment that can be shared by both the REINFORCE and Actor-Critic agents.

### Prompt Summary
Requested a reusable 5x5 GridWorld environment in Python with:
- Start state at (0, 0)
- Goal state at (4, 4)
- Obstacles
- Four actions: up, down, left, right
- Rewards for step cost, obstacle collision, and goal completion

### Implementation Notes
The environment was implemented in `env.py` and included:
- `reset()`
- `step(action)`
- `is_terminal()`

A test block was added under:
```python
if __name__ == "__main__":

## Step 2: REINFORCE Agent Implementation

### Goal
Implement a Policy-Gradient agent using the REINFORCE algorithm to learn optimal navigation in the GridWorld environment.

### Prompt Summary
Requested complete Python code for `reinforce_agent.py` with:
- A PyTorch-based PolicyNetwork
- Input as (row, col) state coordinates
- Output as action probabilities over 4 actions using softmax
- Action sampling using `torch.distributions.Categorical`
- Log-probability tracking for each step
- Discounted return computation using gamma = 0.99
- Return normalization to reduce variance
- Training loop for 1000 episodes
- Progress printing and reward tracking

### Implementation Notes
The REINFORCE agent was implemented with:
- A simple feedforward neural network (2 → 64 → 4)
- Softmax output layer to produce action probabilities
- Sampling from a categorical distribution for stochastic policy
- Storage of log probabilities and rewards for each episode
- Discounted return calculation using backward accumulation
- Normalization of returns before computing policy loss
- Policy update using the log-probability trick

### Issues Encountered
- Initially pasted prompt instructions directly into the Python file instead of generating executable code
- Confusion between Claude-generated code vs manual input
- High variance during training with occasional extreme reward drops (instability)
- Sensitivity to learning rate and reward scaling

### Resolution
- Regenerated correct Python code using Claude
- Verified structure of policy network and training loop
- Ensured correct implementation of discounted returns and normalization
- Ran training using:
```bash
python3 reinforce_agent.py

## Step 3: Actor-Critic Agent Implementation

### Goal
Implement an Actor-Critic reinforcement learning agent that improves learning stability by combining a policy (actor) with a value function (critic).

### Prompt Summary
Requested complete Python code for `actor_critic_agent.py` with:
- A shared neural network backbone
- An actor head producing action probabilities
- A critic head producing a scalar state value
- Action sampling using `torch.distributions.Categorical`
- Temporal Difference (TD) target computation
- Advantage calculation
- Actor and critic loss functions
- Training loop for 1000 episodes
- Progress logging and reward tracking

### Implementation Notes
The Actor-Critic agent was implemented using PyTorch and included:
- A shared hidden layer for feature representation
- A policy output (actor) using softmax over action logits
- A value output (critic) estimating the state value
- Action sampling from a categorical distribution
- TD-based learning for updating both actor and critic
- Combined loss function:
  - Actor loss using policy gradient with advantage
  - Critic loss using mean squared error (MSE)

---

### ⚠️ Major Issues Encountered

#### 1. Tensor vs Integer Conflicts
- Encountered error:
  `AttributeError: 'int' object has no attribute 'detach'`
- Cause: Mixing Python integers (`reward`, `next_value = 0`) with PyTorch tensors

#### 2. Incorrect TD Target Logic
- Initially applied `.detach()` to values that were not tensors
- Terminal states were not handled correctly

#### 3. Invalid Probability Distribution (NaNs)
- Encountered:
  `ValueError: tensor([nan, nan, nan, nan])`
- Cause: Unstable policy outputs due to exploding gradients or poor updates

#### 4. Loss Function Instability
- Improper separation between actor and critic gradients
- Incorrect use of `.detach()` caused unintended gradient flow

#### 5. Training Instability
- Episodes sometimes stalled or failed to improve
- Learning was highly sensitive to hyperparameters

---

### Resolution

The following corrections were applied:

- Proper TD target handling:
```python
if done:
    td_target = torch.tensor(reward, dtype=torch.float32)
else:
    td_target = reward + gamma * next_value.detach()

## Step 4: Model Comparison and Final Analysis

### Goal
Compare the performance, stability, and implementation complexity of the REINFORCE and Actor-Critic agents in the GridWorld environment.

---

### Performance Comparison

| Metric | REINFORCE | Actor-Critic |
|-------|----------|-------------|
| Final Average Reward | ~ **-11** | ~ **-12.47** |
| Learning Speed | Faster | Slower |
| Stability | Low (high variance) | Higher (after fixes) |
| Implementation Complexity | Lower | Higher |
| Sensitivity to Hyperparameters | Moderate | High |

---

### Behavioral Observations

#### REINFORCE
- Learned faster in early episodes
- Achieved slightly better final reward performance
- Exhibited high variance in training
- Experienced a significant instability event (large negative reward spike)

#### Actor-Critic
- Learned more gradually
- Demonstrated smoother and more consistent improvement
- Avoided extreme reward collapses after stabilization
- Required significant debugging and refinement to function correctly

---

### Key Tradeoffs

- **REINFORCE**
  - Simpler and easier to implement
  - More direct policy learning
  - Higher variance → less stable

- **Actor-Critic**
  - More complex architecture
  - Uses value estimation to reduce variance
  - More stable learning once properly implemented
  - Requires careful handling of tensors, TD targets, and gradients

---

### Lessons Learned

- Reinforcement learning implementations are highly sensitive to:
  - Learning rate
  - Numerical stability
  - Correct tensor handling
  - Proper treatment of terminal states

- Actor-Critic methods can improve stability, but only when:
  - TD targets are constructed correctly
  - Gradients are controlled properly
  - Tensor operations are consistent

- Simpler algorithms (REINFORCE) may outperform more advanced methods when:
  - The environment is small
  - The implementation is clean
  - Stability issues are not dominant

---

### Final Conclusion

Both reinforcement learning agents were successfully implemented and trained within the GridWorld environment.

The REINFORCE agent achieved slightly better final reward performance, but at the cost of higher variance and instability.

The Actor-Critic agent, while more complex and requiring extensive debugging, demonstrated more stable and controlled learning behavior once properly implemented.

This comparison highlights a key principle in reinforcement learning:
> More advanced methods do not guarantee better performance without careful implementation and tuning.

---