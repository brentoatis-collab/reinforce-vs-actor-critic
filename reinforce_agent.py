import torch
from torch.distributions import Categorical
import numpy as np
from env import GridWorld

import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size=2, hidden_size=64, action_size=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

def train_reinforce(num_episodes=1000, gamma=0.99, learning_rate=0.01):
    env = GridWorld()
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    reward_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        done = False
        while not done:
            state_tensor = torch.tensor([state[0], state[1]], dtype=torch.float32)
            action_probs = policy(state_tensor)
            
            m = Categorical(action_probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            
            state, reward, done = env.step(action.item())
            rewards.append(reward)
        
        reward_history.append(sum(rewards))
        
        returns = []
        cumulative_return = 0
        for r in reversed(rewards):
            cumulative_return = r + gamma * cumulative_return
            returns.insert(0, cumulative_return)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = 0
        for log_prob, return_val in zip(log_probs, returns):
            policy_loss += -log_prob * return_val
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}")
    
    return reward_history

if __name__ == "__main__":
    reward_history = train_reinforce()
    print(f"Training complete. Final average reward: {np.mean(reward_history[-100:]):.2f}")