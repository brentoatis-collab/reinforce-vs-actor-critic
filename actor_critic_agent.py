import torch
from torch.distributions import Categorical
from env import GridWorld

import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_size=2, hidden_size=64, action_size=4):
        super(ActorCritic, self).__init__()
        
        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        shared_output = self.shared(state)
        action_logits = self.actor(shared_output)
        value = self.critic(shared_output)
        return action_logits, value
    
    def get_action_and_value(self, state):
        action_logits, value = self.forward(state)
        # stabilize logits before sotmax
        action_logits = torch.clamp(action_logits, -10, 10)
        action_probs = torch.softmax(action_logits, dim=-1)
        # 🔥 Stabilize probabilities (prevents collapse)
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum()
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()


def train_actor_critic(episodes=1000, gamma=0.99, lr=1e-4):
    env = GridWorld()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reward_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(100):  # Limit steps per episode
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Get action and value
            action, log_prob, value = model.get_action_and_value(state_tensor)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Get next value
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                _, _, next_value = model.get_action_and_value(next_state_tensor)
            else:
                next_value = 0
            
            # Compute TD target and advantage
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32)
            else:
                td_target = reward + gamma * next_value.detach()
            advantage = td_target - value
            
            
            # Compute losses
            actor_loss = -log_prob * advantage.detach()
            critic_loss = nn.MSELoss()(value, td_target)
            
            # Combine and update
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            state = next_state
            if done:
                break
        reward_history.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(reward_history[-100:]) / 100
            print(f"Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}")
    
    return reward_history


if __name__ == "__main__":
    reward_history = train_actor_critic()
    print(f"Training complete. Final average reward: {sum(reward_history[-100:]) / 100:.2f}")