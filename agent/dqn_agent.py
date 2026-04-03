import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from agent.replay_buffer import ReplayBuffer


# ===== NEURAL NETWORK =====
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ===== AGENT =====
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Model
        self.model = DQN(state_size, action_size)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Loss
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    # ===== ACTION SELECTION =====
    def select_action(self, state):
        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    # ===== TRAINING =====
    def train(self, batch_size=32):
        if self.memory.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Current Q values
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        next_q_values = self.model(next_states)
        next_q_value = torch.max(next_q_values, dim=1)[0]

        # Target
        target = rewards + self.gamma * next_q_value * (1 - dones)

        # Loss
        loss = self.criterion(q_value, target.detach())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ===== SAVE MODEL =====
    def save_model(self, path="models/dqn_model.pth"):
        torch.save(self.model.state_dict(), path)

    # ===== LOAD MODEL =====
    def load_model(self, path="models/dqn_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()