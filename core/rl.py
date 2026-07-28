"""
Simple reinforcement learning utilities for Everything AI.
Presumed need: agents that can learn from rewards inside the sandbox.
Evidence: PolicyHead exists + open-ended "build everything in ai" + pure torch.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import PolicyHead
from .tensor_ops import DEVICE

class REINFORCE:
    """Vanilla REINFORCE policy gradient on a PolicyHead."""
    def __init__(self, policy: PolicyHead, lr: float = 1e-3, gamma: float = 0.99):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.trajectory: List[Tuple[torch.Tensor, int, float]] = []  # state, action, reward

    def select_action(self, state: torch.Tensor) -> int:
        logits, _ = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def store(self, state: torch.Tensor, action: int, reward: float):
        self.trajectory.append((state.detach(), action, reward))

    def update(self) -> float:
        if not self.trajectory:
            return 0.0
        # Compute discounted returns
        returns = []
        G = 0.0
        for _, _, r in reversed(self.trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for (state, action, _), G in zip(self.trajectory, returns):
            logits, value = self.policy(state)
            log_prob = F.log_softmax(logits, dim=-1)[action]
            # policy gradient + value baseline
            loss = loss - log_prob * G + 0.5 * (value.squeeze() - G).pow(2)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.trajectory.clear()
        return loss.item()

class SimpleEnv:
    """Minimal bandit / navigation env for testing RL."""
    def __init__(self, n_actions: int = 4, state_dim: int = 8):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.state = torch.randn(state_dim, device=DEVICE)
        self.step_count = 0

    def reset(self):
        self.state = torch.randn(self.state_dim, device=DEVICE)
        self.step_count = 0
        return self.state

    def step(self, action: int):
        self.step_count += 1
        # Reward if action matches a hidden preferred action derived from state
        preferred = int(self.state[0].item() * 1000) % self.n_actions
        reward = 1.0 if action == preferred else -0.1
        self.state = torch.randn(self.state_dim, device=DEVICE)
        done = self.step_count >= 20
        return self.state, reward, done, {"preferred": preferred}
