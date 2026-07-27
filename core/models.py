"""
Higher-level models: simple transformer, MLP, agent policy head.
"""

import torch
import torch.nn as nn
from .layers import Linear, TransformerBlock
from .tensor_ops import DEVICE

class SimpleTransformer(nn.Module):
    """A tiny transformer for sequence modeling / language / planning."""
    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, max_seq: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)
        self.max_seq = max_seq
        self.to(DEVICE)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_seq
        pos = torch.arange(T, device=DEVICE)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

class MLP(nn.Module):
    """Generic multi-layer perceptron."""
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, act=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PolicyHead(nn.Module):
    """Simple actor-critic style head for agent decisions."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = MLP(state_dim, [hidden], hidden)
        self.actor = Linear(hidden, action_dim)
        self.critic = Linear(hidden, 1)
        self.to(DEVICE)

    def forward(self, state: torch.Tensor):
        h = self.shared(state)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value
