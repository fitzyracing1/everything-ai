"""
Basic neural network layers built from first principles + torch.
"""

import torch
import torch.nn as nn
from typing import Optional
from .tensor_ops import DEVICE, layer_norm

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=DEVICE) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, device=DEVICE)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

class MultiHeadAttention(nn.Module):
    """Minimal multi-head self-attention for transformer-style models."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model)
        self.out = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, T, d_head)
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            Linear(d_model, mult * d_model),
            nn.GELU(),
            Linear(mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
