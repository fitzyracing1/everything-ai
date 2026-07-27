"""
Core tensor operations and utilities.
Presumed need: foundational math for all higher AI components.
Evidence: empty sandbox + torch available + open-ended "build everything in ai".
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x: torch.Tensor) -> torch.Tensor:
    return x.to(DEVICE)

def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax."""
    return F.softmax(x - x.max(dim=dim, keepdim=True).values, dim=dim)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)

def layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

print(f"[everything-ai] tensor_ops loaded on {DEVICE}")
