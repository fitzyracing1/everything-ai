"""
Simple in-memory vector store using torch cosine similarity.
No external DB required — pure sandbox.
"""

from typing import List, Tuple, Optional
import torch
from core.tensor_ops import cosine_similarity, DEVICE

class SimpleVectorStore:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vectors: List[torch.Tensor] = []
        self.metadatas: List[dict] = []
        self.texts: List[str] = []

    def add(self, text: str, vector: Optional[torch.Tensor] = None, metadata: Optional[dict] = None):
        if vector is None:
            # Random embedding for demo (real systems would use encoder)
            vector = torch.randn(self.dim, device=DEVICE)
        else:
            vector = vector.to(DEVICE).flatten()[:self.dim]
            if vector.numel() < self.dim:
                vector = torch.nn.functional.pad(vector, (0, self.dim - vector.numel()))
        self.vectors.append(vector)
        self.texts.append(text)
        self.metadatas.append(metadata or {})

    def search(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        if not self.vectors:
            return []
        q = query.to(DEVICE).flatten()[:self.dim]
        if q.numel() < self.dim:
            q = torch.nn.functional.pad(q, (0, self.dim - q.numel()))
        sims = []
        for i, v in enumerate(self.vectors):
            s = cosine_similarity(q.unsqueeze(0), v.unsqueeze(0)).item()
            sims.append((i, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, s in sims[:top_k]:
            results.append((self.texts[i], s, self.metadatas[i]))
        return results

    def __len__(self):
        return len(self.vectors)
