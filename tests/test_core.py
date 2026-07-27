#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/workdir/artifacts/everything-ai")

import torch
from core.models import SimpleTransformer, MLP, PolicyHead
from core.layers import Linear, MultiHeadAttention
from agents.subsumption import SubsumptionAgent
from memory.vector_store import SimpleVectorStore
from tools.registry import ToolRegistry

def test_transformer():
    m = SimpleTransformer(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq=32)
    x = torch.randint(0, 50, (2, 10))
    out = m(x)
    assert out.shape == (2, 10, 50)
    print("✓ transformer")

def test_subsumption():
    a = SubsumptionAgent()
    a.add_layer("test", 0, lambda o: True, lambda o: "ok")
    assert a.act({}) == "ok"
    print("✓ subsumption")

def test_memory():
    s = SimpleVectorStore(8)
    s.add("hello")
    assert len(s) == 1
    print("✓ memory")

def test_tools():
    r = ToolRegistry()
    assert "echo" in r.list()
    assert r.run("echo", "hi") == "ECHO: hi"
    print("✓ tools")

if __name__ == "__main__":
    test_transformer()
    test_subsumption()
    test_memory()
    test_tools()
    print("All tests passed.")
