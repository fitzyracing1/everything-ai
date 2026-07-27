#!/usr/bin/env python3
"""
Full stack demo: core model + memory + tools + multi-agent.
"""

import sys
sys.path.insert(0, "/home/workdir/artifacts/everything-ai")

import torch
from core.models import SimpleTransformer, PolicyHead
from memory.vector_store import SimpleVectorStore
from tools.registry import ToolRegistry, echo, add, sysinfo
from agents.multi import MultiAgentSystem
from agents.base import Agent

class ToolUsingAgent(Agent):
    def __init__(self, name: str, registry: ToolRegistry):
        super().__init__(name)
        self.registry = registry

    def act(self, observation):
        goal = observation.get("goal", "") if isinstance(observation, dict) else str(observation)
        matches = self.registry.match(goal)
        if matches:
            # naive: run first match
            tool_name = matches[0]
            try:
                if tool_name == "echo":
                    return {"result": self.registry.run("echo", goal)}
                elif tool_name == "add":
                    return {"result": self.registry.run("add", 2, 3)}
                elif tool_name == "sysinfo":
                    return {"result": self.registry.run("sysinfo")}
            except Exception as e:
                return {"error": str(e)}
        return {"status": "no matching tool", "goal": goal}

def main():
    print("=== Everything AI Full Demo ===\n")

    # 1. Core model
    print("1. Instantiating SimpleTransformer...")
    model = SimpleTransformer(vocab_size=100, d_model=32, n_layers=1, n_heads=2)
    x = torch.randint(0, 100, (1, 8))
    logits = model(x)
    print(f"   Input shape {x.shape} -> logits {logits.shape}")

    # 2. Memory
    print("\n2. Vector store...")
    store = SimpleVectorStore(dim=32)
    store.add("safety protocol alpha", metadata={"type": "safety"})
    store.add("mission waypoint beta", metadata={"type": "mission"})
    store.add("energy recharge station", metadata={"type": "energy"})
    q = torch.randn(32)
    results = store.search(q, top_k=2)
    print(f"   Stored {len(store)} items. Top retrieval: {results[0][0] if results else 'none'}")

    # 3. Tools
    print("\n3. Tool registry...")
    reg = ToolRegistry()
    print(f"   Registered tools: {reg.list()}")
    print(f"   sysinfo -> {reg.run('sysinfo')}")

    # 4. Multi-agent
    print("\n4. Multi-agent system...")
    mas = MultiAgentSystem("sandbox-crew")
    mas.add_agent(ToolUsingAgent("tool-bot", reg))
    mas.add_agent(ToolUsingAgent("echo-bot", reg))

    obs = {
        "tool-bot": {"goal": "get system info please"},
        "echo-bot": {"goal": "say hello from the sandbox"},
    }
    results = mas.step(obs)
    print(mas.report())
    print("Step results:", results)

    print("\n=== All components operational ===")

if __name__ == "__main__":
    main()
