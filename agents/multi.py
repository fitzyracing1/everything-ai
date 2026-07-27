"""
Simple multi-agent system with shared blackboard.
"""

from typing import Any, Dict, List
from .base import Agent

class Blackboard:
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: List[Dict] = []

    def write(self, key: str, value: Any, author: str = "unknown"):
        self.data[key] = value
        self.history.append({"key": key, "value": str(value)[:100], "author": author})

    def read(self, key: str, default=None):
        return self.data.get(key, default)

class MultiAgentSystem:
    def __init__(self, name: str = "mas"):
        self.name = name
        self.agents: Dict[str, Agent] = {}
        self.blackboard = Blackboard()

    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def step(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for name, agent in self.agents.items():
            obs = observations.get(name, observations.get("global", {}))
            # Inject blackboard state into obs if dict
            if isinstance(obs, dict):
                obs = {**obs, "blackboard": dict(self.blackboard.data)}
            action = agent.act(obs)
            results[name] = action
            # Agents can write to blackboard via special action format
            if isinstance(action, dict) and "write" in action:
                for k, v in action["write"].items():
                    self.blackboard.write(k, v, author=name)
        return results

    def report(self) -> str:
        lines = [f"MultiAgentSystem '{self.name}' with {len(self.agents)} agents"]
        for a in self.agents.values():
            lines.append("  " + a.report())
        lines.append(f"Blackboard keys: {list(self.blackboard.data.keys())}")
        return "\n".join(lines)
