"""
Subsumption Architecture Agent.
Implements the classic Brooks-style layered control:
  Layer 0 (highest priority): Provide air / life support / safety
  Layer 1: Eat / energy / self-maintenance
  Layer 2: Win / mission goals
  Layer 3 (lowest): Talk / report / human interface

Higher layers inhibit lower ones when active.
Evidence: subsumption-architecture skill + "build everything in ai".
"""

from typing import Any, Callable, Dict, List, Optional
from .base import Agent

class Layer:
    def __init__(self, name: str, priority: int, condition: Callable, action: Callable):
        self.name = name
        self.priority = priority  # lower number = higher priority
        self.condition = condition
        self.action = action
        self.active = False

    def evaluate(self, obs: Any) -> bool:
        self.active = self.condition(obs)
        return self.active

class SubsumptionAgent(Agent):
    def __init__(self, name: str = "subsumption"):
        super().__init__(name)
        self.layers: List[Layer] = []

    def add_layer(self, name: str, priority: int, condition: Callable, action: Callable):
        self.layers.append(Layer(name, priority, condition, action))
        self.layers.sort(key=lambda l: l.priority)

    def act(self, observation: Any) -> Any:
        # Evaluate all layers
        for layer in self.layers:
            layer.evaluate(observation)

        # Highest priority active layer wins (inhibits lower)
        for layer in self.layers:
            if layer.active:
                result = layer.action(observation)
                self.remember({"layer": layer.name, "obs": str(observation)[:80], "action": str(result)[:80]})
                return result

        # Default idle
        return {"status": "idle", "msg": "No layer active"}

    def status(self) -> Dict:
        return {
            "name": self.name,
            "layers": [{"name": l.name, "priority": l.priority, "active": l.active} for l in self.layers]
        }
