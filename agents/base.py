"""
Base agent and environment interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class Environment(ABC):
    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple:
        """Return (observation, reward, done, info)"""
        pass

class Agent(ABC):
    def __init__(self, name: str = "agent"):
        self.name = name
        self.memory: list = []

    @abstractmethod
    def act(self, observation: Any) -> Any:
        pass

    def remember(self, experience: Dict):
        self.memory.append(experience)

    def report(self) -> str:
        return f"Agent {self.name}: {len(self.memory)} experiences"
