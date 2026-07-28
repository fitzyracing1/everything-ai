"""
Simple goal-decomposition planner agent.
Presumed need: higher-level reasoning on top of subsumption and multi-agent.
Evidence: multi-agent blackboard + "build everything in ai".
"""

from typing import Any, Dict, List, Optional
from .base import Agent

class PlannerAgent(Agent):
    """
    Decomposes a high-level goal into ordered subgoals using simple keyword heuristics
    + optional blackboard memory. Can be replaced later by a learned planner.
    """
    def __init__(self, name: str = "planner"):
        super().__init__(name)
        self.plan: List[str] = []
        self.current_idx = 0

    def decompose(self, goal: str) -> List[str]:
        g = goal.lower()
        steps = []
        if any(w in g for w in ("safety", "air", "oxygen", "critical")):
            steps.append("check_life_support")
        if any(w in g for w in ("energy", "battery", "charge", "power")):
            steps.append("ensure_power")
        if any(w in g for w in ("mission", "target", "navigate", "go")):
            steps.append("plan_path")
            steps.append("execute_move")
        if any(w in g for w in ("report", "status", "talk", "log")):
            steps.append("generate_report")
        if not steps:
            steps = ["observe", "decide", "act"]
        return steps

    def act(self, observation: Any) -> Any:
        if isinstance(observation, dict):
            goal = observation.get("goal", "")
            blackboard = observation.get("blackboard", {})
        else:
            goal = str(observation)
            blackboard = {}

        if not self.plan or goal != getattr(self, "_last_goal", None):
            self.plan = self.decompose(goal)
            self.current_idx = 0
            self._last_goal = goal

        if self.current_idx >= len(self.plan):
            result = {"status": "plan_complete", "plan": self.plan}
            self.remember({"goal": goal, "result": "complete"})
            return result

        step = self.plan[self.current_idx]
        self.current_idx += 1
        result = {
            "status": "executing",
            "current_step": step,
            "remaining": self.plan[self.current_idx:],
            "write": {f"plan_{self.name}": self.plan},
        }
        self.remember({"goal": goal, "step": step})
        return result

    def status(self) -> Dict:
        return {
            "name": self.name,
            "plan": self.plan,
            "current_idx": self.current_idx,
            "memory_len": len(self.memory),
        }
