"""
Lightweight tool registry inspired by Starfit header-grabbing philosophy.
Tools declare their needs; registry can introspect.
"""

from typing import Callable, Dict, List, Any, Optional
import inspect
import functools

_REGISTRY: Dict[str, dict] = {}

def tool(name: Optional[str] = None, triggers: Optional[List[str]] = None, description: str = ""):
    """Decorator to register a tool."""
    def decorator(fn: Callable):
        tname = name or fn.__name__
        _REGISTRY[tname] = {
            "fn": fn,
            "triggers": triggers or [tname],
            "description": description or (fn.__doc__ or "").strip(),
            "signature": str(inspect.signature(fn)),
        }
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator

class ToolRegistry:
    def __init__(self):
        self.tools = dict(_REGISTRY)

    def list(self) -> List[str]:
        return list(self.tools.keys())

    def get(self, name: str):
        return self.tools.get(name)

    def match(self, goal: str) -> List[str]:
        goal_l = goal.lower()
        matched = []
        for name, info in self.tools.items():
            for t in info["triggers"]:
                if t.lower() in goal_l:
                    matched.append(name)
                    break
        return matched

    def run(self, name: str, *args, **kwargs) -> Any:
        info = self.get(name)
        if not info:
            raise KeyError(f"Tool {name} not found")
        return info["fn"](*args, **kwargs)

# Example built-in tools
@tool(triggers=["echo", "say"], description="Echo a message")
def echo(msg: str) -> str:
    return f"ECHO: {msg}"

@tool(triggers=["math", "calculate", "add"], description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b

@tool(triggers=["system", "info"], description="Return basic system info")
def sysinfo() -> dict:
    import platform
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
