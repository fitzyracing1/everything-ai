#!/usr/bin/env python3
"""
Demo: Subsumption agent responding to different world states.
"""

import sys
sys.path.insert(0, "/home/workdir/artifacts/everything-ai")

from agents.subsumption import SubsumptionAgent

def main():
    agent = SubsumptionAgent("life-support-bot")

    # Layer 0: Provide air / critical safety (highest priority)
    agent.add_layer(
        "provide_air",
        priority=0,
        condition=lambda obs: obs.get("o2", 100) < 18,
        action=lambda obs: {"action": "VENT_O2", "level": 21.0, "reason": "O2 critical"}
    )

    # Layer 1: Eat / energy
    agent.add_layer(
        "acquire_energy",
        priority=1,
        condition=lambda obs: obs.get("battery", 100) < 20,
        action=lambda obs: {"action": "DOCK_CHARGE", "reason": "low battery"}
    )

    # Layer 2: Win / mission
    agent.add_layer(
        "execute_mission",
        priority=2,
        condition=lambda obs: obs.get("mission_active", False),
        action=lambda obs: {"action": "PROCEED_TO_TARGET", "target": obs.get("target", "unknown")}
    )

    # Layer 3: Talk
    agent.add_layer(
        "report_status",
        priority=3,
        condition=lambda obs: True,  # always available but lowest
        action=lambda obs: {"action": "STATUS_REPORT", "msg": f"All nominal. O2={obs.get('o2')} Battery={obs.get('battery')}"}
    )

    scenarios = [
        {"o2": 21, "battery": 80, "mission_active": False},
        {"o2": 15, "battery": 80, "mission_active": True, "target": "airlock"},
        {"o2": 21, "battery": 12, "mission_active": True, "target": "lab"},
        {"o2": 21, "battery": 90, "mission_active": True, "target": "surface"},
    ]

    print("=== Subsumption Agent Demo ===\n")
    for i, obs in enumerate(scenarios):
        print(f"Scenario {i+1}: {obs}")
        action = agent.act(obs)
        print(f"  -> {action}\n")

    print("Agent memory length:", len(agent.memory))
    print(agent.status())

if __name__ == "__main__":
    main()
