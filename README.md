# Everything AI

**A complete, modular, self-contained AI framework built entirely inside the Grok sandbox.**

No external internet required for core operation. Uses local PyTorch, pure Python agents, in-memory vector store, and a Starfit-inspired tool registry.

## Architecture

```
everything-ai/
├── core/          # Tensor ops, layers, models, REINFORCE + SimpleEnv
├── agents/        # Base, Subsumption, MultiAgent + Blackboard, Planner
├── memory/        # SimpleVectorStore (cosine)
├── tools/         # ToolRegistry + @tool (echo, math, sysinfo, multiply, list_artifacts, sha)
├── examples/      # demos for subsumption, full stack, RL, planner
├── docs/          # ARCHITECTURE.md
└── tests/         # unit tests for all major components
```

## Quick Start

```bash
cd /home/workdir/artifacts/everything-ai
PYTHONPATH=. python3 tests/test_core.py
PYTHONPATH=. python3 examples/demo_subsumption.py
PYTHONPATH=. python3 examples/demo_full.py
PYTHONPATH=. python3 examples/demo_rl.py
PYTHONPATH=. python3 examples/demo_planner.py
```

## Design Goals

- **Everything local**: No cloud LLM calls required for the framework itself.
- **Composable**: Agents, tools, memory and models can be mixed freely.
- **Safety-first layers**: Subsumption enforces priority (air > energy > mission > talk).
- **Inspectable**: Every component is pure Python + torch, readable and editable.
- **Extensible**: Add new tools with `@tool`, new layers with `add_layer`, new models by subclassing.
- **Learning ready**: REINFORCE + PolicyHead for rapid policy experiments.

## Built for

Open-ended AI research, autonomous agent prototyping, phone-first / offline AI systems, and integration with Grok custom skills (subsumption-architecture, starfit, BMD conceptual agents, persistent-self-coder, etc.).

## Status (2026-07-28)

- Core neural modules + tests: complete
- Subsumption + multi-agent + blackboard: complete
- Planner agent + goal decomposition: added
- REINFORCE training loop + toy env: added
- Expanded tool registry: added
- Architecture documentation: added
- All demos and tests passing in sandbox

---

Generated and extended by autonomous sandbox build process ("Build everything in ai").
