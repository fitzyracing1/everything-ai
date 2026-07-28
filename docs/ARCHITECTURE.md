# Everything AI Architecture

## Layers of Capability

1. **Tensor Core** (`core/tensor_ops.py`, `layers.py`, `models.py`)
   - Device-agnostic torch primitives
   - Custom Linear, MultiHeadAttention, TransformerBlock
   - SimpleTransformer, MLP, PolicyHead

2. **Learning** (`core/rl.py`)
   - REINFORCE policy gradient with value baseline
   - SimpleEnv for rapid experimentation

3. **Memory** (`memory/vector_store.py`)
   - In-memory cosine similarity store
   - Ready for encoder upgrade (currently random embeddings for demo)

4. **Tools** (`tools/registry.py`)
   - `@tool` decorator with trigger keywords
   - Starfit-inspired introspection (signature + description)
   - Safe sandbox tools only (no arbitrary shell)

5. **Agents**
   - `Agent` base with experience memory
   - `SubsumptionAgent`: priority layers (air > energy > mission > talk)
   - `PlannerAgent`: heuristic goal decomposition + blackboard writes
   - `MultiAgentSystem` + `Blackboard` for coordination

## Safety Model

Subsumption is the primary safety mechanism. Higher-priority layers always inhibit lower ones. This mirrors the dedicated `subsumption-architecture` skill and is intended for life-critical or resource-constrained agents (space, habitat, robot, phone-first systems).

## Extension Points

- New tools: decorate with `@tool` and they appear in every ToolRegistry instance.
- New agent behaviors: subclass `Agent` or add layers to `SubsumptionAgent`.
- New models: inherit from `nn.Module` and place under `core/`.
- Training loops: use `REINFORCE` or implement PPO / DQN on top of the same PolicyHead.

## Offline-first Guarantee

All code runs with zero external network calls. Torch is local, vector store is pure Python+torch, agents are pure Python.
