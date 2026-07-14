# ChatMD Project Map

A map of the ChatMD project — grouped by what each layer does.

## 🎯 Entry points

| File | Purpose |
|---|---|
| **[src/app.py](../src/app.py)** | FastAPI web server. Serves the chat UI, handles SSE streaming, campaign pause/resume/cancel endpoints. This is what you run with uvicorn. |
| **[index.html](../index.html)** | The single-page frontend — chat interface, campaign banner, trajectory viewer, thermo plots. |
| **[viz_run.py](../viz_run.py)** | Standalone CLI test — render + vision analyze one simulation folder. Bypasses the whole graph. Useful for iterating on viz alone. |

## 🧠 Orchestrator (the ReAct LLM brain)

| File | Purpose |
|---|---|
| **[src/orchestrator/agent.py](../src/orchestrator/agent.py)** | The `create_react_agent` setup. Configures Claude, checkpointer (AsyncSqliteSaver), tool binding. Entry point for `.astream()`. |
| **[src/orchestrator/tools.py](../src/orchestrator/tools.py)** | Standard tools the LLM can call: `simulate`, `analyze_runs`, `visualize_run`, KIM potential lookups, session helpers. This is where `simulate()` orchestrates the simulation subgraph. |
| **[src/orchestrator/campaign_tools.py](../src/orchestrator/campaign_tools.py)** | The 8 campaign-mode tools: `start_campaign`, `get_campaign_state`, `record_trial`, `check_stop`, `measure_metrics`, `generate_report`, `save_report_narrative`, `campaign_visualize_all`. State via `plan.json`/`trials.jsonl`/`control.json`. |
| **[src/orchestrator/system_prompt.py](../src/orchestrator/system_prompt.py)** | The base system prompt for the orchestrator ReAct agent. |
| **[src/orchestrator/skills/campaign.md](../src/orchestrator/skills/campaign.md)** | Campaign-mode skill instructions — teaches the LLM the loop shape, velocity/seed lock, plan-then-confirm pattern. |

## ⚙️ Simulation subgraph (LangGraph pipeline for a single sim)

| File | Purpose |
|---|---|
| **[src/graph/graph.py](../src/graph/graph.py)** | Wires up the LangGraph: generate_lammps → detect_viz → generate_ovito → execute. |
| **[src/graph/nodes.py](../src/graph/nodes.py)** | The node implementations: `generate_lammps_node`, `detect_viz_node`, `generate_ovito_node`, `execute_node`, `review_node`. |
| **[src/graph/routing.py](../src/graph/routing.py)** | Conditional edges — decides which node runs next based on state. |
| **[src/graph/state.py](../src/graph/state.py)** | The `SimulationState` TypedDict — the shared state passed between nodes. |
| **[src/graph/hpc_executor_node.py](../src/graph/hpc_executor_node.py)** | HPC execution path — SSH to cluster, submit PBS job, poll for completion. |

## 🤖 Agents (DSPy structured LLM calls)

| File | Purpose |
|---|---|
| **[src/agents.py](../src/agents.py)** | DSPy signatures + `SimulationHelpers` class. Includes the LAMMPS Generator, Reviewer, OVITO Generator, viz-detection helpers. Called BY the graph nodes. |
| **[src/prompts.py](../src/prompts.py)** | Prompt text and knowledge chunks fed to DSPy agents (LAMMPS conventions, OVITO API notes, review checklist). |

## 📊 Benchmark / campaign metrics

| File | Purpose |
|---|---|
| **[src/benchmark/crack_metrics.py](../src/benchmark/crack_metrics.py)** | The Δa (crack extension) metric. Robust to noise via percentile span + clamp≥0. |
| **[src/benchmark/metrics/__init__.py](../src/benchmark/metrics/__init__.py)** | Metric registry — plugin pattern for adding more metrics later. |
| **[src/benchmark/metrics/delta_a.py](../src/benchmark/metrics/delta_a.py)** | The registered `delta_a` plugin (wraps crack_metrics). |
| **[src/benchmark/reporting.py](../src/benchmark/reporting.py)** | Matplotlib graph rendering for `report.png` (Δa vs iteration + per-knob panels). |

## 🎨 Visualization / rendering

| File | Purpose |
|---|---|
| **[src/ovito_render_server.py](../src/ovito_render_server.py)** | Subprocess OVITO renderer. Reads JSON requests from stdin → outputs base64 PNG. Runs in isolation to avoid semaphore leaks. |
| **[src/ovito_executor.py](../src/ovito_executor.py)** | `DynamicOVITOExecutor` — safely execs the LLM-generated `ovito_script.py` inside the render server. |

## 🔧 Infrastructure / helpers

| File | Purpose |
|---|---|
| **[src/config.py](../src/config.py)** | Config (API keys, LAMMPS binary path, HPC settings, model choices). |
| **[src/config.example.py](../src/config.example.py)** | Template for users to copy → config.py. |
| **[src/memory.py](../src/memory.py)** | Persistent memory store (data/memory.json). Facts the user asked Claude to remember. |
| **[src/run_manager.py](../src/run_manager.py)** | Creates timestamped `runs/<ts>/` folders; manages folder layout. |
| **[src/tools.py](../src/tools.py)** | Low-level shared utilities used by both `orchestrator/tools.py` and the render server. **Note: different from `orchestrator/tools.py`.** |
| **[src/kim_utils.py](../src/kim_utils.py)** | KIM potential lookup and installation helpers. |
| **[kim_potentials_registry.md](../kim_potentials_registry.md)** | Curated list of KIM potentials with links/notes — used as reference by the LAMMPS Generator. |

## 🧪 Tests

| File | Purpose |
|---|---|
| **[tests/](../tests/)** | Pytest suite: agent config, memory, MVPs, visualization, KIM, temperature validation, run folder. |

## 📖 Config / meta

| File | Purpose |
|---|---|
| **[pixi.toml](../pixi.toml)** | Pixi environment (conda + pypi deps). |
| **[README.md](../README.md)** | Project overview. |

---

## The mental model of the flow

```
User types in chat
    ↓
app.py (FastAPI) receives, calls orchestrator/agent.py
    ↓
Orchestrator LLM (ReAct) picks a tool
    ├─ Regular sim → orchestrator/tools.py::simulate()
    │       ↓
    │   graph/graph.py subgraph
    │       ↓ generate_lammps → detect_viz → generate_ovito → execute
    │   Uses agents.py (DSPy) for LLM calls; render_server.py for images
    │
    └─ Campaign → orchestrator/campaign_tools.py::start_campaign() then loop
            ↓
        State: runs/<campaign>/plan.json + trials.jsonl + control.json
        Metrics: benchmark/crack_metrics.py
        Report: benchmark/reporting.py + LLM narrative
```

## Two "tools.py" files (a common gotcha)

- **`src/tools.py`** — low-level shared helpers (render server startup, frame picking)
- **`src/orchestrator/tools.py`** — the LangChain `@tool`-decorated tools the LLM calls

If an import fails with "module tools not found," it's usually confusion between these two.

## In one sentence

**app.py is the door, orchestrator/ is the brain, graph/ is the per-simulation pipeline, agents.py + prompts.py is what the LLM says, benchmark/ is metrics + reports, ovito_*.py is rendering, and the rest is infrastructure.**
