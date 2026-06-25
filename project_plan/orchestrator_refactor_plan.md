# Refactor Plan: `create_react_agent` Orchestrator as Main Entry, Campaign as a Skill

> **One source of truth.** Supersedes scattered notes in [project_plan/tool_calling_orchestrator_refactor.md], `graph_comparison.md`, and `/path/to/.claude/plans/hi-can-you-tell-tingly-matsumoto.md`. Read this file end-to-end.

> **Why not deepagents:** Library injects its own system prompt + middleware that overwrites/competes with our domain instructions. Skills load opaquely via `SkillsMiddleware`. We want full control of the prompt and explicit, traceable skill loading. So we build on `langgraph.prebuilt.create_react_agent` instead — same end-shape, simpler, no new dependency.

---

## 1. Context

The current graph (branch `campaign_mode`, HEAD `ae030a5`) routes every turn through `classify_intent → route_after_intent` — a flat supervisor dispatching to one of five flat targets (`chat_agent`, `analyze_results`, `hpc_executor`, `campaign_executor`, or the LAMMPS subflow). Campaign behavior is bolted on with two seams:
- regex-triggered prompt **addendum** injected into `chat_node` ([src/benchmark/campaign_prompts.py:30-62](src/benchmark/campaign_prompts.py#L30-L62))
- flat-graph `campaign_executor` node the classifier must dispatch to

The intent classifier returns one of 9 hardcoded labels. Every new capability touches 4-5 files (label + router branch + node + graph registration + state fields).

**The change.** One entry node — an `orchestrator` built with `langgraph.prebuilt.create_react_agent` — owns the conversation, decides what to do via its own LLM reasoning, and dispatches work via tools:

| Mechanism | Used for | This project |
|-----------|----------|--------------|
| **Tool** | Any work the orchestrator dispatches | `get_kim_models`, `simulate`, `analyze_runs`, `start_campaign`, `hpc_submit`, `load_skill` |
| **Skill (markdown file)** | Domain-specific instructions loaded on demand via `load_skill(name)` tool | `campaign.md` (Phase 1 setup + Phase 3 iteration mode) |
| **Subflow** | Pre-built LangGraph the orchestrator invokes from inside a tool | `simulate_subgraph` (LAMMPS+OVITO), HPC node logic |

**Outcome.** Kills the intent classifier + flat router. Campaign trigger detection moves out of regex into the orchestrator's system prompt ("if the user mentions benchmarks/sweeps/optimization, call `load_skill('campaign')` first").

---

## 2. Cons of current architecture → solutions

| # | Con | Solution |
|---|-----|----------|
| 1 | **Fixed-vocabulary intent classifier** — `classify_intent` ([src/graph/nodes.py:333-375](src/graph/nodes.py#L333-L375)) returns one of 9 hardcoded labels. New phrasings need prompt tuning; new capabilities need new labels. | Delete `classify_intent` + `route_after_intent`. Let the orchestrator-LLM choose tools/skills/subagents by natural-language descriptions. |
| 2 | **Campaign bolted on in two seams** — regex addendum in `chat_node` + flat-graph `campaign_executor` node. | Campaign = one **skill markdown file** loaded via `load_skill('campaign')` tool + one `start_campaign` tool that spawns `CampaignManager`. |
| 3 | **Chat node carries campaign prompt forever** once `is_campaign_context` triggers — 310 lines injected every turn for the rest of the session. | Skill loads on explicit `load_skill` call and lives only in that turn's tool result. Next turn starts clean. |
| 4 | **Adding a mode touches 4-5 files** — label + router branch + node + graph registration + state fields. | Adding a mode = one new tool function (or one new skill directory). No router/state edits. |
| 5 | **Hand-rolled tool-call loop** — 160 lines of streaming + tool dispatch + history trimming in `chat_node` ([src/graph/nodes.py:89-248](src/graph/nodes.py#L89-L248)). | `create_react_agent` provides the loop. Delete `chat_node`. |
| 6 | **`SmartOrchestrator` is a god-module** — classification + extraction + viz detection in one DSPy module ([src/agents.py](src/agents.py)). | Split: extraction stays in `extract_params_node`; viz stays in `detect_viz_node`; classification disappears (orchestrator-LLM replaces it). |
| 7 | **Bag-of-fields state schema** — 40+ fields in `SimulationState` spanning every concern. | Split: `OrchestratorState` (top-level, `MessagesState`-based), `SimulationState` (LAMMPS subgraph only). Campaign fields leave state — they're tool args. |
| 8 | **Flat router is order-sensitive** — `pbs_pending` short-circuits before `intent` is checked. | No router. LLM picks tool from context. HPC pause stays via `interrupt()` inside the `hpc_submit` tool. |
| 9 | **Phase 2 HPC pays the cost again** — Atomsk + Globus-submit will repeat label + branch + node + state. | Atomsk = one tool. Globus-submit stays inside `hpc_submit`. Zero router edits. |
| 10 | **Campaign block parsing is fragile** — `_parse_campaign_parameters_block` regex ([src/graph/nodes.py:249-332](src/graph/nodes.py#L249-L332)) breaks on LLM formatting drift. | Structured tool call: `start_campaign(template, bounds, n_iterations, target)`. Pydantic validates; no regex. |

---

## 3. Target topology

```
START
  │
  ▼
orchestrator   ← langgraph.prebuilt.create_react_agent
  │              system_prompt: our own — fully owned, no library overrides
  │              tools:
  │                • load_skill(name)     — reads src/orchestrator/skills/<name>.md
  │                • get_kim_models(elem)
  │                • simulate(...)        — invokes simulate_subgraph
  │                • analyze_runs(query)
  │                • start_campaign(...)  — spawns CampaignManager daemon
  │                • hpc_submit(...)      — wraps hpc_executor_node logic
  ▼
END
```

The LAMMPS+OVITO iterative pipeline stays as a compiled `CompiledStateGraph` invoked synchronously by the `simulate` tool. HPC keeps its current two-phase `interrupt()` pause inside the `hpc_submit` tool — [src/app.py](src/app.py)'s `is_paused` check still works since `interrupt()` propagates through the tool call up through `create_react_agent`.

### Current vs target (side-by-side)

```
CURRENT (9 nodes, flat router)            TARGET (1 orchestrator, 3 extension types)
═══════════════════════════════════       ════════════════════════════════════════════

        START                                          START
          │                                              │
          ▼                                              ▼
   classify_intent                              ┌──────────────────┐
   ┌──┬──┬──┬──┐                                │   orchestrator   │
   │  │  │  │  │                                │ (langgraph       │
   │  │  │  │  └─[execute+ready]→END            │  create_react_   │
   │  │  │  │                                   │  agent)          │
   │  │  │  └─[hpc_*]→hpc_executor─→END         └────────┬─────────┘
   │  │  │                                               │
   │  │  └─[campaign_start]→campaign_executor─→END       │
   │  │                                                  ▼
   │  └─[analyze]→analyze_results─→END                  END
   │
   ▼                                          tools (called by LLM):
   chat_agent                                   • load_skill ──→ reads skills/*.md
   │                                            • get_kim_models
   ▼                                            • simulate ──┐
   extract_params                               • analyze_runs│ invokes
   │                                            • start_campaign│ subgraph
   ▼ [has_params]                               • hpc_submit  │
   generate_lammps                                            ▼
   │                                            ┌──────────────────────┐
   ▼ [review_passed]                            │  simulate_subgraph   │
   detect_viz                                   │  extract_params      │
   │                                            │       ↓              │
   ▼ [has_custom_viz]                           │  generate_lammps     │
   generate_ovito                               │       ↓              │
   │                                            │  detect_viz          │
   ▼                                            │       ↓              │
   END                                          │  generate_ovito      │
                                                └──────────────────────┘

                                          skills (loaded on explicit tool call):
                                            • campaign.md

                                          hpc_submit tool internally calls
                                          hpc_executor_node logic — keeps
                                          interrupt() pause for PBS edit
```

### What's actually changing

| Layer | Current | Target |
|-------|---------|--------|
| **Entry** | `classify_intent` (DSPy, 9 fixed labels) | `orchestrator` (`create_react_agent`, NL tool descriptions in our own system prompt) |
| **Routing** | `route_after_intent` (if/elif chain) | LLM tool-choice |
| **Chat** | `chat_node` (160 lines, hand-rolled tool loop) | `create_react_agent` built-in loop |
| **Simulate pipeline** | Graph edges from `chat_agent` | `simulate` tool invokes compiled subgraph |
| **HPC** | Flat node, dispatched by router | `hpc_submit` tool wraps existing node logic |
| **Campaign** | Flat node + 310-line prompt addendum + regex trigger | `campaign.md` skill (loaded via `load_skill`) + `start_campaign` tool |
| **Analyze** | Flat node | `analyze_runs` tool |
| **Pause/resume** | `interrupt()` in HPC node, surfaced by `is_paused = bool(post_state.next)` | Same — `interrupt()` inside `hpc_submit` tool propagates through `create_react_agent` |

---

## 4. Files to change

### New
- **`src/orchestrator/agent.py`** — builds the orchestrator: model + our system prompt + tools list. Exports `orchestrator_graph = create_react_agent(model, tools, prompt=SYSTEM_PROMPT, checkpointer=SqliteSaver(...))`.
- **`src/orchestrator/system_prompt.py`** — the full system prompt: agent role, capability summary, available tools, and **explicit instruction to call `load_skill(name)` when a request matches a known skill** (e.g. "if the user mentions benchmarking, sweeping, optimization, or campaigns, call `load_skill('campaign')` BEFORE responding"). Skill names + 1-line descriptions are listed in this prompt so the LLM knows what's available.
- **`src/orchestrator/tools.py`** — all tools:
  - `load_skill(name: str)` — reads `src/orchestrator/skills/{name}.md` and returns its contents as the tool result. The orchestrator's next LLM turn sees the skill instructions in tool-result message.
  - `get_kim_models(element)` — lifted from [src/graph/nodes.py:77](src/graph/nodes.py#L77) (`_handle_kim_tool_call`).
  - `simulate(material, temperature, ensemble, duration_ps, system_size, kim_model, visualization_notes, camera_angle, execution_target="local", auto_approve=False)` — invokes the LAMMPS+OVITO subgraph. The two trailing args control the review/execution flow (see §4a). Phase 1 only wires `execution_target="local"`; `"hpc"` reserved for Phase 2.
  - `analyze_runs(query)` — wraps the tool-calling loop in [src/graph/nodes.py:779](src/graph/nodes.py#L779) (`analyze_results_node`), exposing `list_runs` + `get_run_log` inside.
  - `start_campaign(template, bounds, n_iterations, target, original_user_message)` — spawns `CampaignManager` daemon thread (see §4b). Always local execution in Phase 1.
  - `hpc_submit(lammps_script, ovito_script, run_folder)` — internally calls `hpc_executor_node` logic ([src/graph/hpc_executor_node.py:246](src/graph/hpc_executor_node.py#L246)). PBS-edit `interrupt()` propagates up through the tool call. See §4b.
  - `hpc_status(job_ids: list[str] | None = None)` — wraps existing `_run_hpc_status` helper ([src/graph/hpc_executor_node.py:218-230](src/graph/hpc_executor_node.py#L218-L230)). Defaults to all jobs in state.hpc_jobs if `job_ids` is None.
- **`src/orchestrator/skills/campaign.md`** — campaign skill file (just a markdown file the `load_skill` tool reads):
  - **Header note** so the orchestrator's LLM knows when to call `load_skill('campaign')` — listed in the system prompt as: "campaign — use for benchmark sweeps, parameter optimization across iterations, or when message says 'Propose iteration N / Past attempts:'".
  - Phase 1 setup instructions (currently the addendum in [src/benchmark/campaign_prompts.py:67-377](src/benchmark/campaign_prompts.py#L67-L377)).
  - Phase 3 iteration-mode instructions: "pick knob values within bounds, state hypothesis, call `simulate(..., auto_approve=True)` with full params".
  - Contract: orchestrator must call `start_campaign(...)` after the user confirms during setup.

### Modified
- **`src/graph/graph.py`** — slim down: build only the LAMMPS+OVITO subgraph starting at `extract_params`. Export as `simulate_subgraph`.
- **`src/graph/routing.py`** — delete `route_after_intent`. Keep `route_after_extract`, `route_after_lammps`, `route_after_detect_viz`.
- **`src/graph/state.py`** — keep `SimulationState` for the LAMMPS subgraph. Orchestrator uses a `MessagesState` subclass carrying top-level fields needed for SSE / HPC pause (`session_id`, `execution_target`, `hpc_jobs`, `hpc_run_folder_pending`, `workflow_stage`). Campaign fields leave state.
- **`src/app.py`** — swap `from graph.graph import simulation_graph` for `from orchestrator.agent import orchestrator_graph`. `is_paused = bool(post_state.next)` check unchanged.
- **`src/benchmark/campaign_prompts.py`** — delete `is_campaign_context()` and the addendum injection. Keep `format_history_message` (still used by `CampaignManager._run_one_iteration`).

### 4a. The `simulate` tool's `execution_target` + `auto_approve` flags

These two flags govern whether the tool emits a review event and how the script gets executed. Designed now so HPC campaigns (Phase 2) drop in without re-plumbing.

| `execution_target` | `auto_approve` | Phase | What happens |
|---|---|---|---|
| `"local"` | `False` | **Phase 1** | Generate script → emit `scripts_ready_for_review` → wait for user Approve → run local |
| `"local"` | `True` | **Phase 1** | Generate script → **skip** review event → return script (CampaignManager runs it) |
| `"hpc"` | `False` | **Phase 1** | Generate script → emit `scripts_ready_for_review` → on Approve, call HPC subagent → PBS preview → user edits → submit |
| `"hpc"` | `True` | **Phase 2** | Generate script → skip review → call HPC subagent with `auto_pbs=True` → submit with default PBS template, no edit pause |

In Phase 1 the only `auto_approve=True` caller is the campaign skill's iteration prompt (always `execution_target="local"`). Phase 2 enables HPC campaigns by flipping that to `"hpc"` and giving CampaignManager a stored `pbs_template` to use.

### 4b. Inside the `hpc_submit` tool

Plain tool that wraps the existing `hpc_executor_node` logic. **No behavior change** — same PBS preview SSE, same two-call pattern (graph ends after Phase 1; browser POSTs again with edited PBS to trigger Phase 2), same Globus MCP submit, same job tracking. Only the entry point changes (from a graph node dispatched by `route_after_intent` to a tool dispatched by the orchestrator-LLM).

**There is no `interrupt()` in the current code.** The pause/resume works at the HTTP layer: after Phase 1 the graph returns, browser shows the PBS preview, user edits, then browser POSTs a new request with `workflow_stage="pbs_pending"` + `pbs_script_edited`. The next orchestrator turn calls `hpc_submit` again, which detects `pbs_pending` and runs Phase 2.

**Signature:**
```python
@tool
def hpc_submit(lammps_script: str, ovito_script: str, run_folder: str) -> str:
    """Submit a LAMMPS job to HPC via Globus.

    Generates a PBS script and shows it to the user for review/edit
    (via SSE pbs_preview event), pauses for user approval, then submits
    the job via Globus on Approve. Returns the job ID once submitted.
    """
```

**Body — wraps the existing node:**
```python
def hpc_submit(lammps_script, ovito_script, run_folder, *, session_id):
    from graph.hpc_executor_node import _run_hpc_executor_phases
    return _run_hpc_executor_phases(
        lammps_script=lammps_script,
        ovito_script=ovito_script,
        run_folder=run_folder,
        session_id=session_id,
    )
```

The existing [`hpc_executor_node`](src/graph/hpc_executor_node.py#L246) body is refactored into a callable `_run_hpc_executor_phases(...)` that the tool invokes. Same code, new entry point.

**What's preserved (verbatim):**
- Phase 1: build PBS script, save scripts to `run_folder`, emit `pbs_preview` SSE event, call `interrupt()` to pause for user edit
- Phase 2: read user's edited PBS, call `MultiServerMCPClient` + `create_react_agent` with Globus MCP tools to submit the job, track job IDs in state
- The `interrupt()` pause propagates: tool → `create_react_agent` → `src/app.py`'s `is_paused = bool(post_state.next)` check works unchanged
- Resume via `Command(resume="__resume__")` against the orchestrator graph still triggers Phase 2

**Status queries** ("what's the status of my HPC job?") become a separate tool — `hpc_status(job_ids: list[str] | None)` — wrapping the existing `_run_hpc_status` helper. Simpler than overloading `hpc_submit`.

**State fields used:** `hpc_jobs`, `hpc_run_folder_pending`, `workflow_stage` — these stay on the orchestrator's `MessagesState` subclass so the tool can read/write them across the pause.

---

### 4c. Inside the `start_campaign` tool

Thin **launcher** — does not run iterations itself. The iteration loop lives in `CampaignManager` (a daemon thread). The tool's only job: set up SSE plumbing, instantiate the manager, kick off the thread, return.

**Signature:**
```python
@tool
def start_campaign(
    template: dict,                  # locked sim spec: material, system_size, kim_model,
                                     #   phases, dump_settings, OVITO colors, camera
    bounds: dict,                    # {strain_rate: [1e-3, 1e-2], temperature: [200,600], ...}
    n_iterations: int,
    target: str,                     # e.g. "minimize delta_a"
    original_user_message: str,      # raw user request for iteration 1's prompt
) -> str:
    """Launch a benchmark campaign as a background thread.

    Use after the user confirms the CAMPAIGN PARAMETERS block during Phase 1
    setup. Returns immediately; the campaign runs asynchronously.
    """
```

**Body — five steps:**
```python
def start_campaign(template, bounds, n_iterations, target, original_user_message, *, session_id):
    import threading, queue
    from datetime import datetime
    from benchmark.campaign import CampaignManager
    from graph.nodes import SESSION_QUEUES, _q   # existing globals
    from config import LAMMPS_CORES

    # 1. Create campaign_id (also serves as LangGraph thread_id for iterations)
    campaign_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 2. Build the _DualQueue tee so iteration events reach BOTH:
    #    - campaign_q (primary, for CampaignManager's own log/summary)
    #    - chat_q (the user's browser SSE stream, so they see live progress)
    chat_q = SESSION_QUEUES[session_id]
    campaign_q = queue.Queue()
    dual_q = _DualQueue(campaign_q, chat_q)

    # 3. Register dual_q under the campaign_id so iteration nodes find it
    SESSION_QUEUES[campaign_id] = dual_q

    # 4. Instantiate manager + spawn daemon thread
    manager = CampaignManager(
        campaign_id=campaign_id,
        n_iterations=n_iterations,
        original_user_message=original_user_message,
        sse_q=dual_q,
        num_cores=LAMMPS_CORES,
        target=target,
        template=template,
        bounds=bounds,
    )
    threading.Thread(target=manager.run, daemon=True).start()

    # 5. Emit campaign_started SSE on user's chat session + return
    _q({"session_id": session_id}, {
        "type": "campaign_started",
        "campaign_id": campaign_id,
        "n_iterations": n_iterations,
    })
    return (
        f"Campaign {campaign_id} launched with {n_iterations} iterations. "
        f"You'll see iteration progress in the chat. You can keep chatting "
        f"about other things in the meantime."
    )
```

**What's lifted from current code** — this is exactly what `campaign_executor_node` ([src/graph/nodes.py:891-1020](src/graph/nodes.py#L891-L1020)) does today. The only differences:

| | Today (`campaign_executor_node`) | New (`start_campaign` tool) |
|---|---|---|
| **Inputs** | Reads `campaign_n_iterations`, `campaign_target`, `campaign_template`, `campaign_bounds`, `campaign_original_message` from `state` (parsed earlier by `_parse_campaign_parameters_block`) | Receives all as typed args directly from orchestrator LLM tool call — Pydantic validates |
| **Trigger** | Dispatched by `route_after_intent` after `classify_intent` returns `campaign_start` | Called when orchestrator decides to launch (after skill-driven Phase 1 setup) |
| **Fallback extraction** | Has a regex fallback that scans `conversation_history` if state fields are empty | Not needed — orchestrator always passes args explicitly |
| **Return** | Returns `{}` to the graph (node convention) | Returns a string for the orchestrator to relay to user |

**What's NOT inside `start_campaign`:**
- ❌ The iteration loop (lives in `CampaignManager.run`)
- ❌ LAMMPS execution (lives in `CampaignManager._run_one_iteration → run_lammps_simulation`)
- ❌ Δa measurement (lives in `compute_delta_a` from [src/benchmark/crack_metrics.py](src/benchmark/crack_metrics.py))
- ❌ Phase 1 setup conversation (lives in the campaign skill's `SKILL.md`)
- ❌ Parsing the `CAMPAIGN PARAMETERS:` block (gone — orchestrator calls tool with typed args)
- ❌ `_DualQueue` class definition (lift it from `nodes.py` as a private helper next to `start_campaign`, or move to a new `src/orchestrator/_dual_queue.py`)

### 4d. Execution responsibility — unchanged

The `simulate` tool **only generates scripts and emits the review event**. It does *not* run LAMMPS. Execution paths stay exactly as they are today:

| Trigger | Who runs LAMMPS | File |
|---|---|---|
| User clicks Approve in browser (chat sim) | `app.py` `/run_simulation` endpoint calls `run_lammps_simulation()` | [src/app.py](src/app.py) + [src/run_manager.py](src/run_manager.py) |
| Campaign iteration (`auto_approve=True`) | `CampaignManager._run_one_iteration` calls `run_lammps_simulation()` directly | [src/benchmark/campaign.py](src/benchmark/campaign.py) |
| HPC (Phase 1, `execution_target="hpc"`) | HPC subagent's existing Globus MCP flow on user's Approve | [src/graph/hpc_executor_node.py](src/graph/hpc_executor_node.py) |

`run_lammps_simulation()`, the `/run_simulation` HTTP endpoint, and the dump-monitor SSE streaming are untouched. The orchestrator owns *deciding what to simulate*; existing infrastructure owns *running it*.

### Untouched
- **`src/run_manager.py`** — `run_lammps_simulation()` and subprocess management unchanged.
- **`src/app.py` `/run_simulation` endpoint** — still the entry point for executing approved scripts.
- **`src/benchmark/campaign.py`** — `CampaignManager` reused as-is in Phase 1 (local execution only).
- **`src/agents.py`** — `IterativeLammpsGeneration`, `IterativeOVITOGeneration`, `SmartOrchestrator.extract_parameters` still used by simulate subgraph.
- **`src/graph/nodes.py`** — `extract_params_node`, `generate_lammps_node`, `detect_viz_node`, `generate_ovito_node` keep their bodies. Delete `chat_node`, `intent_node`, `analyze_results_node`, `campaign_executor_node`.
- **`src/graph/hpc_executor_node.py`** — body unchanged; only the wrapping changes.

---

## 5. Campaign mode flow

### 5a. Setup phase — user starts a campaign

```
User: "run a benchmark campaign on Al crack propagation, 10 iterations"
  │
  ▼
orchestrator (create_react_agent)
  │
  │  System prompt says: "If user mentions benchmarking/sweeping/optimization,
  │                       call load_skill('campaign') BEFORE responding."
  │
  ▼
orchestrator calls tool: load_skill("campaign")
  │
  ▼
Tool reads src/orchestrator/skills/campaign.md → returns its contents
  │
  ▼
orchestrator's next LLM turn sees the skill instructions in the tool result
and replies:
  "I'll set up a Griffith crack template. Knobs I'll vary across 10 runs:
     • strain_rate ∈ [1e-3, 1e-2] 1/ps
     • temperature ∈ [200, 600] K
     • crack_length_frac ∈ [0.1, 0.4]
     • crack_width_a0 ∈ [2, 6]
   Locked: material=Al, kim_model=..., system_size=...
   Target: minimize Δa
   Confirm or edit any field."
  │
  ▼
User: "looks good, go"
  │
  ▼
orchestrator calls tool:
  start_campaign(
    template={material: "Al", system_size: ..., phases: [...], ...},
    bounds={strain_rate: [1e-3, 1e-2], temperature: [200,600], ...},
    n_iterations=10,
    target="minimize delta_a",
    original_user_message="run a benchmark campaign on Al..."
  )
  │
  ▼
start_campaign tool (src/orchestrator/tools.py):
  1. campaign_id = "benchmark_20260622_143055"
  2. Create _DualQueue tee: campaign_q + chat_q (so browser sees live events)
  3. SESSION_QUEUES[campaign_id] = dual_q
  4. manager = CampaignManager(campaign_id, n_iter, original_msg, dual_q,
                               target, template, bounds)
  5. threading.Thread(target=manager.run, daemon=True).start()
  6. Return immediately: "Campaign benchmark_20260622_143055 launched, 10 iterations running"
  │
  ▼
orchestrator returns control to user, emits campaign_started SSE
```

### 5b. Background phase — `CampaignManager.run()` loops

```
For i in 1..10:
  │
  ▼
  CampaignManager._run_one_iteration(i)
    │
    ├─ Emit SSE: campaign_iteration_start {iteration: i}
    │
    ├─ Build user_message:
    │     if i == 1: original_user_message + locked template + bounds
    │     else:      locked template + bounds + format_history_message(past_runs)
    │                (e.g. "Past: 1) strain=7e-3,T=300 → Δa=+12.4Å  Propose iter 2...")
    │
    ▼
  simulation_graph.invoke(
    user_message=...,
    session_id=campaign_id,     ← reuses orchestrator graph
    workflow_stage="initial"
  )
    │
    │  Orchestrator handles this synthesized message:
    │    • Sees "Propose iteration N" + "Past attempts:" — system prompt says
    │      call load_skill('campaign') for these
    │    • Calls load_skill("campaign") → gets Phase 3 iteration instructions
    │    • Skill says: "in iteration mode, propose strict knob values,
    │                   call simulate(..., auto_approve=True)"
    │    • Orchestrator emits proposal + calls simulate(material, T, ensemble, ...,
    │                                                  auto_approve=True)
    │    • simulate tool → simulate_subgraph → extract_params → generate_lammps
    │      → detect_viz → generate_ovito
    │    • Returns lammps_script + ovito_script
    │
    ▼
  CampaignManager:
    • run_lammps_simulation(script, run_folder=runs/benchmark_<id>/run_<ts>/)
      (auto-executes, no human approval needed — campaign mode bypass)
    • compute_delta_a(dump_files) → Δa = e.g. 8.7 Å
    │
    ├─ Append to runs/benchmark_<id>/campaign_log.md
    ├─ Update runs/benchmark_<id>/summary.json
    ├─ Emit SSE: campaign_iteration_complete {iteration: i, delta_a: 8.7, knobs: {...}}
    │
    └─ (loop continues to iteration i+1)

After all 10 iterations:
  │
  ▼
  Emit SSE: campaign_complete {best_iteration: 7, best_delta_a: 2.1, all_runs: [...]}
```

### 5c. Browser sees

Real-time via SSE on the original chat session:
```
agent_content: "Campaign benchmark_20260622_143055 launched, 10 iterations running"
campaign_started: {campaign_id, n_iterations: 10}
campaign_iteration_start: {iteration: 1}
campaign_iteration_complete: {iteration: 1, delta_a: 12.4, knobs: {...}}
campaign_iteration_start: {iteration: 2}
campaign_iteration_complete: {iteration: 2, delta_a: 8.7, knobs: {...}}
...
campaign_complete: {best_iteration: 7, best_delta_a: 2.1}
```

User can keep chatting in the meantime — the daemon thread is independent of their chat turns.

### 5d. How the campaign skill loads

Two separate sessions run after launch:

| Session | Who | Skill loaded? |
|---------|-----|---------------|
| **A — User's chat** | Browser ↔ orchestrator | Loaded once during Phase 1 setup via explicit `load_skill('campaign')` tool call. After `start_campaign` returns, skill content is gone from working context (lived only in that turn's tool result). |
| **B — Campaign daemon** | `CampaignManager` thread | Orchestrator re-calls `load_skill('campaign')` at the start of **every iteration**, because the synthesized iteration message contains the trigger keywords the system prompt watches for. |

**System prompt tells the orchestrator when to load the skill:**
```
Available skills (call load_skill(name) to load):
  • campaign — use when user mentions benchmark, sweep, optimize across iterations,
    OR when the message says "Propose iteration N" and includes "Past attempts:"
```

**Per-iteration loop in detail:**
```
CampaignManager iteration i:
  │
  ▼
synthesized message:
  "Propose iteration 2. Past attempts: 1) strain=7e-3,T=300 → Δa=+12.4Å
   Locked template: ...  Knob bounds: ..."
  │
  ▼
simulation_graph.invoke(message, session_id=campaign_id)
  │
  ▼
orchestrator turn START
  │
  │  System prompt + message keywords → LLM decides to call load_skill('campaign')
  │
  ▼
orchestrator calls load_skill('campaign') → reads campaign.md → tool returns text
  │
  ▼
orchestrator's next LLM call sees skill text in tool-result message:
  "iteration mode — pick knob values within bounds, state your hypothesis,
   then call simulate(..., auto_approve=True) with the full param set"
  │
  ▼
orchestrator calls simulate(material=Al, temperature=350, strain_rate=5e-3, ...,
                            auto_approve=True)
  │
  ▼
simulate_subgraph runs → returns lammps_script + ovito_script
  │
  ▼
orchestrator turn END  ← tool-result messages stay in this turn's checkpoint;
                          fresh invocation next iteration starts new MessagesState
  │
  ▼
CampaignManager grabs scripts from returned state, runs LAMMPS, measures Δa
```

**Why this works:**
- No regex, no `is_campaign_context()`, no library middleware — explicit `load_skill` tool call the LLM decides to make based on a tight system prompt
- Skill file is the single source of truth for "what does campaign mode look like" — covers both user-facing setup and machine-synthesized iterations
- Per-iteration cost is one extra tool call (read file, return text) — cheap
- CampaignManager unchanged — same `simulation_graph.invoke` per iteration, same Δa measurement, same logging
- Failure mode: if the LLM forgets to call `load_skill`, it falls back to base system prompt + tries to handle from there. Mitigate with: (a) explicit instruction in system prompt, (b) iteration message itself can include "(reminder: load_skill('campaign'))" as a hint, (c) eval suite during implementation

---

## 6. Dependencies

**No new deps.** `langgraph` and `langgraph-prebuilt` are already in `pixi.toml`. The whole orchestrator builds on `from langgraph.prebuilt import create_react_agent`.

---

## 7. SSE streaming

Keep `_q(state, event_dict)` pattern ([src/graph/nodes.py:48](src/graph/nodes.py#L48)). Orchestrator tools receive `session_id` via the orchestrator graph's state (a `MessagesState` subclass) and call `_q` directly. `_DualQueue` tee for campaign sessions moves into the `start_campaign` tool.

---

## 8. Verification

1. `rm data/sessions.db` (stale checkpoints don't match new schema).
2. `pixi run run-app`, open browser to `localhost:2002`.
3. **Chat-only** — "what KIM potentials exist for Cu?" → `get_kim_models` tool → response streams. No simulation triggered.
4. **Simulate** — "simulate Al at 300K NVE for 10ps" → `simulate(...)` → LAMMPS+OVITO subgraph → `scripts_ready_for_review` SSE → Approve → run.
5. **HPC** — "run that on HPC" → `hpc_submit(...)` tool → PBS preview → edit → resume → Globus submit → job ID.
6. **Campaign** — "run a benchmark campaign on Al crack propagation" → orchestrator calls `load_skill('campaign')` → Phase 1 setup conversation → user confirms → `start_campaign(...)` → daemon thread → browser sees `campaign_iteration_start/complete` events. Each iteration: orchestrator re-calls `load_skill('campaign')` then `simulate(..., auto_approve=True)`.
7. **Analyze** — "show me the last 5 runs" → `analyze_runs(...)` → list streams.
8. Restart server mid-HPC-pause; resume PBS still works.

---

## 9. Phase 2: HPC campaigns (not in scope now)

Locked out of Phase 1, but the Phase 1 design leaves the door open. When we want it:

**User flow:**
"run a 50-iteration campaign on HPC, 4 nodes each, walltime 4h"

**What changes:**
1. **`start_campaign` tool** gains optional args: `execution_target="hpc"`, `pbs_template={cores:128, walltime:"4:00:00", project:"..."}`.
2. **`CampaignManager.__init__`** stores `execution_target` and `pbs_template`.
3. **`CampaignManager._run_one_iteration`**:
   - Synthesized iteration message now mentions "Submit to HPC with the locked PBS template"
   - After `simulation_graph.invoke` returns, instead of calling `run_lammps_simulation()`, polls Globus job status (via HPC subagent's status tool or direct MCP call)
   - Waits for completion → transfers dump files back → measures Δa
4. **`campaign.md` iteration section** says: `simulate(..., execution_target="hpc", auto_approve=True)`.
5. **`hpc_submit` tool** grows `auto_pbs: bool` param — when True, uses the campaign's locked `pbs_template` instead of `interrupt()`-ing for user edit.

**What doesn't change:**
- Orchestrator code, tool signatures (just new optional args), graph topology, SSE plumbing, skill loading mechanism.

**Why this is cheap to add later:** The `execution_target` + `auto_approve` flags on `simulate` are already in the Phase 1 signature. No retro-fit needed; we just wire the `"hpc"` + `True` branch.

---

## 10. Implementation order

Tracked in TodoWrite:
1. Verify deepagents install + API shape (skills file format, `CompiledSubAgent`, `context_schema`)
2. Add deepagents dependency to `pixi.toml`
3. Build `simulate_subgraph` in `src/graph/graph.py` (slim down to LAMMPS+OVITO only)
4. Write `src/orchestrator/tools.py` (get_kim_models, simulate, analyze_runs, start_campaign)
5. Write `src/orchestrator/skills/campaign/SKILL.md` (with dual setup+iteration triggers)
6. Wrap `hpc_executor_node` as `CompiledSubAgent` in `src/orchestrator/skills/hpc_subagent.py`
7. Write `src/orchestrator/agent.py` (`create_deep_agent` with tools+skills+subagents)
8. Wire `orchestrator_graph` into `src/app.py`
9. Clean up: delete `chat_node` / `intent_node` / `analyze_results_node` / `campaign_executor_node`, drop `route_after_intent`, strip `is_campaign_context`
10. Verify end-to-end (chat, simulate, HPC, campaign, analyze)
