# LangGraph Agent Migration Notes
*Last updated: 2026-04-28*

## Overview

Migrated the ChatMD simulation pipeline from a monolithic sequential `app.py` (~600 lines) into a structured LangGraph graph. `app.py` is now a thin HTTP/WebSocket layer. Business logic lives in `src/graph/`.

---

## Architecture: Before vs After

### Before
- Single function `llm_stream_live()` (~370 lines) handled everything sequentially:
  classify intent → build context → LLM call → extract params → memory → LAMMPS gen → viz gen
- No persistent state between turns
- Queue passed directly inside state (not serializable)

### After
- LangGraph `StateGraph` with 8 nodes and conditional routing
- `SqliteSaver` checkpointing — state persists across HTTP requests via `thread_id = session_id`
- SSE queue kept outside state in a global registry (`SESSION_QUEUES`)
- `app.py` reduced to ~80 lines for the `/api/llm-stream-live` route

---

## File Structure

```
src/
  graph/
    __init__.py        # empty — makes graph/ a Python package
    state.py           # SimulationState TypedDict
    nodes.py           # 8 node functions + SESSION_QUEUES registry
    routing.py         # 4 routing functions
    graph.py           # compiled simulation_graph with SqliteSaver
```

---

## Graph Topology

```
START
  └─► classify_intent
        ├─► [script_ready + execute] → END   (bypass generation, start sim directly)
        ├─► [intent == analyze]      → analyze_results → END
        └─► chat_agent
              └─► extract_params
                    ├─► [no parameters] → END
                    └─► generate_lammps
                          ├─► [review failed] → END
                          └─► detect_viz
                                ├─► [no custom viz] → END
                                └─► generate_ovito
                                      └─► END
```

> Note: `build_context_node` and `retrieve_memory_node` from the original plan were merged into `chat_agent`. KIM registry lookup is now a real tool call inside `chat_agent`. Memory retrieval is handled inside `IterativeLammpsGeneration.forward()` directly.

---

## Nodes

| Node | Wraps | Key state outputs |
|------|-------|-------------------|
| `classify_intent` | `SmartOrchestrator.classify_intent()` (DSPy) | `intent` |
| `chat_agent` | OpenAI streaming + `get_kim_models` tool call | `analysis`, `registry_context` |
| `extract_params` | `SmartOrchestrator.extract_parameters()` (DSPy) | `has_parameters`, `material`, `kim_model`, `temperature`, `ensemble`, `duration_ps`, `system_size`, `visualization_notes`, `camera_angle` |
| `generate_lammps` | `IterativeLammpsGeneration.forward()` (DSPy) — includes memory retrieval internally | `lammps_script`, `dump_filename_pattern`, `review_passed` |
| `detect_viz` | `detect_precomputed_properties()` + `SmartOrchestrator.detect_visualization()` | `viz_request`, `precomputed_properties`, `simulation_context`, `has_custom_viz` |
| `generate_ovito` | `IterativeOVITOGeneration.forward()` (DSPy) | `ovito_script` |
| `analyze_results` | OpenAI tool-calling agent with `list_recent_runs` + `get_run_log` tools | streams analysis to SSE |

---

## Routing Functions

Routing functions are **plain Python functions**, not nodes. They read state and return the next node name (string). No LLM calls happen here.

```python
# src/graph/routing.py

def route_after_classify(state) -> str:
    # Already-approved script + execute intent → skip generation
    if state["workflow_stage"] == "script_ready" and state["intent"] == "execute":
        return END
    return "build_context"

def route_after_extract(state) -> str:
    if not state.get("has_parameters"):
        return END
    return "retrieve_memory"

def route_after_lammps(state) -> str:
    if state.get("review_passed"):
        return "detect_viz"
    return END

def route_after_detect_viz(state) -> str:
    if state.get("has_custom_viz"):
        return "generate_ovito"
    return END
```

---

## SimulationState

All fields must be JSON-serializable — no `queue.Queue` objects.

```python
class SimulationState(TypedDict):
    # Input
    user_message: str
    workflow_stage: str          # "initial" | "parameters_proposed" | "script_ready"
    conversation_history: List[Dict]
    pending_script: Optional[str]
    ovito_script_param: Optional[str]

    # Classification
    intent: Optional[str]        # "execute" | "modify" | "cancel" | "question" | "new_simulation"

    # Orchestrator
    registry_context: Optional[str]
    messages: Optional[List[Dict]]
    analysis: Optional[str]

    # Extracted parameters
    has_parameters: Optional[bool]
    material: Optional[str]
    kim_model: Optional[str]
    temperature: Optional[float]
    ensemble: Optional[str]
    duration_ps: Optional[float]
    system_size: Optional[int]
    visualization_notes: Optional[str]
    camera_angle: Optional[str]

    # Memory
    lammps_memory_context: Optional[str]
    viz_memory_context: Optional[str]

    # LAMMPS outputs
    lammps_script: Optional[str]
    dump_filename_pattern: Optional[str]
    total_steps: Optional[int]
    dump_freq_val: Optional[int]
    review_passed: Optional[bool]

    # Visualization outputs
    has_custom_viz: Optional[bool]
    viz_request: Optional[str]
    precomputed_properties: Optional[Dict]
    simulation_context: Optional[Dict]
    ovito_script: Optional[str]
    target_timesteps: Optional[List[int]]

    # Control
    error: Optional[str]
    session_id: Optional[str]    # used to look up SESSION_QUEUES in nodes.py
```

---

## SSE Streaming with LangGraph

### The Problem
`queue.Queue` is not JSON-serializable. SqliteSaver cannot checkpoint it.

### The Solution: SESSION_QUEUES Registry

```python
# src/graph/nodes.py

SESSION_QUEUES: Dict[str, queue.Queue] = {}

def _q(state: SimulationState, event: dict):
    sid = state.get("session_id")
    if sid and sid in SESSION_QUEUES:
        SESSION_QUEUES[sid].put(event)
```

- `session_id` (a plain string) goes into state — serializable by SqliteSaver
- The actual `queue.Queue` lives in `SESSION_QUEUES[session_id]` — never in state
- `app.py` creates the queue, registers it, starts the graph in a thread, then reads from the queue for SSE

### app.py pattern

```python
session_id = body.get("session_id") or str(uuid.uuid4())
sse_q = queue.Queue()
SESSION_QUEUES[session_id] = sse_q

threading.Thread(target=lambda: simulation_graph.invoke(initial_state, config={"configurable": {"thread_id": session_id}}), daemon=True).start()

def generate():
    while True:
        item = sse_q.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"

return Response(generate(), mimetype="text/event-stream")
```

---

## SqliteSaver: Persistent Conversation Memory

### Setup

```python
# src/graph/graph.py
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect(db_path, check_same_thread=False)  # check_same_thread=False required for Flask threads
checkpointer = SqliteSaver(conn)
graph = g.compile(checkpointer=checkpointer)
```

**Do NOT use `SqliteSaver.from_conn_string()`** — it returns a context manager (generator), not a SqliteSaver instance. Direct `sqlite3.connect()` is correct.

### How Memory Works Across Turns

- Each request uses `thread_id = session_id` in `graph_config`
- LangGraph loads the last checkpoint for that thread before invoking
- `material`, `temperature`, `lammps_script`, etc. from turn 1 are available in turn 2
- `build_context_node` injects previous simulation context into the orchestrator prompt:

```python
prev_material = state.get("material")
prev_temp     = state.get("temperature")
prev_script   = state.get("lammps_script")
if prev_material and prev_script:
    current_message += (
        f"\n\n[PREVIOUS SESSION CONTEXT: Last simulation was {prev_material}"
        f" at {prev_temp}K. The user may be referring to this.]"
    )
```

### Frontend: Sending session_id Back

The frontend must store and re-send `session_id` on every subsequent turn:

```javascript
// Stored when workflow_complete event arrives
let currentSessionId = null;

case 'workflow_complete':
    if (data.session_id) currentSessionId = data.session_id;
    break;

// Sent with every POST
const body = {
    command: message,
    history: conversationHistory,
    session_id: currentSessionId   // null on first turn; backend generates one
};
```

---

## DSPy Thread-Safety

### The Problem

DSPy 3.x locks `dspy.settings` to the thread that first calls `dspy.configure()`. Calling `configure_dspy_lm()` inside a background thread (e.g., inside `classify_intent_node`) after it was already called in the main thread raises:

```
dspy.settings can only be changed by the thread that initially configured it
```

### The Fix

Configure DSPy **once at startup** in the main thread, before Flask routes or background threads run:

```python
# src/app.py — after imports, before route definitions
configure_dspy_lm()
```

Remove all per-request calls:
- `classify_intent_node` in `nodes.py` — removed
- `submit_feedback()` in `app.py` — removed

---

## Post-Migration Bugs and Fixes

### Bug 1: `SqliteSaver.from_conn_string()` returns a context manager, not a saver

**Symptom:** `'_GeneratorContextManager' object has no attribute 'get_next_version'` on the first request.

**Cause:** `SqliteSaver.from_conn_string(path)` is a context manager meant to be used with `with`. Calling it directly returns the generator object, not the SqliteSaver instance.

**Fix:** Create the sqlite connection manually and pass it directly:

```python
# WRONG
checkpointer = SqliteSaver.from_conn_string(db_path)

# CORRECT
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

`check_same_thread=False` is required because Flask serves requests on multiple threads.

---

### Bug 2: DSPy thread-safety error

**Symptom:** `dspy.settings can only be changed by the thread that initially configured it` — triggered when the user typed a follow-up message after a simulation.

**Cause:** DSPy 3.x locks `dspy.settings` to the thread that first calls `dspy.configure()`. `configure_dspy_lm()` was being called in two places:
1. Inside `classify_intent_node` — runs in the background graph thread
2. Inside `submit_feedback()` — runs in the Flask main thread

Whichever ran first "owned" the settings. The other thread then raised this error.

**Fix:** Configure DSPy exactly once at module load time in the main thread, before any routes or background threads start. Remove all per-request calls.

```python
# src/app.py — after imports, before any route definitions
configure_dspy_lm()
```

Removed from:
- `classify_intent_node` in `nodes.py`
- `submit_feedback()` in `app.py`
- Removed unused `configure_dspy_lm` import from `nodes.py`

---

### Bug 3: `initial_state` None values overwriting the SqliteSaver checkpoint

**Symptom:** "what simulation did you just do?" → Orchestrator said it had no memory of any simulation, even though SqliteSaver had correctly saved the state.

**Root cause:** SQLite was working. The bug was in Python. Every request built `initial_state` with every field explicitly set to `None`:

```python
initial_state = {
    "material": None,       # ← overwrites "Aluminum" from checkpoint
    "lammps_script": None,  # ← overwrites the generated script
    "temperature": None,    # ← overwrites 300.0
    ...
}
```

LangGraph merges the incoming state with the checkpoint using last-write-wins. So `None` in `initial_state` won the merge and wiped out everything SqliteSaver had stored.

**Fix:** Only include request-origin fields in `initial_state`. Omit generation outputs entirely — LangGraph restores them from the checkpoint automatically.

```python
# Only fields that come from the HTTP request
initial_state: SimulationState = {
    "user_message":         user_command,
    "workflow_stage":       workflow_stage,
    "conversation_history": history,
    "pending_script":       pending_script_param,
    "ovito_script_param":   ovito_script_param,
    "session_id":           session_id,
    # Per-turn transient fields — always reset each request
    "intent":               None,
    "registry_context":     None,
    "messages":             None,
    "analysis":             None,
    "has_parameters":       None,
    "lammps_memory_context": None,
    "viz_memory_context":   None,
    "review_passed":        None,
    "has_custom_viz":       None,
    "error":                None,
    # material, temperature, lammps_script, etc. NOT included
    # — checkpoint restores them
}
```

---

### Bug 4: Non-simulation turns wiping simulation state from checkpoint

**Symptom:** "hi" → Orchestrator correctly says "you ran Aluminum at 300K last time". Then "what simulation did you just do?" → Orchestrator says it has no memory.

**Root cause:** `extract_params_node` always ran and always wrote back all parameter fields, even when `has_parameters=False`. For a "hi" message:
1. Checkpoint loaded: `material="Aluminum"`, `temperature=300`, `lammps_script="..."`
2. `orchestrator_chat_node` correctly saw the previous context and mentioned Aluminum
3. `extract_params_node` ran on the "hi" analysis, found no parameters, returned `material=""`, `temperature=300.0` (default), etc.
4. Those empty values were written to the checkpoint, overwriting `material="Aluminum"`
5. Next turn: checkpoint had `material=""` — simulation memory gone

**Fix:** When `has_parameters=False`, return only that flag and leave all other fields untouched in the checkpoint:

```python
# src/graph/nodes.py — extract_params_node
if not params["has_parameters"]:
    return {"has_parameters": False}   # everything else stays in checkpoint

return {
    "has_parameters": True,
    "material":       params.get("material"),
    "temperature":    params.get("temperature"),
    ...
}
```

**Key insight:** The problem in bugs 3 and 4 was never SQLite — SqliteSaver was saving and restoring correctly. The bugs were in Python code that accidentally defeated it by overwriting checkpoint values with None or empty defaults.

---

### Bug 5: Frontend not sending `session_id` back on follow-up turns

**Symptom:** Each new message from the user started a brand new LangGraph thread (new UUID), so the checkpoint from the previous turn was never loaded.

**Cause:** The backend sends `session_id` in the `workflow_complete` SSE event, but the frontend was discarding it. The next POST body contained no `session_id`, so the backend generated a fresh UUID every time.

**Fix — backend:** Ensure every `workflow_complete` event in every code path carries `session_id`:

```python
# All early-return paths in run_graph()
sse_q.put({"type": "workflow_complete", "session_id": session_id})
```

**Fix — frontend:**

```javascript
let currentSessionId = null;  // persists across turns

// Capture from workflow_complete
case 'workflow_complete':
    if (data.session_id) currentSessionId = data.session_id;
    break;

// Send back with every POST
const body = {
    command: message,
    history: conversationHistory,
    session_id: currentSessionId   // null on first turn; backend generates and returns one
};
```

---

## Dead Code Removed

### `src/agents.py`
- `OVITOScriptReview` class — deleted
- `OVITOScriptGeneration` class — deleted
- `SmartOrchestrator.analyze_request()` — deleted
- `SmartOrchestrator.select_tools()` — deleted

### `src/tools.py`
- `parse_visualization_request()` — deleted

### `src/run_manager.py`
- `save_ovito_script()` — deleted

---

## WebSocket and Rendering Bugs

### Bug 6: Flask debug reloader killing WebSocket on OVITO subprocess spawn

**Symptom:** WebSocket disconnected immediately when simulation started.

**Cause:** `debug=True` in `socketio.run()` enables Flask's file-change reloader. OVITO's `TachyonRenderer` spawns internal processes; the reloader detected them and restarted Flask, killing the WebSocket.

**Fix:** Disable reloader while keeping debug output:
```python
socketio.run(app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
```

---

### Bug 7: LAMMPS MPI subprocess inheriting WebSocket file descriptor

**Symptom:** WebSocket dropped when `subprocess.Popen` launched `mpirun`.

**Cause:** `Popen` inherits all open file descriptors from the parent Flask process, including the WebSocket socket fd. The OS gave the child a copy, causing the connection to reset.

**Fix:**
```python
process = subprocess.Popen(cmd, ..., close_fds=True)
```

---

### Bug 8: Flask-SocketIO missing explicit async_mode and ping settings

**Symptom:** WebSocket disconnecting on long-running simulations.

**Fix:** Explicitly configure threading mode and extend ping timeouts:
```python
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=120,
    ping_interval=25,
)
```

---

### Bug 9: `trajectoryFrames` not cleared between simulations

**Symptom:** Second simulation's OVITO frames were silently dropped; old image stayed visible.

**Cause:** `trajectoryFrames` array and the viewer element were never reset before a new `start_streaming_simulation` event. The 100-frame cap was already hit from simulation 1, so all new frames were discarded.

**Fix:** Call `hideTrajectoryViewer()` before emitting `start_streaming_simulation`. This resets the array, hides the old image, and shows the placeholder.

---

### Bug 10: `addTypingIndicator` missing `'Visualization'` case → empty CSS class crash

**Symptom:** `SyntaxError: Failed to execute 'add' on 'DOMTokenList': The token provided must not be empty` during the SSE phase of the second simulation.

**Cause:** The switch statement in `addTypingIndicator` had no case for `'Visualization'`, so `colorClass` stayed `''`. `classList.add(...''.split(' '))` passes an empty string token, which throws.

**Fix:** Added `'Visualization'` case and a `default` fallback to the switch.

---

### Bug 11: OVITO `import_file()` leaks OS semaphores → Flask process crash on second simulation

**Symptom:** `transport close` + `ERR_CONNECTION_REFUSED` on every second simulation. The entire Flask server process died.

**Root cause:** OVITO's `import_file()` internally creates OS-level semaphores registered with Python's `multiprocessing.resource_tracker`. These semaphores were not fully released when the first simulation's pipeline was garbage collected. On the second call to `import_file()`, the resource tracker detected the leak, entered "shutdown" mode, and crashed the Python process.

Making `TachyonRenderer` and `Viewport` singletons did not help — `import_file()` itself was the source of the leak.

**Fix:** Run ALL OVITO code in a dedicated subprocess per simulation.

Created `src/ovito_render_server.py` — a standalone script that:
- Imports OVITO once and keeps the pipeline cached across frames
- Reads JSON render requests from stdin
- Writes JSON results (base64 PNG) to stdout

`rendering_worker` in `tools.py` now launches this script with `subprocess.Popen(..., close_fds=True)` and communicates via newline-delimited JSON:

```python
render_proc = subprocess.Popen(
    [sys.executable, _RENDER_SERVER_SCRIPT],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
    close_fds=True,
)
```

Each simulation gets a **fresh subprocess** — OVITO semaphores are fully isolated in the child and cleaned up when the child exits. Flask is never affected.

**Why not `multiprocessing.Process`?**  
macOS Python 3.11 uses `'spawn'` as the default multiprocessing start method. `spawn` requires the target function to be picklable, which means the child must be able to `import tools`. Since `tools.py` depends on `config`, `kim_utils`, etc., this fails silently. The `subprocess.Popen` + standalone script approach avoids all pickling and import path issues.

---

### Bug 12: `apply_basic_visualization` using `'Potential Energy'` property

**Symptom:** `Modifier 'Color coding' reported: The particles property with the name 'Potential Energy' does not exist` — all frames failed to render for simulations that don't compute potential energy (e.g. Argon LJ).

**Fix:** Changed basic viz to color by `'Particle Type'` which is always present in any LAMMPS dump file:

```python
pipeline.modifiers.append(ColorCodingModifier(property='Particle Type'))
```

---

## Proof-of-Concept Test

`test_langgraph_dspy.py` at the project root validates that DSPy + LangGraph + `dspy.ReAct` work together before the full migration was committed.

Key finding: `dspy.ReAct` handles the thought→tool→observation loop internally. Tools are plain Python functions — no `@tool` decorator needed. Fully optimizable with the DSPy compiler.

---

## Phase 2 (Not Yet Done)

- `interrupt()` for human-in-the-loop parameter confirmation
- Streaming tokens directly from LangGraph nodes (instead of putting onto queue)
- DSPy compiler optimization of individual nodes
- Replace `SmartOrchestrator` with `dspy.ReAct` agent that calls tools dynamically
