# HPC Execution Workflow

## Overview

ChatMD supports submitting LAMMPS simulations to the Swing HPC cluster at LCRC via Globus MCP. The same LangGraph SSE pipeline used for local execution handles HPC — no separate WebSocket handler. A sessions sidebar (Claude-style) provides persistent session management so HPC job tracking survives browser closes and server restarts.

---

## Architecture Decisions

| Decision | Choice | Reason |
|---|---|---|
| Execution decision point | Review card ("Run on HPC" button) | User decides at approval time, not during chat |
| HPC routing | Single `hpc_executor_node` | Handles Phase 1 (PBS preview), Phase 2 (Globus submit), and status queries — Globus tools in one place |
| Globus tool registration | Dynamic via `session.list_tools()` | No hardcoded schemas; LLM sees whatever Globus MCP exposes |
| PBS config | `config.py` defaults | No UI form; values are site-specific constants |
| Job persistence | `hpc_jobs: List[Dict]` in `SimulationState` | SqliteSaver persists it per session; no new DB table needed |
| Session persistence | `sessions` SQLite table + localStorage | Sessions survive page reloads; sidebar shows all sessions |
| Orchestrator Globus tools | None | Orchestrator stays clean; all Globus interactions owned by `hpc_executor_node` |

---

## Full Flow

```
1. User: "simulate copper at 500K"
         ↓
2. classify_intent → chat_agent → extract_params → generate_lammps → generate_ovito
         ↓
3. SSE: scripts_ready_for_review
   Review card shows:
     [Approve & Run]  [Run on HPC]  [Edit Scripts ▼]
         ↓
4a. "Approve & Run" → existing local WebSocket path (unchanged)
         ↓
4b. "Run on HPC" → POST /api/llm-stream-live {
        command: "run on hpc",
        pending_script: ...,
        execution_target: "hpc",
        session_id: ...
    }
         ↓
5.  app.py: pending_script present → workflow_stage = "script_ready"
    route_after_intent: execute + script_ready + execution_target=hpc → hpc_executor_node
         ↓
6.  hpc_executor_node — Phase 1 (no Globus yet):
    a. create_run_folder(), save input.lammps + ovito_script.py locally
    b. Generate run.pbs from config.py defaults
    c. SSE: {"type": "pbs_preview", "pbs_script": "...", "run_folder": "..."}
    d. SSE: "Here is the PBS script. Review and confirm to proceed."
    e. Sets workflow_stage = "pbs_pending"
         ↓
   Frontend shows PBS approval card with script content:
     [Confirm & Submit to HPC]  [Cancel]
         ↓
7.  User clicks "Confirm & Submit" → POST /api/llm-stream-live {
        command: "confirm hpc submit",
        workflow_stage: "pbs_pending",
        session_id: ...
    }
         ↓
8.  hpc_executor_node — Phase 2 (Globus MCP ReAct loop):
    e. globus_transfer_submit_task — upload input.lammps + run.pbs to HPC endpoint
    f. globus_compute_register_shell_command + globus_compute_submit_task → PBS submission
    g. globus_compute_get_task_status → get globus_task_id
    h. SSE: agent_content events streaming progress ("Transferring...", "Submitted ✓")
    i. Appends to state["hpc_jobs"]: {globus_task_id, hpc_run_folder, submitted_at, description}
         ↓
9.  Later — user: "what happened to my job?"
    classify_intent → intent = "hpc_status" → hpc_executor_node
    Reads hpc_jobs from state, calls globus_compute_get_task_status per job
    If SUCCEEDED → globus_transfer_submit_task to download log + dump files → OVITO replay
```

---

## Components

### New File: `src/graph/hpc_executor_node.py`

Single LangGraph node that handles all three phases:

- **Phase 1** (`workflow_stage="script_ready"` + `execution_target="hpc"`): saves scripts locally, generates PBS, emits `pbs_preview` SSE, sets `workflow_stage="pbs_pending"`
- **Phase 2** (`workflow_stage="pbs_pending"`): `MultiServerMCPClient` connects to `globus-mcp`, `create_react_agent` drives tool calls to transfer files + submit job
- **Status** (`intent="hpc_status"`): same agent reads `hpc_jobs` from state, checks each job via Globus MCP

**LangChain-native MCP integration (`langchain_mcp_adapters`):**

```python
GLOBUS_MCP_CONFIG = {
    "globus": {
        "command": "uvx",
        "args": ["globus-mcp"],
        "transport": "stdio",
    }
}

async with MultiServerMCPClient(GLOBUS_MCP_CONFIG) as client:
    tools = client.get_tools()   # LangChain BaseTool objects — no format conversion needed
    agent = create_react_agent(llm, tools, prompt=SystemMessage(content=HPC_EXECUTOR_PROMPT))
    async for chunk in agent.astream({"messages": [HumanMessage(content=user_msg)]},
                                      stream_mode="updates"):
        # chunk["agent"] → AIMessage with tool_calls or final text
        # chunk["tools"] → ToolMessage with tool result
        # → forward each to user via _q(state, {"type": "agent_content", ...})
```

**Globus MCP tools (fetched dynamically — no hardcoded schemas):**
Whatever `uvx globus-mcp` exposes is available to the agent. The agent decides which tools to call and in what order — nothing is prescribed in code.

**`HPC_EXECUTOR_PROMPT` — role only, no steps:**
```
You are an HPC agent for ChatMD. You have access to Globus tools to move files between
endpoints and run commands on HPC clusters.

You will be given context: local paths, Globus endpoint IDs, and either a job to submit
or a list of previously submitted jobs to check.

Use the available tools to accomplish the user's goal. Decide for yourself which tools
to call and in what order based on what needs to be done. Report progress clearly after
each action. If something fails, explain what went wrong and stop.
```

**`user_msg` — facts only, no instructions:**
- Phase 2: local run_folder, endpoint UUIDs, HPC remote directory
- Status: original user question + `hpc_jobs` list with task IDs and run folders

**Key function:** `_run_hpc_agent(state, user_msg) → str`
- Sync wrapper around `_run_hpc_agent_async()` via `asyncio.run()`
- Streams each agent step and tool result to user via `_q()`
- Returns `globus_task_id` parsed from `ToolMessage` content

---

### Modified Files

#### `src/graph/state.py`
Three new HPC fields:
```python
execution_target: Optional[str]       # "local" | "hpc" — set from POST body by app.py
hpc_jobs: Optional[List[Dict]]        # [{globus_task_id, hpc_run_folder, submitted_at, description}]
hpc_run_folder_pending: Optional[str] # temp: local folder between Phase 1 and Phase 2
```

#### `src/graph/routing.py` — `route_after_intent()`
```
pbs_pending                              → hpc_executor
intent == "hpc_status"                   → hpc_executor
execute + script_ready + target == "hpc" → hpc_executor
execute + script_ready                   → END (local path)
analyze                                  → analyze_results
otherwise                                → chat_agent
```

#### `src/graph/graph.py`
- Imports and registers `hpc_executor_node`
- Adds `"hpc_executor": "hpc_executor"` to `classify_intent` conditional edges
- Adds `g.add_edge("hpc_executor", END)`

#### `src/prompts.py`
- `ClassifyUserIntent` intent Literal extended with `'hpc_status'`
- `HPC_EXECUTOR_PROMPT` constant added — instructs agent for SUBMISSION MODE and STATUS MODE

#### `src/config.py`
```python
GLOBUS_LOCAL_ENDPOINT = ""            # local Globus endpoint UUID
GLOBUS_HPC_ENDPOINT = ""              # Swing endpoint UUID
GLOBUS_COMPUTE_ENDPOINT_ID = "00000000-0000-0000-0000-000000000000"
HPC_RUN_BASE_DIR = "/path/to/your/hpc/run/dir"
HPC_DEFAULT_NODES = 1
HPC_DEFAULT_CORES = 8
HPC_DEFAULT_WALLTIME = "01:00:00"
HPC_PROJECT_ID = "YOUR_PROJECT_ID"
```

#### `src/app.py`
- Reads `execution_target` from POST body → added to `initial_state`
- Reads `workflow_stage: "pbs_pending"` from POST body (Phase 2 trigger)
- Includes `execution_target` in `scripts_ready_for_review` SSE payload
- Upserts session row in `sessions` table when a new session_id appears
- Auto-titles session from `material + temperature` after first simulation
- Sessions REST endpoints: `GET/POST /api/sessions`, `PATCH/DELETE /api/sessions/<id>`
- `_init_sessions_table()` called at startup

#### `index.html`
- Body restructured to `flex` with 208px sessions sidebar + flexible main content
- Review card shows both **"Approve & Run"** and **"Run on HPC"** buttons (or only "Approve & Run on HPC" when `execution_target="hpc"` from intent)
- `approveAndRunHPC()` — posts `execution_target: "hpc"` + `pending_script`, triggers Phase 1
- `pbs_preview` SSE case → `showPBSApprovalCard()` — renders PBS script with "Confirm & Submit" / "Cancel"
- `confirmHPCSubmit()` — posts `workflow_stage: "pbs_pending"`, triggers Phase 2
- `_hpcStream(body)` — shared fetch+SSE reader for HPC flows
- Sessions sidebar: `loadSessions()`, `refreshSessions()`, `renderSessions()`, `switchSession()`, `createNewSession()`
- `workflow_complete` handler saves `session_id` to `localStorage` and calls `refreshSessions()`

---

## Session Persistence

Sessions are stored in `data/sessions.db` (same file as LangGraph's `SqliteSaver`):

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT DEFAULT 'New Session',
    created_at  TEXT,
    updated_at  TEXT
);
```

- A row is upserted on every POST to `/api/llm-stream-live`
- Title is auto-set from `material + temperature` after the first simulation (e.g. "Cu 500K")
- `hpc_jobs` list lives inside `SimulationState` — persisted by SqliteSaver per `thread_id = session_id`
- On page load, `localStorage.getItem('chatmd_last_session')` restores the last active session

---

## PBS Script Template

Generated by `_generate_pbs_script(run_folder_name, kim_model)` from `config.py` defaults.
`kim_model` is taken from `state["kim_model"]` — the KIM potential selected during parameter extraction.

```bash
#!/bin/bash -l
#PBS -A YOUR_PROJECT_ID
#PBS -l select=1:ncpus=8:mpiprocs=8
#PBS -l walltime=00:30:00
#PBS -N chatmd_<run_folder_name>
#PBS -j oe
#PBS -m be
#PBS -M your-email@institution.edu

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
export PATH="$HOME/.pixi/bin:$PATH"
which lmp
module list
echo Jobid: $PBS_JOBID
echo Running on host `hostname`

pixi run kim-api-collections-management install user <kim_model>
pixi run mpirun -np 8 lmp -in input.lammps -log log.lammps

echo "Simulation complete"
```

**Config values driving the template** (`src/config.py`):
| Variable | Value |
|---|---|
| `GLOBUS_LOCAL_ENDPOINT` | `00000000-0000-0000-0000-000000000000` (Mac) |
| `GLOBUS_HPC_ENDPOINT` | `00000000-0000-0000-0000-000000000000` (LCRC Swing) |
| `GLOBUS_COMPUTE_ENDPOINT_ID` | `00000000-0000-0000-0000-000000000000` |
| `HPC_PROJECT_ID` | `YOUR_PROJECT_ID` |
| `HPC_DEFAULT_NODES` | `1` |
| `HPC_DEFAULT_CORES` | `8` |
| `HPC_DEFAULT_WALLTIME` | `00:30:00` |
| `HPC_USER_EMAIL` | `your-email@institution.edu` |

---

## What Stays Unchanged

- `tools.py` — local LAMMPS execution path
- `ovito_render_server.py` — OVITO render server (results downloaded to local `run_folder`)
- All LangGraph generation nodes (`chat_node`, `extract_params_node`, `generate_lammps_node`, etc.)
- `run_manager.py` — reused as-is (`create_run_folder`, `save_lammps_script`)
- Existing `scripts_ready_for_review` SSE flow for local execution
- WebSocket trajectory streaming

---

## Prerequisites Before Using HPC

1. Endpoint UUIDs are already set in `config.py` (Mac local + LCRC Swing + Compute endpoint)
2. Install LangChain packages (added to `pixi.toml`):
   - `langchain-openai` — `ChatOpenAI` pointing at `config.API_BASE_URL`
   - `langchain-core` — message types (`HumanMessage`, `SystemMessage`, `AIMessage`, `ToolMessage`)
   - `langchain-mcp-adapters` — `MultiServerMCPClient`
3. Ensure `uvx` is available: `pip install uv` or via pixi
4. `uvx globus-mcp` must be runnable — it's pulled from PyPI automatically by uvx on first use
5. Authenticate with Globus once: `globus login` (credentials cached on disk for future runs)

---

## Status

- [x] `src/graph/state.py` — HPC fields added
- [x] `src/config.py` — Globus/HPC config vars added
- [x] `src/prompts.py` — `hpc_status` intent + `HPC_EXECUTOR_PROMPT` added
- [x] `src/graph/hpc_executor_node.py` — created (Phase 1 + Phase 2 + status)
- [x] `src/graph/routing.py` — HPC routing rules added
- [x] `src/graph/graph.py` — `hpc_executor_node` registered
- [x] `src/app.py` — `execution_target` in state + sessions endpoints
- [x] `index.html` — sessions sidebar + "Run on HPC" button + PBS approval card
