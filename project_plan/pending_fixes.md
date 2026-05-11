# Pending Fixes

Issues identified but not yet fixed. Each entry includes root cause and proposed fix.

---

## Bug Fixes

### 1. Feedback submit button hangs
**File:** `src/app.py` — `/api/feedback` endpoint

**Root cause:** Calls `SmartOrchestrator().process_feedback()` which makes a synchronous DSPy LLM call to categorize and rephrase the feedback. If the LLM is busy or slow, the Flask request blocks and the button appears frozen.

**Fix:** Replace LLM categorization with keyword matching and store raw feedback directly:
- Check feedback text for viz keywords (`color`, `render`, `ovito`, `frame`, etc.) → `visualization_issue`
- Otherwise → `lammps_issue`
- Call `memory.add()` directly — no LLM call needed

---

### 2. KIM model re-asked every session
**File:** `src/prompts.py`, `src/graph/nodes.py`

**Root cause:** Two related issues:
1. `registry_context` from previous turn is saved in `SimulationState` but never injected into the next turn's messages — LLM starts fresh each turn with no knowledge of previous model selection
2. Even when `get_kim_models` returns a single ✅ model, the prompt says "ask user to select" — forces unnecessary interaction

**Fix:**
- Update prompt rule: if exactly one ✅ model exists for an element, auto-select it and inform the user — do not ask
- Only ask when multiple ✅ models exist or no ✅ models are available

---

### 3. LAMMPS crash error invisible to user
**File:** `src/tools.py` — `run_lammps_simulation()`

**Root cause:** LAMMPS prints the real error (e.g. "Cannot open KIM model", "Illegal fix command") to stdout. The thermo-reading loop consumes stdout line by line and discards lines it doesn't recognise as thermo data. By the time the crash is detected, the error lines are gone. Only the MPI abort message (useless) remains in stderr.

**Fix:** Buffer all stdout lines into a list alongside thermo parsing. On non-zero exit code, include the last 40 lines of stdout in the error message raised to the user.

---

### 4. Memory feedback loop broken for OVITO render errors
**File:** `src/ovito_render_server.py`, `src/agents.py`

**Root cause:** When a custom OVITO script fails at render time, `ovito_render_server.py` silently falls back to basic viz. The failure is logged to stderr but never written to `memory.json`. So `IterativeOVITOGeneration._store_failure()` is never called for runtime render errors — only review-time failures are stored.

**Fix:** When the render fallback triggers, send the error back in the JSON response with a `status: "warn"` field. In `tools.py`, check for this and call `memory.add()` with the error details so future OVITO generation avoids the same mistake.

---

## Dead Code Removal

### 5. `session_manager.py` — entire file unused
**Files:** `src/session_manager.py`, `src/app.py`

**Root cause:** Pre-LangGraph session persistence. Replaced entirely by LangGraph's `SqliteSaver` checkpointer. Three wrapper functions in `app.py` (`create_session`, `get_session`, `update_session`) are defined but never called anywhere.

**Fix:**
- Delete `src/session_manager.py`
- Remove `from session_manager import get_session_manager` from `app.py`
- Remove `create_session`, `get_session`, `update_session` functions from `app.py`

---

### 6. Dead functions in `run_manager.py`
**File:** `src/run_manager.py`

**Root cause:** `get_run_folder_summary()` and `get_latest_run()` are defined but never imported or called anywhere in the codebase.

**Fix:** Delete both functions from `run_manager.py`

---

## Architectural / Performance

### 7. `intent_node` + `extract_params_node` redundant
**File:** `src/graph/nodes.py`, `src/graph/graph.py`

**Root cause:** `chat_node` (Orchestrator) already understands intent and extracts parameters as part of its response. Then `intent_node` and `extract_params_node` each make separate DSPy LLM calls to re-classify and re-extract from the orchestrator's text output. This costs 2 extra LLM calls per turn.

**Fix:** Have `chat_node` return structured fields directly (via function calling or structured output) so downstream nodes can read from state without re-calling the LLM. Eliminates `intent_node` and `extract_params_node` entirely.

---

### 8. `detect_viz_node` is a separate LLM call
**File:** `src/graph/nodes.py`

**Root cause:** Visualization detection is a separate node that makes its own DSPy LLM call. This information could be extracted during `extract_params_node` at no extra cost since it's looking at the same user message.

**Fix:** Merge viz detection into `extract_params_node` — add `has_custom_viz` and `viz_request` as output fields of the parameter extraction step.

---

### 11. CNA structure fraction not tracked in thermo log for phase transition simulations
**Files:** `src/prompts.py` — `KnowledgeBase`, `src/agents.py` — `generator_knowledge`

**Root cause:** For melting/solidification simulations, the LAMMPS generator does not add CNA-based structure tracking to the thermo output. The analyst can only read temperature/energy from the log — it cannot quantify FCC fraction over time, melting onset, or recrystallization progress without it.

**Fix:**
- Add `KnowledgeBase.PHASE_TRANSITION_OUTPUT_REQUIREMENTS` constant to `prompts.py`:
  - `compute myCNA all cna/atom <cutoff>` (0.854 * a0)
  - `variable fcc_frac equal c_fcc_count/count(all)`
  - `variable liq_frac equal c_liq_count/count(all)`
  - Add `v_fcc_frac v_liq_frac` to `thermo_style`
- Include in `generator_knowledge` in `agents.py`
- Generator will apply it only for melting/heating/solidification simulations (same pattern as `CRACK_OUTPUT_REQUIREMENTS`)

**Benefit:** Analyst can precisely report melting onset, solidification progress, and recrystallization fraction from numbers — no vision model needed.

---

### 10. LAMMPS generator inconsistently uses single vs per-timestep dump files
**Files:** `src/prompts.py` — `GenerateLAMMPSScript.dump_filename_pattern`, `src/agents.py` — `generator_knowledge`

**Root cause:** The `dump_filename_pattern` output field description gives the LLM two valid examples: `dump.Cu.*.dat` (per-timestep) and `traj.lammpstrj` (single file). The LLM probabilistically picks either. Single-file mode breaks `replay_trajectory` in `app.py` which always passes `file_frame_index=0` — only frame 0 would render.

**Fix:**
- Change `dump_filename_pattern` desc to mandate per-timestep `*.dat` format only — never single concatenated files
- Add a rule to `generator_knowledge` in `agents.py`: `dump` command must always use `*` wildcard in the filename (e.g. `dump.Al_melting.*.dat`)
- If single-file replay support is needed, detect `dump_info['mode'] == 'single'` in `run_replay()` and loop `file_frame_index=0,1,...,N-1` against the same file

---

---

## Planned Features

### Session Chat History Restore
**Files:** `src/app.py`, `index.html`

**Problem:** Clicking a session in the sidebar shows a blank chat box. LangGraph memory is intact (LLM picks up where it left off) but the rendered messages are not restored visually.

**Fix:**
- Add `messages TEXT DEFAULT '[]'` column to the `sessions` SQLite table
- `GET /api/sessions/<id>` — return single session including saved messages
- Extend `PATCH /api/sessions/<id>` to accept a `messages` field
- Frontend: maintain `chatMessages = []` array, push every user + agent message into it
- After `workflow_complete` → `PATCH /api/sessions/<session_id>` with `{messages: chatMessages}`
- In `switchSession()` → `GET /api/sessions/<session_id>` → re-render saved messages into chat box

**Notes:** Messages re-appear instantly (no re-streaming). Live trajectory frames and thermo plots do not restore. ~50 lines total across both files.

---

## Implemented Features

### Editable Script Panel
**File:** `index.html` — `showScriptReviewPrompt()`, `approveAndRun()`, new `toggleScriptEdit()`

**What it does:** After the LLM generates scripts and the review card appears, the user can click **Edit Scripts ▼** to expand editable textareas — one for the LAMMPS script (green), one for the OVITO script (purple, only shown if generated). Clicking **Approve & Run** reads the textarea content and runs it directly with no LLM call. If the user never opens the panel, the original generated scripts run as before.

**Two flows coexist:**
- Chat-based change → orchestrator → full LLM regeneration (unchanged)
- Direct textarea edit → Approve & Run → LAMMPS runs immediately (no LLM call)

---

## Planned Features

### HPC Execution via Globus MCP
**Files to create:** `src/graph/hpc_executor_node.py`
**Files to modify:** `src/graph/graph.py`, `src/graph/routing.py`, `src/graph/state.py`, `index.html`

**Design:**
- Add `execution_target: "local" | "hpc"` to `SimulationState`
- UI toggle on the review card — "Run Locally / Run on HPC" — sets this before Approve
- After script approval, `route_after_approval` routes to either existing local executor or new `hpc_executor_node`
- `hpc_executor_node` is a ReAct agent with Globus MCP tools:
  1. `globus transfer` — upload LAMMPS script + inputs to HPC endpoint
  2. `globus compute` — submit LAMMPS job, receive job ID
  3. Poll job status (`squeue` via Globus Compute)
  4. `globus transfer` back — pull log file (for thermo streaming) and dump files (for OVITO replay)
  5. Hand dump files to existing OVITO render server — no change there
- Agent streams progress to user via SSE: "Job submitted: ID 12345", "Transferring results..."

**What stays untouched:** all LangGraph generation nodes, OVITO render server, local execution path, SSE/WebSocket streaming

---

### FastAPI + Async Migration
**Files to change:** `src/app.py` (full rewrite), `pixi.toml` (add fastapi, uvicorn, python-socketio)

**Why migrate from Flask to FastAPI:**

**1. No more global SESSION_QUEUES hack**
- Current: `SESSION_QUEUES: Dict[str, queue.Queue]` — a global registry needed because `queue.Queue` can't go inside LangGraph state (not serializable by SqliteSaver)
- FastAPI: each request owns its own `asyncio.Queue` in local scope — no global registry, no session ID lookup, no cleanup needed

**2. Native async SSE — no background threads**
- Current: LangGraph runs in a `threading.Thread`, puts events into a `threading.Queue`, Flask generator reads it blocking — GIL-limited, fragile
- FastAPI: `simulation_graph.ainvoke()` runs as an `asyncio` coroutine, puts events into `asyncio.Queue`, `StreamingResponse` async generator reads it — all on one event loop, no threads

**3. Native WebSocket support**
- Current: `flask-socketio` + `eventlet`/`gevent` monkey-patching — known to conflict with stdlib threading and cause subtle bugs
- FastAPI: built-in `WebSocket` via Starlette — no monkey-patching, clean async handlers

**4. LangGraph `ainvoke()` compatibility**
- LangGraph natively supports `ainvoke()` for async execution — currently we call synchronous `invoke()` inside a thread, losing async benefits of LangGraph internals
- FastAPI lets us call `ainvoke()` directly from the async route handler

**5. DSPy async support**
- DSPy supports async LM calls — currently blocked because Flask is synchronous
- FastAPI unlocks parallel DSPy calls (e.g. generate + review simultaneously) via `asyncio.gather()`

**6. Better concurrency for multiple users**
- Current: each user request spawns a thread — thread pool exhaustion under load
- FastAPI: async event loop handles many concurrent requests without spawning threads

**7. Cleaner code structure**
- Current `app.py`: 400+ lines mixing routing, threading, queue management, SSE formatting
- FastAPI: routes are thin async functions, queue logic disappears, SSE is a `StreamingResponse`

**Migration scope:**
- `src/app.py` — full rewrite as FastAPI app with `uvicorn`
- All LangGraph nodes — change `invoke()` → `ainvoke()`, `_q()` helper stays the same
- WebSocket handlers — rewrite using Starlette `WebSocket`
- `pixi.toml` — replace `flask`, `flask-socketio`, `eventlet` with `fastapi`, `uvicorn`, `python-socketio`
- Frontend `index.html` — no changes needed (SSE and WebSocket protocols unchanged)

**What stays untouched:** all LangGraph graph/nodes/routing, OVITO render server, tools.py, prompts.py, agents.py

---

### 9. `SmartOrchestrator` partially dead in `agents.py`
**File:** `src/agents.py`

**Root cause:** `SmartOrchestrator` was the pre-LangGraph orchestration class. After migration, the graph uses its own nodes. Several DSPy modules inside it are now dead: `self.analyzer`, `self.tool_selector`, `self.intent_classifier`. Only `self.param_extractor`, `self.viz_detector`, and `self.feedback_extractor` are still called.

**Fix:** Remove dead modules from `SmartOrchestrator` or refactor into standalone functions. If feedback endpoint is simplified (fix #1), `self.feedback_extractor` also becomes dead.

---

## HPC Workflow Fixes (May 10 2026 session)

End-to-end audit and fixes for the HPC submission, status check, and transfer flows. Goal: stop the "graph hangs after submit" symptom and fix the multi-turn agent conversation routing.

### Completed (this session)

#### A. HPC submission terminal event — Fix #1 ✅
**Files:** `src/app.py`

**Problem:** After a successful HPC submit, the post-graph cascade in `app.py` would emit a misleading `review_failed` event because `lammps_script` persists in the SQLite checkpoint and `review_passed` was reset to None by the per-turn initial_state. UI showed "Script review failed" for a job that actually succeeded.

**Fix:** In `run_graph()`, snapshot `prior_hpc_count` before `simulation_graph.invoke()`. After invoke, if `len(final["hpc_jobs"]) > prior_hpc_count`, emit `workflow_complete` and return early — bypassing the broken downstream cascade. No new SSE events, no frontend cards (intentionally minimal).

#### B. Recursive error-message flatten — bonus
**Files:** `src/app.py`

**Problem:** `BaseExceptionGroup` from asyncio TaskGroups was rendered to the user as `"unhandled errors in a TaskGroup (1 sub-exception)"` — useless. Real error (e.g., Globus 404, MCP broken pipe) was buried 5 levels deep inside `.exceptions`.

**Fix:** Added `_flatten` helper in the `except` block that recursively walks `.exceptions` (ExceptionGroup) and `.__cause__` chains up to depth 5 and joins all readable type+message strings.

#### C. Multi-turn agent conversation routing — Fix #6 (replaced) ✅
**Files:** `index.html` (sendMessage SSE loop + addToHistory call), `src/prompts.py` (ClassifyUserIntent docstring)

**Problem:** When the HPC agent asked "which job?" and the user replied "2", the next turn went through `intent_node`, but DSPy classifier had no signal that the previous turn was an HPC clarifying question — it would default to `question` or `new_simulation` and route to the orchestrator.

**Root cause #1:** `accumulatedContent` filter in `sendMessage` only included `Orchestrator` and `Analyst` agent_content events — `HPC` was dropped, so HPC replies never made it into `conversationHistory`.

**Root cause #2:** Even when included, assistant messages were stored as plain `{role: "assistant", content: ...}` with no agent identity tag.

**Fix:**
- `sendMessage` now also accumulates HPC content and tracks `lastAssistantAgent`
- Before pushing to history, prefix the content with `[Orchestrator] ` / `[HPC] ` / `[Analyst] `
- `ClassifyUserIntent` docstring updated with a "MULTI-TURN AGENT CONVERSATIONS" section that tells DSPy to route follow-up replies to the agent that asked the question, but to follow the user if they change topic mid-clarification

**Verified:** `intent=hpc_status confidence=0.97` for "transfer the result to local machine" after the HPC agent's clarifying question.

#### D. Graceful tool error handling + agent-driven job picker — Fix #9+#10 ✅
**Files:** `src/graph/hpc_executor_node.py` (ToolNode wrap), `src/prompts.py` (HPC_EXECUTOR_PROMPT)

**Problem:** When a Globus MCP tool call failed (e.g., 404 because remote folder was deleted), the `ToolException` propagated up through `create_react_agent` and crashed the entire TaskGroup. The whole status check died on a single bad job.

**Plus:** When the user said "check my jobs" with multiple jobs in state, the agent would try to ls all of them at once — guaranteeing a crash if any one had a stale folder.

**Fix:**
- Wrapped tools with `ToolNode(tools, handle_tool_errors=True)` for both submit and status agent calls — failures become ToolMessages the agent reads and responds to conversationally
- Updated `HPC_EXECUTOR_PROMPT` to tell the agent: if the user is vague AND there's >1 job, list them and ask which one — do NOT call any tool; if a tool returns an error, report conversationally and don't retry blindly

#### E. Transfer progress acknowledgement — bonus
**Files:** `src/prompts.py`

**Problem:** When transferring results back, the agent would call `globus_transfer_data`, then silently wait minutes for completion before responding. Felt like a hang.

**Fix:** Updated TRANSFERRING RESULTS BACK section in `HPC_EXECUTOR_PROMPT` to instruct the agent to immediately acknowledge the transfer (with task ID + source/dest paths) before waiting for completion, then report the final result.

#### F. Early-exit on task_id detection — Fix #7 (reopened) ✅
**Files:** `src/graph/hpc_executor_node.py` — `_stream_agent_step`

**Problem:** Even with the right prompt ("get the result once and stop"), the agent would loop 20+ times after qsub success — calling list_directory, get_task_status, etc. to "verify" — eventually hitting LangGraph's recursion limit (10007) or running for minutes. Graph never returned, frontend stayed locked.

**Fix:** Added `exit_when_task_id_found` opt-in parameter to `_stream_agent_step`. When True, the loop breaks on either:
- A ToolMessage parseable as JSON with a `task_id`/`globus_task_id`/`job_id` key, OR
- An AIMessage text containing both "task id" AND ("pbs" OR "job id") substrings

The qsub call site (`_run_hpc_submit_async` step 3) is the only caller that opts in. Other steps (mkdir, transfer, status) keep their default behavior — agent self-terminates as before.

**Detection method:** Plain Python substring matching (`"task id" in text.lower()`). No regex, no JSON parsing of agent text — keywords were chosen to match the labels the agent's prompt explicitly tells it to use.

#### G. Listing prompt cleanup — bonus
**Files:** `src/prompts.py`

**Problem:** Job listing showed `"task ID unknown"` and `"PBS pending"` for every job — confusing UI noise.

**Fix:** Removed Globus task ID and PBS Job ID from the user-facing listing format. Now shows just description + run folder name + date. Removed "all" as a valid identifier (risky — checking every job can fail on stale folders). Asks user to identify by NUMBER or RUN FOLDER NAME — never description alone (could be confused with a new simulation request).

---

### Pending (this session, not done)

#### Fix #2: Tighten `review_failed` cascade condition
**File:** `src/app.py`

**Problem:** Even after Fix #1's HPC early-return, the `review_failed` branch in the cascade still misfires for question/analyze/cancel turns. `lammps_script` is in the checkpoint, `review_passed` is reset to None each turn → branch fires → user sees "Script review failed" status for turns where no script was ever generated.

**Fix:** Change condition from `not final.get("review_passed")` to `final.get("review_passed") is False AND final.get("review_issues")` — distinguishes "reviewer rejected" from "no review ran this turn."

#### Fix #3: Require explicit `confirm_execute` flag for pendingScript bypass
**Files:** `src/app.py` (line 310-311), `index.html` (Approve button handlers)

**Problem:** Frontend's `pendingScript` lingers in JS memory across some flow paths (HPC submit completion, cancel, error). When the user types ANY message while it's set, the body POSTs `pending_script: <script>`, which forces `workflow_stage="script_ready"`. If DSPy classifies the user's message as "execute", the bypass fires the simulation silently with the stale script.

**Fix:** Backend requires both `pending_script` AND `confirm_execute: true` in the request body to set `workflow_stage="script_ready"`. Frontend's Approve buttons set `confirm_execute: true`; sendMessage from chat input does not.

#### Fix #4: Guard against empty `hpc_run_folder` before submit
**File:** `src/graph/hpc_executor_node.py` (Phase 2 branch ~line 295)

**Problem:** If Flask restarts between PBS preview (Phase 1) and Confirm Submit (Phase 2), `hpc_run_folder_pending` may be empty/None. Code does `os.path.basename("")` → `""` → submits to `mkdir -p ""` → silently fails on remote with no clear error.

**Fix:** Add a guard at the top of Phase 2: if `run_folder` is empty or doesn't exist as a local directory, emit a user-facing error and reset `workflow_stage="initial"` instead of attempting submit.

#### Fix #8: Retry MCP stdio session on broken pipe
**File:** `src/graph/hpc_executor_node.py`

**Problem:** First HPC operation after long idle (~1 hour) fails with broken-pipe because the `globus-mcp` uvx subprocess died from inactivity. No retry.

**Fix:** Wrap `MultiServerMCPClient.session("globus")` in a small retry loop (max 2 attempts) with a fresh client on each retry. Log the first failure but don't surface to user unless second attempt also fails.

---

### Deferred (intentional, not blocking)

#### Fix #5: Extract + store real `globus_task_id` and `pbs_job_id` in `hpc_jobs`
**File:** `src/graph/hpc_executor_node.py`

**What it would do:** Parse the agent's text reply for "Globus Compute Task ID:" and "PBS Job ID:" labels, extract the values, store both in `hpc_jobs[i]` instead of the current `"unknown"` fallback for `globus_task_id` and missing `pbs_job_id` field entirely.

**Why deferred:** Current status checks and transfer-back flows work via `hpc_run_folder` (which is always real) — they don't need either ID. The IDs are only useful for:
- Cancelling a running job via `qdel`
- Looking up a task in the Globus web dashboard
- Querying task-level Globus Compute events

None of those features exist in the UI today. When they're added, implement Fix #5 to populate the IDs.

**Implementation sketch (when needed):** Add `_extract_after_label(text, label)` helper. In post-loop scan of `final_messages`, also iterate AIMessages and extract values after "Globus Compute Task ID:" and "PBS Job ID:" labels. Return tuple `(globus_task_id, pbs_job_id)` from `_run_hpc_submit_async`. Add `pbs_job_id` field to `new_job` dict.

---

## May 11 2026 session — additional fixes

Continuation of HPC + multi-turn debugging. Goal: make agent transitions reliable and frontend never permanently lock.

### Completed (this session)

#### A. Tighter classifier multi-turn rule
**File:** `src/prompts.py` — `ClassifyUserIntent` docstring

Routing-by-prior-agent now requires BOTH:
- (a) Last assistant message ends with a clarifying question (text ends with "?", or contains phrases like "which one", "which job", "please specify")
- (b) User reply is a SHORT answer (number, "yes", "the Cu one", folder name)

Otherwise classify based on user's new message. Added explicit DO/DON'T examples for each agent.

#### B. HPC off-topic escape hatch
**File:** `src/prompts.py` — `HPC_EXECUTOR_PROMPT`

Added `HANDLING OFF-TOPIC REQUESTS` as the FIRST rule. Agent recognizes non-HPC questions (e.g. "what is your name", "what is NPT?") and replies with a fixed redirect message instead of calling tools. Defense in depth Layer 2 — backstops the classifier when it misroutes.

#### C. HPC listing format improvements
**File:** `src/prompts.py` — `HPC_EXECUTOR_PROMPT`

- Reverse chronological order (newest at top, oldest at bottom)
- Explicit numbered prefixes "1.", "2.", "3." required on each job line
- Two-step pattern for numeric replies: agent acknowledges the mapping ("Got it — checking job 8: run_xxx") before calling tools
- Transfer requires the user to provide the EXPLICIT run folder name (not number, not description). Numbers acceptable for status check, NOT for transfer — eliminates ambiguity about which folder gets transferred

#### D. Inject `conversation_history` into HPC status agent
**File:** `src/graph/hpc_executor_node.py` — Branch C (status query)

**Problem:** HPC status agent received only `user_message` + jobs list each turn. Had no memory of prior turns. So when user said "yes" after agent asked "would you like to transfer?", agent had no context to interpret "yes".

**Fix:** Build `history_block` from last 6 messages (3 user/assistant pairs) and prepend to `user_msg`. Cap at 400 chars per message to keep prompt bounded. HPC agent now sees prior conversation and can reason about multi-turn replies. Status branch only — submit branch unchanged.

#### E. Frontend `isStreaming` safety net
**File:** `index.html` — `sendMessage` and `_hpcStream` finally blocks

**Problem:** If backend stream closes without firing `workflow_complete` (or any other reset event), `isStreaming` stays `true` forever. User can't type, must hard-refresh.

**Fix:** Added `isStreaming = false;` to the `finally` block of both `sendMessage` and `_hpcStream`. Guarantees the input box always unlocks after the stream ends, regardless of which (or no) SSE events fired. Catches: SSE delivery issues, dropped events, backend exceptions, race conditions.

#### F. Diagnostic prints in `app.py` for visibility
**File:** `src/app.py` — `llm_stream_live` + `run_graph`

Added `print()` statements at key points:
- `[USER MSG]` — when each user message arrives at backend (with session_id, history_len, pending_script status)
- `[POST-GRAPH] calling simulation_graph.invoke()...` — before the graph runs
- `[POST-GRAPH] invoke returned. ...` — after graph returns, with full state summary (intent, workflow_stage, has_params, review_passed, material, has_script, hpc_jobs_count)
- `[CASCADE] branch=...` — for each cascade branch that fires, with the matched condition

Now the terminal output reveals exactly which step hangs (or doesn't). Diagnostic-only, no behavior change.

#### G. Fix #2 — Tighten `review_failed` cascade condition
**File:** `src/app.py` — post-graph cascade

**Problem (root cause of analyst/HPC-then-orchestrator hang):** The `review_failed` cascade branch fired on EVERY non-generation turn (analyst, hpc_status, question), because:
- `lammps_script` persists in checkpoint from any earlier `generate_lammps` run
- `review_passed` is reset to `None` at the start of each turn
- `not None` evaluates to `True` in Python
- So the condition `lammps_script AND not review_passed` was True for any turn AFTER the first successful sim

This emitted misleading `review_failed` events on turns where no review actually happened, leading to broken downstream behavior (frontend got confused, eventually couldn't send next message).

**Fix:** Tightened the condition:
```python
# Before:
if final.get("lammps_script") and not final.get("review_passed"):
# After:
if (final.get("lammps_script")
        and final.get("review_passed") is False     # explicit False, not None
        and final.get("review_issues")):            # populated issues
```

`is False` distinguishes "reviewer ran and rejected" from "no review ran this turn". The `review_issues` check is belt-and-suspenders — real failures always have populated issues.

**Verified:** This fix resolved the "analyst/HPC turn breaks subsequent orchestrator turns" pattern that was the major remaining blocker.

---

### Still pending after May 11

| # | Fix | Notes |
|---|---|---|
| 3 | Explicit `confirm_execute` flag for pendingScript bypass | Prevents accidental sim re-runs from stale frontend state |
| 4 | Guard empty `hpc_run_folder` before submit | HPC submit safety after server restart between PBS preview + confirm |
| 5 (deferred) | Real `globus_task_id` + `pbs_job_id` extraction | Skip until cancel-job or task-detail features needed |
| 8 | MCP stdio retry on broken pipe | First HPC operation after long idle (~1 hour) sometimes fails |
