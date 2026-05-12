# FastAPI Migration — Detailed Phase Guide

Companion to `fastapi_migration_plan.md` (which is the high-level overview).
This document is the working reference when actually executing the migration —
task lists, code patterns, testing, and rollback per phase.

Written: 2026-05-11

---

## Pre-migration prep (1 day, before Phase 1)

### Task 0.1 — Add new dependencies to `pixi.toml`

Add these (don't remove old ones yet — Flask still runs in parallel):

```
fastapi >= 0.115.0
uvicorn[standard] >= 0.32.0
python-multipart >= 0.0.9         # for form data if needed
python-socketio >= 5.11.0         # for ASGI WebSocket (Phase 3)
aiosqlite >= 0.20.0               # for async SQLite (Phase 4)
```

Keep these for now — remove in Phase 5:
```
flask
flask-socketio
eventlet  (or whatever SocketIO backend is used)
```

After updating, run `pixi install` to fetch the new packages.

### Task 0.2 — Choose file structure

**Recommended:** Create a NEW file `src/app_fastapi.py` alongside the existing `src/app.py` (Flask). Don't touch `app.py` until Phase 5.

This means during migration, you have two app files:
- `src/app.py` — old Flask, port 2002
- `src/app_fastapi.py` — new FastAPI, port 2003

Both run independently. Browser can talk to either.

### Task 0.3 — Save baseline test cases

Before starting, document what currently works (manual checklist). After each phase, re-test affected items.

| # | Test |
|---|---|
| 1 | Submit local sim from chat |
| 2 | Submit HPC sim — Phase 1 PBS preview appears |
| 3 | Submit HPC sim — Phase 2 confirm submit works, jobs appear in checkpoint |
| 4 | HPC status check — agent lists jobs and replies |
| 5 | HPC transfer — files arrive locally |
| 6 | Analyst flow — "analyze run X" works |
| 7 | KIM model search by element |
| 8 | KIM model download with progress |
| 9 | Session list / switch |
| 10 | Feedback submission |
| 11 | Trajectory rendering during local sim |
| 12 | Trajectory replay from past run |

Print this checklist. After each phase, tick off what still works.

---

## Phase 1 — Foundation + simple endpoints (1-2 days)

### Goal
FastAPI runs on port 2003 alongside Flask on port 2002. Frontend stays on 2002 for now. FastAPI handles only simple endpoints — no LangGraph, no SSE.

### Tasks

#### 1.1 — Create `src/app_fastapi.py` with bootstrap

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os, sys, json, uuid, sqlite3, asyncio
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

app = FastAPI(title="ChatMD", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"))
```

#### 1.2 — Copy session DB helpers (sync, fine to keep)

```python
def _get_sessions_db():
    db_path = os.path.abspath(config.SESSION_DB_PATH)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_sessions_table():
    conn = _get_sessions_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT DEFAULT 'New Session',
            created_at  TEXT,
            updated_at  TEXT,
            messages    TEXT DEFAULT '[]'
        )
    """)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN messages TEXT DEFAULT '[]'")
    except Exception:
        pass
    conn.commit()
    conn.close()

# Run at startup:
_init_sessions_table()
```

#### 1.3 — Define Pydantic models

```python
class CreateSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str

class PatchSessionRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    messages: Optional[List[Dict[str, Any]]] = None

class FeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=1, max_length=10000)
    simulation_summary: str = Field("", max_length=10000)
    session_id: Optional[str] = None

class FeedbackResponse(BaseModel):
    category: str
    lesson: str
    keywords: str
    stored: bool

class KIMDownloadRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
```

#### 1.4 — Migrate session endpoints

```python
@app.get("/api/sessions")
async def list_sessions():
    conn = _get_sessions_db()
    rows = conn.execute(
        "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session():
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = _get_sessions_db()
    conn.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, 'New Session', now, now)
    )
    conn.commit()
    conn.close()
    return CreateSessionResponse(session_id=session_id, title="New Session", created_at=now)

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conn = _get_sessions_db()
    row = conn.execute(
        "SELECT session_id, title, created_at, updated_at, messages FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    return dict(row)

@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, body: PatchSessionRequest):
    now = datetime.utcnow().isoformat()
    conn = _get_sessions_db()
    if body.title is not None:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (body.title.strip(), now, session_id)
        )
    if body.messages is not None:
        conn.execute(
            "UPDATE sessions SET messages = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(body.messages), now, session_id)
        )
    conn.commit()
    conn.close()
    return {"session_id": session_id}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = _get_sessions_db()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"deleted": session_id}
```

#### 1.5 — Migrate feedback endpoint

```python
from agents import SmartOrchestrator

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    orchestrator = SmartOrchestrator()
    # Sync DSPy call wrapped in to_thread (Phase 4 makes this native async)
    result = await asyncio.to_thread(
        orchestrator.process_feedback,
        body.feedback,
        body.simulation_summary
    )
    return FeedbackResponse(**result)
```

#### 1.6 — Migrate KIM endpoints

```python
import kim_utils

@app.get("/api/kim-potentials/list")
async def list_kim_potentials():
    status = await asyncio.to_thread(kim_utils.get_registry_status)
    return {
        "success": True,
        "potentials": status.get('potentials', []),
        "stats": {
            "total": status.get('total_potentials', 0),
            "downloaded": status.get('downloaded_potentials', 0),
            "pending": status.get('not_downloaded', 0),
            "unique_elements": len(set(p['element'] for p in status.get('potentials', [])))
        }
    }

@app.post("/api/kim-potentials/download")
async def download_kim_potential(body: KIMDownloadRequest):
    success, message = await asyncio.to_thread(
        kim_utils.install_and_verify_kim_model, body.model_id, 180
    )
    if success:
        await asyncio.to_thread(kim_utils.update_registry, body.model_id, downloaded=True)
        return {"success": True, "message": message, "model_id": body.model_id}
    else:
        raise HTTPException(status_code=500, detail=message)

# Similar pattern for /search, /status, /init, /add
```

#### 1.7 — Run command

```bash
pixi run uvicorn src.app_fastapi:app --host 0.0.0.0 --port 2003 --reload
```

Old Flask continues on port 2002:
```bash
pixi run python src/app.py
```

### Testing
- Browser: open `http://localhost:2003/`
- ✅ UI loads
- ✅ Session list shows existing sessions
- ✅ Create new session works
- ✅ Delete session works
- ✅ KIM potentials panel loads
- ❌ Chat won't work yet (POST goes to FastAPI but FastAPI doesn't handle /api/llm-stream-live yet — expected)

### Rollback
Just stop uvicorn. Flask on 2002 untouched.

### Deliverable
FastAPI app serves UI + simple endpoints on port 2003. Chat still uses Flask on 2002.

### Commit
```
git commit -m "FastAPI Phase 1 — foundation + simple endpoints"
```

---

## Phase 2 — SSE streaming endpoint (2-3 days) — THE BIG ONE

### Goal
Migrate `POST /api/llm-stream-live` (the chat endpoint) to FastAPI. Chat works end-to-end through new server. **The recurring frontend hangs from SSE buffering should disappear.**

### Tasks

#### 2.1 — Define request body model

```python
class LLMStreamRequest(BaseModel):
    command: str
    history: List[Dict[str, Any]] = []
    ovito_script: Optional[str] = None
    pending_script: Optional[str] = None
    execution_target: str = "local"
    session_id: Optional[str] = None
    workflow_stage: Optional[str] = None
```

#### 2.2 — Async SSE event stream generator

```python
async def sse_event_stream(sse_queue: asyncio.Queue, request: Request):
    """Yield SSE events from queue. Stop on sentinel or client disconnect."""
    while True:
        # Detect client disconnect
        if await request.is_disconnected():
            print("[SSE] client disconnected, stopping stream")
            break

        try:
            event = await asyncio.wait_for(sse_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue  # check disconnect again

        if event is None:  # sentinel
            break

        yield f"data: {json.dumps(event)}\n\n"
```

#### 2.3 — Async run_graph

```python
from graph.graph import simulation_graph

async def run_graph_async(initial_state, sse_queue, session_id, history, body):
    """Run graph and push events. Sentinel pushed in finally."""
    try:
        # Snapshot prior hpc count
        prior_hpc_count = 0
        try:
            prior_state = await asyncio.to_thread(
                simulation_graph.get_state,
                {"configurable": {"thread_id": session_id}}
            )
            if prior_state and prior_state.values:
                prior_hpc_count = len(prior_state.values.get("hpc_jobs") or [])
        except Exception as e:
            print(f"[POST-GRAPH] get_state failed: {e}")

        print(f"[POST-GRAPH] calling simulation_graph.invoke()...")
        final = await asyncio.to_thread(
            simulation_graph.invoke,
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )
        print(f"[POST-GRAPH] invoke returned. intent={final.get('intent')!r} ...")

        # Run cascade
        await run_cascade(final, sse_queue, session_id, prior_hpc_count, history, body)

    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = format_exception_message(e)  # use _flatten helper
        await sse_queue.put({"type": "error", "message": f"An error occurred: {msg}"})
    finally:
        await sse_queue.put(None)  # sentinel
```

#### 2.4 — Migrate the cascade (port from app.py)

```python
async def run_cascade(final, sse_queue, session_id, prior_hpc_count, history, body):
    # Auto-title
    if final.get("material"):
        title_parts = [final["material"]]
        if final.get("temperature"):
            title_parts.append(f"{final['temperature']}K")
        new_title = " ".join(str(p) for p in title_parts)
        await asyncio.to_thread(_auto_title, session_id, new_title)

    # Branch 1: HPC submission completed
    new_hpc_count = len(final.get("hpc_jobs") or [])
    if new_hpc_count > prior_hpc_count:
        print(f"[CASCADE] branch=hpc_submission_complete")
        await sse_queue.put({"type": "workflow_complete", "session_id": session_id})
        return

    # Branch 2: Direct execute
    if final["workflow_stage"] == "script_ready" and final["intent"] == "execute":
        print(f"[CASCADE] branch=direct_execute")
        await sse_queue.put({"type": "agent_start", "agent": "System"})
        await sse_queue.put({"type": "agent_content", ...})
        await sse_queue.put({"type": "start_streaming_simulation", ...})
        await sse_queue.put({"type": "agent_complete", "agent": "System"})
        await sse_queue.put({"type": "workflow_complete", "session_id": session_id})
        return

    # ... all other branches ported similarly ...

    print(f"[CASCADE] emitting workflow_complete (final fallthrough)")
    await sse_queue.put({"type": "workflow_complete", "session_id": session_id})
```

#### 2.5 — The endpoint

```python
@app.post("/api/llm-stream-live")
async def llm_stream_live(request: Request, body: LLMStreamRequest):
    print(f"\n[USER MSG] {body.command!r}")
    print(f"          session_id={body.session_id!r}  history_len={len(body.history)}  ...")

    # Determine workflow stage
    workflow_stage = "initial"
    if body.workflow_stage == 'pbs_pending':
        workflow_stage = "pbs_pending"
    elif body.pending_script:
        workflow_stage = "script_ready"
    elif body.history:
        workflow_stage = "parameters_proposed"

    session_id = body.session_id or str(uuid.uuid4())
    sse_queue = asyncio.Queue()

    # Ensure session row exists
    await asyncio.to_thread(_ensure_session_row, session_id)

    # Build initial state
    initial_state = {
        "user_message":       body.command,
        "workflow_stage":     workflow_stage,
        "pending_script":     body.pending_script,
        "ovito_script_param": body.ovito_script,
        "execution_target":   body.execution_target,
        "session_id":         session_id,
        "intent":             None,
        "analysis":           None,
        "has_parameters":     None,
        "review_passed":      None,
        "review_issues":      None,
        "review_suggestions": None,
        "has_custom_viz":     None,
        "error":              None,
        **({"conversation_history": body.history} if body.history else {}),
    }

    # Spawn graph task
    asyncio.create_task(
        run_graph_async(initial_state, sse_queue, session_id, body.history, body)
    )

    return StreamingResponse(
        sse_event_stream(sse_queue, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
```

#### 2.6 — Update `_q()` helper in `src/graph/nodes.py`

Two options:

**Option A (smaller diff):** Use `put_nowait()` which works for both `queue.Queue` and `asyncio.Queue`:

```python
def _q(state, event):
    sse_q = SESSION_QUEUES.get(state.get("session_id"))
    if sse_q is not None:
        sse_q.put_nowait(event)  # works for either queue type
```

But also need to update what gets stored in `SESSION_QUEUES` — make it `asyncio.Queue` instances.

**Option B (cleaner):** Pass the queue via state, drop the global registry. Bigger refactor, defer to Phase 5.

#### 2.7 — Helper: `_flatten` for error messages

Copy from current `app.py` as-is. It's pure logic:

```python
def format_exception_message(e):
    def _flatten(exc, depth=0):
        if depth > 5 or exc is None:
            return []
        msgs = []
        inner = getattr(exc, "exceptions", None)
        if inner:
            for sub in inner:
                msgs.extend(_flatten(sub, depth + 1))
        else:
            text = f"{type(exc).__name__}: {exc}".strip()
            msgs.append(text or type(exc).__name__)
        if not inner and getattr(exc, "__cause__", None):
            msgs.extend(_flatten(exc.__cause__, depth + 1))
        return msgs

    parts = _flatten(e) or [repr(e)]
    return " | ".join(parts)
```

### Testing

Point browser at `http://localhost:2003/`. Then:
1. Submit local sim — should work end-to-end
2. Submit HPC sim — Phase 1 PBS preview should appear
3. Confirm HPC submit — should reach Globus
4. Status check — should work
5. Transfer back — should work
6. **Critical: try the patterns that hung in Flask** (analyst → orchestrator → "looks good"). Should NOT hang.
7. Type a message during a long-running turn — should be properly rejected (isStreaming check still works)

### Risks
- LangGraph thread safety: `asyncio.to_thread` runs on a thread pool, multiple concurrent requests may conflict on SqliteSaver. Test concurrent requests.
- Cancellation: ensure `asyncio.create_task(run_graph_async(...))` is cancelled when client disconnects. Detection via `request.is_disconnected()` in the SSE generator.
- `_q()` changes: verify all node code paths still emit events correctly.
- Cascade: verify each branch behaves identically to Flask version. Compare side-by-side.

### Rollback
If broken: continue using Flask on 2002. FastAPI on 2003 broken doesn't affect production.

### Deliverable
Chat works end-to-end through FastAPI on port 2003. SSE buffering issues should be resolved.

### Commit
```
git commit -m "FastAPI Phase 2 — SSE streaming migrated, chat works on port 2003"
```

---

## Phase 3 — WebSocket migration (1-2 days)

### Goal
Migrate WebSocket handlers (local sim trajectory streaming, KIM downloads, replay) to FastAPI ASGI.

### Tasks

#### 3.1 — Choose WebSocket library

**Recommended:** `python-socketio` in ASGI mode. Frontend code stays unchanged (still uses socket.io client).

#### 3.2 — Setup Socket.IO ASGI

```python
import socketio

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

# Run with: uvicorn src.app_fastapi:socket_app --port 2003
```

#### 3.3 — Migrate handlers

```python
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await sio.emit('status', {'message': 'Connected to ChatMD server'}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def start_streaming_simulation(sid, data):
    """Local LAMMPS sim with trajectory streaming."""
    lammps_script = data.get('script')
    ovito_script = data.get('ovito_script')
    dump_filename_pattern = data.get('dump_filename_pattern', '')
    target_timesteps = data.get('target_timesteps', None)
    camera_angle = data.get('camera_angle', 'default')

    if not lammps_script:
        await sio.emit('error', {'message': 'No LAMMPS script provided'}, to=sid)
        return

    await sio.emit('simulation_started', {'message': 'Starting...'}, to=sid)

    # Run LAMMPS in background task
    asyncio.create_task(run_simulation_task(sid, lammps_script, ovito_script, ...))

async def run_simulation_task(sid, lammps_script, ovito_script, ...):
    async def emit_trajectory_frame(data_point):
        if data_point.get('type') == 'trajectory_frame':
            await sio.emit('trajectory_frame', {...}, to=sid)
        elif data_point.get('type') == 'trajectory_complete':
            await sio.emit('trajectory_complete', {...}, to=sid)
        else:
            await sio.emit('thermo_data_point', data_point, to=sid)

    # Wrap sync data_callback for sync LAMMPS execution
    def sync_callback(data_point):
        asyncio.run_coroutine_threadsafe(
            emit_trajectory_frame(data_point),
            asyncio.get_event_loop()
        )

    try:
        result = await asyncio.to_thread(
            run_lammps_simulation,
            lammps_script,
            {'ovito_script': ovito_script, ...},
            sync_callback,
            num_cores=config.LAMMPS_CORES
        )
        await sio.emit('simulation_complete', {...}, to=sid)
    except Exception as e:
        await sio.emit('simulation_error', {'message': str(e)}, to=sid)

# Similar for replay_trajectory, download_kim_potential, etc.
```

#### 3.4 — Update run command

```bash
pixi run uvicorn src.app_fastapi:socket_app --host 0.0.0.0 --port 2003 --reload
```

### Testing
1. Local sim — trajectory frames stream live
2. Replay trajectory from past run
3. KIM download with progress updates

### Risks
- python-socketio ASGI integration with FastAPI sometimes has CORS quirks
- Sync callbacks from LAMMPS subprocess need careful event loop handling
- Verify socket.io client/server version compatibility

### Rollback
WebSocket calls fall back to Flask on 2002 if FastAPI WebSocket fails.

### Deliverable
All WebSocket flows work via FastAPI. Frontend code unchanged.

### Commit
```
git commit -m "FastAPI Phase 3 — WebSocket handlers migrated via python-socketio ASGI"
```

---

## Phase 4 — Async LangGraph + DSPy (OPTIONAL, 2-3 days)

### Goal
Native async throughout. Skip if you want to ship Phases 1+2+3 sooner.

### Tasks

#### 4.1 — Convert nodes to async

Each node in `src/graph/nodes.py`:

```python
# Before:
def chat_node(state):
    response = client.chat.completions.create(...)
    return {"analysis": ...}

# After:
async def chat_node(state):
    response = await client.chat.completions.acreate(...)
    return {"analysis": ...}
```

Most nodes just need `async def` + `await` on LLM calls.

#### 4.2 — Switch to ainvoke

In FastAPI handler:
```python
# Before:
final = await asyncio.to_thread(simulation_graph.invoke, ...)

# After:
final = await simulation_graph.ainvoke(initial_state, config=...)
```

#### 4.3 — Async DSPy

DSPy 3.x supports async. Convert:
```python
# Before:
result = orchestrator.classify_intent(...)

# After:
result = await orchestrator.aclassify_intent(...)  # if available
```

Verify DSPy version supports `aforward()`.

#### 4.4 — Wrap remaining sync subprocess calls

MCP, OVITO render server, LAMMPS execution — keep sync but wrap with `asyncio.to_thread()`:

```python
result = await asyncio.to_thread(some_sync_function, args)
```

#### 4.5 — Update SqliteSaver if possible

If `aiosqlite`-based SqliteSaver is available in LangGraph, switch. Otherwise keep sync version (it's fast enough).

### Testing
- All flows still work after async conversion
- Concurrent requests benefit from no thread overhead
- Performance: time per turn should be similar or slightly faster

### Risks
- DSPy async has edge cases (newer feature)
- Concurrent state access via SqliteSaver — need to test
- More refactoring than other phases

### Rollback
Revert nodes to sync versions. Use `asyncio.to_thread` wrapping from Phase 2.

### Deliverable
Native async pipeline. No thread pool overhead.

### Commit
```
git commit -m "FastAPI Phase 4 — async LangGraph + DSPy throughout"
```

---

## Phase 5 — Cleanup + testing + documentation (1-2 days)

### Goal
Production-ready FastAPI app. Old Flask code removed.

### Tasks

#### 5.1 — Final cutover
- Stop running Flask on 2002
- Move FastAPI from port 2003 to 2002
- Verify frontend works without changes (it should — same SSE format, same WebSocket protocol)

#### 5.2 — Delete old code
```bash
rm src/app.py
mv src/app_fastapi.py src/app.py
```

#### 5.3 — Update dependencies
Remove from `pixi.toml`:
- `flask`
- `flask-socketio`
- `eventlet` (or whatever SocketIO backend was)
- The `werkzeug` `allow_unsafe_werkzeug=True` workaround can go too

Run `pixi install` to update `pixi.lock`.

#### 5.4 — Improve logging
Replace `print()` calls with `logging` module:

```python
import logging
log = logging.getLogger(__name__)

# Instead of:
print(f"[POST-GRAPH] invoke returned. ...")
# Use:
log.debug(f"[POST-GRAPH] invoke returned. ...")
```

Configure at startup:
```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
```

You can later set `level=logging.INFO` to silence diagnostic output.

#### 5.5 — Update README
- Startup command: `pixi run uvicorn src.app:app --host 0.0.0.0 --port 2002`
- Or with WebSocket: `pixi run uvicorn src.app:socket_app --host 0.0.0.0 --port 2002`
- Add to setup instructions

#### 5.6 — Final end-to-end test
Run through all 12 baseline test scenarios from Task 0.3.

#### 5.7 — Performance comparison
Time a typical "looks good" turn under both Flask and FastAPI to verify no regression.

### Risks
- Renaming the file might break imports elsewhere — search and update
- `pixi install` might pick up new versions of things that break behavior

### Rollback
Keep `src/app.py.flask_backup` as a safety net for a few days, then delete.

### Deliverable
Clean FastAPI codebase. Documented. Tested. Old Flask gone.

### Commit
```
git commit -m "FastAPI Phase 5 — Flask removed, cleanup, README updated"
```

---

## Total time estimate

| Phase | Time | Cumulative |
|---|---|---|
| Pre-migration prep | 1 day | 1 |
| Phase 1 | 1-2 days | 2-3 |
| Phase 2 | 2-3 days | 4-6 |
| Phase 3 | 1-2 days | 5-8 |
| Phase 4 (optional) | 2-3 days | 7-11 |
| Phase 5 | 1-2 days | 8-13 |
| **Total (skip P4)** | | **5-9 days** |
| **Total (full)** | | **7-12 days** |

## What stays untouched throughout

- `src/graph/` — LangGraph nodes (Phase 4 makes them async, but structure stays)
- `src/prompts.py` — no changes
- `src/agents.py` — DSPy modules stay (Phase 4 makes some methods async)
- `index.html` — minimal/no changes (Phase 3 might tweak WebSocket URL, otherwise none)
- `src/tools.py` — LAMMPS execution unchanged
- `src/memory.py` — unchanged
- `src/kim_utils.py` — unchanged
- All pending fixes (#3, #4, #8) — apply to either codebase

## Recommended commit cadence

Commit at end of each phase to your `langgraph_hpc` branch (or a dedicated `fastapi_migration` branch). Each commit message:
- "FastAPI Phase N: <summary>"
- List what was migrated
- Note what still uses Flask (until Phase 5)

This gives you:
- Clean rollback points per phase
- Reviewable diffs
- Documentation of when things changed

## Decision points along the way

After Phase 2, ask yourself:
- Do I need Phase 4 (async agents)? If chat is fast enough on Phase 2 + sync agents + asyncio.to_thread wrapping, skip P4 and save 2-3 days.

After Phase 3, ask yourself:
- Is everything working end-to-end? If yes, plan a final cleanup day for Phase 5. If issues, fix before moving on.

## Risks to watch throughout

- **DSPy async support** — newer feature, may have edge cases
- **Globus MCP / langchain_mcp_adapters** — verify it works in async context
- **SqliteSaver** — sync; may need `aiosqlite` migration or careful wrapping
- **Frontend SSE protocol** — verify byte-for-byte the same SSE format as Flask
- **WebSocket via python-socketio ASGI** — version compatibility with frontend client

## When to stop

The migration is "done" when:
- All 12 baseline test scenarios pass on FastAPI
- Old Flask is deleted from the codebase
- README updated with new startup command
- Performance is equal or better than before
- A week of normal usage shows no regressions

---

## Quick reference table

| Phase | What you migrate | Frontend port | Test |
|---|---|---|---|
| 1 | Sessions, KIM, feedback | 2003 (FastAPI) | UI loads, lists work, KIM panel works |
| 2 | Chat (SSE streaming) | 2003 | Submit + status + transfer all work, no hangs |
| 3 | WebSocket (trajectory, KIM downloads) | 2003 | Trajectory frames stream live |
| 4 | (optional) Async agents | 2003 | Slightly faster turns |
| 5 | Final cutover, cleanup | 2002 (FastAPI) | All 12 baseline tests pass |
