"""
ChatMD FastAPI application.

Serves the chat UI, sessions, KIM potentials, feedback, and the LangGraph
simulation pipeline. Provides:
  - HTTP endpoints (FastAPI routes)
  - SSE streaming chat (POST /api/llm-stream-live)
  - WebSocket via python-socketio (local sim trajectory streaming, KIM downloads)

Run:
    pixi run run-app
    # or directly:
    pixi run uvicorn src.app:socket_app --host 0.0.0.0 --port 2002 --reload --reload-dir src
"""
import asyncio
import json
import os
import queue as _queue
import sqlite3
import sys
import threading
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Make sibling modules importable (config, agents, kim_utils, graph)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from agents import configure_dspy_lm
from graph.graph import simulation_graph
from graph.nodes import SESSION_QUEUES, CAMPAIGN_COMPLETION_EVENTS

# Configure DSPy LM once at module load (main thread). DSPy 3.x locks settings
# to the thread that first calls dspy.configure(), so this must happen here —
# NEVER inside request handlers or background threads.
configure_dspy_lm()


# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChatMD",
    version="1.0.0",
    description="ChatMD backend — chat, simulation orchestration, HPC integration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve index.html relative to project root (one level up from src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INDEX_HTML_PATH = os.path.join(PROJECT_ROOT, "index.html")


# ─── Pydantic models ─────────────────────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str


class PatchSessionRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    messages: Optional[List[Dict[str, Any]]] = None


class FeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=1, max_length=10_000)
    simulation_summary: str = Field("", max_length=10_000)
    session_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    category: str
    lesson: str
    keywords: str
    stored: bool


class KIMDownloadRequest(BaseModel):
    model_id: str = Field(..., min_length=1)


class KIMAddRequest(BaseModel):
    element: str = Field(..., min_length=1, max_length=10)
    model_id: str = Field(..., min_length=1)
    type: str = "Unknown"
    description: str = ""


# ─── Session DB helpers ──────────────────────────────────────────────────────

def _get_sessions_db() -> sqlite3.Connection:
    db_path = os.path.abspath(config.SESSION_DB_PATH)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sessions_table() -> None:
    conn = _get_sessions_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT DEFAULT 'New Session',
            created_at  TEXT,
            updated_at  TEXT,
            messages    TEXT DEFAULT '[]'
        )
        """
    )
    # Add messages column if table predates it (no-op if already present)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN messages TEXT DEFAULT '[]'")
    except Exception:
        pass
    conn.commit()
    conn.close()


@app.on_event("startup")
async def _on_startup() -> None:
    _init_sessions_table()
    print("[FastAPI] startup complete — sessions table initialized")


# ─── Static frontend ─────────────────────────────────────────────────────────

@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(INDEX_HTML_PATH)


# ─── Session endpoints ───────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions() -> List[Dict[str, Any]]:
    conn = _get_sessions_db()
    rows = conn.execute(
        "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = _get_sessions_db()
    conn.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, "New Session", now, now),
    )
    conn.commit()
    conn.close()
    return CreateSessionResponse(session_id=session_id, title="New Session", created_at=now)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    conn = _get_sessions_db()
    row = conn.execute(
        "SELECT session_id, title, created_at, updated_at, messages FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    return dict(row)


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, body: PatchSessionRequest) -> Dict[str, str]:
    now = datetime.utcnow().isoformat()
    conn = _get_sessions_db()
    if body.title is not None:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (body.title.strip(), now, session_id),
        )
    if body.messages is not None:
        conn.execute(
            "UPDATE sessions SET messages = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(body.messages), now, session_id),
        )
    conn.commit()
    conn.close()
    return {"session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    conn = _get_sessions_db()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    # Best-effort cleanup of LangGraph checkpoint rows for this thread.
    # These tables may not exist on a brand-new install; ignore failures.
    for stmt in (
        "DELETE FROM checkpoints WHERE thread_id = ?",
        "DELETE FROM writes WHERE thread_id = ?",
    ):
        try:
            conn.execute(stmt, (session_id,))
        except Exception:
            pass
    conn.commit()
    conn.close()
    return {"deleted": session_id}


# ─── Feedback endpoint ───────────────────────────────────────────────────────

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """
    Sync DSPy call wrapped with asyncio.to_thread so the event loop stays free.
    Phase 4 will switch to native async if DSPy supports it.
    """
    from agents import SmartOrchestrator

    orchestrator = SmartOrchestrator()
    result = await asyncio.to_thread(
        orchestrator.process_feedback, body.feedback, body.simulation_summary
    )
    return FeedbackResponse(
        category=result["category"],
        lesson=result["lesson"],
        keywords=result["keywords"],
        stored=result["stored"],
    )


# ─── KIM potentials endpoints ────────────────────────────────────────────────

@app.get("/api/kim-potentials/list")
async def list_kim_potentials() -> Dict[str, Any]:
    import kim_utils

    status = await asyncio.to_thread(kim_utils.get_registry_status)
    potentials = status.get("potentials", [])
    return {
        "success": True,
        "potentials": potentials,
        "stats": {
            "total": status.get("total_potentials", 0),
            "downloaded": status.get("downloaded_potentials", 0),
            "pending": status.get("not_downloaded", 0),
            "unique_elements": len({p["element"] for p in potentials}),
        },
    }


@app.post("/api/kim-potentials/download")
async def download_kim_potential(body: KIMDownloadRequest) -> Dict[str, Any]:
    import kim_utils

    success, message = await asyncio.to_thread(
        kim_utils.install_and_verify_kim_model, body.model_id, 180
    )
    if success:
        await asyncio.to_thread(kim_utils.update_registry, body.model_id, downloaded=True)
        return {"success": True, "message": message, "model_id": body.model_id}
    raise HTTPException(status_code=500, detail={"success": False, "message": message, "model_id": body.model_id})


@app.get("/api/kim-potentials/search")
async def search_kim_potentials(element: str) -> Dict[str, Any]:
    import kim_utils

    if not element:
        raise HTTPException(status_code=400, detail="element parameter is required")
    models = await asyncio.to_thread(kim_utils.query_openkim_models, element, 50)
    return {"element": element, "models": models, "count": len(models)}


@app.get("/api/kim-potentials/status")
async def kim_potentials_status(model_id: str) -> Dict[str, Any]:
    import kim_utils

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id parameter is required")
    installed = await asyncio.to_thread(kim_utils.is_model_installed, model_id)
    return {"model_id": model_id, "installed": installed}


@app.post("/api/kim-potentials/init")
async def initialize_kim_potentials() -> Dict[str, Any]:
    """
    Kick off recommended-potentials initialization in a background thread.
    Returns immediately; progress is currently not surfaced (Flask emitted
    progress via SocketIO; FastAPI's WebSocket migration is in Phase 3).
    """
    import kim_utils

    status = await asyncio.to_thread(kim_utils.get_registry_status)
    if status.get("downloaded_potentials", 0) > 0:
        return {"message": "Potentials already initialized", "status": status}

    def _init() -> None:
        try:
            kim_utils.initialize_recommended_potentials()
        except Exception as e:
            print(f"[FastAPI] KIM init error: {e}")

    threading.Thread(target=_init, daemon=True).start()
    return {
        "message": "Initialization started",
        "potentials": kim_utils.RECOMMENDED_POTENTIALS,
    }


@app.post("/api/kim-potentials/add")
async def add_kim_potential(body: KIMAddRequest) -> Dict[str, Any]:
    import kim_utils

    success = await asyncio.to_thread(
        kim_utils.add_potential_to_registry,
        body.element, body.model_id, body.type, body.description,
    )
    if success:
        return {"success": True, "message": f"Added {body.model_id} to registry"}
    raise HTTPException(status_code=500, detail="Failed to add potential to registry")


# ─── SSE streaming chat endpoint ─────────────────────────────────────────────
# Native FastAPI StreamingResponse for SSE. The graph runs sync (via
# asyncio.to_thread) so the event loop stays free. Graph nodes
# (src/graph/nodes.py) push events to a thread-safe queue.Queue; the SSE
# generator polls that queue from async context via asyncio.to_thread.

class LLMStreamRequest(BaseModel):
    command: str = Field(..., min_length=1)
    history: List[Dict[str, Any]] = []
    ovito_script: Optional[str] = None
    pending_script: Optional[str] = None
    pbs_script: Optional[str] = None
    execution_target: str = "local"
    session_id: Optional[str] = None
    workflow_stage: Optional[str] = None


def _flatten(exc: BaseException, depth: int = 0) -> List[str]:
    """Recursively unpack ExceptionGroup / __cause__ for a useful error message.
    Caps depth at 5 to avoid runaway recursion.
    """
    if depth > 5 or exc is None:
        return []
    msgs: List[str] = []
    inner = getattr(exc, "exceptions", None)  # ExceptionGroup
    if inner:
        for sub in inner:
            msgs.extend(_flatten(sub, depth + 1))
    else:
        text = f"{type(exc).__name__}: {exc}".strip()
        msgs.append(text or type(exc).__name__)
    if not inner and getattr(exc, "__cause__", None):
        msgs.extend(_flatten(exc.__cause__, depth + 1))
    return msgs


async def sse_event_stream(sse_q: _queue.Queue, request: Request):
    """Async SSE generator. Polls the sync queue.Queue via asyncio.to_thread
    so we don't block the event loop. Stops on sentinel (None) or client
    disconnect (whichever comes first)."""
    while True:
        if await request.is_disconnected():
            print("[SSE] client disconnected — stopping stream")
            break
        try:
            event = await asyncio.to_thread(sse_q.get, True, 1.0)
        except _queue.Empty:
            continue
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"


async def run_cascade(final: dict, sse_q: _queue.Queue, session_id: str,
                       prior_hpc_count: int, body: "LLMStreamRequest") -> None:
    """Post-graph cascade — decides which terminal SSE event(s) to emit based
    on the final state. Branches return early once a match fires; otherwise
    falls through to a plain workflow_complete."""

    # Auto-title session from first simulation parameters
    mat = final.get("material")
    temp = final.get("temperature")

    # Campaign mode: material lives inside campaign_template, not at top level
    if not mat and final.get("campaign_template"):
        mat = final["campaign_template"].get("material")

    if mat:
        title_parts = [mat]
        if temp:
            title_parts.append(f"{temp}K")
        if final.get("intent") == "campaign_start":
            title_parts.append("(campaign)")
        new_title = " ".join(str(p) for p in title_parts)
        _conn = _get_sessions_db()
        _conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ? AND title = 'New Session'",
            (new_title, datetime.utcnow().isoformat(), session_id),
        )
        _conn.commit()
        _conn.close()

    # Branch 1: HPC submission completed this turn (Fix #1)
    new_hpc_count = len(final.get("hpc_jobs") or [])
    if new_hpc_count > prior_hpc_count:
        print(f"[CASCADE] branch=hpc_submission_complete (hpc_jobs grew from {prior_hpc_count} to {new_hpc_count})")
        sse_q.put({"type": "workflow_complete", "session_id": session_id})
        return

    # Branch 2: Direct execution path — script already approved
    if final.get("workflow_stage") == "script_ready" and final.get("intent") == "execute":
        print("[CASCADE] branch=direct_execute (script_ready + execute)")
        sse_q.put({"type": "agent_start", "agent": "System"})
        sse_q.put({"type": "agent_content", "agent": "System",
                   "content": "Starting live streaming simulation...", "message_type": "info"})
        sse_q.put({"type": "start_streaming_simulation",
                   "script": final.get("pending_script"),
                   "ovito_script": final.get("ovito_script_param"),
                   "user_command": body.command})
        sse_q.put({"type": "agent_complete", "agent": "System"})
        sse_q.put({"type": "workflow_complete", "session_id": session_id})
        return

    # Branch 3: No simulation parameters extracted — ask for clarification
    if (not final.get("has_parameters") and final.get("analysis")
            and final.get("intent") not in ("analyze", "question", "cancel")):
        event_type = "needs_more_info" if body.history else "awaiting_confirmation"
        print(f"[CASCADE] branch={event_type} (no params + analysis present + intent not analyze/question/cancel)")
        sse_q.put({"type": event_type, "analysis": final["analysis"]})
        sse_q.put({"type": "workflow_complete", "session_id": session_id})
        return

    # Branch 4: LAMMPS review failed (Fix #2 — only fire on actual reviewer rejection)
    if (final.get("lammps_script")
            and final.get("review_passed") is False
            and final.get("review_issues")):
        print("[CASCADE] branch=review_failed (real failure: review_passed=False + issues present)")
        sse_q.put({"type": "review_failed",
                   "script": final["lammps_script"],
                   "issues": final.get("review_issues") or [],
                   "suggestions": final.get("review_suggestions") or []})
        sse_q.put({"type": "workflow_complete", "session_id": session_id})
        return

    # Branch 5: Default viz message when no custom viz
    if final.get("lammps_script") and final.get("review_passed") and not final.get("has_custom_viz"):
        print("[CASCADE] branch=default_viz (script + review_passed + not has_custom_viz)")
        sse_q.put({"type": "agent_content", "agent": "Visualization",
                   "content": "Applying default visualization settings for trajectory rendering.",
                   "message_type": "visualization"})
        sse_q.put({"type": "agent_complete", "agent": "Visualization"})

    # Frame sampling reads from dump files at runtime — no precompute needed
    target_timesteps = None

    # Branch 6: Scripts ready for user approval
    if final.get("lammps_script") and final.get("review_passed"):
        print("[CASCADE] branch=scripts_ready_for_review (script + review_passed)")
        sse_q.put({
            "type":                  "scripts_ready_for_review",
            "lammps_script":         final["lammps_script"],
            "ovito_script":          final.get("ovito_script"),
            "dump_filename_pattern": final.get("dump_filename_pattern", ""),
            "target_timesteps":      target_timesteps,
            "camera_angle":          final.get("camera_angle", "default"),
            "execution_target":      final.get("execution_target", "local"),
        })

    # Final fallthrough — always emit workflow_complete so frontend unlocks
    print("[CASCADE] emitting workflow_complete (final fallthrough)")
    sse_q.put({"type": "workflow_complete", "session_id": session_id})


async def run_graph_async(initial_state: dict, sse_q: _queue.Queue,
                           session_id: str, body: "LLMStreamRequest") -> None:
    """Run the LangGraph in a thread (sync invoke) and emit events via the queue.
    Always pushes a sentinel (None) in finally so the SSE generator exits cleanly,
    and removes the session_id from SESSION_QUEUES."""
    
    try:
        # Snapshot hpc_jobs count before invoke so the cascade can detect a fresh submit
        prior_hpc_count = 0
        try:
            prior_state = await asyncio.to_thread(
                simulation_graph.get_state,
                {"configurable": {"thread_id": session_id}},
            )
            if prior_state and prior_state.values:
                prior_hpc_count = len(prior_state.values.get("hpc_jobs") or [])
        except Exception as e:
            print(f"[POST-GRAPH] get_state failed (likely fresh session): {e}")

        print("[POST-GRAPH] calling simulation_graph.invoke()...")
        final = await asyncio.to_thread(
            simulation_graph.invoke,
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )
        print(f"[POST-GRAPH] invoke returned. "
              f"intent={final.get('intent')!r}  "
              f"workflow_stage={final.get('workflow_stage')!r}  "
              f"has_params={final.get('has_parameters')}  "
              f"review_passed={final.get('review_passed')}  "
              f"material={final.get('material')!r}  "
              f"has_script={'yes' if final.get('lammps_script') else 'no'}  "
              f"hpc_jobs_count={len(final.get('hpc_jobs') or [])}")

        await run_cascade(final, sse_q, session_id, prior_hpc_count, body)

        # ── Campaign mode: keep SSE open until background loop signals completion
        # The campaign_executor_node spawned a daemon thread that emits events into
        # SESSION_QUEUES via a DualQueue. Without this wait, the finally below
        # would close the SSE immediately and the frontend would miss every
        # campaign_iteration_* / campaign_complete event.
        if final.get("intent") == "campaign_start":
            print("[POST-GRAPH] campaign launched — keeping SSE open until campaign_complete")
            ev = _take_campaign_completion_event(session_id)
            if ev is not None:
                # Wait up to 3 hours; on signal, give SSE generator a grace period
                completed = await asyncio.to_thread(ev.wait, 3 * 3600)
                if completed:
                    await asyncio.sleep(0.5)  # let final SSE events flush
                    print("[POST-GRAPH] campaign_complete signaled — closing SSE")
                else:
                    print("[POST-GRAPH] campaign SSE wait timed out")

    except Exception as e:
        traceback.print_exc()
        parts = _flatten(e) or [repr(e)]
        msg = " | ".join(parts)
        sse_q.put({"type": "error", "message": f"An error occurred: {msg}"})
    finally:
        sse_q.put(None)  # sentinel — SSE generator stops
        SESSION_QUEUES.pop(session_id, None)


def _take_campaign_completion_event(chat_session_id: str):
    """Pop and return the completion event for this chat session, if registered.

    CAMPAIGN_COMPLETION_EVENTS is defined in graph.nodes (not here) so worker
    threads can access it without triggering app.py's top-level
    dspy.configure() side effect on a fresh import.
    """
    return CAMPAIGN_COMPLETION_EVENTS.pop(chat_session_id, None)


@app.post("/api/llm-stream-live")
async def llm_stream_live(request: Request, body: LLMStreamRequest) -> StreamingResponse:
    """Streaming chat endpoint — delegates to LangGraph simulation_graph.

    Returns an SSE stream of agent events as the graph executes. The graph
    runs sync (via asyncio.to_thread); event delivery is async via
    StreamingResponse. The frontend reads events through the EventSource /
    fetch ReadableStream API.
    """
    print(f"\n[USER MSG] {body.command!r}")
    print(f"          session_id={body.session_id!r}  history_len={len(body.history)}  "
          f"pending_script={'yes' if body.pending_script else 'no'}  "
          f"body_workflow_stage={body.workflow_stage!r}")

    # Determine workflow_stage from request body / hints
    workflow_stage = "initial"
    if body.workflow_stage == "pbs_pending":
        workflow_stage = "pbs_pending"
    elif body.pending_script:
        workflow_stage = "script_ready"
    elif body.history:
        workflow_stage = "parameters_proposed"

    session_id = body.session_id or str(uuid.uuid4())

    # Per-request queue.Queue — graph nodes call _q from a thread, so it must
    # be thread-safe. queue.Queue is. SSE generator polls it via asyncio.to_thread.
    sse_q: _queue.Queue = _queue.Queue()
    SESSION_QUEUES[session_id] = sse_q

    # Ensure session row exists in SQLite (upsert — no-op if already present)
    now = datetime.utcnow().isoformat()
    _conn = _get_sessions_db()
    _conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, "New Session", now, now),
    )
    _conn.commit()
    _conn.close()

    # Build initial state — same shape as Flask version. Generation outputs
    # (material, lammps_script, etc.) are intentionally omitted so LangGraph
    # restores them from the SqliteSaver checkpoint.
    initial_state: Dict[str, Any] = {
        "user_message":       body.command,
        "workflow_stage":     workflow_stage,
        "pending_script":     body.pending_script,
        "ovito_script_param": body.ovito_script,
        "pbs_script_edited":  body.pbs_script,
        "execution_target":   body.execution_target,
        "session_id":         session_id,
        # Per-turn transient fields — always reset each request
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

    # Spawn graph runner as background asyncio task. The task pushes events
    # to sse_q which the SSE generator drains. The task always pushes a
    # sentinel in its finally block, so the stream closes cleanly.
    asyncio.create_task(
        run_graph_async(initial_state, sse_q, session_id, body)
    )

    return StreamingResponse(
        sse_event_stream(sse_q, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind a proxy
        },
    )


# ─── WebSocket via python-socketio ASGI ──────────────────────────────────────
# Async Socket.IO server mounted alongside FastAPI. Handles local LAMMPS
# trajectory streaming, KIM downloads with progress, and replay events. The
# frontend uses the standard socket.io client and connects to /socket.io.
#
# Cross-thread emits: LAMMPS execution runs in a worker thread (via
# asyncio.to_thread). Its sync data_callback uses
# asyncio.run_coroutine_threadsafe(emit_coro, loop) to schedule emits on the
# main event loop from the worker thread.

import socketio  # noqa: E402

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")


@sio.event
async def connect(sid, environ):
    print(f"[Socket.IO] Client connected: {sid}")
    await sio.emit("status", {"message": "Connected to ChatMD server"}, to=sid)


@sio.event
async def disconnect(sid):
    print(f"[Socket.IO] Client disconnected: {sid}")


@sio.event
async def start_streaming_simulation(sid, data):
    """Local LAMMPS sim with live trajectory + thermo streaming."""
    from tools import run_lammps_simulation

    lammps_script         = data.get("script")
    ovito_script          = data.get("ovito_script")
    dump_filename_pattern = data.get("dump_filename_pattern", "")
    target_timesteps      = data.get("target_timesteps", None)
    camera_angle          = data.get("camera_angle", "default")

    if not lammps_script:
        await sio.emit("error", {"message": "No LAMMPS script provided"}, to=sid)
        return

    await sio.emit(
        "simulation_started",
        {"message": "Starting live streaming simulation with visualization..."},
        to=sid,
    )

    loop = asyncio.get_running_loop()

    def data_callback(data_point):
        """Called from the LAMMPS subprocess thread. Schedules an async
        sio.emit on the main event loop. Safe to call from any thread."""
        if not isinstance(data_point, dict):
            return
        kind = data_point.get("type")
        if kind == "trajectory_frame":
            asyncio.run_coroutine_threadsafe(
                sio.emit("trajectory_frame", {
                    "image_data":   data_point["image_data"],
                    "frame_number": data_point["frame_number"],
                    "timestep":     data_point["timestep"],
                }, to=sid),
                loop,
            )
        elif kind == "trajectory_complete":
            asyncio.run_coroutine_threadsafe(
                sio.emit("trajectory_complete", {
                    "total_frames": data_point["total_frames"],
                    "message":      data_point["message"],
                }, to=sid),
                loop,
            )
        else:
            # Thermodynamic data point
            asyncio.run_coroutine_threadsafe(
                sio.emit("thermo_data_point", data_point, to=sid),
                loop,
            )

    try:
        render_config = {
            "ovito_script":          ovito_script,
            "target_timesteps":      target_timesteps,
            "dump_filename_pattern": dump_filename_pattern,
            "camera_angle":          camera_angle,
        }
        print(f"[FastAPI] Render config: {'Custom OVITO script' if ovito_script else 'Default visualization'}")

        output_files, all_thermo_data, frame_images = await asyncio.to_thread(
            run_lammps_simulation,
            lammps_script,
            render_config,
            data_callback,
            num_cores=config.LAMMPS_CORES,
        )
        await sio.emit("simulation_complete", {
            "output_files":  output_files,
            "total_points":  len(all_thermo_data),
            "total_frames":  len(frame_images),
            "message":       f"Simulation completed successfully with {len(frame_images)} visualization frames!",
        }, to=sid)
    except Exception as e:
        await sio.emit("simulation_error", {"message": f"Simulation failed: {e}"}, to=sid)


@sio.event
async def replay_trajectory(sid, data):
    """Re-render dump files from a past run as trajectory_frame events."""
    from tools import (
        find_dump_file,
        extract_timestep_from_filename,
        _start_render_server_process,
        _send_render_request,
    )

    run_folder = data.get("run_folder", "")
    if not run_folder or not os.path.isdir(run_folder):
        await sio.emit("simulation_error", {"message": f"Run folder not found: {run_folder}"}, to=sid)
        return

    loop = asyncio.get_running_loop()

    def _do_replay():
        dump_info = find_dump_file(run_folder)
        if not dump_info or not dump_info["files"]:
            asyncio.run_coroutine_threadsafe(
                sio.emit("simulation_error", {"message": "No dump files found in run folder"}, to=sid),
                loop,
            )
            return

        pattern = dump_info.get("pattern", "")
        files = sorted(dump_info["files"],
                       key=lambda f: extract_timestep_from_filename(f, pattern))
        # Subsample to at most 50 frames
        if len(files) > 50:
            step = max(1, len(files) // 50)
            files = files[::step][:50]

        # Load saved OVITO script if present
        ovito_script_path = os.path.join(run_folder, "ovito_script.py")
        ovito_script = None
        if os.path.exists(ovito_script_path):
            with open(ovito_script_path) as _f:
                ovito_script = _f.read()
        print(f"[replay_trajectory] rendering {len(files)} frames from {run_folder} "
              f"({'custom viz' if ovito_script else 'default viz'})")

        proc = _start_render_server_process()
        try:
            for i, f in enumerate(files):
                timestep = extract_timestep_from_filename(f, pattern)
                result = _send_render_request(proc, {
                    "dump_file":        f,
                    "ovito_script":     ovito_script,
                    "file_frame_index": 0,
                    "camera_angle":     "default",
                    "render_size":      [600, 450],
                })
                if result.get("status") == "ok":
                    asyncio.run_coroutine_threadsafe(
                        sio.emit("trajectory_frame", {
                            "image_data":   result["image"],
                            "frame_number": i,
                            "timestep":     timestep,
                        }, to=sid),
                        loop,
                    )
                else:
                    print(f"[replay_trajectory] frame {i} error: {result.get('error')}")

            asyncio.run_coroutine_threadsafe(
                sio.emit("trajectory_complete", {
                    "total_frames": len(files),
                    "message":      f"Loaded {len(files)} frames from past run",
                }, to=sid),
                loop,
            )
        finally:
            try:
                _send_render_request(proc, {"type": "shutdown"})
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass

    await asyncio.to_thread(_do_replay)


@sio.event
async def download_kim_potential(sid, data):
    """Download a KIM potential via WebSocket with progress updates."""
    import kim_utils

    model_id = data.get("model_id")
    if not model_id:
        await sio.emit("kim_download_error", {"message": "model_id is required"}, to=sid)
        return

    await sio.emit("kim_download_started", {"model_id": model_id}, to=sid)

    try:
        success, message = await asyncio.to_thread(
            kim_utils.install_and_verify_kim_model, model_id, 180
        )
        if success:
            await asyncio.to_thread(kim_utils.update_registry, model_id, downloaded=True)
            await sio.emit("kim_download_complete", {
                "model_id": model_id,
                "success":  True,
                "message":  message,
            }, to=sid)
        else:
            await sio.emit("kim_download_error", {
                "model_id": model_id,
                "message":  message,
            }, to=sid)
    except Exception as e:
        await sio.emit("kim_download_error", {
            "model_id": model_id or "unknown",
            "message":  str(e),
        }, to=sid)


@sio.event
async def initialize_recommended_potentials(sid, data=None):
    """Initialize recommended KIM potentials with progress callbacks."""
    import kim_utils

    loop = asyncio.get_running_loop()

    def progress_callback(model_id, status_type, message):
        total = len(kim_utils.RECOMMENDED_POTENTIALS)
        try:
            current_index = kim_utils.RECOMMENDED_POTENTIALS.index(model_id) + 1
        except ValueError:
            current_index = 0
        progress = int((current_index / total) * 100) if total else 0
        asyncio.run_coroutine_threadsafe(
            sio.emit("kim_init_progress", {
                "status":         message,
                "current_model":  model_id,
                "progress":       progress,
            }, to=sid),
            loop,
        )

    try:
        results = await asyncio.to_thread(
            kim_utils.initialize_recommended_potentials, progress_callback
        )
        success_count = sum(results.values()) if isinstance(results, dict) else 0
        total = len(results) if isinstance(results, dict) else 0
        await sio.emit("kim_init_complete", {
            "results":       results,
            "success_count": success_count,
            "total_count":   total,
        }, to=sid)
    except Exception as e:
        await sio.emit("kim_download_error", {"message": f"Initialization failed: {e}"}, to=sid)


# Wrap FastAPI with the Socket.IO ASGI app. Run with:
#   uvicorn src.app:socket_app --host 0.0.0.0 --port 2002 --reload --reload-dir src
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
