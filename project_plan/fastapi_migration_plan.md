# FastAPI Migration Plan

Captures the current Flask-based architecture's pain points, why FastAPI fixes them, and a phased plan to migrate.

Written: 2026-05-11

---

## Current problems we're facing

These are the recurring issues we've been chasing across multiple debugging sessions. All of them root back to the Flask + threading + SSE architecture, not the LangGraph or agent code.

### 1. Frontend permanently locks ("frontend not responding")

**Symptom:** User types a message, hits Enter, nothing happens. Input box appears unresponsive. Must hard-refresh the page or run `isStreaming = false` in DevTools console to recover.

**Root cause:** The frontend's `sendMessage` function sets a `isStreaming = true` flag at the start of each turn. It only resets to `false` when specific SSE events arrive (`workflow_complete`, `awaiting_confirmation`, etc.). If the SSE stream from the backend doesn't deliver these events properly OR closes without sending them, `isStreaming` stays `true` forever — and `sendMessage` silently drops every subsequent message.

**Why it happens with Flask + Werkzeug:**
- Flask's threaded model uses `queue.Queue` to pass events from the LangGraph background thread to the SSE generator
- Werkzeug's WSGI server has known SSE buffering issues — events queued for delivery may not flush to the client immediately
- The synchronous generator yields events but Flask/Werkzeug doesn't always close the response cleanly
- The frontend's `reader.read()` loop never gets `done=true`, so the `finally` block (with our newly-added safety net) never runs

**What we tried:**
- Frontend safety net (`isStreaming = false` in `finally` block) — helps when stream closes cleanly, doesn't help when stream physically stays open
- Diagnostic prints (`[CASCADE]` etc.) — confirmed cascade fires correctly but events don't always reach frontend
- Manual workaround (`isStreaming = false` in DevTools) — works but requires user intervention

### 2. SESSION_QUEUES global hack

**Current code:** A global dict `SESSION_QUEUES: Dict[str, queue.Queue]` maps each session to its SSE event queue. The graph nodes look up the queue via `session_id` to push events.

**Why it exists:** The queue can't go inside LangGraph state (not JSON-serializable by SqliteSaver). So we use a global registry as a side-channel.

**Problems:**
- Global mutable state is fragile across threads
- Cleanup happens in `finally` but races can leave stale entries
- If a request fires for the same `session_id` while a previous turn is still running, the queue gets overwritten — old worker's events leak into the new request's stream

### 3. Background threads + queue.Queue lifecycle bugs

**Current code:** Each request spawns a `threading.Thread` to run the graph in the background while the SSE generator pulls from a `queue.Queue`.

**Problems:**
- GIL serializes work — true parallelism not possible
- Thread cleanup depends on `finally` blocks running correctly
- If the client disconnects mid-stream, the background thread keeps running and pushing events to a queue nobody's draining
- Thread pool exhaustion possible under concurrent load
- Cancellation is awkward (Python threads can't be killed cleanly)

### 4. Flask-SocketIO + eventlet/monkey-patching conflicts

**Current code:** Flask-SocketIO (with eventlet or gevent monkey-patching) handles WebSocket for local LAMMPS simulation streaming.

**Problems:**
- Monkey-patching the standard library is known to cause subtle bugs with stdlib threading, subprocess, etc.
- We've had issues like the Werkzeug WSGI compatibility patch (`allow_unsafe_werkzeug=True`)
- WebSocket and SSE can interfere with each other
- Hard to debug

### 5. LangGraph runs synchronously in a thread

**Current code:** `simulation_graph.invoke(initial_state, ...)` runs synchronously inside the background thread.

**Problems:**
- LangGraph natively supports `ainvoke()` for async execution — we're not using it
- Inside nodes, sync DSPy calls block until completion
- No native parallelism between agents (can't generate + review concurrently)
- Wrapping in a thread loses the async benefits LangGraph offers

### 6. DSPy async support unused

**Current code:** All DSPy `Module.forward()` calls are synchronous.

**Problems:**
- DSPy 3.x supports async (`aforward()`) — could enable parallel LLM calls
- Currently every step (intent → chat → extract → generate → review) is sequential
- A single user turn that requires 5-10 LLM calls is fully sequential — even calls that could run in parallel (e.g., generate + review attempts)
- If migrating to async, we could use `asyncio.gather()` for natural parallelism

### 7. `app.py` is a 900-line mix of routing + threading + SSE + queue management

**Current state:** Single file mixing many concerns. Hard to read, hard to test, hard to modify safely.

**Specific clutter:**
- ~80-line post-graph cascade with 5+ branches
- Manual SSE event formatting
- Threading + queue setup
- Session DB management
- WebSocket handlers
- KIM API endpoints
- Background tasks

---

## How FastAPI addresses each problem

| Current Problem | How FastAPI fixes it |
|---|---|
| Frontend locks from SSE buffering | Native async `StreamingResponse` — events flush as they're yielded, no Werkzeug buffering. Stream cleanly closes when generator exits. Frontend's `reader.read()` reliably returns `done=true`. |
| `SESSION_QUEUES` global hack | Each request owns its own `asyncio.Queue` in local scope. No global registry. Cleanup is automatic when the async function returns. |
| Background threads + races | Native asyncio tasks. Single event loop. No GIL contention for I/O. Cancellation is built-in (`Request.is_disconnected()` detects client disconnect). |
| Flask-SocketIO monkey-patching | Native ASGI WebSocket via Starlette. No monkey-patching. Clean async handlers. |
| Sync LangGraph wrapped in thread | Native `await graph.ainvoke()`. LangGraph's async path is well-tested. No thread overhead. |
| Sync DSPy | Migrate to DSPy `aforward()`. Use `asyncio.gather()` for parallel agent calls (e.g., generate + review concurrently). |
| 900-line `app.py` | Routes are thin async functions. SSE is `StreamingResponse(generator)`. Queue logic disappears (use local `asyncio.Queue`). Easier to test, easier to modify. |

### Plus benefits we get for free

- **Pydantic-based input validation** — every endpoint auto-validates request bodies. Catches `{"title": 123}` (wrong type) before it crashes our handler.
- **Auto-generated OpenAPI docs** at `/docs` and `/redoc` — useful for understanding endpoints
- **Better concurrency under load** — async event loop handles many concurrent users without spawning threads
- **Native dependency injection** — cleaner code for shared resources (DB connections, LLM clients)
- **Production-ready ASGI server** (uvicorn) — vs. Werkzeug's dev WSGI server which we have to bypass with `allow_unsafe_werkzeug=True`

---

## Phased migration plan

7-12 days total. Each phase ends with something testable so we don't migrate-and-pray.

### Phase 1 — Foundation + simple endpoints (1-2 days)

**Goal:** FastAPI running alongside Flask, handling the simple endpoints. Old streaming still works.

**Tasks:**
- Add `fastapi`, `uvicorn`, `python-multipart` to `pixi.toml`
- Create `src/app_fastapi.py` (do not delete `app.py` yet)
- Migrate read-only / simple endpoints:
  - `GET /` (serve index.html)
  - `GET /api/sessions`, `POST /api/sessions`, `PATCH /api/sessions/<id>`, `DELETE /api/sessions/<id>`, `GET /api/sessions/<id>`
  - `POST /api/feedback`
  - `GET /api/kim-potentials/list`, `/search`, `/status`
  - `POST /api/kim-potentials/download`, `/init`, `/add`
- Run on port 2003 for parallel testing (Flask stays on 2002)
- Add Pydantic models for request bodies (PatchSession, CreatePotential, etc.)

**Deliverable:** FastAPI app serves the UI, session list works, KIM panel works. Streaming still goes to old Flask on port 2002.

**Risk:** Low. CRUD endpoints translate trivially.

---

### Phase 2 — SSE streaming endpoint (2-3 days) — the BIG one

**Goal:** `POST /api/llm-stream-live` migrated to FastAPI. Chat works end-to-end through new server. **The hangs we've been fighting should be GONE after this.**

**Tasks:**
- Convert `llm_stream_live` from sync to async
- Replace `queue.Queue` with `asyncio.Queue`
- Replace threaded `run_graph` with native asyncio task (`asyncio.create_task`)
- Replace Flask `Response(generate(), mimetype='text/event-stream')` with FastAPI `StreamingResponse`
- Migrate the post-graph cascade as-is (just async)
- Keep `simulation_graph.invoke()` as sync for now — wrap with `asyncio.to_thread()` to call from async context (defer full async LangGraph to phase 4)
- Migrate diagnostic prints + auto-titling SQL
- Update frontend if SSE format needs tweaks (probably no changes — same `data: {...}\n\n` format)
- Remove `SESSION_QUEUES` global hack — each request owns its own asyncio.Queue in local scope
- Add `Request.is_disconnected()` check to detect client disconnect and cancel cleanly

**Deliverable:** Chat flows work through FastAPI. SSE buffering issues should disappear because async StreamingResponse flushes properly. Frontend lock-ups should be gone.

**Risk:** Medium. Need to handle:
- `asyncio.to_thread()` for sync LangGraph calls
- Cancellation when client disconnects
- SqliteSaver is sync — wrap appropriately

---

### Phase 3 — WebSocket migration (1-2 days)

**Goal:** Local simulation streaming + KIM downloads + replay all work via FastAPI.

**Tasks:**
- Choose WebSocket approach:
  - **Option A:** `python-socketio` (ASGI mode) — keeps frontend code identical (still uses socket.io client)
  - **Option B:** Native FastAPI `WebSocket` — cleaner, but requires frontend rewrite
- Migrate WebSocket handlers:
  - `start_streaming_simulation` (local sim with trajectory streaming)
  - `replay_trajectory`
  - `download_kim_potential`
  - `initialize_recommended_potentials`
- Update frontend if Option B chosen
- Test trajectory rendering still works

**Deliverable:** Local sim execution works through FastAPI. OVITO trajectory frames stream correctly.

**Risk:** Medium with Option A (python-socketio + ASGI sometimes has quirks), Low with Option B (native FastAPI WS is well-tested).

**Recommendation:** Option A to avoid frontend changes. Fall back to B if issues.

---

### Phase 4 — Async LangGraph + DSPy (2-3 days)

**Goal:** Native async throughout. No `asyncio.to_thread` wrapping for graph operations.

**Tasks:**
- Switch `simulation_graph.invoke()` → `simulation_graph.ainvoke()`
- Convert graph nodes to async (`async def chat_node`, etc.) — most just need `await` on LLM calls
- Migrate DSPy to async LM calls (DSPy 3.x has `aforward()` pattern)
- Wrap sync subprocess calls (MCP, OVITO render server) with `asyncio.to_thread()` — these stay threaded but don't block event loop
- Update SqliteSaver usage (might use `aiosqlite` if available, or keep sync wrapped)
- Test all flows still work

**Deliverable:** Full async pipeline. Better concurrency. No thread pool overhead.

**Optional:** If time-constrained, skip this phase — `asyncio.to_thread()` wrapping from Phase 2 is "good enough" for correctness. This phase is for performance + cleanliness.

**Risk:** Medium-High. DSPy async support is newer and might have edge cases. Some node refactoring needed.

---

### Phase 5 — Cleanup + testing + documentation (1-2 days)

**Goal:** Production-ready FastAPI app. Old Flask code removed.

**Tasks:**
- Delete `src/app.py` (Flask version)
- Rename `src/app_fastapi.py` → `src/app.py`
- Remove `flask`, `flask-socketio`, `eventlet` from `pixi.toml`
- Remove `werkzeug` `allow_unsafe_werkzeug=True` hack
- Add proper logging (replace `print()` calls with `logging.info/debug`)
- Update README / setup instructions
- Update startup command (`uvicorn src.app:app --host 0.0.0.0 --port 2002` instead of `python src/app.py`)
- End-to-end testing of every flow:
  - New simulation (local + HPC)
  - Status check (HPC)
  - Transfer back
  - Analyst flow
  - KIM download
  - Session switching
- Performance benchmark vs old code

**Deliverable:** Clean FastAPI codebase. Documented. Tested.

**Risk:** Low. Just cleanup and verification.

---

## Total estimate

| Phase | Time | What it fixes |
|---|---|---|
| 1 — Foundation + simple endpoints | 1-2 days | Faster UI loads, cleaner CRUD |
| **2 — SSE streaming (main fix)** | **2-3 days** | **Frontend hangs from SSE buffering — GONE** |
| 3 — WebSocket migration | 1-2 days | Trajectory streaming on FastAPI |
| 4 — Async LangGraph + DSPy | 2-3 days | True async, parallel LLM calls |
| 5 — Cleanup + testing | 1-2 days | Production-ready |
| **Total** | **7-12 days** | |

## Realistic timeline

- **Best case (no surprises, focused work):** 7 days
- **Typical:** 9-10 days
- **With unforeseen issues / debugging:** 12 days
- **If skip Phase 4 (keep sync LangGraph wrapped):** Save 2-3 days. Total 5-9 days.

## What stays untouched (good news)

- `src/graph/` — LangGraph nodes work as-is (Phase 4 makes them async, but they keep their structure)
- `src/prompts.py` — no changes
- `src/agents.py` — DSPy module definitions stay (Phase 4 makes some methods async)
- `index.html` — minimal/no changes (Phase 3 might tweak WebSocket, otherwise none)
- `src/tools.py` — LAMMPS execution unchanged
- All pending fixes (#3, #4, #8) — apply to either codebase, do them in either before or after migration

---

## Recommended next steps

1. **Schedule a focused block** of 7-12 days. Avoid trying to do this incrementally between feature work — context-switching breaks the flow.
2. **Phase 2 is the critical path.** If under time pressure, do Phases 1+2 only (3-5 days). That alone fixes the SSE hang issues.
3. **Test thoroughly between phases.** Each phase should be deployable on its own.
4. **Keep the old code until Phase 5.** Don't delete `app.py` until everything works on the FastAPI side.
5. **Don't skip the proper logging migration** in Phase 5. The `print()` statements were fine for debugging but should be `logging.debug()` calls in production.

## Risks to watch

- **DSPy async support** — DSPy 3.x has it but it's newer. Test incrementally.
- **Globus MCP / langchain_mcp_adapters** — verify they work in async context
- **SqliteSaver** — sync; may need `aiosqlite` migration or careful wrapping
- **Frontend SSE protocol** — verify byte-for-byte the same SSE format as Flask
- **WebSocket path** — depends heavily on Option A vs B choice

## When to revisit this plan

- After Phase 2: re-evaluate whether Phases 3-5 are worth the time, or if you want to ship the partial migration
- If new bugs surface that aren't related to async/SSE — they're probably orthogonal and should be fixed independently
- After a few weeks of running on FastAPI: profile performance, see if there are further optimization opportunities (caching, batching, etc.)
