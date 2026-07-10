# Campaign Mode v2 — Workflow & Ownership

## Purpose

Autonomous MD simulation campaigns driven end-to-end by the LangGraph orchestrator.
The user defines an objective; the orchestrator designs the plan, runs iterations,
analyses results, and produces a final report with minimal human intervention.

This document defines **the workflow** and **who does what** — the source of truth
for the v2 rewrite. Implementation plan lives in `~/.claude/plans/tranquil-yawning-spark.md`.

---

## Actors

| Symbol | Actor | Role |
|---|---|---|
| 👤 | **User** | Defines objective, confirms plan, controls run (pause/resume/cancel), reads report |
| 🤖 | **LLM** (orchestrator, LangGraph ReAct agent) | Plans, reasons, iterates, writes narrative |
| 🐍 | **Python** (tools + HTTP endpoints + background) | Runs simulations, measures metrics, persists state, renders graph |
| 🖥️ | **Frontend** (browser, index.html) | UI, event rendering, control buttons |

---

## End-to-end workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  👤 USER                                                            │
│    │  "Run a campaign to minimize Δa on Al Griffith crack"          │
│    ▼                                                                │
│                                                                     │
│  ─────────────── TURN 1 · DESIGN THE CAMPAIGN ───────────────       │
│                                                                     │
│  🤖 LLM                                                             │
│    • Asks clarifying questions if scope is unclear                  │
│    • Drafts full plan dict (objective, template, knobs+bounds,      │
│      max_iters, stop criteria)                                      │
│    • Prints plan as prose in chat                                   │
│    • Ends turn — waits for user reply                               │
│                                                                     │
│  👤 USER                                                            │
│    • Reads plan, replies "confirm" (or edits: "change max_iter=30") │
│                                                                     │
│  ─────────────── TURN 2 · RUN THE CAMPAIGN ─────────────────────    │
│  ─────────────── (single ReAct trace, all iterations) ──────        │
│                                                                     │
│  🤖 LLM                                                             │
│    ├─ 🐍 campaign_start(plan)                                       │
│    │      • creates runs/<campaign_id>/                             │
│    │      • writes plan.json + control.json (status=running)        │
│    │      • emits campaign_started SSE                              │
│    │      • returns campaign_id                                     │
│    │                                                                │
│    │   ┌──────────  ITERATION LOOP  ────────────┐                   │
│    │   │                                        │                   │
│    │   ├─ 🐍 campaign_get_state()               │                   │
│    │   │      returns plan + trials + best +    │                   │
│    │   │      iterations_since_improvement      │                   │
│    │   │                                        │                   │
│    │   ├─ 🤖 REASON: pick next knobs +          │                   │
│    │   │      write hypothesis (streamed live)  │                   │
│    │   │                                        │                   │
│    │   ├─ 🐍 simulate(knobs, auto_approve=True) │                   │
│    │   │      generates LAMMPS+OVITO, runs,     │                   │
│    │   │      streams frames+thermo,            │                   │
│    │   │      returns run_folder                │                   │
│    │   │                                        │                   │
│    │   ├─ 🐍 campaign_measure_metrics(          │                   │
│    │   │       run_folder, metric_name)         │                   │
│    │   │      dispatches to metric registry     │                   │
│    │   │      returns {delta_a, n_defect_*, ...}│                   │
│    │   │                                        │                   │
│    │   ├─ 🐍 campaign_record_trial(iter, knobs, │                   │
│    │   │       hypothesis, metrics, folder)    │                   │
│    │   │      appends to trials.jsonl           │                   │
│    │   │      emits campaign_iteration_complete │                   │
│    │   │                                        │                   │
│    │   ├─ 🐍 campaign_check_stop()              │                   │
│    │   │      reads plan+trials+control         │                   │
│    │   │      returns {should_stop, reason}     │                   │
│    │   │                                        │                   │
│    │   │      should_stop=False → back to top   │                   │
│    │   │      should_stop=True  → exit loop     │                   │
│    │   └────────────────────────────────────────┘                   │
│    │                                                                │
│    ├─ 🐍 campaign_generate_report(campaign_id)                      │
│    │      • renders report.png (matplotlib)                         │
│    │      • writes report.md skeleton (facts only)                  │
│    │      • returns {md_path, png_path, skeleton, next_step}        │
│    │                                                                │
│    ├─ 🤖 WRITE narrative "What was learned" section                 │
│    │      (streamed live to user in chat)                           │
│    │                                                                │
│    └─ 🐍 campaign_save_narrative(campaign_id, text)                 │
│           appends narrative to report.md                            │
│           emits campaign_complete SSE                               │
│                                                                     │
│  👤 USER                                                            │
│    • Sees final graph + summary inline                              │
│    • Opens report.md / report.png for full detail                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌───────────── PARALLEL CONTROLS (👤 during running loop) ────────────┐
│                                                                     │
│   👤 clicks Pause / Resume / Cancel button                          │
│      ▼                                                              │
│   🖥️ POST /api/campaign/{id}/{pause|resume|cancel}                  │
│      ▼                                                              │
│   🐍 writes control.json ← {status: paused|running|cancelled}       │
│      ▼                                                              │
│   🐍 campaign_check_stop() picks it up on the next call             │
│                                                                     │
│      pause     → should_stop=True (reason=user_paused);             │
│                  loop exits; NO report generated                    │
│      resume    → 🖥️ re-invokes orchestrator on same session;        │
│                  LLM sees status=running, re-enters loop            │
│      cancel    → should_stop=True (reason=user_cancelled);          │
│                  loop exits, report IS generated from trials so far │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌────────────── PERSISTED STATE — runs/<campaign_id>/ ────────────────┐
│                                                                     │
│   plan.json          🐍 written by start_campaign, immutable        │
│   control.json       🐍 written by start_campaign + control API     │
│   trials.jsonl       🐍 appended by record_trial (one line/iter)    │
│   summary.json       🐍 rewritten by record_trial (aggregate)       │
│   campaign_log.md    🐍 appended by record_trial (human-readable)   │
│   report.md          🐍 skeleton + 🤖 narrative                     │
│   report.png         🐍 written by generate_report                  │
│   run_<timestamp>/   🐍 per-iter LAMMPS output (from simulate)      │
│                                                                     │
│   data/orchestrator_sessions.db   🐍 AsyncSqliteSaver checkpoint    │
│                                    (LLM state + full ReAct trace)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Ownership table

### Actions by 👤 USER

| Action | When |
|---|---|
| Types campaign objective in chat | Turn 1, start |
| Answers clarifying questions if asked | Turn 1, mid |
| Types "confirm" or edit request | Turn 1, end |
| Clicks Pause / Resume / Cancel button | Any time during running loop |
| Reads final report + graph | After campaign_complete |

### Actions by 🤖 LLM (inside orchestrator ReAct trace)

| Action | Called via | When |
|---|---|---|
| Ask clarifying questions | Direct chat message | Turn 1 |
| Draft full plan dict (objective + template + knobs + stop criteria) | Direct chat message | Turn 1 |
| Call `campaign_start(plan)` | Tool | Turn 2, once |
| Call `campaign_get_state()` | Tool | Every iteration, step 1 |
| Reason + write hypothesis + pick next knobs | Chat message (streamed) | Every iteration, step 2 |
| Call `simulate(knobs, auto_approve=True)` | Tool | Every iteration, step 3 |
| Call `campaign_measure_metrics(run_folder, metric_name)` | Tool | Every iteration, step 4 |
| Call `campaign_record_trial(...)` | Tool | Every iteration, step 5 |
| Call `campaign_check_stop()` | Tool | Every iteration, step 6 |
| Call `campaign_generate_report(campaign_id)` | Tool | After loop exits |
| Write "What was learned" narrative | Chat message (streamed) | After campaign_generate_report |
| Call `campaign_save_narrative(campaign_id, text)` | Tool | Final tool call |

**LLM does NOT:**
- Decide when to stop (that's `campaign_check_stop()` in Python)
- Compute report facts (that's `campaign_generate_report()` in Python)
- Parse or re-format past-attempt prose (history comes typed from `campaign_get_state()`)
- Persist state (all persistence is in tools)

### Actions by 🐍 PYTHON — Tools

All campaign tools are prefixed `campaign_*` for clarity in `ORCHESTRATOR_TOOLS`.

| Tool | Owner responsibilities |
|---|---|
| `campaign_start(plan)` | Create `runs/<campaign_id>/`, write `plan.json` + `control.json`, emit `campaign_started`, return `campaign_id` |
| `campaign_get_state()` | Read `plan.json` + `trials.jsonl` + `control.json`; compute `best_so_far`, `iterations_since_improvement`; return typed dict (respecting `plan.objective.direction` for min vs max) |
| `simulate(...)` | **Unchanged from today.** Generate LAMMPS + OVITO, run subprocess, stream frames/thermo, return `run_folder`. Honors `auto_approve=True` to skip review card |
| `campaign_measure_metrics(run_folder, metric_name)` | Dispatcher. Looks up `metric_name` in `src/benchmark/metrics/__init__.py` registry, invokes the extractor, returns the metric dict. Ships with `delta_a` only in v2; add-a-metric = drop a new file in `src/benchmark/metrics/` |
| `campaign_record_trial(iteration, knobs, hypothesis, metrics, run_folder, wall_clock_s, failure_reason?)` | Append line to `trials.jsonl`, rewrite `summary.json`, append markdown to `campaign_log.md`, emit `campaign_iteration_complete` or `_failed` |
| `campaign_check_stop()` | Deterministic. Read plan + trials + control. Evaluate 6 criteria in order: `user_cancelled`, `user_paused`, `max_iterations`, `target_hit`, `no_improvement`, `convergence`. Uses `plan.objective.direction` to compare correctly for min vs max campaigns. Return `{should_stop, reason}` |
| `campaign_generate_report(campaign_id)` | Render `report.png` (matplotlib, 2 panels: metric vs iter + per-knob scatter). Y-axis label pulled from `plan.objective.display_name` + `units`. Write `report.md` skeleton (Objective, Result, Trial table, Files, empty "What was learned"). Return paths + skeleton |
| `campaign_save_narrative(campaign_id, text)` | Append narrative under `## What was learned` in `report.md`. Emit `campaign_complete` |

### Actions by 🐍 PYTHON — HTTP endpoints

| Endpoint | Owner responsibilities |
|---|---|
| `POST /api/campaign/{id}/pause` | Write `control.json` with `status="paused"`. Return 200 |
| `POST /api/campaign/{id}/resume` | Write `control.json` with `status="running"`. Re-invoke orchestrator on the paused session (kicks loop back to life). Return 200 |
| `POST /api/campaign/{id}/cancel` | Write `control.json` with `status="cancelled"`. Return 200 |

### Actions by 🐍 PYTHON — Background

| Component | Owner responsibilities |
|---|---|
| `AsyncSqliteSaver` checkpointer | Checkpoints orchestrator state after every ReAct step. Enables restart-safety. **Unchanged from today.** |
| SSE stream (`sse_event_stream` in app.py) | Forwards events from session queue to browser. **Unchanged from today** — no more daemon thread means no more `_DualQueue`, no more `CAMPAIGN_COMPLETION_EVENTS` |

### Actions by 🖥️ FRONTEND (index.html)

| Action | New / Existing |
|---|---|
| Show campaign banner on `campaign_started` | Existing, unchanged |
| Update banner on `campaign_iteration_start` | Existing, unchanged |
| Append row to progress table on `campaign_iteration_complete` | Existing, unchanged |
| Show final graph inline on `campaign_complete` | Existing (needs img src pointed at report.png) |
| Pause / Resume / Cancel buttons on banner | **New** |
| Wire buttons to `POST /api/campaign/{id}/{action}` | **New** |
| Handle `campaign_paused` / `campaign_resumed` / `campaign_cancelled` events | **New** |

---

## `plan.json` schema (LLM authors, Python persists)

Note: `objective` is a **structured object**, not a string. Ships with `metric: "delta_a"` in v2; new metrics registered in `src/benchmark/metrics/` become selectable without changing any tool.

```json
{
  "campaign_id": "benchmark_20260708_143052",
  "created_at": "2026-07-08T14:30:52Z",
  "objective": {
    "metric":       "delta_a",
    "direction":    "minimize",
    "units":        "Å",
    "display_name": "Δa",
    "target":       null
  },
  "template": {
    "material": "Al",
    "system_size": 8000,
    "phase_1": "300K equilibration 10ps",
    "phase_2": "tensile load 50ps",
    "camera_angle": "+z"
  },
  "knobs": {
    "strain_amplitude": [0.001, 0.01],
    "temperature":      [200, 600],
    "crack_length_frac":[0.2, 0.5],
    "crack_width_a0":   [3, 8]
  },
  "max_iterations": 20,
  "stop_criteria": {
    "no_improvement_patience": 5,
    "convergence": {"epsilon_pct": 1.0, "window": 5}
  }
}
```

## `trials.jsonl` schema (Python writes, one line per iteration)

```json
{"iteration": 3, "knobs": {"strain_amplitude": 0.005, "temperature": 400, ...},
 "hypothesis": "Higher T should reduce Δa by ...",
 "metrics": {"delta_a": 1.23, "n_defect_initial": 812, "n_defect_final": 950, ...},
 "run_folder": "runs/benchmark_.../run_2026...",
 "wall_clock_s": 187.4,
 "failure_reason": null,
 "is_new_best": true,
 "created_at": "2026-07-08T14:38:15Z"}
```

## `control.json` schema

```json
{"status": "running", "updated_at": "2026-07-08T14:38:15Z"}
```

## `check_stop()` return

```json
{"should_stop": true, "reason": "no_improvement", "detail": "no new best in 5 trials"}
```

Reasons: `max_iterations` · `no_improvement` · `convergence` · `target_hit` · `user_paused` · `user_cancelled`.

---

## Stopping criteria

Evaluated in `check_stop()` in this order (first match wins):

| Criterion | Condition | Reason |
|---|---|---|
| User cancelled | `control.status == "cancelled"` | `user_cancelled` |
| User paused | `control.status == "paused"` | `user_paused` |
| Max iterations | `len(trials) >= plan.max_iterations` | `max_iterations` |
| Target achieved | `plan.objective.target` is set AND best-so-far meets it (`<=` for minimize, `>=` for maximize) | `target_hit` |
| No improvement | `iterations_since_last_new_best >= plan.stop_criteria.no_improvement_patience` | `no_improvement` |
| Convergence | Best-value change over last N trials < ε% | `convergence` |

**Direction-aware comparison** — `campaign_check_stop` and `campaign_get_state` both use `plan.objective.direction` (`"minimize"` or `"maximize"`) to pick the right comparator. "Best" means smallest for minimize, largest for maximize.

---

## Metric registry (extensibility)

Metrics live as pluggable files under `src/benchmark/metrics/`. Each file exports one function:

```
compute(run_folder: str) -> dict
```

The registry (`src/benchmark/metrics/__init__.py`) is a simple mapping:

```python
METRIC_REGISTRY = {
    "delta_a": compute_delta_a,
    # add new metrics here — one line each
}
```

**v2 ships with `delta_a` only.** Future metrics (fracture_toughness, peak_stress, dxa_dislocation_density, ...) are added by (1) dropping a new file in the folder and (2) adding one line to `METRIC_REGISTRY`. No changes needed to `campaign_measure_metrics`, `campaign_check_stop`, `campaign_generate_report`, or the campaign skill — everything downstream reads `plan.objective` and works generically.

If the LLM asks for a metric that isn't in the registry, `campaign_measure_metrics` returns an error listing the available options. The campaign skill teaches the LLM to advertise only registered metrics during Turn 1 planning.

## Standalone workflow — unaffected

The v2 rewrite is **surgical**: only campaign-specific code is added or removed. The standalone chat → `simulate()` → `analyze_runs()` path is untouched. Verification (§Verification below) requires a standalone smoke test *before* declaring v2 done.

Unaffected components:
- `simulate()` tool + LAMMPS subgraph (Generator / Reviewer / OVITO)
- `analyze_runs()` tool
- `run_lammps_simulation` in `src/tools.py`
- Normal chat SSE plumbing
- All deletes (CampaignManager, _DualQueue, SSE keep-alive block) only fire during campaigns

## What's deleted vs today

| Deleted | Reason |
|---|---|
| `src/benchmark/campaign.py` (729 LOC) — `CampaignManager` class + `_DualQueue` | Loop lives in orchestrator now |
| `src/benchmark/campaign_prompts.py` (56 LOC) — `format_history_message` | History comes typed via `get_campaign_state()` |
| SSE keep-alive block in `app.py` L714-734 | No daemon thread to wait on |
| STRICT ITERATION FORMAT + `PROPOSED_KNOBS:` rules in `campaign.md` skill | Knobs typed end-to-end; no prose parsing |
| Daemon-thread setup in `start_campaign` tool | Orchestrator drives the loop directly |
| `CAMPAIGN_COMPLETION_EVENTS` bookkeeping | No thread to signal completion for |

## What's reused as-is

- `simulate()` tool + LAMMPS subgraph (Generator / Reviewer / OVITO)
- `crack_metrics.compute_delta_a()`
- `run_manager.create_run_folder(parent_dir=...)`
- `AsyncSqliteSaver` checkpointer
- Frontend campaign banner + progress table (event contract unchanged for the 4 existing events)
- SSE plumbing (`_q`, `SESSION_QUEUES`, `sse_event_stream`)

---

## Out of scope for v2

- **Steering messages** (user guidance folded into running loop's next reasoning step) — deferred; additive later via `plan.pending_steering: list[str]`
- **HPC in the campaign loop** — deferred to v2.1; design sketched below
- **Multi-objective / Pareto front** — v2 is single scalar objective
- **BO / Ax optimizer** — future swap-in for the LLM reasoning step (loop shape supports it without changes)

---

## Appendix — HPC-in-campaign (v2.1 sketch)

**Why HPC is fundamentally different from local:** HPC jobs take hours. The orchestrator can't sit blocked on a ReAct trace waiting — it would burn tokens, hold the SSE stream open indefinitely, and lose everything on browser close. So each HPC iteration must **submit-and-exit**, and something has to **wake the orchestrator** when the job finishes.

### Loop shape (v2.1, HPC)

Each iteration becomes a **short turn** (submit + exit) rather than one long ReAct trace containing all iterations. The whole campaign becomes many short turns triggered by a background poller.

```
Iter N (short turn):
  campaign_get_state()  → sees iters 1..N-1 complete
  LLM reason + hypothesis
  simulate(execution_target="hpc", auto_approve=True)
      → auto-submits (no PBS-edit pause), returns {job_id, status: "queued"}
  campaign_record_trial(iteration=N, ..., metrics=null, job_id, status="running")
  campaign_check_stop()  → returns should_stop=True, reason="awaiting_hpc"
  LLM emits "iter N submitted, waiting..." → exits loop, turn ends

[hours pass; browser can be closed]

Background poller notices job_id complete:
  runs measure_metrics(run_folder)
  updates that trial row in trials.jsonl (status="complete", metrics={...})
  emits campaign_iteration_complete SSE
  re-invokes orchestrator on session: "iter N complete, continue"

Iter N+1 (next short turn): same shape as above.
```

### New components v2.1 needs

- **`src/orchestrator/hpc_campaign_poller.py`** — background asyncio task launched at app startup; scans trials with `status="running"`; polls PBS/Globus every ~60s; on completion updates trial + triggers orchestrator continuation
- **Session registry** — `campaign_id → chat_session_id` mapping (written by `campaign_start`) so the poller knows which session to wake
- **`plan.hpc_config`** — `{nodes, walltime, queue, globus_endpoint}` field added to plan schema
- **Trial row extensions** — `job_id`, `status` fields
- **`simulate()` HPC-in-campaign branch** — skip PBS-edit pause when in campaign context; auto-submit; return `{job_id, status: "queued"}`
- **`campaign_check_stop()` new criterion** — if any trial has `status="running"`, return `{should_stop: True, reason: "awaiting_hpc"}` to force graceful loop exit
- **Recursion limit** — HPC turns are ~6 steps; default 25 fine (only local turns need the 200 bump)

### Edge cases handled by design

- **Browser closed during HPC wait** → poller keeps working; user reopens session; `campaign_get_state()` shows progress
- **Server restart** → poller relaunches, scans for `status="running"` trials, resumes polling. No campaigns lost
- **HPC job fails/timeouts** → poller writes trial with `failure_reason="hpc_job_failed"`; orchestrator wake-up message includes the failure; LLM decides retry vs continue
- **User pauses during HPC wait** → `control.status="paused"` — poller still updates trial rows as jobs complete but withholds orchestrator wake-ups until unpaused
- **Multiple parallel HPC jobs** — nothing in the design forces sequential submission (v2.2+ feature; loop shape already supports it)

### What survives from v2 unchanged

- All 7 `campaign_*` tools keep the same names/signatures (gain optional HPC-aware fields)
- Report generation
- Metric registry
- Frontend (adds "awaiting HPC" banner state)
- Local campaign flow is completely untouched

**v2.1 is purely additive** — the local path never gains HPC complexity.
