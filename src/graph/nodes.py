"""
LangGraph node functions for the simulate subgraph + analyze agent.

Each node takes SimulationState and returns a partial dict of updated fields.
SSE events are put onto SESSION_QUEUES[session_id] so app.py's SSE generator
can forward them to the browser.
"""
import json
import queue
import threading
from typing import Dict
from openai import OpenAI

# Global registry: session_id → queue.Queue
# Keeps sse_queue out of LangGraph state (not serializable by SqliteSaver).
SESSION_QUEUES: Dict[str, queue.Queue] = {}

import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run_manager
from agents import (
    IterativeLammpsGeneration,
    IterativeOVITOGeneration,
    SimulationHelpers,
    detect_precomputed_properties,
)
from prompts import ANALYST_SYSTEM_PROMPT
from graph.state import SimulationState

client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.API_BASE_URL,
)


def _q(state: SimulationState, event: dict):
    """Helper: put an SSE event onto the queue for this session."""
    sid = state.get("session_id")
    if sid and sid in SESSION_QUEUES:
        SESSION_QUEUES[sid].put(event)


# ─── generate_lammps ─────────────────────────────────────────────────────────

def generate_lammps_node(state: SimulationState) -> dict:
    _q(state, {"type": "agent_start", "agent": "Generator"})
    _q(state, {"type": "status", "message": "Generating and reviewing script..."})

    iterative_generator = IterativeLammpsGeneration(max_attempts=2)
    progress_queue = queue.Queue()

    def _progress_callback(stage, attempt, max_attempts, **kwargs):
        progress_queue.put((stage, attempt, max_attempts, kwargs))

    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = iterative_generator.forward(
                material=state.get("material") or "unknown",
                kim_model=state.get("kim_model"),
                temperature=state.get("temperature"),
                ensemble=state.get("ensemble"),
                duration_ps=state.get("duration_ps"),
                system_size=state.get("system_size"),
                simulation_details=state.get("analysis", ""),
                visualization_notes=state.get("visualization_notes"),
                progress_callback=_progress_callback,
            )
        except Exception as e:
            error_holder[0] = e
        finally:
            progress_queue.put(("done", 0, 0, {}))

    thread = threading.Thread(target=_run)
    thread.start()

    # Forward progress events to sse_queue
    while True:
        try:
            stage, attempt, max_attempts, kwargs = progress_queue.get(timeout=0.5)
        except queue.Empty:
            _q(state, {"type": "heartbeat"})
            continue

        if stage == "done":
            break
        elif stage == "generating":
            _q(state, {"type": "status", "message": f"Generating script (attempt {attempt}/{max_attempts})..."})
        elif stage == "script_generated":
            script_text = kwargs.get("script", "")
            explanation = kwargs.get("explanation", "")
            _q(state, {"type": "agent_start", "agent": "Generator"})
            _q(state, {"type": "agent_content", "agent": "Generator", "content": script_text,
                       "message_type": "script_trace", "attempt": attempt})
            if explanation:
                _q(state, {"type": "agent_content", "agent": "Generator", "content": explanation,
                           "message_type": "explanation", "attempt": attempt})
            _q(state, {"type": "agent_complete", "agent": "Generator"})
        elif stage == "reviewing":
            _q(state, {"type": "status", "message": f"Reviewing script (attempt {attempt}/{max_attempts})..."})
        elif stage == "review_complete":
            is_valid = kwargs.get("is_valid", True)
            issues = kwargs.get("issues", [])
            suggestions = kwargs.get("suggestions", [])
            severity = kwargs.get("severity", "none")
            verdict = "PASS" if is_valid else "FAIL"
            review_msg = f"**REVIEW: {verdict}** (attempt {attempt}/{max_attempts}, severity: {severity})\n"
            if issues:
                review_msg += "\n**Issues:**\n" + "\n".join(f"- {i}" for i in issues)
            if suggestions:
                review_msg += "\n\n**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions)
            _q(state, {"type": "agent_start", "agent": "Reviewer"})
            _q(state, {"type": "agent_content", "agent": "Reviewer", "content": review_msg,
                       "message_type": "review", "attempt": attempt})
            _q(state, {"type": "agent_complete", "agent": "Reviewer"})

    thread.join()

    if error_holder[0]:
        raise error_holder[0]

    result = result_holder[0]
    print(f"[generate_lammps] review_passed={result.review_passed}  attempts={result.attempts}")
    return {
        "lammps_script":         result.script,
        "dump_filename_pattern": getattr(result, "dump_filename_pattern", "") or "",
        "total_steps":           getattr(result, "total_steps", 0) or 0,
        "dump_freq_val":         getattr(result, "dump_freq_val", 1000) or 1000,
        "review_passed":         result.review_passed,
        "review_issues":         getattr(result, "issues", []) or [],
        "review_suggestions":    getattr(result, "suggestions", []) or [],
    }


# ─── detect_viz ──────────────────────────────────────────────────────────────

def detect_viz_node(state: SimulationState) -> dict:
    _q(state, {"type": "agent_start", "agent": "Visualization"})

    viz_request = state.get("visualization_notes") or state["user_message"]
    lammps_script = state.get("lammps_script", "")

    # Detect pre-computed properties from LAMMPS script
    precomputed = {}
    if lammps_script:
        precomputed = detect_precomputed_properties(lammps_script)
        if precomputed:
            print(f"[detect_viz] precomputed properties: {list(precomputed.keys())}")

    # Build simulation_context for OVITO generator
    simulation_context = {}
    material = state.get("material")
    if material:
        simulation_context["material"] = material
    if precomputed:
        simulation_context["precomputed_properties"] = precomputed

    helpers = SimulationHelpers()
    has_custom_viz = helpers.detect_visualization(viz_request)
    print(f"[detect_viz] has_custom_viz={has_custom_viz}")

    return {
        "viz_request":           viz_request,
        "precomputed_properties": precomputed,
        "simulation_context":    simulation_context,
        "has_custom_viz":        has_custom_viz,
    }


# ─── generate_ovito ──────────────────────────────────────────────────────────

def generate_ovito_node(state: SimulationState) -> dict:
    _q(state, {"type": "status", "message": "Generating and reviewing OVITO script..."})

    ovito_gen = IterativeOVITOGeneration(max_attempts=config.OVITO_SCRIPT_MAX_RETRIES)
    progress_queue = queue.Queue()

    def _progress_callback(stage, attempt, max_attempts, **kwargs):
        progress_queue.put((stage, attempt, max_attempts, kwargs))

    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = ovito_gen.forward(
                visualization_request=state.get("viz_request") or state["user_message"],
                simulation_context=state.get("simulation_context"),
                progress_callback=_progress_callback,
            )
        except Exception as e:
            error_holder[0] = e
        finally:
            progress_queue.put(("done", 0, 0, {}))

    thread = threading.Thread(target=_run)
    thread.start()

    while True:
        try:
            stage, attempt, max_attempts, kwargs = progress_queue.get(timeout=0.5)
        except queue.Empty:
            _q(state, {"type": "heartbeat"})
            continue

        if stage == "done":
            break
        elif stage == "generating":
            _q(state, {"type": "status", "message": f"Generating OVITO script (attempt {attempt}/{max_attempts})..."})
        elif stage == "script_generated":
            script_text = kwargs.get("script", "")
            _q(state, {"type": "agent_start", "agent": "Visualization"})
            _q(state, {"type": "agent_content", "agent": "Visualization", "content": script_text,
                       "message_type": "script_trace", "attempt": attempt})
            _q(state, {"type": "agent_complete", "agent": "Visualization"})
        elif stage == "reviewing":
            _q(state, {"type": "status", "message": f"Reviewing OVITO script (attempt {attempt}/{max_attempts})..."})
        elif stage == "review_complete":
            is_valid = kwargs.get("is_valid", True)
            issues = kwargs.get("issues", [])
            suggestions = kwargs.get("suggestions", [])
            severity = kwargs.get("severity", "none")
            verdict = "PASS" if is_valid else "FAIL"
            review_msg = f"**OVITO Review: {verdict}** (attempt {attempt}/{max_attempts}, severity: {severity})\n"
            if issues:
                review_msg += "\n**Issues:**\n" + "\n".join(f"- {i}" for i in issues)
            if suggestions:
                review_msg += "\n\n**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions)
            _q(state, {"type": "agent_start", "agent": "Reviewer"})
            _q(state, {"type": "agent_content", "agent": "Reviewer", "content": review_msg,
                       "message_type": "ovito_review", "attempt": attempt})
            _q(state, {"type": "agent_complete", "agent": "Reviewer"})

    thread.join()

    if error_holder[0]:
        print(f"[generate_ovito] error: {error_holder[0]}")
        return {"ovito_script": None}

    result = result_holder[0]
    if result and result.review_passed:
        print(f"[generate_ovito] passed after {result.attempts} attempt(s)")
        return {"ovito_script": result.script}
    else:
        _q(state, {"type": "status", "message": "OVITO script failed review, falling back to default visualization"})
        print("[generate_ovito] failed all attempts, falling back to default")
        return {"ovito_script": None}


# ─── Analyst tools (used by analyze_results_node) ────────────────────────────
# Tool SCHEMAS are no longer hand-written — the @tool-decorated functions
# list_recent_runs / get_run_log (defined below with the Analyst agent) provide
# them automatically to create_react_agent. The _exec_* helpers below do the
# actual work and are called by those tools.


def _exec_list_runs() -> str:
    """Execute list_recent_runs: returns formatted string and emits run_list SSE event."""
    runs = run_manager.list_all_runs()
    if not runs:
        return "No simulation runs found on disk."

    recent = runs[:8]
    lines = []
    run_summaries = []

    for i, run_name in enumerate(recent, 1):
        folder_path = os.path.join(run_manager.RUNS_BASE_DIR, run_name)
        summary = run_manager.get_run_folder_summary(folder_path)
        meta = summary.get("metadata", {})
        material = meta.get("material", "unknown")
        status = meta.get("status", "unknown")
        timestamp = meta.get("timestamp", "")
        if timestamp:
            try:
                from datetime import datetime as _dt
                ts = _dt.fromisoformat(timestamp)
                timestamp_str = ts.strftime("%Y-%m-%d %H:%M")
            except Exception:
                timestamp_str = timestamp[:16]
        else:
            timestamp_str = ""
        details = " | ".join(filter(None, [
            f"Material: {material}" if material != "unknown" else None,
            f"Status: {status}" if status != "unknown" else None,
            timestamp_str,
        ]))
        lines.append(f"{i}. {run_name} — {details}")
        run_summaries.append({"name": run_name, "material": material, "status": status})

    return "\n".join(lines) + "\n\nTell the user: reply with 'analyze 1', 'analyze 2', etc. to analyze a run."


def _resolve_run_dir(run_name: str) -> str:
    """Resolve a run identifier to an absolute run-folder path.

    Accepts a folder name under runs/, a 1-based number matching the
    list_recent_runs ordering, 'latest', or a path that is already a directory
    (which is how nested campaign runs like runs/benchmark_*/run_* are reached).

    Returns "" if nothing resolves; callers supply their own error text.
    """
    if not run_name:
        return ""

    # Strip leading "analyze " prefix in case the LLM passes the full user phrase.
    clean = run_name.strip().lower().removeprefix("analyze").strip()
    run_name = clean if clean else run_name.strip()

    if run_name == "latest":
        return run_manager.get_latest_run() or ""

    if run_name.isdigit():
        idx = int(run_name) - 1
        all_runs = run_manager.list_all_runs()
        if 0 <= idx < len(all_runs):
            return os.path.abspath(os.path.join(run_manager.RUNS_BASE_DIR, all_runs[idx]))
        return ""

    # An existing path wins over a name lookup.
    if os.path.isdir(run_name):
        return os.path.abspath(run_name)

    candidate = os.path.abspath(os.path.join(run_manager.RUNS_BASE_DIR, run_name))
    return candidate if os.path.isdir(candidate) else ""


def _exec_get_run_log(run_name: str, state: SimulationState) -> str:
    """Execute get_run_log: resolves folder (name, number, or 'latest'), emits header, returns parsed log."""
    run_folder = _resolve_run_dir(run_name) or run_manager.get_latest_run()

    print(f"[get_run_log] run_name={run_name!r}  resolved_folder={run_folder}")

    if not run_folder:
        return "No simulation runs found."

    run_display = os.path.basename(run_folder)
    _q(state, {"type": "agent_content", "agent": "Analyst",
               "content": f"**Run:** {run_display}\n\n---\n", "message_type": "analysis"})

    log_path = os.path.join(run_folder, "log.lammps")
    if not os.path.exists(log_path):
        candidates = glob.glob(os.path.join(run_folder, "log.*"))
        log_path = candidates[0] if candidates else None

    if not log_path:
        return f"No log file found in {run_folder}."

    print(f"[analyze_results] reading log: {log_path}")
    with open(log_path, "r") as f:
        log_text = f.read()

    thermo_points = _parse_thermo_for_plot(log_text)
    if thermo_points:
        _q(state, {"type": "thermo_data", "data": thermo_points})
        print(f"[analyze_results] emitted {len(thermo_points)} thermo points for chart")

    _q(state, {"type": "load_trajectory", "run_folder": run_folder})

    return log_text


def _parse_thermo_for_plot(log_text: str) -> list:
    """Extract thermo rows from a LAMMPS log into a list of dicts for chart rendering."""
    points = []
    headers = None
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("Step") and any(k in line for k in ("Temp", "Press", "PotEng", "TotEng", "Enthalpy", "Volume")):
            headers = line.split()
            continue
        if headers:
            if line.startswith(("Loop", "Performance", "Total wall")):
                headers = None
                continue
            if not line or line.startswith(("WARNING", "Neighbor", "Setting", "Per MPI", "#", "LAMMPS")):
                continue
            parts = line.split()
            if parts and parts[0].lstrip("-").isdigit() and len(parts) == len(headers):
                try:
                    points.append({h.lower(): float(v) for h, v in zip(headers, parts)})
                except ValueError:
                    continue
    return points


# ─── analyze_results ─────────────────────────────────────────────────────────
#
# The Analyst is a real ReAct agent (create_react_agent), consistent with the
# orchestrator and HPC executor — NOT a hand-rolled fixed-round loop. This lets
# it chain tools naturally (e.g. list_recent_runs → get_run_log → analyze)
# instead of being capped at one tool call per turn.
#
# The two analyst tools (list_recent_runs / get_run_log) emit SSE side-effects
# (thermo chart + trajectory viewer) via _q. Because create_react_agent tools
# don't receive our SESSION_QUEUES-keyed state directly, the current session_id
# is stashed in a ContextVar for the duration of one analyst call so the tools
# can emit to the right browser queue.

import contextvars
from langchain_core.tools import tool as _lc_tool

_ANALYST_SESSION: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "analyst_session_id", default=None
)


def _analyst_q(event: dict):
    """Emit an SSE event to the session bound for the current analyst call."""
    sid = _ANALYST_SESSION.get()
    if sid and sid in SESSION_QUEUES:
        SESSION_QUEUES[sid].put(event)


@_lc_tool
def list_recent_runs() -> str:
    """List recent LAMMPS simulation runs with metadata (timestamp, status).

    Call this when the user wants to browse runs or has NOT specified which run
    to analyze. After this returns, ask the user to reply 'analyze 1', 'analyze
    2', etc. — do NOT fabricate an analysis without reading a run's log first.
    """
    print("[analyze_results] tool: list_recent_runs")
    return _exec_list_runs()


@_lc_tool
def get_run_log(run_name: str) -> str:
    """Read the thermodynamic log for a simulation run. Always call this before
    answering questions about simulation results (temperature, pressure, energy,
    equilibration) unless the log is already in the conversation.

    Args:
        run_name: Run folder name, OR a number ('1', '2', ...) matching the
                  list_recent_runs output, OR 'latest'.
    """
    print(f"[analyze_results] tool: get_run_log(run_name={run_name})")
    # _exec_get_run_log emits thermo_data + load_trajectory SSE events; give it
    # a lightweight state carrying session_id so those reach the browser.
    return _exec_get_run_log(run_name, {"session_id": _ANALYST_SESSION.get()})


@_lc_tool
def stress_strain_curve(run_name: str, axis: str = "y",
                        fit_max_strain: float = 0.01) -> str:
    """Compute the engineering stress-strain curve for a tensile-loading run.

    Call this whenever the user asks about Young's modulus, yield or peak
    stress, strain at failure, stiffness, or wants stress plotted against
    strain. Do NOT try to answer those from raw pressure numbers in the log —
    this tool derives strain from the box length and is the only correct source.

    Only works on runs that contain a `fix deform` loading phase; a pure
    equilibration run has no strain and the tool will say so.

    The plot is shown to the user automatically — interpret the curve in your
    reply rather than announcing that a plot exists.

    Args:
        run_name: Run folder name, OR a number ('1', '2', ...) matching the
                  list_recent_runs output, OR 'latest', OR a path to a run
                  folder (needed for campaign runs under runs/benchmark_*/).
        axis: Loading axis — 'x', 'y', or 'z'. Defaults to 'y'.
        fit_max_strain: Upper strain bound for the elastic (Young's modulus)
                        least-squares fit. Defaults to 0.01.
    """
    print(f"[analyze_results] tool: stress_strain_curve(run_name={run_name}, axis={axis})")
    try:
        return _exec_stress_strain(run_name, axis, fit_max_strain)
    except Exception as e:  # noqa: BLE001 — a raised tool aborts the agent turn
        return f"Stress-strain analysis failed: {e}"


def _exec_stress_strain(run_name: str, axis: str, fit_max_strain: float) -> str:
    """Run stress_strain.py's analysis on one run; write CSV+PNG, emit the plot.

    Reuses the root-level stress_strain module rather than reimplementing the
    thermo parsing / loading-phase detection / modulus fit.
    """
    # stress_strain.py lives at the repo root alongside viz_run.py and
    # kim_search.py. Append (never insert) so root scripts cannot shadow a
    # same-named module under src/.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    import stress_strain as ss

    axis = (axis or "y").strip().lower()
    if axis not in ss._AXIS_COLUMNS:
        return f"Invalid axis '{axis}'. Use 'x', 'y', or 'z'."
    press_col, length_col = ss._AXIS_COLUMNS[axis]

    run_folder = _resolve_run_dir(run_name)
    if not run_folder:
        return (f"Could not resolve a run folder from '{run_name}'. "
                "Call list_recent_runs and pass a number or folder name.")

    log_path = os.path.join(run_folder, "log.lammps")
    if not os.path.exists(log_path):
        candidates = glob.glob(os.path.join(run_folder, "log.*"))
        if not candidates:
            return f"No LAMMPS log file found in {run_folder}."
        log_path = candidates[0]

    blocks = ss.parse_thermo_blocks(log_path)
    if not blocks:
        return f"No thermo tables found in {log_path}."

    try:
        block = ss.pick_loading_block(blocks, length_col)
    except ValueError:
        return (f"This run has no tensile-loading phase: box length '{length_col}' "
                f"never changes, so there is no `fix deform` block along {axis}. "
                "Stress-strain is undefined here (it looks like an equilibration-only "
                f"run). If the pull was along a different axis, retry with that axis.")

    missing = [c for c in (press_col, length_col) if c not in block]
    if missing:
        return (f"Thermo output is missing column(s) {', '.join(missing)}. "
                f"Add them to thermo_style. Available: {', '.join(block)}.")

    strain, stress = ss.compute_stress_strain(block, press_col, length_col)
    modulus = ss.youngs_modulus(strain, stress, fit_max_strain)

    peak_stress = max(stress)
    peak_strain = strain[stress.index(peak_stress)]

    run_display = os.path.basename(os.path.normpath(run_folder))
    prefix = os.path.join(run_folder, "stress_strain")
    csv_path = prefix + ".csv"
    ss.write_csv(csv_path, block, strain, stress, length_col)
    png_path = ss.plot(prefix + ".png", strain, stress, modulus,
                       fit_max_strain)

    # Push the curve into chat. /runs is mounted statically by app.py and
    # agent_content is markdown-rendered, so an <img> needs no frontend change.
    # relpath (not basename) keeps nested campaign runs addressable.
    if png_path:
        try:
            rel = os.path.relpath(run_folder, os.path.abspath(run_manager.RUNS_BASE_DIR))
            if not rel.startswith(".."):
                _analyst_q({
                    "type": "agent_content", "agent": "Analyst",
                    "content": f"\n\n![Stress-strain curve](/runs/{rel}/stress_strain.png)\n\n",
                    "message_type": "analysis",
                })
        except ValueError:
            pass  # run folder on a different drive; skip the image, keep the numbers

    modulus_str = (f"{modulus:.1f} GPa (least-squares fit over strain <= {fit_max_strain})"
                   if modulus is not None else "n/a (not enough low-strain points)")

    return "\n".join([
        f"run:             {run_display}",
        f"loading axis:    {axis}  (stress = -{press_col}, strain from {length_col})",
        f"data points:     {len(strain)}  (loading phase only)",
        f"L0 ({length_col}):          {block[length_col][0]:.4f} Angstrom",
        f"max strain:      {max(strain):.4f}",
        f"Young's modulus: {modulus_str}",
        f"peak stress:     {peak_stress:.3f} GPa at strain {peak_strain:.4f}",
        f"csv:             {csv_path}",
        f"png:             {png_path or 'not written (matplotlib unavailable)'}",
    ])


_ANALYST_TOOLS = [list_recent_runs, get_run_log, stress_strain_curve]


def analyze_results_node(state: SimulationState) -> dict:
    """Analyst ReAct agent: lists runs or analyzes simulation results.

    Uses create_react_agent so tool calls chain naturally (list → read →
    analyze) with no fixed round cap. Streams tokens to the browser via _q.
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    session_id = state.get("session_id")
    _q(state, {"type": "agent_start", "agent": "Analyst"})

    model = ChatOpenAI(
        model=config.DEFAULT_MODEL,
        api_key=config.API_KEY,
        base_url=config.API_BASE_URL,
        **config.temp_kwargs(config.ANALYST_TEMPERATURE),
    )
    agent = create_react_agent(
        model=model,
        tools=_ANALYST_TOOLS,
        prompt=ANALYST_SYSTEM_PROMPT,
        name="chatmd_analyst",
    )

    # Rebuild prior turns so follow-ups ("was it equilibrated?") answer from
    # context without re-reading.
    history_msgs = []
    for m in state.get("conversation_history", []):
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            history_msgs.append(HumanMessage(content=content))
        elif role == "assistant" and content.strip():
            history_msgs.append(AIMessage(content=content))
    agent_input = {
        "messages": history_msgs + [HumanMessage(content=state.get("user_message", ""))]
    }

    # Bind session_id for the tools' SSE side-effects, then stream tokens.
    token = _ANALYST_SESSION.set(session_id)
    try:
        for chunk, _meta in agent.stream(
            agent_input,
            config={"recursion_limit": 12},
            stream_mode="messages",
        ):
            # Only stream final assistant text (AIMessage tokens); tool results
            # are surfaced via their own SSE side-effects.
            if isinstance(chunk, AIMessage) or getattr(chunk, "type", "") == "AIMessageChunk":
                tok = chunk.content if isinstance(chunk.content, str) else ""
                if tok:
                    _q(state, {"type": "agent_content", "agent": "Analyst",
                               "content": tok, "message_type": "analysis"})
    except Exception as e:
        print(f"[analyze_results] agent error: {e}")
        _q(state, {"type": "agent_content", "agent": "Analyst",
                   "content": f"Analysis error: {e}", "message_type": "analysis"})
    finally:
        _ANALYST_SESSION.reset(token)

    _q(state, {"type": "agent_complete", "agent": "Analyst"})
    print("[analyze_results] done")
    return {}
