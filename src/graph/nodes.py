# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

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
import subprocess
import textwrap
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

    iterative_generator = IterativeLammpsGeneration(max_attempts=3)
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

    Call this whenever the user asks about yield or peak stress, strain at
    failure, or wants stress plotted against strain. Do NOT try to answer
    those from raw pressure numbers in the log — this tool derives strain
    from the box length and is the only correct source.

    Only works on runs that contain a `fix deform` loading phase; a pure
    equilibration run has no strain and the tool will say so.

    The plot is shown to the user automatically — interpret the curve in your
    reply rather than announcing that a plot exists.

    Args:
        run_name: Run folder name, OR a number ('1', '2', ...) matching the
                  list_recent_runs output, OR 'latest', OR a path to a run
                  folder (needed for campaign runs under runs/benchmark_*/).
        axis: Loading axis — 'x', 'y', or 'z'. Defaults to 'y'.
        fit_max_strain: Upper strain bound for the elastic least-squares fit.
                        Defaults to 0.01. The fitted slope is not reported.
    """
    print(f"[analyze_results] tool: stress_strain_curve(run_name={run_name}, axis={axis})")
    try:
        return _exec_stress_strain(run_name, axis, fit_max_strain)
    except Exception as e:  # noqa: BLE001 — a raised tool aborts the agent turn
        return f"Stress-strain analysis failed: {e}"


_SS_CURVE_POINTS = 40


def _stress_curve_lines(strain, stress, npts=_SS_CURVE_POINTS):
    """The curve itself, coarse enough to read and cheap enough to send.

    Peak stress alone cannot say whether the material softened afterwards, how
    sharply, or whether the rise was linear — and the agent has no vision, so
    the PNG tells it nothing. Without these points it either stays silent on
    the shape or invents it.

    Block means rather than every Nth row: the per-point thermal scatter on
    run_2026_08_12_102150 is 0.019 GPa, so sampled raw points carry noise that
    averaging over a block removes. 40 points reproduces that run's peak to
    8.649 against a true 8.680, and puts 8 points through the post-peak
    collapse — enough to show it as a smooth acceleration rather than a cliff,
    which is the brittle/ductile distinction. 20 points leaves only 3 there and
    60 adds nothing 40 did not already show. Cost is ~115 tokens against
    ~20,000 for the raw 1001-row CSV.

    The window scales with the run, so this is 40 points whether the log holds
    200 thermo lines or 20000; a run shorter than npts passes its own points.
    """
    n = len(stress)
    if n < 4:
        return []
    k = max(1, n // npts)
    pairs = []
    for i in range(0, n, k):
        es, ss_ = strain[i:i + k], stress[i:i + k]
        pairs.append(f"{sum(es) / len(es):.4f},{sum(ss_) / len(ss_):.2f}")

    pad = " " * 17
    out = [
        f"curve:           {len(pairs)} points, block means over {k} thermo line(s) each.",
        f"{pad}Read SHAPE from these; quote peak stress from the line above.",
        f"{pad}Do NOT infer an elastic modulus from them — the lateral axes",
        f"{pad}are held fixed during loading.",
    ]
    out += [pad + line for line in textwrap.wrap(" ".join(pairs), 88)]
    return out


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

    # The elastic slope is deliberately NOT reported to the agent. It is still
    # computed above, but under this loading the lateral axes are held fixed,
    # so the slope is not Young's modulus and reporting it under that name
    # invites the agent to quote it as one.
    return "\n".join([
        f"run:             {run_display}",
        f"loading axis:    {axis}  (stress = -{press_col}, strain from {length_col})",
        f"data points:     {len(strain)}  (loading phase only)",
        f"L0 ({length_col}):          {block[length_col][0]:.4f} Angstrom",
        f"max strain:      {max(strain):.4f}",
        f"peak stress:     {peak_stress:.3f} GPa at strain {peak_strain:.4f}",
    ] + _stress_curve_lines(strain, stress) + [
        f"csv:             {csv_path}",
        f"png:             {png_path or 'not written (matplotlib unavailable)'}",
    ])


# ── DXA dislocation-density tool ─────────────────────────────────────────────
# Output layout matches dxa_analysis.py's CLI defaults exactly, so a CSV the
# user produced with `python dxa_analysis.py <run>` is picked up as our cache
# and, conversely, we never write anywhere the CLI would not look.
_DXA_SUBDIR = "dxa_analysis"
_DXA_CSV = "dxa_summary.csv"
_DXA_PNG = "dxa_density.png"

# Every frame is analyzed — no subsampling. Measured across this repo's runs:
# median 41 frames, max 63, at ~1.5 s/frame, so ~60-95 s once per run and
# instant from the cache thereafter. Striding would save ~40s on the first call
# and permanently cost onset resolution, which is the headline number here.
# This cap only guards the pathological case (a run dumping every 100 steps);
# past it we stop and say so rather than quietly analyzing a subset.
_DXA_MAX_FRAMES = 200

# LAMMPS `lattice` styles that have a DXA input-crystal-structure equivalent.
# sc/custom/none deliberately absent — DXA cannot take them.
_LAMMPS_TO_DXA_LATTICE = {
    "fcc": "FCC",
    "bcc": "BCC",
    "hcp": "HCP",
    "diamond": "CubicDiamond",
}

# The frame loop runs here, in its own process. OVITO cannot run in the server:
# see the module docstring of dxa_worker.py for the segfault that forced this.
_DXA_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dxa_worker.py")


def _dxa_import():
    """Import the root-level dxa_analysis module.

    Safe to do lazily inside the server: dxa_analysis pulls in nothing but
    stdlib and cna_adaptive at module level — every ovito and matplotlib import
    inside it is function-local. Append (never insert) so root scripts cannot
    shadow a same-named module under src/, same rule as _exec_stress_strain.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    import dxa_analysis as dxa
    return dxa


def _detect_lattice(run_folder: str):
    """Read the crystal structure out of the run's own input.lammps.

    Returns a DXA lattice name, or None if the file is missing or declares a
    style DXA has no equivalent for.

    The agent must not be the one choosing this. A wrong lattice does not
    raise — DXA simply finds almost nothing and returns a plausible-looking
    near-zero result, which is the worst kind of failure. Every run in this
    repo states the answer outright with a `lattice <style>` line.

    Only the style token is parsed. The lattice constant is nearly always a
    variable (`lattice fcc ${a0}`) and one run reads
    `lattice hcp ${a} orient x 1 0 0 ... c/a $(v_c/v_a)`, so everything after
    the style is ignored rather than interpreted.
    """
    path = os.path.join(run_folder, "input.lammps")
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0].lower() == "lattice":
                    return _LAMMPS_TO_DXA_LATTICE.get(parts[1].lower())
    except OSError:
        return None
    return None


def _dxa_rows_from_csv(csv_path: str):
    """Rebuild dxa_analysis's `rows` shape from an existing dxa_summary.csv.

    Returns (rows, families). `families` comes from the header order, which
    write_summary_csv wrote in family_columns() order — so rows reconstructed
    here re-plot identically to the ones that produced the file. That is what
    lets us redraw a deleted dxa_density.png without re-running DXA at all.

    Raises ValueError on an empty, header-only or foreign CSV; the caller turns
    that into tool text, since a raised tool aborts the agent turn.
    """
    import csv as _csv

    with open(csv_path, newline="") as fh:
        reader = _csv.DictReader(fh)
        header = reader.fieldnames or []
        records = list(reader)

    if "timestep" not in header or "density_m-2" not in header:
        raise ValueError("not a dxa_summary.csv (no timestep/density_m-2 columns)")
    if not records:
        raise ValueError("header only, no data rows")

    # 'len_1/6<112>_A' -> '1/6<112>'. Burgers families never contain '_A'.
    families = [c[4:-2] for c in header
                if c.startswith("len_") and c.endswith("_A")]
    count_cols = [c for c in header if c.startswith("count_")]

    def _f(rec, key):
        try:
            return float(rec.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    rows = []
    for rec in records:
        try:
            ts = int(float(rec["timestep"]))
        except (KeyError, TypeError, ValueError):
            continue  # tolerate a partial final line from an interrupted write
        rows.append({
            "timestep": ts,
            "phase": rec.get("phase") or "",
            "path": "",
            "dxa": {
                "total_length": _f(rec, "total_line_length_A"),
                "cell_volume": _f(rec, "cell_volume_A3"),
                # Note the hyphens: the CSV columns are density_A-2 / density_m-2.
                "density_A2": _f(rec, "density_A-2"),
                "density_m2": _f(rec, "density_m-2"),
                "n_lines": int(_f(rec, "n_lines")),
                "lengths": {f: _f(rec, f"len_{f}_A") for f in families},
                "counts": {c[6:]: int(_f(rec, c)) for c in count_cols},
                "segments": [],
            },
        })

    if not rows:
        raise ValueError("no parseable data rows")
    rows.sort(key=lambda r: r["timestep"])
    return rows, families


def _dxa_push_image(run_folder: str, png_path: str) -> bool:
    """Push dxa_density.png into the chat bubble.

    Returns False when the file is not reachable through app.py's /runs
    StaticFiles mount, so the caller can say so rather than let the agent
    narrate a plot the user cannot actually see.

    The ?v=<mtime> query is a cache-buster: this URL is stable across runs, so
    a regenerated plot would otherwise be masked by the browser's old copy.
    StaticFiles ignores the query and marked renders the markdown image as-is.
    """
    try:
        rel = os.path.relpath(run_folder, os.path.abspath(run_manager.RUNS_BASE_DIR))
    except ValueError:
        return False  # different drive; skip the image, keep the numbers
    if rel.startswith(".."):
        return False  # outside runs/, not served
    try:
        ver = int(os.path.getmtime(png_path))
    except OSError:
        ver = 0
    url = f"/runs/{rel.replace(os.sep, '/')}/{_DXA_SUBDIR}/{_DXA_PNG}?v={ver}"
    _analyst_q({
        "type": "agent_content", "agent": "Analyst",
        "content": f"\n\n![Dislocation density and line length by Burgers family]({url})\n\n",
        "message_type": "analysis",
    })
    return True


def _dxa_lattice_warning(rows, lattice: str) -> str:
    """Cross-check the analyzed lattice against what DXA actually identified.

    DXA only classifies the structure it was ASKED for (plus HCP, as FCC's
    stacking-fault partner); everything else lands in 'Other'. Measured on the
    BCC fixture: asked BCC -> 1024 BCC, asked FCC -> 1024 Other. So a wrong
    lattice does not show up as the true structure, it shows up as a pristine
    frame that is overwhelmingly 'Other' — which is itself the tell, since
    frame 0 of a built crystal should be almost entirely its own lattice.

    Catches a bad explicit override; the LAMMPS idiom where `lattice` is
    declared only to set units/spacing before a read_data of something else;
    and a cache poisoned by `python dxa_analysis.py <bcc_run>`, whose --lattice
    still defaults to FCC and which records no lattice in the CSV it writes.

    False positive to accept: a run whose frame 0 is genuinely disordered
    (melt, amorphous). Hence "may be wrong" rather than an assertion — and such
    a run has no dislocations to misreport anyway.
    """
    counts = rows[0]["dxa"].get("counts") or {}
    total = sum(counts.values())
    if total <= 0:
        return ""
    dominant = max(counts, key=counts.get)
    if dominant == lattice or counts[dominant] < 0.5 * total:
        return ""
    return (f"warning:           analyzed as {lattice}, but the first frame is "
            f"{100.0 * counts[dominant] / total:.0f}% {dominant} — the lattice may be "
            f"wrong, which would make these densities far too low.")


def _dxa_summary_text(dxa, rows, families, *, run_display, source,
                      csv_path, png_path, notes) -> str:
    """Format the scalar report.

    The agent has no vision, so the PNG tells it nothing: every claim it might
    make about nucleation, multiplication or deformation mechanism has to
    appear here as a number.
    """
    steps = [r["timestep"] for r in rows]
    dens = [r["dxa"]["density_m2"] for r in rows]
    peak_i = max(range(len(rows)), key=lambda i: dens[i])
    final = rows[-1]
    onset_i = next((i for i, d in enumerate(dens) if d > 0), None)

    def _fams(row, top=4):
        lens = row["dxa"]["lengths"]
        tot = sum(lens.values())
        ranked = sorted(((v, f) for f, v in lens.items() if v > 0), reverse=True)
        if not ranked or tot <= 0:
            return "none (no dislocation lines in this frame)"
        return ", ".join(f"{dxa.BURGERS_LABEL.get(f, f)} {100.0 * v / tot:.0f}% ({v:.0f} A)"
                         for v, f in ranked[:top])

    def _pct(row, name):
        counts = row["dxa"]["counts"]
        tot = sum(counts.values())
        return (100.0 * counts.get(name, 0) / tot) if tot else 0.0

    out = [
        f"run:               {run_display}",
        f"source:            {source}",
        f"frames:            {len(rows)}  (timesteps {steps[0]} to {steps[-1]}"
        + (f", spacing {steps[1] - steps[0]}" if len(steps) > 1 else "") + ")",
    ]

    if onset_i is None:
        out.append("nucleation:        NONE — dislocation density is exactly zero in "
                   "every frame. The crystal stayed defect-free; any deformation here "
                   "was elastic, twinning, or cleavage without dislocation activity.")
    else:
        on = rows[onset_i]
        out += [
            f"nucleation onset:  step {on['timestep']} (frame {onset_i + 1}/{len(rows)}"
            + (f", phase '{on['phase']}'" if on["phase"] else "")
            + f") — first frame with rho > 0: {dens[onset_i]:.3e} m^-2, "
              f"{on['dxa']['n_lines']} line(s)",
            f"defect-free for:   {100.0 * onset_i / len(rows):.0f}% of the frames "
            f"before onset",
            f"first families:    {_fams(on, top=3)}",
        ]

    out += [
        f"peak density:      {dens[peak_i]:.3e} m^-2 at step {steps[peak_i]} "
        f"({rows[peak_i]['dxa']['n_lines']} lines, "
        f"{rows[peak_i]['dxa']['total_length']:.0f} A total line length)",
        f"final density:     {dens[-1]:.3e} m^-2 at step {steps[-1]} "
        f"({final['dxa']['n_lines']} lines)",
    ]

    if onset_i is not None and dens[onset_i] > 0:
        span = steps[peak_i] - steps[onset_i]
        rate = (f", {(dens[peak_i] - dens[onset_i]) / span:.2e} m^-2 per step"
                if span > 0 else "")
        out.append(f"multiplication:    x{dens[peak_i] / dens[onset_i]:.0f} from onset "
                   f"to peak over {span} steps{rate}")

    if dens[peak_i] > 0:
        out.append("post-peak trend:   " + (
            "peak IS the last frame — density was still rising when the run ended"
            if peak_i == len(rows) - 1 else
            f"fell {100.0 * (1 - dens[-1] / dens[peak_i]):.0f}% from peak over the last "
            f"{steps[-1] - steps[peak_i]} steps (annihilation / absorption at free "
            f"surfaces)"))
        out += [
            f"families @ peak:   {_fams(rows[peak_i])}",
            f"families @ final:  {_fams(final)}",
        ]

    out += [
        f"structure @ final: FCC {_pct(final, 'FCC'):.2f}%, HCP {_pct(final, 'HCP'):.2f}% "
        f"(stacking-fault atoms), Other {_pct(final, 'Other'):.2f}% "
        f"(free surfaces, crack faces, dislocation cores)",
        f"csv:               {csv_path}",
        f"png:               {png_path or 'not written'}",
    ]
    return "\n".join(out + notes)


def _dxa_run_worker(run_folder, lattice, csv_path, png_path, n_frames, run_display):
    """Run the DXA frame loop in a child process, streaming progress to the UI.

    Returns (notes, error). `error` is prose to hand straight back to the agent
    when the run produced no CSV; None when it did. Nothing raises: a tool that
    raises aborts the agent turn.

    Why a subprocess at all: OVITO segfaults when called from a thread other
    than the one that initialized it, and this tool runs on a LangGraph
    thread-pool worker. dxa_worker.py's docstring has the crash report. The
    practical payoff is that the failure is now recoverable — the child dies,
    the server keeps serving, and the agent gets a sentence it can relay.

    stderr is merged into stdout rather than read separately: OVITO is chatty,
    and draining two pipes in sequence deadlocks as soon as the one not being
    read fills its 64 KB buffer. Non-JSON lines are kept as diagnostics.
    """
    cmd = [sys.executable, _DXA_WORKER, run_folder,
           "--lattice", lattice, "--csv", csv_path, "--png", png_path]
    notes, chatter, worker_error = [], [], None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",   # stray non-UTF-8 bytes -> "?" rather than a crash
            bufsize=1,          # line-buffered, so progress arrives live
            close_fds=True,     # don't hand the server's sockets to the child
        )
    except OSError as e:
        return notes, f"Could not start the DXA worker process: {e}"

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            if len(chatter) < 40:       # bounded: 53 frames of OVITO output
                chatter.append(line)
            continue
        kind = ev.get("type")
        if kind == "progress":
            _analyst_q({"type": "status",
                        "message": (f"DXA {run_display}: {ev['i']}/{n_frames} frames "
                                    f"— step {ev['ts']}, rho={ev['rho']:.2e} m^-2")})
        elif kind == "skip":
            notes.append(f"note:              skipped {ev['file']} "
                         f"(DXA error: {ev['error']})")
        elif kind == "error":
            worker_error = ev.get("message", "")
    proc.wait()

    # Trust the artifact, not the exit code: OVITO can abort the process after
    # write_summary_csv has already succeeded.
    if os.path.exists(csv_path):
        return notes, None

    msg = worker_error or ""
    if msg.startswith("ovito-missing:"):
        detail = (f"OVITO is not importable in the worker ({msg.split(':', 1)[1].strip()}). "
                  f"Run `python dxa_analysis.py {run_folder}` where OVITO is available; "
                  f"this tool will then read the cached CSV without needing OVITO itself.")
    elif msg.startswith("first-frame:"):
        detail = (f"DXA failed on the first frame — {msg.split(':', 1)[1].strip()}. The "
                  f"dump is likely truncated or missing the x/y/z columns DXA needs.")
    elif msg.startswith("csv-write:"):
        detail = f"DXA ran but could not write its CSV: {msg.split(':', 1)[1].strip()}"
    elif proc.returncode is not None and proc.returncode < 0:
        sig = -proc.returncode
        detail = (f"the worker process was killed by signal {sig}"
                  + (" (segfault inside OVITO)" if sig == 11 else "")
                  + f" after {n_frames} frames were queued")
    else:
        detail = f"the worker exited with code {proc.returncode}"

    tail = " | ".join(chatter[-3:])
    return notes, (f"DXA did not complete for {run_display}: {detail.rstrip('.')}."
                   + (f" Last worker output: {tail}" if tail else "")
                   + " The chat server is unaffected — DXA runs in a separate process.")


def _exec_dxa(run_name: str, lattice: str) -> str:
    """Dislocation density per frame for one run: cached first, compute if not.

    Composes dxa_analysis's pieces (collect_frames / analyze_frame /
    family_columns / write_summary_csv / plot_density) rather than calling its
    main(), which prints instead of returning and raises SystemExit(2) on bad
    input — a BaseException that would escape the tool wrapper's except clause
    and abort the agent turn.

    render_snapshot is deliberately unused: it mutates the global OVITO scene
    via add_to_scene/remove_from_scene and is not safe inside the server.
    """
    from datetime import datetime as _dt

    dxa = _dxa_import()

    run_folder = _resolve_run_dir(run_name)
    if not run_folder:
        return (f"Could not resolve a run folder from '{run_name}'. "
                "Call list_recent_runs and pass a number or folder name.")
    run_display = os.path.basename(os.path.normpath(run_folder))

    notes = []
    # Validate up front: an unknown name would otherwise reach _lattice_enum's
    # getattr and raise per-frame, mid-loop, after status events have fired.
    lattice = (lattice or "auto").strip()
    if lattice.lower() == "auto":
        detected = _detect_lattice(run_folder)
        if detected:
            lattice = detected
            lattice_src = "auto-detected from input.lammps"
        else:
            lattice = "FCC"
            lattice_src = "ASSUMED — no usable `lattice` line in input.lammps"
            notes.append("note:              the run's input.lammps does not declare a "
                         "lattice style DXA understands, so FCC was assumed. Say so if "
                         "you report these numbers.")
    elif lattice in dxa.LATTICES:
        lattice_src = "explicitly requested"
    else:
        return (f"Invalid lattice '{lattice}'. Use one of: {', '.join(dxa.LATTICES)}, "
                f"or 'auto' to read it from the run's input.lammps. FCC is right for "
                f"Cu, Al, Ni, Au, Ag; BCC for Fe, W, Mo, Ta.")

    out_dir = os.path.join(run_folder, _DXA_SUBDIR)
    csv_path = os.path.join(out_dir, _DXA_CSV)
    png_path = os.path.join(out_dir, _DXA_PNG)

    if os.path.exists(csv_path):
        # ---- cached -------------------------------------------------------
        # Never recompute over an existing CSV: it may be one the user produced
        # from the CLI, and DXA costs ~1.5 s/frame.
        try:
            rows, families = _dxa_rows_from_csv(csv_path)
        except (ValueError, OSError) as e:
            return (f"Found {csv_path} but could not read it ({e}) — it is empty or was "
                    f"truncated by an interrupted run. Delete that file and ask again to "
                    f"recompute, or run `python dxa_analysis.py {run_folder}` from the "
                    f"repo root.")
        stamp = _dt.fromtimestamp(os.path.getmtime(csv_path)).strftime("%Y-%m-%d %H:%M")
        source = f"cached CSV (written {stamp}) — DXA not re-run"
        # The CSV records neither the lattice nor the frame selection that
        # produced it, so do not echo either as if it were fact.
        if lattice_src == "explicitly requested":
            notes.append(f"note:              lattice='{lattice}' was requested, but the "
                         f"cached CSV does not record which lattice produced it. The "
                         f"numbers above are whatever the cache holds.")
    else:
        # ---- compute ------------------------------------------------------
        frames = dxa.collect_frames(run_folder)
        if not frames:
            return (f"No LAMMPS dump files found in {run_folder}. DXA reads atomic "
                    f"positions from the per-frame `dump.*` trajectory files; this run "
                    f"has only thermo output, so dislocation density cannot be computed. "
                    f"Use get_run_log for what this run does contain.")
        if len(frames) > _DXA_MAX_FRAMES:
            return (f"{run_display} has {len(frames)} trajectory frames, more than this "
                    f"tool will analyze inline ({_DXA_MAX_FRAMES}) — at ~1.5 s/frame that "
                    f"is over {len(frames) * 1.5 / 60:.0f} minutes. Run `python "
                    f"dxa_analysis.py {run_folder} --stride N` from the repo root, then "
                    f"ask again and this tool will read the result instantly.")

        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            return f"Cannot create the output folder {out_dir}: {e}"

        # Emit one status before the worker starts so the bar moves immediately;
        # the per-frame ones come back over the worker's stdout.
        _analyst_q({"type": "status",
                    "message": f"DXA {run_display}: 0/{len(frames)} frames"})

        worker_notes, err = _dxa_run_worker(run_folder, lattice, csv_path, png_path,
                                            len(frames), run_display)
        notes.extend(worker_notes)
        if err:
            return err

        # Read back what the worker wrote. This is the same parser the cached
        # branch uses, so both branches now converge on one shape and the plot
        # and summary cannot drift between them.
        try:
            rows, families = _dxa_rows_from_csv(csv_path)
        except (ValueError, OSError) as e:
            return (f"The DXA worker finished but its CSV could not be read ({e}). "
                    f"Delete {csv_path} and ask again to recompute.")
        source = (f"computed now — OVITO DXA over {len(rows)} frames in a separate "
                  f"process, lattice={lattice} ({lattice_src})")
        # Otherwise the last per-frame line lingers in a status bar shared with
        # the orchestrator.
        _analyst_q({"type": "status",
                    "message": f"DXA {run_display}: complete, {len(rows)} frames"})

    warning = _dxa_lattice_warning(rows, lattice)
    if warning:
        notes.append(warning)

    # Redraw whenever the PNG is missing, on BOTH branches — rows rebuilt from
    # the CSV replot identically, so a deleted image costs a second of
    # matplotlib instead of a full DXA recompute.
    png_out = png_path if os.path.exists(png_path) else None
    if png_out is None:
        try:
            png_out = dxa.plot_density(png_path, rows, families)
        except Exception as e:  # noqa: BLE001
            png_out = None
            notes.append(f"note:              plot failed ({e}); the numbers above are "
                         f"unaffected.")
        else:
            if png_out is None:
                notes.append("note:              matplotlib is unavailable, so no density "
                             "plot was written. The numbers above are complete — do NOT "
                             "tell the user to look at a plot.")

    if png_out and not _dxa_push_image(run_folder, png_out):
        notes.append("note:              this run is outside runs/, so the plot is not "
                     "served to the browser and the user CANNOT see it. Report the "
                     "numbers and mention the file path.")

    return _dxa_summary_text(dxa, rows, families, run_display=run_display,
                             source=source, csv_path=csv_path, png_path=png_out,
                             notes=notes)


@_lc_tool
def dxa_dislocation_density(run_name: str, lattice: str = "auto") -> str:
    """Dislocation density vs time from the trajectory, via OVITO's DXA.

    Call this for any question about dislocations, plasticity mechanism,
    dislocation nucleation or multiplication, partials, Shockley/stair-rod
    content, stacking faults, work hardening, or "how did this actually
    deform". It reports dislocation density in m^-2 per frame, the timestep
    where dislocations first nucleated, the peak density, and line length split
    by Burgers-vector family.

    Pairs with stress_strain_curve: that gives the mechanical response, this
    gives the defect mechanism underneath it. For a crack or tensile run call
    both — yield in the stress curve should coincide with nucleation onset here.

    Requires per-frame `dump.*` trajectory files; it cannot work from the thermo
    log. Results are cached in <run>/dxa_analysis/dxa_summary.csv, so repeat
    calls are instant, but the first call on a run analyzes every frame and can
    take a minute.

    The density plot is shown to the user automatically — interpret the numbers
    in your reply rather than announcing that a plot exists.

    Args:
        run_name: Run folder name, OR a number ('1', '2', ...) matching the
                  list_recent_runs output, OR 'latest', OR a path to a run
                  folder (needed for campaign runs under runs/benchmark_*/).
        lattice: Leave as 'auto'. The crystal structure is read from the run's
                 own input.lammps; guessing it from the element name is the
                 mistake the auto-detection exists to prevent.
    """
    print(f"[analyze_results] tool: dxa_dislocation_density("
          f"run_name={run_name}, lattice={lattice})")
    try:
        return _exec_dxa(run_name, lattice)
    except Exception as e:  # noqa: BLE001 — a raised tool aborts the agent turn
        return f"Dislocation (DXA) analysis failed: {e}"


_ANALYST_TOOLS = [list_recent_runs, get_run_log, stress_strain_curve,
                  dxa_dislocation_density]


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
            # A crack run legitimately wants get_run_log + stress_strain_curve
            # + dxa_dislocation_density; 12 allows only ~5 tool calls.
            config={"recursion_limit": 16},
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
