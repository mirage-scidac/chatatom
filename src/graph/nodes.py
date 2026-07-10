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

_LIST_RUNS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_recent_runs",
        "description": "List recent LAMMPS simulation runs with metadata (timestamp, status).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_GET_RUN_LOG_TOOL = {
    "type": "function",
    "function": {
        "name": "get_run_log",
        "description": (
            "Read the thermodynamic log for a simulation run. "
            "Always call this before answering questions about simulation results, "
            "temperature, pressure, energy, or equilibration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "run_name": {
                    "type": "string",
                    "description": (
                        "Run folder name, OR a number ('1', '2', ...) to pick from "
                        "the numbered list shown by list_recent_runs, OR 'latest'."
                    ),
                }
            },
            "required": ["run_name"],
        },
    },
}


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


def _exec_get_run_log(run_name: str, state: SimulationState) -> str:
    """Execute get_run_log: resolves folder (name, number, or 'latest'), emits header, returns parsed log."""
    # Strip leading "analyze " prefix in case LLM passes the full user phrase
    clean = run_name.strip().lower().removeprefix("analyze").strip()
    run_name = clean if clean else run_name.strip()

    if run_name == "latest":
        run_folder = run_manager.get_latest_run()
    elif run_name.strip().isdigit():
        idx = int(run_name.strip()) - 1
        all_runs = run_manager.list_all_runs()
        if 0 <= idx < len(all_runs):
            run_folder = os.path.abspath(os.path.join(run_manager.RUNS_BASE_DIR, all_runs[idx]))
        else:
            run_folder = run_manager.get_latest_run()
    else:
        candidate = os.path.abspath(os.path.join(run_manager.RUNS_BASE_DIR, run_name))
        run_folder = candidate if os.path.isdir(candidate) else run_manager.get_latest_run()

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

def analyze_results_node(state: SimulationState) -> dict:
    """Tool-calling agent: lists runs or analyzes simulation results.

    Streams throughout to avoid non-streaming round-trips that may hang on
    proxy LLM endpoints.
    """
    _q(state, {"type": "agent_start", "agent": "Analyst"})

    messages = [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        *state.get("conversation_history", []),
        {"role": "user", "content": state.get("user_message", "")},
    ]

    # ── Round 1: stream with tools — detect if LLM calls a tool ─────────────
    try:
        stream1 = client.chat.completions.create(
            model=config.DEFAULT_MODEL,
            messages=messages,
            tools=[_LIST_RUNS_TOOL, _GET_RUN_LOG_TOOL],
            tool_choice="auto",
            temperature=0.3,
            stream=True,
        )
    except Exception as e:
        print(f"[analyze_results] API error round 1: {e}")
        _q(state, {"type": "agent_content", "agent": "Analyst",
                   "content": f"Error connecting to LLM: {e}", "message_type": "analysis"})
        _q(state, {"type": "agent_complete", "agent": "Analyst"})
        return {}

    tool_calls_acc = {}   # index → {id, name, arguments}
    text_chunks = []

    for chunk in stream1:
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function and tc_delta.function.name:
                    tool_calls_acc[idx]["name"] = tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

        if delta.content:
            text_chunks.append(delta.content)

    if tool_calls_acc:
        reconstructed = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls_acc.values()
        ]
        messages.append({"role": "assistant", "tool_calls": reconstructed})

        for tc in tool_calls_acc.values():
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            if tc["name"] == "list_recent_runs":
                print("[analyze_results] tool: list_recent_runs")
                result = _exec_list_runs()
            elif tc["name"] == "get_run_log":
                run_name = args.get("run_name", "latest")
                print(f"[analyze_results] tool: get_run_log(run_name={run_name})")
                result = _exec_get_run_log(run_name, state)
            else:
                result = f"Unknown tool: {tc['name']}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        try:
            stream2 = client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
                stream=True,
            )
            for chunk in stream2:
                token = chunk.choices[0].delta.content
                if token:
                    _q(state, {"type": "agent_content", "agent": "Analyst",
                               "content": token, "message_type": "analysis"})
        except Exception as e:
            print(f"[analyze_results] API error round 2: {e}")
            _q(state, {"type": "agent_content", "agent": "Analyst",
                       "content": f"Analysis error: {e}", "message_type": "analysis"})

    else:
        for token in text_chunks:
            _q(state, {"type": "agent_content", "agent": "Analyst",
                       "content": token, "message_type": "analysis"})

    _q(state, {"type": "agent_complete", "agent": "Analyst"})
    print("[analyze_results] done")
    return {}
