"""
LangGraph node functions for the ChatMD simulation pipeline.

Each node takes SimulationState and returns a partial dict of updated fields.
SSE events are put onto state["sse_queue"] so the app.py SSE generator can
forward them to the browser.
"""
import json
import queue
import threading
from typing import Dict
from openai import OpenAI

# Global registry: session_id → queue.Queue
# Keeps sse_queue out of LangGraph state (not serializable by SqliteSaver)
SESSION_QUEUES: Dict[str, queue.Queue] = {}

# Campaign completion signaling: chat_session_id → threading.Event
# campaign_executor_node creates the event; CampaignManager sets it when
# campaign_complete fires; app.py waits on it before closing the SSE stream.
# Lives here (not in app.py) so worker threads can access it without
# triggering app.py's top-level dspy.configure() side effects.
CAMPAIGN_COMPLETION_EVENTS: Dict[str, threading.Event] = {}

import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import kim_utils
import run_manager
from agents import (
    SmartOrchestrator,
    IterativeLammpsGeneration,
    IterativeOVITOGeneration,
    detect_precomputed_properties,
)
from prompts import _build_orchestrator_prompt, ANALYST_SYSTEM_PROMPT
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


# ── Tool definitions ─────────────────────────────────────────────────────────

_KIM_TOOL = {
    "type": "function",
    "function": {
        "name": "get_kim_models",
        "description": "Fetch available KIM interatomic potentials for a given element from the registry.",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Chemical symbol of the element, e.g. 'Cu', 'Al', 'Fe'"
                }
            },
            "required": ["element"]
        }
    }
}



def _handle_kim_tool_call(element: str) -> str:
    """Execute the get_kim_models tool and return result as a string."""
    print(f"[chat_node] tool call: get_kim_models(element={element})")
    result = kim_utils.build_registry_context_for_command(element)
    if result:
        return result
    return f"No KIM models found in local registry for element '{element}'."


# ─── Node 1: chat ────────────────────────────────────────────────────────────


def chat_node(state: SimulationState) -> dict:
    """Conversational agent — first node.

    Responds naturally, streams reply to user, and can call get_kim_models()
    as a real tool so the LLM never hallucinates JSON tool-call text.

    Short-circuits for analysis/list-runs requests so the orchestrator
    never refuses them — analyze_results_node handles those turns.
    """
    _q(state, {"type": "agent_start", "agent": "Orchestrator"})

    # Pass session_id + user_message + history so the orchestrator prompt can
    # inject the campaign addendum when in campaign context.
    messages = [{"role": "system", "content": _build_orchestrator_prompt(
        session_id=state.get("session_id"),
        user_message=state.get("user_message", ""),
        conversation_history=state.get("conversation_history"),
    )}]

    trimmed = state.get("conversation_history", [])[-(config.MAX_CONVERSATION_TURNS * 2):]
    for msg in trimmed:
        messages.append({"role": msg["role"], "content": msg["content"]})

    current_message = state["user_message"]

    # Inject previous simulation context if available in checkpoint
    prev_material = state.get("material")
    prev_temp     = state.get("temperature")
    prev_script   = state.get("lammps_script")
    if prev_material and prev_script:
        prev_kim = state.get("kim_model") or "unknown"
        current_message += (
            f"\n\n[PREVIOUS SESSION CONTEXT: Last simulation was {prev_material}"
            f" at {prev_temp}K using KIM model {prev_kim}.\n\n"
            f"LAMMPS script used:\n```\n{prev_script}\n```\n"
            f"The user may be referring to this.]"
        )

    messages.append({"role": "user", "content": current_message})

    registry_context = state.get("registry_context") or ""
    response_chunks = []

    # Agentic loop: stream first, handle tool calls only if they appear.
    # Avoids a wasted non-streaming round-trip on turns with no tool call.
    while True:
        stream = client.chat.completions.create(
            model=config.ORCHESTRATOR_MODEL or config.DEFAULT_MODEL,
            messages=messages,
            temperature=config.ORCHESTRATOR_TEMPERATURE,
            tools=[_KIM_TOOL],
            tool_choice="auto",
            stream=True,
        )

        # Collect streamed chunks — stream text tokens live, accumulate tool call fragments
        tool_calls_acc = {}   # index -> {id, name, arguments}
        text_chunks = []

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # Accumulate tool call argument fragments (arrive across multiple chunks)
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

            # Stream text tokens directly to the frontend
            if delta.content:
                text_chunks.append(delta.content)
                _q(state, {
                    "type": "agent_content",
                    "agent": "Orchestrator",
                    "content": delta.content,
                    "message_type": "analysis",
                })

        # ── Tool call branch ──────────────────────────────────────────────────
        if tool_calls_acc:
            reconstructed_tool_calls = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_acc.values()
            ]
            messages.append({"role": "assistant", "tool_calls": reconstructed_tool_calls})

            for tc in tool_calls_acc.values():
                if tc["name"] == "get_kim_models":
                    args = json.loads(tc["arguments"])
                    element = args.get("element", "")
                    tool_result = _handle_kim_tool_call(element)
                    registry_context = tool_result
                else:
                    tool_result = f"Unknown tool: {tc['name']}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            # Loop back for the follow-up response (now with registry context)
            continue

        # ── No tool call — text already streamed, done ────────────────────────
        response_chunks = text_chunks
        break

    analysis = "".join(response_chunks)
    _q(state, {"type": "agent_complete", "agent": "Orchestrator"})
    print(f"[chat_node] response length={len(analysis)}  registry_fetched={'yes' if registry_context else 'no'}")

    # ── Campaign mode: parse CAMPAIGN PARAMETERS: block (if present) ────────
    # If the orchestrator emitted a structured CAMPAIGN PARAMETERS: block,
    # extract n_iterations/bounds/target and persist to state. These survive
    # in the SqliteSaver checkpoint so the next turn (user says "yes") can
    # be classified as campaign_start and campaign_executor can read them.
    campaign_fields = _parse_campaign_parameters_block(analysis)

    result = {"analysis": analysis, "registry_context": registry_context}
    if campaign_fields:
        # Remember the original user message that started this campaign setup,
        # so iteration 1's synthesized prompt is the user's natural language.
        if "campaign_original_message" not in campaign_fields:
            # Walk backwards through history to find the FIRST user msg that
            # mentioned "campaign" (most reliable indicator)
            from benchmark.campaign_prompts import CAMPAIGN_TRIGGER_PHRASES
            history = state.get("conversation_history") or []
            for msg in history:
                if msg.get("role") == "user":
                    content_lower = (msg.get("content") or "").lower()
                    if any(p in content_lower for p in CAMPAIGN_TRIGGER_PHRASES):
                        campaign_fields["campaign_original_message"] = msg["content"]
                        break
            # Fallback: current user message
            if "campaign_original_message" not in campaign_fields:
                cur = state.get("user_message", "")
                cur_lower = cur.lower()
                if any(p in cur_lower for p in CAMPAIGN_TRIGGER_PHRASES):
                    campaign_fields["campaign_original_message"] = cur

        result.update(campaign_fields)
        print(f"[chat_node] parsed CAMPAIGN PARAMETERS block: "
              f"n_iter={campaign_fields.get('campaign_n_iterations')}, "
              f"target={campaign_fields.get('campaign_target')}")

    return result


def _parse_campaign_parameters_block(text: str) -> dict:
    """Extract campaign settings from the orchestrator's CAMPAIGN PARAMETERS: block.

    Looks for a block of the form:
        CAMPAIGN PARAMETERS:
        n_iterations: 20
        strain_rate_range: [1e-3, 2e-2]
        temperature_range: [100, 600]
        crack_length_frac_range: [0.1, 0.6]
        crack_width_a0_range: [2, 8]
        target: minimize delta_a

    Returns a dict with keys campaign_n_iterations, campaign_target,
    campaign_bounds (only the keys actually present). Empty dict if the
    block is not found.
    """
    import re

    if "CAMPAIGN PARAMETERS:" not in text:
        return {}

    # Take just the block (until blank line or end of text)
    block_start = text.index("CAMPAIGN PARAMETERS:")
    block = text[block_start:]
    # Stop at the first double-newline after the header (end of block)
    end_match = re.search(r"\n\s*\n", block)
    if end_match:
        block = block[:end_match.start()]

    fields: dict = {}

    # n_iterations: 20
    m = re.search(r"n_iterations\s*:\s*(\d+)", block)
    if m:
        fields["campaign_n_iterations"] = int(m.group(1))

    # target: minimize delta_a
    m = re.search(r"target\s*:\s*(.+?)(?:\n|$)", block)
    if m:
        fields["campaign_target"] = m.group(1).strip()

    # Knob ranges: e.g., "strain_rate_range: [1e-3, 2e-2]"
    bounds = {}
    for knob in ("strain_rate", "temperature", "crack_length_frac", "crack_width_a0"):
        m = re.search(
            rf"{knob}_range\s*:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]",
            block,
        )
        if m:
            bounds[knob] = [float(m.group(1)), float(m.group(2))]
    if bounds:
        fields["campaign_bounds"] = bounds

    # Locked template fields — extract whichever ones the agent emitted.
    # Stored as a string dict so the iteration prompts can include them verbatim.
    template: dict = {}

    # Dynamically discover any phase_<N> keys in the block (phase_1, phase_2,
    # ..., phase_5, phase_10, etc.). Sort numerically so phase_10 comes after
    # phase_2, not before. Handles bullets, dashes, varying whitespace.
    phase_keys = sorted(
        set(re.findall(r"^[\s\-•*]*(phase_\d+)\s*[:=]", block, re.MULTILINE | re.IGNORECASE)),
        key=lambda k: int(k.split("_")[1]),
    )

    template_keys = [
        "material", "system_size", "kim_model", "timestep_ps",
        *phase_keys,                                   # dynamically discovered
        "dump_freq_steps", "cna_cutoff_a",
        "ovito_colors", "camera",
    ]

    for template_key in template_keys:
        m = re.search(rf"{template_key}\s*:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
        if m:
            template[template_key] = m.group(1).strip()
    if template:
        fields["campaign_template"] = template

    return fields


# ─── Node 2: intent ───────────────────────────────────────────────────────────

def intent_node(state: SimulationState) -> dict:
    """DSPy intent classification — classifies the user's own message, not the orchestrator response."""
    if state.get("workflow_stage") == "pbs_pending":
        return {}  # routing already handles pbs_pending; don't let classifier overwrite it

    # TEMP: short-circuit disabled — testing whether DSPy classify_intent handles
    # explicit analysis requests correctly on its own. Restore if misrouting returns.
    # _msg_lower = state["user_message"].strip().lower()
    # if any(kw in _msg_lower for kw in {"analyze", "analysis", "interpret", "list runs", "show runs"}):
    #     print(f"[intent_node] explicit analysis keyword → forcing analyze")
    #     return {"intent": "analyze"}

    # NOTE: Removed bare-affirmative HPC shortcut (used to force "yes/ok/sure" → hpc_status
    # when hpc_jobs existed in state). Made obsolete by [Agent]-tagged conversation history
    # + multi-turn classifier rule in ClassifyUserIntent — DSPy now correctly routes
    # short replies like "yes" based on which agent asked the clarifying question.

    orchestrator = SmartOrchestrator()
    result = orchestrator.classify_intent(state["user_message"], state["workflow_stage"], state.get("conversation_history", []))
    print(f"[intent_node] intent={result['intent']}  confidence={result['confidence']}")

    # Hard guard: campaign_start MUST have a CAMPAIGN PARAMETERS block visible
    # in conversation_history (LLM classifier sometimes ignores this rule).
    # Without the block, campaign_n_iterations isn't set and the campaign would
    # launch with default values. Demote to chat_agent so Phase 2 can fire.
    if result["intent"] == "campaign_start":
        history = state.get("conversation_history") or []
        block_seen = False
        for msg in reversed(history):
            if msg.get("role") == "assistant" and "CAMPAIGN PARAMETERS:" in (msg.get("content") or ""):
                block_seen = True
                break
        if not block_seen:
            print("[intent_node] WARNING: campaign_start without CAMPAIGN PARAMETERS block in history — demoting to chat_agent so Phase 2 can fire")
            return {"intent": "modify"}  # routes to chat_agent

    if result["intent"] == "execute_hpc":
        return {"intent": "execute_hpc", "execution_target": "hpc", "workflow_stage": "script_ready"}
    return {"intent": result["intent"]}


# ─── Node 4: extract_params ──────────────────────────────────────────────────

def extract_params_node(state: SimulationState) -> dict:
    _q(state, {"type": "status", "message": "Extracting parameters..."})

    # ── Campaign setup short-circuit ────────────────────────────────────────
    # If chat_node just emitted a CAMPAIGN PARAMETERS: block, this is a
    # campaign-mode SETUP turn (not a simulation request). Skip extraction
    # entirely so we don't trigger generate_lammps + scripts_ready_for_review
    # — the script-approval card during setup confused users into accidentally
    # spawning a standalone (non-campaign) simulation outside the campaign folder.
    analysis = state.get("analysis") or ""
    if "CAMPAIGN PARAMETERS:" in analysis:
        print("[extract_params] skipping — chat response is a CAMPAIGN PARAMETERS block (setup turn)")
        return {"has_parameters": False}

    print(f"[extract_params] analysis text: {state['analysis']}")
    orchestrator = SmartOrchestrator()
    params = orchestrator.extract_parameters(state["analysis"])

    print(f"[extract_params] has_parameters={params['has_parameters']}  material={params.get('material')}")

    # If no simulation parameters were found (e.g. user said "hi" or asked a question),
    # only write has_parameters=False. Do NOT overwrite material/temperature/lammps_script
    # etc. in the checkpoint — those belong to the previous simulation and must survive.
    if not params["has_parameters"]:
        return {"has_parameters": False}

    return {
        "has_parameters":      True,
        "material":            params.get("material"),
        "kim_model":           params.get("kim_model"),
        "temperature":         params.get("temperature"),
        "ensemble":            params.get("ensemble"),
        "duration_ps":         params.get("duration_ps"),
        "system_size":         params.get("system_size"),
        "visualization_notes": params.get("visualization_notes"),
        "camera_angle":        params.get("camera_angle", "default"),
    }



# ─── Node 6: generate_lammps ─────────────────────────────────────────────────

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


# ─── Node 7: detect_viz ──────────────────────────────────────────────────────

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

    orchestrator = SmartOrchestrator()
    has_custom_viz = orchestrator.detect_visualization(viz_request)
    print(f"[detect_viz] has_custom_viz={has_custom_viz}")

    return {
        "viz_request":           viz_request,
        "precomputed_properties": precomputed,
        "simulation_context":    simulation_context,
        "has_custom_viz":        has_custom_viz,
    }


# ─── Node 8: generate_ovito ──────────────────────────────────────────────────

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

    # Forward OVITO progress events to sse_queue
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
        # Non-fatal: fall back to no custom viz
        return {"ovito_script": None}

    result = result_holder[0]
    if result and result.review_passed:
        print(f"[generate_ovito] passed after {result.attempts} attempt(s)")
        return {"ovito_script": result.script}
    else:
        _q(state, {"type": "status", "message": "OVITO script failed review, falling back to default visualization"})
        print("[generate_ovito] failed all attempts, falling back to default")
        return {"ovito_script": None}


# ── Analyst tools ────────────────────────────────────────────────────────────

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
        # User picked by number (e.g. "1", "2") from the listed options
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



# ─── Node 9: analyze_results ─────────────────────────────────────────────────

def analyze_results_node(state: SimulationState) -> dict:
    """Tool-calling agent: lists runs or analyzes simulation results.

    Uses the same streaming-throughout pattern as chat_node to avoid
    non-streaming round-trips that may hang on the custom API proxy.
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

        # Accumulate tool call argument fragments
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

        # Collect text (don't emit yet — may be discarded if tool calls follow)
        if delta.content:
            text_chunks.append(delta.content)

    if tool_calls_acc:
        # ── Tool call detected: execute, then stream final response ──────────
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

        # ── Round 2: stream final response (no tools) ────────────────────────
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
        # ── No tool call — emit whatever text round 1 produced ───────────────
        for token in text_chunks:
            _q(state, {"type": "agent_content", "agent": "Analyst",
                       "content": token, "message_type": "analysis"})

    _q(state, {"type": "agent_complete", "agent": "Analyst"})
    print("[analyze_results] done")
    return {}


# ── Node: campaign_executor — launches background CampaignManager ────────────

def campaign_executor_node(state: SimulationState) -> dict:
    """Launcher node: spawns CampaignManager.run() in a daemon thread
    and returns immediately. The actual N-iteration loop runs asynchronously
    in the background, streaming SSE events through SESSION_QUEUES.
    """
    from datetime import datetime
    from benchmark.campaign import CampaignManager

    # Pull campaign params from checkpoint state (set by chat_node during setup)
    n_iter = state.get("campaign_n_iterations")
    target = state.get("campaign_target") or "minimize delta_a"
    original_msg = state.get("campaign_original_message")
    template = state.get("campaign_template") or {}
    bounds = state.get("campaign_bounds") or {}

    # Fallback: try to extract n_iter from conversation history if not in state.
    # Looks for patterns like "with 2 iterations", "5 iterations", "10 iter"
    # in past user messages (most recent first).
    if not n_iter:
        import re
        history = state.get("conversation_history") or []
        for msg in reversed(history):
            if msg.get("role") == "user":
                m = re.search(r"\b(\d+)\s*iter", (msg.get("content") or ""), re.IGNORECASE)
                if m:
                    n_iter = int(m.group(1))
                    print(f"[campaign_executor] extracted n_iter={n_iter} from user history (fallback)")
                    break
    if not n_iter:
        n_iter = 20
        print(f"[campaign_executor] defaulting n_iter=20 (no value in state or history)")

    # Fallback: walk conversation_history to find the user msg that triggered the campaign
    if not original_msg:
        for msg in state.get("conversation_history", []) or []:
            if msg.get("role") == "user":
                content = (msg.get("content") or "").lower()
                if any(p in content for p in (
                    "run a campaign", "benchmark campaign", "closed-loop campaign",
                    "autonomous campaign", "start a campaign", "begin a campaign",
                )):
                    original_msg = msg["content"]
                    break
    if not original_msg:
        original_msg = state.get("user_message", "")

    # Create a dedicated campaign_id (also serves as LangGraph thread_id for iterations)
    campaign_id = f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Dual-emit: forward ALL events to BOTH the campaign queue and the user's
    # chat session queue (so the open browser SSE connection sees live progress
    # — including agent_content from chat_node, generate_lammps progress,
    # trajectory_frame, etc., not just the high-level campaign_* events).
    chat_sid = state.get("session_id")
    chat_q = SESSION_QUEUES.get(chat_sid) if chat_sid else None
    campaign_q = queue.Queue()

    # Completion signal: app.py waits on this Event to keep the SSE open until
    # campaign_complete fires. Without this, the SSE closes when graph returns.
    completion_event = threading.Event()
    if chat_sid:
        CAMPAIGN_COMPLETION_EVENTS[chat_sid] = completion_event

    class _DualQueue:
        """Tee: putting an event onto this queue mirrors it to both queues.
        Also sets the completion_event when a campaign_complete event flows by."""
        def __init__(self, primary, secondary, completion_event):
            self.primary = primary
            self.secondary = secondary
            self.completion_event = completion_event
            self._tee_count = 0

        def put(self, item, *args, **kwargs):
            try:
                self.primary.put(item)
            except Exception as e:
                print(f"[DualQueue] primary.put FAILED: {e}")
            if self.secondary is not None and self.secondary is not self.primary:
                try:
                    self.secondary.put(item)
                    self._tee_count += 1
                    # Log first few tee operations + every 50th, so we can confirm events flow
                    if self._tee_count <= 5 or self._tee_count % 50 == 0:
                        evt_type = item.get("type") if isinstance(item, dict) else "?"
                        print(f"[DualQueue] teed #{self._tee_count} type={evt_type} to chat queue")
                except Exception as e:
                    print(f"[DualQueue] secondary.put FAILED: {e}")
            else:
                if isinstance(item, dict):
                    evt_type = item.get("type")
                    if evt_type and evt_type.startswith("campaign_"):
                        print(f"[DualQueue] WARNING: no secondary queue; campaign event '{evt_type}' won't reach chat browser")
            # Signal completion AFTER queues have been written so SSE generator
            # has a chance to drain the campaign_complete event.
            if isinstance(item, dict) and item.get("type") == "campaign_complete":
                self.completion_event.set()

    dual_q = _DualQueue(campaign_q, chat_q, completion_event)

    # CRITICAL: register dual_q (NOT plain campaign_q) under campaign_id, so
    # graph nodes that fire inside iterations (using session_id=campaign_id)
    # also tee their events to the user's chat queue. Otherwise chat_node's
    # agent_content, generate_lammps progress, etc. would only land on
    # SESSION_QUEUES[campaign_id] which nobody is draining.
    SESSION_QUEUES[campaign_id] = dual_q

    manager = CampaignManager(
        campaign_id=campaign_id,
        n_iterations=n_iter,
        original_user_message=original_msg,
        sse_q=dual_q,
        num_cores=config.LAMMPS_CORES,
        target=target,
        template=template,
        bounds=bounds,
    )

    t = threading.Thread(target=manager.run, daemon=True)
    t.start()

    _q(state, {
        "type": "campaign_started",
        "campaign_id": campaign_id,
        "n_iterations": n_iter,
        "target": target,
    })
    print(f"[campaign_executor] spawned campaign thread: {campaign_id}, "
          f"n_iter={n_iter}, target={target}")

    return {}
