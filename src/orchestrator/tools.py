"""Orchestrator tools for the ChatMD create_react_agent.

NOTE: this file is `src/orchestrator/tools.py` — distinct from `src/tools.py`
(which contains LAMMPS execution helpers like run_lammps_simulation, dump
monitoring, OVITO rendering). The orchestrator tools live under the
orchestrator/ package; the LAMMPS helpers stay at src/ root.

Tools registered in ORCHESTRATOR_TOOLS:
  - load_skill                 (reads skills/<name>.md, strips frontmatter)
  - get_kim_models             (wraps kim_utils.build_registry_context_for_command)
  - simulate                   (invokes simulate_subgraph, returns Command(update=...))
  - analyze_runs               (invokes analyze_results_node, return_direct)
  - hpc_submit                 (two-phase HPC submit via Globus)
  - hpc_status                 (queries Globus / PBS status)
  - campaign_* (7 tools)       (see orchestrator/campaign_tools.py — v2 orchestrator
                                owns the campaign loop directly, no daemon thread)
"""
import os
import sys
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kim_utils
from prompts import _VIZ_PROMPT


SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def _strip_frontmatter(text: str) -> str:
    """Return everything AFTER the closing `---` of YAML frontmatter.
    If no frontmatter is present, return text unchanged.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    body_start = end + len("\n---")
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    return text[body_start:]


# ── Phase 1a: real tools ─────────────────────────────────────────────────────

@tool
def load_skill(name: str) -> str:
    """Load a skill's instructions into the conversation for this turn.

    Reads src/orchestrator/skills/<name>.md, strips YAML frontmatter, and
    returns the body as a string. The orchestrator's next LLM turn will see
    the skill instructions in the tool-result message.

    Args:
        name: Skill name (e.g. "campaign"). Must match the filename without
              the .md extension.

    Returns:
        The skill body, or an error message if the file does not exist.
    """
    safe_name = os.path.basename(name).removesuffix(".md")
    path = os.path.join(SKILLS_DIR, f"{safe_name}.md")
    if not os.path.exists(path):
        return f"Skill '{safe_name}' not found at {path}."
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError as e:
        return f"Could not read skill '{safe_name}': {e}"
    return _strip_frontmatter(text)


@tool
def get_kim_models(element: str) -> str:
    """Fetch available KIM interatomic potentials for an element.

    Args:
        element: Chemical symbol (e.g. "Cu", "Al", "Fe").

    Returns:
        Formatted registry string (✅ = downloaded, ⬇️ = needs download),
        or a message if no models are found locally.
    """
    result = kim_utils.build_registry_context_for_command(element)
    if result:
        return result
    return f"No KIM models found in local registry for element '{element}'."


# ── Phase 1b-1e: stubs ───────────────────────────────────────────────────────

@tool
def simulate(
    material: str,
    temperature: float,
    ensemble: str,
    duration_ps: float,
    system_size: int,
    kim_model: str,
    simulation_details: str,
    visualization_notes: str | None = None,
    camera_angle: Literal["top", "side", "default"] = "default",
    execution_target: str = "local",
    auto_approve: bool = False,
    # ── Campaign-only optional knob args (set ONLY inside a campaign
    # iteration loop). For chat-driven simulations, leave all of these as
    # None — they will be ignored. The campaign skill instructs the LLM
    # to populate them per iteration; the values are echoed back onto
    # OrchestratorState.campaign_knobs as a convenience mirror, then the
    # LLM passes them explicitly to campaign_record_trial for persistence.
    strain_rate: float | None = None,
    strain_amplitude: float | None = None,
    crack_length_frac: float | None = None,
    crack_width_a0: float | None = None,
    # ToolRuntime is auto-injected by parameter name `runtime`; no
    # Annotated wrapper needed. The LLM-visible schema automatically
    # excludes this parameter.
    runtime: ToolRuntime = None,
) -> Command:
    """Generate LAMMPS + OVITO scripts for a simulation.

    Internally invokes the slim simulate_subgraph (generate_lammps →
    detect_viz → generate_ovito), seeded with the structured args + the
    full simulation_details prose. Returns a Command(update=...) that
    writes the script fields onto the orchestrator's state — the cascade
    in app.py reads them to fire `scripts_ready_for_review`.

    Args:
        material: Element symbol (e.g. "Al", "Cu").
        temperature: Target temperature in K.
        ensemble: "NVE", "NVT", or "NPT".
        duration_ps: Simulation duration in picoseconds.
        system_size: Number of unit cells per dimension (cubic box).
        kim_model: Exact KIM model ID.
        simulation_details: Full proposal prose — phase definitions, pressure
                            targets, crack geometry, special conditions. The
                            downstream generator parses this for multi-phase
                            / crack / fatigue setups.
        visualization_notes: Optional viz preferences (free-form).
        camera_angle: One of "top", "side", or "default" ONLY.
            - "top"     = top-down / bird's-eye view down the +z axis
                          (best for crack plane in xy, layered defects).
            - "side"    = side view down the +x axis
                          (best for layered systems, profile views).
            - "default" = isometric 3/4 perspective (general 3D view).
            Translate user descriptions to these 3 values before passing:
              "+z" / "down +z" / "from above" / "bird's-eye" / "top-down"  → "top"
              "+x" / "from the side" / "side view"                          → "side"
              anything else / unspecified                                   → "default"
        execution_target: "local" or "hpc". Phase 1 wires only "local".
        auto_approve: True only for campaign iterations — suppresses the
                      user Approve prompt in the cascade.

    Returns:
        Command writing lammps_script, ovito_script, dump_filename_pattern,
        review_passed, has_custom_viz, auto_approve, execution_target,
        camera_angle, and target_timesteps onto the orchestrator's state.
        The tool result message contains a short status string for the LLM.
    """
    # Import here to avoid circular imports at module load (graph imports
    # nodes.py which imports kim_utils which... etc.)
    from graph.graph import simulate_subgraph

    state = runtime.state if runtime else {}
    tool_call_id = runtime.tool_call_id if runtime else ""
    session_id = (state or {}).get("session_id")

    # DEBUG: print the EXACT simulation_details prose the LLM passed
    # to the tool — one block per simulate call (i.e. per campaign iteration).
    print("=" * 80)
    print(f"[simulate] simulation_details ({len(simulation_details)} chars):")
    print("-" * 80)
    print(simulation_details)
    print("=" * 80)
    


    # Seed the subgraph's SimulationState with everything generate_lammps_node
    # would read. We skip extract_params_node by setting has_parameters=True
    # and pre-populating the structured fields — avoids a redundant DSPy call.
    subgraph_state = {
        "user_message":         simulation_details,
        "session_id":           session_id,
        "analysis":             simulation_details,
        "has_parameters":       True,
        "material":             material,
        "kim_model":            kim_model,
        "temperature":          temperature,
        "ensemble":             ensemble,
        "duration_ps":          duration_ps,
        "system_size":          system_size,
        "visualization_notes":  visualization_notes,
        "camera_angle":         camera_angle,
        "execution_target":     execution_target,
        "workflow_stage":       "initial",
        "conversation_history": [],
    }

    final = simulate_subgraph.invoke(subgraph_state)

    lammps_script = final.get("lammps_script")
    review_passed = bool(final.get("review_passed"))

    if not lammps_script:
        msg = "Script generation failed: no script produced. Tell the user something went wrong upstream."
    elif not review_passed:
        issues = final.get("review_issues") or []
        msg = (
            f"Script generated but reviewer flagged issues: {'; '.join(issues[:3])}. "
            f"Tell the user the simulation needs adjustment before running."
        )
    elif auto_approve:
        msg = "Script generated and auto-approved (campaign iteration). Tell the user iteration is starting."
    else:
        msg = (
            "Script generated and reviewed successfully. The Run/Approve card "
            "will appear in the chat. Briefly tell the user the script is ready."
        )

    # Echo any campaign-only knob args the LLM passed back into state.
    # In v2 the source of truth for per-trial knobs is trials.jsonl (written
    # by campaign_record_trial). This state mirror is kept for downstream
    # observers/cascades that already read it. All-None for chat-driven sims.
    campaign_knobs = {
        k: v for k, v in {
            "strain_rate":       strain_rate,
            "strain_amplitude":  strain_amplitude,
            "crack_length_frac": crack_length_frac,
            "crack_width_a0":    crack_width_a0,
            "temperature":       temperature,
        }.items() if v is not None
    }

    # Mirror script + viz fields onto OrchestratorState so the cascade can
    # decide what terminal SSE to emit. auto_approve gates the Run/Approve
    # card (campaign iterations skip it).
    return Command(
        update={
            "lammps_script":         lammps_script,
            "ovito_script":          final.get("ovito_script"),
            "dump_filename_pattern": final.get("dump_filename_pattern"),
            "target_timesteps":      final.get("target_timesteps"),
            "camera_angle":          final.get("camera_angle") or camera_angle,
            "execution_target":      execution_target,
            "auto_approve":          auto_approve,
            "review_passed":         review_passed,
            "review_issues":         final.get("review_issues"),
            "review_suggestions":    final.get("review_suggestions"),
            "has_custom_viz":        final.get("has_custom_viz"),
            "campaign_knobs":        campaign_knobs or None,
            # Echo the simulate args back into state so the cascade can
            # auto-title the session ("Cu 300K") and so follow-up turns
            # can read these without walking the message history.
            "material":              material,
            "temperature":           temperature,
            "ensemble":              ensemble,
            "duration_ps":           duration_ps,
            "system_size":           system_size,
            "kim_model":             kim_model,
            # ToolMessage with the REAL tool_call_id satisfies langgraph's
            # invariant: every AIMessage tool_call needs a matching ToolMessage.
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )


@tool(return_direct=True)
def analyze_runs(
    query: str,
    runtime: ToolRuntime = None,
) -> str:
    """List past simulation runs or read their logs to answer questions.

    Internally invokes `analyze_results_node` (tool-calling Analyst agent
    with list_recent_runs + get_run_log). All output streams directly to
    the user via the existing SSE pipeline (Analyst typing indicator + text).

    `return_direct=True`: create_react_agent ends the orchestrator's turn
    immediately after this tool returns — no follow-up LLM call that would
    restate what the Analyst already showed.

    For follow-up analysis questions ("was it equilibrated?", "analyze
    run 4"), the orchestrator will call analyze_runs again on the next
    turn. The conversation_history passed in lets the inside-Analyst
    answer from context without re-loading.

    Args:
        query: Natural-language question or instruction
               (e.g. "list recent runs", "analyze run 2", "what was the
               final temperature in the latest Al run?").
    """
    from graph.nodes import analyze_results_node
    from langchain_core.messages import AIMessage, HumanMessage

    state = runtime.state if runtime else {}
    session_id = (state or {}).get("session_id")

    # Convert orchestrator's LangChain messages → role/content dicts the
    # Analyst agent expects. This lets it see prior listings ("Here are
    # 5 runs: 1. Cu_500K | 2. Al_300K | ...") so follow-ups like "analyze
    # run 2" resolve without re-calling list_recent_runs. Cap at last 12
    # messages to keep the Analyst's prompt bounded.
    orch_messages = (state or {}).get("messages") or []
    history = []
    for m in orch_messages[-12:]:
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            history.append({"role": "assistant", "content": m.content})

    analyst_state = {
        "user_message":         query,
        "session_id":           session_id,
        "conversation_history": history,
    }

    analyze_results_node(analyst_state)

    # Return value satisfies the assistant→ToolMessage invariant. Because
    # return_direct=True, the graph ends immediately after this — the LLM
    # does NOT see this content, so it can be anything.
    return "Analysis streamed via Analyst agent."


@tool(return_direct=True)
def hpc_submit(
    material: str,
    temperature: float,
    kim_model: str = "",
    runtime: ToolRuntime = None,
) -> Command:
    """Submit a LAMMPS job to HPC via Globus — two-phase pattern.

    All script content (LAMMPS, OVITO, edited PBS) is read from
    runtime.state, NOT from LLM-typed tool args. The frontend POSTs the
    user-edited textarea content; app.py threads it into OrchestratorState
    as pending_script / ovito_script_param / pbs_script_edited. This tool
    reads those fields directly — the LLM only needs to provide the
    structured metadata (material, temperature, kim_model).

    PHASE 1 (state.pbs_script_edited is empty — first call):
      - Creates a run folder, writes pending_script as input.lammps.
      - Writes ovito_script_param (if any) as ovito_script.py.
      - Generates default PBS script and writes submit_lammps.sh.
      - Emits "pbs_preview" SSE so the frontend shows the editable card.
      - Returns workflow_stage="pbs_pending" + hpc_run_folder_pending.
        The user edits the PBS, hits Approve, frontend re-POSTs with
        body.pbs_script. app.py threads it into state.pbs_script_edited.

    PHASE 2 (state.pbs_script_edited is set — second call):
      - Reads state.hpc_run_folder_pending to find the saved scripts.
      - Overwrites submit_lammps.sh with the user's edits.
      - Calls _run_hpc_submit (Globus MCP via create_react_agent in
        graph/hpc_executor_node.py).
      - Appends a new hpc_jobs entry with the Globus task ID.

    return_direct=True for both phases — orchestrator turn ends after the
    tool runs. PBS preview / Globus submission output streams directly to
    the user via the existing _q SSE plumbing.

    Args:
        material: For naming + the job description.
        temperature: For the job description.
        kim_model: Optional KIM model ID — added to the PBS script's
                   kim-api install command so HPC has the potential.
    """
    import os
    from datetime import datetime

    from graph.hpc_executor_node import (
        _generate_pbs_script,
        _run_hpc_submit,
    )
    import run_manager
    import config
    from graph.nodes import _q

    state = runtime.state if runtime else {}
    session_id = (state or {}).get("session_id")
    tool_call_id = runtime.tool_call_id if runtime else ""

    # Read user-edited scripts from state (set by app.py from frontend body).
    # Falls back to state.lammps_script / state.ovito_script if the user
    # didn't edit (the originals from the most recent simulate call).
    lammps_script = (
        (state or {}).get("pending_script")
        or (state or {}).get("lammps_script")
    )
    ovito_script = (
        (state or {}).get("ovito_script_param")
        or (state or {}).get("ovito_script")
    )
    pbs_script_edited = (state or {}).get("pbs_script_edited")

    if not lammps_script:
        ack = (
            "ERROR: no LAMMPS script in state. Run simulate(...) first to "
            "generate a script, then ask to submit to HPC."
        )
        return Command(update={"messages": [ToolMessage(content=ack, tool_call_id=tool_call_id)]})

    # ── Phase 2: user has approved + edited the PBS, submit via Globus ───
    if pbs_script_edited:
        run_folder = (state or {}).get("hpc_run_folder_pending") or ""
        if not run_folder:
            ack = "ERROR: no run_folder_pending in state — call hpc_submit without pbs_script_edited first to set up the run."
            return Command(update={"messages": [ToolMessage(content=ack, tool_call_id=tool_call_id)]})

        with open(os.path.join(run_folder, "submit_lammps.sh"), "w") as f:
            f.write(pbs_script_edited)

        run_folder_name = os.path.basename(run_folder)
        hpc_remote_dir = f"{config.HPC_RUN_BASE_DIR}/{run_folder_name}"
        description = f"{material} {temperature}K".strip()

        # _run_hpc_submit reads state for _q(state, ...) SSE forwarding —
        # build a minimal compatible dict.
        submit_state = {
            "session_id": session_id,
            "material":   material,
            "temperature": temperature,
        }
        globus_task_id = _run_hpc_submit(submit_state, run_folder, hpc_remote_dir)

        new_job = {
            "globus_task_id": globus_task_id or "unknown",
            "hpc_run_folder": run_folder,
            "submitted_at":   datetime.now().isoformat(),
            "description":    description,
        }
        existing_jobs = (state or {}).get("hpc_jobs") or []
        ack = f"Submitted to HPC. Globus task ID: {globus_task_id or 'unknown'}."
        return Command(update={
            "messages":               [ToolMessage(content=ack, tool_call_id=tool_call_id)],
            "hpc_jobs":               existing_jobs + [new_job],
            "workflow_stage":         "initial",
            "hpc_run_folder_pending": None,
            "pbs_script_edited":      None,    # consumed — clear for next submit
        })

    # ── Phase 1: write scripts, emit PBS preview, await user edit ────────
    run_folder = run_manager.create_run_folder()
    run_folder_name = os.path.basename(run_folder)
    run_manager.save_lammps_script(run_folder, lammps_script)
    if ovito_script:
        with open(os.path.join(run_folder, "ovito_script.py"), "w") as f:
            f.write(ovito_script)

    pbs_script = _generate_pbs_script(run_folder_name, kim_model=kim_model)
    with open(os.path.join(run_folder, "submit_lammps.sh"), "w") as f:
        f.write(pbs_script)

    # SSE: pbs_preview goes to the frontend's editable card.
    sse_state = {"session_id": session_id}
    _q(sse_state, {
        "type":        "pbs_preview",
        "pbs_script":  pbs_script,
        "run_folder":  run_folder,
    })
    _q(sse_state, {
        "type":    "agent_content",
        "agent":   "HPC",
        "content": "Here is the PBS script that will be submitted. Review it and confirm to proceed.",
    })

    ack = f"PBS preview emitted for run_folder={run_folder}. Awaiting user edit/approval — turn ends."
    return Command(update={
        "messages":               [ToolMessage(content=ack, tool_call_id=tool_call_id)],
        "workflow_stage":         "pbs_pending",
        "hpc_run_folder_pending": run_folder,
    })


@tool(return_direct=True)
def hpc_status(
    query: str = "check status of my HPC jobs",
    runtime: ToolRuntime = None,
) -> str:
    """Check status of previously-submitted HPC jobs (and optionally
    transfer results back from HPC to the local Globus endpoint).

    Wraps `_run_hpc_status_async` (create_react_agent + Globus MCP
    setup). Streams the HPC agent's reasoning + tool calls directly to
    the user via the existing SSE pipeline (HPC agent persona).

    return_direct=True: orchestrator's turn ends after the HPC agent
    finishes streaming. The HPC agent does the user-facing speaking.

    Args:
        query: Natural-language status/transfer request
               (e.g. "check status", "are my jobs done?",
               "transfer the results back").
    """
    import os
    from graph.hpc_executor_node import _run_hpc_status
    from langchain_core.messages import AIMessage, HumanMessage
    import config

    state = runtime.state if runtime else {}
    session_id = (state or {}).get("session_id")
    hpc_jobs = (state or {}).get("hpc_jobs") or []

    if not hpc_jobs:
        sse_state = {"session_id": session_id}
        from graph.nodes import _q
        _q(sse_state, {
            "type": "agent_content",
            "agent": "HPC",
            "content": "No HPC jobs found for this session.",
        })
        return "No HPC jobs to check."

    jobs_summary = "\n".join(
        f"- {j.get('description', 'job')}: "
        f"globus_task_id={j['globus_task_id']}, "
        f"local_run_folder={j['hpc_run_folder']}, "
        f"remote_run_folder={config.HPC_RUN_BASE_DIR}/{os.path.basename(j['hpc_run_folder'])}, "
        f"submitted={j['submitted_at']}"
        for j in hpc_jobs
    )

    # Build recent conversation context (capped) so HPC agent can resolve
    # multi-turn references like "yes transfer it".
    orch_messages = (state or {}).get("messages") or []
    history_lines = []
    for m in orch_messages[-6:]:
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            history_lines.append(f"USER: {m.content[:400]}")
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            history_lines.append(f"ASSISTANT: {m.content[:400]}")
    history_block = (
        f"Recent conversation (most recent at bottom):\n{chr(10).join(history_lines)}\n\n"
        if history_lines else ""
    )

    user_msg = (
        f"{history_block}"
        f"User request: {query}\n\n"
        f"HPC jobs submitted in this session:\n{jobs_summary}\n\n"
        f"Local Globus endpoint: {config.GLOBUS_LOCAL_ENDPOINT}\n"
        f"HPC Globus endpoint: {config.GLOBUS_HPC_ENDPOINT}\n\n"
        f"If the user wants to transfer results back, transfer the entire "
        f"remote_run_folder to the corresponding local_run_folder."
    )

    status_state = {"session_id": session_id}
    _run_hpc_status(status_state, user_msg)

    return "HPC status streamed to user via HPC agent."


# ── visualize_run ────────────────────────────────────────────────────────────

@tool(return_direct=True)
def visualize_run(
    run_folder: str,
    camera_angle: Literal["top", "side", "default"] = "top",
    runtime: ToolRuntime = None,
) -> str:
    """Render key dump frames of a simulation run and produce a visual
    analysis (crack morphology, dislocations, stacking faults, etc.) via a
    vision-capable LLM.

    Reuses the existing OVITO render server + the run's own ovito_script.py
    for consistent coloring. Writes viz_report.md + viz_frames/*.png to the
    run folder, then streams a text summary + a clickable link to the chat.

    return_direct=True — the orchestrator turn ends after this tool.

    Args:
        run_folder: Path or name of a run folder. Accepts:
                    - absolute path or "runs/run_..."
                    - just "run_..." (resolved under runs/)
                    - a number ("1", "2", ...) matching list_recent_runs
                    - "latest"
        camera_angle: "top" (default, best for crack analysis), "side", or "default"
                     (isometric 3D view).
    """
    import base64
    import glob as _glob
    from graph.nodes import _q
    import run_manager

    state = runtime.state if runtime else {}
    session_id = (state or {}).get("session_id")
    tool_call_id = runtime.tool_call_id if runtime else ""

    # ── Resolve run_folder (support name / number / "latest") ─────────
    resolved = _resolve_run_folder(run_folder)
    if not resolved or not os.path.isdir(resolved):
        _q(state, {"type": "agent_content", "agent": "Visualization",
                   "content": f"Could not resolve run folder: {run_folder!r}"})
        return f"Visualization failed: could not resolve run folder {run_folder!r}"

    _q(state, {"type": "agent_start", "agent": "Visualization"})
    _q(state, {"type": "agent_content", "agent": "Visualization",
               "content": f"**Analyzing** `{os.path.basename(resolved)}`\n\n"})

    # ── Frame selection: pristine + seeded-crack (last relax) + last_load ──
    picks = _pick_viz_frames(resolved)
    if not picks:
        _q(state, {"type": "agent_content", "agent": "Visualization",
                   "content": f"No dump files found in {resolved}."})
        _q(state, {"type": "agent_complete", "agent": "Visualization"})
        return f"No dump files in {resolved}"

    # ── Load ovito_script.py if present ───────────────────────────────
    ovito_script = None
    ovito_path = os.path.join(resolved, "ovito_script.py")
    if os.path.exists(ovito_path):
        with open(ovito_path) as f:
            ovito_script = f.read()

    # ── Render 3 PNGs via existing render server ──────────────────────
    from tools import _start_render_server_process, _send_render_request
    frames_dir = os.path.join(resolved, "viz_frames")
    os.makedirs(frames_dir, exist_ok=True)

    proc = _start_render_server_process()
    rendered = []
    try:
        for label, dump_path in picks:
            resp = _send_render_request(proc, {
                "dump_file":        os.path.abspath(dump_path),
                "ovito_script":     ovito_script,
                "camera_angle":     camera_angle,
                "render_size":      (800, 600),
                "file_frame_index": 0,
            })
            if resp.get("status") != "ok":
                _q(state, {"type": "agent_content", "agent": "Visualization",
                           "content": f"Render failed for {label}: {resp.get('error')}\n"})
                continue
            b64 = resp["image"]
            png_path = os.path.join(frames_dir, f"{label}.png")
            with open(png_path, "wb") as f:
                f.write(base64.b64decode(b64))
            rendered.append({"label": label, "png_path": png_path, "b64": b64})
    finally:
        try:
            proc.stdin.close(); proc.wait(timeout=5)
        except Exception:
            proc.kill()

    if not rendered:
        _q(state, {"type": "agent_complete", "agent": "Visualization"})
        return "No frames rendered — cannot analyze."

    # ── Vision LLM call — stream summary to chat ──────────────────────
    from openai import OpenAI
    import config
    client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE_URL)

    content = []
    for f in rendered:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{f['b64']}"},
        })
    if ovito_script:
        content.append({
            "type": "text",
            "text": ("The atom colors in these frames were assigned by the following "
                     "OVITO script. Use its color_map to interpret each color's "
                     "structural meaning (do NOT guess):\n\n```python\n" +
                     ovito_script + "\n```"),
        })
    else:
        content.append({
            "type": "text",
            "text": ("No custom OVITO script was used. Atoms are colored by LAMMPS "
                     "particle type (NOT crystal structure). Do not interpret colors "
                     "as FCC/HCP/BCC. Focus on gross spatial features."),
        })
    content.append({"type": "text", "text": _VIZ_PROMPT})

    model = getattr(config, "ORCHESTRATOR_MODEL", "") or config.DEFAULT_MODEL

    # Stream the summary token-by-token to the chat
    summary_parts = []
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            tok = chunk.choices[0].delta.content
            if tok:
                summary_parts.append(tok)
                _q(state, {"type": "agent_content", "agent": "Visualization",
                           "content": tok, "message_type": "analysis"})
    except Exception as e:
        _q(state, {"type": "agent_content", "agent": "Visualization",
                   "content": f"\nVision call failed: {e}\n"})
        _q(state, {"type": "agent_complete", "agent": "Visualization"})
        return f"Vision LLM call failed: {e}"

    observation = "".join(summary_parts).strip()

    # ── Write viz_report.md + viz_report.html at run folder root ─────
    md_path, html_path = _write_viz_report(resolved, rendered, observation)

    # Link to both reports (.html opens directly in Chrome)
    rel_url = f"/{resolved}" if resolved.startswith("runs/") else f"/runs/{os.path.basename(resolved)}"
    _q(state, {"type": "agent_content", "agent": "Visualization",
               "content": (
                   f"\n\n📄 [Open viz report (HTML — best in Chrome)]({rel_url}/viz_report.html)"
                   f" • [Markdown version]({rel_url}/viz_report.md)\n"
               )})
    _q(state, {"type": "agent_complete", "agent": "Visualization"})

    return "Visual analysis streamed to user."


# ── visualize_run helpers ────────────────────────────────────────────────────
# _VIZ_PROMPT is imported from prompts.py (all LLM prompts live there).


def _timestep_of(dump_path: str) -> int:
    base = os.path.basename(dump_path)
    for token in base.split("."):
        if token.isdigit():
            return int(token)
    return 0


def _resolve_run_folder(run_folder: str) -> str:
    """Accepts a full path, just a name, a number, or 'latest'."""
    import run_manager
    if not run_folder:
        return ""
    val = run_folder.strip().lower()
    if val == "latest":
        return run_manager.get_latest_run() or ""
    if val.isdigit():
        all_runs = run_manager.list_all_runs()
        idx = int(val) - 1
        if 0 <= idx < len(all_runs):
            return os.path.abspath(os.path.join(run_manager.RUNS_BASE_DIR, all_runs[idx]))
        return ""
    # Try as a full path first
    if os.path.isdir(run_folder):
        return os.path.abspath(run_folder)
    # Then as a name under runs/
    candidate = os.path.abspath(os.path.join(run_manager.RUNS_BASE_DIR, run_folder))
    return candidate if os.path.isdir(candidate) else ""


def _pick_viz_frames(run_folder: str):
    """Return list of (label, dump_path).

    Frames chosen to MATCH the Δa metric's baseline→final pair so the viz
    report visually corresponds to the measured crack growth:
      1. pristine   — first equil frame (clean crystal, before crack seeding)
      2. seeded     — LAST relax frame (the seeded crack, pre-loading) =
                      the metric's a_initial baseline
      3. last_load  — LAST load frame = the metric's a_final

    We intentionally do NOT include a "first_load" frame: the dump clock is
    global and does not reset when loading starts, so the first dump.*.load
    file lands an unpredictable number of steps INTO loading and is not a
    reliable "start of load" (same reason the metric uses last-relax, not
    first-load, as its baseline).

    Fallback (no .load dumps): pristine + last dump overall.
    """
    import glob as _glob
    all_dumps = sorted(_glob.glob(os.path.join(run_folder, "dump.*")), key=_timestep_of)
    if not all_dumps:
        return []
    picks = []

    equil = sorted(_glob.glob(os.path.join(run_folder, "dump.*.equil")), key=_timestep_of)
    picks.append(("pristine", equil[0] if equil else all_dumps[0]))

    # Seeded crack (metric baseline) = LAST relax frame.
    relax = sorted(_glob.glob(os.path.join(run_folder, "dump.*.relax")), key=_timestep_of)
    if relax and relax[-1] != picks[0][1]:
        picks.append(("seeded", relax[-1]))

    load = sorted(_glob.glob(os.path.join(run_folder, "dump.*.load")), key=_timestep_of)
    if load:
        if load[-1] != picks[-1][1]:
            picks.append(("last_load", load[-1]))
    else:
        if all_dumps[-1] != picks[-1][1]:
            picks.append(("last", all_dumps[-1]))

    # Log the chosen frames so it's verifiable at runtime which dumps the viz
    # report renders (should be: pristine equil → last relax → last load).
    print(f"[viz] frames for {os.path.basename(run_folder)}: "
          + ", ".join(f"{lbl}={os.path.basename(p)}" for lbl, p in picks))
    return picks


_VIZ_HTML_STYLE = """
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
         margin: 20px auto; padding: 0 20px; color: #222; }
  h1 { color: #333; }
  table { border-collapse: collapse; margin: 20px 0; }
  td { padding: 10px; }
  img { border: 1px solid #ccc; border-radius: 4px; }
  em { color: #666; }
  .summary { line-height: 1.6; margin-top: 30px; }
  hr { border: 0; border-top: 1px solid #ddd; margin: 30px 0; }
</style>
"""


def _write_viz_report(run_folder, rendered, observation):
    """Write BOTH viz_report.md and viz_report.html side-by-side.
    Returns (md_path, html_path)."""
    import html as _html
    title = f"Visual Analysis — {os.path.basename(run_folder.rstrip('/'))}"

    # Side-by-side table (works verbatim in .md and .html)
    img_cells, caption_cells = [], []
    for f in rendered:
        rel = os.path.relpath(f["png_path"], run_folder)
        img_cells.append(f'<td align="center"><img src="{rel}" alt="{f["label"]}" width="320"></td>')
        caption_cells.append(f'<td align="center"><em>{f["label"]}</em></td>')
    table_html = (
        "<table>\n"
        "  <tr>" + "".join(img_cells) + "</tr>\n"
        "  <tr>" + "".join(caption_cells) + "</tr>\n"
        "</table>"
    )

    # .md version
    md_lines = [f"# {title}", "", table_html, "", "---", "", observation, ""]
    md_path = os.path.join(run_folder, "viz_report.md")
    with open(md_path, "w") as fout:
        fout.write("\n".join(md_lines))

    # .html version (opens directly in Chrome, no extension needed)
    observation_html = _html.escape(observation.strip()).replace("\n\n", "</p><p>")
    html_body = (
        "<!DOCTYPE html>\n<html><head>\n"
        f"<meta charset='utf-8'>\n<title>{_html.escape(title)}</title>\n"
        f"{_VIZ_HTML_STYLE}\n"
        "</head><body>\n"
        f"<h1>{_html.escape(title)}</h1>\n"
        f"{table_html}\n<hr>\n"
        f"<div class='summary'><p>{observation_html}</p></div>\n"
        "</body></html>\n"
    )
    html_path = os.path.join(run_folder, "viz_report.html")
    with open(html_path, "w") as fout:
        fout.write(html_body)

    return md_path, html_path


# ── Exported list for create_react_agent ─────────────────────────────────────

from orchestrator.campaign_tools import CAMPAIGN_TOOLS

ORCHESTRATOR_TOOLS = [
    load_skill,
    get_kim_models,
    simulate,
    analyze_runs,
    visualize_run,
    *CAMPAIGN_TOOLS,   # campaign_start, campaign_get_state, campaign_measure_metrics,
                       # campaign_record_trial, campaign_check_stop,
                       # campaign_generate_report, campaign_save_narrative,
                       # campaign_visualize_all
    hpc_submit,
    hpc_status,
]
