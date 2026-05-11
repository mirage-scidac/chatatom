"""
HPC Executor Node for ChatMD.

Uses MultiServerMCPClient (langchain_mcp_adapters) + create_react_agent (langgraph.prebuilt)
to drive Globus MCP tools natively — no manual ReAct loop, no format conversion.

Phases:
  1. PBS preview  — generate PBS script, emit pbs_preview SSE, wait for user confirmation
  2. Submission   — MultiServerMCPClient connects to globus-mcp, create_react_agent submits job
  3. Status query — same agent reads hpc_jobs from state, checks each job's status
"""
import asyncio
import json
import os
import sys
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run_manager
from prompts import HPC_EXECUTOR_PROMPT
from graph.state import SimulationState
from graph.nodes import _q


llm = ChatOpenAI(
    model=config.DEFAULT_MODEL,
    api_key=config.API_KEY,
    base_url=config.API_BASE_URL,
)

GLOBUS_MCP_CONFIG = {
    "globus": {
        "command": "uvx",
        "args": ["globus-mcp"],
        "transport": "stdio",
    }
}


def _generate_pbs_script(run_folder_name: str, kim_model: str = "") -> str:
    """Build PBS script from config defaults, matching the site-specific Swing template."""
    cores = config.HPC_DEFAULT_CORES
    kim_install = (
        f"pixi run kim-api-collections-management install user {kim_model}\n"
        if kim_model else ""
    )
    return (
        "#!/bin/bash -l\n"
        f"#PBS -A {config.HPC_PROJECT_ID}\n"
        f"#PBS -l select={config.HPC_DEFAULT_NODES}:ncpus={cores}:mpiprocs={cores}\n"
        f"#PBS -l walltime={config.HPC_DEFAULT_WALLTIME}\n"
        f"#PBS -N chatmd_{run_folder_name}\n"
        "#PBS -j oe\n"
        "#PBS -m be\n"
        f"#PBS -M {config.HPC_USER_EMAIL}\n"
        "\n"
        "echo Working directory is $PBS_O_WORKDIR\n"
        "cd $PBS_O_WORKDIR\n"
        'export PATH="$HOME/.pixi/bin:$PATH"\n'
        "which lmp\n"
        "module list\n"
        "echo Jobid: $PBS_JOBID\n"
        "echo Running on host `hostname`\n"
        "\n"
        f"{kim_install}"
        f"pixi run mpirun -np {cores} lmp -in input.lammps -log log.lammps\n"
        "\n"
        'echo "Simulation complete"\n'
    )


async def _stream_agent_step(state: SimulationState, agent, messages: list,
                              attempt_offset: int = 0,
                              exit_when_task_id_found: bool = False) -> tuple:
    """
    Stream one agent step, emitting tool calls and responses to the user via _q.
    Returns (all_messages, final_counter) so the next step can continue the attempt counter.

    exit_when_task_id_found: when True, break out of the agent loop the moment a
    Globus task ID appears in any ToolMessage. Used for the qsub step where we
    have everything we need (the task ID + PBS job ID in the tool response) and
    don't want the agent to keep verifying.
    """
    all_messages = list(messages)
    counter = [attempt_offset]
    found_task_id = None

    def _emit(content: str):
        counter[0] += 1
        _q(state, {"type": "agent_content", "agent": "HPC", "content": content, "attempt": counter[0]})

    async for chunk in agent.astream({"messages": messages}, stream_mode="updates"):
        if "agent" in chunk:
            for msg in chunk["agent"].get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            _emit(f"Calling {tc['name']}...")
                    elif msg.content:
                        _emit(msg.content)
                    all_messages.append(msg)

                    # Early-exit signal #1: agent emitted a text reply that mentions
                    # both task ID and PBS job — that's its natural success report.
                    # We break here so the agent doesn't get to "verify" past success.
                    if exit_when_task_id_found and msg.content:
                        text = msg.content.lower()
                        if "task id" in text and ("pbs" in text or "job id" in text):
                            found_task_id = "in_agent_text"
        if "tools" in chunk:
            for msg in chunk["tools"].get("messages", []):
                if isinstance(msg, ToolMessage):
                    # Early-exit signal #2: tool returned JSON with a task_id key
                    try:
                        data = json.loads(msg.content)
                        task_id = data.get("task_id") or data.get("globus_task_id") or data.get("job_id", "")
                        if task_id:
                            _emit(f"Task ID: {task_id}")
                            found_task_id = task_id
                    except Exception:
                        pass
                    all_messages.append(msg)

        # Either signal triggers the break — opt-in via exit_when_task_id_found.
        if exit_when_task_id_found and found_task_id:
            print(f"[HPC] Got IDs ({found_task_id}) — ending agent loop early")
            break

    return all_messages, counter[0]


async def _run_hpc_submit_async(state: SimulationState, run_folder: str, hpc_remote_dir: str) -> str:
    """
    Two-step HPC submission, streaming agent progress to the user at each step:
      Step 1 — transfer run folder to HPC via Globus Transfer
      Step 2 — submit PBS job via Globus Compute (qsub submit_lammps.sh)
    Returns the Globus Compute task ID from step 2.
    """
    globus_task_id = ""

    client = MultiServerMCPClient(GLOBUS_MCP_CONFIG)
    async with client.session("globus") as session:
        tools = await load_mcp_tools(session)
        # Wrap tools so MCP failures (404, auth, broken pipe) become ToolMessages the
        # agent can read and respond to conversationally — instead of crashing the
        # whole TaskGroup when one tool call fails.
        tool_node = ToolNode(tools, handle_tool_errors=True)
        agent = create_react_agent(llm, tool_node, prompt=SystemMessage(content=HPC_EXECUTOR_PROMPT))

        # ── Step 1: Create remote directory ──────────────────────────────────
        _q(state, {"type": "agent_content", "agent": "HPC", "content": "Step 1: Creating run directory on Swing...", "attempt": 1})
        step1 = (
            f"Run the following shell command on Globus Compute endpoint {config.GLOBUS_COMPUTE_ENDPOINT_ID}: "
            f"mkdir -p {hpc_remote_dir}"
        )
        history, attempt = await _stream_agent_step(state, agent, [HumanMessage(content=step1)], attempt_offset=1)

        # ── Step 2: Transfer files ────────────────────────────────────────────
        attempt += 1
        _q(state, {"type": "agent_content", "agent": "HPC", "content": "Step 2: Transferring files to HPC...", "attempt": attempt})
        step2 = (
            f"Transfer the following two files from local Globus endpoint {config.GLOBUS_LOCAL_ENDPOINT} "
            f"to the remote directory {hpc_remote_dir} on HPC Globus endpoint {config.GLOBUS_HPC_ENDPOINT}:\n"
            f"  - {run_folder}/input.lammps\n"
            f"  - {run_folder}/submit_lammps.sh\n"
            f"Wait until the transfer completes successfully before replying."
        )
        history, attempt = await _stream_agent_step(state, agent, history + [HumanMessage(content=step2)], attempt_offset=attempt)

        # ── Step 3: Submit PBS job ───────────────────────────────────────────
        attempt += 1
        _q(state, {"type": "agent_content", "agent": "HPC", "content": "Step 3: Submitting PBS job to Swing...", "attempt": attempt})
        step3 = (
            f"Run the following shell command on Globus Compute endpoint "
            f"{config.GLOBUS_COMPUTE_ENDPOINT_ID}: "
            f"cd {hpc_remote_dir} && qsub submit_lammps.sh. "
            f"After submitting the command, check the result of the submission once. "
            f"Report the Globus Compute task ID and any PBS job ID or job number you find in the response. "
            f"Do NOT poll or check more than once — get the result once and stop immediately."
        )
        # Step 3 (qsub) is the only step where we want early-exit on task_id —
        # the other steps don't have a clean "we're done" signal and the agent
        # naturally terminates when it has the answer.
        final_messages, attempt = await _stream_agent_step(
            state, agent, history + [HumanMessage(content=step3)],
            attempt_offset=attempt,
            exit_when_task_id_found=True,
        )

    # Extract Globus Compute task ID and emit it to the user
    for msg in final_messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                task_id = data.get("task_id") or data.get("globus_task_id") or data.get("job_id", "")
                if task_id:
                    globus_task_id = task_id
                    break
            except Exception:
                pass

    if globus_task_id:
        _q(state, {"type": "agent_content", "agent": "HPC",
                   "content": f"✅ Job submitted! Globus Compute task ID: {globus_task_id}", "attempt": attempt + 1})

    return globus_task_id


async def _run_hpc_status_async(state: SimulationState, user_msg: str) -> None:
    """Single-step status check, streaming agent progress to the user."""
    client = MultiServerMCPClient(GLOBUS_MCP_CONFIG)
    async with client.session("globus") as session:
        tools = await load_mcp_tools(session)
        # Wrap tools so MCP failures (404, auth, broken pipe) become ToolMessages the
        # agent can read and respond to conversationally — instead of crashing the
        # whole TaskGroup when one tool call fails.
        tool_node = ToolNode(tools, handle_tool_errors=True)
        agent = create_react_agent(llm, tool_node, prompt=SystemMessage(content=HPC_EXECUTOR_PROMPT))
        await _stream_agent_step(state, agent, [HumanMessage(content=user_msg)], attempt_offset=0)


def _run_in_new_loop(coro):
    """Run a coroutine in a fresh thread with its own event loop, isolated from LangGraph."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _run_hpc_submit(state: SimulationState, run_folder: str, hpc_remote_dir: str) -> str:
    return _run_in_new_loop(_run_hpc_submit_async(state, run_folder, hpc_remote_dir))


def _run_hpc_status(state: SimulationState, user_msg: str) -> None:
    _run_in_new_loop(_run_hpc_status_async(state, user_msg))


def hpc_executor_node(state: SimulationState) -> dict:
    """
    LangGraph node — handles all HPC lifecycle phases.

    Phase 1 (workflow_stage="script_ready", execution_target="hpc"):
      Generates PBS script, emits pbs_preview SSE — no Globus calls yet.

    Phase 2 (workflow_stage="pbs_pending"):
      Runs create_react_agent with Globus MCP tools to transfer files + submit job.

    Status (intent="hpc_status"):
      Runs agent to check status of all jobs in state["hpc_jobs"].
    """
    intent = state.get("intent", "")
    workflow_stage = state.get("workflow_stage", "initial")
    hpc_jobs = state.get("hpc_jobs") or []

    # ── Phase 1: PBS preview (no Globus) ─────────────────────────────────────
    if workflow_stage == "script_ready" and state.get("execution_target") == "hpc":
        run_folder = run_manager.create_run_folder()
        run_folder_name = os.path.basename(run_folder)

        run_manager.save_lammps_script(run_folder, state["lammps_script"])
        if state.get("ovito_script"):
            with open(os.path.join(run_folder, "ovito_script.py"), "w") as f:
                f.write(state["ovito_script"])

        pbs_script = _generate_pbs_script(run_folder_name, kim_model=state.get("kim_model", ""))
        with open(os.path.join(run_folder, "submit_lammps.sh"), "w") as f:
            f.write(pbs_script)

        _q(state, {
            "type": "pbs_preview",
            "pbs_script": pbs_script,
            "run_folder": run_folder,
        })
        _q(state, {
            "type": "agent_content",
            "agent": "HPC",
            "content": "Here is the PBS script that will be submitted to Swing. Review it and confirm to proceed.",
        })

        return {
            "workflow_stage": "pbs_pending",
            "hpc_run_folder_pending": run_folder,
        }

    # ── Phase 2: Globus submission ────────────────────────────────────────────
    if workflow_stage == "pbs_pending":
        run_folder = state.get("hpc_run_folder_pending", "")
        run_folder_name = os.path.basename(run_folder)
        hpc_remote_dir = f"{config.HPC_RUN_BASE_DIR}/{run_folder_name}"
        description = f"{state.get('material', '')} {state.get('temperature', '')}K".strip()

        globus_task_id = _run_hpc_submit(state, run_folder, hpc_remote_dir)

        new_job = {
            "globus_task_id": globus_task_id or "unknown",
            "hpc_run_folder": run_folder,
            "submitted_at": datetime.now().isoformat(),
            "description": description,
        }
        return {
            "hpc_jobs": hpc_jobs + [new_job],
            "workflow_stage": "initial",
            "hpc_run_folder_pending": None,
        }

    # ── Status query ──────────────────────────────────────────────────────────
    if intent == "hpc_status":
        if not hpc_jobs:
            _q(state, {"type": "agent_content", "agent": "HPC", "content": "No HPC jobs found for this session."})
            return {}

        jobs_summary = "\n".join(
            f"- {j.get('description', 'job')}: "
            f"globus_task_id={j['globus_task_id']}, "
            f"local_run_folder={j['hpc_run_folder']}, "
            f"remote_run_folder={config.HPC_RUN_BASE_DIR}/{os.path.basename(j['hpc_run_folder'])}, "
            f"submitted={j['submitted_at']}"
            for j in hpc_jobs
        )
        user_msg = (
            f"User request: {state.get('user_message', 'check job status')}\n\n"
            f"HPC jobs submitted in this session:\n{jobs_summary}\n\n"
            f"Local Globus endpoint: {config.GLOBUS_LOCAL_ENDPOINT}\n"
            f"HPC Globus endpoint: {config.GLOBUS_HPC_ENDPOINT}\n\n"
            f"If the user wants to transfer results back, transfer the entire "
            f"remote_run_folder to the corresponding local_run_folder."
        )
        _run_hpc_status(state, user_msg)
        return {}

    return {}
