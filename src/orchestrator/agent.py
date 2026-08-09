"""ChatMD orchestrator — a langgraph create_react_agent.

The main entry point for ChatMD. app.py builds this graph at startup
(with an AsyncSqliteSaver checkpointer) and invokes it for every chat
request.

Architecture:
  - Built on langgraph.prebuilt.create_react_agent (battle-tested ReAct loop)
  - System prompt assembled by orchestrator.system_prompt (tool catalogue +
    dynamic skill catalogue + two-turn rule + domain prompt from prompts.py)
  - Tools from orchestrator.tools.ORCHESTRATOR_TOOLS
  - Checkpointer: AsyncSqliteSaver (built in app.py's startup handler so
    it's instantiated inside an event loop, as AsyncSqliteSaver requires)

Streaming + SSE: SSE events emitted from inside tool bodies use the
_q(state, event) helper from graph.nodes which writes to
SESSION_QUEUES[session_id]. session_id flows in via the orchestrator
graph's state (MessagesState subclass; see OrchestratorState below).
"""
import os
import sys

from typing import NotRequired

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import create_react_agent, ToolNode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from orchestrator.system_prompt import build_orchestrator_system_prompt
from orchestrator.tools import ORCHESTRATOR_TOOLS


class OrchestratorState(MessagesState):
    """State for the orchestrator graph.

    Extends MessagesState (which provides `messages: list[BaseMessage]`)
    with top-level fields needed by tools and the SSE plumbing:

    - session_id: routes SSE events through SESSION_QUEUES (set per request
      by app.py; survives across turns via checkpoint).
    - execution_target: "local" or "hpc" (per-turn hint from frontend).
    - workflow_stage: "initial" | "pbs_pending" | "script_ready" — used by
      the HPC tools' two-call pattern (Phase 1e).
    - hpc_jobs / hpc_run_folder_pending: HPC bookkeeping that survives
      across the PBS-edit pause (Phase 1e).
    """
    session_id: str | None
    execution_target: str | None
    workflow_stage: str | None
    hpc_jobs: list | None
    hpc_run_folder_pending: str | None

    # ── User-edited textarea content from the approve cards ──────────────────
    # Threaded in by app.py from body fields so tools can read the real user-
    # edited content without asking the LLM to retype long scripts as args.
    pending_script: str | None       # user-edited LAMMPS script
    ovito_script_param: str | None   # user-edited OVITO script
    pbs_script_edited: str | None    # user-edited PBS script (HPC Phase 2)

    # ── Simulation parameters from the most-recent simulate(...) call ────────
    # Persisted so the cascade can auto-title the session ("Cu 300K") and so
    # follow-up turns can reference the current simulation without re-walking
    # message history.
    material: str | None
    temperature: float | None
    ensemble: str | None
    duration_ps: float | None
    system_size: int | None
    kim_model: str | None

    # ── Written by `simulate` tool, read by the post-graph cascade ────────────
    # The cascade in app.py fires `scripts_ready_for_review` only when both
    # lammps_script is present AND auto_approve is falsy. Campaign iteration
    # calls (auto_approve=True) skip the card so the orchestrator can run
    # the script within its own ReAct loop without user intervention.
    lammps_script: str | None
    ovito_script: str | None
    dump_filename_pattern: str | None
    target_timesteps: list | None
    camera_angle: str | None
    auto_approve: bool | None
    # Campaign-only: typed knob values the LLM passed to simulate() during
    # an iteration. Written by simulate() as a per-iteration echo. In v2
    # this is a convenience mirror — trial history itself lives on disk
    # (trials.jsonl) and is read via campaign_get_state().
    campaign_knobs: dict | None

    # Campaign v2: id of the active campaign. Written by campaign_start via
    # Command(update={"campaign_id": ...}); read by every other campaign_*
    # tool via InjectedState so those tools take no id argument themselves.
    # None for chat-driven turns. Persists across the ReAct trace of a
    # single campaign turn via the AsyncSqliteSaver checkpointer.
    campaign_id: str | None
    # Surfaces the script-review status from the simulate subgraph so the
    # cascade can branch on review_failed (rare but needs explicit handling).
    review_passed: bool | None
    review_issues: list | None
    review_suggestions: list | None
    has_custom_viz: bool | None

    # create_react_agent injects this internally to enforce its recursion
    # cap. NotRequired since callers don't set it.
    remaining_steps: NotRequired[RemainingSteps]


def _build_model() -> ChatOpenAI:
    """Construct the chat model from config. ORCHESTRATOR_MODEL takes
    precedence over DEFAULT_MODEL; empty string falls back to DEFAULT_MODEL.
    """
    model_name = (
        getattr(config, "ORCHESTRATOR_MODEL", "")
        or config.DEFAULT_MODEL
    )
    return ChatOpenAI(
        model=model_name,
        api_key=config.API_KEY,
        base_url=config.API_BASE_URL,
        temperature=getattr(config, "ORCHESTRATOR_TEMPERATURE", 0.1),
    )


# Path for AsyncSqliteSaver checkpoints. Configurable via
# config.ORCHESTRATOR_DB_PATH; defaults to data/orchestrator_sessions.db.
ORCHESTRATOR_DB_PATH = getattr(
    config, "ORCHESTRATOR_DB_PATH", "data/orchestrator_sessions.db"
)


def _prompt(state: OrchestratorState) -> list:
    """Dynamic system prompt: rebuilt per invocation so it can read
    session_id / user_message / history out of state for skill catalogue
    customization (future-proofing — Phase 1a doesn't use these args yet).

    Returns the list of messages create_react_agent will send to the model:
    [SystemMessage(...), *state["messages"]].
    """
    from langchain_core.messages import SystemMessage

    user_msg = ""
    messages = state.get("messages") or []
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            user_msg = m.content if isinstance(m.content, str) else ""
            break

    prompt_text = build_orchestrator_system_prompt(
        session_id=state.get("session_id"),
        user_message=user_msg,
        conversation_history=None,
    )
    return [SystemMessage(content=prompt_text), *messages]


def build_orchestrator_graph(checkpointer=None):
    """Build the orchestrator graph with a caller-supplied checkpointer.

    Checkpointer is passed in (not constructed here) because
    AsyncSqliteSaver must be instantiated inside a running event loop.
    app.py creates one per-process AsyncSqliteSaver at startup and passes
    it here.
    """
    return create_react_agent(
        model=_build_model(),
        tools=ORCHESTRATOR_TOOLS,  
        # tools = ToolNode(ORCHESTRATOR_TOOLS, handle_tool_errors=True),
        prompt=_prompt,
        state_schema=OrchestratorState,
        checkpointer=checkpointer,
        name="chatmd_orchestrator",
    )


# Default compiled graph WITHOUT a checkpointer — usable for smoke tests
# that don't need session persistence. Production should call
# build_orchestrator_graph(checkpointer=...) after the event loop is up.
orchestrator_graph = build_orchestrator_graph()
