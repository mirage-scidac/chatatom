"""
LangGraph routing functions for the ChatMD simulation pipeline.

Each function reads the current SimulationState and returns the name of
the next node to execute, or END to stop the graph.
"""
from langgraph.graph import END
from graph.state import SimulationState


def route_after_intent(state: SimulationState) -> str:
    """
    After classify_intent (runs before chat_agent):
    - execute + script_ready + hpc  → hpc_executor (Phase 1: PBS preview)
    - pbs_pending                   → hpc_executor (Phase 2: Globus submission)
    - hpc_status                    → hpc_executor (status query)
    - execute + script_ready        → END (app.py fires local executor directly)
    - analyze                       → analyze_results (skip chat entirely)
    - everything else               → chat_agent
    """
    intent = state.get("intent", "question")
    workflow_stage = state.get("workflow_stage", "initial")
    execution_target = state.get("execution_target", "local")

    if workflow_stage == "pbs_pending":
        return "hpc_executor"
    if workflow_stage == "script_ready" and execution_target == "hpc":
        return "hpc_executor"
    if intent == "hpc_status":
        return "hpc_executor"
    if intent == "execute_hpc":
        return "hpc_executor"
    if intent == "execute" and workflow_stage == "script_ready":
        return END
    if intent == "analyze":
        return "analyze_results"
    return "chat_agent"


def route_after_extract(state: SimulationState) -> str:
    """
    After extract_params:
    - If no simulation parameters extracted → END (chat_node already asked for more info;
      kim_node will have fetched models into state for next turn)
    - Otherwise → generate_lammps
    """
    if not state.get("has_parameters"):
        return END
    return "generate_lammps"


def route_after_lammps(state: SimulationState) -> str:
    """
    After generate_lammps:
    - If review passed → detect_viz
    - If review failed → END (app.py emits review_failed from final state)
    """
    if state.get("review_passed"):
        return "detect_viz"
    return END


def route_after_detect_viz(state: SimulationState) -> str:
    """
    After detect_viz:
    - If custom visualization requested → generate_ovito
    - Otherwise → END (default viz; app.py emits scripts_ready_for_review)
    """
    if state.get("has_custom_viz"):
        return "generate_ovito"
    return END
