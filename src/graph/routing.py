# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
LangGraph routing functions for the simulate subgraph.

Each function reads the current SimulationState and returns the name of
the next node to execute, or END to stop the graph.
"""
from langgraph.graph import END
from graph.state import SimulationState


def route_after_lammps(state: SimulationState) -> str:
    """
    After generate_lammps:
    - If review passed → detect_viz
    - If review failed → END (cascade in app.py emits review_failed)
    """
    if state.get("review_passed"):
        return "detect_viz"
    return END


def route_after_detect_viz(state: SimulationState) -> str:
    """
    After detect_viz:
    - If custom visualization requested → generate_ovito
    - Otherwise → END (default viz; cascade emits scripts_ready_for_review)
    """
    if state.get("has_custom_viz"):
        return "generate_ovito"
    return END
