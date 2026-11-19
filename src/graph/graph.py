# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
LangGraph compiled subgraph for the orchestrator's `simulate` tool.

Usage:
    from graph.graph import simulate_subgraph
    final_state = simulate_subgraph.invoke(initial_state)

Slim LAMMPS+OVITO pipeline. Starts at generate_lammps (not extract_params)
because the orchestrator-LLM has already extracted structured params and
passes them as tool args — we just seed SimulationState directly. No
checkpointer — invoked synchronously per-tool-call; persistence is handled
at the orchestrator level (AsyncSqliteSaver).

    START → generate_lammps → (review_passed?) → detect_viz
                                               → (has_custom_viz?) → generate_ovito
                                                                   → END
"""
import os
import sys

from langgraph.graph import StateGraph, START, END

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.state import SimulationState
from graph.nodes import (
    generate_lammps_node,
    detect_viz_node,
    generate_ovito_node,
)
from graph.routing import (
    route_after_lammps,
    route_after_detect_viz,
)


def build_simulate_subgraph():
    g = StateGraph(SimulationState)
    g.add_node("generate_lammps", generate_lammps_node)
    g.add_node("detect_viz",      detect_viz_node)
    g.add_node("generate_ovito",  generate_ovito_node)

    g.add_edge(START, "generate_lammps")
    g.add_conditional_edges("generate_lammps", route_after_lammps,
                            {"detect_viz": "detect_viz", END: END})
    g.add_conditional_edges("detect_viz", route_after_detect_viz,
                            {"generate_ovito": "generate_ovito", END: END})
    g.add_edge("generate_ovito", END)
    return g.compile()


simulate_subgraph = build_simulate_subgraph()
