"""
LangGraph compiled simulation graph for ChatMD.

Usage:
    from graph.graph import simulation_graph
    final_state = simulation_graph.invoke(initial_state)
"""
import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from graph.state import SimulationState
from graph.nodes import (
    chat_node,
    intent_node,
    extract_params_node,
    generate_lammps_node,
    detect_viz_node,
    generate_ovito_node,
    analyze_results_node,
)
from graph.hpc_executor_node import hpc_executor_node
from graph.routing import (
    route_after_intent,
    route_after_extract,
    route_after_lammps,
    route_after_detect_viz,
)


def build_simulation_graph():
    g = StateGraph(SimulationState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("chat_agent",       chat_node)
    g.add_node("classify_intent",  intent_node)
    g.add_node("extract_params",   extract_params_node)
    g.add_node("generate_lammps",  generate_lammps_node)
    g.add_node("detect_viz",       detect_viz_node)
    g.add_node("generate_ovito",   generate_ovito_node)
    g.add_node("analyze_results",  analyze_results_node)
    g.add_node("hpc_executor",     hpc_executor_node)

    # ── Wire edges ────────────────────────────────────────────────────────────
    g.add_edge(START, "classify_intent")

    g.add_conditional_edges(
        "classify_intent", route_after_intent,
        {"chat_agent": "chat_agent", "analyze_results": "analyze_results",
         "hpc_executor": "hpc_executor", END: END}
    )

    g.add_edge("chat_agent", "extract_params")

    g.add_edge("analyze_results", END)

    g.add_conditional_edges(
        "extract_params", route_after_extract,
        {"generate_lammps": "generate_lammps", END: END}
    )

    g.add_conditional_edges(
        "generate_lammps", route_after_lammps,
        {"detect_viz": "detect_viz", END: END}
    )

    g.add_conditional_edges(
        "detect_viz", route_after_detect_viz,
        {"generate_ovito": "generate_ovito", END: END}
    )

    g.add_edge("generate_ovito", END)
    g.add_edge("hpc_executor", END)

    db_path = getattr(config, "SESSION_DB_PATH", "data/sessions.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    # check_same_thread=False needed because app uses multiple threads
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=DELETE")  # no .wal/.shm files 
    conn.execute("PRAGMA wal_autocheckpoint=0")  # disable WAL checkpointing
    checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)


simulation_graph = build_simulation_graph()
