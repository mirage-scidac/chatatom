# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
SimulationState — TypedDict used by the simulate subgraph nodes and the
analyze_results_node. Populated by the orchestrator tools that wrap them.

Most top-level orchestrator concerns (hpc_jobs, execution_target,
workflow_stage, lammps_script as cross-turn state, etc.) live on
OrchestratorState in src/orchestrator/agent.py. This state is per-tool-call.
"""
from typing import TypedDict, Optional, List, Dict


class SimulationState(TypedDict):
    # ── Per-tool-call input ──────────────────────────────────────────────────
    user_message: str                     # raw prose context (proposal text)
    session_id: Optional[str]             # SSE queue lookup key
    conversation_history: List[Dict]      # analyst tool reads recent context
    analysis: Optional[str]               # generate_lammps reads this as the
                                           #   simulation_details prose

    # ── Extracted simulation parameters (set by simulate tool args) ──────────
    material: Optional[str]
    kim_model: Optional[str]
    temperature: Optional[float]
    ensemble: Optional[str]
    duration_ps: Optional[float]
    system_size: Optional[int]
    visualization_notes: Optional[str]
    camera_angle: Optional[str]

    # ── LAMMPS generation outputs (set by generate_lammps_node) ──────────────
    lammps_script: Optional[str]
    dump_filename_pattern: Optional[str]
    total_steps: Optional[int]
    dump_freq_val: Optional[int]
    review_passed: Optional[bool]
    review_issues: Optional[List[str]]
    review_suggestions: Optional[List[str]]

    # ── Visualization outputs (set by detect_viz_node + generate_ovito_node) ─
    has_custom_viz: Optional[bool]
    viz_request: Optional[str]
    precomputed_properties: Optional[Dict]
    simulation_context: Optional[Dict]
    ovito_script: Optional[str]

    # ── HPC: read by hpc_executor_node helpers ──────────────────────────────
    # hpc_submit tool builds a minimal SimulationState dict with these for
    # the hpc_executor_node helpers (_run_hpc_submit, _generate_pbs_script).
    execution_target: Optional[str]
    hpc_jobs: Optional[List[Dict]]
    hpc_run_folder_pending: Optional[str]
    pending_script: Optional[str]
    ovito_script_param: Optional[str]
    pbs_script_edited: Optional[str]
