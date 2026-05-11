"""
LangGraph state definition for the ChatMD simulation pipeline.
"""
from typing import TypedDict, Optional, List, Dict


class SimulationState(TypedDict):
    # ── Input (from HTTP request) ─────────────────────────────────────────────
    user_message: str
    workflow_stage: str                   # "initial" | "parameters_proposed" | "script_ready"
    conversation_history: List[Dict]
    pending_script: Optional[str]         # script carried over from previous turn
    ovito_script_param: Optional[str]     # ovito script from previous turn

    # ── Classification ────────────────────────────────────────────────────────
    intent: Optional[str]                 # "execute" | "modify" | "cancel" | "question" | "new_simulation"

    # ── Orchestrator context ──────────────────────────────────────────────────
    registry_context: Optional[str]       # KIM registry string injected into prompt
    analysis: Optional[str]              # full conversational LLM text response

    # ── Extracted simulation parameters ──────────────────────────────────────
    has_parameters: Optional[bool]
    material: Optional[str]
    kim_model: Optional[str]
    temperature: Optional[float]
    ensemble: Optional[str]
    duration_ps: Optional[float]
    system_size: Optional[int]
    visualization_notes: Optional[str]
    camera_angle: Optional[str]

    # ── LAMMPS generation outputs ─────────────────────────────────────────────
    lammps_script: Optional[str]
    dump_filename_pattern: Optional[str]
    total_steps: Optional[int]
    dump_freq_val: Optional[int]
    review_passed: Optional[bool]
    review_issues: Optional[List[str]]
    review_suggestions: Optional[List[str]]

    # ── Visualization outputs ─────────────────────────────────────────────────
    has_custom_viz: Optional[bool]
    viz_request: Optional[str]
    precomputed_properties: Optional[Dict]
    simulation_context: Optional[Dict]    # context dict passed to OVITO generator
    ovito_script: Optional[str]
    target_timesteps: Optional[List[int]]

    # ── HPC Execution ─────────────────────────────────────────────────────────
    execution_target: Optional[str]       # "local" | "hpc" — set from POST body by app.py
    hpc_jobs: Optional[List[Dict]]        # [{globus_task_id, hpc_run_folder, submitted_at, description}]
    hpc_run_folder_pending: Optional[str] # temp: local run_folder between Phase 1 (preview) and Phase 2 (submit)

    # ── Control ───────────────────────────────────────────────────────────────
    error: Optional[str]
    session_id: Optional[str]             # used to look up sse_queue from SESSION_QUEUES in nodes.py
