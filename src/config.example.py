# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

# ChatAtom Configuration Template
# ─────────────────────────────────────────────────────────────────────────────
# Copy this file to `config.py` and fill in your own values:
#     cp src/config.example.py src/config.py
#
# IMPORTANT: src/config.py is gitignored — never commit it.
# All values containing secrets, API keys, or personal info are placeholders here.
# ─────────────────────────────────────────────────────────────────────────────


# ── LLM Endpoint ─────────────────────────────────────────────────────────────
# Base URL of your LLM API (OpenAI-compatible). Examples:
#   - LM Studio (local):  http://localhost:1234/v1
#   - Ollama (local):     http://localhost:11434/v1
#   - OpenAI:             https://api.openai.com/v1
#   - Custom proxy:       https://your-proxy.example.com/v1
API_BASE_URL = "https://your-llm-endpoint.example.com/v1/"
API_KEY      = "your-api-key-here"


# ── Per-Agent Models ─────────────────────────────────────────────────────────
# You can route different agents to different models if your API supports it.
# Empty strings ("") fall back to DEFAULT_MODEL below.
ORCHESTRATOR_MODEL     = ""   # Orchestrator — needs strong reasoning
LAMMPS_GENERATOR_MODEL = ""   # Generator — needs technical accuracy
LAMMPS_REVIEWER_MODEL  = ""   # Reviewer — needs critical analysis
OVITO_SCRIPT_MODEL     = ""   # OVITO script generation

# Default model used when an agent-specific model is empty.
DEFAULT_MODEL = "your-default-model-name"


# ── Per-Agent Temperatures ───────────────────────────────────────────────────
# Lower = more deterministic. 0.1 is a good default for code-heavy tasks.
#
# Reasoning models (the gpt5*/o-series) accept ONLY their own default and
# reject any explicit value with a 400. Flip this one switch for those models:
# every agent below then omits the parameter, values left intact for later.
MODEL_REJECTS_TEMPERATURE = False

ORCHESTRATOR_TEMPERATURE = 0.1   # Orchestrator ReAct agent
DSPY_TEMPERATURE         = 0.1   # LAMMPS + OVITO generator/reviewer (one shared
                                 # dspy.LM, so these cannot differ from each other)
ANALYST_TEMPERATURE      = 0.3   # Results analyst
VISION_TEMPERATURE       = 0.3   # Image / render summaries


def temp_kwargs(value):
    """Spread into an LLM constructor or request; empty when the temperature
    must be omitted:

        ChatOpenAI(model=..., **config.temp_kwargs(config.ANALYST_TEMPERATURE))
    """
    if MODEL_REJECTS_TEMPERATURE or value is None:
        return {}
    return {"temperature": value}


# ── Knowledge Base ───────────────────────────────────────────────────────────
# KnowledgeBase.LAMMPS_ERROR_RULE collects general LAMMPS scripting rules
# (declaration order, lattice-vs-box units, barostat choice for non-cubic cells,
# non-periodic boundaries, delete_atoms). It is injected into both the generator
# and the reviewer contexts.
#
# Set False to withhold it from both, e.g. to A/B whether the rule changes the
# generated scripts. Flipping this needs no code edit; uvicorn --reload-dir src
# picks it up.
ENABLE_LAMMPS_ERROR_RULE = True


# ── LAMMPS Execution ─────────────────────────────────────────────────────────
LAMMPS_CORES         = 4         # Default cores for parallel runs
LAMMPS_USE_MPI       = True      # Use MPI when cores > 1
LAMMPS_MPI_COMMAND   = "mpirun"  # Could be mpirun, mpiexec, srun, etc.
LAMMPS_EXECUTABLE    = "lmp"     # LAMMPS binary on PATH
LAMMPS_USE_HWTHREADS = True      # Use hardware threads on hyperthreaded systems


# ── Trajectory & Visualization ───────────────────────────────────────────────
MAX_TRAJECTORY_FRAMES   = 50    # Max frames to capture during simulation
OVITO_SCRIPT_MAX_RETRIES = 3    # Max generate/review attempts for OVITO
THERMO_STREAM_INTERVAL  = 100   # Steps between live thermo updates


# ── Application Server ───────────────────────────────────────────────────────
APP_PORT = 2002


# ── Memory Backend ───────────────────────────────────────────────────────────
# "auto"   — try Mem0, fall back to SimpleMemory
# "mem0"   — Mem0 only (requires embeddings API)
# "simple" — JSON file storage
MEMORY_BACKEND   = "auto"
MEMORY_FILE_PATH = "data/memory.json"


# ── Conversation & Sessions ──────────────────────────────────────────────────
MAX_CONVERSATION_TURNS = 10                 # User/assistant pairs to retain
SESSION_DB_PATH        = "data/sessions.db" # LangGraph SqliteSaver path
SESSION_EXPIRY_DAYS    = 30                 # Auto-delete older sessions


# ── HPC / Globus (optional — only needed for HPC execution path) ─────────────
# Get endpoint UUIDs from https://app.globus.org/endpoints
# Get a Globus Compute endpoint UUID from your HPC site admin or by
# registering one with `globus-compute-endpoint configure`.
GLOBUS_LOCAL_ENDPOINT      = "your-local-globus-endpoint-uuid"
GLOBUS_HPC_ENDPOINT        = "your-hpc-globus-endpoint-uuid"
GLOBUS_COMPUTE_ENDPOINT_ID = "your-globus-compute-endpoint-uuid"

HPC_RUN_BASE_DIR    = "/path/to/your/runs/on/hpc"   # Where ChatAtom writes run folders on HPC
HPC_DEFAULT_NODES   = 1
HPC_DEFAULT_CORES   = 8
HPC_DEFAULT_WALLTIME = "00:30:00"
HPC_PROJECT_ID      = "YOUR_PROJECT_ID"             # PBS allocation / project ID
HPC_USER_EMAIL      = "you@example.com"             # PBS notification email
