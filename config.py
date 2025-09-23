# LLM Endpoint Configuration
# Example for a local server like LM Studio
API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "not-needed"

# Per-Agent Model Configuration
# You can use different models for different agents based on their requirements
ORCHESTRATOR_MODEL = "gemini25pro"  # Orchestrator needs good reasoning
LAMMPS_GENERATOR_MODEL = "gemini25pro"  # Generator needs technical accuracy
LAMMPS_REVIEWER_MODEL = "gemini25pro"  # Reviewer needs critical analysis

# Default model (fallback if agent-specific model not defined)
DEFAULT_MODEL = "gemini25pro"

# Temperature settings per agent (optional)
ORCHESTRATOR_TEMPERATURE = 0.1
LAMMPS_GENERATOR_TEMPERATURE = 0.1
LAMMPS_REVIEWER_TEMPERATURE = 0.1

# LAMMPS Execution Configuration
LAMMPS_CORES = 4  # Default number of cores for LAMMPS simulations
LAMMPS_USE_MPI = True  # Whether to use MPI for parallel execution (when cores > 1)
LAMMPS_MPI_COMMAND = "mpirun"  # MPI command (could be mpirun, mpiexec, srun, etc.)
LAMMPS_EXECUTABLE = "lmp"  # LAMMPS executable name
