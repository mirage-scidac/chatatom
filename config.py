# LLM Endpoint Configuration
# Example for a local server like LM Studio
API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "not-needed"

# Per-Agent Model Configuration
# You can use different models for different agents based on their requirements
ORCHESTRATOR_MODEL = ""  # Orchestrator needs good reasoning (handles simulation + visualization)
LAMMPS_GENERATOR_MODEL = ""  # Generator needs technical accuracy
LAMMPS_REVIEWER_MODEL = ""  # Reviewer needs critical analysis
OVITO_SCRIPT_MODEL = ""  # OVITO script generation needs visualization expertise

# Default model (fallback if agent-specific model not defined)
DEFAULT_MODEL = "gemini25flash"

# Temperature settings per agent (optional)
ORCHESTRATOR_TEMPERATURE = 0.1
LAMMPS_GENERATOR_TEMPERATURE = 0.1
LAMMPS_REVIEWER_TEMPERATURE = 0.1
OVITO_SCRIPT_TEMPERATURE = 0.1

# LAMMPS Execution Configuration
LAMMPS_CORES = 4  # Default number of cores for LAMMPS simulations
LAMMPS_USE_MPI = True  # Whether to use MPI for parallel execution (when cores > 1)
LAMMPS_MPI_COMMAND = "mpirun"  # MPI command (could be mpirun, mpiexec, srun, etc.)
LAMMPS_EXECUTABLE = "lmp"  # LAMMPS executable name
