# LLM Endpoint Configuration
# Example for a local server like LM Studio
API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "not-needed"

# Per-Agent Model Configuration
# You can use different models for different agents based on their requirements
ORCHESTRATOR_MODEL = "gemini25pro"  # Orchestrator needs good reasoning (handles simulation + visualization)
LAMMPS_GENERATOR_MODEL = "claudesonnet45"  # Generator needs technical accuracy
LAMMPS_REVIEWER_MODEL = "claudeopus41"  # Reviewer needs critical analysis
OVITO_SCRIPT_MODEL = "claudesonnet45"  # OVITO script generation needs visualization expertise

# Default model (fallback if agent-specific model not defined)
DEFAULT_MODEL = "claudeopus41"

# Temperature settings per agent (optional)
ORCHESTRATOR_TEMPERATURE = 0.1
LAMMPS_GENERATOR_TEMPERATURE = 0.1
LAMMPS_REVIEWER_TEMPERATURE = 0.1
OVITO_SCRIPT_TEMPERATURE = 0.1

# LAMMPS Execution Configuration
LAMMPS_CORES = 32  # Default number of cores for LAMMPS simulations
LAMMPS_USE_MPI = True  # Whether to use MPI for parallel execution (when cores > 1)
LAMMPS_MPI_COMMAND = "mpirun"  # MPI command (could be mpirun, mpiexec, srun, etc.)
LAMMPS_EXECUTABLE = "lmp"  # LAMMPS executable name

# Trajectory Visualization Configuration
MAX_TRAJECTORY_FRAMES = 20  # Maximum number of frames to capture during simulation

# Thermodynamic Data Streaming Configuration
THERMO_STREAM_INTERVAL = 100  # Steps between thermo data updates during live streaming (for live plots)

# Application Server Configuration
APP_PORT = 5002  # Port for the Flask application server
