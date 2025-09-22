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
