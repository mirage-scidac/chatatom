# ChatMD

A web-based application that allows users to run molecular dynamics simulations with LAMMPS using KIM (Knowledge Base of Interatomic Models) potentials and visualize the results with OVITO using natural language commands.

**Important**: This application exclusively uses KIM potentials for accurate, reproducible simulations. Traditional LAMMPS pair styles are not supported.

## Installation

This project uses Conda for managing the environment and complex dependencies, and `uv` for fast Python package installation.

### Prerequisites

- **Miniconda/Anaconda**: For the `conda` package manager.
- **uv**: The fast Python package installer. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### 1. Create Environment

Create and activate a Conda environment named `chatmd`.

```bash
conda create -n chatmd python=3.11 -y
conda activate chatmd
```

### 2. Install Dependencies

With the `chatmd` environment active, install the required packages.

- **LAMMPS with KIM support (via Conda):**
  ```bash
  conda install -c conda-forge lammps -y
  ```

- **KIM API (Required for KIM potentials):**
  ```bash
  conda install -c conda-forge kim-api -y
  ```

- **Python Packages (via uv):**
  ```bash
  conda install uv pip
  uv pip install ovito openai flask flask-socketio pytest requests
  ```

### 3. Install Fortran Compiler (Required for KIM Models)

Many KIM potential models contain Fortran code that needs to be compiled during installation. Install gfortran via conda:

```bash
conda install -c conda-forge gfortran -y
```

**Note**: You can skip this step if you already have a Fortran compiler (gfortran) installed on your system. To check if you have gfortran installed, run:
```bash
which gfortran
```

### 4. Install KIM Models

Install commonly used KIM potential models:

```bash
# Install verified working models
kim-api-collections-management install user LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004
kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_005
kim-api-collections-management install user EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
kim-api-collections-management install user EAM_Dynamo_FoilesBaskesDaw_1986Universal3_Cu__MO_666348409573_005
```

**Note**: You can install additional KIM models as needed using:
```bash
kim-api-collections-management install user <MODEL_ID>
```

## Configuration

The application can be configured by editing the `config.py` file. Key configuration options include:

### LLM Configuration
- `API_BASE_URL`: URL of your LLM API endpoint
- `API_KEY`: API key for authentication
- Model settings for different agents (Orchestrator, Generator, Reviewer)

### LAMMPS Execution Configuration
- `LAMMPS_CORES`: Default number of cores for LAMMPS simulations (default: 1)
- `LAMMPS_USE_MPI`: Enable/disable MPI for parallel execution (default: True)
- `LAMMPS_MPI_COMMAND`: MPI command to use (default: "mpirun")
- `LAMMPS_EXECUTABLE`: LAMMPS executable name (default: "lmp")

### Parallel Execution

The application supports parallel LAMMPS execution using MPI. To configure parallel execution:

1. **Set the number of cores** in `config.py`:
   ```python
   LAMMPS_CORES = 4  # Use 4 cores by default
   ```

2. **Ensure MPI is available** on your system:
   ```bash
   # Check if MPI is installed
   mpirun --version
   ```

3. **Install LAMMPS with MPI support** (if not already done):
   ```bash
   conda install -c conda-forge lammps=*=mpi_* -y
   ```

The system will automatically:
- Use serial execution (Python interface) for single core
- Use MPI execution (subprocess) for multiple cores
- Fall back to serial execution if MPI is not available
- Validate core count (must be at least 1)

## Running the Application

With the `chatmd` environment active, run the Flask server:

```bash
python src/app.py
```

The application will be available at `http://localhost:5001`.

### Real-Time Features

The application provides several real-time capabilities:

- **Streaming Agent Responses**: AI agents stream their responses in real-time using Server-Sent Events (SSE)
- **Live Simulation Monitoring**: WebSocket integration provides real-time thermodynamic data during LAMMPS execution
- **Dynamic Visualization**: Charts update automatically as simulation data becomes available
- **Parallel Execution**: Automatic MPI support for faster simulations with graceful fallback to serial execution

## KIM Potentials

This application uses the Knowledge Base of Interatomic Models (KIM) for all molecular dynamics simulations. KIM provides:

- **Standardized interface** for interatomic potentials
- **Reproducible simulations** across different systems
- **Verified models** with proper documentation
- **Automatic parameter handling** (no manual pair coefficients)

Additional KIM models can be installed as needed. See the [KIM documentation](docs/kim_potential/kim_potential_reference.md) for more details.

### Finding More KIM Models

Browse available models at [OpenKIM.org](https://openkim.org) and install them using:

```bash
kim-api-collections-management install user <MODEL_ID>
```

## Running Tests

To verify the installation and run the end-to-end tests for the agent workflow, use `pytest` from the project root:

```bash
python -m pytest
```

## Troubleshooting

- **`ModuleNotFoundError`**: Ensure the `chatmd` Conda environment is active (`conda activate chatmd`).
- **KIM Model Not Found**: Install the required model using `kim-api-collections-management install user <MODEL_ID>`.
- **KIM API Issues**: Ensure KIM API is properly installed with `conda install -c conda-forge kim-api`.
- **LAMMPS KIM Support**: Verify LAMMPS was compiled with KIM support (should be automatic with conda installation).
- **OVITO License**: The `ovito` package requires a license. Place your `ovito.lic` file in a standard location (e.g., `~/.config/ovito/`) or in the project folder. See the OVITO documentation for more details.

## Documentation

- [KIM Potential Reference](docs/kim_potential/kim_potential_reference.md) - Comprehensive guide to using KIM potentials
- [OpenKIM Website](https://openkim.org) - Browse available interatomic models
- [LAMMPS KIM Documentation](https://docs.lammps.org/kim_commands.html) - Official LAMMPS KIM command reference
