<h1><img src="assets/chatatom-title.svg" alt="ChatAtom" height="56"></h1>

A web-based application that allows users to run molecular dynamics simulations with LAMMPS using KIM (Knowledge Base of Interatomic Models) potentials and visualize the results with OVITO using natural language commands.

**Important**: This application exclusively uses KIM potentials for accurate, reproducible simulations. Traditional LAMMPS pair styles are not supported.

## Installation

This project uses Conda for managing the environment and complex dependencies, and `uv` for fast Python package installation.

### Pixi Quickstart

The repository includes a [`pixi.toml`](pixi.toml) environment definition and generated lockfile. To create the environment and run the app:

```bash
pixi install
pixi run run-app
```

The application will be available at `http://localhost:2002`.
API docs (auto-generated) at `http://localhost:2002/docs`.

`run-app` is a pixi task that runs:

```bash
uvicorn src.app:socket_app --host 0.0.0.0 --port 2002 --reload --reload-dir src
```

`--reload-dir src` is important: `--reload` watches the working directory for code changes, but limiting it to `src/` prevents uvicorn from restarting when LAMMPS writes new files into `runs/` during a simulation (which would kill the in-flight subprocess + WebSocket connection).

### Prerequisites

- **pixi**: Pixi manages the entire environment (Python, LAMMPS, KIM API, Python packages, etc.) via `pixi.toml` + `pixi.lock`. Install from https://pixi.sh.

### 1. Install Environment

```bash
pixi install
```

That's it — pixi reads `pixi.toml` + `pixi.lock` and builds the complete environment with LAMMPS, KIM API, FastAPI, all Python packages, and dependencies.

### 2. Install Fortran Compiler (Required for KIM Models)

Many KIM potential models contain Fortran code that needs to be compiled during installation. If pixi didn't pick one up automatically, install gfortran:

```bash
# Easiest — use whatever your OS provides
brew install gcc          # macOS
apt-get install gfortran  # Debian/Ubuntu
```

To check if you already have gfortran:
```bash
which gfortran
```

### 3. Install KIM Models

Install commonly used KIM potential models:

```bash
# Install verified working models (use pixi run so the right kim-api binary is used)
pixi run kim-api-collections-management install user LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004
pixi run kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_005
pixi run kim-api-collections-management install user EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
pixi run kim-api-collections-management install user EAM_Dynamo_FoilesBaskesDaw_1986Universal3_Cu__MO_666348409573_005
```

**Note**: You can install additional KIM models as needed using:
```bash
pixi run kim-api-collections-management install user <MODEL_ID>
```

The chat UI also has a **KIM Potentials panel** for searching and downloading models with progress tracking — usually easier than the CLI.

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

3. **LAMMPS with MPI support** is included in the `pixi install` environment automatically (the `lammps` conda package on conda-forge ships an MPI build).

The system will automatically:
- Use serial execution (Python interface) for single core
- Use MPI execution (subprocess) for multiple cores
- Fall back to serial execution if MPI is not available
- Validate core count (must be at least 1)

## Running the Application

Run the FastAPI / uvicorn server (preferred — uses pixi's environment):

```bash
pixi run run-app
```

Or with a manually-activated environment:

```bash
uvicorn src.app:socket_app --host 0.0.0.0 --port 2002 --reload --reload-dir src
```

The application will be available at `http://localhost:2002`.
Auto-generated API docs at `http://localhost:2002/docs`.

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

Additional KIM models can be installed as needed. 
### Finding More KIM Models

Browse available models at [OpenKIM.org](https://openkim.org) and install them using:

```bash
kim-api-collections-management install user <MODEL_ID>
```

## Running Tests

To verify the installation and run the end-to-end tests for the agent workflow, use `pytest` from the project root:

```bash
pixi run python -m pytest
```

## Troubleshooting

- **`ModuleNotFoundError`**: Run via `pixi run ...` so the pixi environment is active. Or run `pixi shell` to drop into an interactive shell with the env activated.
- **KIM Model Not Found**: Install via `pixi run kim-api-collections-management install user <MODEL_ID>` — or use the KIM Potentials panel in the UI.
- **KIM API Issues**: KIM API ships in the pixi environment automatically. If broken, try `pixi install` to refresh.
- **LAMMPS KIM Support**: The conda-forge LAMMPS package includes KIM support by default; pixi pulls this version.
- **OVITO License**: The `ovito` package requires a license. Place your `ovito.lic` file in a standard location (e.g., `~/.config/ovito/`) or in the project folder. See the OVITO documentation for more details.

## Documentation
- [OpenKIM Website](https://openkim.org) - Browse available interatomic models
- [LAMMPS KIM Documentation](https://docs.lammps.org/kim_commands.html) - Official LAMMPS KIM command reference
