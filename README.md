# LLM Molecular Dynamics Studio

A web-based application that allows users to run molecular dynamics simulations with LAMMPS and visualize the results with OVITO using natural language commands.

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

- **LAMMPS (via Conda):**
  ```bash
  conda install -c conda-forge lammps -y
  ```

- **Python Packages (via uv):**
  ```bash
  uv pip install ovito openai flask pytest
  ```

## Running the Application

With the `chatmd` environment active, run the Flask server:

```bash
python app.py
```

The application will be available at `http://localhost:5001`.

## Running Tests

To verify the installation and run the end-to-end tests for the agent workflow, use `pytest` from the project root:

```bash
python -m pytest
```

## Troubleshooting

- **`ModuleNotFoundError`**: Ensure the `chatmd` Conda environment is active (`conda activate chatmd`).
- **OVITO License**: The `ovito` package requires a license. Place your `ovito.lic` file in a standard location (e.g., `~/.config/ovito/`) or in the project folder. See the OVITO documentation for more details.