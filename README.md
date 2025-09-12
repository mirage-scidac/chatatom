# LAMMPS & OVITO Python APIs — Installation Guide

Set up a robust Python environment to run molecular dynamics simulations with LAMMPS and visualize results with OVITO. This guide uses a hybrid approach that combines Conda and uv, leveraging each where it excels:

- Conda: Installs the complex, C++-based LAMMPS application and its system-level dependencies.
- uv: Rapidly installs pure Python packages like OVITO within the stable environment Conda creates.

## Prerequisites

Install the following first:

- Miniconda or Anaconda (provides the `conda` package manager). Download from the official Anaconda website.
- `uv` (a fast Python package installer from Astral).

Install `uv`:

- macOS, Linux, or WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

## Step 1: Create and activate a Conda environment

Creating an isolated environment prevents conflicts with your other projects.

- Create the environment named `md-studio` with Python 3.11:
```bash
conda create -n md-studio python=3.11
```

- Activate the environment (do this every time you work on this project):
```bash
conda activate md-studio
```

Your terminal prompt should now show `(md-studio)`.

## Step 2: Install LAMMPS (via conda-forge)

With the environment active, install the pre-compiled LAMMPS package from the `conda-forge` channel:
```bash
conda install -c conda-forge lammps
```

Conda will install the core C++ application and its Python wrapper.

## Step 3: Install the OVITO Python API (via uv)

With the same environment still active, use `uv` to install OVITO:
```bash
uv pip install ovito
```

License note: The `ovito` package requires a license. Place your `ovito.lic` file in one of the locations OVITO searches (for example, your user configuration directory like `~/.config/ovito/` on Linux) or directly in your project folder. Refer to OVITO’s documentation for platform-specific license file locations.

## Step 4: Verify your installation

Make sure your `(md-studio)` environment is active, then start Python:
```bash
python
```

In the Python interpreter, run:
```python
# Import the LAMMPS library
from lammps import lammps
print("LAMMPS Python module imported successfully!")

# Import the OVITO library
from ovito.io import import_file
print("OVITO Python module imported successfully!")

# Exit the interpreter
exit()
```

If both success messages appear without errors, your environment is ready.

## Why use both Conda and uv?

Using two package managers lets you use the best tool for each task:

- Conda (the general contractor)
  - Role: Environment and binary package manager
  - Scope: Python + other software (C++, R, etc.)
  - Strength: Manages complex, non-Python dependencies like LAMMPS; creates isolated, reproducible environments
  - Weakness: Can be slower at resolving pure Python dependencies

- uv (the specialist)
  - Role: Python package manager
  - Scope: Python packages from PyPI
  - Strength: Blazing-fast installation and dependency resolution for Python packages
  - Weakness: Cannot install non-Python software like the core LAMMPS application

This hybrid approach yields a robust, stable, and efficient development environment for simulation work.

## Troubleshooting

- Module not found (e.g., `ModuleNotFoundError: No module named 'lammps'` or `ovito`)
  - Ensure the environment is active: `conda activate md-studio`
  - Confirm installation:
    - `conda list | grep lammps`
    - `uv pip list | grep ovito`
- Wrong Python used by `uv`
  - Make sure you run `uv` after activating `(md-studio)` so it installs into the correct environment.
- LAMMPS import issues
  - Reinstall with conda-forge channel: `conda install -c conda-forge lammps`
- OVITO license issues
  - Verify the `ovito.lic` file placement per OVITO’s documentation and restart Python after placing the license.

## Optional: Common next steps

- Install additional Python tools with `uv`:
```bash
uv pip install numpy scipy pandas matplotlib
```

- Export the environment (for reproducibility):
```bash
conda env export --name md-studio --no-builds > environment.yml
```

- Remove the environment (if needed):
```bash
conda remove -n md-studio --all
```
