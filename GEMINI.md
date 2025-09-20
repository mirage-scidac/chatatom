# Gemini Code Assistant Context

This document provides context for the "LLM Molecular Dynamics Studio" project.

## Project Overview

The project is a web-based application that allows users to run molecular dynamics simulations with LAMMPS and visualize the results with OVITO using natural language commands. It uses a system of collaborating AI agents to interpret user requests and execute the necessary steps.

The application consists of a Python backend (using Flask) that serves a single-page web front-end. The backend orchestrates a team of LLM agents that communicate with an OpenAI-compatible API endpoint.

A key component is the `argo_bridge.py` script, which acts as a proxy between the OpenAI API and a custom "Argo" API, allowing the use of various large language models.

## Building and Running

### Environment Setup

The project requires a Conda environment with specific packages. The `README.md` file provides detailed instructions for setting up the environment using Conda and `uv`.

1.  **Create and activate the Conda environment:**
    ```bash
    mamba create -n chatmd python=3.11
    mamba activate chatmd
    ```

2.  **Install LAMMPS:**
    ```bash
    conda install -c conda-forge lammps
    ```

3.  **Install OVITO:**
    ```bash
    uv pip install ovito
    ```

### Running the Application

The main application is in `app.py` (which is not yet created, but is planned). To run the application:

```bash
python app.py
```

The `argo_bridge.py` can be run as a standalone server:

```bash
python Argo_docs/argo_bridge.py --username <your_username>
```

## Development Conventions

*   **Testing:** The project uses `pytest` for testing. The testing strategy focuses on integration tests that interact with the actual tools and services, avoiding mocks. Tests are located in the `tests/` directory.
*   **Configuration:** The application will read its configuration (e.g., LLM endpoint URL, API key) from a `config.py` file, which should be added to `.gitignore`.
*   **Coding Style:** The existing code in `argo_bridge.py` and the test files can be used as a reference for coding style.

## Key Files

*   `README.md`: Provides detailed instructions for setting up the development environment.
*   `project_plan/plan_checklist.md`: Outlines the project vision, architecture, and a phased implementation plan. This is the most important document for understanding the project's goals and structure.
*   `Argo_docs/argo_bridge.py`: A Flask application that acts as a bridge between the OpenAI API and the "Argo" API. This is the core of the LLM integration.
*   `Argo_docs/argoapi_*_test.py`: Example scripts demonstrating how to use the Argo API directly.
*   `tests/install_test.py`: A test file for verifying the installation.
