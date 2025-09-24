Implementation Plan: LLM Molecular Dynamics Studio (Agentic MVP Approach)
This document outlines the step-by-step tasks for a junior software engineer to build the "LLM Molecular Dynamics Studio." Each MVP (Minimum Viable Product) is a self-contained, functional piece of the application.

1. Project Vision
To create a web-based application that allows users to run molecular dynamics simulations with LAMMPS and visualize the results with OVITO using natural language commands, orchestrated by a system of collaborating AI agents.

2. Architecture Overview
We will implement a monolithic backend that serves a single-page web front-end and runs an agentic workflow for task execution.

Front-End: A single HTML file using Tailwind CSS and JavaScript for the UI.

Back-End: A single Python server (Flask/FastAPI) that:

Serves the static front-end files.

Exposes a single API endpoint (/api/llm-command) to the front-end.

Orchestrates a team of LLM agents (Generator, Reviewer) that communicate with any standard OpenAI-compatible API endpoint (e.g., LM Studio, OpenAI, or the provided Argo Bridge).

Provides agents with tools to interact with the filesystem and execute simulations.

Reads its configuration (e.g., LLM endpoint URL, API key) from a local file or environment variables.

3. Startup Health Checks
Before the web server starts, the application will perform a series of critical health checks. If any check fails, the application will exit with a clear, user-friendly error message explaining the problem and how to fix it. This ensures the environment is valid before the user interacts with the application.

Dependency Verification: Checks that the lammps and ovito Python libraries are correctly installed and importable in the current Conda environment.

LLM API Connectivity: Makes a small, real API call to the configured LLM endpoint to verify the URL is correct, the server is running, and the API key is valid.

4. Testing Strategy
We will adopt a multi-layered testing approach focused on real-world integration to ensure the application is robust. We will avoid using mocks and instead test the actual functionality of our components.

Tool-Level Integration Tests: These will test individual functions in tools.py by having them perform their real action. For example, the LAMMPS tool will run a real (but small) simulation.

End-to-End Workflow Tests: These will test the full agent workflow by making small, inexpensive calls to the configured LLM endpoint to verify the entire communication pipeline and logic flow are correct.

Framework: We will use the pytest framework for all backend testing. Install it with uv pip install pytest.

5. Phased MVP Implementation Checklist
MVP 1: The "LAMMPS Scripting Co-pilot" ✅ COMPLETE
Goal: Deliver a usable chat application that can intelligently generate and review LAMMPS input scripts that use modern, standard-compliant potentials (KIM).

Tasks:

[x] Task 1.0: Implement Startup Health Checks

[x] In app.py, create a function check_dependencies(). This function will use subprocess to run a small Python script that attempts to import lammps and ovito. If it fails, print a clear error message and exit.

[x] In app.py, create a function check_llm_api(). This function will use the OpenAI client to make a simple, non-streaming chat completion call (e.g., "respond with only the word 'ok'"). If it fails, catch the exception, print a detailed error message (e.g., "Could not connect to LLM endpoint at {API_BASE_URL}. Is the server running and is the API key valid?"), and exit.

[x] At the bottom of app.py, before app.run(), call both check_dependencies() and check_llm_api().

[x] Task 1.1: LLM Endpoint Configuration

[x] Create config.py.

[x] Define API_BASE_URL and API_KEY.

[x] Add config.py to .gitignore.

[x] Task 1.2: Unified Server Setup

[x] Create app.py with Flask.

[x] Set up a route to serve index.html.

[x] Set up the /api/llm-command endpoint.

[x] Task 1.3: Basic Chat UI

[x] Build the UI in index.html with Tailwind CSS.

[x] Write JavaScript to handle sending user input to the backend via fetch().

[x] Task 1.4: Agent LLM Client

[x] Install openai library.

[x] Create the global OpenAI client in app.py.

[x] Task 1.5: Orchestrator Agent

[x] Create agents.py with a base Agent class.

[x] Create the OrchestratorAgent class.

[x] Task 1.6: LAMMPS Generation Agent

[x] Create LammpsGeneratorAgent in agents.py.

[x] Write its system prompt. Crucially, instruct the agent that it MUST use the kim command to automatically fetch interatomic potentials and should not use external potential files.

[x] Task 1.7: LAMMPS Reviewer Agent

[x] Create LammpsReviewerAgent in agents.py.

[x] Write its system prompt. Add a rule for the reviewer to check that the kim command is used for potentials.

[x] Task 1.8: Wire the Agent Workflow

[x] Implement the core logic in the /api/llm-command endpoint to coordinate the agent calls.

[x] Task 1.9: MVP 1 Testing

[x] Create a tests/ directory.

[x] Write a simple end-to-end test for the agent workflow. The test prompt should ask for a simulation using a specific material (e.g., Silicon). The test should make a real, small LLM API call and assert that the generated script contains the kim command as expected. This verifies the agent logic and communication pipeline is working correctly.

**Additional Completed Features:**
[x] Real-time streaming agent responses via Server-Sent Events (SSE)
[x] Enhanced conversational flow with two-phase interaction (proposal → confirmation)
[x] Automatic execution intent detection from natural language
[x] Comprehensive error handling and user feedback
[x] KIM-only potential enforcement with automatic model installation

MVP 2: The "LAMMPS Execution & Live Monitoring" ✅ COMPLETE
Goal: Execute LAMMPS simulations and display live thermodynamic data with real-time plotting capabilities.

[x] Task 2.1: Execution Tool with In-Memory Data Extraction

[x] In tools.py, implement run_lammps_simulation(script_string).

[x] This tool will use the lammps Python library.

[x] After the simulation runs, use the lammps instance's methods (e.g., callbacks or get_thermo()) to extract thermodynamic data directly into a Python data structure (e.g., a list of dictionaries or a pandas DataFrame). Do not parse the log.lammps file.

[x] The function should return both the file paths to output files (like dump.atom) and the in-memory thermodynamic data.

[x] Task 2.2: UI Thermodynamic Panels

[x] In index.html, add a plot-container div for displaying thermodynamic plots.

[x] Task 2.3: Data Plotting Integration

[x] Add Chart.js to index.html.

[x] In JavaScript, implement drawPlot(data). This function will receive the JSON-formatted thermodynamic data extracted directly from the LAMMPS run.

[x] Task 2.4: Update Orchestrator for Simulation Execution

[x] Teach the OrchestratorAgent to use the new simulation tool and to correctly handle the direct data returned by run_lammps_simulation.

[x] Task 2.5: MVP 2 Testing

[x] Create a tests/test_assets/ directory. Inside, create a very simple, fast-running LAMMPS script (in.test_lj) that uses a KIM potential.

[x] Write a live integration test for tools.run_lammps_simulation. The test should read the in.test_lj script, pass it to the tool, and assert that:

[x] The expected output file (dump.atom) is successfully created.

[x] The function returns a valid data structure (e.g., a list of dictionaries) containing the thermo data.

**Additional Completed Features:**
[x] WebSocket integration for real-time simulation data streaming
[x] Parallel LAMMPS execution with MPI support and automatic fallback
[x] Automatic KIM model detection, installation, and validation
[x] Real-time thermodynamic data parsing and streaming
[x] Enhanced LAMMPS execution with both serial and parallel modes
[x] Comprehensive error handling for simulation failures
[x] Live streaming simulation with run_lammps_simulation_streaming()

MVP 3: The "3D Trajectory Visualization"
Goal: Add 3D molecular visualization capabilities using OVITO integration for rendering atomic trajectories.

[ ] Task 3.1: UI 3D Viewer Panel

[ ] In index.html, add a viewer-container div for displaying 3D molecular visualizations.

[ ] Task 3.2: 3D Viewer Integration

[ ] Add three.js to index.html.

[ ] In JavaScript, implement loadViewer(model_url) to display 3D molecular structures.

[ ] Task 3.3: Trajectory Rendering Tool

[ ] In tools.py, implement render_trajectory(dump_file_path) using ovito.

[ ] This function should convert LAMMPS dump files to web-compatible 3D formats (e.g., .gltf).

[ ] Task 3.4: Update Orchestrator for 3D Visualization

[ ] Teach the OrchestratorAgent to use the trajectory rendering tool when 3D visualization is requested.

[ ] Task 3.5: MVP 3 Testing

[ ] Write an integration test for tools.render_trajectory. This test should use the dump.atom file from the MVP 2 tests, run it through the real ovito library, and assert that a valid .gltf file is created.

MVP 4: The "OVITO Automation Studio"
Goal: Extend the agentic system to also automate post-processing and advanced visualizations using OVITO.

[ ] Task 4.1: OVITO Generation Agent

[ ] In agents.py, create the OvitoGeneratorAgent class and its system prompt.

[ ] Task 4.2: OVITO Reviewer Agent

[ ] In agents.py, create the OvitoReviewerAgent class and its system prompt.

[ ] Task 4.3: OVITO Execution Tool

[ ] In tools.py, implement run_ovito_script(...).

[ ] Task 4.4: Expand Orchestrator Intelligence

[ ] Update the OrchestratorAgent to handle complex, multi-stage LAMMPS + OVITO workflows.

[ ] Task 4.5: MVP 4 Testing

[ ] Write a simple end-to-end test for the new OVITO agent workflow (Generator -> Reviewer), making a real LLM call to generate a trivial OVITO script (e.g., "just load the file").

[ ] In tests/test_assets/, create a simple OVITO script (test_ovito_script.py).

[ ] Write a live integration test for tools.run_ovito_script that uses this script and the dump file from the MVP 2 tests. Assert that it produces the expected output file (e.g., a .gltf or image).

6. Post-MVP Enhancements
Goal: Mature the application from a functional prototype to a robust, optimized, and intelligent tool.

[ ] Task 6.1: Migrate to DSPy for Prompt Optimization

[ ] Task 6.2: Advanced Memory

[ ] Task 6.3: Dynamic Tool Use

[ ] Task 6.4: Package as an Electron Desktop App

7. Technology Stack
Front-End: HTML, Tailwind CSS, JavaScript

JavaScript Libraries: Chart.js, three.js

Back-End: Python (Flask or FastAPI)

Core Libraries: lammps, ovito, pandas

LLM Communication: Standard openai Python client library

Testing: pytest

Package Management: Conda, uv
