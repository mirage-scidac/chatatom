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

Provides agents with tools to interact with external data (NIST) and the filesystem.

Reads its configuration (e.g., LLM endpoint URL, API key) from a local file or environment variables.

3. Startup Health Checks
Before the web server starts, the application will perform a series of critical health checks. If any check fails, the application will exit with a clear, user-friendly error message explaining the problem and how to fix it. This ensures the environment is valid before the user interacts with the application.

Dependency Verification: Checks that the lammps and ovito Python libraries are correctly installed and importable in the current Conda environment.

LLM API Connectivity: Makes a small, real API call to the configured LLM endpoint to verify the URL is correct, the server is running, and the API key is valid.

4. Testing Strategy
We will adopt a multi-layered testing approach focused on real-world integration to ensure the application is robust. We will avoid using mocks and instead test the actual functionality of our components.

Tool-Level Integration Tests: These will test individual functions in tools.py by having them perform their real action. For example, the NIST tool will make a real web request, and the LAMMPS tool will run a real (but small) simulation.

End-to-End Workflow Tests: These will test the full agent workflow by making small, inexpensive calls to the configured LLM endpoint to verify the entire communication pipeline and logic flow are correct.

Framework: We will use the pytest framework for all backend testing. Install it with uv pip install pytest.

5. Phased MVP Implementation Checklist
MVP 1: The "LAMMPS Scripting Co-pilot"
Goal: Deliver a usable chat application that can intelligently generate and review LAMMPS input scripts, data files, and find correct potentials, without actually running the simulation.

Tasks:

[ ] Task 1.0: Implement Startup Health Checks

[ ] In app.py, create a function check_dependencies(). This function will use subprocess to run a small Python script that attempts to import lammps and ovito. If it fails, print a clear error message and exit.

[ ] In app.py, create a function check_llm_api(). This function will use the OpenAI client to make a simple, non-streaming chat completion call (e.g., "respond with only the word 'ok'"). If it fails, catch the exception, print a detailed error message (e.g., "Could not connect to LLM endpoint at {API_BASE_URL}. Is the server running and is the API key valid?"), and exit.

[ ] At the bottom of app.py, before app.run(), call both check_dependencies() and check_llm_api().

[ ] Task 1.1: LLM Endpoint Configuration

[ ] Create config.py.

[ ] Define API_BASE_URL and API_KEY.

[ ] Add config.py to .gitignore.

[ ] Task 1.2: Unified Server Setup

[ ] Create app.py with Flask.

[ ] Set up a route to serve index.html.

[ ] Set up the /api/llm-command endpoint.

[ ] Task 1.3: Basic Chat UI

[ ] Build the UI in index.html with Tailwind CSS.

[ ] Write JavaScript to handle sending user input to the backend via fetch().

[ ] Task 1.4: Agent LLM Client

[ ] Install openai library.

[ ] Create the global OpenAI client in app.py.

[ ] Task 1.5: Orchestrator Agent

[ ] Create agents.py with a base Agent class.

[ ] Create the OrchestratorAgent class.

[ ] Task 1.6: NIST Potential Tool

[ ] Create tools.py.

[ ] Install requests and beautifulsoup4.

[ ] Implement search_nist_potentials(elements).

[ ] Task 1.7: LAMMPS Generation Agent

[ ] Create LammpsGeneratorAgent in agents.py.

[ ] Write its system prompt.

[ ] Task 1.8: LAMMPS Reviewer Agent

[ ] Create LammpsReviewerAgent in agents.py.

[ ] Write its system prompt.

[ ] Task 1.9: Wire the Agent Workflow

[ ] Implement the core logic in the /api/llm-command endpoint to coordinate the agent calls.

[ ] Task 1.10: MVP 1 Testing

[ ] Create a tests/ directory.

[ ] Write a live integration test for tools.search_nist_potentials. The test should make a real web request to the NIST website for a common potential (e.g., ['Ni']) and assert that a valid URL string is returned.

[ ] Write a simple end-to-end test for the agent workflow. The test should make a real, small, and inexpensive LLM API call that requires the agents to generate and review a trivial one-line script. This verifies the agent logic and communication pipeline is working correctly.

MVP 2: The "Local Simulation Runner"
Goal: Build on MVP 1 by adding the ability to execute the validated LAMMPS script and visualize the results directly in the application.

[ ] Task 2.1: Execution Tool

[ ] In tools.py, implement run_lammps_simulation(script_string) using the lammps library.

[ ] Task 2.2: UI Visualization Panels

[ ] In index.html, add plot-container and viewer-container divs.

[ ] Task 2.3: Data Plotting Integration

[ ] Add Chart.js to index.html.

[ ] In tools.py, implement get_log_data(log_file_path) using pandas.

[ ] In JavaScript, implement drawPlot(data).

[ ] Task 2.4: 3D Viewer Integration

[ ] Add three.js to index.html.

[ ] In tools.py, implement render_trajectory(dump_file_path) using ovito.

[ ] In JavaScript, implement loadViewer(model_url).

[ ] Task 2.5: Update Orchestrator Logic

[ ] Teach the OrchestratorAgent to use the new simulation and visualization tools.

[ ] Task 2.6: MVP 2 Testing

[ ] Create a tests/test_assets/ directory. Inside, create a very simple, fast-running LAMMPS script (in.test_lj) for a few hundred atoms and a few dozen steps.

[ ] Write a live integration test for tools.run_lammps_simulation. The test should read the in.test_lj script, pass it to the tool, and assert that the expected output files (log.lammps, dump.atom) are successfully created.

[ ] Write an integration test for tools.get_log_data. This test should use the log.lammps file generated by the previous test to ensure it can be parsed correctly by pandas.

[ ] Write an integration test for tools.render_trajectory. This test should use the dump.atom file from the simulation test, run it through the real ovito library, and assert that a valid .gltf file is created.

MVP 3: The "OVITO Automation Studio"
Goal: Extend the agentic system to also automate post-processing and advanced visualizations using OVITO.

[ ] Task 3.1: OVITO Generation Agent

[ ] In agents.py, create the OvitoGeneratorAgent class and its system prompt.

[ ] Task 3.2: OVITO Reviewer Agent

[ ] In agents.py, create the OvitoReviewerAgent class and its system prompt.

[ ] Task 3.3: OVITO Execution Tool

[ ] In tools.py, implement run_ovito_script(...).

[ ] Task 3.4: Expand Orchestrator Intelligence

[ ] Update the OrchestratorAgent to handle complex, multi-stage LAMMPS + OVITO workflows.

[ ] Task 3.5: MVP 3 Testing

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
