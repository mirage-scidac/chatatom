import subprocess
import sys
import os
import re

# This will be populated with the Flask app later.
# For now, it contains the health checks.

def check_dependencies():
    """Checks if lammps and ovito are installed."""
    print("Running dependency check...")
    try:
        # We check for the conda environment variable to give a more specific error message.
        if "CONDA_PREFIX" not in os.environ:
            print("WARNING: Not running in a Conda environment. Dependency check might not be accurate.", file=sys.stderr)

        # Check for lammps
        subprocess.run(
            [sys.executable, "-c", "import lammps"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Dependency 'lammps' is installed.")

        # Check for ovito
        subprocess.run(
            [sys.executable, "-c", "import ovito"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Dependency 'ovito' is installed.")

    except subprocess.CalledProcessError as e:
        print("❌ Dependency check failed.", file=sys.stderr)
        if "No module named 'lammps'" in e.stderr:
            print("Error: Could not import 'lammps'.", file=sys.stderr)
        elif "No module named 'ovito'" in e.stderr:
            print("Error: Could not import 'ovito'.", file=sys.stderr)
        else:
            print("An unknown error occurred during dependency check.", file=sys.stderr)
        
        print("Please ensure you are in the correct Conda environment and that both libraries are installed.", file=sys.stderr)
        print(f"Python executable: {sys.executable}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: Python executable not found at '{sys.executable}'.", file=sys.stderr)
        sys.exit(1)

def check_llm_api():
    """Checks if the LLM API is reachable and the key is valid."""
    print("Running LLM API connection check...")
    try:
        import config
        from openai import OpenAI

        client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
        completion = client.chat.completions.create(
            model="gpt4o",  # This can be any name for local models
            messages=[{"role": "user", "content": "respond with only the word 'ok'"}],
            max_tokens=5,
            timeout=5, # 5 second timeout
        )
        response = completion.choices[0].message.content.strip().lower()
        if 'ok' in response:
            print("✅ LLM API connection successful.")
        else:
            print(f"⚠️ LLM API responded, but with an unexpected message: {response}", file=sys.stderr)

    except ImportError:
        print("❌ LLM API check failed.", file=sys.stderr)
        print("Error: 'openai' or 'config.py' not found.", file=sys.stderr)
        print("Please install the openai library (`uv pip install openai`) and ensure config.py exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("❌ LLM API connection failed.", file=sys.stderr)
        print(f"Error: Could not connect to LLM endpoint at {config.API_BASE_URL}.", file=sys.stderr)
        print("Please ensure the LLM server is running and the API_BASE_URL and API_KEY in config.py are correct.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

import subprocess
import sys
import os

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_socketio import SocketIO, emit
import json
import threading
import time

import config
from openai import OpenAI

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'chatmd_secret_key'  # Required for SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Create the global OpenAI client
client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


@app.route('/')
def serve_index():
    return send_from_directory('..', 'index.html')

from agents import (
    OrchestratorAgent, LammpsGeneratorAgent, LammpsReviewerAgent,
    IterativeLammpsGeneration, PlanVerifier, SmartOrchestrator,
    configure_dspy_lm
)
from memory import get_memory_backend
from tools import run_lammps_simulation
import kim_utils
import uuid
from datetime import datetime
from session_manager import get_session_manager

# =============================================================================
# Session Storage for Checkpoint Workflow
# =============================================================================

# SQLite-based session persistence (replaces in-memory storage)
# Sessions now survive server restarts


def create_session(user_request: str) -> str:
    """Create a new workflow session using SQLite backend."""
    session_id = str(uuid.uuid4())
    session_manager = get_session_manager()
    session_manager.create(session_id, user_request)
    return session_id


def get_session(session_id: str) -> dict:
    """Get a workflow session by ID from SQLite."""
    session_manager = get_session_manager()
    return session_manager.get(session_id)


def update_session(session_id: str, updates: dict) -> bool:
    """Update a workflow session in SQLite."""
    session_manager = get_session_manager()
    return session_manager.update(session_id, updates)


# =============================================================================
# Checkpoint Workflow API Endpoints
# =============================================================================

@app.route('/api/workflow/start', methods=['POST'])
def workflow_start():
    """
    Start a new agentic workflow with verification.

    This endpoint:
    1. Analyzes the user request
    2. Verifies the simulation plan is feasible
    3. Returns a checkpoint for user approval

    Request body:
        {"request": "simulate copper at 300K for 10ps"}

    Response:
        {
            "session_id": "uuid",
            "stage": "parameters_pending",
            "analysis": {...},
            "verification": {...},
            "checkpoint": {
                "type": "parameters",
                "data": {...},
                "confidence": 0.85
            }
        }
    """
    try:
        data = request.get_json()
        user_request = data.get('request')

        if not user_request:
            return jsonify({"error": "Request not provided"}), 400

        # Configure DSPy
        configure_dspy_lm()

        # Create session
        session_id = create_session(user_request)
        session = get_session(session_id)

        # Build registry context
        registry_context = kim_utils.build_registry_context_for_command(user_request)

        # Analyze request using SmartOrchestrator
        orchestrator = SmartOrchestrator()
        analysis = orchestrator.analyze_request(user_request, registry_context)

        # Check if clarification is needed
        if analysis.needs_clarification:
            update_session(session_id, {
                "stage": "clarification_needed",
                "analysis": {
                    "material": analysis.material,
                    "temperature": analysis.temperature,
                    "ensemble": analysis.ensemble,
                    "duration_ps": analysis.duration_ps,
                    "kim_model": analysis.kim_model,
                    "system_size": analysis.system_size,
                    "confidence": analysis.confidence
                }
            })

            return jsonify({
                "session_id": session_id,
                "stage": "clarification_needed",
                "question": analysis.clarification_question,
                "partial_analysis": {
                    "material": analysis.material,
                    "temperature": analysis.temperature,
                    "confidence": analysis.confidence
                }
            })

        # Verify the plan
        verifier = PlanVerifier()
        verification = verifier.forward(
            material=analysis.material,
            kim_model=analysis.kim_model,
            temperature=analysis.temperature,
            ensemble=analysis.ensemble,
            duration_ps=analysis.duration_ps,
            system_size=analysis.system_size
        )

        # Update session
        update_session(session_id, {
            "stage": "parameters_pending",
            "analysis": {
                "material": analysis.material,
                "temperature": analysis.temperature,
                "ensemble": analysis.ensemble,
                "duration_ps": analysis.duration_ps,
                "kim_model": analysis.kim_model,
                "system_size": analysis.system_size,
                "confidence": analysis.confidence
            },
            "verification": {
                "is_feasible": verification.is_feasible,
                "issues": verification.issues,
                "suggestions": verification.suggestions,
                "confidence": verification.confidence,
                "kim_model_available": verification.kim_model_available
            }
        })

        return jsonify({
            "session_id": session_id,
            "stage": "parameters_pending",
            "analysis": {
                "material": analysis.material,
                "temperature": analysis.temperature,
                "ensemble": analysis.ensemble,
                "duration_ps": analysis.duration_ps,
                "kim_model": analysis.kim_model,
                "system_size": analysis.system_size,
                "confidence": analysis.confidence
            },
            "verification": {
                "is_feasible": verification.is_feasible,
                "issues": verification.issues,
                "suggestions": verification.suggestions,
                "confidence": verification.confidence,
                "kim_model_available": verification.kim_model_available
            },
            "checkpoint": {
                "type": "parameters",
                "message": "Please review and approve these simulation parameters",
                "requires_approval": True
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/checkpoint/parameters', methods=['POST'])
def checkpoint_parameters():
    """
    Handle parameter approval checkpoint.

    Request body:
        {
            "session_id": "uuid",
            "approved": true,
            "modifications": {}  # Optional parameter modifications
        }

    Response:
        {
            "session_id": "uuid",
            "stage": "script_pending",
            "script": "...",
            "review": {...},
            "checkpoint": {...}
        }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        approved = data.get('approved', False)
        modifications = data.get('modifications', {})

        if not session_id:
            return jsonify({"error": "session_id required"}), 400

        session = get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        if session['stage'] != 'parameters_pending':
            return jsonify({"error": f"Invalid stage: {session['stage']}"}), 400

        if not approved:
            return jsonify({
                "session_id": session_id,
                "stage": "cancelled",
                "message": "Workflow cancelled by user"
            })

        # Apply any modifications
        params = session['analysis'].copy()
        params.update(modifications)

        # Configure DSPy
        configure_dspy_lm()

        # Generate script using IterativeLammpsGeneration
        generator = IterativeLammpsGeneration(max_attempts=3)
        result = generator.forward(
            material=params['material'],
            kim_model=params['kim_model'],
            temperature=params['temperature'],
            ensemble=params['ensemble'],
            duration_ps=params['duration_ps'],
            system_size=params.get('system_size', 5)
        )

        # Update session
        update_session(session_id, {
            "stage": "script_pending",
            "parameters": params,
            "script": result.script,
            "review": {
                "passed": result.review_passed,
                "attempts": result.attempts,
                "issues": result.issues,
                "suggestions": result.suggestions,
                "confidence": result.confidence
            },
            "conversation": result.conversation
        })

        return jsonify({
            "session_id": session_id,
            "stage": "script_pending",
            "script": result.script,
            "review": {
                "passed": result.review_passed,
                "attempts": result.attempts,
                "issues": result.issues,
                "suggestions": result.suggestions,
                "confidence": result.confidence
            },
            "checkpoint": {
                "type": "script",
                "message": "Please review and approve this LAMMPS script",
                "requires_approval": True
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/checkpoint/script', methods=['POST'])
def checkpoint_script():
    """
    Handle script approval checkpoint.

    Request body:
        {
            "session_id": "uuid",
            "approved": true,
            "run_simulation": true  # Whether to execute the simulation
        }

    Response:
        {
            "session_id": "uuid",
            "stage": "ready" | "executing",
            "message": "..."
        }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        approved = data.get('approved', False)
        run_simulation = data.get('run_simulation', False)

        if not session_id:
            return jsonify({"error": "session_id required"}), 400

        session = get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        if session['stage'] != 'script_pending':
            return jsonify({"error": f"Invalid stage: {session['stage']}"}), 400

        if not approved:
            return jsonify({
                "session_id": session_id,
                "stage": "cancelled",
                "message": "Workflow cancelled by user"
            })

        if run_simulation:
            # Update stage to executing
            update_session(session_id, {"stage": "executing"})

            return jsonify({
                "session_id": session_id,
                "stage": "executing",
                "script": session['script'],
                "message": "Script approved. Use WebSocket to start streaming simulation.",
                "websocket_event": "start_streaming_simulation"
            })
        else:
            # Just mark as ready
            update_session(session_id, {"stage": "ready"})

            return jsonify({
                "session_id": session_id,
                "stage": "ready",
                "script": session['script'],
                "message": "Script approved and ready for execution"
            })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/workflow/session/<session_id>', methods=['GET'])
def get_workflow_session(session_id):
    """Get the current state of a workflow session."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session)


@app.route('/api/llm-stream', methods=['GET'])
def llm_stream():
    user_command = request.args.get('command')
    conversation_context = request.args.get('context', '')  # For phase 2 conversations
    execution_intent = request.args.get('execution_intent', 'false').lower() == 'true'  # Track execution intent

    if not user_command:
        return jsonify({"error": "Command not provided"}), 400

    def generate():
        try:
            # Instantiate agents
            orchestrator = OrchestratorAgent()
            generator = LammpsGeneratorAgent()
            reviewer = LammpsReviewerAgent()

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting agent workflow...'})}\n\n"

            # Detect execution intent from the original request if this is Phase 1
            original_execution_intent = False
            if not conversation_context:
                # Phase 1: Check if the original request implies execution
                user_command_lower = user_command.lower()
                original_execution_intent = any(keyword in user_command_lower for keyword in [
                    'simulate', 'run', 'execute', 'perform', 'start simulation', 'do simulation'
                ])

            # Extract element from user command and build registry context
            registry_context = kim_utils.build_registry_context_for_command(user_command)

            # Determine if this is Phase 1 (initial request) or Phase 2 (user response)
            if conversation_context:
                # Phase 2: User is responding to orchestrator's proposal
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Orchestrator'})}\n\n"
                
                orchestrator_prompt = f"""Previous analysis and proposal:
{conversation_context}

User response: {user_command}

Process the user's feedback and provide final simulation parameters using the SIMULATION PARAMETERS format."""
                
                finalization_chunks = []
                finalization_generator = orchestrator.execute_stream(orchestrator_prompt)
                for chunk in finalization_generator:
                    finalization_chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Orchestrator', 'content': chunk, 'message_type': 'finalization'})}\n\n"
                
                finalization = ''.join(finalization_chunks)
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Orchestrator'})}\n\n"
                
                # Check if we now have final parameters
                if "SIMULATION PARAMETERS:" not in finalization:
                    # Still need more information
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    info_message = "Please provide more specific details to proceed."
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': info_message, 'message_type': 'info'})}\n\n"
                    yield f"data: {json.dumps({'type': 'needs_more_info', 'analysis': finalization})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
                    yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                    return
                
                analysis = finalization  # Use finalized parameters for script generation
            else:
                # Phase 1: Initial analysis and proposal
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Orchestrator'})}\n\n"
                
                orchestrator_prompt = f"Analyze this simulation request and propose parameters (Phase 1): {user_command}{registry_context}"
                analysis_chunks = []
                analysis_generator = orchestrator.execute_stream(orchestrator_prompt)
                for chunk in analysis_generator:
                    analysis_chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Orchestrator', 'content': chunk, 'message_type': 'analysis'})}\n\n"
                
                analysis = ''.join(analysis_chunks)
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Orchestrator'})}\n\n"

                # Check if orchestrator needs more information or is proposing parameters
                if "SIMULATION PARAMETERS:" not in analysis:
                    # Phase 1: Orchestrator is asking for confirmation or more info
                    # Let the conversation flow naturally without system message
                    yield f"data: {json.dumps({'type': 'awaiting_confirmation', 'analysis': analysis})}\n\n"
                    yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                    return

            # 2. Generator creates the LAMMPS script using structured parameters
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Generator'})}\n\n"
            
            generator_prompt = f"Generate LAMMPS script with these parameters:\n{analysis}"
            lammps_script_chunks = []
            script_generator = generator.execute_stream(generator_prompt)
            for chunk in script_generator:
                lammps_script_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Generator', 'content': chunk, 'message_type': 'script'})}\n\n"
            
            lammps_script = ''.join(lammps_script_chunks)
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Generator'})}\n\n"

            # 3. Reviewer checks the script
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Reviewer'})}\n\n"
            
            reviewer_prompt = f"Review this LAMMPS script for correctness:\n\n{lammps_script}"
            review_generator = reviewer.execute_stream(reviewer_prompt)
            review_chunks = []
            for chunk in review_generator:
                review_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Reviewer', 'content': chunk, 'message_type': 'review'})}\n\n"
            
            review = ''.join(review_chunks)
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Reviewer'})}\n\n"

            # 4. Visualization agent - ALWAYS runs to show visualization is happening
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Visualization'})}\n\n"

            # Parse visualization request from orchestrator analysis
            # The analysis contains VISUALIZATION PREFERENCES if the user requested custom viz
            from tools import parse_visualization_request

            # Extract material from analysis for context
            simulation_context = {}
            if 'Material:' in analysis:
                # Simple extraction - look for material line
                for line in analysis.split('\n'):
                    if 'Material:' in line:
                        material = line.split('Material:')[1].strip().split()[0]
                        simulation_context['material'] = material
                        break

            # Use the analysis (not user_command) to detect visualization preferences
            # The analysis contains structured VISUALIZATION PREFERENCES section
            viz_request = analysis if 'VISUALIZATION' in analysis.upper() else user_command
            ovito_script = parse_visualization_request(viz_request, simulation_context, lammps_script)
            
            print(f"[APP] Visualization script: {'Custom (' + str(len(ovito_script)) + ' chars)' if ovito_script else 'Default'}")
            
            # Generate appropriate message based on whether custom visualization was requested
            if ovito_script:
                # Custom visualization requested - show the script
                viz_message = f"**Custom Visualization Requested**\n\n**Generated OVITO Script:**\n```python\n{ovito_script}\n```"
            else:
                # No custom visualization - show minimal message
                viz_message = "Applying default visualization settings for trajectory rendering."
            
            # Stream the visualization message
            yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Visualization', 'content': viz_message, 'message_type': 'visualization'})}\n\n"
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Visualization'})}\n\n"

            # Check if review passed
            if "REVIEW: PASS" in review:
                # Enhanced execution logic with intent tracking
                user_command_lower = user_command.lower()
                should_run = False
                
                # Check for explicit execution keywords in current command
                should_run = any(keyword in user_command_lower for keyword in ['run', 'execute', 'simulate', 'perform', 'go ahead', 'proceed', 'start'])
                
                # If we're in phase 2 (conversation_context exists), check for confirmation words
                if conversation_context and not should_run:
                    should_run = any(keyword in user_command_lower for keyword in ['yes', 'ok', 'confirm', 'go', 'ahead'])
                
                # NEW: If we're in phase 2 and have execution intent from original request, auto-execute
                if conversation_context and execution_intent and not should_run:
                    # Check if this looks like a parameter modification (not a rejection)
                    is_parameter_change = any(phrase in user_command_lower for phrase in [
                        'make it', 'change', 'set', 'use', 'atoms', 'temperature', 'pressure', 'time', 'duration'
                    ])
                    
                    if is_parameter_change:
                        should_run = True
                        # Auto-execute silently without system message
                
                # NEW: If we're in phase 1 and detected original execution intent, auto-execute
                if not conversation_context and original_execution_intent and not should_run:
                    should_run = True
                    # Auto-execute silently without system message
                
                if should_run:
                    # 4. Execute the simulation
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': 'Running LAMMPS simulation...', 'message_type': 'info'})}\n\n"
                    
                    try:
                        output_files, thermo_data, frame_images = run_lammps_simulation(lammps_script)
                        
                        success_message = f"LAMMPS simulation completed successfully! Output files: {list(output_files.keys())}. Captured {len(thermo_data)} thermodynamic data points and {len(frame_images)} visualization frames."
                        yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': success_message, 'message_type': 'success'})}\n\n"
                        yield f"data: {json.dumps({'type': 'simulation_complete', 'thermo_data': thermo_data, 'output_files': output_files, 'script': lammps_script})}\n\n"
                            
                    except Exception as e:
                        error_message = f"Simulation execution failed: {str(e)}"
                        yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': error_message, 'message_type': 'error'})}\n\n"
                        yield f"data: {json.dumps({'type': 'simulation_failed', 'script': lammps_script})}\n\n"
                    
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    info_message = "LAMMPS script generated and approved. Add 'run' or 'execute' to your command to run the simulation."
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': info_message, 'message_type': 'info'})}\n\n"
                    yield f"data: {json.dumps({'type': 'script_ready', 'script': lammps_script})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                error_message = "Script review failed. Please check the issues identified by the reviewer."
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': error_message, 'message_type': 'error'})}\n\n"
                yield f"data: {json.dumps({'type': 'review_failed', 'script': lammps_script})}\n\n"
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'An error occurred: {str(e)}'})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@app.route('/api/llm-command', methods=['POST'])
def llm_command():
    data = request.get_json()
    user_command = data.get('command')

    if not user_command:
        return jsonify({"error": "Command not provided"}), 400

    # Instantiate agents
    orchestrator = OrchestratorAgent()
    generator = LammpsGeneratorAgent()
    reviewer = LammpsReviewerAgent()

    try:
        # Track the full conversation
        conversation = []
        
        # 1. Orchestrator analyzes the request with registry context
        registry_context = kim_utils.build_registry_context_for_command(user_command)

        orchestrator_prompt = f"Analyze this simulation request: {user_command}{registry_context}"
        analysis = orchestrator.execute(orchestrator_prompt)
        conversation.append({
            "agent": "Orchestrator",
            "content": analysis,
            "type": "analysis"
        })

        # Check if orchestrator needs more information
        if "SIMULATION PARAMETERS:" not in analysis:
            # Orchestrator is asking for more information
            conversation.append({
                "agent": "System",
                "content": "The orchestrator needs more information to proceed. Please provide the requested details.",
                "type": "info"
            })
            
            return jsonify({
                "conversation": conversation,
                "status": "needs_more_info",
                "analysis": analysis
            })

        # 2. Generator creates the LAMMPS script using structured parameters
        generator_prompt = f"Generate LAMMPS script with these parameters:\n{analysis}"
        lammps_script = generator.execute(generator_prompt)
        conversation.append({
            "agent": "Generator",
            "content": lammps_script,
            "type": "script"
        })

        # 3. Reviewer checks the script
        reviewer_prompt = f"Review this LAMMPS script for correctness:\n\n{lammps_script}"
        review = reviewer.execute(reviewer_prompt)
        conversation.append({
            "agent": "Reviewer",
            "content": review,
            "type": "review"
        })

        # Check if review passed (fix: look for "PASS" instead of "APPROVED")
        if "REVIEW: PASS" in review:
            # Check if user wants to run the simulation
            user_command_lower = user_command.lower()
            should_run = any(keyword in user_command_lower for keyword in ['run', 'execute', 'simulate', 'perform'])
            
            if should_run:
                # 4. Execute the simulation
                try:
                    output_files, thermo_data, frame_images = run_lammps_simulation(lammps_script)
                    
                    conversation.append({
                        "agent": "System",
                        "content": f"LAMMPS simulation completed successfully! Output files: {list(output_files.keys())}. Captured {len(thermo_data)} thermodynamic data points and {len(frame_images)} visualization frames.",
                        "type": "success"
                    })
                    
                    return jsonify({
                        "conversation": conversation,
                        "thermo_data": thermo_data,
                        "output_files": output_files,
                        "script": lammps_script,
                        "status": "simulation_complete"
                    })
                    
                except Exception as e:
                    conversation.append({
                        "agent": "System",
                        "content": f"Simulation execution failed: {str(e)}",
                        "type": "error"
                    })
                    
                    return jsonify({
                        "conversation": conversation,
                        "script": lammps_script,
                        "status": "simulation_failed"
                    })
            else:
                conversation.append({
                    "agent": "System",
                    "content": "LAMMPS script generated and approved. Add 'run' or 'execute' to your command to run the simulation.",
                    "type": "info"
                })
                
                return jsonify({
                    "conversation": conversation,
                    "script": lammps_script,
                    "status": "script_ready"
                })
        else:
            conversation.append({
                "agent": "System",
                "content": "Script review failed. Please check the issues identified by the reviewer.",
                "type": "error"
            })
            
            return jsonify({
                "conversation": conversation,
                "script": lammps_script,
                "status": "review_failed"
            })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to ChatMD server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_streaming_simulation')
def handle_streaming_simulation(data):
    """Handle streaming simulation requests via WebSocket with visualization"""
    try:
        lammps_script = data.get('script')
        ovito_script = data.get('ovito_script')  # Get pre-generated OVITO script
        
        if not lammps_script:
            emit('error', {'message': 'No LAMMPS script provided'})
            return
        
        emit('simulation_started', {'message': 'Starting streaming simulation with visualization...'})
        
        def data_callback(data_point):
            """Callback function to emit both thermodynamic and visualization data"""
            if isinstance(data_point, dict):
                if data_point.get('type') == 'trajectory_frame':
                    # This is a visualization frame
                    socketio.emit('trajectory_frame', {
                        'image_data': data_point['image_data'],
                        'frame_number': data_point['frame_number'],
                        'timestep': data_point['timestep']
                    })
                elif data_point.get('type') == 'trajectory_complete':
                    # Visualization is complete
                    socketio.emit('trajectory_complete', {
                        'total_frames': data_point['total_frames'],
                        'message': data_point['message']
                    })
                else:
                    # This is a thermodynamic data point
                    socketio.emit('thermo_data_point', data_point)
        
        def run_simulation():
            try:
                import re
                
                # Extract frame sampling strategy from LAMMPS script comments
                target_timesteps = None
                for line in lammps_script.split('\n'):
                    if 'Target timesteps:' in line or 'target timesteps:' in line.lower():
                        # Extract the list of timesteps from the comment
                        # Format: # Target timesteps: [0, 13000, 26000, ...]
                        match = re.search(r'\[([\d,\s]+)\]', line)
                        if match:
                            timesteps_str = match.group(1)
                            target_timesteps = [int(x.strip()) for x in timesteps_str.split(',')]
                            print(f"[APP] Extracted target timesteps from script: {target_timesteps}")
                            break
                
                # Simple render config with pre-generated OVITO script
                render_config = {
                    'ovito_script': ovito_script,  # OVITO script string or None (already generated)
                    'target_timesteps': target_timesteps,
                }
                
                print(f"[APP] Render config: {'Custom script (' + str(len(ovito_script)) + ' chars)' if ovito_script else 'Default visualization'}")
                
                # Use configured number of cores from config.py
                output_files, all_thermo_data, frame_images = run_lammps_simulation(
                    lammps_script, 
                    render_config,
                    data_callback,
                    num_cores=config.LAMMPS_CORES
                )
                
                socketio.emit('simulation_complete', {
                    'output_files': output_files,
                    'total_points': len(all_thermo_data),
                    'total_frames': len(frame_images),
                    'message': f'Simulation completed successfully with {len(frame_images)} visualization frames!'
                })
                
            except Exception as e:
                socketio.emit('simulation_error', {
                    'message': f'Simulation failed: {str(e)}'
                })
        
        # Run simulation in a separate thread to avoid blocking
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()
        
    except Exception as e:
        emit('error', {'message': f'Error starting simulation: {str(e)}'})

@app.route('/api/llm-stream-live', methods=['GET'])
def llm_stream_live():
    """Enhanced streaming endpoint that uses WebSocket for simulation data"""
    user_command = request.args.get('command')
    conversation_context = request.args.get('context', '')
    execution_intent = request.args.get('execution_intent', 'false').lower() == 'true'

    if not user_command:
        return jsonify({"error": "Command not provided"}), 400

    def generate():
        try:
            # Instantiate agents
            orchestrator = OrchestratorAgent()
            generator = LammpsGeneratorAgent()
            reviewer = LammpsReviewerAgent()

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting agent workflow...'})}\n\n"

            # Configure DSPy for intent classification
            configure_dspy_lm()
            smart_orchestrator = SmartOrchestrator()

            # Use LLM to classify user intent
            workflow_stage = "initial"
            if conversation_context:
                if "SCRIPT READY:" in conversation_context:
                    workflow_stage = "script_ready"
                else:
                    workflow_stage = "parameters_proposed"

            intent_result = smart_orchestrator.classify_intent(user_command, workflow_stage)
            print(f"[INTENT] '{user_command}' -> {intent_result['intent']} (confidence: {intent_result['confidence']})")

            # If script is ready and user wants to execute, skip re-generation
            if workflow_stage == "script_ready" and intent_result["intent"] == "execute":
                # Extract the pending script from context
                pending_script = conversation_context.replace("SCRIPT READY:\n", "", 1)
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': 'Starting live streaming simulation...', 'message_type': 'info'})}\n\n"
                yield f"data: {json.dumps({'type': 'start_streaming_simulation', 'script': pending_script, 'ovito_script': None, 'user_command': user_command})}\n\n"
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
                yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                return

            # Extract element from user command and build registry context
            registry_context = kim_utils.build_registry_context_for_command(user_command)

            # Phase 1 or 2 logic (same as existing endpoint)
            if conversation_context:
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Orchestrator'})}\n\n"
                
                orchestrator_prompt = f"""Previous analysis and proposal:
{conversation_context}

User response: {user_command}

Process the user's feedback and provide final simulation parameters using the SIMULATION PARAMETERS format."""
                
                finalization_chunks = []
                finalization_generator = orchestrator.execute_stream(orchestrator_prompt)
                for chunk in finalization_generator:
                    finalization_chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Orchestrator', 'content': chunk, 'message_type': 'finalization'})}\n\n"
                
                finalization = ''.join(finalization_chunks)
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Orchestrator'})}\n\n"
                
                if "SIMULATION PARAMETERS:" not in finalization:
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    info_message = "Please provide more specific details to proceed."
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': info_message, 'message_type': 'info'})}\n\n"
                    yield f"data: {json.dumps({'type': 'needs_more_info', 'analysis': finalization})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
                    yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                    return
                
                analysis = finalization
            else:
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Orchestrator'})}\n\n"
                
                orchestrator_prompt = f"Analyze this simulation request and propose parameters (Phase 1): {user_command}{registry_context}"
                analysis_chunks = []
                analysis_generator = orchestrator.execute_stream(orchestrator_prompt)
                for chunk in analysis_generator:
                    analysis_chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Orchestrator', 'content': chunk, 'message_type': 'analysis'})}\n\n"
                
                analysis = ''.join(analysis_chunks)
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Orchestrator'})}\n\n"

                if "SIMULATION PARAMETERS:" not in analysis:
                    yield f"data: {json.dumps({'type': 'awaiting_confirmation', 'analysis': analysis})}\n\n"
                    yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                    return

            # Retrieve relevant lessons from memory before generation
            lammps_memory_context = ""
            viz_memory_context = ""
            try:
                memory = get_memory_backend()

                # Extract material from analysis for memory search
                material_for_search = ""
                if 'Material:' in analysis:
                    for line in analysis.split('\n'):
                        if 'Material:' in line:
                            material_for_search = line.split('Material:')[1].strip().split()[0]
                            break

                if material_for_search:
                    material_query = f"{material_for_search} simulation"
                else:
                    material_query = user_command

                lammps_lessons = memory.search(material_query, category="lammps_issue", limit=3)
                viz_lessons = memory.search(user_command, category="visualization_issue", limit=3)

                if lammps_lessons:
                    lammps_memory_context = "LESSONS FROM PAST SIMULATIONS:\n" + "\n".join(
                        f"- {m.content}" for m in lammps_lessons
                    )
                    print(f"[MEMORY] Found {len(lammps_lessons)} relevant LAMMPS lessons")

                if viz_lessons:
                    viz_memory_context = "VISUALIZATION LESSONS:\n" + "\n".join(
                        f"- {m.content}" for m in viz_lessons
                    )
                    print(f"[MEMORY] Found {len(viz_lessons)} relevant visualization lessons")
            except Exception as e:
                print(f"[MEMORY] Memory retrieval failed: {e}")

            # Generate script
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Generator'})}\n\n"

            generator_prompt = f"Generate LAMMPS script with these parameters:\n{analysis}"
            if lammps_memory_context:
                generator_prompt += f"\n\n{lammps_memory_context}"
            lammps_script_chunks = []
            script_generator = generator.execute_stream(generator_prompt)
            for chunk in script_generator:
                lammps_script_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Generator', 'content': chunk, 'message_type': 'script'})}\n\n"

            lammps_script = ''.join(lammps_script_chunks)
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Generator'})}\n\n"

            # Review script
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Reviewer'})}\n\n"

            reviewer_prompt = f"Review this LAMMPS script for correctness:\n\n{lammps_script}"
            review_generator = reviewer.execute_stream(reviewer_prompt)
            review_chunks = []
            for chunk in review_generator:
                review_chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Reviewer', 'content': chunk, 'message_type': 'review'})}\n\n"

            review = ''.join(review_chunks)
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Reviewer'})}\n\n"

            # 4. Visualization agent - ALWAYS runs to show visualization is happening
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Visualization'})}\n\n"

            # Parse visualization request from orchestrator analysis
            # The analysis contains VISUALIZATION PREFERENCES if the user requested custom viz
            from tools import parse_visualization_request

            # Extract material from analysis for context
            simulation_context = {}
            if 'Material:' in analysis:
                # Simple extraction - look for material line
                for line in analysis.split('\n'):
                    if 'Material:' in line:
                        material = line.split('Material:')[1].strip().split()[0]
                        simulation_context['material'] = material
                        break

            # Inject visualization lessons into simulation context
            if viz_memory_context:
                simulation_context['viz_lessons'] = viz_memory_context

            # Use the analysis (not user_command) to detect visualization preferences
            # The analysis contains structured VISUALIZATION PREFERENCES section
            viz_request = analysis if 'VISUALIZATION' in analysis.upper() else user_command
            ovito_script = parse_visualization_request(viz_request, simulation_context, lammps_script)
            
            # Generate appropriate message based on whether custom visualization was requested
            if ovito_script:
                # Custom visualization requested - show the script
                viz_message = f"**Custom Visualization Requested**\n\n**Generated OVITO Script:**\n```python\n{ovito_script}\n```"
            else:
                # No custom visualization - show minimal message
                viz_message = "Applying default visualization settings for trajectory rendering."
            
            # Stream the visualization message
            yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Visualization', 'content': viz_message, 'message_type': 'visualization'})}\n\n"
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Visualization'})}\n\n"

            # Check if review passed and determine execution using LLM intent
            if "REVIEW: PASS" in review:
                # Use the intent we already classified at the top
                should_run = intent_result["intent"] in ("execute", "new_simulation")

                if should_run:
                    # Signal to use streaming simulation with OVITO script
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': 'Starting live streaming simulation...', 'message_type': 'info'})}\n\n"
                    yield f"data: {json.dumps({'type': 'start_streaming_simulation', 'script': lammps_script, 'ovito_script': ovito_script, 'user_command': user_command})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    info_message = "LAMMPS script generated and approved. Would you like to run the simulation?"
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': info_message, 'message_type': 'info'})}\n\n"
                    yield f"data: {json.dumps({'type': 'script_ready', 'script': lammps_script, 'ovito_script': ovito_script})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                error_message = "Script review failed. Please check the issues identified by the reviewer."
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': error_message, 'message_type': 'error'})}\n\n"
                yield f"data: {json.dumps({'type': 'review_failed', 'script': lammps_script})}\n\n"
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"

            yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'An error occurred: {str(e)}'})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

# ============================================================================
# User Feedback API Endpoint
# ============================================================================

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback about a simulation result.

    The LLM extracts a structured lesson from the feedback and stores
    it in memory for future retrieval during similar simulations.

    Request body:
        {
            "feedback": "atom deletion didn't work, Other atoms still visible",
            "simulation_summary": "Cu melting 100ps, CNA coloring, filter crystalline",
            "session_id": ""  (optional)
        }

    Response:
        {
            "category": "visualization_issue",
            "lesson": "...",
            "keywords": "...",
            "stored": true
        }
    """
    try:
        data = request.get_json()
        feedback = data.get('feedback', '').strip()
        simulation_summary = data.get('simulation_summary', '').strip()

        if not feedback:
            return jsonify({"error": "feedback is required"}), 400

        configure_dspy_lm()
        orchestrator = SmartOrchestrator()
        result = orchestrator.process_feedback(feedback, simulation_summary)

        return jsonify({
            "category": result["category"],
            "lesson": result["lesson"],
            "keywords": result["keywords"],
            "stored": result["stored"]
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============================================================================
# KIM Potentials Management API Endpoints
# ============================================================================

@app.route('/api/kim-potentials/list', methods=['GET'])
def list_kim_potentials():
    """Get list of all potentials from the registry"""
    try:
        status = kim_utils.get_registry_status()
        
        # Format response for frontend
        response = {
            "success": True,
            "potentials": status.get('potentials', []),
            "stats": {
                "total": status.get('total_potentials', 0),
                "downloaded": status.get('downloaded_potentials', 0),
                "pending": status.get('not_downloaded', 0),
                "unique_elements": len(set(p['element'] for p in status.get('potentials', [])))
            }
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "potentials": [],
            "stats": {
                "total": 0,
                "downloaded": 0,
                "pending": 0,
                "unique_elements": 0
            }
        }), 500


@app.route('/api/kim-potentials/download', methods=['POST'])
def download_kim_potential():
    """Download a specific KIM potential"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({"error": "model_id is required"}), 400
        
        # Install the model
        success, message = kim_utils.install_and_verify_kim_model(model_id, timeout=180)
        
        if success:
            # Update registry
            kim_utils.update_registry(model_id, downloaded=True)
            return jsonify({
                "success": True,
                "message": message,
                "model_id": model_id
            })
        else:
            return jsonify({
                "success": False,
                "message": message,
                "model_id": model_id
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/kim-potentials/search', methods=['GET'])
def search_kim_potentials():
    """Search OpenKIM for potentials for a specific element"""
    try:
        element = request.args.get('element')
        
        if not element:
            return jsonify({"error": "element parameter is required"}), 400
        
        models = kim_utils.query_openkim_models(element, max_results=50)
        
        return jsonify({
            "element": element,
            "models": models,
            "count": len(models)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/kim-potentials/status', methods=['GET'])
def kim_potentials_status():
    """Get installation status of a specific potential"""
    try:
        model_id = request.args.get('model_id')
        
        if not model_id:
            return jsonify({"error": "model_id parameter is required"}), 400
        
        installed = kim_utils.is_model_installed(model_id)
        
        return jsonify({
            "model_id": model_id,
            "installed": installed
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/kim-potentials/init', methods=['POST'])
def initialize_kim_potentials():
    """Initialize recommended potentials (first-time setup)"""
    try:
        # Check if already initialized
        status = kim_utils.get_registry_status()
        
        if status['downloaded_potentials'] > 0:
            return jsonify({
                "message": "Potentials already initialized",
                "status": status
            })
        
        # Initialize in background thread and emit progress via WebSocket
        def init_with_progress():
            def progress_callback(model_id, status_type, message):
                socketio.emit('kim_init_progress', {
                    'model_id': model_id,
                    'status': status_type,
                    'message': message
                })
            
            results = kim_utils.initialize_recommended_potentials(progress_callback)
            
            socketio.emit('kim_init_complete', {
                'results': results,
                'success_count': sum(results.values()),
                'total_count': len(results)
            })
        
        init_thread = threading.Thread(target=init_with_progress)
        init_thread.daemon = True
        init_thread.start()
        
        return jsonify({
            "message": "Initialization started",
            "potentials": kim_utils.RECOMMENDED_POTENTIALS
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/kim-potentials/add', methods=['POST'])
def add_kim_potential():
    """Add a new potential to the registry"""
    try:
        data = request.get_json()
        element = data.get('element')
        model_id = data.get('model_id')
        model_type = data.get('type', 'Unknown')
        description = data.get('description', '')
        
        if not element or not model_id:
            return jsonify({"error": "element and model_id are required"}), 400
        
        success = kim_utils.add_potential_to_registry(element, model_id, model_type, description)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Added {model_id} to registry"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to add potential to registry"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# WebSocket event for KIM potential downloads
@socketio.on('download_kim_potential')
def handle_kim_download(data):
    """Handle KIM potential download via WebSocket with progress updates"""
    try:
        model_id = data.get('model_id')
        
        if not model_id:
            emit('kim_download_error', {'message': 'model_id is required'})
            return
        
        emit('kim_download_started', {'model_id': model_id})
        
        # Install the model
        success, message = kim_utils.install_and_verify_kim_model(model_id, timeout=180)
        
        if success:
            # Update registry
            kim_utils.update_registry(model_id, downloaded=True)
            emit('kim_download_complete', {
                'model_id': model_id,
                'success': True,
                'message': message
            })
        else:
            emit('kim_download_error', {
                'model_id': model_id,
                'message': message
            })
            
    except Exception as e:
        emit('kim_download_error', {
            'model_id': data.get('model_id', 'unknown'),
            'message': str(e)
        })


@socketio.on('initialize_recommended_potentials')
def handle_initialize_recommended(data=None):
    """Handle initialization of recommended potentials via WebSocket"""
    try:
        def progress_callback(model_id, status_type, message):
            """Callback to emit progress updates"""
            total = len(kim_utils.RECOMMENDED_POTENTIALS)
            current_index = kim_utils.RECOMMENDED_POTENTIALS.index(model_id) + 1
            progress = int((current_index / total) * 100)
            
            socketio.emit('kim_init_progress', {
                'status': message,
                'current_model': model_id,
                'progress': progress
            })
        
        # Run initialization
        results = kim_utils.initialize_recommended_potentials(progress_callback)
        success_count = sum(results.values())
        
        # Emit completion
        socketio.emit('kim_init_complete', {
            'results': results,
            'success_count': success_count,
            'total_count': len(results)
        })
        
    except Exception as e:
        socketio.emit('kim_download_error', {
            'message': f'Initialization failed: {str(e)}'
        })


def run_health_checks():
    """Main function to run health checks."""
    print("--- Starting Application Health Checks ---")
    check_dependencies()
    check_llm_api()
    print("--- All Health Checks Passed ---")


def initialize_kim_registry():
    """Initialize KIM potentials registry on startup"""
    print("Checking KIM potentials registry...")
    
    if not os.path.exists(kim_utils.REGISTRY_FILE):
        print(f"⚠️  Registry file not found at {kim_utils.REGISTRY_FILE}")
        print("Please ensure kim_potentials_registry.md exists in the project directory")
        return
    
    # Verify and sync registry with actual installed models
    print("Verifying registry against installed KIM models...")
    sync_result = kim_utils.verify_and_sync_registry()
    
    if sync_result['mismatches_found'] > 0:
        print(f"⚠️  Found {sync_result['mismatches_found']} registry mismatches")
        print(f"   Updated {sync_result['updates_made']} entries to match actual installation status")
        for mismatch in sync_result['mismatches']:
            print(f"   - {mismatch['model_id']}: registry={mismatch['registry_status']}, actual={mismatch['actual_status']}")
    else:
        print("✅ Registry is in sync with installed models")
    
    status = kim_utils.get_registry_status()
    print(f"✅ Registry loaded: {status['total_potentials']} potentials registered")
    print(f"   Downloaded: {status['downloaded_potentials']}")
    print(f"   Not downloaded: {status['not_downloaded']}")
    
    # Check if this is first run (no potentials downloaded)
    if status['downloaded_potentials'] == 0:
        print("\n🔄 First run detected - initializing recommended potentials...")
        print(f"   Will download: {', '.join([p.split('__')[0] for p in kim_utils.RECOMMENDED_POTENTIALS])}")
        print("   This may take a few minutes...")
        
        results = kim_utils.initialize_recommended_potentials()
        success_count = sum(results.values())
        
        if success_count > 0:
            print(f"✅ Initialized {success_count}/{len(results)} recommended potentials")
        else:
            print("⚠️  No potentials were successfully downloaded")
            print("   You can download them later from the KIM Potentials tab")


if __name__ == "__main__":
    run_health_checks()
    initialize_kim_registry()
    socketio.run(app, host='0.0.0.0', port=config.APP_PORT, debug=True)

