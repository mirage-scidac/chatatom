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
import queue
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
    IterativeLammpsGeneration, SmartOrchestrator,
    IterativeOVITOGeneration,
    configure_dspy_lm,
    detect_precomputed_properties,
)
from prompts import ORCHESTRATOR_SYSTEM_PROMPT
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
        dump_filename_pattern = data.get('dump_filename_pattern', '')
        target_timesteps_from_frontend = data.get('target_timesteps', None)
        camera_angle = data.get('camera_angle', 'default')

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
                # Use target timesteps from frontend (computed from LLM total_steps)
                target_timesteps = target_timesteps_from_frontend
                render_config = {
                    'ovito_script': ovito_script,  # OVITO script string or None (already generated)
                    'target_timesteps': target_timesteps,
                    'dump_filename_pattern': dump_filename_pattern,
                    'camera_angle': camera_angle,
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

@app.route('/api/llm-stream-live', methods=['POST'])
def llm_stream_live():
    """Enhanced streaming endpoint with multi-turn conversation history"""
    body = request.get_json(force=True, silent=True) or {}
    user_command = body.get('command')
    history = body.get('history', [])
    execution_intent = body.get('execution_intent', False)
    ovito_script_param = body.get('ovito_script', None)
    pending_script_param = body.get('pending_script', None)

    if not user_command:
        return jsonify({"error": "Command not provided"}), 400

    def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting agent workflow...'})}\n\n"

            # Configure DSPy for intent classification
            configure_dspy_lm()
            smart_orchestrator = SmartOrchestrator()

            # Determine workflow stage from explicit fields
            workflow_stage = "initial"
            if pending_script_param:
                workflow_stage = "script_ready"
            elif history:
                workflow_stage = "parameters_proposed"

            intent_result = smart_orchestrator.classify_intent(user_command, workflow_stage)
            print(f"[INTENT] '{user_command}' -> {intent_result['intent']} (confidence: {intent_result['confidence']})")

            # If script is ready and user wants to execute, skip re-generation
            if workflow_stage == "script_ready" and intent_result["intent"] == "execute":
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': 'Starting live streaming simulation...', 'message_type': 'info'})}\n\n"
                yield f"data: {json.dumps({'type': 'start_streaming_simulation', 'script': pending_script_param, 'ovito_script': ovito_script_param, 'user_command': user_command})}\n\n"
                yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
                yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                return

            # Extract element from user command and build registry context
            registry_context = kim_utils.build_registry_context_for_command(user_command)

            # Build multi-turn messages array
            messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]

            # Add trimmed conversation history (only previous turns)
            trimmed_history = history[-(config.MAX_CONVERSATION_TURNS * 2):]
            for msg in trimmed_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Add current user message with registry context appended
            current_message = user_command
            if registry_context:
                current_message += f"\n\n{registry_context}"
            messages.append({"role": "user", "content": current_message})

            # Stream LLM response (unified path for initial and follow-up turns)
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Orchestrator'})}\n\n"

            response_chunks = []
            try:
                completion = client.chat.completions.create(
                    model=config.ORCHESTRATOR_MODEL or config.DEFAULT_MODEL,
                    messages=messages,
                    temperature=config.ORCHESTRATOR_TEMPERATURE,
                    stream=True,
                )
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        text = chunk.choices[0].delta.content
                        response_chunks.append(text)
                        yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Orchestrator', 'content': text, 'message_type': 'analysis'})}\n\n"
            except Exception as e:
                # Fallback to non-streaming
                completion = client.chat.completions.create(
                    model=config.ORCHESTRATOR_MODEL or config.DEFAULT_MODEL,
                    messages=messages,
                    temperature=config.ORCHESTRATOR_TEMPERATURE,
                )
                content = completion.choices[0].message.content
                response_chunks.append(content)
                yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Orchestrator', 'content': content, 'message_type': 'analysis'})}\n\n"

            analysis = ''.join(response_chunks)
            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Orchestrator'})}\n\n"

            # --- Structured parameter extraction via DSPy ---
            yield f"data: {json.dumps({'type': 'status', 'message': 'Extracting parameters...'})}\n\n"
            params = smart_orchestrator.extract_parameters(analysis)

            if not params["has_parameters"]:
                event_type = 'needs_more_info' if history else 'awaiting_confirmation'
                yield f"data: {json.dumps({'type': event_type, 'analysis': analysis})}\n\n"
                yield f"data: {json.dumps({'type': 'workflow_complete'})}\n\n"
                return

            material_for_search = params["material"]
            material_for_gen = params["material"] or "unknown"
            kim_model = params["kim_model"]
            temperature = params["temperature"]
            ensemble = params["ensemble"]
            duration_ps = params["duration_ps"]
            system_size = params["system_size"]
            viz_notes = params["visualization_notes"]
            camera_angle = params.get("camera_angle", "default")

            print(f"[PARAMS] Extracted: material={material_for_gen}, kim={kim_model}, "
                  f"T={temperature}K, ensemble={ensemble}, duration={duration_ps}ps, "
                  f"size={system_size}, camera_angle={camera_angle}")

            # Retrieve relevant lessons from memory before generation
            lammps_memory_context = ""
            viz_memory_context = ""
            try:
                memory = get_memory_backend()

                material_query = f"{material_for_search} simulation" if material_for_search else user_command

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

            # Generate + review using DSPy (structured, iterative)
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Generator'})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating and reviewing script...'})}\n\n"

            iterative_generator = IterativeLammpsGeneration(max_attempts=2)

            # Use queue+thread to stream progress events during synchronous DSPy generation
            progress_queue = queue.Queue()

            def _progress_callback(stage, attempt, max_attempts, **kwargs):
                progress_queue.put((stage, attempt, max_attempts, kwargs))

            result_holder = [None]
            gen_error_holder = [None]

            def _run_generation():
                try:
                    result_holder[0] = iterative_generator.forward(
                        material=material_for_gen,
                        kim_model=kim_model,
                        temperature=temperature,
                        ensemble=ensemble,
                        duration_ps=duration_ps,
                        system_size=system_size,
                        simulation_details=analysis,
                        visualization_notes=viz_notes,
                        progress_callback=_progress_callback
                    )
                except Exception as e:
                    gen_error_holder[0] = e
                finally:
                    progress_queue.put(("done", 0, 0, {}))

            gen_thread = threading.Thread(target=_run_generation)
            gen_thread.start()

            # Yield progress events from the queue while generation runs
            while True:
                try:
                    stage, attempt, max_attempts, kwargs = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue

                if stage == "done":
                    break
                elif stage == "generating":
                    msg = f"Generating script (attempt {attempt}/{max_attempts})..."
                    yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"
                elif stage == "script_generated":
                    # Emit intermediate script as a Generator trace
                    script_text = kwargs.get('script', '')
                    explanation = kwargs.get('explanation', '')
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Generator'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Generator', 'content': script_text, 'message_type': 'script_trace', 'attempt': attempt})}\n\n"
                    if explanation:
                        yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Generator', 'content': explanation, 'message_type': 'explanation', 'attempt': attempt})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Generator'})}\n\n"
                elif stage == "reviewing":
                    msg = f"Reviewing script (attempt {attempt}/{max_attempts})..."
                    yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"
                elif stage == "review_complete":
                    # Emit full review details as Reviewer trace
                    is_valid = kwargs.get('is_valid', True)
                    issues = kwargs.get('issues', [])
                    suggestions = kwargs.get('suggestions', [])
                    severity = kwargs.get('severity', 'none')
                    verdict = 'PASS' if is_valid else 'FAIL'
                    review_msg = f"**REVIEW: {verdict}** (attempt {attempt}/{max_attempts}, severity: {severity})\n"
                    if issues:
                        review_msg += "\n**Issues:**\n" + "\n".join(f"- {i}" for i in issues)
                    if suggestions:
                        review_msg += "\n\n**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions)
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Reviewer'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Reviewer', 'content': review_msg, 'message_type': 'review', 'attempt': attempt})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Reviewer'})}\n\n"

                    if not is_valid and attempt < max_attempts:
                        issue_summary = '; '.join(str(i) for i in issues[:2]) if issues else 'unknown issues'
                        msg = f"Review found issues, regenerating... ({issue_summary})"
                        yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"

            gen_thread.join()

            if gen_error_holder[0] is not None:
                raise gen_error_holder[0]

            result = result_holder[0]
            lammps_script = result.script

            # Only run visualization agent if review passed
            if result.review_passed:
                # Visualization agent — use DSPy modules
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Visualization'})}\n\n"

                simulation_context = {}
                if material_for_search:
                    simulation_context['material'] = material_for_search
                if viz_memory_context:
                    simulation_context['viz_lessons'] = viz_memory_context

                viz_request = viz_notes or user_command

                # Detect pre-computed properties from the LAMMPS script
                if lammps_script:
                    precomputed = detect_precomputed_properties(lammps_script)
                    if precomputed:
                        simulation_context['precomputed_properties'] = precomputed
                        print(f"[VIZ] Detected pre-computed properties: {list(precomputed.keys())}")

                # Detect custom visualization intent via DSPy
                ovito_script = None
                has_custom_viz = smart_orchestrator.detect_visualization(viz_request)

                if has_custom_viz:
                    # Iterative OVITO generate+review via thread+queue (mirrors LAMMPS pattern)
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Generating and reviewing OVITO script...'})}\n\n"

                    ovito_gen = IterativeOVITOGeneration(max_attempts=config.OVITO_SCRIPT_MAX_RETRIES)

                    ovito_progress_queue = queue.Queue()

                    def _ovito_progress(stage, attempt, max_attempts, **kwargs):
                        ovito_progress_queue.put((stage, attempt, max_attempts, kwargs))

                    ovito_result_holder = [None]
                    ovito_error_holder = [None]

                    def _run_ovito_generation():
                        try:
                            ovito_result_holder[0] = ovito_gen.forward(
                                visualization_request=viz_request,
                                simulation_context=simulation_context,
                                progress_callback=_ovito_progress,
                            )
                        except Exception as e:
                            ovito_error_holder[0] = e
                        finally:
                            ovito_progress_queue.put(("done", 0, 0, {}))

                    ovito_thread = threading.Thread(target=_run_ovito_generation)
                    ovito_thread.start()

                    # Yield progress events from the queue while OVITO generation runs
                    while True:
                        try:
                            stage, attempt, max_attempts, kwargs = ovito_progress_queue.get(timeout=0.5)
                        except queue.Empty:
                            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                            continue

                        if stage == "done":
                            break
                        elif stage == "generating":
                            msg = f"Generating OVITO script (attempt {attempt}/{max_attempts})..."
                            yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"
                        elif stage == "script_generated":
                            script_text = kwargs.get('script', '')
                            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Visualization'})}\n\n"
                            viz_msg = f"**Custom Visualization Requested**\n\n**Generated OVITO Script (attempt {attempt}):**\n```python\n{script_text}\n```"
                            yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Visualization', 'content': viz_msg, 'message_type': 'visualization', 'attempt': attempt})}\n\n"
                            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Visualization'})}\n\n"
                        elif stage == "reviewing":
                            msg = f"Reviewing OVITO script (attempt {attempt}/{max_attempts})..."
                            yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"
                        elif stage == "review_complete":
                            is_valid = kwargs.get('is_valid', True)
                            issues = kwargs.get('issues', [])
                            suggestions = kwargs.get('suggestions', [])
                            severity = kwargs.get('severity', 'none')
                            verdict = 'PASS' if is_valid else 'FAIL'
                            review_msg = f"**OVITO Review: {verdict}** (attempt {attempt}/{max_attempts}, severity: {severity})\n"
                            if issues:
                                review_msg += "\n**Issues:**\n" + "\n".join(f"- {i}" for i in issues)
                            if suggestions:
                                review_msg += "\n\n**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions)
                            yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Reviewer'})}\n\n"
                            yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Reviewer', 'content': review_msg, 'message_type': 'ovito_review', 'attempt': attempt})}\n\n"
                            yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Reviewer'})}\n\n"

                            if not is_valid and severity == "error" and attempt < max_attempts:
                                issue_summary = '; '.join(str(i) for i in issues[:2]) if issues else 'unknown issues'
                                msg = f"OVITO review found issues, regenerating... ({issue_summary})"
                                yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"

                    ovito_thread.join()

                    if ovito_error_holder[0] is not None:
                        print(f"[OVITO GEN] Error: {ovito_error_holder[0]}")
                        # Non-fatal: fall back to default visualization

                    ovito_gen_result = ovito_result_holder[0]
                    if ovito_gen_result and ovito_gen_result.review_passed:
                        ovito_script = ovito_gen_result.script
                        print(f"[VIZ] OVITO script passed review after {ovito_gen_result.attempts} attempt(s)")
                    else:
                        ovito_script = None
                        yield f"data: {json.dumps({'type': 'status', 'message': 'OVITO script failed review after all attempts, falling back to default visualization'})}\n\n"
                        print("[VIZ] OVITO script failed all review attempts, falling back to default")
                else:
                    print("[VIZ] No custom visualization requested, using default")
                    viz_message = "Applying default visualization settings for trajectory rendering."
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'Visualization', 'content': viz_message, 'message_type': 'visualization'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Visualization'})}\n\n"

                # Compute target_timesteps from LLM-provided total_steps
                dump_filename_pattern = getattr(result, 'dump_filename_pattern', '') or ''
                llm_total_steps = getattr(result, 'total_steps', 0) or 0
                dump_freq_val = getattr(result, 'dump_freq_val', 1000) or 1000
                target_timesteps = None
                if llm_total_steps > 0:
                    n_frames = min(config.MAX_TRAJECTORY_FRAMES, max(1, llm_total_steps // dump_freq_val))
                    if n_frames > 1:
                        step_size = llm_total_steps / (n_frames - 1)
                        raw = [int(round(i * step_size / dump_freq_val) * dump_freq_val) for i in range(n_frames)]
                        target_timesteps = sorted(set(raw))
                    elif n_frames == 1:
                        target_timesteps = [0]
                    print(f"[APP] Computed target_timesteps from LLM total_steps={llm_total_steps}: {len(target_timesteps or [])} frames")

                # Emit scripts_ready_for_review checkpoint instead of auto-executing
                yield f"data: {json.dumps({'type': 'scripts_ready_for_review', 'lammps_script': lammps_script, 'ovito_script': ovito_script, 'dump_filename_pattern': dump_filename_pattern, 'target_timesteps': target_timesteps, 'camera_angle': camera_angle})}\n\n"
            else:
                # Review failed - no OVITO, emit structured failure
                yield f"data: {json.dumps({'type': 'review_failed', 'script': lammps_script, 'issues': result.issues, 'suggestions': result.suggestions})}\n\n"

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

