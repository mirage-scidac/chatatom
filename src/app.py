import subprocess
import sys
import os

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
            max_completion_tokens=5,
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

import sqlite3
import config
from openai import OpenAI

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'chatmd_secret_key'  # Required for SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=120,        # wait 120s before declaring connection dead
    ping_interval=25,        # send ping every 25s to keep connection alive
)

# Create the global OpenAI client
client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


@app.route('/')
def serve_index():
    return send_from_directory('..', 'index.html')

from agents import (
    SmartOrchestrator,
    configure_dspy_lm,
)
from memory import get_memory_backend
from tools import run_lammps_simulation
import kim_utils
import uuid

# Configure DSPy once at startup in the main thread.
# DSPy 3.x locks settings to the thread that first calls dspy.configure(),
# so this must happen here — never inside request handlers or background threads.
configure_dspy_lm()
from datetime import datetime
from graph.graph import simulation_graph
from graph.state import SimulationState
from graph.nodes import SESSION_QUEUES


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
        
        # Use socketio's background task so it's managed within the async context
        socketio.start_background_task(run_simulation)

    except Exception as e:
        emit('error', {'message': f'Error starting simulation: {str(e)}'})


@socketio.on('replay_trajectory')
def handle_replay_trajectory(data):
    """Render dump files from a past run and stream trajectory_frame events."""
    run_folder = data.get('run_folder', '')
    if not run_folder or not os.path.isdir(run_folder):
        emit('simulation_error', {'message': f'Run folder not found: {run_folder}'})
        return

    def run_replay():
        from tools import (find_dump_file, extract_timestep_from_filename,
                           _start_render_server_process, _send_render_request)

        dump_info = find_dump_file(run_folder)
        if not dump_info or not dump_info['files']:
            socketio.emit('simulation_error', {'message': 'No dump files found in run folder'})
            return

        pattern = dump_info.get('pattern', '')
        files = sorted(dump_info['files'],
                       key=lambda f: extract_timestep_from_filename(f, pattern))

        # Subsample to at most 50 frames
        if len(files) > 50:
            step = max(1, len(files) // 50)
            files = files[::step][:50]

        # Load saved OVITO script if present
        ovito_script_path = os.path.join(run_folder, 'ovito_script.py')
        ovito_script = None
        if os.path.exists(ovito_script_path):
            with open(ovito_script_path) as _f:
                ovito_script = _f.read()
        print(f"[replay_trajectory] rendering {len(files)} frames from {run_folder} "
              f"({'custom viz' if ovito_script else 'default viz'})")

        proc = _start_render_server_process()
        try:
            for i, f in enumerate(files):
                timestep = extract_timestep_from_filename(f, pattern)
                result = _send_render_request(proc, {
                    'dump_file': f,
                    'ovito_script': ovito_script,
                    'file_frame_index': 0,
                    'camera_angle': 'default',
                    'render_size': [600, 450],
                })
                if result.get('status') == 'ok':
                    socketio.emit('trajectory_frame', {
                        'image_data': result['image'],
                        'frame_number': i,
                        'timestep': timestep,
                    })
                else:
                    print(f"[replay_trajectory] frame {i} error: {result.get('error')}")

            socketio.emit('trajectory_complete', {
                'total_frames': len(files),
                'message': f'Loaded {len(files)} frames from past run',
            })
        finally:
            try:
                _send_render_request(proc, {'type': 'shutdown'})
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass

    socketio.start_background_task(run_replay)


@app.route('/api/llm-stream-live', methods=['POST'])
def llm_stream_live():
    """Streaming endpoint — delegates pipeline to LangGraph simulation_graph."""
    body = request.get_json(force=True, silent=True) or {}
    user_command = body.get('command')
    history = body.get('history', [])
    ovito_script_param = body.get('ovito_script', None)
    pending_script_param = body.get('pending_script', None)
    execution_target = body.get('execution_target', 'local')

    if not user_command:
        return jsonify({"error": "Command not provided"}), 400

    # Determine workflow stage
    workflow_stage = "initial"
    if body.get('workflow_stage') == 'pbs_pending':
        workflow_stage = "pbs_pending"
    elif pending_script_param:
        workflow_stage = "script_ready"
    elif history:
        workflow_stage = "parameters_proposed"

    # session_id ties this request to a persisted LangGraph checkpoint
    session_id = body.get("session_id") or str(uuid.uuid4())
    sse_q = queue.Queue()
    SESSION_QUEUES[session_id] = sse_q

    # Ensure session row exists (upsert — no-op if already present)
    now = datetime.utcnow().isoformat()
    _sess_conn = _get_sessions_db()
    _sess_conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, 'New Session', now, now)
    )
    _sess_conn.commit()
    _sess_conn.close()

    # Only pass fields that come from this request.
    # Generation outputs (material, lammps_script, temperature, etc.) are intentionally
    # omitted so that LangGraph restores them from the SqliteSaver checkpoint rather
    # than overwriting them with None on every follow-up turn.
    initial_state: SimulationState = {
        "user_message":       user_command,
        "workflow_stage":     workflow_stage,
        "pending_script":     pending_script_param,
        "ovito_script_param": ovito_script_param,
        "execution_target":   execution_target,
        "session_id":         session_id,
        # Per-turn transient fields — always reset each request
        "intent":             None,
        "analysis":           None,
        "has_parameters":     None,
        "review_passed":      None,
        "review_issues":      None,
        "review_suggestions": None,
        "has_custom_viz":     None,
        "error":              None,
        **({"conversation_history": history} if history else {}),
    }

    graph_config = {"configurable": {"thread_id": session_id}}

    def run_graph():
        try:
            # Snapshot hpc_jobs count before invoke so we can detect a fresh submit this turn.
            # Defensive: fresh sessions have no checkpoint yet; old checkpoints may lack hpc_jobs.
            prior_hpc_count = 0
            try:
                prior_state = simulation_graph.get_state(graph_config)
                if prior_state and prior_state.values:
                    prior_hpc_count = len(prior_state.values.get("hpc_jobs") or [])
            except Exception as _e:
                print(f"[POST-GRAPH] get_state failed (likely fresh session): {_e}")

            final = simulation_graph.invoke(initial_state, config=graph_config)

            # Auto-title session from first simulation parameters
            if final.get("material"):
                title_parts = [final["material"]]
                if final.get("temperature"):
                    title_parts.append(f"{final['temperature']}K")
                new_title = " ".join(str(p) for p in title_parts)
                _t_conn = _get_sessions_db()
                _t_conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ? AND title = 'New Session'",
                    (new_title, datetime.utcnow().isoformat(), session_id)
                )
                _t_conn.commit()
                _t_conn.close()

            # HPC submission completed this turn — agent already streamed the success
            # message with task ID + PBS job ID. Just close the stream cleanly and
            # skip the script-related cascade below (which would otherwise misfire
            # as review_failed from stale checkpoint state).
            new_hpc_count = len(final.get("hpc_jobs") or [])
            if new_hpc_count > prior_hpc_count:
                sse_q.put({"type": "workflow_complete", "session_id": session_id})
                return

            # Direct execution path: script already approved, start simulation
            if final["workflow_stage"] == "script_ready" and final["intent"] == "execute":
                sse_q.put({"type": "agent_start", "agent": "System"})
                sse_q.put({"type": "agent_content", "agent": "System",
                           "content": "Starting live streaming simulation...", "message_type": "info"})
                sse_q.put({"type": "start_streaming_simulation",
                           "script": final["pending_script"],
                           "ovito_script": final["ovito_script_param"],
                           "user_command": user_command})
                sse_q.put({"type": "agent_complete", "agent": "System"})
                sse_q.put({"type": "workflow_complete", "session_id": session_id})
                return

            # No simulation parameters extracted — ask for clarification
            # Skip this branch for analyze/question/cancel intents — they handle their own output
            if (not final.get("has_parameters") and final.get("analysis")
                    and final.get("intent") not in ("analyze", "question", "cancel")):
                event_type = "needs_more_info" if history else "awaiting_confirmation"
                sse_q.put({"type": event_type, "analysis": final["analysis"]})
                sse_q.put({"type": "workflow_complete", "session_id": session_id})
                return

            # LAMMPS review failed
            if final.get("lammps_script") and not final.get("review_passed"):
                sse_q.put({"type": "review_failed",
                           "script": final["lammps_script"],
                           "issues": final.get("review_issues") or [],
                           "suggestions": final.get("review_suggestions") or []})
                sse_q.put({"type": "workflow_complete", "session_id": session_id})
                return

            # Default viz message when no custom viz
            if final.get("lammps_script") and final.get("review_passed") and not final.get("has_custom_viz"):
                sse_q.put({"type": "agent_content", "agent": "Visualization",
                           "content": "Applying default visualization settings for trajectory rendering.",
                           "message_type": "visualization"})
                sse_q.put({"type": "agent_complete", "agent": "Visualization"})

            # Compute target_timesteps from the actual LAMMPS script text.
            # DSPy's total_steps and dump_freq_val are unreliable for multi-stage
            # scripts (e.g. equil + heat + hold with separate run commands and
            # separate dump commands per stage). Parse the script directly.
            # Let the dump monitor read actual frames from dump files directly.
            # Pre-computing target_timesteps from the script is unreliable for
            # multi-phase simulations with separate dump commands per phase.
            target_timesteps = None

            # Scripts ready for user approval
            if final.get("lammps_script") and final.get("review_passed"):
                sse_q.put({
                    "type":                 "scripts_ready_for_review",
                    "lammps_script":        final["lammps_script"],
                    "ovito_script":         final.get("ovito_script"),
                    "dump_filename_pattern": final.get("dump_filename_pattern", ""),
                    "target_timesteps":     target_timesteps,
                    "camera_angle":         final.get("camera_angle", "default"),
                    "execution_target":     final.get("execution_target", "local"),
                })

            sse_q.put({"type": "workflow_complete", "session_id": session_id})

        except Exception as e:
            import traceback
            traceback.print_exc()

            def _flatten(exc, depth=0):
                """Recursively unpack ExceptionGroup / __cause__ / __context__ for a useful message."""
                if depth > 5 or exc is None:
                    return []
                msgs = []
                inner = getattr(exc, "exceptions", None)  # ExceptionGroup
                if inner:
                    for sub in inner:
                        msgs.extend(_flatten(sub, depth + 1))
                else:
                    text = f"{type(exc).__name__}: {exc}".strip()
                    msgs.append(text or type(exc).__name__)
                # Also surface __cause__ if no inner — these often carry the real reason
                if not inner and getattr(exc, "__cause__", None):
                    msgs.extend(_flatten(exc.__cause__, depth + 1))
                return msgs

            parts = _flatten(e) or [repr(e)]
            msg = " | ".join(parts)
            sse_q.put({"type": "error", "message": f"An error occurred: {msg}"})
        finally:
            sse_q.put(None)                        # sentinel — SSE generator stops
            SESSION_QUEUES.pop(session_id, None)   # clean up queue reference

    threading.Thread(target=run_graph, daemon=True).start()

    def generate():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting agent workflow...'})}\n\n"
        while True:
            item = sse_q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

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
# Sessions API Endpoints
# ============================================================================

def _get_sessions_db():
    db_path = os.path.abspath(config.SESSION_DB_PATH)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sessions_table():
    conn = _get_sessions_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT DEFAULT 'New Session',
            created_at  TEXT,
            updated_at  TEXT,
            messages    TEXT DEFAULT '[]'
        )
    """)
    # Add messages column if table already exists without it
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN messages TEXT DEFAULT '[]'")
    except Exception:
        pass
    conn.commit()
    conn.close()


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    conn = _get_sessions_db()
    rows = conn.execute(
        "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/sessions', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = _get_sessions_db()
    conn.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, 'New Session', now, now)
    )
    conn.commit()
    conn.close()
    return jsonify({"session_id": session_id, "title": "New Session", "created_at": now})


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    conn = _get_sessions_db()
    row = conn.execute(
        "SELECT session_id, title, created_at, updated_at, messages FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "session not found"}), 404
    return jsonify(dict(row))


@app.route('/api/sessions/<session_id>', methods=['PATCH'])
def update_session(session_id):
    data = request.get_json(force=True, silent=True) or {}
    now = datetime.utcnow().isoformat()
    conn = _get_sessions_db()
    if 'title' in data:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (data['title'].strip(), now, session_id)
        )
    if 'messages' in data:
        conn.execute(
            "UPDATE sessions SET messages = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(data['messages']), now, session_id)
        )
    conn.commit()
    conn.close()
    return jsonify({"session_id": session_id})


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    conn = _get_sessions_db()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": session_id})


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
    _init_sessions_table()
    initialize_kim_registry()
    socketio.run(
        app,
        host='0.0.0.0',
        port=config.APP_PORT,
        debug=True,
        use_reloader=False,      # reloader restarts app when OVITO spawns subprocess → kills WebSocket
        allow_unsafe_werkzeug=True,
    )
