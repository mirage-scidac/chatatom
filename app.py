
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
import json


import config
from openai import OpenAI

app = Flask(__name__, static_folder='.', static_url_path='')

# Create the global OpenAI client
client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

from agents import OrchestratorAgent, LammpsGeneratorAgent, LammpsReviewerAgent
from tools import run_lammps_simulation

@app.route('/api/llm-stream', methods=['GET'])
def llm_stream():
    user_command = request.args.get('command')
    conversation_context = request.args.get('context', '')  # For phase 2 conversations

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
                
                orchestrator_prompt = f"Analyze this simulation request and propose parameters (Phase 1): {user_command}"
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
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    info_message = "Please review the proposed parameters and confirm or request changes."
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': info_message, 'message_type': 'info'})}\n\n"
                    yield f"data: {json.dumps({'type': 'awaiting_confirmation', 'analysis': analysis})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'System'})}\n\n"
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

            # Check if review passed
            if "REVIEW: PASS" in review:
                # Check if user wants to run the simulation
                # Check both the current command and original command for execution intent
                user_command_lower = user_command.lower()
                should_run = any(keyword in user_command_lower for keyword in ['run', 'execute', 'simulate', 'perform', 'go ahead', 'proceed', 'start'])
                
                # If we're in phase 2 (conversation_context exists), also check for confirmation words
                if conversation_context and not should_run:
                    should_run = any(keyword in user_command_lower for keyword in ['yes', 'ok', 'confirm', 'go', 'ahead'])
                
                if should_run:
                    # 4. Execute the simulation
                    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'System'})}\n\n"
                    yield f"data: {json.dumps({'type': 'agent_content', 'agent': 'System', 'content': 'Running LAMMPS simulation...', 'message_type': 'info'})}\n\n"
                    
                    try:
                        output_files, thermo_data = run_lammps_simulation(lammps_script)
                        
                        success_message = f"LAMMPS simulation completed successfully! Output files: {list(output_files.keys())}. Captured {len(thermo_data)} thermodynamic data points."
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
        
        # 1. Orchestrator analyzes the request
        orchestrator_prompt = f"Analyze this simulation request: {user_command}"
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
                    output_files, thermo_data = run_lammps_simulation(lammps_script)
                    
                    conversation.append({
                        "agent": "System",
                        "content": f"LAMMPS simulation completed successfully! Output files: {list(output_files.keys())}. Captured {len(thermo_data)} thermodynamic data points.",
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

def run_health_checks():
    """Main function to run health checks."""
    print("--- Starting Application Health Checks ---")
    check_dependencies()
    check_llm_api()
    print("--- All Health Checks Passed ---")

if __name__ == "__main__":
    run_health_checks()
    app.run(host='0.0.0.0', port=5001, debug=True)
