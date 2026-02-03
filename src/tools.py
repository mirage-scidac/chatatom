"""
Tools for executing LAMMPS simulations with MPI-based parallel execution.

This module provides MPI-only execution with:
- Real-time thermodynamic data streaming
- Asynchronous frame rendering during simulation
- Organized run folder management
"""
import os
import subprocess
import re
import glob
from typing import Dict, List, Any, Tuple, Optional, Callable
import config
import threading
import time
import kim_utils
import run_manager
from visualization_rendering import render_frame_to_image_custom
from queue import Queue


def parse_visualization_request(user_command: str, simulation_context: Optional[Dict[str, Any]] = None, 
                               lammps_script: Optional[str] = None) -> Optional[str]:
    """
    Generate OVITO script if custom visualization is requested.
    
    Uses OrchestratorAgent as the single source of truth for understanding
    visualization intent.
    
    Args:
        user_command: User's natural language command (may include visualization preferences)
        simulation_context: Optional simulation context (material, structure, etc.)
        lammps_script: Optional LAMMPS script to check for pre-computed properties
        
    Returns:
        OVITO Python script as a string, or None for default visualization
    """
    from agents import OrchestratorAgent
    
    try:
        # Check if LAMMPS script has CNA pre-computed
        if lammps_script and simulation_context:
            if 'compute cna' in lammps_script.lower() or 'compute.*cna' in lammps_script.lower():
                simulation_context['cna_precomputed'] = True
                print("[VIZ] Detected pre-computed CNA in LAMMPS script")
        
        orchestrator = OrchestratorAgent()
        # Returns OVITO script string or None
        ovito_script = orchestrator.parse_visualization_request(
            user_command,
            simulation_context
        )
        return ovito_script
        
    except Exception as e:
        print(f"[VIZ] Warning: Failed to generate visualization script: {e}")
        return None  # Fall back to default visualization


def extract_kim_models(script_string: str) -> List[str]:
    """
    Extract KIM model IDs from a LAMMPS script.
    
    Args:
        script_string: LAMMPS input script as a string
        
    Returns:
        List of KIM model IDs found in the script
    """
    kim_models = []
    kim_init_pattern = r'kim\s+init\s+([A-Za-z0-9_\-]+(?:__[A-Za-z0-9_\-]+)*)'
    
    for line in script_string.split('\n'):
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            continue
        matches = re.findall(kim_init_pattern, line, re.IGNORECASE)
        kim_models.extend(matches)
    
    return kim_models


def ensure_kim_models_installed(script_string: str) -> bool:
    """
    Ensure all KIM models referenced in the script are installed.
    
    Args:
        script_string: LAMMPS input script as a string
        
    Returns:
        True if all models are available, False if any installation failed
    """
    kim_models = extract_kim_models(script_string)
    
    if not kim_models:
        print("No KIM models found in script")
        return True
    
    print(f"Checking KIM models: {kim_models}")
    
    all_available = True
    for model_id in kim_models:
        success, message = kim_utils.install_user_selected_model(model_id)
        
        if success:
            print(message)
        else:
            print(message)
            all_available = False
    
    if all_available:
        print("✅ All required KIM models are available")
    else:
        print("❌ Some KIM models could not be installed")
    
    return all_available


def parse_thermo_line(line: str, headers: List[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a single thermodynamic data line.
    
    Args:
        line: Line of thermodynamic data
        headers: List of column headers
        
    Returns:
        Dictionary of parsed data or None if parsing fails
    """
    try:
        values = line.split()
        
        if len(values) == len(headers):
            data_point = {}
            for i, header in enumerate(headers):
                try:
                    if header.lower() == 'step':
                        data_point[header.lower()] = int(values[i])
                    else:
                        data_point[header.lower()] = float(values[i])
                except ValueError:
                    data_point[header.lower()] = values[i]
            
            return data_point
    except (ValueError, IndexError):
        pass
    
    return None


def find_dump_file(run_folder: str) -> Optional[str]:
    """
    Find dump file using flexible pattern matching.
    
    Args:
        run_folder: Directory to search for dump files
        
    Returns:
        Path to dump file or None if not found
    """
    patterns = ['*.lammpstrj', 'dump.*', '*.dump', '*.atom', 'traj.*']
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(run_folder, pattern))
        if matches:
            return matches[0]
    
    return None


def monitor_dump_file(dump_path: str, frame_queue: Queue, simulation_done: threading.Event, 
                      data_callback: Optional[Callable] = None, max_frames: int = 100,
                      target_timesteps: Optional[List[int]] = None):
    """
    Monitor dump file and identify frames to render (async).
    
    Instead of parsing frames, this now just identifies timesteps and queues
    the file path for OVITO's native reader to handle.
    
    Args:
        dump_path: Path to dump file (or directory to search)
        frame_queue: Queue to put frame info (file path + timestep)
        simulation_done: Event signaling simulation completion
        data_callback: Optional callback for frame notifications
        max_frames: Maximum frames to extract (ignored if target_timesteps provided)
        target_timesteps: Optional list of specific timesteps to extract for smart sampling
    """
    print(f"[DUMP MONITOR] Starting with path: {dump_path}")
    
    # Track directory vs file separately
    if os.path.isdir(dump_path):
        search_dir = dump_path
        actual_dump_file = find_dump_file(search_dir)
        if actual_dump_file:
            print(f"[DUMP MONITOR] Found dump file: {actual_dump_file}")
        else:
            print(f"[DUMP MONITOR] No dump file found yet in {search_dir}, waiting...")
    else:
        search_dir = os.path.dirname(dump_path)
        actual_dump_file = dump_path if os.path.exists(dump_path) else None
        print(f"[DUMP MONITOR] Monitoring specific file: {dump_path}")
    
    frame_counter = 0
    extracted_timesteps = set()  # Track which target timesteps we've extracted
    
    # If target timesteps provided, use smart sampling
    use_smart_sampling = target_timesteps is not None and len(target_timesteps) > 0
    if use_smart_sampling:
        print(f"[DUMP MONITOR] Smart sampling enabled - targeting {len(target_timesteps)} specific timesteps: {target_timesteps}")
    
    # Wait for dump file to be created
    max_wait = 30  # seconds
    wait_time = 0
    while not actual_dump_file and not simulation_done.is_set() and wait_time < max_wait:
        # Re-check for dump file in search directory
        found_path = find_dump_file(search_dir)
        if found_path:
            actual_dump_file = found_path
            print(f"[DUMP MONITOR] Found dump file: {actual_dump_file}")
            break
        time.sleep(0.1)
        wait_time += 0.1
    
    # Final check after wait loop (in case simulation completed quickly)
    if not actual_dump_file:
        actual_dump_file = find_dump_file(search_dir)
        if actual_dump_file:
            print(f"[DUMP MONITOR] Found dump file after simulation completed: {actual_dump_file}")
    
    if not actual_dump_file:
        print(f"[DUMP MONITOR] ERROR: Dump file was never created")
        return
    
    print(f"[DUMP MONITOR] Dump file exists, starting timestep identification")

    # Read file to identify timesteps (lightweight - no parsing)
    # We wait for the NEXT frame header before queueing the PREVIOUS frame
    # to ensure the frame data is complete
    current_timestep = None
    pending_frame = None  # Frame waiting to be queued (needs completion confirmation)

    with open(actual_dump_file, 'r') as f:
        while not simulation_done.is_set() or f.tell() < os.path.getsize(actual_dump_file):
            line = f.readline()

            if not line:
                time.sleep(0.05)  # Wait for more data
                continue

            line = line.strip()

            # Look for timestep markers
            if line == "ITEM: TIMESTEP":
                # If we have a pending frame, queue it now (the new TIMESTEP means previous frame is complete)
                if pending_frame is not None:
                    frame_queue.put(pending_frame)
                    print(f"[DUMP MONITOR] Queued frame {pending_frame['frame_number']} (timestep {pending_frame['timestep']})")
                    frame_counter += 1
                    pending_frame = None

                # Next line contains the timestep
                timestep_line = f.readline()
                if timestep_line:
                    try:
                        current_timestep = int(timestep_line.strip())

                        # Decide whether to queue this frame
                        should_extract = False
                        if use_smart_sampling:
                            # Smart sampling: only extract if timestep matches a target
                            if current_timestep in target_timesteps and current_timestep not in extracted_timesteps:
                                should_extract = True
                                extracted_timesteps.add(current_timestep)
                        else:
                            # Sequential sampling: extract up to max_frames
                            if frame_counter < max_frames:
                                should_extract = True

                        if should_extract:
                            # Don't queue immediately - wait for next TIMESTEP to confirm frame is complete
                            pending_frame = {
                                'dump_file': actual_dump_file,
                                'frame_number': frame_counter,
                                'timestep': current_timestep
                            }
                            
                            # Check completion conditions
                            if use_smart_sampling:
                                if len(extracted_timesteps) >= len(target_timesteps):
                                    print(f"[DUMP MONITOR] All {len(target_timesteps)} target frames queued")
                                    break
                            else:
                                if frame_counter >= max_frames:
                                    print(f"[DUMP MONITOR] Reached maximum frame limit ({max_frames})")
                                    break
                    except ValueError:
                        pass

    # Queue the last pending frame if simulation ended (give it a moment to finish writing)
    if pending_frame is not None:
        time.sleep(0.2)  # Brief delay to ensure last frame data is written
        frame_queue.put(pending_frame)
        print(f"[DUMP MONITOR] Queued final frame {pending_frame['frame_number']} (timestep {pending_frame['timestep']})")
        frame_counter += 1

    if use_smart_sampling:
        print(f"[DUMP MONITOR] Completed. Queued {frame_counter} frames from {len(extracted_timesteps)} target timesteps")
    else:
        print(f"[DUMP MONITOR] Completed. Total frames queued: {frame_counter}")


def rendering_worker(frame_queue: Queue, ovito_script: Optional[str],
                    quality: str, camera_angle: str,
                    frame_images: List[Dict], simulation_done: threading.Event,
                    data_callback: Optional[Callable] = None):
    """
    Worker thread that renders frames asynchronously using OVITO's native file reader.
    
    Args:
        frame_queue: Queue containing frame info (dump_file path + timestep)
        ovito_script: Optional OVITO Python script for custom visualization
        quality: Render quality setting
        camera_angle: Camera angle setting
        frame_images: List to append rendered frames
        simulation_done: Event signaling simulation completion
        data_callback: Optional callback for streaming frames
    """
    print("[RENDERER] Starting rendering worker thread")
    
    while not simulation_done.is_set() or not frame_queue.empty():
        try:
            frame_info = frame_queue.get(timeout=0.5)
        except:
            continue
        
        if frame_info is None:
            continue
        
        try:
            # Use OVITO's native file reader to load dump file with all properties
            dump_file = frame_info['dump_file']
            timestep = frame_info['timestep']
            
            # Render directly from file path
            image_b64 = render_frame_to_image_custom(
                dump_file,
                ovito_script=ovito_script,
                quality=quality,
                camera_angle=camera_angle,
                frame_timestep=timestep
            )
            
            if image_b64:
                frame_result = {
                    'image': image_b64,
                    'frame_number': frame_info['frame_number'],
                    'timestep': timestep
                }
                frame_images.append(frame_result)
                print(f"[RENDERER] Rendered frame {frame_info['frame_number']} (timestep {timestep})")
                
                # Stream via callback
                if data_callback:
                    try:
                        data_callback({
                            'type': 'trajectory_frame',
                            'image_data': image_b64,
                            'frame_number': frame_info['frame_number'],
                            'timestep': timestep
                        })
                    except Exception as e:
                        print(f"[RENDERER] Warning: Error in visualization callback: {e}")
            else:
                print(f"[RENDERER] Warning: Failed to render frame {frame_info['frame_number']}")
        
        except Exception as e:
            print(f"Warning: Error rendering frame: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    print(f"[RENDERER] Completed. Total frames rendered: {len(frame_images)}")


def run_lammps_simulation(script_string: str, render_config: Optional[Dict[str, Any]] = None, 
                          data_callback: Optional[Callable] = None, 
                          num_cores: Optional[int] = None) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[Dict]]:
    """
    Execute MPI-based LAMMPS simulation with real-time streaming and async rendering.
    
    ALWAYS uses multicore MPI execution. No serial fallback.
    
    Args:
        script_string: Complete LAMMPS input script
        render_config: Visualization configuration
        data_callback: Callback for real-time data streaming
        num_cores: Number of cores (defaults to config.LAMMPS_CORES)
        
    Returns:
        Tuple of (output_files, thermo_data, frame_images)
    """
    cores = num_cores if num_cores is not None else config.LAMMPS_CORES
    
    print(f"Running MPI-based LAMMPS simulation with {cores} cores")
    
    # Ensure KIM models are installed
    if not ensure_kim_models_installed(script_string):
        raise RuntimeError("Failed to install required KIM models")
    
    # Create run folder
    run_folder = run_manager.create_run_folder()
    print(f"Created run folder: {run_folder}")
    
    # Save script and metadata
    run_manager.save_lammps_script(run_folder, script_string)
    run_manager.save_simulation_metadata(run_folder, {
        'cores': cores,
        'status': 'starting'
    })
    
    # Extract visualization parameters from render_config
    if render_config is None:
        render_config = {}
    
    ovito_script = render_config.get('ovito_script', None)
    quality = render_config.get('quality', 'medium')
    camera_angle = render_config.get('camera_angle', 'default')
    max_queue_size = render_config.get('max_queue_size', 10)
    target_timesteps = render_config.get('target_timesteps', None)
    
    # Shared data structures
    thermo_data = []
    frame_images = []
    frame_queue = Queue(maxsize=max_queue_size)
    simulation_done = threading.Event()
    
    try:
        # Write script to run folder
        script_path = os.path.join(run_folder, "input.lammps")
        
        # Start rendering worker with direct parameters
        render_thread = threading.Thread(
            target=rendering_worker,
            args=(frame_queue, ovito_script, quality, camera_angle, 
                  frame_images, simulation_done, data_callback)
        )
        render_thread.daemon = True
        render_thread.start()
        
        # Prepare MPI command
        cmd = [config.LAMMPS_MPI_COMMAND]
        
        # Add hardware thread flag if enabled (fixes slot allocation on hyperthreaded systems)
        if config.LAMMPS_USE_HWTHREADS:
            cmd.append('--use-hwthread-cpus')
        
        cmd.extend([
            '-np', str(cores),
            config.LAMMPS_EXECUTABLE,
            '-in', script_path
        ])
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Start MPI process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=run_folder,
            bufsize=1,
            universal_newlines=True
        )
        
        # Start dump file monitor (async) - pass run_folder to find dump file dynamically
        monitor_thread = threading.Thread(
            target=monitor_dump_file,
            args=(run_folder, frame_queue, simulation_done, data_callback, 
                  config.MAX_TRAJECTORY_FRAMES, target_timesteps)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Stream thermodynamic data from stdout
        headers = None
        in_thermo_section = False
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line = line.strip()
            
            # Detect thermo header
            if line.startswith('Step') and any(kw in line for kw in ['Temp', 'Press', 'PotEng', 'KinEng', 'TotEng']):
                headers = line.split()
                in_thermo_section = True
                continue
            
            # Parse thermo data
            if in_thermo_section and headers:
                if not line or line.startswith('Loop') or line.startswith('Performance') or line.startswith('WARNING'):
                    in_thermo_section = False
                    continue
                
                data_point = parse_thermo_line(line, headers)
                if data_point:
                    thermo_data.append(data_point)
                    
                    # Stream to frontend
                    if data_callback:
                        try:
                            data_callback(data_point)
                        except Exception as e:
                            print(f"Warning: Error in thermo callback: {e}")
        
        # Wait for process completion
        process.wait()
        
        if process.returncode != 0:
            stderr_output = process.stderr.read()
            raise RuntimeError(f"LAMMPS failed with code {process.returncode}\n{stderr_output}")
        
        print("MPI simulation completed successfully")
        
        # Signal completion
        simulation_done.set()
        
        # Wait for threads
        monitor_thread.join(timeout=10)
        render_thread.join(timeout=30)
        
        # Update run folder status
        run_manager.update_simulation_status(run_folder, 'completed', 
                                            total_frames=len(frame_images),
                                            total_thermo_points=len(thermo_data))
        
        # Collect output files
        output_files = {}
        for filename in os.listdir(run_folder):
            if filename.endswith(('.atom', '.dump', '.lammpstrj')):
                output_files['dump'] = os.path.join(run_folder, filename)
            elif filename.startswith('log.') or filename == 'log.lammps':
                output_files['log'] = os.path.join(run_folder, filename)
        
        # Completion callback
        if data_callback:
            try:
                data_callback({
                    'type': 'trajectory_complete',
                    'total_frames': len(frame_images),
                    'message': f'Simulation complete: {len(frame_images)} frames rendered'
                })
            except Exception as e:
                print(f"Warning: Error in completion callback: {e}")
        
        print(f"Simulation completed: {len(frame_images)} frames, {len(thermo_data)} thermo points")
        print(f"Results saved to: {run_folder}")
        
        return output_files, thermo_data, frame_images
        
    except Exception as e:
        simulation_done.set()
        run_manager.update_simulation_status(run_folder, 'failed', error=str(e))
        raise
