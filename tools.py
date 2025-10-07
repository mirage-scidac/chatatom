"""
Tools for executing LAMMPS simulations with MPI-based parallel execution.

This module provides MPI-only execution with:
- Real-time thermodynamic data streaming
- Asynchronous frame rendering during simulation
- Organized run folder management
"""
import os
import shutil
import subprocess
import re
import glob
from typing import Dict, List, Any, Tuple, Optional, Callable
import config
import threading
import time
import numpy as np
import kim_utils
import run_manager
from visualization_rendering import render_frame_to_image_custom
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


def parse_visualization_request(user_command: str, simulation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse visualization request from user command and generate OVITO script.
    
    Uses OrchestratorAgent as the single source of truth for understanding
    visualization intent.
    
    Args:
        user_command: User's natural language command (may include visualization preferences)
        simulation_context: Optional simulation context (material, structure, etc.)
        
    Returns:
        Dictionary with visualization configuration
    """
    from agents import OrchestratorAgent
    
    try:
        orchestrator = OrchestratorAgent()
        result = orchestrator.parse_visualization_request(
            user_command,
            simulation_context
        )
        return result
        
    except Exception as e:
        print(f"Warning: Failed to parse visualization request: {e}")
        return {
            'mode': 'config',
            'quality': 'medium',
            'camera_angle': 'default',
            'has_custom_request': False
        }


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


def parse_dump_frame(lines: List[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a single frame from LAMMPS dump file.
    
    Args:
        lines: Lines comprising one complete frame
        
    Returns:
        Dictionary with frame data or None if parsing fails
    """
    try:
        frame_data = {}
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "ITEM: TIMESTEP":
                frame_data['timestep'] = int(lines[i + 1].strip())
                i += 2
            elif line == "ITEM: NUMBER OF ATOMS":
                frame_data['natoms'] = int(lines[i + 1].strip())
                i += 2
            elif line.startswith("ITEM: BOX BOUNDS"):
                box_lines = lines[i + 1:i + 4]
                xlo, xhi = map(float, box_lines[0].split()[:2])
                ylo, yhi = map(float, box_lines[1].split()[:2])
                zlo, zhi = map(float, box_lines[2].split()[:2])
                frame_data['box'] = {
                    'xlo': xlo, 'xhi': xhi,
                    'ylo': ylo, 'yhi': yhi,
                    'zlo': zlo, 'zhi': zhi
                }
                i += 4
            elif line.startswith("ITEM: ATOMS"):
                # Parse atom data
                headers = line.split()[2:]  # Skip "ITEM: ATOMS"
                natoms = frame_data.get('natoms', 0)
                atom_lines = lines[i + 1:i + 1 + natoms]
                
                # Initialize arrays
                positions = []
                types = []
                
                for atom_line in atom_lines:
                    values = atom_line.split()
                    atom_dict = dict(zip(headers, values))
                    
                    # Extract position (x, y, z)
                    x = float(atom_dict.get('x', atom_dict.get('xu', 0)))
                    y = float(atom_dict.get('y', atom_dict.get('yu', 0)))
                    z = float(atom_dict.get('z', atom_dict.get('zu', 0)))
                    positions.append([x, y, z])
                    
                    # Extract type
                    types.append(int(atom_dict.get('type', 1)))
                
                frame_data['positions'] = np.array(positions)
                frame_data['types'] = np.array(types, dtype=int)
                i += natoms + 1
            else:
                i += 1
        
        return frame_data if 'positions' in frame_data else None
        
    except Exception as e:
        print(f"Error parsing dump frame: {e}")
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
                      data_callback: Optional[Callable] = None, max_frames: int = 100):
    """
    Monitor dump file and extract frames as they're written (async).
    
    Args:
        dump_path: Path to dump file (or directory to search)
        frame_queue: Queue to put parsed frames
        simulation_done: Event signaling simulation completion
        data_callback: Optional callback for frame notifications
        max_frames: Maximum frames to extract
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
    current_frame_lines = []
    in_frame = False
    
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
    
    print(f"[DUMP MONITOR] Dump file exists, starting frame extraction")
    
    with open(actual_dump_file, 'r') as f:
        while not simulation_done.is_set() or f.tell() < os.path.getsize(actual_dump_file):
            line = f.readline()
            
            if not line:
                time.sleep(0.05)  # Wait for more data
                continue
            
            if line.strip() == "ITEM: TIMESTEP":
                # Start of new frame
                if current_frame_lines and frame_counter < max_frames:
                    # Parse previous frame
                    frame_data = parse_dump_frame(current_frame_lines)
                    if frame_data:
                        frame_queue.put({
                            'data': frame_data,
                            'frame_number': frame_counter,
                            'timestep': frame_data.get('timestep', frame_counter)
                        })
                        print(f"[DUMP MONITOR] Extracted frame {frame_counter} (timestep {frame_data.get('timestep', frame_counter)})")
                        frame_counter += 1
                        
                        if frame_counter >= max_frames:
                            print(f"[DUMP MONITOR] Reached maximum frame limit ({max_frames})")
                            break
                
                current_frame_lines = [line]
                in_frame = True
            elif in_frame:
                current_frame_lines.append(line)
        
        # Parse final frame
        if current_frame_lines and frame_counter < max_frames:
            frame_data = parse_dump_frame(current_frame_lines)
            if frame_data:
                frame_queue.put({
                    'data': frame_data,
                    'frame_number': frame_counter,
                    'timestep': frame_data.get('timestep', frame_counter)
                })
    
    print(f"[DUMP MONITOR] Completed. Total frames extracted: {frame_counter}")


def convert_dump_frame_to_ovito(frame_data: Dict[str, Any]):
    """
    Convert parsed dump frame to OVITO DataCollection.
    
    Args:
        frame_data: Parsed frame data from parse_dump_frame
        
    Returns:
        OVITO DataCollection or None
    """
    try:
        from ovito.data import DataCollection
        
        data = DataCollection()
        particles = data.create_particles()
        particles.count = frame_data['natoms']
        particles.positions = frame_data['positions']
        particles.particle_types = frame_data['types']
        
        # Create simulation cell
        box = frame_data['box']
        cell_matrix = np.array([
            [box['xhi'] - box['xlo'], 0, 0, box['xlo']],
            [0, box['yhi'] - box['ylo'], 0, box['ylo']],
            [0, 0, box['zhi'] - box['zlo'], box['zlo']]
        ])
        
        cell = data.create_cell(matrix=cell_matrix)
        cell.pbc = (True, True, True)
        
        return data
        
    except Exception as e:
        print(f"Error converting dump frame to OVITO: {e}")
        return None


def rendering_worker(frame_queue: Queue, render_config: Dict[str, Any], 
                    frame_images: List[Dict], simulation_done: threading.Event,
                    data_callback: Optional[Callable] = None):
    """
    Worker thread that renders frames asynchronously.
    
    Args:
        frame_queue: Queue containing frames to render
        render_config: Rendering configuration
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
            # Convert to OVITO and render
            ovito_data = convert_dump_frame_to_ovito(frame_info['data'])
            if ovito_data:
                image_b64 = render_frame_to_image_custom(ovito_data, render_config)
                if image_b64:
                    frame_result = {
                        'image': image_b64,
                        'frame_number': frame_info['frame_number'],
                        'timestep': frame_info['timestep']
                    }
                    frame_images.append(frame_result)
                    print(f"[RENDERER] Rendered frame {frame_info['frame_number']}")
                    
                    # Stream via callback
                    if data_callback:
                        try:
                            data_callback({
                                'type': 'trajectory_frame',
                                'image_data': image_b64,
                                'frame_number': frame_info['frame_number'],
                                'timestep': frame_info['timestep']
                            })
                        except Exception as e:
                            print(f"[RENDERER] Warning: Error in visualization callback: {e}")
                else:
                    print(f"[RENDERER] Warning: Failed to render frame {frame_info['frame_number']}")
            else:
                print(f"[RENDERER] Warning: Failed to convert frame {frame_info['frame_number']} to OVITO")
        
        except Exception as e:
            print(f"Warning: Error rendering frame: {e}")
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
    
    # Set default render config
    if render_config is None:
        render_config = {
            'quality': 'medium',
            'camera_angle': 'default',
            'max_queue_size': 10
        }
    
    # Shared data structures
    thermo_data = []
    frame_images = []
    frame_queue = Queue(maxsize=render_config.get('max_queue_size', 10))
    simulation_done = threading.Event()
    
    try:
        # Write script to run folder
        script_path = os.path.join(run_folder, "input.lammps")
        
        # Start rendering worker
        render_thread = threading.Thread(
            target=rendering_worker,
            args=(frame_queue, render_config, frame_images, simulation_done, data_callback)
        )
        render_thread.daemon = True
        render_thread.start()
        
        # Prepare MPI command
        cmd = [
            config.LAMMPS_MPI_COMMAND,
            '-np', str(cores),
            config.LAMMPS_EXECUTABLE,
            '-in', script_path
        ]
        
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
            args=(run_folder, frame_queue, simulation_done, data_callback, config.MAX_TRAJECTORY_FRAMES)
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
