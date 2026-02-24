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
import tempfile
import base64
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

    Uses DSPy modules (SmartOrchestrator for detection, OVITOScriptGeneration
    for script generation) instead of the former OrchestratorAgent.

    Args:
        user_command: User's natural language command (may include visualization preferences)
        simulation_context: Optional simulation context (material, structure, etc.)
        lammps_script: Optional LAMMPS script to check for pre-computed properties

    Returns:
        OVITO Python script as a string, or None for default visualization
    """
    from agents import SmartOrchestrator, IterativeOVITOGeneration, configure_dspy_lm, detect_precomputed_properties

    try:
        configure_dspy_lm()

        if simulation_context is None:
            simulation_context = {}

        # Detect pre-computed properties from the LAMMPS script
        if lammps_script:
            precomputed = detect_precomputed_properties(lammps_script)
            if precomputed:
                simulation_context['precomputed_properties'] = precomputed
                print(f"[VIZ] Detected pre-computed properties: {list(precomputed.keys())}")

        # Detect custom visualization intent via DSPy
        orchestrator = SmartOrchestrator()
        has_custom_viz = orchestrator.detect_visualization(user_command)

        if not has_custom_viz:
            print("[VIZ] No custom visualization requested, using default")
            return None

        # Generate OVITO script via iterative DSPy module
        ovito_gen = IterativeOVITOGeneration(max_attempts=config.OVITO_SCRIPT_MAX_RETRIES)
        result = ovito_gen.forward(user_command, simulation_context)
        if result.review_passed and result.script:
            print(f"[VIZ] Generated custom OVITO script ({len(result.script)} chars) after {result.attempts} attempt(s)")
            return result.script
        else:
            print(f"[VIZ] OVITO script failed review after {result.attempts} attempts, falling back to default")
            return None

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


def find_dump_file(run_folder: str, dump_filename_pattern: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find dump file(s) using flexible pattern matching.

    Detects whether the dump output is a single concatenated file or a
    per-timestep series (LAMMPS ``*`` wildcard).

    Args:
        run_folder: Directory to search for dump files
        dump_filename_pattern: Optional glob pattern from the LLM (e.g.
            ``dump.Cu_solidification.*.dat``).  When provided, this is
            tried first for reliable matching.

    Returns:
        Dict with keys:
            mode: 'single' | 'per_timestep'
            path: str  — path to file (single) or first file (per_timestep)
            pattern: str  — glob pattern used
            files: list[str]  — all matched files sorted by timestep
        Or None if no dump files found.
    """
    # --- Try LLM-provided pattern first ---
    if dump_filename_pattern:
        # Replace the LAMMPS ``*`` with a filesystem glob ``*``
        fs_pattern = dump_filename_pattern  # already uses ``*``
        is_per_timestep = '*' in dump_filename_pattern
        matches = sorted(glob.glob(os.path.join(run_folder, fs_pattern)))
        if matches or is_per_timestep:
            return {
                'mode': 'per_timestep' if is_per_timestep else 'single',
                'path': matches[0] if matches else '',
                'pattern': fs_pattern,
                'files': matches,
            }

    # --- Fallback: filesystem-based detection ---
    patterns = ['*.lammpstrj', 'dump.*', '*.dump', '*.atom', '*.dat', 'traj.*']

    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(run_folder, pattern)))
        if not matches:
            continue

        # Heuristic: if multiple files match and their names contain a
        # numeric component that varies, treat as per-timestep series.
        if len(matches) > 1:
            # Check if filenames differ only by a numeric segment
            basenames = [os.path.basename(m) for m in matches]
            # Extract numeric parts
            nums = []
            for bn in basenames:
                parts = re.findall(r'\d+', bn)
                if parts:
                    nums.append(parts)
            if len(nums) == len(basenames) and len(set(len(n) for n in nums)) == 1:
                return {
                    'mode': 'per_timestep',
                    'path': matches[0],
                    'pattern': pattern,
                    'files': matches,
                }

        # Single file (or ambiguous multi-file that doesn't look per-timestep)
        return {
            'mode': 'single',
            'path': matches[0],
            'pattern': pattern,
            'files': matches[:1],
        }

    return None


def extract_timestep_from_filename(filepath: str, pattern: str = "") -> int:
    """Extract the timestep number from a per-timestep dump filename.

    For a pattern like ``dump.Cu.*.dat`` and a file ``dump.Cu.5000.dat``,
    returns 5000.  Falls back to extracting the largest integer from the
    basename if the pattern doesn't help.

    Args:
        filepath: Full path to the dump file.
        pattern: The glob pattern used (e.g. ``dump.Cu.*.dat``).

    Returns:
        Extracted timestep as an integer, or -1 on failure.
    """
    basename = os.path.basename(filepath)

    if pattern and '*' in pattern:
        # Split pattern on '*' to get prefix and suffix
        pat_base = os.path.basename(pattern)
        parts = pat_base.split('*', 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ''

        # Strip prefix and suffix from the basename to isolate the timestep
        ts_str = basename
        if prefix and ts_str.startswith(prefix):
            ts_str = ts_str[len(prefix):]
        if suffix and ts_str.endswith(suffix):
            ts_str = ts_str[:-len(suffix)]

        try:
            return int(ts_str)
        except ValueError:
            pass

    # Fallback: find all integers in the basename, return the largest
    nums = re.findall(r'\d+', basename)
    if nums:
        return max(int(n) for n in nums)

    return -1


def extract_dump_pattern_from_script(script_string: str) -> Optional[str]:
    """Extract dump filename pattern from LAMMPS script.

    Parses: dump ID group style FREQ filename args...
    Returns the filename (e.g. 'dump.Cu_solidification.*.dat' or 'traj.lammpstrj')
    """
    for line in script_string.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if stripped.startswith('dump ') and 'custom' in stripped:
            parts = stripped.split()
            if len(parts) >= 6:
                return parts[5]  # filename pattern
    return None


def estimate_frame_stride(script_string: str, max_frames: int) -> int:
    """Estimate stride for evenly-spaced frame sampling.

    Parses `run` commands and `dump` frequency from the LAMMPS script
    to estimate total frames, then returns stride = max(1, total // max_frames).
    """
    total_steps = 0
    dump_freq = 1000

    for line in script_string.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if stripped.startswith('run '):
            try:
                total_steps += int(stripped.split()[1])
            except (IndexError, ValueError):
                pass
        # Match: dump ID group style FREQ file ...
        if stripped.startswith('dump ') and 'custom' in stripped:
            try:
                parts = stripped.split()
                dump_freq = int(parts[4])
            except (IndexError, ValueError):
                pass

    if total_steps > 0:
        estimated_frames = total_steps // dump_freq
        stride = max(1, estimated_frames // max_frames)
        print(f"[STRIDE] Estimated {estimated_frames} frames (run={total_steps}, dump_freq={dump_freq}), stride={stride} for max {max_frames}")
        return stride
    return 1  # default: take every frame


def monitor_dump_file(dump_path: str, frame_queue: Queue, simulation_done: threading.Event,
                      data_callback: Optional[Callable] = None, max_frames: int = 100,
                      target_timesteps: Optional[List[int]] = None,
                      frame_stride: int = 1,
                      dump_filename_pattern: Optional[str] = None):
    """
    Monitor dump file(s) and identify frames to render (async).

    Supports two modes:
    - **Single-file mode**: A single dump file with all frames appended.
      Reads ITEM: TIMESTEP markers to identify complete frames.
    - **Per-timestep mode**: Separate files per timestep (LAMMPS ``*`` wildcard).
      Polls the directory for new files matching the glob pattern.

    Args:
        dump_path: Path to dump file (or directory to search)
        frame_queue: Queue to put frame info (file path + timestep)
        simulation_done: Event signaling simulation completion
        data_callback: Optional callback for frame notifications
        max_frames: Maximum frames to extract (ignored if target_timesteps provided)
        target_timesteps: Optional list of specific timesteps to extract for smart sampling
        frame_stride: Extract every Nth frame for even spacing (ignored if target_timesteps provided)
        dump_filename_pattern: Optional dump filename pattern from LLM (e.g. 'dump.Cu.*.dat')
    """
    print(f"[DUMP MONITOR] Starting with path: {dump_path}")

    search_dir = dump_path if os.path.isdir(dump_path) else os.path.dirname(dump_path)

    # --- Wait for dump files to appear ---
    dump_info = None
    max_wait = 30  # seconds
    wait_time = 0
    while dump_info is None and not simulation_done.is_set() and wait_time < max_wait:
        dump_info = find_dump_file(search_dir, dump_filename_pattern)
        if dump_info:
            break
        time.sleep(0.1)
        wait_time += 0.1

    # Final check after wait loop
    if dump_info is None:
        dump_info = find_dump_file(search_dir, dump_filename_pattern)

    # If find_dump_file returns None but we know it's per-timestep from the pattern
    if dump_info is None and dump_filename_pattern and '*' in dump_filename_pattern:
        dump_info = {
            'mode': 'per_timestep',
            'path': '',
            'pattern': dump_filename_pattern,
            'files': [],
        }

    if dump_info is None:
        print(f"[DUMP MONITOR] ERROR: Dump file was never created")
        return

    mode = dump_info['mode']
    print(f"[DUMP MONITOR] Detected mode: {mode}, pattern: {dump_info['pattern']}, initial files: {len(dump_info['files'])}")

    # If target timesteps provided, use smart sampling
    use_smart_sampling = target_timesteps is not None and len(target_timesteps) > 0
    if use_smart_sampling:
        target_set = set(target_timesteps)
        print(f"[DUMP MONITOR] Smart sampling enabled - targeting {len(target_timesteps)} specific timesteps")

    # ===================================================================
    # PER-TIMESTEP MODE: poll directory for new files
    # ===================================================================
    if mode == 'per_timestep':
        pattern_for_glob = dump_filename_pattern or dump_info['pattern']
        glob_path = os.path.join(search_dir, pattern_for_glob)
        seen_files = set()
        frame_counter = 0
        frames_seen = 0
        extracted_timesteps_set = set()

        while not simulation_done.is_set() or True:
            current_files = set(glob.glob(glob_path))
            new_files = current_files - seen_files

            if new_files:
                # Sort new files by extracted timestep (numeric order)
                new_sorted = sorted(
                    new_files,
                    key=lambda f: extract_timestep_from_filename(f, pattern_for_glob)
                )

                for fpath in new_sorted:
                    timestep = extract_timestep_from_filename(fpath, pattern_for_glob)
                    seen_files.add(fpath)

                    # Decide whether to queue this frame
                    should_extract = False
                    if use_smart_sampling:
                        if timestep in target_set and timestep not in extracted_timesteps_set:
                            should_extract = True
                            extracted_timesteps_set.add(timestep)
                    else:
                        if frames_seen % frame_stride == 0 and frame_counter < max_frames:
                            should_extract = True
                        frames_seen += 1

                    if should_extract:
                        frame_info = {
                            'dump_file': fpath,
                            'frame_number': frame_counter,
                            'file_frame_index': 0,  # Each per-timestep file has exactly 1 frame
                            'timestep': timestep,
                        }
                        frame_queue.put(frame_info)
                        print(f"[DUMP MONITOR] Queued per-timestep frame {frame_counter} "
                              f"(timestep {timestep}, file {os.path.basename(fpath)})")
                        frame_counter += 1

                        # Pace queuing to match renderer throughput (~2-3s per frame)
                        # Only pace if queue is getting full (>50% capacity)
                        if frame_queue.qsize() > frame_queue.maxsize // 2:
                            time.sleep(1.0)

                        # Check completion
                        if use_smart_sampling and len(extracted_timesteps_set) >= len(target_timesteps):
                            print(f"[DUMP MONITOR] All {len(target_timesteps)} target frames queued")
                            print(f"[DUMP MONITOR] Completed. Total frames queued: {frame_counter}")
                            return
                        if not use_smart_sampling and frame_counter >= max_frames:
                            print(f"[DUMP MONITOR] Reached max_frames={max_frames}")
                            print(f"[DUMP MONITOR] Completed. Total frames queued: {frame_counter}")
                            return

            # If simulation is done and no new files, do one final sweep and exit
            if simulation_done.is_set():
                # One more sweep
                final_files = set(glob.glob(glob_path)) - seen_files
                if not final_files:
                    break
                # Process final files in the next iteration
                continue

            time.sleep(0.2)  # Poll interval

        print(f"[DUMP MONITOR] Completed (per-timestep). Total frames queued: {frame_counter}")
        return

    # ===================================================================
    # SINGLE-FILE MODE: read ITEM: TIMESTEP markers (existing behavior)
    # ===================================================================
    actual_dump_file = dump_info['path']
    print(f"[DUMP MONITOR] Monitoring single file: {actual_dump_file}")

    frame_counter = 0
    frames_seen = 0
    extracted_timesteps_set = set()
    current_timestep = None
    pending_frame = None

    with open(actual_dump_file, 'r') as f:
        while not simulation_done.is_set() or f.tell() < os.path.getsize(actual_dump_file):
            line = f.readline()

            if not line:
                time.sleep(0.05)
                continue

            line = line.strip()

            if line == "ITEM: TIMESTEP":
                # Queue the previous pending frame (new TIMESTEP means it's complete)
                if pending_frame is not None:
                    frame_queue.put(pending_frame)
                    print(f"[DUMP MONITOR] Queued frame {pending_frame['frame_number']} (timestep {pending_frame['timestep']})")
                    frame_counter += 1
                    pending_frame = None

                timestep_line = f.readline()
                if timestep_line:
                    try:
                        current_timestep = int(timestep_line.strip())

                        should_extract = False
                        if use_smart_sampling:
                            if current_timestep in target_set and current_timestep not in extracted_timesteps_set:
                                should_extract = True
                                extracted_timesteps_set.add(current_timestep)
                        else:
                            if frames_seen % frame_stride == 0 and frame_counter < max_frames:
                                should_extract = True
                            frames_seen += 1

                        if should_extract:
                            pending_frame = {
                                'dump_file': actual_dump_file,
                                'frame_number': frame_counter,
                                'file_frame_index': frame_counter,
                                'timestep': current_timestep,
                            }

                            if use_smart_sampling and len(extracted_timesteps_set) >= len(target_timesteps):
                                print(f"[DUMP MONITOR] All {len(target_timesteps)} target frames queued")
                                break
                    except ValueError:
                        pass

    # Queue last pending frame
    if pending_frame is not None:
        time.sleep(0.2)
        frame_queue.put(pending_frame)
        print(f"[DUMP MONITOR] Queued final frame {pending_frame['frame_number']} (timestep {pending_frame['timestep']})")
        frame_counter += 1

    if use_smart_sampling:
        print(f"[DUMP MONITOR] Completed. Queued {frame_counter} frames from {len(extracted_timesteps_set)} target timesteps")
    else:
        print(f"[DUMP MONITOR] Completed. Total frames queued: {frame_counter}")


def rendering_worker(frame_queue: Queue, ovito_script: Optional[str],
                    quality: str, camera_angle: str,
                    frame_images: List[Dict], simulation_done: threading.Event,
                    data_callback: Optional[Callable] = None):
    """
    Worker thread that renders frames asynchronously using OVITO's native file reader.

    Caches the OVITO pipeline, modifiers, and renderer across frames to minimize
    GIL hold time per frame (import once, apply script once, render per frame).

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

    import ovito
    from ovito.io import import_file
    from ovito.vis import Viewport, TachyonRenderer
    from ovito_executor import DynamicOVITOExecutor
    from visualization_rendering import apply_basic_visualization

    cached_pipeline = None
    cached_file = None
    known_frames = 0

    # Setup renderer once
    renderer = TachyonRenderer()
    renderer.antialiasing = True
    renderer.shadows = True
    renderer.ambient_occlusion = True

    # Determine render size from quality
    size_map = {'high': (800, 600), 'low': (400, 300), 'medium': (600, 450)}
    render_size = size_map.get(quality, (600, 450))

    # Create viewport ONCE (reused across all frames)
    viewport = Viewport()
    viewport.type = Viewport.Type.Perspective
    if camera_angle == 'top':
        viewport.camera_pos = (0, 0, 50)
        viewport.camera_dir = (0, 0, -1)
    elif camera_angle == 'side':
        viewport.camera_pos = (50, 0, 0)
        viewport.camera_dir = (-1, 0, 0)
    else:  # default perspective
        viewport.camera_pos = (30, 30, 30)
        viewport.camera_dir = (-1, -1, -1)

    while not simulation_done.is_set() or not frame_queue.empty():
        try:
            frame_info = frame_queue.get(timeout=0.5)
        except:
            continue

        if frame_info is None:
            continue

        try:
            dump_file = frame_info['dump_file']
            frame_index = frame_info['frame_number']  # Sequential counter for output
            file_frame_index = frame_info.get('file_frame_index', frame_index)  # Index within file
            timestep = frame_info['timestep']

            # Reimport only when file changed or we need frames beyond known count
            if cached_file != dump_file or file_frame_index >= known_frames:
                if cached_pipeline:
                    cached_pipeline.remove_from_scene()

                # Defensive: clear ALL pipelines from scene to prevent ghosting
                for p in list(ovito.scene.pipelines):
                    p.remove_from_scene()
                del cached_pipeline
                cached_pipeline = None

                print(f"[RENDERER] Importing dump file: {dump_file}")
                cached_pipeline = import_file(dump_file)
                known_frames = cached_pipeline.source.num_frames
                cached_pipeline.modifiers.clear()

                if ovito_script:
                    try:
                        data = cached_pipeline.compute(0)
                        executor = DynamicOVITOExecutor()
                        cached_pipeline = executor.execute_script(
                            ovito_script, cached_pipeline, data)
                        print("[RENDERER] Applied custom OVITO script to cached pipeline")
                    except Exception as e:
                        print(f"[RENDERER] Script execution failed: {e}, using basic visualization")
                        apply_basic_visualization(cached_pipeline)
                else:
                    apply_basic_visualization(cached_pipeline)

                cached_pipeline.add_to_scene()
                cached_file = dump_file

            # Fit camera to current frame geometry
            viewport.zoom_all()

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            try:
                viewport.render_image(
                    filename=tmp_path,
                    size=render_size,
                    alpha=True,
                    renderer=renderer,
                    frame=file_frame_index
                )
                with open(tmp_path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode('utf-8')
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            if image_b64:
                frame_result = {
                    'image': image_b64,
                    'frame_number': frame_index,
                    'timestep': timestep
                }
                frame_images.append(frame_result)
                print(f"[RENDERER] Rendered frame {frame_index} (timestep {timestep})")

                # Stream via callback
                if data_callback:
                    try:
                        data_callback({
                            'type': 'trajectory_frame',
                            'image_data': image_b64,
                            'frame_number': frame_index,
                            'timestep': timestep
                        })
                    except Exception as e:
                        print(f"[RENDERER] Warning: Error in visualization callback: {e}")

        except Exception as e:
            print(f"Warning: Error rendering frame: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue

    # Final cleanup: clear ALL pipelines from scene
    for p in list(ovito.scene.pipelines):
        p.remove_from_scene()

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

    # Inject thermo_modify flush yes to prevent buffered stdout under MPI
    if 'thermo_modify' not in script_string or 'flush' not in script_string:
        lines = script_string.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('thermo ') and not stripped.startswith('thermo_'):
                lines.insert(i + 1, 'thermo_modify flush yes')
                print("[LAMMPS] Injected 'thermo_modify flush yes' for real-time streaming")
                break
        script_string = '\n'.join(lines)

    # Compute frame stride for even spacing
    frame_stride = estimate_frame_stride(script_string, config.MAX_TRAJECTORY_FRAMES)

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
    dump_filename_pattern = render_config.get('dump_filename_pattern', None)

    # Deterministic fallback: parse the LAMMPS script for the dump filename pattern
    if not dump_filename_pattern:
        dump_filename_pattern = extract_dump_pattern_from_script(script_string) or ''
        if dump_filename_pattern:
            print(f"[LAMMPS] Extracted dump pattern from script: {dump_filename_pattern}")

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
                  config.MAX_TRAJECTORY_FRAMES, target_timesteps, frame_stride,
                  dump_filename_pattern)
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
                if line.startswith('Loop') or line.startswith('Performance'):
                    in_thermo_section = False
                    continue

                # Skip blank lines and informational lines WITHOUT exiting thermo mode
                if not line or line.startswith('WARNING') or line.startswith('Neighbor') or \
                   line.startswith('Setting up') or line.startswith('Per MPI') or \
                   line.startswith('Total wall'):
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
        
        # Collect output files using find_dump_file for reliable detection
        output_files = {}
        dump_result = find_dump_file(run_folder, dump_filename_pattern)
        if dump_result:
            if dump_result['mode'] == 'per_timestep':
                output_files['dump'] = dump_result['files']  # list of all per-timestep files
                output_files['dump_mode'] = 'per_timestep'
            else:
                output_files['dump'] = dump_result['path']
                output_files['dump_mode'] = 'single'

        for filename in os.listdir(run_folder):
            if filename.startswith('log.') or filename == 'log.lammps':
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
