"""
Tools for executing LAMMPS simulations and extracting data.
"""
import os
import tempfile
import shutil
import subprocess
import re
from typing import Dict, List, Any, Tuple, Optional, Callable
import lammps
import config
import threading
import time
import numpy as np
import kim_utils


def extract_kim_models(script_string: str) -> List[str]:
    """
    Extract KIM model IDs from a LAMMPS script.
    
    Args:
        script_string: LAMMPS input script as a string
        
    Returns:
        List of KIM model IDs found in the script
    """
    kim_models = []
    
    # Pattern to match kim init commands - KIM model IDs can contain letters, numbers, underscores, and dashes
    kim_init_pattern = r'kim\s+init\s+([A-Za-z0-9_\-]+(?:__[A-Za-z0-9_\-]+)*)'
    
    # Process line by line to skip comments
    for line in script_string.split('\n'):
        # Skip comment lines
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            continue
            
        # Apply regex to non-comment lines
        matches = re.findall(kim_init_pattern, line, re.IGNORECASE)
        kim_models.extend(matches)
    
    return kim_models


def is_kim_model_installed(model_id: str) -> bool:
    """
    Check if a KIM model is installed by looking in KIM directories.
    
    Args:
        model_id: KIM model identifier
        
    Returns:
        True if model is installed, False otherwise
    """
    # First try using kim-api-collections-management if available
    try:
        result = subprocess.run(
            ['kim-api-collections-management', 'list', '--installed'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and model_id in result.stdout:
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback: Check common KIM directories
    import os
    kim_dirs = [
        os.path.expanduser("~/.kim-api"),
        "/usr/local/lib/openkim-models",
        "/opt/openkim-models"
    ]
    
    # Also check conda environment if available
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        kim_dirs.append(os.path.join(conda_prefix, 'lib', 'openkim-models'))
    
    # Look for the model in portable-models and simulator-models subdirectories
    for base_dir in kim_dirs:
        for subdir in ['portable-models', 'simulator-models']:
            search_dir = os.path.join(base_dir, subdir)
            if os.path.exists(search_dir):
                for item in os.listdir(search_dir):
                    if model_id in item:
                        print(f"Found KIM model {model_id} in {os.path.join(search_dir, item)}")
                        return True
            
            # Also check versioned directories
            if os.path.exists(base_dir):
                for version_dir in os.listdir(base_dir):
                    version_path = os.path.join(base_dir, version_dir)
                    if os.path.isdir(version_path):
                        for model_subdir in ['portable-models-dir', 'simulator-models-dir']:
                            model_dir = os.path.join(version_path, model_subdir)
                            if os.path.exists(model_dir):
                                for item in os.listdir(model_dir):
                                    if model_id in item:
                                        print(f"Found KIM model {model_id} in {os.path.join(model_dir, item)}")
                                        return True
    
    return False


def install_kim_model(model_id: str) -> bool:
    """
    Install a KIM model using kim-api-collections-management.
    
    Args:
        model_id: KIM model identifier to install
        
    Returns:
        True if installation successful, False otherwise
    """
    try:
        print(f"Installing KIM model: {model_id}")
        print("This may take a moment to download and install...")
        
        result = subprocess.run(
            ['kim-api-collections-management', 'install', 'user', model_id],
            capture_output=True,
            text=True,
            timeout=120  # Give it 120 seconds to install (increased for network delays)
        )
        
        if result.returncode == 0:
            print(f"Installation command completed for KIM model: {model_id}")
            
            # Wait and verify the model is actually available
            max_retries = 10
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                if is_kim_model_installed(model_id):
                    print(f"✅ Successfully verified KIM model: {model_id}")
                    return True
                
                if attempt < max_retries - 1:
                    print(f"Waiting for model to be indexed... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
            
            print(f"⚠️ Installation completed but model not found after {max_retries} attempts")
            return False
        else:
            print(f"Failed to install KIM model {model_id}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout while installing KIM model {model_id} (exceeded 120 seconds)")
        return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error installing KIM model {model_id}: {str(e)}")
        return False


def ensure_kim_models_installed(script_string: str) -> bool:
    """
    Ensure all KIM models referenced in the script are installed.
    Uses the simplified two-step installation process.
    
    Args:
        script_string: LAMMPS input script as a string
        
    Returns:
        True if all models are available, False if any installation failed
    """
    kim_models = extract_kim_models(script_string)
    
    if not kim_models:
        print("No KIM models found in script")
        return True  # No KIM models to install
    
    print(f"Checking KIM models: {kim_models}")
    
    all_available = True
    for model_id in kim_models:
        # Use the simplified installation method (no automatic fallbacks)
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


def check_mpi_available() -> bool:
    """
    Check if MPI is available on the system.
    
    Returns:
        True if MPI is available, False otherwise
    """
    try:
        result = subprocess.run(
            [config.LAMMPS_MPI_COMMAND, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_core_count(num_cores: int) -> int:
    """
    Validate and adjust the number of cores for LAMMPS execution.
    
    Args:
        num_cores: Requested number of cores
        
    Returns:
        Validated number of cores (adjusted if necessary)
        
    Raises:
        ValueError: If core count is invalid
    """
    if num_cores < 1:
        raise ValueError("Number of cores must be at least 1")
    
    if num_cores > 1 and config.LAMMPS_USE_MPI and not check_mpi_available():
        print(f"Warning: MPI not available. Falling back to serial execution.")
        return 1
    
    return num_cores


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """
    Parse LAMMPS log file to extract thermodynamic data.
    
    Args:
        log_path: Path to the LAMMPS log file
        
    Returns:
        List of thermodynamic data dictionaries
    """
    thermo_data = []
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Find the thermodynamic output section
        in_thermo_section = False
        header_line = None
        
        for line in lines:
            line = line.strip()
            
            # Look for thermo_style custom header
            if line.startswith('Step') and any(keyword in line for keyword in ['Temp', 'Press', 'PotEng', 'KinEng', 'TotEng']):
                header_line = line
                in_thermo_section = True
                continue
            
            # If we're in the thermo section and find a data line
            if in_thermo_section and header_line:
                # Skip empty lines and non-numeric lines
                if not line or line.startswith('Loop') or line.startswith('Performance') or line.startswith('WARNING'):
                    in_thermo_section = False
                    continue
                
                # Try to parse the data line
                try:
                    values = line.split()
                    headers = header_line.split()
                    
                    # Make sure we have the same number of values and headers
                    if len(values) == len(headers):
                        data_point = {}
                        for i, header in enumerate(headers):
                            try:
                                # Convert to appropriate type
                                if header.lower() == 'step':
                                    data_point[header.lower()] = int(values[i])
                                else:
                                    data_point[header.lower()] = float(values[i])
                            except ValueError:
                                # If conversion fails, store as string
                                data_point[header.lower()] = values[i]
                        
                        thermo_data.append(data_point)
                except (ValueError, IndexError):
                    # Skip lines that can't be parsed
                    continue
    
    except FileNotFoundError:
        print(f"Warning: Log file {log_path} not found")
    except Exception as e:
        print(f"Warning: Error parsing log file {log_path}: {e}")
    
    return thermo_data


def run_lammps_simulation(script_string: str, num_cores: Optional[int] = None) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Execute a LAMMPS simulation and extract thermodynamic data.
    
    Args:
        script_string: Complete LAMMPS input script as a string
        num_cores: Number of cores to use (defaults to config.LAMMPS_CORES)
        
    Returns:
        Tuple containing:
        - Dictionary of output file paths (e.g., {'dump': 'path/to/dump.atom'})
        - List of thermodynamic data dictionaries
    """
    # Use provided cores or fall back to config
    cores = num_cores if num_cores is not None else config.LAMMPS_CORES
    cores = validate_core_count(cores)
    
    print(f"Running LAMMPS simulation with {cores} core(s)")
    
    # First, ensure all required KIM models are installed
    if not ensure_kim_models_installed(script_string):
        raise RuntimeError("Failed to install required KIM models")
    
    # Create a temporary directory for the simulation
    temp_dir = tempfile.mkdtemp(prefix="lammps_sim_")
    
    try:
        # Write the script to a file
        script_path = os.path.join(temp_dir, "input.lammps")
        with open(script_path, 'w') as f:
            f.write(script_string)
        
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Execute LAMMPS based on core count
        if cores > 1 and config.LAMMPS_USE_MPI:
            # Use MPI execution via subprocess
            cmd = [
                config.LAMMPS_MPI_COMMAND, 
                '-np', str(cores), 
                config.LAMMPS_EXECUTABLE, 
                '-in', script_path
            ]
            
            print(f"Executing MPI command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            
            if result.returncode != 0:
                error_msg = f"LAMMPS MPI execution failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout}"
                raise RuntimeError(error_msg)
            
            print("MPI execution completed successfully")
            
        else:
            # Use Python interface for serial execution
            print("Using Python LAMMPS interface for serial execution")
            lmp = lammps.lammps()
            
            # Execute the LAMMPS script
            lmp.file(script_path)
            
            # Close LAMMPS
            lmp.close()
        
        # Find and copy output files to permanent location
        output_files = {}
        log_file_path = None
        
        for filename in os.listdir(temp_dir):
            if filename.endswith('.atom') or filename.endswith('.dump'):
                # Copy dump files to a permanent location
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['dump'] = permanent_path
            elif filename.startswith('log.') or filename == 'log.lammps':
                # Copy log files and remember the path for parsing
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['log'] = permanent_path
                log_file_path = permanent_path
        
        # Parse thermodynamic data from the log file
        thermo_data = []
        if log_file_path:
            thermo_data = parse_log_file(log_file_path)
        
        # If no thermo data was captured, create a minimal dataset
        if not thermo_data:
            thermo_data = [{"step": 0, "note": "No thermodynamic data captured"}]
        
        return output_files, thermo_data
        
    finally:
        # Clean up: change back to original directory and remove temp directory
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


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
        
        # Make sure we have the same number of values and headers
        if len(values) == len(headers):
            data_point = {}
            for i, header in enumerate(headers):
                try:
                    # Convert to appropriate type
                    if header.lower() == 'step':
                        data_point[header.lower()] = int(values[i])
                    else:
                        data_point[header.lower()] = float(values[i])
                except ValueError:
                    # If conversion fails, store as string
                    data_point[header.lower()] = values[i]
            
            return data_point
    except (ValueError, IndexError):
        pass
    
    return None


def run_lammps_simulation_streaming(script_string: str, data_callback: Callable[[Dict[str, Any]], None], 
                                  num_cores: Optional[int] = None) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Execute a LAMMPS simulation with real-time thermodynamic data streaming.
    
    Args:
        script_string: Complete LAMMPS input script as a string
        data_callback: Function to call with each new thermodynamic data point
        num_cores: Number of cores to use (defaults to config.LAMMPS_CORES)
        
    Returns:
        Tuple containing:
        - Dictionary of output file paths (e.g., {'dump': 'path/to/dump.atom'})
        - List of all thermodynamic data dictionaries
    """
    # Use provided cores or fall back to config
    cores = num_cores if num_cores is not None else config.LAMMPS_CORES
    cores = validate_core_count(cores)
    
    print(f"Running streaming LAMMPS simulation with {cores} core(s)")
    
    # First, ensure all required KIM models are installed
    if not ensure_kim_models_installed(script_string):
        raise RuntimeError("Failed to install required KIM models")
    
    # Create a temporary directory for the simulation
    temp_dir = tempfile.mkdtemp(prefix="lammps_sim_")
    thermo_data = []
    
    try:
        # Write the script to a file
        script_path = os.path.join(temp_dir, "input.lammps")
        with open(script_path, 'w') as f:
            f.write(script_string)
        
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Execute LAMMPS with streaming output
        if cores > 1 and config.LAMMPS_USE_MPI:
            # Use MPI execution via subprocess with streaming
            cmd = [
                config.LAMMPS_MPI_COMMAND, 
                '-np', str(cores), 
                config.LAMMPS_EXECUTABLE, 
                '-in', script_path
            ]
            
            print(f"Executing streaming MPI command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Stream output line by line
            headers = None
            in_thermo_section = False
            
            for line in process.stdout:
                line = line.strip()
                
                # Look for thermo_style custom header
                if line.startswith('Step') and any(keyword in line for keyword in ['Temp', 'Press', 'PotEng', 'KinEng', 'TotEng']):
                    headers = line.split()
                    in_thermo_section = True
                    continue
                
                # If we're in the thermo section and have headers
                if in_thermo_section and headers:
                    # Skip empty lines and non-numeric lines
                    if not line or line.startswith('Loop') or line.startswith('Performance') or line.startswith('WARNING'):
                        in_thermo_section = False
                        continue
                    
                    # Try to parse the data line
                    data_point = parse_thermo_line(line, headers)
                    if data_point:
                        thermo_data.append(data_point)
                        # Call the callback with the new data point
                        try:
                            data_callback(data_point)
                        except Exception as e:
                            print(f"Warning: Error in data callback: {e}")
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                error_msg = f"LAMMPS MPI execution failed with return code {process.returncode}"
                if stderr_output:
                    error_msg += f"\nStderr: {stderr_output}"
                raise RuntimeError(error_msg)
            
            print("Streaming MPI execution completed successfully")
            
        else:
            # For serial execution, we'll use a different approach
            # Since the Python LAMMPS interface doesn't easily support streaming,
            # we'll fall back to the regular execution and then parse the log
            print("Using Python LAMMPS interface for serial execution (no streaming)")
            lmp = lammps.lammps()
            
            # Execute the LAMMPS script
            lmp.file(script_path)
            
            # Close LAMMPS
            lmp.close()
            
            # Parse the log file and call callback for each data point
            log_path = os.path.join(temp_dir, 'log.lammps')
            if os.path.exists(log_path):
                thermo_data = parse_log_file(log_path)
                for data_point in thermo_data:
                    try:
                        data_callback(data_point)
                        # Add a small delay to simulate streaming
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Warning: Error in data callback: {e}")
        
        # Find and copy output files to permanent location
        output_files = {}
        log_file_path = None
        
        for filename in os.listdir(temp_dir):
            if filename.endswith('.atom') or filename.endswith('.dump'):
                # Copy dump files to a permanent location
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['dump'] = permanent_path
            elif filename.startswith('log.') or filename == 'log.lammps':
                # Copy log files and remember the path for parsing
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['log'] = permanent_path
                log_file_path = permanent_path
        
        # If no thermo data was captured, create a minimal dataset
        if not thermo_data:
            thermo_data = [{"step": 0, "note": "No thermodynamic data captured"}]
        
        return output_files, thermo_data
        
    finally:
        # Clean up: change back to original directory and remove temp directory
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


def list_available_kim_models() -> List[str]:
    """
    List all available KIM models on the system.
    
    Returns:
        List of available KIM model IDs
    """
    available_models = []
    
    # Try using kim-api-collections-management first
    try:
        result = subprocess.run(
            ['kim-api-collections-management', 'list', '--installed'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse the output to extract model names
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '__MO_' in line:
                    available_models.append(line)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Also check directories manually
    kim_dirs = [
        os.path.expanduser("~/.kim-api"),
        "/usr/local/lib/openkim-models",
        "/opt/openkim-models"
    ]
    
    # Add conda environment if available
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        kim_dirs.append(os.path.join(conda_prefix, 'lib', 'openkim-models'))
    
    # Search for models in directories
    for base_dir in kim_dirs:
        for subdir in ['portable-models', 'simulator-models']:
            search_dir = os.path.join(base_dir, subdir)
            if os.path.exists(search_dir):
                for item in os.listdir(search_dir):
                    if '__MO_' in item and item not in available_models:
                        available_models.append(item)
        
        # Also check versioned directories
        if os.path.exists(base_dir):
            for version_dir in os.listdir(base_dir):
                version_path = os.path.join(base_dir, version_dir)
                if os.path.isdir(version_path):
                    for model_subdir in ['portable-models-dir', 'simulator-models-dir']:
                        model_dir = os.path.join(version_path, model_subdir)
                        if os.path.exists(model_dir):
                            for item in os.listdir(model_dir):
                                if '__MO_' in item and item not in available_models:
                                    available_models.append(item)
    
    return sorted(list(set(available_models)))


def diagnose_kim_setup() -> None:
    """
    Print diagnostic information about KIM setup.
    """
    print("=== KIM Diagnostic Information ===")
    
    # Check if kim-api-collections-management is available
    try:
        result = subprocess.run(
            ['kim-api-collections-management', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ kim-api-collections-management is available")
        else:
            print("⚠️ kim-api-collections-management returned non-zero exit code")
    except FileNotFoundError:
        print("❌ kim-api-collections-management not found in PATH")
    except Exception as e:
        print(f"❌ Error checking kim-api-collections-management: {e}")
    
    # Check KIM directories
    print("\n--- KIM Directories ---")
    kim_dirs = [
        os.path.expanduser("~/.kim-api"),
        "/usr/local/lib/openkim-models",
        "/opt/openkim-models"
    ]
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        kim_dirs.append(os.path.join(conda_prefix, 'lib', 'openkim-models'))
    
    for kim_dir in kim_dirs:
        if os.path.exists(kim_dir):
            print(f"✅ {kim_dir} exists")
        else:
            print(f"❌ {kim_dir} does not exist")
    
    # List available models
    print("\n--- Available KIM Models ---")
    models = list_available_kim_models()
    if models:
        print(f"Found {len(models)} KIM models:")
        for model in models[:10]:  # Show first 10
            print(f"  - {model}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
    else:
        print("❌ No KIM models found")
    
    print("=== End KIM Diagnostic ===\n")


def validate_lammps_script(script_string: str) -> bool:
    """
    Validate a LAMMPS script by checking for basic syntax and required commands.
    
    Args:
        script_string: LAMMPS input script as a string
        
    Returns:
        True if script appears valid, False otherwise
    """
    # Basic validation checks
    script_lower = script_string.lower()
    
    # Check for essential commands
    required_commands = ['units', 'dimension', 'boundary']
    for cmd in required_commands:
        if cmd not in script_lower:
            return False
    
    # Check for either kim command or pair_style command
    if 'kim' not in script_lower and 'pair_style' not in script_lower:
        return False
    
    return True


def extract_lammps_frame(lmp):
    """
    Extract current frame data from LAMMPS instance.
    
    Args:
        lmp: LAMMPS instance
        
    Returns:
        dict: Contains positions, types, box, velocities, forces
    """
    try:
        # Get number of atoms
        natoms = lmp.get_natoms()
        
        if natoms == 0:
            return None
        
        # Extract atomic positions (3N array)
        positions = lmp.gather_atoms("x", 1, 3)
        positions = np.array(positions).reshape((natoms, 3))
        
        # Extract atom types
        types = lmp.gather_atoms("type", 0, 1)
        types = np.array(types, dtype=int)
        
        # Extract simulation box
        box_data = lmp.extract_box()
        # box_data format: [boxlo, boxhi] for each dimension
        box = {
            'xlo': box_data[0][0], 'xhi': box_data[1][0],
            'ylo': box_data[0][1], 'yhi': box_data[1][1], 
            'zlo': box_data[0][2], 'zhi': box_data[1][2]
        }
        
        # Try to extract velocities (optional)
        velocities = None
        try:
            vel_data = lmp.gather_atoms("v", 1, 3)
            velocities = np.array(vel_data).reshape((natoms, 3))
        except:
            pass
        
        # Try to extract forces (optional)
        forces = None
        try:
            force_data = lmp.gather_atoms("f", 1, 3)
            forces = np.array(force_data).reshape((natoms, 3))
        except:
            pass
        
        frame_data = {
            'natoms': natoms,
            'positions': positions,
            'types': types,
            'box': box,
            'velocities': velocities,
            'forces': forces
        }
        
        return frame_data
        
    except Exception as e:
        print(f"Error extracting LAMMPS frame: {e}")
        return None


def create_ovito_frame(frame_data):
    """
    Create OVITO DataCollection from LAMMPS frame data.
    
    Args:
        frame_data: Dict from extract_lammps_frame()
        
    Returns:
        DataCollection: Ready for rendering
    """
    try:
        from ovito.data import DataCollection, Particles, SimulationCell
        import numpy as np
        
        if frame_data is None:
            return None
        
        # Create OVITO data collection
        data = DataCollection()
        
        # Create particles
        particles = data.create_particles()
        particles.count = frame_data['natoms']
        
        # Set positions
        particles.positions = frame_data['positions']
        
        # Set particle types
        particles.particle_types = frame_data['types']
        
        # Set velocities if available
        if frame_data['velocities'] is not None:
            particles.velocities = frame_data['velocities']
        
        # Set forces if available
        if frame_data['forces'] is not None:
            particles.forces = frame_data['forces']
        
        # Create simulation cell with matrix
        box = frame_data['box']
        
        # Create cell matrix (3x4 matrix: 3x3 for vectors + 1 column for origin)
        cell_matrix = np.array([
            [box['xhi'] - box['xlo'], 0, 0, box['xlo']],
            [0, box['yhi'] - box['ylo'], 0, box['ylo']],
            [0, 0, box['zhi'] - box['zlo'], box['zlo']]
        ])
        
        # Pass matrix directly to create_cell
        cell = data.create_cell(matrix=cell_matrix)
        
        # Set periodic boundary conditions
        cell.pbc = (True, True, True)
        
        return data
        
    except Exception as e:
        print(f"Error creating OVITO frame: {e}")
        return None


def render_frame_to_image(data_collection, settings=None):
    """
    Render OVITO DataCollection to base64 image using thread-safe TachyonRenderer.
    
    Args:
        data_collection: OVITO data
        settings: Rendering parameters (camera, colors, etc.)
        
    Returns:
        str: Base64-encoded PNG image
    """
    try:
        from ovito.vis import Viewport, TachyonRenderer
        from ovito.modifiers import ColorCodingModifier, AssignColorModifier
        from ovito.pipeline import StaticSource, Pipeline
        import tempfile
        import base64
        import os
        import threading
        import platform
        
        if data_collection is None:
            return None
        
        # Create pipeline from data
        pipeline = Pipeline(source=StaticSource(data=data_collection))
        
        # Apply color coding modifier (color by particle type)
        color_modifier = ColorCodingModifier(
            property='Particle Type',
            gradient=ColorCodingModifier.Hot()
        )
        pipeline.modifiers.append(color_modifier)
        
        # Add pipeline to scene (this is the correct way)
        pipeline.add_to_scene()
        
        # Create viewport for rendering
        viewport = Viewport()
        viewport.type = Viewport.Type.Perspective
        
        # Set camera position and orientation
        if settings and 'camera_angle' in settings:
            if settings['camera_angle'] == 'top':
                viewport.camera_pos = (0, 0, 50)
                viewport.camera_dir = (0, 0, -1)
            elif settings['camera_angle'] == 'side':
                viewport.camera_pos = (50, 0, 0)
                viewport.camera_dir = (-1, 0, 0)
            else:  # default perspective
                viewport.camera_pos = (30, 30, 30)
                viewport.camera_dir = (-1, -1, -1)
        else:
            # Default camera position
            viewport.camera_pos = (30, 30, 30)
            viewport.camera_dir = (-1, -1, -1)
        
        # Determine render size based on quality
        quality = settings.get('quality', 'medium') if settings else 'medium'
        if quality == 'high':
            render_size = (800, 600)
        elif quality == 'low':
            render_size = (400, 300)
        else:  # medium
            render_size = (600, 450)
        
        # Zoom to fit all particles with the render size
        viewport.zoom_all(size=render_size)
        
        # Render to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Use TachyonRenderer to avoid NSWindow threading issues on macOS
            # TachyonRenderer is a software ray-tracer that doesn't require OpenGL/GUI
            renderer = TachyonRenderer()
            
            # Configure TachyonRenderer settings
            renderer.antialiasing = True
            renderer.direct_light_intensity = 0.9
            renderer.shadows = True
            renderer.ambient_occlusion = True
            renderer.depth_of_field = False  # Disable for cleaner images
            
            # Check if we're running in a non-main thread (common in Flask/SocketIO)
            is_main_thread = threading.current_thread() is threading.main_thread()
            is_macos = platform.system() == 'Darwin'
            
            if not is_main_thread and is_macos:
                print("Detected non-main thread on macOS - using TachyonRenderer for thread safety")
            
            # Render the image with TachyonRenderer
            viewport.render_image(
                filename=tmp_path, 
                size=render_size, 
                alpha=True,
                renderer=renderer  # Explicitly use TachyonRenderer
            )
            
            # Read and encode as base64
            with open(tmp_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode('utf-8')
            
            return base64_image
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        print(f"Error rendering frame to image: {e}")
        # Print more detailed error information for debugging
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None


import queue
from concurrent.futures import ThreadPoolExecutor


class FrameQueue:
    """Thread-safe queue for frame data with size limits."""
    
    def __init__(self, max_size=10):
        self.queue = queue.Queue(maxsize=max_size)
        self.stopped = False
    
    def put(self, item, timeout=1.0):
        """Put item in queue with timeout to prevent blocking."""
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            print("Warning: Frame queue full, skipping frame")
            return False
    
    def get(self, timeout=1.0):
        """Get item from queue with timeout."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Signal that no more items will be added."""
        self.stopped = True
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.queue.empty()


def extract_thermo_data_realtime(lmp):
    """
    Extract real-time thermodynamic data from LAMMPS instance.
    
    Args:
        lmp: LAMMPS instance
        
    Returns:
        dict: Current thermodynamic data
    """
    try:
        # Get current timestep
        timestep = lmp.extract_global("ntimestep", 0)
        
        # Extract thermodynamic quantities
        thermo_data = {'step': timestep}
        
        # Try to extract common thermodynamic properties
        try:
            # Temperature
            temp = lmp.extract_compute("thermo_temp", 0, 0)
            if temp is not None:
                thermo_data['temp'] = temp
        except:
            pass
        
        try:
            # Pressure
            press = lmp.extract_compute("thermo_press", 0, 0)
            if press is not None:
                thermo_data['press'] = press
        except:
            pass
        
        try:
            # Potential energy
            pe = lmp.extract_compute("thermo_pe", 0, 0)
            if pe is not None:
                thermo_data['poteng'] = pe
        except:
            pass
        
        try:
            # Kinetic energy
            ke = lmp.extract_compute("thermo_ke", 0, 0)
            if ke is not None:
                thermo_data['kineng'] = ke
                # Total energy
                if 'poteng' in thermo_data:
                    thermo_data['toteng'] = ke + thermo_data['poteng']
        except:
            pass
        
        return thermo_data
        
    except Exception as e:
        print(f"Warning: Error extracting real-time thermo data: {e}")
        return {'step': 0, 'note': 'Error extracting data'}


def calculate_smart_dump_interval(total_steps, user_interval=None, max_frames=100):
    """
    Calculate smart dump interval to capture full simulation within frame limit.
    Ensures first and last frames are always included.
    
    Args:
        total_steps: Total number of simulation steps
        user_interval: User-specified interval (overrides calculation)
        max_frames: Maximum number of frames to capture (default: 100)
        
    Returns:
        int: Optimal dump interval
    """
    if user_interval is not None:
        return max(1, user_interval)
    
    # If simulation is shorter than max frames, capture every step
    if total_steps <= max_frames:
        smart_interval = 1
        frames_estimate = total_steps
    else:
        # Calculate interval to get exactly max_frames distributed across the simulation
        # Reserve 2 frames for first and last, distribute remaining frames uniformly
        smart_interval = max(1, total_steps // (max_frames - 1))
        frames_estimate = min(max_frames, (total_steps // smart_interval) + 1)
    
    print(f"Smart dump interval: {smart_interval} steps for {total_steps} total steps (~{frames_estimate} frames, max: {max_frames})")
    return smart_interval


def run_lammps_with_async_visualization(script_string: str, render_config=None, data_callback=None, 
                                       num_cores: Optional[int] = None):
    """
    Execute LAMMPS with asynchronous visualization using producer-consumer pattern.
    
    Args:
        script_string: Complete LAMMPS input script as a string
        render_config: {
            'interval': None,  # Auto-calculate if None
            'quality': 'medium',  # low/medium/high
            'camera_angle': 'default',
            'max_queue_size': 10  # Maximum frames in render queue
        }
        data_callback: Function to call with thermodynamic and visualization data
        num_cores: Number of cores to use (defaults to config.LAMMPS_CORES)
        
    Returns:
        Tuple containing:
        - Dictionary of output file paths
        - List of all thermodynamic data dictionaries
        - List of rendered frame images (base64)
    """
    import numpy as np
    
    # Use provided cores or fall back to config
    cores = num_cores if num_cores is not None else config.LAMMPS_CORES
    cores = validate_core_count(cores)
    
    print(f"Running LAMMPS with async visualization using {cores} core(s)")
    
    # Check if we should use MPI
    use_mpi = cores > 1 and config.LAMMPS_USE_MPI and check_mpi_available()
    if cores > 1 and not use_mpi:
        print(f"Warning: MPI not available or disabled. Falling back to serial execution.")
        cores = 1
    
    # First, ensure all required KIM models are installed
    if not ensure_kim_models_installed(script_string):
        raise RuntimeError("Failed to install required KIM models")
    
    # Set default render config
    if render_config is None:
        render_config = {
            'interval': None,  # Will be auto-calculated
            'quality': 'medium',
            'camera_angle': 'default',
            'max_queue_size': 10
        }
    
    # Create frame queue for async rendering
    frame_queue = FrameQueue(max_size=render_config.get('max_queue_size', 10))
    
    # Shared data structures
    thermo_data = []
    frame_images = []
    simulation_complete = threading.Event()
    
    def rendering_worker():
        """Worker thread that renders frames asynchronously."""
        print("Starting rendering worker thread")
        
        # Import OVITO in the worker thread to avoid threading issues
        try:
            from ovito.data import DataCollection, Particles, SimulationCell
            from ovito.vis import Viewport, TachyonRenderer
            from ovito.modifiers import ColorCodingModifier
            from ovito.pipeline import StaticSource, Pipeline
        except ImportError as e:
            print(f"Error importing OVITO in worker thread: {e}")
            return
        
        while not simulation_complete.is_set() or not frame_queue.is_empty():
            frame_data = frame_queue.get(timeout=0.5)
            
            if frame_data is None:
                continue
            
            try:
                # Create OVITO data and render in the worker thread
                ovito_data = create_ovito_frame(frame_data['data'])
                if ovito_data:
                    image_b64 = render_frame_to_image(ovito_data, render_config)
                    if image_b64:
                        frame_info = {
                            'image': image_b64,
                            'frame_number': frame_data['frame_number'],
                            'timestep': frame_data['timestep']
                        }
                        frame_images.append(frame_info)
                        
                        # Stream frame via callback if provided
                        if data_callback:
                            try:
                                data_callback({
                                    'type': 'trajectory_frame',
                                    'image_data': image_b64,
                                    'frame_number': frame_data['frame_number'],
                                    'timestep': frame_data['timestep']
                                })
                            except Exception as e:
                                print(f"Warning: Error in visualization callback: {e}")
                
            except Exception as e:
                print(f"Warning: Error rendering frame: {e}")
                # Continue processing other frames even if one fails
                continue
        
        print("Rendering worker thread completed")
    
    # Create a temporary directory for the simulation
    temp_dir = tempfile.mkdtemp(prefix="lammps_async_viz_")
    
    try:
        # Write the script to a file
        script_path = os.path.join(temp_dir, "input.lammps")
        with open(script_path, 'w') as f:
            f.write(script_string)
        
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Choose execution method based on core count
        if use_mpi:
            print(f"Using MPI execution with {cores} cores for async visualization")
            # We'll use subprocess with MPI but need to handle visualization differently
            # For now, fall back to Python interface for visualization
            # TODO: Implement MPI-based visualization extraction
            print("Note: MPI visualization extraction not yet implemented, using serial Python interface")
            use_mpi = False
        
        if not use_mpi:
            print("Using Python LAMMPS interface for async visualization")
            lmp = lammps.lammps()
        
        # Start rendering worker thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            render_future = executor.submit(rendering_worker)
            
            # Parse script to identify run commands and calculate intervals
            script_lines = script_string.split('\n')
            modified_script = []
            frame_counter = 0
            MAX_FRAMES = 100  # Hard limit on total frames
            
            for line in script_lines:
                line = line.strip()
                if line.startswith('run '):
                    # Extract number of steps
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            total_steps = int(parts[1])
                            
                            # Calculate smart interval with max_frames limit
                            interval = calculate_smart_dump_interval(
                                total_steps, 
                                render_config.get('interval'),
                                max_frames=MAX_FRAMES
                            )
                            
                            print(f"Using visualization interval: {interval} steps (max {MAX_FRAMES} frames)")
                            
                            # Run in chunks for visualization and thermo data
                            steps_done = 0
                            while steps_done < total_steps:
                                chunk_size = min(interval, total_steps - steps_done)
                                
                                # Execute this chunk
                                modified_script.append(f"run {chunk_size}")
                                
                                # Execute accumulated commands
                                for cmd in modified_script:
                                    if cmd.strip():
                                        lmp.command(cmd)
                                modified_script = []  # Clear for next chunk
                                
                                steps_done += chunk_size
                                
                                # Extract and stream thermodynamic data immediately
                                thermo_point = extract_thermo_data_realtime(lmp)
                                thermo_point['step'] = steps_done
                                thermo_data.append(thermo_point)
                                
                                # Stream thermo data via callback
                                if data_callback:
                                    try:
                                        data_callback(thermo_point)
                                    except Exception as e:
                                        print(f"Warning: Error in thermo callback: {e}")
                                
                                # Extract frame data and queue for async rendering
                                # IMPORTANT: Enforce hard limit of MAX_FRAMES
                                if frame_counter < MAX_FRAMES:
                                    frame_data = extract_lammps_frame(lmp)
                                    if frame_data:
                                        frame_info = {
                                            'data': frame_data,
                                            'frame_number': frame_counter,
                                            'timestep': steps_done
                                        }
                                        
                                        # Try to queue frame (non-blocking)
                                        if frame_queue.put(frame_info, timeout=0.1):
                                            frame_counter += 1
                                            if frame_counter == MAX_FRAMES:
                                                print(f"Reached maximum frame limit ({MAX_FRAMES} frames)")
                                        else:
                                            print(f"Skipped frame at step {steps_done} (queue full)")
                                elif frame_counter == MAX_FRAMES and steps_done == total_steps:
                                    # Always capture the final frame if we haven't exceeded the limit
                                    frame_data = extract_lammps_frame(lmp)
                                    if frame_data:
                                        frame_info = {
                                            'data': frame_data,
                                            'frame_number': frame_counter,
                                            'timestep': steps_done
                                        }
                                        # Replace the last frame with the final frame to ensure we capture the end
                                        if frame_queue.put(frame_info, timeout=0.1):
                                            print(f"Captured final frame at step {steps_done}")
                            
                            continue  # Skip adding the original run command
                        except ValueError:
                            pass
                
                # Add non-run commands to the script
                modified_script.append(line)
            
            # Execute any remaining commands
            for cmd in modified_script:
                if cmd.strip():
                    lmp.command(cmd)
            
            # Signal simulation completion
            simulation_complete.set()
            frame_queue.stop()
            
            # Wait for rendering to complete
            render_future.result(timeout=30)  # 30 second timeout
        
        # Close LAMMPS
        lmp.close()
        
        # Find and copy output files to permanent location
        output_files = {}
        log_file_path = None
        
        for filename in os.listdir(temp_dir):
            if filename.endswith('.atom') or filename.endswith('.dump'):
                # Copy dump files to a permanent location
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['dump'] = permanent_path
            elif filename.startswith('log.') or filename == 'log.lammps':
                # Copy log files and remember the path for parsing
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['log'] = permanent_path
                log_file_path = permanent_path
        
        # Parse additional thermodynamic data from log file if needed
        if log_file_path and not thermo_data:
            thermo_data = parse_log_file(log_file_path)
        
        # If no thermo data was captured, create a minimal dataset
        if not thermo_data:
            thermo_data = [{"step": 0, "note": "No thermodynamic data captured"}]
        
        # Signal completion via callback if provided
        if data_callback:
            try:
                data_callback({
                    'type': 'trajectory_complete',
                    'total_frames': len(frame_images),
                    'message': f'Async visualization complete: {len(frame_images)} frames rendered'
                })
            except Exception as e:
                print(f"Warning: Error in completion callback: {e}")
        
        print(f"Async visualization completed: {len(frame_images)} frames, {len(thermo_data)} thermo points")
        return output_files, thermo_data, frame_images
        
    finally:
        # Clean up: change back to original directory and remove temp directory
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_lammps_with_visualization(script_string: str, render_config=None, data_callback=None, 
                                 num_cores: Optional[int] = None):
    """
    Execute LAMMPS with live visualization using direct API integration.
    
    DEPRECATED: Use run_lammps_with_async_visualization for better performance.
    
    Args:
        script_string: Complete LAMMPS input script as a string
        render_config: {
            'interval': 100,  # Render every N steps
            'quality': 'medium',  # low/medium/high
            'camera_angle': 'default',
            'color_scheme': 'element'
        }
        data_callback: Function to call with each new thermodynamic data point
        num_cores: Number of cores to use (defaults to config.LAMMPS_CORES)
        
    Returns:
        Tuple containing:
        - Dictionary of output file paths
        - List of all thermodynamic data dictionaries
        - List of rendered frame images (base64)
    """
    print("Warning: Using deprecated synchronous visualization. Consider using run_lammps_with_async_visualization for better performance.")
    
    # Delegate to async version for better performance
    return run_lammps_with_async_visualization(script_string, render_config, data_callback, num_cores)
