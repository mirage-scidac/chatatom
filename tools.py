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
    matches = re.findall(kim_init_pattern, script_string, re.IGNORECASE)
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
        result = subprocess.run(
            ['kim-api-collections-management', 'install', 'user', model_id],
            capture_output=True,
            text=True,
            timeout=60  # Give it 60 seconds to install
        )
        
        if result.returncode == 0:
            print(f"Successfully installed KIM model: {model_id}")
            return True
        else:
            print(f"Failed to install KIM model {model_id}: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error installing KIM model {model_id}: {str(e)}")
        return False


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
        return True  # No KIM models to install
    
    print(f"Checking KIM models: {kim_models}")
    
    all_available = True
    for model_id in kim_models:
        if is_kim_model_installed(model_id):
            print(f"✅ KIM model {model_id} is already installed")
        else:
            print(f"⚠️ KIM model {model_id} not found, attempting to install...")
            if install_kim_model(model_id):
                print(f"✅ Successfully installed KIM model {model_id}")
            else:
                print(f"❌ Failed to install KIM model {model_id}")
                # Check one more time if it's available (maybe it was already there)
                if is_kim_model_installed(model_id):
                    print(f"✅ KIM model {model_id} is actually available despite installation failure")
                else:
                    print(f"❌ KIM model {model_id} is definitely not available")
                    all_available = False
    
    if all_available:
        print("✅ All required KIM models are available")
    else:
        print("❌ Some KIM models are missing and could not be installed")
    
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
