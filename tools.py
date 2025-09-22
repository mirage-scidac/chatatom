"""
Tools for executing LAMMPS simulations and extracting data.
"""
import os
import tempfile
import shutil
import subprocess
import re
from typing import Dict, List, Any, Tuple
import lammps


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
    Check if a KIM model is installed.
    
    Args:
        model_id: KIM model identifier
        
    Returns:
        True if model is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ['kim-api-collections-management', 'list', '--installed'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return model_id in result.stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # If kim-api-collections-management is not available, assume model is not installed
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
        return True  # No KIM models to install
    
    all_installed = True
    for model_id in kim_models:
        if not is_kim_model_installed(model_id):
            print(f"KIM model {model_id} not found, attempting to install...")
            if not install_kim_model(model_id):
                all_installed = False
        else:
            print(f"KIM model {model_id} is already installed")
    
    return all_installed


def run_lammps_simulation(script_string: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Execute a LAMMPS simulation and extract thermodynamic data.
    
    Args:
        script_string: Complete LAMMPS input script as a string
        
    Returns:
        Tuple containing:
        - Dictionary of output file paths (e.g., {'dump': 'path/to/dump.atom'})
        - List of thermodynamic data dictionaries
    """
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
        
        # Initialize LAMMPS
        lmp = lammps.lammps()
        
        # Storage for thermodynamic data
        thermo_data = []
        
        # Define a callback to capture thermodynamic output
        def thermo_callback():
            """Callback function to capture thermodynamic data"""
            try:
                # Get current step
                step = lmp.get_thermo("step")
                
                # Get common thermodynamic quantities
                data_point = {"step": step}
                
                # Try to get common thermo quantities, skip if not available
                thermo_quantities = ["temp", "press", "pe", "ke", "etotal", "vol", "density"]
                for quantity in thermo_quantities:
                    try:
                        value = lmp.get_thermo(quantity)
                        data_point[quantity] = value
                    except:
                        # Skip quantities that aren't available
                        pass
                
                thermo_data.append(data_point)
            except Exception as e:
                # If we can't get thermo data, just skip this step
                pass
        
        # Set up the callback (this is a simplified approach)
        # Note: LAMMPS Python interface callback setup can be complex
        # For now, we'll run the simulation and extract data afterwards
        
        # Execute the LAMMPS script
        lmp.file(script_path)
        
        # Try to extract final thermodynamic state
        try:
            final_data = {}
            final_data["step"] = lmp.get_thermo("step")
            
            thermo_quantities = ["temp", "press", "pe", "ke", "etotal", "vol", "density"]
            for quantity in thermo_quantities:
                try:
                    value = lmp.get_thermo(quantity)
                    final_data[quantity] = value
                except:
                    pass
            
            if final_data:
                thermo_data.append(final_data)
                
        except Exception as e:
            # If we can't extract final data, that's okay
            pass
        
        # Close LAMMPS
        lmp.close()
        
        # Find output files in the temporary directory
        output_files = {}
        for filename in os.listdir(temp_dir):
            if filename.endswith('.atom') or filename.endswith('.dump'):
                # Copy dump files to a permanent location
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['dump'] = permanent_path
            elif filename.startswith('log.'):
                # Copy log files
                permanent_path = os.path.join(original_dir, f"output_{filename}")
                shutil.copy2(os.path.join(temp_dir, filename), permanent_path)
                output_files['log'] = permanent_path
        
        # If no thermo data was captured, create a minimal dataset
        if not thermo_data:
            thermo_data = [{"step": 0, "note": "No thermodynamic data captured"}]
        
        return output_files, thermo_data
        
    finally:
        # Clean up: change back to original directory and remove temp directory
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


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
