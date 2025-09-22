"""
Tools for executing LAMMPS simulations and extracting data.
"""
import os
import tempfile
import shutil
from typing import Dict, List, Any, Tuple
import lammps


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
