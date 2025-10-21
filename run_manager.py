"""
Run folder management for organizing LAMMPS simulation outputs.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# Base directory for all simulation runs
RUNS_BASE_DIR = "runs"


def create_run_folder() -> str:
    """
    Create a timestamped run folder for organizing simulation outputs.
    
    Returns:
        str: Absolute path to the created run folder
    """
    # Ensure base runs directory exists
    os.makedirs(RUNS_BASE_DIR, exist_ok=True)
    
    # Create timestamped folder name
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_folder_name = f"run_{timestamp}"
    run_folder_path = os.path.join(RUNS_BASE_DIR, run_folder_name)
    
    # Create the run folder
    os.makedirs(run_folder_path, exist_ok=True)
    
    # Return absolute path
    return os.path.abspath(run_folder_path)


def save_lammps_script(run_folder: str, script_content: str) -> str:
    """
    Save LAMMPS input script to the run folder.
    
    Args:
        run_folder: Path to the run folder
        script_content: LAMMPS script content
        
    Returns:
        str: Path to the saved script file
    """
    script_path = os.path.join(run_folder, "input.lammps")
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path


def save_ovito_script(run_folder: str, script_content: str) -> str:
    """
    Save OVITO visualization script to the run folder.
    
    Args:
        run_folder: Path to the run folder
        script_content: OVITO script content
        
    Returns:
        str: Path to the saved script file
    """
    script_path = os.path.join(run_folder, "visualization.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path


def save_simulation_metadata(run_folder: str, metadata: dict) -> str:
    """
    Save simulation metadata as JSON.
    
    Args:
        run_folder: Path to the run folder
        metadata: Dictionary containing simulation metadata
        
    Returns:
        str: Path to the saved metadata file
    """
    metadata_path = os.path.join(run_folder, "simulation_info.json")
    
    # Add timestamp if not present
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().isoformat()
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def update_simulation_status(run_folder: str, status: str, **kwargs) -> None:
    """
    Update simulation status in metadata file.
    
    Args:
        run_folder: Path to the run folder
        status: Status string (e.g., 'starting', 'running', 'completed', 'failed')
        **kwargs: Additional metadata to update
    """
    metadata_path = os.path.join(run_folder, "simulation_info.json")
    
    # Load existing metadata or create new
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update status and additional fields
    metadata['status'] = status
    metadata['last_updated'] = datetime.now().isoformat()
    metadata.update(kwargs)
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_run_folder_summary(run_folder: str) -> dict:
    """
    Get summary information about a run folder.
    
    Args:
        run_folder: Path to the run folder
        
    Returns:
        dict: Summary information including metadata and file list
    """
    summary = {
        'path': run_folder,
        'exists': os.path.exists(run_folder),
        'files': [],
        'metadata': {}
    }
    
    if not os.path.exists(run_folder):
        return summary
    
    # List all files in the run folder
    summary['files'] = os.listdir(run_folder)
    
    # Load metadata if available
    metadata_path = os.path.join(run_folder, "simulation_info.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            summary['metadata'] = json.load(f)
    
    return summary


def list_all_runs() -> list:
    """
    List all run folders in the runs directory.
    
    Returns:
        list: List of run folder names sorted by timestamp (newest first)
    """
    if not os.path.exists(RUNS_BASE_DIR):
        return []
    
    runs = [d for d in os.listdir(RUNS_BASE_DIR) 
            if os.path.isdir(os.path.join(RUNS_BASE_DIR, d)) and d.startswith('run_')]
    
    # Sort by timestamp (newest first)
    runs.sort(reverse=True)
    
    return runs


def get_latest_run() -> Optional[str]:
    """
    Get the path to the most recent run folder.
    
    Returns:
        Optional[str]: Absolute path to the latest run folder, or None if no runs exist
    """
    runs = list_all_runs()
    if not runs:
        return None
    
    return os.path.abspath(os.path.join(RUNS_BASE_DIR, runs[0]))
