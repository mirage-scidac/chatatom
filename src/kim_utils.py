# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
KIM Model Discovery and Management Utilities

This module provides robust functions for discovering, validating, and installing
KIM (Knowledgebase of Interatomic Models) models for LAMMPS simulations.
"""

import subprocess
import time
import logging
from typing import List, Dict, Optional, Tuple
import requests
import os
from datetime import datetime
import re
import sys

# Define absolute path to kim-api-collections-management
# Assumes it is in the same directory as the python executable
KIM_API_CMD = os.path.join(os.path.dirname(sys.executable), "kim-api-collections-management")

logger = logging.getLogger(__name__)

# Path to the registry file
REGISTRY_FILE = "kim_potentials_registry.md"

# Recommended potentials for auto-download on first startup
RECOMMENDED_POTENTIALS = [
    "EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006",
    "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
    "EAM_Dynamo_AcklandTichyVitek_1987_Ni__MO_977363131043_006"
]

# Element properties for LAMMPS simulations
ELEMENT_PROPERTIES = {
    "Ar": {"mass": 39.948, "lattice": "fcc", "lattice_constant": 5.26},
    "Cu": {"mass": 63.546, "lattice": "fcc", "lattice_constant": 3.615},
    "Al": {"mass": 26.9815, "lattice": "fcc", "lattice_constant": 4.05},
    "Si": {"mass": 28.0855, "lattice": "diamond", "lattice_constant": 5.43},
    "Ni": {"mass": 58.6934, "lattice": "fcc", "lattice_constant": 3.52},
    "Au": {"mass": 196.966, "lattice": "fcc", "lattice_constant": 4.08},
    "Ag": {"mass": 107.868, "lattice": "fcc", "lattice_constant": 4.09},
    "Fe": {"mass": 55.845, "lattice": "bcc", "lattice_constant": 2.87},
    "Ti": {"mass": 47.867, "lattice": "hcp", "lattice_constant": 2.95},
    "W": {"mass": 183.84, "lattice": "bcc", "lattice_constant": 3.16},
    "Mo": {"mass": 95.95, "lattice": "bcc", "lattice_constant": 3.15}
}


# ============================================================================
# Helper Functions
# ============================================================================

def _get_models_from_registry(element: str) -> List[Dict[str, str]]:
    """
    Get models for a specific element from the registry.
    
    Args:
        element: Chemical symbol of the element
        
    Returns:
        List of model dictionaries in the same format as query_openkim_models
    """
    potentials = read_registry()
    element_potentials = [p for p in potentials if p['element'] == element]
    
    models_info = []
    for potential in element_potentials:
        model_id = potential['model_id']
        model_type = potential['type']
        
        # Extract year if present in the ID
        year_match = re.search(r'_(\d{4})', model_id)
        year = year_match.group(1) if year_match else 'N/A'
        
        # Create human-readable name from ID
        name_part = model_id.split('__MO_')[0]
        name = name_part.replace('_', ' ')
        
        models_info.append({
            'id': model_id,
            'name': name,
            'type': model_type,
            'year': year
        })
    
    return models_info


def query_openkim_models(element: str, max_results: int = 50) -> List[Dict[str, str]]:
    """
    Query OpenKIM repository for available models for a specific element.
    Returns detailed information about each model for user selection.
    
    Args:
        element: Chemical symbol of the element (e.g., 'Cu', 'Al')
        max_results: Maximum number of models to return (default: 50)
        
    Returns:
        List of dictionaries with model information:
        [
            {
                'id': 'EAM_Dynamo_...',
                'name': 'Human-readable name',
                'type': 'EAM/LJ/SW/etc',
                'year': '2020' (if available)
            },
            ...
        ]
    """
    try:
        # Query OpenKIM browse page for the element
        url = f"https://openkim.org/browse/models/by-species/{element}"
        logger.info(f"Querying OpenKIM for {element} models: {url}")
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            import re
            
            # Pattern to match KIM model IDs
            pattern = r'([A-Z][A-Za-z0-9_]+__MO_\d+_\d+)'
            model_ids = re.findall(pattern, response.text)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_models = []
            for model_id in model_ids:
                if model_id not in seen:
                    seen.add(model_id)
                    unique_models.append(model_id)
            
            # Parse model information
            models_info = []
            for model_id in unique_models[:max_results]:
                # Extract model type from ID (e.g., EAM_Dynamo, LJ_Shifted, SW_)
                type_match = re.match(r'^([A-Z][A-Za-z_]+?)_', model_id)
                model_type = type_match.group(1) if type_match else 'Unknown'
                
                # Extract year if present in the ID
                year_match = re.search(r'_(\d{4})', model_id)
                year = year_match.group(1) if year_match else 'N/A'
                
                # Create human-readable name from ID
                # Remove the MO number suffix for readability
                name_part = model_id.split('__MO_')[0]
                name = name_part.replace('_', ' ')
                
                models_info.append({
                    'id': model_id,
                    'name': name,
                    'type': model_type,
                    'year': year
                })
            
            logger.info(f"Found {len(models_info)} models for {element} from OpenKIM")
            return models_info
        else:
            logger.warning(f"Failed to query OpenKIM: HTTP {response.status_code}")
            
            # Fallback to registry if web query fails
            logger.info(f"Using registry as fallback for {element}")
            return _get_models_from_registry(element)
            
    except Exception as e:
        logger.warning(f"Error querying OpenKIM for {element}: {e}")
        
        # Fallback to registry on error
        logger.info(f"Using registry as fallback for {element}")
        return _get_models_from_registry(element)


def get_installed_kim_models() -> List[str]:
    """
    Get list of currently installed KIM models.
    
    Returns:
        List of installed model IDs
    """
    try:
        if not os.path.exists(KIM_API_CMD):
            logger.warning(f"KIM API command not found at {KIM_API_CMD}")
            # Fallback to system path lookup if specific path fails
            cmd = "kim-api-collections-management"
        else:
            cmd = KIM_API_CMD
            
        result = subprocess.run(
            [cmd, "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse the output to extract model IDs
            import re
            pattern = r'([A-Z][A-Za-z0-9_]+__MO_\d+_\d+)'
            models = re.findall(pattern, result.stdout)
            return list(set(models))  # Remove duplicates
        else:
            logger.warning("Failed to list installed KIM models")
            return []
            
    except Exception as e:
        logger.warning(f"Error listing installed KIM models: {e}")
        return []


def is_model_installed(model_id: str) -> bool:
    """
    Check if a specific KIM model is installed.
    
    Args:
        model_id: The KIM model ID to check
        
    Returns:
        True if the model is installed, False otherwise
    """
    installed_models = get_installed_kim_models()
    return model_id in installed_models


def find_best_kim_model(element: str, prefer_eam: bool = True) -> Optional[str]:
    """
    Find the best available KIM model for a given element.
    
    Strategy:
    1. Check registry for models for this element
    2. Check if any are already installed
    3. Query OpenKIM for additional options
    4. Return the first available model
    
    Args:
        element: Chemical symbol of the element
        prefer_eam: Prefer EAM potentials over other types
        
    Returns:
        Model ID if found, None otherwise
    """
    logger.info(f"Finding best KIM model for {element}")
    
    # First, check registry for models
    registry_models = _get_models_from_registry(element)
    
    if registry_models:
        logger.info(f"Found {len(registry_models)} models in registry for {element}")
        
        # Check if any are already installed
        for model_info in registry_models:
            model_id = model_info['id']
            if is_model_installed(model_id):
                logger.info(f"Using already installed model: {model_id}")
                return model_id
        
        # Return first registry model (will be installed later)
        logger.info(f"Will use registry model: {registry_models[0]['id']}")
        return registry_models[0]['id']
    
    # If no registry models, query OpenKIM
    logger.info(f"No registry models for {element}, querying OpenKIM...")
    available_models = query_openkim_models(element)
    
    if available_models:
        # Prefer EAM models if requested
        if prefer_eam:
            eam_models = [m for m in available_models if "EAM" in m['type']]
            if eam_models:
                logger.info(f"Found EAM model from OpenKIM: {eam_models[0]['id']}")
                return eam_models[0]['id']
        
        logger.info(f"Using model from OpenKIM: {available_models[0]['id']}")
        return available_models[0]['id']
    
    logger.error(f"No KIM models found for {element}")
    return None


def install_and_verify_kim_model(model_id: str, timeout: int = 120) -> Tuple[bool, str]:
    """
    Install a KIM model and verify it's available.
    
    Args:
        model_id: The KIM model ID to install
        timeout: Maximum time to wait for installation (seconds)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info(f"Installing KIM model: {model_id}")
    
    # Check if already installed
    if is_model_installed(model_id):
        logger.info(f"Model {model_id} is already installed")
        return True, "Model already installed"
    
    try:
        # Install the model
        logger.info(f"Downloading and installing {model_id}...")
        if not os.path.exists(KIM_API_CMD):
            cmd = "kim-api-collections-management"
        else:
            cmd = KIM_API_CMD
        logger.info(f"Running command: {cmd} install user {model_id}")

        result = subprocess.run(
            [cmd, "install", "user", model_id],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            # Many CLI tools write errors to stdout instead of stderr; include both
            # so the failure is actually diagnosable. Truncate to 1 KB each so we
            # don't dump megabytes into the chat.
            stderr_text = (result.stderr or "").strip()[:1024]
            stdout_text = (result.stdout or "").strip()[:1024]
            error_msg = (
                f"Installation failed (returncode={result.returncode}): "
                f"stderr={stderr_text!r}, stdout={stdout_text!r}"
            )
            logger.error(error_msg)
            return False, error_msg
        
        logger.info("Installation command completed, verifying...")
        
        # Verify installation with retries
        max_retries = 10
        retry_delay = 2
        
        for attempt in range(max_retries):
            if is_model_installed(model_id):
                logger.info(f"Model {model_id} successfully installed and verified")
                return True, "Installation successful"
            
            if attempt < max_retries - 1:
                logger.info(f"Verification attempt {attempt + 1}/{max_retries}, waiting...")
                time.sleep(retry_delay)
        
        error_msg = f"Model installed but not found in collection after {max_retries} attempts"
        logger.error(error_msg)
        return False, error_msg
        
    except subprocess.TimeoutExpired:
        error_msg = f"Installation timed out after {timeout} seconds"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Installation error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def get_element_properties(element: str) -> Optional[Dict]:
    """
    Get physical properties for an element.
    
    Args:
        element: Chemical symbol of the element
        
    Returns:
        Dictionary with mass, lattice type, and lattice constant, or None if not found
    """
    return ELEMENT_PROPERTIES.get(element)


def install_user_selected_model(model_id: str) -> Tuple[bool, str]:
    """
    Install a user-selected KIM model without automatic fallbacks.
    This is the simplified two-step approach: user selects, we install.
    
    Args:
        model_id: The KIM model ID chosen by the user
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info(f"Installing user-selected KIM model: {model_id}")
    
    # Check if already installed
    if is_model_installed(model_id):
        logger.info(f"Model {model_id} is already installed")
        return True, f"✅ Model {model_id} is already installed and ready to use."
    
    # Attempt installation
    success, message = install_and_verify_kim_model(model_id, timeout=180)
    
    if success:
        update_registry(model_id, downloaded=True)
        return True, f"✅ Successfully installed {model_id}. Ready to use!"
    else:
        # Provide clear failure message
        error_details = f"❌ Failed to install {model_id}.\n\nReason: {message}\n\n"
        error_details += "Please try selecting a different model from the list."
        return False, error_details


def build_registry_context_for_command(user_command: str) -> str:
    """
    Build KIM registry context string for a user command.

    Searches the user command for known element symbols and returns
    a formatted string with available KIM models for that element,
    suitable for including in LLM prompts.

    Args:
        user_command: Natural language command from user

    Returns:
        Formatted registry context string, or empty string if no element found
    """
    registry_context = ""
    lower_command = user_command.lower()

    for element in ELEMENT_PROPERTIES.keys():
        # Match chemical symbol as a standalone token to avoid false positives
        # (e.g., 'ar' in 'Orchestrator')
        import re
        if re.search(rf"\b{element.lower()}\b", lower_command):
            # Get models for this element from registry
            potentials = read_registry()
            element_models = [p for p in potentials if p['element'] == element]

            if element_models:
                registry_context = f"\n\nAVAILABLE KIM MODELS FOR {element} (from registry):\n"
                downloaded_models = [m for m in element_models if m['downloaded']]
                pending_models = [m for m in element_models if not m['downloaded']]

                if downloaded_models:
                    registry_context += "\nALREADY DOWNLOADED (Ready to use immediately):\n"
                    for i, model in enumerate(downloaded_models, 1):
                        registry_context += f"✅ {i}. {model['model_id']} ({model['type']})\n"
                        registry_context += f"   → This model is installed and ready to go!\n"

                if pending_models:
                    registry_context += "\nAVAILABLE FOR DOWNLOAD:\n"
                    for i, model in enumerate(pending_models, len(downloaded_models) + 1):
                        registry_context += f"⬇️ {i}. {model['model_id']} ({model['type']})\n"
            break

    return registry_context


# ============================================================================
# Registry Management Functions
# ============================================================================

def _create_default_registry() -> bool:
    """
    Create the default registry file if it doesn't exist.
    """
    try:
        content = """
# KIM Potentials Registry

This file tracks the status of KIM potentials used in the application.

## Single Element Potentials

| Element | Model ID | Type | Status | Last Checked | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| Cu | EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006 | EAM | ⬇️ | - | Recommended for Copper |
| Al | EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 | EAM | ⬇️ | - | Recommended for Aluminum |
| Ni | EAM_Dynamo_AcklandTichyVitek_1987_Ni__MO_977363131043_006 | EAM | ⬇️ | - | Recommended for Nickel |
| Si | SW_StillingerWeber_1985_Si__MO_103435434588_006 | SW | ⬇️ | - | Recommended for Silicon |

## Binary/Ternary Potentials

| Elements | Model ID | Type | Status | Last Checked | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| - | - | - | - | - | - |

---
**Last Updated:** 2025-12-08 09:00:00
"""
        with open(REGISTRY_FILE, 'w') as f:
            f.write(content.strip())
        logger.info(f"Created default registry file: {REGISTRY_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to create default registry: {e}")
        return False


def read_registry() -> List[Dict[str, str]]:

    """
    Read the KIM potentials registry from the markdown file.
    
    Returns:
        List of dictionaries containing potential information
    """
    if not os.path.exists(REGISTRY_FILE):
        logger.info(f"Registry file {REGISTRY_FILE} not found, creating default...")
        _create_default_registry()
        
    if not os.path.exists(REGISTRY_FILE):
        logger.warning(f"Registry file {REGISTRY_FILE} not found")
        return []
    
    potentials = []
    
    try:
        with open(REGISTRY_FILE, 'r') as f:
            content = f.read()
        
        # Parse the markdown table
        # Look for lines that match the table format
        lines = content.split('\n')
        in_single_element_section = False
        in_binary_section = False
        
        for line in lines:
            if '## Single Element Potentials' in line:
                in_single_element_section = True
                in_binary_section = False
                continue
            elif '## Binary/Ternary Potentials' in line:
                in_single_element_section = False
                in_binary_section = True
                continue
            elif line.startswith('##') or line.startswith('---'):
                in_single_element_section = False
                in_binary_section = False
                continue
            
            # Skip header and separator lines
            if '|' not in line or 'Element' in line or '---' in line:
                continue
            
            # Parse table row
            if in_single_element_section or in_binary_section:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 6:
                    element = parts[1]
                    model_id = parts[2]
                    model_type = parts[3]
                    downloaded = parts[4]
                    download_date = parts[5]
                    description = parts[6] if len(parts) > 6 else ''
                    
                    # Skip placeholder rows
                    if element == '-' or model_id == '-':
                        continue
                    
                    potentials.append({
                        'element': element,
                        'model_id': model_id,
                        'type': model_type,
                        'downloaded': downloaded == '✅',
                        'download_date': download_date,
                        'description': description,
                        'is_binary': in_binary_section
                    })
        
        logger.info(f"Read {len(potentials)} potentials from registry")
        return potentials
        
    except Exception as e:
        logger.error(f"Error reading registry: {e}")
        return []


def update_registry(model_id: str, downloaded: bool = True) -> bool:
    """
    Update the registry to mark a potential as downloaded.
    
    Args:
        model_id: The KIM model ID to update
        downloaded: Whether the model is downloaded (default: True)
        
    Returns:
        True if update was successful, False otherwise
    """
    if not os.path.exists(REGISTRY_FILE):
        _create_default_registry()

    if not os.path.exists(REGISTRY_FILE):
        logger.error(f"Registry file {REGISTRY_FILE} not found")
        return False
    
    try:
        with open(REGISTRY_FILE, 'r') as f:
            content = f.read()
        
        # Update the status and date for this model
        status_icon = '✅' if downloaded else '⬇️'
        date_str = datetime.now().strftime('%Y-%m-%d') if downloaded else '-'
        
        # Find and replace the line containing this model_id
        lines = content.split('\n')
        updated = False
        
        for i, line in enumerate(lines):
            if model_id in line and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 6:
                    # Update status and date
                    parts[4] = f' {status_icon} '
                    parts[5] = f' {date_str} '
                    lines[i] = '|'.join(parts)
                    updated = True
                    break
        
        if updated:
            # Update the "Last Updated" timestamp
            for i, line in enumerate(lines):
                if '**Last Updated:**' in line:
                    lines[i] = f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    break
            
            # Write back to file
            with open(REGISTRY_FILE, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Updated registry for {model_id}")
            return True
        else:
            logger.warning(f"Model {model_id} not found in registry")
            return False
            
    except Exception as e:
        logger.error(f"Error updating registry: {e}")
        return False


def initialize_recommended_potentials(progress_callback=None) -> Dict[str, bool]:
    """
    Download and install recommended potentials on first startup.
    
    Args:
        progress_callback: Optional callback function(model_id, status, message)
        
    Returns:
        Dictionary mapping model_id to success status
    """
    logger.info("Initializing recommended potentials...")
    results = {}
    
    for model_id in RECOMMENDED_POTENTIALS:
        logger.info(f"Installing recommended potential: {model_id}")
        
        if progress_callback:
            progress_callback(model_id, 'starting', f'Installing {model_id}...')
        
        success, message = install_and_verify_kim_model(model_id, timeout=180)
        results[model_id] = success
        
        if success:
            # Update registry
            update_registry(model_id, downloaded=True)
            if progress_callback:
                progress_callback(model_id, 'success', f'✅ Installed {model_id}')
        else:
            if progress_callback:
                progress_callback(model_id, 'error', f'❌ Failed to install {model_id}: {message}')
    
    logger.info(f"Initialization complete. {sum(results.values())}/{len(results)} potentials installed")
    return results


def verify_and_sync_registry() -> Dict[str, any]:
    """
    Cross-reference registry with actual installed KIM models and sync if needed.
    
    Returns:
        Dictionary with sync results
    """
    logger.info("Verifying registry against installed KIM models...")
    
    potentials = read_registry()
    installed_models = get_installed_kim_models()
    
    updates_made = 0
    mismatches = []
    
    for potential in potentials:
        model_id = potential['model_id']
        registry_says_downloaded = potential['downloaded']
        actually_installed = model_id in installed_models
        
        if registry_says_downloaded != actually_installed:
            mismatches.append({
                'model_id': model_id,
                'registry_status': 'downloaded' if registry_says_downloaded else 'not_downloaded',
                'actual_status': 'installed' if actually_installed else 'not_installed'
            })
            
            # Update registry to match reality
            if update_registry(model_id, downloaded=actually_installed):
                updates_made += 1
                logger.info(f"Updated registry for {model_id}: {actually_installed}")
    
    result = {
        'verified': True,
        'total_checked': len(potentials),
        'mismatches_found': len(mismatches),
        'updates_made': updates_made,
        'mismatches': mismatches
    }
    
    if mismatches:
        logger.warning(f"Found {len(mismatches)} registry mismatches, corrected {updates_made}")
    else:
        logger.info("Registry is in sync with installed models")
    
    return result


def get_registry_status() -> Dict[str, any]:
    """
    Get the current status of the registry.
    
    Returns:
        Dictionary with registry statistics
    """
    potentials = read_registry()
    
    downloaded_count = sum(1 for p in potentials if p['downloaded'])
    total_count = len(potentials)
    
    # Get installed models from KIM API
    installed_models = get_installed_kim_models()
    
    return {
        'total_potentials': total_count,
        'downloaded_potentials': downloaded_count,
        'not_downloaded': total_count - downloaded_count,
        'installed_models': len(installed_models),
        'registry_exists': os.path.exists(REGISTRY_FILE),
        'potentials': potentials
    }


def add_potential_to_registry(element: str, model_id: str, model_type: str, description: str = '') -> bool:
    """
    Add a new potential to the registry.
    
    Args:
        element: Element symbol(s)
        model_id: KIM model ID
        model_type: Type of potential (EAM, LJ, SW, etc.)
        description: Optional description
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(REGISTRY_FILE):
        _create_default_registry()

    if not os.path.exists(REGISTRY_FILE):
        logger.error(f"Registry file {REGISTRY_FILE} not found")
        return False
    
    try:
        with open(REGISTRY_FILE, 'r') as f:
            content = f.read()
        
        # Check if model already exists
        if model_id in content:
            logger.warning(f"Model {model_id} already in registry")
            return False
        
        # Determine if it's a binary/ternary potential
        is_binary = len(element.split('-')) > 1 or len(element.split('/')) > 1
        
        # Create new table row
        downloaded = '✅' if is_model_installed(model_id) else '⬇️'
        date_str = datetime.now().strftime('%Y-%m-%d') if is_model_installed(model_id) else '-'
        new_row = f"| {element} | {model_id} | {model_type} | {downloaded} | {date_str} | {description} |"
        
        # Find the appropriate section and add the row
        lines = content.split('\n')
        section_marker = '## Binary/Ternary Potentials' if is_binary else '## Single Element Potentials'
        
        for i, line in enumerate(lines):
            if section_marker in line:
                # Find the end of the table (next ## or ---)
                j = i + 1
                while j < len(lines) and not lines[j].startswith('##') and not lines[j].startswith('---'):
                    j += 1
                # Insert before the section end
                lines.insert(j - 1, new_row)
                break
        
        # Write back
        with open(REGISTRY_FILE, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Added {model_id} to registry")
        return True
        
    except Exception as e:
        logger.error(f"Error adding potential to registry: {e}")
        return False
