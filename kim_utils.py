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

logger = logging.getLogger(__name__)

# Verified working models - these have been tested and confirmed to work
VERIFIED_KIM_MODELS = {
    "Ar": [
        "LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004"
    ],
    "Cu": [
        "EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006",
        "EAM_Dynamo_AdamsFoilesWolfer_1989Universal6_Cu__MO_145873824897_001",
        "EAM_Dynamo_AcklandVitek_1990_Cu__MO_642748370624_001"
    ],
    "Al": [
        "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
        "EAM_CubicNaturalSpline_ErcolessiAdams_1994_Al__MO_800509458712_003"
    ],
    "Si": [
        "SW_StillingerWeber_1985_Si__MO_405512056662_005"
    ],
    "Ni": [
        "EAM_Dynamo_AcklandTichyVitek_1987_Ni__MO_977363131043_006",
        "EAM_CubicNaturalSpline_AngeloMoodyBaskes_1995_Ni__MO_800536961967_004"
    ],
    "Au": [
        "EAM_Dynamo_Ackland_1987_Au__MO_754413982908_001",
        "EAM_Dynamo_AdamsFoilesWolfer_1989_Au__MO_087738844640_001"
    ],
    "Ag": [
        "EAM_Dynamo_AcklandTichyVitek_1987_Ag__MO_212700056563_006"
    ],
    "Fe": [
        "EAM_Dynamo_AcklandBaconCalder_1997_Fe__MO_142799717516_006"
    ],
    "Ti": [
        "EAM_Dynamo_Ackland_1992_Ti__MO_748534961139_006"
    ],
    "W": [
        "EAM_Dynamo_Ackland_2003_W__MO_141627196590_006"
    ],
    "Mo": [
        "EAM_Dynamo_AcklandThetford_1987_Mo__MO_789280224186_000"
    ]
}

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


def query_openkim_models(element: str, max_results: int = 10) -> List[str]:
    """
    Query OpenKIM repository for available models for a specific element.
    
    Args:
        element: Chemical symbol of the element (e.g., 'Cu', 'Al')
        max_results: Maximum number of models to return
        
    Returns:
        List of model IDs available for the element
    """
    try:
        # Use the fetch tool approach - scrape the OpenKIM browse page
        url = f"https://openkim.org/browse/models?species={element}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Extract model IDs from the HTML
            import re
            # Pattern to match KIM model IDs
            pattern = r'([A-Z][A-Za-z0-9_]+__MO_\d+_\d+)'
            models = re.findall(pattern, response.text)
            
            # Filter for models that contain the element symbol
            element_models = [m for m in models if element in m or element.lower() in m.lower()]
            
            # Remove duplicates and limit results
            unique_models = list(dict.fromkeys(element_models))[:max_results]
            
            logger.info(f"Found {len(unique_models)} models for {element} from OpenKIM")
            return unique_models
        else:
            logger.warning(f"Failed to query OpenKIM: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        logger.warning(f"Error querying OpenKIM for {element}: {e}")
        return []


def get_installed_kim_models() -> List[str]:
    """
    Get list of currently installed KIM models.
    
    Returns:
        List of installed model IDs
    """
    try:
        result = subprocess.run(
            ["kim-api-collections-management", "list"],
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
    1. Check verified models list first
    2. Check if any verified models are already installed
    3. Query OpenKIM for additional options
    4. Return the first available model
    
    Args:
        element: Chemical symbol of the element
        prefer_eam: Prefer EAM potentials over other types
        
    Returns:
        Model ID if found, None otherwise
    """
    logger.info(f"Finding best KIM model for {element}")
    
    # First, check verified models
    if element in VERIFIED_KIM_MODELS:
        verified = VERIFIED_KIM_MODELS[element]
        logger.info(f"Found {len(verified)} verified models for {element}")
        
        # Check if any are already installed
        for model_id in verified:
            if is_model_installed(model_id):
                logger.info(f"Using already installed model: {model_id}")
                return model_id
        
        # Return first verified model (will be installed later)
        logger.info(f"Will use verified model: {verified[0]}")
        return verified[0]
    
    # If no verified models, query OpenKIM
    logger.info(f"No verified models for {element}, querying OpenKIM...")
    available_models = query_openkim_models(element)
    
    if available_models:
        # Prefer EAM models if requested
        if prefer_eam:
            eam_models = [m for m in available_models if "EAM" in m]
            if eam_models:
                logger.info(f"Found EAM model from OpenKIM: {eam_models[0]}")
                return eam_models[0]
        
        logger.info(f"Using model from OpenKIM: {available_models[0]}")
        return available_models[0]
    
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
        result = subprocess.run(
            ["kim-api-collections-management", "install", "user", model_id],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Installation failed: {result.stderr}"
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


def ensure_kim_model_available(element: str, model_id: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Ensure a KIM model is available for the given element.
    
    This is the main entry point for robust KIM model management.
    It will:
    1. Use the provided model_id if given
    2. Otherwise, find the best available model for the element
    3. Install the model if not already installed
    4. Verify the installation
    
    Args:
        element: Chemical symbol of the element
        model_id: Optional specific model ID to use
        
    Returns:
        Tuple of (success: bool, model_id: str, message: str)
    """
    logger.info(f"Ensuring KIM model available for {element}")
    
    # If no model specified, find the best one
    if model_id is None:
        model_id = find_best_kim_model(element)
        if model_id is None:
            return False, "", f"No KIM models found for element {element}"
    
    # Check if model is already installed
    if is_model_installed(model_id):
        logger.info(f"Model {model_id} already available")
        return True, model_id, "Model already installed"
    
    # Install and verify the model
    success, message = install_and_verify_kim_model(model_id)
    
    if success:
        return True, model_id, message
    else:
        # If installation failed, try to find an alternative
        logger.warning(f"Failed to install {model_id}, looking for alternatives...")
        
        if element in VERIFIED_KIM_MODELS:
            # Try other verified models for this element
            for alt_model in VERIFIED_KIM_MODELS[element]:
                if alt_model != model_id:
                    logger.info(f"Trying alternative model: {alt_model}")
                    success, message = install_and_verify_kim_model(alt_model)
                    if success:
                        return True, alt_model, f"Used alternative model: {message}"
        
        return False, model_id, f"All installation attempts failed: {message}"
